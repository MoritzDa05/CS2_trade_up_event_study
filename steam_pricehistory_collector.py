# Collects the Steam Community Market price history (CS2) for every item of a list.
# Each item is saved at once and logged in a manifest, so a stopped run resumes.
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urlencode

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

BASE_URL = "https://steamcommunity.com/market/pricehistory/"
APPID_CS2 = 730

# Requested at start-up to check the cookie.
PROBE_ITEM = "AK-47 | Redline (Field-Tested)"

# Steam sends English month names; strptime("%b") would depend on the locale.
MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

# All thresholds in config.py are in EUR.
EURO = "€"

RATE_LIMIT_BACKOFF = [60, 120, 300, 600, 900]  # seconds, on HTTP 429

# This many 400 responses in a row mean a dead cookie, not unknown items.
CONSECUTIVE_400_AUTH_THRESHOLD = 10

log = logging.getLogger("collector")


class SteamAuthError(RuntimeError):
    pass


class SteamRateLimit(RuntimeError):
    pass


class SteamTransient(RuntimeError):
    pass


def build_session(cookie_value: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://steamcommunity.com/market/",
    })
    s.cookies.set("steamLoginSecure", cookie_value,
                  domain="steamcommunity.com", path="/")
    return s


def _request_once(session: requests.Session, name: str, currency: int,
                  appid: int, timeout: float) -> Optional[dict]:
    # Encoded by hand so that spaces become %20, not +.
    query = urlencode(
        {"appid": appid, "currency": currency, "market_hash_name": name},
        quote_via=quote,
    )
    url = f"{BASE_URL}?{query}"

    try:
        # A redirect means "go to login", i.e. a dead cookie.
        r = session.get(url, timeout=timeout, allow_redirects=False)
    except requests.Timeout as e:
        raise SteamTransient(f"Timeout: {e}") from e
    except requests.RequestException as e:
        raise SteamTransient(f"Network error: {e}") from e

    if r.status_code in (301, 302, 303, 307, 308):
        raise SteamAuthError("Redirect to the login page -> cookie invalid or expired.")
    if r.status_code == 429:
        raise SteamRateLimit("HTTP 429 -- rate limit reached.")
    if r.status_code in (401, 403):
        raise SteamAuthError(f"HTTP {r.status_code} -- not authenticated.")
    if r.status_code == 400:
        # Steam sends 400 for unknown items and dead cookies alike; collect() counts them.
        return None
    if r.status_code >= 500:
        raise SteamTransient(f"HTTP {r.status_code} -- server error.")
    if r.status_code != 200:
        raise SteamTransient(f"Unexpected status {r.status_code}.")

    try:
        payload = r.json()
    except ValueError as e:
        raise SteamTransient(f"Response is not JSON: {e}") from e

    if not isinstance(payload, dict) or not payload.get("success"):
        return None

    return payload


def fetch_with_retry(session: requests.Session, name: str, currency: int,
                     appid: int, timeout: float, max_retries: int) -> Optional[dict]:
    for attempt in range(max_retries + 1):
        try:
            return _request_once(session, name, currency, appid, timeout)

        except SteamRateLimit as e:
            if attempt >= max_retries:
                raise
            wait = RATE_LIMIT_BACKOFF[min(attempt, len(RATE_LIMIT_BACKOFF) - 1)]
            wait *= random.uniform(0.85, 1.15)
            log.warning("%s Waiting %.0f s (attempt %d/%d).",
                        e, wait, attempt + 1, max_retries)
            time.sleep(wait)

        except SteamTransient as e:
            if attempt >= max_retries:
                raise
            wait = min(5 * (2 ** attempt), 120) * random.uniform(0.85, 1.15)
            log.warning("%s Waiting %.0f s (attempt %d/%d).",
                        e, wait, attempt + 1, max_retries)
            time.sleep(wait)

    return None


# Format "Oct 22 2025 01: +0" (UTC): hourly in the last weeks, daily before.
def parse_steam_timestamp(s: str) -> datetime:
    parts = s.split()
    if len(parts) < 4:
        raise ValueError(f"Unexpected time format: {s!r}")
    month = MONTHS[parts[0]]
    day = int(parts[1])
    year = int(parts[2])
    hour = int(parts[3].rstrip(":"))
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


# One row per day; several hourly rows give the volume-weighted mean.
def payload_to_frame(payload: dict, name: str) -> pd.DataFrame:
    rows = payload.get("prices") or []
    records = []
    for entry in rows:
        try:
            ts_raw, price, volume = entry[0], entry[1], entry[2]
            records.append((parse_steam_timestamp(ts_raw), float(price), int(volume)))
        except (ValueError, KeyError, IndexError, TypeError) as e:
            log.debug("Row skipped (%s): %r", e, entry)

    if not records:
        return pd.DataFrame(columns=[
            "market_hash_name", "date", "median_price", "volume",
            "n_obs", "granularity",
        ])

    df = pd.DataFrame(records, columns=["ts_utc", "median_price", "volume"])
    df["date"] = df["ts_utc"].dt.date

    df = df.copy()
    df["_pv"] = df["median_price"] * df["volume"]
    g = df.groupby("date", as_index=False).agg(
        _pv=("_pv", "sum"),
        volume=("volume", "sum"),
        n_obs=("median_price", "size"),
        _mean=("median_price", "mean"),
    )
    g["median_price"] = np.where(
        (g["n_obs"] == 1) | (g["volume"] <= 0),
        g["_mean"],
        g["_pv"] / g["volume"].replace(0, np.nan),
    )
    out = g[["date", "median_price", "volume", "n_obs"]].copy()
    out["granularity"] = np.where(out["n_obs"] > 1, "hourly_agg", "daily")

    out.insert(0, "market_hash_name", name)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date").reset_index(drop=True)


# File-system-safe name; the SHA1 suffix keeps it unique.
def item_slug(name: str) -> str:
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]
    base = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")[:80]
    return f"{base}__{digest}" if base else digest


def append_manifest(manifest_path: Path, record: dict) -> None:
    with manifest_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())  # survives hard crashes


# The last entry per item wins.
def read_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        return pd.DataFrame(columns=["market_hash_name", "status", "n_days",
                                     "first_date", "last_date", "message",
                                     "fetched_at"])
    records = []
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    log.warning("Broken manifest line skipped.")
    if not records:
        return pd.DataFrame(columns=["market_hash_name", "status"])
    df = pd.DataFrame(records)
    return df.drop_duplicates(subset="market_hash_name", keep="last")


def load_item_list(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".csv", ".tsv"):
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep, encoding="utf-8")
        if "market_hash_name" not in df.columns:
            raise SystemExit(
                f"'{path}' has no column 'market_hash_name'. "
                f"Found: {list(df.columns)}"
            )
    else:
        names = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()]
        df = pd.DataFrame({"market_hash_name": [n for n in names if n]})

    df["market_hash_name"] = df["market_hash_name"].astype(str).str.strip()
    before = len(df)
    df = df.drop_duplicates(subset="market_hash_name").reset_index(drop=True)
    if len(df) < before:
        log.info("%d duplicates removed from the item list.", before - len(df))
    return df


def collect(args) -> None:
    load_dotenv()
    cookie = os.getenv("STEAM_LOGIN_SECURE", "").strip()
    if not cookie:
        raise SystemExit(
            "STEAM_LOGIN_SECURE is missing. Create a .env file:\n"
            "    STEAM_LOGIN_SECURE=76561198...%7C%7C...\n"
            "(see README)"
        )

    currency = args.currency or int(os.getenv("STEAM_CURRENCY", "3"))

    outdir = Path(args.outdir)
    items_dir = outdir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / "manifest.jsonl"

    items = load_item_list(Path(args.items))
    meta_cols = [c for c in items.columns if c != "market_hash_name"]

    # Resume: skip items already in the manifest.
    manifest = read_manifest(manifest_path)
    if args.force:
        done = set()
    elif args.retry_failed:
        done = set(manifest.loc[
            manifest["status"].isin(["ok", "empty", "not_found"]),
            "market_hash_name"
        ]) if len(manifest) else set()
    else:
        done = set(manifest["market_hash_name"]) if len(manifest) else set()

    todo = items[~items["market_hash_name"].isin(done)].copy()
    if args.limit:
        todo = todo.head(args.limit)

    log.info("Item list: %d | already done: %d | to fetch: %d",
             len(items), len(done), len(todo))
    if todo.empty:
        log.info("Nothing to do.")
        return

    eta_min = len(todo) * (args.min_delay + args.max_delay) / 2 / 60
    log.info("Estimated run time (without backoffs): ~%.0f minutes.", eta_min)

    session = build_session(cookie)

    # Probe at start-up: fails early if the cookie is dead.
    log.info("Checking the cookie with probe item: %s", PROBE_ITEM)
    try:
        probe = fetch_with_retry(session, PROBE_ITEM, currency,
                                 args.appid, args.timeout, max_retries=2)
    except SteamAuthError as e:
        raise SystemExit(f"Cookie check failed: {e}")
    if probe is None:
        raise SystemExit(
            "The probe returned no result. Most likely the cookie is invalid "
            "or expired; copy it again from the browser (see README).\n"
            "If the value contains '||' instead of '%7C%7C', try the other form."
        )
    # The currency follows the account's wallet, not the currency parameter.
    prefix, suffix = probe.get("price_prefix", ""), probe.get("price_suffix", "")
    log.info("Cookie OK. Currency according to Steam: prefix=%r suffix=%r", prefix, suffix)
    if not args.allow_currency and EURO not in f"{prefix}{suffix}":
        raise SystemExit(
            f"Steam returns prices with prefix={prefix!r} suffix={suffix!r}, "
            f"expected {EURO!r}. The currency follows the wallet of the "
            "account, not STEAM_CURRENCY. With a non-euro account all "
            "prices would be in the wrong currency -- stopping. "
            "Use --allow-currency to override this deliberately."
        )

    consecutive_400 = 0
    counts = {"ok": 0, "empty": 0, "not_found": 0, "failed": 0}
    t_start = time.monotonic()

    try:
        for i, row in enumerate(todo.itertuples(index=False), start=1):
            name = row.market_hash_name
            status, message, frame = "failed", "", None

            try:
                payload = fetch_with_retry(session, name, currency,
                                           args.appid, args.timeout,
                                           args.max_retries)
                if payload is None:
                    consecutive_400 += 1
                    status = "not_found"
                    message = "HTTP 400 / success=false"
                    if consecutive_400 >= CONSECUTIVE_400_AUTH_THRESHOLD:
                        raise SteamAuthError(
                            f"{consecutive_400} consecutive 400 responses. "
                            "The cookie has probably expired -- stopping so that "
                            "the rest of the list is not wrongly marked as "
                            "'not found'."
                        )
                else:
                    consecutive_400 = 0
                    frame = payload_to_frame(payload, name)
                    status = "ok" if len(frame) else "empty"

            except SteamAuthError:
                raise
            except Exception as e:
                status, message = "failed", f"{type(e).__name__}: {e}"
                log.error("[%d/%d] %s -> %s", i, len(todo), name, message)

            # Saved at once, so a crash loses at most this item.
            record = {
                "market_hash_name": name,
                "status": status,
                "n_days": int(len(frame)) if frame is not None else 0,
                "first_date": (str(frame["date"].min().date())
                               if frame is not None and len(frame) else None),
                "last_date": (str(frame["date"].max().date())
                              if frame is not None and len(frame) else None),
                "message": message,
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }

            if frame is not None and len(frame):
                for col in meta_cols:
                    frame[col] = getattr(row, col, None)
                target = items_dir / f"{item_slug(name)}.parquet"
                frame.to_parquet(target, index=False)

            append_manifest(manifest_path, record)
            counts[status] = counts.get(status, 0) + 1

            if i % 10 == 0 or i == len(todo):
                elapsed = time.monotonic() - t_start
                log.info("[%d/%d] ok=%d empty=%d not_found=%d failed=%d "
                         "| %.1f min elapsed",
                         i, len(todo), counts["ok"], counts["empty"],
                         counts["not_found"], counts["failed"], elapsed / 60)

            # Random pause between requests.
            if i < len(todo):
                time.sleep(random.uniform(args.min_delay, args.max_delay))

    except KeyboardInterrupt:
        log.warning("Interrupted by user. Progress is saved in the manifest "
                    "-- restart to resume.")
    except SteamAuthError as e:
        log.error("AUTH ERROR: %s", e)
        log.error("Get a fresh cookie and restart -- the run resumes.")

    log.info("Done. ok=%d empty=%d not_found=%d failed=%d",
             counts["ok"], counts["empty"], counts["not_found"], counts["failed"])


def build_panel(outdir: Path) -> Optional[pd.DataFrame]:
    files = sorted((outdir / "items").glob("*.parquet"))
    if not files:
        log.warning("No item files found.")
        return None
    frames = [pd.read_parquet(f) for f in files]
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["market_hash_name", "date"]).reset_index(drop=True)

    panel.to_parquet(outdir / "panel_long.parquet", index=False)
    panel.to_csv(outdir / "panel_long.csv", index=False, encoding="utf-8")
    log.info("Panel: %d rows, %d items, %s to %s",
             len(panel), panel["market_hash_name"].nunique(),
             panel["date"].min().date(), panel["date"].max().date())
    return panel


# Coverage per item around the event, in calendar days (pilot check).
def qc_report(outdir: Path, event_date: str, est_start: int, est_end: int,
              post_end: int) -> None:
    panel_path = outdir / "panel_long.parquet"
    if not panel_path.exists():
        log.error("No panel found. Run without --qc first.")
        return
    panel = pd.read_parquet(panel_path)

    t0 = pd.Timestamp(event_date)
    win_start = t0 + timedelta(days=est_start)
    win_end = t0 + timedelta(days=post_end)
    n_days_window = (win_end - win_start).days + 1

    log.info("=" * 68)
    log.info("QC REPORT | event t=0: %s", t0.date())
    log.info("Analysis window: %s to %s (%d calendar days)",
             win_start.date(), win_end.date(), n_days_window)
    log.info("Estimation window: t%+d to t%+d | event window: t0 to t%+d",
             est_start, est_end, post_end)
    log.info("=" * 68)

    win = panel[(panel["date"] >= win_start) & (panel["date"] <= win_end)]
    est = panel[(panel["date"] >= win_start)
                & (panel["date"] <= t0 + timedelta(days=est_end))]

    rows = []
    for name, g in win.groupby("market_hash_name"):
        g_est = est[est["market_hash_name"] == name]
        rows.append({
            "market_hash_name": name,
            "days_observed": len(g),
            "coverage": len(g) / n_days_window,
            "days_in_estimation": len(g_est),
            "median_daily_volume": g["volume"].median(),
            "mean_daily_volume": g["volume"].mean(),
            "share_volume_lt_5": (g["volume"] < 5).mean(),
            "price_min": g["median_price"].min(),
            "price_max": g["median_price"].max(),
            "covers_start": g["date"].min() <= win_start + timedelta(days=3),
            "covers_end": g["date"].max() >= win_end - timedelta(days=3),
        })

    if not rows:
        log.error("No observations in the window. Is the event date right?")
        return

    qc = pd.DataFrame(rows).sort_values("coverage")
    qc.to_csv(outdir / "qc_report.csv", index=False, encoding="utf-8")

    log.info("Items in the panel:                   %d", len(qc))
    log.info("  Coverage >= 95%%:                    %d", (qc["coverage"] >= 0.95).sum())
    log.info("  Coverage 80-95%%:                    %d",
             ((qc["coverage"] >= 0.80) & (qc["coverage"] < 0.95)).sum())
    log.info("  Coverage < 80%% (critical):         %d", (qc["coverage"] < 0.80).sum())
    log.info("  Median volume < 5/day (illiquid):   %d", (qc["median_daily_volume"] < 5).sum())
    log.info("  Window start not covered:           %d", (~qc["covers_start"]).sum())
    log.info("  Window end not covered:             %d", (~qc["covers_end"]).sum())

    worst = qc.head(10)
    if len(worst):
        log.info("-- 10 items with the lowest coverage --")
        for r in worst.itertuples(index=False):
            log.info("  %5.1f%%  vol_med=%6.1f  %s",
                     r.coverage * 100, r.median_daily_volume, r.market_hash_name)

    log.info("Details: %s", outdir / "qc_report.csv")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Price history collector for the Steam Community Market (CS2)."
    )
    p.add_argument("--items", required=True,
                   help="CSV with a column 'market_hash_name', or TXT with one name per line.")
    p.add_argument("--outdir", default="data/new",
                   help="Output directory (the data of the thesis are in data/full).")
    p.add_argument("--appid", type=int, default=APPID_CS2)
    p.add_argument("--allow-currency", action="store_true",
                   help="Skip the currency check (use only deliberately).")
    p.add_argument("--currency", type=int, default=None,
                   help="Currency code of the request (default from .env).")
    p.add_argument("--limit", type=int, default=None,
                   help="Fetch only the first N open items (pilot).")
    p.add_argument("--min-delay", type=float, default=3.5,
                   help="Minimum pause between requests (seconds).")
    p.add_argument("--max-delay", type=float, default=5.5,
                   help="Maximum pause between requests (seconds).")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--force", action="store_true",
                   help="Fetch everything again, ignoring the manifest.")
    p.add_argument("--retry-failed", action="store_true",
                   help="Retry only items that failed before.")
    p.add_argument("--no-collect", action="store_true",
                   help="Do not collect, only build the panel and QC report.")
    p.add_argument("--qc", action="store_true", help="Write the quality report.")
    p.add_argument("--event-date", default="2025-10-23",
                   help="Day t = 0 of the QC report: first full UTC day after the "
                        "update (YYYY-MM-DD), as config.T0.")
    p.add_argument("--est-start", type=int, default=-190,
                   help="Start of the estimation window, calendar days relative to t0.")
    p.add_argument("--est-end", type=int, default=-15,
                   help="End of the estimation window (start of the gap).")
    p.add_argument("--post-end", type=int, default=90,
                   help="End of the event window, calendar days.")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(outdir / "collector.log", encoding="utf-8"),
        ],
    )

    if not args.no_collect:
        collect(args)

    build_panel(outdir)

    if args.qc:
        qc_report(outdir, args.event_date, args.est_start,
                  args.est_end, args.post_end)


if __name__ == "__main__":
    main()

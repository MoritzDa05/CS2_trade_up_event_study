# Builds the daily analysis panel from the raw panel. Problems are only flagged;
# later scripts decide (exception: spikes on low-volume days are set to NaN).
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

log = logging.getLogger("00_prepare")

TS_COLS = ["market_hash_name", "date", "median_price", "volume", "n_obs", "granularity"]
META_DROP = {"date", "median_price", "volume", "n_obs", "granularity", "first_sale_date"}


def setup_logging() -> None:
    C.OUT_LOGS.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(C.OUT_LOGS / "00_prepare.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    # Windows consoles (cp1252) cannot print ★ and ™: replace them instead of logging errors.
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)


def day(offset: int) -> pd.Timestamp:
    return C.T0 + pd.Timedelta(days=offset)


def ts_scalar(x: pd.Timestamp) -> pa.Scalar:
    return pa.scalar(x.to_pydatetime(), pa.timestamp("ms"))


def section(title: str) -> None:
    log.info("")
    log.info("-" * 70)
    log.info(title)
    log.info("-" * 70)


# First and last sale per item over the whole history, not just the window.
def load_trade_span(dset: ds.Dataset) -> pd.DataFrame:
    tbl = dset.to_table(columns=["market_hash_name", "date"])
    agg = tbl.group_by("market_hash_name").aggregate([("date", "min"), ("date", "max")])
    span = agg.to_pandas().rename(columns={"date_min": "first_trade", "date_max": "last_trade"})
    for c in ["first_trade", "last_trade"]:
        span[c] = pd.to_datetime(span[c]).dt.normalize().astype("datetime64[ns]")
    log.info("Items in the raw panel: %d  (%d rows, %s to %s)", len(span), tbl.num_rows,
             span.first_trade.min().date(), span.last_trade.max().date())
    return span.set_index("market_hash_name")


def load_meta(dset: ds.Dataset) -> pd.DataFrame:
    cols = [c for c in dset.schema.names if c not in META_DROP]
    parts = []
    for batch in dset.to_batches(columns=cols, batch_size=500_000):
        parts.append(batch.to_pandas().drop_duplicates("market_hash_name"))
    meta = pd.concat(parts).drop_duplicates("market_hash_name").set_index("market_hash_name")

    unknown = sorted(set(meta["group"].unique()) - set(C.ROLE_MAP))
    if unknown:
        raise ValueError(f"Groups without a role in config.ROLE_MAP: {unknown}")

    meta = meta.rename(columns={"pattern": "finish"})
    meta.insert(1, "role", meta["group"].map(C.ROLE_MAP))
    meta.insert(3, "subclass", meta["finish"].map(C.SUBCLASS_MAP).fillna(C.SUBCLASS_DEFAULT))
    log.info("Column 'first_sale_date' dropped (only sparsely filled, and it "
             "disagrees with the actual trading dates)")
    log.info("Column 'pattern' renamed to 'finish'")
    return meta


def load_observations(dset: ds.Dataset, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    filt = (ds.field("date") >= ts_scalar(start)) & (ds.field("date") <= ts_scalar(end))
    obs = dset.to_table(columns=TS_COLS, filter=filt).to_pandas()
    obs["date"] = pd.to_datetime(obs["date"]).dt.normalize().astype("datetime64[ns]")
    log.info("Loaded: %d rows, %s to %s (including %d buffer days)",
             len(obs), start.date(), end.date(), C.LOAD_BUFFER_DAYS)
    return obs


def technical_checks(obs: pd.DataFrame) -> pd.DataFrame:
    dups = obs.duplicated(["market_hash_name", "date"]).sum()
    if dups:
        raise ValueError(f"{dups} duplicate (item, date) rows. Check the collector.")
    log.info("Duplicates (item, date): 0")

    not_daily = (obs["granularity"] != "daily") | (obs["n_obs"] != 1)
    if not_daily.any():
        log.warning("%d rows are not 'daily' or have n_obs != 1 (hourly "
                    "aggregation inside the window!)", not_daily.sum())
    else:
        log.info("Granularity: every row 'daily' with n_obs = 1")

    bad = obs["median_price"] <= 0
    log.info("Rows removed for price <= 0: %d", bad.sum())
    obs = obs.loc[~bad].drop(columns=["n_obs", "granularity"])
    obs = obs.rename(columns={"median_price": "price_raw"})
    obs["volume"] = obs["volume"].astype("int32")
    return obs.sort_values(["market_hash_name", "date"]).reset_index(drop=True)


# Spike: a jump that reverses on the next trading day. Removed only on days with
# <= SPIKE_MAX_VOLUME trades; inside the protection window only if extreme.
def detect_spikes(obs: pd.DataFrame) -> pd.DataFrame:
    g = obs.groupby("market_hash_name", sort=False)
    p = obs["price_raw"]
    obs["p_prev"] = g["price_raw"].shift(1)
    obs["p_next"] = g["price_raw"].shift(-1)
    d_prev = (obs["date"] - g["date"].shift(1)).dt.days
    d_next = (g["date"].shift(-1) - obs["date"]).dt.days

    jump = np.log(p / obs["p_prev"]).abs() > C.SPIKE_JUMP
    revert = np.log(obs["p_next"] / obs["p_prev"]).abs() < C.SPIKE_REVERT
    close = (d_prev <= C.SPIKE_MAX_NEIGHBOR_GAP) & (d_next <= C.SPIKE_MAX_NEIGHBOR_GAP)

    obs["flag_spike"] = (jump & revert & close).fillna(False).astype(bool)
    t = (obs["date"] - C.T0).dt.days
    protected = t.between(*C.SPIKE_PROTECT_WINDOW)
    extreme = np.log(p / obs["p_prev"]).abs() > C.SPIKE_JUMP_PROTECTED
    obs["flag_spike_removed"] = (obs["flag_spike"] & (obs["volume"] <= C.SPIKE_MAX_VOLUME)
                                 & (~protected | extreme))
    obs["price"] = obs["price_raw"].where(~obs["flag_spike_removed"])
    return obs


# One row per item and day from the first sale on; no sale: volume 0, price NaN.
def build_calendar(obs: pd.DataFrame, span: pd.DataFrame,
                   load_start: pd.Timestamp, win_start: pd.Timestamp,
                   win_end: pd.Timestamp) -> pd.DataFrame:
    in_window = obs.loc[obs["date"] >= win_start, "market_hash_name"].unique()
    items = pd.Index(sorted(in_window), name="market_hash_name")

    start = span.loc[items, "first_trade"].clip(lower=load_start)
    lengths = ((win_end - start).dt.days + 1).clip(lower=0).to_numpy()
    total = int(lengths.sum())

    names = np.repeat(items.to_numpy(), lengths)
    first_idx = np.repeat(np.cumsum(lengths) - lengths, lengths)
    offsets = np.arange(total) - first_idx
    dates = np.repeat(start.to_numpy(), lengths) + offsets.astype("timedelta64[D]")

    cats = pd.CategoricalDtype(items)
    grid = pd.DataFrame({"market_hash_name": pd.Categorical(names, dtype=cats),
                         "date": pd.to_datetime(dates).astype("datetime64[ns]")})
    obs = obs[obs["market_hash_name"].isin(items)].copy()
    obs["market_hash_name"] = pd.Categorical(obs["market_hash_name"], dtype=cats)

    keep = ["market_hash_name", "date", "price_raw", "price", "volume",
            "flag_spike", "flag_spike_removed"]
    panel = grid.merge(obs[keep], on=["market_hash_name", "date"], how="left", validate="1:1")
    panel["volume"] = panel["volume"].fillna(0).astype("int32")
    panel["traded"] = panel["volume"] > 0
    for c in ["flag_spike", "flag_spike_removed"]:
        panel[c] = panel[c].fillna(False).astype(bool)
    log.info("Calendar: %d items, %d item-days (including the buffer)", len(items), len(panel))
    return panel


# ret_obs: trade to trade, with its gap in days; ret_ffill: daily, on the last known price.
def add_returns(panel: pd.DataFrame) -> pd.DataFrame:
    g = panel.groupby("market_hash_name", sort=False, observed=True)
    panel["price_ffill"] = g["price"].ffill()

    last_valid = panel["date"].where(panel["price"].notna())
    panel["days_since_price"] = (panel["date"] - last_valid.groupby(panel["market_hash_name"],
                                 observed=True).ffill()).dt.days

    valid = panel.loc[panel["price"].notna(), ["market_hash_name", "date", "price"]]
    gv = valid.groupby("market_hash_name", sort=False, observed=True)
    panel["ret_obs"] = np.nan
    panel["gap"] = np.nan
    panel.loc[valid.index, "ret_obs"] = np.log(valid["price"] / gv["price"].shift(1))
    panel.loc[valid.index, "gap"] = (valid["date"] - gv["date"].shift(1)).dt.days

    panel["ret_ffill"] = np.log(panel["price_ffill"]).groupby(panel["market_hash_name"],
                                                             observed=True).diff()
    return panel


# Liquidity, price level and flags per item, all from the estimation window.
def item_metrics(panel: pd.DataFrame, span: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    lo, hi = C.EST_WINDOW
    est = panel[panel["t"].between(lo, hi)]
    g = est.groupby("market_hash_name", observed=True)

    m = pd.DataFrame({
        "est_exist_days": g.size(),  # the calendar starts at the first sale
        "est_trade_days": g["traded"].sum(),
        "est_volume": g["volume"].sum(),
        "est_median_price": g["price"].median(),
    })
    m["trade_share"] = m["est_trade_days"] / m["est_exist_days"]
    m["vol_per_day"] = m["est_volume"] / m["est_exist_days"]

    traded = est.loc[est["traded"], ["market_hash_name", "date", "price_raw"]].copy()
    traded["gap_trade"] = traded.groupby("market_hash_name", observed=True)["date"].diff().dt.days
    gt = traded.groupby("market_hash_name", observed=True)
    m["p90_gap"] = gt["gap_trade"].quantile(0.9)
    m["high_price_share"] = gt["price_raw"].apply(lambda s: (s >= C.HIGH_PRICE_EUR).mean())

    items = meta.loc[meta.index.isin(panel["market_hash_name"].cat.categories)].copy()
    items = items.join(span).join(m)
    items.loc[items["est_trade_days"].fillna(0) == 0, "trade_share"] = np.nan

    tier = pd.cut(items["trade_share"], C.LIQ_BINS, labels=C.LIQ_LABELS, right=False)
    items["liq_tier"] = tier.astype("object").fillna("no_est_data")
    items["liq_main"] = items["trade_share"] >= C.LIQ_MAIN_MIN
    items["liq_robust"] = items["trade_share"] >= C.LIQ_ROBUST_MIN
    items["flag_high_price"] = items["high_price_share"] > C.HIGH_PRICE_SHARE
    items["flag_below_min_price"] = items["est_median_price"] < C.MIN_PRICE_EUR

    sp = panel.groupby("market_hash_name", observed=True).agg(
        n_spikes_flagged=("flag_spike", "sum"),
        n_spikes_removed=("flag_spike_removed", "sum"))
    # Same window as the spike protection rule.
    ev = panel[panel["t"].between(*C.SPIKE_PROTECT_WINDOW)].groupby("market_hash_name",
                                                                    observed=True)
    sp["n_spikes_removed_event"] = ev["flag_spike_removed"].sum()
    items = items.join(sp)

    for c in ["est_exist_days", "est_trade_days", "est_volume"]:
        items[c] = items[c].fillna(0).astype("int32")
    items.index.name = "market_hash_name"
    return items


# Days with the largest median return: whole window and estimation window.
def jump_days(panel: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    ok = items.index[items["est_median_price"] >= C.MIN_PRICE_EUR]
    r = panel[(panel["gap"] == 1) & panel["market_hash_name"].isin(ok)]
    daily = r.groupby("date").agg(t=("t", "first"), n_items=("ret_obs", "size"),
                                  median_ret=("ret_obs", "median"),
                                  share_negative=("ret_obs", lambda s: (s < 0).mean()))
    vol = panel.groupby("date")["volume"].sum()
    daily["total_volume"] = vol
    daily["volume_vs_median"] = vol / vol.median()
    daily = daily[daily["n_items"] >= C.JUMP_DAYS_MIN_ITEMS]
    daily["median_ret_pct"] = (np.exp(daily["median_ret"]) - 1) * 100

    def top(df: pd.DataFrame, label: str) -> pd.DataFrame:
        out = df.reindex(df["median_ret"].abs().sort_values(ascending=False).index)
        out = out.head(C.JUMP_DAYS_TOP_N).reset_index()
        out.insert(0, "ranking", label)
        out.insert(1, "rank", np.arange(1, len(out) + 1))
        return out

    lo, hi = C.EST_WINDOW
    res = pd.concat([top(daily, "full"),
                     top(daily[daily["t"].between(lo, hi)], "estimation_window")])
    cols = ["ranking", "rank", "date", "t", "n_items", "median_ret_pct", "share_negative",
            "total_volume", "volume_vs_median"]
    res = res[cols].reset_index(drop=True)
    num = ["median_ret_pct", "share_negative", "volume_vs_median"]
    res[num] = res[num].round(3)
    return res


# Last manifest line per item is its final status; also counts the requests.
def read_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        log.warning("%s not found -- no collector status", path.relative_to(C.ROOT))
        return pd.DataFrame(columns=["market_hash_name", "status", "message",
                                     "fetched_at", "requests"])
    m = pd.read_json(path, lines=True, convert_dates=False, encoding="utf-8")
    requests = m.groupby("market_hash_name").size().rename("requests")
    last = m.drop_duplicates("market_hash_name", keep="last")
    return last.join(requests, on="market_hash_name").reset_index(drop=True)


# Requested items that are missing from the raw panel, with the collector's status.
def not_collected(span: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    if not C.ITEMS_UNIVERSE.exists():
        log.warning("%s not found -- cannot report the never-collected items",
                    C.ITEMS_UNIVERSE.name)
        return pd.DataFrame(columns=["market_hash_name", "group"])
    # utf-8-sig: the item lists are written with a BOM.
    requested = pd.read_csv(C.ITEMS_UNIVERSE, encoding="utf-8-sig")
    missing = requested[~requested["market_hash_name"].isin(span.index)]
    missing = missing.rename(columns={"pattern": "finish"})
    keep = [c for c in ["market_hash_name", "group", "category", "weapon", "finish",
                        "wear", "stattrak", "collection", "case"] if c in missing.columns]
    missing = missing[keep].merge(
        manifest[["market_hash_name", "status", "message", "requests", "fetched_at"]],
        on="market_hash_name", how="left")
    return missing.sort_values(["group", "market_hash_name"]).reset_index(drop=True)


# How liquid are collected items of the same kind as the missing ones?
def missing_comparables(missing: pd.DataFrame, meta: pd.DataFrame,
                        items: pd.DataFrame) -> None:
    keys = ["group", "stattrak", "wear", "finish"]
    if missing.empty or not set(keys) <= set(missing.columns):
        return
    want = set(map(tuple, missing[keys].astype(str).to_numpy()))
    m = meta.loc[meta.index.difference(missing["market_hash_name"])]
    m = m[[k in want for k in map(tuple, m[keys].astype(str).to_numpy())]]
    share = items["trade_share"].reindex(m.index)
    log.info("Collected items of the same kind as the missing ones (same group, "
             "StatTrak status, wear and finish): %d, %d with a sale in the "
             "estimation window, highest trade share %.1f %% (main sample: >= %.0f %%)",
             len(m), int(share.notna().sum()),
             100 * share.max() if share.notna().any() else float("nan"),
             100 * C.LIQ_MAIN_MIN)


# Items retrieved per day and group, main run and currency check.
def retrieval_dates(manifest: pd.DataFrame) -> pd.DataFrame:
    if not C.ITEMS_UNIVERSE.exists():
        return pd.DataFrame()
    groups = pd.read_csv(C.ITEMS_UNIVERSE, encoding="utf-8-sig",
                         usecols=["market_hash_name", "group"])
    frames = []
    for run, m in [("main", manifest), ("check", read_manifest(C.CHECK_MANIFEST))]:
        ok = m[m["status"] == "ok"].merge(groups, on="market_hash_name", how="left")
        if ok.empty:
            continue
        ok["date"] = ok["fetched_at"].str[:10]
        agg = ok.groupby(["group", "date"]).agg(
            items=("market_hash_name", "size"), first=("fetched_at", "min"),
            last=("fetched_at", "max")).reset_index()
        agg.insert(0, "run", run)
        frames.append(agg)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(out):
        days = out.groupby(["run", "date"])["items"].sum()
        for (run, date), n in days.items():
            log.info("Retrieved (%s run): %s  %d items", run, date, n)
    return out


# USD listing cap in EUR per retrieval day: the most frequent price above HIGH_PRICE_EUR.
def listing_cap(dset: ds.Dataset, manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return pd.DataFrame()
    day = manifest.loc[manifest["status"] == "ok"].set_index(
        "market_hash_name")["fetched_at"].str[:10]
    tbl = dset.to_table(columns=["market_hash_name", "median_price"],
                        filter=ds.field("median_price") >= C.HIGH_PRICE_EUR
                        ).to_pandas()
    tbl["day"] = tbl["market_hash_name"].map(day)
    tbl["price"] = tbl["median_price"].round(2)
    rows = []
    for d, g in tbl.dropna(subset=["day"]).groupby("day"):
        counts = g["price"].value_counts()
        cap = counts.index[0]
        rows.append({"retrieved": d, "cap_eur": cap, "days_at_cap": int(counts.iloc[0]),
                     "items_at_cap": g.loc[g["price"] == cap, "market_hash_name"].nunique(),
                     "usd_per_eur": round(C.LISTING_CAP_USD / cap, 5)})
    out = pd.DataFrame(rows)
    if len(out):
        out["ratio_to_first"] = (out["cap_eur"] / out["cap_eur"].iloc[0]).round(5)
        for r in out.itertuples(index=False):
            log.info("Listing cap (USD %.0f) for items retrieved %s: EUR %.2f "
                     "(%d days, %d items), ratio to the first day %.4f",
                     C.LISTING_CAP_USD, r.retrieved, r.cap_eur, r.days_at_cap,
                     r.items_at_cap, r.ratio_to_first)
    return out


# Two retrievals of the same items: a constant ratio means one exchange rate per retrieval.
def currency_check(dset: ds.Dataset, manifest: pd.DataFrame) -> pd.DataFrame:
    if not C.CHECK_PANEL.exists():
        log.warning("%s not found -- no currency check",
                    C.CHECK_PANEL.relative_to(C.ROOT))
        return pd.DataFrame()
    cols = ["market_hash_name", "date", "median_price", "granularity"]
    check = pd.read_parquet(C.CHECK_PANEL, columns=cols)
    names = sorted(check["market_hash_name"].unique())
    main = dset.to_table(columns=cols, filter=ds.field("market_hash_name").isin(names)
                         ).to_pandas()
    for df in (check, main):
        df["date"] = pd.to_datetime(df["date"]).dt.normalize().astype("datetime64[ns]")
    m = main.merge(check, on=["market_hash_name", "date"], suffixes=("_main", "_check"))
    m = m[(m["granularity_main"] == "daily") & (m["granularity_check"] == "daily")]
    m["ratio"] = m["median_price_check"] / m["median_price_main"]

    levels = pd.cut(m["median_price_main"], [0, 0.5, 1, 5, np.inf], right=False,
                    labels=["below 0.50", "0.50 to 1", "1 to 5", "5 and more"])
    rows = []
    for breakdown, key in [("all", pd.Series("all", index=m.index)),
                           ("price level (EUR)", levels),
                           ("year", m["date"].dt.year)]:
        for value, r in m.groupby(key, observed=True)["ratio"]:
            rows.append({"breakdown": breakdown, "value": str(value), "days": len(r),
                         "items": m.loc[r.index, "market_hash_name"].nunique(),
                         "median": r.median(), "sd": r.std(), "min": r.min(),
                         "max": r.max()})
    out = pd.DataFrame(rows).round(5)

    first = manifest.loc[manifest["market_hash_name"].isin(names), "fetched_at"]
    second = read_manifest(C.CHECK_MANIFEST)["fetched_at"]
    gap = (pd.to_datetime(second.min()) - pd.to_datetime(first.min()))
    log.info("Currency check: %d items, %d days in both retrievals, retrieved "
             "%s and %s (%.1f days apart)", len(names), len(m), first.min(),
             second.min(), gap.total_seconds() / 86400)
    a = out[out["breakdown"] == "all"].iloc[0]
    log.info("  ratio second / first retrieval: median %.4f, sd %.4f",
             a["median"], a["sd"])
    for r in out[out["breakdown"] != "all"].itertuples(index=False):
        log.info("  %-18s %-11s days %5d  median %.4f  sd %.4f", r.breakdown,
                 r.value, r.days, r.median, r.sd)
    return out


def summarize(panel: pd.DataFrame, items: pd.DataFrame, obs_win: pd.DataFrame,
              span: pd.DataFrame, jumps: pd.DataFrame, missing: pd.DataFrame) -> None:
    section("SUMMARY (for the data section of the thesis)")
    n_requested = len(span) + len(missing)
    log.info("Sample funnel: %d requested -> %d in the raw panel -> %d with data "
             "in the window", n_requested, len(span), len(items))
    log.info("  never collected (failed or empty at Steam): %d  "
             "-> output/tables/00_items_not_collected.csv", len(missing))
    if len(missing):
        by_group = missing["group"].value_counts().to_dict() if "group" in missing else {}
        log.info("  of those, by group: %s", by_group)
    log.info("  in the panel but no sale inside the window: %d", len(span) - len(items))
    log.info("Item-days in the window:          %d  (with a sale: %d)",
             len(panel), int(panel["traded"].sum()))

    n_obs = len(obs_win)
    n_flag = int(obs_win["flag_spike"].sum())
    n_rem = int(obs_win["flag_spike_removed"].sum())
    log.info("Spikes detected:                  %d  (%.3f %% of the trading days)",
             n_flag, 100 * n_flag / n_obs)
    log.info("  of those removed (volume <= %d): %d  (%.3f %%)",
             C.SPIKE_MAX_VOLUME, n_rem, 100 * n_rem / n_obs)
    pw = obs_win["t"].between(*C.SPIKE_PROTECT_WINDOW)
    log.info("  in the protection window %s: %d detected, %d removed (only a factor > %.0f)",
             C.SPIKE_PROTECT_WINDOW, int((obs_win["flag_spike"] & pw).sum()),
             int((obs_win["flag_spike_removed"] & pw).sum()), np.exp(C.SPIKE_JUMP_PROTECTED))
    log.info("  -> inspect output/tables/00_spikes_event_window.csv manually")
    log.info("Price on the cluster at %.2f EUR: %d item-days",
             C.MASS_POINT_EUR, int(panel["flag_mass_point"].sum()))

    section("Roles x liquidity (items)")
    tab = pd.crosstab(items["role"], items["liq_tier"], margins=True, margins_name="total")
    order = [c for c in ["lt10", "10_30", "30_50", "ge50", "no_est_data", "total"]
             if c in tab.columns]
    for line in tab[order].to_string().splitlines():
        log.info(line)

    section("treated_output: subclass x liquidity")
    to = items[items["role"] == "treated_output"]
    tab2 = pd.crosstab(to["subclass"], to["liq_tier"], margins=True, margins_name="total")
    order2 = [c for c in order if c in tab2.columns]
    for line in tab2[order2].to_string().splitlines():
        log.info(line)
    log.info("flag_high_price (treated_output): %d items, of those liq_main: %d",
             int(to["flag_high_price"].sum()), int((to["flag_high_price"] & to["liq_main"]).sum()))

    for label, title in [("full", "Market-wide jump days over the whole window (top 5)"),
                         ("estimation_window",
                          "Market-wide jump days in the estimation window (top 5)")]:
        section(title)
        sub = jumps[jumps["ranking"] == label]
        for r in sub.head(5).itertuples(index=False):
            log.info("  %s  t=%+4d  median %+6.1f %%  negative %4.0f %%  volume x%.1f",
                     pd.Timestamp(r.date).date(), r.t, r.median_ret_pct,
                     100 * r.share_negative, r.volume_vs_median)
    g = jumps[jumps["ranking"] == "full"]
    rank = g.loc[g["t"] == 0, "rank"]
    log.info("t=0 in the overall top %d: %s", C.JUMP_DAYS_TOP_N,
             f"yes, rank {int(rank.iloc[0])}" if len(rank) else "NO -> check the event date")
    log.info("Cross-check the estimation-window list against the patch notes: "
             "output/tables/00_jump_days.csv")


def main() -> None:
    setup_logging()
    for d in [C.CLEAN_DIR, C.OUT_TABLES]:
        d.mkdir(parents=True, exist_ok=True)
    if not C.RAW_PANEL.exists():
        raise FileNotFoundError(f"Raw panel not found: {C.RAW_PANEL}")

    win_start, win_end = day(C.WINDOW[0]), day(C.WINDOW[1])
    load_start = win_start - pd.Timedelta(days=C.LOAD_BUFFER_DAYS)
    log.info("T0 = %s | window %s to %s | estimation window t = %s",
             C.T0.date(), win_start.date(), win_end.date(), C.EST_WINDOW)

    dset = ds.dataset(C.RAW_PANEL)

    section("1. Trading span")
    span = load_trade_span(dset)

    section("2. Metadata")
    meta = load_meta(dset)
    log.info("Roles: %s", meta["role"].value_counts().to_dict())

    section("3. Observations + technical checks")
    obs = technical_checks(load_observations(dset, load_start, win_end))

    section("4. Spikes")
    obs = detect_spikes(obs)
    obs["t"] = (obs["date"] - C.T0).dt.days
    obs_win = obs[obs["t"].between(*C.WINDOW)]
    log.info("Spikes in the window: %d detected, %d removed", obs_win["flag_spike"].sum(),
             obs_win["flag_spike_removed"].sum())

    section("5. Calendar")
    no_data = sorted(set(span.index) - set(obs_win["market_hash_name"].unique()))
    pd.DataFrame({"market_hash_name": no_data}).join(
        meta[["group"]], on="market_hash_name").join(span, on="market_hash_name").to_csv(
        C.OUT_TABLES / "00_items_without_window_data.csv", index=False, encoding="utf-8")
    log.info("Items with no sale inside the window (not in the panel): %d", len(no_data))
    panel = build_calendar(obs, span, load_start, win_start, win_end)

    section("6. Returns")
    panel = add_returns(panel)
    panel["t"] = (panel["date"] - C.T0).dt.days.astype("int16")
    panel = panel[panel["t"].between(*C.WINDOW)].reset_index(drop=True)
    panel["dow"] = panel["date"].dt.dayofweek.astype("int8")
    panel["flag_mass_point"] = (panel["price_raw"] - C.MASS_POINT_EUR).abs() <= C.MASS_POINT_TOL
    log.info("ret_obs present: %d | of those gap > %d: %d",
             panel["ret_obs"].notna().sum(), C.MAX_GAP_RET, (panel["gap"] > C.MAX_GAP_RET).sum())

    section("7. Item metrics")
    items = item_metrics(panel, span, meta)

    section("8. Diagnostics")
    jumps = jump_days(panel, items)
    jumps.to_csv(C.OUT_TABLES / "00_jump_days.csv", index=False, encoding="utf-8")

    manifest = read_manifest(C.MANIFEST)
    missing = not_collected(span, manifest)
    missing.to_csv(C.OUT_TABLES / "00_items_not_collected.csv", index=False, encoding="utf-8")
    log.info("Requested but never collected: %d items "
             "-> output/tables/00_items_not_collected.csv", len(missing))
    if "status" in missing:
        log.info("  collector status: %s", missing["status"].value_counts().to_dict())
    missing_comparables(missing, meta, items)

    retrieval_dates(manifest).to_csv(C.OUT_TABLES / "00_retrieval.csv", index=False,
                                     encoding="utf-8")
    currency_check(dset, manifest).to_csv(C.OUT_TABLES / "00_currency_check.csv",
                                          index=False, encoding="utf-8")
    listing_cap(dset, manifest).to_csv(C.OUT_TABLES / "00_listing_cap.csv",
                                       index=False, encoding="utf-8")

    ev = obs_win[obs_win["flag_spike"] & obs_win["t"].between(*C.SPIKE_PROTECT_WINDOW)]
    ev[["market_hash_name", "date", "t", "p_prev", "price_raw", "p_next", "volume",
        "flag_spike_removed"]].sort_values(["t", "market_hash_name"]).to_csv(
        C.OUT_TABLES / "00_spikes_event_window.csv", index=False, encoding="utf-8")

    overview = items.groupby(["role", "group"]).agg(
        items=("trade_share", "size"),
        with_est_data=("est_trade_days", lambda s: int((s > 0).sum())),
        liq_main=("liq_main", "sum"), liq_robust=("liq_robust", "sum"),
        median_price=("est_median_price", "median"),
        high_price=("flag_high_price", "sum"),
        spikes_removed=("n_spikes_removed", "sum")).round(2)
    overview.to_csv(C.OUT_TABLES / "00_sample_overview.csv", encoding="utf-8")
    pd.crosstab(items["role"], items["liq_tier"]).to_csv(
        C.OUT_TABLES / "00_liquidity_by_role.csv", encoding="utf-8")

    section("9. Save")
    cols = ["market_hash_name", "date", "t", "dow", "price_raw", "price", "volume", "traded",
            "price_ffill", "days_since_price", "ret_obs", "gap", "ret_ffill",
            "flag_spike", "flag_spike_removed", "flag_mass_point"]
    panel[cols].to_parquet(C.PANEL_CLEAN, index=False)
    items.reset_index().to_parquet(C.ITEMS_CLEAN, index=False)
    log.info("-> %s  (%d rows)", C.PANEL_CLEAN.relative_to(C.ROOT), len(panel))
    log.info("-> %s  (%d items)", C.ITEMS_CLEAN.relative_to(C.ROOT), len(items))

    summarize(panel, items, obs_win, span, jumps, missing)
    log.info("")
    log.info("Done.")


if __name__ == "__main__":
    main()

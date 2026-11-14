# Builds the samples from the flagged panel: one boolean column per variant, nothing
# is deleted. All criteria come from the estimation window (no look-ahead).
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

log = logging.getLogger("01_screening")

# Screening order: the first failed step becomes the drop reason.
STEPS = ["est_data", "burn_in", "min_price", "high_price", "liquidity",
         "min_returns", "subclass", "store_capsule"]


def setup_logging() -> None:
    C.OUT_LOGS.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(C.OUT_LOGS / "01_screening.log", mode="w", encoding="utf-8")
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


def section(title: str) -> None:
    log.info("")
    log.info("-" * 70)
    log.info(title)
    log.info("-" * 70)


def show(df: pd.DataFrame) -> None:
    for line in df.to_string().splitlines():
        log.info(line)


# Usable returns (gap <= MAX_GAP_RET) per item, in both estimation windows.
def add_return_counts(panel: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    ok = panel["ret_obs"].notna() & (panel["gap"] <= C.MAX_GAP_RET)

    for label, (lo, hi) in [("", C.EST_WINDOW), ("_robust", C.EST_WINDOW_ROBUST)]:
        sub = panel.loc[ok & panel["t"].between(lo, hi), "market_hash_name"]
        counts = sub.value_counts()
        items[f"n_est_returns{label}"] = (counts.reindex(items.index)
                                          .fillna(0).astype("int32"))

    log.info("Usable returns in the estimation window (gap <= %d): median %.0f, "
             "minimum %d", C.MAX_GAP_RET, items["n_est_returns"].median(),
             items["n_est_returns"].min())
    return items


def add_flags(items: pd.DataFrame) -> pd.DataFrame:
    est_start = C.T0 + pd.Timedelta(days=C.EST_WINDOW[0])
    items["listed_before_est"] = items["first_trade"] <= est_start
    items["days_before_est"] = (est_start - items["first_trade"]).dt.days

    items["is_store_capsule"] = items.index.isin(C.STORE_CAPSULES)
    missing = sorted(set(C.STORE_CAPSULES) - set(items.index))
    if missing:
        log.warning("%d names from STORE_CAPSULES are not in the data: %s",
                    len(missing), ", ".join(missing[:5]))

    n_new = int((~items["listed_before_est"]).sum())
    log.info("Listed after the start of the estimation window: %d items", n_new)
    log.info("Store-anchored capsules found: %d", int(items["is_store_capsule"].sum()))
    return items


# One boolean column per screening step, True = passed.
def screening_steps(items: pd.DataFrame, liq_min: float,
                    subclass_main: bool = True) -> pd.DataFrame:
    is_output = items["role"] == "treated_output"
    is_index = items["group"].isin({g for gs in C.INDEX_DEFS.values() for g in gs})

    steps = pd.DataFrame(index=items.index)
    steps["est_data"] = items["est_trade_days"] > 0
    steps["burn_in"] = items["listed_before_est"] | (not C.REQUIRE_FULL_EST_WINDOW)
    steps["min_price"] = items["est_median_price"] >= C.MIN_PRICE_EUR
    steps["high_price"] = ~items["flag_high_price"]
    steps["liquidity"] = items["trade_share"] >= liq_min
    steps["min_returns"] = items["n_est_returns"] >= C.MIN_EST_RETURNS
    steps["subclass"] = (~is_output) | (items["subclass"] == C.SUBCLASS_MAIN) \
        if subclass_main else True
    steps["store_capsule"] = ~(is_index & items["is_store_capsule"])
    return steps[STEPS].fillna(False).astype(bool)


def first_failure(steps: pd.DataFrame) -> pd.Series:
    reason = pd.Series("passed", index=steps.index, dtype=object)
    for step in reversed(STEPS):
        reason = reason.mask(~steps[step], step)
    return reason


def build_variants(items: pd.DataFrame) -> pd.DataFrame:
    main_steps = screening_steps(items, C.LIQ_MAIN_MIN)
    out = pd.DataFrame(index=items.index)
    out["drop_reason"] = first_failure(main_steps)
    out["main"] = main_steps.all(axis=1)

    out["liq50"] = screening_steps(items, C.LIQ_ROBUST_MIN).all(axis=1)
    out["subclass_all"] = screening_steps(items, C.LIQ_MAIN_MIN,
                                          subclass_main=False).all(axis=1)

    # Illiquid comparison group: like main, but trade share 10 % to 30 %.
    illiquid = screening_steps(items, 0.0)
    illiquid["liquidity"] = (items["trade_share"] >= C.LIQ_BINS[1]) & \
                            (items["trade_share"] < C.LIQ_MAIN_MIN)
    out["illiquid"] = illiquid.all(axis=1)

    # Launch: fails only the burn-in rule. min_returns is skipped here; MIN_REG_OBS applies in 04.
    launch_steps = main_steps.drop(columns=["burn_in", "min_returns"])
    out["launch"] = launch_steps.all(axis=1) & ~items["listed_before_est"]

    # Burn-in robustness: like main, but first sale at least BURN_IN_ROBUST_DAYS days earlier.
    est_start = C.T0 + pd.Timedelta(days=C.EST_WINDOW[0])
    early = items["first_trade"] <= est_start - pd.Timedelta(days=C.BURN_IN_ROBUST_DAYS)
    out["burn_in30"] = out["main"] & early
    log.info("Burn-in robustness (first sale >= %d days before t = %d): drops %d "
             "main-sample items", C.BURN_IN_ROBUST_DAYS, C.EST_WINDOW[0],
             int((out["main"] & ~early).sum()))
    log.info("  by role: %s", items.loc[out["main"] & ~early, "role"]
             .value_counts().to_dict())

    for name in C.SAMPLE_VARIANTS:
        if name not in out.columns:
            raise ValueError(f"Variant '{name}' declared in config but not built.")
    return out


# Cumulative waterfall per role, in the order of STEPS.
def sample_flow(items: pd.DataFrame) -> pd.DataFrame:
    steps = screening_steps(items, C.LIQ_MAIN_MIN)
    alive = pd.Series(True, index=items.index)
    rows = [items.groupby("role").size().rename("items_in_window")]
    for step in STEPS:
        alive = alive & steps[step]
        rows.append(alive.groupby(items["role"]).sum().rename(f"after_{step}"))
    flow = pd.concat(rows, axis=1).fillna(0).astype(int)
    flow.loc["TOTAL"] = flow.sum()
    return flow


def variant_counts(items: pd.DataFrame, variants: pd.DataFrame) -> pd.DataFrame:
    cols = list(C.SAMPLE_VARIANTS)
    tab = variants[cols].groupby(items["role"]).sum().astype(int)
    tab.loc["TOTAL"] = tab.sum()
    return tab


def index_constituents(items: pd.DataFrame, variants: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, groups in C.INDEX_DEFS.items():
        member = items["group"].isin(groups) & variants["main"]
        rows.append(pd.DataFrame({
            "index": name,
            "market_hash_name": items.index[member],
            "group": items.loc[member, "group"].to_numpy(),
            "est_median_price": items.loc[member, "est_median_price"].round(2).to_numpy(),
            "trade_share": items.loc[member, "trade_share"].round(3).to_numpy(),
        }))
    out = pd.concat(rows, ignore_index=True)
    log.info("Index constituents: %s",
             out.groupby("index").size().to_dict())
    return out


# Diagnostic only, never a filter: price before and at the end of each event window.
def event_coverage(panel: pd.DataFrame, items: pd.DataFrame,
                   variants: pd.DataFrame) -> pd.DataFrame:
    keep = variants["main"]
    names = items.index[keep]
    priced = panel.loc[panel["price"].notna(), ["market_hash_name", "t"]]
    priced = priced[priced["market_hash_name"].isin(names)]

    def has_price(lo: int, hi: int) -> pd.Series:
        sub = priced[priced["t"].between(lo, hi)]
        return items.index.isin(sub["market_hash_name"].unique())

    rows = []
    for label, (lo, hi) in C.EVENT_WINDOWS.items():
        anchor = has_price(lo - C.COVERAGE_ANCHOR_DAYS, lo - 1)
        end = has_price(max(hi - C.COVERAGE_END_DAYS + 1, lo), hi)
        df = pd.DataFrame({"role": items["role"], "keep": keep,
                           "anchor": anchor, "end": end, "both": anchor & end})
        df = df[df["keep"]]
        agg = df.groupby("role")[["anchor", "end", "both"]].mean().round(3)
        agg.insert(0, "window", label)
        agg.insert(1, "n_items", df.groupby("role").size())
        rows.append(agg.reset_index())
    return pd.concat(rows, ignore_index=True)


def summarize(items: pd.DataFrame, variants: pd.DataFrame,
              flow: pd.DataFrame, coverage: pd.DataFrame) -> None:
    section("SAMPLE FLOW (main specification, cumulative)")
    show(flow)

    section("ITEMS PER ROLE AND VARIANT")
    show(variant_counts(items, variants))

    section("WHY ITEMS DROP OUT (main specification)")
    tab = pd.crosstab(items["role"], variants["drop_reason"])
    order = [c for c in ["passed"] + STEPS if c in tab.columns]
    show(tab[order])

    section("MAIN SAMPLE: PRICE AND LIQUIDITY")
    m = items[variants["main"]]
    desc = m.groupby("role").agg(
        items=("trade_share", "size"),
        median_price=("est_median_price", "median"),
        trade_share=("trade_share", "median"),
        est_returns=("n_est_returns", "median"),
        stattrak=("stattrak", "sum"))
    show(desc.round(2))

    section("EVENT-WINDOW COVERAGE (diagnostic only)")
    worst = coverage.groupby("role")["both"].min().sort_values()
    log.info("Lowest coverage across all windows, per role:")
    show(worst.to_frame("share_with_anchor_and_end").round(3))

    lo = coverage[coverage["both"] < 0.90]
    if len(lo):
        log.info("")
        log.info("Below 90 %% in %d role/window combinations -> see "
                 "01_event_coverage.csv", len(lo))

    section("CHECKS")
    empty = [v for v in C.SAMPLE_VARIANTS if variants[v].sum() == 0]
    log.info("Variants without a single item: %s", empty or "none")
    for role in C.TREATED_ROLES + C.PLACEBO_ROLES:
        n = int(variants.loc[items["role"] == role, "main"].sum())
        flag = "  <-- thin" if 0 < n < 20 else ""
        log.info("  %-18s %4d items in main%s", role, n, flag)
    n_idx = int((items["group"].isin(C.INDEX_DEFS[C.INDEX_MAIN])
                 & variants["main"]).sum())
    log.info("Main benchmark '%s': %d constituents", C.INDEX_MAIN, n_idx)


def main() -> None:
    setup_logging()
    C.OUT_TABLES.mkdir(parents=True, exist_ok=True)
    for path in (C.PANEL_CLEAN, C.ITEMS_CLEAN):
        if not path.exists():
            raise FileNotFoundError(f"Not found: {path}. Run 00_prepare.py first.")

    log.info("T0 = %s | estimation window t = %s | liquidity main >= %.0f %%",
             C.T0.date(), C.EST_WINDOW, 100 * C.LIQ_MAIN_MIN)

    panel = pd.read_parquet(C.PANEL_CLEAN,
                            columns=["market_hash_name", "t", "price", "ret_obs", "gap"])
    items = pd.read_parquet(C.ITEMS_CLEAN).set_index("market_hash_name")
    log.info("Loaded: %d items, %d item-days", len(items), len(panel))

    section("1. ADDITIONAL METRICS")
    items = add_return_counts(panel, items)
    items = add_flags(items)

    section("2. SAMPLES")
    variants = build_variants(items)
    log.info("Main specification: %d of %d items", int(variants["main"].sum()), len(items))

    section("3. TABLES")
    flow = sample_flow(items)
    flow.to_csv(C.OUT_TABLES / "01_sample_flow.csv", encoding="utf-8")
    variant_counts(items, variants).to_csv(
        C.OUT_TABLES / "01_sample_variants.csv", encoding="utf-8")
    index_constituents(items, variants).to_csv(
        C.OUT_TABLES / "01_index_constituents.csv", index=False, encoding="utf-8")
    coverage = event_coverage(panel, items, variants)
    coverage.to_csv(C.OUT_TABLES / "01_event_coverage.csv", index=False, encoding="utf-8")

    section("4. SAVE")
    keep = ["role", "group", "subclass", "trade_share", "liq_tier",
            "est_median_price", "n_est_returns", "n_est_returns_robust",
            "listed_before_est", "days_before_est", "is_store_capsule",
            "flag_high_price"]
    out = items[keep].join(variants)
    out.reset_index().to_parquet(C.SAMPLES, index=False)
    log.info("-> %s  (%d items, %d variants)", C.SAMPLES.relative_to(C.ROOT),
             len(out), len(C.SAMPLE_VARIANTS))

    summarize(items, variants, flow, coverage)
    log.info("")
    log.info("Done.")


if __name__ == "__main__":
    main()

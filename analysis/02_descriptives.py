# Describes the main sample before the event: sample characteristics, return
# properties, return correlations, data quality and three figures.
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import plotstyle as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("02_descriptives")


def setup_logging() -> None:
    C.OUT_LOGS.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(C.OUT_LOGS / "02_descriptives.log", mode="w", encoding="utf-8")
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


def order_roles(df: pd.DataFrame) -> pd.DataFrame:
    present = [r for r in C.ROLE_ORDER if r in df.index]
    rest = [r for r in df.index if r not in C.ROLE_ORDER]
    return df.loc[present + rest]


# Key figures per role; prices as median and quartiles because they are heavily skewed.
def table1(items: pd.DataFrame, est: pd.DataFrame, by: str = "role") -> pd.DataFrame:
    vol = est.groupby("market_hash_name", observed=True)["volume"].mean()
    items = items.assign(volume_per_day=vol.reindex(items.index))

    keys = [by] if isinstance(by, str) else list(by)
    g = items.groupby(keys, observed=True)

    tab = pd.DataFrame({
        "Items": g.size(),
        "Collections": g["collection"].nunique(),
        "Price median": g["est_median_price"].median(),
        "Price Q1": g["est_median_price"].quantile(0.25),
        "Price Q3": g["est_median_price"].quantile(0.75),
        "Trade share": g["trade_share"].median(),
        "Volume per day": g["volume_per_day"].median(),
        "Est. returns": g["n_est_returns"].median(),
        "StatTrak share": g["stattrak"].mean(),
    })
    return tab.round({"Price median": 2, "Price Q1": 2, "Price Q3": 2,
                      "Trade share": 2, "Volume per day": 1,
                      "Est. returns": 0, "StatTrak share": 2})


# Return moments per item, then the median across items.
def return_properties(rets: pd.DataFrame, roles: pd.Series) -> pd.DataFrame:
    def stats(g: pd.DataFrame) -> pd.Series:
        r = g["ret_obs"]
        ar1 = r.autocorr(lag=1) if len(r) > 10 else np.nan
        return pd.Series({
            "n_returns": len(r),
            "mean_pct": 100 * r.mean(),
            "sd_pct": 100 * r.std(),
            "skewness": r.skew(),
            "excess_kurtosis": r.kurt(),
            "zero_share": (r.abs() < 1e-9).mean(),
            "ar1": ar1,
        })

    per_item = rets.groupby("market_hash_name", observed=True).apply(
        stats, include_groups=False)
    per_item["role"] = roles.reindex(per_item.index)

    g = per_item.groupby("role", observed=True)
    tab = pd.DataFrame({
        "Items": g.size(),
        "Returns (median)": g["n_returns"].median(),
        "Mean % (median)": g["mean_pct"].median(),
        "SD % (median)": g["sd_pct"].median(),
        "Skewness (median)": g["skewness"].median(),
        "Excess kurtosis (median)": g["excess_kurtosis"].median(),
        "Zero returns (share)": g["zero_share"].median(),
        "AR(1) (median)": g["ar1"].median(),
    })
    return tab.round(3)


# Mean pairwise return correlation by how closely two items are related.
def return_correlation(rets: pd.DataFrame, weekly: pd.DataFrame,
                       items: pd.DataFrame) -> pd.DataFrame:
    def pairwise(panel_wide: pd.DataFrame, min_obs: int) -> tuple[np.ndarray, pd.Index]:
        corr = panel_wide.corr(min_periods=min_obs)
        vals = corr.to_numpy().copy()
        np.fill_diagonal(vals, np.nan)
        return vals, corr.columns

    daily, names = pairwise(
        rets.pivot_table(index="t", columns="market_hash_name",
                         values="ret_obs", observed=True), C.MIN_CORR_OVERLAP)
    weekly_vals, names_w = pairwise(weekly, C.MIN_CORR_OVERLAP // 5)
    assert names.equals(names_w), "column order differs between frequencies"

    meta = items.reindex(names)
    same_skin = meta["weapon"].fillna("?") + "|" + meta["finish"].fillna("?")
    has_finish = meta["finish"].notna().to_numpy()

    def matrix(values: pd.Series) -> np.ndarray:
        arr = values.to_numpy()
        return arr[:, None] == arr[None, :]

    same_case = matrix(meta["case"].fillna("?")) & meta["case"].notna().to_numpy()[:, None]
    same_role = matrix(meta["role"].fillna("?"))
    same_item = matrix(same_skin) & has_finish[:, None] & has_finish[None, :]

    cats = {
        "Same skin, different wear": same_item,
        "Same case, different skin": same_case & ~same_item,
        "Same role, different case": same_role & ~same_case & ~same_item,
        "Different role": ~same_role,
    }

    rows = []
    for name, mask in cats.items():
        d = daily[mask & ~np.isnan(daily)]
        w = weekly_vals[mask & ~np.isnan(weekly_vals)]
        rows.append({"Pair type": name, "Pairs": len(d) // 2,
                     "Mean corr (daily)": np.mean(d) if len(d) else np.nan,
                     "Mean corr (5-day)": np.mean(w) if len(w) else np.nan,
                     "Median corr (5-day)": np.median(w) if len(w) else np.nan})
    return pd.DataFrame(rows).round(3)


# Non-overlapping 5-day returns on the last known price; blocks without a trade are dropped,
# and so is the incomplete last block of the estimation window.
def weekly_returns(est: pd.DataFrame) -> pd.DataFrame:
    sub = est.loc[est["price_ffill"].notna(), ["market_hash_name", "t", "price_ffill"]].copy()
    sub["block"] = (sub["t"] - C.EST_WINDOW[0]) // 5
    n_full = (C.EST_WINDOW[1] - C.EST_WINDOW[0] + 1) // 5
    sub = sub[sub["block"] < n_full]
    last = (sub.sort_values("t")
               .groupby(["market_hash_name", "block"], observed=True)["price_ffill"].last()
               .unstack(0))
    rets = np.log(last).diff()
    return rets.replace(0.0, np.nan)


def data_quality(panel: pd.DataFrame, items: pd.DataFrame,
                 items_all: pd.DataFrame, est: pd.DataFrame) -> pd.DataFrame:
    n_days = len(est)
    rows = [
        ("Items in the main sample", len(items)),
        ("Items with estimation-window data (before screening)",
         int((items_all["est_trade_days"] > 0).sum())),
        ("Item-days in the estimation window", n_days),
        ("Item-days with a trade", int(est["traded"].sum())),
        ("Share of days without a trade", round(1 - est["traded"].mean(), 3)),
        ("90th percentile of the gap between trades in days "
         "(median across items)",
         round(items["p90_gap"].median(), 1)),
        ("Spike days flagged (data window)", int(items["n_spikes_flagged"].sum())),
        ("Spike days removed (volume <= %d, data window)" % C.SPIKE_MAX_VOLUME,
         int(items["n_spikes_removed"].sum())),
        ("Item-days at the %.2f EUR price cluster (data window)" % C.MASS_POINT_EUR,
         int(panel.loc[panel["market_hash_name"].isin(items.index),
                       "flag_mass_point"].sum())),
        ("Items flagged as high-price (all items with estimation data)",
         int(items_all["flag_high_price"].sum())),
        ("Items below the price floor (all items with estimation data)",
         int(items_all["flag_below_min_price"].sum())),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def fig_price_distribution(items: pd.DataFrame) -> None:
    roles = [r for r in C.ROLE_ORDER if r in set(items["role"])]
    data = [items.loc[items["role"] == r, "est_median_price"].dropna() for r in roles]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    # No vert=: deprecated in newer Matplotlib, vertical is the default.
    bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                    medianprops=dict(color="white", linewidth=1.5),
                    flierprops=dict(marker=".", markersize=3,
                                    markerfacecolor=ps.NEUTRAL,
                                    markeredgecolor="none", alpha=0.4))
    for patch, role in zip(bp["boxes"], roles):
        patch.set_facecolor(ps.color(role))
        patch.set_edgecolor(ps.NEUTRAL)
        patch.set_alpha(0.85)

    ax.set_yscale("log")
    ax.axhline(C.MIN_PRICE_EUR, color=ps.NEUTRAL, linestyle=":", linewidth=1)
    ax.text(0.4, C.MIN_PRICE_EUR * 1.1, f"screening floor {C.MIN_PRICE_EUR:.2f} EUR",
            fontsize=8, color=ps.NEUTRAL)
    ax.set_xticks(range(1, len(roles) + 1))
    ax.set_xticklabels([f"{ps.short(r)}\nn = {len(d)}" for r, d in zip(roles, data)],
                       fontsize=7.5)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_ylabel("Median price in the estimation window (EUR, log scale)")
    ps.save(fig, "02_fig1_price_distribution", C.OUT_FIGURES)


# Trade share before screening, all items with estimation data (appendix figure).
def fig_liquidity_distribution(items_all: pd.DataFrame) -> None:
    roles = [r for r in C.ROLE_ORDER
             if r in set(items_all["role"]) and r not in ("excluded",)]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for role in roles:
        x = np.sort(items_all.loc[items_all["role"] == role, "trade_share"].dropna())
        if len(x) < 5:
            continue
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", color=ps.color(role), label=ps.label(role),
                linestyle=ps.linestyle(role))

    ax.axvline(C.LIQ_MAIN_MIN, color=ps.NEUTRAL, linestyle="--", linewidth=1)
    ax.text(C.LIQ_MAIN_MIN + 0.01, 0.05,
            f"main sample:\n\u2265 {C.LIQ_MAIN_MIN:.0%}", fontsize=8,
            color=ps.NEUTRAL)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Trade share in the estimation window")
    ax.set_ylabel("Share of items at or below")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    ps.save(fig, "02_fig2_liquidity_distribution", C.OUT_FIGURES)


# Per role: the item closest to the median price among the more liquid half.
def pick_examples(items: pd.DataFrame) -> list[str]:
    if C.EXAMPLE_ITEMS:
        return [n for n in C.EXAMPLE_ITEMS if n in items.index]

    picks = []
    for role in C.EXAMPLE_ROLES:
        sub = items[items["role"] == role]
        if sub.empty:
            continue
        target = sub["est_median_price"].median()
        sub = sub[sub["trade_share"] >= sub["trade_share"].median()]
        picks.append((sub["est_median_price"] - target).abs().idxmin())
    return picks


def fig_example_series(panel: pd.DataFrame, items: pd.DataFrame,
                       names: list[str]) -> None:
    lo, hi = C.EST_WINDOW
    fig, axes = plt.subplots(len(names), 1, figsize=(7.5, 2.2 * len(names)),
                             sharex=True)
    axes = np.atleast_1d(axes)

    for ax, name in zip(axes, names):
        sub = panel[(panel["market_hash_name"] == name)
                    & panel["t"].between(lo, hi)]
        role = items.loc[name, "role"]
        col = ps.color(role)

        ax.plot(sub["t"], sub["price_ffill"], color=col, linewidth=1.0, alpha=0.45)
        traded = sub[sub["traded"]]
        ax.plot(traded["t"], traded["price"], color=col, linewidth=0,
                marker=".", markersize=3)

        vol = ax.twinx()
        vol.bar(traded["t"], traded["volume"], color=ps.GRID, width=1.0, zorder=0)
        vol.set_ylim(0, max(traded["volume"].max() * 4, 1))
        vol.set_yticks([])
        vol.grid(False)
        ax.set_zorder(vol.get_zorder() + 1)
        ax.patch.set_visible(False)

        share = items.loc[name, "trade_share"]
        ax.set_title(f"{name}  --  {ps.label(role)}, traded on {share:.0%} of days",
                     loc="left", fontsize=9)
        ax.set_ylabel("EUR")

    axes[-1].set_xlabel("Event time t (days relative to October 23, 2025)")
    fig.tight_layout()
    ps.save(fig, "02_fig3_example_series", C.OUT_FIGURES)


def main() -> None:
    setup_logging()
    ps.apply_style()
    C.OUT_TABLES.mkdir(parents=True, exist_ok=True)
    if not C.SAMPLES.exists():
        raise FileNotFoundError(f"{C.SAMPLES} missing. Run 01_screening.py first.")

    panel = pd.read_parquet(C.PANEL_CLEAN,
                            columns=["market_hash_name", "t", "price", "price_ffill",
                                     "volume", "traded", "ret_obs", "gap",
                                     "flag_mass_point"])
    items_all = pd.read_parquet(C.ITEMS_CLEAN).set_index("market_hash_name")
    samples = pd.read_parquet(C.SAMPLES).set_index("market_hash_name")

    items = items_all.loc[samples.index[samples["main"]]].copy()
    items["n_est_returns"] = samples.loc[items.index, "n_est_returns"]
    log.info("Main sample: %d items | estimation window t = %s", len(items), C.EST_WINDOW)

    lo, hi = C.EST_WINDOW
    in_est = panel["t"].between(lo, hi)
    est = panel[in_est & panel["market_hash_name"].isin(items.index)]
    rets = est[est["ret_obs"].notna() & (est["gap"] <= C.MAX_GAP_RET)]
    log.info("Estimation window: %d item-days, %d usable returns", len(est), len(rets))

    section("SAMPLE CHARACTERISTICS (estimation window; basis of thesis Table 2)")
    t1 = order_roles(table1(items, est))
    show(t1)
    t1.to_csv(C.OUT_TABLES / "02_table1_sample_characteristics.csv", encoding="utf-8")

    t1b = table1(items, est, by=["role", "liq_tier"])
    t1b.to_csv(C.OUT_TABLES / "02_table1_by_liquidity.csv", encoding="utf-8")

    section("RETURN PROPERTIES (daily log returns, trade to trade; thesis Table C1)")
    t3 = order_roles(return_properties(rets, items["role"]))
    show(t3)
    t3.to_csv(C.OUT_TABLES / "02_table3_return_properties.csv", encoding="utf-8")

    section("CROSS-SECTIONAL CORRELATION OF RETURNS")
    corr = return_correlation(rets, weekly_returns(est), items)
    show(corr)
    corr.to_csv(C.OUT_TABLES / "02_return_correlation.csv", index=False, encoding="utf-8")

    section("DATA QUALITY (appendix)")
    dq = data_quality(panel, items, items_all, est)
    show(dq)
    dq.to_csv(C.OUT_TABLES / "02_data_quality.csv", index=False, encoding="utf-8")

    section("FIGURES")
    fig_price_distribution(items)
    log.info("02_fig1_price_distribution")
    fig_liquidity_distribution(items_all[items_all["est_trade_days"] > 0])
    log.info("02_fig2_liquidity_distribution")
    names = pick_examples(items)
    fig_example_series(panel, items, names)
    log.info("02_fig3_example_series: %s", " | ".join(names))

    log.info("")
    log.info("Done. Tables in %s, figures in %s",
             C.OUT_TABLES.relative_to(C.ROOT), C.OUT_FIGURES.relative_to(C.ROOT))


if __name__ == "__main__":
    main()

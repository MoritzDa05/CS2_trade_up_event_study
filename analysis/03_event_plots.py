# Model-free event paths: every item is normalised to its last price on or before
# NORM_DAY, then the median per role. No tests here (see 04).
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

log = logging.getLogger("03_event_plots")

XLABEL = "Event time t (days relative to October 23, 2025)"
# Typographic minus (U+2212), as in matplotlib's tick labels.
MINUS = "−"


def setup_logging() -> None:
    C.OUT_LOGS.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(C.OUT_LOGS / "03_event_plots.log", mode="w", encoding="utf-8")
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


# Log return since the last price on or before NORM_DAY; items without one are dropped.
def normalise(panel: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    sub = panel[panel["market_hash_name"].isin(items.index)].copy()

    before_event = (sub["t"] <= C.NORM_DAY) & sub["price"].notna()
    anchor = (sub.loc[before_event]
              .sort_values("t")
              .groupby("market_hash_name", observed=True)["price"].last())

    missing = items.index.difference(anchor.dropna().index)
    if len(missing):
        log.warning("%d items have no observed price on or before t = %d and "
                    "are not in the paths", len(missing), C.NORM_DAY)

    # astype(float): mapping a categorical column can return a categorical under pandas 3.
    sub["anchor"] = sub["market_hash_name"].map(anchor).astype(float)
    sub = sub[sub["anchor"].notna() & (sub["anchor"] > 0)]

    sub["cum_ffill"] = np.log(sub["price_ffill"] / sub["anchor"])
    sub["cum_traded"] = np.log(sub["price"] / sub["anchor"])
    sub["role"] = sub["market_hash_name"].map(items["role"])
    return sub


def aggregate(paths: pd.DataFrame) -> pd.DataFrame:
    g = paths.groupby(["role", "t"], observed=True)
    out = pd.DataFrame({
        "n_items": g["cum_ffill"].count(),
        "median": g["cum_ffill"].median(),
        "mean": g["cum_ffill"].mean(),
        "q1": g["cum_ffill"].quantile(0.25),
        "q3": g["cum_ffill"].quantile(0.75),
        "n_traded": g["cum_traded"].count(),
        "median_traded": g["cum_traded"].median(),
        "volume": g["volume"].sum(),
    }).reset_index()

    # Group volume vs. the estimation window; CSV and log only (08 has the per-item measure).
    base = (out[out["t"].between(*C.EST_WINDOW)]
            .groupby("role", observed=True)["volume"].median())
    out["volume_ratio"] = out["volume"] / out["role"].map(base)
    return out


def to_pct(x: pd.Series | np.ndarray) -> np.ndarray:
    return (np.exp(np.asarray(x, dtype=float)) - 1) * 100


# Raw (not abnormal) return per window; window "0" is re-anchored at t = -1.
def window_table(paths: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, (lo, hi) in C.EVENT_WINDOWS.items():
        day = paths[paths["t"] == hi]
        if lo - 1 > C.NORM_DAY:
            anchor = (paths[paths["t"] == lo - 1]
                      .set_index("market_hash_name")["cum_ffill"])
            shift = day["market_hash_name"].map(anchor).to_numpy()
            day = day.assign(cum_ffill=day["cum_ffill"].to_numpy() - shift,
                             cum_traded=day["cum_traded"].to_numpy() - shift)
        g = day.groupby("role", observed=True)
        rows.append(pd.DataFrame({
            "window": label,
            "start_day": lo,
            "end_day": hi,
            "n_items": g["cum_ffill"].count(),
            "median_pct": to_pct(g["cum_ffill"].median()),
            "mean_pct": to_pct(g["cum_ffill"].mean()),
            "q1_pct": to_pct(g["cum_ffill"].quantile(0.25)),
            "q3_pct": to_pct(g["cum_ffill"].quantile(0.75)),
            "n_traded_that_day": g["cum_traded"].count(),
        }).reset_index())
    out = pd.concat(rows, ignore_index=True)
    return out.round(2)


# The y-axis follows the lines, not the quartile band, which would squash the medians.
def draw_paths(ax, agg: pd.DataFrame, roles: list[str], window: tuple[int, int],
               value: str = "median", bands: bool = True,
               traded_only: bool = True) -> None:
    lo, hi = window
    span = []
    for role in roles:
        sub = agg[(agg["role"] == role) & agg["t"].between(lo, hi)].sort_values("t")
        if sub.empty:
            continue
        col = ps.color(role)
        ax.plot(sub["t"], to_pct(sub[value]), color=col, label=ps.label(role),
                linestyle=ps.linestyle(role))
        span.append(to_pct(sub[value]))

        if bands and role in C.PLOT_BAND_ROLES:
            ax.fill_between(sub["t"], to_pct(sub["q1"]), to_pct(sub["q3"]),
                            color=col, alpha=0.12, linewidth=0)

        if traded_only and role in C.PLOT_TRADED_ONLY_ROLES:
            ax.plot(sub["t"], to_pct(sub["median_traded"]), color=col,
                    linestyle=":", linewidth=1.0, alpha=0.9)

    if span:
        flat = np.concatenate(span)
        lo_y, hi_y = np.nanmin(flat), np.nanmax(flat)
        pad = max(5.0, 0.15 * (hi_y - lo_y))
        ax.set_ylim(lo_y - pad, hi_y + pad)

    ax.axvline(0, color=ps.NEUTRAL, linewidth=1)
    ax.axhline(0, color=ps.GRID, linewidth=1)
    norm_day = str(C.NORM_DAY).replace("-", MINUS)
    ax.set_ylabel(f"Cumulative return since t = {norm_day} (%)")


# Legend below the axes; its space is reserved first.
def legend_below(fig, ax, handles=None, labels=None) -> None:
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    rows = -(-len(handles) // 3)  # ceil(n / 3)
    fig.tight_layout(rect=(0, 0.045 * rows, 1, 1))
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=3)


def fig_short(agg: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    draw_paths(ax, agg, C.PLOT_ROLES_MAIN, C.PLOT_WINDOW_SHORT)
    ax.set_xlabel(XLABEL)

    handles, labels = ax.get_legend_handles_labels()
    proxy = plt.Line2D([], [], color=ps.NEUTRAL, linestyle=":", linewidth=1.0)
    # Legend fills by column: the dotted-line entry goes right below output and input.
    k = len(C.PLOT_TRADED_ONLY_ROLES)
    handles = handles[:k] + [proxy] + handles[k:]
    labels = labels[:k] + ["Items traded that day"] + labels[k:]
    legend_below(fig, ax, handles, labels)
    ps.save(fig, "03_fig1_event_path_short", C.OUT_FIGURES)


def fig_long(agg: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    draw_paths(ax, agg, C.PLOT_ROLES_MAIN, C.PLOT_WINDOW_LONG,
               bands=False, traded_only=False)
    ax.axvspan(*C.EST_WINDOW, color=ps.GRID, alpha=0.35, linewidth=0)
    ax.text(np.mean(C.EST_WINDOW), ax.get_ylim()[1], "estimation window",
            ha="center", va="top", fontsize=8, color=ps.NEUTRAL)
    ax.set_xlabel(XLABEL)
    legend_below(fig, ax)
    ps.save(fig, "03_fig2_event_path_long", C.OUT_FIGURES)


def fig_mean(agg: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    draw_paths(ax, agg, C.PLOT_ROLES_MAIN, C.PLOT_WINDOW_SHORT,
               value="mean", bands=False, traded_only=False)
    ax.set_xlabel(XLABEL)
    legend_below(fig, ax)
    ps.save(fig, "03_fig3_event_path_mean", C.OUT_FIGURES)


def fig_other(agg: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    draw_paths(ax, agg, C.PLOT_ROLES_APPENDIX, C.PLOT_WINDOW_SHORT,
               bands=False, traded_only=False)
    ax.set_xlabel(XLABEL)
    legend_below(fig, ax)
    ps.save(fig, "03_fig4_event_path_other_roles", C.OUT_FIGURES)


def main() -> None:
    setup_logging()
    ps.apply_style()
    C.OUT_TABLES.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(C.PANEL_CLEAN,
                            columns=["market_hash_name", "t", "price",
                                     "price_ffill", "volume"])
    items = pd.read_parquet(C.ITEMS_CLEAN).set_index("market_hash_name")
    samples = pd.read_parquet(C.SAMPLES).set_index("market_hash_name")
    items = items.loc[samples.index[samples["main"]]]
    log.info("Main sample: %d items | normalised to t = %d", len(items), C.NORM_DAY)

    paths = normalise(panel, items)
    agg = aggregate(paths)
    agg.to_csv(C.OUT_TABLES / "03_event_paths.csv", index=False, encoding="utf-8")

    section("RAW CUMULATIVE RETURNS BY EVENT WINDOW (median, %)")
    windows = window_table(paths)
    pivot = windows.pivot(index="role", columns="window", values="median_pct")
    pivot = pivot[[w for w in C.EVENT_WINDOWS if w in pivot.columns]]
    show(pivot.reindex([r for r in C.ROLE_ORDER if r in pivot.index]))
    windows.to_csv(C.OUT_TABLES / "03_raw_event_windows.csv", index=False,
                   encoding="utf-8")

    section("EVENT DAY DETAIL (t = -1 to +5, median in %)")
    detail = agg[agg["t"].between(-1, 5)].pivot(index="t", columns="role",
                                                values="median")
    detail = detail[[r for r in C.PLOT_ROLES_MAIN if r in detail.columns]]
    show(detail.apply(to_pct).round(1))

    vol = agg[agg["t"].between(-1, 5)].pivot(index="t", columns="role",
                                             values="volume_ratio")
    log.info("")
    log.info("Volume as a multiple of the estimation-window level "
             "(group aggregate, not plotted):")
    show(vol[[r for r in C.PLOT_ROLES_MAIN if r in vol.columns]].round(1))

    section("STALENESS CHECK (median, %)")
    st = agg[agg["t"].isin([0, 1, 2, 5, 10])]
    st = st[st["role"].isin(C.PLOT_TRADED_ONLY_ROLES)]
    st = pd.DataFrame({
        "role": st["role"], "t": st["t"],
        "last_known_price": to_pct(st["median"]).round(1),
        "traded_only": to_pct(st["median_traded"]).round(1),
        "items_traded": st["n_traded"], "items_total": st["n_items"],
    })
    show(st.set_index(["role", "t"]))

    section("FIGURES")
    fig_short(agg)
    fig_long(agg)
    fig_mean(agg)
    fig_other(agg)
    for name in ["03_fig1_event_path_short", "03_fig2_event_path_long",
                 "03_fig3_event_path_mean", "03_fig4_event_path_other_roles"]:
        log.info(name)

    log.info("")
    log.info("Done. Tables in %s, figures in %s",
             C.OUT_TABLES.relative_to(C.ROOT), C.OUT_FIGURES.relative_to(C.ROOT))


if __name__ == "__main__":
    main()

# Volume: abnormal log volume with BMP, KP and rank tests (also relative to the control
# group), the long-run regime, the echo of the seven-day hold and the unit balance.
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import plotstyle as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("08_volume")

# Control group = Placebo Map + Placebo Classified (term of the thesis).
GROUP_LABELS = {"control": "Control group"}
GROUP_SHORT = {"control": "Control\ngroup"}
GROUP_COLORS = {"control": ps.color("placebo")}
TEST_GROUPS = ["treated_output", "treated_input", "control", "placebo_st",
               "index_main", "index_alt", "case"]
ECHO_GROUPS = ["control", "placebo_st", "treated_input", "treated_output"]


def glabel(g: str) -> str:
    return GROUP_LABELS.get(g, ps.label(g))


def gcolor(g: str) -> str:
    return GROUP_COLORS.get(g, ps.color(g))


def gshort(g: str) -> str:
    return GROUP_SHORT.get(g, ps.short(g))


def setup_logging() -> None:
    C.OUT_LOGS.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(C.OUT_LOGS / "08_volume.log", mode="w",
                             encoding="utf-8")
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


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    for level, mark in C.SIG_LEVELS:
        if p < level:
            return mark
    return ""


# Items of all groups, daily units and EUR turnover (t x item), median knife price path.
def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    samples = pd.read_parquet(C.SAMPLES)
    items = pd.read_parquet(C.ITEMS_CLEAN)[["market_hash_name", "skin_id"]]
    role_to_group = {r: g for g, roles in C.VOL_GROUPS.items() for r in roles}
    d = samples[["market_hash_name", "role", "main"]].merge(
        items, on="market_hash_name", how="left")
    d["group"] = d["role"].map(role_to_group)
    d = d.dropna(subset=["group"])
    d["cluster"] = d["skin_id"].astype("string").fillna(d["market_hash_name"])
    d = d.set_index("market_hash_name")

    panel = pd.read_parquet(C.PANEL_CLEAN, columns=[
        "market_hash_name", "t", "volume", "price_raw", "price_ffill"])
    panel["market_hash_name"] = panel["market_hash_name"].astype(str)
    panel = panel[panel["market_hash_name"].isin(d.index)]
    days = range(C.WINDOW[0], C.WINDOW[1] + 1)
    units = (panel.pivot(index="t", columns="market_hash_name", values="volume")
             .reindex(days).fillna(0.0))
    # Raw median for turnover: spikes are removed only for returns; the units were sold.
    eur = units * panel.pivot(index="t", columns="market_hash_name",
                              values="price_raw").reindex(days).fillna(0.0)

    knives = d.index[(d["group"] == "treated_output") & d["main"]]
    pf = np.log(panel[panel["market_hash_name"].isin(knives)].pivot(
        index="t", columns="market_hash_name", values="price_ffill").reindex(days))
    pre = pf.loc[C.LEVEL_PRE[0]:C.LEVEL_PRE[1]].median()
    knife_price = (pf - pre).median(axis=1)
    return d, units, eur, knife_price


# AV = log(c + units) - estimation-window mean of the item (Campbell and Wasley 1996).
def abnormal(level: pd.DataFrame) -> pd.DataFrame:
    lv = np.log(C.VOLUME_LOG_CONST + level)
    return lv - lv.loc[C.EST_WINDOW[0]:C.EST_WINDOW[1]].mean()


# Variance of an L-day mean from the autocovariances up to VOLUME_ACOV_LAGS
# (weights 1 - k/L); gamma0 / L where the sum is not positive.
def var_of_mean(x: np.ndarray, L: int) -> np.ndarray:
    xc = x - np.nanmean(x, axis=0)
    g0 = np.nanmean(xc ** 2, axis=0)
    v = g0.copy()
    for k in range(1, min(L - 1, C.VOLUME_ACOV_LAGS) + 1):
        gk = np.nanmean(xc[k:] * xc[:-k], axis=0)
        v = v + 2 * (1 - k / L) * gk
    v = np.where(v > 0, v, g0)
    return v / L


# Mean pairwise correlation of the columns (KP r-bar).
def mean_correlation(x: np.ndarray) -> float:
    sd = np.nanstd(x, axis=0)
    x = x[:, sd > 0]
    if x.shape[1] < 3:
        return np.nan
    corr = np.corrcoef(x, rowvar=False)
    np.fill_diagonal(corr, np.nan)
    return float(np.nanmean(corr))


# KP r-bar for an L-day window: mean correlation of non-overlapping L-day block means of the
# estimation window; at most T / VOLUME_KP_MIN_BLOCKS days per block (longer windows).
def block_rbar(est: np.ndarray, L: int) -> tuple[float, int]:
    block = max(1, min(L, est.shape[0] // C.VOLUME_KP_MIN_BLOCKS))
    nb = est.shape[0] // block
    means = est[: nb * block].reshape(nb, block, -1).mean(axis=1)
    return mean_correlation(means), block


# BMP, KP and rank test for one group, window and reference (av: days x items).
def window_tests(av: pd.DataFrame, window: tuple[int, int]) -> dict:
    lo, hi = window
    L = hi - lo + 1
    est = av.loc[C.EST_WINDOW[0]:C.EST_WINDOW[1]].to_numpy(float)
    m = av.loc[lo:hi].mean()
    n = int(m.notna().sum())
    # exp(mean AV): index of log(1 + units), log diagnostic only (not a multiple of units).
    out = {"n": n, "mean_log": m.mean(), "multiple_av": np.exp(m.mean())}
    if n < 3:
        return out

    sav = m / np.sqrt(var_of_mean(est, L))
    t_bmp = sav.mean() / (sav.std(ddof=1) / np.sqrt(n))
    out["t_bmp"] = t_bmp
    out["p_bmp"] = 2 * stats.t.sf(abs(t_bmp), n - 1)
    rbar, block = block_rbar(est, L)
    out["rbar"], out["rbar_block_days"] = rbar, block
    if np.isfinite(rbar) and rbar > 0:
        t_kp = t_bmp * np.sqrt((1 - rbar) / (1 + (n - 1) * rbar))
        out["t_kp"] = t_kp
        out["p_kp"] = 2 * stats.t.sf(abs(t_kp), n - 1)

    # Rank test: ranks within item over [EST_WINDOW[0], window end], U = rank / (M + 1) - 0.5.
    sub = av.loc[C.EST_WINDOW[0]:hi]
    ranks = sub.rank(axis=0) / (sub.count(axis=0) + 1) - 0.5
    K = ranks.mean(axis=1)
    k_est = K.loc[C.EST_WINDOW[0]:C.EST_WINDOW[1]].to_numpy(float)[:, None]
    z = K.loc[lo:hi].mean() / np.sqrt(var_of_mean(k_est, L)[0])
    out["z_rank"] = z
    out["p_rank"] = 2 * stats.norm.sf(abs(z))
    return out


# Descriptive multiple of normal volume in units: median over items of mean daily units
# in the window over mean daily units in the estimation window.
def units_multiple(units: pd.DataFrame, idx: pd.Index, window: tuple[int, int]) -> float:
    lo, hi = window
    est = units.loc[C.EST_WINDOW[0]:C.EST_WINDOW[1], idx].mean()
    ratio = (units.loc[lo:hi, idx].mean() / est).replace([np.inf, -np.inf], np.nan)
    return float(ratio.median())


# Every group x window x reference; a reference's daily mean AV is subtracted (DiD).
# multiple: units multiple of the group, over that of the reference (descriptive).
def test_table(av: pd.DataFrame, units: pd.DataFrame,
               members: dict[str, pd.Index]) -> pd.DataFrame:
    rows = []
    refs = [None] + C.VOLUME_REFERENCES
    mult = {(g, label): units_multiple(units, members[g], window)
            for label, window in C.VOLUME_WINDOWS.items()
            for g in set(TEST_GROUPS) | set(C.VOLUME_REFERENCES)}
    for ref in refs:
        adj = av if ref is None else av.sub(av[members[ref]].mean(axis=1), axis=0)
        for label, window in C.VOLUME_WINDOWS.items():
            for g in TEST_GROUPS:
                if g == ref:
                    continue
                st = window_tests(adj[members[g]], window)
                multiple = mult[(g, label)] / (mult[(ref, label)] if ref else 1.0)
                rows.append({"reference": ref or "none",
                             "window": label, "group": g, "multiple": multiple, **st})
    return pd.DataFrame(rows)


# Units above the item's estimation-window daily mean.
def excess_units(units: pd.DataFrame) -> pd.DataFrame:
    return units - units.loc[C.EST_WINDOW[0]:C.EST_WINDOW[1]].mean()


# Per item: wave, echo (ECHO_WINDOW minus ECHO_PRE) and the unit-balance windows.
def echo_parts(ex: pd.DataFrame) -> pd.DataFrame:
    s = lambda w: ex.loc[w[0]:w[1]].sum()  # noqa: E731
    return pd.DataFrame({"wave": s(C.ECHO_WAVE),
                         "echo": s(C.ECHO_WINDOW) - s(C.ECHO_PRE),
                         "buy": s(C.BALANCE_BUY), "sell": s(C.BALANCE_SELL)})


# Column sums over a group; with rng over a resample of its skins.
def draw_sums(parts: pd.DataFrame, clusters: pd.Series,
              rng: np.random.Generator | None) -> pd.Series:
    if rng is None:
        return parts.sum()
    cl = clusters.loc[parts.index].to_numpy()
    codes, pos = np.unique(cl, return_inverse=True)
    X = parts.to_numpy(float)
    grouped = np.zeros((len(codes), X.shape[1]))
    np.add.at(grouped, pos, X)
    pick = rng.integers(0, len(codes), len(codes))
    return pd.Series(grouped[pick].sum(axis=0), index=parts.columns)


# Echo share per group and Covert units per knife unit.
def balance(sums: dict[str, pd.Series]) -> dict:
    share = {g: sums[g]["echo"] / sums[g]["wave"] for g in sums}
    return {"share": share, "ratio": sums["treated_input"]["buy"]
            / sums["treated_output"]["sell"]}


# Days 30-190 vs. estimation window, per item: log ratio of units, EUR and price
# (EUR = units x price); mean over items, CI and difference to control by skin.
def long_run(units: pd.DataFrame, eur: pd.DataFrame, members: dict,
             clusters: pd.Series) -> pd.DataFrame:
    lo, hi = C.VOLUME_WINDOWS["+30_+190"]
    est = slice(*C.EST_WINDOW)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.log(units.loc[lo:hi].mean() / units.loc[est].mean())
        e = np.log(eur.loc[lo:hi].mean() / eur.loc[est].mean())
    ratios = pd.DataFrame({"units": u, "eur": e, "price": e - u})
    ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()

    rng = np.random.default_rng(C.BOOTSTRAP_SEED)
    boot = {}
    for g in TEST_GROUPS:
        idx = members[g].intersection(ratios.index)
        X = ratios.loc[idx].to_numpy(float)
        codes, pos = np.unique(clusters.loc[idx].to_numpy(), return_inverse=True)
        rows = [np.flatnonzero(pos == i) for i in range(len(codes))]
        draws = np.empty((C.BOOTSTRAP_REPS, 3))
        for r in range(C.BOOTSTRAP_REPS):
            pick = np.concatenate([rows[i] for i in
                                   rng.integers(0, len(rows), len(rows))])
            draws[r] = X[pick].mean(axis=0)
        boot[g] = (len(idx), X.mean(axis=0), draws)

    out = []
    for g in TEST_GROUPS:
        n, est_, draws = boot[g]
        row = {"group": g, "items": n}
        for j, name in enumerate(["units", "eur", "price"]):
            lo_, hi_ = ci(draws[:, j])
            row[f"{name}_multiple"] = np.exp(est_[j])
            row[f"{name}_lo"], row[f"{name}_hi"] = np.exp(lo_), np.exp(hi_)
            if g != C.VOL_CONTROL:
                d = draws[:, j] - boot[C.VOL_CONTROL][2][:, j]
                row[f"{name}_vs_control"] = np.exp(est_[j] - boot[C.VOL_CONTROL][1][j])
                row[f"{name}_vs_control_p"] = p_two_sided(d)
        out.append(row)
    return pd.DataFrame(out).set_index("group")


def ci(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    return tuple(np.percentile(x, [2.5, 97.5])) if len(x) else (np.nan, np.nan)


def p_two_sided(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return min(1.0, 2 * min((x <= 0).mean(), (x >= 0).mean())) if len(x) else np.nan


def event_lines(ax, label_hold: bool = False) -> None:
    ax.axvline(0, color=ps.NEUTRAL, linewidth=1)
    ax.axvline(C.TRADE_HOLD_DAYS, color=ps.NEUTRAL, linewidth=0.8, linestyle=":")
    if label_hold:
        ax.text(C.TRADE_HOLD_DAYS + 0.7, 0.97, f"{C.TRADE_HOLD_DAYS}-day hold ends",
                transform=ax.get_xaxis_transform(), fontsize=8,
                color=ps.NEUTRAL, va="top")


def plain_log(axis) -> None:
    from matplotlib.ticker import FuncFormatter, NullFormatter
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def fig_volume_path(path: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for g in ["treated_output", "treated_input", "placebo_st", "control"]:
        ax.plot(path.index, path[g], color=gcolor(g), label=glabel(g))
    event_lines(ax, label_hold=True)
    ax.axhline(1, color=ps.NEUTRAL, linewidth=0.8)
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 5, 10, 20, 50])
    plain_log(ax.yaxis)
    ax.set_xlabel("Event time t (days relative to October 23, 2025)")
    ax.set_ylabel("Volume, multiple of normal\n(typical item)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    ps.save(fig, "08_fig1_volume_path", C.OUT_FIGURES)


def fig_echo(echo: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    order = [g for g in ECHO_GROUPS if g in echo.index]
    x = np.arange(len(order))
    for i, g in enumerate(order):
        r = echo.loc[g]
        ax.bar(i, 100 * r["share"], color=gcolor(g), width=0.6)
        ax.errorbar(i, 100 * r["share"],
                    yerr=[[100 * (r["share"] - r["share_lo"])],
                          [100 * (r["share_hi"] - r["share"])]],
                    color=ps.NEUTRAL, capsize=3)
    ax.axhline(0, color=ps.NEUTRAL, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([gshort(g) for g in order], fontsize=8.5)
    ax.set_ylabel("Units back after 7 days\n(% of the first wave, 95% CI)")
    fig.tight_layout()
    ps.save(fig, "08_fig2_echo", C.OUT_FIGURES)


def fig_knives(path: pd.DataFrame, price: pd.Series) -> None:
    lo, hi = -5, C.VOLUME_PATH[1]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    v = path.loc[lo:hi, "treated_output"]
    ax.bar(v.index, v, color=ps.color("treated_output"), alpha=0.35, width=0.9,
           label="Volume, multiple of normal (left)")
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20])
    plain_log(ax.yaxis)
    ax.set_ylabel("Volume, multiple of normal")
    ax2 = ax.twinx()
    pr = 100 * (np.exp(price.loc[lo:hi]) - 1)
    ax2.plot(pr.index, pr, color=ps.NEUTRAL, linewidth=1.6,
             label="Median price change (right)")
    ax2.set_ylabel("Median price change since before the update (%)")
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    event_lines(ax, label_hold=True)
    ax.set_xlabel("Event time t (days relative to October 23, 2025)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    fig.tight_layout()
    ps.save(fig, "08_fig3_knives_volume_price", C.OUT_FIGURES)


def fig_long_run(lr: pd.DataFrame) -> None:
    groups = ["treated_output", "treated_input", "control", "placebo_st",
              "index_alt"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(groups))
    for off, m, alpha, lab in [(-0.27, "units", 1.0, "Units sold"),
                               (0.0, "price", 0.6, "Average price"),
                               (0.27, "eur", 0.3, "Turnover in EUR")]:
        vals = lr.loc[groups, f"{m}_multiple"].to_numpy()
        err = np.vstack([vals - lr.loc[groups, f"{m}_lo"].to_numpy(),
                         lr.loc[groups, f"{m}_hi"].to_numpy() - vals])
        ax.bar(x + off, vals, width=0.25, color=[gcolor(g) for g in groups],
               alpha=alpha, edgecolor=ps.NEUTRAL, linewidth=0.5, label=lab)
        ax.errorbar(x + off, vals, yerr=err, fmt="none", color=ps.NEUTRAL,
                    capsize=2, linewidth=0.8)
    ax.axhline(1, color=ps.NEUTRAL, linewidth=0.8)
    ax.set_yscale("log")
    ax.set_yticks([0.3, 0.5, 1, 2, 3, 5])
    plain_log(ax.yaxis)
    ax.set_xticks(x)
    ax.set_xticklabels([gshort(g) for g in groups], fontsize=8.5)
    ax.set_ylabel("Days 30-190 against before\n(multiple, 95% CI)")
    ax.legend(loc="upper right", fontsize=8,
              title="Left to right, same color", title_fontsize=8)
    fig.tight_layout()
    ps.save(fig, "08_fig4_long_run", C.OUT_FIGURES)


def main() -> None:
    setup_logging()
    ps.apply_style()
    C.OUT_TABLES.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 170)

    section("1. DATA")
    items, units, eur, knife_price = load()
    main_items = items[items["main"]]
    members = {g: main_items.index[main_items["group"] == g] for g in C.VOL_GROUPS}
    everyone = {g: items.index[items["group"] == g] for g in C.VOL_GROUPS}
    zero = (units.loc[C.EST_WINDOW[0]:C.EST_WINDOW[1]] == 0).mean()
    overview = pd.DataFrame({
        "main_items": {g: len(members[g]) for g in C.VOL_GROUPS},
        "all_items": {g: len(everyone[g]) for g in C.VOL_GROUPS},
        "zero_days_pct": {g: 100 * zero[members[g]].mean() for g in C.VOL_GROUPS},
    })
    show(overview.round(1))
    log.info("Measure: log(%g + units), minus the estimation-window mean %s",
             C.VOLUME_LOG_CONST, C.EST_WINDOW)

    av_units = abnormal(units)

    section("2. ABNORMAL VOLUME (tests on log(1 + units); multiple = median ratio "
            "of mean daily units, window over estimation window)")
    tab = test_table(av_units, units, members)
    tab["sig"] = [stars(p) for p in tab["p_kp"].fillna(tab["p_bmp"])]
    cols = ["n", "multiple", "multiple_av", "t_bmp", "t_kp", "p_kp",
            "z_rank", "p_rank", "sig"]
    for ref in ["none"] + C.VOLUME_REFERENCES:
        log.info("")
        log.info("Reference: %s", ref if ref != "none"
                 else "none (own estimation-window mean)")
        sub = tab[tab["reference"] == ref]
        show(sub.set_index(["window", "group"])[cols].round(3))
    log.info("(multiple_av = exp(mean abnormal log volume), diagnostic only)")
    tab.drop(columns=["multiple_av"]).to_csv(
        C.OUT_TABLES / "08_abnormal_volume.csv", index=False, encoding="utf-8")

    section("3. THE NEW REGIME: UNITS, PRICE AND EUR, DAYS 30-190 AGAINST BEFORE")
    lr = long_run(units, eur, members, items["cluster"])
    show(lr[["items", "units_multiple", "units_lo", "units_hi",
             "price_multiple", "eur_multiple", "eur_lo", "eur_hi"]].round(3))
    log.info("")
    show(lr[["units_vs_control", "units_vs_control_p", "price_vs_control",
             "eur_vs_control", "eur_vs_control_p"]].drop(C.VOL_CONTROL).round(3))
    log.info("(multiple = exp(mean log ratio per item); units x price = EUR "
             "exactly in logs; 95 %% CI by resampling skins)")
    lr.to_csv(C.OUT_TABLES / "08_long_run.csv", encoding="utf-8")

    section("4. THE ECHO OF THE SEVEN-DAY HOLD")
    ex = excess_units(units)
    # Echo and unit balance use all items of a group, not only the main sample.
    parts = {g: echo_parts(ex[everyone[g]]) for g in ECHO_GROUPS}
    point = balance({g: parts[g].sum() for g in ECHO_GROUPS})
    rng = np.random.default_rng(C.BOOTSTRAP_SEED)
    draws = [balance({g: draw_sums(parts[g], items["cluster"], rng)
                      for g in ECHO_GROUPS}) for _ in range(C.BOOTSTRAP_REPS)]
    rows = []
    for g in ECHO_GROUPS:
        sh = np.array([d["share"][g] for d in draws])
        diff = sh - np.array([d["share"]["control"] for d in draws])
        lo, hi = ci(sh)
        dlo, dhi = ci(diff)
        rows.append({"group": g, "items": len(parts[g]),
                     "wave_units": parts[g]["wave"].sum(),
                     "echo_units": parts[g]["echo"].sum(),
                     "share": point["share"][g], "share_lo": lo, "share_hi": hi,
                     "diff_vs_control": point["share"][g] - point["share"]["control"],
                     "diff_lo": dlo, "diff_hi": dhi,
                     "p_diff": p_two_sided(diff) if g != "control" else np.nan})
    echo = pd.DataFrame(rows).set_index("group")
    show(echo.round(3))
    log.info("(share = excess units %s minus %s, over excess units %s; all items "
             "of a group; 95 %% CI by resampling skins)", C.ECHO_WINDOW,
             C.ECHO_PRE, C.ECHO_WAVE)
    echo.to_csv(C.OUT_TABLES / "08_echo.csv", encoding="utf-8")

    section("5. UNIT BALANCE: COVERTS IN, KNIVES OUT (descriptive)")
    d = np.array([x["ratio"] for x in draws])
    lo_b, hi_b = ci(d)
    bal = pd.DataFrame([{"balance": f"excess Covert units {C.BALANCE_BUY} / "
                         f"excess knife units {C.BALANCE_SELL}",
                         "coverts_per_knife": point["ratio"],
                         "ci_lo": lo_b, "ci_hi": hi_b,
                         "covert_units": sum(parts["treated_input"]["buy"]),
                         "knife_units": sum(parts["treated_output"]["sell"])}]
                       ).set_index("balance")
    show(bal.round(2))
    log.info("A trade-up turns %d Coverts into 1 knife. Assumes every crafted knife "
             "is sold and every Covert bought on Steam, and that resales cancel.",
             C.CRAFT_INPUTS)
    log.info("A resale-corrected version is not identified: the knives' echo is "
             "no larger than the control group's (section 4).")
    bal.to_csv(C.OUT_TABLES / "08_unit_balance.csv", encoding="utf-8")

    section("6. FIGURES")
    lo, hi = C.VOLUME_PATH
    path = pd.DataFrame({g: np.exp(av_units.loc[lo:hi, members[g]].mean(axis=1))
                         for g in C.VOL_GROUPS})
    path.index.name = "t"
    path.join(knife_price.rename("knife_price_log").loc[lo:hi]).to_csv(
        C.OUT_TABLES / "08_path.csv", encoding="utf-8")
    fig_volume_path(path)
    fig_echo(echo)
    fig_knives(path, knife_price)
    fig_long_run(lr)
    log.info("08_fig1_volume_path, 08_fig2_echo, 08_fig3_knives_volume_price, "
             "08_fig4_long_run")
    log.info("")
    log.info("Done.")


if __name__ == "__main__":
    main()

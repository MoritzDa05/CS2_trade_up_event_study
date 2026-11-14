# Volatility: the signature V(k) = k * sigma^2 + 2 * eta^2 of weekly medians splits
# efficient volatility from noise. Before vs. after, vs. control, by binding status.
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
from inference import pool_clusters  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("07_volatility")

# Display names for groups that are not a single role (terms of the thesis).
GROUP_LABELS = {
    "control": "Control group",
    "input, binding": "Input, binding",
    "input, slack": "Input, slack",
    "input, no pre-event ratio": "Input, no pre-event ratio",
    "input, all": "Input, all",
    "placebo_st": "Placebo ST (pseudo route)",
}
GROUP_COLORS = {"control": ps.color("placebo")}
PANEL_TITLES = {"treated_output": "Output", "treated_input": "Input",
                "control": "Control group", "placebo_st": "Placebo ST"}
# Earlier updates (event days), dotted in 07_fig2_event_time; VOL_PRE_CALM ends before both.
EARLIER_UPDATES = {name: C.EVENT_DAYS[name] for name in C.EARLIER_UPDATES}


# Label with the days actually used: the full weeks inside the window.
def window_label(word: str, window: tuple[int, int]) -> str:
    weeks = window_weeks(window)
    lo, hi = 7 * weeks[0], 7 * weeks[-1] + 6
    return f"{word} (days {lo} to {hi})".replace("-", "−")


def glabel(g: str) -> str:
    return GROUP_LABELS.get(g, ps.label(g))


def gcolor(g: str) -> str:
    return GROUP_COLORS.get(g, ps.color(g))


def setup_logging() -> None:
    C.OUT_LOGS.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(C.OUT_LOGS / "07_volatility.log", mode="w",
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


# Weeks whose seven days all lie inside the window (week 0 = days 0 to 6).
def window_weeks(window: tuple[int, int]) -> range:
    lo, hi = window
    return range(-(-lo // 7), (hi - 6) // 7 + 1)


NO_RATIO = "no pre-event ratio"


# Binding: R_pre < 0.870; slack: R_pre >= 0.870; otherwise no pre-event ratio.
# 06 stores only a boolean, which would count the last group as slack.
def binding_status(routes: pd.DataFrame) -> pd.Series:
    r = routes["r_pre"]
    status = pd.Series(NO_RATIO, index=routes.index, dtype=object)
    status[r < C.BOUND_STRICT] = "binding"
    status[r >= C.BOUND_STRICT] = "slack"
    differ = (status == "binding") != routes["binding"].astype(bool)
    if differ.any():
        raise ValueError(f"{int(differ.sum())} routes are binding in 06 but not "
                         "here, or vice versa -- re-run 06_arbitrage.py.")
    return status


def load_items() -> pd.DataFrame:
    samples = pd.read_parquet(C.SAMPLES)
    items = pd.read_parquet(C.ITEMS_CLEAN)[["market_hash_name", "skin_id"]]
    role_to_group = {r: g for g, roles in C.VOL_GROUPS.items() for r in roles}
    d = samples[["market_hash_name", "role", "main", "liq50"]].merge(
        items, on="market_hash_name", how="left")
    d["group"] = d["role"].map(role_to_group)
    d = d.dropna(subset=["group"])
    # Cluster = skin (its wears are correlated); items without a skin are their own.
    d["cluster"] = d["skin_id"].astype("string").fillna(d["market_hash_name"])
    return d.set_index("market_hash_name")


# Weekly log median prices (week x item) and the daily panel.
def load_prices(names: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_parquet(C.PANEL_CLEAN, columns=[
        "market_hash_name", "t", "price", "volume", "ret_obs", "gap"])
    panel["market_hash_name"] = panel["market_hash_name"].astype(str)
    panel = panel[panel["market_hash_name"].isin(names)]
    traded = panel.dropna(subset=["price"]).assign(week=lambda d: d["t"] // 7)
    weekly = (traded.groupby(["week", "market_hash_name"])["price"].median()
              .unstack())
    weeks = range(C.WINDOW[0] // 7, C.WINDOW[1] // 7 + 1)
    return np.log(weekly.reindex(weeks)), panel


# Variance of the k-week log change per item, k = 1..K, demeaned with the item's mean
# weekly change and corrected for overlap (Lo and MacKinlay 1988):
# V(k) = sum (D_k - k mu)^2 / [n_k (1 - k / N)]; items with N <= K get no signature.
def signature_items(lw: pd.DataFrame, window: tuple[int, int], K: int
                    ) -> pd.DataFrame:
    sub = lw.loc[list(window_weeks(window))]
    d1 = sub.diff(1)
    N = d1.notna().sum()
    mu = d1.mean()
    out = {}
    for k in range(1, K + 1):
        dk = sub.diff(k)
        nk = dk.notna().sum()
        denom = (nk * (1 - k / N)).where((N > K) & (nk > 0))
        out[f"V{k}"] = ((dk - k * mu) ** 2).sum() / denom
    out["n1"] = N
    return pd.DataFrame(out)


# Robustness "not demeaned": mean squared k-week log change.
def signature_items_raw(lw: pd.DataFrame, window: tuple[int, int], K: int
                        ) -> pd.DataFrame:
    sub = lw.loc[list(window_weeks(window))]
    out = {f"V{k}": (sub.diff(k) ** 2).mean() for k in range(1, K + 1)}
    out["n1"] = sub.diff(1).notna().sum()
    return pd.DataFrame(out)


# Robustness: split from the first-order autocovariance (Zhou 1996), demeaned returns.
def weekly_ac_items(lw: pd.DataFrame, window: tuple[int, int]) -> pd.DataFrame:
    r = lw.loc[list(window_weeks(window))].diff(1)
    r = r - r.mean()
    ac = (r * r.shift(1)).mean()
    m2 = r.var(ddof=1)
    return pd.DataFrame({"raw": m2, "eff": m2 + 2 * ac, "noise": -ac,
                         "n1": r.notna().sum()})


# The same on daily trade-to-trade returns (demeaned): efficient variance / mean gap, per week.
def daily_ac_items(panel: pd.DataFrame, window: tuple[int, int]) -> pd.DataFrame:
    lo, hi = window
    d = panel.dropna(subset=["ret_obs"])
    d = d[d["t"].between(lo, hi) & (d["t"] - d["gap"] >= lo)]
    d = d.sort_values(["market_hash_name", "t"])
    d = d.assign(r=d["ret_obs"] - d.groupby("market_hash_name")["ret_obs"]
                 .transform("mean"))
    d["lag"] = d.groupby("market_hash_name")["r"].shift(1)
    g = d.groupby("market_hash_name")
    m2 = g["r"].var(ddof=1)
    ac = (d["r"] * d["lag"]).groupby(d["market_hash_name"]).mean()
    gap = g["gap"].mean()
    return pd.DataFrame({"raw": 7 * m2 / gap, "eff": 7 * (m2 + 2 * ac) / gap,
                         "noise": -ac, "n1": g.size()})


def fit_line(V: np.ndarray) -> tuple[float, float]:
    k = np.arange(1, len(V) + 1, dtype=float)
    ok = np.isfinite(V)
    k, V = k[ok], V[ok]
    b = ((k - k.mean()) * (V - V.mean())).sum() / ((k - k.mean()) ** 2).sum()
    return b, V.mean() - b * k.mean()


# Group (raw, efficient, noise): line through the median V(k), or medians of the splits.
def group_stat(X: np.ndarray, kind: str) -> np.ndarray:
    with np.errstate(all="ignore"):
        med = np.nanmedian(X, axis=0)
    if kind == "signature":
        b, a = fit_line(med)
        return np.array([med[0], b, a / 2])
    return med


# One way of measuring; items need min_returns weekly returns before and min_post
# (default: min_returns) after the update.
class Variant:
    def __init__(self, name: str, kind: str, pre: pd.DataFrame,
                 post: pd.DataFrame, min_returns: int, min_post: int | None = None):
        self.name, self.kind = name, kind
        self.cols = ([c for c in pre.columns if c.startswith("V")]
                     if kind == "signature" else ["raw", "eff", "noise"])
        min_post = min_returns if min_post is None else min_post
        ok = (pre["n1"] >= min_returns) & (post["n1"].reindex(pre.index)
                                           >= min_post)
        self.items = pre.index[ok.fillna(False)]
        self.pre = pre.loc[self.items, self.cols]
        self.post = post.loc[self.items, self.cols]


# Cluster bootstrap over skins, both windows on the same draw; groups independent.
def summarise(variant: Variant, members: dict[str, pd.Index],
              clusters: pd.Series, reps: int) -> dict[str, dict]:
    rng = np.random.default_rng(C.BOOTSTRAP_SEED)
    out = {}
    for g, idx in members.items():
        idx = idx.intersection(variant.items)
        if len(idx) == 0:
            continue
        Xpre = variant.pre.loc[idx].to_numpy(float)
        Xpost = variant.post.loc[idx].to_numpy(float)
        cl = clusters.loc[idx].to_numpy()
        codes, pos = np.unique(cl, return_inverse=True)
        rows = [np.flatnonzero(pos == i) for i in range(len(codes))]
        est = np.concatenate([group_stat(Xpre, variant.kind),
                              group_stat(Xpost, variant.kind)])
        draws = np.empty((reps, 6))
        for r in range(reps):
            pick = np.concatenate([rows[i] for i in
                                   rng.integers(0, len(rows), len(rows))])
            draws[r] = np.concatenate([group_stat(Xpre[pick], variant.kind),
                                       group_stat(Xpost[pick], variant.kind)])
        out[g] = {"n": len(idx), "clusters": len(codes), "est": est,
                  "draws": draws}
    return out


def to_sd(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return np.where(v > 0, np.sqrt(np.clip(v, 0, None)), np.nan)


MEASURES = {"raw": 0, "efficient": 1, "noise": 2}


def log_change(x: np.ndarray, m: int) -> np.ndarray:
    x = np.atleast_2d(x)
    with np.errstate(all="ignore"):
        return np.log(to_sd(x[:, 3 + m]) / to_sd(x[:, m]))


def ci(draws: np.ndarray) -> tuple[float, float]:
    d = draws[np.isfinite(draws)]
    return tuple(np.percentile(d, [2.5, 97.5])) if len(d) else (np.nan, np.nan)


def p_two_sided(draws: np.ndarray) -> float:
    d = draws[np.isfinite(draws)]
    if not len(d):
        return np.nan
    return min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))


def pct(x):
    return 100 * (np.exp(x) - 1)


def volatility_table(res: dict[str, dict], volume: pd.DataFrame
                     ) -> pd.DataFrame:
    rows = []
    for g, r in res.items():
        e = r["est"]
        row = {"group": g, "items": r["n"], "skins": r["clusters"]}
        if g in volume.index:
            row["trades_week_pre"] = volume.loc[g, "pre"]
            row["trades_week_post"] = volume.loc[g, "post"]
        for name, m in MEASURES.items():
            row[f"{name}_pre"] = 100 * to_sd(e[m])
            row[f"{name}_post"] = 100 * to_sd(e[3 + m])
            chg = log_change(e, m)[0]
            lo, hi = ci(log_change(r["draws"], m))
            row[f"{name}_chg_pct"] = pct(chg)
            row[f"{name}_chg_lo"], row[f"{name}_chg_hi"] = pct(lo), pct(hi)
        # Noise share = 2 eta^2 / V(1).
        with np.errstate(all="ignore"):
            row["noise_share_pre"] = 100 * 2 * e[2] / e[0]
            row["noise_share_post"] = 100 * 2 * e[5] / e[3]
        rows.append(row)
    return pd.DataFrame(rows).set_index("group")


# Change in log sd relative to the reference group (DiD), in %; bootstrap p-value.
def did_table(res: dict[str, dict], treated: list[str],
              reference: str = C.VOL_CONTROL) -> pd.DataFrame:
    c = res[reference]
    rows = []
    for g in treated:
        if g not in res:
            continue
        for name, m in MEASURES.items():
            d = log_change(res[g]["est"], m)[0] - log_change(c["est"], m)[0]
            dd = log_change(res[g]["draws"], m) - log_change(c["draws"], m)
            lo, hi = ci(dd)
            rows.append({"group": g, "reference": reference,
                         "measure": name, "did_pct": pct(d),
                         "ci_lo": pct(lo), "ci_hi": pct(hi),
                         "p_boot": p_two_sided(dd),
                         "draws_finite": int(np.isfinite(dd).sum())})
    return pd.DataFrame(rows)


# Binding and slack share skins: one draw resamples the skins for both groups.
def joint_difference(variant: Variant, idx_a: pd.Index, idx_b: pd.Index,
                     clusters: pd.Series, reps: int) -> pd.DataFrame:
    idx_a = idx_a.intersection(variant.items)
    idx_b = idx_b.intersection(variant.items)
    parts = {}
    for key, idx in (("a", idx_a), ("b", idx_b)):
        parts[key] = (variant.pre.loc[idx].to_numpy(float),
                      variant.post.loc[idx].to_numpy(float),
                      clusters.loc[idx].to_numpy())
    codes = np.unique(np.concatenate([parts["a"][2], parts["b"][2]]))
    rows = {k: {c: np.flatnonzero(v[2] == c) for c in codes}
            for k, v in parts.items()}

    def stat(key, pick=None):
        Xpre, Xpost, _ = parts[key]
        if pick is not None:
            sel = np.concatenate([rows[key][c] for c in pick]).astype(int)
            Xpre, Xpost = Xpre[sel], Xpost[sel]
        return np.concatenate([group_stat(Xpre, variant.kind),
                               group_stat(Xpost, variant.kind)])

    rng = np.random.default_rng(C.BOOTSTRAP_SEED)
    draws = {"a": np.empty((reps, 6)), "b": np.empty((reps, 6))}
    for r in range(reps):
        pick = codes[rng.integers(0, len(codes), len(codes))]
        for key in ("a", "b"):
            draws[key][r] = stat(key, pick)
    out = []
    for name, m in MEASURES.items():
        d = log_change(stat("a"), m)[0] - log_change(stat("b"), m)[0]
        dd = log_change(draws["a"], m) - log_change(draws["b"], m)
        lo, hi = ci(dd)
        out.append({"measure": name, "diff_pct": pct(d), "ci_lo": pct(lo),
                    "ci_hi": pct(hi), "p_boot": p_two_sided(dd)})
    return pd.DataFrame(out)


# Per route: sums of cross- and autoproducts of demeaned weekly returns.
def coupling_sums(c: pd.DataFrame, p: pd.DataFrame, window: tuple[int, int]
                  ) -> pd.DataFrame:
    wk = list(window_weeks(window))
    rc = c.loc[wk].diff(1)
    rp = p.loc[wk].diff(1)
    both = rc.notna() & rp.notna()
    rc = (rc - rc.where(both).mean()).where(rc.notna())
    rp = (rp - rp.where(both).mean()).where(rp.notna())
    out = {}
    for lag in range(-C.COUPLING_LAGS, C.COUPLING_LAGS + 1):
        prod = rc * rp.shift(-lag)
        out[f"cp{lag}"] = prod.sum()
        out[f"n_cp{lag}"] = prod.notna().sum()
    out["pp0"] = (rp ** 2).sum()
    out["n_pp0"] = rp.notna().sum()
    pp1 = rp * rp.shift(1)
    out["pp1"] = pp1.sum()
    out["n_pp1"] = pp1.notna().sum()
    return pd.DataFrame(out)


# Dimson beta (lead, same week, lag) with noise-corrected pool variance; naive beta.
def coupling_beta(S: pd.DataFrame) -> tuple[float, float]:
    tot = S.sum()
    cov = sum(tot[f"cp{l}"] / tot[f"n_cp{l}"]
              for l in range(-C.COUPLING_LAGS, C.COUPLING_LAGS + 1))
    var = tot["pp0"] / tot["n_pp0"]
    var_eff = var + 2 * tot["pp1"] / tot["n_pp1"]
    naive = (tot["cp0"] / tot["n_cp0"]) / var
    return cov / var_eff, naive


# Bootstrap clusters of routes: pools; with fewer than MIN_POOL_CLUSTERS pools the cases.
def route_clusters(r: pd.DataFrame, label: str) -> tuple[pd.Series, str]:
    pools = pool_clusters(r.index, r["case"])
    g_case, g_pool = r["case"].nunique(), pools.nunique()
    if g_pool < C.MIN_POOL_CLUSTERS:
        log.info("Clusters %s: %d cases -> %d pools; fewer than %d pools, case "
                 "clusters kept", label, g_case, g_pool, C.MIN_POOL_CLUSTERS)
        return r["case"], "cases"
    log.info("Clusters %s: %d cases -> %d pools", label, g_case, g_pool)
    return pools, "pools"


COUPLING_DIFF = "binding - slack (same pools)"


# Beta of the Covert on its pool's expected value, before and after; bootstrap by pool.
def coupling(lw: pd.DataFrame, routes: pd.DataFrame, reps: int
             ) -> pd.DataFrame:
    ev = pd.read_parquet(C.TRADEUP_EV, columns=["t", "market_hash_name", "ev"])
    # Expected value on the last day of each week.
    ev = ev[(ev["t"] % 7) == 6].assign(week=lambda d: d["t"] // 7)
    pool = np.log(ev.pivot(index="week", columns="market_hash_name",
                           values="ev")).reindex(lw.index)
    r = routes[routes["main"]]
    status = binding_status(r)
    inp = r["role"] == "treated_input"
    groups = {
        "input, all": r.index[inp],
        "input, binding": r.index[inp & (status == "binding")],
        "input, slack": r.index[inp & (status == "slack")],
        "placebo_st": r.index[r["role"] == C.PLACEBO_ROLE],
        # Last, so that the groups above do not depend on it; the binding - slack draws
        # below continue the same generator and do (appendix only).
        f"input, {NO_RATIO}": r.index[inp & (status == NO_RATIO)],
    }
    names = r.index.intersection(lw.columns).intersection(pool.columns)
    sums = {w: coupling_sums(lw[names], pool[names], win)
            for w, win in [("pre", C.VOL_PRE), ("post", C.VOL_POST)]}
    ok = ((sums["pre"]["n_cp0"] >= C.MIN_WEEKLY_RETURNS)
          & (sums["post"]["n_cp0"] >= C.MIN_WEEKLY_RETURNS))
    eligible = ok.index[ok]

    rng = np.random.default_rng(C.BOOTSTRAP_SEED)
    rows = []
    for g, idx in groups.items():
        idx = idx.intersection(eligible)
        if len(idx) == 0:
            continue
        cl, unit = route_clusters(r.loc[idx], g)
        codes, pos = np.unique(cl.to_numpy(), return_inverse=True)
        members = [idx[pos == i] for i in range(len(codes))]
        est = {w: coupling_beta(sums[w].loc[idx]) for w in sums}
        draws = np.empty((reps, 2))
        for k in range(reps):
            pick = np.concatenate([members[i] for i in
                                   rng.integers(0, len(members), len(members))])
            draws[k] = [coupling_beta(sums[w].loc[pick])[0] for w in ("pre", "post")]
        diff = draws[:, 1] - draws[:, 0]
        row = {"group": g, "routes": len(idx), "clusters": len(codes),
               "cluster_unit": unit}
        for j, w in enumerate(("pre", "post")):
            row[f"beta_{w}"] = est[w][0]
            row[f"beta_{w}_lo"], row[f"beta_{w}_hi"] = ci(draws[:, j])
            row[f"naive_{w}"] = est[w][1]
        row["change"] = est["post"][0] - est["pre"][0]
        row["change_lo"], row["change_hi"] = ci(diff)
        row["p_change"] = p_two_sided(diff)
        row["p_post_eq_1"] = p_two_sided(draws[:, 1] - 1)
        rows.append(row)

    # Binding vs. slack within the same pools: one draw resamples the pools for both.
    a = groups["input, binding"].intersection(eligible)
    b = groups["input, slack"].intersection(eligible)
    case_of, unit = route_clusters(r.loc[a.union(b)], "binding - slack")
    codes = np.unique(case_of.loc[a.union(b)])
    by_case = {k: {c: idx[case_of.loc[idx].to_numpy() == c] for c in codes}
               for k, idx in (("a", a), ("b", b))}

    def change(key, pick):
        sel = pd.Index(np.concatenate([by_case[key][c] for c in pick]))
        if len(sel) == 0:
            return np.nan
        return (coupling_beta(sums["post"].loc[sel])[0]
                - coupling_beta(sums["pre"].loc[sel])[0])

    est = change("a", codes) - change("b", codes)
    dd = np.empty(reps)
    for k in range(reps):
        pick = codes[rng.integers(0, len(codes), len(codes))]
        dd[k] = change("a", pick) - change("b", pick)
    lo, hi = ci(dd)
    rows.append({"group": COUPLING_DIFF, "routes": len(a) + len(b),
                 "clusters": len(codes), "cluster_unit": unit, "change": est,
                 "change_lo": lo,
                 "change_hi": hi, "p_change": p_two_sided(dd)})
    return pd.DataFrame(rows).set_index("group")


# Signature on rolling windows of VOL_ROLLING_WEEKS weeks, never across t = 0.
def rolling(lw: pd.DataFrame, members: dict[str, pd.Index]) -> pd.DataFrame:
    K, L = C.VOL_SIGNATURE_K, C.VOL_ROLLING_WEEKS
    first, last = C.VOL_ROLLING_RANGE
    rows = []
    for start in range(first, last - L + 2):
        end = start + L - 1
        if start < 0 <= end:
            continue
        sig = signature_items(lw, (7 * start, 7 * end + 6), K)
        V, n1 = sig[[f"V{k}" for k in range(1, K + 1)]], sig["n1"]
        for g, idx in members.items():
            idx = idx.intersection(n1.index[n1 >= C.VOL_ROLLING_MIN_RETURNS])
            if len(idx) < C.VOL_ROLLING_MIN_ITEMS:
                continue
            raw, eff, noise = group_stat(V.loc[idx].to_numpy(float), "signature")
            rows.append({"group": g, "mid_day": 7 * start + (7 * L - 1) / 2,
                         "regime": "post" if start >= 0 else "pre",
                         "items": len(idx), "slope": eff, "intercept": 2 * noise,
                         "raw_pct": 100 * to_sd(raw),
                         "efficient_pct": 100 * to_sd(eff),
                         "noise_pct": 100 * to_sd(noise)})
    out = pd.DataFrame(rows)
    # Share of group windows with a negative slope or intercept (fit not interpretable).
    neg = out.assign(neg_slope=out["slope"] < 0, neg_intercept=out["intercept"] < 0)
    log.info("Rolling windows with a negative slope / intercept (share per group "
             "and regime):")
    show(neg.groupby(["group", "regime"])[["neg_slope", "neg_intercept"]].mean().round(3))
    return out


def fig_signature(sig: pd.DataFrame, groups: list[str]) -> None:
    # No figure title: the caption explains slope and intercept.
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.2), sharex=True)
    K = C.VOL_SIGNATURE_K
    k = np.arange(1, K + 1)
    for ax, g in zip(axes.flat, groups):
        for w, col, mk in [("pre", ps.NEUTRAL, "o"), ("post", gcolor(g), "s")]:
            V = sig.loc[(g, w)].to_numpy(float) * 1e3
            b, a = fit_line(V)
            xs = np.linspace(0, K, 50)
            ax.plot(xs, a + b * xs, color=col, linewidth=1.2,
                    linestyle="--" if w == "pre" else "-")
            ax.scatter(k, V, color=col, marker=mk, s=22, zorder=3,
                       label=window_label("before", C.VOL_PRE) if w == "pre"
                       else window_label("after", C.VOL_POST))
        ax.set_title(PANEL_TITLES.get(g, glabel(g)), fontsize=10)
        ax.set_xlim(0, K + 0.3)
        ax.set_ylim(bottom=0)
        ax.set_xticks(k)
    for ax in axes[1]:
        ax.set_xlabel(r"horizon $k$ (weeks)")
    fig.supylabel(r"variance of $k$-week change ($\times 10^{-3}$)",
                  fontsize=10, color=ps.NEUTRAL)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    ps.save(fig, "07_fig1_signature", C.OUT_FIGURES)


def fig_event_time(roll: pd.DataFrame, groups: list[str]) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 5.6), sharex=True)
    for g in groups:
        d = roll[roll["group"] == g]
        for regime in ("pre", "post"):
            s = d[d["regime"] == regime]
            kw = {"color": gcolor(g),
                  "label": glabel(g) if regime == "post" else None}
            ax1.plot(s["mid_day"], s["efficient_pct"], **kw)
            # No rolling noise for the control group: too few items.
            if g != C.VOL_CONTROL:
                ax2.plot(s["mid_day"], s["noise_pct"], **kw)
    for ax in (ax1, ax2):
        ax.axvline(0, color=ps.NEUTRAL, linewidth=1)
        ax.axvspan(0, 30, color=ps.GRID, alpha=0.5, linewidth=0)
        for day in EARLIER_UPDATES.values():
            ax.axvline(day, color=ps.NEUTRAL, linewidth=0.8, linestyle=":")
        ax.set_ylim(bottom=0)
    for name, day in EARLIER_UPDATES.items():
        ax1.text(day - 1.5, 0.97, name, transform=ax1.get_xaxis_transform(),
                 rotation=90, ha="right", va="top", fontsize=8,
                 color=ps.NEUTRAL)
    ax1.set_ylabel("efficient volatility\n(weekly sd, %)")
    ax2.set_ylabel("noise\n(sd of weekly median, %)")
    ax2.set_xlabel(f"days relative to the update (middle of an "
                   f"{C.VOL_ROLLING_WEEKS}-week window; shaded: transition)")
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    ps.save(fig, "07_fig2_event_time", C.OUT_FIGURES)


def fig_coupling(cp: pd.DataFrame) -> None:
    order = [g for g in ["input, binding", "input, slack", "placebo_st"]
             if g in cp.index]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for i, g in enumerate(order):
        r = cp.loc[g]
        for off, w, mk in [(-0.12, "pre", "o"), (0.12, "post", "s")]:
            col = ps.NEUTRAL if w == "pre" else gcolor(
                "treated_input" if g.startswith("input") else g)
            ax.errorbar(i + off, r[f"beta_{w}"],
                        yerr=[[r[f"beta_{w}"] - r[f"beta_{w}_lo"]],
                              [r[f"beta_{w}_hi"] - r[f"beta_{w}"]]],
                        fmt=mk, color=col, capsize=3,
                        label=(window_label("before", C.VOL_PRE) if w == "pre"
                               else window_label("after", C.VOL_POST))
                        if i == 0 else None)
    ax.axhline(0, color=ps.NEUTRAL, linewidth=0.8)
    ax.axhline(1, color=ps.NEUTRAL, linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([glabel(g) + f"\n({int(cp.loc[g, 'clusters'])} "
                        f"{cp.loc[g, 'cluster_unit']})" for g in order], fontsize=8.5)
    ax.set_ylabel(r"$\beta$ on the own pool (95% CI)")
    ax.set_ylim(top=1.3)
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    fig.tight_layout()
    ps.save(fig, "07_fig3_coupling", C.OUT_FIGURES)


def main() -> None:
    setup_logging()
    ps.apply_style()
    C.OUT_TABLES.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 170)

    section("1. DATA")
    items = load_items()
    lw, panel = load_prices(items.index)
    main_items = items[items["main"]]
    members = {g: main_items.index[main_items["group"] == g]
               for g in C.VOL_GROUPS}
    clusters = items["cluster"]
    log.info("Weekly price series: %d weeks x %d items | windows %s and %s "
             "(%d and %d full weeks)", *lw.shape, C.VOL_PRE, C.VOL_POST,
             len(window_weeks(C.VOL_PRE)), len(window_weeks(C.VOL_POST)))

    K = C.VOL_SIGNATURE_K
    sig_pre, sig_post = signature_items(lw, C.VOL_PRE, K), signature_items(lw, C.VOL_POST, K)
    for w, sg in (("pre", sig_pre), ("post", sig_post)):
        short = sg.index[(sg["n1"] <= K) & sg.index.isin(main_items.index)]
        log.info("Items without a signature in the %s window (N <= K = %d one-week "
                 "changes): %d", w, K, len(short))
    main_v = Variant("main: signature 1-%d weeks" % K, "signature",
                     sig_pre, sig_post, C.MIN_WEEKLY_RETURNS)

    # Median weekly units per group (liquidity context), items in the signature only.
    vol = panel.assign(week=panel["t"] // 7)
    in_sig = main_items.index.intersection(main_v.items)
    volume = {}
    for w, win in [("pre", C.VOL_PRE), ("post", C.VOL_POST)]:
        v = vol[vol["week"].isin(window_weeks(win))]
        per_item = v.groupby(["market_hash_name", "week"])["volume"].sum() \
            .groupby("market_hash_name").median()
        per_item = per_item.reindex(in_sig)
        volume[w] = per_item.groupby(main_items.loc[in_sig, "group"]).median()
    volume = pd.DataFrame(volume)
    counts = pd.DataFrame({
        "main sample": main_items.groupby("group").size(),
        "eligible": main_items.loc[main_items.index.intersection(main_v.items)]
        .groupby("group").size()}).reindex(list(C.VOL_GROUPS)).fillna(0).astype(int)
    log.info("Items with >= %d weekly returns in both windows:",
             C.MIN_WEEKLY_RETURNS)
    show(counts)

    section("2. SIGNATURE: MEDIAN VARIANCE OF k-WEEK CHANGES")
    sig_rows = {}
    for g, idx in members.items():
        idx = idx.intersection(main_v.items)
        for w, X in [("pre", main_v.pre), ("post", main_v.post)]:
            sig_rows[(g, w)] = np.nanmedian(X.loc[idx].to_numpy(float), axis=0)
    sig = pd.DataFrame(sig_rows, index=main_v.cols).T
    sig.index.names = ["group", "window"]
    show((sig * 1e3).round(3))
    log.info("(x 1e-3; a straight line in k means the model fits)")
    sig.to_csv(C.OUT_TABLES / "07_signature.csv", encoding="utf-8")

    section("3. VOLATILITY BEFORE AND AFTER (weekly sd in %)")
    res = summarise(main_v, members, clusters, C.BOOTSTRAP_REPS)
    vt = volatility_table(res, volume)
    show(vt[["items", "trades_week_pre", "trades_week_post", "raw_pre",
             "raw_post", "raw_chg_pct"]].round(2))
    log.info("")
    show(vt[["efficient_pre", "efficient_post", "efficient_chg_pct",
             "efficient_chg_lo", "efficient_chg_hi"]].round(2))
    log.info("")
    show(vt[["noise_pre", "noise_post", "noise_chg_pct", "noise_chg_lo",
             "noise_chg_hi", "noise_share_pre", "noise_share_post"]].round(2))
    vt.to_csv(C.OUT_TABLES / "07_volatility.csv", encoding="utf-8")

    section("4. RELATIVE TO THE CONTROL GROUP")
    did = pd.concat([did_table(res, C.VOL_TREATED + ["placebo_st"]),
                     did_table(res, C.VOL_TREATED + ["placebo_st"],
                               reference="index_alt")],
                    ignore_index=True)
    show(did.round(3))
    log.info("(reference index_alt = music kits and agents, a second check "
             "because the control group has only %d skins; draws_finite: bootstrap "
             "draws with a positive variance in both groups, out of %d)",
             res[C.VOL_CONTROL]["clusters"], C.BOOTSTRAP_REPS)
    did.to_csv(C.OUT_TABLES / "07_did.csv", index=False, encoding="utf-8")

    section("5. COVERTS AT THE BOUND VERSUS THE OTHERS")
    routes = pd.read_parquet(C.TRADEUP_ROUTES).set_index("market_hash_name")
    status = binding_status(
        routes.loc[routes.index.intersection(members["treated_input"])])
    n_status = pd.DataFrame({
        "main sample": status.value_counts(),
        "eligible": status.loc[status.index.intersection(main_v.items)]
        .value_counts()}).reindex(["binding", "slack", NO_RATIO]).fillna(0)
    log.info("Input Coverts by binding status (R before the update < %.3f "
             "binding, >= %.3f slack; without a pre-event ratio neither):",
             C.BOUND_STRICT, C.BOUND_STRICT)
    show(n_status.astype(int))
    log.info("")
    # No-ratio group last, so that the draws of the others do not depend on it.
    het_members = {"input, binding": status.index[status == "binding"],
                   "input, slack": status.index[status == "slack"],
                   C.VOL_CONTROL: members[C.VOL_CONTROL],
                   f"input, {NO_RATIO}": status.index[status == NO_RATIO]}
    het = summarise(main_v, het_members, clusters, C.BOOTSTRAP_REPS)
    het_tab = volatility_table(het, pd.DataFrame())
    show(het_tab[["items", "efficient_pre", "efficient_post",
                  "efficient_chg_pct", "noise_chg_pct"]].round(2))
    het_did = did_table(het, ["input, binding", "input, slack",
                              f"input, {NO_RATIO}"])
    show(het_did.round(3))
    joint = joint_difference(main_v, het_members["input, binding"],
                             het_members["input, slack"], clusters,
                             C.BOOTSTRAP_REPS)
    log.info("")
    log.info("Binding minus slack, same skins resampled together:")
    show(joint.round(3))
    het_tab.join(het_did.pivot(index="group", columns="measure",
                               values="did_pct").add_prefix("did_")
                 ).to_csv(C.OUT_TABLES / "07_heterogeneity.csv", encoding="utf-8")
    het_did.to_csv(C.OUT_TABLES / "07_heterogeneity_did.csv", index=False,
                   encoding="utf-8")
    joint.to_csv(C.OUT_TABLES / "07_heterogeneity_joint.csv", index=False,
                 encoding="utf-8")

    section("6. COUPLING OF COVERTS TO THEIR OWN POOL")
    cp = coupling(lw, routes, C.BOOTSTRAP_REPS)
    show(cp[["routes", "clusters", "beta_pre", "beta_pre_lo", "beta_pre_hi",
             "beta_post", "beta_post_lo", "beta_post_hi"]].dropna(how="all",
             subset=["beta_pre"]).round(3))
    log.info("")
    show(cp[["naive_pre", "naive_post", "change", "change_lo", "change_hi",
             "p_change", "p_post_eq_1"]].round(3))
    log.info("(beta: one lead and one lag, noise-corrected pool variance; "
             "naive: contemporaneous OLS. The placebo has only %d %s, so its "
             "interval is not reliable.)", cp.loc["placebo_st", "clusters"],
             cp.loc["placebo_st", "cluster_unit"])
    cp.to_csv(C.OUT_TABLES / "07_coupling.csv", encoding="utf-8")

    section("7. ROBUSTNESS")
    liq50 = items[items["liq50"]]
    members_liq50 = {g: liq50.index[liq50["group"] == g] for g in C.VOL_GROUPS}
    K2 = C.VOL_SIGNATURE_K_ALT
    variants = [
        (main_v, members),
        (Variant(f"signature 1-{K2} weeks", "signature",
                 signature_items(lw, C.VOL_PRE, K2),
                 signature_items(lw, C.VOL_POST, K2), C.MIN_WEEKLY_RETURNS),
         members),
        (Variant(f"post window {C.VOL_POST_SHORT}", "signature",
                 signature_items(lw, C.VOL_PRE, K),
                 signature_items(lw, C.VOL_POST_SHORT, K),
                 C.MIN_WEEKLY_RETURNS, C.MIN_WEEKLY_RETURNS_SHORT), members),
        # Pre window ending before Genesis and Armory.
        (Variant(f"pre window {C.VOL_PRE_CALM}", "signature",
                 signature_items(lw, C.VOL_PRE_CALM, K),
                 signature_items(lw, C.VOL_POST, K), C.MIN_WEEKLY_RETURNS),
         members),
        (Variant("per item, weekly autocovariance", "ac",
                 weekly_ac_items(lw, C.VOL_PRE),
                 weekly_ac_items(lw, C.VOL_POST), C.MIN_WEEKLY_RETURNS),
         members),
        (Variant("per item, daily trade-to-trade", "ac",
                 daily_ac_items(panel, C.VOL_PRE),
                 daily_ac_items(panel, C.VOL_POST), C.MIN_WEEKLY_RETURNS),
         members),
        (main_v, members_liq50),
        (Variant("not demeaned", "signature",
                 signature_items_raw(lw, C.VOL_PRE, K),
                 signature_items_raw(lw, C.VOL_POST, K), C.MIN_WEEKLY_RETURNS),
         members),
    ]
    labels = [v.name for v, _ in variants]
    labels[-2] = "liquidity >= 50 %"
    rob_rows = []
    for (v, mem), label in zip(variants, labels):
        keep = {g: mem[g] for g in C.VOL_TREATED + [C.VOL_CONTROL]}
        r = summarise(v, keep, clusters, C.BOOTSTRAP_REPS)
        d = did_table(r, C.VOL_TREATED).set_index(["group", "measure"])
        row = {"variant": label}
        for g in C.VOL_TREATED + [C.VOL_CONTROL]:
            short = g.replace("treated_", "")
            row[f"{short}_n"] = r[g]["n"]
            for mname, m in [("eff", 1), ("noise", 2)]:
                row[f"{short}_{mname}_chg"] = pct(log_change(r[g]["est"], m)[0])
        for g in C.VOL_TREATED:
            short = g.replace("treated_", "")
            row[f"{short}_did_eff"] = d.loc[(g, "efficient"), "did_pct"]
            row[f"{short}_did_eff_p"] = d.loc[(g, "efficient"), "p_boot"]
        rob_rows.append(row)
    rob = pd.DataFrame(rob_rows).set_index("variant")
    show(rob.round(2).T)
    rob.to_csv(C.OUT_TABLES / "07_robustness.csv", encoding="utf-8")

    section("8. FIGURES")
    roll = rolling(lw, {g: members[g] for g in C.VOL_TREATED + [C.VOL_CONTROL]})
    roll.to_csv(C.OUT_TABLES / "07_rolling.csv", index=False, encoding="utf-8")
    fig_signature(sig, C.VOL_TREATED + [C.VOL_CONTROL, "placebo_st"])
    fig_event_time(roll, C.VOL_TREATED + [C.VOL_CONTROL])
    fig_coupling(cp)
    log.info("07_fig1_signature, 07_fig2_event_time, 07_fig3_coupling")
    log.info("")
    log.info("Done.")


if __name__ == "__main__":
    main()

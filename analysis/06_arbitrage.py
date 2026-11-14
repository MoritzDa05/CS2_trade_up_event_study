# Arbitrage: R = 5 x Covert price / expected output value per route, with output wear
# tiers from the float rule. Bound, speed and side of the adjustment, placebo.
from __future__ import annotations

import logging
import sys
from math import comb, factorial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import plotstyle as ps  # noqa: E402
from inference import (OLS, design, pool_clusters, star_p,  # noqa: E402
                       weight_scheme, wild_bootstrap)
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("06_arbitrage")

N_IN = C.CRAFT_INPUTS
WEAR_ORDER = list(C.WEAR_BOUNDS)
# Upper edges of the first four wear tiers.
WEAR_EDGES = np.array([hi for _, hi in C.WEAR_BOUNDS.values()][:-1])
TREATED = "treated_input"


def setup_logging() -> None:
    C.OUT_LOGS.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(C.OUT_LOGS / "06_arbitrage.log", mode="w",
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


# Irwin-Hall coefficients for the mean of N_IN uniform input positions.
_IH_K = np.arange(N_IN + 1)
_IH_COEF = np.array([(-1) ** k * comb(N_IN, k) for k in _IH_K],
                    dtype=float) / factorial(N_IN)


# P(mean of N_IN uniform draws on [lo, hi] <= q), closed form; lo == hi is a point mass.
def mean_cdf(q: np.ndarray, lo: float, hi: float) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if hi - lo < 1e-12:
        return (q >= lo).astype(float)
    x = np.clip(N_IN * (q - lo) / (hi - lo), 0.0, N_IN)
    terms = _IH_COEF * np.clip(x[..., None] - _IH_K, 0.0, None) ** N_IN
    return np.clip(terms.sum(axis=-1), 0.0, 1.0)


# One input's contribution: position in its skin's range (normalized) or its float (raw).
def input_interval(wear: str, fmin: float, fmax: float, float_rule: str,
                   input_float: str) -> tuple[float, float] | None:
    w_lo, w_hi = C.WEAR_BOUNDS[wear]
    lo, hi = max(w_lo, fmin), min(w_hi, fmax)
    if hi < lo:
        return None
    if input_float == "best":
        hi = lo
    if float_rule == "normalized":
        span = fmax - fmin
        return (lo - fmin) / span, (hi - fmin) / span
    return lo, hi


# Probability of each wear tier for an output skin with range [lo, hi].
def output_wear_probs(out_lo: float, out_hi: float,
                      interval: tuple[float, float]) -> np.ndarray:
    q = (WEAR_EDGES - out_lo) / (out_hi - out_lo)
    cdf = mean_cdf(q, *interval)
    return np.diff(np.concatenate([[0.0], cdf, [1.0]]))


# Outcome probabilities of a route (5 copies of one input): each finish 1/n, its
# wear tier from the float rule. Unmatched outcomes keep output = NaN (unpriced).
def route_weights(routes: pd.DataFrame, pools: pd.DataFrame, float_rule: str,
                  input_float: str) -> pd.DataFrame:
    groups = {k: g for k, g in pools.groupby(["case", "stattrak"])}
    rows = []
    for r in routes.itertuples(index=False):
        g = groups[(r.case, r.pool_stattrak)]
        names = dict(zip(zip(g["skin"], g["wear"].fillna("")),
                         g["market_hash_name"]))
        finishes = g.drop_duplicates("skin")
        p_finish = 1.0 / len(finishes)
        interval = input_interval(r.wear, r.min_float, r.max_float,
                                  float_rule, input_float)
        for f in finishes.itertuples(index=False):
            if pd.isna(f.min_float):
                rows.append((r.market_hash_name, names.get((f.skin, "")),
                             p_finish, "vanilla"))
                continue
            probs = output_wear_probs(f.min_float, f.max_float, interval)
            for wear, p in zip(WEAR_ORDER, probs):
                if p > 1e-12:
                    rows.append((r.market_hash_name, names.get((f.skin, wear)),
                                 p_finish * p, wear))
    w = pd.DataFrame(rows, columns=["route", "output", "weight", "out_wear"])
    total = w.groupby("route")["weight"].sum()
    assert np.allclose(total, 1.0), "route probabilities do not sum to one"
    return w


# Input routes with the pool each one feeds, and the pool table.
def load_routes() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not C.CASE_POOLS.exists():
        raise FileNotFoundError(
            f"{C.CASE_POOLS} missing. Re-run build_item_universe.py.")
    items = pd.read_parquet(C.ITEMS_CLEAN)
    samples = pd.read_parquet(C.SAMPLES).set_index("market_hash_name")
    # utf-8-sig: the item lists are written with a BOM.
    pools = pd.read_csv(C.CASE_POOLS, encoding="utf-8-sig")

    r = items[items["role"].isin([TREATED, C.PLACEBO_ROLE])].copy()
    r["main"] = (samples["main"].reindex(r["market_hash_name"])
                 .fillna(False).astype(bool).to_numpy())
    # StatTrak feeds the StatTrak pool; the placebo gets a pseudo route into the glove pool.
    r["pool_stattrak"] = np.where(r["role"] == C.PLACEBO_ROLE, False,
                                  r["stattrak"])
    keys = set(zip(pools["case"], pools["stattrak"]))
    missing = [k for k in zip(r["case"], r["pool_stattrak"]) if k not in keys]
    if missing:
        raise ValueError(f"{len(missing)} routes without a pool, e.g. {missing[:3]}")
    # Knife or glove pool (no case has both).
    pool_type = pools.groupby("case")["category"].agg(
        lambda c: c.iloc[0] if c.nunique() == 1 else "mixed")
    r["pool_type"] = r["case"].map(pool_type)
    return r.reset_index(drop=True), pools


# Raw daily prices and a trailing median over the traded days of PRICE_SMOOTH_DAYS,
# carried forward up to PRICE_MAX_STALE days; never across t = 0.
def daily_prices(names: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_parquet(C.PANEL_CLEAN,
                            columns=["market_hash_name", "t", "price"])
    panel["market_hash_name"] = panel["market_hash_name"].astype(str)
    panel = panel[panel["market_hash_name"].isin(names)]
    wide = (panel.pivot(index="t", columns="market_hash_name", values="price")
            .sort_index().reindex(range(C.WINDOW[0], C.WINDOW[1] + 1)))
    return wide, smooth_prices(wide, C.PRICE_SMOOTH_DAYS)


# Trailing median over `days` calendar days (traded days only), carried forward up to
# PRICE_MAX_STALE days, never across t = 0; days = 1 keeps the last traded price.
def smooth_prices(wide: pd.DataFrame, days: int) -> pd.DataFrame:
    parts = []
    for seg in (wide.loc[:-1], wide.loc[0:]):
        smooth = seg.rolling(days, min_periods=1).median()
        pos = pd.DataFrame(np.repeat(np.arange(len(seg))[:, None],
                                     seg.shape[1], axis=1),
                           index=seg.index, columns=seg.columns)
        age = pos - pos.where(seg.notna()).ffill()
        parts.append(smooth.ffill().where(age <= C.PRICE_MAX_STALE))
    return pd.concat(parts)


# Cluster per route or item: its pool (inference.pool_clusters). With fewer than
# MIN_POOL_CLUSTERS pools, the cases are the clusters.
def cluster_ids(d: pd.DataFrame, label: str) -> np.ndarray:
    pools = pool_clusters(d.index, d["case"])
    g_case, g_pool = d["case"].nunique(), pools.nunique()
    if g_pool < C.MIN_POOL_CLUSTERS:
        log.info("Clusters %s: %d cases -> %d pools; fewer than %d pools, case "
                 "clusters kept", label, g_case, g_pool, C.MIN_POOL_CLUSTERS)
        return d["case"].to_numpy()
    log.info("Clusters %s: %d cases -> %d pools", label, g_case, g_pool)
    return pools.to_numpy()


# Median traded price per item over a window (NaN without enough trading days).
def window_prices(raw: pd.DataFrame, window: tuple[int, int],
                  shift: int = 0) -> pd.Series:
    sub = raw.loc[window[0] + shift:window[1] + shift]
    return sub.median().where(sub.notna().sum() >= C.MIN_WINDOW_TRADES)


# EV, priced mass and capped share per route from one price per outcome.
def window_ev(weights: pd.DataFrame, px: pd.Series) -> pd.DataFrame:
    w = weights.assign(p=weights["output"].map(px))
    w["priced"] = w["weight"].where(w["p"].notna(), 0.0)
    w["value"] = (w["weight"] * w["p"]).fillna(0.0)
    w["capped"] = w["value"].where(w["p"] >= C.HIGH_PRICE_EUR, 0.0)
    g = w.groupby("route")[["priced", "value", "capped"]].sum()
    out = pd.DataFrame({"mass": g["priced"]})
    out["ev"] = (g["value"] / g["priced"]).where(g["priced"] >= C.MIN_PRICED_MASS)
    out["cap_share"] = (g["capped"] / g["value"]).where(g["value"] > 0)
    return out


# Daily EV (priced part rescaled to one, NaN below MIN_PRICED_MASS), mass, capped share.
def expected_values(weights: pd.DataFrame, prices: pd.DataFrame
                    ) -> dict[str, pd.DataFrame]:
    routes = weights["route"].unique()
    w = weights[weights["output"].isin(prices.columns)]
    W = (w.pivot_table(index="route", columns="output", values="weight",
                       aggfunc="sum", fill_value=0.0)
         .reindex(index=routes, fill_value=0.0))
    P = prices[W.columns].to_numpy()
    have = np.isfinite(P).astype(float)
    Pv = np.nan_to_num(P)
    Wt = W.to_numpy().T
    mass = have @ Wt
    value = Pv @ Wt
    capped = (Pv * (Pv >= C.HIGH_PRICE_EUR)) @ Wt
    with np.errstate(invalid="ignore", divide="ignore"):
        ev = np.where(mass >= C.MIN_PRICED_MASS, value / mass, np.nan)
        cap_share = np.where(value > 0, capped / value, np.nan)
    frame = lambda a: pd.DataFrame(a, index=prices.index, columns=routes)  # noqa: E731
    return {"ev": frame(ev), "mass": frame(mass), "cap": frame(cap_share)}


# EVs, ratios and route summary for one variant. Binding status always comes from
# the main specification; the ex post variant reuses its EVs.
def run_spec(routes: pd.DataFrame, pools: pd.DataFrame, raw: pd.DataFrame,
             prices: pd.DataFrame, float_rule: str = C.FLOAT_RULE, input_float: str = C.INPUT_FLOAT,
             ex_post: bool = False, base: dict | None = None,
             prices_hl: pd.DataFrame | None = None) -> dict:
    if base is not None and ex_post:
        weights, evs = base["weights"], base["evs"]
    else:
        weights = route_weights(routes, pools, float_rule, input_float)
        evs = expected_values(weights, prices)
    names = routes["market_hash_name"].tolist()
    p_in = prices[names]
    ev = evs["ev"][names]
    ev_used = ev.shift(-C.TRADE_HOLD_DAYS) if ex_post else ev
    ratio = N_IN * p_in / ev_used

    # Half-life on prices smoothed over HALF_LIFE_SMOOTH_DAYS (1 = unsmoothed).
    hl = {}
    if prices_hl is not None:
        evs_hl = (base["evs_hl"] if base is not None and ex_post
                  else expected_values(weights, prices_hl))
        ev_hl = evs_hl["ev"][names]
        hl = {"evs_hl": evs_hl, "p_in_hl": prices_hl[names],
              "ev_used_hl": ev_hl.shift(-C.TRADE_HOLD_DAYS) if ex_post else ev_hl}
        hl["ratio_hl"] = N_IN * hl["p_in_hl"] / hl["ev_used_hl"]

    s = routes.set_index("market_hash_name")[
        ["role", "main", "case", "wear", "stattrak", "pool_type"]].copy()
    pre_px = window_prices(raw, C.LEVEL_PRE)
    post_px = window_prices(raw, C.LEVEL_POST)
    hold = C.TRADE_HOLD_DAYS if ex_post else 0
    ev_pre = window_ev(weights, pre_px)  # always ex ante
    ev_post = window_ev(weights, window_prices(raw, C.LEVEL_POST, hold))
    s["p_pre"] = pre_px.reindex(s.index)
    s["p_post"] = post_px.reindex(s.index)
    s["ev_pre"] = ev_pre["ev"].reindex(s.index)
    s["ev_post"] = ev_post["ev"].reindex(s.index)
    s["r_pre"] = N_IN * s["p_pre"] / s["ev_pre"]
    s["r_post"] = N_IN * s["p_post"] / s["ev_post"]
    s["mass_pre"] = ev_pre["mass"].reindex(s.index)
    s["mass_post"] = ev_post["mass"].reindex(s.index)
    s["cap_share_pre"] = ev_pre["cap_share"].reindex(s.index)
    s["n_outcomes"] = (weights.groupby("route")["output"].size()
                       .reindex(s.index))
    if base is None:
        s["binding"] = s["r_pre"] < C.BOUND_STRICT
    else:
        s["binding"] = base["summary"]["binding"].reindex(s.index)
    return {"weights": weights, "evs": evs, "p_in": p_in, "ev": ev,
            "ev_used": ev_used, "ratio": ratio, "summary": s, **hl}


# Routes by role, sample, binding status (needs a pre-event ratio) and cap filter.
def select(s: pd.DataFrame, role: str = TREATED, sample: str = "main",
           binding: bool | None = None, cap_filter: bool = False) -> pd.Index:
    m = s["role"] == role
    if sample == "main":
        m &= s["main"]
    if binding is not None:
        m &= (s["binding"] == binding) & s["r_pre"].notna()
    if cap_filter:
        m &= s["cap_share_pre"] <= C.CAP_SHARE_MAX
    return s.index[m]


# Output wear tiers by input wear, main treated routes (the float rule made visible).
def output_wear_table(weights: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    idx = select(s)
    w = weights[weights["route"].isin(idx)]
    tab = (w.groupby(["route", "out_wear"])["weight"].sum().unstack(fill_value=0)
           .join(s["wear"]).groupby("wear").mean() * 100)
    cols = [c for c in WEAR_ORDER + ["vanilla"] if c in tab.columns]
    return tab.reindex(WEAR_ORDER)[cols].round(1)


# Shares of routes below, in and above the band, before and after.
def bound_table(s: pd.DataFrame, idx: pd.Index) -> pd.DataFrame:
    d = s.loc[idx]
    groups = [("all", d), ("binding", d[d["binding"]]),
              ("slack", d[~d["binding"]])]
    groups += [(f"wear: {w}", d[d["wear"] == w]) for w in WEAR_ORDER]
    # Binding routes by knife or glove pool: where does the overshoot come from?
    for kind, label in [("Knives", "knife cases"), ("Gloves", "glove cases")]:
        groups.append((f"binding: {label}",
                       d[d["binding"] & (d["pool_type"] == kind)]))
    rows = []
    for label, g in groups:
        g = g.dropna(subset=["r_pre", "r_post"])
        if g.empty:
            continue
        row = {"group": label, "n": len(g)}
        for per in ("pre", "post"):
            r = g[f"r_{per}"]
            row[f"median_r_{per}"] = r.median()
            row[f"below_{per}_pct"] = 100 * (r < C.BOUND_STRICT).mean()
            row[f"band_{per}_pct"] = 100 * r.between(C.BOUND_STRICT,
                                                     C.BOUND_HOLDER).mean()
            row[f"above_{per}_pct"] = 100 * (r > C.BOUND_HOLDER).mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("group")


# Covert price on the exact EV in EUR vs. 0.174 and 0.2; also binding routes only.
# Before and after on the same routes (price and EV in both windows).
def level_test(s: pd.DataFrame, idx: pd.Index) -> pd.DataFrame:
    rows = []
    for label, sub in [("all", s.loc[idx]),
                       ("binding", s.loc[idx][s.loc[idx, "binding"]])]:
        sub = sub.dropna(subset=["p_pre", "ev_pre", "p_post", "ev_post"])
        for per in ("pre", "post"):
            d = sub
            X, names = design(d, [f"ev_{per}"])
            m = OLS(d[f"p_{per}"].to_numpy(float), X,
                    cluster_ids(d, f"level test {label} {per}"), names)
            slope, se = m.beta[1], m.se[1]
            row = {"routes": label, "period": per, "n": m.n,
                   "clusters": m.n_clusters, "intercept": m.beta[0],
                   "slope": slope, "se": se, "p_vs_zero": m.p[1], "r2": m.r2}
            for tag, bound in [("strict", C.BOUND_STRICT / N_IN),
                               ("holder", C.BOUND_HOLDER / N_IN)]:
                t = (slope - bound) / se
                row[f"p_vs_{tag}"] = 2 * stats.t.sf(abs(t), m.df)
            rows.append(row)
    return pd.DataFrame(rows)


# Change in log R = Covert side + output side (a fall in EV counts positive).
def decomposition(spec: dict, idx: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = spec["summary"].loc[idx]
    lp = np.log(spec["p_in"][idx]) - np.log(s["p_pre"])
    le = np.log(spec["ev_used"][idx]) - np.log(s["ev_pre"])
    ok = lp.notna() & le.notna()
    lp, le = lp.where(ok), le.where(ok)
    daily = pd.DataFrame({"input": lp.mean(axis=1), "output": -le.mean(axis=1),
                          "n": ok.sum(axis=1)})
    daily["ratio"] = daily["input"] + daily["output"]

    gap = np.log(C.BOUND_STRICT) - np.log(s["r_pre"])
    rows = []
    for label, (lo, hi) in C.DECOMP_WINDOWS.items():
        a = lp.loc[lo:hi].mean()
        b = -le.loc[lo:hi].mean()
        keep = a.notna() & b.notna() & gap.notna()
        if not keep.any():
            continue
        inp, out, g = a[keep].mean(), b[keep].mean(), gap[keep].mean()
        rows.append({"window": label, "n": int(keep.sum()), "gap_pre": g,
                     "input": inp, "output": out, "ratio": inp + out,
                     "share_closed_pct": 100 * (inp + out) / g,
                     "input_share_pct": 100 * inp / (inp + out)})
    return pd.DataFrame(rows).set_index("window"), daily


# CARs of 04 by binding status; difference binding - slack as in 05
# (winsorised, clustered by pool, wild bootstrap), in log points.
def car_by_binding(s: pd.DataFrame) -> pd.DataFrame:
    path = C.CLEAN_DIR / "car_items.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run 04_abnormal_returns.py first.")
    car = pd.read_parquet(path)

    sub = s.loc[select(s)].copy()
    sub["status"] = np.where(sub["r_pre"].isna(), "no pre-event ratio",
                             np.where(sub["binding"], "binding", "slack"))
    order = ["binding", "slack", "no pre-event ratio"]
    rows = []
    for win in C.EVENT_WINDOWS:
        c = car.loc[car["window"] == win].set_index("market_hash_name")["car"]
        d = sub.assign(car=c.reindex(sub.index)).dropna(subset=["car"])
        if d.empty:
            log.warning("No CARs for window %s in %s", win, path.name)
            continue
        for status in order:
            g = d.loc[d["status"] == status, "car"]
            if g.empty:
                continue
            rows.append({"window": win, "status": status, "items": len(g),
                         "caar_pct": 100 * (np.exp(g.mean()) - 1),
                         "median_pct": 100 * (np.exp(g.median()) - 1),
                         "share_positive": (g > 0).mean()})

        lo_q, hi_q = d["car"].quantile([C.WINSOR, 1 - C.WINSOR])
        dd = d[d["status"].isin(["binding", "slack"])].copy()
        dd["car_w"] = dd["car"].clip(lo_q, hi_q)
        dd["binding_d"] = (dd["status"] == "binding").astype(float)
        X, names = design(dd, ["binding_d"])
        y, cl = dd["car_w"].to_numpy(float), cluster_ids(dd, f"binding - slack {win}")
        m = OLS(y, X, cl, names)
        j = names.index("binding_d")
        p_wild = wild_bootstrap(y, X, cl, j)
        rows.append({"window": win, "status": "difference binding - slack",
                     "items": m.n, "clusters": m.n_clusters,
                     "weights": weight_scheme(m.n_clusters),
                     "diff_log": m.beta[j], "se": m.se[j], "p": m.p[j],
                     "p_wild": p_wild, "p_star": star_p(m.p[j], p_wild, m.n_clusters)})
    return pd.DataFrame(rows)


# Half-life and settling level from log R(t) = y* + (y0 - y*) exp(-k t).
def _fit_path(med: pd.Series) -> tuple[float, float]:
    med = med.dropna()
    tt = med.index.to_numpy(float) - med.index[0]
    try:
        (ys, _, kap), _ = curve_fit(
            lambda t, ys, y0, kap: ys + (y0 - ys) * np.exp(-kap * t),
            tt, med.to_numpy(), p0=(med.iloc[-7:].mean(), med.iloc[0], 0.1),
            maxfev=10000)
    except RuntimeError:
        return np.nan, np.nan
    return (np.log(2) / kap if kap > 0 else np.nan), np.exp(ys)


# Speed on binding routes: share closed at t = 0 and day the bound is cleared (smoothed
# prices, as in 06_fig1_ratio_path), half-life and level fitted on HALF_LIFE_SMOOTH_DAYS prices.
def convergence(spec: dict, idx: pd.Index, bootstrap: bool = True) -> dict:
    lo, hi = C.CONVERGENCE_WINDOW
    s = spec["summary"]
    y = np.log(spec["ratio"].loc[lo:hi, idx])
    y_fit = np.log(spec["ratio_hl"].loc[lo:hi, idx])
    hl, r_star = _fit_path(y_fit.median(axis=1))
    out = {"routes": len(idx), "half_life_days": hl, "r_star": r_star}

    gap = np.log(C.BOUND_STRICT) - np.log(s.loc[idx, "r_pre"])
    step0 = y.loc[lo] - np.log(s.loc[idx, "r_pre"])
    k = step0.notna() & gap.notna()
    out["event_day_closed_pct"] = 100 * step0[k].mean() / gap[k].mean()

    below = 100 * (y < np.log(C.BOUND_STRICT)).sum(axis=1) / y.notna().sum(axis=1)
    ok = below <= C.BOUND_CLEARED_PCT
    cleared = ok[::-1].cummin()[::-1]  # true from here to the end
    out["bound_cleared_day"] = cleared.idxmax() if cleared.any() else np.nan

    if bootstrap:
        rng = np.random.default_rng(C.BOOTSTRAP_SEED)
        cl = pd.Series(cluster_ids(s.loc[idx], "half-life bootstrap"), index=idx)
        groups = {c: cl.index[cl == c] for c in cl.unique()}
        keys = list(groups)
        draws = []
        for _ in range(C.BOOTSTRAP_REPS):
            pick = rng.choice(len(keys), size=len(keys), replace=True)
            cols = np.concatenate([groups[keys[i]] for i in pick])
            draws.append(_fit_path(y_fit[cols].median(axis=1))[0])
        draws = np.array(draws, dtype=float)
        out["half_life_draws_finite"] = int(np.isfinite(draws).sum())
        draws = draws[np.isfinite(draws)]
        out["half_life_lo"], out["half_life_hi"] = np.percentile(draws, [2.5, 97.5])
    return out


# Routes with pre and post prices; dlogp winsorised within role (shared with 06_fig3_placebo).
def placebo_base(s: pd.DataFrame) -> pd.DataFrame:
    base = s[s["main"]].dropna(subset=["p_pre", "p_post", "r_pre"]).copy()
    base["dlogp"] = np.log(base["p_post"]) - np.log(base["p_pre"])
    for role in [TREATED, C.PLACEBO_ROLE]:
        m_ = base["role"] == role
        lo_q, hi_q = base.loc[m_, "dlogp"].quantile([C.WINSOR, 1 - C.WINSOR])
        base.loc[m_, "dlogp"] = base.loc[m_, "dlogp"].clip(lo_q, hi_q)
    return base


# Does the pre-event gap predict the price change? (a) kink at the bound, (b) value vs.
# own price. The placebo shares regression to the mean, so compare the groups.
def placebo_tests(s: pd.DataFrame) -> pd.DataFrame:
    base = placebo_base(s)
    base["log_r_pre"] = np.log(base["r_pre"])
    dist = base["log_r_pre"] - np.log(C.BOUND_STRICT)
    base["gap_below"] = dist.clip(upper=0)  # negative below the bound
    base["gap_above"] = dist.clip(lower=0)
    base["log_ev_pre"] = np.log(base["ev_pre"])
    base["log_p_pre"] = np.log(base["p_pre"])
    groups = {
        "treated, all": base["role"] == TREATED,
        "treated, StatTrak": (base["role"] == TREATED) & base["stattrak"],
        "placebo, StatTrak glove cases": base["role"] == C.PLACEBO_ROLE,
    }
    specs = [("(a) kink at the bound", ["gap_below", "gap_above"], None,
              ["gap_below", "gap_above"]),
             ("(b) value, price, wear FE", ["log_ev_pre", "log_p_pre"],
              ["wear"], ["log_ev_pre"])]
    rows = []
    for gname, mask in groups.items():
        d = base[mask].copy()
        cl = cluster_ids(d, gname)
        for sname, cols, fe, keys in specs:
            X, names = design(d, cols, fe)
            y = d["dlogp"].to_numpy(float)
            m = OLS(y, X, cl, names)
            for key in keys:
                j = names.index(key)
                t_one = (m.beta[j] + 1) / m.se[j]
                p_wild = wild_bootstrap(y, X, cl, j)
                rows.append({"group": gname, "spec": sname, "regressor": key,
                             "n": m.n, "clusters": m.n_clusters,
                             "weights": weight_scheme(m.n_clusters),
                             "coef": m.beta[j], "se": m.se[j], "p": m.p[j],
                             "p_wild": p_wild,
                             "p_star": star_p(m.p[j], p_wild, m.n_clusters),
                             "p_vs_minus1": (2 * stats.t.sf(abs(t_one), m.df)
                                             if key == "gap_below" else np.nan),
                             "r2": m.r2})

    # Difference-in-slopes test: treated StatTrak and placebo in one regression.
    d = base[groups["treated, StatTrak"]
             | groups["placebo, StatTrak glove cases"]].copy()
    d["treated"] = (d["role"] == TREATED).astype(float)
    y, cl = d["dlogp"].to_numpy(float), cluster_ids(d, "difference StatTrak - placebo")
    for sname, cols, fe, keys in specs:
        inter = []
        for c in cols:
            d[f"{c} x treated"] = d[c] * d["treated"]
            inter.append(f"{c} x treated")
        fe_cols = None
        reg = ["treated"] + cols + inter
        if fe:
            d["wear_group"] = d["wear"] + "_" + d["treated"].astype(int).astype(str)
            fe_cols = ["wear_group"]
            # wear_group already holds the group effect; keeping treated would be collinear.
            reg = cols + inter
        X, names = design(d, reg, fe_cols)
        m = OLS(y, X, cl, names)
        j = names.index(f"{keys[0]} x treated")
        p_wild = wild_bootstrap(y, X, cl, j)
        rows.append({"group": "difference: treated StatTrak - placebo",
                     "spec": sname, "regressor": f"{keys[0]} x treated",
                     "n": m.n, "clusters": m.n_clusters,
                     "weights": weight_scheme(m.n_clusters), "coef": m.beta[j],
                     "se": m.se[j], "p": m.p[j], "p_wild": p_wild,
                     "p_star": star_p(m.p[j], p_wild, m.n_clusters),
                     "p_vs_minus1": np.nan, "r2": m.r2})
    return pd.DataFrame(rows)


# Mean, CV and skewness of the output value per route at one set of prices.
def outcome_shape(weights: pd.DataFrame, px: pd.Series) -> pd.DataFrame:
    w = weights.assign(p=weights["output"].map(px)).dropna(subset=["p"])
    mass = w.groupby("route")["weight"].sum()
    w["q"] = w["weight"] / w["route"].map(mass)
    mean = (w["q"] * w["p"]).groupby(w["route"]).sum()
    dev = w["p"] - w["route"].map(mean)
    var = (w["q"] * dev ** 2).groupby(w["route"]).sum()
    m3 = (w["q"] * dev ** 3).groupby(w["route"]).sum()
    sd = np.sqrt(var)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = pd.DataFrame({"mass": mass, "mean": mean, "cv": sd / mean,
                            "skew": m3 / sd ** 3})
    out.loc[out["mass"] < C.MIN_PRICED_MASS, ["mean", "cv", "skew"]] = np.nan
    return out


# Do binding routes with a more lottery-like outcome (CV, skewness at pre-event
# prices) settle at a higher R? Regressor: percentile rank of the measure.
def overshoot_tests(weights: pd.DataFrame, raw: pd.DataFrame,
                    s: pd.DataFrame) -> pd.DataFrame:
    shape = outcome_shape(weights, window_prices(raw, C.LEVEL_PRE))
    # Check: the mean must equal ev_pre.
    rel = (shape["mean"].reindex(s.index) / s["ev_pre"] - 1).abs().max()
    if np.isfinite(rel) and rel > 1e-8:
        log.warning("outcome_shape: mean differs from ev_pre by up to %.2e", rel)

    d = (s.loc[select(s, binding=True)].join(shape[["cv", "skew"]])
         .dropna(subset=["r_post", "cv", "skew"]).copy())
    d["log_r_post"] = np.log(d["r_post"])
    rows = []
    for key, label in [("cv", "coefficient of variation"),
                       ("skew", "skewness")]:
        dd = d.assign(rank=(d[key].rank() - 1) / (len(d) - 1))
        X, names = design(dd, ["rank"])
        y, cl = dd["log_r_post"].to_numpy(float), cluster_ids(dd, f"overshoot {key}")
        m = OLS(y, X, cl, names)
        j = names.index("rank")
        p_wild = wild_bootstrap(y, X, cl, j)
        rows.append({"measure": label, "routes": m.n, "clusters": m.n_clusters,
                     "weights": weight_scheme(m.n_clusters),
                     "median_measure": dd[key].median(),
                     "spearman_rho": stats.spearmanr(dd[key], dd["r_post"])[0],
                     "coef_rank": m.beta[j], "se": m.se[j], "p": m.p[j],
                     "p_wild": p_wild, "p_star": star_p(m.p[j], p_wild, m.n_clusters)})
    return pd.DataFrame(rows)


def ratio_path(spec: dict, idx_bind: pd.Index, idx_slack: pd.Index,
               idx_placebo: pd.Index) -> pd.DataFrame:
    lo, hi = C.PATH_WINDOW
    r = spec["ratio"].loc[lo:hi]
    out = pd.DataFrame({
        "median_binding": r[idx_bind].median(axis=1),
        "median_slack": r[idx_slack].median(axis=1),
        "median_placebo": r[idx_placebo].median(axis=1),
        "below_bound_binding_pct": 100 * (r[idx_bind] < C.BOUND_STRICT).sum(axis=1)
        / r[idx_bind].notna().sum(axis=1),
        "n_binding": r[idx_bind].notna().sum(axis=1),
    })
    out.index.name = "t"
    return out


def key_numbers(spec: dict, idx: pd.Index) -> dict:
    s = spec["summary"].loc[idx].dropna(subset=["r_post"])
    b = s[s["binding"]]
    lvl = level_test(spec["summary"], idx)
    slope = lvl.query("routes == 'all' and period == 'post'")["slope"].iloc[0]
    # Convergence on all binding routes of idx, as in section 4 (not only those with r_post).
    sa = spec["summary"].loc[idx]
    cv = convergence(spec, sa.index[sa["binding"].astype(bool) & sa["r_pre"].notna()],
                     bootstrap=False)
    return {"routes": len(s), "binding": len(b),
            "median_r_post_all": s["r_post"].median(),
            "median_r_post_binding": b["r_post"].median(),
            "binding_below_pct": 100 * (b["r_post"] < C.BOUND_STRICT).mean(),
            "binding_band_pct": 100 * b["r_post"].between(
                C.BOUND_STRICT, C.BOUND_HOLDER).mean(),
            "level_slope_post": slope,
            "event_day_closed_pct": cv["event_day_closed_pct"],
            "bound_cleared_day": cv["bound_cleared_day"],
            "half_life_days": cv["half_life_days"],
            "r_star": cv["r_star"]}


# Plain numbers (0.5, 1, 2) on a log axis.
def plain_log(axis) -> None:
    from matplotlib.ticker import FuncFormatter, NullFormatter
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def shade_band(ax) -> None:
    ax.axhspan(C.BOUND_STRICT, C.BOUND_HOLDER, color=ps.GRID, alpha=0.7,
               linewidth=0, zorder=0)
    ax.axhline(C.BOUND_STRICT, color=ps.NEUTRAL, linestyle="--", linewidth=1)


# Dotted line at t = +7: end of the trade hold on purchases from t = 0.
def event_lines(ax) -> None:
    ax.axvline(0, color=ps.NEUTRAL, linewidth=1)
    ax.axvline(C.TRADE_HOLD_DAYS, color=ps.NEUTRAL, linewidth=0.8,
               linestyle=":")


def minus(v) -> str:
    return f"{v:g}".replace("-", "−")


X_LABEL_DAYS = f"Event time t (days relative to {C.T0_TEXT})"
BOUND_TEXT = f"{C.BOUND_STRICT:.3f}"
BAND_LABEL = (f"Band from the bound ({C.BOUND_STRICT:.3f}) to the holder line "
              f"({C.BOUND_HOLDER:.1f})")


def fig_ratio_path(path: pd.DataFrame) -> None:
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.0), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1]})
    shade_band(ax)
    ax.plot(path.index, path["median_binding"], color=ps.color(TREATED),
            label="Input, binding")
    ax.plot(path.index, path["median_slack"], color=ps.color(TREATED),
            linestyle="--", linewidth=1.1, alpha=0.8, label="Input, slack")
    ax.plot(path.index, path["median_placebo"], color=ps.color(C.PLACEBO_ROLE),
            label=f"{ps.label(C.PLACEBO_ROLE)} (pseudo route)")
    event_lines(ax)
    ax.set_yscale("log")
    ax.set_yticks([0.2, 0.3, 0.5, round(C.BOUND_STRICT, 2), C.BOUND_HOLDER, 2, 3])
    plain_log(ax.yaxis)
    ax.set_ylabel("Median R = 5 × Covert price / expected value")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=ps.GRID))
    labels.append(BAND_LABEL)
    ax.legend(handles, labels, loc="lower right", fontsize=8)

    ax2.plot(path.index, path["below_bound_binding_pct"],
             color=ps.color(TREATED))
    event_lines(ax2)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel(f"Binding routes\nbelow {BOUND_TEXT} (%)")
    ax2.set_xlabel(X_LABEL_DAYS)
    # The CSV keeps the full PATH_WINDOW.
    ax2.set_xlim(*C.PATH_FIG_WINDOW)
    fig.tight_layout()
    ps.save(fig, "06_fig1_ratio_path", C.OUT_FIGURES)


def fig_decomposition(daily: pd.DataFrame, gap: float) -> None:
    d = daily.loc[C.PATH_WINDOW[0]:C.PATH_WINDOW[1]]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(d.index, d["input"], color=ps.color(TREATED),
            label="Covert prices rose")
    ax.plot(d.index, d["output"], color=ps.color("treated_output"),
            label="Expected output value fell")
    ax.plot(d.index, d["ratio"], color=ps.NEUTRAL, linestyle="--",
            label="Total change in log R")
    ax.axhline(gap, color=ps.NEUTRAL, linestyle=":", linewidth=1)
    # Label at the right end, below the line, where no series crosses it.
    ax.text(d.index.max(), gap, f"gap to {BOUND_TEXT} before the update",
            ha="right", va="top", fontsize=8, color=ps.NEUTRAL)
    ax.axhline(0, color=ps.NEUTRAL, linewidth=0.8)
    event_lines(ax)
    ax.set_xlabel(X_LABEL_DAYS)
    ax.set_ylabel("Change since the pre-event window (log points)")
    ax.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    ps.save(fig, "06_fig2_decomposition", C.OUT_FIGURES)


def fig_placebo(s: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    base = placebo_base(s)
    for role, label in [(TREATED, ps.label(TREATED)),
                        (C.PLACEBO_ROLE, ps.label(C.PLACEBO_ROLE)
                         + " (pseudo route)")]:
        d = base[base["role"] == role]
        x = np.log(d["r_pre"]) - np.log(C.BOUND_STRICT)
        y = d["dlogp"]
        ax.scatter(d["r_pre"], y, s=12, alpha=0.55, color=ps.color(role),
                   edgecolor="none", label=label)
        # Kinked fit as in specification (a) of the placebo table.
        X = np.column_stack([np.ones(len(x)), x.clip(upper=0), x.clip(lower=0)])
        a, b_lo, b_hi = np.linalg.lstsq(X, y.to_numpy(), rcond=None)[0]
        xs = np.linspace(x.min(), x.max(), 200)
        ys = a + b_lo * np.clip(xs, None, 0) + b_hi * np.clip(xs, 0, None)
        ax.plot(C.BOUND_STRICT * np.exp(xs), ys, color=ps.color(role),
                linewidth=1.6)
    ax.axvline(C.BOUND_STRICT, color=ps.NEUTRAL, linestyle="--", linewidth=1,
               label=f"Bound ({C.BOUND_STRICT:.3f})")
    ax.axhline(0, color=ps.NEUTRAL, linewidth=0.8)
    ax.set_xscale("log")
    plain_log(ax.xaxis)
    ax.set_xlabel("R before the update (log scale)")
    ax.set_ylabel(f"Change in log price, days {minus(C.LEVEL_PRE[0])} to "
                  f"{minus(C.LEVEL_PRE[1])}\nvs. days {C.LEVEL_POST[0]} to "
                  f"{C.LEVEL_POST[1]}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    ps.save(fig, "06_fig3_placebo", C.OUT_FIGURES)


def fig_by_wear(s: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    shade_band(ax)
    pos = np.arange(len(WEAR_ORDER))
    for off, per, col in [(-0.2, "pre", ps.GRID), (0.2, "post", ps.color(TREATED))]:
        data = [s.loc[s["wear"] == w, f"r_{per}"].dropna() for w in WEAR_ORDER]
        bp = ax.boxplot(data, positions=pos + off, widths=0.35, whis=(5, 95),
                        showfliers=False, patch_artist=True)
        for box in bp["boxes"]:
            box.set_facecolor(col)
            box.set_edgecolor(ps.NEUTRAL)
        for med in bp["medians"]:
            med.set_color(ps.NEUTRAL)
    ax.set_yscale("log")
    ax.set_ylim(0.02, 80)
    plain_log(ax.yaxis)
    ax.set_xticks(pos)
    ax.set_xticklabels(WEAR_ORDER, fontsize=8.5)
    ax.set_ylabel("R = 5 × Covert price / expected value")
    ax.legend([plt.Rectangle((0, 0), 1, 1, facecolor=ps.GRID),
               plt.Rectangle((0, 0), 1, 1, facecolor=ps.color(TREATED))],
              [f"before (days {minus(C.LEVEL_PRE[0])} to "
               f"{minus(C.LEVEL_PRE[1])})",
               f"after (days {C.LEVEL_POST[0]} to {C.LEVEL_POST[1]})"],
              loc="upper center", ncol=2, fontsize=8)
    fig.tight_layout()
    ps.save(fig, "06_fig4_ratio_by_wear", C.OUT_FIGURES)


def main() -> None:
    setup_logging()
    ps.apply_style()
    C.OUT_TABLES.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 160)

    routes, pools = load_routes()
    names = set(routes["market_hash_name"]) | set(pools["market_hash_name"])
    raw, prices = daily_prices(names)
    prices_hl = smooth_prices(raw, C.HALF_LIFE_SMOOTH_DAYS)
    log.info("Routes: %d (%d treated, %d placebo) | price matrix %d days x %d items",
             len(routes), (routes["role"] == TREATED).sum(),
             (routes["role"] == C.PLACEBO_ROLE).sum(), *prices.shape)
    log.info("Float rule: %s | input float: %s | bound %.3f, holder line %.2f",
             C.FLOAT_RULE, C.INPUT_FLOAT, C.BOUND_STRICT, C.BOUND_HOLDER)

    section("1. ROUTES: WHAT FIVE COPIES OF A COVERT PRODUCE")
    main_spec = run_spec(routes, pools, raw, prices, prices_hl=prices_hl)
    s = main_spec["summary"]
    w = main_spec["weights"]
    unmapped = w.loc[w["output"].isna(), "weight"].sum() / len(routes)
    log.info("Probability on outcomes missing from the pool table: %.4f %%",
             100 * unmapped)
    idx_t = select(s)
    log.info("Main treated routes: %d | priced mass median pre %.3f, post %.3f | "
             "valid R pre %d, post %d",
             len(idx_t), s.loc[idx_t, "mass_pre"].median(),
             s.loc[idx_t, "mass_post"].median(),
             s.loc[idx_t, "r_pre"].notna().sum(), s.loc[idx_t, "r_post"].notna().sum())
    wear_tab = output_wear_table(w, s)
    log.info("Outcome wear tier by input wear (% of probability, main routes):")
    show(wear_tab)
    wear_tab.to_csv(C.OUT_TABLES / "06_output_wear.csv", encoding="utf-8")

    ex = s.loc[[e for e in C.TRADEUP_EXAMPLES if e in s.index],
               ["case", "p_pre", "ev_pre", "r_pre", "p_post", "ev_post", "r_post"]]
    log.info("")
    log.info("Example routes (EUR):")
    show(ex.round(3))

    section("2. THE BOUND BEFORE AND AFTER THE UPDATE")
    bt = bound_table(s, idx_t)
    show(bt.round(2))
    bt.to_csv(C.OUT_TABLES / "06_bound_test.csv", encoding="utf-8")
    s.to_csv(C.OUT_TABLES / "06_routes.csv", encoding="utf-8")

    section("3. LEVEL RELATION WITH THE EXACT EXPECTED VALUE")
    lvl = level_test(s, idx_t)
    show(lvl.round(4))
    lvl.to_csv(C.OUT_TABLES / "06_level_test.csv", index=False, encoding="utf-8")
    log.info("Reference slopes: %.3f (strict bound), %.3f (holder line)",
             C.BOUND_STRICT / N_IN, C.BOUND_HOLDER / N_IN)

    section("4. CONVERGENCE: HOW FAST, AND FROM WHICH SIDE")
    idx_b = select(s, binding=True)
    idx_s = select(s, binding=False)
    idx_p = select(s, role=C.PLACEBO_ROLE)
    dec, daily = decomposition(main_spec, idx_b)
    show(dec.round(3))
    dec.to_csv(C.OUT_TABLES / "06_decomposition.csv", encoding="utf-8")
    hl = convergence(main_spec, idx_b)
    log.info("")
    log.info("Speed, %d binding routes, window %s:", hl["routes"],
             C.CONVERGENCE_WINDOW)
    log.info("  event day closes %.0f %% of the pre-event gap to the bound",
             hl["event_day_closed_pct"])
    log.info("  from day %s on, at most %.0f %% of the routes are below the bound",
             hl["bound_cleared_day"], C.BOUND_CLEARED_PCT)
    log.info("  afterwards R keeps rising: half-life %.1f days "
             "[%.1f, %.1f, pool bootstrap, %d finite draws], settles at R = %.3f "
             "(fit on prices smoothed over %d day(s))",
             hl["half_life_days"], hl["half_life_lo"], hl["half_life_hi"],
             hl["half_life_draws_finite"], hl["r_star"], C.HALF_LIFE_SMOOTH_DAYS)
    pd.DataFrame([hl]).to_csv(C.OUT_TABLES / "06_half_life.csv", index=False,
                              encoding="utf-8")
    # Check only: do bound cleared and the ratchet days move on the fit prices?
    hl_spec = dict(main_spec, ratio=main_spec["ratio_hl"], p_in=main_spec["p_in_hl"],
                   ev_used=main_spec["ev_used_hl"])
    hl_chk = convergence(hl_spec, idx_b, bootstrap=False)
    _, daily_hl = decomposition(hl_spec, idx_b)
    log.info("  check on prices smoothed over %d day(s): bound cleared from day %s "
             "(main %s); event day closes %.0f %%", C.HALF_LIFE_SMOOTH_DAYS,
             hl_chk["bound_cleared_day"], hl["bound_cleared_day"],
             hl_chk["event_day_closed_pct"])
    ratchet = pd.DataFrame({"input_main": daily["input"], "output_main": daily["output"],
                            "input_fit": daily_hl["input"], "output_fit": daily_hl["output"]})
    show(ratchet.loc[C.RATCHET_DAYS].round(3))
    path = ratio_path(main_spec, idx_b, idx_s, idx_p)
    path.join(daily.add_prefix("decomp_")).to_csv(
        C.OUT_TABLES / "06_path.csv", encoding="utf-8")

    section("4b. EVENT-WINDOW CARs OF THE INPUT BY BINDING STATUS (CARs FROM 04)")
    cb = car_by_binding(s)
    show(cb.set_index(["window", "status"]).round(3))
    log.info("(CAAR and median as e^x - 1 in %, unwinsorised; difference in log "
             "points, CAR winsorised as in 05, clustered by pool, wild bootstrap)")
    cb.to_csv(C.OUT_TABLES / "06_car_by_binding.csv", index=False,
              encoding="utf-8")

    section("5. PLACEBO: DOES THE GAP PREDICT PRICES WITHOUT CRAFTING?")
    pl = placebo_tests(s)
    show(pl.assign(sig=[stars(p) for p in pl["p"]]).round(3))
    pl.to_csv(C.OUT_TABLES / "06_placebo.csv", index=False, encoding="utf-8")

    section("5b. OVERSHOOT: WHY BINDING ROUTES SETTLE ABOVE THE HOLDER LINE")
    log.info("Binding routes by pool, R in the post window (days %d to %d):",
             *C.LEVEL_POST)
    pools_rows = [g for g in bt.index if g.startswith("binding: ")]
    show(bt.loc[pools_rows, ["n", "median_r_pre", "median_r_post",
                             "below_post_pct", "above_post_pct"]].round(2))
    log.info("")
    log.info("log R after the update on the shape of the outcome distribution "
             "(binding routes, outcome prices of days %d to %d):", *C.LEVEL_PRE)
    ov = overshoot_tests(main_spec["weights"], raw, s)
    show(ov.round(3))
    log.info("(coef_rank: difference in log R between the lowest and the "
             "highest route by the measure; clustered by pool, wild bootstrap. "
             "A taste for gambling predicts a positive coefficient.)")
    ov.to_csv(C.OUT_TABLES / "06_overshoot.csv", index=False, encoding="utf-8")

    section("6. ROBUSTNESS")
    variants = {
        "main": (main_spec, idx_t),
        "input float: best in tier": (
            run_spec(routes, pools, raw, prices, input_float="best",
                     base=main_spec, prices_hl=prices_hl),
            idx_t),
        "float rule: raw average": (
            run_spec(routes, pools, raw, prices, float_rule="raw",
                     base=main_spec, prices_hl=prices_hl),
            idx_t),
        f"ex post: knife prices at t+{C.TRADE_HOLD_DAYS}": (
            run_spec(routes, pools, raw, prices, ex_post=True,
                     base=main_spec, prices_hl=prices_hl), idx_t),
        "all input items": (main_spec, select(s, sample="all")),
        f"without routes > {C.CAP_SHARE_MAX:.0%} capped": (
            main_spec, select(s, cap_filter=True)),
    }
    rob = pd.DataFrame({k: key_numbers(sp, ix) for k, (sp, ix) in variants.items()}).T
    show(rob.round(3))
    rob.to_csv(C.OUT_TABLES / "06_robustness.csv", encoding="utf-8")

    section("7. FILES AND FIGURES")
    keep = routes["market_hash_name"].tolist()
    frames = {
        "price": main_spec["p_in"],
        "ev": main_spec["ev"],
        "ev_expost": main_spec["ev"].shift(-C.TRADE_HOLD_DAYS),
        "priced_mass": main_spec["evs"]["mass"][keep],
        "cap_share": main_spec["evs"]["cap"][keep],
        "ratio": main_spec["ratio"],
    }
    long = None
    for col, f in frames.items():
        f = f.copy()
        f.index.name = "t"
        part = f.reset_index().melt(id_vars="t", var_name="market_hash_name",
                                    value_name=col)
        long = part if long is None else long.merge(
            part, on=["t", "market_hash_name"])
    long = long.dropna(subset=["price", "ev"], how="all")
    long["role"] = long["market_hash_name"].map(s["role"])
    long.to_parquet(C.TRADEUP_EV, index=False)
    log.info("%s: %d rows", C.TRADEUP_EV.name, len(long))
    s.reset_index().to_parquet(C.TRADEUP_ROUTES, index=False)
    log.info("%s: %d routes", C.TRADEUP_ROUTES.name, len(s))
    main_spec["weights"].to_parquet(C.TRADEUP_WEIGHTS, index=False)
    log.info("%s: %d route-outcome pairs", C.TRADEUP_WEIGHTS.name,
             len(main_spec["weights"]))

    fig_ratio_path(path)
    # Gap line in 06_fig2_decomposition (thesis Figure D1): the decomposition window
    # that equals LEVEL_POST.
    post_key = next((k for k, w in C.DECOMP_WINDOWS.items()
                     if tuple(w) == tuple(C.LEVEL_POST)), None)
    gap = dec.loc[post_key, "gap_pre"] if post_key in dec.index else np.nan
    fig_decomposition(daily, gap)
    fig_placebo(s)
    fig_by_wear(s.loc[idx_t])
    log.info("06_fig1_ratio_path, 06_fig2_decomposition, 06_fig3_placebo, "
             "06_fig4_ratio_by_wear")
    log.info("")
    log.info("Done.")


if __name__ == "__main__":
    main()

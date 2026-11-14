# Abnormal returns: market model on trade-to-trade returns (Maynes-Rumsey scaling),
# CAARs per role and window with BMP, Kolari-Pynnoenen, rank and sign tests.
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

log = logging.getLogger("04_abnormal_returns")

# Rank test with the Corrado-Zivney variance: written to 04_rank_sensitivity only.
CZ_COLS = ["z_rank_cz", "p_rank_cz"]


def setup_logging() -> None:
    C.OUT_LOGS.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(C.OUT_LOGS / "04_abnormal_returns.log", mode="w",
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


def to_pct(x) -> float:
    return (np.exp(x) - 1) * 100


# Equal-weighted log-return index, cumulated to a level (no free float to weight by).
def build_index(panel: pd.DataFrame, items: pd.DataFrame,
                groups: list[str]) -> pd.Series:
    members = items.index[items["group"].isin(groups)]
    sub = panel[panel["market_hash_name"].isin(members)]
    daily = sub.groupby("t", observed=True)["ret_ffill"].mean()
    daily = daily.reindex(range(C.WINDOW[0], C.WINDOW[1] + 1)).fillna(0.0)
    return daily.cumsum()


# Per item: market model with and without alpha on returns scaled by sqrt(gap);
# the alpha = 0 model gets its own beta and sigma.
def estimate(panel: pd.DataFrame, items: pd.DataFrame, index: pd.Series,
             est_window: tuple[int, int]) -> pd.DataFrame:
    lo, hi = est_window
    r = panel[(panel["t"].between(lo, hi)) & panel["ret_obs"].notna()
              & (panel["gap"] <= C.MAX_GAP_RET)
              & panel["market_hash_name"].isin(items.index)].copy()

    lvl = index.to_dict()
    r["m"] = [lvl.get(t, np.nan) - lvl.get(t - int(g), np.nan)
              for t, g in zip(r["t"], r["gap"])]
    r = r.dropna(subset=["m"])

    rows = {}
    for name, g in r.groupby("market_hash_name", observed=True):
        n = len(g)
        if n < C.MIN_REG_OBS:
            continue
        w = np.sqrt(g["gap"].to_numpy())
        y = g["ret_obs"].to_numpy() / w
        xm = g["m"].to_numpy() / w

        # With alpha: R / sqrt(g) = alpha * sqrt(g) + beta * R_m / sqrt(g).
        X = np.column_stack([w, xm])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        sigma = resid.std(ddof=2)
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else np.nan

        # Without alpha: beta under the alpha = 0 restriction.
        beta0 = float(xm @ y / (xm @ xm)) if (xm @ xm) > 0 else np.nan
        resid0 = y - beta0 * xm
        # Uncentred: without an intercept the residual mean is not zero.
        sigma0 = float(np.sqrt((resid0 ** 2).sum() / (n - 1)))

        rows[name] = dict(n_obs=n, alpha=coef[0], beta=coef[1], sigma=sigma,
                          r2=r2, beta_alpha0=beta0, sigma_alpha0=sigma0)

    out = pd.DataFrame.from_dict(rows, orient="index")
    out["role"] = items.loc[out.index, "role"]
    log.info("Estimated: %d of %d items (minimum %d returns)",
             len(out), len(items), C.MIN_REG_OBS)
    return out


# Standardised residuals of a return model, for ranks, r-bar and the sign test.
# placebo_st_adj: main-model residual minus the day's mean of Placebo ST; market_adj and
# placebo_st_adj use the sigma of the model with alpha, as their CARs do.
def std_residuals(r: pd.DataFrame, params: pd.DataFrame,
                  model: str = C.MODEL_MAIN) -> np.ndarray:
    name = r["market_hash_name"]
    ret, g, m = r["ret_obs"].to_numpy(), r["gap"].to_numpy(), r["m"].to_numpy()
    if model == "mm_alpha0":
        e = ret - name.map(params["beta_alpha0"]).to_numpy() * m
        s = name.map(params["sigma_alpha0"]).to_numpy()
    elif model == "mm":
        e = (ret - name.map(params["alpha"]).to_numpy() * g
             - name.map(params["beta"]).to_numpy() * m)
        s = name.map(params["sigma"]).to_numpy()
    elif model == "market_adj":
        e = ret - m
        s = name.map(params["sigma"]).to_numpy()
    elif model == "placebo_st_adj":
        e0 = pd.Series(ret - name.map(params["beta_alpha0"]).to_numpy() * m,
                       index=r.index)
        bench = name.map(params["role"]).to_numpy() == C.PLACEBO_BENCHMARK_ROLE
        day_mean = e0[bench].groupby(r.loc[bench, "t"]).mean()
        e = (e0 - r["t"].map(day_mean)).to_numpy()
        s = name.map(params["sigma"]).to_numpy()
    else:
        raise ValueError(f"unknown model: {model}")
    return e / (s * np.sqrt(g))


# Standardised residuals of a model, day x item, for t in [lo, hi].
def model_residuals(panel: pd.DataFrame, params: pd.DataFrame, index: pd.Series,
                    lo: int, hi: int, model: str) -> pd.DataFrame:
    r = panel[(panel["t"].between(lo, hi)) & panel["ret_obs"].notna()
              & (panel["gap"] <= C.MAX_GAP_RET)
              & panel["market_hash_name"].isin(params.index)].copy()
    lvl = index.to_dict()
    r["m"] = [lvl.get(t, np.nan) - lvl.get(t - int(g), np.nan)
              for t, g in zip(r["t"], r["gap"])]
    r = r.dropna(subset=["m"])
    r["sar"] = std_residuals(r, params, model)
    return r.pivot_table(index="t", columns="market_hash_name", values="sar",
                         observed=True)


# Standardised estimation-window residuals of a model: input for r-bar and the sign test.
def residual_series(panel: pd.DataFrame, params: pd.DataFrame, index: pd.Series,
                    est_window: tuple[int, int], model: str = C.MODEL_MAIN
                    ) -> pd.DataFrame:
    return model_residuals(panel, params, index, *est_window, model)


# Mean residual correlation per role (KP r-bar), shared by the tables and 04_fig1_caar_path.
def rbar_by_role(params: pd.DataFrame, resid: pd.DataFrame,
                 roles: list[str]) -> dict[str, float]:
    out = {}
    for role in roles:
        cols = [c for c in params.index[params["role"] == role] if c in resid.columns]
        sub = resid[cols]
        out[role] = mean_correlation(sub) if sub.shape[1] > 2 else np.nan
    return out


# KP factor sqrt((1 - r) / (1 + (n - 1) r)); NaN (no correction) where r-bar is missing or <= 0.
def kp_factor(rbar: float, n):
    if rbar is None or not np.isfinite(rbar) or rbar <= 0:
        return np.nan * np.ones_like(np.asarray(n, dtype=float))
    denom = 1 + (np.asarray(n, dtype=float) - 1) * rbar
    return np.sqrt((1 - rbar) / denom)


# Mean pairwise correlation; NaN if no pair shares KP_MIN_OVERLAP days (illiquid variant).
def mean_correlation(resid: pd.DataFrame) -> float:
    if resid.shape[1] < 2:
        return float("nan")
    corr = resid.corr(min_periods=C.KP_MIN_OVERLAP).to_numpy().copy()
    np.fill_diagonal(corr, np.nan)
    if np.isnan(corr).all():
        return float("nan")
    return float(np.nanmean(corr))


# Last traded price and its day, per item and day (the holding period needs both).
def anchors(panel: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    sub = panel[panel["market_hash_name"].isin(items.index)]
    price = sub.pivot_table(index="t", columns="market_hash_name",
                            values="price", observed=True)
    price = price.reindex(range(C.WINDOW[0], C.WINDOW[1] + 1))
    day = price.notna().astype(float).mul(price.index.to_numpy(), axis=0)
    day = day.where(price.notna())
    return price.ffill(), day.ffill()


# CAR from the last trade before the window to the last trade inside it.
def car_for_window(price_ff: pd.DataFrame, day_ff: pd.DataFrame,
                   params: pd.DataFrame, index: pd.Series,
                   window: tuple[int, int], model: str) -> pd.DataFrame:
    t1, t2 = window
    names = params.index

    p_anchor = price_ff.loc[t1 - 1, names]
    d_anchor = day_ff.loc[t1 - 1, names]
    p_end = price_ff.loc[t2, names]
    d_end = day_ff.loc[t2, names]

    raw = np.log(p_end / p_anchor)
    days = d_end - d_anchor
    m = d_end.map(index) - d_anchor.map(index)

    if model == "mm":
        alpha, beta = params["alpha"], params["beta"]
    elif model in ("mm_alpha0", "placebo_st_adj"):
        # placebo_st_adj: CAR of the main model; caar_table subtracts the Placebo ST mean.
        alpha, beta = 0.0, params["beta_alpha0"]
    elif model == "market_adj":
        alpha, beta = 0.0, 1.0
    else:
        raise ValueError(f"unknown model: {model}")

    ar = raw - alpha * days - beta * m

    # market_adj and placebo_st_adj borrow the market-model sigma: t-values only indicative.
    sigma = params["sigma_alpha0"] if model == "mm_alpha0" else params["sigma"]

    # scale lets placebo_st_adj restandardise its adjusted CAR.
    scale = sigma * np.sqrt(days.where(days > 0))
    out = pd.DataFrame({
        "car": ar, "raw": raw, "days": days, "scale": scale,
        "sar": ar / scale,
        "role": params["role"],
    })
    return out[out["car"].notna() & np.isfinite(out["car"])]


# Rank test (Corrado 1989; Corrado and Zivney 1992): ranks of each item's standardised
# returns over [EST_WINDOW[0], t2], U = rank / (M + 1) - 0.5. Main statistic: variance of the
# daily mean U in the estimation window (Section 4.4); z_rank_cz: Corrado-Zivney variance
# over estimation and event window, days weighted by N_t (sensitivity only).
def rank_test(sar: pd.DataFrame, window: tuple[int, int]) -> dict:
    t1, t2 = window
    sub = sar.loc[C.EST_WINDOW[0]:t2]
    U = sub.rank(axis=0) / (sub.count(axis=0) + 1) - 0.5
    daily_mean = U.mean(axis=1)
    n_t = U.notna().sum(axis=1)
    est = daily_mean.loc[slice(*C.EST_WINDOW)].dropna()
    win = daily_mean.loc[t1:t2].dropna()
    out = {}
    if len(est) <= C.RANK_MIN_EST_DAYS or len(win) == 0:
        return out
    s = est.std(ddof=1)
    z = np.sqrt(len(win)) * win.mean() / s if s > 0 else np.nan
    out["z_rank"] = z
    out["p_rank"] = 2 * stats.norm.sf(abs(z))
    days = est.index.union(win.index)
    weighted = np.sqrt(n_t.loc[days]) * daily_mean.loc[days]
    s_cz = np.sqrt((weighted ** 2).mean())
    z_cz = weighted.loc[win.index].sum() / (np.sqrt(len(win)) * s_cz) if s_cz > 0 else np.nan
    out["z_rank_cz"] = z_cz
    out["p_rank_cz"] = 2 * stats.norm.sf(abs(z_cz))
    return out


def test_stats(car: pd.DataFrame, rbar: float, sar_wide: pd.DataFrame | None,
               window: tuple[int, int], est_pos_share: float | None) -> dict:
    n = len(car)
    # n: all items with a CAR; n_sar: those that traded in the window (days > 0).
    out = {"n": n, "n_sar": int(car["sar"].notna().sum()),
           "caar_pct": to_pct(car["car"].mean()),
           "median_pct": to_pct(car["car"].median())}
    if n < 3:
        return out

    t_plain = car["car"].mean() / (car["car"].std(ddof=1) / np.sqrt(n))
    out["t"] = t_plain
    out["p_t"] = 2 * stats.t.sf(abs(t_plain), n - 1)

    sar = car["sar"].dropna()
    if len(sar) >= 3:
        t_bmp = sar.mean() / (sar.std(ddof=1) / np.sqrt(len(sar)))
        out["t_bmp"] = t_bmp
        out["p_bmp"] = 2 * stats.t.sf(abs(t_bmp), len(sar) - 1)

        # KP only for r-bar > 0 (not for the benchmark itself, whose residuals offset).
        out["rbar"] = rbar
        adj = float(kp_factor(rbar, len(sar)))
        if np.isfinite(adj):
            out["t_kp"] = t_bmp * adj
            out["p_kp"] = 2 * stats.t.sf(abs(out["t_kp"]), len(sar) - 1)

    # Rank test: mean rank in the window vs. its variability in the estimation window.
    if sar_wide is not None and not sar_wide.empty:
        cols = [c for c in car.index if c in sar_wide.columns]
        if len(cols) >= 3:
            out.update(rank_test(sar_wide[cols], window))

    # Generalised sign test against the estimation-window share of positive residuals;
    # only items with a trade in the window (CAR = 0 is no observation).
    if est_pos_share is not None and 0 < est_pos_share < 1:
        traded = car["days"] > 0
        n_s = int(traded.sum())
        if n_s >= 3:
            pos = int((car.loc[traded, "car"] > 0).sum())
            z = ((pos / n_s - est_pos_share) * np.sqrt(n_s)
                 / np.sqrt(est_pos_share * (1 - est_pos_share)))
            out["z_sign"] = z
            out["p_sign"] = 2 * stats.norm.sf(abs(z))
    return out


# Standardised returns (day x item) from the start of the estimation window on; the rank
# test ranks them per window (rank_test).
def sar_matrix(panel: pd.DataFrame, params: pd.DataFrame, index: pd.Series,
               model: str = C.MODEL_MAIN) -> pd.DataFrame:
    return model_residuals(panel, params, index, C.EST_WINDOW[0], C.WINDOW[1], model)


# CAAR and all tests per role and window (EVENT_WINDOWS unless windows is given).
def caar_table(price_ff, day_ff, params, index, sar_wide, resid,
               roles: list[str], model: str,
               windows: dict[str, tuple[int, int]] | None = None) -> pd.DataFrame:
    rbar = rbar_by_role(params, resid, roles)
    pos_share = {}
    for role in roles:
        cols = params.index[params["role"] == role]
        sub = resid[[c for c in cols if c in resid.columns]]
        # Count only observed residuals (non-trading days are NaN).
        v = sub.to_numpy(dtype=float).ravel()
        v = v[~np.isnan(v)]
        pos_share[role] = float((v > 0).mean()) if v.size else 0.5

    rows = []
    for label, window in (windows or C.EVENT_WINDOWS).items():
        car = car_for_window(price_ff, day_ff, params, index, window, model)
        for role in roles:
            sub = car[car["role"] == role]
            if sub.empty:
                continue
            if model == "placebo_st_adj":
                ctrl = car[car["role"] == C.PLACEBO_BENCHMARK_ROLE]["car"].mean()
                # Restandardise, so that BMP and KP test the adjusted CAR.
                sub = sub.assign(car=sub["car"] - ctrl)
                sub = sub.assign(sar=sub["car"] / sub["scale"])
            st = test_stats(sub, rbar.get(role), sar_wide, window, pos_share.get(role))
            rows.append({"model": model, "window": label, "role": role, **st})
    return pd.DataFrame(rows)


# Daily CAAR per role with the same anchor as the window table.
def caar_path(price_ff, day_ff, params, index, model: str) -> pd.DataFrame:
    t1 = C.EVENT_WINDOWS["-1_+1"][0]
    p_anchor = price_ff.loc[t1 - 1]
    d_anchor = day_ff.loc[t1 - 1]

    if model == "mm":
        alpha, beta = params["alpha"], params["beta"]
    elif model == "mm_alpha0":
        alpha, beta = 0.0, params["beta_alpha0"]
    elif model == "market_adj":
        alpha, beta = 0.0, 1.0
    else:
        raise ValueError(f"caar_path does not support model {model}")

    rows = []
    for t in range(C.PLOT_WINDOW_LONG[0], C.PLOT_WINDOW_LONG[1] + 1):
        raw = np.log(price_ff.loc[t] / p_anchor)
        days = day_ff.loc[t] - d_anchor
        m = day_ff.loc[t].map(index) - d_anchor.map(index)
        ar = raw - alpha * days - beta * m
        df = pd.DataFrame({"ar": ar, "role": params["role"]}).dropna()
        g = df.groupby("role", observed=True)["ar"]
        rows.append(pd.DataFrame({"t": t, "caar": g.mean(), "median": g.median(),
                                  "se": g.std(ddof=1) / np.sqrt(g.count()),
                                  "n": g.count()}).reset_index())
    return pd.concat(rows, ignore_index=True)


# One equal-weighted portfolio per role: immune to cross-correlation, less power.
def portfolio_test(panel, items, index, roles) -> pd.DataFrame:
    rows = []
    daily_index = index.diff()
    for role in roles:
        members = items.index[items["role"] == role]
        sub = panel[panel["market_hash_name"].isin(members)]
        port = sub.groupby("t", observed=True)["ret_ffill"].mean()
        df = pd.DataFrame({"r": port, "m": daily_index}).dropna()
        est = df.loc[slice(*C.EST_WINDOW)]
        if len(est) < C.MIN_REG_OBS:
            continue
        beta = float(est["m"] @ est["r"] / (est["m"] @ est["m"]))
        e = est["r"] - beta * est["m"]
        g0 = e.var(ddof=1)
        for label, (t1, t2) in C.EVENT_WINDOWS.items():
            win = df.loc[t1:t2]
            L = len(win)
            car = float((win["r"] - beta * win["m"]).sum())
            # Variance of a sum of L days: sum over |k| < L of (L - |k|) gamma_k.
            q = min(L - 1, C.PORTFOLIO_ACOV_LAGS)
            var = L * g0 + 2 * sum((L - k) * e.autocorr(k) * g0 for k in range(1, q + 1))
            if var <= 0:
                log.info("Portfolio %s, window %s: autocovariance variance not positive, "
                         "using gamma0 * L", role, label)
                var = L * g0
            se = np.sqrt(var)
            t_stat = car / se if se > 0 else np.nan
            rows.append({"role": role, "window": label, "beta": beta,
                         "caar_pct": to_pct(car), "t": t_stat,
                         "p": 2 * stats.norm.sf(abs(t_stat))})
    return pd.DataFrame(rows)


# CAAR path; the band is widened by the KP factor, consistent with t_KP.
def fig_caar_path(path: pd.DataFrame, rbar: dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    lo, hi = C.PLOT_WINDOW_SHORT
    for role in C.PLOT_ROLES_MAIN:
        sub = path[(path["role"] == role) & path["t"].between(lo, hi)]
        if sub.empty:
            continue
        col = ps.color(role)
        ax.plot(sub["t"], to_pct(sub["caar"]), color=col, label=ps.label(role))
        if role in C.PLOT_BAND_ROLES:
            adj = kp_factor(rbar.get(role), sub["n"].to_numpy(dtype=float))
            half = 1.96 * sub["se"].to_numpy(dtype=float) / adj
            ax.fill_between(sub["t"], to_pct(sub["caar"] - half),
                            to_pct(sub["caar"] + half),
                            color=col, alpha=0.15, linewidth=0)
    ax.axvline(0, color=ps.NEUTRAL, linewidth=1)
    ax.axhline(0, color=ps.GRID, linewidth=1)
    ax.set_xlabel("Event time t (days relative to October 23, 2025)")
    ax.set_ylabel("CAAR (%)")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    ps.save(fig, "04_fig1_caar_path", C.OUT_FIGURES)


# Labels for 04_fig2_models.
MODEL_NAMES = {
    "mm_alpha0": "Market model, α = 0",
    "mm": "Market model",
    "market_adj": "Market-adjusted (β = 1)",
    "placebo_st_adj": "Relative to Placebo ST",
}
WINDOW_NAMES = {
    "0": "t = 0",
    "-1_+1": "[−1, +1]",
    "-1_+5": "[−1, +5]",
    "-1_+30": "[−1, +30]",
    "-1_+60": "[−1, +60]",
}


def fig_models(models: pd.DataFrame) -> None:
    roles = [r for r in C.PLOT_ROLES_MAIN]
    windows = list(C.EVENT_WINDOWS)
    fig, axes = plt.subplots(1, len(C.RETURN_MODELS),
                             figsize=(11, 3.6), sharey=True)
    x = np.arange(len(windows))
    width = 0.8 / len(roles)
    for ax, model in zip(np.atleast_1d(axes), C.RETURN_MODELS):
        sub = models[models["model"] == model]
        for k, role in enumerate(roles):
            vals = [sub[(sub["window"] == w) & (sub["role"] == role)]["caar_pct"].squeeze()
                    for w in windows]
            vals = [v if np.ndim(v) == 0 else np.nan for v in vals]
            # Bars centred on the tick: offsets symmetric around zero.
            ax.bar(x + (k - (len(roles) - 1) / 2) * width, vals, width, color=ps.color(role),
                   label=ps.label(role) if model == C.RETURN_MODELS[0] else None)
        ax.axhline(0, color=ps.NEUTRAL, linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([WINDOW_NAMES.get(w, w) for w in windows], rotation=45, ha="right", fontsize=8)
        ax.set_title(MODEL_NAMES.get(model, model), fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel("CAAR (%)")
    fig.tight_layout(rect=(0, 0.12, 1, 0.97))
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    ps.save(fig, "04_fig2_models", C.OUT_FIGURES)


def main() -> None:
    setup_logging()
    ps.apply_style()
    C.OUT_TABLES.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(C.PANEL_CLEAN,
                            columns=["market_hash_name", "t", "price",
                                     "ret_obs", "ret_ffill", "gap"])
    items_all = pd.read_parquet(C.ITEMS_CLEAN).set_index("market_hash_name")
    samples = pd.read_parquet(C.SAMPLES).set_index("market_hash_name")
    items = items_all.loc[samples.index[samples["main"]]]
    roles = [r for r in C.ROLE_ORDER if r in set(items["role"])]
    log.info("Main sample: %d items | model: %s | index: %s",
             len(items), C.MODEL_MAIN, C.INDEX_MAIN)

    section("1. BENCHMARK AND ESTIMATION")
    index = build_index(panel, items, C.INDEX_DEFS[C.INDEX_MAIN])
    log.info("Index level at t=0: %.1f %%, at t=+60: %.1f %% (relative to t=-200)",
             to_pct(index.loc[0]), to_pct(index.loc[60]))

    params = estimate(panel, items, index, C.EST_WINDOW)
    diag = params.groupby("role").agg(
        items=("beta", "size"), alpha_pct=("alpha", "median"),
        beta=("beta", "median"), beta_alpha0=("beta_alpha0", "median"),
        r2=("r2", "median"), sigma_pct=("sigma", "median"))
    diag["alpha_pct"] *= 100
    diag["sigma_pct"] *= 100
    show(diag.reindex([r for r in roles if r in diag.index]).round(3))
    diag.to_csv(C.OUT_TABLES / "04_estimation_diagnostics.csv", encoding="utf-8")

    # Residuals per return model: r-bar, ranks and the sign test use the model's own.
    resid_by = {m: residual_series(panel, params, index, C.EST_WINDOW, m)
                for m in C.RETURN_MODELS}
    sar_by = {m: sar_matrix(panel, params, index, m) for m in C.RETURN_MODELS}
    resid, sar_wide = resid_by[C.MODEL_MAIN], sar_by[C.MODEL_MAIN]
    price_ff, day_ff = anchors(panel, items)

    section("2. MAIN SPECIFICATION")
    main_tab = caar_table(price_ff, day_ff, params, index, sar_wide, resid,
                          roles, C.MODEL_MAIN)
    sens = [main_tab]
    main_tab = main_tab.drop(columns=CZ_COLS, errors="ignore")
    main_tab.to_csv(C.OUT_TABLES / "04_caar_main.csv", index=False, encoding="utf-8")

    for label in C.EVENT_WINDOWS:
        sub = main_tab[main_tab["window"] == label].set_index("role")
        cols = ["n", "n_sar", "caar_pct", "median_pct", "rbar", "t_bmp",
                "t_kp", "p_kp", "z_rank", "z_sign"]
        cols = [c for c in cols if c in sub.columns]
        view = sub.reindex([r for r in roles if r in sub.index])[cols].round(3)
        view["sig(KP)"] = [stars(p) for p in sub.reindex(view.index).get(
            "p_kp", pd.Series(np.nan, index=view.index))]
        log.info("")
        log.info("Window %s", label)
        show(view)

    section("2b. PRE-TREND TEST")
    # Pre-trend test, main model and estimated alpha (with alpha = 0 an estimation-window
    # drift would look like a pre-trend). Own file, so 04_caar_main stays unchanged.
    pre = pd.concat([caar_table(price_ff, day_ff, params, index, sar_by[m], resid_by[m],
                                roles, m, windows=C.PRE_WINDOWS)
                     for m in (C.MODEL_MAIN, "mm")], ignore_index=True)
    sens.append(pre[pre["model"] == C.MODEL_MAIN])
    pre = pre.drop(columns=CZ_COLS, errors="ignore")
    pre.to_csv(C.OUT_TABLES / "04_pretrend.csv", index=False, encoding="utf-8")

    # Rank test with the Corrado-Zivney variance: sensitivity only, not in the test standard.
    sens = pd.concat(sens, ignore_index=True)[
        ["model", "window", "role", "n_sar", "z_rank", "p_rank"] + CZ_COLS]
    sens.to_csv(C.OUT_TABLES / "04_rank_sensitivity.csv", index=False, encoding="utf-8")
    log.info("")
    log.info("Rank test, Corrado-Zivney variance (sensitivity, 04_rank_sensitivity.csv):")
    show(sens.set_index(["window", "role"])[["z_rank", "z_rank_cz", "p_rank_cz"]].round(3))
    cols = ["n_sar", "caar_pct", "median_pct", "t_bmp", "t_kp", "p_kp",
            "z_rank", "p_rank"]
    for (m, label), sub in pre.groupby(["model", "window"], sort=False):
        sub = sub.set_index("role")
        view = sub.reindex([r for r in roles if r in sub.index])
        view = view[[c for c in cols if c in view.columns]].round(3)
        view["sig(KP)"] = [stars(p) for p in sub.reindex(view.index).get(
            "p_kp", pd.Series(np.nan, index=view.index))]
        log.info("")
        log.info("Model %s, window %s", m, label)
        show(view)

    section("3. ROBUSTNESS: RETURN MODELS")
    models = pd.concat([caar_table(price_ff, day_ff, params, index, sar_by[m],
                                   resid_by[m], roles, m) for m in C.RETURN_MODELS],
                       ignore_index=True).drop(columns=CZ_COLS, errors="ignore")
    rb = pd.DataFrame({m: rbar_by_role(params, resid_by[m], roles)
                       for m in C.RETURN_MODELS})
    log.info("r-bar of the residuals per return model:")
    show(rb.round(4))
    models.to_csv(C.OUT_TABLES / "04_robustness_models.csv", index=False,
                  encoding="utf-8")
    piv = models[models["window"] == "-1_+60"].pivot(index="role", columns="model",
                                                     values="caar_pct")
    show(piv.reindex([r for r in roles if r in piv.index]).round(1))

    section("4. ROBUSTNESS: INDICES AND SAMPLE VARIANTS")
    grid = []
    for idx_name, groups in C.INDEX_DEFS.items():
        idx = build_index(panel, items, groups)
        pr = estimate(panel, items, idx, C.EST_WINDOW)
        # Residuals per benchmark, so that r-bar matches the index used.
        rs = residual_series(panel, pr, idx, C.EST_WINDOW)
        tab = caar_table(price_ff, day_ff, pr, idx, None, rs, roles, C.MODEL_MAIN)
        grid.append(tab.assign(index=idx_name, variant="main"))

    for variant in [v for v in C.SAMPLE_VARIANTS if v != "main"]:
        names = samples.index[samples[variant]]
        if len(names) < 10:
            continue
        sub_items = items_all.loc[names]
        pr = estimate(panel, sub_items, index, C.EST_WINDOW)
        if pr.empty:
            continue
        p_ff, d_ff = anchors(panel, sub_items)
        # Roles from the variant itself (launch has placebo_souvenir items).
        sub_roles = [r for r in C.ROLE_ORDER if r in set(pr["role"])]
        # Residual correlation of the variant's own items.
        rs = residual_series(panel, pr, index, C.EST_WINDOW)
        tab = caar_table(p_ff, d_ff, pr, index, None, rs, sub_roles, C.MODEL_MAIN)
        grid.append(tab.assign(index=C.INDEX_MAIN, variant=variant))

    # Shorter estimation window after Trade Protection, same sample; below MIN_REG_OBS drops out.
    lo_r, hi_r = C.EST_WINDOW_ROBUST
    log.info("")
    log.info("Robustness: estimation window %s instead of %s", C.EST_WINDOW_ROBUST,
             C.EST_WINDOW)
    pr_rob = estimate(panel, items, index, C.EST_WINDOW_ROBUST)
    if not pr_rob.empty:
        rs_rob = residual_series(panel, pr_rob, index, C.EST_WINDOW_ROBUST)
        tab = caar_table(price_ff, day_ff, pr_rob, index, None, rs_rob, roles,
                         C.MODEL_MAIN)
        grid.append(tab.assign(index=C.INDEX_MAIN, variant="est_window_robust"))
        log.info("  %d of %d main-sample items keep at least %d returns in "
                 "t = %d..%d", len(pr_rob), len(items), C.MIN_REG_OBS, lo_r, hi_r)
        by_role = pr_rob.groupby("role").size().to_dict()
        log.info("  per role: %s", by_role)

    grid = pd.concat(grid, ignore_index=True)
    grid.to_csv(C.OUT_TABLES / "04_robustness_grid.csv", index=False, encoding="utf-8")
    view = grid[(grid["window"] == "-1_+60")
                & grid["role"].isin(["treated_output", "treated_input"])]
    show(view.pivot_table(index=["variant", "index"], columns="role",
                          values="caar_pct").round(1))

    section("5. PORTFOLIO APPROACH")
    port = portfolio_test(panel, items, index, roles)
    port.to_csv(C.OUT_TABLES / "04_portfolio.csv", index=False, encoding="utf-8")
    show(port[port["window"].isin(["0", "-1_+60"])]
         .pivot_table(index="role", columns="window",
                      values=["caar_pct", "t"]).round(2))

    section("6. SAVE AND FIGURES")
    car_items = pd.concat(
        [car_for_window(price_ff, day_ff, params, index, w, C.MODEL_MAIN)
         .assign(window=label) for label, w in C.EVENT_WINDOWS.items()])
    car_items.reset_index(names="market_hash_name").to_parquet(
        C.CLEAN_DIR / "car_items.parquet", index=False)
    log.info("-> data/clean/car_items.parquet (%d rows, input for 05)", len(car_items))

    path = caar_path(price_ff, day_ff, params, index, C.MODEL_MAIN)
    path.to_csv(C.OUT_TABLES / "04_caar_path.csv", index=False, encoding="utf-8")
    fig_caar_path(path, rbar_by_role(params, resid, roles))
    fig_models(models)
    log.info("04_fig1_caar_path, 04_fig2_models")
    log.info("")
    log.info("Done.")


if __name__ == "__main__":
    main()

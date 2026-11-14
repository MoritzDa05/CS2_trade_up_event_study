# Cross-section: does the expected pool value explain the CARs (OLS clustered by pool,
# wild bootstrap), and do Covert prices line up with the pool value after the update?
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
from inference import (OLS, design, pool_clusters, star_p,  # noqa: E402
                       weight_scheme, wild_bootstrap)
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("05_cross_section")


def setup_logging() -> None:
    C.OUT_LOGS.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(C.OUT_LOGS / "05_cross_section.log", mode="w",
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


# Cluster per item: its pool (inference.pool_clusters); with fewer than MIN_POOL_CLUSTERS
# pools, the cases are the clusters.
def cluster_ids(sub: pd.DataFrame, label: str) -> np.ndarray:
    pools = pool_clusters(sub.index, sub["case"])
    g_case, g_pool = sub["case"].nunique(), pools.nunique()
    if g_pool < C.MIN_POOL_CLUSTERS:
        log.info("Clusters %s: %d cases -> %d pools; fewer than %d pools, case "
                 "clusters kept", label, g_case, g_pool, C.MIN_POOL_CLUSTERS)
        return sub["case"].to_numpy()
    log.info("Clusters %s: %d cases -> %d pools", label, g_case, g_pool)
    return pools.to_numpy()


def fit(df: pd.DataFrame, dep: str, cols: list[str], fe: list[str] | None,
        label: str, boot_on: str | None = None) -> tuple[pd.DataFrame, dict]:
    sub = df.dropna(subset=[dep] + cols)
    X, names = design(sub, cols, fe)
    y = sub[dep].to_numpy(dtype=float)
    cl = cluster_ids(sub, f"{sub['role'].iloc[0]} {label}")
    m = OLS(y, X, cl, names)
    info = {"spec": label, "n": m.n, "clusters": m.n_clusters, "r2": m.r2,
            "weights": weight_scheme(m.n_clusters)}
    if boot_on and boot_on in names:
        j = names.index(boot_on)
        info["p_wild"] = wild_bootstrap(y, X, cl, j)
        # Tells 09 which coefficient the bootstrap p-value belongs to.
        info["p_wild_on"] = boot_on
    tab = m.table().assign(spec=label)
    # Stars: CR1 from WEBB_MAX_CLUSTERS clusters on, below that the wild bootstrap.
    if m.n_clusters >= C.WEBB_MAX_CLUSTERS:
        tab["p_star"] = tab["p"]
    else:
        tab["p_star"] = [star_p(np.nan, wild_bootstrap(y, X, cl, names.index(n)),
                                m.n_clusters) if n in cols else np.nan for n in names]
    return tab, info


# Expected pool value per case and StatTrak status: mean pre-event price of the pool
# items (from case_pools.csv, because a knife can sit in several cases).
def pool_values(items: pd.DataFrame) -> pd.DataFrame:
    if not C.CASE_POOLS.exists():
        raise FileNotFoundError(
            f"{C.CASE_POOLS} missing. Run build_item_universe.py: it "
            "writes the case -> rare pool mapping that this script needs.")
    # utf-8-sig: the item lists are written with a BOM.
    pools = pd.read_csv(C.CASE_POOLS, encoding="utf-8-sig")
    pools["price"] = pools["market_hash_name"].map(items["est_median_price"])
    # astype(bool): ~ on an object column would not negate.
    pools["capped"] = pools["market_hash_name"].map(
        items["flag_high_price"]).fillna(False).astype(bool)

    priced = pools.dropna(subset=["price"])
    out = priced.groupby(["case", "stattrak"]).agg(
        pool_value=("price", "mean"),
        pool_value_median=("price", "median"),
        pool_items=("price", "size"),
        pool_capped=("capped", "sum")).reset_index()
    # Robustness: without price-capped items, whose prices are censored.
    alt = (priced[~priced["capped"]].groupby(["case", "stattrak"])["price"]
           .mean().rename("pool_value_nocap").reset_index())
    out = out.merge(alt, on=["case", "stattrak"], how="left")

    # Only the wear tiers a trade-up output lands in; vanilla knives stay in.
    realistic = priced[priced["wear"].isin(C.POOL_WEARS_REALISTIC)
                       | priced["wear"].isna()]
    wr = (realistic.groupby(["case", "stattrak"])["price"].mean()
          .rename("pool_value_wear").reset_index())
    out = out.merge(wr, on=["case", "stattrak"], how="left")
    log.info("Pool values for %d case/StatTrak cells, median %d pool items",
             len(out), int(out["pool_items"].median()))
    return out


# Cheapest input cost per knife: 5 x mean Covert price, minimum over its cases.
def input_cost_per_output(items: pd.DataFrame) -> pd.Series:
    pools = pd.read_csv(C.CASE_POOLS, encoding="utf-8-sig")
    inputs = items[items["role"] == "treated_input"]
    cost = (inputs.groupby(["case", "stattrak"])["est_median_price"].mean()
            * C.CRAFT_INPUTS)
    pools["cost"] = list(map(lambda c, s: cost.get((c, s), np.nan),
                             pools["case"], pools["stattrak"]))
    return pools.groupby("market_hash_name")["cost"].min()


# Main-sample items with CAR, pool value, input cost and controls.
def build_panel(window: str) -> pd.DataFrame:
    items = pd.read_parquet(C.ITEMS_CLEAN).set_index("market_hash_name")
    samples = pd.read_parquet(C.SAMPLES).set_index("market_hash_name")
    car = pd.read_parquet(C.CLEAN_DIR / "car_items.parquet")
    car = car[car["window"] == window].set_index("market_hash_name")

    df = items.loc[samples.index[samples["main"]]].copy()
    df["car"] = car["car"].reindex(df.index)
    df["log_price"] = np.log(df["est_median_price"])

    pv = pool_values(items)
    key = pd.MultiIndex.from_arrays([df["case"], df["stattrak"]])
    pv_idx = pv.set_index(["case", "stattrak"])
    for col in ["pool_value", "pool_value_nocap", "pool_value_wear", "pool_items"]:
        df[col] = pv_idx[col].reindex(key).to_numpy()
    df["log_pool"] = np.log(df["pool_value"])
    df["log_pool_nocap"] = np.log(df["pool_value_nocap"])
    df["log_pool_wear"] = np.log(df["pool_value_wear"])

    routes = (pd.read_csv(C.CASE_POOLS, encoding="utf-8-sig")
              .groupby("market_hash_name")["case"].nunique())
    df["n_routes"] = routes.reindex(df.index)
    df["log_input_cost"] = np.log(
        input_cost_per_output(items).reindex(df.index))

    df["is_glove"] = (df["category"] == "Gloves").astype(float)
    df["stattrak_d"] = df["stattrak"].astype(float)
    df["wear_g"] = df["wear"].fillna("vanilla")
    df["n_covert"] = df["n_covert_in_case"].astype(float)
    return df


def window_label(w: str) -> str:
    if "_" not in w:
        return f"t = {w}"
    lo, hi = w.split("_")
    return f"[{lo}, {hi}]".replace("-", "−")


def winsorize(s: pd.Series, p: float = None) -> pd.Series:
    p = p if p is not None else C.WINSOR
    lo, hi = s.quantile([p, 1 - p])
    return s.clip(lo, hi)


def window_medians(panel: pd.DataFrame, window: tuple[int, int]) -> pd.Series:
    return (panel[panel["t"].between(*window)]
            .groupby("market_hash_name", observed=True)["price"].median())


# Pool value per case and StatTrak status from the median prices of a window.
def pool_value_window(panel: pd.DataFrame, window: tuple[int, int]) -> pd.DataFrame:
    pools = pd.read_csv(C.CASE_POOLS, encoding="utf-8-sig")
    pools["price"] = pools["market_hash_name"].map(window_medians(panel, window))
    priced = pools.dropna(subset=["price"])
    out = priced.groupby(["case", "stattrak"])["price"].mean().rename(
        "pool_value").to_frame()
    realistic = priced[priced["wear"].isin(C.POOL_WEARS_REALISTIC)
                       | priced["wear"].isna()]
    out["pool_value_wear"] = realistic.groupby(["case", "stattrak"])["price"].mean()
    return out


# Covert price in EUR on the pool value, before and after the update. Before: pool value
# of the estimation window; after: pool value of the same window as the Covert price.
# In levels, because the substitution condition is linear in euros.
def level_relation(df: pd.DataFrame, panel: pd.DataFrame
                   ) -> tuple[pd.DataFrame, dict[str, pd.Series], pd.DataFrame]:
    inputs = df[df["role"] == "treated_input"].copy()
    rows = []
    # Same windows as 06.
    periods = {"before": C.LEVEL_PRE, "after": C.LEVEL_POST}
    prices = {lab: window_medians(panel, w) for lab, w in periods.items()}
    key = pd.MultiIndex.from_arrays([inputs["case"], inputs["stattrak"]])
    post = pool_value_window(panel, C.LEVEL_POST)
    pool_after = pd.DataFrame({c: post[c].reindex(key).to_numpy() for c in post.columns},
                              index=inputs.index)
    pool_by = {"before": inputs, "after": pool_after}
    combos = [(lab, pool) for lab in periods
              for pool in ["pool_value", "pool_value_wear"]]
    for label, pool_col in combos:
        px = prices[label]
        sub = inputs.drop(columns=["pool_value", "pool_value_wear"]).assign(
            price=px.reindex(inputs.index),
            **{pool_col: pool_by[label][pool_col].reindex(inputs.index)}).dropna(
            subset=["price", pool_col])
        X, names = design(sub, [pool_col])
        m = OLS(sub["price"].to_numpy(float), X,
                cluster_ids(sub, f"level {label} {pool_col}"), names)
        slope, se = m.beta[1], m.se[1]
        t_vs_bound = (slope - C.ARBITRAGE_SLOPE) / se if se > 0 else np.nan
        rows.append({
            "period": label, "pool": pool_col, "n": m.n, "clusters": m.n_clusters,
            "intercept": m.beta[0], "slope": slope, "se": se,
            "t_vs_zero": m.t[1], "p_vs_zero": m.p[1],
            "implied_inputs": C.SELLER_NET / slope if slope > 0 else np.nan,
            "t_vs_bound": t_vs_bound,
            "p_vs_bound": 2 * stats.t.sf(abs(t_vs_bound), m.n_clusters - 1),
            "r2": m.r2})
    return pd.DataFrame(rows), prices, pool_after


# CAR by wear tier: crafting demand should hit the cheap tiers (WW, BS), not FN.
def wear_pattern(df: pd.DataFrame) -> pd.DataFrame:
    order = ["Factory New", "Minimal Wear", "Field-Tested", "Well-Worn",
             "Battle-Scarred"]
    rows = []
    for role in ["treated_input", "treated_output"]:
        sub = df[df["role"] == role]
        g = sub.groupby("wear", observed=True)
        t = pd.DataFrame({
            "role": role,
            "items": g.size(),
            "car_median_pct": (np.exp(g["car"].median()) - 1) * 100,
            "car_mean_pct": (np.exp(g["car"].mean()) - 1) * 100,
            "price_pre": g["est_median_price"].median(),
        }).reindex(order).dropna(subset=["items"])
        rows.append(t.reset_index())
    return pd.concat(rows, ignore_index=True).round(2)


def fig_wear(wear: pd.DataFrame) -> None:
    order = ["Factory New", "Minimal Wear", "Field-Tested", "Well-Worn",
             "Battle-Scarred"]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(order))
    width = 0.38
    for k, (role, off) in enumerate([("treated_input", -width / 2),
                                     ("treated_output", width / 2)]):
        sub = wear[wear["role"] == role].set_index("wear").reindex(order)
        ax.bar(x + off, sub["car_median_pct"], width, color=ps.color(role),
               label=ps.label(role))
    ax.axhline(0, color=ps.NEUTRAL, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=8.5)
    ax.set_xlabel("Wear level")
    ax.set_ylabel(f"Median CAR over {window_label(C.CAR_WINDOW_MAIN)} (%)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    ps.save(fig, "05_fig4_wear_pattern", C.OUT_FIGURES)


def fig_dose(case_level: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, role in [(axes[0], "treated_input"), (axes[1], "treated_output")]:
        title = ps.label(role)
        sub = case_level[case_level["role"] == role].dropna(
            subset=["log_pool", "car"])
        if sub.empty:
            continue
        col = ps.color(role)
        ax.scatter(sub["log_pool"], 100 * (np.exp(sub["car"]) - 1),
                   s=np.sqrt(sub["n_items"]) * 8, color=col, alpha=0.75,
                   edgecolor="white", linewidth=0.5)
        b, a = np.polyfit(sub["log_pool"], sub["car"], 1)
        xs = np.linspace(sub["log_pool"].min(), sub["log_pool"].max(), 50)
        ax.plot(xs, 100 * (np.exp(a + b * xs) - 1), color=ps.NEUTRAL,
                linewidth=1.2, linestyle="--")
        r = np.corrcoef(sub["log_pool"], sub["car"])[0, 1]
        # Panel label (group, correlation, cases), not a figure title.
        ax.set_title(f"{title}: r = {r:.2f}, {len(sub)} cases".replace(
            "-", "−"), fontsize=9)
        ax.axhline(0, color=ps.GRID, linewidth=1)
        ax.set_xlabel("log pool value of the case")
    axes[0].set_ylabel(
        f"mean CAR over {window_label(C.CAR_WINDOW_MAIN)} (%)")
    fig.tight_layout()
    ps.save(fig, "05_fig1_dose_response", C.OUT_FIGURES)


def fig_coefficients(coefs: pd.DataFrame) -> None:
    sub = coefs[coefs.index.isin(["log_pool", "log_price", "trade_share",
                                  "log_input_cost", "n_routes"])].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels, y = [], 0
    for (grp, spec), row in sub.groupby(["group", "spec"], sort=False):
        for name, r in row.iterrows():
            ax.errorbar(r["coef"], y, xerr=1.96 * r["se"], fmt="o",
                        color=ps.color(grp), markersize=5, capsize=3)
            labels.append(f"{name}  [{spec}]")
            y += 1
    ax.axvline(0, color=ps.NEUTRAL, linewidth=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("coefficient, dependent variable log CAR; "
                  "95% cluster-robust interval")
    fig.tight_layout()
    ps.save(fig, "05_fig2_coefficients", C.OUT_FIGURES)


def fig_level(df: pd.DataFrame, prices: dict[str, pd.Series],
              lvl: pd.DataFrame, pool_after: pd.DataFrame) -> None:
    inputs = df[df["role"] == "treated_input"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for label, col in [("before", ps.GRID),
                       ("after", ps.color("treated_input"))]:
        px = prices[label]
        pv = inputs["pool_value"] if label == "before" else pool_after["pool_value"]
        sub = inputs.assign(price=px.reindex(inputs.index),
                            pool_value=pv.reindex(inputs.index)).dropna(
            subset=["price", "pool_value"])
        ax.scatter(sub["pool_value"], sub["price"], s=8, alpha=0.5,
                   color=col, edgecolor="none",
                   label=f"{label} the update")
        # Line of the pool_value fit, matching the scatter.
        row = lvl[(lvl["period"] == label)
                  & (lvl["pool"] == "pool_value")].iloc[0]
        xs = np.linspace(sub["pool_value"].min(), sub["pool_value"].max(), 50)
        ax.plot(xs, row["intercept"] + row["slope"] * xs, color=col,
                linewidth=1.5)

    xs = np.linspace(0, inputs["pool_value"].max(), 50)
    ax.plot(xs, C.ARBITRAGE_SLOPE * xs, color=ps.NEUTRAL, linestyle="--",
            linewidth=1.2,
            label=f"no-arbitrage bound, slope {C.ARBITRAGE_SLOPE:.3f}")
    ax.set_ylim(0, inputs["est_median_price"].quantile(0.99) * 2.5)
    ax.set_xlabel("Expected value of the case pool (EUR)")
    ax.set_ylabel("Covert price (EUR)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    ps.save(fig, "05_fig3_level_relation", C.OUT_FIGURES)


def main() -> None:
    setup_logging()
    ps.apply_style()
    C.OUT_TABLES.mkdir(parents=True, exist_ok=True)

    df = build_panel(C.CAR_WINDOW_MAIN)
    df["car_long"] = pd.read_parquet(C.CLEAN_DIR / "car_items.parquet").pipe(
        lambda d: d[d["window"] == C.CAR_WINDOW_LONG].set_index(
            "market_hash_name")["car"]).reindex(df.index)
    # Winsorise within role: the tails differ (knives fell, Coverts rose).
    df["car"] = df.groupby("role")["car"].transform(winsorize)
    df["car_long"] = df.groupby("role")["car_long"].transform(winsorize)
    log.info("Panel: %d items | dependent variable: CAR %s",
             len(df), C.CAR_WINDOW_MAIN)

    inputs = df[df["role"] == "treated_input"]
    outputs = df[df["role"] == "treated_output"]

    section("1. COVERTS: DOES THE POOL VALUE EXPLAIN THE REACTION?")
    specs_in = [
        ("(1) controls only", ["log_price", "trade_share", "stattrak_d",
                               "n_covert"], ["wear_g"]),
        ("(2) + pool value", ["log_pool", "log_price", "trade_share",
                              "stattrak_d", "n_covert"], ["wear_g"]),
        ("(3) + category FE", ["log_pool", "log_price", "trade_share",
                               "stattrak_d", "n_covert"], ["wear_g", "category"]),
        ("(4) pool without capped items", ["log_pool_nocap", "log_price",
                                           "trade_share", "stattrak_d",
                                           "n_covert"], ["wear_g"]),
        ("(5) pool weighted by realistic wear", ["log_pool_wear", "log_price",
                                                 "trade_share", "stattrak_d",
                                                 "n_covert"], ["wear_g"]),
    ]
    tabs, infos_in = [], []
    for label, cols, fe in specs_in:
        boot = "log_pool" if "log_pool" in cols else None
        t, i = fit(inputs, "car", cols, fe, label, boot_on=boot)
        tabs.append(t.assign(group="treated_input"))
        infos_in.append(i)
    in_tab = pd.concat(tabs)

    view = in_tab[in_tab.index.isin(["log_pool", "log_pool_nocap",
                                     "log_pool_wear", "log_price",
                                     "trade_share", "n_covert"])]
    view = view.assign(sig=[stars(p) for p in view["p"]])
    show(view[["spec", "coef", "se", "t", "p", "sig"]].round(3))
    show(pd.DataFrame(infos_in).round(3))

    # Long window as its own specification (6), so the number has a table row.
    long_label = f"(6) long window {C.CAR_WINDOW_LONG}"
    t_long, i_long = fit(inputs, "car_long", ["log_pool", "log_price",
                                              "trade_share", "stattrak_d",
                                              "n_covert"], ["wear_g"],
                         long_label, boot_on="log_pool")
    log.info("")
    log.info("Long window %s: log_pool coefficient %.3f (p = %.3f, wild %.3f)",
             C.CAR_WINDOW_LONG, t_long.loc["log_pool", "coef"],
             t_long.loc["log_pool", "p"], i_long.get("p_wild", np.nan))
    pd.concat([in_tab, t_long.assign(group="treated_input")]).to_csv(
        C.OUT_TABLES / "05_regression_input.csv", encoding="utf-8")

    section("2. KNIVES AND GLOVES: DOES CHEAP INPUT MEAN A DEEPER FALL?")
    specs_out = [
        ("(1) controls only", ["log_price", "trade_share", "stattrak_d",
                               "is_glove"], ["wear_g"]),
        ("(2) + input cost", ["log_input_cost", "log_price", "trade_share",
                              "stattrak_d", "is_glove"], ["wear_g"]),
        ("(3) + routes", ["log_input_cost", "n_routes", "log_price",
                          "trade_share", "stattrak_d", "is_glove"], ["wear_g"]),
    ]
    tabs, infos_out = [], []
    for label, cols, fe in specs_out:
        boot = "log_input_cost" if "log_input_cost" in cols else None
        t, i = fit(outputs, "car", cols, fe, label, boot_on=boot)
        tabs.append(t.assign(group="treated_output"))
        infos_out.append(i)
    out_tab = pd.concat(tabs)
    out_tab.to_csv(C.OUT_TABLES / "05_regression_output.csv", encoding="utf-8")
    view = out_tab[out_tab.index.isin(["log_input_cost", "n_routes",
                                       "log_price", "trade_share", "is_glove"])]
    view = view.assign(sig=[stars(p) for p in view["p"]])
    show(view[["spec", "coef", "se", "t", "p", "sig"]].round(3))
    show(pd.DataFrame(infos_out).round(3))

    # Fit statistics per specification, read by 09.
    fit_tab = pd.concat([
        pd.DataFrame(infos_in).assign(group="treated_input",
                                      window=C.CAR_WINDOW_MAIN),
        pd.DataFrame([i_long]).assign(group="treated_input",
                                      window=C.CAR_WINDOW_LONG),
        pd.DataFrame(infos_out).assign(group="treated_output",
                                       window=C.CAR_WINDOW_MAIN),
    ], ignore_index=True)
    fit_tab = fit_tab[["group", "spec", "window", "n", "clusters", "r2",
                       "p_wild", "p_wild_on"]]
    fit_tab.to_csv(C.OUT_TABLES / "05_regression_fit.csv", index=False,
                   encoding="utf-8")
    log.info("")
    log.info("-> output/tables/05_regression_fit.csv (%d specifications)",
             len(fit_tab))

    section("3. LEVEL RELATION: DO PRICES MOVE TO THE ARBITRAGE BOUND?")
    panel = pd.read_parquet(C.PANEL_CLEAN,
                            columns=["market_hash_name", "t", "price"])
    lvl, level_prices, pool_after = level_relation(df, panel)
    show(lvl.round(4))
    lvl.to_csv(C.OUT_TABLES / "05_level_test.csv", index=False, encoding="utf-8")
    log.info("")
    log.info("Predicted slope from the fee structure: %.3f "
             "(sell at %.3f of the buyer price, %d inputs)",
             C.ARBITRAGE_SLOPE, C.SELLER_NET, C.CRAFT_INPUTS)
    log.info("Level windows: before %s, after %s (config.LEVEL_PRE/LEVEL_POST); pool "
             "value before from the estimation window, after from %s",
             C.LEVEL_PRE, C.LEVEL_POST, C.LEVEL_POST)

    section("4. CASE LEVEL AND FIGURES")
    case_level = (df[df["role"].isin(["treated_input", "treated_output"])]
                  .groupby(["role", "case"], observed=True)
                  .agg(car=("car", "mean"), log_pool=("log_pool", "mean"),
                       log_input_cost=("log_input_cost", "mean"),
                       n_items=("car", "size")).reset_index())
    case_level.to_csv(C.OUT_TABLES / "05_case_level.csv", index=False,
                      encoding="utf-8")
    for role in ["treated_input", "treated_output"]:
        sub = case_level[case_level["role"] == role].dropna(
            subset=["log_pool", "car"])
        if len(sub) > 2:
            log.info("%-16s %2d cases, correlation of CAR with log pool value: %.3f",
                     role, len(sub), np.corrcoef(sub["log_pool"], sub["car"])[0, 1])

    section("5. REACTION BY WEAR TIER")
    wear = wear_pattern(df)
    show(wear.set_index(["role", "wear"]))
    wear.to_csv(C.OUT_TABLES / "05_wear_pattern.csv", index=False, encoding="utf-8")
    fig_wear(wear)

    coefs = pd.concat([in_tab, out_tab])
    fig_dose(case_level)
    fig_coefficients(coefs)
    fig_level(df, level_prices, lvl, pool_after)
    log.info("05_fig1_dose_response, 05_fig2_coefficients, "
             "05_fig3_level_relation, 05_fig4_wear_pattern")
    log.info("")
    log.info("Done.")


if __name__ == "__main__":
    main()

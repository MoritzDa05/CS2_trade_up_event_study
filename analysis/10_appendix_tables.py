# Builds the appendix tables (A to E) from the outputs of 00-08, check_item_list.py
# and items/case_pools.csv; estimates nothing. Table A2 comes from trade_up_contracts/observed.py.
# Optional argument: the appendices to build, e.g. A or C D (default: all).
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

log = logging.getLogger("10_appendix_tables")

# Groups in table order: analysis groups, benchmark, appendix-only groups.
APPENDIX_ROLES = {
    "treated_output": "Output",
    "treated_input": "Input",
    "placebo_st": "Placebo ST",
    "placebo": "Placebo Map",
    "placebo_low": "Placebo Classified",
    "index_main": "Benchmark (sticker capsules)",
    "index_alt": "Music kits and agents",
    "case": "Cases",
    "excluded": "Graffiti",
}
# Pair types of 02 in the thesis term ("group", not "role").
PAIR_LABELS = {
    "Same role, different case": "Same group, different case",
    "Different role": "Different group",
}
ANALYSIS_ROLES = ["treated_output", "treated_input", "placebo_st", "placebo",
                  "placebo_low"]
BENCHMARK_ROLE = "index_main"

EVENT_LABELS = {
    "0": "t = 0",
    "-1_+1": "[−1, +1]",
    "-1_+5": "[−1, +5]",
    "-1_+30": "[−1, +30]",
    "-1_+60": "[−1, +60]",
}
PRE_LABELS = {
    "-14_-2": "[−14, −2]",
    "-5_-2": "[−5, −2]",
}

# Return models of 04 as the panels of Table C4.
MODEL_PANELS = {
    "mm_alpha0": "Panel A. Market model with α = 0 (main specification)",
    "mm": "Panel B. Market model with estimated α",
    "market_adj": "Panel C. Market-adjusted (β = 1)",
    "placebo_st_adj": "Panel D. Difference from the mean CAR of Placebo ST",
}
ALPHA_MODEL = "mm"  # second model of the pre-trend test in 04

# Rows of Table C5: (label, index, variant); None marks a heading row.
# The launch variant is left out (4 output items, no input).
TABLE_C5_ROWS = [
    ("Main specification (sticker capsules)", "capsules", "main"),
    ("Benchmark index", None, None),
    ("Music kits", "musickits", "main"),
    ("Agents", "agents", "main"),
    ("Sticker capsules, music kits, and agents", "combined", "main"),
    ("Sample", None, None),
    ("Trade share of at least 50%", "capsules", "liq50"),
    ("Including Doppler, Gamma Doppler, and Case Hardened", "capsules",
     "subclass_all"),
    ("Illiquid items (trade share 10% to 30%)", "capsules", "illiquid"),
    (f"First sale at least {C.BURN_IN_ROBUST_DAYS} days before the estimation window",
     "capsules", "burn_in30"),
    ("Estimation window", None, None),
    ("t = −85 to −15", "capsules", "est_window_robust"),
]
TABLE_C5_ROLES = ["treated_output", "treated_input"]
TABLE_C5_WINDOWS = ["0", "-1_+60"]
# Marks t_BMP where KP cannot be estimated (r-bar missing).
BMP_MARK = "ᵃ"

# Steam lists these under one market name; the game file has one paint kit per colour.
PHASE_FINISHES = {"Doppler": 7, "Gamma Doppler": 5}
CHECK_KITS_ROW = "Float range of the knife and glove finishes"
A1_EXAMPLES = 2
VANILLA = "vanilla"  # 06's wear level for vanilla knives (no float)
NARROW_MAX = 0.08  # highest float of the finishes that are always FN or MW
EXAMPLE_ROUTE = "AK-47 | Head Shot (Battle-Scarred)"  # example route of Section 3.1

# Groups of the item list in the order of Table B1.
LIST_GROUPS = {
    "output_knife": "Output, knives",
    "output_gloves": "Output, gloves",
    "input_covert": "Input",
    "placebo_covert_st": "Placebo ST",
    "placebo_covert": "Placebo Map",
    "placebo_classified": "Placebo Classified",
    "placebo_souvenir": "Souvenir Coverts",
    "capsule_sticker": "Sticker capsules",
    "control_musickit": "Music kits",
    "control_agent": "Agents",
    "case": "Cases",
    "control_graffiti": "Graffiti",
}
# Pipeline groups in Tables B2 and B3.
B_ROLES = {
    "treated_output": "Output",
    "treated_input": "Input",
    "placebo_st": "Placebo ST",
    "placebo": "Placebo Map",
    "placebo_low": "Placebo Classified",
    "index_main": "Benchmark (sticker capsules)",
    "index_alt": "Music kits and agents",
    "case": "Cases",
    "excluded": "Graffiti",
    "placebo_souvenir": "Souvenir Coverts",
}
B2_PANELS = {
    "Panel A. Analysis groups": ["treated_output", "treated_input", "placebo_st",
                                 "placebo", "placebo_low"],
    "Panel B. Other groups": ["index_main", "index_alt", "case", "excluded",
                              "placebo_souvenir", "TOTAL"],
}
# Columns of 01_sample_flow -> row labels of Table B2, in the order of 01.
FLOW_STEPS = {
    "items_in_window": "In data window",
    "after_est_data": "Sale in estimation window",
    "after_burn_in": "Listed before estimation window",
    "after_min_price": f"Median price ≥ EUR {C.MIN_PRICE_EUR:.2f}",
    "after_high_price": "No censored prices",
    "after_liquidity": f"Trade share ≥ {C.LIQ_MAIN_MIN:.0%}",
    "after_min_returns": f"At least {C.MIN_EST_RETURNS} returns",
    "after_subclass": "Standard finishes",
    "after_store_capsule": "Not sold in store",
}
B2_HEADS = {"index_main": "Benchmark (sticker capsules)", "index_alt": "Music kits and agents",
            "placebo_souvenir": "Souvenir", "TOTAL": "All groups"}
# Trade-share tiers of 00 as columns of Table B3.
LIQ_COLUMNS = {
    "no_est_data": "No sale",
    "lt10": "Below 10%",
    "10_30": "10% to 30%",
    "30_50": "30% to 50%",
    "ge50": "50% or more",
}
VARIANT_COLUMNS = {
    "main": "Main sample",
    "liq50": f"Trade share ≥ {C.LIQ_ROBUST_MIN:.0%}",
    "subclass_all": "With Doppler and Case Hardened",
    "illiquid": "Trade share 10% to 30%",
    "burn_in30": f"First sale ≥ {C.BURN_IN_ROBUST_DAYS} days before the estimation window",
}
# Table B4: rows per panel and the events of Section 2.3. The event day and the day
# after are marked (an evening announcement moves prices the next day).
JUMP_ROWS = {"full": 15, "estimation_window": 10}
EVENT_MARK = "ᵃ"

WEAR_ABBR = {"Factory New": "FN", "Minimal Wear": "MW", "Field-Tested": "FT",
             "Well-Worn": "WW", "Battle-Scarred": "BS"}


# 'A, B, and C' (serial comma, as in the thesis).
def and_list(items: list[str]) -> str:
    items = list(items)
    if len(items) < 3:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


# Table D1: every input specification of 05 (Table 4 shows only (1) to (3)).
TABLE_D1_SPECS = {
    "(1) controls only": "Controls only",
    "(2) + pool value": "Pool value",
    "(3) + category FE": "Category fixed effects",
    "(4) pool without capped items": "Pool without capped items",
    "(5) pool weighted by realistic wear":
        "Pool of " + and_list([WEAR_ABBR[w] for w in C.POOL_WEARS_REALISTIC]),
    f"(6) long window {C.CAR_WINDOW_LONG}": "Long window",
}
# The pool value enters (2) to (6) under three names; one row holds all of them.
POOL_VARIABLES = ["log_pool", "log_pool_nocap", "log_pool_wear"]
TABLE_D1_REGRESSORS = [
    (POOL_VARIABLES, "Pool value (log)"),
    (["log_price"], "Own price (log)"),
    (["trade_share"], "Trade share"),
    (["stattrak_d"], "StatTrak"),
    (["n_covert"], "Coverts in the case"),
]
WEAR_FE_PREFIX = "wear_g_"
CATEGORY_FE_PREFIX = "category_"
# Base of the category FE: get_dummies(drop_first=True) in 05 drops the first category
# alphabetically; the CSV does not record it.
CATEGORY_BASE = "Heavy"

# Table D2: binding status of 06_car_by_binding, in this order.
BINDING_STATUS = {
    "binding": "Binding",
    "slack": "Slack",
    "no pre-event ratio": "No ratio before the update",
}
BINDING_DIFF = "difference binding - slack"

WEAR_ROW_PREFIX = "wear: "
# Table D3: rows of 06_bound_test beyond the three of Table 5.
POOL_ROWS = {
    "binding: knife cases": "Knife cases",
    "binding: glove cases": "Glove cases",
}

# Table D4, Panel B: days of the ratchet paragraph (Section 5.3) and the changes quoted.
RATCHET_DAYS = C.RATCHET_DAYS
# Non-breaking spaces, so that Word keeps "Δ ln P" on one line.
D_LN_P, D_LN_EV, D_LN_R = (f"Δ ln {x}" for x in ("P", "E[V]", "R"))
RATCHET_CHANGES = [(7, 16), (16, 28)]

# Table D5: rows of 06_robustness (labels built from config as in 06) -> column heads.
ROBUSTNESS_LABELS = {
    "main": "Main",
    "input float: best in tier": "Lowest float",
    "float rule: raw average": "Raw average",
    f"ex post: knife prices at t+{C.TRADE_HOLD_DAYS}":
        f"Output prices of t + {C.TRADE_HOLD_DAYS}",
    "all input items": "All input Coverts",
    f"without routes > {C.CAP_SHARE_MAX:.0%} capped": "Without capped value",
}
OVERSHOOT_LABELS = {
    "coefficient of variation": "Coefficient of variation",
    "skewness": "Skewness",
}

VOL_LABELS = {
    "treated_output": "Output",
    "treated_input": "Input",
    "control": "Control group",
    "placebo_st": "Placebo ST",
    "index_main": "Benchmark (sticker capsules)",
    "index_alt": "Music kits and agents",
    "case": "Cases",
}
VOL_REFERENCES = {"control": "control group", "index_alt": "music kits and agents"}

# Table E1: references as panels; windows of Table 3 plus echo and new regime.
VOLUME_PANELS = {
    "none": "Panel A. Relative to the item's own normal volume",
    "control": "Panel B. Relative to the control group",
    "index_alt": "Panel C. Relative to music kits and agents",
}
VOLUME_WINDOWS = ["0", "-1_+1", "-1_+5", "-1_+60", "+7_+9", "+30_+190"]

ECHO_GROUPS = ["control", "placebo_st", "treated_input", "treated_output"]

ROBUSTNESS_GROUPS = {"output": "Output", "input": "Input",
                     "control": "Control group"}

HET_GROUPS = {
    "input, binding": "Binding",
    "input, slack": "Slack",
    "input, no pre-event ratio": "Without ratio",
}
COUPLING_GROUPS = {
    "input, all": "Input, all",
    "input, binding": "Binding",
    "input, slack": "Slack",
    "placebo_st": "Placebo ST",
}
COUPLING_DIFF = "binding - slack (same pools)"  # row name in 07_coupling

BUDAPEST_MAJOR = C.EVENT_DAYS["Budapest Major"]

# Test standard of Section 4.4, as in 09.
ESTABLISHED_LEVEL = 0.05
ESTABLISHED_MARK = "†"

MINUS = "−"  # typographic minus, as in the thesis
MISSING = "–"  # not defined


def setup_logging() -> None:
    C.OUT_LOGS.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(C.OUT_LOGS / "10_appendix_tables.log", mode="w",
                             encoding="utf-8")
    fh.setFormatter(fmt)
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


def show(df: pd.DataFrame, header: bool = True) -> None:
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        for line in df.to_string(index=False, header=header).splitlines():
            log.info(line)


def read_table(name: str, script: str, index_col: int | list[int] | None = None
               ) -> pd.DataFrame:
    path = C.OUT_TABLES / name
    if not path.exists():
        raise SystemExit(f"{path} not found -- run {script} first.")
    return pd.read_csv(path, index_col=index_col, encoding="utf-8")


# CSV and TSV (for Word) of the same frame.
def save(df: pd.DataFrame, stem: str, header: bool = True) -> None:
    df.to_csv(C.OUT_TABLES / f"{stem}.csv", index=False, header=header,
              encoding="utf-8")
    df.to_csv(C.OUT_TABLES / f"{stem}.tsv", index=False, header=header,
              sep="\t", encoding="utf-8")
    log.info("-> output/tables/%s.csv and .tsv", stem)


def emit(title: str, df: pd.DataFrame, stem: str, header: bool = True) -> None:
    section(title)
    show(df, header=header)
    save(df, stem, header=header)


def ok(x) -> bool:
    return x is not None and pd.notna(x) and np.isfinite(float(x))


# Number formats as in 09: American English, typographic minus.
def fmt_int(x) -> str:
    return f"{int(round(x)):,}" if ok(x) else MISSING


def fmt_dec(x, digits: int) -> str:
    return f"{x:,.{digits}f}".replace("-", MINUS) if ok(x) else MISSING


def fmt_signed(x, digits: int) -> str:
    return f"{x:+,.{digits}f}".replace("-", MINUS) if ok(x) else MISSING


def fmt_cell(caar, t, mark: str = "") -> str:
    return f"{fmt_signed(caar, 1)} ({fmt_signed(t, 2)}){mark}"


def fmt_coef(x, digits: int = 3) -> str:
    return f"{x:.{digits}f}".replace("-", MINUS) if ok(x) else ""


def fmt_p(p) -> str:
    if not ok(p):
        return MISSING
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


# Wild bootstrap p = (hits + 1) / (reps + 1): no hit at all is shown as "≤ 0.001".
def fmt_p_wild(p) -> str:
    if not ok(p):
        return MISSING
    floor = 1 / (C.BOOTSTRAP_REPS + 1)
    return f"≤ {floor:.3f}" if p <= floor * (1 + 1e-9) else f"{p:.3f}"


# Bootstrap p from 999 draws: below 0.01 shown as "< 0.01" (as in Table 6).
def fmt_p_boot(p) -> str:
    if not ok(p):
        return MISSING
    return "< 0.01" if p < 0.01 else f"{p:.3f}"


def fmt_ci(lo, hi, digits: int, signed: bool = False) -> str:
    if not (ok(lo) and ok(hi)):
        return ""
    f = (lambda x: fmt_signed(x, digits)) if signed else (
        lambda x: fmt_dec(x, digits))
    return f"[{f(lo)}; {f(hi)}]"


def sig_stars(p) -> str:
    if not ok(p):
        return ""
    for level, mark in C.SIG_LEVELS:
        if p < level:
            return mark
    return ""


# Stars from p_star: CR1 from WEBB_MAX_CLUSTERS clusters on, wild bootstrap below.
def star_of(r) -> str:
    return sig_stars(r["p_star"] if "p_star" in r.index else r["p"])


def to_pct(x):
    return (np.exp(x) - 1) * 100


def days_label(lo: int, hi: int) -> str:
    s = f"t = {lo}" if lo == hi else f"t = {lo} to {hi}"
    return s.replace("-", MINUS)


def window_label(lo: int, hi: int) -> str:
    def f(x: int) -> str:
        return "0" if x == 0 else f"{x:+d}".replace("-", MINUS)
    return f"[{f(lo)}, {f(hi)}]"


# Section 4.4: t_KP and the rank test both reject at 5 % with the same sign.
def is_established(row) -> bool:
    vals = [row.get(k) for k in ("p_kp", "p_rank", "t_kp", "z_rank")]
    if not all(ok(v) for v in vals):
        return False
    p_kp, p_rank, t_kp, z_rank = vals
    return (p_kp < ESTABLISHED_LEVEL and p_rank < ESTABLISHED_LEVEL
            and np.sign(t_kp) == np.sign(z_rank))


def group_label(role: str, n) -> str:
    return f"{APPENDIX_ROLES[role]} (n = {fmt_int(n)})"


def pick(df: pd.DataFrame, **keys) -> pd.Series | None:
    mask = np.ones(len(df), dtype=bool)
    for k, v in keys.items():
        mask &= (df[k].astype(str) == str(v)).to_numpy()
    sel = df[mask]
    if len(sel) > 1:
        log.warning("%d rows for %s, using the first", len(sel), keys)
    return sel.iloc[0] if len(sel) else None


def main_n(caar: pd.DataFrame) -> dict[str, int]:
    out = {}
    for role, g in caar.groupby("role"):
        ns = g["n"].unique()
        if len(ns) != 1:
            log.warning("%s: n differs across windows (%s), using the largest",
                        role, ns)
        out[role] = int(ns.max())
    return out


# Rows of a table without header line, padded to one width so that panels line up in Word.
class Rows:
    def __init__(self, width: int):
        self.width = width
        self.rows: list[list[str]] = []

    def add(self, cells: list[str]) -> None:
        if len(cells) > self.width:
            raise ValueError(f"row wider than the table: {cells}")
        self.rows.append(list(cells) + [""] * (self.width - len(cells)))

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def read_clean(path: Path, script: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"{path} not found -- run {script} first.")
    return pd.read_parquet(path)


# Case pools with the finish of each item ('★ Karambit | Doppler' -> 'Doppler').
def load_pools() -> pd.DataFrame:
    if not C.CASE_POOLS.exists():
        raise SystemExit(f"{C.CASE_POOLS} not found -- run "
                         "build_item_universe.py first.")
    # utf-8-sig: the item lists are written with a BOM.
    pools = pd.read_csv(C.CASE_POOLS, encoding="utf-8-sig")
    pools["finish"] = pools["skin"].str.split(" | ", n=1, regex=False).str[1]
    return pools


def load_routes() -> tuple[pd.DataFrame, pd.DataFrame]:
    script = "06_arbitrage.py"
    return (read_clean(C.TRADEUP_ROUTES, script),
            read_clean(C.TRADEUP_WEIGHTS, script))


# Main-sample input routes, as select() in 06.
def main_input(routes: pd.DataFrame) -> pd.DataFrame:
    return routes[(routes["role"] == "treated_input") & routes["main"]]


# Wear levels a float range reaches: 'FN, MW' for two, 'FN to FT' for three or more.
def wear_text(lo: float, hi: float) -> str:
    levels = [WEAR_ABBR[w] for w, (a, b) in C.WEAR_BOUNDS.items()
              if min(b, hi) > max(a, lo)]
    if len(levels) >= 3:
        return f"{levels[0]} to {levels[-1]}"
    return ", ".join(levels)


# Table A1: float ranges of the knife and glove finishes.
def table_a1() -> pd.DataFrame:
    pools = load_pools()
    fin = pools.drop_duplicates("skin")
    knives = fin["category"] == "Knives"
    gloves = fin["category"] == "Gloves"
    vanilla = fin["min_float"].isna()
    if (knives | gloves).sum() != len(fin):
        log.warning("Finishes that are neither knives nor gloves: %s",
                    sorted(fin.loc[~(knives | gloves), "skin"]))
    ranged = fin[~vanilla]

    t = Rows(5)
    t.add(["Float range", "Knives", "Gloves", "Wear levels", "Examples"])
    classes = (ranged[["min_float", "max_float"]].drop_duplicates()
               .sort_values(["max_float", "min_float"]))
    for lo, hi in classes.itertuples(index=False):
        g = ranged[(ranged["min_float"] == lo) & (ranged["max_float"] == hi)]
        names = (g["finish"].value_counts().rename_axis("finish")
                 .reset_index(name="n")
                 .sort_values(["n", "finish"], ascending=[False, True]))
        t.add([f"{lo:.2f} to {hi:.2f}",
               fmt_int((g["category"] == "Knives").sum()),
               fmt_int((g["category"] == "Gloves").sum()), wear_text(lo, hi),
               ", ".join(names["finish"].head(A1_EXAMPLES))])
    t.add(["Vanilla knives", fmt_int((vanilla & knives).sum()),
           fmt_int((vanilla & gloves).sum()), MISSING, MISSING])
    t.add(["All finishes", fmt_int(knives.sum()), fmt_int(gloves.sum())])

    # Counts quoted in Section 2.1 and Appendix B (log only).
    narrow = (knives & (fin["max_float"] <= NARROW_MAX + 1e-9)).sum()
    log.info("Check: %d finishes in %d pools: %d knives (%d of them vanilla), "
             "%d gloves; %d knife finishes end at %.2f",
             len(fin), pools["case"].nunique(), knives.sum(),
             (vanilla & knives).sum(), gloves.sum(), narrow, NARROW_MAX)
    # Paint kits: one per finish with a float, plus the extra colour variants.
    phase = {f: int((ranged["finish"] == f).sum()) for f in PHASE_FINISHES}
    kits = len(ranged) + sum((k - 1) * phase[f] for f, k in PHASE_FINISHES.items())
    log.info("Check: %d finishes with a float; %s; that makes %d paint kits",
             len(ranged), ", ".join(f"{f} {n} x {PHASE_FINISHES[f]} kits"
                                    for f, n in phase.items()), kits)
    check = read_table("check_item_list.csv", "check_item_list.py")
    row = pick(check, check=CHECK_KITS_ROW)
    if row is None:
        log.warning("check_item_list.csv lacks the row '%s'", CHECK_KITS_ROW)
    elif int(row["game_file"]) != kits:
        log.warning("check_item_list.csv counts %d paint kits, not %d",
                    int(row["game_file"]), kits)
    else:
        log.info("Check: agrees with check_item_list.csv (%d paint kits)", kits)
    return t.frame()


# Table A3: wear level of the output by wear level of the input (routes of 06).
def table_a3() -> pd.DataFrame:
    routes, weights = load_routes()
    wear = main_input(routes).set_index("market_hash_name")["wear"]
    w = weights[weights["route"].isin(wear.index)]
    missing = set(wear.index) - set(w["route"])
    if missing:
        raise SystemExit(f"{len(missing)} main input routes without weights -- "
                         "re-run 06_arbitrage.py.")
    cols = list(C.WEAR_BOUNDS) + [VANILLA]
    unknown = set(w["out_wear"]) - set(cols)
    if unknown:
        raise SystemExit(f"Unknown output wear levels in the weights: {unknown}")
    # Probability of each output wear level per route, in percent.
    prob = (w.groupby(["route", "out_wear"])["weight"].sum()
            .unstack(fill_value=0.0)
            .reindex(index=wear.index, columns=cols, fill_value=0.0)) * 100
    by_wear = prob.groupby(wear).mean().reindex(list(C.WEAR_BOUNDS))
    overall = prob.mean()
    n = wear.value_counts().reindex(list(C.WEAR_BOUNDS))

    # Cross-check against 06_output_wear.csv (rounded to one decimal).
    ref = read_table("06_output_wear.csv", "06_arbitrage.py", index_col=0)
    gap = (by_wear.round(1) - ref.reindex(index=by_wear.index, columns=cols)
           ).abs().max().max()
    if not gap <= 0.05 + 1e-9:
        log.warning("Rows differ from 06_output_wear.csv by up to %.2f "
                    "percentage points -- different runs of 06?", gap)
    else:
        log.info("Check: rows agree with 06_output_wear.csv")

    t = Rows(1 + len(cols))
    t.add(["Input wear level", "Wear level of the output (%)"])
    t.add([""] + [WEAR_ABBR[c] for c in C.WEAR_BOUNDS] + ["Vanilla"])
    for level in C.WEAR_BOUNDS:
        t.add([level] + [fmt_dec(x, 1) for x in by_wear.loc[level, cols]])
    t.add(["All routes"] + [fmt_dec(x, 1) for x in overall[cols]])

    # Numbers quoted in Sections 2.1, 5.2 and Appendix D (log only).
    log.info("Check: row sums %s", ", ".join(
        f"{s:.2f}" for s in by_wear.sum(axis=1)))
    used = list(C.POOL_WEARS_REALISTIC)
    log.info("Check: all routes -- %s together %.1f %%, with vanilla %.1f %%; "
             "%s alone %.1f %%", and_list([WEAR_ABBR[x] for x in used]),
             overall[used].sum(), overall[used + [VANILLA]].sum(),
             WEAR_ABBR[cols[0]], overall[cols[0]])
    best, worst = cols[0], list(C.WEAR_BOUNDS)[-1]
    pools = load_pools()
    hi = pools.drop_duplicates("market_hash_name").set_index(
        "market_hash_name")["max_float"]
    sub = w[w["route"].map(wear) == worst]
    narrow = sub["output"].map(hi) <= NARROW_MAX + 1e-9
    k = (wear == worst).sum() / 100  # sums over routes -> mean in %
    to_best = sub["out_wear"] == best
    log.info("Check: five %s inputs -> %s %.2f %%, of which %.2f %% from "
             "finishes ending at %.2f; these finishes carry %.2f %% of the "
             "probability of %s routes (%s %.2f %%, %s %.2f %%)", worst, best,
             sub.loc[to_best, "weight"].sum() / k,
             sub.loc[to_best & narrow, "weight"].sum() / k, NARROW_MAX,
             sub.loc[narrow, "weight"].sum() / k, worst, best,
             sub.loc[narrow & to_best, "weight"].sum() / k, cols[1],
             sub.loc[narrow & (sub["out_wear"] == cols[1]), "weight"].sum() / k)
    log.info("Check: %d routes (%s)", len(wear),
             ", ".join(f"{WEAR_ABBR[x]} {n[x]}" for x in C.WEAR_BOUNDS))
    return t.frame()


# Numbers of the Appendix A text that are not in a table (log only): Doppler shares,
# the example route of Section 3.1 and the pseudo routes of Placebo ST.
def checks_a() -> None:
    section("CHECKS FOR THE TEXT OF APPENDIX A")
    routes, weights = load_routes()
    pools = load_pools()
    main = main_input(routes)

    # Doppler and Gamma Doppler
    regular = pools[~pools["stattrak"].astype(bool)].drop_duplicates(["case", "skin"])
    rows = []
    for case, g in regular.groupby("case"):
        cnt = {f: int((g["finish"] == f).sum()) for f in PHASE_FINISHES}
        if not sum(cnt.values()):
            continue
        extra = sum((k - 1) * cnt[f] for f, k in PHASE_FINISHES.items())
        kits = sum(k * cnt[f] for f, k in PHASE_FINISHES.items())
        rows.append({"case": case, "finishes": len(g),
                     **{f: cnt[f] for f in PHASE_FINISHES},
                     "share": sum(cnt.values()) / len(g),
                     "share_if_each_kit": kits / (len(g) + extra)})
    tab = pd.DataFrame(rows)
    log.info("Doppler and Gamma Doppler: %d of %d pools (share of the "
             "finishes, and the share if every paint kit were a finish):",
             len(tab), pools["case"].nunique())
    show(tab.round(3))
    log.info("Share of the finishes from %.3f to %.3f; if every paint kit "
             "were a finish, from %.3f to %.3f", tab["share"].min(),
             tab["share"].max(), tab["share_if_each_kit"].min(),
             tab["share_if_each_kit"].max())
    finish = pools.drop_duplicates("market_hash_name").set_index(
        "market_hash_name")["finish"]
    w = weights[weights["route"].isin(main["market_hash_name"])]
    on_phase = w["output"].map(finish).isin(list(PHASE_FINISHES))
    share = w["weight"].where(on_phase, 0.0).groupby(w["route"]).sum()
    hit = share[share > 0]
    log.info("Check: %d of %d main-sample input routes can yield Doppler or "
             "Gamma Doppler; on these they carry %.1f %% of the probability "
             "on average (%.1f %% to %.1f %%)", len(hit), len(main),
             100 * hit.mean(), 100 * hit.min(), 100 * hit.max())

    # Example route of Section 3.1
    r = routes.set_index("market_hash_name")
    if EXAMPLE_ROUTE not in r.index:
        log.warning("Example route %s not among the routes of 06", EXAMPLE_ROUTE)
    else:
        e = r.loc[EXAMPLE_ROUTE]
        items = read_clean(C.ITEMS_CLEAN, "00_prepare.py").set_index(
            "market_hash_name")
        fmin, fmax = (float(items.loc[EXAMPLE_ROUTE, c])
                      for c in ("min_float", "max_float"))
        w_lo, w_hi = C.WEAR_BOUNDS[e["wear"]]
        a = (max(w_lo, fmin) - fmin) / (fmax - fmin)
        b = (min(w_hi, fmax) - fmin) / (fmax - fmin)
        pool = pools[(pools["case"] == e["case"])
                     & (pools["stattrak"].astype(bool) == bool(e["stattrak"]))
                     ].drop_duplicates("skin")
        log.info("Check, example route %s: %s, input range %.2f to %.2f, "
                 "positions from %.3f to %.3f; pool of %d finishes (%s)",
                 EXAMPLE_ROUTE, e["case"], fmin, fmax, a, b, len(pool),
                 ", ".join(f"{c} {v}" for c, v in
                           pool["category"].value_counts().items()))
        for (lo, hi), g in pool.groupby(["min_float", "max_float"]):
            log.info("  %d finishes with range %.2f to %.2f: output float from "
                     "%.3f to %.3f", len(g), lo, hi, lo + (hi - lo) * a,
                     lo + (hi - lo) * b)
        probs = (weights.loc[weights["route"] == EXAMPLE_ROUTE]
                 .groupby("out_wear")["weight"].sum() * 100)
        log.info("  output wear level (%%): %s", ", ".join(
            f"{k} {v:.2f}" for k, v in probs.items()))
        n_in = C.CRAFT_INPUTS
        log.info("  before: %d x EUR %.3f = EUR %.2f, E[V] EUR %.2f, R = %.3f; "
                 "bound reached at a Covert price of EUR %.2f (E[V] unchanged) "
                 "or at E[V] = EUR %.2f (price unchanged)", n_in, e["p_pre"],
                 n_in * e["p_pre"], e["ev_pre"], e["r_pre"],
                 C.BOUND_STRICT * e["ev_pre"] / n_in,
                 n_in * e["p_pre"] / C.BOUND_STRICT)
        log.info("  after: Covert EUR %.2f, E[V] EUR %.2f, R = %.3f",
                 e["p_post"], e["ev_post"], e["r_post"])

    # Pseudo routes of Placebo ST
    ps = routes[routes["role"] == C.PLACEBO_ROLE]
    pm = ps[ps["main"]]
    pr = pm[pm["r_pre"].notna()]
    log.info("Check: pseudo routes of Placebo ST -- %d from %d glove cases; "
             "%d in the main sample (%d cases); %d with R before the update "
             "(%d cases)", len(ps), ps["case"].nunique(), len(pm),
             pm["case"].nunique(), len(pr), pr["case"].nunique())


# '2026-08-28' -> 'Aug 28' or 'Aug 28, 2026'.
def fmt_day(date: str, year: bool = False) -> str:
    d = pd.Timestamp(date)
    return f"{d:%b} {d.day}" + (f", {d.year}" if year else "")


# Table B1: from the item list to the data window.
def table_b1() -> pd.DataFrame:
    script = "00_prepare.py"
    overview = read_table("00_sample_overview.csv", script)
    no_sale = read_table("00_items_without_window_data.csv", script)
    missing = read_table("00_items_not_collected.csv", script)
    retrieval = read_table("00_retrieval.csv", script)
    retrieval = retrieval[retrieval["run"] == "main"]
    unknown = (set(overview["group"]) | set(no_sale["group"])
               | set(missing["group"])) - set(LIST_GROUPS)
    if unknown:
        raise SystemExit(f"Groups without a label in LIST_GROUPS: {sorted(unknown)}")

    in_window = overview.groupby("group")["items"].sum()
    spikes = overview.groupby("group")["spikes_removed"].sum()
    n_no_sale = no_sale["group"].value_counts()
    n_missing = missing["group"].value_counts()

    t = Rows(7)
    t.add(["Panel A. Items by group"])
    t.add(["Group", "Item list", "Price history", "Not traded",
           "In data window", "Spikes removed", "Retrieved"])
    totals = np.zeros(5, dtype=int)
    for g, label in LIST_GROUPS.items():
        w, ns, nm = (int(in_window.get(g, 0)), int(n_no_sale.get(g, 0)),
                     int(n_missing.get(g, 0)))
        vals = np.array([w + ns + nm, w + ns, ns, w, int(spikes.get(g, 0))])
        totals += vals
        days = sorted(retrieval.loc[retrieval["group"] == g, "date"].unique())
        t.add([label] + [fmt_int(v) for v in vals]
              + [", ".join(fmt_day(d) for d in days) or MISSING])
    t.add(["All groups"] + [fmt_int(v) for v in totals])

    # Panel B: items without a sale in the window, by when they traded.
    lo, hi = (C.T0 + pd.Timedelta(days=d) for d in C.WINDOW)
    first = pd.to_datetime(no_sale["first_trade"])
    last = pd.to_datetime(no_sale["last_trade"])
    no_sale["when"] = np.select([last < lo, first > hi], ["before", "after"],
                                "around")
    tab = pd.crosstab(no_sale["group"], no_sale["when"])
    for c in ["before", "after", "around"]:
        if c not in tab:
            tab[c] = 0
    t.add(["Panel B. Items not traded in the data window, by when they traded"])
    t.add(["Group", "Only before", "Only after", "Before and after", "Total"])
    for g, label in LIST_GROUPS.items():
        if g not in tab.index:
            continue
        r = tab.loc[g]
        t.add([label, fmt_int(r["before"]), fmt_int(r["after"]),
               fmt_int(r["around"]), fmt_int(r[["before", "after", "around"]].sum())])
    t.add(["All groups"]
          + [fmt_int(tab[c].sum()) for c in ["before", "after", "around"]]
          + [fmt_int(tab[["before", "after", "around"]].to_numpy().sum())])

    status = (missing["status"].value_counts() if "status" in missing
              else pd.Series(dtype=int))
    log.info("Check: %d items in the list, %d with a price history, %d without "
             "a sale in the window, %d in the data window, %d spikes removed",
             *totals)
    log.info("Check: collector status of the missing items -- %s",
             status.to_dict())
    return t.frame()


# Table B2: sample selection by group and screening step.
def table_b2() -> pd.DataFrame:
    flow = read_table("01_sample_flow.csv", "01_screening.py", index_col=0)
    missing = [c for c in FLOW_STEPS if c not in flow.columns]
    if missing:
        raise SystemExit(f"01_sample_flow.csv lacks columns {missing}")
    width = 1 + max(len(v) for v in B2_PANELS.values())
    t = Rows(width)
    for panel, roles in B2_PANELS.items():
        t.add([panel])
        t.add(["Step"] + [B2_HEADS.get(r, B_ROLES.get(r)) for r in roles])
        for col, label in FLOW_STEPS.items():
            t.add([label] + [fmt_int(flow.loc[r, col]) if r in flow.index
                             else MISSING for r in roles])

    main = flow[list(FLOW_STEPS)[-1]]
    log.info("Check: main sample %d items (%s)", main["TOTAL"],
             ", ".join(f"{B_ROLES[r]} {main[r]}"
                       for r in B2_PANELS["Panel A. Analysis groups"]))
    return t.frame()


# Table B3: trade share before screening, and the sample variants.
def table_b3() -> pd.DataFrame:
    liq = read_table("00_liquidity_by_role.csv", "00_prepare.py", index_col=0)
    var = read_table("01_sample_variants.csv", "01_screening.py", index_col=0)
    for c in LIQ_COLUMNS:
        if c not in liq.columns:
            liq[c] = 0
    roles = [r for r in B_ROLES if r in liq.index]

    t = Rows(1 + len(LIQ_COLUMNS) + 1)
    t.add(["Panel A. Items in the data window by trade share in the "
           "estimation window"])
    t.add(["Group"] + list(LIQ_COLUMNS.values()) + ["Total"])
    for r in roles:
        vals = [int(liq.loc[r, c]) for c in LIQ_COLUMNS]
        t.add([B_ROLES[r]] + [fmt_int(v) for v in vals] + [fmt_int(sum(vals))])
    tot = [int(liq[c].sum()) for c in LIQ_COLUMNS]
    t.add(["All groups"] + [fmt_int(v) for v in tot] + [fmt_int(sum(tot))])

    t.add(["Panel B. Items by sample variant"])
    t.add(["Group"] + list(VARIANT_COLUMNS.values()))
    for r in roles + ["TOTAL"]:
        t.add([B_ROLES.get(r, "All groups")]
              + [fmt_int(var.loc[r, c]) if r in var.index else MISSING
                 for c in VARIANT_COLUMNS])

    log.info("Check: main sample in Panel B %d, illiquid variant %d",
             var.loc["TOTAL", "main"], var.loc["TOTAL", "illiquid"])
    # Trade shares the illiquid comparison group has in effect, and the main-sample
    # items the burn-in variant drops.
    s = pd.read_parquet(C.SAMPLES)
    ts = s.loc[s["illiquid"], "trade_share"]
    log.info("Check: illiquid variant in effect %.0f %% to %.0f %% trade share; "
             "burn-in variant drops %d main-sample items", 100 * ts.min(),
             100 * ts.max(), int((s["main"] & ~s["burn_in30"]).sum()))
    return t.frame()


# Table B4: largest market-wide daily price moves.
def table_b4() -> pd.DataFrame:
    jd = read_table("00_jump_days.csv", "00_prepare.py")
    marked = {e + k for e in C.EVENT_DAYS.values() for k in (0, 1)}
    t = Rows(7)
    panels = {"full": "Panel A. Data window",
              "estimation_window": "Panel B. Estimation window"}
    for key, n in JUMP_ROWS.items():
        sub = jd[jd["ranking"] == key].sort_values("rank").head(n)
        t.add([panels[key]])
        t.add(["Rank", "Date", "t", "Items", "Median return (%)", "Falling (%)",
               "Volume (multiple)"])
        for r in sub.itertuples(index=False):
            tt = "0" if r.t == 0 else f"{r.t:+d}".replace("-", MINUS)
            t.add([fmt_int(r.rank), fmt_day(r.date, True),
                   tt + (EVENT_MARK if r.t in marked else ""), fmt_int(r.n_items),
                   fmt_signed(r.median_ret_pct, 1),
                   fmt_dec(100 * r.share_negative, 1),
                   fmt_dec(r.volume_vs_median, 2)])

    full = jd[jd["ranking"] == "full"].sort_values("rank")
    est = jd[jd["ranking"] == "estimation_window"]
    after = full[full["t"].between(0, BUDAPEST_MAJOR)]
    log.info("Check: largest day t = %d (%+.1f %%, %.1f %% falling); %d of the "
             "top %d days lie in t = 0 to +%d; largest |median| in the "
             "estimation window %.1f %%", full.iloc[0]["t"],
             full.iloc[0]["median_ret_pct"], 100 * full.iloc[0]["share_negative"],
             len(after), len(full), BUDAPEST_MAJOR,
             est["median_ret_pct"].abs().max())
    return t.frame()


# Table C1: return properties (02), market model (04), return correlation (02).
def table_c1() -> pd.DataFrame:
    props = read_table("02_table3_return_properties.csv", "02_descriptives.py")
    diag = read_table("04_estimation_diagnostics.csv", "04_abnormal_returns.py")
    caar = read_table("04_caar_main.csv", "04_abnormal_returns.py")
    corr = read_table("02_return_correlation.csv", "02_descriptives.py")

    props = props.set_index(props.columns[0])
    diag = diag.set_index(diag.columns[0])
    rbar = (caar[caar["model"] == C.MODEL_MAIN].groupby("role")["rbar"].first())
    roles = [r for r in APPENDIX_ROLES if r in props.index]

    width = 8
    rows: list[list[str]] = []

    def row(cells: list[str]) -> None:
        rows.append(cells + [""] * (width - len(cells)))

    row(["Panel A. Returns in the estimation window"])
    row(["Group", "Items", "Returns", "SD (%)", "Skewness", "Excess kurtosis",
         "Zero returns (%)", "AR(1)"])
    for r in roles:
        p = props.loc[r]
        row([APPENDIX_ROLES[r], fmt_int(p["Items"]), fmt_int(p["Returns (median)"]),
             fmt_dec(p["SD % (median)"], 2), fmt_dec(p["Skewness (median)"], 2),
             fmt_dec(p["Excess kurtosis (median)"], 2),
             fmt_dec(100 * p["Zero returns (share)"], 1),
             fmt_dec(p["AR(1) (median)"], 2)])

    row(["Panel B. Market model estimates"])
    row(["Group", "β (α = 0)", "α (% per day)", "R²", "σ (%)", "r̄"])
    for r in roles:
        if r not in diag.index:
            log.warning("%s missing from 04_estimation_diagnostics", r)
            continue
        d = diag.loc[r]
        row([APPENDIX_ROLES[r], fmt_dec(d["beta_alpha0"], 3),
             fmt_dec(d["alpha_pct"], 3), fmt_dec(d["r2"], 3),
             fmt_dec(d["sigma_pct"], 2), fmt_dec(rbar.get(r), 3)])

    row(["Panel C. Correlation of returns between two items"])
    row(["Pair of items", "Pairs", "Mean, daily", "Mean, five-day",
         "Median, five-day"])
    for _, c in corr.iterrows():
        label = PAIR_LABELS.get(c["Pair type"], c["Pair type"])
        row([label, fmt_int(c["Pairs"]), fmt_dec(c["Mean corr (daily)"], 3),
             fmt_dec(c["Mean corr (5-day)"], 3), fmt_dec(c["Median corr (5-day)"], 3)])
    return pd.DataFrame(rows)


# Table C2: pre-event windows (anticipation), main model and estimated alpha.
def table_c2() -> pd.DataFrame:
    pre = read_table("04_pretrend.csv", "04_abnormal_returns.py")
    caar = read_table("04_caar_main.csv", "04_abnormal_returns.py")
    n_main = main_n(caar[caar["model"] == C.MODEL_MAIN])

    rows = []
    for win, win_label in PRE_LABELS.items():
        first = True
        for role in APPENDIX_ROLES:
            a = pick(pre, model=C.MODEL_MAIN, window=win, role=role)
            b = pick(pre, model=ALPHA_MODEL, window=win, role=role)
            if a is None:
                continue
            mark = ESTABLISHED_MARK if is_established(a) else ""
            rows.append({
                "Window": win_label if first else "",
                "Group": group_label(role, n_main.get(role, a["n"])),
                "CAAR (%)": fmt_signed(a["caar_pct"], 1) + mark,
                "Median (%)": fmt_signed(a["median_pct"], 1),
                "t_BMP": fmt_signed(a.get("t_bmp"), 2),
                "t_KP": fmt_signed(a.get("t_kp"), 2),
                "Rank z": fmt_signed(a.get("z_rank"), 2),
                "Sign z": fmt_signed(a.get("z_sign"), 2),
                "CAAR, α estimated (%)": (fmt_signed(b["caar_pct"], 1)
                                          if b is not None else MISSING),
                "t_KP, α estimated": (fmt_signed(b.get("t_kp"), 2)
                                      if b is not None else MISSING),
            })
            first = False
    return pd.DataFrame(rows)


# Table C3: all event windows and tests, plus raw median and portfolio t.
def table_c3() -> pd.DataFrame:
    caar = read_table("04_caar_main.csv", "04_abnormal_returns.py")
    raw = read_table("03_raw_event_windows.csv", "03_event_plots.py")
    port = read_table("04_portfolio.csv", "04_abnormal_returns.py")
    caar = caar[caar["model"] == C.MODEL_MAIN]
    n_main = main_n(caar)

    rows = []
    for win, win_label in EVENT_LABELS.items():
        first = True
        for role in APPENDIX_ROLES:
            a = pick(caar, window=win, role=role)
            if a is None:
                log.warning("No row for window %s, role %s", win, role)
                continue
            r = pick(raw, window=win, role=role)
            p = pick(port, window=win, role=role)
            # The benchmark portfolio regressed on itself is identically zero.
            t_port = (MISSING if role == BENCHMARK_ROLE or p is None
                      else fmt_signed(p["t"], 2))
            mark = ESTABLISHED_MARK if is_established(a) else ""
            rows.append({
                "Window": win_label if first else "",
                "Group": group_label(role, n_main[role]),
                "CAAR (%)": fmt_signed(a["caar_pct"], 1) + mark,
                "Median (%)": fmt_signed(a["median_pct"], 1),
                "Raw median (%)": (fmt_signed(r["median_pct"], 1)
                                   if r is not None else MISSING),
                "t_BMP": fmt_signed(a.get("t_bmp"), 2),
                "t_KP": fmt_signed(a.get("t_kp"), 2),
                "Rank z": fmt_signed(a.get("z_rank"), 2),
                "Sign z": fmt_signed(a.get("z_sign"), 2),
                "t, portfolio": t_port,
            })
            first = False
    return pd.DataFrame(rows)


# Table C4: CAAR and t_KP for every return model of 04.
def table_c4() -> pd.DataFrame:
    models = read_table("04_robustness_models.csv", "04_abnormal_returns.py")
    n_main = main_n(models[models["model"] == C.MODEL_MAIN])

    cols = ["Group"] + list(EVENT_LABELS.values())
    rows: list[list[str]] = [cols]
    for model, panel in MODEL_PANELS.items():
        if not (models["model"] == model).any():
            log.warning("Model %s missing from 04_robustness_models", model)
            continue
        rows.append([panel] + [""] * len(EVENT_LABELS))
        for role in ANALYSIS_ROLES:
            # Placebo ST minus its own mean is zero by construction.
            if model == "placebo_st_adj" and role == C.PLACEBO_BENCHMARK_ROLE:
                continue
            cells = []
            for win in EVENT_LABELS:
                r = pick(models, model=model, window=win, role=role)
                cells.append(MISSING if r is None
                             else fmt_cell(r["caar_pct"], r.get("t_kp")))
            rows.append([group_label(role, n_main[role])] + cells)
    return pd.DataFrame(rows)


# Table C5: benchmark indices, sample variants and the shorter estimation window.
def table_c5() -> pd.DataFrame:
    grid = read_table("04_robustness_grid.csv", "04_abnormal_returns.py")

    win_labels = [EVENT_LABELS[w] for w in TABLE_C5_WINDOWS]
    width = 1 + len(TABLE_C5_ROLES) * (1 + len(TABLE_C5_WINDOWS))
    rows: list[list[str]] = []
    top, sub = [""], [""]
    for role in TABLE_C5_ROLES:
        top += [APPENDIX_ROLES[role]] + [""] * len(TABLE_C5_WINDOWS)
        sub += ["n"] + win_labels
    rows += [top, sub]

    for label, index, variant in TABLE_C5_ROWS:
        if index is None:
            rows.append([label] + [""] * (width - 1))
            continue
        cells = [label]
        for role in TABLE_C5_ROLES:
            n = None
            vals = []
            for win in TABLE_C5_WINDOWS:
                r = pick(grid, index=index, variant=variant, window=win, role=role)
                if r is None:
                    vals.append(MISSING)
                    continue
                n = r["n"]
                if ok(r.get("t_kp")):
                    vals.append(fmt_cell(r["caar_pct"], r["t_kp"]))
                else:
                    vals.append(fmt_cell(r["caar_pct"], r.get("t_bmp"), BMP_MARK))
            cells += [fmt_int(n) if n is not None else MISSING] + vals
        rows.append(cells)
    return pd.DataFrame(rows)


# Table D1: input regressions of 05, all specifications and coefficients.
def table_d1() -> pd.DataFrame:
    script = "05_cross_section.py"
    coefs = read_table("05_regression_input.csv", script, index_col=0)
    fit = read_table("05_regression_fit.csv", script)
    fit = fit[fit["group"] == "treated_input"]

    cols = []
    for spec in TABLE_D1_SPECS:
        tab = coefs[coefs["spec"] == spec]
        info = pick(fit, spec=spec)
        if tab.empty or info is None:
            raise SystemExit(f"No results for '{spec}' -- re-run {script}.")
        cols.append((tab, info))

    names = set(coefs.index.astype(str))
    wears = [w for w in C.WEAR_BOUNDS if f"{WEAR_FE_PREFIX}{w}" in names]
    wear_base = [w for w in C.WEAR_BOUNDS if w not in wears]
    if len(wear_base) != 1:
        log.warning("Wear base level not unique: %s", wear_base)
    cats = sorted(n[len(CATEGORY_FE_PREFIX):] for n in names
                  if n.startswith(CATEGORY_FE_PREFIX))
    if CATEGORY_BASE in cats or not cats or CATEGORY_BASE > cats[0]:
        log.warning("Category base '%s' does not fit the dummies %s -- check "
                    "CATEGORY_BASE", CATEGORY_BASE, cats)

    t = Rows(1 + len(cols))
    # Word drops a leading empty cell in the first pasted line, hence the stub head.
    t.add(["Variable"] + [spec.split(" ")[0] for spec in TABLE_D1_SPECS])
    t.add([""] + list(TABLE_D1_SPECS.values()))

    def coef_rows(label: str, keys: list[str]) -> None:
        cells, ses = [label], [""]
        for tab, _ in cols:
            hit = [k for k in keys if k in tab.index]
            if hit:
                r = tab.loc[hit[0]]
                cells.append(fmt_coef(r["coef"]) + star_of(r))
                ses.append(f"({fmt_coef(r['se'])})")
            else:
                cells.append("")
                ses.append("")
        t.add(cells)
        t.add(ses)

    for keys, label in TABLE_D1_REGRESSORS:
        coef_rows(label, keys)
    # Fixed effects under a heading row each.
    t.add(["Wear level"])
    for w in wears:
        coef_rows(w, [f"{WEAR_FE_PREFIX}{w}"])
    t.add(["Weapon category"])
    for c in cats:
        coef_rows(c, [f"{CATEGORY_FE_PREFIX}{c}"])
    coef_rows("Constant", ["const"])

    def wild(tab: pd.DataFrame, info: pd.Series) -> str:
        if not any(k in tab.index for k in POOL_VARIABLES):
            return ""
        return fmt_p_wild(info["p_wild"])

    t.add(["CAR window"] + [EVENT_LABELS.get(str(i["window"]), str(i["window"]))
                            for _, i in cols])
    t.add(["Items"] + [fmt_int(i["n"]) for _, i in cols])
    t.add(["Clusters (pools)"] + [fmt_int(i["clusters"]) for _, i in cols])
    t.add(["R²"] + [fmt_dec(i["r2"], 3) for _, i in cols])
    t.add(["Wild bootstrap p"] + [wild(tab, i) for tab, i in cols])

    # Range quoted in Section 5.2 for (3) to (5) (log only).
    gam = [tab.loc[[k for k in POOL_VARIABLES if k in tab.index][0], "coef"]
           for tab, i in cols[2:5]]
    log.info("Check: pool value in (3) to (5) from %.3f to %.3f", min(gam), max(gam))

    # Base of the weapon-category effects: the weapons of that category in the input.
    items = pd.read_parquet(C.ITEMS_CLEAN).set_index("market_hash_name")
    s = pd.read_parquet(C.SAMPLES).set_index("market_hash_name")
    inp = items.loc[s.index[s["main"]]]
    base = inp[(inp["role"] == "treated_input") & (inp["category"] == CATEGORY_BASE)]
    log.info("Check: base levels %s and %s (%s; %d items)",
             wear_base[0] if wear_base else MISSING, CATEGORY_BASE,
             ", ".join(base["weapon"].value_counts().index), len(base))
    return t.frame()


# Table D2: event-window CARs of the input by binding status (06).
def table_d2() -> pd.DataFrame:
    cb = read_table("06_car_by_binding.csv", "06_arbitrage.py")
    windows = [w for w in EVENT_LABELS
               if (cb["window"].astype(str) == w).any()]
    missing = [s for s in list(BINDING_STATUS) + [BINDING_DIFF]
               if not (cb["status"] == s).any()]
    if missing:
        raise SystemExit(f"06_car_by_binding.csv lacks {missing} -- re-run "
                         "06_arbitrage.py.")

    t = Rows(1 + len(windows))
    t.add(["Measure"] + [EVENT_LABELS[w] for w in windows])
    letters = iter("ABCDEFG")
    for status, label in BINDING_STATUS.items():
        ns = cb.loc[cb["status"] == status, "items"].unique()
        if len(ns) != 1:
            log.warning("%s: items differ across windows (%s), using the largest",
                        status, ns)
        t.add([f"Panel {next(letters)}. {label} (n = {fmt_int(ns.max())})"])
        caar, med, pos = ["CAAR (%)"], ["Median (%)"], ["Positive (%)"]
        for w in windows:
            r = pick(cb, window=w, status=status)
            caar.append(fmt_signed(r["caar_pct"], 1) if r is not None else MISSING)
            med.append(fmt_signed(r["median_pct"], 1) if r is not None else MISSING)
            pos.append(fmt_dec(100 * r["share_positive"], 1)
                       if r is not None else MISSING)
        t.add(caar)
        t.add(med)
        t.add(pos)

    t.add([f"Panel {next(letters)}. Binding minus slack"])
    diff, se, wild, n = (["Difference (log points)"], [""],
                         ["Wild bootstrap p"], ["Items (clusters)"])
    for w in windows:
        r = pick(cb, window=w, status=BINDING_DIFF)
        if r is None:
            for lst in (diff, se, wild, n):
                lst.append(MISSING)
            continue
        diff.append(fmt_coef(r["diff_log"]) + star_of(r))
        se.append(f"({fmt_coef(r['se'])})")
        wild.append(fmt_p_wild(r["p_wild"]))
        n.append(f"{fmt_int(r['items'])} ({fmt_int(r['clusters'])})")
    for lst in (diff, se, wild, n):
        t.add(lst)
    return t.frame()


# Table D3: routes relative to the bound by input wear and output pool.
def table_d3() -> pd.DataFrame:
    bt = read_table("06_bound_test.csv", "06_arbitrage.py", index_col=0)
    wear_rows = {f"{WEAR_ROW_PREFIX}{w}": w for w in C.WEAR_BOUNDS}
    missing = [g for g in list(wear_rows) + list(POOL_ROWS) if g not in bt.index]
    if missing:
        raise SystemExit(f"06_bound_test.csv lacks rows {missing} -- re-run "
                         "06_arbitrage.py.")

    # Both panels split a row of Table 5; the parts must add up.
    for parts, whole in [(list(wear_rows), "all"), (list(POOL_ROWS), "binding")]:
        if whole in bt.index and bt.loc[parts, "n"].sum() != bt.loc[whole, "n"]:
            log.warning("Rows %s add up to %d routes, '%s' has %d", parts,
                        bt.loc[parts, "n"].sum(), whole, bt.loc[whole, "n"])

    b, h = f"{C.BOUND_STRICT:.3f}", f"{C.BOUND_HOLDER:.1f}"
    t = Rows(8)
    t.add(["Subgroup", "Routes", "Median R, before", f"Below {b}, before (%)",
           "Median R, after", f"Below {b}, after (%)",
           f"Between {b} and {h}, after (%)", f"Above {h}, after (%)"])

    def line(label: str, r: pd.Series) -> list[str]:
        return [label, fmt_int(r["n"]), fmt_dec(r["median_r_pre"], 2),
                fmt_dec(r["below_pre_pct"], 1), fmt_dec(r["median_r_post"], 2),
                fmt_dec(r["below_post_pct"], 1), fmt_dec(r["band_post_pct"], 1),
                fmt_dec(r["above_post_pct"], 1)]

    t.add(["Panel A. All routes, by wear level of the input"])
    for key, label in wear_rows.items():
        t.add(line(label, bt.loc[key]))
    t.add(["Panel B. Binding routes, by output pool of the case"])
    for key, label in POOL_ROWS.items():
        t.add(line(label, bt.loc[key]))
    return t.frame()


# Table D4: decomposition of the change in log R (A) and the ratchet days (B).
def table_d4() -> pd.DataFrame:
    script = "06_arbitrage.py"
    dec = read_table("06_decomposition.csv", script, index_col=0)
    path = read_table("06_path.csv", script, index_col=0)
    b = f"{C.BOUND_STRICT:.3f}"

    t = Rows(8)
    t.add(["Panel A. Change against the level before the update, by window"])
    t.add(["Window", "Routes", f"Gap to {b}, before", D_LN_P, D_LN_EV, D_LN_R,
           "Gap closed (%)", "From Coverts (%)"])
    for key, (lo, hi) in C.DECOMP_WINDOWS.items():
        if key not in dec.index:
            log.warning("Window '%s' missing from 06_decomposition", key)
            continue
        r = dec.loc[key]
        # 06 gives the output side the sign that makes it add up; shown here as Δ ln E[V].
        if not np.isclose(r["input"] + r["output"], r["ratio"]):
            log.warning("%s: input + output != ratio in 06_decomposition", key)
        t.add([days_label(lo, hi), fmt_int(r["n"]), fmt_dec(r["gap_pre"], 2),
               fmt_signed(r["input"], 2), fmt_signed(-r["output"], 2),
               fmt_signed(r["ratio"], 2), fmt_dec(r["share_closed_pct"], 1),
               fmt_dec(r["input_share_pct"], 1)])

    # Panel B keeps the columns of Panel A; the gap has no daily value and stays empty.
    t.add(["Panel B. Daily path during the interim recovery of the output"])
    t.add(["Day", "Routes", "", D_LN_P, D_LN_EV, D_LN_R, "Median R",
           f"Below {b} (%)"])
    days = [d for d in RATCHET_DAYS if d in path.index]
    for d in days:
        p = path.loc[d]
        if p["decomp_n"] != p["n_binding"]:
            log.warning("t = %d: %d routes in the decomposition, %d in the "
                        "median", d, p["decomp_n"], p["n_binding"])
        t.add([days_label(d, d), fmt_int(p["decomp_n"]), "",
               fmt_signed(p["decomp_input"], 2),
               fmt_signed(-p["decomp_output"], 2),
               fmt_signed(p["decomp_ratio"], 2),
               fmt_dec(p["median_binding"], 2),
               fmt_dec(p["below_bound_binding_pct"], 1)])
    t.add(["Change (%)"])
    for a, z in RATCHET_CHANGES:
        if a not in path.index or z not in path.index:
            log.warning("Ratchet change %d to %d: day missing from 06_path", a, z)
            continue
        pa, pz = path.loc[a], path.loc[z]
        t.add([days_label(a, z), "", "",
               fmt_signed(to_pct(pz["decomp_input"] - pa["decomp_input"]), 1),
               fmt_signed(to_pct(pa["decomp_output"] - pz["decomp_output"]), 1),
               fmt_signed(to_pct(pz["decomp_ratio"] - pa["decomp_ratio"]), 1)])
    return t.frame()


# Table D5: overshoot, calculation variants (A) and outcome shape (B).
def table_d5() -> pd.DataFrame:
    script = "06_arbitrage.py"
    rob = read_table("06_robustness.csv", script, index_col=0)
    ov = read_table("06_overshoot.csv", script)
    missing = [k for k in ROBUSTNESS_LABELS if k not in rob.index]
    if missing:
        raise SystemExit(f"06_robustness.csv lacks rows {missing} -- re-run "
                         f"{script} or check ROBUSTNESS_LABELS.")
    b, h = f"{C.BOUND_STRICT:.3f}", f"{C.BOUND_HOLDER:.1f}"

    # Variants as columns, results as rows (otherwise the table needs a landscape page).
    t = Rows(1 + len(ROBUSTNESS_LABELS))
    variants = list(ROBUSTNESS_LABELS)
    rb = rob.loc[variants]

    t.add(["Panel A. Calculation variants"])
    t.add(["Statistic"] + list(ROBUSTNESS_LABELS.values()))
    panel_a = [
        ("Routes", "routes", fmt_int),
        ("Binding routes", "binding", fmt_int),
        ("Median R, all", "median_r_post_all", lambda v: fmt_dec(v, 2)),
        ("Median R, binding", "median_r_post_binding", lambda v: fmt_dec(v, 2)),
        (f"Binding below {b} (%)", "binding_below_pct", lambda v: fmt_dec(v, 1)),
        (f"Binding between {b} and {h} (%)", "binding_band_pct",
         lambda v: fmt_dec(v, 1)),
        ("Gap closed at t = 0 (%)", "event_day_closed_pct",
         lambda v: fmt_dec(v, 1)),
        ("Bound cleared (day)", "bound_cleared_day", fmt_int),
        ("Half-life (days)", "half_life_days", lambda v: fmt_dec(v, 1)),
    ]
    for label, col, f in panel_a:
        t.add([label] + [f(v) for v in rb[col]])

    # In Word each measure spans two of the six columns (merged cells).
    shapes = []
    for key in OVERSHOOT_LABELS:
        r = pick(ov, measure=key)
        if r is None:
            log.warning("Measure '%s' missing from 06_overshoot", key)
        shapes.append(r)
    t.add(["Panel B. Shape of the outcome distribution, binding routes"])
    t.add(["Statistic"] + list(OVERSHOOT_LABELS.values()))

    def cells(f) -> list[str]:
        return [MISSING if r is None else f(r) for r in shapes]

    t.add(["Routes"] + cells(lambda r: fmt_int(r["routes"])))
    t.add(["Clusters (pools)"] + cells(lambda r: fmt_int(r["clusters"])))
    t.add(["Median of the measure"]
          + cells(lambda r: fmt_dec(r["median_measure"], 2)))
    t.add(["Spearman ρ with R after"]
          + cells(lambda r: fmt_signed(r["spearman_rho"], 2)))
    t.add(["Coefficient on the rank"]
          + cells(lambda r: fmt_coef(r["coef_rank"]) + star_of(r)))
    t.add([""] + cells(lambda r: f"({fmt_coef(r['se'])})"))
    t.add(["Wild bootstrap p"] + cells(lambda r: fmt_p_wild(r["p_wild"])))

    # Range quoted in Section 5.3 across the six variants (log only).
    log.info("Check: binding median R after from %.2f to %.2f, at most %.1f %% "
             "below the bound", rb["median_r_post_binding"].min(),
             rb["median_r_post_binding"].max(), rb["binding_below_pct"].max())
    return t.frame()


# Table E1: abnormal volume (multiple of normal) by reference, group and window.
def table_e1() -> pd.DataFrame:
    av = read_table("08_abnormal_volume.csv", "08_volume.py")
    labels = {w: window_label(*C.VOLUME_WINDOWS[w]) for w in VOLUME_WINDOWS}
    labels["0"] = "t = 0"

    t = Rows(1 + len(VOLUME_WINDOWS))
    t.add(["Group"] + list(labels.values()))
    n_items: dict[str, int] = {}
    for ref, panel in VOLUME_PANELS.items():
        t.add([panel])
        for g, label in VOL_LABELS.items():
            if g == ref:
                continue
            # Multiple, test statistic on the line below; item counts in the log.
            cells, stats = [label], [""]
            for w in VOLUME_WINDOWS:
                r = pick(av, reference=ref, window=w, group=g)
                if r is None:
                    log.warning("No row for reference %s, window %s, group %s",
                                ref, w, g)
                    cells.append(MISSING)
                    stats.append("")
                    continue
                n_items.setdefault(g, int(r["n"]))
                if n_items[g] != int(r["n"]):
                    log.warning("%s: n differs across windows or references", g)
                mark = ESTABLISHED_MARK if is_established(r) else ""
                cells.append(fmt_dec(r["multiple"], 2) + mark)
                if ok(r.get("t_kp")):
                    stats.append(f"({fmt_signed(r['t_kp'], 2)})")
                else:
                    stats.append(f"({fmt_signed(r.get('t_bmp'), 2)}){BMP_MARK}")
            t.add(cells)
            t.add(stats)

    # Multiples quoted in Sections 5.1 and 6.2 (log only).
    at0 = av[(av["reference"] == "none") & (av["window"].astype(str) == "0")]
    log.info("Check: multiples at t = 0 -- %s", ", ".join(
        f"{VOL_LABELS[g]} {m:.2f}" for g, m in zip(at0["group"], at0["multiple"])))
    log.info("Check: items -- %s", ", ".join(
        f"{VOL_LABELS[g]} {n_items[g]}" for g in VOL_LABELS if g in n_items))
    return t.frame()


# Table E2: echo of the seven-day hold (A) and unit balance (B).
def table_e2() -> pd.DataFrame:
    script = "08_volume.py"
    echo = read_table("08_echo.csv", script, index_col=0)
    bal = read_table("08_unit_balance.csv", script)
    missing = [g for g in ECHO_GROUPS if g not in echo.index]
    if missing or bal.empty:
        raise SystemExit(f"08_echo.csv lacks {missing} or 08_unit_balance.csv "
                         f"is empty -- re-run {script}.")
    wave = window_label(*C.ECHO_WAVE)
    back = window_label(*C.ECHO_WINDOW)

    t = Rows(7)
    t.add(["Panel A. Echo of the seven-day trade hold"])
    t.add(["Group", "Items", f"First wave, {wave}", f"Returned, {back}",
           "Echo share (%)", "Relative to control group (pp)", "p-value"])
    for g in ECHO_GROUPS:
        r = echo.loc[g]
        ref = g == C.VOL_CONTROL
        t.add([VOL_LABELS[g], fmt_int(r["items"]), fmt_int(r["wave_units"]),
               fmt_int(r["echo_units"]), fmt_dec(100 * r["share"], 1),
               "" if ref else fmt_signed(100 * r["diff_vs_control"], 1),
               "" if ref else fmt_p_boot(r["p_diff"])])
        t.add(["", "", "", "",
               fmt_ci(100 * r["share_lo"], 100 * r["share_hi"], 1),
               "" if ref else fmt_ci(100 * r["diff_lo"], 100 * r["diff_hi"], 1,
                                     signed=True), ""])

    # Unit balance in the columns of Panel A, so that Word needs no merged cells.
    b = bal.iloc[0]
    t.add(["Panel B. Unit balance"])
    t.add(["Statistic", "", f"Input, {window_label(*C.BALANCE_BUY)}",
           f"Output, {window_label(*C.BALANCE_SELL)}",
           "Input units per output unit"])
    t.add(["Excess units", "", fmt_int(b["covert_units"]),
           fmt_int(b["knife_units"]), fmt_dec(b["coverts_per_knife"], 2)])
    t.add(["", "", "", "", fmt_ci(b["ci_lo"], b["ci_hi"], 2)])
    return t.frame()


# Table E3: robustness of the change in efficient volatility (07).
def table_e3() -> pd.DataFrame:
    script = "07_volatility.py"
    rob = read_table("07_robustness.csv", script, index_col=0)
    K, K2 = C.VOL_SIGNATURE_K, C.VOL_SIGNATURE_K_ALT
    # Row labels of 07, built from the same config values.
    variants = {
        f"main: signature 1-{K} weeks": "Main",
        f"signature 1-{K2} weeks": f"k = 1 to {K2}",
        f"post window {C.VOL_POST_SHORT}": "After: " + days_label(*C.VOL_POST_SHORT),
        f"pre window {C.VOL_PRE_CALM}": "Before: " + days_label(*C.VOL_PRE_CALM),
        "per item, weekly autocovariance": "Weekly autocovariance",
        "per item, daily trade-to-trade": "Daily trade-to-trade",
        "liquidity >= 50 %": f"Trade share ≥ {C.LIQ_ROBUST_MIN:.0%}",
        "not demeaned": "Not demeaned",
    }
    missing = [v for v in variants if v not in rob.index]
    if missing:
        raise SystemExit(f"07_robustness.csv lacks rows {missing} -- re-run "
                         f"{script} or check the labels in table_e3.")

    t = Rows(6)
    t.add(["Variant", "Items", "Efficient, change (%)", "Noise, change (%)",
           "Relative to control group (%)", "p-value"])
    for i, (key, label) in enumerate(ROBUSTNESS_GROUPS.items()):
        t.add([f"Panel {'ABC'[i]}. {label}"])
        for v, vlabel in variants.items():
            r = rob.loc[v]
            cells = [vlabel, fmt_int(r[f"{key}_n"]),
                     fmt_signed(r[f"{key}_eff_chg"], 1),
                     fmt_signed(r[f"{key}_noise_chg"], 1)]
            if key != C.VOL_CONTROL:
                cells += [fmt_signed(r[f"{key}_did_eff"], 1),
                          fmt_p_boot(r[f"{key}_did_eff_p"])]
            t.add(cells)

    # Variants that Section 5.4 calls only marginally significant (log only).
    weak = [f"{variants[v]} ({k}, p = {rob.loc[v, f'{k}_did_eff_p']:.3f})"
            for v in variants for k in ("output", "input")
            if rob.loc[v, f"{k}_did_eff_p"] >= ESTABLISHED_LEVEL]
    log.info("Check: p >= 0.05 relative to the control group -- %s",
             "; ".join(weak) or "none")
    return t.frame()


# Table E4: volatility of all groups (A) and the median V(k) (B, C).
def table_e4() -> pd.DataFrame:
    script = "07_volatility.py"
    vol = read_table("07_volatility.csv", script, index_col=0)
    did = read_table("07_did.csv", script)
    sig = read_table("07_signature.csv", script, index_col=[0, 1])
    missing = [g for g in VOL_LABELS if g not in vol.index]
    if missing:
        raise SystemExit(f"07_volatility.csv lacks rows {missing} -- re-run "
                         f"{script}.")
    vcols = [c for c in sig.columns if c.startswith("V")]

    # As in Table 6: differs from both references at 5 % with the same sign.
    def established(g: str) -> bool:
        r = [pick(did, group=g, reference=ref, measure="efficient")
             for ref in VOL_REFERENCES]
        if any(x is None for x in r):
            return False
        return (all(x["p_boot"] < ESTABLISHED_LEVEL for x in r)
                and len({np.sign(x["did_pct"]) for x in r}) == 1)

    def share(x) -> str:
        # A negative intercept of the signature leaves the noise undefined.
        return fmt_dec(x, 1) if ok(x) and x >= 0 else MISSING

    t = Rows(7)
    t.add(["Panel A. Weekly volatility before and after the update"])
    t.add(["Group", "Items (skins)", "Efficient, before (%)",
           "Efficient, after (%)", "Change (%)", "Noise share, before (%)",
           "Noise share, after (%)"])
    for g, label in VOL_LABELS.items():
        r = vol.loc[g]
        mark = ESTABLISHED_MARK if established(g) else ""
        t.add([label, f"{fmt_int(r['items'])} ({fmt_int(r['skins'])})",
               fmt_dec(r["efficient_pre"], 2), fmt_dec(r["efficient_post"], 2),
               fmt_signed(r["efficient_chg_pct"], 1) + mark,
               share(r["noise_share_pre"]), share(r["noise_share_post"])])
        t.add(["", "", "", "",
               fmt_ci(r["efficient_chg_lo"], r["efficient_chg_hi"], 1,
                      signed=True)])

    for letter, w, word in (("B", "pre", "before"), ("C", "post", "after")):
        t.add([f"Panel {letter}. Median variance of k-week changes, {word} "
               "the update (× 10⁻³)"])
        t.add(["Group"] + [f"V({c[1:]})" for c in vcols])
        for g, label in VOL_LABELS.items():
            if (g, w) not in sig.index:
                log.warning("07_signature lacks %s, %s", g, w)
                continue
            t.add([label] + [fmt_dec(1e3 * sig.loc[(g, w), c], 2)
                             for c in vcols])

    undefined = [VOL_LABELS[g] for g in VOL_LABELS
                 if not (ok(vol.loc[g, "noise_share_post"])
                         and vol.loc[g, "noise_share_post"] >= 0)]
    log.info("Check: noise not defined after the update for %s",
             ", ".join(undefined) or "no group")
    return t.frame()


# Table E5: input by binding status (A) and coupling to the own pool (B).
def table_e5() -> pd.DataFrame:
    script = "07_volatility.py"
    het = read_table("07_heterogeneity.csv", script, index_col=0)
    hdid = read_table("07_heterogeneity_did.csv", script)
    cp = read_table("07_coupling.csv", script, index_col=0)
    for name, frame, need in [("07_heterogeneity.csv", het, list(HET_GROUPS)),
                              ("07_coupling.csv", cp,
                               list(COUPLING_GROUPS) + [COUPLING_DIFF])]:
        missing = [g for g in need if g not in frame.index]
        if missing:
            raise SystemExit(f"{name} lacks rows {missing} -- re-run {script}.")

    def rel(g: str) -> pd.Series | None:
        return pick(hdid, group=g, reference=C.VOL_CONTROL, measure="efficient")

    t = Rows(1 + len(COUPLING_GROUPS) + 1)
    h = het.loc[list(HET_GROUPS)]
    t.add(["Panel A. Efficient volatility of the input by binding status"])
    t.add(["Statistic"] + list(HET_GROUPS.values()))
    t.add(["Items (skins)"] + [f"{fmt_int(a)} ({fmt_int(b)})"
                               for a, b in zip(h["items"], h["skins"])])
    t.add(["Efficient volatility, before (%)"]
          + [fmt_dec(x, 2) for x in h["efficient_pre"]])
    t.add(["Efficient volatility, after (%)"]
          + [fmt_dec(x, 2) for x in h["efficient_post"]])
    t.add(["Change (%)"] + [fmt_signed(x, 1) for x in h["efficient_chg_pct"]])
    t.add([""] + [fmt_ci(a, b, 1, signed=True)
                  for a, b in zip(h["efficient_chg_lo"], h["efficient_chg_hi"])])
    r = {g: rel(g) for g in HET_GROUPS}
    for g in HET_GROUPS:
        if r[g] is None:
            log.warning("07_heterogeneity_did lacks %s", g)
        elif not np.isclose(r[g]["did_pct"], het.loc[g, "did_efficient"]):
            log.warning("%s: 07_heterogeneity and 07_heterogeneity_did differ "
                        "-- they come from different runs of 07", g)
    t.add(["Relative to control group (%)"]
          + [MISSING if r[g] is None else fmt_signed(r[g]["did_pct"], 1)
             for g in HET_GROUPS])
    t.add([""] + ["" if r[g] is None else
                  fmt_ci(r[g]["ci_lo"], r[g]["ci_hi"], 1, signed=True)
                  for g in HET_GROUPS])
    t.add(["p-value"] + [MISSING if r[g] is None else fmt_p_boot(r[g]["p_boot"])
                         for g in HET_GROUPS])

    c = cp.loc[list(COUPLING_GROUPS)]
    d = cp.loc[COUPLING_DIFF]
    t.add(["Panel B. Coupling to the own pool"])
    t.add(["Statistic"] + list(COUPLING_GROUPS.values()) + ["Difference"])
    t.add(["Routes (clusters)"] + [f"{fmt_int(a)} ({fmt_int(b)})"
                                   for a, b in zip(c["routes"], c["clusters"])]
          + [f"{fmt_int(d['routes'])} ({fmt_int(d['clusters'])})"])
    for w, word in (("pre", "before"), ("post", "after")):
        t.add([f"β, {word}"] + [fmt_dec(x, 2) for x in c[f"beta_{w}"]])
        t.add([""] + [fmt_ci(a, b, 2) for a, b in zip(c[f"beta_{w}_lo"],
                                                      c[f"beta_{w}_hi"])])
    t.add(["Change in β"] + [fmt_signed(x, 2) for x in c["change"]]
          + [fmt_signed(d["change"], 2)])
    t.add([""] + [fmt_ci(a, b, 2, signed=True)
                  for a, b in zip(c["change_lo"], c["change_hi"])]
          + [fmt_ci(d["change_lo"], d["change_hi"], 2, signed=True)])
    t.add(["p-value"] + [fmt_p_boot(x) for x in c["p_change"]]
          + [fmt_p_boot(d["p_change"])])
    t.add(["p-value, β after = 1"] + [fmt_p_boot(x) for x in c["p_post_eq_1"]])
    for w, word in (("pre", "before"), ("post", "after")):
        t.add([f"Naive β, {word}"] + [fmt_dec(x, 2) for x in c[f"naive_{w}"]])
    return t.frame()


# Table E6: units sold, average price and turnover, days 30 to 190 vs. before.
def table_e6() -> pd.DataFrame:
    script = "08_volume.py"
    lr = read_table("08_long_run.csv", script, index_col=0)
    missing = [g for g in VOL_LABELS if g not in lr.index]
    if missing:
        raise SystemExit(f"08_long_run.csv lacks rows {missing} -- re-run "
                         f"{script}.")
    lo, hi = C.VOLUME_WINDOWS["+30_+190"]
    measures = {"units": "Units sold", "price": "Average price",
                "eur": "Turnover in EUR"}

    t = Rows(7)
    t.add([f"Panel A. Days {lo} to {hi} against before the update (multiple)"])
    t.add(["Group", "Items"] + list(measures.values()))
    for g, label in VOL_LABELS.items():
        r = lr.loc[g]
        t.add([label, fmt_int(r["items"])]
              + [fmt_dec(r[f"{m}_multiple"], 2) for m in measures])
        t.add(["", ""] + [fmt_ci(r[f"{m}_lo"], r[f"{m}_hi"], 2)
                          for m in measures])

    t.add(["Panel B. Relative to the control group"])
    head = ["Group"]
    for m in measures.values():
        head += [m, "p-value"]
    t.add(head)
    for g, label in VOL_LABELS.items():
        if g == C.VOL_CONTROL:
            continue
        r = lr.loc[g]
        cells = [label]
        for m in measures:
            cells += [fmt_dec(r[f"{m}_vs_control"], 2),
                      fmt_p_boot(r[f"{m}_vs_control_p"])]
        t.add(cells)

    log.info("Check: price multiples -- %s", ", ".join(
        f"{VOL_LABELS[g]} {lr.loc[g, 'price_multiple']:.2f}" for g in VOL_LABELS))
    return t.frame()


def main() -> None:
    p = argparse.ArgumentParser(description="Builds the appendix tables.")
    p.add_argument("appendix", nargs="*", default=["A", "B", "C", "D", "E"],
                   help="Appendices to build, e.g. A or C D (default: all).")
    todo = {a.upper() for a in p.parse_args().appendix}
    setup_logging()
    C.OUT_TABLES.mkdir(parents=True, exist_ok=True)

    if "A" in todo:
        appendix_a()
    if "B" in todo:
        appendix_b()
    if "C" in todo:
        appendix_c()
    if "D" in todo:
        appendix_d()
    if "E" in todo:
        appendix_e()

    log.info("")
    log.info("Done.")


def appendix_a() -> None:
    t = table_a1()
    emit("TABLE A1: FLOAT RANGES OF THE KNIFE AND GLOVE FINISHES", t,
         "thesis_tableA1", header=False)
    # Table A2 (own trade-up contracts) is written by trade_up_contracts/observed.py.
    t = table_a3()
    emit("TABLE A3: WEAR LEVEL OF THE OUTPUT BY WEAR LEVEL OF THE INPUT", t,
         "thesis_tableA3", header=False)
    checks_a()


def appendix_b() -> None:
    t = table_b1()
    emit("TABLE B1: FROM THE ITEM LIST TO THE DATA WINDOW", t,
         "thesis_tableB1", header=False)
    t = table_b2()
    emit("TABLE B2: SAMPLE SELECTION", t, "thesis_tableB2", header=False)
    t = table_b3()
    emit("TABLE B3: TRADE SHARE AND SAMPLE VARIANTS", t, "thesis_tableB3",
         header=False)
    t = table_b4()
    emit("TABLE B4: LARGEST MARKET-WIDE DAILY PRICE MOVES", t,
         "thesis_tableB4", header=False)


def appendix_c() -> None:
    t = table_c1()
    emit("TABLE C1: RETURN PROPERTIES, MARKET MODEL, CORRELATION", t,
         "thesis_tableC1", header=False)
    t = table_c2()
    emit("TABLE C2: ABNORMAL RETURNS BEFORE THE ANNOUNCEMENT", t,
         "thesis_tableC2")
    t = table_c3()
    emit("TABLE C3: ALL EVENT WINDOWS AND TESTS", t, "thesis_tableC3")
    t = table_c4()
    emit("TABLE C4: RETURN MODELS", t, "thesis_tableC4", header=False)
    t = table_c5()
    emit("TABLE C5: INDICES, SAMPLES, ESTIMATION WINDOW", t,
         "thesis_tableC5", header=False)


def appendix_d() -> None:
    t = table_d1()
    emit("TABLE D1: CROSS-SECTIONAL REGRESSIONS OF THE INPUT", t,
         "thesis_tableD1", header=False)
    t = table_d2()
    emit("TABLE D2: EVENT-WINDOW CARs OF THE INPUT BY BINDING STATUS", t,
         "thesis_tableD2", header=False)
    t = table_d3()
    emit("TABLE D3: THE BOUND BY WEAR LEVEL AND OUTPUT POOL", t,
         "thesis_tableD3", header=False)
    t = table_d4()
    emit("TABLE D4: DECOMPOSITION OF THE CHANGE IN LOG R, RATCHET", t,
         "thesis_tableD4", header=False)
    t = table_d5()
    emit("TABLE D5: OVERSHOOT, VARIANTS AND OUTCOME SHAPE", t,
         "thesis_tableD5", header=False)


def appendix_e() -> None:
    t = table_e1()
    emit("TABLE E1: ABNORMAL VOLUME", t, "thesis_tableE1", header=False)
    t = table_e2()
    emit("TABLE E2: ECHO AND UNIT BALANCE", t, "thesis_tableE2",
         header=False)
    t = table_e3()
    emit("TABLE E3: ROBUSTNESS OF THE VOLATILITY CHANGE", t,
         "thesis_tableE3", header=False)
    t = table_e4()
    emit("TABLE E4: VOLATILITY OF ALL GROUPS AND SIGNATURE", t,
         "thesis_tableE4", header=False)
    t = table_e5()
    emit("TABLE E5: BINDING STATUS AND COUPLING", t, "thesis_tableE5",
         header=False)
    t = table_e6()
    emit("TABLE E6: UNITS, PRICE AND TURNOVER, DAYS 30 TO 190", t,
         "thesis_tableE6", header=False)


if __name__ == "__main__":
    main()

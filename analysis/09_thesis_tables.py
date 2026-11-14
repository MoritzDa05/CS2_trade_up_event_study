# Builds the tables of the thesis main text from the CSVs of 01-07; estimates nothing.
# Per table: thesis_tableN.csv and .tsv (for pasting into Word).
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from inference import pool_of_case  # noqa: E402

log = logging.getLogger("09_thesis_tables")

# Group names of the thesis, in table order.
THESIS_ROLES = {
    "treated_output": "Output",
    "treated_input": "Input",
    "placebo_st": "Placebo ST",
    "placebo": "Placebo Map",
    "placebo_low": "Placebo Classified",
    "index_main": "Benchmark (sticker capsules)",
    "index_alt": "Music kits and agents",
}

MINUS = "−"  # typographic minus, as in the thesis

# Table 3 windows ([-1, +30] only in Appendix C). The benchmark is left out (AR ~ 0 by
# construction, KP undefined); Table C3 shows it.
TABLE3_WINDOWS = {
    "0": "t = 0",
    "-1_+1": "[−1, +1]",
    "-1_+5": "[−1, +5]",
    "-1_+60": "[−1, +60]",
}
TABLE3_ROLES = ["treated_output", "treated_input", "placebo_st", "placebo",
                "placebo_low"]

# Section 4.4: established if t_KP and the rank test reject at 5 % with the same sign.
ESTABLISHED_LEVEL = 0.05
ESTABLISHED_MARK = "†"

# Table 4 columns; (4), (5) and the long window of 05 are in Appendix D.
TABLE4_COLUMNS = [
    ("treated_input", "(1) controls only"),
    ("treated_input", "(2) + pool value"),
    ("treated_input", "(3) + category FE"),
    ("treated_output", "(1) controls only"),
    ("treated_output", "(2) + input cost"),
    ("treated_output", "(3) + routes"),
]
# Regressors shown (name in 05 -> row label); fixed effects get a Yes/No row.
TABLE4_REGRESSORS = {
    "log_pool": "Pool value (log)",
    "log_input_cost": "Input cost (log)",
    "n_routes": "Cases containing the item",
    "log_price": "Own price (log)",
    "trade_share": "Trade share",
    "stattrak_d": "StatTrak",
    "n_covert": "Coverts in the case",
    "is_glove": "Gloves",
}
TABLE4_FE = {
    "wear_g_": "Wear fixed effects",
    "category_": "Category fixed effects",
}

# Table 5: Panel A from 06_bound_test, Panel B from 06_placebo.
TABLE5_PANEL_A = {
    "all": "All routes",
    "binding": "Binding",
    "slack": "Slack",
}
TABLE5_PANEL_B = {
    "treated, all": "Input",
    "treated, StatTrak": "Input, StatTrak",
    "placebo, StatTrak glove cases": "Placebo ST",
    "difference: treated StatTrak - placebo": "Difference",
}
TABLE5_KINK = "(a) kink at the bound"
TABLE5_VALUE = "(b) value, price, wear FE"

# Table 6: Panel A per group (07), Panel B input Coverts by binding status.
TABLE6_GROUPS = {
    "treated_output": "Output",
    "treated_input": "Input",
    "control": "Control group",
    "placebo_st": "Placebo ST",
    "index_alt": "Music kits and agents",
}
TABLE6_REFERENCES = {"control": "control group",
                     "index_alt": "music kits and agents"}
TABLE6_BINDING = {"input, binding": "Binding", "input, slack": "Slack"}
TABLE6_DIFF = "binding - slack (same pools)"  # row name in 07_coupling


def setup_logging() -> None:
    C.OUT_LOGS.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(C.OUT_LOGS / "09_thesis_tables.log", mode="w",
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


def read_table(name: str, script: str, index_col: int | None = 0) -> pd.DataFrame:
    path = C.OUT_TABLES / name
    if not path.exists():
        raise SystemExit(f"{path} not found -- run {script} first.")
    return pd.read_csv(path, index_col=index_col, encoding="utf-8")


# CSV and TSV of the same frame; TSV for Word, because the cells contain commas.
def save(df: pd.DataFrame, stem: str, header: bool = True) -> None:
    df.to_csv(C.OUT_TABLES / f"{stem}.csv", index=False, header=header,
              encoding="utf-8")
    df.to_csv(C.OUT_TABLES / f"{stem}.tsv", index=False, header=header,
              sep="\t", encoding="utf-8")
    log.info("-> output/tables/%s.csv and .tsv", stem)


# Number formats: American English and typographic minus, as in the thesis.
def fmt_int(x) -> str:
    if x is None or not np.isfinite(x):
        return ""
    return f"{int(round(x)):,}"


def fmt_dec(x, digits: int) -> str:
    if x is None or not np.isfinite(x):
        return ""
    return f"{x:,.{digits}f}".replace("-", MINUS)


def fmt_quartiles(med, q1, q3) -> str:
    return f"{med:,.2f} [{q1:,.2f}; {q3:,.2f}]"


def fmt_signed(x, digits: int) -> str:
    if x is None or not np.isfinite(x):
        return ""
    s = f"{x:+,.{digits}f}"
    return s.replace("-", MINUS)


def fmt_coef(x, digits: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return ""
    return f"{x:.{digits}f}".replace("-", MINUS)


def fmt_p(p) -> str:
    if p is None or not np.isfinite(p):
        return ""
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


# Wild bootstrap p = (hits + 1) / (reps + 1): no hit at all is shown as "≤ 0.001".
def fmt_p_wild(p) -> str:
    if p is None or not np.isfinite(p):
        return ""
    floor = 1 / (C.BOOTSTRAP_REPS + 1)
    return f"≤ {floor:.3f}" if p <= floor * (1 + 1e-9) else f"{p:.3f}"


# Percentile bootstrap p from 999 draws: below 0.01 shown as "< 0.01".
def fmt_p_boot(p) -> str:
    if p is None or not np.isfinite(p):
        return ""
    return "< 0.01" if p < 0.01 else f"{p:.3f}"


def fmt_ci(lo, hi, digits: int, signed: bool = False) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return ""
    f = (lambda x: fmt_signed(x, digits)) if signed else (
        lambda x: f"{x:.{digits}f}".replace("-", MINUS))
    return f"[{f(lo)}; {f(hi)}]"


def sig_stars(p) -> str:
    if p is None or not np.isfinite(p):
        return ""
    for level, mark in C.SIG_LEVELS:
        if p < level:
            return mark
    return ""


# Stars from p_star: CR1 from WEBB_MAX_CLUSTERS clusters on, wild bootstrap below.
def star_of(r) -> str:
    return sig_stars(r["p_star"] if "p_star" in r.index else r["p"])


def thesis_table2() -> pd.DataFrame:
    flow = read_table("01_sample_flow.csv", "01_screening.py")
    chars = read_table("02_table1_sample_characteristics.csv", "02_descriptives.py")

    # The last column of the sample flow is the main sample.
    final_step = flow.columns[-1]
    log.info("Main sample = column '%s' of 01_sample_flow", final_step)

    roles = [r for r in THESIS_ROLES if r in chars.index]
    missing = [r for r in THESIS_ROLES if r not in chars.index]
    if missing:
        log.warning("Roles missing from 02 and left out of the table: %s", missing)

    # 01 and 02 count the main sample independently; a mismatch means mixed runs.
    for r in roles:
        a, b = int(flow.loc[r, final_step]), int(chars.loc[r, "Items"])
        if a != b:
            log.warning("%s: main sample %d in 01 but %d in 02 -- re-run 01 and 02",
                        r, a, b)

    rows = []
    for r in roles:
        c = chars.loc[r]
        rows.append({
            "Group": THESIS_ROLES[r],
            "Items in data window": fmt_int(flow.loc[r, "items_in_window"]),
            "Main sample": fmt_int(flow.loc[r, final_step]),
            "Price in EUR, median [Q1; Q3]": fmt_quartiles(
                c["Price median"], c["Price Q1"], c["Price Q3"]),
            "Trade share": fmt_dec(c["Trade share"], 2),
            "Units sold per day": fmt_dec(c["Volume per day"], 1),
            "Returns in estimation window": fmt_int(c["Est. returns"]),
            "StatTrak share": fmt_dec(c["StatTrak share"], 2),
        })
    return pd.DataFrame(rows)


def is_established(row: pd.Series) -> bool:
    p_kp, p_rank = row.get("p_kp"), row.get("p_rank")
    t_kp, z_rank = row.get("t_kp"), row.get("z_rank")
    if not all(pd.notna(v) for v in (p_kp, p_rank, t_kp, z_rank)):
        return False
    return (p_kp < ESTABLISHED_LEVEL and p_rank < ESTABLISHED_LEVEL
            and np.sign(t_kp) == np.sign(z_rank))


def thesis_table3() -> pd.DataFrame:
    caar = read_table("04_caar_main.csv", "04_abnormal_returns.py", index_col=None)

    caar = caar[caar["model"] == C.MODEL_MAIN]
    if caar.empty:
        raise SystemExit(f"04_caar_main.csv has no rows for model {C.MODEL_MAIN}")

    # n (items with a CAR) must be equal in every window; it goes into the group label.
    n_by_role = {}
    for role in TABLE3_ROLES:
        ns = caar.loc[caar["role"] == role, "n"].unique()
        if len(ns) != 1:
            log.warning("%s: n differs across windows (%s), using the largest",
                        role, ns)
        n_by_role[role] = int(ns.max())

    rows = []
    for win, win_label in TABLE3_WINDOWS.items():
        first = True
        for role in TABLE3_ROLES:
            sel = caar[(caar["window"] == win) & (caar["role"] == role)]
            if sel.empty:
                log.warning("No row for window %s, role %s", win, role)
                continue
            r = sel.iloc[0]
            mark = ESTABLISHED_MARK if is_established(r) else ""
            rows.append({
                "Window": win_label if first else "",
                "Group": f"{THESIS_ROLES[role]} (n = {fmt_int(n_by_role[role])})",
                "CAAR (%)": fmt_signed(r["caar_pct"], 1) + mark,
                "Median (%)": fmt_signed(r["median_pct"], 1),
                "t_KP": fmt_signed(r["t_kp"], 2),
                "Rank z": fmt_signed(r["z_rank"], 2),
            })
            first = False
    return pd.DataFrame(rows)


def thesis_table4() -> pd.DataFrame:
    script = "05_cross_section.py"
    coefs = {
        "treated_input": read_table("05_regression_input.csv", script),
        "treated_output": read_table("05_regression_output.csv", script),
    }
    fit = read_table("05_regression_fit.csv", script, index_col=None)

    columns = []
    for group, spec in TABLE4_COLUMNS:
        tab = coefs[group][coefs[group]["spec"] == spec]
        info = fit[(fit["group"] == group) & (fit["spec"] == spec)]
        if tab.empty or info.empty:
            raise SystemExit(f"No results for {group}, {spec} -- re-run {script}.")
        header = f"{THESIS_ROLES[group]} {spec.split(' ')[0]}"
        columns.append((header, tab, info.iloc[0]))

    rows = []
    for name, label in TABLE4_REGRESSORS.items():
        if not any(name in tab.index for _, tab, _ in columns):
            continue
        coef_row, se_row = {"Variable": label}, {"Variable": ""}
        for header, tab, _ in columns:
            if name in tab.index:
                r = tab.loc[name]
                coef_row[header] = fmt_coef(r["coef"]) + star_of(r)
                se_row[header] = f"({fmt_coef(r['se'])})"
            else:
                coef_row[header] = se_row[header] = ""
        rows += [coef_row, se_row]

    for prefix, label in TABLE4_FE.items():
        row = {"Variable": label}
        for header, tab, _ in columns:
            has = any(str(i).startswith(prefix) for i in tab.index)
            row[header] = "Yes" if has else "No"
        rows.append(row)

    stat_rows = [("Items", lambda i: fmt_int(i["n"])),
                 ("Clusters (pools)", lambda i: fmt_int(i["clusters"])),
                 ("R²", lambda i: fmt_dec(i["r2"], 3)),
                 ("Wild bootstrap p", lambda i: fmt_p_wild(i["p_wild"]))]
    for label, f in stat_rows:
        row = {"Variable": label}
        for header, _, info in columns:
            row[header] = f(info)
        rows.append(row)
    return pd.DataFrame(rows)


def thesis_table5() -> pd.DataFrame:
    script = "06_arbitrage.py"
    bound = read_table("06_bound_test.csv", script)
    plac = read_table("06_placebo.csv", script, index_col=None)
    cols = ["", "c1", "c2", "c3", "c4"]
    rows: list[list[str]] = []

    def row(label: str, cells: list[str]) -> None:
        rows.append([label] + cells + [""] * (4 - len(cells)))

    # Panel A: routes relative to the bound.
    missing = [g for g in TABLE5_PANEL_A if g not in bound.index]
    if missing:
        raise SystemExit(f"06_bound_test.csv lacks rows {missing} -- re-run {script}.")
    a = bound.loc[list(TABLE5_PANEL_A)]
    row("Panel A. Routes relative to the bound", [])
    row("", list(TABLE5_PANEL_A.values()))
    row("Routes", [fmt_int(v) for v in a["n"]])
    row("Median R, before", [fmt_dec(v, 2) for v in a["median_r_pre"]])
    row("Median R, after", [fmt_dec(v, 2) for v in a["median_r_post"]])
    row(f"Below {C.BOUND_STRICT:.3f}, before (%)",
        [fmt_dec(v, 1) for v in a["below_pre_pct"]])
    row(f"Below {C.BOUND_STRICT:.3f}, after (%)",
        [fmt_dec(v, 1) for v in a["below_post_pct"]])
    row(f"Between {C.BOUND_STRICT:.3f} and {C.BOUND_HOLDER:.1f}, after (%)",
        [fmt_dec(v, 1) for v in a["band_post_pct"]])
    row(f"Above {C.BOUND_HOLDER:.1f}, after (%)",
        [fmt_dec(v, 1) for v in a["above_post_pct"]])

    # Panel B: kink and placebo tests.
    def get(group: str, spec: str, regressor: str) -> pd.Series | None:
        sel = plac[(plac["group"] == group) & (plac["spec"] == spec)
                   & (plac["regressor"].str.startswith(regressor))]
        return sel.iloc[0] if len(sel) else None

    groups = list(TABLE5_PANEL_B)
    for g in groups:
        if get(g, TABLE5_KINK, "gap_below") is None:
            raise SystemExit(f"06_placebo.csv lacks group '{g}' -- re-run {script}.")
    row("Panel B. Kink and placebo tests", [])
    row("", list(TABLE5_PANEL_B.values()))

    def coef_rows(label: str, spec: str, regressor: str) -> None:
        cells, ses = [], []
        for g in groups:
            r = get(g, spec, regressor)
            cells.append("" if r is None else fmt_coef(r["coef"]) + star_of(r))
            ses.append("" if r is None else f"({fmt_coef(r['se'])})")
        row(label, cells)
        row("", ses)

    def stat_row(label: str, spec: str, regressor: str, key: str, f) -> None:
        cells = []
        for g in groups:
            r = get(g, spec, regressor)
            cells.append("" if r is None or pd.isna(r[key]) else f(r[key]))
        row(label, cells)

    coef_rows("Slope below the bound, b⁻", TABLE5_KINK, "gap_below")
    coef_rows("Slope above the bound, b⁺", TABLE5_KINK, "gap_above")
    stat_row("p-value, b⁻ = −1", TABLE5_KINK, "gap_below", "p_vs_minus1", fmt_p)
    stat_row("Wild bootstrap p, b⁻", TABLE5_KINK, "gap_below", "p_wild", fmt_p_wild)
    stat_row("R², Equation (9)", TABLE5_KINK, "gap_below", "r2",
             lambda v: fmt_dec(v, 3))
    coef_rows("Expected output value (log)", TABLE5_VALUE, "log_ev_pre")
    stat_row("Wild bootstrap p, expected output value", TABLE5_VALUE,
             "log_ev_pre", "p_wild", fmt_p_wild)
    stat_row("Routes", TABLE5_KINK, "gap_below", "n", fmt_int)
    stat_row("Clusters", TABLE5_KINK, "gap_below", "clusters", fmt_int)

    # Cases and pools of the Placebo ST regression (routes as in 06's placebo test).
    rt = pd.read_parquet(C.TRADEUP_ROUTES)
    pl = rt[(rt["role"] == C.PLACEBO_ROLE) & rt["main"]].dropna(
        subset=["p_pre", "p_post", "r_pre"])
    log.info("Check: Placebo ST in Panel B -- %d cases in %d pools",
             pl["case"].nunique(), len({pool_of_case(c) for c in pl["case"].unique()}))
    return pd.DataFrame(rows, columns=cols)


def thesis_table6() -> pd.DataFrame:
    script = "07_volatility.py"
    vol = read_table("07_volatility.csv", script)
    did = read_table("07_did.csv", script, index_col=None)
    het = read_table("07_heterogeneity.csv", script)
    joint = read_table("07_heterogeneity_joint.csv", script,
                       index_col=None).set_index("measure")
    cp = read_table("07_coupling.csv", script)

    groups = list(TABLE6_GROUPS)
    binding = list(TABLE6_BINDING)
    for name, frame, need in [("07_volatility.csv", vol, groups),
                              ("07_heterogeneity.csv", het, binding),
                              ("07_coupling.csv", cp, binding + [TABLE6_DIFF])]:
        missing = [g for g in need if g not in frame.index]
        if missing:
            raise SystemExit(f"{name} lacks rows {missing} -- re-run {script}.")

    n_cols = len(groups)
    cols = [""] + [f"c{i}" for i in range(1, n_cols + 1)]
    rows: list[list[str]] = []

    def row(label: str, cells: list[str]) -> None:
        rows.append([label] + cells + [""] * (n_cols - len(cells)))

    def rel(group: str, ref: str) -> pd.Series | None:
        sel = did[(did["group"] == group) & (did["reference"] == ref)
                  & (did["measure"] == "efficient")]
        return sel.iloc[0] if len(sel) else None

    # Section 5.4: differs from both references at 5 % with the same sign.
    def established(group: str) -> bool:
        r = [rel(group, ref) for ref in TABLE6_REFERENCES]
        if any(x is None for x in r):
            return False
        return (all(x["p_boot"] < ESTABLISHED_LEVEL for x in r)
                and len({np.sign(x["did_pct"]) for x in r}) == 1)

    # Panel A: volatility per group.
    v = vol.loc[groups]
    row("Panel A. Weekly volatility before and after the update", [])
    row("", list(TABLE6_GROUPS.values()))
    row("Items (skins)", [f"{fmt_int(a)} ({fmt_int(b)})"
                          for a, b in zip(v["items"], v["skins"])])
    row("Units sold per week, before", [fmt_dec(x, 1) for x in v["trades_week_pre"]])
    row("Units sold per week, after", [fmt_dec(x, 1) for x in v["trades_week_post"]])
    row("Raw volatility, change (%)", [fmt_signed(x, 1) for x in v["raw_chg_pct"]])
    row("Efficient volatility, before (%)",
        [fmt_dec(x, 2) for x in v["efficient_pre"]])
    row("Efficient volatility, after (%)",
        [fmt_dec(x, 2) for x in v["efficient_post"]])
    row("Efficient volatility, change (%)",
        [fmt_signed(v.loc[g, "efficient_chg_pct"], 1)
         + (ESTABLISHED_MARK if established(g) else "") for g in groups])
    row("", [fmt_ci(v.loc[g, "efficient_chg_lo"], v.loc[g, "efficient_chg_hi"], 1,
                    signed=True) for g in groups])
    # A negative intercept leaves the noise undefined (as in Table E4).
    def share(x) -> str:
        return fmt_dec(x, 1) if pd.notna(x) and x >= 0 else "–"

    row("Noise share, before (%)", [share(x) for x in v["noise_share_pre"]])
    row("Noise share, after (%)", [share(x) for x in v["noise_share_post"]])
    for ref, ref_label in TABLE6_REFERENCES.items():
        r = {g: rel(g, ref) for g in groups}
        row(f"Efficient volatility relative to {ref_label} (%)",
            ["" if r[g] is None else fmt_signed(r[g]["did_pct"], 1) for g in groups])
        row("p-value", ["" if r[g] is None else fmt_p_boot(r[g]["p_boot"])
                        for g in groups])

    # Panel B: input Coverts by binding status.
    h = het.loc[binding]
    e = joint.loc["efficient"]
    c = cp.loc[binding]
    d = cp.loc[TABLE6_DIFF]
    row("Panel B. Input Coverts by binding status", [])
    row("", list(TABLE6_BINDING.values()) + ["Difference"])
    row("Items (skins)", [f"{fmt_int(a)} ({fmt_int(b)})"
                          for a, b in zip(h["items"], h["skins"])])
    row("Efficient volatility, before (%)", [fmt_dec(x, 2) for x in h["efficient_pre"]])
    row("Efficient volatility, after (%)", [fmt_dec(x, 2) for x in h["efficient_post"]])
    row("Efficient volatility, change (%)",
        [fmt_signed(x, 1) for x in h["efficient_chg_pct"]]
        + [fmt_signed(e["diff_pct"], 1)])
    row("", [fmt_ci(a, b, 1, signed=True)
             for a, b in zip(h["efficient_chg_lo"], h["efficient_chg_hi"])]
        + [fmt_ci(e["ci_lo"], e["ci_hi"], 1, signed=True)])
    row("p-value", ["", "", fmt_p_boot(e["p_boot"])])
    for w, label in (("pre", "before"), ("post", "after")):
        row(f"β on the own pool, {label}", [fmt_dec(x, 2) for x in c[f"beta_{w}"]])
        row("", [fmt_ci(a, b, 2) for a, b in zip(c[f"beta_{w}_lo"],
                                                  c[f"beta_{w}_hi"])])
    row("Change in β", [fmt_signed(x, 2) for x in c["change"]]
        + [fmt_signed(d["change"], 2)])
    row("", [fmt_ci(a, b, 2, signed=True) for a, b in zip(c["change_lo"],
                                                           c["change_hi"])]
        + [fmt_ci(d["change_lo"], d["change_hi"], 2, signed=True)])
    row("p-value", [fmt_p_boot(x) for x in c["p_change"]]
        + [fmt_p_boot(d["p_change"])])
    row("p-value, β after = 1", [fmt_p_boot(x) for x in c["p_post_eq_1"]])
    row("Routes (pools)", [f"{fmt_int(a)} ({fmt_int(b)})"
                           for a, b in zip(c["routes"], c["clusters"])]
        + [f"{fmt_int(d['routes'])} ({fmt_int(d['clusters'])})"])

    return pd.DataFrame(rows, columns=cols)


def main() -> None:
    setup_logging()
    C.OUT_TABLES.mkdir(parents=True, exist_ok=True)

    section("THESIS TABLE 2: SAMPLE AND GROUP CHARACTERISTICS (SECTION 4.2)")
    t2 = thesis_table2()
    show(t2)
    save(t2, "thesis_table2")

    section("THESIS TABLE 3: CUMULATIVE ABNORMAL RETURNS (SECTION 5.1)")
    t3 = thesis_table3()
    show(t3)
    save(t3, "thesis_table3")

    section("THESIS TABLE 4: CROSS-SECTION OF THE EVENT REACTION (SECTION 5.2)")
    t4 = thesis_table4()
    show(t4)
    save(t4, "thesis_table4")

    section("THESIS TABLE 5: THE NO-ARBITRAGE BOUND AND THE KINK (SECTION 5.3)")
    t5 = thesis_table5()
    show(t5, header=False)
    save(t5, "thesis_table5", header=False)

    section("THESIS TABLE 6: VOLATILITY, NOISE AND COUPLING (SECTION 5.4)")
    t6 = thesis_table6()
    show(t6, header=False)
    save(t6, "thesis_table6", header=False)

    log.info("")
    log.info("Done.")


if __name__ == "__main__":
    main()

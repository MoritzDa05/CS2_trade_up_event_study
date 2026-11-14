# Central configuration: every script reads its parameters from here.
# A methodological change is made only here, then the pipeline is re-run.
from pathlib import Path
import math

import pandas as pd

# Paths
ROOT = Path(__file__).resolve().parent

RAW_PANEL = ROOT / "data" / "full" / "panel_long.parquet"  # raw data, never overwritten
ITEMS_UNIVERSE = ROOT / "items" / "items_full.csv"  # what the collector was asked for
MANIFEST = ROOT / "data" / "full" / "manifest.jsonl"  # one line per collector request
CHECK_PANEL = ROOT / "data" / "check" / "panel_long.parquet"  # currency check, second retrieval
CHECK_MANIFEST = ROOT / "data" / "check" / "manifest.jsonl"
CLEAN_DIR = ROOT / "data" / "clean"
OUT_TABLES = ROOT / "output" / "tables"
OUT_FIGURES = ROOT / "output" / "figures"
OUT_LOGS = ROOT / "output" / "logs"

PANEL_CLEAN = CLEAN_DIR / "panel_clean.parquet"
ITEMS_CLEAN = CLEAN_DIR / "items.parquet"
SAMPLES = CLEAN_DIR / "samples.parquet"  # written by 01, read by 02 ff.

# Event and windows (calendar days relative to T0)
T0 = pd.Timestamp("2025-10-23")  # t = 0 (UTC), confirmed by the volume spike
T0_TEXT = f"{T0:%B} {T0.day}, {T0.year}"  # for axis labels and notes

# Other events of Section 2.3 (first UTC day after the announcement)
EVENT_DAYS = {
    "Austin Major": -153,
    "Trade Protection": -99,
    "Genesis": -36,
    "Armory": -21,
    "Budapest Major": 21,  # capsule release; the tournament ran from t = +32 to +52 (Nov 24 to Dec 14)
}
EARLIER_UPDATES = ["Genesis", "Armory"]  # dotted in the event-time figure of 07

WINDOW = (-200, 200)  # kept in panel_clean
EST_WINDOW = (-190, -15)  # market model, liquidity, screening
EVENT_WINDOWS = {
    "0": (0, 0),
    "-1_+1": (-1, 1),
    "-1_+5": (-1, 5),
    "-1_+30": (-1, 30),
    "-1_+60": (-1, 60),
}

# Pre-event windows for the anticipation test in 04
PRE_WINDOWS = {
    "-14_-2": (-14, -2),
    "-5_-2": (-5, -2),
}

EST_WINDOW_ROBUST = (-85, -15)  # robustness in 04: starts after Trade Protection

LOAD_BUFFER_DAYS = 60  # loaded before the window, cut off afterwards

# Returns
MAX_GAP_RET = 5  # longest gap (days) for a usable return

# Spikes (single outlier days)
SPIKE_JUMP = math.log(1.5)  # |return| > +50 % or < -33 %
SPIKE_REVERT = math.log(1.1)  # next price within +-10 % of the previous one
SPIKE_MAX_NEIGHBOR_GAP = 7  # days to the previous and next trade
SPIKE_MAX_VOLUME = 1  # only 1-trade days -> NaN (robustness: 2)

SPIKE_PROTECT_WINDOW = (-1, 10)  # around t = 0: only extreme spikes are removed
SPIKE_JUMP_PROTECTED = math.log(3.0)  # factor 3 (+200 % / -67 %)

# Liquidity, measured in the estimation window only
LIQ_BINS = [0.0, 0.10, 0.30, 0.50, 1.0 + 1e-9]  # trade share = trading days / days listed
LIQ_LABELS = ["lt10", "10_30", "30_50", "ge50"]
LIQ_MAIN_MIN = 0.30
LIQ_ROBUST_MIN = 0.50

MIN_EST_RETURNS = 30  # usable returns in the estimation window

# Burn-in: tradable before the estimation window, otherwise variant "launch"
REQUIRE_FULL_EST_WINDOW = True

# High prices and the listing cap
HIGH_PRICE_EUR = 1400.0  # Steam prices distorted above this
HIGH_PRICE_SHARE = 0.25  # flag if more trading days lie above it
MASS_POINT_EUR = 1544.73  # USD 1,800 listing cap on the main retrieval day
MASS_POINT_TOL = 0.015
LISTING_CAP_USD = 1800.0

MIN_PRICE_EUR = 0.50  # screening floor: median price in the estimation window

# Groups -> roles
ROLE_MAP = {
    "output_knife": "treated_output",
    "output_gloves": "treated_output",
    "input_covert": "treated_input",
    "placebo_covert": "placebo",
    "placebo_covert_st": "placebo_st",
    "placebo_classified": "placebo_low",
    "placebo_souvenir": "placebo_souvenir",
    "capsule_sticker": "index_main",
    "control_musickit": "index_alt",
    "control_agent": "index_alt",
    "case": "case",
    "control_graffiti": "excluded",
}

TREATED_ROLES = ["treated_output", "treated_input"]
PLACEBO_ROLES = ["placebo", "placebo_low", "placebo_st"]

# Knife and glove finishes with noisy prices
SUBCLASS_MAP = {
    "Doppler": "phase",
    "Gamma Doppler": "phase",
    "Case Hardened": "pattern_strong",
}
SUBCLASS_DEFAULT = "standard"
SUBCLASS_MAIN = "standard"

# Benchmark indices (by group, so that music kits and agents stay separate)
INDEX_DEFS = {
    "capsules": ["capsule_sticker"],
    "musickits": ["control_musickit"],
    "agents": ["control_agent"],
    "combined": ["capsule_sticker", "control_musickit", "control_agent"],
}
INDEX_MAIN = "capsules"

# Capsules still sold in the in-game store: excluded from every index.
STORE_CAPSULES = [
    "10 Year Birthday Sticker Capsule",
    "2021 Community Sticker Capsule",
    "Ambush Sticker Capsule",
    "Battlefield 2042 Sticker Capsule",
    "Bestiary Capsule",
    "CS20 Sticker Capsule",
    "Chicken Capsule",
    "Community Capsule 2018",
    "Enfu Sticker Capsule",
    "Espionage Sticker Capsule",
    "Feral Predators Capsule",
    "Half-Life: Alyx Sticker Capsule",
    "Halo Capsule",
    "Perfect World Sticker Capsule 1",
    "Perfect World Sticker Capsule 2",
    "Pinups Capsule",
    "Poorly Drawn Capsule",
    "Skill Groups Capsule",
    "Slid3 Capsule",
    "Sugarface Capsule",
    "Team Roles Capsule",
    "The Boardroom Sticker Capsule",
    "Warhammer 40,000 Sticker Capsule",
]

# Sample variants (01): each changes one decision of "main".
SAMPLE_VARIANTS = {
    "main": "liquidity >= 30 %, full estimation window, standard subclass, "
            "no high-price flag",
    "liq50": "like main, but liquidity >= 50 %",
    "subclass_all": "like main, but phase and pattern_strong included",
    "illiquid": "like main, but liquidity tier 10-30 % instead of >= 30 % "
                "(comparison group, not a robustness check)",
    "launch": "items that fail only the burn-in rule (listed during the "
              "estimation window); model-free and alpha = 0 only",
    "burn_in30": "like main, but the first sale at least BURN_IN_ROBUST_DAYS days "
                 "before the estimation window",
}
BURN_IN_ROBUST_DAYS = 30  # robustness: run-in time before the estimation window

# Descriptives (02)
ROLE_ORDER = ["treated_output", "treated_input", "placebo", "placebo_low",
              "placebo_st", "placebo_souvenir", "index_main", "index_alt",
              "case", "excluded"]

# Empty: 02 picks the item closest to the median price of each role.
EXAMPLE_ITEMS: list[str] = []
EXAMPLE_ROLES = ["treated_output", "treated_input", "placebo_low", "index_main"]

MIN_CORR_OVERLAP = 40  # common trading days per pair

# Event plots (03)
NORM_DAY = -2  # last day certainly free of the event

# Roles in thesis Figure 2; the others go into the appendix figure.
PLOT_ROLES_MAIN = ["treated_output", "treated_input", "placebo_st",
                   "placebo", "placebo_low", "index_main"]
PLOT_ROLES_APPENDIX = ["index_alt", "case", "excluded"]

PLOT_BAND_ROLES = ["treated_output"]  # interquartile band

PLOT_TRADED_ONLY_ROLES = ["treated_output", "treated_input"]  # dotted "traded only" line

PLOT_WINDOW_SHORT = (-30, 60)
PLOT_WINDOW_LONG = (-200, 200)

# Abnormal returns (04); mm_alpha0 is the main model, see Section 4.3.
RETURN_MODELS = ["mm_alpha0", "mm", "market_adj", "placebo_st_adj"]
MODEL_MAIN = "mm_alpha0"
PLACEBO_BENCHMARK_ROLE = "placebo_st"  # subtracted by placebo_st_adj (not VOL_CONTROL)

MIN_REG_OBS = 30  # returns needed for the regression

KP_MIN_OVERLAP = 40  # common days per pair for Kolari-Pynnoenen

SIG_LEVELS = [(0.01, "***"), (0.05, "**"), (0.10, "*")]

RANK_MIN_EST_DAYS = 10  # rank test needs more estimation-window days than this
PORTFOLIO_ACOV_LAGS = 10  # autocovariance lags for the variance of a portfolio window sum

# Cross-section (05)
CASE_POOLS = ROOT / "items" / "case_pools.csv"  # written by build_item_universe.py

CAR_WINDOW_MAIN = "-1_+1"
CAR_WINDOW_LONG = "-1_+60"

WINSOR = 0.01  # 1st and 99th percentile

# Wear tiers a trade-up output realistically lands in.
POOL_WEARS_REALISTIC = ["Field-Tested", "Well-Worn", "Battle-Scarred"]

SELLER_NET = 1 / 1.15  # Steam fee: the seller receives 1/1.15
CRAFT_INPUTS = 5
ARBITRAGE_SLOPE = SELLER_NET / CRAFT_INPUTS  # 0.870 / 5 = 0.174

BOOTSTRAP_REPS = 999  # wild cluster bootstrap
BOOTSTRAP_SEED = 20251023
WEBB_MAX_CLUSTERS = 12  # below: Webb (2023) six-point weights and stars from the wild bootstrap
MIN_POOL_CLUSTERS = 4  # fewer pool clusters: keep the case clusters of that regression

# Arbitrage (06): files written by 06, read by 07
TRADEUP_EV = CLEAN_DIR / "tradeup_ev.parquet"
TRADEUP_ROUTES = CLEAN_DIR / "tradeup_routes.parquet"
TRADEUP_WEIGHTS = CLEAN_DIR / "tradeup_weights.parquet"

# Wear tiers as float intervals (fixed by the game)
WEAR_BOUNDS = {
    "Factory New": (0.00, 0.07),
    "Minimal Wear": (0.07, 0.15),
    "Field-Tested": (0.15, 0.38),
    "Well-Worn": (0.38, 0.45),
    "Battle-Scarred": (0.45, 1.00),
}

FLOAT_RULE = "normalized"  # "normalized" (main) or "raw"

INPUT_FLOAT = "uniform"  # "uniform" (main) or "best"

PRICE_SMOOTH_DAYS = 7  # trailing median over traded days (figures, bound shares, decomposition)
HALF_LIFE_SMOOTH_DAYS = 1  # half-life fit and its interval: unsmoothed, last traded price
PRICE_MAX_STALE = 30  # older last trade -> no price

MIN_PRICED_MASS = 0.90  # priced share of the outcome probability

# No-arbitrage bound on R = 5 * P_covert / E[output value]
BOUND_STRICT = SELLER_NET  # buy, craft, sell
BOUND_HOLDER = 1.0  # owner of the Coverts

TRADE_HOLD_DAYS = 7  # crafted outputs inherit the remaining trade hold of their inputs (ex post variant)

# Level windows, as in 05
LEVEL_PRE = (-60, -2)
LEVEL_POST = (30, 60)
MIN_WINDOW_TRADES = 1

CONVERGENCE_WINDOW = (0, 60)  # half-life fitted on this window
PATH_WINDOW = (-60, 200)
PATH_FIG_WINDOW = (-30, 90)  # thesis Figure 4
BOUND_CLEARED_PCT = 5.0  # % of binding routes still below the bound

# Windows of the decomposition table
DECOMP_WINDOWS = {
    "t = 0": (0, 0),
    "week 1": (1, 7),
    "[30, 60]": (30, 60),
    "[150, 200]": (150, 200),
}

CAP_SHARE_MAX = 0.20  # robustness: max. value share in the distorted price zone

RATCHET_DAYS = [7, 10, 16, 28]  # days of the ratchet paragraph (Table D4, Panel B)

PLACEBO_ROLE = "placebo_st"  # pseudo route into the glove pool

# Worked examples in the 06 log
TRADEUP_EXAMPLES = [
    "AK-47 | Asiimov (Battle-Scarred)",
    "AK-47 | Head Shot (Battle-Scarred)",
    "Galil AR | Chatterbox (Field-Tested)",
    "Galil AR | Chatterbox (Battle-Scarred)",
]

# Volatility (07): signature V(k) = k * sigma^2 + 2 * eta^2 on weekly medians
VOL_PRE = EST_WINDOW
VOL_POST = (30, 190)  # after the adjustment phase
VOL_POST_SHORT = (30, 104)  # robustness, fewer confounders: exactly ten full weeks (days 35 to 104)
VOL_PRE_CALM = (-190, -40)  # robustness: ends before Genesis and Armory

VOL_SIGNATURE_K = 4  # horizons 1..K weeks
VOL_SIGNATURE_K_ALT = 6  # robustness
MIN_WEEKLY_RETURNS = 10  # per item and window
MIN_WEEKLY_RETURNS_SHORT = 5  # the short post window has at most 9

# Reference of the difference-in-differences: "control"
VOL_GROUPS = {
    "treated_output": ["treated_output"],
    "treated_input": ["treated_input"],
    "control": ["placebo_low", "placebo"],
    "placebo_st": ["placebo_st"],
    "index_main": ["index_main"],
    "index_alt": ["index_alt"],
    "case": ["case"],
}
VOL_CONTROL = "control"
VOL_TREATED = ["treated_output", "treated_input"]

# Rolling signature for the event-time figure
VOL_ROLLING_WEEKS = 8
VOL_ROLLING_MIN_RETURNS = 5
VOL_ROLLING_RANGE = (-28, 27)  # first and last week
VOL_ROLLING_MIN_ITEMS = 10  # items a group needs in a rolling window

# Coupling of Coverts to their own pool (Dimson leads and lags)
COUPLING_LAGS = 1

# Volume (08)
VOLUME_LOG_CONST = 1.0  # abnormal volume: log(units + 1)

VOLUME_WINDOWS = {
    **EVENT_WINDOWS,
    "+7_+9": (7, 9),  # echo of the seven-day hold
    "+30_+190": (30, 190),  # new regime, as in 07
}

VOLUME_ACOV_LAGS = 20  # autocovariance lags for the variance of window means
VOLUME_KP_MIN_BLOCKS = 6  # KP r-bar from block means: at least this many blocks (<= 29 days)

VOLUME_REFERENCES = ["control", "index_alt"]

# Seven-day hold: purchase wave, echo, level just before the echo
ECHO_WAVE = (0, 2)
ECHO_WINDOW = (7, 9)
ECHO_PRE = (4, 6)

# Unit balance: Coverts bought vs. knives sold seven days later
BALANCE_BUY = (0, 60)
BALANCE_SELL = (7, 67)

VOLUME_PATH = (-10, 60)  # figure window

# Item list check (check_item_list.py)
ITEM_CHECK_UNTIL = "2026-09-19T16:47:00Z"  # ByMykel download, Sep 19, 2026, 18:47 CEST
ITEM_CHECK_COMMIT = "39df8b2a5bdd43f358c0a63a92adee2e220db505"  # game files in meta_cache/valve
ITEM_CHECK_CACHE = ROOT / "meta_cache" / "valve"

# Diagnostics
JUMP_DAYS_TOP_N = 15
JUMP_DAYS_MIN_ITEMS = 100  # items that must trade for a jump day

# Event-window coverage: diagnostic only, never a filter
COVERAGE_ANCHOR_DAYS = 5
COVERAGE_END_DAYS = 5

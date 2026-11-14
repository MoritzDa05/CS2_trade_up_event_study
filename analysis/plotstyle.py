# Shared figure style for all scripts: colours, group labels, rcParams and export.
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no interactive backend: scripts run headless
import matplotlib.pyplot as plt

# Okabe-Ito palette (colour-blind safe): treated groups warm, controls cool.
COLORS = {
    "treated_output": "#D55E00",
    "treated_input": "#E69F00",
    "placebo": "#0072B2",
    "placebo_low": "#56B4E9",
    "placebo_st": "#009E73",
    "placebo_souvenir": "#7F7F7F",
    "index_main": "#000000",
    "index_alt": "#555555",
    "case": "#CC79A7",
    "excluded": "#BBBBBB",
}
NEUTRAL = "#333333"
GRID = "#DDDDDD"

# Group names as in the thesis text and tables.
ROLE_LABELS = {
    "treated_output": "Output (knives, gloves)",
    "treated_input": "Input (case Coverts)",
    "placebo": "Placebo Map",
    "placebo_low": "Placebo Classified",
    "placebo_st": "Placebo ST",
    "placebo_souvenir": "Souvenir Coverts",
    "index_main": "Benchmark (sticker capsules)",
    "index_alt": "Music kits and agents",
    "case": "Cases",
    "excluded": "Graffiti (excluded)",
}

# Two-line versions for axis ticks.
SHORT_LABELS = {
    "treated_output": "Output\n(knives, gloves)",
    "treated_input": "Input\n(case Coverts)",
    "placebo": "Placebo\nMap",
    "placebo_low": "Placebo\nClassified",
    "placebo_st": "Placebo\nST",
    "placebo_souvenir": "Souvenir\nCoverts",
    "index_main": "Benchmark\n(sticker capsules)",
    "index_alt": "Music kits\nand agents",
    "case": "Cases",
    "excluded": "Graffiti",
}

# Groups that are shown but not analysed get a dashed line.
DASHED = {"placebo_souvenir", "excluded"}


# rcParams for all figures; sizes chosen for print at about 15 cm width.
def apply_style() -> None:
    plt.rcParams.update({
        "figure.figsize": (7.5, 4.5),
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "svg.hashsalt": "cs2-event-study",  # fixed SVG ids, so that reruns give identical files
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.edgecolor": NEUTRAL,
        "axes.labelcolor": NEUTRAL,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": NEUTRAL,
        "ytick.color": NEUTRAL,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
    })


def color(role: str) -> str:
    return COLORS.get(role, NEUTRAL)


def label(role: str) -> str:
    return ROLE_LABELS.get(role, role)


def short(role: str) -> str:
    return SHORT_LABELS.get(role, ROLE_LABELS.get(role, role))


def linestyle(role: str) -> str:
    return "--" if role in DASHED else "-"


# Writes <name>.png (300 dpi, for Word) and <name>.svg (vector), closes the figure.
# The SVG gets no creation date, so that reruns give identical files.
def save(fig, name: str, outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "svg"):
        path = outdir / f"{name}.{ext}"
        fig.savefig(path, format=ext, metadata={"Date": None} if ext == "svg" else None)
        paths.append(path)
    plt.close(fig)
    return paths

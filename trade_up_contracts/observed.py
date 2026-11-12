"""Compares the outcome of the own trade-up contracts (observed.csv) with the predictions
made before the contracts (predictions.csv) and writes Table A2."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs.csv"
PREDICTIONS = HERE / "predictions.csv"
OBSERVED = HERE / "observed.csv"
TABLE = "thesis_tableA2"

HIT = 1e-6
MINUS = "−"


def signed(x: float) -> str:
    return f"{x:+.9f}".replace("-", MINUS)


def when(ts, time: bool = True) -> str:
    d = pd.Timestamp(ts)
    return f"{d:%b} {d.day}" + (f", {d:%H:%M}" if time else "")


def main() -> None:
    inputs = pd.read_csv(INPUTS, encoding="utf-8", dtype={"contract": str})
    pred = pd.read_csv(PREDICTIONS, encoding="utf-8", dtype={"contract": str})
    obs = pd.read_csv(OBSERVED, encoding="utf-8", dtype={"contract": str}).sort_values("contract")

    cols = {}
    for _, o in obs.iterrows():
        c = o["contract"]
        p = pred[(pred["contract"] == c) & (pred["output"] == o["output"])]
        if p.empty:
            sys.exit(f"Contract {c}: {o['output']} is not a possible output in {PREDICTIONS.name}")
        p = p.iloc[0]
        d_norm, d_raw = o["float"] - p["float_normalized"], o["float"] - p["float_raw"]
        holds = sorted(inputs.loc[inputs["contract"] == c, "tradable_after"].unique())
        print(f"Contract {c}: {o['output']}, observed {o['float']:.9f}, Equation (1) "
              f"{p['float_normalized']:.9f} ({'hit' if abs(d_norm) < HIT else 'miss'}), raw average "
              f"{p['float_raw']:.9f} ({'hit' if abs(d_raw) < HIT else 'miss'}); tradable after: "
              f"output {o['tradable_after']}, inputs {', '.join(holds)}")

        days = " and ".join(sorted({when(h, False) for h in holds}))
        times = " and ".join(f"{pd.Timestamp(h):%H:%M}" for h in holds)
        cols[f"Contract {c}"] = {
            "Output": o["output"],
            "Range of the output": f"{p['float_min']:.2f} to {p['float_max']:.2f}",
            "Observed float": f"{o['float']:.9f}",
            "Equation (1)": f"{p['float_normalized']:.9f}",
            "Difference, Equation (1)": signed(d_norm),
            "Raw average": f"{p['float_raw']:.9f}",
            "Difference, raw average": signed(d_raw),
            "Contract executed": when(o["executed"]),
            "Inputs tradable after": f"{days}, {times}",
            "Output tradable after": when(o["tradable_after"]),
            "Contract + 7 days": when(pd.Timestamp(o["executed"]) + pd.Timedelta(days=7), False),
        }

    panels = {
        "Panel A. Output float": ["Output", "Range of the output", "Observed float", "Equation (1)",
                                  "Difference, Equation (1)", "Raw average", "Difference, raw average"],
        "Panel B. Trade hold": ["Contract executed", "Inputs tradable after",
                                "Output tradable after", "Contract + 7 days"],
    }
    names = list(cols)
    rows = []
    for title, labels in panels.items():
        rows.append([title] + [""] * len(names))
        rows.append(["Statistic"] + names)
        for label in labels:
            shown = "Difference" if label.startswith("Difference") else label
            rows.append([shown] + [cols[n][label] for n in names])

    table = pd.DataFrame(rows)
    C.OUT_TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(C.OUT_TABLES / f"{TABLE}.csv", index=False, header=False, encoding="utf-8")
    table.to_csv(C.OUT_TABLES / f"{TABLE}.tsv", index=False, header=False, sep="\t", encoding="utf-8")
    print(f"Written: output/tables/{TABLE}.csv and .tsv")


if __name__ == "__main__":
    main()

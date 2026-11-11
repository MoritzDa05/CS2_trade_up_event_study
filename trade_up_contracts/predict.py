"""Predicts the output float of the own trade-up contracts (Appendix A) from inputs.csv."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs.csv"
OUTPUT = HERE / "predictions.csv"
SKINS = C.ROOT / "meta_cache" / "skins.json"

TIERS = ["Consumer Grade", "Industrial Grade", "Mil-Spec Grade", "Restricted", "Classified", "Covert"]
N_INPUTS = 10


def wear(f: float) -> str:
    for name, (lo, hi) in C.WEAR_BOUNDS.items():
        if lo <= f < hi:
            return name
    return "Battle-Scarred"


def short(collection: str) -> str:
    return collection.removeprefix("The ").removesuffix(" Collection")


def load_skins() -> pd.DataFrame:
    skins = json.loads(SKINS.read_text(encoding="utf-8"))
    rows = [{"item": s["name"], "rarity": s["rarity"]["name"],
             "collection": s["collections"][0]["name"],
             "float_min": s["min_float"], "float_max": s["max_float"]}
            for s in skins if s.get("collections") and s.get("min_float") is not None]
    return pd.DataFrame(rows).drop_duplicates("item")


def check_inputs(contract: str, inp: pd.DataFrame) -> None:
    def stop(msg: str) -> None:
        sys.exit(f"Contract {contract}: {msg}")

    if len(inp) != N_INPUTS:
        stop(f"{len(inp)} inputs instead of {N_INPUTS}")
    if inp["float_min"].isna().any():
        stop(f"not in {SKINS.name}: {list(inp.loc[inp['float_min'].isna(), 'item'])}")
    if inp["stattrak"].nunique() > 1:
        stop("StatTrak and regular items mixed")
    if inp["rarity"].nunique() > 1:
        stop(f"different rarities: {sorted(inp['rarity'].unique())}")
    outside = inp[(inp["float"] < inp["float_min"]) | (inp["float"] > inp["float_max"])]
    if len(outside):
        stop(f"float outside the range of the skin: {list(outside['pattern'])}")
    wrong = inp[inp["float"].map(wear) != inp["wear"]]
    if len(wrong):
        stop(f"float does not match the wear level: {list(wrong['pattern'])}")


def predict(contract: str, inp: pd.DataFrame, skins: pd.DataFrame) -> pd.DataFrame:
    check_inputs(contract, inp)
    out_tier = TIERS[TIERS.index(inp["rarity"].iloc[0]) + 1]
    out = skins[(skins["rarity"] == out_tier) & skins["collection"].isin(inp["collection"])].copy()
    no_output = set(inp["collection"]) - set(out["collection"])
    if no_output:
        sys.exit(f"Contract {contract}: no {out_tier} skin in {sorted(no_output)}")

    # Equation (1): each input float is normalized to the range of its own skin
    position = (inp["float"] - inp["float_min"]) / (inp["float_max"] - inp["float_min"])
    m, a = position.mean(), inp["float"].mean()
    span = out["float_max"] - out["float_min"]
    out["float_normalized"] = out["float_min"] + m * span
    out["float_raw"] = out["float_min"] + a * span
    n_in = inp["collection"].value_counts()
    n_out = out["collection"].value_counts()
    out["probability"] = out["collection"].map(n_in) / N_INPUTS / out["collection"].map(n_out)

    gap = (out["float_normalized"] - out["float_raw"]).abs().min()
    print(f"Contract {contract}: m = {m:.9f}, raw average = {a:.9f}, "
          f"{len(out)} possible outputs, rules differ by at least {gap:.4f}")

    out.insert(0, "contract", contract)
    out["collection"] = out["collection"].map(short)
    out["wear_normalized"] = out["float_normalized"].map(wear)
    out["wear_raw"] = out["float_raw"].map(wear)
    for col in ("float_normalized", "float_raw"):
        out[col] = out[col].map("{:.9f}".format)
    out["probability"] = out["probability"].map("{:.4f}".format)
    out = out.rename(columns={"item": "output"}).sort_values(["collection", "output"])
    return out[["contract", "output", "collection", "float_min", "float_max", "probability",
                "float_normalized", "wear_normalized", "float_raw", "wear_raw"]]


def main() -> None:
    skins = load_skins()
    inputs = pd.read_csv(INPUTS, encoding="utf-8", dtype={"contract": str})
    inputs = inputs[inputs["contract"].notna()].merge(skins, on="item", how="left")
    tables = [predict(c, inp, skins) for c, inp in inputs.groupby("contract", sort=True)]
    pd.concat(tables).to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f"Written: {OUTPUT}")


if __name__ == "__main__":
    main()

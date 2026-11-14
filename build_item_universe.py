# Builds the item list of the event study from the ByMykel CSGO-API metadata
# and assigns every market item to an analysis group.
from __future__ import annotations

import argparse
import json
import logging
import sys
import unicodedata
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

log = logging.getLogger("universe")

API_BASE = "https://raw.githubusercontent.com/ByMykel/CSGO-API/main/public/api/en"
API_FALLBACK = "https://bymykel.github.io/CSGO-API/api/en"

FILES = ["skins", "crates", "collections", "agents", "music_kits", "graffiti"]

# Weapon categories only: knives carry the Covert rarity id as well.
WEAPON_CATEGORIES = {"Rifles", "Pistols", "SMGs", "Heavy"}

RARITY_COVERT = "rarity_ancient_weapon"
RARITY_CLASSIFIED = "rarity_legendary_weapon"

# Weapon rarities from Consumer Grade (0) to Contraband (6).
RARITY_ORDER = {
    "rarity_common_weapon": 0,
    "rarity_uncommon_weapon": 1,
    "rarity_rare_weapon": 2,
    "rarity_mythical_weapon": 3,
    "rarity_legendary_weapon": 4,
    "rarity_ancient_weapon": 5,
    "rarity_contraband": 6,
}

PLACEBO_LOW_RELEASED_BEFORE = "2025-04-16"  # t = -190, start of the estimation window

# Tiers below Covert: (label, trade-up steps to a knife).
LADDER = {
    "rarity_legendary_weapon": ("classified", 2),
    "rarity_mythical_weapon":  ("restricted", 3),
    "rarity_rare_weapon":      ("milspec",    4),
}

# Written as escapes so that editors cannot swap them for look-alikes.
STAR = "\u2605"
TM = "\u2122"

# Rough knife price label (column price_tier); not used anywhere in the analysis.
CHEAP_KNIVES = {
    "Gut Knife", "Falchion Knife", "Navaja Knife", "Shadow Daggers",
    "Bowie Knife", "Huntsman Knife", "Nomad Knife", "Survival Knife",
    "Paracord Knife", "Classic Knife", "Ursus Knife", "Stiletto Knife",
}
EXPENSIVE_KNIVES = {
    "Karambit", "Butterfly Knife", "M9 Bayonet", "Talon Knife",
    "Skeleton Knife", "Kukri Knife",
}


def load_metadata(cache_dir: Path, refresh: bool = False) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    data = {}

    for name in FILES:
        target = cache_dir / f"{name}.json"

        if target.exists() and not refresh:
            log.info("Cache: %s", target.name)
        else:
            ok = False
            for base in (API_BASE, API_FALLBACK):
                url = f"{base}/{name}.json"
                try:
                    r = requests.get(url, timeout=60)
                    r.raise_for_status()
                    target.write_bytes(r.content)
                    log.info("Downloaded: %-14s %7.1f KB", name, len(r.content) / 1024)
                    ok = True
                    break
                except requests.RequestException as e:
                    log.warning("Failed (%s): %s", base.split("//")[1][:20], e)
            if not ok:
                raise SystemExit(
                    f"Could not load '{name}.json'. Check the internet connection "
                    f"or place the file manually at {target}."
                )

        with target.open(encoding="utf-8") as fh:
            data[name] = json.load(fh)

    return data


def add_stattrak(name: str) -> str:
    # For star items, StatTrak goes after the star.
    prefix = f"{STAR} "
    if name.startswith(prefix):
        return f"{prefix}StatTrak{TM} {name[len(prefix):]}"
    return f"StatTrak{TM} {name}"


def expand_variants(skin: dict, souvenir_ok: bool = False) -> Iterator[dict]:
    # One market item per wear tier x StatTrak x Souvenir.
    base = skin["name"]
    category = skin["category"]["name"]
    wears = [w["name"] for w in (skin.get("wears") or [])]
    has_st = bool(skin.get("stattrak")) and category != "Gloves"
    # skin["souvenir"] is unreliable, so the caller says whether a Souvenir exists.
    has_souv = bool(souvenir_ok)

    # Vanilla knives have no wear suffix.
    if category == "Knives" and not wears:
        yield {"market_hash_name": base, "wear": None,
               "stattrak": False, "souvenir": False, "is_vanilla": True}
        if has_st:
            yield {"market_hash_name": add_stattrak(base), "wear": None,
                   "stattrak": True, "souvenir": False, "is_vanilla": True}
        return

    if not wears:
        log.debug("No wears and not a vanilla knife, skipped: %s", base)
        return

    for wear in wears:
        yield {"market_hash_name": f"{base} ({wear})", "wear": wear,
               "stattrak": False, "souvenir": False, "is_vanilla": False}
        if has_st:
            yield {"market_hash_name": f"{add_stattrak(base)} ({wear})", "wear": wear,
                   "stattrak": True, "souvenir": False, "is_vanilla": False}
        if has_souv:
            yield {"market_hash_name": f"Souvenir {base} ({wear})", "wear": wear,
                   "stattrak": False, "souvenir": True, "is_vanilla": False}


# Maps every case to its collection and its star pool (knives, gloves or none).
def build_case_index(meta: dict) -> dict:
    skins = meta["skins"]
    by_id = {s["id"]: s for s in skins}
    cases = [c for c in meta["crates"] if c.get("type") == "Case"]

    # Only items in souvenir packages have a Souvenir version.
    souvenir_ids: set[str] = set()
    for crate in meta["crates"]:
        if crate.get("type") not in ("Souvenir", "Souvenir Highlight"):
            continue
        for entry in (crate.get("contains") or []) + (crate.get("contains_rare") or []):
            souvenir_ids.add(entry["id"])

    case_of_skin: dict[str, str] = {}
    rare_type: dict[str, Optional[str]] = {}
    collection_of: dict[str, Optional[str]] = {}
    rare_skin_ids: set[str] = set()
    case_stats: dict[str, dict] = {}

    for case in cases:
        cname = case["name"]

        cats = set()
        for entry in case.get("contains_rare") or []:
            skin = by_id.get(entry["id"])
            if skin:
                cats.add(skin["category"]["name"])
                if skin["category"]["name"] in ("Knives", "Gloves"):
                    rare_skin_ids.add(entry["id"])
        if "Knives" in cats:
            rare_type[cname] = "Knives"
        elif "Gloves" in cats:
            rare_type[cname] = "Gloves"
        else:
            rare_type[cname] = None

        collections = set()
        n_covert = 0
        for entry in case.get("contains") or []:
            case_of_skin[entry["id"]] = cname
            skin = by_id.get(entry["id"])
            if not skin:
                continue
            for col in skin.get("collections") or []:
                collections.add(col["name"])
            if (skin["rarity"]["id"] == RARITY_COVERT
                    and skin["category"]["name"] in WEAPON_CATEGORIES):
                n_covert += 1

        collection_of[cname] = sorted(collections)[0] if collections else None

        case_stats[cname] = {
            "rare_type": rare_type[cname],
            "collection": collection_of[cname],
            "n_covert": n_covert,
            "n_rare_items": len({e["id"] for e in (case.get("contains_rare") or [])}),
            "first_sale_date": case.get("first_sale_date"),
            "market_hash_name": case.get("market_hash_name"),
        }

    return {
        "by_id": by_id,
        "cases": cases,
        "case_of_skin": case_of_skin,
        "rare_type": rare_type,
        "collection_of": collection_of,
        "rare_skin_ids": rare_skin_ids,
        "case_stats": case_stats,
        "souvenir_ids": souvenir_ids,
    }


# Case -> rare pool, many-to-many (a knife can sit in several cases); read by 05 and 06.
def build_case_pools(idx: dict) -> pd.DataFrame:
    by_id = idx["by_id"]
    rows = []
    for case in idx["cases"]:
        rare = case.get("contains_rare") or []
        finishes = {by_id[e["id"]]["name"] for e in rare
                    if e["id"] in by_id
                    and by_id[e["id"]]["category"]["name"] in ("Knives", "Gloves")}
        for entry in rare:
            skin = by_id.get(entry["id"])
            if not skin or skin["category"]["name"] not in ("Knives", "Gloves"):
                continue
            for var in expand_variants(skin, souvenir_ok=False):
                rows.append({
                    "case": case["name"],
                    "market_hash_name": unicodedata.normalize("NFC", var["market_hash_name"]),
                    "skin": skin["name"],
                    "category": skin["category"]["name"],
                    "wear": var["wear"],
                    "stattrak": var["stattrak"],
                    "min_float": skin.get("min_float"),
                    "max_float": skin.get("max_float"),
                    "pool_size_skins": len(finishes),
                })
    out = pd.DataFrame(rows).drop_duplicates(subset=["case", "market_hash_name"])
    log.info("Case pools: %d cases, %d case-item pairs, %d distinct items",
             out["case"].nunique(), len(out), out["market_hash_name"].nunique())
    return out


# Per collection: star pool, highest weapon rarity, release date.
def build_collection_index(meta: dict, idx: dict) -> dict:
    by_id = idx["by_id"]

    star_collections: set[str] = set()
    for case in idx["cases"]:
        if idx["rare_type"].get(case["name"]) is None:
            continue
        col = idx["collection_of"].get(case["name"])
        if col:
            star_collections.add(col)
        for entry in (case.get("contains") or []):
            skin = by_id.get(entry["id"])
            for c in (skin.get("collections") or []) if skin else []:
                star_collections.add(c["name"])

    top: dict[str, int] = {}
    for skin in meta["skins"]:
        if skin["category"]["name"] not in WEAPON_CATEGORIES:
            continue
        rank = RARITY_ORDER.get(skin["rarity"]["id"])
        if rank is None:
            continue
        for c in (skin.get("collections") or []):
            name = c["name"]
            top[name] = max(top.get(name, -1), rank)

    release = {c["name"]: c.get("release_date") for c in meta["collections"]}

    return {name: {"star": name in star_collections,
                   "top_rarity": rank,
                   "release": release.get(name)}
            for name, rank in top.items()}


# Market names that mix phases (Doppler) or high-variance patterns.
def detect_blended_patterns(meta: dict) -> dict[str, str]:
    counts: dict[str, set] = defaultdict(set)
    for s in meta["skins"]:
        if s["category"]["name"] in ("Knives", "Gloves"):
            counts[s["name"]].add(s.get("paint_index"))

    blended = {name: "phase" for name, idx in counts.items() if len(idx) > 1}

    high_variance = ("Case Hardened", "Marble Fade", "Fade")
    for s in meta["skins"]:
        pat = (s.get("pattern") or {}).get("name") or ""
        if pat in high_variance and s["name"] not in blended:
            blended[s["name"]] = "pattern"

    return blended


def build_universe(meta: dict, include_ladder: bool = False,
                   graffiti_limit: int = 150, placebo_low: bool = True,
                   placebo_low_souvenir: bool = False,
                   released_before: str = PLACEBO_LOW_RELEASED_BEFORE
                   ) -> pd.DataFrame:
    idx = build_case_index(meta)
    by_id = idx["by_id"]
    blended = detect_blended_patterns(meta)

    knife_cases = {c for c, t in idx["rare_type"].items() if t == "Knives"}
    glove_cases = {c for c, t in idx["rare_type"].items() if t == "Gloves"}
    star_cases = knife_cases | glove_cases

    rows: list[dict] = []

    def emit(skin: dict, group: str, treated: bool,
             souvenir: Optional[bool] = None, **extra) -> None:
        case_name = idx["case_of_skin"].get(skin["id"])
        # Star items are listed in contains_rare, not in contains.
        if case_name is None and skin["id"] in idx["rare_skin_ids"]:
            for c in idx["cases"]:
                if any(e["id"] == skin["id"] for e in (c.get("contains_rare") or [])):
                    case_name = c["name"]
                    break

        collections = [c["name"] for c in (skin.get("collections") or [])]
        collection = collections[0] if collections else idx["collection_of"].get(case_name)
        souvenir_ok = (skin["id"] in idx["souvenir_ids"]) if souvenir is None else souvenir

        for var in expand_variants(skin, souvenir_ok=souvenir_ok):
            # Souvenir items cannot be used in trade-ups.
            row_treated = treated and not var["souvenir"]
            rows.append({
                **var,
                "group": "placebo_souvenir" if var["souvenir"] else group,
                "treated": row_treated,
                "category": skin["category"]["name"],
                "weapon": (skin.get("weapon") or {}).get("name"),
                "pattern": (skin.get("pattern") or {}).get("name"),
                "rarity": skin["rarity"]["name"],
                "collection": collection,
                "case": case_name,
                "case_rare_type": idx["rare_type"].get(case_name),
                "n_covert_in_case": (idx["case_stats"].get(case_name) or {}).get("n_covert"),
                "blend_warning": blended.get(skin["name"]),
                "min_float": skin.get("min_float"),
                "max_float": skin.get("max_float"),
                "skin_id": skin["id"],
                **extra,
            })

    # 1) Output: knives and gloves from the case pools
    seen_names: set[str] = set()
    # sorted() keeps runs reproducible (set order depends on the hash seed).
    for skin_id in sorted(idx["rare_skin_ids"]):
        skin = by_id.get(skin_id)
        if not skin or skin["name"] in seen_names:
            continue
        seen_names.add(skin["name"])
        cat = skin["category"]["name"]
        emit(skin,
             group="output_knife" if cat == "Knives" else "output_gloves",
             treated=True,
             price_tier=("cheap" if (skin.get("weapon") or {}).get("name") in CHEAP_KNIVES
                         else "expensive" if (skin.get("weapon") or {}).get("name") in EXPENSIVE_KNIVES
                         else "mid"))

    # 2) Input / placebo: Covert weapons with or without a star pool in their case
    for skin in meta["skins"]:
        if skin["rarity"]["id"] != RARITY_COVERT:
            continue
        if skin["category"]["name"] not in WEAPON_CATEGORIES:
            continue

        case_name = idx["case_of_skin"].get(skin["id"])
        in_star_case = case_name in star_cases if case_name else False

        if in_star_case:
            is_glove_case = case_name in glove_cases
            emit(skin, group="input_covert", treated=True,
                 st_no_output=is_glove_case,
                 ladder_steps=1)
        else:
            emit(skin, group="placebo_covert", treated=False, ladder_steps=None)

    # StatTrak Coverts of glove cases have no output (no StatTrak gloves).
    for r in rows:
        if r.get("st_no_output") and r.get("stattrak"):
            r["treated"] = False
            r["group"] = "placebo_covert_st"

    # 2b) Placebo Classified: Classified skins in collections without a higher tier
    if placebo_low:
        cidx = build_collection_index(meta, idx)
        cutoff = pd.Timestamp(released_before)
        n_low = 0
        for skin in meta["skins"]:
            if skin["rarity"]["id"] != RARITY_CLASSIFIED:
                continue
            if skin["category"]["name"] not in WEAPON_CATEGORIES:
                continue
            cols = [c["name"] for c in (skin.get("collections") or [])]
            if not cols:
                continue
            info = cidx.get(cols[0])
            if not info or info["star"]:
                continue
            if info["top_rarity"] != RARITY_ORDER[RARITY_CLASSIFIED]:
                continue
            rel = pd.Timestamp(info["release"]) if info["release"] else pd.NaT
            if pd.isna(rel) or rel >= cutoff:
                continue
            emit(skin, group="placebo_classified", treated=False,
                 souvenir=placebo_low_souvenir, ladder_steps=None)
            n_low += 1
        log.info("Second placebo layer: %d skins from collections without a "
                 "higher tier (released before %s)", n_low, released_before)

    # 3) Optional ladder: lower rarities in star cases
    if include_ladder:
        for skin in meta["skins"]:
            tier = LADDER.get(skin["rarity"]["id"])
            if not tier:
                continue
            if skin["category"]["name"] not in WEAPON_CATEGORIES:
                continue
            case_name = idx["case_of_skin"].get(skin["id"])
            if case_name not in star_cases:
                continue
            label, steps = tier
            emit(skin, group=f"ladder_{label}", treated=True, ladder_steps=steps)

    # 4) Cases: own group, the expected effect has no clear sign
    for case in idx["cases"]:
        mhn = case.get("market_hash_name")
        if not mhn:
            continue
        st = idx["case_stats"][case["name"]]
        rows.append({
            "market_hash_name": mhn, "wear": None, "stattrak": False,
            "souvenir": False, "is_vanilla": False,
            "group": "case", "treated": True,
            "category": "Container", "weapon": None, "pattern": None,
            "rarity": case.get("rarity", {}).get("name"),
            "collection": st["collection"], "case": case["name"],
            "case_rare_type": st["rare_type"],
            "n_covert_in_case": st["n_covert"],
            "blend_warning": None, "min_float": None, "max_float": None,
            "skin_id": case["id"], "first_sale_date": st["first_sale_date"],
        })

    # 5) Controls: sticker capsules, agents, music kits, graffiti
    def simple(entries, group, name_field="market_hash_name",
               category=None, limit=None):
        n = 0
        for e in entries:
            mhn = e.get(name_field)
            if not mhn:
                continue
            rows.append({
                "market_hash_name": mhn, "wear": None, "stattrak": False,
                "souvenir": False, "is_vanilla": False,
                "group": group, "treated": False,
                "category": category, "weapon": None, "pattern": None,
                "rarity": (e.get("rarity") or {}).get("name"),
                "collection": ((e.get("collections") or [{}])[0].get("name")
                               if e.get("collections") else None),
                "case": None, "case_rare_type": None, "n_covert_in_case": None,
                "blend_warning": None, "min_float": None, "max_float": None,
                "skin_id": e.get("id"),
            })
            n += 1
            if limit and n >= limit:
                break

    sticker_caps = [c for c in meta["crates"] if c.get("type") == "Sticker Capsule"]
    simple(sticker_caps, "capsule_sticker", category="Container")
    simple(meta["agents"], "control_agent", category="Agent")
    simple(meta["music_kits"], "control_musickit", category="MusicKit")
    simple(meta["graffiti"], "control_graffiti", category="Graffiti",
           limit=graffiti_limit)

    df = pd.DataFrame(rows)

    # NFC so that ★ and ™ match Steam's spelling byte for byte.
    df["market_hash_name"] = (df["market_hash_name"]
                              .astype(str)
                              .map(lambda s: unicodedata.normalize("NFC", s).strip()))

    before = len(df)
    df = df.drop_duplicates(subset="market_hash_name", keep="first")
    if len(df) < before:
        log.info("%d duplicate market names removed (e.g. Doppler phases).",
                 before - len(df))

    cols = ["market_hash_name", "group", "treated", "category", "weapon",
            "pattern", "wear", "stattrak", "souvenir", "is_vanilla", "rarity",
            "collection", "case", "case_rare_type", "n_covert_in_case",
            "ladder_steps", "st_no_output", "price_tier", "blend_warning",
            "min_float", "max_float", "first_sale_date", "skin_id"]
    for c in cols:
        if c not in df.columns:
            df[c] = None

    return df[cols].sort_values(["group", "market_hash_name"]).reset_index(drop=True)


# Stratified pilot: covers every group and name construction, not representative.
def draw_pilot(df: pd.DataFrame, size: int = 150, seed: int = 20251023) -> pd.DataFrame:
    quota = {
        "output_knife": 0.24, "output_gloves": 0.09,
        "input_covert": 0.20, "placebo_covert": 0.09,
        "placebo_covert_st": 0.06, "placebo_souvenir": 0.05,
        "case": 0.08, "capsule_sticker": 0.06,
        "control_agent": 0.04, "control_musickit": 0.04,
        "control_graffiti": 0.05,
    }

    parts = []
    for group, share in quota.items():
        sub = df[df["group"] == group]
        if sub.empty:
            continue
        n = max(2, int(round(size * share)))
        n = min(n, len(sub))

        strata = sub.groupby([sub["wear"].fillna("_none"),
                              sub["stattrak"].astype(str)], dropna=False)
        picked = strata.apply(
            lambda g: g.sample(min(len(g), max(1, n // max(1, strata.ngroups))),
                               random_state=seed),
            include_groups=False,
        )
        picked = sub.loc[picked.index.get_level_values(-1)] if len(picked) else sub.head(0)

        if len(picked) < n:
            rest = sub.drop(index=picked.index)
            extra = rest.sample(min(len(rest), n - len(picked)), random_state=seed)
            picked = pd.concat([picked, extra])
        parts.append(picked.head(n))

    pilot = pd.concat(parts).drop_duplicates(subset="market_hash_name")

    # Always include the name constructions that break most easily.
    must_have = [
        df[df["is_vanilla"] & ~df["stattrak"]],
        df[df["is_vanilla"] & df["stattrak"]],
        df[(df["group"] == "output_knife") & df["stattrak"]],
        df[df["group"] == "output_gloves"],
        df[df["souvenir"]],
        df[df["blend_warning"] == "phase"],
        df[df["group"] == "control_graffiti"],
    ]
    for block in must_have:
        if not block.empty:
            pilot = pd.concat([pilot, block.sample(min(3, len(block)), random_state=seed)])

    return (pilot.drop_duplicates(subset="market_hash_name")
                 .sort_values(["group", "market_hash_name"])
                 .reset_index(drop=True))


def report(df: pd.DataFrame, idx_meta: dict) -> None:
    log.info("=" * 70)
    log.info("ITEM UNIVERSE")
    log.info("=" * 70)

    for group, sub in df.groupby("group"):
        st = sub["stattrak"].sum()
        log.info("  %-20s %6d items   (StatTrak: %d, treated: %s)",
                 group, len(sub), st,
                 "yes" if sub["treated"].iloc[0] else "no")
    log.info("  %-20s %6d", "TOTAL", len(df))

    log.info("")
    log.info("--- Trade-up structure ---")
    stats = idx_meta["case_stats"]
    kn = sum(1 for v in stats.values() if v["rare_type"] == "Knives")
    gl = sum(1 for v in stats.values() if v["rare_type"] == "Gloves")
    log.info("  Cases with a knife pool   : %d", kn)
    log.info("  Cases with a glove pool   : %d", gl)
    log.info("  Cases without a ★ pool    : %d", len(stats) - kn - gl)

    blend = df[df["blend_warning"] == "phase"]
    log.info("")
    log.info("--- Warnings ---")
    log.info("  Blended phase prices (Doppler & co.): %d items", len(blend))
    log.info("  High pattern dispersion (Fade, Case Hardened): %d items",
             (df["blend_warning"] == "pattern").sum())
    st_asym = df[df["group"] == "placebo_covert_st"]
    log.info("  StatTrak without a valid output (glove cases): %d items",
             len(st_asym))

    est_hours = len(df) * 4.5 / 3600
    log.info("")
    log.info("  Estimated collection time at 4.5 s/request: %.1f hours", est_hours)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Builds the item list for the CS2 trade-up event study.")
    p.add_argument("--outdir", default="items")
    p.add_argument("--cache", default="meta_cache")
    p.add_argument("--refresh", action="store_true",
                   help="Download the metadata again instead of using the cache.")
    p.add_argument("--include-ladder", action="store_true",
                   help="Add Classified/Restricted/Mil-Spec from ★ cases "
                        "(dose-response analysis; many more items).")
    p.add_argument("--no-placebo-low", action="store_true",
                   help="Drop the second placebo layer (Classified without a "
                        "higher tier).")
    p.add_argument("--placebo-low-souvenir", action="store_true",
                   help="Also include the souvenir variants of the second "
                        "placebo layer (default: no, very thinly traded).")
    p.add_argument("--placebo-low-released-before",
                   default=PLACEBO_LOW_RELEASED_BEFORE,
                   help="Only collections released before this date "
                        "(default: start of the estimation window).")
    p.add_argument("--pilot-size", type=int, default=150)
    p.add_argument("--graffiti-limit", type=int, default=150)
    p.add_argument("--seed", type=int, default=20251023)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    meta = load_metadata(Path(args.cache), refresh=args.refresh)
    log.info("")

    df = build_universe(meta, include_ladder=args.include_ladder,
                        graffiti_limit=args.graffiti_limit,
                        placebo_low=not args.no_placebo_low,
                        placebo_low_souvenir=args.placebo_low_souvenir,
                        released_before=args.placebo_low_released_before)
    idx = build_case_index(meta)
    report(df, idx)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # utf-8-sig so that Excel shows ★ and ™ correctly.
    full = outdir / "items_full.csv"
    df.to_csv(full, index=False, encoding="utf-8-sig")

    pilot = draw_pilot(df, size=args.pilot_size, seed=args.seed)
    ppath = outdir / "items_pilot.csv"
    pilot.to_csv(ppath, index=False, encoding="utf-8-sig")

    cdf = pd.DataFrame([
        {"case": k, **v} for k, v in idx["case_stats"].items()
    ]).sort_values(["rare_type", "first_sale_date"])
    cpath = outdir / "collections.csv"
    cdf.to_csv(cpath, index=False, encoding="utf-8-sig")

    pools = build_case_pools(idx)
    pool_path = outdir / "case_pools.csv"
    pools.to_csv(pool_path, index=False, encoding="utf-8-sig")

    # Separate file so that the collector can fetch only this group.
    low = df[df["group"] == "placebo_classified"]
    lpath = None
    if not low.empty:
        lpath = outdir / "items_placebo_classified.csv"
        low.to_csv(lpath, index=False, encoding="utf-8-sig")

    log.info("")
    log.info("Written:")
    log.info("  %-34s %6d rows", str(full), len(df))
    log.info("  %-34s %6d rows", str(ppath), len(pilot))
    log.info("  %-34s %6d rows", str(cpath), len(cdf))
    log.info("  %-34s %6d rows", str(pool_path), len(pools))
    if lpath is not None:
        log.info("  %-34s %6d rows", str(lpath), len(low))
    log.info("")
    log.info("Next step: collect the prices, see README.md (Collecting the Data Again).")


if __name__ == "__main__":
    main()

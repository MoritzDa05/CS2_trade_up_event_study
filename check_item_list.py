# Checks the item list of build_item_universe.py against Valve's game files
# (SteamDatabase/GameTracking-CS2) and writes one table and a log.
#
# The item list comes from ByMykel's CSGO-API. The reference side of every
# check comes only from Valve's files of one commit: items_game.txt, the
# English localization, and pak01_dir.txt, the list of all files in the game.
# The game ships one generated icon for every knife or glove with a paint kit
# that exists, so this list shows which finishes exist.
#
# The files do not define which finishes a knife or glove pool contains: a case
# names its pool (e.g. set_community_33_unusual), but the list behind the name
# is not in the public files. ByMykel fills these lists by hand. Checks 7, 9
# and 10 test everything around them that the files do define; which pool a
# finish belongs to cannot be checked.
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

import config as C

log = logging.getLogger("check_item_list")

REPO = "SteamDatabase/GameTracking-CS2"
GAME_FILES = {
    "items_game": "game/csgo/pak01_dir/scripts/items/items_game.txt",
    "english": "game/csgo/pak01_dir/resource/csgo_english.txt",
    "file_list": "game/csgo/pak01_dir.txt",
}
API_COMMITS = f"https://api.github.com/repos/{REPO}/commits"
RAW = "https://raw.githubusercontent.com/" + REPO + "/{sha}/{path}"

# The rarity of a skin in the game file is the suffix of its loot list.
RARITY = {
    "rarity_common_weapon": "common",
    "rarity_uncommon_weapon": "uncommon",
    "rarity_rare_weapon": "rare",
    "rarity_mythical_weapon": "mythical",
    "rarity_legendary_weapon": "legendary",
    "rarity_ancient_weapon": "ancient",
    "rarity_contraband": "immortal",
}
RARITY_SUFFIX = re.compile(r"_(common|uncommon|rare|mythical|legendary|ancient|immortal)$")
ENTRY = re.compile(r"^\[(?P<kit>[^\]]+)\](?P<weapon>weapon_\w+)$")


# Paint kit names are case-insensitive in the game file, so both sides are lower-cased.
def entry(text: str) -> tuple[str, str] | None:
    hit = ENTRY.match(text)
    return (hit["kit"].lower(), hit["weapon"].lower()) if hit else None


POOL_LISTS = ("unusual_revolving_list",)  # pool of the oldest cases
GLOVE_POOL_NAME = "★ Gloves ★"
GENERIC_POOL_NAME = "★ Rare Special Item ★"

# Knives and gloves that take a paint kit, by prefab in items_game.txt.
KNIFE_PREFAB = "melee_unusual"
GLOVE_PREFAB = "hands_paintable"
# Generated icon of a weapon with a paint kit: <item>_<kit>_<light|medium|heavy>.
ICON = re.compile(r"^panorama/images/econ/default_generated/(\w+?)_(?:light|medium|heavy)_png\.vtex_c\b")

WEAPON_GROUPS = ["input_covert", "placebo_covert_st", "placebo_covert",
                 "placebo_classified", "placebo_souvenir"]
CASE_COVERT_GROUPS = ["input_covert", "placebo_covert_st"]
FLOAT_TOL = 1e-6
SHOW = 5  # differences listed per check in the table


def setup_logging() -> None:
    C.OUT_LOGS.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(C.OUT_LOGS / "check_item_list.log", mode="w",
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


def resolve_commit(until: str) -> tuple[str, str]:
    r = requests.get(API_COMMITS, params={"path": GAME_FILES["items_game"],
                                          "until": until, "per_page": 1},
                     timeout=60)
    r.raise_for_status()
    commits = r.json()
    if not commits:
        raise SystemExit(f"No commit of items_game.txt before {until}.")
    c = commits[0]
    return c["sha"], c["commit"]["committer"]["date"]


def load_game_files(sha: str) -> dict[str, str]:
    cache = C.ITEM_CHECK_CACHE / sha
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, path in GAME_FILES.items():
        target = cache / Path(path).name
        if not target.exists():
            r = requests.get(RAW.format(sha=sha, path=path), timeout=120)
            r.raise_for_status()
            target.write_bytes(r.content)
            log.info("Downloaded %s (%.1f MB)", target.name, len(r.content) / 1e6)
        out[key] = target.read_text(encoding="utf-8-sig")
    return out


# Valve KeyValues parser; a list of pairs because items_game.txt repeats keys.
TOKEN = re.compile(r'"((?:[^"\\]|\\.)*)"|(\{)|(\})|(//[^\n]*)|([^\s{}"]+)')


def parse_kv(text: str) -> list:
    stack: list[list] = [[]]
    key = None
    for m in TOKEN.finditer(text):
        quoted, opening, closing, comment, bare = m.groups()
        if comment is not None:
            continue
        if opening:
            block: list = []
            stack[-1].append((key, block))
            stack.append(block)
            key = None
        elif closing:
            stack.pop()
        else:
            tok = quoted if quoted is not None else bare
            if key is None:
                key = tok
            else:
                stack[-1].append((key, tok))
                key = None
    return stack[0]


def blocks(pairs: list, name: str) -> list:
    out = []
    for k, v in pairs:
        if k == name and isinstance(v, list):
            out.extend(v)
    return out


def scalars(pairs: list) -> dict:
    return {k: v for k, v in pairs if isinstance(v, str)}


# A key defined twice keeps its first definition.
def first_wins(pairs: list, label: str) -> dict:
    out: dict = {}
    dupes = 0
    for k, v in pairs:
        if k in out:
            dupes += out[k] != v
            continue
        out[k] = v
    if dupes:
        log.warning("%s: %d keys defined twice with different content",
                    label, dupes)
    return out


class GameFile:
    def __init__(self, text: str, english: str, file_list: str):
        root = blocks(parse_kv(text), "items_game")
        lang = blocks(blocks(parse_kv(english), "lang"), "Tokens")
        self.loc = {k.lower(): v for k, v in lang if isinstance(v, str)}

        self.items = first_wins(blocks(root, "items"), "items")
        self.prefabs = first_wins(blocks(root, "prefabs"), "prefabs")
        self.loot = first_wins(blocks(root, "client_loot_lists"), "client_loot_lists")
        self.series = first_wins(blocks(root, "revolving_loot_lists"),
                                 "revolving_loot_lists")
        self.sets = first_wins(blocks(root, "item_sets"), "item_sets")
        self.kits = {}
        for _, kit in blocks(root, "paint_kits"):
            if isinstance(kit, list):
                d = scalars(kit)
                self.kits.setdefault((d.get("name") or "").lower(), d)
        default = self.kits.get("default", {})
        self.default_range = (float(default.get("wear_remap_min", 0.06)),
                              float(default.get("wear_remap_max", 0.80)))

        # Rarity of every (paint kit, weapon) from the loot lists it is in.
        self.rarity: dict[tuple, set] = defaultdict(set)
        for name, entries in self.loot.items():
            m = RARITY_SUFFIX.search(name)
            if not m or not isinstance(entries, list):
                continue
            for e, _ in entries:
                key = entry(e)
                if key:
                    self.rarity[key].add(m.group(1))

        self.containers = self._containers()

        # Display name of every weapon, knife and glove item, e.g.
        # weapon_knife_karambit -> "Karambit"; knives and gloves by prefab.
        self.item_names: dict[str, str | None] = {}
        self.knives: set[str] = set()
        self.gloves: set[str] = set()
        for block in self.items.values():
            if not isinstance(block, list):
                continue
            d = scalars(block)
            name = (d.get("name") or "").lower()
            if not name or name in self.item_names:
                continue
            self.item_names[name] = self.text(self.attr(d, "item_name"))
            if d.get("prefab") == KNIFE_PREFAB:
                self.knives.add(name)
            elif d.get("prefab") == GLOVE_PREFAB:
                self.gloves.add(name)

        # Knife and glove finishes that exist, from the generated icons in the
        # file list: market name -> paint kits, e.g. "★ Karambit | Doppler" ->
        # the seven Doppler kits. The longest item name is matched first,
        # because some item names begin with another one.
        self.icon_files = 0
        self.star_finishes: dict[str, set] = defaultdict(set)
        self.unnamed_icons: list[str] = []
        star = sorted(self.knives | self.gloves, key=len, reverse=True)
        for line in file_list.splitlines():
            hit = ICON.match(line)
            if not hit:
                continue
            self.icon_files += 1
            stem = hit.group(1).lower()
            item = next((s for s in star if stem.startswith(s + "_")), None)
            if item is None:
                continue
            kit = stem[len(item) + 1:]
            finish = self.kit_name(kit)
            if finish is None or self.item_names.get(item) is None:
                self.unnamed_icons.append(stem)
                continue
            self.star_finishes[f"★ {self.item_names[item]} | {finish}"].add(kit)

    def text(self, token: str | None) -> str | None:
        return self.loc.get((token or "").lstrip("#").lower())

    def attr(self, d: dict, key: str) -> str | None:
        """Value of `key` in an item, or else in its prefabs, depth first."""
        if key in d:
            return d[key]
        for name in (d.get("prefab") or "").split():
            prefab = self.prefabs.get(name)
            if isinstance(prefab, list):
                value = self.attr(scalars(prefab), key)
                if value is not None:
                    return value
        return None

    def kit_name(self, kit: str) -> str | None:
        """Display name of a paint kit, e.g. am_doppler_phase2 -> "Doppler"."""
        d = self.kits.get((kit or "").lower())
        return self.text(d.get("description_tag")) if d else None

    def skin_name(self, kit: str, weapon: str) -> str | None:
        """Market name of a weapon with a paint kit, e.g. "AK-47 | Asiimov"."""
        item, finish = self.item_names.get(weapon), self.kit_name(kit)
        return f"{item} | {finish}" if item and finish else None

    # Every item that opens into a loot list with a knife or glove pool.
    def _containers(self) -> dict[str, dict]:
        out = {}
        for def_index, block in self.items.items():
            if not isinstance(block, list):
                continue
            attrs = first_wins(blocks(block, "attributes"), "attributes")
            series = attrs.get("set supply crate series")
            if not isinstance(series, list):
                continue
            lootlist = self.series.get(scalars(series).get("value"))
            subs = [e for e, _ in self.loot.get(lootlist, [])] \
                if isinstance(self.loot.get(lootlist), list) else []
            pool_lists = [s for s in subs if s.endswith("_unusual") or s in POOL_LISTS]
            if not pool_lists:
                continue
            d = scalars(block)
            contents = []
            for sub in subs:
                m = RARITY_SUFFIX.search(sub)
                entries = self.loot.get(sub)
                if not m or not isinstance(entries, list):
                    continue
                for e, _ in entries:
                    key = entry(e)
                    if key:
                        contents.append((*key, m.group(1)))
            pool_name = self.text(d.get("loot_list_rare_item_name"))
            out[def_index] = {
                "name": self.text(d.get("item_name")) or d.get("name"),
                "prefab": d.get("prefab", ""),
                "pool_name": pool_name,
                # Internal name of the pool, e.g. set_community_33_unusual.
                "pool_list": "+".join(sorted(pool_lists)),
                "contents": contents,
            }
        return out

    def float_range(self, kit: str) -> tuple[float, float] | None:
        d = self.kits.get((kit or "").lower())
        if d is None:
            return None
        return (float(d.get("wear_remap_min", self.default_range[0])),
                float(d.get("wear_remap_max", self.default_range[1])))


# The item list as the pipeline uses it (ByMykel files, items_full.csv).
class ItemList:
    def __init__(self):
        meta = C.ROOT / "meta_cache"
        self.skins = json.loads((meta / "skins.json").read_text(encoding="utf-8"))
        self.crates = json.loads((meta / "crates.json").read_text(encoding="utf-8"))
        self.items = pd.read_csv(C.ITEMS_UNIVERSE, encoding="utf-8-sig")
        self.pools = pd.read_csv(C.CASE_POOLS, encoding="utf-8-sig")

        self.by_id = {s["id"]: s for s in self.skins}
        self.by_name: dict[str, list] = defaultdict(list)
        for s in self.skins:
            self.by_name[s["name"]].append(s)
        self.cases = {c["def_index"]: c for c in self.crates
                      if c.get("type") == "Case"}
        self.case_index = {c["name"]: k for k, c in self.cases.items()}

    @staticmethod
    def key(skin: dict) -> tuple[str, str]:
        return (((skin.get("pattern") or {}).get("id") or "").lower(),
                ((skin.get("weapon") or {}).get("id") or "").lower())

    def weapon_skins(self, groups: list[str]) -> pd.DataFrame:
        sub = self.items[self.items["group"].isin(groups)]
        sub = sub.drop_duplicates(["skin_id", "group"])
        rows = []
        for r in sub.itertuples(index=False):
            s = self.by_id.get(r.skin_id)
            if s is None:
                log.warning("skin_id %s not in skins.json", r.skin_id)
                continue
            kit, weapon = self.key(s)
            rows.append({"group": r.group, "skin": s["name"], "kit": kit,
                         "weapon": weapon, "case": r.case,
                         "collection": r.collection,
                         "rarity": RARITY.get(s["rarity"]["id"]),
                         "min_float": s.get("min_float"),
                         "max_float": s.get("max_float")})
        return pd.DataFrame(rows)


def result(check: str, game: int, listed: int, agree: int,
           differences: list[str]) -> dict:
    for d in differences:
        log.info("  differs: %s", d)
    return {"check": check, "game_file": game, "item_list": listed,
            "agree": agree, "differ": len(differences),
            "detail": "; ".join(differences[:SHOW])
                      + (f"; ... ({len(differences)} in total)"
                         if len(differences) > SHOW else "")}


def check_containers(g: GameFile, il: ItemList) -> list[dict]:
    section("1. Containers with a knife or glove pool")
    cases_game = {k for k, v in g.containers.items() if v["prefab"] == "weapon_case"}
    other = {k: v for k, v in g.containers.items() if v["prefab"] != "weapon_case"}
    listed = set(il.cases)
    diff = ([f"{g.containers[k]['name']} ({k}) not in the list"
             for k in sorted(cases_game - listed)]
            + [f"{il.cases[k]['name']} ({k}) has no pool in the game file"
               for k in sorted(listed - cases_game)])
    log.info("Cases with a pool: %d in the game file, %d in the list",
             len(cases_game), len(listed))
    rows = [result("Cases with a knife or glove pool", len(cases_game),
                   len(listed), len(cases_game & listed), diff)]
    diff = [f"{v['name']} ({k}, {v['prefab']}, pool: {v['pool_name']})"
            for k, v in sorted(other.items())]
    log.info("Other containers with a pool: %d", len(other))
    rows.append(result("Other containers with a knife or glove pool",
                       len(other), 0, 0, diff))
    return rows


def check_pool_type(g: GameFile, il: ItemList) -> list[dict]:
    section("2. Knives or gloves, knife type")
    rows, diff, agree = [], [], 0
    pool_type = (il.items[il.items["group"] == "case"]
                 .set_index("case")["case_rare_type"])
    for k, c in sorted(il.cases.items()):
        game = g.containers.get(k)
        if game is None:
            continue
        want = "Gloves" if game["pool_name"] == GLOVE_POOL_NAME else "Knives"
        have = pool_type.get(c["name"])
        if have == want:
            agree += 1
        else:
            diff.append(f"{c['name']}: list {have}, game file {want}")
    rows.append(result("Pool holds knives or gloves", len(il.cases),
                       len(il.cases), agree, diff))

    # Cases whose pool name gives a single knife type, e.g. "★ Falchion Knife ★".
    diff, agree, n = [], 0, 0
    for k, c in sorted(il.cases.items()):
        game = g.containers.get(k)
        name = (game or {}).get("pool_name") or ""
        if name in (GLOVE_POOL_NAME, GENERIC_POOL_NAME) or not name.startswith("★"):
            continue
        n += 1
        want = name.strip("★ ").strip()
        have = set(il.pools.loc[il.pools["case"] == c["name"], "skin"]
                   .str.replace("★ ", "", regex=False)
                   .str.split(" | ", regex=False).str[0])
        if have == {want}:
            agree += 1
        else:
            diff.append(f"{c['name']}: list {sorted(have)}, game file {want}")
    log.info("Cases with a single knife type: %d", n)
    rows.append(result("Knife type of cases with a single knife type", n, n,
                       agree, diff))
    return rows


def check_case_coverts(g: GameFile, il: ItemList) -> list[dict]:
    section("3. Covert weapon skins per case")
    skins = il.weapon_skins(CASE_COVERT_GROUPS).drop_duplicates(["kit", "weapon"])
    listed = {(il.case_index.get(r.case), r.kit, r.weapon)
              for r in skins.itertuples(index=False)}
    game = {(k, kit, weapon)
            for k in il.cases if k in g.containers
            for kit, weapon, rar in g.containers[k]["contents"] if rar == "ancient"}
    name = {k: c["name"] for k, c in il.cases.items()}
    diff = ([f"{name.get(c, c)}: [{kit}]{w} only in the game file"
             for c, kit, w in sorted(game - listed)]
            + [f"{name.get(c, c)}: [{kit}]{w} only in the list"
               for c, kit, w in sorted(listed - game)])
    log.info("Covert skins in cases: %d in the game file, %d in the list",
             len(game), len(listed))
    return [result("Covert weapon skins in the cases", len(game), len(listed),
                   len(game & listed), diff)]


def check_rarity(g: GameFile, il: ItemList) -> list[dict]:
    section("4. Rarity of the weapon skins")
    skins = il.weapon_skins(WEAPON_GROUPS).drop_duplicates(["kit", "weapon"])
    diff, agree, missing = [], 0, []
    for r in skins.itertuples(index=False):
        seen = g.rarity.get((r.kit, r.weapon))
        if not seen:
            missing.append(f"{r.skin} ({r.group}) in no loot list")
        elif r.rarity in seen:
            agree += 1
        else:
            diff.append(f"{r.skin}: list {r.rarity}, game file {sorted(seen)}")
    log.info("Weapon skins: %d, rarity agrees for %d, %d in no loot list",
             len(skins), agree, len(missing))
    return [result("Rarity of the weapon skins", len(skins) - len(missing),
                   len(skins), agree, diff + missing)]


def check_placebos(g: GameFile, il: ItemList) -> list[dict]:
    section("5. Placebo Map and Placebo Classified")
    rows = []
    in_pool = {(kit, w): v["name"] for v in g.containers.values()
               for kit, w, _ in v["contents"]}
    skins = il.weapon_skins(["placebo_covert"]).drop_duplicates(["kit", "weapon"])
    diff = [f"{r.skin} is in {in_pool[(r.kit, r.weapon)]}"
            for r in skins.itertuples(index=False) if (r.kit, r.weapon) in in_pool]
    rows.append(result("Map Coverts outside containers with a pool", len(skins),
                       len(skins), len(skins) - len(diff), diff))

    # Collections of Placebo Classified must not contain a Covert.
    skins = il.weapon_skins(["placebo_classified"])
    members = {}
    for _, entries in g.sets.items():
        if not isinstance(entries, list):
            continue
        listed = [e for e, _ in blocks(entries, "items")]
        members[g.text(scalars(entries).get("name"))] = listed
    diff, agree, cols = [], 0, sorted(skins["collection"].dropna().unique())
    for col in cols:
        entries = members.get(col)
        if entries is None:
            diff.append(f"{col}: no item set of this name in the game file")
            continue
        top = set()
        for e in entries:
            key = entry(e)
            if key:
                top |= g.rarity.get(key, set())
        if top & {"ancient", "immortal"}:
            diff.append(f"{col}: contains {sorted(top & {'ancient', 'immortal'})}")
        else:
            agree += 1
    rows.append(result("Collections of Placebo Classified without a Covert",
                       len(cols), len(cols), agree, diff))
    return rows


def check_floats(g: GameFile, il: ItemList) -> list[dict]:
    section("6. Float ranges")
    rows = []

    def compare(label: str, frame: pd.DataFrame) -> dict:
        diff, agree = [], 0
        for r in frame.itertuples(index=False):
            rng = g.float_range(r.kit)
            if rng is None:
                diff.append(f"{r.skin}: paint kit {r.kit} not in the game file")
            elif (abs(rng[0] - float(r.min_float)) <= FLOAT_TOL
                  and abs(rng[1] - float(r.max_float)) <= FLOAT_TOL):
                agree += 1
            else:
                diff.append(f"{r.skin}: list {r.min_float}-{r.max_float}, "
                            f"game file {rng[0]:g}-{rng[1]:g}")
        log.info("%s: %d compared, %d agree", label, len(frame), agree)
        return result(label, len(frame), len(frame), agree, diff)

    weapons = il.weapon_skins(WEAPON_GROUPS).drop_duplicates(["kit", "weapon"])
    rows.append(compare("Float range of the weapon skins", weapons))

    # Doppler phases are separate paint kits with one market name; each is compared.
    finishes = (il.pools.dropna(subset=["min_float"])
                .drop_duplicates("skin")[["skin", "min_float", "max_float"]])
    recs = []
    for r in finishes.itertuples(index=False):
        for s in il.by_name.get(r.skin, []):
            kit, _ = il.key(s)
            recs.append({"skin": r.skin, "kit": kit, "min_float": r.min_float,
                         "max_float": r.max_float})
    frame = pd.DataFrame(recs)
    rows.append(compare("Float range of the knife and glove finishes", frame))
    log.info("Knife and glove finishes: %d market names, %d paint kits",
             finishes["skin"].nunique(), len(frame))
    return rows


def check_pool_identity(g: GameFile, il: ItemList) -> list[dict]:
    """Cases with the same pool name in the game file must have the same pool in
    the list, and cases with different pool names different pools. ByMykel
    fills its pools by these names, so this finds a case linked to the wrong
    pool, not an error inside a pool."""
    section("7. Cases that share a pool")
    finishes = il.pools.groupby("case")["skin"].apply(frozenset).to_dict()
    by_name: dict[str, list] = defaultdict(list)
    for k, c in sorted(il.cases.items()):
        if k in g.containers:
            by_name[g.containers[k]["pool_list"]].append(c["name"])
    owner: dict[frozenset, set] = defaultdict(set)   # pool in the list -> names
    for name, cases in by_name.items():
        for case in cases:
            owner[finishes.get(case, frozenset())].add(name)
    diff, agree = [], 0
    for name, cases in sorted(by_name.items()):
        pools = {finishes.get(case, frozenset()) for case in cases}
        shared = set().union(*(owner[p] for p in pools)) - {name}
        log.info("  %-28s %2d cases, %s finishes", name, len(cases),
                 "/".join(str(len(p)) for p in pools))
        if len(pools) > 1:
            diff.append(f"{name}: cases with different pools in the list "
                        f"({', '.join(cases)})")
        elif shared:
            diff.append(f"{name}: same pool in the list as {', '.join(sorted(shared))}")
        else:
            agree += 1
    distinct = len({finishes.get(c["name"], frozenset()) for k, c in il.cases.items()
                    if k in g.containers})
    log.info("Pool names in the game file: %d; distinct pools in the list: %d",
             len(by_name), distinct)
    return [result("Cases sharing a pool", len(by_name), distinct, agree, diff)]


def check_names(g: GameFile, il: ItemList) -> list[dict]:
    """The market name in the list must be the name the game file gives the
    paint kit and weapon the list assigns to it. Checks 3 to 6 rely on this
    assignment."""
    section("8. Names of the weapon skins")
    skins = il.weapon_skins(WEAPON_GROUPS).drop_duplicates(["kit", "weapon"])
    diff, agree = [], 0
    for r in skins.itertuples(index=False):
        name = g.skin_name(r.kit, r.weapon)
        if name == r.skin:
            agree += 1
        else:
            diff.append(f"{r.skin}: game file {name} ([{r.kit}]{r.weapon})")
    log.info("Weapon skins: %d, name agrees for %d", len(skins), agree)
    return [result("Names of the weapon skins", len(skins), len(skins), agree, diff)]


def star_items(g: GameFile, il: ItemList) -> tuple[set, set]:
    """Knife and glove finishes plus vanilla knives: in the game, in the pools."""
    game = set(g.star_finishes) | {f"★ {g.item_names[k]}" for k in g.knives}
    listed = set(il.pools["skin"])
    return game, listed


def check_star_finishes(g: GameFile, il: ItemList) -> list[dict]:
    """Every knife and glove finish of the pools must exist in the game, under
    the same name and with the same paint kits. The game side comes only from
    the file list and the localization, not from the item list."""
    section("9. Knife and glove finishes of the pools in the game")
    if g.unnamed_icons:
        log.warning("Icons without a name in the game file: %d (%s)",
                    len(g.unnamed_icons), ", ".join(g.unnamed_icons[:SHOW]))
    log.info("Game files: %d generated icons; %d knife and glove finishes with "
             "%d paint kits; %d knife types", g.icon_files, len(g.star_finishes),
             sum(len(v) for v in g.star_finishes.values()), len(g.knives))
    game, listed = star_items(g, il)
    missing = sorted(listed - game)
    log.info("Finishes and vanilla knives: %d in the game, %d in the pools, "
             "%d of these in the game", len(game), len(listed),
             len(listed) - len(missing))
    rows = [result("Pool finishes that exist in the game", len(game), len(listed),
                   len(listed & game), [f"{s} not in the game files" for s in missing])]

    # Paint kits per finish: Doppler and Gamma Doppler have one kit per color
    # variant. Game side from the icons, list side from ByMykel's skins.
    diff, n_game, n_list, agree = [], 0, 0, 0
    for name in sorted(s for s in listed if " | " in s):
        want = g.star_finishes.get(name, set())
        have = {il.key(s)[0] for s in il.by_name.get(name, [])}
        n_game, n_list, agree = n_game + len(want), n_list + len(have), \
            agree + len(want & have)
        if want != have:
            diff.append(f"{name}: list {sorted(have)}, game files {sorted(want)}")
    log.info("Paint kits of the pool finishes: %d in the game, %d in the list",
             n_game, n_list)
    rows.append(result("Paint kits of the pool finishes", n_game, n_list, agree, diff))
    return rows


def check_star_completeness(g: GameFile, il: ItemList) -> list[dict]:
    """Every knife and glove finish in the game must be in at least one pool.
    Known exception (Appendix B): the gloves of the Dead Hand Collection, which
    come from a terminal that the list does not have as a case."""
    section("10. Knife and glove finishes of the game in the pools")
    game, listed = star_items(g, il)
    extra = sorted(game - listed)
    log.info("Finishes and vanilla knives in the game: %d, in no pool: %d",
             len(game), len(extra))
    return [result("Knife and glove finishes of the game in a pool", len(game),
                   len(listed), len(game & listed),
                   [f"{s} in no pool" for s in extra])]


def main() -> None:
    p = argparse.ArgumentParser(description="Checks the item list against Valve's game files.")
    p.add_argument("--commit", default=C.ITEM_CHECK_COMMIT,
                   help="Commit of GameTracking-CS2 (default: ITEM_CHECK_COMMIT in "
                        "config.py); an empty string uses the last commit before "
                        "ITEM_CHECK_UNTIL.")
    args = p.parse_args()

    setup_logging()
    C.OUT_TABLES.mkdir(parents=True, exist_ok=True)
    il = ItemList()
    log.info("Item list: %d items, %d cases (ByMykel files of %s UTC)",
             len(il.items), len(il.cases),
             pd.Timestamp(C.ITEM_CHECK_UNTIL).strftime("%Y-%m-%d %H:%M"))

    if args.commit:
        sha, date = args.commit, "given on the command line or in config"
    else:
        sha, date = resolve_commit(C.ITEM_CHECK_UNTIL)
    log.info("Game files: %s, commit %s (%s)", REPO, sha, date)
    log.info("  last commit before %s", C.ITEM_CHECK_UNTIL)
    files = load_game_files(sha)
    g = GameFile(files["items_game"], files["english"], files["file_list"])
    log.info("Parsed: %d items, %d loot lists, %d item sets, %d paint kits, "
             "%d localization tokens, %d files in the file list", len(g.items),
             len(g.loot), len(g.sets), len(g.kits), len(g.loc),
             len(files["file_list"].splitlines()))

    rows = (check_containers(g, il) + check_pool_type(g, il)
            + check_case_coverts(g, il) + check_rarity(g, il)
            + check_placebos(g, il) + check_floats(g, il)
            + check_pool_identity(g, il) + check_names(g, il)
            + check_star_finishes(g, il) + check_star_completeness(g, il))
    out = pd.DataFrame(rows)
    out.to_csv(C.OUT_TABLES / "check_item_list.csv", index=False, encoding="utf-8")

    section("SUMMARY")
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        for line in out.drop(columns="detail").to_string(index=False).splitlines():
            log.info(line)
    log.info("-> output/tables/check_item_list.csv")
    log.info("")
    log.info("Done.")


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: MIT
"""Synthetic staff ledgers with exact, programmatically verified answers.

Each world is one ledger about eight live staff members. Every member has a
team, a locker number, a badge code, and a mentor: 32 live facts. Four of the
facts are corrected later in the ledger, so the live value is the last one
stated. Four archived members and unrelated filler lines are distractors. The
rendered source has exactly ``source_tokens`` tokens under the pinned
tokenizer. Eight questions per world cover four families: direct fact,
updated fact, linked facts (two records), and exact identifier.

Pools share no full names, badge codes, or source digests. Worlds in the
held-out surface stratum use a second set of record and question templates.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import re
import string
from pathlib import Path
from typing import Any, Iterable

DATA_SEED = 20261025
DATASET_VERSION = 3
SOURCE_TOKENS = 1024
LIVE_STAFF = 8
ARCHIVED_STAFF = 4
UPDATES_PER_WORLD = 4
QUESTIONS_PER_WORLD = 8
FAMILIES = ("direct", "updated", "linked", "identifier")
ATTRIBUTES = ("team", "locker", "badge", "mentor")
ATTRIBUTE_LABEL = {
    "team": "team",
    "locker": "locker number",
    "badge": "badge code",
    "mentor": "mentor",
}

POOLS = {
    "calibration": 64,
    "tiny": 32,
    "train": 2048,
    "validation": 128,
    "test_ordinary": 128,
    "test_shifted": 128,
    "diagnostic": 16,
}
POOL_ORDER = tuple(POOLS)
POOL_TEMPLATE = {pool: ("B" if pool == "test_shifted" else "A") for pool in POOLS}

FIRST_NAMES = [
    "Ansel",
    "Brielle",
    "Cato",
    "Dahlia",
    "Emeric",
    "Faye",
    "Garrick",
    "Hollis",
    "Imara",
    "Jasper",
    "Kiera",
    "Lucan",
    "Maren",
    "Nico",
    "Orla",
    "Pascal",
    "Quill",
    "Rosalind",
    "Soren",
    "Talia",
    "Ulric",
    "Vesper",
    "Wren",
    "Xander",
    "Yara",
    "Zephyr",
    "Alden",
    "Bryony",
    "Cassius",
    "Delia",
    "Elowen",
    "Fenwick",
    "Greta",
    "Hugo",
    "Isolde",
    "Jorah",
    "Kendra",
    "Leopold",
    "Mira",
    "Nerys",
    "Oberon",
    "Perrin",
    "Quenby",
    "Rhett",
    "Sable",
    "Thea",
    "Una",
    "Viggo",
    "Willa",
    "Xenia",
    "Yusuf",
    "Zelda",
    "Amos",
    "Beatrix",
    "Corvin",
    "Dorian",
    "Esme",
    "Florian",
    "Gwen",
    "Hamish",
    "Ines",
    "Jules",
    "Kasimir",
    "Leda",
    "Magnus",
    "Nadia",
    "Osric",
    "Petra",
    "Quincy",
    "Rafferty",
    "Sybil",
    "Tobias",
    "Ursula",
    "Valen",
    "Wilhelm",
    "Ximena",
    "Yvette",
    "Zane",
    "Aurelio",
    "Bronwyn",
    "Caspian",
    "Dagny",
    "Edric",
    "Fiona",
    "Gideon",
    "Hester",
    "Ivo",
    "Juniper",
    "Kieran",
    "Lorcan",
    "Maud",
    "Niall",
    "Ottoline",
    "Piers",
    "Roderick",
    "Saskia",
]
SURNAMES = [
    "Ashford",
    "Blackwood",
    "Carraway",
    "Dunmore",
    "Ellery",
    "Fairbanks",
    "Greaves",
    "Hawthorne",
    "Ingram",
    "Jessop",
    "Kingsley",
    "Lockhart",
    "Marlowe",
    "Northam",
    "Oakes",
    "Pemberton",
    "Quintrell",
    "Ravenscroft",
    "Sinclair",
    "Thornbury",
    "Underhill",
    "Vance",
    "Whitlock",
    "Yardley",
    "Abernathy",
    "Bellamy",
    "Cromwell",
    "Darrow",
    "Everly",
    "Fairweather",
    "Galloway",
    "Harrington",
    "Ivers",
    "Jennings",
    "Kestrel",
    "Langford",
    "Montague",
    "Netherby",
    "Ormsby",
    "Prescott",
    "Radley",
    "Stanhope",
    "Tremont",
    "Upton",
    "Vickery",
    "Wexford",
    "Yates",
    "Ashcombe",
    "Brightwater",
    "Castellan",
    "Devereux",
    "Eastwick",
    "Fenwright",
    "Gilmore",
    "Holloway",
    "Innsmouth",
    "Jarrow",
    "Kimberley",
    "Lindqvist",
    "Merriweather",
    "Nightingale",
    "Oakhurst",
    "Pennington",
    "Quarrie",
    "Rutherford",
    "Stirling",
    "Tolliver",
    "Umbridge",
    "Valcourt",
    "Winterbourne",
    "Yeoman",
    "Archibald",
    "Blakely",
    "Coldwater",
    "Dalrymple",
    "Ellsworth",
    "Farrington",
    "Grimshaw",
    "Hartwell",
    "Islington",
    "Jordane",
    "Kirkpatrick",
    "Lowell",
    "Mortlake",
    "Norwood",
    "Overton",
    "Pickering",
    "Quayle",
    "Redgrave",
    "Southwell",
    "Thackeray",
    "Ulverston",
    "Vanterpool",
    "Wycliffe",
    "Youngblood",
    "Amberley",
    "Birchwood",
    "Copperfield",
    "Draycott",
    "Edgecombe",
    "Foxworth",
    "Glendower",
    "Hazelwood",
    "Iverson",
    "Jephson",
    "Kettering",
    "Ludlow",
    "Maybrook",
    "Nettleton",
    "Oxley",
    "Pinkerton",
    "Quennell",
    "Rosewood",
    "Sandringham",
    "Tillingham",
    "Uxbridge",
    "Verity",
    "Wainwright",
    "Yelverton",
    "Aldersgate",
    "Beaumont",
    "Chadwick",
    "Danvers",
    "Elphinstone",
    "Fitzroy",
    "Gresham",
    "Hathaway",
]
SURNAME_SHARE = {
    "calibration": 12,
    "tiny": 12,
    "train": 44,
    "validation": 14,
    "test_ordinary": 14,
    "test_shifted": 14,
    "diagnostic": 12,
}
TEAMS = [
    "Harbor",
    "Ridge",
    "Juniper",
    "Cobalt",
    "Meridian",
    "Saffron",
    "Granite",
    "Willow",
    "Falcon",
    "Ember",
    "Quartz",
    "Lantern",
    "Orchid",
    "Summit",
    "Tundra",
    "Velvet",
    "Beacon",
    "Cinder",
    "Delta",
    "Fjord",
    "Heron",
    "Ivory",
    "Kestrel",
    "Mosaic",
]
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

RECORD_TEMPLATES = {
    "A": {
        "team": "{name} is assigned to team {team}.",
        "locker": "{name} uses locker {locker}.",
        "badge": "{name} holds badge code {badge}.",
        "mentor": "{name} is mentored by {mentor}.",
        "update": "Correction: {name}'s {label} changed from {old} to {new}.",
        "archived": (
            "Archived entry: {name} was on team {team}, locker {locker}, "
            "badge {badge}."
        ),
        "header": "Staff ledger, site {site}.",
    },
    "B": {
        "team": "Team assignment for {name}: {team}.",
        "locker": "Locker {locker} is registered to {name}.",
        "badge": "Badge {badge} belongs to {name}.",
        "mentor": "{mentor} mentors {name}.",
        "update": "Update notice for {name}: {label} is now {new} (previously {old}).",
        "archived": "Former staff {name} (team {team}, locker {locker}, badge {badge}).",
        "header": "Site {site} personnel register.",
    },
}

QUESTION_TEMPLATES = {
    "A": {
        "team": "What team is {name} on?",
        "locker": "What is {name}'s locker number?",
        "badge": "What is {name}'s badge code?",
        "mentor": "Who mentors {name}?",
        "updated": "What is {name}'s current {label}?",
        "linked_team": "Who mentors {name}, and what team is that mentor on?",
        "linked_locker": "Who mentors {name}, and what is that mentor's locker number?",
        "linked_mentor": "Who mentors {name}, and who mentors that mentor?",
    },
    "B": {
        "team": "Which team has {name} been assigned to?",
        "locker": "Which locker does {name} use?",
        "badge": "Give the badge code held by {name}.",
        "mentor": "Name the mentor of {name}.",
        "updated": "After all corrections, what is {name}'s {label}?",
        "linked_team": "Name the mentor of {name} and the team that mentor is on.",
        "linked_locker": "Name the mentor of {name} and the locker that mentor uses.",
        "linked_mentor": "Name the mentor of {name} and that mentor's own mentor.",
    },
    "C": {
        "team": "According to the ledger, which team is {name} assigned to?",
        "locker": "Locker number for {name}?",
        "badge": "Badge code for {name}?",
        "mentor": "{name}'s mentor is whom?",
        "updated": (
            "What does the ledger list as {name}'s {label} after the correction?"
        ),
        "linked_team": "Give {name}'s mentor and the team of that mentor.",
        "linked_locker": "Give {name}'s mentor and that mentor's locker number.",
        "linked_mentor": "Give {name}'s mentor and the mentor of that mentor.",
        "linked_badge": "Give {name}'s mentor and that mentor's badge code.",
    },
}

FILLER_TEMPLATES = {
    "A": [
        "The east stairwell lights were replaced on day {n1}.",
        "Rainfall for week {n1} measured {n2} millimeters.",
        "Boiler pressure read {n2} kilopascals at the {n3} o'clock check.",
        "Visitor parking was repainted over {n3} days in spring.",
        "The cafeteria served {n2} lunches on day {n1}.",
        "Window {n1} on the north face was resealed.",
        "Delivery bay {n3} received {n2} crates of paper.",
        "The generator test ran for {n3} minutes without fault.",
        "Corridor {n1} carpet was cleaned twice this quarter.",
        "The roof drain near column {n1} was cleared of leaves.",
        "Humidity in the archive room stayed near {n2} percent.",
        "The elevator inspection certificate was renewed on day {n1}.",
        "A total of {n2} chairs were moved to the lower hall.",
        "Fire drill {n3} finished in {n1} minutes.",
        "The loading ramp was salted {n3} times during the cold week.",
        "Meter {n1} recorded {n2} kilowatt hours.",
        "The courtyard hedge was trimmed on day {n1}.",
        "Supply order {n2} contained {n3} boxes of toner.",
        "Thermostat {n3} was recalibrated to {n1} degrees.",
        "The bicycle rack holds {n2} bicycles after the extension.",
    ],
    "B": [
        "Facility log: stairwell lamp swap completed, day {n1}.",
        "Weather note, week {n1}: {n2} mm of rain.",
        "Boiler check at {n3}:00 showed {n2} kPa.",
        "Parking lines repainted; {n3} working days.",
        "Kitchen count for day {n1}: {n2} meals.",
        "North window {n1} sealed again.",
        "Bay {n3} intake: {n2} paper crates.",
        "Backup generator ran {n3} minutes, no faults.",
        "Carpet in corridor {n1} cleaned, second pass.",
        "Roof drain by column {n1} cleared.",
        "Archive humidity held at about {n2} percent.",
        "Lift certificate renewed, day {n1}.",
        "Moved {n2} chairs to the lower hall.",
        "Drill {n3} completed in {n1} minutes.",
        "Ramp salted {n3} times in the cold spell.",
        "Meter {n1}: {n2} kWh.",
        "Hedge trimming done, day {n1}.",
        "Order {n2}: {n3} toner boxes.",
        "Thermostat {n3} set to {n1} degrees.",
        "Bike rack capacity now {n2}.",
    ],
}
FILLER_PAD_WORDS = [" again", " today", " early", " as logged", " on site", " later"]


def stable_seed(*parts: Any) -> int:
    text = ":".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclasses.dataclass
class Record:
    rid: str
    kind: str
    text: str
    fields: dict[str, Any]
    token_start: int = -1
    quartile: int = -1


@dataclasses.dataclass
class Question:
    qid: str
    family: str
    text: str
    answer: str
    support: list[str]
    attribute: str
    entity: str
    quartile: int
    template: str


@dataclasses.dataclass
class World:
    world_id: str
    pool: str
    stratum: str
    template_set: str
    source_text: str
    source_ids: list[int]
    source_digest: str
    n_tokens: int
    filler_fraction: float
    live: dict[str, dict[str, Any]]
    updates: list[dict[str, Any]]
    archived: list[dict[str, Any]]
    records: list[dict[str, Any]]
    questions: list[dict[str, Any]]
    fit_questions: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class NameRegistry:
    """Names unique within a world and disjoint across pools.

    Each pool owns a disjoint block of surnames, so no full name can appear in
    two pools. Inside one world every first name starts with a different
    letter and every surname is distinct. Badge codes are unique globally.
    """

    def __init__(self) -> None:
        self.codes: set[str] = set()
        self.surnames: dict[str, list[str]] = {}
        cursor = 0
        for pool in POOL_ORDER:
            share = SURNAME_SHARE[pool]
            self.surnames[pool] = SURNAMES[cursor : cursor + share]
            cursor += share
        if cursor > len(SURNAMES):
            raise RuntimeError("not enough surnames for the pool shares")
        self.names: set[str] = set()

    def world_names(self, rng: random.Random, pool: str, count: int) -> list[str]:
        firsts: list[str] = []
        initials: set[str] = set()
        for name in rng.sample(FIRST_NAMES, len(FIRST_NAMES)):
            if name[0] not in initials:
                firsts.append(name)
                initials.add(name[0])
            if len(firsts) == count:
                break
        lasts = rng.sample(self.surnames[pool], count)
        names = [f"{a} {b}" for a, b in zip(firsts, lasts)]
        self.names.update(names)
        return names

    def new_code(self, rng: random.Random) -> str:
        for _ in range(10_000):
            head = "".join(rng.choice(CODE_ALPHABET) for _ in range(3))
            tail = "".join(rng.choice(CODE_ALPHABET) for _ in range(4))
            code = f"{head}-{tail}"
            if code not in self.codes:
                self.codes.add(code)
                return code
        raise RuntimeError("code space exhausted")


def normalize_answer(text: str, attribute: str) -> str:
    value = text.strip().lower()
    value = value.replace("’", "'")
    value = value.strip("\"'` ")
    value = re.sub(r"[.!?]+$", "", value).strip()
    prefixes = {
        "team": ("team ",),
        "locker": ("locker number ", "locker "),
        "badge": ("badge code ", "badge "),
        "mentor": (),
    }.get(attribute, ())
    for prefix in prefixes:
        if value.startswith(prefix):
            value = value[len(prefix) :].strip()
    value = re.sub(r"\s+", " ", value)
    return value


def _split_pair(text: str) -> list[str]:
    parts = re.split(r"\s*[,;]\s*", text.strip(), maxsplit=1)
    return parts


def exact_match(prediction: str, gold: str, attribute: str) -> bool:
    """Exact match after normalization. A linked gold answer has two parts,
    "mentor, value"; both must match in order."""
    if "," in gold:
        gold_parts = _split_pair(gold)
        pred_parts = _split_pair(prediction)
        if len(pred_parts) != 2 or len(gold_parts) != 2:
            return False
        return normalize_answer(pred_parts[0], "mentor") == normalize_answer(
            gold_parts[0], "mentor"
        ) and normalize_answer(pred_parts[1], attribute) == normalize_answer(
            gold_parts[1], attribute
        )
    return normalize_answer(prediction, attribute) == normalize_answer(gold, attribute)


class Tokenizer:
    """Thin wrapper so the generator can run with any tokenizer interface."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def offsets(self, text: str) -> list[tuple[int, int]]:
        out = self.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
        return [tuple(pair) for pair in out["offset_mapping"]]

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)


def _build_world_content(
    rng: random.Random, registry: NameRegistry, pool: str
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    names = registry.world_names(rng, pool, LIVE_STAFF + ARCHIVED_STAFF)
    live_names, archived_names = names[:LIVE_STAFF], names[LIVE_STAFF:]
    lockers = rng.sample(
        range(100, 1000), LIVE_STAFF + ARCHIVED_STAFF + UPDATES_PER_WORLD
    )
    live: dict[str, dict[str, Any]] = {}
    # mentor graph: a random derangement over live staff with no self
    # mentoring and no two-cycles, so "the mentor of X's mentor" is never X
    mentors = live_names[:]
    while True:
        rng.shuffle(mentors)
        mentor_of = dict(zip(live_names, mentors))
        if all(m != n and mentor_of[m] != n for n, m in mentor_of.items()):
            break
    for idx, name in enumerate(live_names):
        live[name] = {
            "team": rng.choice(TEAMS),
            "locker": str(lockers[idx]),
            "badge": registry.new_code(rng),
            "mentor": mentors[idx],
        }
    archived = []
    for idx, name in enumerate(archived_names):
        archived.append(
            {
                "name": name,
                "team": rng.choice(TEAMS),
                "locker": str(lockers[LIVE_STAFF + idx]),
                "badge": registry.new_code(rng),
            }
        )
    # updates: four distinct (entity, attribute) pairs over team/locker/mentor
    candidates = [(n, a) for n in live_names for a in ("team", "locker", "mentor")]
    rng.shuffle(candidates)
    updates = []
    used_entities: set[str] = set()
    spare_lockers = lockers[LIVE_STAFF + ARCHIVED_STAFF :]
    for name, attribute in candidates:
        if len(updates) == UPDATES_PER_WORLD:
            break
        if name in used_entities:
            continue
        new_value = live[name][attribute]
        if attribute == "team":
            old_value = rng.choice([t for t in TEAMS if t != new_value])
        elif attribute == "locker":
            old_value = str(spare_lockers[len(updates)])
        else:
            others = [n for n in live_names if n not in (name, new_value)]
            old_value = rng.choice(others)
        used_entities.add(name)
        updates.append(
            {"name": name, "attribute": attribute, "old": old_value, "new": new_value}
        )
    assert len(updates) == UPDATES_PER_WORLD
    return live, updates, archived


def _render_records(
    rng: random.Random,
    live: dict[str, dict[str, Any]],
    updates: list[dict[str, Any]],
    archived: list[dict[str, Any]],
    template_set: str,
) -> list[Record]:
    tpl = RECORD_TEMPLATES[template_set]
    updated_keys = {(u["name"], u["attribute"]): u for u in updates}
    records: list[Record] = []
    for name, facts in live.items():
        for attribute in ATTRIBUTES:
            update = updated_keys.get((name, attribute))
            shown = dict(facts)
            if update is not None:
                shown[attribute] = update["old"]
            text = tpl[attribute].format(name=name, **shown)
            records.append(
                Record(
                    rid=f"live:{name}:{attribute}",
                    kind="live",
                    text=text,
                    fields={
                        "name": name,
                        "attribute": attribute,
                        "value": shown[attribute],
                    },
                )
            )
    for update in updates:
        text = tpl["update"].format(
            name=update["name"],
            label=ATTRIBUTE_LABEL[update["attribute"]],
            old=update["old"],
            new=update["new"],
        )
        records.append(
            Record(
                rid=f"update:{update['name']}:{update['attribute']}",
                kind="update",
                text=text,
                fields=dict(update),
            )
        )
    for entry in archived:
        records.append(
            Record(
                rid=f"archived:{entry['name']}",
                kind="archived",
                text=tpl["archived"].format(**entry),
                fields=dict(entry),
            )
        )
    rng.shuffle(records)
    # event order: every correction must follow the record it corrects
    order = records[:]
    for update in updates:
        base_id = f"live:{update['name']}:{update['attribute']}"
        upd_id = f"update:{update['name']}:{update['attribute']}"
        base_idx = next(i for i, r in enumerate(order) if r.rid == base_id)
        upd_idx = next(i for i, r in enumerate(order) if r.rid == upd_id)
        if upd_idx < base_idx:
            item = order.pop(upd_idx)
            base_idx = next(i for i, r in enumerate(order) if r.rid == base_id)
            insert_at = rng.randint(base_idx + 1, len(order))
            order.insert(insert_at, item)
    return order


def _filler_sentence(rng: random.Random, template_set: str) -> str:
    template = rng.choice(FILLER_TEMPLATES[template_set])
    return template.format(
        n1=rng.randint(2, 60), n2=rng.randint(10, 990), n3=rng.randint(1, 12)
    )


def _assemble(
    header: str, blocks: list[tuple[str, list[str]]], filler_tail: list[str]
) -> str:
    lines = [header]
    for text, fillers in blocks:
        lines.extend(fillers)
        lines.append(text)
    lines.extend(filler_tail)
    return "\n".join(lines)


def render_source(
    rng: random.Random,
    tok: Tokenizer,
    records: list[Record],
    template_set: str,
    site: str,
    target_tokens: int = SOURCE_TOKENS,
) -> tuple[str, list[int], float]:
    """Interleave filler lines until the source is exactly ``target_tokens``."""
    header = RECORD_TEMPLATES[template_set]["header"].format(site=site)
    blocks: list[tuple[str, list[str]]] = [(r.text, []) for r in records]
    tail: list[str] = []
    content_tokens = len(tok.encode(_assemble(header, blocks, tail)))
    if content_tokens > target_tokens - 48:
        raise ValueError(
            f"content uses {content_tokens} tokens; too little filler room"
        )
    # coarse fill: add whole sentences to random gaps until close to the target
    while True:
        text = _assemble(header, blocks, tail)
        count = len(tok.encode(text))
        if count >= target_tokens - 10:
            break
        sentence = _filler_sentence(rng, template_set)
        slot = rng.randint(0, len(blocks))
        if slot == len(blocks):
            tail.append(sentence)
        else:
            blocks[slot][1].append(sentence)

    # fine fill: whole sentences while the deficit is large, then pad the
    # last filler line word by word to land exactly on the target
    def last_filler() -> list[str] | None:
        if tail:
            return tail
        for slot in range(len(blocks) - 1, -1, -1):
            if blocks[slot][1]:
                return blocks[slot][1]
        return None

    for _ in range(600):
        text = _assemble(header, blocks, tail)
        count = len(tok.encode(text))
        if count == target_tokens:
            break
        if count > target_tokens:
            lines = last_filler()
            if lines is None:
                raise ValueError("content alone exceeds the token target")
            lines.pop()
            continue
        deficit = target_tokens - count
        if deficit >= 12:
            tail.append(_filler_sentence(rng, template_set))
            continue
        lines = last_filler()
        if lines is None:
            tail.append(_filler_sentence(rng, template_set))
            continue
        placed = False
        for word in rng.sample(FILLER_PAD_WORDS, len(FILLER_PAD_WORDS)):
            original = lines[-1]
            lines[-1] = original.rstrip(".") + word + "."
            trial_count = len(tok.encode(_assemble(header, blocks, tail)))
            if count < trial_count <= target_tokens:
                placed = True
                break
            lines[-1] = original
        if not placed:
            tail.append(_filler_sentence(rng, template_set))
    text = _assemble(header, blocks, tail)
    ids = tok.encode(text)
    if len(ids) != target_tokens:
        raise RuntimeError(f"could not land on {target_tokens} tokens: {len(ids)}")
    if tok.decode(ids) != text:
        raise RuntimeError("tokenization is not lossless for this source")
    filler_lines = sum(len(f) for _, f in blocks) + len(tail)
    filler_text = "\n".join([s for _, f in blocks for s in f] + tail)
    filler_tokens = len(tok.encode(filler_text)) if filler_lines else 0
    return text, ids, filler_tokens / target_tokens


def _locate_records(tok: Tokenizer, text: str, records: list[Record]) -> None:
    offsets = tok.offsets(text)
    starts = [s for s, _ in offsets]
    cursor = 0
    n = len(offsets)
    for record in records:
        char_start = text.index(record.text, cursor)
        cursor = char_start + len(record.text)
        token_index = 0
        while token_index + 1 < n and starts[token_index + 1] <= char_start:
            token_index += 1
        record.token_start = token_index
        record.quartile = min(3, (4 * token_index) // n)


def _question(
    qid: str,
    family: str,
    template_set: str,
    key: str,
    answer: str,
    support: list[str],
    attribute: str,
    entity: str,
    quartile: int,
    **fmt: Any,
) -> Question:
    text = QUESTION_TEMPLATES[template_set][key].format(**fmt)
    return Question(
        qid=qid,
        family=family,
        text=text,
        answer=answer,
        support=support,
        attribute=attribute,
        entity=entity,
        quartile=quartile,
        template=template_set,
    )


def _candidates(
    live: dict[str, dict[str, Any]],
    updates: list[dict[str, Any]],
    quartiles: dict[str, int],
    template_set: str,
    prefix: str,
) -> dict[str, list[Question]]:
    updated = {(u["name"], u["attribute"]) for u in updates}
    out: dict[str, list[Question]] = {f: [] for f in FAMILIES}
    for name, facts in live.items():
        for attribute in ("team", "locker", "mentor"):
            if (name, attribute) in updated:
                continue
            rid = f"live:{name}:{attribute}"
            out["direct"].append(
                _question(
                    f"{prefix}direct:{name}:{attribute}",
                    "direct",
                    template_set,
                    attribute,
                    facts[attribute],
                    [rid],
                    attribute,
                    name,
                    quartiles[rid],
                    name=name,
                )
            )
        rid = f"live:{name}:badge"
        out["identifier"].append(
            _question(
                f"{prefix}identifier:{name}",
                "identifier",
                template_set,
                "badge",
                facts["badge"],
                [rid],
                "badge",
                name,
                quartiles[rid],
                name=name,
            )
        )
        mentor = facts["mentor"]
        if (name, "mentor") not in updated:
            for attribute, key in (
                ("team", "linked_team"),
                ("locker", "linked_locker"),
                ("mentor", "linked_mentor"),
            ):
                if (mentor, attribute) in updated:
                    continue
                rids = [f"live:{name}:mentor", f"live:{mentor}:{attribute}"]
                out["linked"].append(
                    _question(
                        f"{prefix}linked:{name}:{attribute}",
                        "linked",
                        template_set,
                        key,
                        f"{mentor}, {live[mentor][attribute]}",
                        rids,
                        attribute,
                        name,
                        quartiles[rids[1]],
                        name=name,
                    )
                )
    for update in updates:
        name, attribute = update["name"], update["attribute"]
        rids = [f"live:{name}:{attribute}", f"update:{name}:{attribute}"]
        out["updated"].append(
            _question(
                f"{prefix}updated:{name}:{attribute}",
                "updated",
                template_set,
                "updated",
                update["new"],
                rids,
                attribute,
                name,
                quartiles[rids[1]],
                name=name,
                label=ATTRIBUTE_LABEL[attribute],
            )
        )
    return out


def _select_balanced(
    rng: random.Random, candidates: dict[str, list[Question]], per_family: int
) -> list[Question]:
    """Two questions per family, preferring under-covered source quartiles."""
    coverage = [0, 0, 0, 0]
    chosen: list[Question] = []
    used_entities: set[str] = set()
    for family in FAMILIES:
        pool = candidates[family][:]
        rng.shuffle(pool)
        for _ in range(per_family):
            remaining = [q for q in pool if q not in chosen]
            fresh = [q for q in remaining if q.entity not in used_entities]
            options = fresh or remaining
            if not options:
                raise RuntimeError(f"not enough {family} candidates")
            # least covered source quartile first; random tie break
            best = min(options, key=lambda q: (coverage[q.quartile], rng.random()))
            chosen.append(best)
            used_entities.add(best.entity)
            coverage[best.quartile] += 1
    return chosen


def _diagnostic_questions(
    rng: random.Random,
    live: dict[str, dict[str, Any]],
    updates: list[dict[str, Any]],
    quartiles: dict[str, int],
) -> tuple[list[Question], list[Question]]:
    """32 fit questions (8 per family, template sets A and B) and 8 held-out
    evaluation questions in template set C with a new linked combination."""
    fit: list[Question] = []
    cand_a = _candidates(live, updates, quartiles, "A", "fit:A:")
    cand_b = _candidates(live, updates, quartiles, "B", "fit:B:")
    # direct: 8 distinct entities, template A
    by_entity: dict[str, list[Question]] = {}
    for q in cand_a["direct"]:
        by_entity.setdefault(q.entity, []).append(q)
    for name in sorted(by_entity):
        fit.append(rng.choice(by_entity[name]))
    while len([q for q in fit if q.family == "direct"]) < 8:
        q = rng.choice(cand_a["direct"])
        if q not in fit:
            fit.append(q)
    # updated: four facts x two phrasings
    fit.extend(cand_a["updated"])
    fit.extend(cand_b["updated"])
    # linked: eight with template A (template B tops up rare short worlds)
    linked = cand_a["linked"][:]
    rng.shuffle(linked)
    picked: list[Question] = []
    seen: set[str] = set()
    for q in linked:
        if q.entity not in seen:
            picked.append(q)
            seen.add(q.entity)
    for q in linked + cand_b["linked"]:
        if len(picked) == 8:
            break
        if q not in picked and q.text not in {p.text for p in picked}:
            picked.append(q)
    if len(picked) < 8:
        raise RuntimeError("not enough linked fit candidates")
    fit.extend(picked)
    # identifier: all eight badges, template A
    fit.extend(cand_a["identifier"])
    assert len(fit) == 32, len(fit)
    cand_c = _candidates(live, updates, quartiles, "C", "eval:C:")
    linked_badge = []
    updated_mentor = {u["name"] for u in updates if u["attribute"] == "mentor"}
    for name, facts in live.items():
        if name in updated_mentor:
            continue
        mentor = facts["mentor"]
        rids = [f"live:{name}:mentor", f"live:{mentor}:badge"]
        linked_badge.append(
            _question(
                f"eval:C:linked_badge:{name}",
                "linked",
                "C",
                "linked_badge",
                f"{mentor}, {live[mentor]['badge']}",
                rids,
                "badge",
                name,
                quartiles[rids[1]],
                name=name,
            )
        )
    cand_c["linked"] = linked_badge + cand_c["linked"]
    # an evaluation question may depend only on records the fit questions
    # supervise, so changing an unseen fact cannot move the evaluation
    covered = fit_support(fit)
    cand_c = {
        family: [q for q in qs if set(q.support) <= covered]
        for family, qs in cand_c.items()
    }
    evaluation = _select_balanced(rng, cand_c, 2)
    fit_texts = {q.text for q in fit}
    for q in evaluation:
        if q.text in fit_texts:
            raise RuntimeError("evaluation question duplicates a fit question")
    uncovered = uncovered_support(fit, evaluation)
    if uncovered:
        raise RuntimeError(f"evaluation depends on unsupervised records {uncovered}")
    return fit, evaluation


def fit_support(questions: Iterable[Any]) -> set[str]:
    """Record ids whose values the questions' answers depend on."""
    out: set[str] = set()
    for q in questions:
        support = q.support if hasattr(q, "support") else q["support"]
        out.update(support)
    return out


def uncovered_support(
    fit: Iterable[Any], evaluation: Iterable[Any]
) -> dict[str, list[str]]:
    """Evaluation questions whose support is not fully covered by the fit
    questions, keyed by question id with the missing record ids."""
    covered = fit_support(fit)
    out: dict[str, list[str]] = {}
    for q in evaluation:
        qid = q.qid if hasattr(q, "qid") else q["qid"]
        support = q.support if hasattr(q, "support") else q["support"]
        missing = sorted(set(support) - covered)
        if missing:
            out[qid] = missing
    return out


def generate_world(
    world_id: str,
    pool: str,
    index: int,
    tok: Tokenizer,
    registry: NameRegistry,
    data_seed: int = DATA_SEED,
    source_tokens: int = SOURCE_TOKENS,
) -> World:
    rng = random.Random(stable_seed(data_seed, pool, index))
    template_set = POOL_TEMPLATE[pool]
    live, updates, archived = _build_world_content(rng, registry, pool)
    records = _render_records(rng, live, updates, archived, template_set)
    site = "".join(rng.choice(string.ascii_uppercase) for _ in range(2)) + str(
        rng.randint(10, 99)
    )
    text, ids, filler_fraction = render_source(
        rng, tok, records, template_set, site, source_tokens
    )
    _locate_records(tok, text, records)
    quartiles = {r.rid: r.quartile for r in records}
    candidates = _candidates(live, updates, quartiles, template_set, "")
    questions = _select_balanced(rng, candidates, QUESTIONS_PER_WORLD // len(FAMILIES))
    fit_questions: list[Question] = []
    if pool == "diagnostic":
        fit_questions, questions = _diagnostic_questions(rng, live, updates, quartiles)
    for q in questions + fit_questions:
        _verify_answer(q, live, updates)
    stratum = "shifted" if template_set == "B" else "ordinary"
    return World(
        world_id=world_id,
        pool=pool,
        stratum=stratum,
        template_set=template_set,
        source_text=text,
        source_ids=ids,
        source_digest=sha256_text(text),
        n_tokens=len(ids),
        filler_fraction=filler_fraction,
        live=live,
        updates=updates,
        archived=archived,
        records=[dataclasses.asdict(r) for r in records],
        questions=[dataclasses.asdict(q) for q in questions],
        fit_questions=[dataclasses.asdict(q) for q in fit_questions],
    )


def _verify_answer(
    q: Question, live: dict[str, dict[str, Any]], updates: list[dict]
) -> None:
    """Recompute the answer from the structured facts; the live table holds
    final values, so corrected facts resolve to the latest value."""
    facts = live[q.entity]
    if q.family in ("direct", "identifier", "updated"):
        expected = facts[q.attribute]
    else:
        expected = f"{facts['mentor']}, {live[facts['mentor']][q.attribute]}"
    if expected != q.answer:
        raise RuntimeError(f"answer mismatch for {q.qid}: {expected} != {q.answer}")
    if q.family == "updated":
        assert any(
            u["name"] == q.entity and u["attribute"] == q.attribute for u in updates
        )


def generate_pool(
    pool: str, tok: Tokenizer, registry: NameRegistry, count: int | None = None
) -> list[World]:
    count = POOLS[pool] if count is None else count
    return [
        generate_world(f"{pool}-{i:04d}", pool, i, tok, registry) for i in range(count)
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            line = canonical_json(row) + "\n"
            stream.write(line)
            digest.update(line.encode())
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def questions_only(
    worlds: list[dict[str, Any]], key: str = "questions"
) -> list[dict[str, Any]]:
    """Question rows with no answers, support records, or source: the only
    task input a fresh reader-only consumer may receive."""
    rows = []
    for world in worlds:
        for q in world[key]:
            rows.append(
                {
                    "world_id": world["world_id"],
                    "qid": q["qid"],
                    "family": q["family"],
                    "attribute": q["attribute"],
                    "text": q["text"],
                }
            )
    return rows


def validation_panel(
    worlds: list[dict[str, Any]], count: int = 64
) -> list[tuple[str, str]]:
    """Fixed 64 worlds x 4 questions, one per family, from the validation pool."""
    rng = random.Random(stable_seed(DATA_SEED, "validation-panel"))
    chosen = sorted(rng.sample(range(len(worlds)), count))
    panel = []
    for idx in chosen:
        world = worlds[idx]
        for family in FAMILIES:
            options = [q for q in world["questions"] if q["family"] == family]
            panel.append((world["world_id"], rng.choice(options)["qid"]))
    return panel


def wrong_world_differs(world: dict[str, Any], other: dict[str, Any]) -> bool:
    """A wrong cartridge must come from a world that cannot supply any of this
    world's answers: no shared staff names, and no question answer present
    among the other world's live values for the same attribute."""
    if set(world["live"]) & set(other["live"]):
        return False
    other_values: dict[str, set[str]] = {a: set() for a in ATTRIBUTES}
    for facts in other["live"].values():
        for attribute in ATTRIBUTES:
            other_values[attribute].add(facts[attribute])
    for q in world["questions"]:
        if q["answer"] in other_values[q["attribute"]]:
            return False
    return True


def derangement(
    worlds: list[dict[str, Any]], seed_tag: str, strict: bool = True
) -> dict[str, str]:
    """Fixed wrong-cartridge mapping: every world receives the cartridge of a
    different world whose answers differ (``wrong_world_differs``). Solved as
    a bipartite matching so the mapping is a permutation with no fixed point.
    ``strict=False`` only requires disjoint staff names (small smoke pools)."""
    rng = random.Random(stable_seed(DATA_SEED, "derangement", seed_tag))
    n = len(worlds)
    compatible = [
        [
            j
            for j in range(n)
            if j != i
            and (
                wrong_world_differs(worlds[i], worlds[j])
                if strict
                else not (set(worlds[i]["live"]) & set(worlds[j]["live"]))
            )
        ]
        for i in range(n)
    ]
    for row in compatible:
        rng.shuffle(row)
    match_of: list[int] = [-1] * n  # donor j -> receiver i

    def assign(i: int, seen: set[int]) -> bool:
        for j in compatible[i]:
            if j in seen:
                continue
            seen.add(j)
            if match_of[j] == -1 or assign(match_of[j], seen):
                match_of[j] = i
                return True
        return False

    for i in range(n):
        if not assign(i, set()):
            raise RuntimeError(
                f"no compatible wrong-world donor for {worlds[i]['world_id']}"
            )
    mapping = {}
    for j, i in enumerate(match_of):
        mapping[worlds[i]["world_id"]] = worlds[j]["world_id"]
    assert all(k != v for k, v in mapping.items()) and len(mapping) == n
    return mapping


def pool_stats(worlds: list[World]) -> dict[str, Any]:
    tokens = [w.n_tokens for w in worlds]
    filler = [w.filler_fraction for w in worlds]
    quartiles = [0, 0, 0, 0]
    families: dict[str, int] = {}
    for w in worlds:
        for q in w.questions:
            quartiles[q["quartile"]] += 1
            families[q["family"]] = families.get(q["family"], 0) + 1
    return {
        "worlds": len(worlds),
        "min_tokens": min(tokens),
        "max_tokens": max(tokens),
        "filler_fraction_mean": sum(filler) / len(filler),
        "filler_fraction_min": min(filler),
        "filler_fraction_max": max(filler),
        "question_quartiles": quartiles,
        "question_families": families,
    }

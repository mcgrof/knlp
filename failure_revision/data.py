"""GSM8K-Platinum loading, duplicate grouping and hash-seeded splits.

GSM8K-Platinum (madrylab/gsm8k-platinum) is the GSM8K test set with
label errors corrected and ambiguous questions removed.  Items are split
once, before any screening, by a seeded hash of their duplicate group so
near-identical questions never straddle two splits:

* ``dev`` -- the first ``n_dev`` items; the first 16 are for bring-up.
* ``pilot`` -- a ``pilot_fraction`` share of the remainder, screened to
  find observed-zero items for the pilot.
* ``confirm`` -- the rest, untouched until a frozen confirmation run.
"""

from __future__ import annotations

import hashlib
import re

from .grader import parse_gold

DATASET = "madrylab/gsm8k-platinum"
DATASET_CONFIG = "main"
DATASET_SPLIT = "test"


def normalize_question(q: str) -> str:
    q = q.lower()
    q = re.sub(r"[^a-z0-9$%.]+", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def question_hash(q: str) -> str:
    return hashlib.sha256(normalize_question(q).encode()).hexdigest()[:16]


def load_items(revision: str | None = None) -> tuple[list[dict], dict]:
    """Return (items, info).  Items carry id, question, gold (as str)."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, DATASET_CONFIG, split=DATASET_SPLIT, revision=revision)
    items, dropped = [], []
    seen: dict[str, int] = {}
    for idx, row in enumerate(ds):
        status = row.get("cleaning_status")
        if status is not None and str(status).lower() == "rejected":
            dropped.append(idx)
            continue
        gold = parse_gold(row["answer"])
        qid = question_hash(row["question"])
        n = seen.get(qid, 0)
        seen[qid] = n + 1
        items.append(
            {
                "id": qid if n == 0 else f"{qid}-{n}",
                "group": qid,
                "row": idx,
                "question": row["question"],
                "gold": f"{gold.numerator}/{gold.denominator}",
                "reference_solution": row["answer"].split("####")[0],
            }
        )
    info = {
        "rows": len(ds),
        "rejected_dropped": len(dropped),
        "items": len(items),
        "columns": list(ds.column_names),
    }
    return items, info


def make_splits(
    items: list[dict], seed: int, n_dev: int = 32, pilot_fraction: float = 0.6
) -> dict:
    groups: dict[str, list[str]] = {}
    for it in items:
        groups.setdefault(it["group"], []).append(it["id"])
    order = sorted(
        groups, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest()
    )
    dev: list[str] = []
    rest: list[str] = []
    n_dev_items = 0
    for g in order:
        if n_dev_items < n_dev:
            dev.append(g)
            n_dev_items += len(groups[g])
        else:
            rest.append(g)
    n_pilot = round(len(rest) * pilot_fraction)
    pilot, confirm = rest[:n_pilot], rest[n_pilot:]

    def ids(gs: list[str]) -> list[str]:
        return [i for g in gs for i in groups[g]]

    dup_groups = {g: v for g, v in groups.items() if len(v) > 1}
    return {
        "seed": seed,
        "dev": ids(dev),
        "pilot": ids(pilot),
        "confirm": ids(confirm),
        "duplicate_groups": dup_groups,
    }

#!/usr/bin/env python3
"""Write the train, development and confirmation parquets of a frozen split.

cas_freeze_split.py fixes which Self-Study rows are held out, as a
manifest. This script turns that manifest into the files a run reads,
without recomputing membership: rows are taken by their original row
index, so repairs or filtering of the source can never move a row to
the other side.

The held-out rows are divided once more, by whole prompt group, into
a development set (held-out loss and answer-task curation) and a sealed
confirmation set that no callback, fit or checkpoint choice reads.
A group's ID is the first 16 hex digits of the SHA-256 of its
normalized prompt; the sorted IDs are shuffled with CONF_SEED and the
first CONF_GROUPS become confirmation. Rows are never split from their
group, so the confirmation row count follows from the groups chosen.

Before writing, the parquet hash and every recorded held-out and fit
row hash are checked against the manifest. After writing, the three
files are read back and checked for row counts, row hashes and zero
shared prompt groups. The receipt records, per file, the SHA-256, the
ordered original row indices, the row hashes and the group IDs.

Env: PARQUET, MANIFEST (cas_freeze_split.py JSON), OUT_DIR,
CONF_GROUPS (64), CONF_SEED (0).
"""

import hashlib
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

PARQUET = os.environ["PARQUET"]
MANIFEST = os.environ["MANIFEST"]
OUT_DIR = Path(os.environ["OUT_DIR"])
CONF_GROUPS = int(os.environ.get("CONF_GROUPS", "64"))
CONF_SEED = int(os.environ.get("CONF_SEED", "0"))


def norm(s):
    return re.sub(r"\W+", " ", s.lower()).strip()


def group_id(prompt):
    return hashlib.sha256(norm(prompt).encode()).hexdigest()[:16]


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def rows_of(msgs):
    users, shas = [], []
    for ms in msgs:
        u = next(m["content"] for m in ms if m["role"] == "user")
        a = next(m for m in ms if m["role"] == "assistant")
        h = hashlib.sha256(u.encode() + b"\0")
        h.update(np.asarray(a["token_ids"], dtype=np.int64).tobytes())
        users.append(u)
        shas.append(h.hexdigest()[:16])
    return users, shas


def main():
    man = json.loads(Path(MANIFEST).read_text())
    assert file_sha(PARQUET) == man["parquet_sha256"], "parquet hash differs"
    full = pq.read_table(PARQUET)
    users, shas = rows_of(full.column("messages").to_pylist())
    assert len(users) == man["rows"]
    for key in ("dev", "fit", "fit_check"):
        got = [shas[i] for i in man[f"{key}_rows"]]
        assert got == man[f"{key}_row_sha"], f"{key} row hashes differ"

    held = man["dev_rows"]
    heldset = set(held)
    train = [i for i in range(len(users)) if i not in heldset]
    assert (
        hashlib.sha256(",".join(map(str, train)).encode()).hexdigest()
        == man["train_rows_sha256"]
    )
    groups = {}
    for i in held:
        groups.setdefault(group_id(users[i]), []).append(i)
    assert len(groups) == man["counts"]["dev_prompt_groups"]
    assert CONF_GROUPS < len(groups)
    order = sorted(groups)
    random.Random(CONF_SEED).shuffle(order)
    conf_g = set(order[:CONF_GROUPS])
    conf = sorted(i for g in conf_g for i in groups[g])
    dev = sorted(i for g in groups if g not in conf_g for i in groups[g])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(PARQUET).stem
    parts = {"train": train, "dev": dev, "confirm": conf}
    rec = {
        "source": {
            "parquet": os.path.abspath(PARQUET),
            "sha256": man["parquet_sha256"],
            "rows": len(users),
        },
        "manifest": {"path": os.path.abspath(MANIFEST), "sha256": file_sha(MANIFEST)},
        "settings": dict(CONF_GROUPS=CONF_GROUPS, CONF_SEED=CONF_SEED),
        "files": {},
    }
    seen = {}
    for name, idx in parts.items():
        path = OUT_DIR / f"{stem}_{name}.parquet"
        pq.write_table(full.take(idx), path)
        back_users, back_shas = rows_of(
            pq.read_table(path, columns=["messages"]).column("messages").to_pylist()
        )
        assert back_shas == [shas[i] for i in idx], f"{name} read-back differs"
        gids = sorted({group_id(u) for u in back_users})
        for g in gids:
            assert seen.setdefault(g, name) == name, f"group {g} leaks into {name}"
        rec["files"][name] = {
            "path": path.name,
            "sha256": file_sha(path),
            "rows": len(idx),
            "prompt_groups": len(gids),
            "row_index": idx,
            "row_sha": back_shas,
            "group_ids": gids,
        }
    assert sum(len(v) for v in parts.values()) == len(users)
    out = OUT_DIR / f"{stem}_materialized.json"
    out.write_text(json.dumps(rec, indent=1))
    summary = {
        n: {k: f[k] for k in ("rows", "prompt_groups", "sha256")}
        for n, f in rec["files"].items()
    }
    print("MATERIALIZE " + json.dumps({"parquet": stem, **summary}), flush=True)


if __name__ == "__main__":
    main()

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

REPAIRS, when set, is a cas_teacher_repair.py receipt (its
<stem>_repairs.parquet beside it). Each row it reports complete
replaces the original row at the same index, after checking the old
row hash against the source and the new one against the repaired row;
rows it reports still capped or failed keep their original content and
are listed as unresolved. Membership is unchanged by a repair. The
repaired whole-patient file is also written as <stem>_master.parquet,
row for row in source order, for tools that index the source by row
(the fitted start's capture).

EXCLUDE, when set, is a JSON file whose rows (a list of objects with a
row index) are removed from the train and development files: an
eligibility overlay on the frozen split. Group assignment and the
confirmation file are unchanged; an exclusion that names a confirmation
row stops the run.

Env: PARQUET, MANIFEST (cas_freeze_split.py JSON), OUT_DIR,
CONF_GROUPS (64), CONF_SEED (0), REPAIRS (optional), EXCLUDE (optional).
"""

import hashlib
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PARQUET = os.environ["PARQUET"]
MANIFEST = os.environ["MANIFEST"]
OUT_DIR = Path(os.environ["OUT_DIR"])
CONF_GROUPS = int(os.environ.get("CONF_GROUPS", "64"))
CONF_SEED = int(os.environ.get("CONF_SEED", "0"))
REPAIRS = os.environ.get("REPAIRS", "")
EXCLUDE = os.environ.get("EXCLUDE", "")


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

    repaired, unresolved, rep_rec = {}, [], None
    if REPAIRS:
        rep_rec = json.loads(Path(REPAIRS).read_text())
        assert rep_rec["source_sha256"] == man["parquet_sha256"], "repair source"
        rp = Path(REPAIRS).with_name(Path(REPAIRS).name.replace(".json", ".parquet"))
        rt = pq.read_table(rp).cast(full.schema)
        _, rshas = rows_of(rt.column("messages").to_pylist())
        by_sha = {h: j for j, h in enumerate(rshas)}
        for r in rep_rec["rows"]:
            i = r["row"]
            assert shas[i] == r["old_row_sha"], f"repair row {i} source differs"
            if r["status"] == "complete":
                repaired[i] = rt.slice(by_sha[r["new_row_sha"]], 1)
            elif r["status"] in ("still_capped", "failed"):
                unresolved.append(dict(row=i, status=r["status"]))
        full = pa.concat_tables(
            [
                repaired[i] if i in repaired else full.slice(i, 1)
                for i in range(len(users))
            ]
        )
        users2, shas2 = rows_of(full.column("messages").to_pylist())
        assert users2 == users
        for i in repaired:
            assert shas2[i] != shas[i]
        shas = shas2

    excluded, train0, dev0 = [], list(train), list(dev)
    if EXCLUDE:
        ex = json.loads(Path(EXCLUDE).read_text())
        assert ex["source_sha256"] == man["parquet_sha256"], "exclusion source"
        excluded = sorted({r["row"] for r in ex["rows"]})
        confset = set(conf)
        assert not confset & set(excluded), "exclusions may not touch confirmation"
        drop = set(excluded)
        train = [i for i in train if i not in drop]
        dev = [i for i in dev if i not in drop]

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
    if EXCLUDE:
        rec["exclusions"] = {
            "file": os.path.abspath(EXCLUDE),
            "sha256": file_sha(EXCLUDE),
            "rows": excluded,
            "by_file": {
                n: sum(1 for i in excluded if i in set(idx0))
                for n, idx0 in (("train", train0), ("dev", dev0))
            },
        }
    if REPAIRS:
        master = OUT_DIR / f"{stem}_master.parquet"
        pq.write_table(full, master)
        _, back = rows_of(
            pq.read_table(master, columns=["messages"]).column("messages").to_pylist()
        )
        assert back == shas, "master read-back differs"
        rec["repairs"] = {
            "master": master.name,
            "master_sha256": file_sha(master),
            "receipt": os.path.abspath(REPAIRS),
            "receipt_sha256": file_sha(REPAIRS),
            "repaired_rows": sorted(repaired),
            "unresolved": unresolved,
            "by_file": {
                n: dict(
                    repaired=sum(1 for i in idx if i in repaired),
                    unresolved=sum(1 for u in unresolved if u["row"] in set(idx)),
                )
                for n, idx in parts.items()
            },
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
    assert sum(len(v) for v in parts.values()) + len(excluded) == len(users)
    out = OUT_DIR / f"{stem}_materialized.json"
    out.write_text(json.dumps(rec, indent=1))
    summary = {
        n: {k: f[k] for k in ("rows", "prompt_groups", "sha256")}
        for n, f in rec["files"].items()
    }
    print("MATERIALIZE " + json.dumps({"parquet": stem, **summary}), flush=True)


if __name__ == "__main__":
    main()

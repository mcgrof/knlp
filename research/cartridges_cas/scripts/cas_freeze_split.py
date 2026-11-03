#!/usr/bin/env python3
"""Freeze one patient's Self-Study rows into train, fit and dev sets.

A recipe comparison needs one held-out population that no arm trains
on, that the fitted start never saw, and that every checkpoint is
scored on. This script fixes that population as a manifest, as a
function of the parquet and the settings below only.

The fitted start (cas_value_fit.py) picks its rows by hashing each
prompt with its seed into a fit side and a check side, then draws
N_FIT and N_VAL rows from them with numpy's default_rng(seed). This
script reproduces that choice exactly from the parquet, without the
model, so existing fits stay usable: those rows remain in the training
set (they were always training rows) and are excluded from dev.

Dev is chosen by prompt group, not by row: rows sharing a normalized
prompt go to the same side, so no dev prompt appears in training.
Groups that contain a fit row are never dev. Groups are shuffled with
DEV_SEED and taken until dev holds at least DEV_N rows.

Env: PARQUET, OUT_JSON, DEV_N (256), DEV_SEED (0), FIT_SEED (0),
FIT_VAL_FRAC (0.2), N_FIT (192), N_VAL (64), WRITE_DIR (optional: also
write <stem>_train.parquet and <stem>_dev.parquet there).
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
OUT_JSON = os.environ["OUT_JSON"]
DEV_N = int(os.environ.get("DEV_N", "256"))
DEV_SEED = int(os.environ.get("DEV_SEED", "0"))
FIT_SEED = int(os.environ.get("FIT_SEED", "0"))
FIT_VAL_FRAC = float(os.environ.get("FIT_VAL_FRAC", "0.2"))
N_FIT = int(os.environ.get("N_FIT", "192"))
N_VAL = int(os.environ.get("N_VAL", "64"))


def norm(s):
    return re.sub(r"\W+", " ", s.lower()).strip()


def fit_rows(users):
    """cas_value_fit.group_split plus its seeded draw, on raw prompts."""
    fit, val = [], []
    for i, u in enumerate(users):
        h = hashlib.sha256(f"{FIT_SEED}:{u}".encode()).digest()
        (val if h[0] / 256.0 < FIT_VAL_FRAC else fit).append(i)
    rng = np.random.default_rng(FIT_SEED)
    fit = sorted(int(x) for x in rng.choice(fit, min(N_FIT, len(fit)), replace=False))
    val = sorted(int(x) for x in rng.choice(val, min(N_VAL, len(val)), replace=False))
    return fit, val


def main():
    t = pq.read_table(PARQUET, columns=["messages"])
    msgs = t.column("messages").to_pylist()
    users = [next(m["content"] for m in ms if m["role"] == "user") for ms in msgs]
    fit, fitval = fit_rows(users)
    fitset = set(fit) | set(fitval)
    groups = {}
    for i, u in enumerate(users):
        groups.setdefault(norm(u), []).append(i)
    eligible = [g for g, rows in groups.items() if not fitset & set(rows)]
    eligible.sort()
    random.Random(DEV_SEED).shuffle(eligible)
    dev = []
    for g in eligible:
        if len(dev) >= DEV_N:
            break
        dev.extend(groups[g])
    dev = sorted(dev)
    devset = set(dev)
    train = [i for i in range(len(users)) if i not in devset]
    assert not devset & fitset
    assert len(train) + len(dev) == len(users)
    dev_prompts = {norm(users[i]) for i in dev}
    assert not any(norm(users[i]) in dev_prompts for i in train)

    def row_sha(i):
        a = next(m for m in msgs[i] if m["role"] == "assistant")
        h = hashlib.sha256(users[i].encode() + b"\0")
        h.update(np.asarray(a["token_ids"], dtype=np.int64).tobytes())
        return h.hexdigest()[:16]

    rec = {
        "parquet": os.path.abspath(PARQUET),
        "parquet_sha256": hashlib.sha256(open(PARQUET, "rb").read()).hexdigest(),
        "rows": len(users),
        "settings": dict(
            DEV_N=DEV_N,
            DEV_SEED=DEV_SEED,
            FIT_SEED=FIT_SEED,
            FIT_VAL_FRAC=FIT_VAL_FRAC,
            N_FIT=N_FIT,
            N_VAL=N_VAL,
        ),
        "counts": dict(
            train=len(train),
            dev=len(dev),
            dev_prompt_groups=len(dev_prompts),
            fit=len(fit),
            fit_check=len(fitval),
        ),
        "dev_rows": dev,
        "fit_rows": fit,
        "fit_check_rows": fitval,
        "dev_row_sha": [row_sha(i) for i in dev],
        "fit_row_sha": [row_sha(i) for i in fit],
        "fit_check_row_sha": [row_sha(i) for i in fitval],
        "train_rows_sha256": hashlib.sha256(
            ",".join(map(str, train)).encode()
        ).hexdigest(),
    }
    Path(OUT_JSON).write_text(json.dumps(rec, indent=1))
    print("FREEZE " + json.dumps({"parquet": PARQUET, **rec["counts"]}), flush=True)
    if os.environ.get("WRITE_DIR"):
        out = Path(os.environ["WRITE_DIR"])
        out.mkdir(parents=True, exist_ok=True)
        full = pq.read_table(PARQUET)
        stem = Path(PARQUET).stem
        pq.write_table(full.take(train), out / f"{stem}_train.parquet")
        pq.write_table(full.take(dev), out / f"{stem}_dev.parquet")


if __name__ == "__main__":
    main()

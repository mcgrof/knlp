"""Immutable confirmation inputs and atomic, standard-library records."""

import hashlib
import json
import os
from pathlib import Path

ROWS = (
    ("V00", "V00", "ordered"),
    ("V01", "V01", "ordered"),
    ("V02", "V02", "ordered"),
    ("V03", "V03", "ordered"),
    ("V30", "V30", "ordered"),
    ("V30-S", "V30", "shuffle"),
    ("V30-C", "V30", "constant"),
)
SEEDS = tuple(range(76411, 76419))
GUIDANCE = "paired-depth-code-confirmation"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 2**20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def source_guard(root):
    r = Path(root) / "records"
    source = read(r / "STAGE-C-SOURCE.json")
    for n, h in source["files"].items():
        assert sha(Path(root) / "code" / n) == h, n
    assert source["files"]["research/mo_variants/model.py"] == MODEL_SHA
    return source


def plan_guard(root):
    root = Path(root)
    r = root / "records"
    p = read(r / "STAGE-C-PLAN.json")
    auth = read(r / "STAGE-C-RESOURCE-DECISION.json")
    assert auth["status"] == "approved" and auth["scope"] == "C"
    assert auth["invoked_by_user"]
    assert p["frozen_before_confirmation_outcomes"]
    assert p["D_enabled"] is False and len(p["jobs"]) == 56
    assert sha(r / "STAGE-C-PLAN.json") == auth["plan_sha256"]
    source_guard(root)
    return p, auth


MODEL_SHA = sha(Path(__file__).with_name("model.py"))

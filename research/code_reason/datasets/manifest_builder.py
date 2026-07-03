#!/usr/bin/env python3
"""Build execution-free task manifests for the code-reason harness.

A manifest is a JSONL file of `task_manifest`-schema rows, one per task the
runner will attempt. Which task types and which target models appear is
decided entirely by the Kconfig flags carried in `build/code_reason/config.json`
-- never by constants in this file. The builder is deterministic: given the
same dataset and the same flags it emits byte-identical output (stable row
order, sorted keys, seed recorded), so a manifest is a reproducible artifact.

A dataset is a directory holding a `tasks.json` of the form the bundled
`fixtures/smoke` dataset uses: a header (dataset name, repo subdir, language,
source_commit) plus a list of raw records (id, task_type, payload, gold).
No target code is executed to build a manifest; patch equivalence rows carry
a stable `patch_hash` of the reference patch text so runs are traceable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "tools"))
from patch_utils import patch_hash  # noqa: E402

_SCHEMA_DIR = os.path.join(_ROOT, "schemas")
_FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

# Kconfig flag -> canonical task_type (schema enum value).
TASK_FLAGS = {
    "CONFIG_CODE_REASON_TASK_PATCH_EQUIV": "patch_equiv",
    "CONFIG_CODE_REASON_TASK_FAULT_LOCALIZATION": "fault_localization",
    "CONFIG_CODE_REASON_TASK_CODE_QA": "code_qa",
}

# Kconfig flag -> model id. Order here is the stable model_ids order in a row.
MODEL_FLAGS = [
    ("CONFIG_CODE_REASON_MODEL_OPUS_4_5", "claude-opus-4-5"),
    ("CONFIG_CODE_REASON_MODEL_SONNET_4_5", "claude-sonnet-4-5"),
    ("CONFIG_CODE_REASON_MODEL_GEMINI_3_PRO_GRADER", "gemini-3-pro"),
    ("CONFIG_CODE_REASON_MODEL_GPT_5_2_GRADER", "gpt-5.2"),
]

try:
    import jsonschema  # type: ignore

    _HAVE_JSONSCHEMA = True
except Exception:  # pragma: no cover - optional dependency
    _HAVE_JSONSCHEMA = False

_SCHEMA_CACHE: dict = {}


def _load_schema(name):
    if name not in _SCHEMA_CACHE:
        with open(os.path.join(_SCHEMA_DIR, f"{name}.schema.json")) as fh:
            _SCHEMA_CACHE[name] = json.load(fh)
    return _SCHEMA_CACHE[name]


def _validate(obj, schema_name):
    if not _HAVE_JSONSCHEMA:
        return
    jsonschema.validate(obj, _load_schema(schema_name))


def load_config_flags(config_path):
    """Return the flags dict from a gen_config_json.py artifact."""
    with open(config_path) as fh:
        cfg = json.load(fh)
    return cfg.get("flags", {})


def enabled_task_types(flags):
    """Task types whose Kconfig flag is truthy, in schema-enum order."""
    return [tt for flag, tt in TASK_FLAGS.items() if flags.get(flag) in (True, "y")]


def enabled_models(flags):
    return [m for flag, m in MODEL_FLAGS if flags.get(flag) in (True, "y")]


def load_dataset(dataset_dir):
    """Parse a dataset directory's tasks.json header + records."""
    with open(os.path.join(dataset_dir, "tasks.json")) as fh:
        return json.load(fh)


def build_manifest(dataset, flags, seed=0):
    """Build validated manifest rows for the enabled task types + models.

    Rows preserve the dataset's record order and are filtered to the task
    types the flags enable. Returns a list of dicts (not yet written).
    """
    name = dataset["dataset"]
    repo = dataset.get("repo", "repo")
    language = dataset.get("language", "unknown")
    source_commit = dataset.get("source_commit", "")
    want = set(enabled_task_types(flags))
    models = enabled_models(flags)
    rows = []
    for rec in dataset["records"]:
        tt = rec["task_type"]
        if tt not in want:
            continue
        row = {
            "task_id": f"{name}-{rec['id']}",
            "task_type": tt,
            "dataset": name,
            "repo": repo,
            "language": language,
            "source_commit": source_commit,
            "model_ids": models,
            "seed": seed,
            "payload": rec.get("payload", {}),
            "gold": rec.get("gold", {}),
        }
        if tt == "patch_equiv":
            ref = rec.get("payload", {}).get("patch_reference", "")
            row["patch_hash"] = patch_hash(ref)
        _validate(row, "task_manifest")
        rows.append(row)
    return rows


def write_manifest(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def _self_test():
    smoke = load_dataset(os.path.join(_FIXTURES, "smoke"))
    all_on = {flag: True for flag in TASK_FLAGS}
    all_on["CONFIG_CODE_REASON_MODEL_OPUS_4_5"] = True
    all_on["CONFIG_CODE_REASON_MODEL_SONNET_4_5"] = True
    rows = build_manifest(smoke, all_on, seed=7)
    assert len(rows) == 4, [r["task_id"] for r in rows]
    peq = [r for r in rows if r["task_type"] == "patch_equiv"]
    assert len(peq) == 2 and all(r.get("patch_hash") for r in peq), peq
    assert all(r["model_ids"] == ["claude-opus-4-5", "claude-sonnet-4-5"] for r in rows)
    assert all(r["seed"] == 7 for r in rows)
    # task filtering: drop code_qa
    partial = dict(all_on)
    partial["CONFIG_CODE_REASON_TASK_CODE_QA"] = False
    rows2 = build_manifest(smoke, partial, seed=0)
    assert len(rows2) == 3 and not any(r["task_type"] == "code_qa" for r in rows2), [
        r["task_id"] for r in rows2
    ]
    # determinism: same inputs -> identical bytes
    import io

    def dump(rs):
        buf = io.StringIO()
        for r in rs:
            buf.write(json.dumps(r, sort_keys=True) + "\n")
        return buf.getvalue()

    assert dump(build_manifest(smoke, all_on, 7)) == dump(rows)
    js = "with-schema" if _HAVE_JSONSCHEMA else "no-jsonschema (soft)"
    print(f"[manifest_builder] self-test PASS ({js}): 4 rows, filter 3, deterministic")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", default=os.path.join(_FIXTURES, "smoke"))
    ap.add_argument(
        "--config",
        default=os.path.join(_ROOT, "..", "..", "build", "code_reason", "config.json"),
    )
    ap.add_argument("--out", default=None, help="manifest.jsonl path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return
    if os.path.exists(args.config):
        flags = load_config_flags(args.config)
    else:
        # No config.json yet: enable everything the paper path implies.
        flags = {f: True for f in TASK_FLAGS}
        flags["CONFIG_CODE_REASON_MODEL_OPUS_4_5"] = True
        flags["CONFIG_CODE_REASON_MODEL_SONNET_4_5"] = True
        print(
            f"[manifest_builder] no config at {args.config}; enabling all tasks",
            file=sys.stderr,
        )
    dataset = load_dataset(args.dataset_dir)
    rows = build_manifest(dataset, flags, seed=args.seed)
    out = args.out or os.path.join(args.dataset_dir, "manifest.jsonl")
    write_manifest(rows, out)
    print(f"[manifest_builder] wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()

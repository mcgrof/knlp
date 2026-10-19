"""CLI for a pinned, external MoT author-code baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from .audit import run_runtime_audit
from .attribution_eval import replay_attribution_jsonl, run_attribution_eval
from .data import freeze_smoke_manifest
from .diagnostic_data import freeze_diagnostic_manifest
from .controls import run_diagnostic_controls
from .diagnostic_eval import replay_paired_jsonl, run_capped_diagnostic_eval
from .mechanics import run_mechanics_preflight
from .preflight import run_real_model_preflight
from .recipe import (
    DATASET_REVISIONS,
    MODEL_REVISIONS,
    MODIFIED_MOT_SHA,
    MODIFIED_MOT_UPSTREAM_FILE_SHA256,
    UPSTREAM_FILE_SHA256,
    UPSTREAM_SHA,
    UPSTREAM_URL,
    build_channel_profile_config,
    build_eval_config,
    build_reproduction_matrix,
    build_train_config,
)
from .smoke import run_training_smoke


def _read_kconfig(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("CONFIG_") or "=" not in line:
            continue
        name, raw_value = line.split("=", 1)
        key = name.removeprefix("CONFIG_")
        if raw_value == "y":
            value: Any = True
        elif raw_value == "n":
            value = False
        elif raw_value.startswith('"') and raw_value.endswith('"'):
            value = json.loads(raw_value)
        else:
            try:
                value = int(raw_value)
            except ValueError:
                value = raw_value
        values[key] = value
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(
    *args: str, cwd: Path | None = None, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=cwd,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _source_from_args(args) -> Path:
    values = _read_kconfig(args.config)
    configured = str(values.get("MOT_AUTHOR_SOURCE", "")).strip()
    source = args.source or os.environ.get("MOT_AUTHOR_SOURCE") or configured
    if not source:
        raise SystemExit(
            "Set MOT_AUTHOR_SOURCE=/path/to/pinned/MoT or "
            "CONFIG_MOT_AUTHOR_SOURCE in the defconfig."
        )
    return Path(source).expanduser().resolve()


def _output_from_args(args) -> Path:
    values = _read_kconfig(args.config)
    configured = str(values.get("MOT_AUTHOR_OUTPUT_DIR", "mot-author-output"))
    return Path(args.output or configured).expanduser().resolve()


def inspect_source(source: Path) -> dict[str, Any]:
    if not (source / ".git").exists():
        raise RuntimeError(f"Not a git checkout: {source}")
    head = _run("git", "rev-parse", "HEAD", cwd=source).stdout.strip()
    status = _run("git", "status", "--porcelain", cwd=source).stdout.strip()
    remotes = _run("git", "remote", "-v", cwd=source).stdout.splitlines()

    expected_files = (
        MODIFIED_MOT_UPSTREAM_FILE_SHA256
        if head == MODIFIED_MOT_SHA
        else UPSTREAM_FILE_SHA256
    )
    hashes = {}
    for relative, expected in expected_files.items():
        file_path = source / relative
        actual = _sha256(file_path) if file_path.is_file() else None
        hashes[relative] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matches": actual == expected,
        }

    tracked_checkpoints = _run(
        "git",
        "ls-tree",
        "-r",
        "--name-only",
        "HEAD",
        cwd=source,
    ).stdout.splitlines()
    tracked_checkpoints = [
        name
        for name in tracked_checkpoints
        if Path(name).suffix.lower() in {".bin", ".ckpt", ".pt", ".pth", ".safetensors"}
    ]
    return {
        "upstream_url": UPSTREAM_URL,
        "expected_sha": [UPSTREAM_SHA, MODIFIED_MOT_SHA],
        "actual_sha": head,
        "sha_matches": head in {UPSTREAM_SHA, MODIFIED_MOT_SHA},
        "source_variant": (
            "modified-mot" if head == MODIFIED_MOT_SHA else "unchanged-author"
        ),
        "worktree_clean": not status,
        "remotes": remotes,
        "file_hashes": hashes,
        "all_file_hashes_match": all(item["matches"] for item in hashes.values()),
        "tracked_checkpoint_files": tracked_checkpoints,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _require_valid_source(receipt: dict[str, Any]) -> None:
    if not receipt["sha_matches"]:
        raise RuntimeError(
            f"MoT source SHA mismatch: {receipt['actual_sha']} not in {receipt['expected_sha']}"
        )
    if not receipt["worktree_clean"]:
        raise RuntimeError(
            "MoT source checkout is modified; preserve an unchanged tree."
        )
    if not receipt["all_file_hashes_match"]:
        raise RuntimeError("One or more pinned MoT source-file hashes do not match.")


def command_doctor(args) -> int:
    receipt = inspect_source(_source_from_args(args))
    _require_valid_source(receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def command_upstream_test(args) -> int:
    source = _source_from_args(args)
    receipt = inspect_source(source)
    _require_valid_source(receipt)
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=source,
        check=False,
    )
    return int(completed.returncode)


def command_audit(args) -> int:
    source = _source_from_args(args)
    receipt = inspect_source(source)
    _require_valid_source(receipt)
    result = run_runtime_audit(source)
    result["source"] = receipt
    if args.output_json:
        _write_json(args.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_real_preflight(args) -> int:
    source = _source_from_args(args)
    receipt = inspect_source(source)
    _require_valid_source(receipt)
    if not args.source_model or not args.target_model:
        raise SystemExit(
            "Real-model preflight requires exact local snapshot paths through "
            "--source-model and --target-model."
        )
    result = run_real_model_preflight(
        source=source,
        source_model=Path(args.source_model),
        target_model=Path(args.target_model),
    )
    result["source"] = receipt
    output_json = args.output_json or (
        _output_from_args(args) / "REAL_MODEL_PREFLIGHT.json"
    )
    _write_json(output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"]["gpt2_author_training_preflight_pass"] else 1


def command_mechanics_preflight(args) -> int:
    if not args.source_model or not args.target_model:
        raise SystemExit(
            "Mechanics preflight requires exact local snapshot paths through "
            "--source-model and --target-model."
        )
    result = run_mechanics_preflight(
        source=_source_from_args(args),
        source_model=Path(args.source_model),
        target_model=Path(args.target_model),
        variant=args.variant,
    )
    output_json = args.output_json or (
        _output_from_args(args) / "MECHANICS_PREFLIGHT.json"
    )
    _write_json(output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"]["mechanics_preflight_pass"] else 1


def command_training_smoke(args) -> int:
    if not args.source_model or not args.target_model:
        raise SystemExit(
            "Training smoke requires exact local snapshot paths through "
            "--source-model and --target-model."
        )
    result = run_training_smoke(
        source=_source_from_args(args),
        source_model=Path(args.source_model),
        target_model=Path(args.target_model),
        data_manifest=Path(args.data_manifest),
        output=_output_from_args(args),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"]["two_step_training_smoke_pass"] else 1


def command_freeze_smoke_data(args) -> int:
    result = freeze_smoke_manifest(
        target_model=Path(args.target_model),
        output_json=Path(args.output_json),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_freeze_diagnostic_data(args) -> int:
    if not args.source_model or not args.target_model:
        raise SystemExit("Diagnostic data freezing requires --source-model and --target-model")
    result = freeze_diagnostic_manifest(
        source=_source_from_args(args),
        source_model=Path(args.source_model),
        target_model=Path(args.target_model),
        output_json=Path(args.output_json),
        smoke_manifest=(None if not args.smoke_manifest else Path(args.smoke_manifest)),
    )
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "manifest_content_sha256": result["manifest_content_sha256"],
                "partition_rows": {
                    name: partition["row_count"]
                    for name, partition in result["partitions"].items()
                },
                "evaluation_items": {
                    name: cohort["item_count"]
                    for name, cohort in result["evaluation"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_diagnostic_controls(args) -> int:
    if not args.source_model or not args.target_model:
        raise SystemExit("Diagnostic controls require --source-model and --target-model")
    result = run_diagnostic_controls(
        source=_source_from_args(args),
        source_model=Path(args.source_model),
        target_model=Path(args.target_model),
        data_manifest=Path(args.data_manifest),
    )
    output_json = args.output_json or (_output_from_args(args) / "DIAGNOSTIC_CONTROLS.json")
    _write_json(output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"]["diagnostic_controls_pass"] else 1


def command_diagnostic_eval(args) -> int:
    result = run_capped_diagnostic_eval(
        source=_source_from_args(args),
        checkpoint=Path(args.checkpoint),
        data_manifest=Path(args.data_manifest),
        output=_output_from_args(args) / "capped_evaluation",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["operational_complete"] else 1


def command_replay_diagnostic(args) -> int:
    result = replay_paired_jsonl(Path(args.input_jsonl), Path(args.output_json))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_attribution_eval(args) -> int:
    result = run_attribution_eval(
        source=_source_from_args(args),
        checkpoint=Path(args.checkpoint),
        checkpoint_file_sha256=args.checkpoint_file_sha256,
        data_manifest=Path(args.data_manifest),
        anchor_records=Path(args.anchor_records),
        output=_output_from_args(args) / "attribution_evaluation",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["operational_complete"] else 1


def command_replay_attribution(args) -> int:
    result = replay_attribution_jsonl(
        Path(args.input_jsonl), Path(args.output_json)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _render(args) -> dict[str, Path]:
    source = _source_from_args(args)
    source_receipt = inspect_source(source)
    _require_valid_source(source_receipt)
    output = _output_from_args(args)
    inputs = output / "inputs"
    run_path = output / "run"
    eval_path = output / "evaluation"
    inputs.mkdir(parents=True, exist_ok=True)

    source_model = args.source_model or "gpt2"
    target_model = args.target_model or "gpt2-medium"
    train_config = build_train_config(
        output_path=run_path,
        source_model=source_model,
        target_model=target_model,
    )
    channel_config = build_channel_profile_config()
    eval_config = build_eval_config(
        output_path=eval_path,
        checkpoint_dir_path=run_path,
    )
    train_path = inputs / "train.json"
    channel_path = inputs / "channel_profile.json"
    eval_config_path = inputs / "eval.json"
    provenance_path = output / "PROVENANCE.json"
    frozen_path = output / "FROZEN_BASELINE.json"
    matrix_path = output / "REPRODUCTION_MATRIX.md"

    _write_json(train_path, train_config)
    _write_json(channel_path, channel_config)
    _write_json(eval_config_path, eval_config)
    _write_json(
        provenance_path,
        {
            "schema": "knlp_mot_author_provenance_v1",
            "source": source_receipt,
            "model_revisions": MODEL_REVISIONS,
            "dataset_revisions": DATASET_REVISIONS,
            "generated_by": "research.kv_translate.mot_author",
        },
    )
    _write_json(
        frozen_path,
        {
            "schema": "knlp_mot_author_frozen_baseline_v1",
            "label": "author_code_mechanics_baseline_not_paper_result_reproduction",
            "paper_anchor": {
                "direction": "gpt2-to-gpt2-medium",
                "top_k": 1,
                "num_translators": 2,
                "reported_f1": 0.34,
                "agreement_tolerance": None,
                "reason_no_gate": (
                    "The paper does not fully specify the Table 8 evaluation "
                    "row or downloadable checkpoint."
                ),
            },
            "native_control": "author evaluator native target path",
            "source_cache_work": "count source prefill and attention extraction",
            "receiver_work": "count complete receiver context replay",
            "known_readiness_limits": [
                "Top-1 routing has zero task-loss gradient in the pinned code.",
                "The pinned forward path executes every translator expert.",
                "Cross-token sparse alignment can create an all-masked causal row.",
                "The same-tokenizer GPT-2 baseline does not exercise that token-grid defect.",
            ],
            "train_config_sha256": _sha256(train_path),
            "channel_profile_config_sha256": _sha256(channel_path),
            "eval_config_sha256": _sha256(eval_config_path),
        },
    )
    matrix_path.write_text(build_reproduction_matrix(), encoding="utf-8")
    return {
        "output": output,
        "train": train_path,
        "channel": channel_path,
        "eval": eval_config_path,
        "provenance": provenance_path,
        "frozen": frozen_path,
        "matrix": matrix_path,
        "source": source,
    }


def command_render(args) -> int:
    paths = _render(args)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    return 0


def command_train(args) -> int:
    if not args.source_model or not args.target_model:
        raise SystemExit(
            "Training requires explicit pinned model snapshot paths through "
            "--source-model and --target-model."
        )
    paths = _render(args)
    command = [
        sys.executable,
        "train.py",
        "mot",
        "--default-config-path",
        str(paths["train"]),
        "--channel-profile-config-path",
        str(paths["channel"]),
        "--output-path",
        str(paths["output"] / "run"),
    ]
    return subprocess.run(command, cwd=paths["source"], check=False).returncode


def command_eval(args) -> int:
    paths = _render(args)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_dir():
        raise SystemExit(f"Checkpoint directory does not exist: {checkpoint}")
    eval_config = json.loads(paths["eval"].read_text(encoding="utf-8"))
    eval_config["checkpoint_dir_path"] = str(checkpoint)
    _write_json(paths["eval"], eval_config)
    command = [
        sys.executable,
        "eval.py",
        "mot",
        "--default-config-path",
        str(paths["eval"]),
        "--checkpoint-dir-path",
        str(checkpoint),
        "--output-path",
        str(paths["output"] / "evaluation"),
    ]
    return subprocess.run(command, cwd=paths["source"], check=False).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(".config"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor").set_defaults(func=command_doctor)
    subparsers.add_parser("upstream-test").set_defaults(func=command_upstream_test)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--output-json", type=Path)
    audit.set_defaults(func=command_audit)

    real_preflight = subparsers.add_parser("real-preflight")
    real_preflight.add_argument("--source-model", required=True)
    real_preflight.add_argument("--target-model", required=True)
    real_preflight.add_argument("--output-json", type=Path)
    real_preflight.set_defaults(func=command_real_preflight)

    mechanics = subparsers.add_parser("mechanics-preflight")
    mechanics.add_argument("--source-model", required=True)
    mechanics.add_argument("--target-model", required=True)
    mechanics.add_argument(
        "--variant",
        choices=("unchanged-author", "modified-mot"),
        required=True,
    )
    mechanics.add_argument("--output-json", type=Path)
    mechanics.set_defaults(func=command_mechanics_preflight)

    smoke = subparsers.add_parser("training-smoke")
    smoke.add_argument("--source-model", required=True)
    smoke.add_argument("--target-model", required=True)
    smoke.add_argument("--data-manifest", required=True)
    smoke.set_defaults(func=command_training_smoke)

    freeze_data = subparsers.add_parser("freeze-smoke-data")
    freeze_data.add_argument("--target-model", required=True)
    freeze_data.add_argument("--output-json", required=True)
    freeze_data.set_defaults(func=command_freeze_smoke_data)

    freeze_diagnostic = subparsers.add_parser("freeze-diagnostic-data")
    freeze_diagnostic.add_argument("--source-model", required=True)
    freeze_diagnostic.add_argument("--target-model", required=True)
    freeze_diagnostic.add_argument("--output-json", required=True)
    freeze_diagnostic.add_argument("--smoke-manifest")
    freeze_diagnostic.set_defaults(func=command_freeze_diagnostic_data)

    diagnostic_controls = subparsers.add_parser("diagnostic-controls")
    diagnostic_controls.add_argument("--source-model", required=True)
    diagnostic_controls.add_argument("--target-model", required=True)
    diagnostic_controls.add_argument("--data-manifest", required=True)
    diagnostic_controls.add_argument("--output-json", type=Path)
    diagnostic_controls.set_defaults(func=command_diagnostic_controls)

    diagnostic_eval = subparsers.add_parser("diagnostic-eval")
    diagnostic_eval.add_argument("--checkpoint", required=True)
    diagnostic_eval.add_argument("--data-manifest", required=True)
    diagnostic_eval.set_defaults(func=command_diagnostic_eval)

    replay_diagnostic = subparsers.add_parser("replay-diagnostic")
    replay_diagnostic.add_argument("--input-jsonl", required=True)
    replay_diagnostic.add_argument("--output-json", required=True)
    replay_diagnostic.set_defaults(func=command_replay_diagnostic)

    attribution_eval = subparsers.add_parser("attribution-eval")
    attribution_eval.add_argument("--checkpoint", required=True)
    attribution_eval.add_argument("--checkpoint-file-sha256", required=True)
    attribution_eval.add_argument("--data-manifest", required=True)
    attribution_eval.add_argument("--anchor-records", required=True)
    attribution_eval.set_defaults(func=command_attribution_eval)

    replay_attribution = subparsers.add_parser("replay-attribution")
    replay_attribution.add_argument("--input-jsonl", required=True)
    replay_attribution.add_argument("--output-json", required=True)
    replay_attribution.set_defaults(func=command_replay_attribution)

    for name, function in (
        ("render", command_render),
        ("train", command_train),
        ("eval", command_eval),
    ):
        child = subparsers.add_parser(name)
        child.add_argument("--source-model")
        child.add_argument("--target-model")
        if name == "eval":
            child.add_argument("--checkpoint", required=True)
        child.set_defaults(func=function)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

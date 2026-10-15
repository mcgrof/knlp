"""Profile and evaluate a paper-shaped DroidSpeak baseline."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .data import (
    LONGBENCH_REVISION,
    deterministic_split,
    download_hotpotqa,
    load_jsonl,
    sha256_file,
)
from .profile import (
    ProfilePoint,
    RecomputeInterval,
    enumerate_intervals,
    pareto_frontier,
    quality_floor,
    select_interval,
)
from .qwen2_backend import Qwen2DroidSpeak, architecture_signature
from .scoring import best_qa_f1

DEFAULT_SENDER = "Qwen/Qwen2.5-0.5B"
DEFAULT_RECEIVER = "Qwen/Qwen2.5-0.5B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sender-model", default=DEFAULT_SENDER)
    parser.add_argument("--receiver-model", default=DEFAULT_RECEIVER)
    parser.add_argument("--sender-revision")
    parser.add_argument("--receiver-revision")
    parser.add_argument("--sender-adapter")
    parser.add_argument("--receiver-adapter")
    parser.add_argument("--sender-adapter-revision")
    parser.add_argument("--receiver-adapter-revision")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-samples", type=int, default=50)
    parser.add_argument("--eval-samples", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-relative-quality-drop", type=float, default=0.05)
    parser.add_argument("--max-context-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def resolve_revision(repo_id: str, requested: str | None) -> str:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, revision=requested)
    if not info.sha:
        raise RuntimeError(f"could not resolve a commit for {repo_id}")
    return info.sha


def git_commit() -> str | None:
    supplied = os.environ.get("KNLP_COMMIT")
    if supplied:
        return supplied
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_record(pair: Qwen2DroidSpeak) -> dict[str, Any]:
    import torch
    import transformers

    device = None
    if torch.cuda.is_available():
        device = {
            "name": torch.cuda.get_device_name(pair.device),
            "capability": list(torch.cuda.get_device_capability(pair.device)),
            "total_memory": torch.cuda.get_device_properties(pair.device).total_memory,
        }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": device,
        "knlp_commit": git_commit(),
    }


def sample_id(sample: dict[str, Any], fallback: int) -> str:
    return str(sample.get("_id", fallback))


def score_row(
    phase: str,
    sample: dict[str, Any],
    fallback_id: int,
    strategy: str,
    interval: RecomputeInterval | None,
    result: Any,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "sample_id": sample_id(sample, fallback_id),
        "strategy": strategy,
        "interval": interval.as_dict() if interval is not None else None,
        "prediction": result.text,
        "answers": sample["answers"],
        "f1": best_qa_f1(result.text, sample["answers"]),
        "ttft_s": result.ttft_s,
        "prefill_s": result.prefill_s,
        "decode_s": result.decode_s,
        "context_tokens": result.context_tokens,
        "query_tokens": result.query_tokens,
        "generated_tokens": len(result.token_ids),
    }


def aggregate_profile(rows: list[dict[str, Any]]) -> list[ProfilePoint]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        interval = row["interval"]
        grouped.setdefault((interval["start"], interval["end"]), []).append(row)
    points = []
    for (start, end), interval_rows in sorted(grouped.items()):
        points.append(
            ProfilePoint(
                RecomputeInterval(start, end),
                statistics.fmean(row["f1"] for row in interval_rows),
                statistics.fmean(row["ttft_s"] for row in interval_rows),
            )
        )
    return points


def point_record(point: ProfilePoint) -> dict[str, Any]:
    return {
        "interval": point.interval.as_dict(),
        "mean_f1": point.quality,
        "mean_ttft_s": point.latency_s,
    }


def aggregate_eval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    strategies = sorted({row["strategy"] for row in rows})
    for strategy in strategies:
        selected = [row for row in rows if row["strategy"] == strategy]
        result[strategy] = {
            "samples": len(selected),
            "mean_f1": statistics.fmean(row["f1"] for row in selected),
            "mean_ttft_s": statistics.fmean(row["ttft_s"] for row in selected),
            "mean_prefill_s": statistics.fmean(row["prefill_s"] for row in selected),
        }
    receiver = result["receiver_full_prefill"]
    for strategy, values in result.items():
        values["ttft_speedup_vs_receiver"] = (
            receiver["mean_ttft_s"] / values["mean_ttft_s"]
        )
        values["relative_f1_vs_receiver"] = (
            values["mean_f1"] / receiver["mean_f1"] if receiver["mean_f1"] else None
        )
    return result


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    if args.max_context_tokens <= 0:
        raise ValueError("max-context-tokens must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(
            f"output directory is not empty; preserve or move it: {args.output}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    dataset_path = download_hotpotqa(args.dataset, force=args.force_download)
    records = load_jsonl(dataset_path)
    profile_samples, eval_samples = deterministic_split(
        records,
        args.profile_samples,
        args.eval_samples,
        args.seed,
    )
    sender_revision = resolve_revision(args.sender_model, args.sender_revision)
    receiver_revision = resolve_revision(args.receiver_model, args.receiver_revision)
    sender_adapter_revision = (
        resolve_revision(args.sender_adapter, args.sender_adapter_revision)
        if args.sender_adapter
        else None
    )
    receiver_adapter_revision = (
        resolve_revision(args.receiver_adapter, args.receiver_adapter_revision)
        if args.receiver_adapter
        else None
    )
    pair = Qwen2DroidSpeak.load(
        sender_model=args.sender_model,
        receiver_model=args.receiver_model,
        sender_revision=sender_revision,
        receiver_revision=receiver_revision,
        sender_adapter=args.sender_adapter,
        receiver_adapter=args.receiver_adapter,
        sender_adapter_revision=sender_adapter_revision,
        receiver_adapter_revision=receiver_adapter_revision,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    manifest = {
        "schema_version": 1,
        "started_unix_s": time.time(),
        "paper_contract": {
            "intervals": "all contiguous intervals on grouped boundaries",
            "group_size": args.group_size,
            "quality_budget_relative": args.max_relative_quality_drop,
            "profile_dataset": "LongBench hotpotqa",
            "profile_samples": args.profile_samples,
            "held_out_samples": args.eval_samples,
        },
        "scope": {
            "algorithm": "in-process Transformers DroidSpeak mechanism",
            "transport": "not timed; no LMCache or network in this run",
            "claim": "mechanism baseline, not paper-scale performance reproduction",
        },
        "models": {
            "sender": {
                "id": args.sender_model,
                "revision": sender_revision,
                "adapter": args.sender_adapter,
                "adapter_revision": sender_adapter_revision,
            },
            "receiver": {
                "id": args.receiver_model,
                "revision": receiver_revision,
                "adapter": args.receiver_adapter,
                "adapter_revision": receiver_adapter_revision,
            },
            "architecture": architecture_signature(pair.receiver.config),
        },
        "dataset": {
            "path": str(dataset_path.resolve()),
            "sha256": sha256_file(dataset_path),
            "longbench_revision": LONGBENCH_REVISION,
        },
        "parameters": vars(args)
        | {"dataset": str(args.dataset), "output": str(args.output)},
        "environment": environment_record(pair),
    }
    write_json(args.output / "manifest.json", manifest)

    intervals = enumerate_intervals(pair.num_layers, args.group_size)
    profile_rows: list[dict[str, Any]] = []
    audit = None
    for sample_number, sample in enumerate(profile_samples):
        prompt = pair.tokenize_hotpot(
            sample["context"], sample["input"], args.max_context_tokens
        )
        sender_state = pair.capture_sender(prompt.context_ids)
        if audit is None:
            audit = pair.audit_full_recompute(prompt, sender_state)
            write_json(args.output / "full_recompute_audit.json", audit)
            if audit["max_key_abs_error"] != 0 or audit["max_value_abs_error"] != 0:
                raise RuntimeError(f"full recompute parity failed: {audit}")
        for interval in intervals:
            generated = pair.run_interval(
                prompt, sender_state, interval, args.max_new_tokens
            )
            row = score_row(
                "profile",
                sample,
                sample_number,
                "receiver_hybrid",
                interval,
                generated,
            )
            profile_rows.append(row)
            append_jsonl(args.output / "profile_samples.jsonl", row)

    points = aggregate_profile(profile_rows)
    direct_interval = RecomputeInterval(0, 0)
    full_interval = RecomputeInterval(0, pair.num_layers)
    direct_point = next(point for point in points if point.interval == direct_interval)
    full_point = next(point for point in points if point.interval == full_interval)
    selected = select_interval(
        points,
        full_point.quality,
        args.max_relative_quality_drop,
    )
    profile_summary = {
        "quality_floor": quality_floor(
            full_point.quality, args.max_relative_quality_drop
        ),
        "direct_full_reuse": point_record(direct_point),
        "receiver_full_prefill": point_record(full_point),
        "selected": point_record(selected),
        "pareto_frontier": [point_record(point) for point in pareto_frontier(points)],
        "all_points": [point_record(point) for point in points],
    }
    write_json(args.output / "profile_summary.json", profile_summary)

    eval_rows: list[dict[str, Any]] = []
    strategies = {
        "droidspeak_selected": selected.interval,
        "direct_full_reuse": direct_interval,
        "receiver_full_prefill": full_interval,
    }
    for sample_number, sample in enumerate(eval_samples):
        prompt = pair.tokenize_hotpot(
            sample["context"], sample["input"], args.max_context_tokens
        )
        sender_state = pair.capture_sender(prompt.context_ids)
        for strategy, interval in strategies.items():
            generated = pair.run_interval(
                prompt, sender_state, interval, args.max_new_tokens
            )
            row = score_row(
                "held_out",
                sample,
                sample_number,
                strategy,
                interval,
                generated,
            )
            eval_rows.append(row)
            append_jsonl(args.output / "eval_samples.jsonl", row)
        sender_result = pair.run_native(pair.sender, prompt, args.max_new_tokens)
        sender_row = score_row(
            "held_out",
            sample,
            sample_number,
            "sender_full_prefill",
            None,
            sender_result,
        )
        eval_rows.append(sender_row)
        append_jsonl(args.output / "eval_samples.jsonl", sender_row)

    held_out = aggregate_eval(eval_rows)
    receiver_f1 = held_out["receiver_full_prefill"]["mean_f1"]
    direct_f1 = held_out["direct_full_reuse"]["mean_f1"]
    selected_f1 = held_out["droidspeak_selected"]["mean_f1"]
    sender_f1 = held_out["sender_full_prefill"]["mean_f1"]
    held_out_floor = quality_floor(receiver_f1, args.max_relative_quality_drop)
    summary = {
        "finished_unix_s": time.time(),
        "selected_interval": selected.interval.as_dict(),
        "quality_floor_profile_f1": profile_summary["quality_floor"],
        "profile_receiver_f1": full_point.quality,
        "profile_quality_nonzero": full_point.quality > 0,
        "profile_direct_reuse_worse_than_receiver": (
            direct_point.quality < full_point.quality
        ),
        "held_out_quality_floor_f1": held_out_floor,
        "held_out_selected_meets_quality_budget": selected_f1 >= held_out_floor,
        "held_out_direct_reuse_worse_than_receiver": direct_f1 < receiver_f1,
        "paper_pair_receiver_better_than_sender": receiver_f1 > sender_f1,
        "held_out": held_out,
        "transport_included": False,
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

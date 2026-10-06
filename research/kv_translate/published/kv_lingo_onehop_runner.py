#!/usr/bin/env python3
"""Run fixed-history one-hop diagnostics for KV-Lingo checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch

from . import coqa
from .capture import make_cache
from .coqa_runner import write_json, write_jsonl
from .kv_lingo import capture_pre_norm_span
from .kv_lingo_coqa_runner import (
    clean_decode,
    load_translator,
    native_generate,
    should_stop,
)
from .kv_lingo_data import iter_jsonl, token_sha256
from .kv_lingo_train import PINS, load_models


@torch.inference_mode()
def translated_generate(
    tokenizer, target, translator, token_ids: list[int], source_span
):
    device = next(target.parameters()).device
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    prefix_without_last = ids[:, :-1]
    positions = torch.arange(
        prefix_without_last.shape[1], dtype=torch.long, device=device
    ).unsqueeze(0)
    started = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        keys, values = translator(source_span, target, positions)
        cache = make_cache(list(zip(keys, values, strict=True)))
        position = prefix_without_last.shape[1]
        output = target(
            input_ids=ids[:, -1:],
            attention_mask=torch.ones(1, position + 1, dtype=torch.long, device=device),
            position_ids=torch.tensor([[position]], dtype=torch.long, device=device),
            past_key_values=cache,
            use_cache=True,
        )
    cache, logits = output.past_key_values, output.logits
    generated = []
    stop = None
    while stop is None:
        scores = logits[0, -1].float()
        if not torch.isfinite(scores).all():
            raise FloatingPointError(
                "one-hop translated receiver produced nonfinite logits"
            )
        token = int(scores.argmax())
        generated.append(token)
        stop = should_stop(tokenizer, generated, coqa.MAX_NEW_TOKENS)
        if stop is None:
            position += 1
            output = target(
                input_ids=torch.tensor([[token]], dtype=torch.long, device=device),
                attention_mask=torch.ones(
                    1, position + 1, dtype=torch.long, device=device
                ),
                position_ids=torch.tensor(
                    [[position]], dtype=torch.long, device=device
                ),
                past_key_values=cache,
                use_cache=True,
            )
            cache, logits = output.past_key_values, output.logits
    torch.cuda.synchronize()
    raw, answer = clean_decode(tokenizer, generated)
    result = {
        "answer": answer,
        "raw_text": raw,
        "token_ids": generated,
        "stop_reason": stop,
        "health_events": coqa.health_events(
            answer, raw, stop_reason=stop, new_tokens=len(generated)
        ),
        "seconds": time.perf_counter() - started,
    }
    del keys, values, cache, logits, output
    return result


def scored_fields(prefix: str, result: dict, references: list[str]):
    score = coqa.turn_score(references, result["answer"])
    return {
        f"{prefix}_answer": result["answer"],
        f"{prefix}_raw_text": result["raw_text"],
        f"{prefix}_token_ids": result["token_ids"],
        f"{prefix}_stop_reason": result["stop_reason"],
        f"{prefix}_health": result["health_events"],
        f"{prefix}_f1": score["f1"],
        f"{prefix}_em": score["em"],
    }


def parse_translator(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("translator must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("translator must be LABEL=PATH")
    return label, Path(path)


def artifact_direction(path: Path) -> str:
    value = torch.load(path, map_location="cpu", weights_only=False)
    return value["direction"]


def learning_gate(
    stage1_mean_kl: float,
    step1000_mean_kl: float,
    stage1: dict,
    step1000: dict,
) -> dict:
    """Apply the prospectively frozen cumulative-step-1000 learning gate.

    KL must improve by at least ten percent.  The one-hop arm must either avoid
    regressing more than three F1 points/two health points from Stage 1, or pass
    those same absolute point screens against the native receiver.  The latter
    deliberately does not inherit Q1's extra per-domain descriptive screen.
    """
    base_f1 = stage1["equal_domain"]["transfer_f1"]
    trained_f1 = step1000["equal_domain"]["transfer_f1"]
    base_health = (
        stage1["pooled_official_turn_weighted"]["transfer_unhealthy"]
        / stage1["pooled_official_turn_weighted"]["questions"]
    )
    trained_pool = step1000["pooled_official_turn_weighted"]
    trained_health = trained_pool["transfer_unhealthy"] / trained_pool["questions"]
    f1_change = trained_f1 - base_f1
    health_change = trained_health - base_health
    epsilon = 1e-12
    stable = f1_change >= -0.03 - epsilon and health_change <= 0.02 + epsilon
    direct = (
        step1000["equal_domain"]["native_minus_transfer_f1"] <= 0.03 + epsilon
        and trained_pool["health_excess"] <= 0.02 + epsilon
    )
    kl_fraction = step1000_mean_kl / stage1_mean_kl
    kl_passed = kl_fraction <= 0.90 + epsilon
    return {
        "schema": "kv_lingo_step1000_learning_gate_v1",
        "stage1_mean_kl": stage1_mean_kl,
        "step1000_mean_kl": step1000_mean_kl,
        "step1000_over_stage1_kl": kl_fraction,
        "kl_improvement_fraction": 1.0 - kl_fraction,
        "kl_improved_at_least_10_percent": kl_passed,
        "step1000_vs_stage1": {
            "equal_domain_transfer_f1_change": f1_change,
            "unhealthy_rate_change": health_change,
            "f1_decrease_at_most_3pp": f1_change >= -0.03 - epsilon,
            "unhealthy_increase_at_most_2pp": health_change <= 0.02 + epsilon,
            "passed": stable,
        },
        "direct_step1000": {
            "equal_domain_native_minus_transfer_f1": step1000["equal_domain"][
                "native_minus_transfer_f1"
            ],
            "pooled_health_excess": trained_pool["health_excess"],
            "mean_deficit_at_most_3pp": step1000["equal_domain"][
                "native_minus_transfer_f1"
            ]
            <= 0.03 + epsilon,
            "health_excess_at_most_2pp": trained_pool["health_excess"]
            <= 0.02 + epsilon,
            "passed": direct,
        },
        "onehop_passed": stable or direct,
        "passed": kl_passed and (stable or direct),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--direction", choices=("4b-to-8b", "8b-to-4b"), required=True)
    parser.add_argument(
        "--translator", action="append", type=parse_translator, required=True
    )
    parser.add_argument("--stage1-validation", type=Path, required=True)
    parser.add_argument("--step1000-validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    labels = [label for label, _path in args.translator]
    if len(set(labels)) != len(labels):
        parser.error("translator labels must be unique")
    for _label, path in args.translator:
        if artifact_direction(path) != args.direction:
            parser.error(f"{path} is not a {args.direction} translator")

    from transformers import AutoTokenizer

    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-4B",
        revision=PINS["Qwen/Qwen3-4B"],
        local_files_only=True,
    )
    models = load_models("cuda")
    source_name, target_name = (
        ("4b", "8b") if args.direction == "4b-to-8b" else ("8b", "4b")
    )
    translators = {
        label: load_translator(path, "cuda")[0] for label, path in args.translator
    }
    rows = list(iter_jsonl(args.rows))
    outputs = {}
    completed = {}
    for label in labels:
        path = args.out / f"RAW_{label}.jsonl"
        outputs[label] = list(iter_jsonl(path)) if path.exists() else []
        completed[label] = {row["row_id"] for row in outputs[label]}

    started = time.time()
    for serial, row in enumerate(rows, 1):
        missing = [label for label in labels if row["row_id"] not in completed[label]]
        if not missing:
            continue
        token_ids = row["token_ids"]
        if token_sha256(token_ids) != row["token_ids_sha256"]:
            raise RuntimeError(f"one-hop token hash mismatch at {row['row_id']}")
        native = native_generate(
            tokenizer, models[target_name], token_ids, coqa.MAX_NEW_TOKENS
        )
        device = next(models[source_name].parameters()).device
        prefix = torch.tensor([token_ids[:-1]], dtype=torch.long, device=device)
        source_span = capture_pre_norm_span(models[source_name], prefix)
        for label in missing:
            translated = translated_generate(
                tokenizer,
                models[target_name],
                translators[label],
                token_ids,
                source_span,
            )
            outputs[label].append(
                {
                    "row_id": row["row_id"],
                    "conversation_id": row["conversation_id"],
                    "domain": row["domain"],
                    "turn": row["turn"],
                    "token_ids_sha256": row["token_ids_sha256"],
                    "prefix_tokens": row["prefix_tokens"],
                    **scored_fields("native", native, row["references"]),
                    **scored_fields("transfer", translated, row["references"]),
                    "transfer_seconds": translated["seconds"],
                }
            )
            completed[label].add(row["row_id"])
            write_jsonl(args.out / f"RAW_{label}.jsonl", outputs[label])
        del prefix, source_span
        print(f"one-hop {serial}/{len(rows)} {row['row_id']}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    order = {row["row_id"]: serial for serial, row in enumerate(rows)}
    summaries = {}
    for label in labels:
        outputs[label].sort(key=lambda row: order[row["row_id"]])
        write_jsonl(args.out / f"RAW_{label}.jsonl", outputs[label])
        summaries[label] = coqa.summarize_q1(outputs[label])
        write_json(args.out / f"SUMMARY_{label}.json", summaries[label])
    result = {
        "schema": "kv_lingo_onehop_comparison_v1",
        "direction": args.direction,
        "rows": len(rows),
        "labels": labels,
        "summaries": summaries,
        "wall_seconds_this_invocation": time.time() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "Generated-by": "OpenAI Codex",
    }
    if "stage1" in summaries and "step1000" in summaries:
        base = summaries["stage1"]
        trained = summaries["step1000"]
        base_f1 = base["equal_domain"]["transfer_f1"]
        trained_f1 = trained["equal_domain"]["transfer_f1"]
        base_health = base["pooled_official_turn_weighted"]["transfer_unhealthy"] / len(
            rows
        )
        trained_health = trained["pooled_official_turn_weighted"][
            "transfer_unhealthy"
        ] / len(rows)
        result["step1000_vs_stage1"] = {
            "equal_domain_transfer_f1_change": trained_f1 - base_f1,
            "unhealthy_rate_change": trained_health - base_health,
            "f1_decrease_at_most_3pp": bool(base_f1 - trained_f1 <= 0.03),
            "unhealthy_increase_at_most_2pp": bool(
                trained_health - base_health <= 0.02
            ),
            "direct_step1000_screen_passed": bool(trained["screen"]["promising"]),
        }
        with args.stage1_validation.open(encoding="utf-8") as stream:
            stage1_validation = json.load(stream)
        with args.step1000_validation.open(encoding="utf-8") as stream:
            step1000_validation = json.load(stream)
        if stage1_validation["direction"] != args.direction:
            raise ValueError("Stage 1 validation has the wrong direction")
        if step1000_validation["direction"] != args.direction:
            raise ValueError("step-1000 validation has the wrong direction")
        result["learning_gate"] = learning_gate(
            stage1_validation["mean_kl"],
            step1000_validation["mean_kl"],
            base,
            trained,
        )
    write_json(args.out / "RESULT.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

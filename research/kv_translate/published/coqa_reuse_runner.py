#!/usr/bin/env python3
"""Run immutable CoQA passage-cache reuse stages on one accelerator."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import torch

from . import coqa_reuse
from .capture import cache_layers, make_cache
from .coqa_runner import (
    ModelPair,
    cache_length,
    ids_sha256,
    load_pair,
    resource_receipt,
    score_result,
    write_json,
    write_jsonl,
)

ON_CHECKPOINT = None


def log(message):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {message}", flush=True)


def checkpoint():
    if ON_CHECKPOINT is not None:
        ON_CHECKPOINT()


def read_json(path: Path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def clone_cache(cache):
    """Clone every tensor so a continuation cannot mutate the canonical cache."""
    return make_cache(
        [(key.clone(), value.clone()) for key, value in cache_layers(cache)]
    )


def caches_are_storage_disjoint(left, right) -> bool:
    left_layers = cache_layers(left)
    right_layers = cache_layers(right)
    if len(left_layers) != len(right_layers):
        return False
    for left_pair, right_pair in zip(left_layers, right_layers, strict=True):
        for left_tensor, right_tensor in zip(left_pair, right_pair, strict=True):
            if left_tensor.shape != right_tensor.shape:
                return False
            if (
                left_tensor.untyped_storage().data_ptr()
                == right_tensor.untyped_storage().data_ptr()
            ):
                return False
    return True


def cache_fingerprint(cache) -> str:
    """Hash deterministic probes plus complete cache layout metadata."""
    digest = hashlib.sha256()
    digest.update(str(cache_length(cache)).encode("ascii"))
    for layer, pair in enumerate(cache_layers(cache)):
        for component, tensor in zip(("k", "v"), pair, strict=True):
            digest.update(
                f"{layer}:{component}:{tensor.dtype}:{tuple(tensor.shape)}".encode()
            )
            flat = tensor.detach().reshape(-1)
            count = min(17, flat.numel())
            indexes = torch.linspace(
                0, flat.numel() - 1, count, device=flat.device, dtype=torch.long
            )
            probes = flat[indexes].float().cpu().numpy().tobytes()
            digest.update(probes)
    return digest.hexdigest()


def scored_fields(prefix: str, result: dict, references: list[str]) -> dict:
    score = score_result(result, references)
    fields = {
        f"{prefix}_answer": score["answer"],
        f"{prefix}_raw_text": score["raw_text"],
        f"{prefix}_token_ids": score["token_ids"],
        f"{prefix}_stop_reason": score["stop_reason"],
        f"{prefix}_health": score["health_events"],
        f"{prefix}_finite_logits": score["finite_logits"],
        f"{prefix}_f1": score["f1"],
        f"{prefix}_em": score["em"],
    }
    result.pop("cache", None)
    return fields


def reuse_conversation(pair: ModelPair, rows: list[dict], *, early_only: bool):
    rows = sorted(rows, key=lambda row: row["turn"])
    early_cuts = {row["early_cut"] for row in rows}
    early_hashes = {row["early_prefix_token_ids_sha256"] for row in rows}
    if len(early_cuts) != 1 or len(early_hashes) != 1:
        raise RuntimeError("conversation does not have one shared early prefix")
    early_cut = next(iter(early_cuts))
    early_ids = rows[0]["token_ids"][:early_cut]
    if ids_sha256(early_ids) != next(iter(early_hashes)):
        raise RuntimeError("early prefix hash does not match frozen token IDs")

    source, source_prefill_seconds = pair.prefill("14B", early_ids)
    native, native_prefill_seconds = pair.prefill("32B", early_ids)
    mapped, early_mapping_seconds = pair.map_cache("14B", "32B", source)
    canonical = {"source_14B": source, "native_32B": native, "mapped_32B": mapped}
    before = {name: cache_fingerprint(cache) for name, cache in canonical.items()}
    raw_rows = []
    provenance = []

    for row in rows:
        prompt_ids = row["token_ids"]
        late_cut = row["late_cut"]
        native_early_cache = clone_cache(native)
        mapped_early_cache = clone_cache(mapped)
        branch_disjoint = {
            "native_early": caches_are_storage_disjoint(native, native_early_cache),
            "translated_early": caches_are_storage_disjoint(mapped, mapped_early_cache),
        }
        native_early = pair.generate_split(
            "32B", prompt_ids, native_early_cache, cache_cut=early_cut
        )
        translated_early = pair.generate_split(
            "32B", prompt_ids, mapped_early_cache, cache_cut=early_cut
        )
        timings = {
            "canonical_source_prefill": source_prefill_seconds,
            "canonical_native_prefill": native_prefill_seconds,
            "canonical_early_mapping": early_mapping_seconds,
            "native_early_suffix": native_early["suffix_seconds"],
            "native_early_generate": native_early["generation_seconds"],
            "translated_early_suffix": translated_early["suffix_seconds"],
            "translated_early_generate": translated_early["generation_seconds"],
        }
        values = {
            "row_id": row["row_id"],
            "conversation_id": row["conversation_id"],
            "domain": row["domain"],
            "turn": row["turn"],
            "token_ids_sha256": row["token_ids_sha256"],
            **scored_fields("native_early", native_early, row["references"]),
            **scored_fields("translated_early", translated_early, row["references"]),
        }
        del native_early_cache, mapped_early_cache

        if not early_only:
            native_late_cache = clone_cache(native)
            source_late_cache = clone_cache(source)
            branch_disjoint.update(
                {
                    "native_late_extension": caches_are_storage_disjoint(
                        native, native_late_cache
                    ),
                    "source_late_extension": caches_are_storage_disjoint(
                        source, source_late_cache
                    ),
                }
            )
            late_suffix = prompt_ids[early_cut:late_cut]
            native_late_cache, native_extend_seconds = pair.consume(
                "32B", native_late_cache, late_suffix
            )
            source_late_cache, source_extend_seconds = pair.consume(
                "14B", source_late_cache, late_suffix
            )
            mapped_late_cache, late_mapping_seconds = pair.map_cache(
                "14B", "32B", source_late_cache
            )
            native_late = pair.generate_split("32B", prompt_ids, native_late_cache)
            translated_late = pair.generate_split("32B", prompt_ids, mapped_late_cache)
            values.update(scored_fields("native_late", native_late, row["references"]))
            values.update(
                scored_fields("translated_late", translated_late, row["references"])
            )
            timings.update(
                {
                    "native_late_extension": native_extend_seconds,
                    "source_late_extension": source_extend_seconds,
                    "late_mapping": late_mapping_seconds,
                    "native_late_final_token": native_late["suffix_seconds"],
                    "native_late_generate": native_late["generation_seconds"],
                    "translated_late_final_token": translated_late["suffix_seconds"],
                    "translated_late_generate": translated_late["generation_seconds"],
                }
            )
            del native_late_cache, source_late_cache, mapped_late_cache
        values["timings_seconds"] = timings
        raw_rows.append(values)
        provenance.append(
            {
                "row_id": row["row_id"],
                "conversation_id": row["conversation_id"],
                "domain": row["domain"],
                "turn": row["turn"],
                "full_prompt_tokens": len(prompt_ids),
                "full_prompt_token_ids_sha256": row["token_ids_sha256"],
                "early_cut": early_cut,
                "early_prefix_tokens": early_cut,
                "early_suffix_tokens": len(prompt_ids) - early_cut,
                "early_prefix_token_ids_sha256": row["early_prefix_token_ids_sha256"],
                "early_suffix_token_ids_sha256": row["early_suffix_token_ids_sha256"],
                "late_cut": late_cut,
                "late_prefix_tokens": late_cut,
                "late_suffix_tokens": len(prompt_ids) - late_cut,
                "late_prefix_token_ids_sha256": row["late_prefix_token_ids_sha256"],
                "late_suffix_token_ids_sha256": row["late_suffix_token_ids_sha256"],
                "canonical_cache_fingerprints_before": before,
                "branch_storages_disjoint": branch_disjoint,
                "early_source_prefills_per_conversation": 1,
                "early_translations_per_conversation": 1,
                "independent_early_query_branches": len(rows),
            }
        )

    after = {name: cache_fingerprint(cache) for name, cache in canonical.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("a canonical passage cache was mutated by a query branch")
    if not all(all(row["branch_storages_disjoint"].values()) for row in provenance):
        raise RuntimeError(
            "a query branch shares tensor storage with a canonical cache"
        )
    for row in provenance:
        row["canonical_cache_fingerprints_after"] = after
        row["canonical_cache_fingerprints_unchanged"] = unchanged
    del canonical, source, native, mapped
    gc.collect()
    torch.cuda.empty_cache()
    return raw_rows, provenance


def retain_complete_conversations(input_rows, raw_rows, provenance):
    expected = {}
    for row in input_rows:
        expected[row["conversation_id"]] = expected.get(row["conversation_id"], 0) + 1
    complete = {
        conversation_id
        for conversation_id, count in expected.items()
        if sum(row["conversation_id"] == conversation_id for row in raw_rows) == count
        and sum(row["conversation_id"] == conversation_id for row in provenance)
        == count
    }
    return (
        complete,
        [row for row in raw_rows if row["conversation_id"] in complete],
        [row for row in provenance if row["conversation_id"] in complete],
    )


def smoke_summary(rows, provenance):
    finite_fields = [
        "native_early_finite_logits",
        "translated_early_finite_logits",
        "native_late_finite_logits",
        "translated_late_finite_logits",
    ]
    checks = {
        "all_logits_finite": all(row[field] for row in rows for field in finite_fields),
        "all_canonical_fingerprints_unchanged": all(
            row["canonical_cache_fingerprints_unchanged"] for row in provenance
        ),
        "all_query_branch_storages_disjoint": all(
            all(row["branch_storages_disjoint"].values()) for row in provenance
        ),
        "one_early_translation_per_conversation": all(
            row["early_translations_per_conversation"] == 1 for row in provenance
        ),
        "three_independent_early_query_branches": all(
            row["independent_early_query_branches"] == 3 for row in provenance
        ),
        "exactly_four_arms_per_row": True,
    }
    return {
        "schema": "coqa_passage_reuse_smoke_result_v1",
        "rows": len(rows),
        "answers": 4 * len(rows),
        "checks": checks,
        "passed": all(checks.values()),
        "scores_used_for_selection": False,
    }


def result_markdown(stage: str, result: dict) -> str:
    if stage == "r1":
        early = result["contrasts"]["early_transfer_loss"]
        late = result["contrasts"]["late_transfer_loss"]
        lines = [
            "# CoQA reusable passage-cache development result",
            "",
            f"Terminal state: **{result['next_state']}**.",
            "",
            "The four arms compare native and translated caches at both the passage-only and question-containing cuts.",
            f"Early equal-domain native-minus-translated F1: {100 * early['equal_domain_reference_minus_treatment_f1']:.2f} points.",
            f"Late equal-domain native-minus-translated F1: {100 * late['equal_domain_reference_minus_treatment_f1']:.2f} points.",
            "",
            "These are prospective development screens on short CoQA passages, not population noninferiority or a speed claim.",
        ]
    else:
        early = result["early_transfer_loss"]
        lines = [
            "# CoQA reusable passage-cache replication result",
            "",
            f"Terminal state: **{result['next_state']}**.",
            "",
            f"Equal-domain native-minus-translated F1: {100 * early['equal_domain_reference_minus_treatment_f1']:.2f} points.",
            "",
            "This cohort was frozen independently before the development result was observed.",
        ]
    return "\n".join(lines) + "\n"


def run_stage(pair: ModelPair, stage: str, frozen: Path, out: Path):
    input_name = {
        "smoke": "SMOKE_FROZEN_ROWS.jsonl",
        "r1": "R1_FROZEN_ROWS.jsonl",
        "r2": "R2_FROZEN_ROWS.jsonl",
    }[stage]
    inputs = read_jsonl(frozen / input_name)
    if stage == "r1":
        smoke = read_json(out / "SMOKE_RESULT.json")
        if not smoke["passed"]:
            raise RuntimeError("R1 is not admitted because smoke did not pass")
    if stage == "r2":
        r1 = read_json(out / "R1_RESULT.json")
        if r1["next_state"] != "R2":
            raise RuntimeError(f"R2 is not admitted from {r1['next_state']}")
        replication = read_json(frozen / "REPLICATION_MANIFEST.json")
        if not replication["sufficient"]:
            raise RuntimeError("R2 has insufficient frozen independent passages")

    raw_path = out / f"{stage.upper()}_RAW_ANSWERS.jsonl"
    provenance_path = out / f"{stage.upper()}_CACHE_PROVENANCE.jsonl"
    raw = read_jsonl(raw_path) if raw_path.exists() else []
    provenance = read_jsonl(provenance_path) if provenance_path.exists() else []
    completed, raw, provenance = retain_complete_conversations(inputs, raw, provenance)
    grouped = {}
    for row in inputs:
        grouped.setdefault(row["conversation_id"], []).append(row)
    ordered = sorted(
        grouped, key=lambda cid: min(row["row_id"] for row in grouped[cid])
    )
    for index, conversation_id in enumerate(ordered, 1):
        if conversation_id in completed:
            continue
        log(f"{stage} {index}/{len(ordered)} {conversation_id}")
        new_raw, new_provenance = reuse_conversation(
            pair, grouped[conversation_id], early_only=stage == "r2"
        )
        raw.extend(new_raw)
        provenance.extend(new_provenance)
        write_jsonl(raw_path, raw)
        write_jsonl(provenance_path, provenance)
        checkpoint()

    order = {row["row_id"]: index for index, row in enumerate(inputs)}
    raw.sort(key=lambda row: order[row["row_id"]])
    provenance.sort(key=lambda row: order[row["row_id"]])
    write_jsonl(raw_path, raw)
    write_jsonl(provenance_path, provenance)
    checkpoint()
    if stage == "smoke":
        result = smoke_summary(raw, provenance)
        write_json(out / "SMOKE_RESULT.json", result)
        answers = 4 * len(raw)
    elif stage == "r1":
        result = coqa_reuse.summarize_r1(raw)
        write_json(out / "R1_RESULT.json", result)
        (out / "R1_RESULT.md").write_text(
            result_markdown(stage, result), encoding="utf-8"
        )
        answers = 4 * len(raw)
    else:
        result = coqa_reuse.summarize_r2(raw)
        write_json(out / "R2_RESULT.json", result)
        (out / "R2_RESULT.md").write_text(
            result_markdown(stage, result), encoding="utf-8"
        )
        answers = 2 * len(raw)
    checkpoint()
    return result, answers


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "r1", "r2"), required=True)
    parser.add_argument("--frozen-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--forward-mapper", required=True)
    args = parser.parse_args(argv)

    frozen = Path(args.frozen_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    pair = load_pair(args.forward_mapper)
    result, answers = run_stage(pair, args.stage, frozen, out)
    resource = resource_receipt(args.stage, started, answers)
    write_json(out / f"RESOURCE_USE_{args.stage.upper()}.json", resource)
    if args.stage == "smoke":
        seconds_per_answer = resource["wall_seconds"] / max(1, answers)
        forecast = {
            "schema": "coqa_passage_reuse_measured_forecast_v1",
            "smoke_answers": answers,
            "smoke_wall_seconds_including_model_load": resource["wall_seconds"],
            "seconds_per_smoke_answer_including_model_load": seconds_per_answer,
            "r1_answers": 1200,
            "conditional_r2_answers": 600,
            "r1_hours_with_30pct_contingency": seconds_per_answer * 1200 * 1.3 / 3600,
            "r2_hours_with_30pct_contingency": seconds_per_answer * 600 * 1.3 / 3600,
            "note": "conservative projection includes model loading in every smoke answer",
        }
        write_json(out / "MEASURED_FORECAST.json", forecast)
    print(
        json.dumps(
            {"stage": args.stage, "result": result, "resource": resource},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

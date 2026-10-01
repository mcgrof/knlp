#!/usr/bin/env python3
"""Run CoQA with persistent native 14B state and one-way 32B answers."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import torch

from . import coqa, coqa_owner
from .capture import cache_layers
from .coqa_reuse_runner import (
    cache_fingerprint,
    caches_are_storage_disjoint,
    clone_cache,
)
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


def full_cache_sha256(cache) -> str:
    """Hash every cache byte plus its tensor metadata."""
    digest = hashlib.sha256()
    for layer, pair in enumerate(cache_layers(cache)):
        for component, tensor in zip(("k", "v"), pair, strict=True):
            digest.update(
                f"{layer}:{component}:{tensor.dtype}:{tuple(tensor.shape)}".encode()
            )
            raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
            digest.update(raw.tobytes())
    return digest.hexdigest()


def _frozen_suffix(story: dict, turn: int, prior_stop: str) -> list[int]:
    key = "newline" if prior_stop == "newline" else "other"
    return story["next_turn_suffix_token_ids"][str(turn)][key]


def _extend_to_late_cut(
    pair: ModelPair,
    model: str,
    cache,
    ledger: list[int],
    suffix: list[int] | None,
):
    if suffix is not None:
        if cache_length(cache) != len(ledger):
            raise RuntimeError(f"{model} cache does not cover the old ledger")
        ledger.extend(suffix)
        pending = suffix[:-1]
    else:
        pending = ledger[cache_length(cache) : -1]
    cache, seconds = pair.consume(model, cache, pending)
    if cache_length(cache) != len(ledger) - 1:
        raise RuntimeError(f"{model} cache does not end at the frozen late cut")
    return cache, seconds


def _ingest_complete_ledger(pair, model, cache, ledger, generated_ids):
    before = cache_length(cache)
    ledger.extend(generated_ids)
    pending = ledger[before:]
    cache, seconds = pair.consume(model, cache, pending)
    if cache_length(cache) != len(ledger):
        raise RuntimeError(f"{model} did not ingest every returned token exactly once")
    return cache, pending, seconds


def _answer_fields(prefix: str, result: dict, references: list[str]) -> dict:
    scored = score_result(result, references)
    return {
        f"{prefix}_answer": scored["answer"],
        f"{prefix}_raw_text": scored["raw_text"],
        f"{prefix}_token_ids": scored["token_ids"],
        f"{prefix}_stop_reason": scored["stop_reason"],
        f"{prefix}_health": scored["health_events"],
        f"{prefix}_finite_logits": scored["finite_logits"],
        f"{prefix}_f1": scored["f1"],
        f"{prefix}_em": scored["em"],
    }


def owner_conversation(
    pair: ModelPair,
    story: dict,
    *,
    turns: int,
    full_root_hash: bool,
):
    passage_ids = story["passage_prefix_token_ids"]
    turn_one_ids = story["turn_1_token_ids"]
    passage_cut = story["passage_cut"]
    if len(passage_ids) != passage_cut or turn_one_ids[:passage_cut] != passage_ids:
        raise RuntimeError("frozen passage prefix does not match turn-one tokens")

    root14, root14_seconds = pair.prefill("14B", passage_ids)
    root32, root32_seconds = pair.prefill("32B", passage_ids)
    roots = {"14B": root14, "32B": root32}
    sampled_before = {name: cache_fingerprint(cache) for name, cache in roots.items()}
    full_before = (
        {name: full_cache_sha256(cache) for name, cache in roots.items()}
        if full_root_hash
        else None
    )

    source_cache = clone_cache(root14)
    shadow_cache = clone_cache(root32)
    native_cache = clone_cache(root32)
    branch_disjoint = {
        "source_from_14B_root": caches_are_storage_disjoint(root14, source_cache),
        "shadow_from_32B_root": caches_are_storage_disjoint(root32, shadow_cache),
        "native_only_from_32B_root": caches_are_storage_disjoint(root32, native_cache),
        "two_32B_branches": caches_are_storage_disjoint(shadow_cache, native_cache),
    }
    if not all(branch_disjoint.values()):
        raise RuntimeError("a writable conversation branch aliases an immutable root")

    shared_ledger = list(turn_one_ids)
    native_ledger = list(turn_one_ids)
    shared_stop = native_stop = None
    rows, ledgers, provenance = [], [], []

    for turn in range(1, turns + 1):
        if turn == 1:
            shared_suffix = native_suffix = None
        else:
            shared_suffix = _frozen_suffix(story, turn, shared_stop)
            native_suffix = _frozen_suffix(story, turn, native_stop)

        source_cache, source_question_seconds = _extend_to_late_cut(
            pair, "14B", source_cache, shared_ledger, shared_suffix
        )
        shadow_cache, shadow_question_seconds = _extend_to_late_cut(
            pair, "32B", shadow_cache, list(shared_ledger), None
        )
        native_cache, native_question_seconds = _extend_to_late_cut(
            pair, "32B", native_cache, native_ledger, native_suffix
        )
        if cache_length(source_cache) != cache_length(shadow_cache):
            raise RuntimeError("source and same-transcript shadow cuts disagree")

        shared_prompt_hash = ids_sha256(shared_ledger)
        native_prompt_hash = ids_sha256(native_ledger)
        mapped, mapping_seconds = pair.map_cache("14B", "32B", source_cache)
        translated = pair.generate_split("32B", shared_ledger, mapped)
        shadow_branch = clone_cache(shadow_cache)
        shadow = pair.generate_split("32B", shared_ledger, shadow_branch)
        native = pair.generate_split("32B", native_ledger, native_cache)
        references = coqa.answer_references(story, turn)

        rows.append(
            {
                "row_id": f"{story['id']}:{turn}",
                "conversation_id": story["id"],
                "domain": story["source"],
                "turn": turn,
                "shared_prompt_token_ids_sha256": shared_prompt_hash,
                "native_only_prompt_token_ids_sha256": native_prompt_hash,
                **_answer_fields("translated", translated, references),
                **_answer_fields("same_transcript_native", shadow, references),
                **_answer_fields("native_32b_only", native, references),
                "timings_seconds": {
                    "source_question_ingest": source_question_seconds,
                    "shadow_question_ingest": shadow_question_seconds,
                    "native_only_question_ingest": native_question_seconds,
                    "mapping": mapping_seconds,
                    "translated_final_token": translated["suffix_seconds"],
                    "translated_generation": translated["generation_seconds"],
                    "shadow_final_token": shadow["suffix_seconds"],
                    "shadow_generation": shadow["generation_seconds"],
                    "native_only_final_token": native["suffix_seconds"],
                    "native_only_generation": native["generation_seconds"],
                },
            }
        )

        shared_generated = list(translated["token_ids"])
        native_generated = list(native["token_ids"])
        if not shared_generated or not native_generated:
            raise RuntimeError("the frozen decoder returned an empty token sequence")
        source_cache, source_ingested, source_answer_seconds = _ingest_complete_ledger(
            pair, "14B", source_cache, shared_ledger, shared_generated
        )
        shadow_cache, shadow_ingested, shadow_answer_seconds = _ingest_complete_ledger(
            pair,
            "32B",
            shadow_cache,
            list(shared_ledger[: -len(shared_generated)]),
            shared_generated,
        )
        native_cache = native["cache"]
        native_cache, native_ingested, native_answer_seconds = _ingest_complete_ledger(
            pair, "32B", native_cache, native_ledger, native_generated
        )
        if source_ingested != shadow_ingested:
            raise RuntimeError(
                "native source and shadow ingested different shared tokens"
            )
        if source_ingested[-len(shared_generated) :] != shared_generated:
            raise RuntimeError("actual generated answer IDs were not returned to 14B")

        shared_stop = translated["stop_reason"]
        native_stop = native["stop_reason"]
        ledgers.append(
            {
                "row_id": f"{story['id']}:{turn}",
                "conversation_id": story["id"],
                "domain": story["source"],
                "turn": turn,
                "shared_prompt_token_ids_sha256": shared_prompt_hash,
                "shared_generated_token_ids": shared_generated,
                "shared_generated_stop_reason": shared_stop,
                "source_ingested_token_ids": source_ingested,
                "shadow_ingested_token_ids": shadow_ingested,
                "shared_ledger_tokens_after_answer": len(shared_ledger),
                "shared_ledger_token_ids_after_answer": list(shared_ledger),
                "shared_ledger_token_ids_sha256_after_answer": ids_sha256(
                    shared_ledger
                ),
                "native_only_prompt_token_ids_sha256": native_prompt_hash,
                "native_only_generated_token_ids": native_generated,
                "native_only_generated_stop_reason": native_stop,
                "native_only_ingested_token_ids": native_ingested,
                "native_only_ledger_tokens_after_answer": len(native_ledger),
                "native_only_ledger_token_ids_after_answer": list(native_ledger),
                "native_only_ledger_token_ids_sha256_after_answer": ids_sha256(
                    native_ledger
                ),
            }
        )
        provenance.append(
            {
                "row_id": f"{story['id']}:{turn}",
                "conversation_id": story["id"],
                "domain": story["source"],
                "turn": turn,
                "persistent_owner": "native_14B",
                "receiver": "disposable_mapped_32B",
                "reverse_mapping": False,
                "source_cache_tokens_after_return": cache_length(source_cache),
                "shadow_cache_tokens_after_return": cache_length(shadow_cache),
                "native_only_cache_tokens_after_return": cache_length(native_cache),
                "shared_ledger_tokens_after_answer": len(shared_ledger),
                "native_only_ledger_tokens_after_answer": len(native_ledger),
                "returned_actual_generated_ids": True,
                "source_answer_ingest_seconds": source_answer_seconds,
                "shadow_answer_ingest_seconds": shadow_answer_seconds,
                "native_only_answer_ingest_seconds": native_answer_seconds,
                "root_branch_storages_disjoint": branch_disjoint,
            }
        )
        del mapped, translated, shadow_branch, shadow, native

    sampled_after = {name: cache_fingerprint(cache) for name, cache in roots.items()}
    if sampled_before != sampled_after:
        raise RuntimeError("an immutable passage root changed")
    full_after = (
        {name: full_cache_sha256(cache) for name, cache in roots.items()}
        if full_root_hash
        else None
    )
    if full_root_hash and full_before != full_after:
        raise RuntimeError("a smoke passage root changed at the byte level")
    for row in provenance:
        row.update(
            {
                "root_prefill_seconds": {"14B": root14_seconds, "32B": root32_seconds},
                "root_sampled_fingerprints_before": sampled_before,
                "root_sampled_fingerprints_after": sampled_after,
                "root_sampled_fingerprints_unchanged": True,
                "root_full_sha256_before": full_before,
                "root_full_sha256_after": full_after,
                "root_full_sha256_unchanged": (
                    full_before == full_after if full_root_hash else None
                ),
            }
        )
    del roots, root14, root32, source_cache, shadow_cache, native_cache
    gc.collect()
    torch.cuda.empty_cache()
    return rows, ledgers, provenance


def _retain_complete(stories, rows, ledgers, provenance, turns):
    complete = set()
    for story in stories:
        conversation_id = story["id"]
        if all(
            sum(row["conversation_id"] == conversation_id for row in values) == turns
            for values in (rows, ledgers, provenance)
        ):
            complete.add(conversation_id)

    def keep(values):
        return [row for row in values if row["conversation_id"] in complete]

    return complete, keep(rows), keep(ledgers), keep(provenance)


def _smoke_result(rows, ledgers, provenance):
    checks = {
        "all_logits_finite": all(
            row[f"{stream}_finite_logits"]
            for row in rows
            for stream in coqa_owner.STREAMS
        ),
        "all_actual_generated_ids_returned": all(
            row["returned_actual_generated_ids"] for row in provenance
        ),
        "all_cache_lengths_match_ledgers": all(
            row["source_cache_tokens_after_return"]
            == row["shadow_cache_tokens_after_return"]
            == row["shared_ledger_tokens_after_answer"]
            and row["native_only_cache_tokens_after_return"]
            == row["native_only_ledger_tokens_after_answer"]
            for row in provenance
        ),
        "all_roots_byte_unchanged": all(
            row["root_full_sha256_unchanged"] for row in provenance
        ),
        "all_writable_branches_storage_disjoint": all(
            all(row["root_branch_storages_disjoint"].values()) for row in provenance
        ),
        "exactly_three_streams_per_turn": len(rows) * 3 == 30,
        "complete_token_ledgers": len(ledgers) == len(rows) == 10,
    }
    return {
        "schema": "coqa_native_source_owner_smoke_result_v1",
        "rows": len(rows),
        "answers": len(rows) * 3,
        "checks": checks,
        "passed": all(checks.values()),
        "scores_used_for_selection": False,
        "next_state": "S1" if all(checks.values()) else "FIX_TARGETED_DEFECT",
    }


def _result_markdown(result: dict) -> str:
    lines = [
        f"# CoQA native-source-owner {result['stage'].upper()} result",
        "",
        f"Terminal state: **{result['next_state']}**.",
        "",
    ]
    for reference, contrast in result["contrasts"].items():
        loss = 100 * contrast["equal_domain_reference_minus_translated_f1"]
        late = (
            100 * contrast["turns_6_to_10_equal_domain_reference_minus_translated_f1"]
        )
        lines.append(
            f"Against {reference}: all-turn deficit {loss:.2f} points; turns 6-10 deficit {late:.2f} points."
        )
    lines += [
        "",
        "These are prospective point screens on the frozen CoQA cohort, not population noninferiority or an economics result.",
    ]
    return "\n".join(lines) + "\n"


def run_stage(pair, stage: str, frozen: Path, out: Path):
    names = {"smoke": "SMOKE", "s1": "S1", "s2": "S2"}
    upper = names[stage]
    stories = read_json(frozen / f"{upper}_CONVERSATIONS.json")
    turns = 2 if stage == "smoke" else 10
    if stage == "s1":
        smoke = read_json(out / "SMOKE_RESULT.json")
        if not smoke["passed"]:
            raise RuntimeError("S1 is not admitted because smoke did not pass")
    if stage == "s2":
        s1 = read_json(out / "S1_RESULT.json")
        if s1["next_state"] != "S2":
            raise RuntimeError(f"S2 is not admitted from {s1['next_state']}")

    raw_path = out / f"{upper}_RAW_ANSWERS.jsonl"
    ledger_path = out / f"{upper}_TOKEN_LEDGER.jsonl"
    provenance_path = out / f"{upper}_CACHE_PROVENANCE.jsonl"
    rows = read_jsonl(raw_path) if raw_path.exists() else []
    ledgers = read_jsonl(ledger_path) if ledger_path.exists() else []
    provenance = read_jsonl(provenance_path) if provenance_path.exists() else []
    complete, rows, ledgers, provenance = _retain_complete(
        stories, rows, ledgers, provenance, turns
    )
    for index, story in enumerate(stories, 1):
        if story["id"] in complete:
            continue
        log(f"{stage} {index}/{len(stories)} {story['id']} turns={turns}")
        new_rows, new_ledgers, new_provenance = owner_conversation(
            pair, story, turns=turns, full_root_hash=stage == "smoke"
        )
        rows.extend(new_rows)
        ledgers.extend(new_ledgers)
        provenance.extend(new_provenance)
        write_jsonl(raw_path, rows)
        write_jsonl(ledger_path, ledgers)
        write_jsonl(provenance_path, provenance)
        checkpoint()

    order = {
        (story["id"], turn): index
        for index, story in enumerate(stories)
        for turn in range(1, turns + 1)
    }
    for values in (rows, ledgers, provenance):
        values.sort(key=lambda row: order[(row["conversation_id"], row["turn"])])
    write_jsonl(raw_path, rows)
    write_jsonl(ledger_path, ledgers)
    write_jsonl(provenance_path, provenance)
    checkpoint()

    if stage == "smoke":
        result = _smoke_result(rows, ledgers, provenance)
    else:
        result = coqa_owner.summarize(rows, stage)
        (out / f"{upper}_RESULT.md").write_text(
            _result_markdown(result), encoding="utf-8"
        )
    write_json(out / f"{upper}_RESULT.json", result)
    checkpoint()
    return result, len(rows) * 3


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "s1", "s2"), required=True)
    parser.add_argument("--frozen-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--forward-mapper", required=True)
    parser.add_argument("--combined-rate", type=float, required=True)
    args = parser.parse_args(argv)

    frozen, out = Path(args.frozen_dir), Path(args.out_dir)
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
            "schema": "coqa_native_source_owner_measured_forecast_v1",
            "smoke_answers": answers,
            "smoke_wall_seconds_including_model_load": resource["wall_seconds"],
            "seconds_per_smoke_answer_including_model_load": seconds_per_answer,
            "combined_rate_usd_per_hour": args.combined_rate,
            "s1_answers": 600,
            "conditional_s2_answers": 3000,
            "s1_hours_with_30pct_contingency": seconds_per_answer * 600 * 1.3 / 3600,
            "s2_hours_with_30pct_contingency": seconds_per_answer * 3000 * 1.3 / 3600,
            "note": "conservative projection includes fixed model loading in every smoke answer",
        }
        forecast["conditional_total_usd"] = args.combined_rate * (
            forecast["s1_hours_with_30pct_contingency"]
            + forecast["s2_hours_with_30pct_contingency"]
        )
        write_json(out / "MEASURED_FORECAST.json", forecast)
    print(
        json.dumps(
            {"stage": args.stage, "result": result, "resource": resource},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

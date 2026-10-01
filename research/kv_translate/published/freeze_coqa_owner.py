#!/usr/bin/env python3
"""Freeze native-14B-owner CoQA cohorts, token recipes, and receipts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from . import coqa, coqa_owner, coqa_reuse
from .freeze_coqa import file_sha256
from .freeze_coqa_reuse import git_commit_exists


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def read_json(path: Path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def artifact(path: Path) -> dict:
    return {"sha256": file_sha256(path), "bytes": path.stat().st_size}


def token_cut(offsets, boundary: int) -> dict:
    cut = 0
    crossing = None
    for index, (start, end) in enumerate(offsets):
        if start < boundary < end:
            crossing = index
        if end <= boundary and end > start:
            cut = index + 1
    if cut < 1 or cut >= len(offsets):
        raise ValueError(f"invalid passage cut {cut} for {len(offsets)} tokens")
    return {
        "passage_cut": cut,
        "token_crossing_boundary": crossing,
        "crossing_token_left_in_suffix": crossing is None or crossing >= cut,
    }


def freeze_story(story: dict, source_tokenizer, target_tokenizer) -> dict:
    if len(story["questions"]) < 10 or len(story["answers"]) < 10:
        raise ValueError(f"conversation {story['id']} lacks ten complete turns")
    prompt = coqa.render_prompt(story, 1, [])
    boundary = len(f"{coqa.INSTRUCTION}\n\nPassage: {story['story']}\n\n")
    if not prompt[boundary:].startswith("Question:"):
        raise RuntimeError(f"renderer boundary is wrong for {story['id']}")
    target = target_tokenizer(
        prompt, add_special_tokens=False, return_offsets_mapping=True
    )
    target_ids = list(target["input_ids"])
    source_ids = source_tokenizer.encode(prompt, add_special_tokens=False)
    if source_ids != target_ids:
        raise ValueError(f"tokenizers disagree on turn one for {story['id']}")
    cut = token_cut(list(target["offset_mapping"]), boundary)

    suffixes = {}
    for turn in range(2, 11):
        question = story["questions"][turn - 1]["input_text"]
        suffixes[str(turn)] = {}
        for key, stop_reason in (("newline", "newline"), ("other", "cap")):
            text = coqa.next_turn_suffix(question, stop_reason)
            target_suffix = target_tokenizer.encode(text, add_special_tokens=False)
            source_suffix = source_tokenizer.encode(text, add_special_tokens=False)
            if source_suffix != target_suffix:
                raise ValueError(
                    f"tokenizers disagree on {story['id']} turn {turn} {key}"
                )
            suffixes[str(turn)][key] = target_suffix

    passage_ids = target_ids[: cut["passage_cut"]]
    return {
        "id": story["id"],
        "source": story["source"],
        "selection_sha256": coqa.selection_digest(story["id"]),
        "story": story["story"],
        "questions": story["questions"][:10],
        "answers": story["answers"][:10],
        "additional_answers": {
            key: answers[:10]
            for key, answers in story.get("additional_answers", {}).items()
        },
        "first_question_character_offset": boundary,
        **cut,
        "passage_prefix_token_ids": passage_ids,
        "passage_prefix_token_ids_sha256": coqa.sha256_bytes(
            b"".join(
                int(token).to_bytes(8, "little", signed=False) for token in passage_ids
            )
        ),
        "turn_1_token_ids": target_ids,
        "next_turn_suffix_token_ids": suffixes,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-q0", required=True)
    parser.add_argument("--reuse-r0", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--guidance-commit", required=True)
    parser.add_argument("--source-repo")
    parser.add_argument("--provider", required=True)
    parser.add_argument("--gpu-rate", type=float, required=True)
    parser.add_argument("--cpu-rate", type=float, required=True)
    parser.add_argument("--memory-rate", type=float, required=True)
    parser.add_argument("--alternative-provider", required=True)
    parser.add_argument("--alternative-gpu-rate", type=float, required=True)
    parser.add_argument("--provider-reason", required=True)
    args = parser.parse_args(argv)

    previous = Path(args.previous_q0)
    reuse = Path(args.reuse_r0)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    source_repo = (
        Path(args.source_repo)
        if args.source_repo
        else Path(__file__).resolve().parents[3]
    )
    if not git_commit_exists(source_repo, args.source_commit):
        raise ValueError(f"source commit {args.source_commit} is unavailable")

    previous_conversations = read_json(previous / "FROZEN_CONVERSATIONS.json")
    smoke_raw = [
        story for story in previous_conversations if "smoke" in story["selected_for"]
    ]
    s1_raw = [
        story for story in previous_conversations if "q2" in story["selected_for"]
    ]
    s2_raw = read_json(reuse / "R2_FROZEN_CONVERSATIONS.json")
    if (len(smoke_raw), len(s1_raw), len(s2_raw)) != (5, 20, 100):
        raise ValueError(
            f"unexpected cohort sizes {(len(smoke_raw), len(s1_raw), len(s2_raw))}"
        )

    from transformers import AutoTokenizer

    source_name = "Qwen/Qwen3-14B"
    source_revision = "40c069824f4251a91eefaf281ebe4c544efd3e18"
    target_name = "Qwen/Qwen3-32B"
    target_revision = "9216db5781bf21249d130ec9da846c4624c16137"
    source_tokenizer = AutoTokenizer.from_pretrained(
        source_name, revision=source_revision
    )
    target_tokenizer = AutoTokenizer.from_pretrained(
        target_name, revision=target_revision
    )
    cohorts = {
        "SMOKE": [
            freeze_story(story, source_tokenizer, target_tokenizer)
            for story in smoke_raw
        ],
        "S1": [
            freeze_story(story, source_tokenizer, target_tokenizer) for story in s1_raw
        ],
        "S2": [
            freeze_story(story, source_tokenizer, target_tokenizer) for story in s2_raw
        ],
    }
    for name, stories in cohorts.items():
        write_json(out / f"{name}_CONVERSATIONS.json", stories)

    ids = {
        name: {story["id"] for story in stories} for name, stories in cohorts.items()
    }
    normalized_passages = {
        name: [coqa_reuse.normalize_passage(story["story"]) for story in stories]
        for name, stories in cohorts.items()
    }
    disjoint = (
        ids["SMOKE"].isdisjoint(ids["S1"])
        and ids["SMOKE"].isdisjoint(ids["S2"])
        and ids["S1"].isdisjoint(ids["S2"])
    )
    passage_disjoint = len(
        {passage for passages in normalized_passages.values() for passage in passages}
    ) == sum(len(passages) for passages in normalized_passages.values())
    if not disjoint or not passage_disjoint:
        raise RuntimeError("frozen cohorts are not ID- and passage-disjoint")

    protocol = {
        "schema": "coqa_native_source_owner_protocol_v1",
        "guidance_commit": args.guidance_commit,
        "models": {
            "source": {"name": source_name, "revision": source_revision},
            "receiver": {"name": target_name, "revision": target_revision},
        },
        "dtype": "bfloat16",
        "mapper": "unchanged forward Full-Head affine ridge k=8 artifact",
        "persistent_owner": "14B cache containing only native 14B forward outputs",
        "handoff": "map the entire native source prefix at prompt-minus-final-token",
        "receiver": "disposable mapped 32B branch for one answer",
        "return_path": (
            "ingest the withheld prompt token and every actual generated answer "
            "token ID exactly once through native 14B inference"
        ),
        "reverse_mapping": False,
        "streams": list(coqa_owner.STREAMS),
        "history": "generated token IDs only; references are scoring-only",
        "turns": list(range(1, 11)),
        "decoding": read_json(previous / "PROTOCOL.json")["decoding"],
        "statistics": {
            "bootstrap_unit": "complete conversations resampled within domain",
            "seed": coqa.BOOTSTRAP_SEED,
            "replicates": coqa.BOOTSTRAP_REPLICATES,
            "interpretation": "descriptive only",
        },
        "screens": {
            "both_native_references": True,
            "equal_domain_mean_deficit_max": 0.03,
            "pooled_health_excess_max": 0.02,
            "turns_6_to_10_equal_domain_mean_deficit_max": 0.05,
            "s2_each_domain_all_turn_deficit_max": 0.05,
        },
        "forbidden": [
            "mapper fitting or reverse mapping",
            "reference answers in generated history",
            "outcome-driven cohort or threshold changes",
            "delta-only mapping optimization",
            "extra cuts, prompts, model pairs, or rescue sweeps",
            "storage or transport benchmarking",
        ],
    }
    write_json(out / "OWNER_PROTOCOL.json", protocol)

    cohort_manifest = {
        "schema": "coqa_native_source_owner_cohort_manifest_v1",
        "selection_salt": coqa.SELECTION_SALT,
        "cohorts": {
            name.lower(): {
                "conversations": len(stories),
                "turns": 2 if name == "SMOKE" else 10,
                "maximum_answers": len(stories) * (6 if name == "SMOKE" else 30),
                "conversation_ids": [story["id"] for story in stories],
                "artifact": artifact(out / f"{name}_CONVERSATIONS.json"),
            }
            for name, stories in cohorts.items()
        },
        "all_cohort_ids_disjoint": disjoint,
        "all_cohort_passages_disjoint_after_casefold_whitespace": passage_disjoint,
        "s1_identity": "exact prior 20 Q2 development conversations",
        "s2_identity": "exact previously reserved and unused 100-conversation R2 cohort",
        "all_quality_conversations_have_ten_turns": True,
    }
    write_json(out / "COHORT_MANIFEST.json", cohort_manifest)

    previous_mapper = read_json(previous / "MAPPER_MANIFEST.json")["forward"]
    source_manifest = {
        "schema": "coqa_native_source_owner_source_manifest_v1",
        "knlp_commit": args.source_commit,
        "commit_exists_in_source_repo": True,
        "guidance_commit": args.guidance_commit,
        "official_data": artifact(previous / "inputs" / "coqa-dev-v1.0.json"),
        "official_evaluator": artifact(previous / "inputs" / "evaluate-v1.0.py"),
        "forward_mapper": previous_mapper,
        "forward_mapper_sha256_expected": "27cedb0c157ad9f28c8cae1892794ddc2fc1edab528b12d8106e24550f63283a",
        "previous_protocol": artifact(previous / "PROTOCOL.json"),
        "reserved_cohort_manifest": artifact(reuse / "REPLICATION_MANIFEST.json"),
        "entrypoints": [
            "research/kv_translate/published/freeze_coqa_owner.py",
            "research/kv_translate/published/coqa_owner.py",
            "research/kv_translate/published/coqa_owner_runner.py",
            "research/kv_translate/published/coqa_owner_modal_app.py",
        ],
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if previous_mapper["sha256"] != source_manifest["forward_mapper_sha256_expected"]:
        raise RuntimeError(
            "the inherited forward mapper is not the frozen k=8 artifact"
        )
    write_json(out / "SOURCE_MANIFEST.json", source_manifest)

    combined_rate = args.gpu_rate + 4 * args.cpu_rate + 48 * args.memory_rate
    forecast = {
        "schema": "coqa_native_source_owner_cost_forecast_v1",
        "provider": args.provider,
        "resource": {"gpu": "H200", "cpu_cores": 4, "memory_gib": 48},
        "live_rates_usd_per_hour": {
            "gpu": args.gpu_rate,
            "cpu_per_core": args.cpu_rate,
            "memory_per_gib": args.memory_rate,
            "combined_requested": combined_rate,
        },
        "alternative": {
            "provider": args.alternative_provider,
            "gpu_rate_usd_per_hour": args.alternative_gpu_rate,
        },
        "provider_reason": args.provider_reason,
        "planning_gpu_hours_if_both_quality_stages_run": [0.75, 1.25],
        "planning_usd_range": [0.75 * combined_rate, 1.25 * combined_rate],
        "maximum_answers": {"smoke": 30, "s1": 600, "conditional_s2": 3000},
        "first_allocation": "smoke only",
        "reforecast": "replace projections with measured smoke work and 30% contingency",
        "lifecycle_checkpoint_hours": 1,
        "credits": "Modal monthly personal evaluation credit is exhausted; new ephemeral compute is billed spend",
        "not_a_completion_time_or_spending_cap": True,
    }
    write_json(out / "COST_FORECAST.json", forecast)

    cpu_checks = {
        "schema": "coqa_native_source_owner_cpu_checks_v1",
        "status": "PASS",
        "cohort_sizes": {
            name.lower(): len(stories) for name, stories in cohorts.items()
        },
        "all_cohort_ids_disjoint": disjoint,
        "all_cohort_passages_disjoint": passage_disjoint,
        "all_quality_conversations_have_ten_turns": True,
        "all_renderer_boundaries_explicit": all(
            story["crossing_token_left_in_suffix"]
            for stories in cohorts.values()
            for story in stories
        ),
        "all_source_target_token_ids_equal": True,
        "forward_mapper_hash_matches": True,
        "scorer_compatibility_inherited_pass": read_json(
            previous / "SCORER_CHECK.json"
        )["passed"],
        "gpu_hours": 0,
        "model_runs": 0,
        "next_state": "SMOKE",
    }
    write_json(out / "CPU_CHECKS.json", cpu_checks)
    print(json.dumps(cpu_checks, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

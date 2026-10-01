#!/usr/bin/env python3
"""Freeze the CPU-only inputs and receipts for the CoQA handoff study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import time

from . import coqa


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def write_jsonl(path: Path, rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def mapper_info(path: str, direction: str):
    from safetensors import safe_open

    with safe_open(path, framework="pt") as handle:
        metadata = handle.metadata()
        w_shape = list(handle.get_slice("W").get_shape())
        b_shape = list(handle.get_slice("b").get_shape())
    selected = json.loads(metadata["selected"])
    distinct_k = sorted({len(row) for row in selected})
    if len(distinct_k) != 1:
        raise ValueError(f"{direction} mapper has inconsistent selected-layer counts")
    return {
        "direction": direction,
        "path_at_freeze": os.path.realpath(path),
        "bytes": os.path.getsize(path),
        "sha256": file_sha256(path),
        "method": metadata["method"],
        "coefficient_dtype": metadata.get("coefficient_dtype"),
        "feature_width": int(metadata["feature_width"]),
        "lambda": float(metadata["lambda"]),
        "selected_layers_per_target": distinct_k[0],
        "selected": selected,
        "W_shape": w_shape,
        "b_shape": b_shape,
        "metadata": metadata,
    }


def model_info(name: str, revision: str):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(name, revision=revision)
    if not hasattr(cfg, "head_dim"):
        raise ValueError(
            f"{name} config has no explicit head_dim; this protocol forbids deriving it"
        )
    return {
        "name": name,
        "revision": revision,
        "model_type": cfg.model_type,
        "layers": int(cfg.num_hidden_layers),
        "attention_heads": int(cfg.num_attention_heads),
        "kv_heads": int(cfg.num_key_value_heads),
        "head_dim": int(cfg.head_dim),
        "rope_theta": float(cfg.rope_theta),
        "vocab_size": int(cfg.vocab_size),
        "max_position_embeddings": int(cfg.max_position_embeddings),
    }


def scorer_check(evaluator_path: str, dataset: dict):
    official = runpy.run_path(evaluator_path)["CoQAEvaluator"]
    first = dataset["data"][0]
    fixtures = [
        ("unknown", ["unknown"], "unknown"),
        ("empty", [""], ""),
        ("normalization", ["The red, fox."], "red fox"),
        ("leave_one_out", ["red", "blue", "green"], "red"),
        (
            "dataset_multiple_reference",
            coqa.answer_references(first, 1),
            first["answers"][0]["input_text"],
        ),
    ]
    rows = []
    for name, references, prediction in fixtures:
        ours = coqa.turn_score(references, prediction)
        theirs = official._compute_turn_score(references, prediction)
        delta = {
            key: abs(float(ours[key]) - float(theirs[key])) for key in ("em", "f1")
        }
        rows.append(
            {
                "name": name,
                "references": references,
                "prediction": prediction,
                "ours": ours,
                "official": theirs,
                "absolute_delta": delta,
                "passed": max(delta.values()) <= 1e-15,
            }
        )
    return {
        "schema": "coqa_scorer_check_v1",
        "official_evaluator_sha256": file_sha256(evaluator_path),
        "fixtures": rows,
        "passed": all(row["passed"] for row in rows),
        "multiple_reference_rule": (
            "hold out each reference in turn, score against the best remaining "
            "reference, then average over held-out references"
        ),
    }


def compact_story(story: dict, selected_for: list[str]):
    return {
        "id": story["id"],
        "source": story["source"],
        "selection_sha256": coqa.selection_digest(story["id"]),
        "selected_for": selected_for,
        "story": story["story"],
        "questions": story["questions"],
        "answers": story["answers"],
        "additional_answers": story.get("additional_answers", {}),
    }


def row_for(story: dict, turn: int, source_tok, target_tok, split: str):
    prompt = coqa.render_prompt(story, turn)
    source_ids = source_tok.encode(prompt, add_special_tokens=False)
    target_ids = target_tok.encode(prompt, add_special_tokens=False)
    if source_ids != target_ids:
        raise ValueError(
            f"tokenizers disagree for {story['id']} turn {turn}; transfer needs shared IDs"
        )
    if len(target_ids) < 2:
        raise ValueError(f"prompt {story['id']} turn {turn} has fewer than two tokens")
    return {
        "row_id": f"{story['id']}:{turn}",
        "split": split,
        "conversation_id": story["id"],
        "domain": story["source"],
        "turn": turn,
        "prompt": prompt,
        "token_ids": target_ids,
        "prefix_token_count": len(target_ids) - 1,
        "final_token_id": target_ids[-1],
        "token_ids_sha256": hashlib.sha256(
            b"".join(
                int(token).to_bytes(8, "little", signed=False) for token in target_ids
            )
        ).hexdigest(),
        "references": coqa.answer_references(story, turn),
        "primary_reference_history": coqa.primary_answers(story)[: turn - 1],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--forward-mapper", required=True)
    parser.add_argument("--reverse-mapper", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--source-repo",
        help="git checkout/object store in which --source-commit must exist",
    )
    parser.add_argument(
        "--source-bundle", help="optional replay bundle to archive with Q0"
    )
    parser.add_argument(
        "--guidance-commit",
        required=True,
        help="provenance identifier for the experiment protocol",
    )
    parser.add_argument("--modal-gpu-rate", type=float, default=4.54)
    parser.add_argument("--modal-cpu-rate", type=float, default=0.0473)
    parser.add_argument("--modal-memory-rate", type=float, default=0.008)
    args = parser.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    inputs = out / "inputs"
    inputs.mkdir()

    with open(args.data, encoding="utf-8") as stream:
        dataset = json.load(stream)
    if dataset.get("version") != "1.0":
        raise ValueError(f"expected CoQA 1.0, got {dataset.get('version')!r}")

    from transformers import AutoTokenizer

    source_name = "Qwen/Qwen3-14B"
    source_revision = "40c069824f4251a91eefaf281ebe4c544efd3e18"
    target_name = "Qwen/Qwen3-32B"
    target_revision = "9216db5781bf21249d130ec9da846c4624c16137"
    source_tok = AutoTokenizer.from_pretrained(source_name, revision=source_revision)
    target_tok = AutoTokenizer.from_pretrained(target_name, revision=target_revision)

    selection = coqa.select_conversations(dataset)
    q1_rows = [
        row_for(story, turn, source_tok, target_tok, "q1")
        for story in selection["q1"]
        for turn in coqa.Q1_TURNS
        if turn <= len(story["questions"])
    ]
    smoke_rows = [
        row_for(story, turn, source_tok, target_tok, "smoke")
        for story in selection["smoke"]
        for turn in coqa.Q1_TURNS
        if turn <= len(story["questions"])
    ]

    q1_ids = {story["id"] for story in selection["q1"]}
    q2_ids = {story["id"] for story in selection["q2"]}
    smoke_ids = {story["id"] for story in selection["smoke"]}
    conversations = []
    for story in dataset["data"]:
        tags = []
        if story["id"] in q1_ids:
            tags.append("q1")
        if story["id"] in q2_ids:
            tags.append("q2")
        if story["id"] in smoke_ids:
            tags.append("smoke")
        if tags:
            conversations.append(compact_story(story, tags))
    conversations.sort(
        key=lambda row: (row["source"], row["selection_sha256"], row["id"])
    )

    q1_path = out / "Q1_FROZEN_ROWS.jsonl"
    smoke_path = out / "SMOKE_FROZEN_ROWS.jsonl"
    conv_path = out / "FROZEN_CONVERSATIONS.json"
    write_jsonl(q1_path, q1_rows)
    write_jsonl(smoke_path, smoke_rows)
    write_json(conv_path, conversations)
    shutil.copy2(args.data, inputs / "coqa-dev-v1.0.json")
    shutil.copy2(args.evaluator, inputs / "evaluate-v1.0.py")

    scorer = scorer_check(args.evaluator, dataset)
    if not scorer["passed"]:
        raise RuntimeError("internal CoQA scorer differs from the official evaluator")
    write_json(out / "SCORER_CHECK.json", scorer)

    source_model = model_info(source_name, source_revision)
    target_model = model_info(target_name, target_revision)
    if source_model["kv_heads"] != target_model["kv_heads"]:
        raise ValueError("model pair does not have matching KV head counts")
    if source_model["head_dim"] != 128 or target_model["head_dim"] != 128:
        raise ValueError(
            f"unexpected explicit head dimensions: {source_model}, {target_model}"
        )

    mapper_manifest = {
        "schema": "coqa_mapper_manifest_v1",
        "forward": mapper_info(args.forward_mapper, "Qwen3-14B_to_Qwen3-32B"),
        "reverse": mapper_info(args.reverse_mapper, "Qwen3-32B_to_Qwen3-14B"),
        "fitting_permitted": False,
    }
    if mapper_manifest["forward"]["selected_layers_per_target"] != 8:
        raise ValueError("forward mapper is not the frozen k=8 artifact")
    if mapper_manifest["reverse"]["selected_layers_per_target"] != 20:
        raise ValueError("reverse mapper is not the frozen k=20 artifact")
    write_json(out / "MAPPER_MANIFEST.json", mapper_manifest)

    input_manifest = {
        "schema": "coqa_input_manifest_v1",
        "official_data": {
            "url": "https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json",
            "sha256": file_sha256(args.data),
            "bytes": os.path.getsize(args.data),
            "version": dataset["version"],
            "conversations": len(dataset["data"]),
        },
        "official_evaluator": {
            "url": "https://nlp.stanford.edu/data/coqa/evaluate-v1.0.py",
            "sha256": file_sha256(args.evaluator),
            "bytes": os.path.getsize(args.evaluator),
        },
        "models": {"source": source_model, "target": target_model},
        "tokenizer": {
            "source_class": type(source_tok).__name__,
            "target_class": type(target_tok).__name__,
            "vocab_size": len(target_tok),
            "eos_token_id": target_tok.eos_token_id,
            "bos_token_id": target_tok.bos_token_id,
            "same_token_ids_on_all_frozen_q1_and_smoke_rows": True,
        },
        "selection": {
            "salt": coqa.SELECTION_SALT,
            "ordering": "ascending SHA256(salt + conversation_id), then id",
            "q1_per_domain": 20,
            "smoke_per_domain": 1,
            "smoke_position_per_domain_one_indexed": 21,
            "q2_per_domain": 4,
            "q2_is_first_four_q1_per_domain": True,
            "q1_conversation_ids": [story["id"] for story in selection["q1"]],
            "smoke_conversation_ids": [story["id"] for story in selection["smoke"]],
            "q2_conversation_ids": [story["id"] for story in selection["q2"]],
        },
        "artifacts": {
            path.name: {"sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in (q1_path, smoke_path, conv_path)
        },
        "counts": {
            "q1_conversations": len(selection["q1"]),
            "q1_paired_questions": len(q1_rows),
            "q1_max_answers": 2 * len(q1_rows),
            "smoke_conversations": len(selection["smoke"]),
            "smoke_q1_style_questions": len(smoke_rows),
            "q2_conversations": len(selection["q2"]),
            "q2_max_turn_positions": sum(
                min(10, len(story["questions"])) for story in selection["q2"]
            ),
        },
    }
    write_json(out / "INPUT_MANIFEST.json", input_manifest)

    protocol = {
        "schema": "coqa_cross_model_protocol_v1",
        "guidance_commit": args.guidance_commit,
        "prompt": {
            "instruction": coqa.INSTRUCTION,
            "rendering": (
                "instruction, blank line, 'Passage: {story}', blank line, then ordered "
                "'Question: ...'/'Answer: ...' lines; current prompt ends exactly in 'Answer:'"
            ),
            "chat_wrapper": False,
            "thinking_tokens_added": False,
        },
        "decoding": {
            "greedy": True,
            "temperature": None,
            "max_new_tokens": coqa.MAX_NEW_TOKENS,
            "stop": ["first decoded newline", "model EOS", "64-token cap"],
            "newline_token_is_retained_in_raw_token_ids": True,
            "answer_to_scorer": "decoded text strictly before the first newline, stripped",
        },
        "q1": {
            "turns": list(coqa.Q1_TURNS),
            "past_answers": "primary official reference answers",
            "arms": ["native_32B_split_prefill", "FullHead_14B_to_32B"],
            "cache_cut": "all prompt tokens except final token; current question included",
            "fresh_source_history_per_question": True,
            "screen": {
                "equal_domain_mean_f1_deficit_max": 0.03,
                "each_domain_f1_deficit_max": 0.05,
                "pooled_unhealthy_excess_max": 0.02,
            },
        },
        "smoke": {
            "q1_style_turns": list(coqa.Q1_TURNS),
            "native_control": "direct full-prompt generation versus native split-prefill",
            "recursive_ledger_check": "both alternating schedules through two turns on all five smoke conversations",
            "scores_are_not_selection_inputs": True,
        },
        "q2": {
            "admission": "Q1 point-estimate screen only",
            "turns": "up to first ten available",
            "schedules": ["first_receiver_32B", "first_receiver_14B"],
            "streams": [
                "translated alternating trajectory",
                "same-input native receiver shadow",
                "native 14B-only trajectory",
                "native 32B-only trajectory",
            ],
            "token_ledger": (
                "append generated token IDs including newline/EOS/cap-final token; never "
                "re-tokenize cached history; before handoff the owner consumes every pending "
                "token and next question/cue except the final cue token"
            ),
            "native_shadow": "recompute from exact ledger; generated shadow answer discarded",
            "history": "generated trajectory-specific answers, never references",
        },
        "health": {
            "unhealthy_events": [
                "empty",
                "unterminated at cap",
                "unterminated <think> block",
                "six repeats of a 24-character span",
            ],
            "reported_separately": ["newline stop", "EOS stop", "cap hit"],
        },
        "statistics": {
            "official_score": "official CoQA leave-one-reference-out F1",
            "bootstrap_unit": "complete conversations resampled within domain",
            "seed": coqa.BOOTSTRAP_SEED,
            "replicates": coqa.BOOTSTRAP_REPLICATES,
            "interval_use": "descriptive only",
        },
        "forbidden": [
            "mapper fitting or tuning",
            "sample expansion",
            "prompt or stop-rule retry after outcomes",
            "CacheBridge or another method/model/task",
            "storage benchmark as quality evidence",
        ],
    }
    write_json(out / "PROTOCOL.json", protocol)

    paper_gaps = {
        "schema": "coqa_paper_gaps_v1",
        "primary_source": "arXiv:2608.03893v1 section 4.6 and Figure 4",
        "paper_facts": [
            "Qwen3-14B and Qwen3-32B alternate across turns in both directions",
            "100 CoQA conversations across five domains, approximately 15 turns each",
            "answers are scored with F1 against ground truth",
            "drift is standalone-target F1 minus mapped-cache F1 at the same turn",
            "reported single-task k values are 8 forward and 20 reverse",
        ],
        "not_specified_in_paper": [
            "conversation IDs or selection rule",
            "prompt text and whether a chat template was used",
            "whether past answers are gold or generated",
            "precise cache cut and stop-token handling",
            "whether mappings compose recursively or histories are rebuilt",
            "decoding parameters and answer-token cap",
            "official evaluator version and multiple-reference handling",
            "absolute per-domain quality and generation-health criteria",
        ],
        "reconstruction_defaults": [
            "mechanically selected official dev conversations",
            "plain completion prompt frozen in PROTOCOL.json",
            "Q1 gold history and Q2 generated recursive history",
            "question-containing prefix cut one token before prompt end",
            "greedy generation through newline, EOS, or 64 tokens",
            "official evaluator SHA pinned in INPUT_MANIFEST.json",
        ],
        "claim": "disclosed reconstruction, not exact author-code replication",
    }
    write_json(out / "PAPER_GAPS.json", paper_gaps)

    repo = (
        Path(args.source_repo)
        if args.source_repo
        else Path(__file__).resolve().parents[3]
    )
    commit_exists = (
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "cat-file",
                "-e",
                f"{args.source_commit}^{{commit}}",
            ]
        ).returncode
        == 0
    )
    if not commit_exists:
        raise ValueError(f"source commit {args.source_commit} is unavailable in {repo}")
    bundle = None
    if args.source_bundle:
        bundle_path = Path(args.source_bundle)
        bundle_target = out / f"knlp-{args.source_commit[:12]}.bundle"
        shutil.copy2(bundle_path, bundle_target)
        bundle = {
            "file": bundle_target.name,
            "bytes": bundle_target.stat().st_size,
            "sha256": file_sha256(bundle_target),
        }
    execution = {
        "schema": "coqa_execution_source_v1",
        "knlp_commit": args.source_commit,
        "commit_exists_in_source_repo": commit_exists,
        "source_bundle": bundle,
        "entrypoints": [
            "research/kv_translate/published/freeze_coqa.py",
            "research/kv_translate/published/coqa_runner.py",
            "research/kv_translate/published/coqa_modal_app.py",
        ],
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(out / "EXECUTION_SOURCE.json", execution)

    hourly = args.modal_gpu_rate + 4 * args.modal_cpu_rate + 48 * args.modal_memory_rate
    stages = {
        "cached_startup_and_load": 0.15,
        "smoke": 0.50,
        "q1": 1.50,
        "conditional_q2": 2.50,
        "export": 0.25,
    }
    base_hours = sum(stages.values())
    contingency_hours = base_hours * 0.30
    forecast = {
        "schema": "coqa_cost_forecast_v1",
        "provider": "modal",
        "resource": {"gpu": "H200", "cpu_cores": 4, "memory_gib": 48},
        "live_rates_usd_per_hour": {
            "gpu": args.modal_gpu_rate,
            "cpu_per_core": args.modal_cpu_rate,
            "memory_per_gib": args.modal_memory_rate,
            "combined_requested": hourly,
        },
        "planning_hours_before_contingency": stages,
        "contingency_fraction": 0.30,
        "planning_envelope_hours": base_hours + contingency_hours,
        "planning_envelope_usd": round((base_hours + contingency_hours) * hourly, 2),
        "first_allocation": "smoke only; serverless function ends when smoke finishes",
        "initial_lifecycle_checkpoint_hours": 1,
        "watchdog": "each Modal function has a stage-specific timeout; Q1/Q2 launch only after prior receipt",
        "reforecast": "replace Q1/Q2 planning times with measured smoke throughput before Q1",
        "not_a_completion_time_claim": True,
        "models_already_cached_on_knlp_hf_volume": True,
        "billing_note": "CPU and memory are additive to the card rate",
    }
    write_json(out / "COST_FORECAST.json", forecast)

    validation = {
        "schema": "coqa_q0_validation_v1",
        "status": "PASS",
        "receipts": sorted(path.name for path in out.iterdir() if path.is_file()),
        "scorer_matches_official": scorer["passed"],
        "q1_answer_bound": 2 * len(q1_rows),
        "q2_answer_bound": 6
        * sum(min(10, len(story["questions"])) for story in selection["q2"]),
        "gpu_hours": 0,
        "model_runs": 0,
        "next_state": "SMOKE",
    }
    write_json(out / "Q0_VALIDATION.json", validation)
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

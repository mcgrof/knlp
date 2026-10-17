"""Capped five-task evaluation with durable paired modified-MoT records."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(_sha256(item)))
    return digest.hexdigest()


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return float("nan")
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def aggregate_paired_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_task.setdefault(record["task"], []).append(record)
    result: dict[str, Any] = {}
    for task, rows in sorted(by_task.items()):
        complete = len(rows) == 100 and all(
            row["candidate"].get("score") is not None
            and row["native"].get("score") is not None
            for row in rows
        )
        task_result: dict[str, Any] = {
            "item_count": len(rows),
            "cluster_count": len({row["document_group"] for row in rows}),
            "operational_complete": complete,
            "candidate_mean": None,
            "native_mean": None,
            "candidate_minus_native": None,
            "paired_bootstrap_95_ci": None,
            "invalid_candidate_count": sum(row["candidate"].get("parse_status") != "parsed" for row in rows),
            "invalid_native_count": sum(row["native"].get("parse_status") != "parsed" for row in rows),
            "candidate_truncated_count": sum(bool(row.get("truncation_status")) for row in rows),
        }
        if complete:
            candidate = [float(row["candidate"]["score"]) for row in rows]
            native = [float(row["native"]["score"]) for row in rows]
            differences = [left - right for left, right in zip(candidate, native)]
            task_result["candidate_mean"] = sum(candidate) / len(candidate)
            task_result["native_mean"] = sum(native) / len(native)
            task_result["candidate_minus_native"] = sum(differences) / len(differences)
            clusters: dict[str, list[float]] = {}
            for row, difference in zip(rows, differences):
                clusters.setdefault(str(row["document_group"]), []).append(difference)
            cluster_names = sorted(clusters)
            rng = random.Random(42)
            replicates = []
            for _ in range(10_000):
                sampled = [rng.choice(cluster_names) for _ in cluster_names]
                values = [value for name in sampled for value in clusters[name]]
                replicates.append(sum(values) / len(values))
            replicates.sort()
            task_result["paired_bootstrap_95_ci"] = [
                _percentile(replicates, 0.025),
                _percentile(replicates, 0.975),
            ]
            if task in {"boolq", "pubmedqa", "mmlu_redux"}:
                task_result["native_only_correct"] = sum(n == 1.0 and c == 0.0 for c, n in zip(candidate, native))
                task_result["candidate_only_correct"] = sum(c == 1.0 and n == 0.0 for c, n in zip(candidate, native))
            else:
                task_result["candidate_win_tie_loss"] = {
                    "win": sum(value > 0 for value in differences),
                    "tie": sum(value == 0 for value in differences),
                    "loss": sum(value < 0 for value in differences),
                }
        result[task] = task_result
    return result


def replay_paired_jsonl(input_jsonl: Path, output_json: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in input_jsonl.read_text(encoding="utf-8").splitlines() if line]
    aggregates = aggregate_paired_records(records)
    payload = {
        "schema": "knlp_mot_capped_diagnostic_replay_v1",
        "raw_records_sha256": _sha256(input_jsonl),
        "record_count": len(records),
        "aggregates": aggregates,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def run_capped_diagnostic_eval(
    *, source: Path, checkpoint: Path, data_manifest: Path, output: Path
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve()
    data_manifest = data_manifest.expanduser().resolve()
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    torch = importlib.import_module("torch")
    common = importlib.import_module("core.common")
    eval_util = importlib.import_module("core.eval_util")
    mot_train = importlib.import_module("alg.mot.train")

    ctx, translator_pool = mot_train.load_translator_pool_from_checkpoint(
        str(checkpoint), device_override="cuda", dtype_override="bfloat16"
    )
    translator_pool.eval()
    for node in ctx.nodes:
        ctx.tp.get_model(node.id).eval()
    if len(ctx.edges) != 1:
        raise RuntimeError("The capped diagnostic expects exactly one direction")
    edge = ctx.edges[0]
    source_node = edge.src_id
    target_node = edge.tgt_id
    source_model = ctx.tp.get_model(source_node)
    target_model = ctx.tp.get_model(target_node)
    manifest = json.loads(data_manifest.read_text(encoding="utf-8"))
    task_specs = {
        "boolq": eval_util.get_boolq_dataset_spec(),
        "pubmedqa": eval_util.get_pubmedqa_dataset_spec(),
        "mmlu_redux": eval_util.get_mmlu_redux_dataset_spec(),
        "squad_v1_1": eval_util.get_squad_v11_dataset_spec(),
        "newsqa": eval_util.get_newsqa_generation_dataset_spec(),
    }
    code_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, check=True, text=True, capture_output=True
    ).stdout.strip()
    checkpoint_sha256 = _tree_hash(checkpoint)
    scorer_hash = hashlib.sha256(
        (Path(eval_util.__file__).read_bytes() + Path(__file__).read_bytes())
    ).hexdigest()
    run_manifest = {
        "schema": "knlp_mot_capped_diagnostic_run_v1",
        "label": "modified_mot_capped_diagnostic_not_paper_reproduction",
        "source_commit": code_commit,
        "data_manifest_sha256": _sha256(data_manifest),
        "checkpoint_tree_sha256": checkpoint_sha256,
        "scorer_sha256": scorer_hash,
        "generation": {"method": "greedy", "max_new_tokens": 64},
        "omitted_auxiliary_jobs": [
            "openwebtext_validation",
            "stock_inference_profiling",
            "tsne",
        ],
        "task_item_counts": {
            task: cohort["item_count"] for task, cohort in manifest["evaluation"].items()
        },
        "mmlu_subject_counts": manifest["evaluation"]["mmlu_redux"]["subject_counts"],
    }
    run_manifest_path = output / "RUN_MANIFEST.json"
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run_manifest_sha256 = _sha256(run_manifest_path)

    def token_ids(values: list[int] | None, model_id: str):
        if values is None:
            return None
        return common.TokenIDs(torch.tensor([values], dtype=torch.long, device="cuda"), model_id=model_id)

    @torch.inference_mode()
    def generate(model, past, seed, prompt):
        if prompt is not None:
            past = eval_util.append_token_ids_to_past(
                model=model, past_key_values=past, token_ids=prompt
            )
        generated = []
        current = seed
        eos = model.tokenizer.eos_token_id
        eos_reached = False
        for _ in range(64):
            outputs = model(
                input_ids=current.as_tensor(), past_key_values=past, use_cache=True
            )
            next_token = common.TokenIDs(
                outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True),
                model_id=current.model_id,
            )
            value = int(next_token.item())
            if eos is not None and value == eos:
                eos_reached = True
                break
            generated.append(value)
            current = next_token
            past = outputs.past_key_values
        raw = model.tokenizer.decode(generated, skip_special_tokens=True)
        return {
            "generated_token_ids": generated,
            "raw_decoded_text": raw,
            "postprocessed_text": eval_util.postprocess_generated_answer(raw),
            "eos_reached": eos_reached,
            "length_limit_reached": len(generated) == 64 and not eos_reached,
        }

    records: list[dict[str, Any]] = []
    raw_path = output / "PAIRED_RECORDS.jsonl"
    started = time.monotonic()
    with raw_path.open("w", encoding="utf-8", buffering=1) as raw_handle:
        for task, cohort in manifest["evaluation"].items():
            spec = task_specs[task]
            for item in cohort["items"]:
                row_started = time.monotonic()
                target_ledger = item["token_ledgers"]["target"]
                source_ledger = item["token_ledgers"]["source"]
                target_context = token_ids(target_ledger["context_token_ids"], target_model.id)
                source_context = token_ids(source_ledger["context_token_ids"], source_model.id)
                prompt = token_ids(target_ledger["prompt_token_ids"], target_model.id)
                seed = token_ids(target_ledger["seed_token_ids"], target_model.id)
                record = {
                    "schema": "knlp_mot_capped_diagnostic_pair_v1",
                    "item_id": item["item_id"],
                    "task": task,
                    "subject": item.get("subject"),
                    "document_group": item["document_group"],
                    "run_manifest_sha256": run_manifest_sha256,
                    "data_manifest_sha256": run_manifest["data_manifest_sha256"],
                    "code_commit": code_commit,
                    "checkpoint_tree_sha256": checkpoint_sha256,
                    "scorer_sha256": scorer_hash,
                    "gold_answers": item["input"].get("answers", item["input"].get("answer")),
                    "question": item["input"]["question"],
                    "choices": item["input"].get("choices"),
                    "context_token_ids": target_ledger["context_token_ids"],
                    "prompt_token_ids": target_ledger["prompt_token_ids"],
                    "seed_token_ids": target_ledger["seed_token_ids"],
                    "executed_context_lengths": {
                        "source": len(source_ledger["context_token_ids"]),
                        "target": len(target_ledger["context_token_ids"]),
                        "prompt": 0 if target_ledger["prompt_token_ids"] is None else len(target_ledger["prompt_token_ids"]),
                    },
                    "truncation_status": bool(target_ledger["was_truncated"]),
                    "alignment_decision": "identity",
                    "branch_counts": {
                        "bottom_full_attention_layers": int(ctx.config.num_bottom_full_attn),
                        "reduced_sparse_layers": (
                            max(0, ctx.tp.get_model_spec(target_node).num_layers - int(ctx.config.num_bottom_full_attn))
                            if len(target_ledger["context_token_ids"]) > int(ctx.config.topk_sparse_attn)
                            else 0
                        ),
                        "full_key_or_native_layers": (
                            int(ctx.config.num_bottom_full_attn)
                            if len(target_ledger["context_token_ids"]) > int(ctx.config.topk_sparse_attn)
                            else ctx.tp.get_model_spec(target_node).num_layers
                        ),
                    },
                    "candidate": {"score": None, "parse_status": "operational_failure"},
                    "native": {"score": None, "parse_status": "operational_failure"},
                    "operational_error": None,
                }
                try:
                    if source_context.as_tensor().tolist() != target_context.as_tensor().tolist():
                        raise RuntimeError("Frozen source/target context ledgers are not identical")
                    source_past = common.extract_past_key_values(source_model, source_context)
                    native_past = common.extract_past_key_values(target_model, target_context)
                    candidate_past, _ = mot_train.build_replayed_target_past(
                        ctx,
                        source_past_key_values=source_past,
                        source_context_token_ids=source_context,
                        target_context_token_ids=target_context,
                        source_model=source_model,
                        target_model=target_model,
                        src_node_id=source_node,
                        tgt_node_id=target_node,
                        tgt_spec=ctx.tp.get_model_spec(target_node),
                    )
                    candidate = generate(target_model, candidate_past, seed, prompt)
                    native = generate(target_model, native_past, seed, prompt)
                    if task in {"boolq", "pubmedqa", "mmlu_redux"}:
                        candidate_parsed = eval_util.parse_generated_logit_answer(
                            spec, candidate["postprocessed_text"], choices=item["input"].get("choices")
                        )
                        native_parsed = eval_util.parse_generated_logit_answer(
                            spec, native["postprocessed_text"], choices=item["input"].get("choices")
                        )
                        gold = item["input"]["answer"]
                        candidate_score = 1.0 if candidate_parsed is not None and eval_util.is_logit_answer_correct(candidate_parsed, gold) else 0.0
                        native_score = 1.0 if native_parsed is not None and eval_util.is_logit_answer_correct(native_parsed, gold) else 0.0
                    else:
                        candidate_parsed = candidate["postprocessed_text"] or None
                        native_parsed = native["postprocessed_text"] or None
                        gold = item["input"]["answers"]
                        candidate_score = float(eval_util.compute_generation_f1(candidate["postprocessed_text"], gold))
                        native_score = float(eval_util.compute_generation_f1(native["postprocessed_text"], gold))
                    candidate.update(
                        parsed_answer=candidate_parsed,
                        parse_status="parsed" if candidate_parsed is not None else "invalid_answer",
                        score=candidate_score,
                    )
                    native.update(
                        parsed_answer=native_parsed,
                        parse_status="parsed" if native_parsed is not None else "invalid_answer",
                        score=native_score,
                    )
                    record["candidate"] = candidate
                    record["native"] = native
                except Exception as error:
                    record["operational_error"] = f"{type(error).__name__}: {error}"
                record["elapsed_seconds"] = time.monotonic() - row_started
                raw_handle.write(json.dumps(record, sort_keys=True) + "\n")
                records.append(record)

    aggregates = aggregate_paired_records(records)
    summary = {
        "schema": "knlp_mot_capped_diagnostic_summary_v1",
        "record_count": len(records),
        "expected_record_count": 500,
        "operational_complete": len(records) == 500 and all(
            record["operational_error"] is None for record in records
        ),
        "raw_records_sha256": _sha256(raw_path),
        "run_manifest_sha256": run_manifest_sha256,
        "elapsed_seconds": time.monotonic() - started,
        "aggregates": aggregates,
    }
    summary_path = output / "SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    replay_paired_jsonl(raw_path, output / "REPLAY.json")
    return summary

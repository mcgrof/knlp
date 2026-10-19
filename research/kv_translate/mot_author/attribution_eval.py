"""Frozen four-arm attribution evaluation for modified MoT replay."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Iterable

from .diagnostic_data import SCHEMA as DIAGNOSTIC_DATA_SCHEMA
from .diagnostic_eval import _canonical_sha256, _percentile, _sha256, _tree_hash
from .recipe import FROZEN_DIAGNOSTIC_CONTENT_SHA256


ARMS = ("N", "T", "R", "O")
PAIRS = (("T", "R"), ("T", "O"), ("R", "N"), ("O", "R"))


def _git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".zst":
        text = subprocess.run(
            ["zstd", "-q", "-dc", str(path)],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
    else:
        text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


def _tensor_ledger_sha256(tensors: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for layer_index, tensor in enumerate(tensors):
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(str(layer_index).encode("ascii") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _cluster_key(row: dict[str, Any]) -> str:
    if row["task"] == "squad_v1_1":
        return "context:" + _canonical_sha256(row["context_token_ids"])
    if row["task"] == "mmlu_redux":
        return "subject:" + str(row["subject"])
    return "document:" + str(row["document_group"])


def _paired_cluster_interval(
    rows: list[dict[str, Any]], left: str, right: str
) -> dict[str, Any]:
    clusters: dict[str, list[float]] = {}
    for row in rows:
        difference = float(row["arms"][left]["score"]) - float(
            row["arms"][right]["score"]
        )
        clusters.setdefault(_cluster_key(row), []).append(difference)
    names = sorted(clusters)
    rng = random.Random(42)
    replicates: list[float] = []
    for _ in range(10_000):
        sampled = [rng.choice(names) for _ in names]
        values = [value for name in sampled for value in clusters[name]]
        replicates.append(sum(values) / len(values))
    replicates.sort()
    differences = [
        float(row["arms"][left]["score"]) - float(row["arms"][right]["score"])
        for row in rows
    ]
    return {
        "difference": sum(differences) / len(differences),
        "paired_cluster_bootstrap_95_ci": [
            _percentile(replicates, 0.025),
            _percentile(replicates, 0.975),
        ],
        "cluster_count": len(names),
        "cluster_sizes": sorted(len(clusters[name]) for name in names),
        "win_tie_loss": {
            "win": sum(value > 0 for value in differences),
            "tie": sum(value == 0 for value in differences),
            "loss": sum(value < 0 for value in differences),
        },
    }


def aggregate_attribution_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_task.setdefault(record["task"], []).append(record)
    result: dict[str, Any] = {}
    for task, rows in sorted(by_task.items()):
        complete = len(rows) == 100 and all(
            row.get("operational_error") is None
            and isinstance(row["arms"][arm].get("score"), (int, float))
            and math.isfinite(float(row["arms"][arm]["score"]))
            and 0.0 <= float(row["arms"][arm]["score"]) <= 1.0
            for row in rows
            for arm in ARMS
        )
        task_result: dict[str, Any] = {
            "item_count": len(rows),
            "cluster_count": len({_cluster_key(row) for row in rows}),
            "operational_complete": complete,
            "arm_means": {},
            "invalid_answer_counts": {},
            "paired_differences": {},
        }
        if complete:
            for arm in ARMS:
                scores = [float(row["arms"][arm]["score"]) for row in rows]
                task_result["arm_means"][arm] = sum(scores) / len(scores)
                task_result["invalid_answer_counts"][arm] = sum(
                    row["arms"][arm].get("parse_status") != "parsed" for row in rows
                )
            for left, right in PAIRS:
                task_result["paired_differences"][f"{left}-{right}"] = (
                    _paired_cluster_interval(rows, left, right)
                )
        result[task] = task_result
    return result


def replay_attribution_jsonl(input_jsonl: Path, output_json: Path) -> dict[str, Any]:
    records = _read_jsonl(input_jsonl)
    item_ids = [str(record["item_id"]) for record in records]
    payload = {
        "schema": "knlp_mot_attribution_replay_v1",
        "raw_records_sha256": _sha256(input_jsonl),
        "record_count": len(records),
        "scored_output_count": len(records) * len(ARMS),
        "unique_item_id_count": len(set(item_ids)),
        "item_ids_unique": len(item_ids) == len(set(item_ids)),
        "all_arm_score_contract_pass": all(
            record.get("operational_error") is None
            and isinstance(record["arms"][arm].get("score"), (int, float))
            and math.isfinite(float(record["arms"][arm]["score"]))
            and 0.0 <= float(record["arms"][arm]["score"]) <= 1.0
            for record in records
            for arm in ARMS
        ),
        "aggregates": aggregate_attribution_records(records),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def _checkpoint_model_file(checkpoint: Path) -> Path:
    files = sorted((checkpoint / "translators").glob("*.pt"))
    if len(files) != 1:
        raise RuntimeError(
            f"Expected exactly one translator checkpoint file, found {len(files)}"
        )
    return files[0]


def build_attribution_pasts(
    *,
    mot_train,
    common,
    ctx,
    source_model,
    target_model,
    source_node: str,
    target_node: str,
    source_spec,
    target_spec,
    source_context,
    target_context,
    source_past,
    native_past,
) -> tuple[dict[str, Any], list[Any], dict[str, float]]:
    """Build T/R/O from one immutable selection ledger.

    N is the caller-provided native cache. R passes no injection layers and no
    injected tensors. O extracts only the configured target layers from N.
    """
    source_layers = ctx.cm.get_src_layer_indices(ctx.edges[0].id)
    target_layers = ctx.cm.get_tgt_layer_indices(ctx.edges[0].id)
    phase_seconds: dict[str, float] = {}

    phase_started = time.monotonic()
    selected_indices = mot_train.build_extrapolated_sparse_attention_indices(
        source_model,
        source_context,
        source_layer_indices=source_layers,
        target_layer_indices=target_layers,
        num_source_layers=source_spec.num_layers,
        num_target_layers=target_spec.num_layers,
        source_model_id=source_model.id,
        top_k=ctx.config.topk_sparse_attn,
    )
    phase_seconds["source_selection"] = time.monotonic() - phase_started

    phase_started = time.monotonic()
    translated_past, _ = mot_train.build_replayed_target_past(
        ctx,
        source_past_key_values=source_past,
        source_context_token_ids=source_context,
        target_context_token_ids=target_context,
        source_model=source_model,
        target_model=target_model,
        src_node_id=source_node,
        tgt_node_id=target_node,
        tgt_spec=target_spec,
        sparse_attention_indices_override=selected_indices,
    )
    phase_seconds["T_replay"] = time.monotonic() - phase_started

    phase_started = time.monotonic()
    recomputed_past = mot_train.replay_target_prefill_with_injected_window(
        target_model=target_model,
        target_model_id=target_model.id,
        context_token_ids=target_context,
        target_layer_indices=[],
        injected_key_block=None,
        injected_value_block=None,
        tgt_spec=target_spec,
        sparse_attention_indices=selected_indices,
        num_bottom_full_attn=ctx.config.num_bottom_full_attn,
        attention_backend=ctx.config.attention_backend,
    )
    phase_seconds["R_replay"] = time.monotonic() - phase_started

    native_window = tuple(native_past[index] for index in target_layers)
    native_key, native_value = common.past_key_values_to_blocks(native_window)
    phase_started = time.monotonic()
    oracle_past = mot_train.replay_target_prefill_with_injected_window(
        target_model=target_model,
        target_model_id=target_model.id,
        context_token_ids=target_context,
        target_layer_indices=target_layers,
        injected_key_block=native_key,
        injected_value_block=native_value,
        tgt_spec=target_spec,
        sparse_attention_indices=selected_indices,
        num_bottom_full_attn=ctx.config.num_bottom_full_attn,
        attention_backend=ctx.config.attention_backend,
    )
    phase_seconds["O_replay"] = time.monotonic() - phase_started
    return (
        {
            "N": native_past,
            "T": translated_past,
            "R": recomputed_past,
            "O": oracle_past,
        },
        selected_indices,
        phase_seconds,
    )


def run_attribution_eval(
    *,
    source: Path,
    checkpoint: Path,
    checkpoint_file_sha256: str,
    data_manifest: Path,
    anchor_records: Path,
    output: Path,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve()
    data_manifest = data_manifest.expanduser().resolve()
    anchor_records = anchor_records.expanduser().resolve()
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    torch = importlib.import_module("torch")
    common = importlib.import_module("core.common")
    eval_util = importlib.import_module("core.eval_util")
    mot_train = importlib.import_module("alg.mot.train")

    checkpoint_model_file = _checkpoint_model_file(checkpoint)
    observed_checkpoint_file_sha256 = _sha256(checkpoint_model_file)
    if observed_checkpoint_file_sha256 != checkpoint_file_sha256:
        raise RuntimeError(
            "Translator checkpoint SHA-256 mismatch: "
            f"expected={checkpoint_file_sha256} "
            f"observed={observed_checkpoint_file_sha256}"
        )

    anchor_rows = _read_jsonl(anchor_records)
    anchors = {str(row["item_id"]): row for row in anchor_rows}
    if len(anchor_rows) != 500 or len(anchors) != 500:
        raise RuntimeError("Anchor records must contain 500 unique items")

    ctx, translator_pool = mot_train.load_translator_pool_from_checkpoint(
        str(checkpoint), device_override="cuda", dtype_override="bfloat16"
    )
    translator_pool.eval()
    for node in ctx.nodes:
        ctx.tp.get_model(node.id).eval()
    if len(ctx.edges) != 1:
        raise RuntimeError("The attribution diagnostic expects exactly one direction")
    edge = ctx.edges[0]
    source_node = edge.src_id
    target_node = edge.tgt_id
    source_model = ctx.tp.get_model(source_node)
    target_model = ctx.tp.get_model(target_node)
    source_spec = ctx.tp.get_model_spec(source_node)
    target_spec = ctx.tp.get_model_spec(target_node)
    source_layers = ctx.cm.get_src_layer_indices(edge.id)
    target_layers = ctx.cm.get_tgt_layer_indices(edge.id)

    manifest = json.loads(data_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != DIAGNOSTIC_DATA_SCHEMA:
        raise RuntimeError("Modified-MoT diagnostic manifest schema mismatch")
    declared_content_sha = manifest.get("manifest_content_sha256")
    manifest_without_sha = dict(manifest)
    manifest_without_sha.pop("manifest_content_sha256", None)
    actual_content_sha = _canonical_sha256(manifest_without_sha)
    if (
        declared_content_sha != actual_content_sha
        or actual_content_sha != FROZEN_DIAGNOSTIC_CONTENT_SHA256
    ):
        raise RuntimeError("Modified-MoT diagnostic manifest content is not admitted")
    expected_item_ids = [
        str(item["item_id"])
        for cohort in manifest["evaluation"].values()
        for item in cohort["items"]
    ]
    if len(expected_item_ids) != 500 or len(set(expected_item_ids)) != 500:
        raise RuntimeError("Frozen evaluation IDs must contain 500 unique items")
    if set(expected_item_ids) != set(anchors):
        raise RuntimeError("Anchor and frozen-manifest item IDs differ")

    task_specs = {
        "boolq": eval_util.get_boolq_dataset_spec(),
        "pubmedqa": eval_util.get_pubmedqa_dataset_spec(),
        "mmlu_redux": eval_util.get_mmlu_redux_dataset_spec(),
        "squad_v1_1": eval_util.get_squad_v11_dataset_spec(),
        "newsqa": eval_util.get_newsqa_generation_dataset_spec(),
    }
    knlp_root = Path(__file__).resolve().parents[3]
    source_commit = _git_head(source)
    knlp_commit = _git_head(knlp_root)
    checkpoint_tree_sha256 = _tree_hash(checkpoint)
    scorer_sha256 = hashlib.sha256(
        Path(eval_util.__file__).read_bytes() + Path(__file__).read_bytes()
    ).hexdigest()
    run_manifest = {
        "schema": "knlp_mot_attribution_run_v1",
        "label": "modified_mot_four_arm_attribution_not_paper_reproduction",
        "arms": {
            "N": "exact native receiver context cache",
            "T": "selected-key replay with translated KV injection",
            "R": "selected-key replay without KV injection",
            "O": "selected-key replay with native receiver KV injection",
        },
        "source_commit": source_commit,
        "knlp_commit": knlp_commit,
        "attention_backend": str(ctx.config.attention_backend),
        "attention_backend_source_sha256": _sha256(source / "alg/mot/train.py"),
        "source_layer_indices": [int(value) for value in source_layers],
        "target_layer_indices": [int(value) for value in target_layers],
        "top_s": int(ctx.config.topk_sparse_attn),
        "num_bottom_full_attention_layers": int(ctx.config.num_bottom_full_attn),
        "data_manifest_sha256": _sha256(data_manifest),
        "data_manifest_content_sha256": actual_content_sha,
        "expected_item_ids_sha256": _canonical_sha256(sorted(expected_item_ids)),
        "anchor_records_sha256": _sha256(anchor_records),
        "checkpoint_tree_sha256": checkpoint_tree_sha256,
        "checkpoint_model_file": str(checkpoint_model_file.relative_to(checkpoint)),
        "checkpoint_model_file_sha256": observed_checkpoint_file_sha256,
        "scorer_sha256": scorer_sha256,
        "generation": {"method": "greedy", "max_new_tokens": 64},
        "task_item_counts": {
            task: cohort["item_count"]
            for task, cohort in manifest["evaluation"].items()
        },
        "selection_contract": "one source-derived tensor ledger per item reused by T, R, and O",
        "squad_uncertainty_grouping": "executed context-token digest",
        "mmlu_uncertainty_grouping": "subject",
    }
    run_manifest_path = output / "RUN_MANIFEST.json"
    run_manifest_path.write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run_manifest_sha256 = _sha256(run_manifest_path)

    def token_ids(values: list[int] | None, model_id: str):
        if values is None:
            return None
        return common.TokenIDs(
            torch.tensor([values], dtype=torch.long, device="cuda"), model_id=model_id
        )

    @torch.inference_mode()
    def generate(model, past, seed, prompt):
        if prompt is not None:
            past = eval_util.append_token_ids_to_past(
                model=model, past_key_values=past, token_ids=prompt
            )
        generated: list[int] = []
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

    def score(task: str, spec, item: dict[str, Any], generated: dict[str, Any]):
        if task in {"boolq", "pubmedqa", "mmlu_redux"}:
            parsed = eval_util.parse_generated_logit_answer(
                spec,
                generated["postprocessed_text"],
                choices=item["input"].get("choices"),
            )
            gold = item["input"]["answer"]
            value = (
                1.0
                if parsed is not None
                and eval_util.is_logit_answer_correct(parsed, gold)
                else 0.0
            )
        else:
            parsed = generated["postprocessed_text"] or None
            value = float(
                eval_util.compute_generation_f1(
                    generated["postprocessed_text"], item["input"]["answers"]
                )
            )
        generated.update(
            parsed_answer=parsed,
            parse_status="parsed" if parsed is not None else "invalid_answer",
            score=value,
        )
        return generated

    records: list[dict[str, Any]] = []
    selection_receipts: list[dict[str, str]] = []
    raw_path = output / "ATTRIBUTION_RECORDS.jsonl"
    started = time.monotonic()
    with raw_path.open("w", encoding="utf-8", buffering=1) as raw_handle:
        for task, cohort in manifest["evaluation"].items():
            spec = task_specs[task]
            for item in cohort["items"]:
                row_started = time.monotonic()
                target_ledger = item["token_ledgers"]["target"]
                source_ledger = item["token_ledgers"]["source"]
                target_context = token_ids(
                    target_ledger["context_token_ids"], target_model.id
                )
                source_context = token_ids(
                    source_ledger["context_token_ids"], source_model.id
                )
                prompt = token_ids(target_ledger["prompt_token_ids"], target_model.id)
                seed = token_ids(target_ledger["seed_token_ids"], target_model.id)
                record: dict[str, Any] = {
                    "schema": "knlp_mot_attribution_record_v1",
                    "item_id": item["item_id"],
                    "task": task,
                    "subject": item.get("subject"),
                    "document_group": item["document_group"],
                    "run_manifest_sha256": run_manifest_sha256,
                    "context_token_ids": target_ledger["context_token_ids"],
                    "prompt_token_ids": target_ledger["prompt_token_ids"],
                    "seed_token_ids": target_ledger["seed_token_ids"],
                    "gold_answers": item["input"].get(
                        "answers", item["input"].get("answer")
                    ),
                    "executed_context_lengths": {
                        "source": len(source_ledger["context_token_ids"]),
                        "target": len(target_ledger["context_token_ids"]),
                        "prompt": 0
                        if target_ledger["prompt_token_ids"] is None
                        else len(target_ledger["prompt_token_ids"]),
                    },
                    "truncation_status": bool(target_ledger["was_truncated"]),
                    "selected_indices_sha256": None,
                    "branch_counts": {
                        "bottom_full_attention_layers": int(
                            ctx.config.num_bottom_full_attn
                        ),
                        "reduced_selected_replay_layers": (
                            max(
                                0,
                                target_spec.num_layers
                                - int(ctx.config.num_bottom_full_attn),
                            )
                            if len(target_ledger["context_token_ids"])
                            > int(ctx.config.topk_sparse_attn)
                            else 0
                        ),
                        "full_key_replay_layers": (
                            max(
                                0,
                                target_spec.num_layers
                                - int(ctx.config.num_bottom_full_attn),
                            )
                            if len(target_ledger["context_token_ids"])
                            <= int(ctx.config.topk_sparse_attn)
                            else 0
                        ),
                        "translated_injection_layers": len(target_layers),
                        "recompute_only_injection_layers": 0,
                        "native_injection_layers": len(target_layers),
                        "downstream_effective_injection_layers": sum(
                            int(index) < target_spec.num_layers - 1
                            for index in target_layers
                        ),
                    },
                    "arms": {
                        arm: {"score": None, "parse_status": "operational_failure"}
                        for arm in ARMS
                    },
                    "anchor_match": {"N": False, "T": False},
                    "phase_seconds": {},
                    "operational_error": None,
                }
                try:
                    if source_context.as_tensor().tolist() != target_context.as_tensor().tolist():
                        raise RuntimeError(
                            "Frozen source/target context ledgers are not identical"
                        )
                    phase_started = time.monotonic()
                    source_past = common.extract_past_key_values(
                        source_model, source_context
                    )
                    record["phase_seconds"]["source_prefill"] = (
                        time.monotonic() - phase_started
                    )
                    phase_started = time.monotonic()
                    native_past = common.extract_past_key_values(
                        target_model, target_context
                    )
                    record["phase_seconds"]["native_prefill"] = (
                        time.monotonic() - phase_started
                    )
                    arm_pasts, selected_indices, control_phase_seconds = (
                        build_attribution_pasts(
                            mot_train=mot_train,
                            common=common,
                            ctx=ctx,
                            source_model=source_model,
                            target_model=target_model,
                            source_node=source_node,
                            target_node=target_node,
                            source_spec=source_spec,
                            target_spec=target_spec,
                            source_context=source_context,
                            target_context=target_context,
                            source_past=source_past,
                            native_past=native_past,
                        )
                    )
                    record["phase_seconds"].update(control_phase_seconds)
                    selection_sha256 = _tensor_ledger_sha256(selected_indices)
                    record["selected_indices_sha256"] = selection_sha256
                    selection_receipts.append(
                        {"item_id": str(item["item_id"]), "sha256": selection_sha256}
                    )

                    for arm in ARMS:
                        phase_started = time.monotonic()
                        generated = generate(
                            target_model, arm_pasts[arm], seed, prompt
                        )
                        record["phase_seconds"][f"{arm}_generation"] = (
                            time.monotonic() - phase_started
                        )
                        record["arms"][arm] = score(task, spec, item, generated)

                    anchor = anchors[str(item["item_id"])]
                    for arm, old_arm in (("N", "native"), ("T", "candidate")):
                        record["anchor_match"][arm] = (
                            record["arms"][arm]["generated_token_ids"]
                            == anchor[old_arm]["generated_token_ids"]
                            and float(record["arms"][arm]["score"])
                            == float(anchor[old_arm]["score"])
                        )
                except Exception as error:
                    record["operational_error"] = f"{type(error).__name__}: {error}"
                record["elapsed_seconds"] = time.monotonic() - row_started
                raw_handle.write(json.dumps(record, sort_keys=True) + "\n")
                records.append(record)

    aggregates = aggregate_attribution_records(records)
    replay = replay_attribution_jsonl(raw_path, output / "REPLAY.json")
    observed_item_ids = [str(record["item_id"]) for record in records]
    id_completeness = (
        len(observed_item_ids) == 500
        and set(observed_item_ids) == set(expected_item_ids)
        and len(set(observed_item_ids)) == 500
    )
    arm_score_contract = replay["all_arm_score_contract_pass"]
    anchor_matches = {
        arm: sum(bool(record["anchor_match"][arm]) for record in records)
        for arm in ("N", "T")
    }
    expected_truncations = sum(
        bool(item["token_ledgers"]["target"]["was_truncated"])
        for cohort in manifest["evaluation"].values()
        for item in cohort["items"]
    )
    observed_truncations = sum(bool(record["truncation_status"]) for record in records)
    full_key_rows = [
        row
        for row in records
        if row["executed_context_lengths"]["target"] <= int(ctx.config.topk_sparse_attn)
        and row["operational_error"] is None
    ]
    full_key_sanity = {
        "item_count": len(full_key_rows),
        "R_N_generated_token_ids_exact_match_count": sum(
            row["arms"]["R"]["generated_token_ids"]
            == row["arms"]["N"]["generated_token_ids"]
            for row in full_key_rows
        ),
        "R_minus_N_mean_score": (
            sum(
                float(row["arms"]["R"]["score"])
                - float(row["arms"]["N"]["score"])
                for row in full_key_rows
            )
            / len(full_key_rows)
            if full_key_rows
            else None
        ),
    }
    selection_ledger_sha256 = _canonical_sha256(selection_receipts)
    offline_replay_matches = replay["aggregates"] == aggregates
    operational_complete = (
        id_completeness
        and arm_score_contract
        and expected_truncations == observed_truncations
        and offline_replay_matches
        and anchor_matches == {"N": 500, "T": 500}
        and len(selection_receipts) == 500
    )
    summary = {
        "schema": "knlp_mot_attribution_summary_v1",
        "record_count": len(records),
        "scored_output_count": len(records) * len(ARMS),
        "expected_record_count": 500,
        "expected_scored_output_count": 2000,
        "operational_complete": operational_complete,
        "integrity": {
            "expected_ids_exactly_once": id_completeness,
            "all_four_arms_finite_and_in_range": arm_score_contract,
            "anchor_token_and_score_matches": anchor_matches,
            "expected_truncation_count": expected_truncations,
            "observed_truncation_count": observed_truncations,
            "selection_receipt_count": len(selection_receipts),
            "selection_ledger_sha256": selection_ledger_sha256,
            "offline_replay_matches": offline_replay_matches,
        },
        "full_key_R_vs_N_sanity": full_key_sanity,
        "raw_records_sha256": _sha256(raw_path),
        "run_manifest_sha256": run_manifest_sha256,
        "elapsed_seconds": time.monotonic() - started,
        "aggregates": aggregates,
    }
    (output / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary

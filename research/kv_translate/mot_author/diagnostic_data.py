"""Freeze the bounded modified-MoT diagnostic inputs before GPU execution."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

from .data import _token_ids_sha256, tokenizer_receipt
from .recipe import DATASET_REVISIONS

SCHEMA = "knlp_mot_diagnostic_data_v1"
SEQUENCE_LENGTH = 128
PREFIX_TOKENS_CONFIGURED = 64
CONTEXT_TOKENS_EXECUTED = 63
SHUFFLE_SEED = 42
SHUFFLE_BUFFER = 50_000
PARTITION_ROWS = {
    "profile_train": 128,
    "profile_validation": 64,
    "main_train": 8_000,
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )


def _tokenizer_sha256(tokenizer) -> str:
    return _canonical_sha256(
        {
            "vocab": tokenizer.get_vocab(),
            "special_tokens_map": tokenizer.special_tokens_map,
        }
    )


def _load_smoke_document_hashes(path: Path | None) -> set[str]:
    if path is None:
        return set()
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    return {
        str(source_row["text_sha256"])
        for item in payload.get("items", [])
        for source_row in item.get("source_rows", [])
    }


def _freeze_openwebtext_partitions(*, tokenizer, smoke_document_hashes: set[str]) -> dict[str, Any]:
    from datasets import load_dataset

    dataset_id = "Skylion007/openwebtext"
    revision = DATASET_REVISIONS[dataset_id]
    stream = load_dataset(dataset_id, split="train", streaming=True, revision=revision)
    stream = stream.shuffle(seed=SHUFFLE_SEED, buffer_size=SHUFFLE_BUFFER)
    partition_names = list(PARTITION_ROWS)
    partitions = {
        name: {"row_count": 0, "rows": [], "source_documents": []}
        for name in partition_names
    }
    partition_index = 0
    token_buffer: list[int] = []
    provenance_buffer: list[str] = []
    source_documents: dict[str, dict[str, Any]] = {}

    for shuffled_row_index, example in enumerate(stream):
        if partition_index >= len(partition_names):
            break
        text = example.get("text", "")
        if not isinstance(text, str) or not text or text.isspace():
            continue
        text_sha256 = _sha256_bytes(text.encode("utf-8"))
        if text_sha256 in smoke_document_hashes:
            continue
        token_ids = tokenizer(text, add_special_tokens=False, verbose=False).input_ids
        if len(token_ids) < 8:
            continue

        partition_name = partition_names[partition_index]
        document_id = f"owt-{shuffled_row_index}-{text_sha256[:16]}"
        source_documents[document_id] = {
            "document_id": document_id,
            "shuffled_row_index": shuffled_row_index,
            "text_sha256": text_sha256,
            "token_count_without_eos": len(token_ids),
        }
        partitions[partition_name]["source_documents"].append(source_documents[document_id])
        document_tokens = [*token_ids, tokenizer.eos_token_id]
        token_buffer.extend(document_tokens)
        provenance_buffer.extend([document_id] * len(document_tokens))

        target_rows = PARTITION_ROWS[partition_name]
        while len(token_buffer) >= SEQUENCE_LENGTH and len(partitions[partition_name]["rows"]) < target_rows:
            chunk = token_buffer[:SEQUENCE_LENGTH]
            provenance = provenance_buffer[:SEQUENCE_LENGTH]
            del token_buffer[:SEQUENCE_LENGTH]
            del provenance_buffer[:SEQUENCE_LENGTH]
            row_index = len(partitions[partition_name]["rows"])
            partitions[partition_name]["rows"].append(
                {
                    "row_index": row_index,
                    "token_ids": chunk,
                    "token_ids_sha256": _token_ids_sha256(chunk),
                    "source_document_ids": list(dict.fromkeys(provenance)),
                }
            )

        if len(partitions[partition_name]["rows"]) == target_rows:
            partitions[partition_name]["row_count"] = target_rows
            partitions[partition_name]["source_document_count"] = len(
                partitions[partition_name]["source_documents"]
            )
            # Never carry tokens from one source-document partition into another.
            token_buffer.clear()
            provenance_buffer.clear()
            partition_index += 1

    if partition_index != len(partition_names):
        missing = partition_names[partition_index:]
        raise RuntimeError(f"OpenWebText ended before frozen partitions were complete: {missing}")

    document_sets = {
        name: {doc["text_sha256"] for doc in partition["source_documents"]}
        for name, partition in partitions.items()
    }
    overlaps = {
        f"{left}__{right}": sorted(document_sets[left] & document_sets[right])
        for left_idx, left in enumerate(partition_names)
        for right in partition_names[left_idx + 1 :]
    }
    if any(overlaps.values()):
        raise RuntimeError("Frozen OpenWebText source-document partitions overlap")
    if any(document_sets[name] & smoke_document_hashes for name in partition_names):
        raise RuntimeError("Frozen OpenWebText partition reuses a smoke document")
    return {
        "dataset": {
            "id": dataset_id,
            "revision": revision,
            "split": "train",
            "streaming": True,
            "shuffle_seed": SHUFFLE_SEED,
            "shuffle_buffer": SHUFFLE_BUFFER,
        },
        "partitions": partitions,
        "document_overlap": overlaps,
        "excluded_smoke_document_hashes": sorted(smoke_document_hashes),
    }


def _load_dataset_rows(path: str, *, split: str, revision: str, name: str | None = None):
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": split, "revision": revision}
    if name is None:
        return load_dataset(path, **kwargs)
    return load_dataset(path, name, **kwargs)


def _stable_raw_record(raw: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(raw), sort_keys=True, ensure_ascii=False, default=str))


def _freeze_evaluation_cohorts(
    source: Path, *, source_model: Path, target_model: Path
) -> dict[str, Any]:
    import sys
    from types import SimpleNamespace

    from transformers import AutoConfig, AutoTokenizer

    sys.path.insert(0, str(source))
    try:
        from core.eval_util import (
            MMLU_REDUX_SUBJECTS,
            compute_benchmark_context_budget,
            compute_logit_task_token_budgets,
            extract_generation_examples,
            extract_question_and_answer,
            get_boolq_dataset_spec,
            get_mmlu_redux_dataset_spec,
            get_newsqa_generation_dataset_spec,
            get_pubmedqa_dataset_spec,
            get_squad_v11_dataset_spec,
            prepare_generation_task_inputs,
            prepare_logit_task_inputs,
        )
    finally:
        sys.path.pop(0)

    class TokenizerModel:
        def __init__(self, model_id: str, snapshot: Path):
            self.id = model_id
            self.tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
            self.config = AutoConfig.from_pretrained(str(snapshot), local_files_only=True)

    models = {
        "source": TokenizerModel("gpt2", source_model),
        "target": TokenizerModel("gpt2-medium", target_model),
    }
    eval_config = SimpleNamespace(generation_max_new_tokens=64)
    spec_by_task: dict[str, Any] = {}
    result: dict[str, Any] = {}
    simple_specs = [
        ("boolq", get_boolq_dataset_spec(), DATASET_REVISIONS["google/boolq"]),
        ("pubmedqa", get_pubmedqa_dataset_spec(), DATASET_REVISIONS["qiaojin/PubMedQA"]),
        ("squad_v1_1", get_squad_v11_dataset_spec(), DATASET_REVISIONS["rajpurkar/squad"]),
        ("newsqa", get_newsqa_generation_dataset_spec(), DATASET_REVISIONS["gabrieltorresgamez/newsqa"]),
    ]
    for task, spec, revision in simple_specs:
        spec_by_task[task] = spec
        dataset = _load_dataset_rows(
            spec.dataset_path,
            split=spec.split,
            revision=revision,
            name=spec.dataset_name,
        )
        items: list[dict[str, Any]] = []
        for raw_index, raw in enumerate(dataset):
            extracted: Iterable[dict[str, Any]]
            if task in {"squad_v1_1", "newsqa"}:
                extracted = extract_generation_examples(spec, dict(raw))
            else:
                pair = extract_question_and_answer(spec, dict(raw))
                extracted = [] if pair is None else [pair]
            for sub_index, item in enumerate(extracted):
                stable_raw = _stable_raw_record(dict(raw))
                item_id = f"{task}-{raw_index}-{sub_index}"
                items.append(
                    {
                        "item_id": item_id,
                        "raw_index": raw_index,
                        "sub_index": sub_index,
                        "document_group": str(raw.get("id", raw_index)),
                        "input": item,
                        "raw_sha256": _canonical_sha256(stable_raw),
                    }
                )
                if len(items) == 100:
                    break
            if len(items) == 100:
                break
        if len(items) != 100:
            raise RuntimeError(f"Could not freeze 100 eligible {task} items")
        result[task] = {
            "dataset": {
                "id": spec.dataset_path,
                "name": spec.dataset_name,
                "revision": revision,
                "split": spec.split,
            },
            "scorer": "accuracy" if task in {"boolq", "pubmedqa"} else "normalized_token_f1",
            "item_count": len(items),
            "items": items,
        }

    spec = get_mmlu_redux_dataset_spec()
    spec_by_task["mmlu_redux"] = spec
    revision = DATASET_REVISIONS["edinburgh-dawg/mmlu-redux-2.0"]
    second_subjects = set(random.Random(SHUFFLE_SEED).sample(list(MMLU_REDUX_SUBJECTS), 43))
    items = []
    subject_counts: dict[str, int] = {}
    for subject in MMLU_REDUX_SUBJECTS:
        dataset = _load_dataset_rows(
            spec.dataset_path,
            split=spec.split,
            revision=revision,
            name=subject,
        )
        wanted = 2 if subject in second_subjects else 1
        for raw_index, raw in enumerate(dataset):
            example = dict(raw)
            example.setdefault(spec.subject_field or "subject", subject)
            pair = extract_question_and_answer(spec, example)
            if pair is None:
                continue
            item_id = f"mmlu_redux-{subject}-{raw_index}"
            items.append(
                {
                    "item_id": item_id,
                    "raw_index": raw_index,
                    "subject": subject,
                    "document_group": subject,
                    "input": pair,
                    "raw_sha256": _canonical_sha256(_stable_raw_record(example)),
                }
            )
            subject_counts[subject] = subject_counts.get(subject, 0) + 1
            if subject_counts[subject] == wanted:
                break
        if subject_counts.get(subject, 0) != wanted:
            raise RuntimeError(f"Could not freeze {wanted} eligible MMLU items for {subject}")
    if len(items) != 100 or len(subject_counts) != len(MMLU_REDUX_SUBJECTS):
        raise RuntimeError("Frozen MMLU cohort must contain 100 items across all 57 subjects")
    result["mmlu_redux"] = {
        "dataset": {
            "id": spec.dataset_path,
            "revision": revision,
            "split": spec.split,
        },
        "scorer": "accuracy",
        "selection": {
            "one_per_subject": True,
            "second_item_subjects": sorted(second_subjects),
            "seed": SHUFFLE_SEED,
        },
        "subject_counts": subject_counts,
        "item_count": len(items),
        "items": items,
    }

    for task, cohort in result.items():
        task_spec = spec_by_task[task]
        for item in cohort["items"]:
            example = item["input"]
            ledgers: dict[str, Any] = {}
            for role, model in models.items():
                if task in {"squad_v1_1", "newsqa"}:
                    max_context_tokens = compute_benchmark_context_budget(
                        None,
                        task_spec,
                        example["question"],
                        eval_config,
                        model=model,
                    )
                    prepared = prepare_generation_task_inputs(
                        spec=task_spec,
                        model=model,
                        context=example["context"],
                        question=example["question"],
                        device="cpu",
                        max_input_tokens=max_context_tokens,
                    )
                else:
                    budgets = compute_logit_task_token_budgets(
                        None,
                        task_spec,
                        example["question"],
                        eval_config,
                        model=model,
                        choices=example.get("choices"),
                        choice_texts=example.get("choice_texts"),
                        subject=example.get("subject"),
                    )
                    prepared = prepare_logit_task_inputs(
                        spec=task_spec,
                        model=model,
                        context=example.get("context"),
                        question=example["question"],
                        device="cpu",
                        choices=example.get("choices"),
                        choice_texts=example.get("choice_texts"),
                        subject=example.get("subject"),
                        max_context_tokens=budgets["max_context_tokens"],
                        max_prefix_tokens=budgets["max_prefix_tokens"],
                    )

                def ids(name: str) -> list[int] | None:
                    value = prepared.get(name)
                    return None if value is None else value.as_tensor()[0].tolist()

                ledgers[role] = {
                    "model_id": model.id,
                    "context_token_ids": ids("context_token_ids"),
                    "prompt_token_ids": ids("prompt_token_ids"),
                    "seed_token_ids": ids("seed_token"),
                    "was_truncated": bool(prepared.get("was_truncated", False)),
                    "formatted_context_text": prepared.get("prefix", {}).get("text"),
                    "formatted_suffix_text": prepared.get("suffix", {}).get("text"),
                }
            if ledgers["source"]["context_token_ids"] != ledgers["target"]["context_token_ids"]:
                raise RuntimeError(f"Evaluation context token grid differs for {item['item_id']}")
            if ledgers["source"]["prompt_token_ids"] != ledgers["target"]["prompt_token_ids"]:
                raise RuntimeError(f"Evaluation prompt token grid differs for {item['item_id']}")
            if ledgers["source"]["seed_token_ids"] != ledgers["target"]["seed_token_ids"]:
                raise RuntimeError(f"Evaluation seed token grid differs for {item['item_id']}")
            item["token_ledgers"] = ledgers
            item["token_ledgers_sha256"] = _canonical_sha256(ledgers)

    if sum(task["item_count"] for task in result.values()) != 500:
        raise RuntimeError("Frozen diagnostic must contain exactly 500 evaluation items")
    return result


def freeze_diagnostic_manifest(
    *,
    source: Path,
    source_model: Path,
    target_model: Path,
    output_json: Path,
    smoke_manifest: Path | None,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    source = source.expanduser().resolve()
    source_model = source_model.expanduser().resolve()
    target_model = target_model.expanduser().resolve()
    tokenizer = AutoTokenizer.from_pretrained(str(target_model), local_files_only=True)
    smoke_document_hashes = _load_smoke_document_hashes(smoke_manifest)
    openwebtext = _freeze_openwebtext_partitions(
        tokenizer=tokenizer,
        smoke_document_hashes=smoke_document_hashes,
    )
    evaluations = _freeze_evaluation_cohorts(
        source,
        source_model=source_model,
        target_model=target_model,
    )
    manifest = {
        "schema": SCHEMA,
        "seed": SHUFFLE_SEED,
        "sequence_contract": {
            "total_tokens": SEQUENCE_LENGTH,
            "configured_prefix_tokens": PREFIX_TOKENS_CONFIGURED,
            "executed_cached_context_tokens": CONTEXT_TOKENS_EXECUTED,
            "suffix_input_tokens": 64,
            "predicted_label_tokens": 64,
        },
        "tokenizer": tokenizer_receipt(target_model),
        "tokenizer_sha256": _tokenizer_sha256(tokenizer),
        "dataset": openwebtext["dataset"],
        "partitions": openwebtext["partitions"],
        "document_overlap": openwebtext["document_overlap"],
        "excluded_smoke_document_hashes": openwebtext["excluded_smoke_document_hashes"],
        "evaluation": evaluations,
    }
    manifest["manifest_content_sha256"] = _canonical_sha256(manifest)
    output_json = output_json.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest

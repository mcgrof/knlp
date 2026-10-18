"""Admission controls for the frozen modified-MoT diagnostic."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any

from .diagnostic_data import CONTEXT_TOKENS_EXECUTED, PARTITION_ROWS, SCHEMA
from .mechanics import inspect_mechanics_source
from .recipe import FROZEN_DIAGNOSTIC_CONTENT_SHA256


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _load_author(source: Path):
    source_text = str(source.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    return {
        "common": importlib.import_module("core.common"),
        "model": importlib.import_module("core.model"),
        "train": importlib.import_module("alg.mot.train"),
    }


def run_diagnostic_controls(
    *, source: Path, source_model: Path, target_model: Path, data_manifest: Path
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    source_receipt = inspect_mechanics_source(source, variant="modified-mot")
    modules = _load_author(source)
    common = modules["common"]
    train = modules["train"]
    source_tokenizer = common.load_tokenizer(str(source_model.expanduser().resolve()))
    target_tokenizer = common.load_tokenizer(str(target_model.expanduser().resolve()))

    class TokenizerModel:
        def __init__(self, model_id, tokenizer):
            self.id = model_id
            self.tokenizer = tokenizer
            self.device = "cpu"

    source_stub = TokenizerModel("gpt2", source_tokenizer)
    target_stub = TokenizerModel("gpt2-medium", target_tokenizer)
    manifest = json.loads(data_manifest.expanduser().resolve().read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise RuntimeError("Modified-MoT diagnostic manifest schema mismatch")
    declared_manifest_sha = manifest.get("manifest_content_sha256")
    content_without_sha = dict(manifest)
    content_without_sha.pop("manifest_content_sha256", None)
    actual_manifest_sha = _canonical_sha256(content_without_sha)
    if declared_manifest_sha != actual_manifest_sha:
        raise RuntimeError("Modified-MoT diagnostic manifest content hash mismatch")
    if actual_manifest_sha != FROZEN_DIAGNOSTIC_CONTENT_SHA256:
        raise RuntimeError("Modified-MoT diagnostic is not the frozen admitted data")

    scan_rows = []
    mismatch_item_ids = []

    def scan(item_id: str, target_values: list[int], expected_source_values: list[int] | None = None):
        target_ids = common.TokenIDs([target_values], model_id=target_stub.id)
        source_ids = train._retokenize_target_context_row_for_source(
            source_model=source_stub,
            target_model=target_stub,
            target_context_token_ids=target_ids,
        )
        source_values = source_ids.as_tensor()[0].tolist()
        identity = bool(
            train.token_ledgers_have_identity_alignment(
                source_stub, target_stub, source_ids, target_ids
            )
        )
        weights, target_to_source, source_to_target = train.build_identity_token_alignment(
            source_stub, target_stub, source_ids, target_ids
        )
        expected_source_matches = expected_source_values is None or source_values == expected_source_values
        if not identity or not expected_source_matches:
            mismatch_item_ids.append(item_id)
        scan_rows.append(
            {
                "item_id": item_id,
                "token_count": len(target_values),
                "identity": identity,
                "expected_source_matches": expected_source_matches,
                "alignment_shape": list(weights.shape),
                "target_mapping_identity": target_to_source[0].tolist() == list(range(len(target_values))),
                "source_mapping_identity": source_to_target[0].tolist() == list(range(len(target_values))),
            }
        )

    for partition_name, expected_count in PARTITION_ROWS.items():
        partition = manifest["partitions"][partition_name]
        if partition.get("row_count") != expected_count or len(partition.get("rows", [])) != expected_count:
            raise RuntimeError(f"Frozen partition count mismatch: {partition_name}")
        for row in partition["rows"]:
            scan(
                f"{partition_name}-{row['row_index']}",
                row["token_ids"][:CONTEXT_TOKENS_EXECUTED],
            )

    evaluation_count = 0
    evaluation_truncated = 0
    for task, cohort in manifest["evaluation"].items():
        if cohort.get("item_count") != 100 or len(cohort.get("items", [])) != 100:
            raise RuntimeError(f"Frozen evaluation cohort count mismatch: {task}")
        for item in cohort["items"]:
            target_ledger = item["token_ledgers"]["target"]
            source_ledger = item["token_ledgers"]["source"]
            scan(
                item["item_id"],
                target_ledger["context_token_ids"],
                expected_source_values=source_ledger["context_token_ids"],
            )
            if target_ledger["was_truncated"]:
                evaluation_truncated += 1
            evaluation_count += 1
    if evaluation_count != 500:
        raise RuntimeError("Frozen evaluation must contain exactly 500 items")

    unicode_text = "Unicode boundary: naïve café — 你好🙂"
    unicode_ids = target_tokenizer(unicode_text, add_special_tokens=False).input_ids
    partial_boundary = None
    for prefix_len in range(1, len(unicode_ids)):
        prefix_ids = unicode_ids[:prefix_len]
        decoded = train._decode_token_ids(target_tokenizer, prefix_ids)
        roundtrip_ids = target_tokenizer(decoded, add_special_tokens=False).input_ids
        if roundtrip_ids != prefix_ids:
            partial_boundary = {
                "text": unicode_text,
                "prefix_token_count": prefix_len,
                "prefix_token_ids": prefix_ids,
                "decoded_prefix": decoded,
                "decode_reencode_token_ids": roundtrip_ids,
                "decode_reencode_preserves_ids": False,
            }
            target_ids = common.TokenIDs([prefix_ids], model_id=target_stub.id)
            source_ids = train._retokenize_target_context_row_for_source(
                source_model=source_stub,
                target_model=target_stub,
                target_context_token_ids=target_ids,
            )
            partial_boundary["production_exact_transfer_preserves_ids"] = (
                source_ids.as_tensor()[0].tolist() == prefix_ids
            )
            weights, _, _ = train.build_identity_token_alignment(
                source_stub, target_stub, source_ids, target_ids
            )
            future_mask = importlib.import_module("torch").triu(
                importlib.import_module("torch").ones(prefix_len, prefix_len, dtype=importlib.import_module("torch").bool),
                diagonal=1,
            ).unsqueeze(0)
            partial_boundary["future_source_weight_max"] = (
                float(weights[future_mask].max().item()) if bool(future_mask.any().item()) else 0.0
            )
            break
    if partial_boundary is None:
        raise RuntimeError("Could not construct a partial-Unicode decode/re-encode boundary")

    unequal_grid_blocked = False
    unequal_source = common.TokenIDs([[unicode_ids[0]]], model_id=source_stub.id)
    unequal_target = common.TokenIDs([[unicode_ids[0], unicode_ids[1]]], model_id=target_stub.id)
    try:
        train.build_identity_token_alignment(
            source_stub, target_stub, unequal_source, unequal_target
        )
    except ValueError:
        unequal_grid_blocked = True

    document_overlap_empty = all(
        not hashes for hashes in manifest.get("document_overlap", {}).values()
    )
    all_identity = not mismatch_item_ids and all(
        row["identity"]
        and row["expected_source_matches"]
        and row["target_mapping_identity"]
        and row["source_mapping_identity"]
        for row in scan_rows
    )
    passed = (
        all_identity
        and document_overlap_empty
        and unequal_grid_blocked
        and partial_boundary["production_exact_transfer_preserves_ids"]
        and partial_boundary["future_source_weight_max"] == 0.0
    )
    return {
        "schema": "knlp_mot_diagnostic_controls_v1",
        "source": source_receipt,
        "data_manifest": {
            "path": str(data_manifest.expanduser().resolve()),
            "declared_content_sha256": declared_manifest_sha,
            "recomputed_content_sha256": actual_manifest_sha,
        },
        "partition_row_counts": {
            name: manifest["partitions"][name]["row_count"] for name in PARTITION_ROWS
        },
        "evaluation_item_count": evaluation_count,
        "evaluation_truncated_count": evaluation_truncated,
        "ledger_scan_count": len(scan_rows),
        "ledger_mismatch_count": len(mismatch_item_ids),
        "ledger_mismatch_item_ids": mismatch_item_ids,
        "partial_unicode_boundary": partial_boundary,
        "unequal_grid_blocked": unequal_grid_blocked,
        "document_overlap_empty": document_overlap_empty,
        "decision": {
            "identity_contract_pass": all_identity,
            "frozen_data_contract_pass": document_overlap_empty,
            "diagnostic_controls_pass": passed,
        },
    }

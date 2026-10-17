"""Freeze an exact OpenWebText token manifest for the MoT GPU smoke."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .recipe import DATASET_REVISIONS, MODEL_REVISIONS

DATASET_ID = "Skylion007/openwebtext"
DATASET_REVISION = DATASET_REVISIONS[DATASET_ID]
DATASET_SPLIT = "train"
SMOKE_ITEM_COUNT = 32
SMOKE_TOTAL_TOKENS = 128
SMOKE_SHUFFLE_BUFFER = 50_000
SMOKE_SEED = 42

TOKENIZER_FILES = (
    "config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _token_ids_sha256(token_ids: list[int]) -> str:
    encoded = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tokenizer_receipt(snapshot: Path) -> dict[str, Any]:
    snapshot = snapshot.expanduser().resolve()
    if snapshot.name != MODEL_REVISIONS["gpt2-medium"]:
        raise RuntimeError("GPT-2-medium tokenizer snapshot revision mismatch")
    files = []
    for name in TOKENIZER_FILES:
        path = snapshot / name
        if path.is_file():
            files.append(
                {
                    "name": name,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    if not files:
        raise RuntimeError(f"No tokenizer files found in {snapshot}")
    return {
        "model": "gpt2-medium",
        "revision": snapshot.name,
        "files": files,
    }


def freeze_smoke_manifest(*, target_model: Path, output_json: Path) -> dict[str, Any]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    target_model = target_model.expanduser().resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        str(target_model),
        local_files_only=True,
    )
    stream = load_dataset(
        DATASET_ID,
        split=DATASET_SPLIT,
        streaming=True,
        revision=DATASET_REVISION,
    )
    stream = stream.shuffle(seed=SMOKE_SEED, buffer_size=SMOKE_SHUFFLE_BUFFER)

    token_buffer: list[int] = []
    provenance_buffer: list[int] = []
    source_rows: dict[int, dict[str, Any]] = {}
    items = []
    for shuffled_row_index, example in enumerate(stream):
        text = example.get("text", "")
        if not text or text.isspace():
            continue
        token_ids = tokenizer(
            text,
            add_special_tokens=False,
            verbose=False,
        ).input_ids
        if len(token_ids) < 8:
            continue
        source_rows[shuffled_row_index] = {
            "shuffled_row_index": shuffled_row_index,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "token_count_without_eos": len(token_ids),
        }
        row_token_ids = [*token_ids, tokenizer.eos_token_id]
        token_buffer.extend(row_token_ids)
        provenance_buffer.extend([shuffled_row_index] * len(row_token_ids))

        while len(token_buffer) >= SMOKE_TOTAL_TOKENS:
            chunk = token_buffer[:SMOKE_TOTAL_TOKENS]
            provenance = provenance_buffer[:SMOKE_TOTAL_TOKENS]
            del token_buffer[:SMOKE_TOTAL_TOKENS]
            del provenance_buffer[:SMOKE_TOTAL_TOKENS]
            ordered_rows = list(dict.fromkeys(provenance))
            items.append(
                {
                    "index": len(items),
                    "token_ids": chunk,
                    "token_ids_sha256": _token_ids_sha256(chunk),
                    "source_rows": [source_rows[index] for index in ordered_rows],
                }
            )
            if len(items) == SMOKE_ITEM_COUNT:
                manifest = {
                    "schema": "knlp_mot_training_smoke_manifest_v2",
                    "dataset": {
                        "id": DATASET_ID,
                        "revision": DATASET_REVISION,
                        "split": DATASET_SPLIT,
                        "streaming": True,
                        "shuffle": True,
                        "shuffle_seed": SMOKE_SEED,
                        "shuffle_buffer": SMOKE_SHUFFLE_BUFFER,
                    },
                    "tokenizer": tokenizer_receipt(target_model),
                    "item_count": len(items),
                    "total_tokens_per_item": SMOKE_TOTAL_TOKENS,
                    "items": items,
                }
                output_json = output_json.expanduser().resolve()
                output_json.parent.mkdir(parents=True, exist_ok=True)
                output_json.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                return manifest
    raise RuntimeError(
        f"OpenWebText ended before {SMOKE_ITEM_COUNT} fixed token chunks"
    )


def load_smoke_manifest(
    *, manifest_path: Path, target_model: Path
) -> tuple[list[list[int]], dict[str, Any]]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "knlp_mot_training_smoke_manifest_v2":
        raise RuntimeError("Training smoke manifest schema mismatch")
    dataset = manifest.get("dataset", {})
    expected_dataset = {
        "id": DATASET_ID,
        "revision": DATASET_REVISION,
        "split": DATASET_SPLIT,
        "streaming": True,
        "shuffle": True,
        "shuffle_seed": SMOKE_SEED,
        "shuffle_buffer": SMOKE_SHUFFLE_BUFFER,
    }
    if dataset != expected_dataset:
        raise RuntimeError("Training smoke dataset receipt mismatch")
    if manifest.get("tokenizer") != tokenizer_receipt(target_model):
        raise RuntimeError("Training smoke tokenizer receipt mismatch")
    items = manifest.get("items", [])
    if manifest.get("item_count") != SMOKE_ITEM_COUNT or len(items) != SMOKE_ITEM_COUNT:
        raise RuntimeError("Training smoke item count mismatch")
    if manifest.get("total_tokens_per_item") != SMOKE_TOTAL_TOKENS:
        raise RuntimeError("Training smoke token count declaration mismatch")

    rows = []
    for expected_index, item in enumerate(items):
        token_ids = item.get("token_ids")
        if item.get("index") != expected_index:
            raise RuntimeError("Training smoke item ordering mismatch")
        if not isinstance(token_ids, list) or len(token_ids) != SMOKE_TOTAL_TOKENS:
            raise RuntimeError("Training smoke token row length mismatch")
        if not all(isinstance(token_id, int) for token_id in token_ids):
            raise RuntimeError("Training smoke token row contains a non-integer")
        if item.get("token_ids_sha256") != _token_ids_sha256(token_ids):
            raise RuntimeError("Training smoke token row hash mismatch")
        rows.append(token_ids)
    return rows, manifest

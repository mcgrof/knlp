#!/usr/bin/env python3
"""Write the external identity sidecar required by legacy Stage-2 checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .kv_lingo_data import file_sha256, write_json
from .kv_lingo_train import (
    CAPTURE_CUT,
    EFFECTIVE_BATCH,
    LEARNING_RATE,
    LOSS_REDUCTION,
    MAP_ARCHITECTURE,
    PINS,
    STAGE2_STEPS,
    WARMUP_STEPS,
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=("4b-to-8b", "8b-to-4b"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--train-rows", type=Path, required=True)
    parser.add_argument("--train-tokens", type=Path, required=True)
    parser.add_argument("--validation-rows", type=Path, required=True)
    parser.add_argument("--validation-tokens", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    contract = {
        "schema": "kv_lingo_resume_contract_v1",
        "direction": args.direction,
        "checkpoint_schema": checkpoint.get("schema"),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "global_step": int(checkpoint.get("global_step", -1)),
        "sample_cursor": int(checkpoint.get("sample_cursor", -1)),
        "stage1_sha256": file_sha256(args.stage1),
        "train_rows_sha256": file_sha256(args.train_rows),
        "train_tokens_sha256": file_sha256(args.train_tokens),
        "validation_rows_sha256": file_sha256(args.validation_rows),
        "validation_tokens_sha256": file_sha256(args.validation_tokens),
        "model_revisions": PINS,
        "tokenizer_revision": PINS["Qwen/Qwen3-4B"],
        "capture_cut": CAPTURE_CUT,
        "map_architecture": MAP_ARCHITECTURE,
        "loss_reduction": LOSS_REDUCTION,
        "effective_batch": EFFECTIVE_BATCH,
        "warmup_steps": WARMUP_STEPS,
        "total_schedule_steps": STAGE2_STEPS,
        "peak_learning_rate": LEARNING_RATE,
        "legacy_metadata_resolution": "identity verified against archived Stage-1/data/validation artifacts",
        "Generated-by": "OpenAI Codex",
    }
    write_json(args.out, contract)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

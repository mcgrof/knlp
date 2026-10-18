"""Frozen inputs for the first MoT author-code mechanics baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

UPSTREAM_URL = "https://github.com/prosopher/MoT.git"
UPSTREAM_SHA = "9753626a11b49b86ab940d87d9e5640958eb87d2"

MODIFIED_MOT_COMMITS = (
    "dd5c49d19781a9f49cc7d573c7ff735cc860c86a",
    "29b5b638b7ab343c553d9c1f18563c2c9b9d39a1",
    "bf13548f0fa2c13c7308ef0e7c3e2cf755bcbdf8",
    "98ae69ee1b0fa74841963ffb7262eefe922c2fb8",
    "663b0e865abcd23fd543f306aaa8e9f6a9f340d2",
    "94c7f53c7498e765eaa7aea74f937493b8ed21fa",
)
MODIFIED_MOT_SHA = MODIFIED_MOT_COMMITS[-1]
MODIFIED_MOT_TRAIN_SHA256 = (
    "07a3d37d74f47717e57730b936f4cdeb0debbc967c037b3b70cb94b808b36a81"
)

MODEL_REVISIONS = {
    "gpt2": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
    "gpt2-medium": "6dcaa7a952f72f9298047fd5137cd6e4f05f41da",
}

DATASET_REVISIONS = {
    "Skylion007/openwebtext": "79d93d786212f7344586290adb811d4ae6a1762c",
    "google/boolq": "35b264d03638db9f4ce671b711558bf7ff0f80d5",
    "qiaojin/PubMedQA": "9001f2853fb87cab8d220904e0de81ac6973b318",
    "edinburgh-dawg/mmlu-redux-2.0": "372ea425445d51e1ba1188c56e5e893f8138621f",
    "rajpurkar/squad": "7b6d24c440a36b6815f21b70d25016731768db1f",
    "gabrieltorresgamez/newsqa": "f26b00ef528914415cbc2013180ce52ec0831fc2",
}

FROZEN_DIAGNOSTIC_CONTENT_SHA256 = (
    "919f6ea78f021a8807a52b7f5cecfdbebacca568c09bb20cc3e60c651da107f7"
)

UPSTREAM_FILE_SHA256 = {
    "LICENSE": "28b7c8dfe3fca5af8d4faec8b413f4b5a09ac37111d09e9e1fcc855ab5cdfef2",
    "README.md": "d4963b2871291c80bcf5339bca68a8ee0283d1b2f5ade4452a77535a0cf7e2ce",
    "configs/channel_profile.json": "16971fce9e8a6002f0a94e0ccb77d3bf9743d101b95846e2abc3709ffdc8acbb",
    "configs/eval.json": "d65d99796303e34cdcdbfb93e1f9bf249e09cb96e636c963129d1f15d943050a",
    "configs/train_mot.json": "044a4c4ba894076f8868688c93f09ff6b82b393910cf3b099bea5a061c43d2ed",
    "requirements.txt": "959e311e83cc9ac0a70bfa44b8d0f15403551db01623ac5f5a70237e5459ee31",
}

MODIFIED_MOT_UPSTREAM_FILE_SHA256 = {
    **UPSTREAM_FILE_SHA256,
    "configs/train_mot.json": "e9ff8f051c4fa769286a5759bfe0d72eeab0fdd15f1380f31547affbd71a8bfa",
}


def build_train_config(
    *,
    output_path: Path,
    source_model: str = "gpt2",
    target_model: str = "gpt2-medium",
) -> dict[str, Any]:
    """Return the explicit author-code training configuration.

    The paper states bfloat16 training. The upstream default JSON uses
    float32, so this recipe records the paper value as an explicit override.
    """

    return {
        "timestamp": None,
        "output_path": str(output_path),
        "model_ids": f"{source_model},{target_model}",
        "model_directions": "A_to_B",
        "max_steps": 500,
        "batch_size": 4,
        "grad_accum_steps": 4,
        "total_tokens": 128,
        "prefix_tokens": 64,
        "learning_rate": 0.0001,
        "weight_decay": 0.01,
        "warmup_steps": 50,
        "grad_clip_norm": 1.0,
        "log_every": 25,
        "seed": 42,
        "shuffle_buffer": 50000,
        "layer_alignment": "depth-ratio",
        "min_window_size_ratio": 0.3,
        "max_window_size_ratio": 0.3,
        "translator_dim": 1024,
        "translator_heads": 16,
        "translator_depth": 2,
        "translator_mlp_ratio": 4,
        "device": "cuda",
        "dtype": "bfloat16",
        "variant": "mot",
        "mot_num_translators": 2,
        "mot_top_k": 1,
        "topk_sparse_attn": 128,
        "num_bottom_full_attn": 2,
        "attention_backend": "dense_selected",
        "require_identity_token_ledgers": True,
    }


def build_channel_profile_config() -> dict[str, Any]:
    return {
        "num_train_examples": 128,
        "num_val_examples": 64,
        "max_steps": 128,
        "min_window_size": 1,
        "translator_dim": 1024,
        "translator_heads": 16,
        "translator_depth": 2,
        "translator_mlp_ratio": 4,
    }


def build_eval_config(
    *, output_path: Path, checkpoint_dir_path: Path
) -> dict[str, Any]:
    return {
        "outputs_path": str(output_path.parent),
        "timestamp": None,
        "output_path": str(output_path),
        "checkpoint_dir_path": str(checkpoint_dir_path),
        "device": "cuda",
        "batch_size": 4,
        "num_workers": 0,
        "max_examples_per_dataset": 100,
        "seed": 42,
        "shuffle_eval_stream": False,
        "shuffle_buffer": 50000,
        "generation_max_new_tokens": 64,
    }


def build_reproduction_matrix() -> str:
    return """# MoT reproduction matrix

This record distinguishes the paper, the pinned author code, and this
mechanics baseline. It is generated before model execution.

| Item | Paper | Pinned author code | Frozen baseline |
| --- | --- | --- | --- |
| Direction | GPT-2 to GPT-2-medium ablation | Configurable | GPT-2 to GPT-2-medium |
| Training data | OpenWebText | OpenWebText stream | Pinned OpenWebText revision |
| Sequence | 64 context + 64 suffix tokens | Configurable | 64 + 64 |
| Optimizer | AdamW, 500 steps, effective batch 16 | Configurable | Same |
| Precision | bfloat16 | Default JSON is float32 | bfloat16 override |
| Router | Top-1 of two translators | Top-1 of two | Unchanged author behavior |
| Channel map | Depth ratio, ratio 0.3 | Depth ratio | Same |
| Sparse replay | Top-S, two full bottom layers | Top-128, two layers | Same |
| Attention backend | Not specified | Gathered selected keys | Dense selected-key correctness path |
| Evaluation | F1 0.34 in Table 8 | Five datasets, 100-item cap | Same author evaluator |
| Source work | Source context cache | KV prefill plus attention-guidance forward | Count both |
| Receiver work | Context replay | Complete target stack replay | Count all layers |

The paper does not specify enough detail to treat the Table 8 F1 value as an
exact reproduction target. The first run is therefore an author-code mechanics baseline.
It includes the matched native target path from the author evaluator.
The dense selected-key backend preserves the selection policy but uses
quadratic working storage; it makes no sparse-compute or serving-speed claim.
"""

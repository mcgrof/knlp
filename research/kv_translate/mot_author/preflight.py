"""Real-model CPU preflight for the pinned MoT author implementation."""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any

from .recipe import MODEL_REVISIONS

PREFLIGHT_THRESHOLDS = {
    "native_continuation_max_abs": 1.0e-4,
    "author_replay_logits_max_abs": 1.0e-4,
    "author_replay_logits_min_cosine": 0.99999,
    "author_replay_cache_max_abs": 2.0e-5,
}

TOKENIZER_CASES = (
    "KV cache reuse starts here.",
    " leading space and two  spaces\nnext line",
    "Unicode: café, 東京, Δx, and 🙂.",
    "<|endoftext|> literal marker",
)

MODEL_PROMPT = (
    "KV cache continuation must match a full causal decoder forward pass "
    "exactly enough for a frozen preflight."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_model_snapshot(
    path: Path, *, model_name: str, expected_revision: str
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise RuntimeError(f"Model snapshot directory does not exist: {resolved}")
    if resolved.name != expected_revision:
        raise RuntimeError(
            f"{model_name} snapshot revision mismatch: "
            f"{resolved.name} != {expected_revision}"
        )

    config_path = resolved / "config.json"
    if not config_path.is_file():
        raise RuntimeError(f"Missing model config: {config_path}")
    weight_files = sorted(
        path
        for pattern in ("*.safetensors", "pytorch_model*.bin")
        for path in resolved.glob(pattern)
        if path.is_file()
    )
    if not weight_files:
        raise RuntimeError(f"No model weights found in {resolved}")

    return {
        "model_name": model_name,
        "expected_revision": expected_revision,
        "snapshot_revision": resolved.name,
        "revision_matches": True,
        "path": str(resolved),
        "config_sha256": _sha256(config_path),
        "weight_files": [
            {"name": item.name, "size_bytes": item.stat().st_size}
            for item in weight_files
        ],
    }


def _load_author_modules(source: Path) -> dict[str, Any]:
    source_text = str(source.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    return {
        "common": importlib.import_module("core.common"),
        "model": importlib.import_module("core.model"),
        "model_spec": importlib.import_module("core.model_spec"),
        "train": importlib.import_module("alg.mot.train"),
    }


def _tokenizer_ledger(common, snapshots: dict[str, Path]) -> dict[str, Any]:
    tokenizers = {
        name: common.load_tokenizer(str(path)) for name, path in snapshots.items()
    }
    cases: list[dict[str, Any]] = []
    for text in TOKENIZER_CASES:
        encodings = {
            name: tokenizer.encode(text, add_special_tokens=False)
            for name, tokenizer in tokenizers.items()
        }
        decoded = {
            name: tokenizer.decode(encodings[name])
            for name, tokenizer in tokenizers.items()
        }
        cases.append(
            {
                "text": text,
                "token_ids": encodings,
                "decoded": decoded,
                "token_ids_equal": len({tuple(ids) for ids in encodings.values()}) == 1,
                "decoded_equal": len(set(decoded.values())) == 1,
            }
        )

    vocab_hashes = {}
    for name, tokenizer in tokenizers.items():
        vocabulary = json.dumps(
            tokenizer.get_vocab(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        vocab_hashes[name] = hashlib.sha256(vocabulary).hexdigest()

    return {
        "models": {
            name: {
                "class": type(tokenizer).__name__,
                "vocab_size": len(tokenizer),
                "vocab_sha256": vocab_hashes[name],
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
            }
            for name, tokenizer in tokenizers.items()
        },
        "cases": cases,
        "all_token_ids_equal": all(item["token_ids_equal"] for item in cases),
        "all_decoded_text_equal": all(item["decoded_equal"] for item in cases),
        "vocabulary_equal": len(set(vocab_hashes.values())) == 1,
    }


def _tensor_max_abs(left, right) -> float:
    return float((left.float() - right.float()).abs().max().item())


def _model_trial(
    *,
    torch,
    common,
    model_module,
    model_spec_module,
    author_train,
    model_name: str,
    snapshot: Path,
) -> dict[str, Any]:
    tokenizer = common.load_tokenizer(str(snapshot))
    raw_model = common.load_frozen_model(str(snapshot), "cpu", "float32")
    wrapped = model_module.Model(str(snapshot), raw_model, tokenizer)
    input_ids = tokenizer(
        MODEL_PROMPT,
        return_tensors="pt",
        add_special_tokens=False,
    )[
        "input_ids"
    ][:, :20]
    if input_ids.shape[1] < 2:
        raise RuntimeError("The fixed preflight prompt produced fewer than two tokens")

    with torch.inference_mode():
        direct = raw_model(input_ids=input_ids, use_cache=True, return_dict=True)
        split = input_ids.shape[1] - 1
        prefix = raw_model(
            input_ids=input_ids[:, :split], use_cache=True, return_dict=True
        )
        continuation = raw_model(
            input_ids=input_ids[:, split:],
            past_key_values=prefix.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        continuation_max_abs = _tensor_max_abs(
            direct.logits[:, split:, :], continuation.logits
        )

        author_ids = common.TokenIDs(input_ids, model_id=wrapped.id)
        hidden_states = author_train.build_gpt2_input_hidden_states(wrapped, author_ids)
        replay_cache = []
        transformer = author_train.require_gpt2_transformer(wrapped)
        for block in transformer.h:
            hidden_states, present = author_train.run_gpt2_block(block, hidden_states)
            replay_cache.append(present)
        replay_logits = raw_model.lm_head(transformer.ln_f(hidden_states))
        replay_logits_max_abs = _tensor_max_abs(direct.logits, replay_logits)
        replay_logits_cosine = float(
            torch.nn.functional.cosine_similarity(
                direct.logits.flatten().float(),
                replay_logits.flatten().float(),
                dim=0,
            ).item()
        )
        replay_cache_max_abs = max(
            _tensor_max_abs(native_tensor, replay_tensor)
            for native_layer, replay_layer in zip(direct.past_key_values, replay_cache)
            for native_tensor, replay_tensor in zip(native_layer, replay_layer)
        )

    spec = model_spec_module.infer_model_spec_from_config(raw_model.config)
    result = {
        "model_name": model_name,
        "family": author_train.resolve_target_model_family(
            wrapped, target_model_id=model_name
        ),
        "prompt_token_ids": input_ids.tolist(),
        "prompt_token_count": int(input_ids.shape[1]),
        "model_spec": {
            "num_layers": spec.num_layers,
            "hidden_size": spec.hidden_size,
            "num_heads": spec.num_heads,
            "num_key_value_heads": spec.num_key_value_heads,
            "head_dim": spec.head_dim,
        },
        "native_continuation_max_abs": continuation_max_abs,
        "author_replay_logits_max_abs": replay_logits_max_abs,
        "author_replay_logits_cosine": replay_logits_cosine,
        "author_replay_cache_max_abs": replay_cache_max_abs,
    }
    result["native_continuation_pass"] = (
        continuation_max_abs <= PREFLIGHT_THRESHOLDS["native_continuation_max_abs"]
    )
    result["author_replay_pass"] = (
        replay_logits_max_abs <= PREFLIGHT_THRESHOLDS["author_replay_logits_max_abs"]
        and replay_logits_cosine
        >= PREFLIGHT_THRESHOLDS["author_replay_logits_min_cosine"]
        and replay_cache_max_abs <= PREFLIGHT_THRESHOLDS["author_replay_cache_max_abs"]
    )

    del direct
    del prefix
    del continuation
    del replay_cache
    del replay_logits
    del hidden_states
    del wrapped
    del raw_model
    gc.collect()
    return result


def run_real_model_preflight(
    *, source: Path, source_model: Path, target_model: Path
) -> dict[str, Any]:
    """Run the no-training GPT-2 author-code preflight on CPU."""

    snapshots = {
        "gpt2": source_model.expanduser().resolve(),
        "gpt2-medium": target_model.expanduser().resolve(),
    }
    snapshot_receipts = {
        name: inspect_model_snapshot(
            path,
            model_name=name,
            expected_revision=MODEL_REVISIONS[name],
        )
        for name, path in snapshots.items()
    }
    modules = _load_author_modules(source)
    torch = importlib.import_module("torch")
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))

    tokenizer = _tokenizer_ledger(modules["common"], snapshots)
    trials = {
        name: _model_trial(
            torch=torch,
            common=modules["common"],
            model_module=modules["model"],
            model_spec_module=modules["model_spec"],
            author_train=modules["train"],
            model_name=name,
            snapshot=path,
        )
        for name, path in snapshots.items()
    }
    tokenizer_pass = (
        tokenizer["all_token_ids_equal"]
        and tokenizer["all_decoded_text_equal"]
        and tokenizer["vocabulary_equal"]
    )
    model_pass = all(
        trial["native_continuation_pass"] and trial["author_replay_pass"]
        for trial in trials.values()
    )

    return {
        "schema": "knlp_mot_author_real_model_preflight_v1",
        "scope": "cpu_no_training_same_tokenizer_gpt2_lane",
        "thresholds": PREFLIGHT_THRESHOLDS,
        "snapshots": snapshot_receipts,
        "tokenizer": tokenizer,
        "models": trials,
        "decision": {
            "shared_token_grid_confirmed": tokenizer_pass,
            "native_kv_continuation_confirmed": all(
                trial["native_continuation_pass"] for trial in trials.values()
            ),
            "author_replay_parity_confirmed": all(
                trial["author_replay_pass"] for trial in trials.values()
            ),
            "gpt2_author_training_preflight_pass": tokenizer_pass and model_pass,
            "paper_result_reproduction_admitted": False,
            "cross_token_alignment_safety_admitted": False,
        },
        "limitations": [
            "This CPU preflight does not train or evaluate a translator.",
            "The shared GPT-2 tokenizer does not exercise unequal token grids.",
            "The Top-1 router and execution-cost findings remain unchanged.",
            "Thresholds bound numerical parity only; they are not quality gates.",
        ],
        "runtime": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "transformers": str(importlib.import_module("transformers").__version__),
        },
    }

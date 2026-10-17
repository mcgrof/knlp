"""Targeted sparse-replay and token-alignment controls for MoT."""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from .preflight import (
    MODEL_PROMPT,
    PREFLIGHT_THRESHOLDS,
    TOKENIZER_CASES,
    inspect_model_snapshot,
)
from .recipe import (
    MODEL_REVISIONS,
    MODIFIED_MOT_COMMITS,
    MODIFIED_MOT_SHA,
    MODIFIED_MOT_TRAIN_SHA256,
    UPSTREAM_SHA,
)

TRAINING_FIXTURE = (
    "Mixture of Translators sparse replay must preserve the native GPT-2 "
    "attention temperature when every causal key is selected. "
)

SOURCE_FILES = (
    "alg/mot/train.py",
    "alg/mot/eval.py",
    "core/common.py",
    "core/channel_profiler.py",
    "core/model.py",
    "core/model_spec.py",
    "core/gemma3_compat.py",
    "core/llama32_compat.py",
    "core/qwen2_compat.py",
    "core/qwen3_compat.py",
)

ORIGINAL_TRAIN_SHA256 = (
    "14f3c48bd94ebb8eb6a76ac1bce75d47d9a223152574f848e00964c094a6feed"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(source: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=source,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def inspect_mechanics_source(source: Path, *, variant: str) -> dict[str, Any]:
    source = source.expanduser().resolve()
    if not (source / ".git").exists():
        raise RuntimeError(f"Not a git checkout: {source}")
    head = _git(source, "rev-parse", "HEAD")
    clean = not _git(source, "status", "--porcelain")
    expected_head = UPSTREAM_SHA if variant == "unchanged-author" else MODIFIED_MOT_SHA
    if head != expected_head:
        raise RuntimeError(f"{variant} source SHA mismatch: {head} != {expected_head}")
    if not clean:
        raise RuntimeError(f"{variant} source checkout is modified")

    patch_commits: list[str] = []
    if variant == "modified-mot":
        patch_commits = _git(
            source,
            "rev-list",
            "--reverse",
            f"{UPSTREAM_SHA}..{head}",
        ).splitlines()
        if tuple(patch_commits) != MODIFIED_MOT_COMMITS:
            raise RuntimeError(
                "Modified-MoT patch chain differs from the pinned commit chain"
            )

    file_hashes = {relative: _sha256(source / relative) for relative in SOURCE_FILES}
    expected_train_hash = (
        ORIGINAL_TRAIN_SHA256
        if variant == "unchanged-author"
        else MODIFIED_MOT_TRAIN_SHA256
    )
    if file_hashes["alg/mot/train.py"] != expected_train_hash:
        raise RuntimeError("Pinned MoT train.py content hash mismatch")
    return {
        "variant": variant,
        "source": str(source),
        "upstream_base": UPSTREAM_SHA,
        "expected_head": expected_head,
        "actual_head": head,
        "worktree_clean": clean,
        "patch_commits": patch_commits,
        "file_sha256": file_hashes,
    }


def _load_author_modules(source: Path) -> dict[str, Any]:
    source_text = str(source.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    modules = {
        "common": importlib.import_module("core.common"),
        "model": importlib.import_module("core.model"),
        "model_spec": importlib.import_module("core.model_spec"),
        "train": importlib.import_module("alg.mot.train"),
        "eval": importlib.import_module("alg.mot.eval"),
        "channel_profiler": importlib.import_module("core.channel_profiler"),
    }
    return modules


def _snapshot_content_receipt(path: Path, *, model_name: str) -> dict[str, Any]:
    receipt = inspect_model_snapshot(
        path,
        model_name=model_name,
        expected_revision=MODEL_REVISIONS[model_name],
    )
    selected_names = {
        "config.json",
        "merges.txt",
        "model.safetensors",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
    selected = sorted(
        item
        for item in path.expanduser().resolve().iterdir()
        if item.is_file()
        and (
            item.name in selected_names
            or item.name.startswith("model-")
            or item.name.startswith("pytorch_model-")
        )
    )
    receipt["content_files"] = [
        {
            "name": item.name,
            "size_bytes": item.stat().st_size,
            "sha256": _sha256(item),
        }
        for item in selected
    ]
    return receipt


def _max_abs(left, right) -> float:
    return float((left.float() - right.float()).abs().max().item())


def _fixed_prompt(tokenizer, token_count: int):
    text = MODEL_PROMPT if token_count == 20 else TRAINING_FIXTURE * 32
    input_ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )[
        "input_ids"
    ][:, :token_count]
    if input_ids.shape[1] != token_count:
        raise RuntimeError(
            f"Fixed prompt produced {input_ids.shape[1]} tokens, expected {token_count}"
        )
    return input_ids


def _sparse_model_trial(
    *, torch, common, model_module, author_train, model_name: str, snapshot: Path
) -> dict[str, Any]:
    tokenizer = common.load_tokenizer(str(snapshot))
    raw_model = common.load_frozen_model(str(snapshot), "cpu", "float32")
    wrapped = model_module.Model(str(snapshot), raw_model, tokenizer)
    transformer = author_train.require_gpt2_transformer(wrapped)
    lengths: dict[str, Any] = {}

    for token_count in (20, 64):
        input_ids = _fixed_prompt(tokenizer, token_count)
        token_ids = common.TokenIDs(input_ids, model_id=wrapped.id)
        full_indices = (
            torch.arange(token_count, dtype=torch.long)
            .view(1, 1, 1, token_count)
            .expand(1, 1, token_count, token_count)
        )
        with torch.inference_mode():
            native = raw_model(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True,
            )
            dense_hidden = author_train.build_gpt2_input_hidden_states(
                wrapped, token_ids
            )
            sparse_hidden = dense_hidden.clone()
            block_output_max_abs = 0.0
            returned_cache_max_abs = 0.0
            attention_flags = []
            for layer_index, block in enumerate(transformer.h):
                block_input = dense_hidden
                dense_hidden, dense_present = author_train.run_gpt2_block(
                    block, block_input
                )
                sparse_block_output, sparse_present = author_train.run_gpt2_block(
                    block,
                    block_input,
                    sparse_attention_indices=full_indices,
                    injected_key=dense_present[0],
                    injected_value=dense_present[1],
                )
                block_output_max_abs = max(
                    block_output_max_abs,
                    _max_abs(dense_hidden, sparse_block_output),
                )
                returned_cache_max_abs = max(
                    returned_cache_max_abs,
                    *(
                        _max_abs(dense_tensor, sparse_tensor)
                        for dense_tensor, sparse_tensor in zip(
                            dense_present, sparse_present
                        )
                    ),
                )
                native_key, native_value = native.past_key_values[layer_index]
                sparse_hidden, _ = author_train.run_gpt2_block(
                    block,
                    sparse_hidden,
                    sparse_attention_indices=full_indices,
                    injected_key=native_key,
                    injected_value=native_value,
                )
                attention_flags.append(
                    {
                        "layer": layer_index,
                        "scale_attn_weights": bool(
                            getattr(block.attn, "scale_attn_weights", True)
                        ),
                        "scale_attn_by_inverse_layer_idx": bool(
                            getattr(
                                block.attn,
                                "scale_attn_by_inverse_layer_idx",
                                False,
                            )
                        ),
                    }
                )

            dense_logits = raw_model.lm_head(transformer.ln_f(dense_hidden))
            sparse_logits = raw_model.lm_head(transformer.ln_f(sparse_hidden))
            dense_native_max_abs = _max_abs(dense_logits, native.logits)
            sparse_native_max_abs = _max_abs(sparse_logits, native.logits)
            sparse_native_cosine = float(
                torch.nn.functional.cosine_similarity(
                    sparse_logits.flatten().float(),
                    native.logits.flatten().float(),
                    dim=0,
                ).item()
            )

        pass_gate = (
            block_output_max_abs <= PREFLIGHT_THRESHOLDS["author_replay_logits_max_abs"]
            and returned_cache_max_abs
            <= PREFLIGHT_THRESHOLDS["author_replay_cache_max_abs"]
            and sparse_native_max_abs
            <= PREFLIGHT_THRESHOLDS["author_replay_logits_max_abs"]
            and sparse_native_cosine
            >= PREFLIGHT_THRESHOLDS["author_replay_logits_min_cosine"]
        )
        lengths[str(token_count)] = {
            "token_count": token_count,
            "prompt_token_ids": input_ids.tolist(),
            "sparse_indices_shape": list(full_indices.shape),
            "every_key_supplied_once_per_query": True,
            "native_kv_injected_at_every_layer": True,
            "sparse_branch_calls": len(transformer.h),
            "injected_branch_calls": len(transformer.h),
            "block_output_max_abs": block_output_max_abs,
            "returned_cache_max_abs": returned_cache_max_abs,
            "dense_author_vs_native_logits_max_abs": dense_native_max_abs,
            "full_sparse_vs_native_logits_max_abs": sparse_native_max_abs,
            "full_sparse_vs_native_logits_cosine": sparse_native_cosine,
            "pass": pass_gate,
            "attention_flags": attention_flags,
        }

        del native
        del dense_hidden
        del sparse_hidden
        del dense_logits
        del sparse_logits
        gc.collect()

    result = {
        "model_name": model_name,
        "layer_count": len(transformer.h),
        "lengths": lengths,
        "pass": all(item["pass"] for item in lengths.values()),
    }
    del wrapped
    del raw_model
    gc.collect()
    return result


def _gpt2_reduced_reference_block(
    *, torch, author_train, block, hidden_states, sparse_indices, injected_key, injected_value
):
    """Independent dense-mask reference for the same selected causal keys."""
    residual = hidden_states
    attn = block.attn
    attn_input = block.ln_1(hidden_states)
    qkv = attn.c_attn(attn_input)
    split_size = getattr(attn, "split_size", qkv.shape[-1] // 3)
    query, native_key, native_value = qkv.split(split_size, dim=2)
    batch_size, seq_len, _ = query.shape
    num_heads = int(attn.num_heads)
    head_dim = int(attn.head_dim)
    query = query.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    native_key = native_key.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    native_value = native_value.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

    indices = author_train.expand_sparse_attention_indices(sparse_indices, num_heads)
    selected = torch.zeros(
        batch_size,
        num_heads,
        seq_len,
        seq_len,
        dtype=torch.bool,
        device=hidden_states.device,
    )
    selected.scatter_(dim=-1, index=indices.clamp(0, seq_len - 1), value=True)
    query_positions = torch.arange(seq_len, device=hidden_states.device).view(1, 1, seq_len, 1)
    key_positions = torch.arange(seq_len, device=hidden_states.device).view(1, 1, 1, seq_len)
    valid = selected & (key_positions <= query_positions)
    if not bool(valid.any(dim=-1).all().item()):
        raise RuntimeError("Reduced-key dense reference contains an all-masked query row")

    scores = torch.matmul(query, injected_key.transpose(-1, -2))
    if bool(getattr(attn, "scale_attn_weights", True)):
        scores = scores / (head_dim ** 0.5)
    if bool(getattr(attn, "scale_attn_by_inverse_layer_idx", False)):
        layer_idx = getattr(attn, "layer_idx", None)
        if layer_idx is None:
            raise RuntimeError("GPT-2 inverse-layer scaling requires layer_idx")
        scores = scores / float(int(layer_idx) + 1)
    scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    output = torch.matmul(weights, injected_value)
    output = output.permute(0, 2, 1, 3).contiguous().view(
        batch_size, seq_len, num_heads * head_dim
    )
    output = attn.c_proj(output)
    output = attn.resid_dropout(output)
    hidden_states = residual + output
    hidden_states = hidden_states + block.mlp(block.ln_2(hidden_states))
    return hidden_states, (native_key, native_value), valid


def _reduced_sparse_model_trial(
    *, torch, common, model_module, author_train, model_name: str, snapshot: Path
) -> dict[str, Any]:
    token_count = 160
    top_k = 128
    bottom_full_layers = 2
    tokenizer = common.load_tokenizer(str(snapshot))
    raw_model = common.load_frozen_model(str(snapshot), "cpu", "float32")
    wrapped = model_module.Model(str(snapshot), raw_model, tokenizer)
    transformer = author_train.require_gpt2_transformer(wrapped)
    if any(bool(getattr(block.attn, "reorder_and_upcast_attn", False)) for block in transformer.h):
        raise RuntimeError("Reduced GPT-2 control requires reorder_and_upcast_attn=False")
    input_ids = _fixed_prompt(tokenizer, token_count)
    token_ids = common.TokenIDs(input_ids, model_id=wrapped.id)
    layer_indices = list(range(len(transformer.h)))
    sparse_by_layer = author_train.build_extrapolated_sparse_attention_indices(
        wrapped,
        token_ids,
        source_layer_indices=layer_indices,
        target_layer_indices=layer_indices,
        num_source_layers=len(layer_indices),
        num_target_layers=len(layer_indices),
        source_model_id=model_name,
        top_k=top_k,
    )
    selector_rows = []
    for layer_idx, indices in enumerate(sparse_by_layer):
        invalid = author_train.build_sparse_query_mask(
            indices, seq_len=token_count, num_heads=int(transformer.h[layer_idx].attn.num_heads)
        )
        valid_counts = (~invalid).sum(dim=-1)
        selector_rows.append(
            {
                "layer": layer_idx,
                "shape": list(indices.shape),
                "min_valid_keys_per_query": int(valid_counts.min().item()),
                "max_valid_keys_per_query": int(valid_counts.max().item()),
                "all_masked_query_count": int((valid_counts == 0).sum().item()),
            }
        )
    if any(row["all_masked_query_count"] for row in selector_rows):
        raise RuntimeError(f"{model_name} selector produced an all-masked query")

    with torch.inference_mode():
        native = raw_model(input_ids=input_ids, use_cache=True, return_dict=True)
        actual_hidden = author_train.build_gpt2_input_hidden_states(wrapped, token_ids)
        reference_hidden = actual_hidden.clone()
        block_max_abs = 0.0
        block_max_rel = 0.0
        cache_max_abs = 0.0
        reduced_branch_calls = 0
        injected_branch_calls = 0
        for layer_idx, block in enumerate(transformer.h):
            if layer_idx < bottom_full_layers:
                actual_hidden, actual_present = author_train.run_gpt2_block(block, actual_hidden)
                reference_hidden, reference_present = author_train.run_gpt2_block(block, reference_hidden)
            else:
                native_key, native_value = native.past_key_values[layer_idx]
                actual_hidden, actual_present = author_train.run_gpt2_block(
                    block,
                    actual_hidden,
                    sparse_attention_indices=sparse_by_layer[layer_idx],
                    injected_key=native_key,
                    injected_value=native_value,
                )
                reference_hidden, reference_present, _ = _gpt2_reduced_reference_block(
                    torch=torch,
                    author_train=author_train,
                    block=block,
                    hidden_states=reference_hidden,
                    sparse_indices=sparse_by_layer[layer_idx],
                    injected_key=native_key,
                    injected_value=native_value,
                )
                reduced_branch_calls += 1
                injected_branch_calls += 1
            block_abs = _max_abs(actual_hidden, reference_hidden)
            reference_scale = max(float(reference_hidden.float().abs().max().item()), 1e-12)
            block_max_abs = max(block_max_abs, block_abs)
            block_max_rel = max(block_max_rel, block_abs / reference_scale)
            cache_max_abs = max(
                cache_max_abs,
                *(
                    _max_abs(actual_tensor, reference_tensor)
                    for actual_tensor, reference_tensor in zip(actual_present, reference_present)
                ),
            )
        actual_logits = raw_model.lm_head(transformer.ln_f(actual_hidden))
        reference_logits = raw_model.lm_head(transformer.ln_f(reference_hidden))
        logits_max_abs = _max_abs(actual_logits, reference_logits)
        logits_scale = max(float(reference_logits.float().abs().max().item()), 1e-12)
        logits_max_rel = logits_max_abs / logits_scale
        logits_cosine = float(
            torch.nn.functional.cosine_similarity(
                actual_logits.flatten().float(),
                reference_logits.flatten().float(),
                dim=0,
            ).item()
        )

    all_masked_indices = torch.full((1, 1, token_count, 1), token_count - 1, dtype=torch.long)
    all_masked = author_train.build_sparse_query_mask(
        all_masked_indices, seq_len=token_count, num_heads=1
    )
    negative_control = bool(all_masked[0, 0, 0].all().item())
    pass_gate = (
        block_max_abs <= PREFLIGHT_THRESHOLDS["author_replay_logits_max_abs"]
        and cache_max_abs <= PREFLIGHT_THRESHOLDS["author_replay_cache_max_abs"]
        and logits_max_abs <= PREFLIGHT_THRESHOLDS["author_replay_logits_max_abs"]
        and logits_cosine >= PREFLIGHT_THRESHOLDS["author_replay_logits_min_cosine"]
        and negative_control
    )
    result = {
        "model_name": model_name,
        "token_count": token_count,
        "top_s": top_k,
        "bottom_full_attention_layers": bottom_full_layers,
        "reduced_branch_calls": reduced_branch_calls,
        "native_kv_injection_calls": injected_branch_calls,
        "attention_flags": {
            "reorder_and_upcast_attn": False,
            "scale_attn_weights": [
                bool(getattr(block.attn, "scale_attn_weights", True)) for block in transformer.h
            ],
            "scale_attn_by_inverse_layer_idx": [
                bool(getattr(block.attn, "scale_attn_by_inverse_layer_idx", False))
                for block in transformer.h
            ],
        },
        "selector": selector_rows,
        "block_output_max_abs": block_max_abs,
        "block_output_max_rel": block_max_rel,
        "returned_cache_max_abs": cache_max_abs,
        "full_stack_logits_max_abs": logits_max_abs,
        "full_stack_logits_max_rel": logits_max_rel,
        "full_stack_logits_cosine": logits_cosine,
        "all_masked_negative_control": negative_control,
        "pass": pass_gate,
    }
    del native, wrapped, raw_model, actual_hidden, reference_hidden, actual_logits, reference_logits
    gc.collect()
    return result


def _identity_selected(
    author_train, source_model, target_model, source_ids, target_ids
):
    identity_helper = getattr(
        author_train, "token_ledgers_have_identity_alignment", None
    )
    if identity_helper is not None:
        return bool(identity_helper(source_model, target_model, source_ids, target_ids))
    return not (
        source_model.id != target_model.id
        or source_ids.shape != target_ids.shape
        or not author_train.torch.equal(
            source_ids.as_tensor().detach().cpu(),
            target_ids.as_tensor().detach().cpu(),
        )
    )


def _effective_alignment(
    *, torch, author_train, source_model, target_model, source_ids, target_ids
):
    weights, target_to_source, source_to_target = (
        author_train.build_cross_token_alignment(
            source_model,
            target_model,
            source_ids,
            target_ids,
        )
    )
    identity = _identity_selected(
        author_train, source_model, target_model, source_ids, target_ids
    )
    if identity:
        seq_len = source_ids.shape[1]
        weights = torch.eye(seq_len, dtype=torch.float32).unsqueeze(0)
        target_to_source = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        source_to_target = target_to_source.clone()
    return weights, target_to_source, source_to_target, identity


def _future_weight(weights, torch) -> float:
    seq_len = weights.shape[1]
    future = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1
    ).unsqueeze(0)
    return float(weights[future].max().item()) if torch.any(weights[future]) else 0.0


def _alignment_trial(
    *,
    torch,
    common,
    model_module,
    author_train,
    source_snapshot: Path,
    target_snapshot: Path,
) -> dict[str, Any]:
    source_tokenizer = common.load_tokenizer(str(source_snapshot))
    target_tokenizer = common.load_tokenizer(str(target_snapshot))
    source_raw = common.load_frozen_model(str(source_snapshot), "cpu", "float32")
    source_model = model_module.Model(
        str(source_snapshot), source_raw, source_tokenizer
    )

    class TokenizerModel:
        def __init__(self, model_id, tokenizer):
            self.id = model_id
            self.tokenizer = tokenizer

    target_model = TokenizerModel(str(target_snapshot), target_tokenizer)
    cases = []
    unicode_case = None
    for text in TOKENIZER_CASES:
        source_tensor = source_tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        target_tensor = target_tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        source_ids = common.TokenIDs(source_tensor, model_id=source_model.id)
        target_ids = common.TokenIDs(target_tensor, model_id=target_model.id)
        decoded = author_train._decode_token_ids(
            target_tokenizer, target_tensor[0].tolist()
        )
        source_spans = author_train._token_spans_for_ids(
            source_tokenizer, source_tensor[0].tolist(), decoded
        )
        target_spans = author_train._token_spans_for_ids(
            target_tokenizer, target_tensor[0].tolist(), decoded
        )
        raw_weights, _, _ = author_train.build_cross_token_alignment(
            source_model, target_model, source_ids, target_ids
        )
        weights, target_to_source, source_to_target, identity = _effective_alignment(
            torch=torch,
            author_train=author_train,
            source_model=source_model,
            target_model=target_model,
            source_ids=source_ids,
            target_ids=target_ids,
        )
        source_sparse = author_train.extract_source_attention_topk_indices(
            source_model,
            source_ids,
            [0],
            source_model_id="gpt2",
            top_k=source_ids.shape[1],
        )[0]
        aligned_sparse = (
            source_sparse
            if identity
            else author_train.align_sparse_attention_indices_to_target_tokens(
                source_sparse,
                target_to_source=target_to_source,
                source_to_target=source_to_target,
            )
        )
        invalid = author_train.build_sparse_query_mask(
            aligned_sparse,
            seq_len=target_ids.shape[1],
            num_heads=1,
        )
        case = {
            "text": text,
            "token_ids": source_tensor.tolist(),
            "source_spans": source_spans,
            "target_spans": target_spans,
            "raw_alignment_is_identity": bool(
                torch.equal(
                    raw_weights,
                    torch.eye(source_ids.shape[1]).unsqueeze(0),
                )
            ),
            "identity_path_selected": identity,
            "effective_future_source_weight_max": _future_weight(weights, torch),
            "raw_future_source_weight_max": _future_weight(raw_weights, torch),
            "target_to_source": target_to_source.tolist(),
            "source_to_target": source_to_target.tolist(),
            "sparse_all_masked_query_count": int(invalid.all(dim=-1).sum().item()),
        }
        cases.append(case)
        if text.startswith("Unicode:"):
            unicode_case = (source_ids, target_ids)

    if unicode_case is None:
        raise RuntimeError("Unicode alignment fixture is missing")
    full_source_ids, full_target_ids = unicode_case
    prefix_len = 7
    prefix_source_ids = full_source_ids[:, :prefix_len]
    prefix_target_ids = full_target_ids[:, :prefix_len]
    prefix_weights, _, _, prefix_identity = _effective_alignment(
        torch=torch,
        author_train=author_train,
        source_model=source_model,
        target_model=target_model,
        source_ids=prefix_source_ids,
        target_ids=prefix_target_ids,
    )
    full_weights, _, _, full_identity = _effective_alignment(
        torch=torch,
        author_train=author_train,
        source_model=source_model,
        target_model=target_model,
        source_ids=full_source_ids,
        target_ids=full_target_ids,
    )
    with torch.inference_mode():
        prefix_native = source_raw(
            prefix_source_ids.as_tensor(), use_cache=True, return_dict=True
        ).past_key_values
        full_native = source_raw(
            full_source_ids.as_tensor(), use_cache=True, return_dict=True
        ).past_key_values
    prefix_key, prefix_value = common.past_key_values_to_blocks(prefix_native)
    full_key, full_value = common.past_key_values_to_blocks(full_native)
    aligned_prefix_key, aligned_prefix_value = (
        author_train.align_cache_blocks_to_target_tokens(
            prefix_key,
            prefix_value,
            alignment_weights=prefix_weights,
        )
    )
    aligned_full_key, aligned_full_value = (
        author_train.align_cache_blocks_to_target_tokens(
            full_key,
            full_value,
            alignment_weights=full_weights,
        )
    )
    native_prefix_max_abs = max(
        _max_abs(prefix_key, full_key[:, :prefix_len]),
        _max_abs(prefix_value, full_value[:, :prefix_len]),
    )
    aligned_prefix_max_abs = max(
        _max_abs(aligned_prefix_key, aligned_full_key[:, :prefix_len]),
        _max_abs(aligned_prefix_value, aligned_full_value[:, :prefix_len]),
    )

    _, target_to_source, source_to_target = author_train._build_overlap_weights(
        [(0, 4)], [(0, 1), (1, 4)]
    )
    unequal_indices = torch.tensor([[[[0]]]], dtype=torch.long)
    unequal_aligned = author_train.align_sparse_attention_indices_to_target_tokens(
        unequal_indices,
        target_to_source=target_to_source.unsqueeze(0),
        source_to_target=source_to_target.unsqueeze(0),
    )
    unequal_invalid = author_train.build_sparse_query_mask(
        unequal_aligned,
        seq_len=2,
        num_heads=1,
    )

    max_future = max(item["effective_future_source_weight_max"] for item in cases)
    all_identity = all(item["identity_path_selected"] for item in cases)
    prefix_pass = (
        aligned_prefix_max_abs <= PREFLIGHT_THRESHOLDS["author_replay_cache_max_abs"]
    )
    result = {
        "cases": cases,
        "all_same_grid_cases_use_identity_path": all_identity,
        "effective_future_source_weight_max": max_future,
        "prefix_integrity": {
            "fixture": TOKENIZER_CASES[2],
            "prefix_token_count": prefix_len,
            "prefix_token_ids": prefix_source_ids.tolist(),
            "prefix_identity_path_selected": prefix_identity,
            "full_identity_path_selected": full_identity,
            "native_prefix_cache_max_abs": native_prefix_max_abs,
            "effective_aligned_prefix_cache_max_abs": aligned_prefix_max_abs,
            "threshold": PREFLIGHT_THRESHOLDS["author_replay_cache_max_abs"],
            "pass": prefix_pass,
        },
        "unequal_grid_witness": {
            "aligned_indices": unequal_aligned.tolist(),
            "invalid_mask": unequal_invalid.tolist(),
            "first_query_all_candidates_invalid": bool(
                unequal_invalid[0, 0, 0].all().item()
            ),
            "cross_family_admitted": False,
        },
        "pass": all_identity and max_future == 0.0 and prefix_pass,
    }
    del source_model
    del source_raw
    gc.collect()
    return result


def run_mechanics_preflight(
    *,
    source: Path,
    source_model: Path,
    target_model: Path,
    variant: str,
) -> dict[str, Any]:
    if variant not in {"unchanged-author", "modified-mot"}:
        raise RuntimeError(f"Unsupported MoT variant: {variant}")
    source_receipt = inspect_mechanics_source(source, variant=variant)
    snapshots = {
        "gpt2": source_model.expanduser().resolve(),
        "gpt2-medium": target_model.expanduser().resolve(),
    }
    snapshot_receipts = {
        name: _snapshot_content_receipt(path, model_name=name)
        for name, path in snapshots.items()
    }
    modules = _load_author_modules(source)
    torch = importlib.import_module("torch")
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))

    source_receipt["imported_modules"] = {
        name: {
            "path": str(Path(module.__file__).resolve()),
            "sha256": _sha256(Path(module.__file__).resolve()),
        }
        for name, module in modules.items()
    }
    sparse = {
        name: _sparse_model_trial(
            torch=torch,
            common=modules["common"],
            model_module=modules["model"],
            author_train=modules["train"],
            model_name=name,
            snapshot=path,
        )
        for name, path in snapshots.items()
    }
    reduced_sparse = None
    if variant == "modified-mot":
        reduced_sparse = {
            name: _reduced_sparse_model_trial(
                torch=torch,
                common=modules["common"],
                model_module=modules["model"],
                author_train=modules["train"],
                model_name=name,
                snapshot=path,
            )
            for name, path in snapshots.items()
        }
    alignment = _alignment_trial(
        torch=torch,
        common=modules["common"],
        model_module=modules["model"],
        author_train=modules["train"],
        source_snapshot=snapshots["gpt2"],
        target_snapshot=snapshots["gpt2-medium"],
    )
    sparse_pass = all(item["pass"] for item in sparse.values())
    reduced_sparse_pass = (
        True
        if reduced_sparse is None
        else all(item["pass"] for item in reduced_sparse.values())
    )
    mechanics_pass = sparse_pass and alignment["pass"] and reduced_sparse_pass
    return {
        "schema": "knlp_mot_sparse_alignment_preflight_v1",
        "variant": variant,
        "scope": "cpu_real_models_no_training",
        "source": source_receipt,
        "snapshots": snapshot_receipts,
        "thresholds": PREFLIGHT_THRESHOLDS,
        "sparse_full_key_replay": sparse,
        "sparse_reduced_key_replay": reduced_sparse,
        "same_grid_alignment": alignment,
        "decision": {
            "sparse_full_key_replay_pass": sparse_pass,
            "same_grid_alignment_and_prefix_integrity_pass": alignment["pass"],
            "sparse_reduced_key_replay_pass": reduced_sparse_pass,
            "mechanics_preflight_pass": mechanics_pass,
            "two_step_training_smoke_admitted": (
                variant == "modified-mot" and mechanics_pass
            ),
            "five_hundred_step_baseline_admitted": False,
            "cross_family_execution_admitted": False,
            "paper_reproduction_claim_admitted": False,
        },
        "limitations": [
            "This preflight does not train or evaluate a translator.",
            "The full-key control matches the frozen 64-token prefix where Top-S=128 selects all keys.",
            "The unequal-grid all-masked witness remains blocked and uncorrected.",
            "Modified MoT is not an exact reproduction of the author implementation.",
        ],
        "runtime": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "transformers": str(importlib.import_module("transformers").__version__),
        },
    }

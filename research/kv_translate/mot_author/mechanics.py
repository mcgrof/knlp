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


def _tensor_sha256(tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _tensor_diagnostics(torch, left, right, *, atol: float) -> dict[str, Any]:
    left64 = left.detach().to(dtype=torch.float64)
    right64 = right.detach().to(dtype=torch.float64)
    absolute = (left64 - right64).abs()
    relative = absolute / right64.abs().clamp_min(1.0e-30)
    return {
        "element_count": int(absolute.numel()),
        "exact_mismatch_count": int((absolute != 0).sum().item()),
        "above_absolute_limit_count": int((absolute > atol).sum().item()),
        "max_abs": float(absolute.max().item()),
        "rms": float(torch.sqrt(torch.mean(absolute.square())).item()),
        "max_elementwise_relative": float(relative.max().item()),
    }


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

    indices = author_train.validate_selected_attention_indices(
        sparse_indices,
        batch_size=batch_size,
        num_heads=num_heads,
        query_len=seq_len,
        key_len=seq_len,
    ).to(device=hidden_states.device)
    attention_key = native_key if injected_key is None else injected_key
    attention_value = native_value if injected_value is None else injected_value
    selected = torch.zeros(
        batch_size,
        num_heads,
        seq_len,
        seq_len,
        dtype=torch.bool,
        device=hidden_states.device,
    )
    selected.scatter_(dim=-1, index=indices, value=True)
    query_positions = torch.arange(seq_len, device=hidden_states.device).view(1, 1, seq_len, 1)
    key_positions = torch.arange(seq_len, device=hidden_states.device).view(1, 1, 1, seq_len)
    valid = selected & (key_positions <= query_positions)
    if not bool(valid.any(dim=-1).all().item()):
        raise RuntimeError("Reduced-key dense reference contains an all-masked query row")

    scores = torch.matmul(query, attention_key.transpose(-1, -2))
    if bool(getattr(attn, "scale_attn_weights", True)):
        scores = scores / (head_dim ** 0.5)
    if bool(getattr(attn, "scale_attn_by_inverse_layer_idx", False)):
        layer_idx = getattr(attn, "layer_idx", None)
        if layer_idx is None:
            raise RuntimeError("GPT-2 inverse-layer scaling requires layer_idx")
        scores = scores / float(int(layer_idx) + 1)
    scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    output = torch.matmul(weights, attention_value)
    output = output.permute(0, 2, 1, 3).contiguous().view(
        batch_size, seq_len, num_heads * head_dim
    )
    output = attn.c_proj(output)
    output = attn.resid_dropout(output)
    hidden_states = residual + output
    hidden_states = hidden_states + block.mlp(block.ln_2(hidden_states))
    return hidden_states, (native_key, native_value), valid


def _fp64_selected_key_oracle(
    *, torch, query, key, value, sparse_indices
) -> dict[str, Any]:
    """Compare independent dense and explicit selected-key attention in FP64."""
    atol = 1.0e-10
    rtol = 1.0e-8
    query64 = query.detach().to(dtype=torch.float64)
    key64 = key.detach().to(dtype=torch.float64)
    value64 = value.detach().to(dtype=torch.float64)
    batch_size, num_heads, query_len, head_dim = query64.shape
    key_len = key64.shape[-2]
    indices = sparse_indices.detach().to(dtype=torch.long).expand(
        batch_size, num_heads, query_len, -1
    )
    query_positions = torch.arange(query_len, dtype=torch.long).view(1, 1, query_len, 1)
    key_positions = torch.arange(key_len, dtype=torch.long).view(1, 1, 1, key_len)

    def dense_selected(q, k, v, selected_indices, *, apply_scale: bool = True):
        selected = torch.zeros(
            batch_size, num_heads, query_len, key_len, dtype=torch.bool
        )
        selected.scatter_(dim=-1, index=selected_indices, value=True)
        valid = selected & (key_positions <= query_positions)
        scores = torch.matmul(q, k.transpose(-1, -2))
        if apply_scale:
            scores = scores / (head_dim ** 0.5)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        return torch.matmul(torch.softmax(scores, dim=-1), v)

    def explicit_selected(q, k, v, selected_indices, *, apply_scale: bool = True):
        top_s = selected_indices.shape[-1]
        expanded_key = k.unsqueeze(2).expand(
            batch_size, num_heads, query_len, key_len, head_dim
        )
        expanded_value = v.unsqueeze(2).expand_as(expanded_key)
        gather_index = selected_indices.unsqueeze(-1).expand(
            batch_size, num_heads, query_len, top_s, head_dim
        )
        selected_key = torch.gather(expanded_key, dim=3, index=gather_index)
        selected_value = torch.gather(expanded_value, dim=3, index=gather_index)
        scores = (q.unsqueeze(-2) * selected_key).sum(dim=-1)
        if apply_scale:
            scores = scores / (head_dim ** 0.5)
        valid = selected_indices <= query_positions
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return (weights.unsqueeze(-1) * selected_value).sum(dim=-2)

    dense64 = dense_selected(query64, key64, value64, indices)
    explicit64 = explicit_selected(query64, key64, value64, indices)
    permutation = torch.arange(indices.shape[-1] - 1, -1, -1, dtype=torch.long)
    permuted64 = explicit_selected(
        query64, key64, value64, indices.index_select(-1, permutation)
    )
    gathered32 = explicit_selected(
        query64.float(), key64.float(), value64.float(), indices
    ).to(dtype=torch.float64)
    wrong_scale64 = explicit_selected(
        query64, key64, value64, indices, apply_scale=False
    )
    wrong_selected = torch.zeros(
        batch_size, num_heads, query_len, key_len, dtype=torch.bool
    )
    wrong_selected.scatter_(dim=-1, index=indices, value=True)
    wrong_mask_scores = torch.matmul(query64, key64.transpose(-1, -2)) / (head_dim ** 0.5)
    wrong_mask_scores = wrong_mask_scores.masked_fill(
        ~wrong_selected, torch.finfo(torch.float64).min
    )
    wrong_mask64 = torch.matmul(torch.softmax(wrong_mask_scores, dim=-1), value64)

    dense_vs_explicit = _tensor_diagnostics(
        torch, dense64, explicit64, atol=atol
    )
    dense_vs_permuted = _tensor_diagnostics(
        torch, dense64, permuted64, atol=atol
    )
    wrong_scale = _tensor_diagnostics(torch, dense64, wrong_scale64, atol=atol)
    wrong_mask = _tensor_diagnostics(torch, dense64, wrong_mask64, atol=atol)
    gathered = _tensor_diagnostics(torch, dense64, gathered32, atol=atol)
    dense_explicit_close = bool(torch.allclose(dense64, explicit64, atol=atol, rtol=rtol))
    permutation_close = bool(torch.allclose(dense64, permuted64, atol=atol, rtol=rtol))
    wrong_scale_detected = not bool(
        torch.allclose(dense64, wrong_scale64, atol=atol, rtol=rtol)
    )
    wrong_mask_detected = not bool(
        torch.allclose(dense64, wrong_mask64, atol=atol, rtol=rtol)
    )
    return {
        "input_shapes": {
            "query": list(query64.shape),
            "key": list(key64.shape),
            "value": list(value64.shape),
            "indices": list(indices.shape),
        },
        "cast_to_float64_before_qk_softmax_and_value_reduction": True,
        "oracle_softmax_dtype": str(dense64.dtype),
        "atol": atol,
        "rtol": rtol,
        "dense_vs_explicit": dense_vs_explicit,
        "dense_vs_selected_key_permutation": dense_vs_permuted,
        "gathered_fp32_vs_fp64_oracle_diagnostic_only": gathered,
        "wrong_scale_negative_control": {
            **wrong_scale,
            "detected": wrong_scale_detected,
        },
        "wrong_mask_negative_control": {
            **wrong_mask,
            "detected": wrong_mask_detected,
        },
        "pass": (
            dense_explicit_close
            and permutation_close
            and wrong_scale_detected
            and wrong_mask_detected
        ),
    }


def _dense_selected_model_trial(
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
        expanded = author_train.validate_selected_attention_indices(
            indices,
            batch_size=1,
            num_heads=int(transformer.h[layer_idx].attn.num_heads),
            query_len=token_count,
            key_len=token_count,
        )
        query_positions = torch.arange(token_count).view(1, 1, token_count, 1)
        valid_counts = (expanded <= query_positions).sum(dim=-1)
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
        cache_max_abs = 0.0
        reduced_branch_calls = 0
        injected_branch_calls = 0
        per_layer = []
        oracle = None
        for layer_idx, block in enumerate(transformer.h):
            if layer_idx < bottom_full_layers:
                actual_hidden, actual_present = author_train.run_gpt2_block(block, actual_hidden)
                reference_hidden, reference_present = author_train.run_gpt2_block(block, reference_hidden)
            else:
                native_key, native_value = native.past_key_values[layer_idx]
                shared_input = reference_hidden.detach().clone()
                shared_actual, shared_actual_present = author_train.run_gpt2_block(
                    block,
                    shared_input,
                    sparse_attention_indices=sparse_by_layer[layer_idx],
                    injected_key=native_key,
                    injected_value=native_value,
                    attention_backend="dense_selected",
                )
                shared_reference, shared_reference_present, _ = _gpt2_reduced_reference_block(
                    torch=torch,
                    author_train=author_train,
                    block=block,
                    hidden_states=shared_input,
                    sparse_indices=sparse_by_layer[layer_idx],
                    injected_key=native_key,
                    injected_value=native_value,
                )
                actual_hidden, actual_present = author_train.run_gpt2_block(
                    block,
                    actual_hidden,
                    sparse_attention_indices=sparse_by_layer[layer_idx],
                    injected_key=native_key,
                    injected_value=native_value,
                    attention_backend="dense_selected",
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
                qkv = block.attn.c_attn(block.ln_1(shared_input))
                shared_output_diag = _tensor_diagnostics(
                    torch,
                    shared_actual,
                    shared_reference,
                    atol=PREFLIGHT_THRESHOLDS["author_replay_logits_max_abs"],
                )
                shared_cache_diag = [
                    _tensor_diagnostics(
                        torch,
                        actual_tensor,
                        reference_tensor,
                        atol=PREFLIGHT_THRESHOLDS["author_replay_cache_max_abs"],
                    )
                    for actual_tensor, reference_tensor in zip(
                        shared_actual_present, shared_reference_present
                    )
                ]
                per_layer.append(
                    {
                        "layer": layer_idx,
                        "identical_inputs": {
                            "hidden_states_sha256": _tensor_sha256(shared_input),
                            "qkv_sha256": _tensor_sha256(qkv),
                            "selected_indices_sha256": _tensor_sha256(
                                sparse_by_layer[layer_idx]
                            ),
                            "injected_key_sha256": _tensor_sha256(native_key),
                            "injected_value_sha256": _tensor_sha256(native_value),
                        },
                        "shared_input_output": shared_output_diag,
                        "shared_input_returned_cache": shared_cache_diag,
                    }
                )
                if oracle is None:
                    split_size = getattr(block.attn, "split_size", qkv.shape[-1] // 3)
                    query, _, _ = qkv.split(split_size, dim=2)
                    num_heads = int(block.attn.num_heads)
                    head_dim = int(block.attn.head_dim)
                    query = query.view(1, token_count, num_heads, head_dim).permute(0, 2, 1, 3)
                    oracle = _fp64_selected_key_oracle(
                        torch=torch,
                        query=query[:, :2, :16, :],
                        key=native_key[:, :2, :, :],
                        value=native_value[:, :2, :, :],
                        sparse_indices=sparse_by_layer[layer_idx][:, :, :16, :],
                    )
                reduced_branch_calls += 1
                injected_branch_calls += 1
            block_abs = _max_abs(actual_hidden, reference_hidden)
            block_max_abs = max(block_max_abs, block_abs)
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
        logits_diagnostics = _tensor_diagnostics(
            torch,
            actual_logits,
            reference_logits,
            atol=PREFLIGHT_THRESHOLDS["author_replay_logits_max_abs"],
        )
        logits_cosine = float(
            torch.nn.functional.cosine_similarity(
                actual_logits.flatten().double(),
                reference_logits.flatten().double(),
                dim=0,
            ).item()
        )

    shared_layer_max_abs = max(
        row["shared_input_output"]["max_abs"] for row in per_layer
    )
    shared_layer_max_rms = max(
        row["shared_input_output"]["rms"] for row in per_layer
    )
    pass_gate = (
        block_max_abs <= PREFLIGHT_THRESHOLDS["author_replay_logits_max_abs"]
        and cache_max_abs <= PREFLIGHT_THRESHOLDS["author_replay_cache_max_abs"]
        and logits_max_abs <= PREFLIGHT_THRESHOLDS["author_replay_logits_max_abs"]
        and logits_cosine >= PREFLIGHT_THRESHOLDS["author_replay_logits_min_cosine"]
        and oracle is not None
        and oracle["pass"]
    )
    result = {
        "model_name": model_name,
        "token_count": token_count,
        "top_s": top_k,
        "attention_backend": "dense_selected",
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
        "per_layer_identical_input_diagnostics": per_layer,
        "shared_input_layer_max_abs": shared_layer_max_abs,
        "shared_input_layer_max_rms": shared_layer_max_rms,
        "block_output_max_abs": block_max_abs,
        "returned_cache_max_abs": cache_max_abs,
        "full_stack_logits_max_abs": logits_max_abs,
        "full_stack_logits_diagnostics": logits_diagnostics,
        "full_stack_logits_cosine": logits_cosine,
        "full_stack_logits_cosine_in_valid_range": -1.0 <= logits_cosine <= 1.0,
        "fp64_selected_key_oracle": oracle,
        "pass": pass_gate,
    }
    del native, wrapped, raw_model, actual_hidden, reference_hidden, actual_logits, reference_logits
    gc.collect()
    return result


def _malformed_selection_trial(*, torch, author_train) -> dict[str, Any]:
    cases = {
        "duplicate": torch.tensor([[[[0, 0]]]], dtype=torch.long),
        "out_of_range": torch.tensor([[[[0, 4]]]], dtype=torch.long),
        "empty_causal_row": torch.tensor([[[[1, 2]]]], dtype=torch.long),
    }
    results = {}
    for name, indices in cases.items():
        try:
            author_train.validate_selected_attention_indices(
                indices,
                batch_size=1,
                num_heads=1,
                query_len=1,
                key_len=4,
            )
        except ValueError as error:
            results[name] = {"rejected": True, "error": str(error)}
        else:
            results[name] = {"rejected": False, "error": None}
    return {
        "cases": results,
        "pass": all(row["rejected"] for row in results.values()),
    }


def _mixed_window_trial(
    *, torch, common, model_module, model_spec_module, author_train, source_snapshot, target_snapshot
) -> dict[str, Any]:
    token_count = 160
    top_k = 128
    bottom_full_layers = 2
    source_layers = [4, 5, 6, 7]
    target_layers = [8, 10, 13, 15]
    source_tokenizer = common.load_tokenizer(str(source_snapshot))
    target_tokenizer = common.load_tokenizer(str(target_snapshot))
    source_raw = common.load_frozen_model(str(source_snapshot), "cpu", "float32")
    target_raw = common.load_frozen_model(str(target_snapshot), "cpu", "float32")
    source_model = model_module.Model(str(source_snapshot), source_raw, source_tokenizer)
    target_model = model_module.Model(str(target_snapshot), target_raw, target_tokenizer)
    source_transformer = author_train.require_gpt2_transformer(source_model)
    target_transformer = author_train.require_gpt2_transformer(target_model)
    source_input = _fixed_prompt(source_tokenizer, token_count)
    target_input = _fixed_prompt(target_tokenizer, token_count)
    if not bool(torch.equal(source_input, target_input)):
        raise RuntimeError("Mixed-window same-grid fixture produced different token IDs")
    source_ids = common.TokenIDs(source_input, model_id=source_model.id)
    target_ids = common.TokenIDs(target_input, model_id=target_model.id)
    sparse_by_layer = author_train.build_extrapolated_sparse_attention_indices(
        source_model,
        source_ids,
        source_layer_indices=source_layers,
        target_layer_indices=target_layers,
        num_source_layers=len(source_transformer.h),
        num_target_layers=len(target_transformer.h),
        source_model_id="gpt2",
        top_k=top_k,
    )
    with torch.inference_mode():
        native = target_raw(input_ids=target_input, use_cache=True, return_dict=True)
        actual_hidden = author_train.build_gpt2_input_hidden_states(target_model, target_ids)
        reference_hidden = actual_hidden.clone()
        actual_cache = []
        reference_cache = []
        branch_counts = {
            "bottom_full_attention_layers": 0,
            "dense_selected_injected_layers": 0,
            "dense_selected_recomputed_layers": 0,
        }
        for layer_idx, block in enumerate(target_transformer.h):
            if layer_idx < bottom_full_layers:
                actual_hidden, actual_present = author_train.run_gpt2_block(block, actual_hidden)
                reference_hidden, reference_present = author_train.run_gpt2_block(block, reference_hidden)
                branch_counts["bottom_full_attention_layers"] += 1
            else:
                injected_key = None
                injected_value = None
                if layer_idx in target_layers:
                    injected_key, injected_value = native.past_key_values[layer_idx]
                    branch_counts["dense_selected_injected_layers"] += 1
                else:
                    branch_counts["dense_selected_recomputed_layers"] += 1
                actual_hidden, actual_present = author_train.run_gpt2_block(
                    block,
                    actual_hidden,
                    sparse_attention_indices=sparse_by_layer[layer_idx],
                    injected_key=injected_key,
                    injected_value=injected_value,
                    attention_backend="dense_selected",
                )
                reference_hidden, reference_present, _ = _gpt2_reduced_reference_block(
                    torch=torch,
                    author_train=author_train,
                    block=block,
                    hidden_states=reference_hidden,
                    sparse_indices=sparse_by_layer[layer_idx],
                    injected_key=injected_key,
                    injected_value=injected_value,
                )
            actual_cache.append(actual_present)
            reference_cache.append(reference_present)

        actual_logits = target_raw.lm_head(target_transformer.ln_f(actual_hidden))
        reference_logits = target_raw.lm_head(target_transformer.ln_f(reference_hidden))
        cache_diagnostics = [
            _tensor_diagnostics(
                torch,
                actual_tensor,
                reference_tensor,
                atol=PREFLIGHT_THRESHOLDS["author_replay_cache_max_abs"],
            )
            for actual_present, reference_present in zip(actual_cache, reference_cache)
            for actual_tensor, reference_tensor in zip(actual_present, reference_present)
        ]
        logits_diagnostics = _tensor_diagnostics(
            torch,
            actual_logits,
            reference_logits,
            atol=PREFLIGHT_THRESHOLDS["author_replay_logits_max_abs"],
        )

        injected_past = tuple(native.past_key_values[index] for index in target_layers)
        injected_key_block, injected_value_block = common.past_key_values_to_blocks(
            injected_past
        )
        target_spec = model_spec_module.infer_model_spec_from_config(
            target_raw.config, default_model_id="gpt2-medium"
        )
        integrated_cache = author_train.replay_target_prefill_with_injected_window(
            target_model=target_model,
            context_token_ids=target_ids,
            target_layer_indices=target_layers,
            injected_key_block=injected_key_block,
            injected_value_block=injected_value_block,
            tgt_spec=target_spec,
            target_model_id="gpt2-medium",
            sparse_attention_indices=sparse_by_layer,
            num_bottom_full_attn=bottom_full_layers,
            attention_backend="dense_selected",
        )
        integration_cache_max_abs = max(
            _max_abs(integrated_tensor, manual_tensor)
            for integrated_present, manual_present in zip(integrated_cache, actual_cache)
            for integrated_tensor, manual_tensor in zip(integrated_present, manual_present)
        )

    cache_max_abs = max(row["max_abs"] for row in cache_diagnostics)
    passed = (
        logits_diagnostics["max_abs"]
        <= PREFLIGHT_THRESHOLDS["author_replay_logits_max_abs"]
        and cache_max_abs <= PREFLIGHT_THRESHOLDS["author_replay_cache_max_abs"]
        and integration_cache_max_abs
        <= PREFLIGHT_THRESHOLDS["author_replay_cache_max_abs"]
        and branch_counts
        == {
            "bottom_full_attention_layers": 2,
            "dense_selected_injected_layers": 4,
            "dense_selected_recomputed_layers": 18,
        }
    )
    result = {
        "token_count": token_count,
        "top_s": top_k,
        "source_target_layer_pairs": list(zip(source_layers, target_layers)),
        "injected_kv_fixture": "captured target KV for fixed mechanics window",
        "translated_checkpoint_used": False,
        "branch_counts": branch_counts,
        "logits": logits_diagnostics,
        "returned_cache_max_abs": cache_max_abs,
        "integration_vs_manual_cache_max_abs": integration_cache_max_abs,
        "pass": passed,
    }
    del source_raw, target_raw, source_model, target_model, native
    del actual_hidden, reference_hidden, actual_logits, reference_logits
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
    if variant == "unchanged-author":
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
        alignment = _alignment_trial(
            torch=torch,
            common=modules["common"],
            model_module=modules["model"],
            author_train=modules["train"],
            source_snapshot=snapshots["gpt2"],
            target_snapshot=snapshots["gpt2-medium"],
        )
        sparse_pass = all(item["pass"] for item in sparse.values())
        mechanics_pass = sparse_pass and alignment["pass"]
        return {
            "schema": "knlp_mot_sparse_alignment_preflight_v1",
            "variant": variant,
            "scope": "cpu_real_models_no_training",
            "source": source_receipt,
            "snapshots": snapshot_receipts,
            "thresholds": PREFLIGHT_THRESHOLDS,
            "sparse_full_key_replay": sparse,
            "same_grid_alignment": alignment,
            "decision": {
                "sparse_full_key_replay_pass": sparse_pass,
                "same_grid_alignment_and_prefix_integrity_pass": alignment["pass"],
                "mechanics_preflight_pass": mechanics_pass,
                "two_step_training_smoke_admitted": False,
                "five_hundred_step_baseline_admitted": False,
                "cross_family_execution_admitted": False,
                "paper_reproduction_claim_admitted": False,
            },
            "limitations": [
                "This preflight does not train or evaluate a translator.",
                "The unchanged-author control has no reduced selected-key backend.",
                "Cross-family execution is not admitted.",
            ],
            "runtime": {
                "python": sys.version,
                "torch": str(torch.__version__),
                "transformers": str(importlib.import_module("transformers").__version__),
            },
        }

    first_failed_stage = None
    malformed = _malformed_selection_trial(
        torch=torch, author_train=modules["train"]
    )
    if not malformed["pass"]:
        first_failed_stage = "malformed_selection_rejection"

    dense_selected: dict[str, Any] = {}
    if first_failed_stage is None:
        for name, path in snapshots.items():
            dense_selected[name] = _dense_selected_model_trial(
                torch=torch,
                common=modules["common"],
                model_module=modules["model"],
                author_train=modules["train"],
                model_name=name,
                snapshot=path,
            )
            if not dense_selected[name]["pass"]:
                first_failed_stage = f"dense_selected_key_replay:{name}"
                break

    mixed_window = None
    if first_failed_stage is None:
        mixed_window = _mixed_window_trial(
            torch=torch,
            common=modules["common"],
            model_module=modules["model"],
            model_spec_module=modules["model_spec"],
            author_train=modules["train"],
            source_snapshot=snapshots["gpt2"],
            target_snapshot=snapshots["gpt2-medium"],
        )
        if not mixed_window["pass"]:
            first_failed_stage = "mixed_window_replay"

    mechanics_pass = first_failed_stage is None
    return {
        "schema": "knlp_mot_dense_selected_preflight_v2",
        "variant": variant,
        "scope": "cpu_real_models_no_training",
        "source": source_receipt,
        "snapshots": snapshot_receipts,
        "thresholds": {
            **PREFLIGHT_THRESHOLDS,
            "fp64_oracle_atol": 1.0e-10,
            "fp64_oracle_rtol": 1.0e-8,
        },
        "reused_unchanged_controls": {
            "full_key_replay": "prior frozen pass; branch unchanged and not rerun",
            "same_grid_identity_and_prefix_integrity": (
                "prior frozen pass; contract unchanged and not rerun"
            ),
            "frozen_data_content": "prior frozen pass; data not resampled",
        },
        "historical_gathered_backend": {
            "status": "failed frozen reduced-key gate",
            "rerun": False,
            "admitted": False,
        },
        "attention_backend": "dense_selected",
        "malformed_selection_rejection": malformed,
        "dense_selected_key_replay": dense_selected,
        "mixed_window_replay": mixed_window,
        "first_failed_stage": first_failed_stage,
        "decision": {
            "dense_selected_backend_pass": mechanics_pass,
            "mechanics_preflight_pass": mechanics_pass,
            "quality_diagnostic_admitted": mechanics_pass,
            "five_hundred_step_baseline_admitted": mechanics_pass,
            "cross_family_execution_admitted": False,
            "paper_reproduction_claim_admitted": False,
            "sparse_compute_or_speed_claim_admitted": False,
        },
        "limitations": [
            "This preflight does not train or evaluate a translator.",
            "The gathered reduced-key backend remains failed and was not rerun.",
            "Dense selected-key attention has quadratic working storage.",
            "The mixed-window fixture uses captured target KV, not a trained translator.",
            "Same-grid GPT-2 to GPT-2-medium is the only admitted lane.",
            "Modified MoT is not an exact reproduction of the author implementation.",
        ],
        "runtime": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "transformers": str(importlib.import_module("transformers").__version__),
        },
    }

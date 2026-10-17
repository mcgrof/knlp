"""Bounded two-step GPU training smoke for the pinned modified-MoT variant."""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

from .data import SMOKE_TOTAL_TOKENS, load_smoke_manifest
from .mechanics import inspect_mechanics_source
from .recipe import MODEL_REVISIONS

SMOKE_STEPS = 2
MICROBATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
TOTAL_TOKENS = SMOKE_TOTAL_TOKENS
PREFIX_TOKENS = 64
CHANNELS = (
    {"src_layer_idx": 8, "dst_layer_idx": 20},
    {"src_layer_idx": 9, "dst_layer_idx": 21},
    {"src_layer_idx": 10, "dst_layer_idx": 22},
    {"src_layer_idx": 11, "dst_layer_idx": 23},
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_modules(source: Path) -> dict[str, Any]:
    source_text = str(source.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    return {
        "common": importlib.import_module("core.common"),
        "context": importlib.import_module("core.context"),
        "topology": importlib.import_module("core.topology"),
        "train": importlib.import_module("alg.mot.train"),
    }


def _parameter_fingerprints(module, torch) -> dict[str, list[float]]:
    result = {}
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            values = parameter.detach().float()
            result[name] = [
                float(values.sum().item()),
                float(values.abs().sum().item()),
                float(values.square().sum().item()),
                float(values.abs().max().item()),
            ]
    return result


def _fingerprint_changed(before, after, prefix: str | None = None) -> bool:
    names = before if prefix is None else [name for name in before if prefix in name]
    return any(before[name] != after[name] for name in names)


def _language_model_sample_hash(models, torch) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for node_id, model in sorted(models.items()):
            digest.update(node_id.encode("utf-8"))
            for name, parameter in model.named_parameters():
                flat = parameter.detach().reshape(-1)
                if flat.numel() == 0:
                    continue
                indices = torch.tensor(
                    [0, flat.numel() // 2, flat.numel() - 1],
                    device=flat.device,
                    dtype=torch.long,
                )
                sample = flat.index_select(0, indices).float().cpu().numpy().tobytes()
                digest.update(name.encode("utf-8"))
                digest.update(sample)
    return digest.hexdigest()


def _language_model_versions(models) -> dict[str, list[int]]:
    return {
        node_id: [parameter._version for parameter in model.parameters()]
        for node_id, model in models.items()
    }


def _gradient_metrics(translator_pool, author_train) -> dict[str, Any]:
    total_l1 = 0.0
    all_finite = True
    router_l1 = 0.0
    expert_l1: dict[str, float] = {}
    for translator_id, translator in translator_pool.translators.items():
        for name, parameter in translator.named_parameters():
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            value = float(gradient.float().abs().sum().item())
            total_l1 += value
            all_finite = all_finite and bool(
                author_train.torch.isfinite(gradient).all().item()
            )
            if ".router." in f".{name}.":
                router_l1 += value
            for expert_index in range(2):
                marker = f".translators.{expert_index}."
                if marker in f".{name}.":
                    key = f"{translator_id}:expert-{expert_index}"
                    expert_l1[key] = expert_l1.get(key, 0.0) + value

    selections = []
    for translator_id, translator in translator_pool.translators.items():
        for module_name, module in translator.named_modules():
            if not isinstance(module, author_train.MixtureOfTranslators):
                continue
            weights = module.last_mixture_weights
            counts = (
                []
                if weights is None
                else (weights > 0).sum(dim=(0, 1)).detach().cpu().tolist()
            )
            selections.append(
                {
                    "translator_id": translator_id,
                    "module": module_name,
                    "selected_token_counts": counts,
                }
            )
    return {
        "total_gradient_l1": total_l1,
        "all_gradients_finite": all_finite,
        "router_gradient_l1": router_l1,
        "expert_gradient_l1": expert_l1,
        "selection_counts": selections,
    }


def _loss_components(ctx, author_train, common, batch_rows, torch):
    target_model = ctx.tp.get_model("B")
    target_ids = common.TokenIDs(
        batch_rows.to(ctx.config.device), model_id=target_model.id
    )
    context_ids, prompt_ids, label_ids = common.split_context_and_prompt_token_ids(
        target_ids, context_tokens=PREFIX_TOKENS
    )
    with torch.no_grad():
        native_target_past = common.extract_past_key_values(target_model, context_ids)
    mixed_target_past, _ = author_train.build_replayed_target_past_from_target_context(
        ctx,
        target_context_token_ids=context_ids,
        source_model=ctx.tp.get_model("A"),
        target_model=target_model,
        src_node_id="A",
        tgt_node_id="B",
        tgt_spec=ctx.tp.get_model_spec("B"),
    )
    combined = author_train.compute_prefix_correction_and_suffix_lm_loss(
        target_model=target_model,
        past_key_values=mixed_target_past,
        prompt_token_ids=prompt_ids,
        label_token_ids=label_ids,
        native_target_past_key_values=native_target_past,
        target_layer_indices=ctx.cm.get_tgt_layer_indices("A_to_B"),
    )
    start_layer = ctx.cm.get_tgt_layer_indices("A_to_B")[0]
    mixed_key, mixed_value = common.past_key_values_to_blocks(
        mixed_target_past[start_layer:]
    )
    native_key, native_value = common.past_key_values_to_blocks(
        native_target_past[start_layer:]
    )
    correction = torch.nn.functional.mse_loss(
        mixed_key, native_key, reduction="mean"
    ) + torch.nn.functional.mse_loss(mixed_value, native_value, reduction="mean")
    suffix = combined - correction
    return combined, correction.detach(), suffix.detach()


def _checkpoint_and_reload(
    *, output: Path, ctx, author_train, torch, reference_batch, common
) -> dict[str, Any]:
    checkpoint_root = output / "checkpoints"
    checkpoint_dir = author_train.save_translator_checkpoints(checkpoint_root, ctx.tp)
    with torch.no_grad():
        reference_loss, _, _ = _loss_components(
            ctx, author_train, common, reference_batch, torch
        )
        reference_value = float(reference_loss.float().item())

    config = ctx.config
    source_spec = ctx.tp.get_model_spec("A")
    target_spec = ctx.tp.get_model_spec("B")
    translator_id = next(iter(ctx.tp.translators))
    reloaded = author_train.LayerWindowDirectionalTranslator(
        src_hidden_size=source_spec.kv_hidden_size,
        tgt_hidden_size=target_spec.kv_hidden_size,
        num_layers=len(CHANNELS),
        translator_dim=config.translator_dim,
        translator_heads=config.translator_heads,
        translator_depth=config.translator_depth,
        mlp_ratio=config.translator_mlp_ratio,
        variant=config.variant,
        mot_num_translators=config.mot_num_translators,
        mot_top_k=config.mot_top_k,
    ).to(device=config.device, dtype=author_train.get_training_dtype(config))
    checkpoint_path = author_train.get_translator_checkpoint_path(
        checkpoint_root, translator_id
    )
    reloaded.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
    reloaded.eval()
    original = ctx.tp.translators[translator_id]
    ctx.tp.translators[translator_id] = reloaded
    with torch.no_grad():
        reloaded_loss, _, _ = _loss_components(
            ctx, author_train, common, reference_batch, torch
        )
        reloaded_value = float(reloaded_loss.float().item())
    ctx.tp.translators[translator_id] = original
    return {
        "checkpoint_directory": str(checkpoint_dir),
        "files": [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(checkpoint_dir.iterdir())
            if path.is_file()
        ],
        "same_batch_loss_before_save": reference_value,
        "same_batch_loss_after_reload": reloaded_value,
        "absolute_difference": abs(reference_value - reloaded_value),
        "pass": math.isclose(
            reference_value, reloaded_value, rel_tol=0.0, abs_tol=1.0e-5
        ),
    }


def run_training_smoke(
    *,
    source: Path,
    source_model: Path,
    target_model: Path,
    data_manifest: Path,
    output: Path,
) -> dict[str, Any]:
    run_id = os.environ.get("KNLP_GPU_RUN_ID")
    run_dir_value = os.environ.get("KNLP_GPU_RUN_DIR")
    if not run_id or not run_dir_value:
        raise RuntimeError(
            "Training smoke requires KNLP_GPU_RUN_ID and KNLP_GPU_RUN_DIR"
        )
    run_dir = Path(run_dir_value).expanduser().resolve()
    output = output.expanduser().resolve()
    if output != run_dir and run_dir not in output.parents:
        raise RuntimeError(f"Output must stay within KNLP_GPU_RUN_DIR: {run_dir}")
    output.mkdir(parents=True, exist_ok=True)

    source = source.expanduser().resolve()
    source_receipt = inspect_mechanics_source(source, variant="modified-mot")
    source_model = source_model.expanduser().resolve()
    target_model = target_model.expanduser().resolve()
    if source_model.name != MODEL_REVISIONS["gpt2"]:
        raise RuntimeError("GPT-2 source snapshot revision mismatch")
    if target_model.name != MODEL_REVISIONS["gpt2-medium"]:
        raise RuntimeError("GPT-2-medium target snapshot revision mismatch")

    modules = _load_modules(source)
    common = modules["common"]
    author_train = modules["train"]
    Context = modules["context"].Context
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("The allocated payload has no available HIP/CUDA device")
    torch.cuda.reset_peak_memory_stats()
    author_train.set_seed(42)

    config = author_train.TrainConfig(
        alg="mot",
        timestamp=None,
        output_path=str(output / "author-output"),
        device="cuda",
        model_ids=f"{source_model},{target_model}",
        model_directions="A_to_B",
        max_steps=SMOKE_STEPS,
        batch_size=MICROBATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        total_tokens=TOTAL_TOKENS,
        prefix_tokens=PREFIX_TOKENS,
        learning_rate=0.0001,
        weight_decay=0.01,
        warmup_steps=50,
        grad_clip_norm=1.0,
        log_every=1,
        seed=42,
        shuffle_buffer=50000,
        layer_alignment="injection",
        min_window_size_ratio=0.3,
        max_window_size_ratio=0.3,
        translator_dim=1024,
        translator_heads=16,
        translator_depth=2,
        translator_mlp_ratio=4,
        dtype="bfloat16",
        variant="mot",
        mot_num_translators=2,
        mot_top_k=1,
        topk_sparse_attn=128,
        num_bottom_full_attn=2,
    )
    ctx = Context(config)
    for channel in CHANNELS:
        ctx.cm.add_channel("A_to_B", **channel)
    translator_pool = author_train.build_translator_pool(ctx)
    translator_pool.train()
    optimizer = torch.optim.AdamW(
        translator_pool.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    manifest_rows, manifest = load_smoke_manifest(
        manifest_path=data_manifest,
        target_model=target_model,
    )
    token_rows = torch.tensor(manifest_rows, dtype=torch.long)
    manifest_path = output / "TRAINING_SMOKE_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    language_models = ctx.tp.models
    lm_hash_before = _language_model_sample_hash(language_models, torch)
    lm_versions_before = _language_model_versions(language_models)
    lm_parameter_ids = {
        id(parameter)
        for model in language_models.values()
        for parameter in model.parameters()
    }
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    optimizer_disjoint = not bool(lm_parameter_ids & optimizer_parameter_ids)
    all_lm_frozen = all(
        not parameter.requires_grad
        for model in language_models.values()
        for parameter in model.parameters()
    )

    step_records = []
    last_batch = None
    for step in range(SMOKE_STEPS):
        optimizer.zero_grad(set_to_none=True)
        translator = next(iter(translator_pool.translators.values()))
        before = _parameter_fingerprints(translator, torch)
        loss_values = []
        correction_values = []
        suffix_values = []
        for accumulation in range(GRAD_ACCUM_STEPS):
            batch_index = step * GRAD_ACCUM_STEPS + accumulation
            start = batch_index * MICROBATCH_SIZE
            batch = token_rows[start : start + MICROBATCH_SIZE]
            last_batch = batch
            combined, correction, suffix = _loss_components(
                ctx, author_train, common, batch, torch
            )
            scaled = combined / GRAD_ACCUM_STEPS
            scaled.backward()
            loss_values.append(float(combined.detach().float().item()))
            correction_values.append(float(correction.float().item()))
            suffix_values.append(float(suffix.float().item()))

        gradients = _gradient_metrics(translator_pool, author_train)
        unclipped_norm = float(
            torch.nn.utils.clip_grad_norm_(
                list(translator_pool.parameters()), config.grad_clip_norm
            )
            .float()
            .item()
        )
        optimizer.step()
        after = _parameter_fingerprints(translator, torch)
        expert_updates = {
            str(index): _fingerprint_changed(
                before, after, prefix=f".translators.{index}."
            )
            for index in range(config.mot_num_translators)
        }
        step_records.append(
            {
                "step": step + 1,
                "microbatches": GRAD_ACCUM_STEPS,
                "losses": loss_values,
                "prefix_correction_losses": correction_values,
                "suffix_lm_losses": suffix_values,
                "all_losses_finite": all(
                    math.isfinite(value)
                    for value in loss_values + correction_values + suffix_values
                ),
                "gradients": gradients,
                "unclipped_gradient_norm": unclipped_norm,
                "translator_parameters_changed": _fingerprint_changed(before, after),
                "expert_parameters_changed": expert_updates,
                "router_parameters_changed": _fingerprint_changed(
                    before, after, prefix=".router."
                ),
            }
        )

    if last_batch is None:
        raise RuntimeError("Training smoke produced no batch")
    checkpoint = _checkpoint_and_reload(
        output=output,
        ctx=ctx,
        author_train=author_train,
        torch=torch,
        reference_batch=last_batch,
        common=common,
    )
    lm_hash_after = _language_model_sample_hash(language_models, torch)
    lm_versions_after = _language_model_versions(language_models)
    lm_gradients_absent = all(
        parameter.grad is None
        for model in language_models.values()
        for parameter in model.parameters()
    )

    selected_expert_update = False
    for step in step_records:
        selected_indices = set()
        for selection in step["gradients"]["selection_counts"]:
            selected_indices.update(
                index
                for index, count in enumerate(selection["selected_token_counts"])
                if count > 0
            )
        selected_expert_update = selected_expert_update or any(
            step["expert_parameters_changed"].get(str(index), False)
            for index in selected_indices
        )

    all_steps_finite = all(
        step["all_losses_finite"]
        and step["gradients"]["all_gradients_finite"]
        and math.isfinite(step["unclipped_gradient_norm"])
        for step in step_records
    )
    nonzero_task_gradient = all(
        step["gradients"]["total_gradient_l1"] > 0.0 for step in step_records
    )
    base_unchanged = (
        lm_hash_before == lm_hash_after
        and lm_versions_before == lm_versions_after
        and lm_gradients_absent
        and all_lm_frozen
        and optimizer_disjoint
    )
    smoke_pass = (
        all_steps_finite
        and nonzero_task_gradient
        and selected_expert_update
        and base_unchanged
        and checkpoint["pass"]
    )

    result = {
        "schema": "knlp_mot_modified_training_smoke_v1",
        "variant": "modified-mot",
        "source": source_receipt,
        "allocation": {
            "provider": os.environ.get("KNLP_GPU_PROVIDER"),
            "run_id": run_id,
            "run_directory": str(run_dir),
        },
        "runtime": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "transformers": str(importlib.import_module("transformers").__version__),
            "device_name": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "recipe": {
            "steps": SMOKE_STEPS,
            "microbatch_size": MICROBATCH_SIZE,
            "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
            "effective_batch_size": MICROBATCH_SIZE * GRAD_ACCUM_STEPS,
            "total_tokens": TOTAL_TOKENS,
            "prefix_tokens_configured": PREFIX_TOKENS,
            "context_cache_tokens_executed": PREFIX_TOKENS - 1,
            "dtype": config.dtype,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "gradient_clip_norm": config.grad_clip_norm,
            "channels": list(CHANNELS),
            "profiled_channels": False,
            "synthetic_fixed_data": False,
            "dataset": manifest["dataset"],
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
        },
        "steps": step_records,
        "base_models": {
            "all_parameters_frozen": all_lm_frozen,
            "optimizer_parameter_sets_disjoint": optimizer_disjoint,
            "all_gradients_absent": lm_gradients_absent,
            "sample_hash_before": lm_hash_before,
            "sample_hash_after": lm_hash_after,
            "version_counters_unchanged": lm_versions_before == lm_versions_after,
            "pass": base_unchanged,
        },
        "checkpoint_reload": checkpoint,
        "decision": {
            "all_losses_and_gradients_finite": all_steps_finite,
            "nonzero_translator_task_gradient_each_step": nonzero_task_gradient,
            "selected_expert_parameter_update_observed": selected_expert_update,
            "frozen_language_models_unchanged": base_unchanged,
            "save_reload_same_batch_pass": checkpoint["pass"],
            "two_step_training_smoke_pass": smoke_pass,
            "five_hundred_step_baseline_admitted": smoke_pass,
            "paper_reproduction_claim_admitted": False,
        },
        "limitations": [
            "This is a two-step synthetic-data execution check, not quality evidence.",
            "The channel window is frozen terminal alignment and was not profiled.",
            "Default Top-1 router task gradient is expected to remain zero.",
            "Cross-family execution remains blocked.",
        ],
    }
    result_path = output / "TRAINING_SMOKE.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    gc.collect()
    torch.cuda.empty_cache()
    return result

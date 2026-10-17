"""Runtime audits for the pinned MoT implementation."""

from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any


def _load_author_train(source: Path):
    source_text = str(source.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    return importlib.import_module("alg.mot.train")


def _router_trial(
    author_train, torch, *, top_k: int, weight_decay: float
) -> dict[str, Any]:
    class CountingTranslator(torch.nn.Module):
        instances: list[Any] = []

        def __init__(self, **kwargs) -> None:
            super().__init__()
            self.calls = 0
            self.num_layers = int(kwargs["num_layers"])
            self.tgt_hidden_size = int(kwargs["tgt_hidden_size"])
            expert_index = len(self.instances) + 1
            self.scale = torch.nn.Parameter(torch.tensor(float(expert_index)))
            self.instances.append(self)

        def forward(self, value):
            self.calls += 1
            reduced = value.mean(dim=-1, keepdim=True)
            return self.scale * reduced.expand(
                -1, -1, self.num_layers, self.tgt_hidden_size
            )

    torch.manual_seed(42)
    CountingTranslator.instances = []
    module = author_train.MixtureOfTranslators(
        src_hidden_size=4,
        tgt_hidden_size=2,
        num_layers=2,
        translator_dim=8,
        translator_heads=1,
        translator_depth=1,
        mlp_ratio=1,
        num_translators=2,
        top_k=top_k,
        translator_cls=CountingTranslator,
    )
    for parameter in module.router.parameters():
        parameter.data.zero_()
    module.router[-1].bias.data.copy_(torch.tensor([1.0, 0.0]))

    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=0.001,
        weight_decay=weight_decay,
    )
    before = [parameter.detach().clone() for parameter in module.router.parameters()]
    value = torch.arange(48, dtype=torch.float32).reshape(2, 3, 2, 4) / 48.0
    loss = module(value).square().mean()
    loss.backward()

    gradient_l1 = 0.0
    gradient_tensors = 0
    for parameter in module.router.parameters():
        if parameter.grad is not None:
            gradient_l1 += float(parameter.grad.detach().abs().sum().item())
            gradient_tensors += 1
    optimizer.step()
    update_l1 = sum(
        float((old - new.detach()).abs().sum().item())
        for old, new in zip(before, module.router.parameters())
    )
    return {
        "top_k": top_k,
        "weight_decay": weight_decay,
        "router_task_gradient_l1": gradient_l1,
        "router_gradient_tensor_count": gradient_tensors,
        "router_parameter_update_l1": update_l1,
        "expert_forward_calls": [item.calls for item in CountingTranslator.instances],
        "mixture_weights": module.last_mixture_weights.detach().tolist(),
    }


def _causal_alignment_trial(author_train, torch) -> dict[str, Any]:
    source_spans = [(0, 4)]
    target_spans = [(0, 1), (1, 4)]
    _, target_to_source, source_to_target = author_train._build_overlap_weights(
        source_spans,
        target_spans,
    )
    sparse_source_indices = torch.tensor([[[[0]]]], dtype=torch.long)
    aligned = author_train.align_sparse_attention_indices_to_target_tokens(
        sparse_source_indices,
        target_to_source=target_to_source.unsqueeze(0),
        source_to_target=source_to_target.unsqueeze(0),
    )
    invalid = author_train.build_sparse_query_mask(
        aligned,
        seq_len=len(target_spans),
        num_heads=1,
    )
    scores = torch.zeros_like(aligned, dtype=torch.float32)
    masked_scores = scores.masked_fill(invalid, torch.finfo(scores.dtype).min)
    probabilities = torch.softmax(masked_scores, dim=-1)
    first_query_invalid_probability = float(
        probabilities[0, 0, 0][invalid[0, 0, 0]].sum().item()
    )
    return {
        "source_spans": source_spans,
        "target_spans": target_spans,
        "target_to_source": target_to_source.tolist(),
        "source_to_target": source_to_target.tolist(),
        "aligned_sparse_indices": aligned.tolist(),
        "invalid_mask": invalid.tolist(),
        "first_query_all_candidates_invalid": bool(invalid[0, 0, 0].all().item()),
        "first_query_invalid_probability": first_query_invalid_probability,
    }


def _static_path_audit(source: Path) -> dict[str, Any]:
    train_path = source / "alg" / "mot" / "train.py"
    train_text = train_path.read_text(encoding="utf-8")
    default_train = json.loads(
        (source / "configs" / "train_mot.json").read_text(encoding="utf-8")
    )
    source_kv_prefill = (
        "source_past = extract_past_key_values(source_model, source_row)" in train_text
    )
    source_guidance_forward = (
        "unique_sparse_indices = extract_source_attention_topk_indices(" in train_text
        and "outputs = source_model(**model_kwargs)" in train_text
    )
    receiver_lower_replay = (
        "for lower_idx in range(target_start_layer_idx):" in train_text
    )
    receiver_translated_replay = (
        "for layer_idx, injected_present in zip(target_layer_indices, injected_window):"
        in train_text
    )
    receiver_upper_replay = (
        "for upper_idx in range(previous_layer_idx + 1, len(target_blocks)):"
        in train_text
    )
    return {
        "method": "exact source-pattern audit at the pinned file hash",
        "source_kv_prefill_present": source_kv_prefill,
        "separate_source_attention_guidance_forward_present": source_guidance_forward,
        "minimum_source_model_forward_roles_per_item": (
            2 if source_kv_prefill and source_guidance_forward else None
        ),
        "receiver_lower_layers_replayed": receiver_lower_replay,
        "receiver_translated_layers_replayed": receiver_translated_replay,
        "receiver_upper_layers_replayed": receiver_upper_replay,
        "complete_receiver_stack_replayed": (
            receiver_lower_replay
            and receiver_translated_replay
            and receiver_upper_replay
        ),
        "kv_only_handoff_supported_by_inspected_path": False,
        "upstream_default_train_config": {
            "model_ids": default_train["model_ids"],
            "dtype": default_train["dtype"],
            "max_steps": default_train["max_steps"],
            "mot_num_translators": default_train["mot_num_translators"],
            "mot_top_k": default_train["mot_top_k"],
        },
        "paper_recipe_difference": (
            "The paper states bfloat16 training; the upstream default JSON "
            "uses float32. The frozen KNLP recipe applies bfloat16 explicitly."
        ),
    }


def run_runtime_audit(source: Path) -> dict[str, Any]:
    author_train = _load_author_train(source)
    torch = importlib.import_module("torch")

    top1 = _router_trial(author_train, torch, top_k=1, weight_decay=0.0)
    top1_decay = _router_trial(author_train, torch, top_k=1, weight_decay=0.01)
    top2 = _router_trial(author_train, torch, top_k=2, weight_decay=0.0)
    causal = _causal_alignment_trial(author_train, torch)
    path_audit = _static_path_audit(source)

    top1_zero = math.isclose(top1["router_task_gradient_l1"], 0.0, abs_tol=0.0)
    top2_nonzero = top2["router_task_gradient_l1"] > 0.0
    all_experts_execute = all(
        calls == 1 for trial in (top1, top2) for calls in trial["expert_forward_calls"]
    )
    causal_leak = causal["first_query_invalid_probability"] > 0.0

    return {
        "schema": "knlp_mot_author_cpu_readiness_v1",
        "observations": {
            "top1": top1,
            "top1_with_weight_decay": top1_decay,
            "top2_diagnostic": top2,
            "cross_token_causal_case": causal,
            "execution_path": path_audit,
        },
        "findings": {
            "top1_router_has_zero_task_gradient": top1_zero,
            "top2_router_has_nonzero_task_gradient": top2_nonzero,
            "all_experts_execute_for_top1_and_top2": all_experts_execute,
            "weight_decay_can_move_top1_router_without_task_signal": (
                top1_decay["router_parameter_update_l1"] > 0.0 and top1_zero
            ),
            "cross_token_all_masked_row_leaks_future_value": causal_leak,
            "candidate_path_is_not_a_kv_only_handoff": (
                not path_audit["kv_only_handoff_supported_by_inspected_path"]
            ),
            "candidate_path_replays_complete_receiver_stack": path_audit[
                "complete_receiver_stack_replayed"
            ],
        },
        "decision": {
            "author_behavior_mechanics_baseline_admitted": (
                top1_zero and top2_nonzero and all_experts_execute
            ),
            "paper_fidelity_claim_admitted": False,
            "cross_family_evaluation_admitted": False,
            "same_tokenizer_gpt2_baseline_scope": (
                "The constructed alignment defect requires unequal token grids. "
                "GPT-2 and GPT-2-medium share a tokenizer, but the unchanged "
                "Top-1 router limitation still applies."
            ),
        },
        "limitations": [
            "This audit uses synthetic tensors and the real author helpers.",
            "It does not load a language model or measure defect incidence on data.",
            "It does not change the author implementation.",
        ],
        "runtime": {
            "python": sys.version,
            "torch": str(torch.__version__),
        },
    }

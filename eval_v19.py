#!/usr/bin/env python
"""
BPA v19 Evaluation: BitterKV Signal Bakeoff.

Systematic evaluation of candidate sensitivity signals for automated
KV cache INT4/INT8 bit allocation. Tests activation-noise, attention
statistics, norm proxies, spectrum, and parameter-curvature signals
against an empirical oracle.

Phases:
  0: Baseline lock + oracle ground truth
  1: Compute all BitterKV signals
  2: Correlation analysis (all signals vs oracle)
  3: Schedule generation + full eval
  5: Bandwidth-bound validation

Usage:
    python eval_v19.py --phase 0
    python eval_v19.py --phase 1
    python eval_v19.py --phase 2
    python eval_v19.py --phase 3
    python eval_v19.py --phase 5
"""

import argparse
import json
import math
import os
import sys
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from scripts.bpa_v11_bench import (
    DTYPE,
    compute_ppl,
    get_cpu_rss_mb,
    get_gpu_mem,
    get_text_batch,
    gpu_sync,
    load_validation_tokens,
    reset_gpu_mem,
)
from backends.base import DenseBackend, V14StepStats
from backends.quant import (
    QuantBackend,
    quantize_int4_block,
    quantize_int8_symmetric,
    dequantize_int4_block,
    dequantize_int8_symmetric,
)
from eval_v15 import (
    V15Result,
    apply_quality_gating,
    build_scoreboard,
    gpu_preflight,
    load_model,
    run_single_eval,
)
from eval_v16 import (
    MixedPrecisionBackend,
    build_schedules,
    run_backend_sweep,
    run_dense_baselines,
    save_results,
)


# ============================================================
# BitterKV Signal Registry
# ============================================================

SIGNALS = OrderedDict()


def register_signal(name, category, cost, needs_backward=False):
    """Decorator to register a signal function."""

    def decorator(fn):
        SIGNALS[name] = {
            "fn": fn,
            "category": category,
            "cost": cost,
            "needs_backward": needs_backward,
        }
        return fn

    return decorator


# ============================================================
# Calibration data helpers
# ============================================================


def get_calibration_batches(token_data, n_seqs, seq_len, device, rng=None):
    """Get calibration data as list of input_ids tensors."""
    if rng is None:
        rng = np.random.RandomState(42)
    batches = []
    for _ in range(n_seqs):
        idx = get_text_batch(token_data, 1, seq_len, rng).to(device)
        batches.append(idx)
    return batches


# ============================================================
# A) Activation-noise / Fisher-on-activations signals
# ============================================================


def _hook_kv_and_measure(
    model, input_ids, target_layer, noise_fn, device, measure="attn_kl"
):
    """Hook into a specific layer, inject noise into K/V, measure impact.

    noise_fn(k, v) -> (k_noisy, v_noisy)
    Returns scalar score (higher = more sensitive).

    Strategy: hook k_proj and v_proj outputs to capture K/V tensors,
    then also capture Q from q_proj. Qwen2 attention returns
    (attn_output, attn_weights_or_None) with no past_kv in the tuple,
    so we must intercept projections directly.
    """
    n_layers = model.config.num_hidden_layers
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    n_heads = model.config.num_attention_heads
    n_kv_heads = getattr(model.config, "num_key_value_heads", n_heads)
    repeat = n_heads // n_kv_heads
    scale = head_dim**-0.5

    # Storage for clean K, V, Q from target layer
    captured = {}

    def make_proj_hook(layer_idx, proj_name):
        def hook(module, input, output):
            if layer_idx == target_layer:
                captured[proj_name] = output.detach().clone()

        return hook

    # First pass: capture clean K, V, Q projections
    hooks = []
    try:
        layer = model.model.layers[target_layer]
        hooks.append(
            layer.self_attn.q_proj.register_forward_hook(
                make_proj_hook(target_layer, "q")
            )
        )
        hooks.append(
            layer.self_attn.k_proj.register_forward_hook(
                make_proj_hook(target_layer, "k")
            )
        )
        hooks.append(
            layer.self_attn.v_proj.register_forward_hook(
                make_proj_hook(target_layer, "v")
            )
        )

        with torch.no_grad():
            outputs_clean = model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    if "k" not in captured or "v" not in captured:
        return 0.0

    # Reshape projections to [B, n_heads/n_kv_heads, T, head_dim]
    B, T, _ = captured["k"].shape
    k_clean = captured["k"].view(B, T, n_kv_heads, head_dim).transpose(1, 2)
    v_clean = captured["v"].view(B, T, n_kv_heads, head_dim).transpose(1, 2)

    # Apply noise
    k_noisy, v_noisy = noise_fn(k_clean, v_clean)

    if measure == "attn_kl":
        # Compute attention with clean vs noisy K/V using captured Q
        q_raw = captured["q"].view(B, T, n_heads, head_dim).transpose(1, 2)  # [B,H,T,D]

        # Expand KV for GQA
        k_clean_exp = k_clean.repeat_interleave(repeat, dim=1)
        k_noisy_exp = k_noisy.repeat_interleave(repeat, dim=1)
        v_clean_exp = v_clean.repeat_interleave(repeat, dim=1)
        v_noisy_exp = v_noisy.repeat_interleave(repeat, dim=1)

        # Sample query positions (last 8) to reduce cost
        q_positions = list(range(max(0, T - 8), T))
        q_sampled = q_raw[:, :, q_positions, :]  # [B, H, 8, D]

        # Attention logits
        logits_clean = (
            torch.matmul(q_sampled.float(), k_clean_exp.float().transpose(-2, -1))
            * scale
        )
        logits_noisy = (
            torch.matmul(q_sampled.float(), k_noisy_exp.float().transpose(-2, -1))
            * scale
        )

        # Softmax attention
        attn_clean = F.softmax(logits_clean, dim=-1)
        attn_noisy = F.softmax(logits_noisy, dim=-1)

        # Output MSE with clean/noisy attention and V
        out_clean = torch.matmul(attn_clean, v_clean_exp[:, :, :T, :].float())
        out_noisy = torch.matmul(attn_noisy, v_noisy_exp[:, :, :T, :].float())

        score = F.mse_loss(out_noisy, out_clean).item()
        return score

    elif measure == "logit_kl":
        # For logit KL: inject noisy K/V via hooks and do a second forward pass
        def make_inject_hook(layer_idx):
            def hook(module, input, output):
                if layer_idx != target_layer:
                    return
                # output is projection output [B, T, kv_dim]
                # Replace k_proj output with noisy version
                return output  # placeholder, see below

            return hook

        # Simpler: hook self_attn to replace its output using
        # a forward pass with modified hidden states is complex.
        # Instead, compute logit KL from the attention output difference.
        # Use the two-pass approach: measure clean logits vs logits
        # when we perturb the KV at the target layer.

        # We already have clean logits. For noisy: hook k_proj and v_proj
        # to return noisy projections.
        noisy_k_flat = k_noisy.transpose(1, 2).reshape(B, T, -1)
        noisy_v_flat = v_noisy.transpose(1, 2).reshape(B, T, -1)

        def make_replace_hook(layer_idx, proj_name):
            def hook(module, input, output):
                if layer_idx != target_layer:
                    return
                if proj_name == "k":
                    return noisy_k_flat
                elif proj_name == "v":
                    return noisy_v_flat

            return hook

        hooks2 = []
        try:
            layer = model.model.layers[target_layer]
            hooks2.append(
                layer.self_attn.k_proj.register_forward_hook(
                    make_replace_hook(target_layer, "k")
                )
            )
            hooks2.append(
                layer.self_attn.v_proj.register_forward_hook(
                    make_replace_hook(target_layer, "v")
                )
            )
            with torch.no_grad():
                outputs_noisy = model(input_ids)
        finally:
            for h in hooks2:
                h.remove()

        logits_c = outputs_clean.logits[:, -1:, :].float()
        logits_n = outputs_noisy.logits[:, -1:, :].float()
        p = F.softmax(logits_c, dim=-1)
        q = F.softmax(logits_n, dim=-1)
        kl = (p * (p.clamp(min=1e-10).log() - q.clamp(min=1e-10).log())).sum(-1)
        return kl.mean().item()

    return 0.0


def _gaussian_noise_fn(sigma):
    """Create noise function that adds Gaussian noise to K and/or V."""

    def noise_k(k, v):
        noise = torch.randn_like(k) * sigma
        return k + noise, v

    def noise_v(k, v):
        noise = torch.randn_like(v) * sigma
        return k, v + noise

    def noise_kv(k, v):
        nk = torch.randn_like(k) * sigma
        nv = torch.randn_like(v) * sigma
        return k + nk, v + nv

    return noise_k, noise_v, noise_kv


def _quant_noise_fn(bits=4, block_size=32):
    """Create noise function that applies quant-dequant to K and/or V."""

    def noise_k(k, v):
        if bits == 4:
            kq, ks = quantize_int4_block(k.float(), block_size)
            k_noisy = dequantize_int4_block(kq, ks, block_size).to(k.dtype)
        else:
            kq, ks = quantize_int8_symmetric(k.float())
            k_noisy = dequantize_int8_symmetric(kq, ks).to(k.dtype)
        return k_noisy, v

    def noise_v(k, v):
        if bits == 4:
            vq, vs = quantize_int4_block(v.float(), block_size)
            v_noisy = dequantize_int4_block(vq, vs, block_size).to(v.dtype)
        else:
            vq, vs = quantize_int8_symmetric(v.float())
            v_noisy = dequantize_int8_symmetric(vq, vs).to(v.dtype)
        return k, v_noisy

    def noise_kv(k, v):
        k_n, _ = noise_k(k, v)
        _, v_n = noise_v(k, v)
        return k_n, v_n

    return noise_k, noise_v, noise_kv


def _compute_signal_over_layers(
    model, batches, n_layers, noise_fn_factory, measure, device
):
    """Compute a signal for all layers using provided noise function."""
    scores = {}
    for li in range(n_layers):
        layer_scores = []
        for input_ids in batches:
            try:
                noise_fn = noise_fn_factory()
                s = _hook_kv_and_measure(
                    model, input_ids, li, noise_fn, device, measure
                )
                layer_scores.append(s)
            except Exception as e:
                layer_scores.append(0.0)
        scores[li] = float(np.mean(layer_scores))
    return scores


@register_signal("A1_attnKL_noise_K", "activation_noise", "medium")
def signal_a1(model, token_data, device, n_layers, **kwargs):
    """Inject Gaussian noise into K, measure attention output MSE."""
    batches = get_calibration_batches(token_data, 8, 512, device)
    sigma = 0.1  # Scale relative to typical K magnitude

    def factory():
        return _gaussian_noise_fn(sigma)[0]  # noise_k only

    return _compute_signal_over_layers(
        model, batches, n_layers, factory, "attn_kl", device
    )


@register_signal("A2_attnKL_noise_V", "activation_noise", "medium")
def signal_a2(model, token_data, device, n_layers, **kwargs):
    """Inject Gaussian noise into V, measure attention output MSE."""
    batches = get_calibration_batches(token_data, 8, 512, device)
    sigma = 0.1

    def factory():
        return _gaussian_noise_fn(sigma)[1]  # noise_v only

    return _compute_signal_over_layers(
        model, batches, n_layers, factory, "attn_kl", device
    )


@register_signal("A3_attnKL_noise_KV", "activation_noise", "medium")
def signal_a3(model, token_data, device, n_layers, **kwargs):
    """Inject Gaussian noise into both K and V."""
    batches = get_calibration_batches(token_data, 8, 512, device)
    sigma = 0.1

    def factory():
        return _gaussian_noise_fn(sigma)[2]  # noise_kv

    return _compute_signal_over_layers(
        model, batches, n_layers, factory, "attn_kl", device
    )


@register_signal("A4_logitKL_noise", "activation_noise", "high")
def signal_a4(model, token_data, device, n_layers, **kwargs):
    """Inject noise into KV, measure logit KL divergence."""
    batches = get_calibration_batches(token_data, 4, 512, device)
    sigma = 0.1

    def factory():
        return _gaussian_noise_fn(sigma)[2]

    return _compute_signal_over_layers(
        model, batches, n_layers, factory, "logit_kl", device
    )


@register_signal("A5_slope_attnKL", "activation_noise", "high")
def signal_a5(model, token_data, device, n_layers, **kwargs):
    """Compute slope of attnKL at two noise scales."""
    batches = get_calibration_batches(token_data, 4, 512, device)
    sigma1, sigma2 = 0.05, 0.1

    scores1 = {}
    scores2 = {}
    for li in range(n_layers):
        s1_vals, s2_vals = [], []
        for input_ids in batches:
            try:
                fn1 = _gaussian_noise_fn(sigma1)[2]
                fn2 = _gaussian_noise_fn(sigma2)[2]
                s1 = _hook_kv_and_measure(model, input_ids, li, fn1, device, "attn_kl")
                s2 = _hook_kv_and_measure(model, input_ids, li, fn2, device, "attn_kl")
                s1_vals.append(s1)
                s2_vals.append(s2)
            except Exception:
                pass
        scores1[li] = float(np.mean(s1_vals)) if s1_vals else 0.0
        scores2[li] = float(np.mean(s2_vals)) if s2_vals else 0.0

    # Slope = (KL2 - KL1) / delta_sigma
    slopes = {}
    for li in range(n_layers):
        slopes[li] = (scores2[li] - scores1[li]) / (sigma2 - sigma1)
    return slopes


@register_signal("A6_fakeQuant_attnKL", "activation_noise", "medium")
def signal_a6(model, token_data, device, n_layers, **kwargs):
    """Apply actual INT4 quant-dequant to K/V, measure attention MSE."""
    batches = get_calibration_batches(token_data, 8, 512, device)

    def factory():
        return _quant_noise_fn(bits=4)[2]  # noise_kv

    return _compute_signal_over_layers(
        model, batches, n_layers, factory, "attn_kl", device
    )


# ============================================================
# B) Attention statistics proxies (cheap forward-only)
# ============================================================


def _collect_attention_stats(model, batches, n_layers):
    """Run forward passes collecting attention weights.

    SDPA does not return attention weights, so we temporarily switch
    to eager attention implementation for this collection.
    """
    all_attns = [[] for _ in range(n_layers)]

    # Force eager attention so we actually get attention weights
    orig_impl = getattr(model.config, "_attn_implementation", "sdpa")
    model.config._attn_implementation = "eager"
    # Also set on each layer's self_attn
    orig_layer_impls = []
    for li in range(n_layers):
        layer = model.model.layers[li]
        impl = getattr(layer.self_attn, "config", None)
        orig_layer_impls.append(
            getattr(impl, "_attn_implementation", "sdpa") if impl else "sdpa"
        )
        if hasattr(layer.self_attn, "config"):
            layer.self_attn.config._attn_implementation = "eager"

    try:
        for input_ids in batches:
            with torch.no_grad():
                outputs = model(input_ids, output_attentions=True)
            if outputs.attentions is not None:
                for li in range(min(n_layers, len(outputs.attentions))):
                    # [B, n_heads, T, T]
                    attn_w = outputs.attentions[li].detach()
                    all_attns[li].append(attn_w)
    finally:
        # Restore original attention implementation
        model.config._attn_implementation = orig_impl
        for li in range(n_layers):
            layer = model.model.layers[li]
            if hasattr(layer.self_attn, "config"):
                layer.self_attn.config._attn_implementation = orig_layer_impls[li]

    return all_attns


@register_signal("B1_attn_entropy", "attention_stats", "low")
def signal_b1(model, token_data, device, n_layers, **kwargs):
    """Average attention entropy per layer (lower entropy -> more sensitive)."""
    batches = get_calibration_batches(token_data, 16, 512, device)
    all_attns = _collect_attention_stats(model, batches, n_layers)

    scores = {}
    for li in range(n_layers):
        if not all_attns[li]:
            scores[li] = 0.0
            continue
        entropies = []
        for attn_w in all_attns[li]:
            # attn_w: [B, H, T, T]
            # Entropy: -sum(p * log(p))
            p = attn_w.float().clamp(min=1e-10)
            ent = -(p * p.log()).sum(dim=-1)  # [B, H, T]
            entropies.append(ent.mean().item())
        # Lower entropy = more peaked = more sensitive to perturbation
        # Invert so higher score = more sensitive
        avg_ent = np.mean(entropies)
        scores[li] = 1.0 / (avg_ent + 1e-8)
    return scores


@register_signal("B2_attn_maxprob", "attention_stats", "low")
def signal_b2(model, token_data, device, n_layers, **kwargs):
    """Average max attention probability (higher = more sensitive)."""
    batches = get_calibration_batches(token_data, 16, 512, device)
    all_attns = _collect_attention_stats(model, batches, n_layers)

    scores = {}
    for li in range(n_layers):
        if not all_attns[li]:
            scores[li] = 0.0
            continue
        maxprobs = []
        for attn_w in all_attns[li]:
            mp = attn_w.max(dim=-1).values.mean().item()
            maxprobs.append(mp)
        scores[li] = float(np.mean(maxprobs))
    return scores


@register_signal("B3_attn_margin", "attention_stats", "low")
def signal_b3(model, token_data, device, n_layers, **kwargs):
    """Average top1-top2 margin in attention logits."""
    batches = get_calibration_batches(token_data, 16, 512, device)
    all_attns = _collect_attention_stats(model, batches, n_layers)

    scores = {}
    for li in range(n_layers):
        if not all_attns[li]:
            scores[li] = 0.0
            continue
        margins = []
        for attn_w in all_attns[li]:
            # Sort attention values
            sorted_a, _ = attn_w.sort(dim=-1, descending=True)
            margin = (sorted_a[:, :, :, 0] - sorted_a[:, :, :, 1]).mean().item()
            margins.append(margin)
        scores[li] = float(np.mean(margins))
    return scores


@register_signal("B4_late_binding", "attention_stats", "low")
def signal_b4(model, token_data, device, n_layers, **kwargs):
    """Ratio of far-attention mass (positions beyond W_min=256)."""
    batches = get_calibration_batches(token_data, 16, 512, device)
    all_attns = _collect_attention_stats(model, batches, n_layers)
    W_min = 256

    scores = {}
    for li in range(n_layers):
        if not all_attns[li]:
            scores[li] = 0.0
            continue
        far_ratios = []
        for attn_w in all_attns[li]:
            T = attn_w.shape[-1]
            if T <= W_min:
                far_ratios.append(0.0)
                continue
            # For each query position, sum attention to positions beyond W_min
            # back from that position
            far_mass = 0.0
            total = 0.0
            for qi in range(W_min, T):
                far_mass += attn_w[:, :, qi, : qi - W_min].sum().item()
                total += attn_w[:, :, qi, :qi].sum().item()
            if total > 0:
                far_ratios.append(far_mass / total)
            else:
                far_ratios.append(0.0)
        # Higher far-attention ratio = more reliance on far tokens = more sensitive
        scores[li] = float(np.mean(far_ratios))
    return scores


# ============================================================
# C) Activation norms / scaling proxies
# ============================================================


def _collect_kv_norms(model, batches, n_layers, device):
    """Collect K and V norms per layer via hooks."""
    k_norms = [[] for _ in range(n_layers)]
    v_norms = [[] for _ in range(n_layers)]
    q_norms = [[] for _ in range(n_layers)]
    residual_norms = [[] for _ in range(n_layers)]
    attn_out_norms = [[] for _ in range(n_layers)]

    for input_ids in batches:
        # Collect via hooks on k_proj, v_proj outputs
        kv_data = {}

        def make_proj_hook(layer_idx, proj):
            def hook(module, input, output):
                kv_data[(layer_idx, proj)] = output.detach()

            return hook

        def make_attn_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    kv_data[(layer_idx, "attn_out")] = output[0].detach()

            return hook

        def make_layer_hook(layer_idx):
            def hook(module, input):
                if isinstance(input, tuple):
                    kv_data[(layer_idx, "residual")] = input[0].detach()

            return hook

        hooks = []
        try:
            for li in range(n_layers):
                layer = model.model.layers[li]
                hooks.append(
                    layer.self_attn.q_proj.register_forward_hook(
                        make_proj_hook(li, "q")
                    )
                )
                hooks.append(
                    layer.self_attn.k_proj.register_forward_hook(
                        make_proj_hook(li, "k")
                    )
                )
                hooks.append(
                    layer.self_attn.v_proj.register_forward_hook(
                        make_proj_hook(li, "v")
                    )
                )
                hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(li)))
                hooks.append(layer.register_forward_pre_hook(make_layer_hook(li)))

            with torch.no_grad():
                _ = model(input_ids)
        finally:
            for h in hooks:
                h.remove()

        for li in range(n_layers):
            if (li, "k") in kv_data:
                k_norms[li].append(
                    kv_data[(li, "k")].float().norm(dim=-1).mean().item()
                )
            if (li, "v") in kv_data:
                v_norms[li].append(
                    kv_data[(li, "v")].float().norm(dim=-1).mean().item()
                )
            if (li, "q") in kv_data:
                q_norms[li].append(
                    kv_data[(li, "q")].float().norm(dim=-1).mean().item()
                )
            if (li, "attn_out") in kv_data:
                attn_out_norms[li].append(
                    kv_data[(li, "attn_out")].float().norm(dim=-1).mean().item()
                )
            if (li, "residual") in kv_data:
                residual_norms[li].append(
                    kv_data[(li, "residual")].float().norm(dim=-1).mean().item()
                )

    return k_norms, v_norms, q_norms, attn_out_norms, residual_norms


@register_signal("C1_norm_QK", "activation_norms", "low")
def signal_c1(model, token_data, device, n_layers, **kwargs):
    """Average ||Q||*||K|| scale per layer."""
    batches = get_calibration_batches(token_data, 16, 512, device)
    k_norms, v_norms, q_norms, _, _ = _collect_kv_norms(
        model, batches, n_layers, device
    )
    scores = {}
    for li in range(n_layers):
        qn = np.mean(q_norms[li]) if q_norms[li] else 1.0
        kn = np.mean(k_norms[li]) if k_norms[li] else 1.0
        scores[li] = float(qn * kn)
    return scores


@register_signal("C2_norm_V", "activation_norms", "low")
def signal_c2(model, token_data, device, n_layers, **kwargs):
    """Average ||V|| per layer."""
    batches = get_calibration_batches(token_data, 16, 512, device)
    _, v_norms, _, _, _ = _collect_kv_norms(model, batches, n_layers, device)
    scores = {}
    for li in range(n_layers):
        scores[li] = float(np.mean(v_norms[li])) if v_norms[li] else 0.0
    return scores


@register_signal("C3_residual_ratio", "activation_norms", "low")
def signal_c3(model, token_data, device, n_layers, **kwargs):
    """||attn_out|| / ||residual|| per layer."""
    batches = get_calibration_batches(token_data, 16, 512, device)
    _, _, _, attn_out_norms, residual_norms = _collect_kv_norms(
        model, batches, n_layers, device
    )
    scores = {}
    for li in range(n_layers):
        ao = np.mean(attn_out_norms[li]) if attn_out_norms[li] else 1.0
        rn = np.mean(residual_norms[li]) if residual_norms[li] else 1.0
        scores[li] = float(ao / max(rn, 1e-8))
    return scores


# ============================================================
# D) Spectrum / effective rank proxies
# ============================================================


def _collect_kv_tensors(model, batches, n_layers, device):
    """Collect raw K and V tensors from a forward pass.

    Hooks k_proj and v_proj outputs to capture K/V projections directly,
    since Qwen2 attention does not return past_kv in its output tuple.
    """
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    n_kv_heads = getattr(
        model.config, "num_key_value_heads", model.config.num_attention_heads
    )

    k_all = [[] for _ in range(n_layers)]
    v_all = [[] for _ in range(n_layers)]

    for input_ids in batches:
        kv_data = {}

        def make_proj_hook(layer_idx, proj_name):
            def hook(module, input, output):
                kv_data[(layer_idx, proj_name)] = output.detach().cpu()

            return hook

        hooks = []
        try:
            for li in range(n_layers):
                layer = model.model.layers[li]
                hooks.append(
                    layer.self_attn.k_proj.register_forward_hook(
                        make_proj_hook(li, "k")
                    )
                )
                hooks.append(
                    layer.self_attn.v_proj.register_forward_hook(
                        make_proj_hook(li, "v")
                    )
                )

            with torch.no_grad():
                _ = model(input_ids)
        finally:
            for h in hooks:
                h.remove()

        for li in range(n_layers):
            if (li, "k") in kv_data:
                # Reshape from [B, T, kv_dim] to [B, n_kv_heads, T, head_dim]
                k_flat = kv_data[(li, "k")]
                B, T, _ = k_flat.shape
                k_shaped = k_flat.view(B, T, n_kv_heads, head_dim).transpose(1, 2)
                k_all[li].append(k_shaped)
            if (li, "v") in kv_data:
                v_flat = kv_data[(li, "v")]
                B, T, _ = v_flat.shape
                v_shaped = v_flat.view(B, T, n_kv_heads, head_dim).transpose(1, 2)
                v_all[li].append(v_shaped)

    return k_all, v_all


def _effective_rank(tensor_list, energy_thresh=0.99):
    """Compute effective rank: min rank capturing energy_thresh of SVD energy."""
    if not tensor_list:
        return 0.0
    ranks = []
    for t in tensor_list:
        # t: [B, H, T, D] -> flatten to [H*T, D]
        mat = t.float().reshape(-1, t.shape[-1])
        # Sample rows if too many
        if mat.shape[0] > 2048:
            idx = torch.randperm(mat.shape[0])[:2048]
            mat = mat[idx]
        try:
            _, s, _ = torch.svd(mat)
            energy = (s**2).cumsum(0) / (s**2).sum()
            rank = (energy < energy_thresh).sum().item() + 1
            ranks.append(rank)
        except Exception:
            ranks.append(t.shape[-1])
    return float(np.mean(ranks))


@register_signal("D1_K_effective_rank", "spectrum", "medium")
def signal_d1(model, token_data, device, n_layers, **kwargs):
    """Effective rank of K per layer (higher rank = harder to compress = more sensitive)."""
    batches = get_calibration_batches(token_data, 8, 512, device)
    k_all, _ = _collect_kv_tensors(model, batches, n_layers, device)
    scores = {}
    for li in range(n_layers):
        scores[li] = _effective_rank(k_all[li])
    return scores


@register_signal("D2_V_effective_rank", "spectrum", "medium")
def signal_d2(model, token_data, device, n_layers, **kwargs):
    """Effective rank of V per layer."""
    batches = get_calibration_batches(token_data, 8, 512, device)
    _, v_all = _collect_kv_tensors(model, batches, n_layers, device)
    scores = {}
    for li in range(n_layers):
        scores[li] = _effective_rank(v_all[li])
    return scores


@register_signal("D3_K_condnum", "spectrum", "medium")
def signal_d3(model, token_data, device, n_layers, **kwargs):
    """Condition number proxy for K (ratio of top singular values)."""
    batches = get_calibration_batches(token_data, 8, 512, device)
    k_all, _ = _collect_kv_tensors(model, batches, n_layers, device)
    scores = {}
    for li in range(n_layers):
        if not k_all[li]:
            scores[li] = 0.0
            continue
        conds = []
        for t in k_all[li]:
            mat = t.float().reshape(-1, t.shape[-1])
            if mat.shape[0] > 2048:
                idx = torch.randperm(mat.shape[0])[:2048]
                mat = mat[idx]
            try:
                _, s, _ = torch.svd(mat)
                cond = (s[0] / s[-1].clamp(min=1e-10)).item()
                conds.append(min(cond, 1e6))
            except Exception:
                conds.append(1.0)
        scores[li] = float(np.mean(conds))
    return scores


# ============================================================
# E) Parameter-curvature proxies (from v18)
# ============================================================


@register_signal("E1_adam_vhat_KV", "param_curvature", "high", needs_backward=True)
def signal_e1(model, token_data, device, n_layers, **kwargs):
    """Mean Adam v-hat over K/V projection weights."""
    from eval_v18 import extract_adam_vhat, apply_root4_transform

    raw_scores = extract_adam_vhat(
        model, token_data, device, n_steps=kwargs.get("calib_steps", 100)
    )
    return raw_scores


@register_signal("E2_adam_vhat_allattn", "param_curvature", "high", needs_backward=True)
def signal_e2(model, token_data, device, n_layers, **kwargs):
    """Mean Adam v-hat over all attention weights (Q, K, V, O)."""
    orig_dtype = next(model.parameters()).dtype
    try:
        model.float()
        use_fp32 = True
    except RuntimeError:
        use_fp32 = False
    model.train()

    attn_params = []
    param_id_to_name = {}
    for name, p in model.named_parameters():
        if any(proj in name for proj in ["k_proj", "v_proj", "q_proj", "o_proj"]):
            p.requires_grad = True
            attn_params.append(p)
            param_id_to_name[id(p)] = name
        else:
            p.requires_grad = False

    optimizer = torch.optim.Adam(attn_params, lr=1e-5, betas=(0.9, 0.999))
    rng = np.random.RandomState(42)
    n_steps = kwargs.get("calib_steps", 100)

    for step in range(n_steps):
        idx = get_text_batch(token_data, 1, 513, rng).to(device)
        input_ids = idx[:, :512]
        labels = idx[:, 1:513]
        outputs = model(input_ids)
        logits = outputs.logits.float()
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.to(logits.device).view(-1)
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    layer_vhat = {}
    for pg in optimizer.param_groups:
        for p in pg["params"]:
            state = optimizer.state[p]
            if "exp_avg_sq" not in state:
                continue
            pname = param_id_to_name.get(id(p))
            if pname is None:
                continue
            parts = pname.split(".")
            layer_idx = None
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    layer_idx = int(parts[i + 1])
                    break
            if layer_idx is None:
                continue
            if layer_idx not in layer_vhat:
                layer_vhat[layer_idx] = []
            layer_vhat[layer_idx].append(state["exp_avg_sq"].mean().item())

    raw_scores = {}
    for li in sorted(layer_vhat.keys()):
        raw_scores[li] = float(np.mean(layer_vhat[li]))

    model.eval()
    if use_fp32:
        model.to(orig_dtype)
    for p in model.parameters():
        p.requires_grad = False

    return raw_scores


@register_signal("E3_gradnorm_KV", "param_curvature", "medium", needs_backward=True)
def signal_e3(model, token_data, device, n_layers, **kwargs):
    """E[||grad||^2] over K/V projection params on calibration set."""
    model.train()
    for p in model.parameters():
        p.requires_grad = False

    kv_params = {}
    for name, p in model.named_parameters():
        if "k_proj" in name or "v_proj" in name:
            p.requires_grad = True
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    layer_idx = int(parts[i + 1])
                    if layer_idx not in kv_params:
                        kv_params[layer_idx] = []
                    kv_params[layer_idx].append(p)
                    break

    rng = np.random.RandomState(42)
    n_steps = 50
    grad_norms = {li: [] for li in kv_params}

    for step in range(n_steps):
        idx = get_text_batch(token_data, 1, 513, rng).to(device)
        input_ids = idx[:, :512]
        labels = idx[:, 1:513]
        outputs = model(input_ids)
        logits = outputs.logits.float()
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.to(logits.device).view(-1)
        )
        loss.backward()

        for li, params in kv_params.items():
            gnorm = sum(
                p.grad.float().norm().item() ** 2 for p in params if p.grad is not None
            )
            grad_norms[li].append(gnorm)

        model.zero_grad()

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    scores = {}
    for li in range(n_layers):
        if li in grad_norms and grad_norms[li]:
            scores[li] = float(np.mean(grad_norms[li]))
        else:
            scores[li] = 0.0
    return scores


# ============================================================
# F) Transform ablations
# ============================================================

TRANSFORMS = {
    "raw": lambda x: x,
    "sqrt": lambda x: x**0.5,
    "root4": lambda x: x**0.25,
    "root8": lambda x: x**0.125,
    "log1p": lambda x: np.log1p(x),
}


def apply_transforms(raw_scores):
    """Apply all transforms to raw scores, return dict of transformed scores."""
    results = {}
    for tname, tfn in TRANSFORMS.items():
        transformed = {}
        for li, score in raw_scores.items():
            transformed[li] = float(tfn(max(score, 0)))
        results[tname] = transformed
    return results


# ============================================================
# Oracle computation
# ============================================================


def compute_oracle_sensitivity(
    model, token_data, device, max_ctx, model_config, n_layers, L=8192, seeds=(0, 1, 2)
):
    """Compute per-layer INT4 sensitivity (oracle ground truth).

    For each layer, quantize only that layer's KV to INT4, measure PPL delta.
    """
    print(f"\n  Computing oracle sensitivity at L={L}...")

    # First get dense baselines
    dense_be = DenseBackend()
    dense_ppls = {}
    for seed in seeds:
        dense_be.configure(L, model_config)
        r = run_single_eval(
            dense_be,
            model,
            token_data,
            L,
            256,
            seed,
            device,
            max_ctx,
            model_config,
        )
        dense_ppls[seed] = r.ppl

    avg_dense = float(np.mean(list(dense_ppls.values())))
    print(f"  Dense avg PPL: {avg_dense:.3f}")

    # Per-layer INT4 sensitivity
    sensitivity = []
    for li in range(n_layers):
        # Schedule: INT8 everywhere except layer li which gets INT4
        sched = [8] * n_layers
        sched[li] = 4
        be = MixedPrecisionBackend(layer_bits=sched)
        be._name = f"layer{li}_int4"

        ppls = []
        for seed in seeds:
            be.configure(L, model_config)
            be.calibrate(model, token_data, L, device, model_config)
            try:
                r = run_single_eval(
                    be, model, token_data, L, 256, seed, device, max_ctx, model_config
                )
                ppls.append(r.ppl)
            except Exception as e:
                ppls.append(float("inf"))

        avg_ppl = float(np.mean(ppls))
        delta_pct = 100.0 * (avg_ppl - avg_dense) / avg_dense if avg_dense > 0 else 0
        sensitivity.append(
            {
                "layer": li,
                "bits": 4,
                "avg_ppl": round(avg_ppl, 4),
                "ppl_delta_pct": round(delta_pct, 3),
                "ppls": [round(p, 3) for p in ppls],
            }
        )
        print(f"    layer {li:2d}: delta={delta_pct:+.3f}%")

    # Sort tolerant to sensitive
    ranked = sorted(sensitivity, key=lambda e: e["ppl_delta_pct"])

    oracle = {
        "model": "qwen05b",
        "L": L,
        "n_layers": n_layers,
        "avg_dense_ppl": round(avg_dense, 3),
        "dense_ppls": {str(s): round(p, 4) for s, p in dense_ppls.items()},
        "int4_sensitivity": sensitivity,
        "int4_ranked_tolerant_to_sensitive": [
            {"layer": e["layer"], "ppl_delta_pct": e["ppl_delta_pct"]} for e in ranked
        ],
    }

    # Also create oracle scores (higher = more sensitive)
    oracle_scores = {}
    for entry in sensitivity:
        oracle_scores[entry["layer"]] = entry["ppl_delta_pct"]

    return oracle, oracle_scores


# ============================================================
# Schedule generation (greedy allocator)
# ============================================================


def greedy_allocate(scores, n_layers, n_int8):
    """Sort layers by score descending, top n_int8 get INT8."""
    order = sorted(scores.keys(), key=lambda l: scores[l], reverse=True)
    schedule = [4] * n_layers
    for i in range(min(n_int8, len(order))):
        schedule[order[i]] = 8
    return schedule


def build_random_schedule(n_layers, n_int8, seed=42):
    """Random schedule with exactly n_int8 INT8 layers."""
    rng = np.random.RandomState(seed)
    int8_layers = set(rng.choice(n_layers, n_int8, replace=False))
    return [8 if i in int8_layers else 4 for i in range(n_layers)]


# ============================================================
# Phase runners
# ============================================================


def run_phase0(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 0: Baseline lock + oracle ground truth."""
    outdir = os.path.join(args.outdir, "phase0")
    oracle_dir = os.path.join(args.outdir, "artifacts", "bitterkv")
    os.makedirs(oracle_dir, exist_ok=True)

    # 0.1 Baselines
    dense_results, dense_ppls = run_dense_baselines(
        model,
        token_data,
        valid_L,
        args.decode_steps,
        args.seeds,
        args.device,
        max_ctx,
        model_config,
    )

    from backends.quant import QuantBackend

    int8_be = QuantBackend()
    schedules = build_schedules(args.sensitivity_path)
    s2_sched = schedules["S2"]
    s2_be = MixedPrecisionBackend(layer_bits=s2_sched)
    s2_be._name = "S2_manual"

    backends = [("INT8", int8_be), ("S2_manual", s2_be)]
    comp_results = run_backend_sweep(
        backends,
        model,
        token_data,
        valid_L,
        args.decode_steps,
        args.seeds,
        args.device,
        max_ctx,
        model_config,
        dense_ppls,
    )

    all_results = dense_results + comp_results
    save_results(all_results, dense_ppls, outdir, 0, {"gpu_info": gpu_info})

    # 0.2 Oracle
    n_layers = model_config["n_layers"]
    oracle, oracle_scores = compute_oracle_sensitivity(
        model,
        token_data,
        args.device,
        max_ctx,
        model_config,
        n_layers,
        L=8192,
        seeds=args.seeds,
    )

    with open(os.path.join(oracle_dir, "oracle_empirical.json"), "w") as f:
        json.dump(oracle, f, indent=2)

    print(f"\nSaved oracle to {oracle_dir}/oracle_empirical.json")


def run_phase1(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 1: Compute ALL BitterKV signals."""
    sig_dir = os.path.join(args.outdir, "artifacts", "bitterkv", "signals")
    os.makedirs(sig_dir, exist_ok=True)
    outdir = os.path.join(args.outdir, "phase1")
    os.makedirs(outdir, exist_ok=True)

    n_layers = model_config["n_layers"]

    print(f"\n{'=' * 60}")
    print(f"Phase 1: Computing {len(SIGNALS)} BitterKV signals")
    print("=" * 60)

    results_summary = {}
    for sig_name, sig_info in SIGNALS.items():
        print(f"\n  [{sig_info['category']}] {sig_name} (cost={sig_info['cost']})...")
        t0 = time.time()
        try:
            # Ensure clean model state before each signal
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            torch.cuda.empty_cache()

            raw_scores = sig_info["fn"](
                model,
                token_data,
                args.device,
                n_layers,
                calib_steps=args.calib_steps,
            )

            # Apply transforms
            transformed = apply_transforms(raw_scores)

            # Save
            artifact = {
                "signal": sig_name,
                "category": sig_info["category"],
                "cost": sig_info["cost"],
                "needs_backward": sig_info["needs_backward"],
                "raw_scores": {str(k): v for k, v in raw_scores.items()},
                "transformed": {
                    tname: {str(k): v for k, v in tscores.items()}
                    for tname, tscores in transformed.items()
                },
                "compute_time_s": round(time.time() - t0, 1),
            }
            with open(os.path.join(sig_dir, f"{sig_name}.json"), "w") as f:
                json.dump(artifact, f, indent=2)

            # Print ranking
            ranking = sorted(
                raw_scores.keys(), key=lambda l: raw_scores[l], reverse=True
            )
            top5 = ranking[:5]
            elapsed = time.time() - t0
            print(
                f"    top-5: {top5}  "
                f"range=[{min(raw_scores.values()):.4g}, {max(raw_scores.values()):.4g}]  "
                f"({elapsed:.1f}s)"
            )

            results_summary[sig_name] = {
                "status": "OK",
                "top5": top5,
                "compute_time_s": round(elapsed, 1),
            }

        except Exception as e:
            import traceback

            traceback.print_exc()
            elapsed = time.time() - t0
            print(f"    FAILED: {e} ({elapsed:.1f}s)")
            results_summary[sig_name] = {
                "status": "FAILED",
                "error": str(e),
                "compute_time_s": round(elapsed, 1),
            }

    meta = {
        "timestamp": datetime.now().isoformat(),
        "phase": 1,
        "version": "v19",
        "gpu_info": gpu_info,
        "n_signals": len(SIGNALS),
        "summary": results_summary,
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nPhase 1 complete. Saved {len(SIGNALS)} signals to {sig_dir}/")


def run_phase2(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 2: Correlation analysis."""
    from scipy.stats import spearmanr

    outdir = os.path.join(args.outdir, "phase2")
    os.makedirs(outdir, exist_ok=True)
    bk_dir = os.path.join(args.outdir, "artifacts", "bitterkv")
    sig_dir = os.path.join(bk_dir, "signals")

    n_layers = model_config["n_layers"]

    print(f"\n{'=' * 60}")
    print("Phase 2: Correlation analysis")
    print("=" * 60)

    # Load oracle
    with open(os.path.join(bk_dir, "oracle_empirical.json")) as f:
        oracle_data = json.load(f)

    oracle_scores = {}
    for entry in oracle_data["int4_sensitivity"]:
        oracle_scores[entry["layer"]] = entry["ppl_delta_pct"]

    oracle_ranking = [
        e["layer"]
        for e in sorted(
            oracle_data["int4_ranked_tolerant_to_sensitive"],
            key=lambda e: e["ppl_delta_pct"],
            reverse=True,
        )
    ]

    # Load all signals
    correlations = {}
    sig_files = sorted(f for f in os.listdir(sig_dir) if f.endswith(".json"))

    for sf in sig_files:
        with open(os.path.join(sig_dir, sf)) as f:
            sig_data = json.load(f)

        sig_name = sig_data["signal"]

        # Test raw and all transforms
        for tname in ["raw"] + list(TRANSFORMS.keys()):
            if tname == "raw":
                scores = {int(k): v for k, v in sig_data["raw_scores"].items()}
            elif tname in sig_data.get("transformed", {}):
                scores = {int(k): v for k, v in sig_data["transformed"][tname].items()}
            else:
                continue

            key = f"{sig_name}_{tname}" if tname != "raw" else sig_name

            # Align layers
            layers = sorted(set(oracle_scores.keys()) & set(scores.keys()))
            if len(layers) < 3:
                continue

            oracle_vec = [oracle_scores[l] for l in layers]
            sig_vec = [scores[l] for l in layers]

            rho, p_val = spearmanr(oracle_vec, sig_vec)

            # Top-K overlap
            sig_ranking = sorted(layers, key=lambda l: scores[l], reverse=True)
            overlaps = {}
            for K in [2, 4, 6, 8]:
                oracle_topK = set(oracle_ranking[:K])
                sig_topK = set(sig_ranking[:K])
                overlaps[K] = len(oracle_topK & sig_topK)

            correlations[key] = {
                "signal": sig_name,
                "transform": tname,
                "category": sig_data["category"],
                "spearman_rho": round(rho, 4) if not np.isnan(rho) else 0.0,
                "p_value": round(p_val, 4) if not np.isnan(p_val) else 1.0,
                "top_K_overlap": overlaps,
                "ranking": sig_ranking[:8],
                "cost": sig_data["cost"],
            }

    # Sort by Spearman rho
    sorted_corrs = sorted(
        correlations.items(), key=lambda x: abs(x[1]["spearman_rho"]), reverse=True
    )

    print(
        f"\n  {'Signal':<40s} {'rho':>6s} {'p':>7s} {'top2':>4s} {'top4':>4s} {'top6':>4s}"
    )
    print("  " + "-" * 75)
    for key, c in sorted_corrs[:30]:
        print(
            f"  {key:<40s} {c['spearman_rho']:>6.3f} {c['p_value']:>7.4f} "
            f"{c['top_K_overlap'][2]:>4d} {c['top_K_overlap'][4]:>4d} {c['top_K_overlap'][6]:>4d}"
        )

    # Save
    corr_output = {
        "oracle_ranking": oracle_ranking,
        "oracle_scores": {str(k): v for k, v in oracle_scores.items()},
        "correlations": correlations,
        "sorted_by_rho": [k for k, _ in sorted_corrs],
    }
    with open(os.path.join(bk_dir, "correlations.json"), "w") as f:
        json.dump(corr_output, f, indent=2)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "phase": 2,
        "version": "v19",
        "gpu_info": gpu_info,
        "n_signals_tested": len(correlations),
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved correlations to {bk_dir}/correlations.json")


def run_phase3(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 3: Schedule generation from top signals + full eval."""
    outdir = os.path.join(args.outdir, "phase3")
    sched_dir = os.path.join(args.outdir, "artifacts", "bitterkv", "schedules")
    os.makedirs(sched_dir, exist_ok=True)
    bk_dir = os.path.join(args.outdir, "artifacts", "bitterkv")

    n_layers = model_config["n_layers"]

    print(f"\n{'=' * 60}")
    print("Phase 3: Schedule generation from signals")
    print("=" * 60)

    # Load correlations to pick top signals
    with open(os.path.join(bk_dir, "correlations.json")) as f:
        corr_data = json.load(f)

    # Load oracle
    with open(os.path.join(bk_dir, "oracle_empirical.json")) as f:
        oracle_data = json.load(f)
    oracle_scores = {}
    for entry in oracle_data["int4_sensitivity"]:
        oracle_scores[entry["layer"]] = entry["ppl_delta_pct"]

    # Select top signals: best overall, best per category, plus controls
    sorted_keys = corr_data["sorted_by_rho"]
    signal_candidates = {}

    # Best overall
    for key in sorted_keys:
        c = corr_data["correlations"][key]
        cat = c["category"]
        if cat not in signal_candidates or abs(c["spearman_rho"]) > abs(
            signal_candidates[cat]["spearman_rho"]
        ):
            signal_candidates[cat] = {**c, "key": key}

    # Always include oracle and controls
    selected_signals = {}

    # Load signal scores for selected signals
    sig_dir = os.path.join(bk_dir, "signals")
    for cat, info in signal_candidates.items():
        sig_name = info["signal"]
        tname = info["transform"]
        sig_path = os.path.join(sig_dir, f"{sig_name}.json")
        if not os.path.exists(sig_path):
            continue
        with open(sig_path) as f:
            sig_data = json.load(f)
        if tname == "raw":
            scores = {int(k): v for k, v in sig_data["raw_scores"].items()}
        else:
            scores = {
                int(k): v for k, v in sig_data["transformed"].get(tname, {}).items()
            }
        key = info["key"]
        selected_signals[key] = scores

    # Add oracle and controls
    selected_signals["oracle"] = oracle_scores
    # Adam root4 from v18 as baseline control
    adam_path = os.path.join(sig_dir, "E1_adam_vhat_KV.json")
    if os.path.exists(adam_path):
        with open(adam_path) as f:
            adam_data = json.load(f)
        if "root4" in adam_data.get("transformed", {}):
            selected_signals["E1_adam_root4"] = {
                int(k): v for k, v in adam_data["transformed"]["root4"].items()
            }

    print(f"\n  Selected {len(selected_signals)} signals for schedule evaluation:")
    for key in selected_signals:
        print(f"    {key}")

    # Step 1: Find minimal k for each signal at L=8192
    print(f"\n{'=' * 60}")
    print("Step 1: Search for minimal k per signal")
    print("=" * 60)

    dense_results_quick, dense_ppls_quick = run_dense_baselines(
        model,
        token_data,
        [8192],
        args.decode_steps,
        args.seeds,
        args.device,
        max_ctx,
        model_config,
    )

    schedule_results = {}
    for sig_key, scores in selected_signals.items():
        print(f"\n  Signal: {sig_key}")
        best_k = None
        for k in range(0, n_layers + 1):
            sched = greedy_allocate(scores, n_layers, k)
            be = MixedPrecisionBackend(layer_bits=sched)
            be._name = f"{sig_key}_k{k}"

            results = run_backend_sweep(
                [(f"{sig_key}_k{k}", be)],
                model,
                token_data,
                [8192],
                args.decode_steps,
                args.seeds,
                args.device,
                max_ctx,
                model_config,
                dense_ppls_quick,
            )
            apply_quality_gating(results, dense_ppls_quick)
            n_pass = sum(1 for r in results if r.passed_3pct)
            n_total = len(results)
            n4 = sum(1 for b in sched if b == 4)
            print(f"    k={k}: {n_pass}/{n_total} @3%, {n4}xINT4 + {k}xINT8")
            if n_pass == n_total:
                best_k = k
                break

        if best_k is None:
            best_k = n_layers
            print(f"    WARNING: No k found, using all INT8")

        best_sched = greedy_allocate(scores, n_layers, best_k)
        int8_layers = [i for i, b in enumerate(best_sched) if b == 8]
        schedule_results[sig_key] = {
            "k": best_k,
            "schedule": best_sched,
            "int8_layers": int8_layers,
        }
        print(f"    Best k={best_k}, INT8 layers={int8_layers}")

    # Add random control and S2 manual
    best_k_oracle = schedule_results.get("oracle", {}).get("k", 6)
    schedule_results["random"] = {
        "k": best_k_oracle,
        "schedule": build_random_schedule(n_layers, best_k_oracle),
        "int8_layers": [
            i
            for i, b in enumerate(build_random_schedule(n_layers, best_k_oracle))
            if b == 8
        ],
    }

    v16_scheds = build_schedules(args.sensitivity_path)
    s2_sched = v16_scheds["S2"]
    schedule_results["S2_manual"] = {
        "k": sum(1 for b in s2_sched if b == 8),
        "schedule": s2_sched,
        "int8_layers": [i for i, b in enumerate(s2_sched) if b == 8],
    }

    # Save all schedules
    with open(os.path.join(sched_dir, "all_schedules.json"), "w") as f:
        json.dump(schedule_results, f, indent=2)

    # Step 2: Full evaluation of all schedules
    print(f"\n{'=' * 60}")
    print("Step 2: Full evaluation at all context lengths")
    print("=" * 60)

    dense_results, dense_ppls = run_dense_baselines(
        model,
        token_data,
        valid_L,
        args.decode_steps,
        args.seeds,
        args.device,
        max_ctx,
        model_config,
    )

    backends = []
    for sig_key, sinfo in schedule_results.items():
        be = MixedPrecisionBackend(layer_bits=sinfo["schedule"])
        be._name = sig_key
        backends.append((sig_key, be))

    # Also INT8 everywhere
    int8_be = QuantBackend()
    backends.append(("INT8", int8_be))

    comp_results = run_backend_sweep(
        backends,
        model,
        token_data,
        valid_L,
        args.decode_steps,
        args.seeds,
        args.device,
        max_ctx,
        model_config,
        dense_ppls,
    )

    all_results = dense_results + comp_results
    save_results(
        all_results,
        dense_ppls,
        outdir,
        3,
        {"gpu_info": gpu_info, "schedules": schedule_results},
    )


def run_phase5(args, model, token_data, valid_L, max_ctx, model_config, gpu_info):
    """Phase 5: Bandwidth-bound validation."""
    outdir = os.path.join(args.outdir, "phase5")
    os.makedirs(outdir, exist_ok=True)
    bk_dir = os.path.join(args.outdir, "artifacts", "bitterkv")

    print(f"\n{'=' * 60}")
    print("Phase 5: Bandwidth-bound validation")
    print("=" * 60)

    # Load best schedules from Phase 3
    sched_path = os.path.join(bk_dir, "schedules", "all_schedules.json")
    if os.path.exists(sched_path):
        with open(sched_path) as f:
            sched_data = json.load(f)
    else:
        print("  No schedules found, using S2 manual")
        v16_scheds = build_schedules(args.sensitivity_path)
        sched_data = {
            "S2_manual": {
                "schedule": v16_scheds["S2"],
                "k": sum(1 for b in v16_scheds["S2"] if b == 8),
            }
        }

    # Pick top-2 best schedules by k (lowest k = best compression)
    sorted_scheds = sorted(sched_data.items(), key=lambda x: x[1]["k"])
    top_scheds = sorted_scheds[:3]

    L_test = 8192
    decode_steps = 256
    microbatches = [1, 4, 8]

    results = {}
    for mb in microbatches:
        print(f"\n  Microbatch={mb}, L={L_test}")
        rng = np.random.RandomState(0)
        total_len = L_test + decode_steps
        idx = get_text_batch(token_data, mb, total_len, rng).to(args.device)
        prefix = idx[:, :L_test]
        continuation = idx[:, L_test : L_test + decode_steps]

        dense_be = DenseBackend()
        dense_be.configure(L_test, model_config)

        mb_results = {}
        test_backends = [("dense", dense_be)]
        for sname, sinfo in top_scheds:
            be = MixedPrecisionBackend(layer_bits=sinfo["schedule"])
            be._name = sname
            be.configure(L_test, model_config)
            be.calibrate(model, token_data, L_test, args.device, model_config)
            test_backends.append((sname, be))

        for bname, be in test_backends:
            times = []
            for trial in range(3):
                torch.cuda.empty_cache()
                gpu_sync(args.device)
                t0 = time.perf_counter()
                try:
                    with torch.no_grad():
                        logits, stats = be.run_decode(
                            model, prefix, continuation, args.device, max_ctx
                        )
                    gpu_sync(args.device)
                    elapsed = time.perf_counter() - t0
                    times.append(elapsed)
                    del logits
                except Exception as e:
                    print(f"    {bname} mb={mb}: OOM ({e})")
                    break

            if times:
                avg_ms = np.mean(times) * 1000
                p50_per_tok = avg_ms / decode_steps
                mb_results[bname] = {
                    "total_ms": round(avg_ms, 1),
                    "p50_per_tok_ms": round(p50_per_tok, 2),
                }
                print(f"    {bname}: {avg_ms:.1f}ms total, {p50_per_tok:.2f}ms/tok")
            else:
                mb_results[bname] = {"status": "OOM"}

        results[f"mb{mb}"] = mb_results

    with open(os.path.join(outdir, "bandwidth_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "phase": 5,
        "version": "v19",
        "gpu_info": gpu_info,
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved bandwidth results to {outdir}/")


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="BPA v19: BitterKV Signal Bakeoff")
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--model", default="qwen05b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decode_steps", type=int, default=256)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--L",
        nargs="+",
        type=int,
        default=[8192, 16384, 32768],
    )
    parser.add_argument("--outdir", default="results/v19")
    parser.add_argument(
        "--sensitivity_path",
        default="results/v15/phase4/layer_sensitivity.json",
    )
    parser.add_argument("--calib_steps", type=int, default=100)
    args = parser.parse_args()

    gpu_info = gpu_preflight(args.device)
    model, tokenizer, max_ctx, model_config = load_model(args.model, args.device)

    print("Loading validation data...")
    token_data = load_validation_tokens(tokenizer)
    valid_L = [L for L in args.L if L <= max_ctx]

    if args.phase == 0:
        run_phase0(args, model, token_data, valid_L, max_ctx, model_config, gpu_info)
    elif args.phase == 1:
        run_phase1(args, model, token_data, valid_L, max_ctx, model_config, gpu_info)
    elif args.phase == 2:
        run_phase2(args, model, token_data, valid_L, max_ctx, model_config, gpu_info)
    elif args.phase == 3:
        run_phase3(args, model, token_data, valid_L, max_ctx, model_config, gpu_info)
    elif args.phase == 5:
        run_phase5(args, model, token_data, valid_L, max_ctx, model_config, gpu_info)
    else:
        print(f"Unknown phase: {args.phase}")


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: MIT
"""Measure residual distributions and separate floating-format damage by coverage."""

import contextlib

import numpy as np
import torch
from torch import nn

from research.tinystories_depth.float_quant import scaled_float
from research.tinystories_depth.kv_cache import (
    cases,
    evaluate,
    fakequant,
    bytes_per_token,
)
from research.tinystories_depth.train import state_hash


@torch.no_grad()
def describe(x):
    source = x.detach()
    x = source.float()
    flat = x.flatten()
    sample = flat[
        torch.linspace(0, len(flat) - 1, min(4096, len(flat)), device=x.device).long()
    ]
    rms = x.square().mean().sqrt().item()
    return dict(
        shape=list(x.shape),
        count=x.numel(),
        mean=x.mean().item(),
        variance=x.var(unbiased=False).item(),
        mean_channel_variance=x.var(-1, unbiased=False).mean().item(),
        rms=rms,
        absmax=x.abs().max().item(),
        absmax_over_rms=x.abs().max().item() / max(rms, 1e-30),
        sampled_abs_quantiles=torch.quantile(
            sample.abs(), sample.new_tensor([0.5, 0.99, 0.999])
        )
        .cpu()
        .tolist(),
        quantile_sample_size=sample.numel(),
        relative_quantization_mse={
            f"FP{bits}": (scaled_float(source, bits).float() - x).square().mean().item()
            / max(rms * rms, 1e-30)
            for bits in (4, 8)
        },
    )


@torch.no_grad()
def distributions(model, tokens, device, window=256, batch_size=8):
    rows = []

    def observe(layer, site, x):
        row = dict(layer=layer, site=site, per_stream=describe(x))
        if x.ndim == 4:
            row["stream_mean"] = describe(x.float().mean(-2))
        rows.append(row)

    ids = torch.as_tensor(
        np.array(tokens[: batch_size * window], dtype=np.int64).reshape(
            batch_size, window
        ),
        device=device,
    )
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        reference = model(ids)
    model.observer = observe
    try:
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            actual = model(ids)
        if not torch.equal(reference, actual):
            raise RuntimeError("Residual observation changed the model logits")
    finally:
        model.observer = None
    weights = {
        name: describe(module.weight)
        for name, module in model.lm.named_modules()
        if isinstance(module, (nn.Linear, nn.Embedding))
    }
    rope = model.lm.model.layerrope
    return dict(
        predicted_tokens=batch_size * window,
        residual_scope=f"One fixed evaluation batch of {batch_size} windows of {window} tokens; per-stream and collapsed-mean statistics remain separate",
        residuals=rows,
        weights=weights,
        schedule=(
            {
                name: value.detach().cpu().tolist()
                for name, value in rope.state_dict().items()
            }
            if rope
            else None
        ),
        gains_by_layer=(
            {
                site: [
                    rope(site, torch.tensor(np.log(layer + 1), device=device))
                    .cpu()
                    .tolist()
                    for layer in range(model.depth)
                ]
                for site in rope.sites
            }
            if rope
            else None
        ),
        observer_logits_identical=True,
    )


def linears(model, scope):
    return [
        (name, module)
        for name, module in model.lm.named_modules()
        if isinstance(module, (nn.Linear, nn.Embedding))
        and (scope == "all" or name.startswith("model.layers."))
    ]


@contextlib.contextmanager
def intervention(
    model,
    *,
    weight_bits=None,
    activation_bits=None,
    residual_bits=None,
    scope="blocks",
    integer=False,
):
    originals, handles = [], []

    def quantize(x, bits):
        return fakequant(x, bits, "symmetric") if integer else scaled_float(x, bits)

    try:
        with torch.no_grad():
            if weight_bits is not None:
                for _, module in linears(model, scope):
                    originals.append((module, module.weight.detach().clone()))
                    module.weight.copy_(quantize(module.weight, weight_bits))
        if activation_bits is not None:

            def prehook(module, args):
                return (quantize(args[0], activation_bits),) + args[1:]

            for _, module in linears(model, "blocks"):
                handles.append(module.register_forward_pre_hook(prehook))
        model.residual_bits = residual_bits
        yield
    finally:
        for handle in handles:
            handle.remove()
        model.residual_bits = None
        with torch.no_grad():
            for module, weight in originals:
                module.weight.copy_(weight)


def policies():
    rows = []
    for bits in (4, 8):
        for scope in ("blocks", "all"):
            for integer in (False, True):
                rows.append(
                    dict(
                        name=f"{scope}-{'INT' if integer else 'FP'}{bits}-weights",
                        weight_bits=bits,
                        scope=scope,
                        integer=integer,
                    )
                )
        rows.extend(
            [
                dict(name=f"block-inputs-FP{bits}", activation_bits=bits),
                dict(name=f"residual-streams-FP{bits}", residual_bits=bits),
                dict(
                    name=f"block-W{bits}A{bits}", weight_bits=bits, activation_bits=bits
                ),
            ]
        )
    return rows


@torch.no_grad()
def quantization(model, tokens, device, window=256, batch_size=32):
    before = state_hash(model.state_dict())
    clean_case = cases()[0]

    def evaluate_case(case):
        return evaluate(
            model, tokens, case, device, window=window, batch_size=batch_size
        )

    clean = evaluate_case(clean_case)
    rows = []
    for policy in policies():
        options = {k: v for k, v in policy.items() if k != "name"}
        with intervention(model, **options):
            result = evaluate_case(clean_case)
        result.update(policy, delta_ce=result["ce"] - clean["ce"])
        result["weight_parameter_fraction"] = (
            sum(
                m.weight.numel()
                for _, m in linears(model, policy.get("scope", "blocks"))
            )
            / sum(p.numel() for p in model.parameters())
            if policy.get("weight_bits") is not None
            else 0.0
        )
        if policy.get("weight_bits") is not None:
            selected = linears(model, policy.get("scope", "blocks"))
            quantized = sum(m.weight.numel() for _, m in selected)
            row_count = sum(m.weight.numel() // m.weight.shape[-1] for _, m in selected)
            total = sum(p.numel() for p in model.parameters())
            result["ideal_weight_bytes"] = (
                quantized * policy["weight_bits"] / 8
                + row_count * 4
                + (total - quantized) * 4
            )
        if state_hash(model.state_dict()) != before:
            raise RuntimeError("Weight intervention did not restore the checkpoint")
        rows.append(result)
    cache_rows = []
    cache_cases = cases() + [
        dict(name="fp4_e2m1-K4V4", k_bits=4, v_bits=4, method="fp4_e2m1"),
        dict(name="fp8_e4m3fn-K8V8", k_bits=8, v_bits=8, method="fp8_e4m3fn"),
    ]
    for case in cache_cases:
        result = clean.copy() if case["name"] == "clean" else evaluate_case(case)
        result.update(
            case,
            delta_ce=result["ce"] - clean["ce"],
            storage=bytes_per_token(case, depth=model.depth),
        )
        cache_rows.append(result)
    if state_hash(model.state_dict()) != before:
        raise RuntimeError("Cache evaluation changed the checkpoint")
    return dict(
        clean=clean,
        weight_activation_residual=rows,
        cache=cache_rows,
        weights_unchanged=True,
        formats="E2M1 FP4 and E4M3FN FP8; per-final-axis absmax scales in FP32, nearest-even, saturation, dequantization to incoming dtype; no physical packing",
        activation_scope="Inputs to every backbone-block linear matrix, excluding embedding, output-head, routing, norms and gains",
        residual_scope="After each attention/MLP update; each mHC stream is quantized separately before its next read",
    )

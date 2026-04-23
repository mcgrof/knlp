#!/usr/bin/env python3.12
"""Collect per-tensor and per-channel K absmax scales from a WikiText-2
calibration forward pass.

Outputs JSON files under /workspace/results/kv_scales/<model_slug>.json
with both per-tensor and per-channel scales so downstream scripts can
pick either.

Methodology: the vLLM/TensorRT-LLM standard is per-tensor absmax scale
    scale = absmax(activation) / 448.0

for e4m3 (max representable value 448.0).  Per-channel is the same
but with one scale per channel (head_dim position).  We run a forward
pass over N calibration tokens, register hooks on the K and V
projection outputs, collect absmax per layer, and serialise.

Calibration set: WikiText-2 validation split, first N_TOKENS tokens.
Standard in the quantisation literature.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


FP8_MAX = 448.0  # e4m3 max representable magnitude


def collect_stats(model_id: str, n_tokens: int, device: str = "cuda"):
    """Run a forward pass; return per-layer per-tensor and per-channel
    absmax for K and V activations."""
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16,
        attn_implementation="eager",
    ).to(device).eval()

    # Calibration data: WikiText-2 validation, first n_tokens
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join([r for r in ds["text"] if r.strip()])
    enc = tok(text, return_tensors="pt")
    input_ids = enc.input_ids[:, :n_tokens].to(device)
    print(f"[calib] {model_id}: collecting over {input_ids.shape[1]} tokens")

    # Stats collectors: one list per layer
    n_layers = model.config.num_hidden_layers
    k_abs_pertensor = [0.0] * n_layers
    v_abs_pertensor = [0.0] * n_layers
    k_abs_perchannel = [None] * n_layers  # filled on first call
    v_abs_perchannel = [None] * n_layers

    hooks = []

    def make_hook(layer_idx, kind):
        def hook(module, inputs, outputs):
            # outputs: [B, T, hidden] — from a Linear. The hidden dim is
            # (n_kv_heads * head_dim) for K/V projections.
            x = outputs.detach()
            if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
                x = x.float()
            flat = x.view(-1, x.shape[-1])
            pt = flat.abs().max().item()
            pc = flat.abs().amax(dim=0).cpu()  # [hidden]
            if kind == "k":
                if pt > k_abs_pertensor[layer_idx]:
                    k_abs_pertensor[layer_idx] = pt
                prev = k_abs_perchannel[layer_idx]
                k_abs_perchannel[layer_idx] = (
                    pc if prev is None else torch.maximum(prev, pc)
                )
            else:
                if pt > v_abs_pertensor[layer_idx]:
                    v_abs_pertensor[layer_idx] = pt
                prev = v_abs_perchannel[layer_idx]
                v_abs_perchannel[layer_idx] = (
                    pc if prev is None else torch.maximum(prev, pc)
                )
        return hook

    # Register K/V projection hooks on every layer.  Attribute paths
    # differ across architectures; we probe a few.
    for i, layer in enumerate(_iter_attention_layers(model)):
        k_proj, v_proj = _find_kv_projections(layer)
        hooks.append(k_proj.register_forward_hook(make_hook(i, "k")))
        hooks.append(v_proj.register_forward_hook(make_hook(i, "v")))

    with torch.no_grad():
        # Slide a 2048-token window (matches WikiText eval convention)
        stride = 2048
        for start in range(0, input_ids.shape[1], stride):
            chunk = input_ids[:, start : start + stride]
            if chunk.shape[1] < 8:
                break
            _ = model(chunk)

    for h in hooks:
        h.remove()

    return {
        "model_id": model_id,
        "n_layers": n_layers,
        "n_tokens": n_tokens,
        "k_pertensor_absmax": k_abs_pertensor,
        "v_pertensor_absmax": v_abs_pertensor,
        "k_perchannel_absmax": [x.tolist() for x in k_abs_perchannel],
        "v_perchannel_absmax": [x.tolist() for x in v_abs_perchannel],
    }


def compute_scales(stats: dict) -> dict:
    """Convert absmax to fp8 e4m3 scales: scale = absmax / 448."""
    out = {
        "model_id": stats["model_id"],
        "n_layers": stats["n_layers"],
        "n_tokens": stats["n_tokens"],
        "fp8_max": FP8_MAX,
        "k_pertensor_scale": [x / FP8_MAX for x in stats["k_pertensor_absmax"]],
        "v_pertensor_scale": [x / FP8_MAX for x in stats["v_pertensor_absmax"]],
        "k_perchannel_scale": [
            [v / FP8_MAX for v in row] for row in stats["k_perchannel_absmax"]
        ],
        "v_perchannel_scale": [
            [v / FP8_MAX for v in row] for row in stats["v_perchannel_absmax"]
        ],
    }
    return out


def _iter_attention_layers(model):
    """Yield each attention-containing transformer block."""
    # HF transformers stores blocks under model.model.layers for most
    # modern decoder-only architectures (LLaMA / Mistral / Qwen / DS).
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        yield from model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        yield from model.transformer.h
    else:
        raise RuntimeError("could not find transformer blocks on model")


def _find_kv_projections(layer):
    """Return (k_proj, v_proj) Linear modules of a transformer block."""
    # LLaMA / Mistral / Qwen / DS-R1 shape: layer.self_attn.{k_proj, v_proj}
    attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
    if attn is None:
        raise RuntimeError("no self_attn on transformer block")
    for kname in ("k_proj", "key_proj", "Wk"):
        if hasattr(attn, kname):
            k = getattr(attn, kname)
            break
    else:
        raise RuntimeError("no k_proj on attention module")
    for vname in ("v_proj", "value_proj", "Wv"):
        if hasattr(attn, vname):
            v = getattr(attn, vname)
            break
    else:
        raise RuntimeError("no v_proj on attention module")
    return k, v


def slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-tokens", type=int, default=2048)
    ap.add_argument("--out-dir", default="/workspace/results/kv_scales")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out_dir) / f"{slug(args.model)}.json"

    t0 = time.time()
    stats = collect_stats(args.model, args.n_tokens)
    scales = compute_scales(stats)
    with open(out_path, "w") as f:
        json.dump(scales, f, indent=2)
    dt = time.time() - t0
    print(f"[calib] wrote {out_path} in {dt:.1f}s")


if __name__ == "__main__":
    main()

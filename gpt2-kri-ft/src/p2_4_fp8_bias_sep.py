#!/usr/bin/env python3
"""P2.4 — FP8 bias-content separation: does the K-bias CAUSE the symmetric-FP8 failure?

The FP8 atlas showed symmetric per-tensor FP8-K quantization collapses on Qwen2.5 (K-bias
magnitude predicts fragility). The mechanism: the huge-magnitude rotated-bias term R_j b_K
dominates the per-tensor amax, so the quantizer scale is set by the bias and the small
content channels get almost no bits -> garbled attention. Causal test: instead of
quantizing the full post-RoPE key R_j(c_k + b_K), quantize only the CONTENT R_j c_k and
add the rotated bias R_j b_K back in high precision. If that fixes the FP8 attention while
leaving BF16 behaviour identical, b_K is causally responsible for the FP8 failure.

Arms (per layer, sampled query positions):
  bf16          : reference attention (no quant).
  fp8_full      : symmetric per-tensor e4m3 fake-quant of the full post-RoPE key.
  fp8_content   : quantize only content (key - R_j b_K), add R_j b_K back in bf16.
  fp8_perchan   : per-(kv-head,channel) e4m3 of the full key (atlas's known partial fix).
Metrics: attention-logit KL(bf16 || arm) at sampled queries, key reconstruction rel-err,
quantizer saturation (amax / rms of what's quantized), top-block flip rate. Run on
Qwen2.5-0.5B/1.5B (big b_K) vs Qwen3-0.6B / SmolLM2-1.7B (no b_K, FP8 should be fine).
Fake-quant only, no serving stack; free W7900, no_grad.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch

from niah_task import load_filler_sentences
from diag_perhead_oracle import capture_q
from niah_evict_perhead import get_keys

FP8_MAX = 448.0


def fq_pertensor(x):
    scale = x.abs().amax().clamp_min(1e-8) / FP8_MAX
    return (x / scale).to(torch.float8_e4m3fn).float() * scale


def fq_perchan(x):  # x [Hkv, T, D] -> per (kv-head, channel) scale
    scale = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / FP8_MAX
    return (x / scale).to(torch.float8_e4m3fn).float() * scale


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def kl(p, q, eps=1e-9):
    return (p.clamp_min(eps) * (p.clamp_min(eps).log() - q.clamp_min(eps).log())).sum(-1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--eval", type=int, default=8)
    ap.add_argument("--n-probe", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype)
    ).to(device)
    model.eval()
    torch.set_grad_enabled(False)
    Hq = model.config.num_attention_heads
    Hkv = getattr(model.config, "num_key_value_heads", Hq)
    group = Hq // Hkv
    nL = model.config.num_hidden_layers
    Dh = getattr(model.config, "head_dim", model.config.hidden_size // Hq)
    sents = load_filler_sentences(args.seed)
    rng = random.Random(args.seed)

    bK = []
    has_bias = False
    for li in range(nL):
        b = getattr(model.model.layers[li].self_attn.k_proj, "bias", None)
        if b is None:
            bK.append(None)
        else:
            bK.append(b.detach().float().view(Hkv, Dh).to(device))
            has_bias = True

    acc = {}

    def push(k, v):
        acc.setdefault(k, []).append(float(v))

    n_used = 0
    for ei in range(args.eval):
        text, need = "", args.length * 5
        while len(text) < need:
            text += sents[rng.randrange(len(sents))] + " "
        ids = tok(text, add_special_tokens=False)["input_ids"][: args.length]
        if len(ids) < 256:
            continue
        ids_t = torch.tensor(ids)
        T = len(ids) - 1
        past = model(ids_t[:-1].unsqueeze(0).to(device), use_cache=True).past_key_values
        keys = get_keys(past)
        probes = sorted(set(int(x) for x in torch.linspace(T // 8, T - 1, args.n_probe).tolist()))
        qs, _ = capture_q(model, ids_t[:-1], probes, device)
        pos_ids = torch.arange(T, device=device).unsqueeze(0)
        dummy = torch.zeros(1, T, Dh, device=device, dtype=next(model.parameters()).dtype)
        cos, sin = model.model.rotary_emb(dummy, pos_ids)
        cos, sin = cos[0].float(), sin[0].float()
        n_used += 1

        for li in range(nL):
            kfull = keys[li][0].float()  # [Hkv, T, D]
            if bK[li] is not None:
                rb = bK[li].unsqueeze(1) * cos.unsqueeze(0) + rotate_half(bK[li].unsqueeze(1)) * sin.unsqueeze(0)
                kcont = kfull - rb
            else:
                rb = None
                kcont = kfull
            # build the four key variants
            k_bf16 = kfull
            k_fp8full = fq_pertensor(kfull)
            k_fp8perc = fq_perchan(kfull)
            k_fp8cont = (fq_pertensor(kcont) + rb) if rb is not None else fq_pertensor(kfull)

            # saturation / dynamic range of what's quantized
            push("amax_over_rms_full", (kfull.abs().amax() / kfull.pow(2).mean().sqrt()).item())
            push("amax_over_rms_content", (kcont.abs().amax() / kcont.pow(2).mean().sqrt()).item())
            push("recon_err_fp8full", ((k_fp8full - kfull).norm() / kfull.norm()).item())
            push("recon_err_fp8content", ((k_fp8cont - kfull).norm() / kfull.norm()).item())

            ql = qs[li].float()  # [Hq, P, D]
            for pi, ppos in enumerate(probes):
                q = ql[:, pi, :]  # [Hq, D]
                def attn(kv):
                    ke = kv.repeat_interleave(group, dim=0)[:, : ppos + 1, :]
                    return (torch.einsum("hd,htd->ht", q, ke) / math.sqrt(Dh)).softmax(-1)
                a_ref = attn(k_bf16)
                for name, kv in (("fp8full", k_fp8full), ("fp8content", k_fp8cont), ("fp8perchan", k_fp8perc)):
                    a = attn(kv)
                    push(f"attn_kl_{name}", kl(a_ref, a).mean().item())
                    # top-block flip: does the argmax key change? (proxy for ranking damage)
                    push(f"argmax_flip_{name}", (a_ref.argmax(-1) != a.argmax(-1)).float().mean().item())
            torch.cuda.empty_cache()
        print(f"  [{n_used}/{args.eval}] T={T}", flush=True)

    def summ(xs):
        if not xs:
            return None
        t = torch.tensor(xs, dtype=torch.float)
        return {"n": len(xs), "mean": round(t.mean().item(), 5), "median": round(t.median().item(), 5)}

    m = {k: (summ(v)["mean"] if summ(v) else None) for k, v in acc.items()}
    out = {
        "model": args.model, "Hq": Hq, "Hkv": Hkv, "group": group, "nL": nL,
        "has_k_bias": has_bias, "length": args.length, "n_used": n_used,
        "metrics": {k: summ(v) for k, v in sorted(acc.items())},
        "fp8_read": {
            "amax_over_rms_full": m.get("amax_over_rms_full"),
            "amax_over_rms_content": m.get("amax_over_rms_content"),
            "attn_kl_fp8full": m.get("attn_kl_fp8full"),
            "attn_kl_fp8content": m.get("attn_kl_fp8content"),
            "attn_kl_fp8perchan": m.get("attn_kl_fp8perchan"),
            "kl_rescue_ratio": (
                round(m["attn_kl_fp8content"] / m["attn_kl_fp8full"], 4)
                if m.get("attn_kl_fp8full") and m["attn_kl_fp8full"] > 1e-9 else None
            ),
            "argmax_flip_fp8full": m.get("argmax_flip_fp8full"),
            "argmax_flip_fp8content": m.get("argmax_flip_fp8content"),
            "note": "b_K causes the FP8 failure if fp8_full has high attn KL / flip and "
            "fp8_content (bias separated) collapses it (kl_rescue_ratio << 1); no-bias "
            "models should already have low fp8_full KL",
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["fp8_read"], indent=2))
    print(f"P2_4_FP8_DONE {args.out}")


if __name__ == "__main__":
    main()

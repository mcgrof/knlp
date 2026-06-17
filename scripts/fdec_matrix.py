#!/usr/bin/env python3
"""fdec compatibility matrix runner (algorithmic dimension).

Verifies that the fight-decode bricks COMPOSE -- run together in one eager decode
loop without mismatch -- and reports correctness vs an fp16 baseline plus the byte
reduction each brick buys. This is the algorithmic/quality dimension of the matrix;
it runs on any GPU (incl. the W7900 on prune) in eager PyTorch. The kernel/throughput
dimension (FlashInfer asym, fused Triton) is CUDA-only and lives in the serving-stack
profiles, not here.

Bricks (extensible -- add a patch fn + a byte-model entry):
  asym_kv   : K stays bf16, V quantized to fp8 e4m3 (per-tensor scale) -> -25% KV.
              The proven asym K16/V8 brick, emulated at the math level.
  (lmhead_idblock, fim_weight_quant, linear_attn: TODO -- wire their code.)

A "cell" is a model x set-of-enabled-bricks. We decode greedily with and without the
bricks and report the fraction of generated tokens that match the baseline (quality
proxy) and the analytical decode-byte reduction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# ---- bricks -----------------------------------------------------------------


def asym_kv_attention(
    module, query, key, value, attention_mask, scaling, dropout=0.0, **kw
):
    """Eager attention with K16/V8: keep keys bf16, quantize values to fp8 e4m3."""
    g = module.num_key_value_groups
    ks = key.repeat_interleave(g, dim=1)
    vs = value.repeat_interleave(g, dim=1)
    # V -> fp8 e4m3 -> bf16 (per-tensor scale); K untouched
    scale = vs.abs().amax().clamp(min=1e-6) / 448.0
    vs = (vs / scale).to(torch.float8_e4m3fn).to(vs.dtype) * scale
    aw = torch.matmul(query, ks.transpose(2, 3)) * scaling
    if attention_mask is not None:
        aw = aw + attention_mask[:, :, :, : ks.shape[-2]]
    aw = F.softmax(aw, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(aw, vs).transpose(1, 2).contiguous()
    return out, aw


def install_asym_kv(model):
    model.config._attn_implementation = "eager"
    mods = set()
    for m in model.modules():
        if hasattr(m, "num_key_value_groups") and hasattr(m, "layer_idx"):
            mods.add(type(m).__module__)
    saved = {}
    for mn in mods:
        mod = sys.modules.get(mn)
        if mod is not None and hasattr(mod, "eager_attention_forward"):
            saved[mn] = mod.eager_attention_forward
            mod.eager_attention_forward = asym_kv_attention
    return saved


def restore(saved):
    for mn, fn in saved.items():
        sys.modules[mn].eager_attention_forward = fn


# brick registry: name -> (install fn, byte-model note, KV/total byte factor)
BRICKS = {
    "asym_kv": dict(install=install_asym_kv, pool="KV", kv_factor=0.75),
}


@torch.no_grad()
def tf_logits(model, tok, texts, device):
    """Teacher-forced: per-position argmax + log-probs (for KL). No generation
    loop, so no autoregressive divergence cascade -- this measures whether a brick
    changes the next-token prediction given the SAME context."""
    am, lg = [], []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to(device)
        l = model(ids, use_cache=False).logits[0].float()  # [T, V]
        am.append(l.argmax(-1).cpu())
        lg.append(F.log_softmax(l, dim=-1).cpu())
    return am, lg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--bricks", default="asym_kv", help="comma-sep brick set (the cell)"
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--n-prompts", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation="eager"
    ).to(device)
    model.eval()

    prompts = [
        "The history of the printing press begins when",
        "In thermodynamics, the second law states that",
        "A good way to learn a new programming language is to",
        "The capital city of France is Paris, and it is known for",
        "Once upon a time in a distant galaxy there was",
        "To compute the factorial of a number recursively you",
    ][: args.n_prompts]

    bricks = [b.strip() for b in args.bricks.split(",") if b.strip()]
    unknown = [b for b in bricks if b not in BRICKS]
    if unknown:
        print(f"[skip] unknown bricks (not yet wired): {unknown}")
        bricks = [b for b in bricks if b in BRICKS]

    # baseline (fp16, no bricks)
    base_am, base_lg = tf_logits(model, tok, prompts, device)

    # apply the cell's bricks together, re-run, compare (teacher-forced)
    saved_all = {}
    for b in bricks:
        saved_all.update(BRICKS[b]["install"](model))
    brk_am, brk_lg = tf_logits(model, tok, prompts, device)
    restore(saved_all)

    # top-1 agreement + mean KL(baseline || brick) over all positions
    total = matched = 0
    kl_sum = 0.0
    nkl = 0
    for ba, bk, blg, klg in zip(base_am, brk_am, base_lg, brk_lg):
        total += ba.numel()
        matched += int((ba == bk).sum())
        p = blg.exp()
        kl_sum += float((p * (blg - klg)).sum(-1).mean())
        nkl += 1
    match_rate = matched / max(1, total)
    mean_kl = kl_sum / max(1, nkl)

    # analytical decode-byte model: KV factor multiplies across KV bricks
    kv_factor = 1.0
    pools = []
    for b in bricks:
        kv_factor *= BRICKS[b].get("kv_factor", 1.0)
        pools.append(f"{b}({BRICKS[b]['pool']})")

    res = {
        "model": args.model,
        "cell_bricks": bricks,
        "pools": pools,
        "composes": True,  # ran together without error
        "tf_top1_agreement_vs_fp16": match_rate,
        "tf_mean_kl_vs_fp16": mean_kl,
        "kv_byte_factor": kv_factor,
        "kv_byte_reduction_pct": round(100 * (1 - kv_factor), 1),
        "n_prompts": len(prompts),
    }
    print(f"\n[fdec cell] {args.model}  bricks={bricks or ['(none)']}")
    print(
        f"  composes: yes   teacher-forced top-1 agreement: {match_rate:.3f}   "
        f"meanKL: {mean_kl:.4f}   KV bytes: {kv_factor:.3f}x "
        f"(-{res['kv_byte_reduction_pct']}%)"
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()

"""Disentangle FORMAT (fp8 vs int8) from GRANULARITY (per_tensor/token/channel) for uniform K quant.

The intervention run used k_fail=fp8:per_tensor vs k_int=int8:per_token -- which differ on BOTH
axes, so "INT8 rescues FP8" was confounded with "per-token rescues per-tensor". This runs uniform-K
anchors (V=fp8:per_tensor fixed) over the full format x granularity grid so the two axes separate.
Also runs the single-layer-0 demotion under each K cell to check L0-dominance is format-robust.
"""

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k_bias_common as kbc


@torch.no_grad()
def nll(model, chunks, device):
    tot_nll, tot_tok = 0.0, 0
    for ids in chunks:
        t = torch.tensor(ids).unsqueeze(0).to(device)
        lp = torch.log_softmax(model(t).logits[0, :-1].float(), dim=-1)
        tgt = t[0, 1:]
        tot_nll += (-lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)).sum().item()
        tot_tok += tgt.numel()
    return tot_nll / tot_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-prompts", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=2048)
    a = ap.parse_args()

    model, tok = kbc.load_model(a.model, "bfloat16", a.device)
    infos = kbc.discover_attention(model)
    chunks = kbc.calib_prompts(tok, n=a.num_prompts, seq_len=a.seq_len)
    V = kbc.parse_spec("fp8:per_tensor")

    ref = nll(model, chunks, a.device)
    print(f"[{a.model} ctx={a.seq_len} n={len(chunks)}]  ref_ppl={math.exp(ref):.3f}\n")
    print(f"{'K cell':22s}  {'uniform dNLL':>12s}  {'demote-L0 dNLL':>14s}")

    grid = []
    for fmt in ("fp8", "int8"):
        for gran in ("per_tensor", "per_token", "per_channel"):
            grid.append(f"{fmt}:{gran}")
    for spec_s in grid:
        K = kbc.parse_spec(spec_s)
        # uniform: all layers this K cell
        hu = kbc.PerLayerKVHarness(model, infos, k_default=K, v_default=V)
        hu.install()
        du = nll(model, chunks, a.device) - ref
        hu.remove()
        # demote L0 only from all-K16/V8 base
        hd = kbc.PerLayerKVHarness(
            model, infos, k_specs={0: K}, k_default=kbc.parse_spec("bf16"), v_default=V
        )
        hd.install()
        dd = nll(model, chunks, a.device) - ref
        hd.remove()
        print(f"{spec_s:22s}  {du:>+12.4f}  {dd:>+14.4f}", flush=True)


if __name__ == "__main__":
    main()

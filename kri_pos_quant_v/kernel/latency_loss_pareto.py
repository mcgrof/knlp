#!/usr/bin/env python3
"""Latency vs loss Pareto for low-rank V: sweep the rank (which sets BOTH the
decode-attention V-read latency AND the model loss) and join the two halves.

Latency: the low-rank-V decode-attention kernel (reads K[T,dh] + proj[T,r] instead
of K + V[T,dh]) vs the fusion-matched dense flash-decode, at a realistic decode
shape, per rank. Loss: held-out delta-NLL per rank from the PEFT sweep jsons
(training-free and compression-aware LoRA, uniform rank on all layers, Qwen-7B).
The Pareto frontier is decode-attention speedup vs model loss.

Scope note: latency is the ATTENTION V-read component; full-model decode impact is
diluted (V is half the KV cache, KV a fraction of total decode traffic -- Amdahl).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch

from vlowrank_kernel import dense_decode_triton, lowrank_decode


def cuda_p50(fn, warmup=15, iters=120):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--peft-dir", required=True)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--ranks", default="16,24,32,48,64,96,128")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = torch.device(args.device)
    dh, H, T = args.head_dim, args.heads, args.context
    ranks = [int(x) for x in args.ranks.split(",")]

    # loss per rank from the PEFT sweep jsons (uniform, held-out val ppl)
    loss = {}
    for f in Path(args.peft_dir).glob("peft_rank*.json"):
        d = json.load(open(f))
        r = d["rank"]
        b = d["ppl"]["base"]
        loss[r] = {
            "trainfree_dnll": math.log(d["ppl"]["trainfree_rankk"]) - math.log(b),
            "peft_dnll": math.log(d["ppl"]["lora_rankk"]) - math.log(b),
        }

    # dense decode-attention latency (the baseline; rank-independent)
    Q = torch.randn(H, 1, dh, device=dev, dtype=torch.bfloat16)
    K = torch.randn(H, T, dh, device=dev, dtype=torch.bfloat16)
    V = torch.randn(H, T, dh, device=dev, dtype=torch.bfloat16)
    t_dense = cuda_p50(lambda: dense_decode_triton(Q, K, V))

    rows = []
    for r in ranks:
        proj = torch.randn(H, T, r, device=dev, dtype=torch.bfloat16)
        Bb = torch.randn(H, dh, r, device=dev, dtype=torch.float32)
        t_low = cuda_p50(lambda: lowrank_decode(Q, K, proj, Bb))
        v_read = (dh + r) / (2.0 * dh)        # total KV traffic ratio (bf16 coeff)
        rows.append({
            "rank": r,
            "v_read_ratio": v_read,
            "attn_speedup": t_dense / t_low,
            "lowrank_ms": t_low,
            "trainfree_dnll": loss.get(r, {}).get("trainfree_dnll"),
            "peft_dnll": loss.get(r, {}).get("peft_dnll"),
        })

    print(f"decode-attn latency/loss Pareto  (H={H} T={T} dh={dh}; dense {t_dense:.3f} ms)")
    print(f"{'rank':>4} {'KVread':>7} {'attn_spdup':>10} {'tf_dNLL':>9} {'peft_dNLL':>10}")
    for x in rows:
        tf = "  n/a" if x["trainfree_dnll"] is None else f"{x['trainfree_dnll']:+.3f}"
        pe = "  n/a" if x["peft_dnll"] is None else f"{x['peft_dnll']:+.3f}"
        print(f"{x['rank']:>4} {x['v_read_ratio']:>6.2f}  {x['attn_speedup']:>9.2f}x "
              f"{tf:>9} {pe:>10}")
    Path(args.out).write_text(json.dumps(
        {"H": H, "T": T, "head_dim": dh, "dense_ms": t_dense, "rows": rows}, indent=2))
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()

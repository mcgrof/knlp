#!/usr/bin/env python3
"""Tune the fused MoE kernel's tile sizes for EMO's expert shapes (GPU).

vLLM ships tuned tile tables only for some data-center GPUs; on others it
falls back to a default table. This script times the fused routed-expert
computation (both kernels plus the activation and the slot sum) for every
tile configuration in a fixed search space, at each token count and expert
count, and writes the fastest configuration per point in the format that
``fused_moe.ConfigTable`` reads. Routing is random top-k over the routed
experts with fixed seeds, as in vLLM's own tuning script. Every candidate
output must match the default configuration's output to a relative L2
error of 1e-2, or the candidate is rejected.

Usage:
    tune_fused_moe.py OUT.json [--experts 127 32] [--tokens 1 16 ...]
"""

import argparse
import itertools
import os
import sys
import time

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import runtime_identity, write_json  # noqa
from fused_moe import VLLM_SOURCE, default_config, fused_experts  # noqa

SPACE = {
    "BLOCK_SIZE_M": [16, 32, 64, 128],
    "BLOCK_SIZE_N": [32, 64, 128, 256],
    "BLOCK_SIZE_K": [32, 64, 128],
    "GROUP_SIZE_M": [1, 8],
    "num_warps": [4, 8],
    "num_stages": [2],
}


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--experts", type=int, nargs="+", default=[127, 32])
    ap.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 16, 64, 128, 512, 1024, 2048]
    )
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--inter", type=int, default=1024)
    ap.add_argument("--topk", type=int, default=7)
    args = ap.parse_args()
    H, I, k = args.hidden, args.inter, args.topk
    keys = list(SPACE)
    cands = [dict(zip(keys, v)) for v in itertools.product(*SPACE.values())]
    out = {
        "runtime": runtime_identity(),
        "kernel_source": VLLM_SOURCE,
        "space": SPACE,
        "shape": {"hidden": H, "inter": I, "topk": k},
        "configs": {},
        "points": [],
    }
    t_start = time.time()
    for E in args.experts:
        g = torch.Generator(device="cuda").manual_seed(E)
        w13 = (torch.randn(E, 2 * I, H, device="cuda", generator=g) * 0.02).bfloat16()
        w2 = (torch.randn(E, H, I, device="cuda", generator=g) * 0.02).bfloat16()
        out["configs"][str(E)] = {}
        for M in args.tokens:
            x = torch.randn(M, H, device="cuda", generator=g).bfloat16()
            logits = torch.randn(M, E, device="cuda", generator=g)
            w, ids = torch.topk(torch.softmax(logits, -1), k, dim=-1)
            w = w.bfloat16().float().contiguous()
            dcfg = default_config(M, E)
            ref = fused_experts(x, w13, w2, w, ids, dcfg)

            def bench(cfg):
                fn = lambda: fused_experts(x, w13, w2, w, ids, cfg)  # noqa
                return triton.testing.do_bench(
                    fn, warmup=10, rep=40, return_mode="median"
                )

            t_def = bench(dcfg)
            best, best_t, n_ok, n_bad = dcfg, t_def, 0, 0
            for cfg in cands:
                try:
                    y = fused_experts(x, w13, w2, w, ids, cfg)
                    torch.cuda.synchronize()
                except Exception:
                    n_bad += 1
                    continue
                if rel(y, ref) > 1e-2:
                    n_bad += 1
                    continue
                t = bench(cfg)
                n_ok += 1
                if t < best_t:
                    best, best_t = cfg, t
            out["configs"][str(E)][str(M)] = best
            pt = {
                "experts": E,
                "tokens": M,
                "default_ms": t_def,
                "best_ms": best_t,
                "speedup": t_def / best_t,
                "best": best,
                "default": dcfg,
                "candidates_ok": n_ok,
                "candidates_rejected": n_bad,
            }
            out["points"].append(pt)
            print(
                f"E={E} M={M}: default {t_def:.3f} ms, best {best_t:.3f} ms "
                f"({n_ok} ok, {n_bad} rejected)",
                flush=True,
            )
            write_json(args.out, out)
        del w13, w2
        torch.cuda.empty_cache()
    out["seconds"] = time.time() - t_start
    write_json(args.out, out)


if __name__ == "__main__":
    main()

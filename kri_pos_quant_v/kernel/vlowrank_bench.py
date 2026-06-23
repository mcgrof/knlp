#!/usr/bin/env python3
"""Validate + benchmark the low-rank-V decode kernel against dense attention.

Loads the V artifact (gen_v_artifact.py), then:
  1. correctness: Triton low-rank output == torch low-rank reference;
  2. fidelity: low-rank output vs DENSE attention output -- the error is the
     rank-r V reconstruction error, the V6 claim (rank-32 ~ retrieval-lossless);
  3. latency: dense decode attention vs the low-rank kernel (+ lift);
  4. traffic: V-read and total-KV bytes, dense vs low-rank.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from vlowrank_kernel import (
    dense_decode,
    dense_decode_triton,
    lowrank_decode,
    lowrank_decode_ref,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = torch.device(args.device)
    A = Path(args.artifact)
    meta = json.loads((A / "meta.json").read_text())
    K = torch.load(A / "K.pt").to(dev)
    V = torch.load(A / "V.pt").to(dev)
    Bbasis = torch.load(A / "Bbasis.pt").to(dev)
    proj = torch.load(A / "proj.pt").to(dev)
    Q = torch.load(A / "Q.pt").to(dev)
    Hkv, T, dh = V.shape
    r = proj.shape[2]
    nq = Q.shape[1]
    print(f"Hkv={Hkv} T={T} dh={dh} r={r} nq={nq}", flush=True)

    # --- 1. correctness: kernel == ref ---
    o_k = lowrank_decode(Q, K, proj, Bbasis)
    o_r = lowrank_decode_ref(Q, K, proj, Bbasis)
    kern_vs_ref = (o_k - o_r).norm() / o_r.norm()
    # --- 2. fidelity: low-rank vs dense ---
    o_d = dense_decode(Q, K, V)
    fid_err = (o_k - o_d).norm() / o_d.norm()
    print(
        f"[correctness] kernel-vs-ref rel={kern_vs_ref:.3e}  "
        f"[fidelity] lowrank-vs-dense rel-err={fid_err:.4f} "
        f"(== rank-{r} V recon {meta.get('rank_r_V_recon_rel_error', float('nan')):.4f})",
        flush=True,
    )

    # --- 2b. SELECTIVE fidelity (the V6 scheme): keep high-residual tokens FULL,
    # compress only the low-residual ones. delta_i = ||v_i - B B^T v_i|| per head;
    # compress the lowest-delta fraction f, keep the rest full. Output error is
    # bounded by sum_{compressed} a_i delta_i. This is what makes rank-32
    # retrieval-lossless, vs the all-compressed 0.61 above.
    recon = torch.einsum("htr,hdr->htd", proj.float(), Bbasis)  # [H,T,dh]
    delta = (V.float() - recon).norm(dim=-1)  # [H,T] per-token residual
    s_all = torch.einsum("hqd,htd->hqt", Q.float(), K.float()) / (dh ** 0.5)
    a_all = torch.softmax(s_all, dim=-1)  # [H,nq,T]
    o_dense_ref = torch.einsum("hqt,htd->hqd", a_all, V.float())
    sel = {}
    for f in (0.90, 0.95, 0.99, 1.0):
        # per head, lowest-delta f*T tokens -> compressed; rest full
        nkeep_full = int(round((1.0 - f) * T))
        err_heads = []
        for h in range(Hkv):
            order = torch.argsort(delta[h])  # ascending
            comp_mask = torch.ones(T, dtype=torch.bool, device=dev)
            if nkeep_full > 0:
                comp_mask[order[-nkeep_full:]] = False  # keep highest-delta full
            vh = V[h].float()
            vmix = torch.where(comp_mask.unsqueeze(1), recon[h], vh)  # [T,dh]
            o_sel = a_all[h] @ vmix  # [nq,dh]
            err_heads.append((o_sel - o_dense_ref[h]).norm() / o_dense_ref[h].norm())
        sel[f] = float(torch.tensor(err_heads).mean())
    print(
        "[selective fidelity] compress-frac -> attn-output rel-err: "
        + "  ".join(f"{int(f*100)}%:{e:.4f}" for f, e in sel.items()),
        flush=True,
    )

    # --- 2c. rank sweep: how compressible is THIS layer's V at the attention-
    # output level? SVD V per head at increasing rank, measure attn-output error.
    rank_sweep = {}
    for rr in (32, 64, 96, 128):
        errs = []
        for h in range(Hkv):
            Vh = V[h].float()
            U, Sv, Wt = torch.linalg.svd(Vh, full_matrices=False)
            basis = Wt[:rr].t()
            recon_h = (Vh @ basis) @ basis.t()
            o_sel = a_all[h] @ recon_h
            errs.append((o_sel - o_dense_ref[h]).norm() / o_dense_ref[h].norm())
        rank_sweep[rr] = float(torch.tensor(errs).mean())
    print(
        "[rank sweep, all-compressed] rank -> attn-output rel-err: "
        + "  ".join(f"r{rr}:{e:.4f}" for rr, e in rank_sweep.items()),
        flush=True,
    )

    # --- 3. latency ---
    def timeit(fn):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.iters):
            fn()
        torch.cuda.synchronize()
        return (time.time() - t0) / args.iters * 1e3

    # fusion-matched: dense Triton flash-decode vs low-rank Triton flash-decode
    o_dt = dense_decode_triton(Q, K, V)
    dt_vs_dense = (o_dt - o_d).norm() / o_d.norm()
    t_dense = timeit(lambda: dense_decode_triton(Q, K, V))
    t_low = timeit(lambda: lowrank_decode(Q, K, proj, Bbasis))
    print(
        f"[latency ms] dense_triton={t_dense:.3f} (vs einsum rel {dt_vs_dense:.1e}) "
        f"lowrank_kernel={t_low:.3f}  speedup={t_dense/t_low:.2f}x",
        flush=True,
    )

    # --- 4. traffic (per query, per head; bytes) ---
    # dense: read K + V, both bf16; lowrank: read K bf16 + proj (bf16 or int8)
    K_bytes = T * dh * 2
    V_bytes = T * dh * 2
    proj_bf16 = T * r * 2
    proj_int8 = T * r * 1
    dense_kv = K_bytes + V_bytes
    low_kv_bf16 = K_bytes + proj_bf16
    low_kv_int8 = K_bytes + proj_int8
    print(
        f"[traffic/query/head] dense KV={dense_kv/1e3:.0f}KB  "
        f"V-read {V_bytes/1e3:.0f}KB->proj {proj_bf16/1e3:.0f}KB ({dh/r:.0f}x)  "
        f"total-KV lowrank(bf16 coeff)={low_kv_bf16/dense_kv*100:.1f}%  "
        f"lowrank(int8 coeff)={low_kv_int8/dense_kv*100:.1f}%",
        flush=True,
    )

    result = {
        "meta": meta,
        "kernel_vs_ref_rel": float(kern_vs_ref),
        "lowrank_vs_dense_rel_err_all_compressed": float(fid_err),
        "selective_fidelity": {str(f): e for f, e in sel.items()},
        "rank_sweep_attn_err": {str(rr): e for rr, e in rank_sweep.items()},
        "latency_ms": {
            "dense": t_dense,
            "lowrank_kernel": t_low,
            "speedup": t_dense / t_low,
        },
        "traffic": {
            "V_read_reduction_x": dh / r,
            "total_kv_ratio_bf16_coeff": low_kv_bf16 / dense_kv,
            "total_kv_ratio_int8_coeff": low_kv_int8 / dense_kv,
        },
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[wrote] {args.out}", flush=True)


if __name__ == "__main__":
    main()

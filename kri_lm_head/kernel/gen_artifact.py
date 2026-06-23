#!/usr/bin/env python3
"""Generate the shadow-bound certified-decode artifact for kernel development.

Runs the qwen-7b LM head through the shadow-bound precompute ONCE and saves every
tensor the deployment kernel needs, plus a dense-argmax ground truth so the kernel
path can be checked lossless without recomputing the model. Targets the headline
op point: hidden_pca basis, r=1280, int8 shadow -> 24.6% bytes, argmax_match 1.0.

Saved (torch .pt):
  aq      int8   [V, r]   per-column-quantized shadow coefficients (codes)
  scale   fp32   [r]      per-column dequant scale (a ~= aq * scale)
  delta   fp32   [V]      residual norm ||w_v - B B^T w_v||
  B       fp32   [d, r]   orthonormal basis
  cof     int32  [V]      idblock slab id per token
  Wq      int8   [V, d]   per-row int8 of W_U (for the exact-fetch kernel) + Wscale
  Wscale  fp32   [V]
  H       fp16   [N, d]   captured hidden states (decode positions)
  gt      int64  [N]      dense argmax per position (ground truth)
  meta    dict            V,d,r,C,bits,aq_err_norm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from h6_oracle_scan import build_idblock
from predictor_baseline import load_split, capture
from shadow_bound_replay import build_basis, shadow_precompute


def quantize_cols_codes(a, bits):
    """Per-column symmetric quant -> (int codes, scale, err_norm)."""
    qmax = 2 ** (bits - 1) - 1
    scale = (a.abs().amax(0) / qmax).clamp(min=1e-12)  # [r]
    codes = torch.round(a / scale).clamp(-qmax, qmax).to(torch.int8)
    err_norm = (scale / 2).norm().item()
    return codes, scale, err_norm


def quantize_rows_int8(W, vchunk=16384):
    """Per-row symmetric int8 of W_U -> (codes int8 [V,d], scale fp32 [V]). Exact
    fetch uses w_v.h ~= (codes_v . h) * scale_v; for CERTIFICATION the exact dot
    must use the TRUE W_U, so this is only a memory-traffic stand-in for the
    fetch-cost benchmark, not the certify path."""
    V, d = W.shape
    codes = torch.empty(V, d, dtype=torch.int8)
    scale = torch.empty(V)
    for s in range(0, V, vchunk):
        Wc = W[s : s + vchunk].float()
        sc = (Wc.abs().amax(1) / 127).clamp(min=1e-12)
        codes[s : s + vchunk] = torch.round(Wc / sc.unsqueeze(1)).clamp(-127, 127).to(
            torch.int8
        )
        scale[s : s + vchunk] = sc
    return codes, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seqs", type=int, default=24)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--positions", type=int, default=1500)
    ap.add_argument("--clusters", type=int, default=256)
    ap.add_argument("--basis", default="hidden_pca")
    ap.add_argument("--r", type=int, default=1280)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--h-dtype", default="float16", dest="h_dtype")
    ap.add_argument("--save-wq", action="store_true", help="also save int8 W_U "
                    "(1GB+) for the fetch-traffic benchmark")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(device).eval()
    torch.set_grad_enabled(False)
    head = model.get_output_embeddings()
    W_U = head.weight.detach()
    V, d = W_U.shape
    C = args.clusters
    cof = build_idblock(V, C).to(device)

    ids = load_split(tok, "val", args.seqs, args.seq_len)
    H, _, _ = capture(model, head, ids, device, pos_cap=args.positions)
    print(f"[{args.model}] V={V} d={d} positions={H.shape[0]} C={C}", flush=True)

    B = build_basis(args.basis, H, W_U, args.r, device)
    a, delta = shadow_precompute(W_U, B, device)
    codes, scale, err_norm = quantize_cols_codes(a, args.bits)
    print(f"shadow: a[{V},{args.r}] int{args.bits} aq_err_norm={err_norm:.4e}", flush=True)

    # dense argmax ground truth (chunked)
    Wt = W_U.float().to(device).t().contiguous()
    gt = torch.empty(H.shape[0], dtype=torch.int64)
    for s in range(0, H.shape[0], 256):
        lg = H[s : s + 256].float().to(device) @ Wt
        gt[s : s + 256] = lg.argmax(1).cpu()
    del Wt

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(codes.cpu(), out / "aq.pt")
    torch.save(scale.cpu(), out / "scale.pt")
    torch.save(delta.cpu(), out / "delta.pt")
    torch.save(B.cpu(), out / "B.pt")
    torch.save(cof.cpu().to(torch.int32), out / "cof.pt")
    hdt = getattr(torch, getattr(args, "h_dtype", "float16"))
    torch.save(H.to(hdt).cpu(), out / "H.pt")
    torch.save(gt, out / "gt.pt")
    if args.save_wq:
        Wq, Wscale = quantize_rows_int8(W_U)
        torch.save(Wq, out / "Wq.pt")
        torch.save(Wscale, out / "Wscale.pt")
    # the TRUE W_U in its NATIVE dtype (bf16 for a bf16 model) -- .half() would
    # silently retype a bf16 head to fp16 and change the deployed argmax.
    torch.save(W_U.detach().cpu(), out / "W_U.pt")
    meta = {
        "model": args.model,
        "V": int(V),
        "d": int(d),
        "r": int(args.r),
        "C": int(C),
        "bits": int(args.bits),
        "basis": args.basis,
        "aq_err_norm": float(err_norm),
        "N": int(H.shape[0]),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[wrote] artifact -> {out}", flush=True)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

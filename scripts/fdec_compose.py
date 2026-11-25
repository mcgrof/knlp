#!/usr/bin/env python3
"""fdec composition matrix on held-out text.

Runs every subset of three decode bricks on one model and reports whether they
compose. A brick is one optimization that reduces the bytes a decode step moves:

  w8    int8 weights for every linear layer in the decoder blocks (group-wise
        symmetric scales, calibration-free). The LM head keeps its native type.
  kv    asymmetric KV cache: keys stay 16-bit, values go to FP8 (e4m3) with
        a direct cast, as the vLLM FP8 cache does by default. `--kv-scale
        tensor` divides each value tensor by one scale first.
  cert  certified LM-head decode: a low-rank int8 shadow of the output head
        gives an upper bound per token; vocabulary slabs open in bound order
        until the dense argmax is proven. The returned token is exact.

For each of the eight subsets the script measures, on evaluation text that no
calibration step has seen:

  - perplexity, top-1 agreement and KL divergence against the unmodified model;
  - for cells with `cert`: whether the certificate returns the dense argmax of
    that cell's own model at every position, and what fraction of the output
    head it reads (shadow bytes plus fetched rows);
  - the analytical decode bytes per token (weights, KV cache, LM head).

The certificate basis comes from calibration hidden states (WikiText-2 train).
Evaluation uses WikiText-2 test. The script also reports the in-sample control
(basis built from the evaluation hidden states), because a basis that has seen
the evaluation states reads far fewer rows than a deployed one can.

Two shadow artifacts are measured for each `cert` cell: one built from the
unmodified model ("reused") and one rebuilt from the cell's own model
("matched"). The certificate is exact with either; only the bytes differ.

Attention in every cell, the unmodified model included, accumulates the
query-key scores and the weighted value sum in float32 and rounds only the
output to the model dtype, as a fused serving kernel does.  The stock eager
attention function rounds the scores to bf16 first, which degrades a model
with large attention scores (see fp32_attention in fdec_matrix.py).

Composition rule: a cell composes when it runs, its certificate (if present)
matches the dense argmax at every position, and its log-perplexity increase is
within `--interaction-tol` of the sum of its bricks' solo increases.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "lm_head", "kernel"))

from fdec_matrix import (  # noqa: E402
    _fake_quant_groupwise,
    _find_decoder_layers,
    install_asym_kv,
    install_fp32_attention,
    restore,
)
from h6_oracle_scan import build_idblock  # noqa: E402
from shadow_bound_replay import (  # noqa: E402
    build_basis,
    quantize_cols,
    replay,
    shadow_precompute,
)

BRICK_ORDER = ("w8", "kv", "cert")
KV_FACTOR = 0.75  # keys 16-bit + values 8-bit, against 16-bit both


def decode_bytes(model, ctx, kv_f, lm_f, w_f):
    """Bytes one decode step reads, in MB, by pool, for a 16-bit model.

    The decoder blocks are read once per token and are what the w8 brick
    quantizes. The KV cache grows with context. The LM head is the output
    matrix. The input embedding is left out: a step reads one row of it."""
    cfg = model.config
    blocks = sum(p.numel() for p in _find_decoder_layers(model).parameters())
    hkv = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    w = blocks * 2 * w_f
    kv = cfg.num_hidden_layers * hkv * ctx * hd * 2 * 2 * kv_f
    lm = cfg.vocab_size * cfg.hidden_size * 2 * lm_f
    return dict(
        weights_mb=w / 1e6,
        kv_mb=kv / 1e6,
        lmhead_mb=lm / 1e6,
        total_mb=(w + kv + lm) / 1e6,
    )


def load_ids(tok, split, num_seqs, seq_len):
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    enc = tok(text, return_tensors="pt").input_ids[0]
    num_seqs = min(num_seqs, enc.numel() // seq_len)
    return enc[: num_seqs * seq_len].view(num_seqs, seq_len)


@torch.no_grad()
def quantize_weights_int8(model, group):
    """In-place int8 fake quantization of every decoder-block linear layer.
    Returns the weight byte factor (bits per weight plus scale overhead)."""
    n = 0
    for layer in _find_decoder_layers(model):
        for m in layer.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = _fake_quant_groupwise(
                    m.weight.data.float(), 8, group
                ).to(m.weight.dtype)
                n += 1
    assert n > 0, "w8 brick found no linear layers to quantize"
    return (8.0 + 16.0 / group) / 16.0


@torch.no_grad()
def run_pass(model, head, ids, device, want_logp=False):
    """Teacher-forced pass. Returns the hidden states the LM head consumed, the
    dense argmax, the gold-token negative log-likelihood, and (optionally) the
    full log-probabilities in half precision for the KL reference."""
    grab = {}
    handle = head.register_forward_pre_hook(
        lambda mod, inp: grab.__setitem__("h", inp[0].detach())
    )
    Hs, AMs, NLLs, LPs = [], [], [], []
    for b in range(ids.shape[0]):
        x = ids[b : b + 1].to(device)
        logits = model(x, use_cache=False).logits[0, :-1].float()
        lp = F.log_softmax(logits, dim=-1)
        gold = x[0, 1:]
        NLLs.append(-lp.gather(1, gold.unsqueeze(1)).squeeze(1).cpu())
        AMs.append(logits.argmax(-1).cpu())
        Hs.append(grab["h"][0, :-1].float().cpu())
        if want_logp:
            LPs.append(lp.half().cpu())
    handle.remove()
    out = dict(H=torch.cat(Hs), am=torch.cat(AMs), nll=torch.cat(NLLs))
    if want_logp:
        out["lp"] = LPs
    return out


@torch.no_grad()
def kl_against(model, ids, device, ref_lp):
    """Mean KL(reference || cell) over all positions."""
    tot, n = 0.0, 0
    for b in range(ids.shape[0]):
        x = ids[b : b + 1].to(device)
        lp = F.log_softmax(model(x, use_cache=False).logits[0, :-1].float(), dim=-1)
        r = ref_lp[b].to(device).float()
        tot += float((r.exp() * (r - lp)).sum(-1).sum())
        n += r.shape[0]
    return tot / max(1, n)


def cert_measure(H_basis, H_eval, W_U, r, bits, clusters, device, basis="hidden_pca"):
    """Build the shadow from H_basis, replay the certificate on H_eval."""
    V, d = W_U.shape
    cof = build_idblock(V, clusters).to(device)
    B = build_basis(basis, H_basis, W_U, r, device)
    a, delta = shadow_precompute(W_U, B, device)
    err = 0.0
    if bits < 16:
        a, err = quantize_cols(a, bits)
    m = replay(H_eval, W_U, B, a, delta, cof, clusters, device, aq_err_norm=err)
    shadow = (r * (bits / 16.0) + 1) / d
    m["shadow_byteratio"] = shadow
    m["total_byteratio"] = shadow + m["fetched_mean"]
    del a, delta, B
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--eval-seqs", type=int, default=16)
    ap.add_argument("--calib-seqs", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--w8-group", type=int, default=128)
    ap.add_argument(
        "--cert-rank-frac",
        type=float,
        default=1.0,
        help="shadow rank as a fraction of the hidden size; held-out hidden "
        "states need full rank before the certificate reads less than the head",
    )
    ap.add_argument(
        "--kv-scale",
        choices=("one", "tensor"),
        default="one",
        help="how the values reach FP8: a direct cast, the vLLM default, or "
        "one scale per tensor (largest magnitude over 448)",
    )
    ap.add_argument("--cert-bits", type=int, default=8)
    ap.add_argument("--cert-clusters", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=32768, help="context for the byte model")
    ap.add_argument(
        "--interaction-tol",
        type=float,
        default=0.01,
        help="allowed excess log-perplexity over the sum of solo increases",
    )
    ap.add_argument(
        "--bricks",
        default=",".join(BRICK_ORDER),
        help="bricks to compose; every subset of this set is one cell",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    enabled = tuple(b for b in BRICK_ORDER if b in args.bricks.split(","))
    unknown = set(args.bricks.split(",")) - set(BRICK_ORDER)
    assert not unknown, f"unknown bricks: {sorted(unknown)}"

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    torch.manual_seed(0)
    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation="eager"
    ).to(device)
    model.eval()
    # Every cell, the unmodified model included, computes attention the way a
    # fused serving kernel does: float32 inside, output in the model dtype.
    install_fp32_attention(model)
    head = model.get_output_embeddings()
    assert getattr(head, "bias", None) is None, "certificate assumes a bias-free head"
    W_U = head.weight.detach()
    V, d = W_U.shape
    r = max(8, int(round(args.cert_rank_frac * d)))

    eval_ids = load_ids(tok, "test", args.eval_seqs, args.seq_len)
    calib_ids = load_ids(tok, "train", args.calib_seqs, args.seq_len)
    print(
        f"[fdec-compose] {args.model} V={V} d={d} cert r={r} int{args.cert_bits} "
        f"eval={eval_ids.numel()} tok calib={calib_ids.numel()} tok",
        flush=True,
    )

    # Subsets ordered so every cell without int8 weights runs before the
    # in-place weight quantization.
    subsets = [
        tuple(b for b in enabled if b in s)
        for n in range(len(enabled) + 1)
        for s in itertools.combinations(enabled, n)
    ]
    subsets.sort(key=lambda s: ("w8" in s, len(s)))

    base = run_pass(model, head, eval_ids, device, want_logp=True)
    base_calib_H = run_pass(model, head, calib_ids, device)["H"]
    base_ppl = math.exp(base["nll"].mean().item())
    ref_lp = base["lp"]

    cells = {}
    weight_f = 1.0
    for s in subsets:
        name = "+".join(s) if s else "none"
        rec = {"bricks": list(s), "ran": False}
        try:
            if "w8" in s and weight_f == 1.0:
                weight_f = quantize_weights_int8(model, args.w8_group)
            saved = install_asym_kv(model, args.kv_scale) if "kv" in s else {}
            if "kv" in s:
                assert saved, "kv brick did not patch any attention module"
            try:
                p = run_pass(model, head, eval_ids, device)
                rec["ppl"] = math.exp(p["nll"].mean().item())
                rec["dlogppl"] = math.log(rec["ppl"] / base_ppl)
                rec["top1_agreement"] = (p["am"] == base["am"]).float().mean().item()
                rec["mean_kl"] = kl_against(model, eval_ids, device, ref_lp)
                if "cert" in s:
                    Hc = run_pass(model, head, calib_ids, device)["H"]
                    kw = dict(
                        r=r,
                        bits=args.cert_bits,
                        clusters=args.cert_clusters,
                        device=device,
                    )
                    rec["cert_matched"] = cert_measure(Hc, p["H"], W_U, **kw)
                    rec["cert_reused"] = cert_measure(base_calib_H, p["H"], W_U, **kw)
                    rec["cert_in_sample"] = cert_measure(p["H"], p["H"], W_U, **kw)
                    del Hc
            finally:
                restore(saved)
            rec["ran"] = True
        except Exception as e:  # a brick conflict is a result, not a crash
            rec["error"] = f"{type(e).__name__}: {e}"
        lm_f = rec["cert_matched"]["total_byteratio"] if "cert_matched" in rec else 1.0
        lm_f = min(lm_f, 1.0)
        bm = decode_bytes(
            model,
            args.ctx,
            KV_FACTOR if "kv" in s else 1.0,
            lm_f,
            weight_f if "w8" in s else 1.0,
        )
        rec["decode_mb_per_token"] = bm["total_mb"]
        rec["byte_pools_mb"] = {k: bm[k] for k in ("weights_mb", "kv_mb", "lmhead_mb")}
        cells[name] = rec
        if device.type == "cuda":
            torch.cuda.empty_cache()
        msg = f"  {name:<12} "
        if rec["ran"]:
            msg += (
                f"ppl={rec['ppl']:.3f} dlog={rec['dlogppl']:+.4f} "
                f"top1={rec['top1_agreement']:.3f} KL={rec['mean_kl']:.4f}"
            )
            if "cert_matched" in rec:
                cm, ci = rec["cert_matched"], rec["cert_in_sample"]
                msg += (
                    f" | cert exact={cm['argmax_match']:.4f} "
                    f"head bytes={cm['total_byteratio']:.3f} "
                    f"(reused {rec['cert_reused']['total_byteratio']:.3f}, "
                    f"in-sample {ci['total_byteratio']:.3f})"
                )
        else:
            msg += f"FAILED {rec['error']}"
        print(msg, flush=True)

    # Composition verdict per cell.
    base_mb = cells["none"]["decode_mb_per_token"]
    for name, rec in cells.items():
        rec["bytes_saved_pct"] = round(
            100 * (1 - rec["decode_mb_per_token"] / base_mb), 2
        )
        if not rec["ran"]:
            rec["composes"] = False
            continue
        quality = [b for b in rec["bricks"] if b != "cert"]
        solo = sum(cells[b]["dlogppl"] for b in quality if cells[b]["ran"])
        rec["interaction_dlogppl"] = rec["dlogppl"] - solo
        exact = all(
            rec[k]["argmax_match"] == 1.0
            for k in ("cert_matched", "cert_reused")
            if k in rec
        )
        rec["cert_exact"] = exact if "cert" in rec["bricks"] else None
        rec["composes"] = bool(
            exact and rec["interaction_dlogppl"] <= args.interaction_tol
        )

    res = {
        "model": args.model,
        "vocab_size": int(V),
        "hidden_size": int(d),
        "eval": {"source": "wikitext-2 test", "tokens": int(eval_ids.numel())},
        "calibration": {"source": "wikitext-2 train", "tokens": int(calib_ids.numel())},
        "cert": {"rank": r, "bits": args.cert_bits, "clusters": args.cert_clusters},
        "w8_group": args.w8_group,
        "kv_scale": args.kv_scale,
        "attention": "float32 scores, softmax and value sum; output in model dtype",
        "dtype": args.dtype,
        "torch": torch.__version__,
        "ctx": args.ctx,
        "interaction_tol": args.interaction_tol,
        "baseline_ppl": base_ppl,
        "cells": cells,
        "all_compose": all(c["composes"] for c in cells.values()),
        "wall_seconds": round(time.time() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(
        f"[fdec-compose] all cells compose: {res['all_compose']}  "
        f"-> {args.out} ({res['wall_seconds']}s)"
    )


if __name__ == "__main__":
    main()

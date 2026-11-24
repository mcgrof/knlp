#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Held-out latency of the certified LM-head decode against the dense head.

The byte sweep (fdec_cert_heldout.py) shows that a shadow built from one set
of hidden states and used on another reads less than the dense head only near
full rank.  This script times that configuration on a GPU, one token at a
time, with the Triton kernels in lm_head/kernel:

  dense       bf16 matrix-vector product over the whole head, then argmax;
              its logits are rounded to bf16, so near-ties can change the
              argmax;
  loop        int8 shadow bound, then open vocabulary slabs one at a time
              until the argmax is certified;
  oneshot-K   int8 shadow bound, then one fused gather over the top fraction
              K of slabs and one certificate check.  A token that does not
              certify falls back to the dense head and takes its answer.

Two shadows are timed on the same GPU:

  held-out   basis from WikiText-2 train hidden states, timed on WikiText-2
             test hidden states, at --rank-frac of the hidden size;
  in-sample  basis from the first --in-sample-positions test hidden states,
             timed on those same states, at --in-sample-rank-frac.

Every row reports "exact": the share of tokens whose argmax equals the
argmax of float32 logits computed from the same head weights.  A certified
token always matches it.  The dense bf16 head does not always match it, so
a one-shot row is below 1.0 by the tokens that fell back and hit a near-tie.

The in-sample row repeats how the earlier latency figures were measured.  The
held-out row is what a deployed shadow can reach.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "lm_head", "kernel"))

from fdec_compose import load_ids, run_pass  # noqa: E402
from gen_artifact import quantize_cols_codes  # noqa: E402
from shadow_bound_replay import build_basis, shadow_precompute  # noqa: E402


def timed(fn, hs):
    """Per-token wall time in ms, device synchronized around each call."""
    for h in hs[:8]:
        fn(h)
    out, ts = [], []
    for h in hs:
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = fn(h)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
        out.append(r)
    return ts, out


def stats(ts):
    q = statistics.quantiles(ts, n=20)
    return dict(
        mean_ms=statistics.fmean(ts), median_ms=statistics.median(ts), p95_ms=q[18]
    )


def bench(tag, H_basis, H_time, W_U, WUt, r, bits, clusters, k_fracs, device):
    from certdecode_kernel import certified_decode, certified_decode_oneshot

    V, d = W_U.shape
    # The kernels view the vocabulary as equal contiguous slabs, so use the
    # largest slab count up to the request that divides the vocabulary.
    clusters = max(c for c in range(1, clusters + 1) if V % c == 0)
    S = V // clusters
    B = build_basis("hidden_pca", H_basis, W_U, r, device)
    a, delta = shadow_precompute(W_U, B, device)
    aq, scale, err = quantize_cols_codes(a, bits)
    del a
    aq, scale, delta, B = (
        aq.to(device).contiguous(),
        scale.to(device),
        delta.to(device).float(),
        B.to(device).float().contiguous(),
    )
    hs = [h.to(device) for h in H_time]
    dense_dtype = WUt.dtype

    # Reference argmax from float32 logits.  The timed dense head rounds its
    # logits to the head dtype, which makes ties, so it is scored too.
    Wf = W_U.float()
    gt = [int((Wf @ h.float()).argmax()) for h in hs]
    del Wf
    t_dense, am = timed(lambda h: int((h.to(dense_dtype) @ WUt).argmax()), hs)
    rows = {"dense": stats(t_dense)}
    rows["dense"]["exact"] = sum(int(a == g) for a, g in zip(am, gt)) / len(gt)
    base = rows["dense"]["mean_ms"]

    def loop(h):
        bid, fetched, _ = certified_decode(h, B, aq, scale, delta, W_U, S, err)
        return bid, fetched, True

    variants = [("loop", loop)]
    for k in k_fracs:
        variants.append(
            (
                f"oneshot-{k:g}",
                lambda h, k=k: certified_decode_oneshot(
                    h, B, aq, scale, delta, W_U, S, err, WUt, k_frac=k
                ),
            )
        )
    shadow = (r * (bits / 16.0) + 2) / d
    for name, fn in variants:
        ts, out = timed(fn, hs)
        s = stats(ts)
        s.update(
            exact=sum(int(o[0] == g) for o, g in zip(out, gt)) / len(gt),
            head_read=shadow + statistics.fmean(o[1] for o in out) / V,
            certified=sum(bool(o[2]) for o in out) / len(out),
            speedup_mean=base / s["mean_ms"],
            speedup_median=rows["dense"]["median_ms"] / s["median_ms"],
        )
        rows[name] = s
    print(f"[cert-latency] {tag} r={r} slabs={clusters} tokens={len(hs)}", flush=True)
    for name, s in rows.items():
        extra = (
            f" exact={s['exact']:.4f}"
            if name == "dense"
            else f" x{s['speedup_mean']:.2f} mean x{s['speedup_median']:.2f} median "
            f"read={s['head_read']:.3f} certified={s['certified']:.3f} "
            f"exact={s['exact']:.4f}"
        )
        print(
            f"  {name:<14} mean={s['mean_ms']:.3f} ms median={s['median_ms']:.3f} ms"
            + extra,
            flush=True,
        )
    del aq, scale, delta, B
    torch.cuda.empty_cache()
    return dict(rank=r, slabs=clusters, tokens=len(hs), rows=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--eval-seqs", type=int, default=16)
    ap.add_argument("--calib-seqs", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--tokens", type=int, default=1000, help="tokens timed per row")
    ap.add_argument("--rank-frac", type=float, default=1.0)
    ap.add_argument("--in-sample-rank-frac", type=float, default=0.357)
    ap.add_argument("--in-sample-positions", type=int, default=1500)
    ap.add_argument("--k-fracs", default="0.0625,0.125,0.25")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--clusters", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype)
    ).to(device)
    model.eval()
    head = model.get_output_embeddings()
    assert getattr(head, "bias", None) is None, "certificate assumes a bias-free head"
    W_U = head.weight.detach().clone()
    V, d = W_U.shape
    H_eval = run_pass(
        model, head, load_ids(tok, "test", args.eval_seqs, args.seq_len), device
    )["H"]
    H_calib = run_pass(
        model, head, load_ids(tok, "train", args.calib_seqs, args.seq_len), device
    )["H"]
    del model, head
    torch.cuda.empty_cache()
    WUt = W_U.t().contiguous()
    k_fracs = [float(x) for x in args.k_fracs.split(",")]
    kw = dict(bits=args.bits, clusters=args.clusters, k_fracs=k_fracs, device=device)
    pick = torch.randperm(H_eval.shape[0])[: args.tokens]
    H_small = H_eval[: args.in_sample_positions]
    res = {
        "model": args.model,
        "vocab_size": int(V),
        "hidden_size": int(d),
        "gpu": torch.cuda.get_device_name(device),
        "bits": args.bits,
        "clusters": args.clusters,
        "held_out": bench(
            "held-out",
            H_calib,
            H_eval[pick],
            W_U,
            WUt,
            min(d, int(round(args.rank_frac * d))),
            **kw,
        ),
        "in_sample": bench(
            "in-sample",
            H_small,
            H_small[: args.tokens],
            W_U,
            WUt,
            int(round(args.in_sample_rank_frac * d)),
            **kw,
        ),
        "wall_seconds": round(time.time() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[cert-latency] -> {args.out} ({res['wall_seconds']}s)")


if __name__ == "__main__":
    main()

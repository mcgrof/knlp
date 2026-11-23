#!/usr/bin/env python3
"""Held-out byte cost of the certified LM-head decode.

The certificate bounds every logit with a rank-r shadow of the output head
plus a residual term rho * delta_v, where rho is the part of the hidden state
outside the shadow basis. The decoder reads vocabulary slabs until the dense
argmax is proven, so the token is exact for any basis. The bytes it reads
depend on rho, and rho depends on whether the basis has seen the hidden
states it is used on.

This script captures hidden states once (WikiText-2 train for the basis,
WikiText-2 test for evaluation) and sweeps the shadow rank and the basis
kind. For each point it reports two numbers:

  held-out   basis from train hidden states, replay on test hidden states;
  in-sample  basis from the first `--in-sample-positions` test hidden
             states, replay on those same states.

The in-sample number is what a replay measures when it builds the basis from
the positions it then scores. The held-out number is what a deployed shadow
can reach.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "lm_head", "kernel"))

from fdec_compose import cert_measure, load_ids, run_pass  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--eval-seqs", type=int, default=16)
    ap.add_argument("--calib-seqs", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--in-sample-positions", type=int, default=1500)
    ap.add_argument("--rank-fracs", default="0.357,0.5,0.75,0.9,1.0")
    ap.add_argument("--bases", default="hidden_pca,w_svd")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--clusters", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    torch.manual_seed(0)
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
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    H_small = H_eval[: args.in_sample_positions]
    print(
        f"[cert-heldout] {args.model} V={V} d={d} eval={H_eval.shape[0]} "
        f"calib={H_calib.shape[0]} in-sample={H_small.shape[0]}",
        flush=True,
    )

    rows = []
    for basis in args.bases.split(","):
        for frac in (float(x) for x in args.rank_fracs.split(",")):
            r = min(d, max(8, int(round(frac * d))))
            kw = dict(r=r, bits=args.bits, clusters=args.clusters, device=device)
            ho = cert_measure(H_calib, H_eval, W_U, basis=basis, **kw)
            ins = cert_measure(H_small, H_small, W_U, basis=basis, **kw)
            rows.append(dict(basis=basis, rank=r, held_out=ho, in_sample=ins))
            print(
                f"  {basis:<10} r={r:<5} held-out: exact={ho['argmax_match']:.4f} "
                f"fetched={ho['fetched_mean']:.3f} total={ho['total_byteratio']:.3f}"
                f" | in-sample: exact={ins['argmax_match']:.4f} "
                f"fetched={ins['fetched_mean']:.3f} "
                f"total={ins['total_byteratio']:.3f}",
                flush=True,
            )
    res = {
        "model": args.model,
        "vocab_size": int(V),
        "hidden_size": int(d),
        "bits": args.bits,
        "clusters": args.clusters,
        "eval_positions": int(H_eval.shape[0]),
        "calib_positions": int(H_calib.shape[0]),
        "in_sample_positions": int(H_small.shape[0]),
        "rows": rows,
        "wall_seconds": round(time.time() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[cert-heldout] -> {args.out} ({res['wall_seconds']}s)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""H2 supervised routability: co-train a routing head WITH the LoRA so the
hidden vector the LM head consumes becomes linearly separable by idblock
cluster.

Steps 1/2 located the LM-head routing gap precisely: a small router from the
hidden state falls 0.12-0.26 short of the oracle at m=32 and the gap does not
shrink with router capacity or model scale -- it is a ROUTABILITY limit (the
hidden state does not expose idblock-cluster identity to a linear router), not a
starved bottleneck. H1's unsupervised cluster-entropy aux was a NEGATIVE: it
sharpens mass onto already-dominant clusters without correcting misses.

H2 attacks the gap at its actual location. A trainable routing head reads the
exact hidden the LM head consumes and is trained with cross-entropy against the
TRUE next-token cluster, jointly with the LoRA. That routing gradient flows
back through the LoRA into the hidden states, pressuring them to expose cluster
identity. The language-model loss holds perplexity -- the hard gate.

Decisive readout (before adapter vs after): dense ppl (must not degrade), oracle
truenext@m32 (the ceiling), the CO-TRAINED router's truenext (deployable), and a
FRESHLY-trained router on the adapted hiddens (does ANY linear router now reach
the oracle = is the routability limit overcome, not just this one router fit).

Usage:
  python3 lm_head/h2_supervised_routing.py --model EleutherAI/pythia-410m \
      --clusters 256 --lambda-route 0.5 --steps 300 --out OUT/h2_pythia410m.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from h6_oracle_scan import build_idblock, oracle_scan, dense_ppl
from predictor_baseline import (
    Router,
    capture,
    eval_router,
    load_split,
    train_router,
)


def lora_targets(model):
    names = {n.split(".")[-1] for n, _ in model.named_modules()}
    cand = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    hit = [c for c in cand if c in names]
    # pythia/gpt-neox fused attention + mlp naming fallback
    return hit or [
        c
        for c in ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        if c in names
    ]


def measure(model, tag, tr_ids, va_ids, cof, C, device, head, epochs, cotrained=None):
    """Dense ppl + oracle truenext@32 + a freshly-trained predictor truenext@32
    on THIS model's hiddens; optionally also a supplied co-trained router.

    The fresh router is trained on TRAIN hiddens and evaluated on held-out VAL
    hiddens -- training and evaluating on the same split overfits and spuriously
    closes the gap."""
    ppl = dense_ppl(model, va_ids, device)
    orc = oracle_scan(model, va_ids, cof, C, [32], device)[32]
    Htr, Ytr, _ = capture(model, head, tr_ids, device, pos_cap=120000)
    Hva, Yva, T10 = capture(model, head, va_ids, device)
    d = Hva.shape[1]
    V = cof.numel()
    fresh = train_router(Htr, cof.cpu()[Ytr], d, C, 64, device, epochs, 4096, 2e-3, 0)
    pred = eval_router(fresh, Hva, Yva, T10, cof, C, [32], device, V)[32]
    out = {
        "ppl": ppl,
        "oracle_truenext": orc["truenext"],
        "oracle_ppl_nofb": orc["ppl_nofb"],
        "fresh_pred_truenext": pred["truenext"],
        "fresh_pred_fetch": pred["fetch"],
        "gap_to_oracle": orc["truenext"] - pred["truenext"],
    }
    line = (
        f"  [{tag}] ppl={ppl:.2f}  oracle_tn={orc['truenext']:.3f}  "
        f"fresh_pred_tn={pred['truenext']:.3f}  gap={out['gap_to_oracle']:.3f}"
    )
    if cotrained is not None:
        ct = eval_router(cotrained, Hva, Yva, T10, cof, C, [32], device, V)[32]
        out["cotrained_pred_truenext"] = ct["truenext"]
        out["cotrained_pred_fetch"] = ct["fetch"]
        out["cotrained_gap_to_oracle"] = orc["truenext"] - ct["truenext"]
        line += (
            f"  cotrained_tn={ct['truenext']:.3f} "
            f"(gap {out['cotrained_gap_to_oracle']:.3f})"
        )
    print(line, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--train-seqs", type=int, default=400)
    ap.add_argument("--val-seqs", type=int, default=80)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--clusters", type=int, default=256)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lambda-route", type=float, default=0.5)
    ap.add_argument("--router-hidden", type=int, default=64)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--router-lr", type=float, default=2e-3)
    ap.add_argument("--pred-epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    t0 = time.time()
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    head = base.get_output_embeddings()
    V, d = head.weight.shape
    C = args.clusters
    cof = build_idblock(V, C).to(device)
    cof_cpu = cof.cpu()

    tr_ids = load_split(tok, "train", args.train_seqs, args.seq_len)
    va_ids = load_split(tok, "val", args.val_seqs, args.seq_len)

    lcfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=lora_targets(base),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lcfg)
    model.print_trainable_parameters()

    # ---- BEFORE (adapter disabled): the step-1 routability baseline ----
    print(f"[{args.model}] measuring BASE ...", flush=True)
    model.eval()
    with model.disable_adapter():
        before = measure(
            model, "base", tr_ids, va_ids, cof, C, device, head, args.pred_epochs
        )

    # ---- co-trained router over the head-input hidden ----
    router = Router(d, args.router_hidden, C).to(device)
    grab = {}

    def pre_hook(mod, inp):
        grab["h"] = inp[0]  # KEEP grad: this is the route gradient's path into LoRA

    handle = head.register_forward_pre_hook(pre_hook)

    params = [p for p in model.parameters() if p.requires_grad] + list(
        router.parameters()
    )
    opt = torch.optim.AdamW(
        [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": args.lr,
            },
            {"params": list(router.parameters()), "lr": args.router_lr},
        ]
    )
    mb = args.micro_batch
    nseq = tr_ids.shape[0]
    model.train()
    step = 0
    print(
        f"[{args.model}] co-training LoRA + router (lambda_route={args.lambda_route}) ...",
        flush=True,
    )
    while step < args.steps:
        perm = torch.randperm(nseq)
        for s in range(0, nseq, mb):
            if step >= args.steps:
                break
            batch = tr_ids[perm[s : s + mb]].to(device)
            out = model(batch)
            logits = out.logits[:, :-1, :]
            y = batch[:, 1:]
            lm = F.cross_entropy(logits.reshape(-1, V).float(), y.reshape(-1))
            # route: the head-input hidden at each position -> true next cluster
            h = grab["h"][:, :-1, :].reshape(-1, d).float()
            tgt = cof[y.reshape(-1)]
            route = F.cross_entropy(router(h), tgt)
            loss = lm + args.lambda_route * route
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            if step % 50 == 0 or step == 1:
                with torch.no_grad():
                    racc = (router(h).argmax(-1) == tgt).float().mean().item()
                print(
                    f"    step {step}/{args.steps}  lm={lm.item():.3f}  "
                    f"route={route.item():.3f}  route_acc={racc:.3f}",
                    flush=True,
                )
    handle.remove()

    # ---- AFTER (adapter enabled): does the gap close, ppl gate hold ----
    print(f"[{args.model}] measuring ADAPTED ...", flush=True)
    model.eval()
    router.eval()
    after = measure(
        model,
        "adapted",
        tr_ids,
        va_ids,
        cof,
        C,
        device,
        head,
        args.pred_epochs,
        cotrained=router,
    )

    result = {
        "model": args.model,
        "vocab_size": int(V),
        "clusters": C,
        "lambda_route": args.lambda_route,
        "lora_rank": args.lora_rank,
        "steps": args.steps,
        "m": 32,
        "before": before,
        "after": after,
        "deltas": {
            "ppl": after["ppl"] - before["ppl"],
            "oracle_truenext": after["oracle_truenext"] - before["oracle_truenext"],
            "fresh_pred_truenext": after["fresh_pred_truenext"]
            - before["fresh_pred_truenext"],
            "gap_to_oracle": after["gap_to_oracle"] - before["gap_to_oracle"],
        },
        "wall_seconds": round(time.time() - t0, 1),
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(result, indent=2))
    print(
        f"[{args.model}] H2 RESULT  dppl={result['deltas']['ppl']:+.2f}  "
        f"fresh_gap {before['gap_to_oracle']:.3f}->{after['gap_to_oracle']:.3f}  "
        f"cotrained_gap={after.get('cotrained_gap_to_oracle', float('nan')):.3f}  "
        f"(WIN = gap shrinks AND dppl ~ 0)",
        flush=True,
    )
    print(f"[wrote] {outp}  ({result['wall_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()

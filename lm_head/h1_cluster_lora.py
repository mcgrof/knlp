#!/usr/bin/env python3
"""H1 cluster-coherence LoRA: make the model route its own output mass.

Fine-tune a LoRA adapter (base weights frozen) with the usual language-model
loss plus an auxiliary term that minimizes the entropy of the next-token
distribution *aggregated over idblock clusters*. In words: reward the model for
putting its probability mass inside few clusters, so the true next token's
cluster becomes more dominant. The bet from the H6/baseline measurements is that
the routing ceiling is high but the deployable router can't reach it; H1 tries to
raise the ceiling and, by sharpening the cluster structure, also pull the small
router up toward it.

It reports, before (adapter disabled) and after the adapter, three things on a
held-out split: dense perplexity (the hard gate -- must not degrade), the oracle
true-next coverage at m=32, and a freshly-trained deployable router's true-next
coverage at m=32. The headline is whether the adapter raises the deployable
router without hurting perplexity.

Usage:
  python3 lm_head/h1_cluster_lora.py --model EleutherAI/pythia-410m \
      --lambda-aux 0.1 --steps 300 --out OUT/h1_pythia-410m.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from h6_oracle_scan import build_idblock, oracle_scan, dense_ppl
from predictor_baseline import load_split, capture, train_router, eval_router


def lora_targets(model):
    name = model.config.model_type
    if name in ("gpt_neox", "gptj"):
        return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    # llama / qwen2 / mistral family
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def cluster_entropy(logits, cluster_of, C):
    """Mean entropy of the cluster-aggregated next-token distribution."""
    N, V = logits.shape
    p = F.softmax(logits, dim=-1)
    idx = cluster_of.unsqueeze(0).expand(N, V)
    q = torch.zeros(N, C, device=logits.device, dtype=p.dtype).scatter_add_(1, idx, p)
    ent = -(q * (q + 1e-9).log()).sum(1)
    return ent.mean()


def measure(model, tag, va_ids, cof, C, device, head, epochs):
    """Dense ppl + oracle truenext + freshly-trained predictor truenext (m=32)."""
    ppl = dense_ppl(model, va_ids, device)
    orc = oracle_scan(model, va_ids, cof, C, [32], device)[32]
    # deployable predictor on this model's hidden states
    Htr, Ytr, _ = capture(model, head, va_ids[: len(va_ids) // 2], device)
    Hva, Yva, T10 = capture(model, head, va_ids[len(va_ids) // 2 :], device)
    V = cof.numel()
    d = Htr.shape[1]
    r = train_router(Htr, cof[Ytr], d, C, 64, device, epochs, 4096, 2e-3, 0)
    pred = eval_router(r, Hva, Yva, T10, cof, C, [32], device, V)[32]
    print(
        f"  [{tag}] ppl={ppl:.2f}  oracle_tn={orc['truenext']:.3f}  "
        f"pred_tn={pred['truenext']:.3f}  pred_top10={pred['top10']:.3f}",
        flush=True,
    )
    return {
        "ppl": ppl,
        "oracle_truenext": orc["truenext"],
        "oracle_ppl_nofb": orc["ppl_nofb"],
        "pred_truenext": pred["truenext"],
        "pred_top10": pred["top10"],
    }


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
    ap.add_argument("--lambda-aux", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
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
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    base.to(device)
    head = base.get_output_embeddings()
    V, d = head.weight.shape
    C = args.clusters
    cof = build_idblock(V, C).to(device)

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

    # ---- before (adapter disabled) ----
    print(f"[{args.model}] measuring BASE ...", flush=True)
    model.eval()
    # No outer no_grad: the model forwards inside measure() are already
    # @torch.no_grad, but train_router() needs gradients for the small router.
    with model.disable_adapter():
        before = measure(model, "base", va_ids, cof, C, device, head, args.pred_epochs)

    # ---- train LoRA with LM loss + lambda * cluster entropy ----
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    mb = args.micro_batch
    nseq = tr_ids.shape[0]
    model.train()
    step = 0
    print(f"[{args.model}] training LoRA (lambda={args.lambda_aux}) ...", flush=True)
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
            aux = cluster_entropy(logits.reshape(-1, V).float(), cof, C)
            loss = lm + args.lambda_aux * aux
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            step += 1
            if step % 50 == 0 or step == 1:
                print(
                    f"    step {step}/{args.steps}  lm={lm.item():.3f}  "
                    f"clus_ent={aux.item():.3f}",
                    flush=True,
                )

    # ---- after (adapter enabled) ----
    print(f"[{args.model}] measuring ADAPTED ...", flush=True)
    model.eval()
    after = measure(model, "adapted", va_ids, cof, C, device, head, args.pred_epochs)

    result = {
        "model": args.model,
        "vocab_size": int(V),
        "lambda_aux": args.lambda_aux,
        "lora_rank": args.lora_rank,
        "steps": args.steps,
        "m": 32,
        "before": before,
        "after": after,
        "deltas": {
            "ppl": after["ppl"] - before["ppl"],
            "oracle_truenext": after["oracle_truenext"] - before["oracle_truenext"],
            "pred_truenext": after["pred_truenext"] - before["pred_truenext"],
        },
        "wall_seconds": round(time.time() - t0, 1),
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(result, indent=2))
    dppl = result["deltas"]["ppl"]
    print(
        f"[{args.model}] DELTAS  ppl={dppl:+.2f}  "
        f"oracle_tn={result['deltas']['oracle_truenext']:+.3f}  "
        f"pred_tn={result['deltas']['pred_truenext']:+.3f}  "
        f"(gate: ppl must not rise much)",
        flush=True,
    )
    print(f"[wrote] {outp}  ({result['wall_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()

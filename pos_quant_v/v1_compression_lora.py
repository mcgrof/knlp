#!/usr/bin/env python3
"""V1 compression-aware LoRA: train the model to tolerate low-rank V.

V6 showed a training-free rank-32 V compression is retrieval-lossless but that
lower ranks (rank-8/16) break. V1 trains a LoRA adapter (base frozen) with the
rank-k V compression APPLIED during the forward pass (straight-through estimator:
compress in forward, identity gradient), so the model learns to route information
in a way that survives aggressive low-rank V. The aim is to push the deployable
operating rank below the training-free rank-32.

The compression is a forward hook on each layer's v_proj that projects the value
onto a fixed per-(layer,kv-head) rank-k PCA basis (fit on a calibration set) and
reconstructs, with STE. It compresses ALL positions uniformly (the hardest case;
the deployable filler-only version is strictly easier).

Reports perplexity on held-out text for: base (no compression), training-free
rank-k (compression, no LoRA), LoRA+rank-k (the result), and LoRA without
compression (the base-quality gate). The win is LoRA+rank-k perplexity
approaching base while the LoRA-no-compress gate stays near base.

Usage:
  python3 pos_quant_v/v1_compression_lora.py --model Qwen/Qwen2.5-7B \
      --rank 8 --steps 300 --out OUT/v1_rank8.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from predictor_baseline import load_split
from v6_compress_eval import prefill_values, get_values, fit_bases


def attn_layers(model):
    return model.model.layers


def make_ste_hook(basis_layer, enabled):
    """Forward hook on v_proj: STE rank-k compression of the value projection.
    basis_layer [H, hd, k]; enabled is a 1-element list toggle."""
    H, hd, k = basis_layer.shape

    def hook(module, inp, out):
        if not enabled[0]:
            return out
        B, T, _ = out.shape
        v = out.view(B, T, H, hd).float()
        proj = torch.einsum("bthd,hdk->bthk", v, basis_layer)
        recon = torch.einsum("bthk,hdk->bthd", proj, basis_layer)
        comp = v + (recon - v).detach()  # straight-through
        return comp.view(B, T, H * hd).to(out.dtype)

    return hook


@torch.no_grad()
def perplexity(model, ids_2d, device):
    tot_nll, tot = 0.0, 0
    for i in range(ids_2d.shape[0]):
        ids = ids_2d[i : i + 1].to(device)
        logits = model(ids).logits[0, :-1].float()
        y = ids[0, 1:]
        nll = F.cross_entropy(logits, y, reduction="sum")
        tot_nll += nll.item()
        tot += y.numel()
    return math.exp(tot_nll / max(tot, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--calib", type=int, default=6)
    ap.add_argument("--train-seqs", type=int, default=300)
    ap.add_argument("--val-seqs", type=int, default=40)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=16)
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
    base.eval()

    tr = load_split(tok, "train", args.train_seqs, args.seq_len)
    va = load_split(tok, "val", args.val_seqs, args.seq_len)

    # calibration: fit per-(layer,kv-head) rank-k basis on cache V of calib text
    calib_vals = []
    for i in range(args.calib):
        past = prefill_values(base, tr[i : i + 1], device)
        calib_vals.append([v.cpu() for v in get_values(past)])
    bases = fit_bases(calib_vals, [args.rank], device)[args.rank]
    print(f"[{args.model}] fit rank-{args.rank} V bases", flush=True)

    # register STE compression hooks
    enabled = [False]
    handles = []
    for L, layer in enumerate(attn_layers(base)):
        h = layer.self_attn.v_proj.register_forward_hook(
            make_ste_hook(bases[L], enabled)
        )
        handles.append(h)

    def ppl(compress):
        enabled[0] = compress
        p = perplexity(base, va, device)
        enabled[0] = False
        return p

    base_ppl = ppl(False)
    free_ppl = ppl(True)  # training-free rank-k compression
    print(
        f"  base ppl={base_ppl:.2f}  training-free rank{args.rank} ppl={free_ppl:.2f}",
        flush=True,
    )

    # LoRA + compression-aware training
    lcfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=2 * args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lcfg)
    model.print_trainable_parameters()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    mb, V = args.micro_batch, base.config.vocab_size
    nseq = tr.shape[0]
    model.train()
    enabled[0] = True  # compression on during training
    step = 0
    print(f"  training LoRA with rank-{args.rank} V (STE) ...", flush=True)
    while step < args.steps:
        perm = torch.randperm(nseq)
        for s in range(0, nseq, mb):
            if step >= args.steps:
                break
            batch = tr[perm[s : s + mb]].to(device)
            logits = model(batch).logits[:, :-1, :]
            loss = F.cross_entropy(
                logits.reshape(-1, V).float(), batch[:, 1:].reshape(-1)
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            step += 1
            if step % 50 == 0 or step == 1:
                print(
                    f"    step {step}/{args.steps} loss={loss.item():.3f}", flush=True
                )
    model.eval()
    enabled[0] = False

    lora_comp_ppl = ppl(True)  # LoRA + rank-k compression (the result)
    lora_free_ppl = ppl(False)  # LoRA, no compression (base-quality gate)
    print(
        f"  LoRA+rank{args.rank} ppl={lora_comp_ppl:.2f}  "
        f"LoRA-no-compress ppl={lora_free_ppl:.2f} (gate)",
        flush=True,
    )

    for h in handles:
        h.remove()
    result = {
        "model": args.model,
        "rank": args.rank,
        "steps": args.steps,
        "ppl": {
            "base": base_ppl,
            "trainfree_rankk": free_ppl,
            "lora_rankk": lora_comp_ppl,
            "lora_no_compress_gate": lora_free_ppl,
        },
        "recovered_frac": (free_ppl - lora_comp_ppl) / max(free_ppl - base_ppl, 1e-9),
        "wall_seconds": round(time.time() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(
        f"[recovered {result['recovered_frac']*100:.0f}% of the rank-{args.rank} "
        f"ppl gap] wrote {args.out} ({result['wall_seconds']}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()

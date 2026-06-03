"""Phase 4: distill a TrellisRetrofit from a pretrained GPT-2 teacher.

loss = CE(labels) + lambda_kl * KL(teacher_full_cache || student)
The teacher is a frozen HF GPT-2 (full softmax attention). Student is a Trellis
bounded-memory LM warm-started from the teacher (retrofit). LoRA or full FT.

Honesty: this trains in the fast stale-gradient mode; the per-token recurrence
still caps practical context length on the W7900 (a full matched-budget Phase-4
vs KRI-FT at long context is gated on the chunked kernel). This script proves
the retrofit+distill mechanism and gives a short matched-budget read at moderate
context.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis_lm.retrofit import TrellisRetrofit, freeze_to_lora


def _dtype(n):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[n]


def packed(dataset, split, seq_len, n, tok):
    from datasets import load_dataset
    ds = load_dataset(dataset, split=split, streaming=True)
    buf, out = [], []
    for ex in ds:
        buf.extend(tok(ex.get("text") or "").input_ids + [tok.eos_token_id])
        while len(buf) >= seq_len:
            out.append(buf[:seq_len]); buf = buf[seq_len:]
            if len(out) >= n:
                return out
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpt2", default="openai-community/gpt2")
    p.add_argument("--n_slots", type=int, default=64)
    p.add_argument("--mode", default="lora", choices=["lora", "full"])
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lambda_kl", type=float, default=1.0)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--dataset", default="roneneldan/TinyStories")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--out", default=None)
    p.add_argument("--save_ckpt", default=None)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = _dtype(a.dtype)
    from transformers import AutoTokenizer, GPT2LMHeadModel
    tok = AutoTokenizer.from_pretrained(a.gpt2)
    teacher = GPT2LMHeadModel.from_pretrained(a.gpt2).to(device).eval()
    for pr in teacher.parameters():
        pr.requires_grad_(False)
    student = TrellisRetrofit.from_gpt2(a.gpt2, n_slots=a.n_slots, dtype=a.dtype).to(device)
    if a.dtype != "fp32":
        student = student.to(dt); teacher = teacher.to(dt)
    if a.mode == "lora":
        trainable = freeze_to_lora(student, rank=a.lora_rank)
    else:
        trainable = [pr for pr in student.parameters() if pr.requires_grad]
    n_train = sum(pr.numel() for pr in trainable)
    opt = torch.optim.AdamW(trainable, lr=a.lr)
    data = packed(a.dataset, "train", a.seq_len, a.steps * a.batch, tok)
    print(f"retrofit mode={a.mode} trainable={n_train:,} "
          f"mem_state_bytes/seq={student.memory_state_bytes(1)} seq={a.seq_len}", flush=True)

    student.train()
    hist = []
    t0 = time.time(); ntok = 0; bi = 0
    for step in range(1, a.steps + 1):
        rows = data[bi:bi + a.batch]; bi += a.batch
        if len(rows) < a.batch:
            break
        idx = torch.tensor(rows, device=device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dt, enabled=a.dtype != "fp32"):
            t_logits = teacher(idx).logits
        with torch.autocast(device_type=device.type, dtype=dt, enabled=a.dtype != "fp32"):
            s_logits, ce = student(idx, labels=idx, training=True)
            lp_s = F.log_softmax(s_logits.float(), -1)
            p_t = F.softmax(t_logits.float(), -1)
            kl = (p_t * (F.log_softmax(t_logits.float(), -1) - lp_s)).sum(-1).mean()
            loss = ce + a.lambda_kl * kl
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        ntok += idx.numel()
        if step % a.log_every == 0 or step == 1:
            tps = ntok / (time.time() - t0)
            print(f"  step {step:5d}  ce {ce.item():.4f}  kl {kl.item():.4f}  "
                  f"ppl {math.exp(min(20, ce.item())):.2f}  tok/s {tps:.0f}", flush=True)
            hist.append({"step": step, "ce": ce.item(), "kl": kl.item(), "tok_s": tps})
    if a.save_ckpt:
        torch.save({"cfg": student.cfg.to_dict(), "model": student.state_dict(),
                    "kind": "trellis"}, a.save_ckpt)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"args": vars(a), "trainable": n_train,
             "mem_state_bytes_per_seq": student.memory_state_bytes(1),
             "history": hist}, indent=2))
        print(f"wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Train TrellisLM / DenseTransformerTiny.

Two tasks:
  --task recall : synthetic associative recall (Phase-0 correctness proof).
                  Sequence of (key,value) pairs then a repeated key; the model
                  must recall the paired value. We track loss AND recall
                  accuracy at the query position, plus the bounded memory size.
  --task lm     : language modeling on TinyStories (or any HF text dataset),
                  gpt2-tokenized and packed to --seq_len.

Phase 0 uses the exact sequential memory (slow but correct); keep batch/seq
small. Reports parameter count, memory-state bytes, and tokens/sec for fair
comparison against the dense baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis_lm.config import TrellisConfig
from trellis_lm.model import build_model


def _dtype(name):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


# ---------------- synthetic associative recall ----------------

def gen_recall_batch(B, n_pairs, n_keys, n_vals, device, gen):
    """Returns idx [B,L] and recall_pos (int). Layout per row:
    k1 v1 k2 v2 ... kN vN  QUERY  k_j  v_j     (predict v_j from k_j)
    keys in [0,n_keys); values in [n_keys, n_keys+n_vals); QUERY = last id.
    """
    K, V = n_keys, n_vals
    query_tok = K + V
    L = 2 * n_pairs + 3
    idx = torch.zeros(B, L, dtype=torch.long, device=device)
    for bi in range(B):
        keys = torch.randperm(K, generator=gen)[:n_pairs]
        vals = torch.randint(0, V, (n_pairs,), generator=gen) + K
        seq = []
        for i in range(n_pairs):
            seq += [int(keys[i]), int(vals[i])]
        j = int(torch.randint(0, n_pairs, (1,), generator=gen))
        seq += [query_tok, int(keys[j]), int(vals[j])]
        idx[bi] = torch.tensor(seq, device=device)
    recall_pos = 2 * n_pairs + 1  # position of k_j; its next-token target is v_j
    return idx, recall_pos


def run_recall(args, device, dt):
    n_keys, n_vals, n_pairs = 16, 16, args.n_pairs
    vocab = n_keys + n_vals + 1
    cfg = TrellisConfig(
        vocab_size=vocab, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_head=args.d_head, n_slots=args.n_slots,
        max_seq_len=2 * n_pairs + 8, dtype=args.dtype, activation=args.activation,
        alpha_mode=args.alpha_mode, beta_mode=args.beta_mode,
        forget_gate=not args.no_forget, use_short_conv_qk=not args.no_conv,
        exact_inner=not args.stale,
    )
    model = build_model(cfg, args.model).to(device)
    if args.dtype != "fp32":
        model = model.to(dt)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    gen = torch.Generator().manual_seed(args.seed)
    print(f"model={args.model} params={model.get_num_params():,} "
          f"mem_state_bytes/seq={model.memory_state_bytes(1)}", flush=True)
    model.train()
    hist = []
    for step in range(1, args.steps + 1):
        idx, rp = gen_recall_batch(args.batch, n_pairs, n_keys, n_vals, device, gen)
        # MQAR: supervise ONLY the answer position (logits[rp] predicts idx[rp+1]);
        # full-sequence CE drowns the recall signal under the value prior.
        labels = torch.full_like(idx, -100)
        labels[:, rp + 1] = idx[:, rp + 1]
        with torch.autocast(device_type=device.type, dtype=dt, enabled=args.dtype != "fp32"):
            logits, loss = model(idx, labels=labels, training=True)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                pred = logits[:, rp].argmax(-1)
                acc = (pred == idx[:, rp + 1]).float().mean().item()
            print(f"  step {step:5d}  loss {loss.item():.4f}  recall_acc {acc:.3f}", flush=True)
            hist.append({"step": step, "loss": loss.item(), "recall_acc": acc})
    return cfg, model, hist


# ---------------- TinyStories LM ----------------

def lm_batches(args, device):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    buf, batches = [], []
    for ex in ds:
        ids = tok(ex.get("text") or "").input_ids + [tok.eos_token_id]
        buf.extend(ids)
        while len(buf) >= args.seq_len:
            batches.append(buf[:args.seq_len]); buf = buf[args.seq_len:]
            if len(batches) >= args.steps * args.batch:
                return tok.vocab_size, batches
    return tok.vocab_size, batches


def run_lm(args, device, dt):
    vocab, packed = lm_batches(args, device)
    cfg = TrellisConfig(
        vocab_size=vocab, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_head=args.d_head, n_slots=args.n_slots,
        max_seq_len=args.seq_len, dtype=args.dtype, activation=args.activation,
        alpha_mode=args.alpha_mode, beta_mode=args.beta_mode,
        forget_gate=not args.no_forget, use_short_conv_qk=not args.no_conv,
        exact_inner=not args.stale,
    )
    model = build_model(cfg, args.model).to(device)
    if args.dtype != "fp32":
        model = model.to(dt)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"model={args.model} params={model.get_num_params():,} "
          f"mem_state_bytes/seq={model.memory_state_bytes(1)} seq_len={args.seq_len}", flush=True)
    model.train()
    hist = []
    t0 = time.time(); ntok = 0
    bi = 0
    for step in range(1, args.steps + 1):
        rows = packed[bi:bi + args.batch]; bi += args.batch
        if len(rows) < args.batch:
            break
        idx = torch.tensor(rows, device=device)
        with torch.autocast(device_type=device.type, dtype=dt, enabled=args.dtype != "fp32"):
            _, loss = model(idx, labels=idx, training=True)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ntok += idx.numel()
        if step % args.log_every == 0 or step == 1:
            tps = ntok / (time.time() - t0)
            ppl = math.exp(min(20, loss.item()))
            print(f"  step {step:5d}  loss {loss.item():.4f}  ppl {ppl:.2f}  tok/s {tps:.0f}", flush=True)
            hist.append({"step": step, "loss": loss.item(), "ppl": ppl, "tok_s": tps})
    return cfg, model, hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="recall", choices=["recall", "lm"])
    p.add_argument("--model", default="trellis", choices=["trellis", "dense"])
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--n_pairs", type=int, default=8)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_head", type=int, default=32)
    p.add_argument("--n_slots", type=int, default=64)
    p.add_argument("--activation", default="ln_silu")
    p.add_argument("--alpha_mode", default="linear")
    p.add_argument("--beta_mode", default="scalar_per_head")
    p.add_argument("--no_forget", action="store_true")
    p.add_argument("--no_conv", action="store_true")
    p.add_argument("--stale", action="store_true", help="stale-gradient inner step (exact_inner=False); fast path for long context")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--dtype", default="fp32", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--dataset", default="roneneldan/TinyStories")
    p.add_argument("--split", default="train")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--save_ckpt", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = _dtype(args.dtype)
    runner = run_recall if args.task == "recall" else run_lm
    cfg, model, hist = runner(args, device, dt)

    if args.save_ckpt:
        torch.save({"cfg": cfg.to_dict(), "model": model.state_dict(),
                    "kind": args.model}, args.save_ckpt)
        print(f"saved checkpoint -> {args.save_ckpt}", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"args": vars(args), "params": model.get_num_params(),
             "mem_state_bytes_per_seq": model.memory_state_bytes(1),
             "history": hist}, indent=2))
        print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

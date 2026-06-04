"""Firm up the Trellis-vs-dense matched-scratch comparison with held-out
validation PPL and multi-seed variance, on a real corpus.

One invocation = one "cell": a fixed (dataset, seq_len, mode), swept over
seeds x {trellis, dense}. For each run it trains from scratch on a packed
TRAIN token stream and reports perplexity on a DISJOINT held-out VAL stream
(the train PPL the bare trainer logs is not a fair model-selection signal).
Results (per-run + mean/std aggregate) are written as JSON so a bash driver
can sequence many cells into one background job and partial progress survives.

mode:
  seq      chunk_size=1, exact_inner=False  -- stale sequential, exact forward,
           1st-order grad. Trellis at full strength; use at short lengths.
  chunk16  chunk_size=16, chunk_refine=0    -- true-stale chunked (per-head beta),
           11-40x faster, +35-46% PPL handicap. Use at long lengths; a win here
           is a *handicapped* win (conservative).

Matched comparison: trellis and dense share d_model/n_layers/n_heads/d_head and
see the same tokens (steps x batch x seq_len); param counts are reported so the
size match is honest.
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


def pack_stream(ds_iter, tok, seq_len, n_rows, eos):
    """Pack a streamed text dataset into n_rows rows of seq_len gpt2 tokens."""
    buf, rows = [], []
    for ex in ds_iter:
        text = ex.get("text") or ""
        if not text:
            continue
        buf.extend(tok(text).input_ids + [eos])
        while len(buf) >= seq_len:
            rows.append(buf[:seq_len])
            buf = buf[seq_len:]
            if len(rows) >= n_rows:
                return rows
    return rows


def load_packed(dataset, seq_len, n_train, n_val, tok):
    """Return (train_rows, val_rows) packed from disjoint splits."""
    from datasets import load_dataset

    eos = tok.eos_token_id
    if dataset == "wikitext103":
        tr = load_dataset("wikitext", "wikitext-103-raw-v1", split="train",
                           streaming=True)
        va = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation",
                           streaming=True)
    elif dataset == "pg19":
        tr = load_dataset("pg19", split="train", streaming=True,
                           trust_remote_code=True)
        va = load_dataset("pg19", split="validation", streaming=True,
                          trust_remote_code=True)
    else:
        raise ValueError(dataset)
    train_rows = pack_stream(iter(tr), tok, seq_len, n_train, eos)
    val_rows = pack_stream(iter(va), tok, seq_len, n_val, eos)
    return train_rows, val_rows


def make_cfg(vocab, seq_len, mode, args):
    if mode == "seq":
        chunk_size, chunk_refine, stale = 1, 0, True
    elif mode == "chunk16":
        chunk_size, chunk_refine, stale = 16, 0, True
    else:
        raise ValueError(mode)
    return TrellisConfig(
        vocab_size=vocab, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_head=args.d_head, n_slots=args.n_slots,
        max_seq_len=seq_len, dtype="fp32", activation="ln_silu",
        alpha_mode="linear", beta_mode="scalar_per_head",
        forget_gate=True, use_short_conv_qk=True,
        exact_inner=not stale, chunk_size=chunk_size, chunk_refine=chunk_refine,
    )


@torch.no_grad()
def eval_ppl(model, val_rows, batch, device):
    model.eval()
    tot_loss, tot_tok = 0.0, 0
    for i in range(0, len(val_rows) - batch + 1, batch):
        idx = torch.tensor(val_rows[i:i + batch], device=device)
        _, loss = model(idx, labels=idx, training=False)
        ntok = idx.numel()
        tot_loss += loss.item() * ntok
        tot_tok += ntok
    return math.exp(min(20, tot_loss / max(1, tot_tok)))


def train_one(kind, seed, cfg, train_rows, val_rows, args, device):
    torch.manual_seed(seed)
    model = build_model(cfg, kind).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    nb = len(train_rows) // args.batch
    t0 = time.time()
    for step in range(args.steps):
        bi = (step % nb) * args.batch
        idx = torch.tensor(train_rows[bi:bi + args.batch], device=device)
        _, loss = model(idx, labels=idx, training=True)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 0 or (step + 1) % args.log_every == 0:
            print(f"    {kind} seed{seed} step {step + 1}/{args.steps} "
                  f"loss {loss.item():.3f}", flush=True)
    vppl = eval_ppl(model, val_rows, args.batch, device)
    return {
        "kind": kind, "seed": seed, "val_ppl": vppl,
        "params": model.get_num_params(),
        "mem_state_bytes_per_seq": model.memory_state_bytes(1)
        if hasattr(model, "memory_state_bytes") else None,
        "train_min": (time.time() - t0) / 60.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["wikitext103", "pg19"])
    p.add_argument("--seq_len", type=int, required=True)
    p.add_argument("--mode", required=True, choices=["seq", "chunk16"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--models", nargs="+", default=["trellis", "dense"])
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_head", type=int, default=64)
    p.add_argument("--n_slots", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--n_val", type=int, default=48)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    n_train = args.steps * args.batch + args.batch
    print(f"[{args.dataset} L{args.seq_len} {args.mode}] packing "
          f"{n_train} train + {args.n_val} val rows ...", flush=True)
    train_rows, val_rows = load_packed(args.dataset, args.seq_len, n_train,
                                       args.n_val, tok)
    print(f"  packed train={len(train_rows)} val={len(val_rows)}", flush=True)
    vocab = tok.vocab_size

    runs = []
    for kind in args.models:
        cfg = make_cfg(vocab, args.seq_len, args.mode, args)
        for seed in args.seeds:
            r = train_one(kind, seed, cfg, train_rows, val_rows, args, device)
            print(f"  -> {kind} seed{seed} val_ppl {r['val_ppl']:.2f} "
                  f"({r['params']:,} params, {r['train_min']:.1f} min)", flush=True)
            runs.append(r)
            _dump(args, train_rows, val_rows, runs)  # incremental
    _dump(args, train_rows, val_rows, runs, final=True)
    return 0


def _agg(runs):
    out = {}
    for kind in sorted({r["kind"] for r in runs}):
        ppls = [r["val_ppl"] for r in runs if r["kind"] == kind]
        mean = sum(ppls) / len(ppls)
        std = (sum((x - mean) ** 2 for x in ppls) / len(ppls)) ** 0.5
        out[kind] = {"val_ppl_mean": mean, "val_ppl_std": std, "n": len(ppls),
                     "ppls": ppls, "params": next(r["params"] for r in runs
                                                   if r["kind"] == kind)}
    if "trellis" in out and "dense" in out:
        out["trellis_vs_dense_pct"] = (
            100.0 * (out["trellis"]["val_ppl_mean"] - out["dense"]["val_ppl_mean"])
            / out["dense"]["val_ppl_mean"])
    return out


def _dump(args, train_rows, val_rows, runs, final=False):
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "dataset": args.dataset, "seq_len": args.seq_len, "mode": args.mode,
        "steps": args.steps, "batch": args.batch, "lr": args.lr,
        "seeds": args.seeds, "n_train_rows": len(train_rows),
        "n_val_rows": len(val_rows), "tokens_per_run": args.steps * args.batch
        * args.seq_len, "dims": {"d_model": args.d_model, "n_layers": args.n_layers,
        "n_heads": args.n_heads, "d_head": args.d_head, "n_slots": args.n_slots},
        "runs": runs, "aggregate": _agg(runs), "final": final,
    }, indent=2))
    if final:
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

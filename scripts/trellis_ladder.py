"""Trellis-vs-linear-cousins scaling ladder (the "does it cross over at scale" test).

Our toy 4-5M Trellis loses to Gated DeltaNet; the paper (arXiv:2512.23852)
claims Trellis WINS at 125M-1B. That is a testable scaling hypothesis, not a
conclusion. This driver climbs a parameter ladder with a *leading* training
recipe (AdamW betas 0.9/0.95, wd 0.1, linear warmup + cosine decay, grad clip,
bf16 autocast, near-Chinchilla tokens, tuned inner-lr gamma=0.1, chunk16
operator) and trains the same four models -- dense / DeltaNet / Gated DeltaNet /
Trellis -- at each rung on C4, reporting val PPL and the Trellis-minus-GatedDelta
gap vs model size. If that gap closes toward 125M the scale story holds; if it
stays flat the paper does not reproduce at the scale we can reach.

Runs rungs smallest-first and dumps a per-rung JSON the moment each rung
finishes, so the trend is visible as it builds and the run can be called early.

Example:
  python scripts/trellis_ladder.py --dataset c4 --seq_len 2048 \
      --rungs 256x4,384x6,512x8,640x10,768x12,1024x16 \
      --tokens_per_param 20 --out /root/trellis-ladder-results
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

from trellis_lm.config import TrellisConfig  # noqa: E402
from trellis_lm.model import build_model  # noqa: E402

KINDS = ["dense", "delta", "gated_delta", "trellis"]


def get_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    return tok


def stream_text(dataset: str, split: str):
    """Yield raw text strings from a streaming corpus, re-opening forever."""
    from datasets import load_dataset

    while True:
        if dataset == "c4":
            ds = load_dataset("allenai/c4", "en", split=split, streaming=True)
        elif dataset == "wikitext103":
            ds = load_dataset(
                "wikitext", "wikitext-103-raw-v1", split=split, streaming=True
            )
        elif dataset == "pile":
            ds = load_dataset(
                "monology/pile-uncopyrighted", split=split, streaming=True
            )
        else:
            raise ValueError(dataset)
        for ex in ds:
            t = ex.get("text") or ""
            if t:
                yield t
        if split != "train":
            return  # don't loop a finite val stream


def batch_stream(dataset, split, tok, seq_len, micro_batch):
    """Infinite generator of [micro_batch, seq_len] LongTensors packed from text."""
    eos = tok.eos_token_id
    buf, rows = [], []
    for text in stream_text(dataset, split):
        buf.extend(tok(text).input_ids + [eos])
        while len(buf) >= seq_len:
            rows.append(buf[:seq_len])
            buf = buf[seq_len:]
            if len(rows) >= micro_batch:
                yield torch.tensor(rows, dtype=torch.long)
                rows = []


def pack_val(dataset, tok, seq_len, n_rows):
    rows, buf = [], []
    eos = tok.eos_token_id
    for text in stream_text(dataset, "validation"):
        buf.extend(tok(text).input_ids + [eos])
        while len(buf) >= seq_len:
            rows.append(buf[:seq_len])
            buf = buf[seq_len:]
            if len(rows) >= n_rows:
                return torch.tensor(rows, dtype=torch.long)
    return torch.tensor(rows, dtype=torch.long)


def make_cfg(d_model, n_layers, seq_len, args):
    return TrellisConfig(
        vocab_size=args.vocab,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=d_model // 64,
        d_head=64,
        n_slots=args.n_slots,
        max_seq_len=seq_len,
        dtype="fp32",  # master weights fp32; compute is bf16 via autocast
        activation="ln_silu",
        alpha_mode="linear",
        beta_mode="scalar_per_head",
        forget_gate=True,
        use_short_conv_qk=True,
        gamma_init=args.gamma_init,
        exact_inner=False,  # chunk16 stale-gradient operator (fast)
        chunk_size=16,
        chunk_refine=0,
    )


def lr_at(step, total, peak, warmup, min_frac):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return peak * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


@torch.no_grad()
def eval_ppl(model, val, micro_batch, device):
    model.eval()
    tot_loss, tot_tok = 0.0, 0
    for i in range(0, val.size(0) - micro_batch + 1, micro_batch):
        idx = val[i : i + micro_batch].to(device)
        # fp32 throughout: bf16 is slower + ~41% inaccurate for the Trellis
        # memory (overhead-bound, sensitive recurrence), so the matched
        # comparison runs fp32 for every model -- fair and accurate.
        _, loss = model(idx, labels=idx, training=False)
        tot_loss += loss.float().item() * idx.numel()
        tot_tok += idx.numel()
    return math.exp(min(20, tot_loss / max(1, tot_tok)))


def train_one(kind, cfg, gen, val, total_tokens, args, device):
    torch.manual_seed(args.seed)
    model = build_model(cfg, kind).to(device)
    micro = args.micro_batch
    tok_per_micro = micro * cfg.max_seq_len
    accum = max(1, args.eff_tokens // tok_per_micro)
    total_steps = max(1, total_tokens // (accum * tok_per_micro))
    warmup = int(args.warmup_frac * total_steps)
    peak = args.base_lr * math.sqrt(512.0 / cfg.d_model)
    opt = torch.optim.AdamW(
        model.parameters(), lr=peak, betas=(0.9, 0.95), weight_decay=0.1
    )
    model.train()
    t0 = time.time()
    seen_tokens = 0
    for step in range(total_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total_steps, peak, warmup, args.min_lr_frac)
        opt.zero_grad()
        loss_val = 0.0
        for _ in range(accum):
            idx = next(gen).to(device)
            _, loss = model(idx, labels=idx, training=True)  # fp32 (see eval_ppl)
            (loss / accum).backward()
            loss_val += loss.float().item() / accum
            seen_tokens += idx.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 0 or (step + 1) % args.log_every == 0:
            print(
                f"    {kind} step {step + 1}/{total_steps} "
                f"lr {opt.param_groups[0]['lr']:.2e} loss {loss_val:.3f} "
                f"({seen_tokens / 1e6:.0f}M tok, {(time.time() - t0) / 60:.1f} min)",
                flush=True,
            )
    ppl = eval_ppl(model, val, micro, device)
    total_params = sum(p.numel() for p in model.parameters())
    res = {
        "kind": kind,
        "val_ppl": ppl,
        "params_total": total_params,
        "params_nonembed": model.get_num_params(),
        "tokens_trained": seen_tokens,
        "total_steps": total_steps,
        "peak_lr": peak,
        "train_min": (time.time() - t0) / 60.0,
    }
    del model, opt
    torch.cuda.empty_cache()
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="c4", choices=["c4", "wikitext103", "pile"])
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument(
        "--rungs",
        default="256x4,384x6,512x8,640x10,768x12,1024x16",
        help="comma list of d_modelxn_layers",
    )
    p.add_argument("--kinds", default=",".join(KINDS))
    p.add_argument("--tokens_per_param", type=float, default=20.0)
    p.add_argument("--base_lr", type=float, default=3e-3)
    p.add_argument("--gamma_init", type=float, default=0.1)
    p.add_argument("--n_slots", type=int, default=64)
    p.add_argument("--eff_tokens", type=int, default=262144)
    p.add_argument("--micro_batch", type=int, default=8)
    p.add_argument("--warmup_frac", type=float, default=0.02)
    p.add_argument("--min_lr_frac", type=float, default=0.1)
    p.add_argument("--val_rows", type=int, default=128)
    p.add_argument("--vocab", type=int, default=50257)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = get_tokenizer()
    kinds = args.kinds.split(",")
    print(f"packing {args.val_rows} val rows from {args.dataset} ...", flush=True)
    val = pack_val(args.dataset, tok, args.seq_len, args.val_rows)
    print(f"  val rows: {val.size(0)}", flush=True)

    for rung in args.rungs.split(","):
        d_model, n_layers = (int(x) for x in rung.lower().split("x"))
        cfg = make_cfg(d_model, n_layers, args.seq_len, args)
        probe = build_model(cfg, "trellis")
        ptot = sum(p.numel() for p in probe.parameters())
        del probe
        total_tokens = int(args.tokens_per_param * ptot)
        print(
            f"\n=== RUNG d{d_model} L{n_layers}: {ptot / 1e6:.1f}M total params, "
            f"{total_tokens / 1e9:.2f}B tokens budget ===",
            flush=True,
        )
        gen = batch_stream(args.dataset, "train", tok, args.seq_len, args.micro_batch)
        rung_res = {
            "d_model": d_model,
            "n_layers": n_layers,
            "seq_len": args.seq_len,
            "dataset": args.dataset,
            "tokens_per_param": args.tokens_per_param,
            "total_tokens_budget": total_tokens,
            "runs": [],
        }
        for kind in kinds:
            r = train_one(kind, cfg, gen, val, total_tokens, args, device)
            print(
                f"  -> {kind}: ppl {r['val_ppl']:.3f} "
                f"({r['params_total'] / 1e6:.1f}M, {r['train_min']:.1f} min)",
                flush=True,
            )
            rung_res["runs"].append(r)
        ppls = {r["kind"]: r["val_ppl"] for r in rung_res["runs"]}
        if "trellis" in ppls and "gated_delta" in ppls:
            gap = ppls["trellis"] - ppls["gated_delta"]
            rung_res["trellis_minus_gated"] = gap
            print(
                f"  RUNG SUMMARY d{d_model}: trellis {ppls['trellis']:.2f} "
                f"vs gated_delta {ppls['gated_delta']:.2f} -> gap {gap:+.2f} "
                f"({'trellis wins' if gap < 0 else 'gated wins'})",
                flush=True,
            )
        fn = out / f"rung_d{d_model}_L{n_layers}.json"
        fn.write_text(json.dumps(rung_res, indent=2))
        print(f"  wrote {fn}", flush=True)
    print("LADDER_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()

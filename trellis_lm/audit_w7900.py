"""Trellis reconstruction-faithfulness audit (ChatGPT-Pro directed).

Cheap d256/L4 screen on the W7900: does fixing the paper-faithful output
path (PostNorm -> GeLU gate -> out_proj), the forget-gate retention init
(beta near 1, not 0.5), and the value-pass readout phi pull our Trellis out
of the ~205-vs-79 hole BEFORE we spend a cloud GPU dollar on scaling.

All configs see byte-identical tokenized TinyStories (packed once, cached,
reused), same optimizer/seed/budget; only the Trellis knobs differ. We report
held-out val PPL and a cheap forget-gate-retention diagnostic. delta = the
hand-rolled (bmm) DeltaNet floor (fla has no ROCm path).

Run on prune: HIP_VISIBLE_DEVICES=0, ~/envs/w7900-ml.
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


# ---- the audit matrix: (label, kind, knob-overrides) -----------------------
# kind "trellis" uses the overrides; "delta" is the hand-rolled DeltaNet floor.
MATRIX = [
    (
        "trellis_current",
        "trellis",
        dict(output_path="current", beta_init=0.5, value_readout_act="none"),
    ),
    (
        "trellis_current_beta99",
        "trellis",
        dict(output_path="current", beta_init=0.99, value_readout_act="none"),
    ),
    (
        "trellis_paper",
        "trellis",
        dict(output_path="paper", beta_init=0.5, value_readout_act="none"),
    ),
    (
        "trellis_paper_beta99",
        "trellis",
        dict(output_path="paper", beta_init=0.99, value_readout_act="none"),
    ),
    (
        "trellis_paper_beta99_valphi",
        "trellis",
        dict(output_path="paper", beta_init=0.99, value_readout_act="ln_silu"),
    ),
    ("delta_floor", "delta", dict()),
]


def build_data(args, device):
    """Pack TinyStories train + val once; cache the token ids to disk so reruns
    and every config share byte-identical data."""
    from transformers import AutoTokenizer

    cache = Path(args.cache)
    if cache.exists():
        blob = torch.load(cache)
        print(f"loaded packed data from {cache}", flush=True)
        return blob["vocab"], blob["train"], blob["val"]

    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    need_train = args.steps * args.batch * args.seq_len
    need_val = args.val_rows * args.seq_len

    def pack(split, n_tokens):
        buf, rows = [], []
        ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
        for ex in ds:
            buf.extend(tok(ex.get("text") or "").input_ids + [tok.eos_token_id])
            while len(buf) >= args.seq_len:
                rows.append(buf[: args.seq_len])
                buf = buf[args.seq_len :]
            if len(rows) * args.seq_len >= n_tokens:
                return rows
        return rows

    train = pack("train", need_train)
    val = pack("validation", need_val)
    blob = {"vocab": tok.vocab_size, "train": train, "val": val}
    torch.save(blob, cache)
    print(
        f"packed+cached train={len(train)} val={len(val)} rows -> {cache}", flush=True
    )
    return tok.vocab_size, train, val


def make_cfg(args, vocab, overrides):
    base = dict(
        vocab_size=vocab,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.d_model // 64,
        d_head=64,
        n_slots=args.n_slots,
        max_seq_len=args.seq_len,
        dtype="bf16",
        activation="ln_silu",
        alpha_mode="linear",
        beta_mode="scalar_per_head",
        chunk_size=args.chunk_size,
        chunk_refine=0,
    )
    base.update(overrides)
    return TrellisConfig(**base)


@torch.no_grad()
def eval_val(model, val_rows, args, device):
    model.eval()
    tot, ntok = 0.0, 0
    for i in range(0, len(val_rows), args.batch):
        rows = val_rows[i : i + args.batch]
        if not rows:
            break
        idx = torch.tensor(rows, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, loss = model(idx, labels=idx, training=False)
        n = idx[:, 1:].numel()
        tot += loss.item() * n
        ntok += n
    model.train()
    return math.exp(min(20, tot / max(1, ntok)))


def beta_retention(model):
    """Cheap data-independent retention proxy: mean sigmoid(beta_proj.bias)
    across Trellis layers (where the forget gate sits at init/after training)."""
    vals = []
    for m in model.modules():
        if hasattr(m, "beta_proj") and m.beta_proj.bias is not None:
            vals.append(torch.sigmoid(m.beta_proj.bias.detach()).mean().item())
    return sum(vals) / len(vals) if vals else None


def train_one(label, kind, overrides, args, vocab, train_rows, val_rows, device):
    torch.manual_seed(args.seed)
    cfg = make_cfg(args, vocab, overrides)
    # fp32 master weights + bf16 autocast (matches the clean ladder run). The
    # Trellis mixer forces its decay/state path to fp32 internally, so casting
    # params to bf16 would clash with that fp32 path.
    model = build_model(cfg, kind).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    warmup = max(1, int(0.05 * args.steps))

    def lr_lambda(s):  # linear warmup -> cosine decay to 10% (matches the ladder)
        if s < warmup:
            return (s + 1) / warmup
        p = (s - warmup) / max(1, args.steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    nparam = model.get_num_params()
    beta0 = beta_retention(model)
    model.train()
    t0 = time.time()
    last = None
    for step in range(args.steps):
        rows = train_rows[(step * args.batch) % (len(train_rows) - args.batch) :][
            : args.batch
        ]
        idx = torch.tensor(rows, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, loss = model(idx, labels=idx, training=True)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        last = loss.item()
        if step % args.log_every == 0:
            print(
                f"  [{label}] step {step:4d} loss {last:.4f} ppl {math.exp(min(20,last)):.1f}",
                flush=True,
            )
    dt = time.time() - t0
    val_ppl = eval_val(model, val_rows, args, device)
    beta1 = beta_retention(model)
    rec = dict(
        label=label,
        kind=kind,
        overrides=overrides,
        params=nparam,
        train_loss=last,
        val_ppl=val_ppl,
        beta_init_mean=beta0,
        beta_final_mean=beta1,
        train_sec=round(dt, 1),
    )
    print(
        f"== {label}: val_ppl {val_ppl:.2f}  beta {beta0:.3f}->{beta1 and round(beta1,3)}  ({dt:.0f}s)",
        flush=True,
    )
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=700)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_slots", type=int, default=64)
    p.add_argument("--chunk_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--val_rows", type=int, default=64)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", default="/tmp/trellis_audit_data.pt")
    p.add_argument("--out", default="/tmp/trellis_audit_results.json")
    p.add_argument("--only", default=None, help="comma-list of labels to run")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'})",
        flush=True,
    )
    vocab, train_rows, val_rows = build_data(args, device)

    matrix = MATRIX
    if args.only:
        want = set(args.only.split(","))
        matrix = [m for m in MATRIX if m[0] in want]

    results = []
    for label, kind, ov in matrix:
        try:
            results.append(
                train_one(label, kind, ov, args, vocab, train_rows, val_rows, device)
            )
        except Exception as e:  # keep the screen going if one config dies
            import traceback

            traceback.print_exc()
            results.append(dict(label=label, kind=kind, overrides=ov, error=str(e)))
        Path(args.out).write_text(json.dumps(results, indent=2))

    print("\n=== AUDIT SUMMARY (val PPL, lower better) ===", flush=True)
    for r in sorted([x for x in results if "val_ppl" in x], key=lambda x: x["val_ppl"]):
        print(
            f"  {r['val_ppl']:8.2f}  {r['label']:32s} beta {r.get('beta_init_mean')}->{r.get('beta_final_mean')}",
            flush=True,
        )
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Trellis fidelity-closure sweep (the linear-attention page's itemized fixes).

Self-consistent comparison at one matched protocol (d256/L4, C4, fixed
token budget, same seed): does turning the paper-faithful toggles on, and the
same-shell identity-phi control, change the verdict (our Trellis 152.54 losing
to fla Gated DeltaNet)? Each config trains from scratch and is scored on a fixed
held-out C4 slice, so absolute PPL won't match the doc's 352M-token run but the
ordering is fair. The fla Gated-DeltaNet reference is re-run at the SAME budget
as the anchor (not the doc's number).

The key questions:
  - identity-phi vs ln_silu: does the nonlinear write help in our harness?
  - paper-faithful (ln_silu + output_path=paper + value_readout + beta=0.9):
    do the turned-on defaults close the gap?
  - softmax-matched (alpha=softmax + phi=softmax): the simplex-objective variant
    (alpha=softmax + phi=ln_silu explodes -- see the alpha/phi pairing finding).

Usage:
  python trellis_phi_sweep.py --train-tokens 60000000 --seq-len 2048 \
      --batch 8 --chunk-size 16 --dataset allenai/c4 --c4-config en \
      --out /data/knlp-key-results/trellis-fidelity-20260624
"""
import argparse, json, math, sys, time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trellis_lm.config import TrellisConfig
from trellis_lm.model import build_model

# (label, kind, cfg-overrides). Trellis configs share the d256/L4 geometry.
# fla isn't installable on ROCm/W7900, so the anchors here are the FLA-FREE
# hand-rolled DeltaNet/Gated-DeltaNet (under-featured vs the paper's fla
# reference -- the doc's fla GDN 78.49 stays the external reference). They give a
# same-budget anchor for the Trellis-internal ablation, which is the point here.
CONFIGS = [
    ("gated_delta_handroll", "gated_delta", {}),                    # GDN anchor (fla-free)
    ("deltanet_handroll", "delta", {}),                            # DeltaNet anchor (fla-free)
    ("trellis_ln_silu", "trellis", dict(activation="ln_silu")),     # our default
    # matched-LR control vs identity (identity only stable at lr 3e-4): isolates
    # the phi nonlinearity from the LR difference.
    ("trellis_ln_silu_lr3e4", "trellis", dict(activation="ln_silu", lr=3e-4)),
    ("trellis_identity", "trellis", dict(activation="identity")),   # delta-rule control
    # identity (delta rule) diverges at lr 3e-3 chunk16 (phi=identity removes the
    # LN that bounds u=Mw-alpha -> state blows up). Stabilized variants:
    ("trellis_identity_lr1e3", "trellis", dict(activation="identity", lr=1e-3)),
    ("trellis_identity_lr3e4", "trellis", dict(activation="identity", lr=3e-4)),
    ("trellis_identity_exact", "trellis",
     dict(activation="identity", lr=1e-3, chunk_size=1)),
    ("trellis_paper_stable", "trellis",
     dict(activation="ln_silu", output_path="paper",
          value_readout_act="ln_silu", beta_init=0.9)),             # fixes on
    ("trellis_softmax_matched", "trellis",
     dict(activation="softmax", alpha_mode="softmax")),             # simplex objective
]


def get_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    return tok


def pack_stream(dataset, c4_config, split, seq_len, n_seqs, tok, seed=0):
    from datasets import load_dataset
    kw = {"streaming": True}
    if c4_config:
        ds = load_dataset(dataset, c4_config, split=split, **kw)
    else:
        ds = load_dataset(dataset, split=split, **kw)
    out, buf = [], []
    for ex in ds:
        t = ex.get("text") or ex.get("content") or ""
        if not t:
            continue
        buf.extend(tok(t).input_ids + [tok.eos_token_id or 0])
        while len(buf) >= seq_len:
            out.append(buf[:seq_len]); buf = buf[seq_len:]
            if len(out) >= n_seqs:
                return out
    return out


@torch.no_grad()
def val_ppl(model, val_seqs, device, batch, dt):
    model.eval()
    tot, ntok = 0.0, 0
    for i in range(0, len(val_seqs), batch):
        rows = val_seqs[i:i + batch]
        if not rows:
            break
        idx = torch.tensor(rows, device=device)
        with torch.autocast(device_type=device.type, dtype=dt, enabled=dt != torch.float32):
            _, loss = model(idx, labels=idx, training=False)
        tot += loss.item() * len(rows); ntok += len(rows)
    model.train()
    return math.exp(min(20, tot / max(1, ntok)))


def train_one(label, kind, ov, args, train_seqs, val_seqs, device, dt):
    torch.manual_seed(args.seed)
    ov = dict(ov)  # copy: don't mutate the module-level CONFIGS
    lr = ov.pop("lr", args.lr)            # per-config LR override
    chunk = ov.pop("chunk_size", args.chunk_size)
    base = dict(
        vocab_size=50257, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_head=args.d_head, n_slots=args.n_slots,
        max_seq_len=args.seq_len, dtype="bf16",
        chunk_size=chunk, exact_inner=(chunk <= 1),
    )
    base.update(ov)
    cfg = TrellisConfig(**base)
    # fp32 master weights + bf16 autocast (the doc's regime). The Trellis mixer
    # forces the recurrence/decay to fp32 internally via autocast(enabled=False),
    # so the model must stay fp32 -- casting it to bf16 mismatches the float()
    # beta/gamma path. autocast (below) gives the bf16 matmul speed.
    model = build_model(cfg, kind).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    nparams = model.get_num_params()
    steps = max(1, args.train_tokens // (args.batch * args.seq_len))
    hist = []
    t0 = time.time(); ntok = 0; bi = 0
    model.train()
    for step in range(1, steps + 1):
        rows = train_seqs[bi:bi + args.batch]; bi += args.batch
        if len(rows) < args.batch:
            bi = 0; rows = train_seqs[bi:bi + args.batch]; bi += args.batch
        idx = torch.tensor(rows, device=device)
        with torch.autocast(device_type=device.type, dtype=dt, enabled=dt != torch.float32):
            _, loss = model(idx, labels=idx, training=True)
        opt.zero_grad(); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        ntok += idx.numel()
        if step % args.log_every == 0 or step == 1:
            tps = ntok / (time.time() - t0)
            print(f"    [{label}] step {step}/{steps} loss {loss.item():.4f} "
                  f"gnorm {gnorm:.2f} tok/s {tps:.0f}", flush=True)
            hist.append({"step": step, "loss": loss.item(), "tok_s": tps})
        if not math.isfinite(loss.item()):
            print(f"    [{label}] NON-FINITE loss at step {step} -- aborting config", flush=True)
            return {"label": label, "kind": kind, "overrides": ov,
                    "params": nparams, "status": "diverged", "history": hist}
    vp = val_ppl(model, val_seqs, device, args.batch, dt)
    wall = time.time() - t0
    print(f"  [{label}] DONE val_ppl={vp:.2f} params={nparams} {wall:.0f}s", flush=True)
    return {"label": label, "kind": kind, "overrides": ov, "params": nparams,
            "val_ppl": round(vp, 3), "train_steps": steps, "wall_s": round(wall, 1),
            "status": "ok", "history": hist}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-tokens", type=int, default=60_000_000)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-head", type=int, default=64)
    ap.add_argument("--n-slots", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--dataset", default="allenai/c4")
    ap.add_argument("--c4-config", default="en")
    ap.add_argument("--val-seqs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--only", default=None, help="comma-list of labels to run")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.bfloat16
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    tok = get_tokenizer()
    n_train = args.train_tokens // args.seq_len + args.batch + 1
    print(f"[data] streaming {args.dataset}/{args.c4_config} "
          f"train_seqs~{n_train} val_seqs={args.val_seqs} seq_len={args.seq_len}", flush=True)
    train_seqs = pack_stream(args.dataset, args.c4_config, "train", args.seq_len,
                             n_train, tok, args.seed)
    val_seqs = pack_stream(args.dataset, args.c4_config, "validation", args.seq_len,
                           args.val_seqs, tok, args.seed)
    print(f"[data] got train={len(train_seqs)} val={len(val_seqs)}", flush=True)

    only = set(args.only.split(",")) if args.only else None
    results = []
    for label, kind, ov in CONFIGS:
        if only and label not in only:
            continue
        print(f"=== {label} ({kind}) {ov} ===", flush=True)
        try:
            r = train_one(label, kind, ov, args, train_seqs, val_seqs, device, dt)
        except Exception as e:
            import traceback; traceback.print_exc()
            r = {"label": label, "kind": kind, "overrides": ov,
                 "status": "error", "error": str(e)}
        results.append(r)
        (outdir / "SWEEP_RESULTS.json").write_text(json.dumps(
            {"args": vars(args), "results": results}, indent=2))
    print("\n=== SWEEP SUMMARY (val PPL, lower=better) ===")
    for r in sorted(results, key=lambda x: x.get("val_ppl", 9e9)):
        print(f"  {r['label']:26s} {r.get('status'):9s} "
              f"val_ppl={r.get('val_ppl','-')} params={r.get('params','-')}")
    print(f"[sweep] -> {outdir}/SWEEP_RESULTS.json")


if __name__ == "__main__":
    main()

"""Trellis bounded-memory CAPACITY test — the gate for Trellis-for-KRI.

The decisive, LM-free falsification (Codex gpt-5.5, 2026-06): can a Trellis
bounded memory store and cleanly retrieve MORE than one fact per unit memory at
matched budget? If not, using it as a "residual / side-channel" memory for
dropped KV cannot beat simply keeping more exact KV at the same budget, and
Trellis-for-KRI is dead (ship Trellis-from-scratch instead).

Task: symmetric associative recall (NO marker — the hard case). A row is
  k1 v1 k2 v2 ... kD vD  QUERY  k_j   (predict v_j)
keys distinct, values from a V-vocab, the queried pair chosen uniformly. With no
marker the write-side cannot know which pair will be queried, so the m-slot
memory must behave like an associative table over all D pairs.

Matched-budget reference. A 1-layer Trellis memory state is 2*H*m*d_head floats
(two passes); an exact (key,value) pair costs 2*H*d_head floats, so the same
budget buys C = m exact pairs. A query-agnostic exact-retention policy that keeps
m of D pairs recalls min(1, m/D). The Trellis memory must BEAT that line, not
merely beat chance (1/V), for the side-channel to be worth more than retaining
more exact KV. We sweep D/m and print recall vs the m/D line.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis_lm.config import TrellisConfig
from trellis_lm.model import build_model


def gen_batch(B, D, n_keys, n_vals, device, gen, mark=False):
    """Rows: k1 v1 ... kD vD [MARK?] QUERY k_j  -> predict v_j at last position."""
    K, V = n_keys, n_vals
    query_tok = K + V
    mark_tok = K + V + 1 if mark else None
    L = 2 * D + 3
    idx = torch.zeros(B, L, dtype=torch.long, device=device)
    for bi in range(B):
        keys = torch.randperm(K, generator=gen)[:D]
        vals = torch.randint(0, V, (D,), generator=gen) + K
        seq = []
        j = int(torch.randint(0, D, (1,), generator=gen))
        for i in range(D):
            seq += [int(keys[i]), int(vals[i])]
        seq += [query_tok, int(keys[j]), int(vals[j])]
        idx[bi] = torch.tensor(seq, device=device)
    recall_pos = 2 * D + 1            # position of k_j; predict its next token v_j
    return idx, recall_pos


def run_cell(m, D, args, device):
    n_keys = max(D, 4)
    n_vals = args.n_vals
    vocab = n_keys + n_vals + 2
    cfg = TrellisConfig(
        vocab_size=vocab, d_model=args.d_model, n_layers=1,
        n_heads=args.n_heads, d_head=args.d_head, n_slots=m,
        max_seq_len=2 * D + 8, dtype="fp32", activation="ln_silu",
        alpha_mode="linear", beta_mode="scalar_per_head",
        forget_gate=True, use_short_conv_qk=True,
        exact_inner=False, chunk_size=args.chunk_size, chunk_refine=0,
    )
    model = build_model(cfg, "trellis").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    gen = torch.Generator().manual_seed(args.seed)
    model.train()
    for step in range(args.steps):
        idx, rp = gen_batch(args.batch, D, n_keys, n_vals, device, gen)
        labels = torch.full_like(idx, -100)
        labels[:, rp + 1] = idx[:, rp + 1]
        _, loss = model(idx, labels=labels, training=True)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    # eval recall accuracy over fresh batches
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(args.eval_batches):
            idx, rp = gen_batch(args.batch, D, n_keys, n_vals, device, gen)
            logits, _ = model(idx, training=False)
            pred = logits[:, rp].argmax(-1)
            correct += (pred == idx[:, rp + 1]).sum().item()
            total += idx.shape[0]
    acc = correct / total
    exact_line = min(1.0, m / D)
    chance = 1.0 / n_vals
    return {"slots": m, "pairs": D, "ratio": D / m, "recall_acc": acc,
            "exact_retention_line": exact_line, "chance": chance,
            "beats_exact": acc > exact_line, "params": model.get_num_params()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slots", type=int, nargs="+", default=[16, 64])
    p.add_argument("--ratios", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--chunk_size", type=int, default=16,
                   help="fast chunked operator; capacity is a memory property")
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_head", type=int, default=32)
    p.add_argument("--n_vals", type=int, default=64)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--eval_batches", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  slots={args.slots} ratios={args.ratios}", flush=True)
    print(f"{'slots':>5} {'pairs':>5} {'D/m':>4} {'recall':>7} {'exact m/D':>9} "
          f"{'beats?':>7}", flush=True)
    runs = []
    for m in args.slots:
        for r in args.ratios:
            D = m * r
            res = run_cell(m, D, args, device)
            runs.append(res)
            mark = "YES" if res["beats_exact"] else "no"
            print(f"{m:>5} {D:>5} {r:>4} {res['recall_acc']:>7.3f} "
                  f"{res['exact_retention_line']:>9.3f} {mark:>7}", flush=True)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(
                {"args": vars(args), "runs": runs}, indent=2))
    n_beat = sum(1 for x in runs if x["beats_exact"] and x["ratio"] > 1)
    n_tot = sum(1 for x in runs if x["ratio"] > 1)
    print(f"\nVERDICT: beats matched-budget exact retention in {n_beat}/{n_tot} "
          f"compression cells (D/m>1). Gate PASS needs a clear majority at D/m>=4.",
          flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

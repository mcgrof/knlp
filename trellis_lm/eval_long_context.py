"""Long-context associative recall (RULER/NIAH-style) for a trained model.

Builds key/value pairs, then a variable amount of random filler ("distractors")
to push the queried pair far from the query, then a repeated key; measures
exact-match recall accuracy vs context length. Includes a repeated-key
condition (the same key appears twice with different values — forget gates can
hurt here). Memory state is bounded for Trellis; the eval reports accuracy vs
the bytes the model needs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis_lm.eval_ppl import load_ckpt


def gen(B, n_pairs, filler, n_keys, n_vals, device, gen_, repeated_key=False):
    K, V = n_keys, n_vals
    query_tok, filler_tok = K + V, K + V + 1
    rows, ans_pos = [], []
    for _ in range(B):
        keys = torch.randperm(K, generator=gen_)[:n_pairs]
        vals = torch.randint(0, V, (n_pairs,), generator=gen_) + K
        seq = []
        for i in range(n_pairs):
            seq += [int(keys[i]), int(vals[i])]
            if filler:
                seq += [filler_tok] * int(torch.randint(0, filler + 1, (1,), generator=gen_))
        j = int(torch.randint(0, n_pairs, (1,), generator=gen_))
        if repeated_key:  # same key reappears with a NEW value mid-sequence
            seq += [int(keys[j]), int(torch.randint(0, V, (1,), generator=gen_)) + K]
        seq += [query_tok, int(keys[j])]
        ans = int(vals[j])
        rows.append((seq, ans))
    L = max(len(s) for s, _ in rows)
    idx = torch.full((B, L + 1), filler_tok, dtype=torch.long, device=device)
    pos = []
    for bi, (s, ans) in enumerate(rows):
        idx[bi, :len(s)] = torch.tensor(s, device=device)
        idx[bi, len(s)] = ans
        pos.append(len(s) - 1)  # position of the final key; predict ans next
    return idx, pos


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n_pairs", type=int, default=8)
    p.add_argument("--fillers", default="0,8,32,64")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--repeated_key", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None)
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, model = load_ckpt(a.ckpt, device)
    n_keys, n_vals = 16, 16
    g = torch.Generator().manual_seed(a.seed)
    rows = []
    for filler in (int(x) for x in a.fillers.split(",")):
        idx, pos = gen(a.batch, a.n_pairs, filler, n_keys, n_vals, device, g, a.repeated_key)
        logits, _ = model(idx, training=False)
        correct = 0
        for bi, p_ in enumerate(pos):
            if int(logits[bi, p_].argmax()) == int(idx[bi, p_ + 1]):
                correct += 1
        acc = correct / a.batch
        approx_len = idx.shape[1]
        rows.append({"filler": filler, "approx_ctx": approx_len, "recall_acc": acc,
                     "mem_state_bytes": model.memory_state_bytes(1),
                     "repeated_key": a.repeated_key})
        print(f"filler={filler:4d} ctx~{approx_len:5d}  recall_acc={acc:.3f}", flush=True)
    if a.output:
        Path(a.output).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

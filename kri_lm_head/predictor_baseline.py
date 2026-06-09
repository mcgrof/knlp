#!/usr/bin/env python3
"""Baseline LM-head cluster-routing predictor (deployable, no oracle).

Trains a tiny bottleneck router that, from the hidden state the LM head sees,
predicts which clusters contain the next token, then measures how far that
deployable predictor falls short of the H6 oracle ceiling. This is the gauge for
how much of the gap the PEFT program (H1/H2) has to close: the oracle ranks
clusters by the dense model's own answer mass (cheating), while this router sees
only the hidden state.

Router: h(d) -> Linear(d,B) -> GELU -> Linear(B,C), trained with cross-entropy
to the cluster that holds the true next token. Eval reports, at each m, the
fetch fraction, true-next coverage, and top-10 coverage of the predicted top-m
clusters, plus the gap to the oracle (read from the H6 scan JSON if present).

Usage:
  python3 kri_lm_head/predictor_baseline.py --model EleutherAI/pythia-410m \
      --oracle-json OUT/pythia-410m.json --out OUT/pred_pythia-410m.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from h6_oracle_scan import build_idblock, build_kmeans_l2


def load_split(tok, split, num_seqs, seq_len):
    from datasets import load_dataset

    cfg = {"train": "train", "val": "test"}[split]
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=cfg)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    enc = tok(text, return_tensors="pt").input_ids[0]
    need = num_seqs * seq_len
    if enc.numel() < need:
        num_seqs = enc.numel() // seq_len
        need = num_seqs * seq_len
    return enc[:need].view(num_seqs, seq_len)


@torch.no_grad()
def capture(model, head, input_ids, device, pos_cap=None):
    """Capture the hidden vector the LM head consumes, the true next-token id,
    and the dense top-10 token ids, for every position."""
    grab = {}

    def pre_hook(mod, inp):
        grab["h"] = inp[0].detach()

    handle = head.register_forward_pre_hook(pre_hook)
    Hs, Ys, T10 = [], [], []
    for b in range(input_ids.shape[0]):
        ids = input_ids[b : b + 1].to(device)
        logits = model(ids).logits[0, :-1, :]
        hid = grab["h"][0, :-1, :]  # [T-1, d] -- exactly what the head saw
        Hs.append(hid.float().cpu())
        Ys.append(ids[0, 1:].cpu())
        T10.append(logits.topk(10, dim=-1).indices.cpu())
    handle.remove()
    H = torch.cat(Hs)
    Y = torch.cat(Ys)
    T = torch.cat(T10)
    if pos_cap and H.shape[0] > pos_cap:
        H, Y, T = H[:pos_cap], Y[:pos_cap], T[:pos_cap]
    return H, Y, T


class Router(nn.Module):
    def __init__(self, d, B, C):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, B), nn.GELU(), nn.Linear(B, C))

    def forward(self, x):
        return self.net(x)


def train_router(H, cl_target, d, C, B, device, epochs, bs, lr, seed):
    torch.manual_seed(seed)
    r = Router(d, B, C).to(device)
    opt = torch.optim.AdamW(r.parameters(), lr=lr, weight_decay=1e-4)
    Hd = H.to(device)
    td = cl_target.to(device)
    N = Hd.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(N, device=device)
        for s in range(0, N, bs):
            idx = perm[s : s + bs]
            loss = F.cross_entropy(r(Hd[idx]), td[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return r


@torch.no_grad()
def eval_router(r, H, Y, T10, cluster_of, C, m_list, device, V):
    cluster_of = cluster_of.to(device)
    cl_sizes = torch.bincount(cluster_of, minlength=C).float()
    Hd = H.to(device)
    y_cl = cluster_of[Y.to(device)]
    t10_cl = cluster_of[T10.to(device)]
    N = Hd.shape[0]
    scores = r(Hd)
    res = {}
    for m in m_list:
        sel = scores.topk(m, dim=-1).indices
        sel_mask = torch.zeros(N, C, dtype=torch.bool, device=device)
        sel_mask.scatter_(1, sel, True)
        fetch = (sel_mask.float() * cl_sizes).sum(1) / V
        tn = sel_mask.gather(1, y_cl.unsqueeze(1)).squeeze(1)
        t10 = sel_mask.gather(1, t10_cl).all(1)
        res[m] = {
            "fetch": fetch.mean().item(),
            "truenext": tn.float().mean().item(),
            "top10": t10.float().mean().item(),
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--train-seqs", type=int, default=400)
    ap.add_argument("--val-seqs", type=int, default=150)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--pos-cap", type=int, default=250000)
    ap.add_argument("--clusters", type=int, default=256)
    ap.add_argument("--m", default="8,16,32,64")
    ap.add_argument("--methods", default="idblock")
    ap.add_argument("--bottlenecks", default="32,64")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--oracle-json", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(device).eval()
    head = model.get_output_embeddings()
    W_U = head.weight
    V, d = W_U.shape
    C = args.clusters
    m_list = [int(x) for x in args.m.split(",")]
    methods = args.methods.split(",")
    bottlenecks = [int(x) for x in args.bottlenecks.split(",")]

    tr_ids = load_split(tok, "train", args.train_seqs, args.seq_len)
    va_ids = load_split(tok, "val", args.val_seqs, args.seq_len)
    print(f"[{args.model}] V={V} d={d} capturing train/val ...", flush=True)
    Htr, Ytr, _ = capture(model, head, tr_ids, device, pos_cap=args.pos_cap)
    Hva, Yva, T10va = capture(model, head, va_ids, device, pos_cap=args.pos_cap)
    print(f"  train_pos={Htr.shape[0]} val_pos={Hva.shape[0]}", flush=True)

    oracle = {}
    if args.oracle_json and Path(args.oracle_json).exists():
        oracle = json.loads(Path(args.oracle_json).read_text())

    result = {
        "model": args.model,
        "vocab_size": int(V),
        "d_model": int(d),
        "C": C,
        "m_list": m_list,
        "train_pos": int(Htr.shape[0]),
        "val_pos": int(Hva.shape[0]),
        "methods": {},
    }

    for method in methods:
        cof = (
            build_idblock(V, C)
            if method == "idblock"
            else build_kmeans_l2(W_U, C, args.seed)
        )
        ctr = cof[Ytr]
        result["methods"][method] = {"bottlenecks": {}}
        for B in bottlenecks:
            r = train_router(
                Htr, ctr, d, C, B, device, args.epochs, 4096, 2e-3, args.seed
            )
            ev = eval_router(r, Hva, Yva, T10va, cof, C, m_list, device, V)
            nparams = sum(p.numel() for p in r.parameters())
            result["methods"][method]["bottlenecks"][str(B)] = {
                "params": int(nparams),
                "by_m": {str(m): ev[m] for m in m_list},
            }
            for m in m_list:
                e = ev[m]
                orc = ""
                try:
                    o = oracle["methods"][method]["by_m"][str(m)]["truenext"]
                    orc = f"  oracle={o:.3f}  gap={o - e['truenext']:+.3f}"
                except Exception:
                    pass
                print(
                    f"  {method} B={B} m={m:>2}  fetch={e['fetch']:.3f}  "
                    f"pred_truenext={e['truenext']:.3f}  top10={e['top10']:.3f}{orc}",
                    flush=True,
                )

    result["wall_seconds"] = round(time.time() - t0, 1)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(result, indent=2))
    print(f"[wrote] {outp}  ({result['wall_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()

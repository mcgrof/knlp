"""Validation perplexity for a trained TrellisLM/Dense checkpoint at one or
more context lengths, on a packed HF text dataset (gpt2 tokenizer)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis_lm.config import TrellisConfig
from trellis_lm.model import build_model


def load_ckpt(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = TrellisConfig.from_dict(ck["cfg"])
    model = build_model(cfg, ck["kind"]).to(device).eval()
    model.load_state_dict(ck["model"])
    return cfg, model


def packed(dataset, split, seq_len, n_seqs):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    ds = load_dataset(dataset, split=split, streaming=True)
    buf, out = [], []
    for ex in ds:
        buf.extend(tok(ex.get("text") or "").input_ids + [tok.eos_token_id])
        while len(buf) >= seq_len:
            out.append(buf[:seq_len]); buf = buf[seq_len:]
            if len(out) >= n_seqs:
                return out
    return out


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", default="roneneldan/TinyStories")
    p.add_argument("--split", default="validation")
    p.add_argument("--seq_lens", default="256,512,1024")
    p.add_argument("--n_seqs", type=int, default=64)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--output", default=None)
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, model = load_ckpt(a.ckpt, device)
    rows = []
    for L in (int(x) for x in a.seq_lens.split(",")):
        seqs = packed(a.dataset, a.split, L, a.n_seqs)
        nll, ntok = 0.0, 0
        for i in range(0, len(seqs), a.batch):
            idx = torch.tensor(seqs[i:i + a.batch], device=device)
            _, loss = model(idx, labels=idx, training=False)
            n = idx.numel() - idx.shape[0]
            nll += loss.item() * n; ntok += n
        ce = nll / max(1, ntok)
        ppl = math.exp(min(20, ce))
        rows.append({"seq_len": L, "ce": round(ce, 5), "ppl": round(ppl, 4),
                     "n_seqs": len(seqs)})
        print(f"L={L:5d}  CE={ce:.4f}  PPL={ppl:.3f}  ({len(seqs)} seqs)", flush=True)
    if a.output:
        Path(a.output).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

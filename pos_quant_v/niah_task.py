#!/usr/bin/env python3
"""Needle-in-a-haystack key->value retrieval with content/filler token labels.

Grounds the position-aware Value-compression idea on REAL long context. The
synthetic key->value generator is ~87% "content" tokens, which understates the
savings of compressing filler-position Values; natural long context is the
opposite (mostly filler with sparse retrievable anchors). This builds NIAH-style
contexts -- a handful of "key: value" needles buried in real English filler --
labels each token as content (part of a stored key->value binding) or filler
(everything else: prose, scaffolding, punctuation), and reports the real filler
fraction vs context length. That fraction is the ceiling on filler-V compression
savings.

Usage:
  python3 pos_quant_v/niah_task.py --tokenizer Qwen/Qwen2.5-7B \
      --lengths 1024,2048,4096,8192 --needles 4 --out OUT/filler_fraction.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

NOUNS = [
    "silver lighthouse",
    "copper kettle",
    "marble fountain",
    "velvet curtain",
    "cedar bridge",
    "granite tower",
    "amber lantern",
    "iron gateway",
    "crystal harbor",
    "willow garden",
    "bronze compass",
    "ivory staircase",
    "scarlet meadow",
    "golden archive",
    "slate observatory",
    "jade pavilion",
]


def load_filler_sentences(seed):
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    sents = []
    for t in ds["text"]:
        t = t.strip()
        if len(t) > 80 and not t.startswith("="):
            sents.append(t)
        if len(sents) >= 20000:
            break
    rng = random.Random(seed)
    rng.shuffle(sents)
    return sents


def build_context(tok, target_len, n_needles, sents, rng):
    """Return (text, content_char_spans, needles). Needles are
    'The access code for the {key} is {value}.' with {key},{value} = content."""
    keys = rng.sample(NOUNS, n_needles)
    needles = [(k, f"{rng.randint(100000, 999999)}") for k in keys]
    # interleave: build chunks of filler, drop needles at random depths
    parts = []  # list of (text, is_content) — content marks key/value spans only
    si = 0

    def add_filler(nchars):
        nonlocal si
        acc = 0
        while acc < nchars and si < len(sents):
            parts.append((sents[si] + " ", False))
            acc += len(sents[si]) + 1
            si += 1

    # rough chars-per-token ~4; aim a bit over then trim by tokens later
    seg = max(1, (target_len * 4) // (n_needles + 1))
    for i in range(n_needles):
        add_filler(seg)
        k, v = needles[i]
        parts.append(("The access code for the ", False))
        parts.append((k, True))  # key span = content
        parts.append((" is ", False))
        parts.append((v, True))  # value span = content
        parts.append((". ", False))
    add_filler(seg)

    # assemble text + char spans of content
    text = ""
    content_spans = []
    for s, is_c in parts:
        if is_c:
            content_spans.append((len(text), len(text) + len(s)))
        text += s

    # append the query (filler) for the LAST needle
    qk, qv = needles[-1]
    query = f"\n\nQuestion: What is the access code for the {qk}? Answer:"
    text += query
    return text, content_spans, needles, (qk, qv)


def token_content_mask(tok, text, content_spans, max_tokens):
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"][:max_tokens]
    offs = enc["offset_mapping"][:max_tokens]
    mask = []
    for a, b in offs:
        is_c = any(a < ce and b > cs for cs, ce in content_spans)
        mask.append(is_c)
    return ids, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--lengths", default="1024,2048,4096,8192")
    ap.add_argument("--needles", type=int, default=4)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    sents = load_filler_sentences(args.seed)
    lengths = [int(x) for x in args.lengths.split(",")]
    rng = random.Random(args.seed)

    result = {"tokenizer": args.tokenizer, "needles": args.needles, "by_length": {}}
    for L in lengths:
        fr = []
        ntok = []
        for _ in range(args.samples):
            text, spans, needles, qa = build_context(tok, L, args.needles, sents, rng)
            ids, mask = token_content_mask(tok, text, spans, L)
            cfrac = sum(mask) / max(len(mask), 1)
            fr.append(1.0 - cfrac)
            ntok.append(len(ids))
        import statistics as st

        result["by_length"][str(L)] = {
            "filler_fraction_mean": st.mean(fr),
            "content_fraction_mean": 1 - st.mean(fr),
            "tokens_mean": st.mean(ntok),
        }
        print(
            f"  L={L:>5}  tokens~{int(st.mean(ntok))}  "
            f"filler={st.mean(fr)*100:.1f}%  content={(1-st.mean(fr))*100:.1f}%",
            flush=True,
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()

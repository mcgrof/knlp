#!/usr/bin/env python3
"""Whole-vs-part prefix test with REAL KVzip scores at long context.

A tiered-serving evaluation reports content-based KVZap-family scoring evicts the
prefix (so destroying prefix-cache reuse), evidenced by a regional MEAN ("first 3
blocks score low") on length-bucketed proxy scores. This script answers the
finer, decision-relevant question with the actual KVzip oracle (the query-agnostic
reconstruction scorer KVZap approximates), on real per-token scores at the long
context where the effect lives:

  Within the prefix, does KVzip prune the WHOLE region (blanket — important
  content lost, reuse genuinely destroyed) or only the redundant FILLER while
  keeping the distinctive, high-information sentences (selective — the reuse
  damage depends on whether the reused span is filler or content)?

We build a long context whose prefix contains a few DISTINCTIVE sentences (named
entities + specific unusual statements — the kind a query-agnostic reconstruction
scorer should value because they are non-redundant) interleaved with GENERIC
filler (redundant, predictable — easy to reconstruct, so low value). We capture
KVzip's real per-(layer,kv-head,token) importance via a hook on `score_kvzip`,
aggregate to per-token, and report, by region (prefix / middle / recency) and
within the prefix (distinctive vs generic), the retention rate at several
keep-budgets. Blanket => distinctive-prefix retention ~= generic-prefix retention
(both low). Selective => distinctive-prefix retention >> generic-prefix retention.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

# Distinctive, non-redundant sentences (named entities + specific claims).
DISTINCT = [
    "Dr. Imelda Vasquez-Toh patented the nitrogen-cooled lattice valve in Reykjavik.",
    "The frigate Anaximander sank at coordinates 47.3 north, 9.1 east in 1894.",
    "Quarterly revenue at Brindlewick Foundry rose to 38.4 million krona.",
    "The Zephyrine Protocol requires exactly seventeen independent signatories.",
    "Professor Okonkwo measured the half-life of dysprosium-154 at 6.2 hours.",
]
# Generic, redundant filler.
FILLER = [
    "The weather stayed mild and the day passed without any particular event.",
    "People walked along the street and the shops opened as they usually do.",
    "It was a normal morning and nothing unusual happened to anyone there.",
    "The committee met and discussed the usual topics before going home.",
    "Things continued as before and the routine carried on much the same.",
]


def build(tok, length, rng, n_distinct=4):
    """Prefix = a few distinct sentences + generic filler; then a long generic
    body to reach `length`. Returns ids and boolean token masks for the
    distinctive-prefix tokens and the generic-prefix tokens."""
    distinct = rng.sample(DISTINCT, n_distinct)
    # interleave distinct sentences with generic filler inside the prefix
    prefix_parts = []
    for d in distinct:
        prefix_parts.append(rng.choice(FILLER))
        prefix_parts.append(d)
    prefix_parts.append(rng.choice(FILLER))
    prefix_text = " ".join(prefix_parts)
    # body filler to fill to length
    body = []
    while len(tok(prefix_text + " " + " ".join(body)).input_ids) < length:
        body.append(rng.choice(FILLER))
    text = prefix_text + " " + " ".join(body)
    ids = tok(text, return_tensors="pt").input_ids[0][:length]
    T = ids.shape[0]
    # locate distinctive-sentence token spans within the prefix
    distinct_mask = torch.zeros(T, dtype=torch.bool)
    idl = ids.tolist()
    for d in distinct:
        dids = tok(" " + d, add_special_tokens=False).input_ids
        L = len(dids)
        for i in range(len(idl) - L):
            if idl[i : i + L] == dids:
                distinct_mask[i : i + L] = True
                break
    prefix_end = len(tok(prefix_text).input_ids)
    return ids, distinct_mask, prefix_end


@torch.no_grad()
def kvzip_scores(model, tok, ids, device, ratio=0.5):
    """Capture KVzip's real per-(layer,kv-head,token) reconstruction scores
    (self.score_val) and return the full tensor [n_layer, 1, n_kv_head, T]. We
    apply KVzip's OWN keep rule to this tensor downstream (compress_post,
    layerwise=False: prune the global bottom-k entries across the flattened
    [layer,head,token] tensor), so retention numbers are KVzip's actual decision,
    not a proxy aggregation."""
    from kvpress import KVzipPress

    press = KVzipPress(compression_ratio=ratio)
    cap = []
    orig = press.score_kvzip

    def patched(*a, **k):
        r = orig(*a, **k)
        sv = press.score_val
        if sv is not None:
            cap.append(sv.detach().float().cpu().clone())
        return r

    press.score_kvzip = patched
    with press(model):
        model(ids.unsqueeze(0).to(device), use_cache=True)
    if not cap:
        return None
    # element-wise max across captures = the fully-filled final score tensor
    sv = cap[0]
    for c in cap[1:]:
        sv = torch.maximum(sv, c)
    return sv  # [n_layer, 1, n_kv_head, T]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--length", type=int, default=8192)
    ap.add_argument(
        "--block", type=int, default=512, help="block size (prefix=first 3)"
    )
    ap.add_argument("--recency", type=int, default=128)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--keep", default="0.05,0.10,0.20,0.30")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation="sdpa"
    ).to(device)
    model.eval()
    rng = random.Random(args.seed)
    keeps = [float(x) for x in args.keep.split(",")]
    prefix_tok = 3 * args.block  # "first 3 blocks" per the paper

    # retention[region][keep] = [hits, total tokens]
    regions = ["prefix_distinct", "prefix_generic", "middle", "recency"]
    ret = {r: {k: [0.0, 0.0] for k in keeps} for r in regions}
    n_used = 0
    for i in range(args.n):
        ids, distinct_mask, _ = build(tok, args.length, rng)
        T = ids.shape[0]
        if T < prefix_tok + args.recency + 256:
            continue
        sv = kvzip_scores(model, tok, ids, device)  # [L,1,H,T]
        if sv is None or sv.shape[-1] != T:
            continue
        n_used += 1
        kept_layhead = sv[:, 0]  # [L,H,T]
        flat = sv.reshape(-1)
        numel = flat.numel()
        # region masks (over token positions)
        pref = torch.zeros(T, dtype=torch.bool)
        pref[4:prefix_tok] = True  # skip 4 sink tokens (forced-kept by KVzip)
        m_pref_d = pref & distinct_mask
        m_pref_g = pref & ~distinct_mask
        m_mid = torch.zeros(T, dtype=torch.bool)
        m_mid[prefix_tok : T - args.recency] = True
        m_rec = torch.zeros(T, dtype=torch.bool)
        m_rec[T - args.recency :] = True
        masks = {
            "prefix_distinct": m_pref_d,
            "prefix_generic": m_pref_g,
            "middle": m_mid,
            "recency": m_rec,
        }
        for k in keeps:  # k = keep fraction; KVzip prunes the global bottom (1-k)
            n_prune = int(numel * (1.0 - k))
            if n_prune <= 0:
                thr = flat.min() - 1.0
            else:
                thr = torch.kthvalue(flat, n_prune).values  # largest pruned score
            kept = kept_layhead > thr  # [L,H,T] entries KVzip keeps
            for rname, m in masks.items():
                if m.any():
                    sub = kept[:, :, m]  # [L,H,n_region] entries
                    ret[rname][k][0] += float(sub.float().sum())
                    ret[rname][k][1] += float(sub.numel())

    res = {
        "model": args.model,
        "length": args.length,
        "n_used": n_used,
        "retention": {},
    }
    print(
        f"\n[{args.model}] n={n_used}  len={args.length}  prefix=first {prefix_tok} tok"
    )
    print(f"  retention rate by region (fraction of region's tokens kept):")
    print(f"  {'region':16s} " + " ".join(f"keep{int(k*100):>2d}%" for k in keeps))
    for rname in regions:
        row = {}
        cells = []
        for k in keeps:
            r = ret[rname][k][0] / max(1e-9, ret[rname][k][1])
            row[f"{k}"] = r
            cells.append(f"{r:7.2f}")
        res["retention"][rname] = row
        print(f"  {rname:16s} " + " ".join(cells))
    print(
        "\n  whole-vs-part: compare prefix_distinct vs prefix_generic.\n"
        "    distinct >> generic  => SELECTIVE (KVzip keeps the high-info prefix,\n"
        "                            drops prefix filler; reuse loss is filler-only).\n"
        "    distinct ~= generic (both low) => BLANKET (prunes whole prefix)."
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()

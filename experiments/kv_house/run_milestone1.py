# SPDX-License-Identifier: GPL-2.0
"""Aggregate milestone-1 measurements into pre-registered verdicts.

Reads the temporal-structure CSV and the transform-ablation JSONL
and answers the seven milestone questions with PASS / FAIL /
INCONCLUSIVE plus a GO / GO_WITH_MODIFIED_HYPOTHESIS / NO_GO
recommendation. Every threshold is fixed here, in code, before any
pod data is collected; a failed hypothesis is a result, not a bug.

Thresholds (pre-registered):
- Q1 temporal structure: on wikitext, the LOCALITY gap (real minus
  scattered-pseudo-block top-quarter singular energy) >= 0.03 with
  real > scattered in >= 70% of blocks, for the majority of
  (tensor, B) cells; the random-token class must show a smaller
  mean locality gap than wikitext. (A within-block shuffle cannot
  move singular values, so the spectral null is the scattered
  pseudo-block; the shuffle null backs the order metrics.)
- Q2 DC concentration: mean DC-Householder mode-0 energy >= 2/B.
- Q4 RoPE: post-RoPE K keeps >= 50% of the pre-RoPE locality gap.
- Q5 equal-rate value: scored at the uniform4 allocation ONLY
  (front-loaded allocations cripple the identity baseline into a
  token-truncation control, so they are reported but never
  scored): best transform's joint attention-output error <= 0.8x
  identity.
- Q6 identity: max transformed-attention identity error <= 1e-6.
- Q7 prefix: real-data encode determinism, and every PIA budget
  entry's status is PASS or WARN (parsed, not substring-matched).
- GO gates, all at uniform4: the KLT oracle family (pca_oracle or
  corpus_pca) reaching <= 0.8x identity attention error shows the
  temporal hypothesis has headroom. GO additionally requires a
  DEPLOYABLE lane at <= 0.8x identity whose total bytes (payload
  plus transform metadata) stay <= 1.10x identity's bytes —
  pca_oracle's per-block basis fails this by construction,
  poweriter's B floats pass it, corpus_pca ships no per-block
  metadata. Oracle-only headroom downgrades to
  GO_WITH_MODIFIED_HYPOTHESIS.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

DEPLOYABLE = (
    "dc_householder",
    "dct2",
    "haar",
    "hadamard",
    "poweriter_householder",
    "corpus_pca",
    "anchor_delta",
)
ORACLES = ("pca_oracle", "corpus_pca")


def load_structure(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_ablation(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def mean(xs):
    xs = list(xs)
    return statistics.fmean(xs) if xs else float("nan")


def _locality_gaps(rows, klass):
    cells = defaultdict(list)
    for r in rows:
        if r["class"] != klass:
            continue
        gap = float(r["real_top_quarter_energy"]) - float(r["scat_top_quarter_energy"])
        cells[(r["tensor"], int(r["block_size"]))].append(gap)
    return cells


def q1_structure(rows):
    wik = {
        k: (mean(g), mean([x > 0 for x in g]))
        for k, g in _locality_gaps(rows, "wikitext").items()
    }
    rnd_gap = mean(g for gs in _locality_gaps(rows, "random").values() for g in gs)
    wik_gap = mean(g for g, _ in wik.values())
    order_gap = mean(
        float(r["real_adjacent_cosine"]) - float(r["shuf_adjacent_cosine"])
        for r in rows
        if r["class"] == "wikitext"
    )
    ok_cells = sum(1 for g, frac in wik.values() if g >= 0.03 and frac >= 0.7)
    verdict = (
        "PASS"
        if wik and ok_cells > len(wik) / 2 and wik_gap > rnd_gap
        else ("FAIL" if wik else "INCONCLUSIVE")
    )
    return verdict, {
        "wikitext_cells": {f"{t}_B{b}": g for (t, b), (g, _) in wik.items()},
        "wikitext_mean_locality_gap": wik_gap,
        "random_mean_locality_gap": rnd_gap,
        "wikitext_mean_order_gap_adjcos": order_gap,
        "cells_passing": f"{ok_cells}/{len(wik)}",
    }


def q2_dc_concentration(ab):
    by_b = defaultdict(list)
    for r in ab:
        if r["transform"] == "dc_householder" and r["class"] == "wikitext":
            by_b[r["block_size"]].append(r["k_top1_mode_energy"])
    detail = {f"B{b}": mean(v) for b, v in by_b.items()}
    ok = all(mean(v) >= 2.0 / b for b, v in by_b.items()) if by_b else False
    return ("PASS" if ok else "FAIL" if by_b else "INCONCLUSIVE"), detail


def q3_ranking(ab, alloc="uniform4"):
    """Rank transforms by joint attention-output error, carrying
    mean total bytes per token per side. alloc is 'uniform4' or the
    structural family 'frontloaded' (8-bit head / 2-bit tail,
    reported but never scored)."""

    def match(name):
        if alloc == "frontloaded":
            return name.startswith("front8x") and name.endswith("tail2")
        return name == alloc

    err, byt = defaultdict(list), defaultdict(list)
    for r in ab:
        if (
            match(r.get("alloc", ""))
            and r["class"] == "wikitext"
            and "joint_attn_out_rel_err" in r
        ):
            err[r["transform"]].append(r["joint_attn_out_rel_err"])
            byt[r["transform"]].append(r["bytes_per_token_side"])
    ranking = sorted((mean(v), t) for t, v in err.items())
    return [(t, e, mean(byt[t])) for e, t in ranking]


def q4_rope(rows):
    gaps = defaultdict(list)
    for r in rows:
        if r["class"] != "wikitext" or r["tensor"] not in ("k_pre", "k_post"):
            continue
        gap = float(r["real_top_quarter_energy"]) - float(r["scat_top_quarter_energy"])
        gaps[r["tensor"]].append(gap)
    pre, post = mean(gaps["k_pre"]), mean(gaps["k_post"])
    if not gaps["k_pre"]:
        return "INCONCLUSIVE", {}
    ratio = post / pre if pre > 0 else float("inf")
    return ("PASS" if ratio >= 0.5 else "FAIL"), {
        "pre_rope_locality_gap": pre,
        "post_rope_locality_gap": post,
        "retained_fraction": ratio,
    }


def q5_equal_rate(ranking_u4):
    d = {t: (e, b) for t, e, b in ranking_u4}
    if "identity" not in d or not ranking_u4:
        return "INCONCLUSIVE", {}
    ident_err, _ = d["identity"]
    best_t, best_e, best_b = ranking_u4[0]
    return ("PASS" if best_e <= 0.8 * ident_err else "FAIL"), {
        "scored_alloc": "uniform4",
        "best": best_t,
        "best_err": best_e,
        "identity_err": ident_err,
        "ratio": best_e / ident_err if ident_err else float("nan"),
    }


def q6_identity(ab):
    errs = [r["identity_max_err"] for r in ab if "identity_max_err" in r]
    if not errs:
        return "INCONCLUSIVE", {}
    return ("PASS" if max(errs) <= 1e-6 else "FAIL"), {"max_err": max(errs)}


def q7_prefix(capture_dir, pia_summary):
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from kv_house.prefix_codec import CodecConfig, SealedBlockCodec

    cap = Path(capture_dir)
    manifest = json.loads((cap / "manifest.json").read_text())
    rec = torch.load(cap / manifest[0]["file"], map_location="cpu")
    layer = sorted(rec)[0]
    k = rec[layer]["k_post"][0].float()
    detail = {}
    ok = True
    for name in ("dc_householder", "dct2", "poweriter_householder"):
        codec = SealedBlockCodec(
            CodecConfig(block_size=16, transform=name, quant_alloc=(8,) * 16)
        )
        arts1, _ = codec.seal_context(k[:64])
        arts2, _ = codec.seal_context(k[:80])
        same = all(a.digest == b.digest for a, b in zip(arts1, arts2[:4]))
        detail[name] = "deterministic+extension-stable" if same else "UNSTABLE"
        ok = ok and same
    if pia_summary:
        entries = json.loads(Path(pia_summary).read_text())
        statuses = [e.get("status", "FAIL") for e in entries]
        detail["pia_statuses"] = statuses
        detail["pia_classifications"] = [e.get("classification", "") for e in entries]
        ok = ok and bool(statuses) and all(s in ("PASS", "WARN") for s in statuses)
    return ("PASS" if ok else "FAIL"), detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure-csv", required=True)
    ap.add_argument("--ablation-jsonl", required=True)
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--pia-summary", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_structure(args.structure_csv)
    ab = load_ablation(args.ablation_jsonl)

    verdicts = {}
    verdicts["q1_temporal_structure"] = q1_structure(rows)
    verdicts["q2_dc_concentration"] = q2_dc_concentration(ab)
    ranking_u4 = q3_ranking(ab, "uniform4")
    verdicts["q3_ranking_uniform4"] = ("TABLE", ranking_u4)
    verdicts["q3_ranking_frontloaded_unscored"] = (
        "TABLE",
        q3_ranking(ab, "frontloaded"),
    )
    verdicts["q4_rope_survival"] = q4_rope(rows)
    verdicts["q5_equal_rate_value"] = q5_equal_rate(ranking_u4)
    verdicts["q6_attention_identity"] = q6_identity(ab)
    verdicts["q7_prefix_stability"] = q7_prefix(args.capture_dir, args.pia_summary)

    d = {t: (e, b) for t, e, b in ranking_u4}
    ident = d.get("identity")
    oracle_headroom = ident is not None and any(
        t in d and d[t][0] <= 0.8 * ident[0] for t in ORACLES
    )
    deployable_headroom = ident is not None and any(
        t in d and d[t][0] <= 0.8 * ident[0] and d[t][1] <= 1.10 * ident[1]
        for t in DEPLOYABLE
    )
    gates = [
        verdicts["q1_temporal_structure"][0],
        verdicts["q2_dc_concentration"][0],
        verdicts["q6_attention_identity"][0],
        verdicts["q7_prefix_stability"][0],
    ]
    if not d:
        reco = "INCONCLUSIVE"
    elif "FAIL" in gates or not oracle_headroom:
        reco = "NO_GO"
    elif deployable_headroom:
        reco = "GO"
    else:
        reco = "GO_WITH_MODIFIED_HYPOTHESIS"
    out = {"verdicts": verdicts, "recommendation": reco}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

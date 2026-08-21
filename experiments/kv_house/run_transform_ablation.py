# SPDX-License-Identifier: GPL-2.0
"""Transform x allocation ablation over captured KV blocks.

For every (layer, kv head, block) this quantizes each temporal
transform's modes under matched byte budgets and measures both
reconstruction error and — the ranking signal that matters —
attention damage from swapping the reconstructed block into an
otherwise full-precision cache, probed with the model's own
captured queries. K and V are damaged separately and jointly.

Controls are built in: identity (no transform), anchor-delta
(CacheGen), random Householder (placebo), flat Hadamard (the
KVLinC-rejected token rotation), per-block PCA (oracle ceiling),
corpus PCA (deployable KLT, fitted on the wikitext calibration
split only). The transformed-attention identity is verified
numerically on real data for every orthogonal transform before its
quantization rows are trusted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from kv_house.attention_sim import (  # noqa: E402
    block_replacement_metrics,
    reference_pass,
)
from kv_house.quant import quantize_rows, standard_allocations  # noqa: E402
from kv_house.transforms import TRANSFORMS, make_transform  # noqa: E402


def attn_alloc(name):
    """Allocations whose attention damage is measured: the uniform
    4-bit reference, the 8-bit-head/2-bit-tail front-loaded family,
    and the FreqKV-style 16-bit-head truncation family. Front-loaded
    names embed B-dependent head-row counts, so match structurally."""
    return (
        name == "uniform4"
        or (name.startswith("front8x") and name.endswith("tail2"))
        or (name.startswith("front16x") and name.endswith("tail0"))
    )


def spaced_blocks(t, b, max_blocks):
    n = t // b
    take = min(n, max_blocks)
    return [(j * n // take) for j in range(take)]


def fit_corpus_bases(cap_dir, manifest, calib_files, block_sizes, layer_stride):
    """Corpus KLT per (layer, head, tensor, B) from calibration
    samples only."""
    from kv_house.transforms import CorpusPCA

    pool = {}
    for entry in manifest:
        if entry["file"] not in calib_files:
            continue
        rec = torch.load(cap_dir / entry["file"], map_location="cpu")
        for layer in sorted(rec)[::layer_stride]:
            for tname in ("k_post", "v"):
                x_all = rec[layer][tname].float()
                for h in range(x_all.shape[0]):
                    x = x_all[h]
                    for b in block_sizes:
                        key = (layer, h, tname, b)
                        blocks = pool.setdefault(key, [])
                        for j in range(x.shape[0] // b):
                            blocks.append(x[j * b : (j + 1) * b])
    bases = {}
    for key, blocks in pool.items():
        t = CorpusPCA(key[3])
        t.fit_corpus(blocks)
        bases[key] = t
    return bases


def identity_check(t, k, q):
    r = t.matrix(dtype=torch.float64)
    if r is None:
        return None
    k64, q64 = k.to(torch.float64), q.to(torch.float64)
    ref = q64 @ k64.T
    got = (q64 @ t.forward(k64).T) @ r
    return float((ref - got).abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--block-sizes", default="8,16,32")
    ap.add_argument("--transforms", default=",".join(TRANSFORMS))
    ap.add_argument("--max-blocks", type=int, default=3)
    ap.add_argument("--max-heads", type=int, default=2)
    ap.add_argument("--layer-stride", type=int, default=2)
    ap.add_argument("--calib-fraction", type=float, default=0.5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cap_dir = Path(args.capture_dir)
    manifest = json.loads((cap_dir / "manifest.json").read_text())
    meta = json.loads((cap_dir / "meta.json").read_text())
    n_q = meta["heads"]["n_q"]
    n_kv = meta["heads"]["n_kv"]
    group = n_q // n_kv
    q_stride = meta["q_stride"]
    block_sizes = [int(b) for b in args.block_sizes.split(",")]
    names = [n for n in args.transforms.split(",") if n]

    per_class = {}
    for e in manifest:
        per_class.setdefault((e["class"], e["ctx"]), []).append(e)
    calib_files = set()
    for (cls, _), entries in per_class.items():
        if cls != "wikitext":
            continue
        k = max(1, int(len(entries) * args.calib_fraction))
        calib_files.update(e["file"] for e in entries[:k])

    corpus = (
        fit_corpus_bases(cap_dir, manifest, calib_files, block_sizes, args.layer_stride)
        if "corpus_pca" in names
        else {}
    )
    print(f"corpus bases fitted: {len(corpus)} (from {sorted(calib_files)})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("w")
    n_rows = 0

    for entry in manifest:
        if entry["file"] in calib_files:
            continue
        rec = torch.load(cap_dir / entry["file"], map_location=args.device)
        for layer in sorted(rec)[:: args.layer_stride]:
            tensors = rec[layer]
            t_len = tensors["k_post"].shape[1]
            q_positions = list(range(0, t_len, q_stride))
            for h in range(min(n_kv, args.max_heads)):
                k_full = tensors["k_post"][h].float()
                v_full = tensors["v"][h].float()
                q_heads = [
                    tensors["q_post"][h * group + g].float() for g in range(group)
                ]
                q_stack = torch.cat(q_heads, 0)
                pos_stack_base = list(q_positions) * len(q_heads)
                for b in block_sizes:
                    for j in spaced_blocks(t_len, b, args.max_blocks):
                        s, e = j * b, (j + 1) * b
                        kb, vb = k_full[s:e], v_full[s:e]
                        ref = reference_pass(q_stack, pos_stack_base, k_full, v_full, e)
                        for name in names:
                            key = (layer, h, "k_post", b)
                            if name == "corpus_pca":
                                if key not in corpus:
                                    continue
                                tk = corpus[key]
                                tv = corpus[(layer, h, "v", b)]
                            else:
                                tk = make_transform(name, b).fit(kb)
                                tv = make_transform(name, b).fit(vb)
                            base = {
                                "file": entry["file"],
                                "class": entry["class"],
                                "ctx": entry["ctx"],
                                "layer": layer,
                                "head": h,
                                "block_size": b,
                                "block_index": j,
                                "transform": name,
                            }
                            if j == 0 and tk.is_orthogonal:
                                err = identity_check(tk, kb.cpu(), q_heads[0].cpu())
                                if err is not None:
                                    base["identity_max_err"] = err
                            yk, yv = tk.forward(kb), tv.forward(vb)
                            ek = yk.pow(2).sum(-1)
                            base["k_top1_mode_energy"] = float(
                                ek[0] / (ek.sum() + 1e-30)
                            )
                            base["k_topq_mode_energy"] = float(
                                ek[: max(1, b // 4)].sum() / (ek.sum() + 1e-30)
                            )
                            for alloc in standard_allocations(b):
                                k_hat = tk.inverse(quantize_rows(yk, alloc.bits))
                                v_hat = tv.inverse(quantize_rows(yv, alloc.bits))
                                row = dict(base)
                                row["alloc"] = alloc.name
                                d = kb.shape[-1]
                                row["bytes_per_token_side"] = (
                                    alloc.total_bytes(d, tk.metadata_floats()) / b
                                )
                                row["k_rel_err"] = float(
                                    (k_hat - kb).norm() / (kb.norm() + 1e-12)
                                )
                                row["v_rel_err"] = float(
                                    (v_hat - vb).norm() / (vb.norm() + 1e-12)
                                )
                                if attn_alloc(alloc.name) and ref is not None:
                                    # the GQA query group is stacked into one
                                    # call (equal probe counts per head, so
                                    # row-mean == head-mean) and the clean
                                    # reference pass is shared across every
                                    # candidate for this block
                                    for tag, kh, vh in (
                                        ("kside", k_hat, vb),
                                        ("vside", kb, v_hat),
                                        ("joint", k_hat, v_hat),
                                    ):
                                        m = block_replacement_metrics(
                                            q_stack,
                                            pos_stack_base,
                                            k_full,
                                            v_full,
                                            s,
                                            e,
                                            kh,
                                            vh,
                                            ref=ref,
                                        )
                                        if m is None:
                                            continue
                                        for mk, mv in m.items():
                                            row[f"{tag}_{mk}"] = mv
                                fh.write(json.dumps(row) + "\n")
                                n_rows += 1
        print(f"{entry['file']}: {n_rows} rows so far")
    fh.close()
    print(f"wrote {n_rows} rows to {out}")


if __name__ == "__main__":
    main()

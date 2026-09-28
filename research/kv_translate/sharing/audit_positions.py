#!/usr/bin/env python3
"""Does a translated cache get worse the further into the prefix it goes?

The mappers were fitted on sequences of 1,024 tokens. A prefix of 16,384 asks
them to work at positions they never saw. If the translated cache departs
from the receiver's own cache more at late positions than at early ones, the
loss is one of extrapolation. If the departure is the same everywhere, the
map is as good at position 30,000 as at position 300, and a loss of quality
on long prefixes has to be explained some other way.

For one context this reads the stored source cache and the stored receiver
cache, translates the first, and reports the relative error and the cosine
between the two by position band, for keys and values, over all layers and
for the first, middle and last thirds of the receiver's layers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from ..published import fit_pair
from . import cache_store

BANDS = (0, 64, 256, 1024, 2048, 4096, 8192, 16384, 32768)


def bands(tokens):
    edges = [b for b in BANDS if b < tokens] + [tokens]
    return list(zip(edges[:-1], edges[1:]))


@torch.no_grad()
def compare(mapped, native, tokens):
    """Per band: relative squared error and cosine, averaged over heads."""
    out = []
    m = mapped.float()
    n = native.float()
    err = ((m - n) ** 2).sum(-1)  # [heads, tokens]
    ref = (n**2).sum(-1)
    dot = (m * n).sum(-1)
    mm = (m**2).sum(-1)
    for lo, hi in bands(tokens):
        e, r = err[:, lo:hi].sum(), ref[:, lo:hi].sum()
        cos = dot[:, lo:hi] / (mm[:, lo:hi].sqrt() * ref[:, lo:hi].sqrt()).clamp_min(
            1e-12
        )
        out.append(
            {
                "from": lo,
                "to": hi,
                "relative_error": float(e / r.clamp_min(1e-12)),
                "cosine": float(cos.mean()),
            }
        )
    return out


def mean_rows(rows_list):
    out = []
    for group in zip(*rows_list):
        out.append(
            {
                "from": group[0]["from"],
                "to": group[0]["to"],
                "relative_error": sum(g["relative_error"] for g in group) / len(group),
                "cosine": sum(g["cosine"] for g in group) / len(group),
            }
        )
    return out


@torch.no_grad()
def audit(source_path, target_path, mapper, theta, device="cuda"):
    sh, th = cache_store.header(source_path), cache_store.header(target_path)
    if sh["prefix_tokens_sha256"] != th["prefix_tokens_sha256"]:
        raise cache_store.WrongCache("the two caches are not for the same prefix")
    size = max(sh["payload_bytes"], th["payload_bytes"])
    reader = cache_store.Reader(size, device)
    head, _ = reader.read(source_path, direct=False)
    src = cache_store.as_pairs(reader.tensor(head))
    moved = fit_pair.transfer(mapper, src, theta, theta, dtype=torch.float32)
    del src
    head, _ = reader.read(target_path, direct=False)
    tgt = cache_store.as_pairs(reader.tensor(head))
    tokens, layers = th["tokens"], th["layers"]
    per = {"keys": [], "values": []}
    for l in range(layers):
        per["keys"].append(compare(moved[l][0][0], tgt[l][0][0], tokens))
        per["values"].append(compare(moved[l][1][0], tgt[l][1][0], tokens))
    thirds = {
        "first_third": range(0, layers // 3),
        "middle_third": range(layers // 3, 2 * layers // 3),
        "last_third": range(2 * layers // 3, layers),
    }
    out = {"tokens": tokens, "layers": layers, "method": mapper["method"]}
    for comp, rows in per.items():
        out[comp] = {"all_layers": mean_rows(rows)}
        for name, idx in thirds.items():
            out[comp][name] = mean_rows([rows[i] for i in idx])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--contexts", nargs="+", required=True)
    ap.add_argument("--mapper", action="append", required=True, help="name=path")
    ap.add_argument("--theta", type=float, default=1_000_000.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    results = {}
    for spec in args.mapper:
        name, path = spec.split("=", 1)
        mapper = fit_pair.load_mapper(path)
        mapper = {**mapper, "W": mapper["W"].cuda(), "b": mapper["b"].cuda()}
        for cid in args.contexts:
            r = audit(
                os.path.join(args.cache_dir, f"{cid}.source.kv"),
                os.path.join(args.cache_dir, f"{cid}.target.kv"),
                mapper,
                args.theta,
            )
            results[f"{name}/{cid}"] = r
            k = r["keys"]["all_layers"]
            v = r["values"]["all_layers"]
            print(
                name,
                cid,
                "keys",
                [round(x["relative_error"], 3) for x in k],
                "values",
                [round(x["relative_error"], 3) for x in v],
                flush=True,
            )
        del mapper
        torch.cuda.empty_cache()
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

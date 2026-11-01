#!/usr/bin/env python3
"""Measure how far a cartridge's keys and values move away from the
sizes the reader produces itself.

Rotary position embedding preserves each key's norm, so the stored key
norm needs no derotation. The first-token cartridge holds real keys and
values computed by the reader; per layer its median slot norms are the
reference. For each cartridge the script reports per layer the median
and largest slot norm of keys and of values divided by that reference
(1 means the reader's own scale), with the frozen sink (slot 0) apart.

An earlier version divided derotated keys by k_norm's weight (a real
key gives exactly 1); that is ill-conditioned in bf16, since rounding
from undoing the rotation, divided by near-zero weight entries, blows up.

Env:
    REF       path of the reader-computed (first-token) cartridge
    CARTS     comma list of label=path
    LAYERS    layers to print in full (default 0,1,2,18,35)
    OUT_JSON  per-layer statistics for every cartridge
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def norms(keys, values):
    kn = [k.float().norm(dim=-1)[0] for k in keys]  # [H, P] per layer
    vn = [v.float().norm(dim=-1)[0] for v in values]
    return kn, vn


def stats(keys, values, ref):
    kn, vn = norms(keys, values)
    out = []
    for i in range(len(kn)):
        rk, rv = ref[i]
        k, v = kn[i] / rk, vn[i] / rv
        out.append(
            dict(
                layer=i,
                key_sink=float(k[:, 0].mean()),
                key_med=float(k[:, 1:].median()),
                key_max=float(k[:, 1:].max()),
                val_sink=float(v[:, 0].mean()),
                val_med=float(v[:, 1:].median()),
                val_max=float(v[:, 1:].max()),
            )
        )
    return out


def main():
    from cas_cart_init import load_cart

    show = [int(x) for x in os.environ.get("LAYERS", "0,1,2,18,35").split(",")]
    rk, rv, _ = load_cart(os.environ["REF"])
    kn, vn = norms(rk, rv)
    ref = [(float(k[:, 1:].median()), float(v[:, 1:].median())) for k, v in zip(kn, vn)]
    res = {}
    for item in os.environ["CARTS"].split(","):
        label, path = item.split("=", 1)
        keys, values, nf = load_cart(path)
        s = stats(keys, values, ref)
        res[label] = dict(path=path, frozen=nf, slots=keys[0].shape[2], layers=s)
        km = [x["key_med"] for x in s]
        print(
            f"NORMS {label}: key norm / reader median, mean over layers {sum(km)/len(km):.3f} "
            f"(range {min(km):.3f}-{max(km):.3f}), largest slot "
            f"{max(x['key_max'] for x in s):.2f}",
            flush=True,
        )
        for x in s:
            if x["layer"] in show:
                print(
                    f"  L{x['layer']:>2} key med {x['key_med']:.3f} max {x['key_max']:.2f} "
                    f"sink {x['key_sink']:.3f} | value med {x['val_med']:.3f} "
                    f"max {x['val_max']:.2f} sink {x['val_sink']:.3f}",
                    flush=True,
                )
    if os.environ.get("OUT_JSON"):
        json.dump(res, open(os.environ["OUT_JSON"], "w"), indent=1)


if __name__ == "__main__":
    main()

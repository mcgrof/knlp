#!/usr/bin/env python3
"""Cross-document cartridge controls for the meta-initialization study:
start one document's training from ANOTHER document's cartridge.

Two kinds are built, both truncated to the target's trainable slot
count with the frozen sink kept (the sink is the KV of the first
special token at position 0, bit-identical across documents):

    foreign   the donor's trained cartridge as is, so every content
              slot holds the wrong document. If it still beats the
              target's own first-p anchor at step 0, what training put
              into the donor's cartridge is partly generic self-study
              behaviour, not document content.
    delta     the target's own anchor plus the donor's slot-wise
              displacement (trained cartridge minus the donor's own
              anchor), per layer, head and slot, with no fit and no
              shrinkage. Slot i sits at position i in both cartridges,
              so the stored-key difference is the rotated pre-RoPE
              difference and adds consistently. Against ``foreign`` it
              asks whether the anchor's content is worth anything at
              step 0.

Files are the library's cartridge dict (trainable_keys /
trainable_values / frozen_keys / frozen_values, one [1, H, P, D] bf16
parameter per layer), readable by ``cas_train_isolated.py``
(``CART_INIT=``) and ``cas_cart_loss.py``. A ``.json`` sidecar records
the sources and, for ``delta``, the per-layer relative value norm of
the transplanted displacement.

Usage:
    cas_cart_transplant.py foreign --anchor T.pt --donor-cart C.pt --out O.pt
    cas_cart_transplant.py delta --anchor T.pt --donor-cart C.pt \\
        --donor-anchor A.pt --out O.pt
"""

import argparse
import json
import os

import torch
from torch.nn import Parameter, ParameterList

TRAINABLE = ("trainable_keys", "trainable_values")
FROZEN = ("frozen_keys", "frozen_values")


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def plist(tensors):
    return ParameterList(
        [Parameter(t.contiguous(), requires_grad=False) for t in tensors]
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("kind", choices=("foreign", "delta"))
    ap.add_argument("--anchor", required=True, help="the target's first-p cartridge")
    ap.add_argument("--donor-cart", required=True, help="the donor's trained cartridge")
    ap.add_argument(
        "--donor-anchor", default="", help="the donor's first-p cartridge (delta)"
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tgt, cart = load(a.anchor), load(a.donor_cart)
    T = tgt[TRAINABLE[0]][0].shape[2]
    assert cart[TRAINABLE[0]][0].shape[2] >= T, "donor is shorter than the target"
    for name in FROZEN:
        for t, c in zip(tgt[name], cart[name]):
            assert torch.equal(t.data, c.data), f"{name}: frozen sink differs"
    out = {name: plist([p.data.clone() for p in tgt[name]]) for name in FROZEN}
    meta = {
        "kind": a.kind,
        "anchor": a.anchor,
        "donor_cart": a.donor_cart,
        "truncated_to": T,
    }
    if a.kind == "foreign":
        for name in TRAINABLE:
            out[name] = plist([p.data[:, :, :T].clone() for p in cart[name]])
    else:
        assert a.donor_anchor, "delta needs --donor-anchor"
        don = load(a.donor_anchor)
        meta["donor_anchor"] = a.donor_anchor
        rel = {}
        for name in TRAINABLE:
            ps, r = [], []
            for t, c, d in zip(tgt[name], cart[name], don[name]):
                delta = c.data[:, :, :T].float() - d.data[:, :, :T].float()
                ps.append((t.data.float() + delta).to(torch.bfloat16))
                r.append((delta.norm() / t.data.float().norm()).item())
            out[name] = plist(ps)
            rel[name] = r
        meta["delta_rel_norm_per_layer"] = rel
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    torch.save(out, a.out)
    with open(a.out + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"TRANSPLANT_DONE {a.kind} trainable={T} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()

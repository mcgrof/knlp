"""Cross-cartridge trajectory-structure diagnostic (meta-init gate, init-free half).

Asks whether trained Cartridges of the SAME document under different
recipes, and of DIFFERENT documents under the same recipe, share
reusable update structure -- without needing the (reconstructible but
unsaved) step-0 tensors: for carts sharing one init, the difference
Z_a - Z_b cancels it exactly, so all statistics here are computed on
reference-relative offsets D_a = Z_a - Z_ref.

Per layer and per K/V it reports:
  1. within-document, cross-recipe alignment: mean pairwise cosine of
     flattened offsets (positions are aligned -- same document, same
     init -- so elementwise cosine is meaningful);
  2. cross-document, same-recipe FEATURE-subspace overlap: offsets
     reshaped to [tokens, heads*head_dim]; overlap = mean squared
     cosine between the top-k right singular subspaces of two
     documents' offsets (position-independent), against a
     moment-matched Gaussian control.

A positive gate signal is (1) substantially above zero and (2) above
its Gaussian control on held-out pairs.

  python cas_meta_delta_diag.py --cas-out ~/cas_out \
      --group kv1024 --out delta_diag_kv1024.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

GROUPS = {
    # same-geometry KV=1024 arms (patients 01,02,03,05,06 unless noted)
    "kv1024": {
        "ref": "iso_diverse_10x",
        "arms": [
            "iso_lhq",
            "iso_diverse_lr05",
            "iso_disagree_hi",
            "iso_disagree_rand",
            "iso_diverse_corrected",
            "iso_finished",
        ],
        "patients": ["01", "02", "03", "05", "06"],
    },
    # auto-sized doc/20 arms: shapes identical per patient across arms
    "auto": {
        "ref": "iso5_decoupled",
        "arms": [
            "iso5_decoupled_4k",
            "iso2k_e80",
            "iso_diverse",
            "iso_faithful",
            "iso_hardneg",
        ],
        "patients": ["01", "02", "03", "05", "06"],
    },
}


def load_cart(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for kv in ("keys", "values"):
        layers = []
        for t in obj[f"trainable_{kv}"]:
            layers.append(t.detach().to(torch.float32).squeeze(0))  # [H,T,D]
        out[kv] = layers
    return out


def offsets(cas_out, group):
    g = GROUPS[group]
    refs, offs = {}, {}
    for p in g["patients"]:
        ref_path = Path(cas_out) / g["ref"] / "carts" / f"patient_{p}.pt"
        if not ref_path.exists():
            continue
        refs[p] = load_cart(ref_path)
        for arm in g["arms"]:
            path = Path(cas_out) / arm / "carts" / f"patient_{p}.pt"
            if not path.exists():
                continue
            cart = load_cart(path)
            offs[(arm, p)] = {
                kv: [a - b for a, b in zip(cart[kv], refs[p][kv])]
                for kv in ("keys", "values")
            }
    return offs


def flat_cos(a, b):
    a, b = a.flatten(), b.flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def feat_subspace(x, k=8):
    # x: [H,T,D] -> [T, H*D]; top-k right singular vectors of the
    # token-by-feature matrix = the feature-space update basis
    m = x.permute(1, 0, 2).reshape(x.shape[1], -1)
    m = m - m.mean(0, keepdim=True)
    try:
        v = torch.linalg.svd(m, full_matrices=False).Vh[:k]
    except Exception:
        return None
    return v  # [k, F]


def subspace_overlap(va, vb):
    # mean squared singular value of Va Vb^T in [0,1]; ~k/F for random
    s = torch.linalg.svdvals(va @ vb.T)
    return float((s**2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cas-out", required=True)
    ap.add_argument("--group", choices=sorted(GROUPS), required=True)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    offs = offsets(args.cas_out, args.group)
    arms = sorted({a for a, _ in offs})
    pats = sorted({p for _, p in offs})
    n_layers = len(next(iter(offs.values()))["keys"])
    res = {"group": args.group, "arms": arms, "patients": pats, "layers": {}}
    gen = torch.Generator().manual_seed(0)
    for li in range(n_layers):
        lrow = {}
        for kv in ("keys", "values"):
            # 1. within-document cross-recipe cosines
            wc = []
            for p in pats:
                ds = [offs[(a, p)][kv][li] for a in arms if (a, p) in offs]
                wc += [
                    flat_cos(ds[i], ds[j])
                    for i in range(len(ds))
                    for j in range(i + 1, len(ds))
                ]
            # 2. cross-document same-recipe subspace overlap + control
            xo, ctrl = [], []
            for a in arms:
                subs = {}
                for p in pats:
                    if (a, p) in offs:
                        v = feat_subspace(offs[(a, p)][kv][li], args.topk)
                        if v is not None:
                            subs[p] = v
                ps = sorted(subs)
                for i in range(len(ps)):
                    for j in range(i + 1, len(ps)):
                        xo.append(subspace_overlap(subs[ps[i]], subs[ps[j]]))
                        d = offs[(a, ps[i])][kv][li]
                        r = torch.randn(d.shape, generator=gen) * d.std() + d.mean()
                        rv = feat_subspace(r, args.topk)
                        ctrl.append(subspace_overlap(rv, subs[ps[j]]))
            lrow[kv] = {
                "within_doc_cross_recipe_cos_mean": (
                    round(sum(wc) / len(wc), 4) if wc else None
                ),
                "cross_doc_subspace_overlap_mean": (
                    round(sum(xo) / len(xo), 4) if xo else None
                ),
                "gaussian_control_overlap_mean": (
                    round(sum(ctrl) / len(ctrl), 4) if ctrl else None
                ),
                "n_pairs": len(xo),
            }
        res["layers"][li] = lrow
    # aggregate
    for kv in ("keys", "values"):
        for stat in (
            "within_doc_cross_recipe_cos_mean",
            "cross_doc_subspace_overlap_mean",
            "gaussian_control_overlap_mean",
        ):
            xs = [
                res["layers"][li][kv][stat]
                for li in res["layers"]
                if res["layers"][li][kv][stat] is not None
            ]
            res[f"agg_{kv}_{stat}"] = round(sum(xs) / len(xs), 4) if xs else None
    Path(args.out).write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        if k.startswith("agg_"):
            print(k, v)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

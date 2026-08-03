"""Reconstruct CAS cartridge first_k (KVFromText) inits and profile
absolute per-layer update statistics Delta = Z_final - Z_init.

The trained Cartridges save only the final trainable/frozen KV tensors;
the step-0 initialization (a frozen bf16 forward of the chat-templated
patient record through Qwen3-8B, truncated to KV_TOKENS) is
deterministic and reconstructible.  Stage "recon" replays the exact
KVFromText code path from the ~/cartridges fork (same tokenizer wrap,
same FlexQwen3ForCausalLM forward with seq_ids/position_ids under bf16
autocast, frozen sink row 0) and saves the init in the same cart dict
schema.  Fidelity check: the carts' frozen row is never trained, so
(cart.frozen - recon.frozen) measures pure reconstruction error.

Stage "profile" computes, per layer x {K,V}, on absolute deltas:
  - delta RMS (per cart, aggregated),
  - per-token-position RMS profile (first/middle/last third means),
  - cross-document pairwise cosine of Delta within an arm (flattened;
    truncated to the common min T for the auto-sized group),
  - cross-document feature-subspace overlap (top-k right singular
    subspaces of [T, heads*dim], reusing cas_meta_delta_diag), with a
    moment-matched Gaussian control,
  - correlation across carts between delta RMS and the arm's strict
    eval accuracy (per-record strict fields or per_patient_acc).

Usage (gpu1, cas_venv, GPU 0 only):
  CUDA_VISIBLE_DEVICES=0 ~/cas_venv/bin/python cas_meta_init_recon.py \
      --stage recon --patients 01 --sizes 1024   # single-job sanity
  CUDA_VISIBLE_DEVICES=0 ~/cas_venv/bin/python cas_meta_init_recon.py \
      --stage recon                              # all 15 inits
  CUDA_VISIBLE_DEVICES=0 ~/cas_venv/bin/python cas_meta_init_recon.py \
      --stage profile --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CARTRIDGES_DIR", os.path.expanduser("~/cartridges"))
os.environ.setdefault(
    "CARTRIDGES_OUTPUT_DIR", os.path.expanduser("~/cas_out/meta_init_recon")
)

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cas_meta_delta_diag import feat_subspace, subspace_overlap

MODEL = "Qwen/Qwen3-8B"
PATIENTS = ["01", "02", "03", "05", "06"]

KV1024_ARMS = [
    "iso_lhq",
    "iso_diverse_10x",
    "iso_diverse_lr05",
    "iso_disagree_hi",
    "iso_disagree_rand",
    "iso_diverse_corrected",
    "iso_finished",
]

# size: "1024"/"512" = fixed KV_TOKENS; "auto" = per-patient, recovered
# from the iso5_decoupled cart shapes (trainable + frozen rows).
GROUPS = {
    "kv1024": {"arms": KV1024_ARMS, "size": "1024"},
    "iso5": {"arms": ["iso5"], "size": "512"},
    "iso5_decoupled": {"arms": ["iso5_decoupled"], "size": "auto"},
}

# arm -> eval json candidates (first = primary for pooled correlation)
EVAL_SOURCES = {
    "iso_lhq": ["eval_lhq_heldout.json", "eval_lhq_trained.json"],
    "iso_diverse_10x": ["eval_10x.json"],
    "iso_diverse_lr05": ["eval_lr05.json"],
    "iso_disagree_hi": ["eval_disagree_hi.json"],
    "iso_disagree_rand": ["eval_disagree_rand.json"],
    "iso_diverse_corrected": ["eval_corrected_4pt.json"],
    "iso_finished": ["eval_finished_4pt.json"],
    "iso5": ["eval_t15_cart_iso5.json"],
    "iso5_decoupled": ["eval_t15_cart_iso5_decoupled.json"],
}

# arm whose carts share the target geometry, for recon fidelity checks
SIZE_TO_REF_ARM = {"1024": "iso_diverse_10x", "512": "iso5", "auto": "iso5_decoupled"}


def cart_path(cas_out, arm, p):
    return Path(cas_out) / arm / "carts" / f"patient_{p}.pt"


def init_path(out_dir, p, kv):
    return Path(out_dir) / f"init_p{p}_kv{kv}.pt"


def load_cart_dict(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for field in ("trainable_keys", "trainable_values", "frozen_keys", "frozen_values"):
        out[field] = [t.detach().clone() for t in obj[field]]
    return out


def auto_sizes(cas_out):
    sizes = {}
    for p in PATIENTS:
        obj = torch.load(
            cart_path(cas_out, "iso5_decoupled", p),
            map_location="cpu",
            weights_only=False,
        )
        t = obj["trainable_keys"][0].shape[2]
        f = obj["frozen_keys"][0].shape[2] if len(obj["frozen_keys"]) else 0
        sizes[p] = t + f
    return sizes


# ----------------------------------------------------------------- recon


def reconstruct(args, out_dir):
    sys.path.insert(0, os.environ["CARTRIDGES_DIR"])
    from transformers import AutoTokenizer
    from cartridges.cache import AttnConfig
    from cartridges.initialization.text import KVFromText
    from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM

    cas_out = Path(args.cas_out)
    autos = auto_sizes(cas_out)
    jobs = []
    for p in args.patients:
        for size in args.sizes:
            kv = autos[p] if size == "auto" else int(size)
            jobs.append((p, size, kv))
    # dedupe identical (p, kv) pairs (auto could collide with 512)
    seen, uniq = set(), []
    for p, size, kv in jobs:
        if (p, kv) not in seen:
            seen.add((p, kv))
            uniq.append((p, size, kv))
    jobs = uniq
    print(f"[recon] {len(jobs)} jobs: {[(p, kv) for p, _, kv in jobs]}", flush=True)

    t0 = time.time()
    # mirror cartridges/train.py: tokenizer, model.to(rank).to(bf16),
    # AttnConfig from model config; no model.eval() (Qwen3 has no
    # active dropout, and train.py runs the init forward in train mode)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = FlexQwen3ForCausalLM.from_pretrained(MODEL)
    model = model.to(0).to(torch.bfloat16)
    attn_config = AttnConfig(
        n_layers=model.config.num_hidden_layers,
        n_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
    )
    print(f"[recon] model loaded in {time.time() - t0:.1f}s", flush=True)

    meta = {"model": MODEL, "jobs": [], "auto_sizes": autos}
    for p, size, kv in jobs:
        t1 = time.time()
        rec = cas_out / "records" / f"patient_{p}.txt"
        cfg = KVFromText.Config(max_tokens=kv, text_source=str(rec))
        cache = cfg.instantiate().initialize_kv_cache(
            tokenizer=tokenizer, model=model, attn_config=attn_config
        )
        out = {}
        for field in (
            "trainable_keys",
            "trainable_values",
            "frozen_keys",
            "frozen_values",
        ):
            out[field] = [
                t.detach().to(torch.bfloat16).cpu() for t in getattr(cache, field)
            ]
        tshape = tuple(out["trainable_keys"][0].shape)
        assert tshape[2] == kv - 1, f"trainable rows {tshape} != kv-1 for kv={kv}"
        path = init_path(out_dir, p, kv)
        torch.save(out, path)
        # fidelity probe: the frozen sink row is never trained, so any
        # diff vs a real cart of the same geometry is pure recon error
        ref_arm = SIZE_TO_REF_ARM[size]
        frozen_max = None
        ref = cart_path(cas_out, ref_arm, p)
        if ref.exists():
            robj = torch.load(ref, map_location="cpu", weights_only=False)
            diffs = []
            for field in ("frozen_keys", "frozen_values"):
                for a, b in zip(out[field], robj[field]):
                    diffs.append((a.float() - b.detach().float()).abs().max().item())
            frozen_max = max(diffs)
        dt = time.time() - t1
        meta["jobs"].append(
            {
                "patient": p,
                "kv": kv,
                "size": size,
                "trainable_shape": list(tshape),
                "frozen_max_abs_diff_vs_" + ref_arm: frozen_max,
                "seconds": round(dt, 2),
            }
        )
        print(
            f"[recon] p{p} kv={kv} shape={tshape} "
            f"frozen_maxdiff={frozen_max} ({dt:.1f}s)",
            flush=True,
        )
    meta["total_seconds"] = round(time.time() - t0, 1)
    (Path(out_dir) / "recon_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[recon] done in {meta['total_seconds']}s", flush=True)


# --------------------------------------------------------------- profile


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx < 1e-12 or sy < 1e-12:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(xs, ys):
    if len(xs) < 3:
        return None
    return pearson(ranks(xs), ranks(ys))


def per_patient_strict(cas_out, fname):
    """Extract per-patient strict accuracy from an eval json."""
    path = Path(cas_out) / fname
    if not path.exists():
        return None
    d = json.load(open(path))
    if "per_patient_acc" in d:
        return {
            k.replace("patient_", ""): float(v) for k, v in d["per_patient_acc"].items()
        }
    if "records" in d:
        num, den = {}, {}
        for r in d["records"]:
            p = str(r["patient"]).replace("patient_", "")
            num[p] = num.get(p, 0) + int(r["strict"])
            den[p] = den.get(p, 0) + 1
        return {p: num[p] / den[p] for p in num}
    return None


def thirds_mean(v):
    t = v.shape[0]
    a, b = t // 3, 2 * t // 3
    return [
        round(v[:a].mean().item(), 6),
        round(v[a:b].mean().item(), 6),
        round(v[b:].mean().item(), 6),
    ]


def profile_group(gname, spec, cas_out, out_dir, device, topk):
    autos = auto_sizes(cas_out)
    carts, inits = {}, {}
    for p in PATIENTS:
        kv = autos[p] if spec["size"] == "auto" else int(spec["size"])
        ip = init_path(out_dir, p, kv)
        if not ip.exists():
            print(f"[profile:{gname}] missing init {ip}, skipping patient")
            continue
        inits[p] = load_cart_dict(ip)
        for arm in spec["arms"]:
            cp = cart_path(cas_out, arm, p)
            if cp.exists():
                carts[(arm, p)] = load_cart_dict(cp)
    pats = sorted({p for _, p in carts})
    arms = sorted({a for a, _ in carts})
    n_layers = len(next(iter(carts.values()))["trainable_keys"])

    # recon fidelity: frozen sink row is untrained -> diff is recon error
    frozen = {}
    for (arm, p), c in carts.items():
        diffs = []
        for field in ("frozen_keys", "frozen_values"):
            for a, b in zip(c[field], inits[p][field]):
                diffs.append((a.float() - b.float()).abs().max().item())
        frozen[f"{arm}/p{p}"] = round(max(diffs), 6)

    ss = {}  # (arm, p, kv) -> [sum_sq, count] across layers
    layers = {}
    gen = torch.Generator().manual_seed(0)
    kvmap = {"keys": "trainable_keys", "values": "trainable_values"}
    for li in range(n_layers):
        lrow = {}
        for kv, field in kvmap.items():
            deltas = {}
            for (arm, p), c in carts.items():
                d = (c[field][li].float() - inits[p][field][li].float()).squeeze(
                    0
                )  # [H, T, D]
                deltas[(arm, p)] = d.to(device)
                acc = ss.setdefault((arm, p, kv), [0.0, 0])
                acc[0] += d.pow(2).sum().item()
                acc[1] += d.numel()
            # delta RMS + positionwise profile
            rms = [d.pow(2).mean().sqrt().item() for d in deltas.values()]
            th = [
                thirds_mean(d.pow(2).mean(dim=(0, 2)).sqrt()) for d in deltas.values()
            ]
            th_mean = [round(sum(t[i] for t in th) / len(th), 6) for i in range(3)]
            # cross-document pairwise cosine of Delta, same arm
            cos, truncated = [], False
            for arm in arms:
                ds = [deltas[(arm, p)] for p in pats if (arm, p) in deltas]
                for i in range(len(ds)):
                    for j in range(i + 1, len(ds)):
                        a, b = ds[i], ds[j]
                        if a.shape != b.shape:
                            t = min(a.shape[1], b.shape[1])
                            a, b = a[:, :t], b[:, :t]
                            truncated = True
                        a, b = a.flatten(), b.flatten()
                        cos.append((a @ b / (a.norm() * b.norm() + 1e-12)).item())
            # cross-document feature-subspace overlap + Gaussian control
            subs = {key: feat_subspace(d, topk) for key, d in deltas.items()}
            xo, ctrl = [], []
            for arm in arms:
                ps = [
                    p for p in pats if (arm, p) in subs and subs[(arm, p)] is not None
                ]
                for i in range(len(ps)):
                    for j in range(i + 1, len(ps)):
                        va, vb = subs[(arm, ps[i])], subs[(arm, ps[j])]
                        xo.append(subspace_overlap(va, vb))
                        d = deltas[(arm, ps[i])]
                        r = torch.randn(d.shape, generator=gen).to(d.device)
                        r = r * d.std() + d.mean()
                        rv = feat_subspace(r, topk)
                        ctrl.append(subspace_overlap(rv, vb))
            lrow[kv] = {
                "delta_rms_mean": round(sum(rms) / len(rms), 6),
                "pos_rms_thirds": th_mean,
                "cross_doc_delta_cos_mean": (
                    round(sum(cos) / len(cos), 4) if cos else None
                ),
                "cos_truncated_to_min_T": truncated,
                "cross_doc_subspace_overlap_mean": (
                    round(sum(xo) / len(xo), 4) if xo else None
                ),
                "gaussian_control_overlap_mean": (
                    round(sum(ctrl) / len(ctrl), 4) if ctrl else None
                ),
                "n_pairs": len(xo),
            }
            del deltas
        layers[li] = lrow

    # per-cart scalar RMS (across all layers)
    cart_rms = {}
    for arm, p in carts:
        k = ss[(arm, p, "keys")]
        v = ss[(arm, p, "values")]
        cart_rms[(arm, p)] = {
            "keys": math.sqrt(k[0] / k[1]),
            "values": math.sqrt(v[0] / v[1]),
            "combined": math.sqrt((k[0] + v[0]) / (k[1] + v[1])),
        }

    # eval accuracy + correlations
    evals, corrs = {}, {}
    pooled = {"combined": ([], []), "keys": ([], []), "values": ([], [])}
    for arm in arms:
        for fname in EVAL_SOURCES.get(arm, []):
            accs = per_patient_strict(cas_out, fname)
            if accs is None:
                continue
            evals.setdefault(arm, {})[fname] = accs
            shared = [p for p in pats if (arm, p) in cart_rms and p in accs]
            row = {"n": len(shared), "patients": shared}
            for which in ("combined", "keys", "values"):
                xs = [cart_rms[(arm, p)][which] for p in shared]
                ys = [accs[p] for p in shared]
                row[which] = {
                    "pearson": (
                        round(pearson(xs, ys), 3)
                        if pearson(xs, ys) is not None
                        else None
                    ),
                    "spearman": (
                        round(spearman(xs, ys), 3)
                        if spearman(xs, ys) is not None
                        else None
                    ),
                }
                if fname == EVAL_SOURCES[arm][0]:
                    pooled[which][0].extend(xs)
                    pooled[which][1].extend(ys)
            corrs.setdefault(arm, {})[fname] = row
    pooled_out = {}
    for which, (xs, ys) in pooled.items():
        pooled_out[which] = {
            "n": len(xs),
            "pearson": (
                round(pearson(xs, ys), 3) if pearson(xs, ys) is not None else None
            ),
            "spearman": (
                round(spearman(xs, ys), 3) if spearman(xs, ys) is not None else None
            ),
        }

    # group-level aggregates over layers
    agg = {}
    for kv in ("keys", "values"):
        for stat in (
            "delta_rms_mean",
            "cross_doc_delta_cos_mean",
            "cross_doc_subspace_overlap_mean",
            "gaussian_control_overlap_mean",
        ):
            xs = [
                layers[li][kv][stat]
                for li in layers
                if layers[li][kv][stat] is not None
            ]
            agg[f"{kv}_{stat}"] = round(sum(xs) / len(xs), 4) if xs else None
        th = [layers[li][kv]["pos_rms_thirds"] for li in layers]
        agg[f"{kv}_pos_rms_thirds"] = [
            round(sum(t[i] for t in th) / len(th), 6) for i in range(3)
        ]

    structure = {}
    for kv in ("keys", "values"):
        scored = [
            (
                li,
                round(
                    layers[li][kv]["cross_doc_subspace_overlap_mean"]
                    - layers[li][kv]["gaussian_control_overlap_mean"],
                    4,
                ),
            )
            for li in layers
            if layers[li][kv]["cross_doc_subspace_overlap_mean"] is not None
        ]
        structure[kv] = sorted(scored, key=lambda t: -t[1])[:5]

    return {
        "arms": arms,
        "patients": pats,
        "size": spec["size"],
        "n_layers": n_layers,
        "frozen_row_max_abs_diff": frozen,
        "cart_rms": {
            f"{arm}/p{p}": {k: round(v, 6) for k, v in r.items()}
            for (arm, p), r in cart_rms.items()
        },
        "evals": evals,
        "correlations_rms_vs_strict_acc": corrs,
        "pooled_correlation_primary_evals": pooled_out,
        "aggregate": agg,
        "top5_structure_layers_overlap_minus_control": structure,
        "layers": layers,
    }


def print_table(res):
    for gname, g in res["groups"].items():
        print(f"\n=== group {gname} (arms={len(g['arms'])}, size={g['size']}) ===")
        a = g["aggregate"]
        print(f"{'':22s} {'keys':>12s} {'values':>12s}")
        for stat, label in (
            ("delta_rms_mean", "delta RMS"),
            ("cross_doc_delta_cos_mean", "xdoc delta cos"),
            ("cross_doc_subspace_overlap_mean", "xdoc overlap@8"),
            ("gaussian_control_overlap_mean", "gaussian control"),
        ):
            print(
                f"{label:22s} {a['keys_' + stat]!s:>12s} {a['values_' + stat]!s:>12s}"
            )
        for kv in ("keys", "values"):
            print(f"pos RMS thirds {kv:7s} {a[kv + '_pos_rms_thirds']}")
        print(
            "top5 structure layers:",
            {
                kv: g["top5_structure_layers_overlap_minus_control"][kv]
                for kv in ("keys", "values")
            },
        )
        fr = g["frozen_row_max_abs_diff"]
        print(
            f"frozen-row recon fidelity: max abs diff over carts = "
            f"{max(fr.values()):.6f} (min {min(fr.values()):.6f})"
        )
        for arm, rows in g["correlations_rms_vs_strict_acc"].items():
            for fname, row in rows.items():
                print(
                    f"corr[{arm} / {fname}] n={row['n']} combined "
                    f"r={row['combined']['pearson']} "
                    f"rho={row['combined']['spearman']} "
                    f"(K r={row['keys']['pearson']}, V r={row['values']['pearson']})"
                )
        po = g["pooled_correlation_primary_evals"]
        print(
            f"pooled corr (primary evals) n={po['combined']['n']}: "
            f"combined r={po['combined']['pearson']} rho={po['combined']['spearman']} "
            f"K r={po['keys']['pearson']} V r={po['values']['pearson']}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["recon", "profile", "all"], default="all")
    ap.add_argument("--cas-out", default=os.path.expanduser("~/cas_out"))
    ap.add_argument(
        "--out-dir", default=os.path.expanduser("~/cas_out/meta_init_recon")
    )
    ap.add_argument("--patients", nargs="+", default=PATIENTS)
    ap.add_argument("--sizes", nargs="+", default=["1024", "512", "auto"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--topk", type=int, default=8)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("recon", "all"):
        reconstruct(args, out_dir)
    if args.stage in ("profile", "all"):
        t0 = time.time()
        res = {"cas_out": args.cas_out, "topk": args.topk, "groups": {}}
        for gname, spec in GROUPS.items():
            print(f"[profile] group {gname} ...", flush=True)
            res["groups"][gname] = profile_group(
                gname, spec, args.cas_out, out_dir, args.device, args.topk
            )
        res["profile_seconds"] = round(time.time() - t0, 1)
        out = out_dir / "absolute_delta_profile.json"
        out.write_text(json.dumps(res, indent=1))
        print_table(res)
        print(f"\nwrote {out} ({res['profile_seconds']}s)")


if __name__ == "__main__":
    main()

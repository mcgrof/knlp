# SPDX-License-Identifier: GPL-2.0
"""SymKV head-axis spectrum probe: is the KV-head second moment anisotropic
enough to compress at all?

The model-lane result showed head-mode compression is catastrophically lossy at
every rank for every basis, and that u0 carries ~1/H of the energy. This probe
tests the direct explanation: the head second moment C (H x H, per layer and per
K/V) is nearly ISOTROPIC -- its eigenvalues are all ~trace/H, so no rank-m basis
(consensus, PCA, or otherwise) can keep most of the energy with m < H. Reports the
normalized eigenvalue spectrum (eig / trace) averaged over layers, the consensus
energy u0^T C u0 / trace, and a flatness ratio lambda_max/lambda_min. A flat
spectrum near 1/H is the mechanism behind the null.
"""

from __future__ import annotations

import argparse
import json
import sys

import torch


def run(model_id, n_calib, ctx, corpus_path, device, pkg, out_path):
    sys.path.insert(0, pkg)
    from symkv import HeadCovariance, consensus_mode
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import importlib

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="eager").to(device).eval()
    mm = importlib.import_module(model.__class__.__module__)
    orig_repeat_kv, orig_eager = mm.repeat_kv, mm.eager_attention_forward
    H = model.config.num_key_value_heads
    cov = {}

    def hook(orig):
        def fn(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
            li = module.layer_idx
            for kind, T in (("k", key), ("v", value)):
                acc = cov.setdefault((li, kind), HeadCovariance(T.shape[1]))
                acc.update(T.permute(0, 2, 1, 3).reshape(-1, T.shape[1], T.shape[3]))
            ks = orig(key, module.num_key_value_groups)
            vs = orig(value, module.num_key_value_groups)
            aw = torch.matmul(query, ks.transpose(2, 3)) * scaling
            if attention_mask is not None:
                aw = aw + attention_mask[:, :, :, : ks.shape[-2]]
            aw = torch.softmax(aw, dim=-1, dtype=torch.float32).to(query.dtype)
            return torch.matmul(aw, vs).transpose(1, 2).contiguous(), aw
        return fn

    mm.eager_attention_forward = hook(orig_repeat_kv)
    ids0 = tok(open(corpus_path).read()).input_ids
    with torch.no_grad():
        for i in range(n_calib):
            a = (i * 917) % max(1, len(ids0) - ctx)
            model(torch.tensor([ids0[a:a + ctx]], device=device), use_cache=False)
    mm.eager_attention_forward = orig_eager

    u0 = consensus_mode(H)
    rows = []
    for kind in ("k", "v"):
        specs, cons, flat = [], [], []
        for (li, k2), acc in cov.items():
            if k2 != kind:
                continue
            C = acc.covariance()
            tr = torch.diagonal(C).sum()
            w = torch.linalg.eigvalsh(C).flip(0)  # descending
            specs.append((w / tr))
            cons.append(float((u0 @ C @ u0) / tr))
            flat.append(float(w[0] / w[-1].clamp_min(1e-12)))
        spec = torch.stack(specs).mean(0)
        rows.append({
            "kind": kind, "n_layers": len(specs), "H": H,
            "mean_norm_spectrum": [round(float(x), 4) for x in spec],
            "isotropic_ref_1_over_H": round(1.0 / H, 4),
            "consensus_energy_mean": round(sum(cons) / len(cons), 4),
            "flatness_lmax_over_lmin_mean": round(sum(flat) / len(flat), 2),
            "top1_energy_mean": round(float(spec[0]), 4),
            "top_m_cumenergy": {m: round(float(spec[:m].sum()), 4) for m in (1, 2, 4, H)},
        })
        print(f"[{model_id.split('/')[-1]}|{kind}] top1={spec[0]:.3f} (isotropic={1/H:.3f}) "
              f"consensus={rows[-1]['consensus_energy_mean']:.3f} "
              f"flatness={rows[-1]['flatness_lmax_over_lmin_mean']:.1f} "
              f"cum(m=4)={rows[-1]['top_m_cumenergy'][4]:.2f}", flush=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--n-calib", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--pkg", default="/tmp/symkv-stage")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.model, a.n_calib, a.ctx, a.corpus, a.device, a.pkg, a.out)

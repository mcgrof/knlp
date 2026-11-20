#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Gate the architecture port: vendored maths must equal the reference.

Runs on CPU. Nothing here trains, and no sweep should start until every
check below passes, because a silently wrong port produces plausible
numbers rather than an error.
"""

import json

import modal

from modal_utils import VOLUME_MOUNTS, app, build_image


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, timeout=60 * 20)
def check():
    import torch

    import cs312_arch as A
    import mhc_reference as R
    import layerrope_reference as LR

    torch.manual_seed(20261120)
    out = {"equivalence": {}, "models": {}, "failures": []}

    def same(name, a, b):
        ok = torch.equal(a, b)
        out["equivalence"][name] = {
            "bit_exact": bool(ok),
            "max_abs_diff": float((a.float() - b.float()).abs().max()),
        }
        if not ok:
            out["failures"].append(name)

    B, T, M, D = 2, 5, A.MULT, 16
    hidden = torch.randn(B, T, D)
    same("expand_to_mhc", A.expand_to_mhc_ref(hidden, M), R.expand_to_mhc_ref(hidden, M))

    logits = torch.randn(B, T, M, M)
    same("sinkhorn", A.sinkhorn_normalize_ref(logits, 20, 1e-6),
         R.sinkhorn_normalize_ref(logits, 20, 1e-6))

    state = A.expand_to_mhc_ref(hidden, M)
    phi = torch.randn(2 * M + M * M, M * D)
    nw = torch.randn(M * D)
    same("pre_norm_fn", A.mhc_pre_norm_fn_ref(state, phi, nw, 1e-5),
         R.mhc_pre_norm_fn_ref(state, phi, nw, 1e-5))

    mixes = A.mhc_pre_norm_fn_ref(state, phi, nw, 1e-5)
    scale = torch.full((3,), 0.01)
    base = torch.randn(2 * M + M * M)
    a1, a2, a3 = A.mhc_pre_split_mixes_ref(mixes, scale, base, M, 2.0, 0.0)
    b1, b2, b3 = R.mhc_pre_split_mixes_ref(mixes, scale, base, M, 2.0, 0.0)
    same("pre_split.pre", a1, b1)
    same("pre_split.post", a2, b2)
    same("pre_split.carry", a3, b3)

    same("pre_apply_mix", A.mhc_pre_apply_mix_ref(state, a1), R.mhc_pre_apply_mix_ref(state, b1))

    branch = torch.randn(B, T, D)
    carry = A.sinkhorn_normalize_ref(a3, 20, 1e-6)
    same("post", A.mhc_post_ref(branch, state, a2, carry),
         R.mhc_post_ref(branch, state, b2, carry))

    g = torch.randn(D)
    mag = torch.tensor(-0.3)
    th = torch.randn(D // 2)
    same("complex_rotate", A.complex_rotate(g, mag, th), LR.complex_rotate(g, mag, th))

    # ---- the ported models build, run, and differ only where they should ----
    from model_config import depth_model_config
    cfg = depth_model_config(4)
    ids = torch.randint(0, cfg.vocab_size, (2, 32))

    built = {}
    for v in ("baseline", "hc", "mhc", "layerrope", "mhc_layerrope"):
        torch.manual_seed(7)
        m = A.build_arch_model(cfg, dtype=torch.float32, qk_norm=True,
                               tie_word_embeddings=False, dropout=0.0, variant=v,
                               layerrope_alpha=-0.5, layerrope_beta=0.0)
        m.eval()
        with torch.no_grad():
            y = m(ids)
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        built[v] = y
        out["models"][v] = {
            "logits_shape": list(y.shape),
            "finite": bool(torch.isfinite(y).all()),
            "trainable_params": int(n),
            "mean_abs_logit": float(y.float().abs().mean()),
        }
        if not torch.isfinite(y).all():
            out["failures"].append(f"{v}: non-finite logits")

    # hc and mhc must differ (the Sinkhorn constraint is the only difference)
    d_hc_mhc = float((built["hc"].float() - built["mhc"].float()).abs().max())
    out["models"]["hc_vs_mhc_max_abs_diff"] = d_hc_mhc
    if d_hc_mhc == 0.0:
        out["failures"].append("hc and mhc produced identical output")
    # layerrope must change the baseline
    d_base_lr = float((built["baseline"].float() - built["layerrope"].float()).abs().max())
    out["models"]["baseline_vs_layerrope_max_abs_diff"] = d_base_lr
    if d_base_lr == 0.0:
        out["failures"].append("layerrope did not change the model")

    out["passed"] = not out["failures"]
    print(json.dumps(out, indent=2)[:3000], flush=True)
    return out


@app.local_entrypoint()
def main(out: str = ""):
    from pathlib import Path
    payload = check.remote()
    text = json.dumps(payload, indent=2, sort_keys=True)
    if out:
        Path(out).write_text(text + "\n")
        print(f"wrote {out}")
    print("PASSED" if payload.get("passed") else f"FAILED: {payload.get('failures')}")

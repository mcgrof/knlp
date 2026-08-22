# SPDX-License-Identifier: GPL-2.0
"""Probe the RoPE amplification of K temporal locality.

The KV-House milestone measured post-RoPE K carrying 2.3-3.2x the
temporal locality gap of pre-RoPE K. This probe answers three
questions the closure left open, offline and on free compute:

1. Mechanism: is the amplification the shared K component (bias
   and/or mean) being rotated into a smooth position-local
   trajectory? Test: subtract the k_proj bias and, separately, the
   per-head mean in the de-rotated frame, re-rotate, and see
   whether the amplification collapses.
2. Generality: is it Qwen-only (huge K biases) or does any model
   with a large mean K component show it? Run a biased model
   (Qwen2.5), a biasless same-family model (Qwen3), and a biasless
   other-family model (Llama-3.2).
3. Exploitation: does rotation-predictive anchor-delta coding of K
   (de-rotate the block, delta against the anchor, re-rotate on
   decode; zero metadata, positions are known) beat plain
   CacheGen-style anchor-delta at identical bytes on
   attention-output error?

Pre-RoPE K is derived by exact inverse rotation of the cached
post-RoPE K using the model's own cos/sin (hooked from the rotary
module), which sidesteps Qwen3's k_norm sitting between k_proj and
RoPE. On models without k_norm the k_proj output is captured too
and used as a self-check that the rotation math matches the model.
Every coder here keeps quantization noise token-local, per the
milestone mechanism finding.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

import tools.kv.k_bias_common as kbc  # noqa: E402
from kv_house.attention_sim import block_replacement_metrics  # noqa: E402
from kv_house.temporal_stats import (  # noqa: E402
    scattered_block,
    temporal_spectrum,
    top_r_energy,
)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rot(x, cos, sin):
    return x * cos + rotate_half(x) * sin


def derot(x, cos, sin):
    return apply_rot(x, cos, -sin)


def quant_rows(x, bits_first, bits_rest):
    """Token-local symmetric fake-quant: row 0 (anchor) at
    bits_first, remaining rows at bits_rest, one absmax scale per
    row. fp16 rows model a stored fp16 anchor."""
    y = torch.empty_like(x)
    for i in range(x.shape[0]):
        bits = bits_first if i == 0 else bits_rest
        row = x[i]
        if bits >= 16:
            y[i] = row.to(torch.float16).to(row.dtype)
        else:
            qmax = float(2 ** (bits - 1) - 1)
            scale = row.abs().max().clamp_min(1e-12) / qmax
            y[i] = torch.clamp(torch.round(row / scale), -qmax - 1, qmax) * scale
    return y


def locality_gap(stream, b, n_blocks, seed):
    gaps = []
    t = stream.shape[0]
    n = t // b
    take = min(n, n_blocks)
    r = max(1, b // 4)
    for j in range(take):
        idx = j * n // take
        blk = stream[idx * b : (idx + 1) * b]
        real = top_r_energy(temporal_spectrum(blk), r)
        scat = top_r_energy(temporal_spectrum(scattered_block(stream, b, seed + j)), r)
        gaps.append(real - scat)
    return sum(gaps) / len(gaps)


def spaced(t, b, n_blocks):
    n = t // b
    take = min(n, n_blocks)
    return [j * n // take for j in range(take)]


def capture(model_id, device, ctx, n_samples, q_stride, smoke):
    model, tok = kbc.load_model(model_id, "bfloat16", device)
    infos = kbc.discover_attention(model)
    info_by_mod = {id(i["attn_module"]): i for i in infos}
    has_knorm = any(hasattr(i["attn_module"], "k_norm") for i in infos)

    rotary = None
    for m in model.modules():
        if hasattr(m, "inv_freq") and callable(m):
            rotary = m
            break
    assert rotary is not None, "no rotary module found"

    cap = {}
    rope = {}

    def rotary_hook(mod, inp, out):
        cos, sin = out
        rope["cos"] = cos.detach().squeeze(0).float().cpu()
        rope["sin"] = sin.detach().squeeze(0).float().cpu()

    handles = [rotary.register_forward_hook(rotary_hook)]

    def mk_kproj_hook(idx, info):
        def hook(mod, inp, out):
            o = out
            if info["fused"]:
                s0, s1 = info["k_slice"]
                o = out[..., s0:s1]
            t = o.shape[1]
            cap.setdefault(idx, {})["k_proj"] = (
                o.detach()
                .reshape(1, t, -1, info["head_dim"])
                .transpose(1, 2)
                .float()
                .cpu()
                .squeeze(0)
            )

        return hook

    if not has_knorm:
        for info in infos:
            proj = info["k_proj"] if not info["fused"] else info["qkv_proj"]
            if proj is not None:
                handles.append(
                    proj.register_forward_hook(mk_kproj_hook(info["layer_idx"], info))
                )

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    impl = model.config._attn_implementation
    orig = ALL_ATTENTION_FUNCTIONS[impl]

    def attn_hook(module, q, k, v, attention_mask, scaling=None, dropout=0.0, **kw):
        info = info_by_mod.get(id(module))
        if info is not None:
            d = cap.setdefault(info["layer_idx"], {})
            d["k_post"] = k.detach().float().cpu().squeeze(0)
            d["v"] = v.detach().float().cpu().squeeze(0)
            d["q_post"] = q.detach()[:, :, ::q_stride, :].float().cpu().squeeze(0)
        return orig(
            module, q, k, v, attention_mask, dropout=dropout, scaling=scaling, **kw
        )

    ALL_ATTENTION_FUNCTIONS[impl] = attn_hook

    if smoke:
        gen = torch.Generator().manual_seed(0)
        chunks = [
            torch.randint(0, model.config.vocab_size, (ctx,), generator=gen)
            for _ in range(n_samples)
        ]
    else:
        chunks = [
            torch.tensor(c, dtype=torch.long)
            for c in kbc.calib_prompts(tok, n=n_samples, seq_len=ctx)
        ]

    samples = []
    for ids in chunks:
        cap.clear()
        with torch.no_grad():
            model(ids.unsqueeze(0).to(device), use_cache=False)
        samples.append(
            {
                "layers": {k: dict(v) for k, v in cap.items()},
                "cos": rope["cos"].clone(),
                "sin": rope["sin"].clone(),
            }
        )
    for h in handles:
        h.remove()
    ALL_ATTENTION_FUNCTIONS[impl] = orig

    biases = {}
    for info in infos:
        if info.get("has_k_bias") and info.get("k_bias") is not None:
            biases[info["layer_idx"]] = (
                info["k_bias"]
                .detach()
                .float()
                .cpu()
                .reshape(info["n_kv_heads"], info["head_dim"])
            )
    meta = {
        "model": model_id,
        "has_knorm": has_knorm,
        "has_k_bias": bool(biases),
        "n_q": infos[0]["n_q_heads"],
        "n_kv": infos[0]["n_kv_heads"],
        "head_dim": infos[0]["head_dim"],
    }
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return samples, biases, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--q-stride", type=int, default=4)
    ap.add_argument("--layer-stride", type=int, default=2)
    ap.add_argument("--max-heads", type=int, default=4)
    ap.add_argument("--n-blocks", type=int, default=6)
    ap.add_argument("--smoke", action="store_true", help="random tokens, no datasets")
    ap.add_argument("--metrics-device", default="cpu")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples, biases, meta = capture(
        args.model, args.device, args.ctx, args.num_samples, args.q_stride, args.smoke
    )
    print(json.dumps(meta))

    group = meta["n_q"] // meta["n_kv"]
    heads = range(min(meta["n_kv"], args.max_heads))
    gap_rows, coder_rows = [], []
    selfcheck = []

    for si, s in enumerate(samples):
        cos, sin = s["cos"], s["sin"]
        for layer in sorted(s["layers"])[:: args.layer_stride]:
            d = s["layers"][layer]
            for h in heads:
                kp = d["k_post"][h]
                pre = derot(kp, cos, sin)
                if "k_proj" in d:
                    ref = d["k_proj"][h]
                    selfcheck.append(float((pre - ref).norm() / (ref.norm() + 1e-12)))
                variants = {"post": kp, "pre": pre}
                if layer in biases:
                    b_vec = biases[layer][h]
                    variants["pre_nobias"] = pre - b_vec
                    variants["post_nobias"] = apply_rot(pre - b_vec, cos, sin)
                mu = pre.mean(0)
                variants["pre_nomean"] = pre - mu
                variants["post_nomean"] = apply_rot(pre - mu, cos, sin)
                for vname, stream in variants.items():
                    for b in (8, 16, 32):
                        gap_rows.append(
                            {
                                "sample": si,
                                "layer": layer,
                                "head": h,
                                "variant": vname,
                                "block_size": b,
                                "gap": locality_gap(
                                    stream, b, args.n_blocks, seed=si * 97 + layer
                                ),
                            }
                        )

                # phase B: coders at matched bytes, B=16, K only
                b = 16
                mdev = args.metrics_device
                v_full = d["v"][h].to(mdev)
                kp_m = kp.to(mdev)
                q_heads = [d["q_post"][h * group + g] for g in range(group)]
                q_stack = torch.cat(q_heads, 0).to(mdev)
                t_len = kp.shape[0]
                pos = list(range(0, t_len, args.q_stride)) * group
                for j in spaced(t_len, b, min(args.n_blocks, 4)):
                    st, en = j * b, (j + 1) * b
                    blk_post = kp[st:en]
                    blk_pre = pre[st:en]
                    c_blk, s_blk = cos[st:en], sin[st:en]
                    for rest_bits in (4, 2):
                        recs = {}
                        plain = blk_post.clone()
                        plain[1:] = blk_post[1:] - blk_post[0:1]
                        pq = quant_rows(plain, 16, rest_bits)
                        rec = pq.clone()
                        rec[1:] = pq[1:] + pq[0:1]
                        recs["plain_delta"] = rec

                        rot = blk_pre.clone()
                        rot[1:] = blk_pre[1:] - blk_pre[0:1]
                        rq = quant_rows(rot, 16, rest_bits)
                        rdec = rq.clone()
                        rdec[1:] = rq[1:] + rq[0:1]
                        recs["rot_delta"] = apply_rot(rdec, c_blk, s_blk)

                        recs["identity"] = quant_rows(blk_post, 16, rest_bits)

                        res_plain = float(
                            plain[1:].norm() / (blk_post[1:].norm() + 1e-12)
                        )
                        res_rot = float(rot[1:].norm() / (blk_pre[1:].norm() + 1e-12))
                        for cname, k_hat in recs.items():
                            m = block_replacement_metrics(
                                q_stack,
                                pos,
                                kp_m,
                                v_full,
                                st,
                                en,
                                k_hat.to(mdev),
                                v_full[st:en],
                            )
                            if m is None:
                                continue
                            coder_rows.append(
                                {
                                    "sample": si,
                                    "layer": layer,
                                    "head": h,
                                    "block_index": j,
                                    "rest_bits": rest_bits,
                                    "coder": cname,
                                    "res_rel_plain": res_plain,
                                    "res_rel_rot": res_rot,
                                    "k_rel_err": m["k_rel_err"],
                                    "attn_out_rel_err": m["attn_out_rel_err"],
                                    "attn_kl": m["attn_kl"],
                                    "top1": m["top1_agreement"],
                                }
                            )
        print(
            f"sample {si} done: {len(gap_rows)} gap rows, {len(coder_rows)} coder rows"
        )

    (out_dir / "gap_rows.json").write_text(json.dumps(gap_rows))
    (out_dir / "coder_rows.json").write_text(json.dumps(coder_rows))

    def mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else float("nan")

    summary = {
        "meta": meta,
        "rotation_selfcheck_max_rel_err": (max(selfcheck) if selfcheck else None),
    }
    gaps = defaultdict(list)
    for r in gap_rows:
        if r["block_size"] == 16:
            gaps[r["variant"]].append(r["gap"])
    summary["gap_B16"] = {k: mean(v) for k, v in gaps.items()}
    g = summary["gap_B16"]
    for tag in ("", "_nobias", "_nomean"):
        if f"pre{tag}" in g and f"post{tag}" in g and g[f"pre{tag}"]:
            summary[f"amplification{tag or '_raw'}"] = g[f"post{tag}"] / g[f"pre{tag}"]
    coders = defaultdict(list)
    for r in coder_rows:
        coders[(r["coder"], r["rest_bits"])].append(r["attn_out_rel_err"])
    summary["attn_err_by_coder"] = {
        f"{c}_{b}b": mean(v) for (c, b), v in sorted(coders.items())
    }
    summary["residual_rel"] = {
        "plain": mean(r["res_rel_plain"] for r in coder_rows),
        "rot": mean(r["res_rel_rot"] for r in coder_rows),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

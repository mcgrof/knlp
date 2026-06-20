"""Deliverable 2: K activation collection + bias-stress index.

Captures, per layer/head over a small calibration set, the K signal at three points -- pre-bias
(X@W_K), post-bias pre-RoPE (X@W_K + b_K), and post-RoPE -- plus V. Answers Q3: is the K bias
large RELATIVE to the dynamic pre-bias K activation (the thing that actually crushes a per-tensor
FP8 scale)? The stress index is a transparent score (all raw components kept) combining how much
the bias dominates the pre-bias K signal, how much post-RoPE amax inflates over pre-bias amax, and
the per-tensor FP8 scale-loss proxy. Low memory: batch 1, stats aggregated online per prompt.
"""

import argparse
import os

import torch

import k_bias_common as kbc

_CAP = {}  # layer_idx -> dict of captured tensors for the CURRENT prompt
_INFO_BY_MOD = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--short-name", default=None)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--num-prompts", type=int, default=16)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-dir", default="artifacts/k_bias_audit/activation")
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()
    sn = args.short_name or args.model.split("/")[-1]
    trc = args.trust_remote_code or any(
        x in args.model.lower() for x in ("deepseek", "phi")
    )

    model, tok = kbc.load_model(args.model, args.dtype, args.device, trc)
    infos = kbc.discover_attention(model)
    info_by_mod = {id(i["attn_module"]): i for i in infos}
    kproj_to_layer = {}
    handles = []

    def mk_kproj_hook(L, info):
        def hook(mod, inp, out):
            # out: [B,T, n_kv*head_dim] post-bias pre-RoPE K (fused: slice K)
            o = out
            if info["fused"]:
                s0, s1 = info["k_slice"]
                o = out[..., s0:s1]
            _CAP.setdefault(L, {})["post_bias_prerope"] = o.detach().float()

        return hook

    # forward hooks on k_proj / fused qkv for post-bias pre-RoPE K
    for info in infos:
        L = info["layer_idx"]
        proj = info["k_proj"] if not info["fused"] else info["qkv_proj"]
        if proj is not None:
            handles.append(proj.register_forward_hook(mk_kproj_hook(L, info)))

    # attention-interface hook for post-RoPE K + V
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    impl = model.config._attn_implementation
    orig = ALL_ATTENTION_FUNCTIONS[impl]

    def attn_hook(module, q, k, v, attention_mask, scaling=None, dropout=0.0, **kw):
        info = info_by_mod.get(id(module))
        if info is not None:
            L = info["layer_idx"]
            _CAP.setdefault(L, {})["post_rope"] = k.detach().float()  # [B,n_kv,T,hd]
            _CAP[L]["v"] = v.detach().float()
        return orig(
            module, q, k, v, attention_mask, dropout=dropout, scaling=scaling, **kw
        )

    ALL_ATTENTION_FUNCTIONS[impl] = attn_hook

    # online accumulators per (layer): we average per-prompt stats
    nL = len(infos)
    acc = {L: {} for L in range(nL)}
    headacc = {}

    def push(d, key, val):
        d.setdefault(key, []).append(val)

    prompts = kbc.calib_prompts(tok, n=args.num_prompts, seq_len=args.seq_len)
    print(f"[act] {sn}: {len(prompts)} prompts x {args.seq_len} tok, {nL} layers")
    with torch.no_grad():
        for pi, ids in enumerate(prompts):
            _CAP.clear()
            t = torch.tensor(ids).unsqueeze(0).to(args.device)
            model(t)
            for info in infos:
                L = info["layer_idx"]
                cap = _CAP.get(L, {})
                if "post_bias_prerope" not in cap or "post_rope" not in cap:
                    continue
                kb = kbc.k_bias_vector(
                    info, device=args.device, dtype=torch.float32
                )  # [n_kv*hd]
                hd = info["head_dim"]
                n_kv = info["n_kv_heads"]
                post_bias = cap["post_bias_prerope"][0]  # [T, n_kv*hd]
                pre_bias = post_bias - kb  # subtract bias -> dynamic component
                post_rope = cap["post_rope"][0]  # [n_kv, T, hd]
                v = cap["v"][0]  # [n_kv, T, hd]

                def amax(x):
                    return x.abs().amax().item()

                def rms(x):
                    return x.pow(2).mean().sqrt().item()

                def p99(x):
                    a = x.abs().flatten()
                    return (
                        a.sort()
                        .values[min(a.numel() - 1, int(0.99 * a.numel()))]
                        .item()
                    )

                rms_pre, rms_post, rms_rope = (
                    rms(pre_bias),
                    rms(post_bias),
                    rms(post_rope),
                )
                amax_pre, amax_post, amax_rope = (
                    amax(pre_bias),
                    amax(post_bias),
                    amax(post_rope),
                )
                p99_pre = p99(pre_bias)
                bias_rms = kb.pow(2).mean().sqrt().item()
                bias_max = kb.abs().amax().item()
                # dominant channel: channel index of max |bias|; fraction of post_bias amax it owns
                dom_ch = kb.abs().argmax().item()
                ch_amax = post_bias[:, dom_ch].abs().amax().item()
                dom_frac = ch_amax / max(amax_post, 1e-9)
                step_post = amax_post / kbc.FP8_MAX
                step_pre = amax_pre / kbc.FP8_MAX
                fp8_scale_loss = step_post / max(step_pre, 1e-9)
                bulk_loss = p99_pre / max(step_post, 1e-9)
                bias_max_over_preK_p99 = bias_max / max(p99_pre, 1e-9)
                post_rope_over_pre = amax_rope / max(amax_pre, 1e-9)
                stress = (
                    torch.log1p(torch.tensor(bias_max_over_preK_p99)).item()
                    + torch.log1p(torch.tensor(post_rope_over_pre)).item()
                    + torch.log1p(torch.tensor(fp8_scale_loss)).item()
                )
                a = acc[L]
                for k_, v_ in dict(
                    rms_pre_bias_K=rms_pre,
                    rms_post_bias_K=rms_post,
                    rms_post_rope_K=rms_rope,
                    amax_pre_bias_K=amax_pre,
                    amax_post_bias_K=amax_post,
                    amax_post_rope_K=amax_rope,
                    p99_pre_bias_K=p99_pre,
                    bias_rms=bias_rms,
                    bias_max=bias_max,
                    bias_rms_over_preK_rms=bias_rms / max(rms_pre, 1e-9),
                    bias_max_over_preK_p99=bias_max_over_preK_p99,
                    post_bias_amax_over_pre_bias_amax=amax_post / max(amax_pre, 1e-9),
                    post_rope_amax_over_pre_bias_amax=post_rope_over_pre,
                    dominant_channel_fraction=dom_frac,
                    fp8_scale_loss_proxy=fp8_scale_loss,
                    bulk_precision_loss_proxy=bulk_loss,
                    k_bias_stress_index=stress,
                ).items():
                    push(a, k_, v_)

    ALL_ATTENTION_FUNCTIONS[impl] = orig
    for h in handles:
        h.remove()

    # average per-prompt -> per-layer rows
    import statistics as st

    layer_rows = []
    for L in range(nL):
        a = acc[L]
        if not a:
            continue
        row = dict(model=sn, layer=L)
        for k_, vals in a.items():
            row[k_] = st.mean(vals)
        layer_rows.append(row)

    # model summary: mean + max of stress and the key ratios over layers
    def col(name, fn):
        vals = [r[name] for r in layer_rows if name in r]
        return fn(vals) if vals else 0.0

    summary = dict(
        model=sn,
        model_id=args.model,
        num_layers=nL,
        mean_stress=col("k_bias_stress_index", st.mean),
        max_stress=col("k_bias_stress_index", max),
        mean_bias_rms_over_preK_rms=col("bias_rms_over_preK_rms", st.mean),
        max_bias_rms_over_preK_rms=col("bias_rms_over_preK_rms", max),
        mean_post_rope_over_pre=col("post_rope_amax_over_pre_bias_amax", st.mean),
        max_fp8_scale_loss=col("fp8_scale_loss_proxy", max),
        mean_dominant_channel_fraction=col("dominant_channel_fraction", st.mean),
    )
    fields = list(layer_rows[0].keys()) if layer_rows else ["model", "layer"]
    base = os.path.join(args.output_dir, sn)
    kbc.write_csv(os.path.join(base, "activation_stats_layer.csv"), layer_rows, fields)
    kbc.write_json(os.path.join(base, "activation_summary.json"), summary)
    _append_summary(
        os.path.join(args.output_dir, "model_activation_summary.csv"), summary
    )
    _plots(base, layer_rows, sn)
    print(
        f"[act] {sn}: mean_stress={summary['mean_stress']:.2f} "
        f"max_stress={summary['max_stress']:.2f} "
        f"mean_bias_rms/preK={summary['mean_bias_rms_over_preK_rms']:.3f}"
    )


def _append_summary(path, summary):
    import csv

    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if not exists:
            w.writeheader()
        w.writerow(summary)


def _plots(base, rows, sn):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    L = [r["layer"] for r in rows]
    for field, fname in [
        ("bias_max", "max_abs_k_bias_by_layer.png"),
        ("bias_rms_over_preK_rms", "bias_rms_over_preK_by_layer.png"),
        ("post_rope_amax_over_pre_bias_amax", "post_rope_amax_over_pre_by_layer.png"),
        ("k_bias_stress_index", "stress_by_layer.png"),
    ]:
        if field not in rows[0]:
            continue
        plt.figure(figsize=(7, 3))
        plt.plot(L, [r[field] for r in rows], marker="o", ms=3)
        plt.title(f"{sn}: {field}")
        plt.xlabel("layer")
        plt.ylabel(field)
        plt.tight_layout()
        plt.savefig(os.path.join(base, fname), dpi=120)
        plt.close()


if __name__ == "__main__":
    main()

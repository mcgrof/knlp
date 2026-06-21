"""72B audit (device_map sharded, for the H200 lane): bias magnitude + activation stress + FP8
probe in one load. Answers the Tier-2 primary Q1: does Qwen2.5-72B have LOW K-bias stress matching
its observed attenuation, despite carrying a QKV bias? measurement_level=fake_quant + activation_audit.
"""

import argparse
import json
import os

import torch

import _t2common as t2
from _t2common import kbc


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-72B-Instruct")
    ap.add_argument("--short-name", default="qwen25_72b")
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    sn = args.short_name

    model, tok = kbc.load_model(
        args.model, "bfloat16", "cuda:0", t2.trc_for(args.model), device_map="auto"
    )
    infos = kbc.discover_attention(model)
    in_dev = model.get_input_embeddings().weight.device

    # --- bias magnitude ---
    max_abs, rms_list, over_p99 = [], [], []
    for info in infos:
        kb = kbc.k_bias_vector(info, dtype=torch.float32)
        st = kbc.bias_stats(kb)
        max_abs.append(st["max_abs"])
        rms_list.append(st["rms"])
        over_p99.append(st["max_over_p99"])
    has_bias = any(i["has_k_bias"] for i in infos)
    import statistics as S

    bias_summary = dict(
        model=sn,
        model_id=args.model,
        has_k_bias=has_bias,
        n_layers=len(infos),
        max_abs_k_bias=max(max_abs),
        mean_max_abs_k_bias=S.mean(max_abs),
        max_rms_k_bias=max(rms_list),
    )

    # --- activation stress (a few prompts) ---
    info_by_mod = {id(i["attn_module"]): i for i in infos}
    cap = {}
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    impl = model.config._attn_implementation
    orig = ALL_ATTENTION_FUNCTIONS[impl]

    def ah(module, q, k, v, attention_mask, scaling=None, dropout=0.0, **kw):
        info = info_by_mod.get(id(module))
        if info is not None:
            cap[info["layer_idx"]] = k.detach().float()
        return orig(
            module, q, k, v, attention_mask, dropout=dropout, scaling=scaling, **kw
        )

    handles = []
    for info in infos:
        proj = info["k_proj"] if not info["fused"] else info["qkv_proj"]
        if proj is not None:

            def mk(info):
                def h(mod, inp, out):
                    o = (
                        out[..., info["k_slice"][0] : info["k_slice"][1]]
                        if info["fused"]
                        else out
                    )
                    cap[("pb", info["layer_idx"])] = o.detach().float()

                return h

            handles.append(proj.register_forward_hook(mk(info)))
    ALL_ATTENTION_FUNCTIONS[impl] = ah
    stress = []
    ids_list = kbc.calib_prompts(tok, n=args.num_prompts, seq_len=args.seq_len)
    for ids in ids_list:
        cap.clear()
        model(torch.tensor(ids).unsqueeze(0).to(in_dev))
        for info in infos:
            L = info["layer_idx"]
            if ("pb", L) not in cap:
                continue
            kb = kbc.k_bias_vector(
                info, device=cap[("pb", L)].device, dtype=torch.float32
            )
            post_bias = cap[("pb", L)][0]
            pre = post_bias - kb
            p99 = pre.abs().flatten().sort().values
            p99 = p99[min(p99.numel() - 1, int(0.99 * p99.numel()))].item()
            stress.append(kb.abs().amax().item() / max(p99, 1e-9))
    ALL_ATTENTION_FUNCTIONS[impl] = orig
    for h in handles:
        h.remove()
    bias_summary["mean_bias_max_over_preK_p99"] = S.mean(stress) if stress else 0.0

    # --- FP8 probe (normal vs prebias vs k16v8) ---
    base = t2.logits_list(model, ids_list, in_dev)
    fp8 = {}
    for name, (k, v, pb) in {
        "normal_fp8": ("fp8:per_tensor", "fp8:per_tensor", False),
        "prebias_fp8": ("fp8:per_tensor", "fp8:per_tensor", True),
        "k16v8": ("bf16", "fp8:per_tensor", False),
    }.items():
        h = kbc.FlexKVHarness(
            model, infos, kbc.parse_spec(k), kbc.parse_spec(v), prebias=pb
        )
        h.install()
        try:
            fp8[name] = t2.metrics(base, t2.logits_list(model, ids_list, in_dev))[
                "mean_logit_err"
            ]
        finally:
            h.remove()
    bias_summary.update(
        fp8_normal_err=fp8["normal_fp8"],
        fp8_prebias_err=fp8["prebias_fp8"],
        fp8_k16v8_err=fp8["k16v8"],
    )
    json.dump(
        bias_summary,
        open(os.path.join(args.output_dir, "qwen72b_audit.json"), "w"),
        indent=2,
    )
    print(
        f"[72b] {sn}: has_bias={has_bias} max|Kbias|={bias_summary['max_abs_k_bias']:.1f} "
        f"stress(bias_max/preKp99)={bias_summary['mean_bias_max_over_preK_p99']:.2f} "
        f"fp8_normal={fp8['normal_fp8']:.3f} prebias={fp8['prebias_fp8']:.3f} k16v8={fp8['k16v8']:.3f}"
    )


if __name__ == "__main__":
    main()

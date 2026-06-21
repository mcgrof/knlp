"""72B audit (H200 lane): K-bias magnitude + FP8 tolerance. Answers Tier-2 Q1 -- does Qwen2.5-72B
attenuate (small / non-dominant K-bias despite carrying a QKV bias)?

NB on device_map: accelerate keeps CPU/disk-offloaded params as META even inside a forward hook, so
the bias CANNOT be read from the live module. We therefore read the K-bias magnitudes straight from
the safetensors shards (no model load -- robust), and run only the bias-INDEPENDENT FP8 cells
(normal-FP8, K16/V8) through a device_map forward to confirm tolerance. Pre-bias / bias-stress need
the live bias and are skipped on the 72B lane (the magnitude + normal-FP8 err already answer Q1).
measurement_level=weights + fake_quant.
"""

import argparse
import glob
import json
import os
import statistics as S

import torch

import _t2common as t2
from _t2common import kbc


def bias_from_safetensors(model_id):
    """Read max|k_proj.bias| per layer directly from the downloaded safetensors (no model load)."""
    from safetensors import safe_open

    pat = "models--" + model_id.replace("/", "--")
    snaps = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/{pat}/snapshots/*"))
    if not snaps:
        return None
    maxabs, rms = [], []
    for fp in glob.glob(os.path.join(snaps[0], "*.safetensors")):
        with safe_open(fp, framework="pt") as f:
            for k in f.keys():
                if k.endswith("k_proj.bias"):
                    tns = f.get_tensor(k).float()
                    maxabs.append(tns.abs().max().item())
                    rms.append(tns.pow(2).mean().sqrt().item())
    if not maxabs:
        return dict(has_k_bias=False, n_kbias_layers=0, max_abs_k_bias=0.0)
    return dict(
        has_k_bias=True,
        n_kbias_layers=len(maxabs),
        max_abs_k_bias=max(maxabs),
        mean_max_abs_k_bias=S.mean(maxabs),
        max_rms_k_bias=max(rms),
    )


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-72B-Instruct")
    ap.add_argument("--short-name", default="qwen25_72b")
    ap.add_argument("--num-prompts", type=int, default=6)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    summ = dict(model=args.short_name, model_id=args.model)
    b = bias_from_safetensors(args.model)
    if b:
        summ.update(b)

    model, tok = kbc.load_model(
        args.model, "bfloat16", "cuda:0", t2.trc_for(args.model), device_map="auto"
    )
    infos = kbc.discover_attention(model)
    dev = model.get_input_embeddings().weight.device
    ids = kbc.calib_prompts(tok, n=args.num_prompts, seq_len=args.seq_len)
    base = t2.logits_list(model, ids, dev)
    # only bias-INDEPENDENT cells (normal-FP8 / K16-V8); pre-bias needs the live bias -> skip on 72B
    for name, (k, v) in {
        "normal_fp8": ("fp8:per_tensor", "fp8:per_tensor"),
        "k16v8": ("bf16", "fp8:per_tensor"),
    }.items():
        h = kbc.FlexKVHarness(model, infos, kbc.parse_spec(k), kbc.parse_spec(v))
        h.install()
        try:
            summ[f"fp8_{name}_err" if name == "normal_fp8" else "fp8_k16v8_err"] = (
                t2.metrics(base, t2.logits_list(model, ids, dev))["mean_logit_err"]
            )
        finally:
            h.remove()
    summ["fp8_normal_err"] = summ.pop("fp8_normal_fp8_err", summ.get("fp8_normal_err"))
    json.dump(
        summ, open(os.path.join(args.output_dir, "qwen72b_audit.json"), "w"), indent=2
    )
    print(
        f"[72b] max|Kbias|={summ.get('max_abs_k_bias'):.2f} (n={summ.get('n_kbias_layers')}) "
        f"normal_fp8_err={summ.get('fp8_normal_err'):.3f} k16v8_err={summ.get('fp8_k16v8_err'):.3f} "
        f"-> {'TOLERANT' if summ.get('fp8_normal_err', 1) < 0.30 else 'sensitive'}"
    )


if __name__ == "__main__":
    main()

"""Phase 2: larger fake-quant probe across FP8 variants + recovery fraction.

Variants: bf16, normal-FP8 unit-scale, normal-FP8 absmax-per-tensor, pre-bias K FP8, per-channel K
FP8, K16/V8, K16/V4. Metrics vs BF16: NLL delta, mean/max logit err, top-1/top-5, INT8-vs-INT6 K
ratio. recovery = (normal_err - prebias_err)/(normal_err - bf16_err). measurement_level=fake_quant.
"""

import argparse
import csv
import json
import os

import torch

import _t2common as t2
from _t2common import kbc

VARIANTS = {
    "bf16_baseline": dict(k="bf16", v="bf16", prebias=False, unit=False),
    "normal_fp8_unit_scale": dict(
        k="fp8:per_tensor", v="fp8:per_tensor", prebias=False, unit=True
    ),
    "normal_fp8_absmax_per_tensor": dict(
        k="fp8:per_tensor", v="fp8:per_tensor", prebias=False, unit=False
    ),
    "prebias_k_fp8_v_fp8": dict(
        k="fp8:per_tensor", v="fp8:per_tensor", prebias=True, unit=False
    ),
    "perchannel_k_fp8_v_fp8": dict(
        k="fp8:per_channel", v="fp8:per_tensor", prebias=False, unit=False
    ),
    "k16_v8": dict(k="bf16", v="fp8:per_tensor", prebias=False, unit=False),
    "k16_v4": dict(k="bf16", v="int4:per_token", prebias=False, unit=False),
}


@torch.no_grad()
def run_variant(model, infos, ids_list, device, cfg):
    k, v = kbc.parse_spec(cfg["k"]), kbc.parse_spec(cfg["v"])
    h = kbc.FlexKVHarness(
        model, infos, k, v, prebias=cfg["prebias"], unit_scale=cfg["unit"]
    )
    h.install()
    try:
        return t2.logits_list(model, ids_list, device)
    finally:
        h.remove()


@torch.no_grad()
def k_only(model, infos, ids_list, device, spec):
    h = kbc.FlexKVHarness(model, infos, kbc.parse_spec(spec), kbc.parse_spec("bf16"))
    h.install()
    try:
        return t2.logits_list(model, ids_list, device)
    finally:
        h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-file")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--num-prompts", type=int, default=64)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--skip-large", action="store_true")
    args = ap.parse_args()
    specs = t2.load_models_spec(args.models_file, args.models, args.only)
    if args.skip_large:
        specs = [m for m in specs if m.get("tier") != "large"]
    os.makedirs(args.output_dir, exist_ok=True)
    summ_path = os.path.join(args.output_dir, "fp8_variant_summary.csv")
    jsonl = open(os.path.join(args.output_dir, "fp8_variant_probe.jsonl"), "a")
    rec_md = [
        "# Pre-bias recovery table (measurement_level=fake_quant)",
        "",
        "| model | normal-FP8 err | prebias err | K16/V8 err | recovery frac | INT8/INT6 |",
        "|---|---|---|---|---|---|",
    ]

    for m in specs:
        mid, sn = m["model_id"], m["short_name"]
        try:
            model, tok = kbc.load_model(mid, args.dtype, args.device, t2.trc_for(mid))
            infos = kbc.discover_attention(model)
            ids = kbc.calib_prompts(tok, n=args.num_prompts, seq_len=args.seq_len)
            base = run_variant(
                model, infos, ids, args.device, VARIANTS["bf16_baseline"]
            )
            summary = dict(
                model=sn,
                has_k_bias=any(i["has_k_bias"] for i in infos),
                n_prompts=len(ids),
                seq_len=args.seq_len,
            )
            errs = {}
            for name, cfg in VARIANTS.items():
                if name == "bf16_baseline":
                    continue
                lg = run_variant(model, infos, ids, args.device, cfg)
                mt = t2.metrics(base, lg)
                errs[name] = mt["mean_logit_err"]
                for kk, vv in mt.items():
                    summary[f"{name}__{kk}"] = vv
                jsonl.write(json.dumps(dict(model=sn, variant=name, **mt)) + "\n")
            # INT8/INT6 K-only ratio
            e8 = t2.metrics(
                base, k_only(model, infos, ids, args.device, "int8:per_tensor")
            )["mean_logit_err"]
            e6 = t2.metrics(
                base, k_only(model, infos, ids, args.device, "int6:per_tensor")
            )["mean_logit_err"]
            ratio = e6 / max(e8, 1e-9)
            normal = errs["normal_fp8_absmax_per_tensor"]
            preb = errs["prebias_k_fp8_v_fp8"]
            rec = t2.recovery(normal, preb)
            summary.update(int8_int6_ratio=ratio, recovery_fraction=rec)
            rec_md.append(
                f"| {sn} | {normal:.3f} | {preb:.3f} | {errs['k16_v8']:.3f} | "
                f"{rec:.2f} | {ratio:.1f} |"
            )
            _append(summ_path, summary)
            print(
                f"[fp8var] {sn}: normal={normal:.3f} prebias={preb:.3f} k16v8={errs['k16_v8']:.3f} "
                f"recovery={rec:.2f} int8/6={ratio:.1f}"
            )
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[FAIL] {sn}: {type(e).__name__}: {str(e)[:140]}")
    jsonl.close()
    with open(os.path.join(args.output_dir, "prebias_recovery_table.md"), "w") as f:
        f.write("\n".join(rec_md))
    print(f"[fp8var] -> {args.output_dir}")


def _append(path, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    main()

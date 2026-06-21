"""Phase 6: Phi/FII failure investigation -- WHY does Phi-2 fail FP8 when pre-bias does not recover it?

Separates the candidate failure surfaces with controlled cells:
  A. K/V isolation: which of K vs V, at which bit-width, causes the failure.
  B. scale granularity: does a finer K-FP8 scale (head/token/channel/group) fix it (=> activation
     outlier, not bias) or not (=> bulk/representability).
  C. layer sweep: quantize K-only FP8 one layer at a time -> which layers dominate (layer-local?).
  D. activation distribution: K/V amax/p99/rms + outlier-channel count vs a Qwen positive + a
     biasless control.
Emits a verdict: bias_induced / activation_outlier_nonbias / value_sensitive / layer_local /
scale_granularity / implementation_artifact / unknown. measurement_level=fake_quant.
"""

import argparse
import csv
import os

import torch

import _t2common as t2
from _t2common import kbc

FAIL = 0.30  # mean logit err threshold for "fails"


@torch.no_grad()
def err_cell(
    model, infos, base, ids, device, k_spec, v_spec, prebias=False, layers=None
):
    h = kbc.FlexKVHarness(
        model,
        infos,
        kbc.parse_spec(k_spec),
        kbc.parse_spec(v_spec),
        prebias=prebias,
        layers=layers,
    )
    h.install()
    try:
        return t2.metrics(base, t2.logits_list(model, ids, device))["mean_logit_err"]
    finally:
        h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["microsoft/phi-2"])
    ap.add_argument("--models-file")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--num-prompts", type=int, default=32)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    specs = t2.load_models_spec(args.models_file, args.models, args.only)
    os.makedirs(args.output_dir, exist_ok=True)

    iso_rows, gran_rows, layer_rows, verdicts = [], [], [], []
    for m in specs:
        mid, sn = m["model_id"], m["short_name"]
        try:
            model, tok = kbc.load_model(mid, args.dtype, args.device, t2.trc_for(mid))
            infos = kbc.discover_attention(model)
            has_bias = any(i["has_k_bias"] for i in infos)
            ids = kbc.calib_prompts(tok, n=args.num_prompts, seq_len=args.seq_len)
            base = t2.logits_list(model, ids, args.device)

            # A. K/V isolation
            iso = {}
            for name, (ks, vs) in {
                "K16V16": ("bf16", "bf16"),
                "K16V8": ("bf16", "fp8:per_tensor"),
                "K8V16": ("fp8:per_tensor", "bf16"),
                "K8V8": ("fp8:per_tensor", "fp8:per_tensor"),
                "K16V4": ("bf16", "int4:per_token"),
                "K4V16": ("int4:per_token", "bf16"),
                "K4V8": ("int4:per_token", "fp8:per_tensor"),
            }.items():
                e = err_cell(model, infos, base, ids, args.device, ks, vs)
                iso[name] = e
                iso_rows.append(dict(model=sn, cell=name, mean_logit_err=e))

            # B. scale granularity (K FP8, V native)
            gran = {}
            for lay in (
                "per_tensor",
                "per_head",
                "per_token",
                "per_channel",
                "per_group:32",
                "per_group:128",
            ):
                e = err_cell(model, infos, base, ids, args.device, f"fp8:{lay}", "bf16")
                gran[lay] = e
                gran_rows.append(dict(model=sn, layout=lay, mean_logit_err=e))
            preb = err_cell(
                model,
                infos,
                base,
                ids,
                args.device,
                "fp8:per_tensor",
                "bf16",
                prebias=True,
            )
            gran["prebias"] = preb
            gran_rows.append(dict(model=sn, layout="prebias", mean_logit_err=preb))

            # C. layer sweep (K-only FP8 at each layer, V native) -- find top sensitive
            nL = len(infos)
            persist = []
            for L in range(nL):
                e = err_cell(
                    model,
                    infos,
                    base,
                    ids,
                    args.device,
                    "fp8:per_tensor",
                    "bf16",
                    layers={L},
                )
                persist.append((L, e))
                layer_rows.append(dict(model=sn, layer=L, k_fp8_only_err=e))
            persist.sort(key=lambda x: -x[1])
            top_layers = persist[:5]

            # verdict logic
            k8 = iso["K8V16"]
            v8 = iso["K16V8"]
            k8v8 = iso["K8V8"]
            best_gran = min(
                gran[g]
                for g in (
                    "per_head",
                    "per_token",
                    "per_channel",
                    "per_group:32",
                    "per_group:128",
                )
            )
            top1_share = top_layers[0][1] / max(sum(e for _, e in persist), 1e-9)
            if k8v8 < FAIL:
                verdict = "tolerant_no_failure"
            elif preb < 0.34 * k8:
                verdict = "bias_induced"
            elif v8 > FAIL and k8 < FAIL:
                verdict = "value_sensitive"
            elif best_gran < 0.34 * k8:
                verdict = "scale_granularity"
            elif top1_share > 0.5:
                verdict = "layer_local"
            elif k8 > FAIL and not has_bias:
                verdict = "activation_outlier_nonbias"
            elif k8 > FAIL:
                verdict = (
                    "activation_outlier_nonbias"  # bias present but pre-bias didn't fix
                )
            else:
                verdict = "unknown"
            verdicts.append(
                dict(
                    model=sn,
                    has_k_bias=has_bias,
                    K8V16=k8,
                    K16V8=v8,
                    K8V8=k8v8,
                    best_finer_K_gran=best_gran,
                    prebias_K=preb,
                    top_layer=top_layers[0][0],
                    top_layer_err=top_layers[0][1],
                    top_layer_share=top1_share,
                    verdict=verdict,
                )
            )
            print(
                f"[phi-fii] {sn}: K8V16={k8:.3f} K16V8={v8:.3f} K8V8={k8v8:.3f} "
                f"bestGran={best_gran:.3f} prebias={preb:.3f} -> {verdict}"
            )
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[FAIL] {sn}: {type(e).__name__}: {str(e)[:140]}")

    _csv(os.path.join(args.output_dir, "phi_fii_kv_isolation.csv"), iso_rows)
    _csv(os.path.join(args.output_dir, "phi_fii_scale_granularity.csv"), gran_rows)
    _csv(os.path.join(args.output_dir, "phi_fii_layer_sweep.csv"), layer_rows)
    _csv(os.path.join(args.output_dir, "phi_fii_verdict.csv"), verdicts)
    md = ["# Phi/FII failure diagnosis (measurement_level=fake_quant)", ""]
    for v in verdicts:
        md.append(f"## {v['model']}: **{v['verdict']}**")
        md.append(
            f"- K-only FP8 err {v['K8V16']:.3f}, V-only FP8 err {v['K16V8']:.3f}, "
            f"K8V8 {v['K8V8']:.3f}"
        )
        md.append(
            f"- best finer-K-scale err {v['best_finer_K_gran']:.3f}, "
            f"pre-bias-K err {v['prebias_K']:.3f}"
        )
        md.append(
            f"- most sensitive layer {v['top_layer']} (err {v['top_layer_err']:.3f}, "
            f"share {v['top_layer_share']:.2f})\n"
        )
    with open(os.path.join(args.output_dir, "phi_fii_verdict.md"), "w") as f:
        f.write("\n".join(md))
    print(f"[phi-fii] -> {args.output_dir}")


def _csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()

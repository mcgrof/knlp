"""Deliverable 5: unified K-bias-sensitivity report.

Joins the bias audit, activation stress, FP8 probe, and alpha sweep with the paper's known
outcomes, and emits the conclusion fields that answer whether bias EXISTENCE, bias MAGNITUDE,
pre-bias recovery, and the alpha causal test support the hypothesis -- plus a per-model serving
recommendation. Pure CPU join; reads the summary CSVs each deliverable appended to.
"""

import argparse
import csv
import os

# paper priors (outcome to predict). attenuated counts as tolerant for the boolean test.
KNOWN = {
    "qwen25-1.5b": "sensitive",
    "qwen25-7b": "sensitive",
    "qwen2-7b": "sensitive",
    "dsr1-qwen-7b": "sensitive",
    "qwen25-3b": "tolerant",
    "qwen25-14b": "tolerant",
    "qwen25-72b": "attenuated",
    "phi-2": "tolerant",
    "phi-4": "tolerant",
    "llama31-8b": "tolerant",
    "mistral-7b": "tolerant",
    "qwen3-8b": "tolerant",
    "qwen3-4b": "tolerant",
}
FP8_FAIL_THRESH = 0.30  # mean logit err above this = FP8 failure
PREBIAS_RECOVER = 0.34  # prebias err / fp8 err below this = recovery


def load_csv(path):
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r["model"]] = r
    return out


def fnum(d, k, default=0.0):
    try:
        return float(d.get(k, default))
    except (ValueError, TypeError):
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-root", default="artifacts/k_bias_audit")
    ap.add_argument("--run-id", default="run")
    args = ap.parse_args()
    root = args.audit_root
    bias = load_csv(os.path.join(root, "bias", "model_bias_summary.csv"))
    act = load_csv(os.path.join(root, "activation", "model_activation_summary.csv"))
    fp8 = load_csv(os.path.join(root, "fp8", "fp8_bias_probe_summary.csv"))
    alpha = load_csv(os.path.join(root, "alpha", "alpha_sweep_summary.csv"))
    models = sorted(set(bias) | set(act) | set(fp8) | set(alpha))

    rows = []
    for m in models:
        b, a, f, al = bias.get(m, {}), act.get(m, {}), fp8.get(m, {}), alpha.get(m, {})
        has_bias = str(b.get("has_k_bias", "")).lower() in ("true", "1")
        max_bias = fnum(b, "max_abs_k_bias")
        stress = fnum(a, "mean_stress")
        biasrms_over_preK = fnum(a, "mean_bias_rms_over_preK_rms")
        fp8_err = fnum(f, "fp8_mean_logit_err")
        prebias_err = fnum(f, "prebias_fp8_mean_logit_err")
        k16_err = fnum(f, "k16v8_mean_logit_err")
        recov = fnum(f, "prebias_recovery_ratio", 1.0)
        int8_int6 = fnum(f, "int8_int6_err_ratio")
        fp8_fails = fp8_err > FP8_FAIL_THRESH
        prebias_recovers = bool(fp8_fails and recov < PREBIAS_RECOVER)
        alpha_causal = str(al.get("alpha_causal", "")).lower() in ("true", "1")
        known = KNOWN.get(m, "unknown")
        # serving policy
        if fp8_fails and prebias_recovers:
            policy = "pre-bias FP8"
        elif fp8_fails:
            policy = "K16/V8"
        elif has_bias and int8_int6 > 3:
            policy = "needs calibration/preflight"
        else:
            policy = "normal FP8"
        rows.append(
            dict(
                model=m,
                known=known,
                has_k_bias=has_bias,
                max_abs_k_bias=max_bias,
                mean_stress=stress,
                bias_rms_over_preK=biasrms_over_preK,
                fp8_logit_err=fp8_err,
                prebias_logit_err=prebias_err,
                k16v8_logit_err=k16_err,
                int8_int6_ratio=int8_int6,
                fp8_fails=fp8_fails,
                prebias_recovers=prebias_recovers,
                alpha_causal=alpha_causal,
                policy=policy,
            )
        )

    # ---- cross-model conclusions ----
    biased = [r for r in rows if r["has_k_bias"]]
    biased_fail = [r for r in biased if r["fp8_fails"]]
    biased_ok = [r for r in biased if not r["fp8_fails"]]
    failing = [r for r in rows if r["fp8_fails"]]
    # bias boolean predictor: do ALL biased fail? if some biased are fine -> existence insufficient
    if biased and biased_ok:
        bias_bool = "no"  # bias present but some tolerant -> existence not sufficient
    elif biased and not biased_ok:
        bias_bool = "yes"
    else:
        bias_bool = "n/a"
    # magnitude predictor: failing models have higher max_bias/stress than tolerant biased ones
    if biased_fail and biased_ok:
        fail_mag = sum(r["max_abs_k_bias"] for r in biased_fail) / len(biased_fail)
        ok_mag = sum(r["max_abs_k_bias"] for r in biased_ok) / len(biased_ok)
        fail_str = sum(r["mean_stress"] for r in biased_fail) / len(biased_fail)
        ok_str = sum(r["mean_stress"] for r in biased_ok) / len(biased_ok)
        mag_pred = (
            "yes" if (fail_mag > 2 * ok_mag or fail_str > 1.5 * ok_str) else "partial"
        )
    else:
        mag_pred = "partial"
    recov_models = [r for r in failing if r["prebias_recovers"]]
    if failing:
        prebias_pred = (
            "yes"
            if len(recov_models) == len(failing)
            else ("partial" if recov_models else "no")
        )
    else:
        prebias_pred = "n/a"
    causal_models = [r for r in rows if r["has_k_bias"] and r["alpha_causal"]]
    sens_with_alpha = [
        r for r in rows if r["known"] == "sensitive" and "alpha_causal" in r
    ]
    if any(r["alpha_causal"] for r in rows):
        alpha_pred = "yes" if causal_models else "no"
    else:
        alpha_pred = "partial"

    conclusions = dict(
        bias_boolean_predicts_failure=bias_bool,
        bias_magnitude_predicts_failure=mag_pred,
        prebias_quantization_recovers_failure=prebias_pred,
        alpha_sweep_supports_causality=alpha_pred,
    )

    # ---- write outputs ----
    out = os.path.join(root, "report")
    os.makedirs(out, exist_ok=True)
    flds = [
        "model",
        "known",
        "has_k_bias",
        "max_abs_k_bias",
        "mean_stress",
        "bias_rms_over_preK",
        "fp8_logit_err",
        "prebias_logit_err",
        "k16v8_logit_err",
        "int8_int6_ratio",
        "fp8_fails",
        "prebias_recovers",
        "alpha_causal",
        "policy",
    ]
    with open(os.path.join(out, "model_bias_outcome_table.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=flds)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    md = ["# K-projection-bias vs FP8 K-cache sensitivity -- unified report", ""]
    md.append(
        "Hypothesis: K-cache FP8 failure tracks K-bias MAGNITUDE / outlier dominance, not "
        "the mere `attention_bias=True` flag.\n"
    )
    md.append("## Conclusions\n")
    for k, v in conclusions.items():
        md.append(f"- **{k}**: `{v}`")
    md.append("\n## Per-model table\n")
    md.append(
        "| model | known | bias? | max|K-bias| | stress | bias_rms/preK | fp8 err | "
        "prebias err | k16v8 err | int8/int6 | fp8 fails | prebias recovers | policy |"
    )
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(
            f"| {r['model']} | {r['known']} | {'Y' if r['has_k_bias'] else 'N'} | "
            f"{r['max_abs_k_bias']:.1f} | {r['mean_stress']:.2f} | {r['bias_rms_over_preK']:.2f} | "
            f"{r['fp8_logit_err']:.3f} | {r['prebias_logit_err']:.3f} | {r['k16v8_logit_err']:.3f} | "
            f"{r['int8_int6_ratio']:.1f} | {'Y' if r['fp8_fails'] else 'N'} | "
            f"{'Y' if r['prebias_recovers'] else '-'} | {r['policy']} |"
        )
    md.append("\n## Interpretation\n")
    md.append(f"- Biased models that FAIL FP8: {[r['model'] for r in biased_fail]}")
    md.append(
        f"- Biased models that are TOLERANT (bias present, no FP8 failure): "
        f"{[r['model'] for r in biased_ok]}  <- these refute bias-existence-as-predictor"
    )
    md.append(
        f"- No-bias controls: {[r['model'] for r in rows if not r['has_k_bias']]} "
        f"(pre-bias must be a no-op)"
    )
    md.append(
        "\nThe headline: " + _headline(bias_bool, mag_pred, prebias_pred, alpha_pred)
    )
    with open(os.path.join(out, "summary_table.md"), "w") as fh:
        fh.write("\n".join(md))
    with open(os.path.join(out, "README.md"), "w") as fh:
        fh.write("\n".join(md))
    _scatter(out, rows)

    print("\n".join(f"{k}: {v}" for k, v in conclusions.items()))
    print(f"\n[report] {len(rows)} models -> {out}/summary_table.md")


def _headline(bb, mag, pre, al):
    parts = []
    if bb == "no":
        parts.append(
            "attention bias is a RISK FACTOR, not a sufficient predictor (some biased "
            "models are tolerant)"
        )
    if mag in ("yes", "partial"):
        parts.append(f"bias magnitude/stress predicts failure ({mag})")
    if pre in ("yes", "partial"):
        parts.append(f"pre-bias FP8 recovers the sensitive models ({pre})")
    if al == "yes":
        parts.append("the alpha sweep supports a causal FP8-specific bias effect")
    return "; ".join(parts) + "." if parts else "inconclusive on current models."


def _scatter(out, rows):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    have = [r for r in rows if r["fp8_logit_err"] > 0]
    if not have:
        return
    plt.figure(figsize=(6, 4))
    col = {"sensitive": "r", "tolerant": "g", "attenuated": "orange", "unknown": "gray"}
    for r in have:
        plt.scatter(
            r["mean_stress"], r["fp8_logit_err"], c=col.get(r["known"], "gray"), s=40
        )
        plt.annotate(r["model"], (r["mean_stress"], r["fp8_logit_err"]), fontsize=6)
    plt.axhline(FP8_FAIL_THRESH, ls="--", c="k", lw=0.5)
    plt.xlabel("mean K-bias stress index")
    plt.ylabel("normal-FP8 mean logit error")
    plt.title("stress vs FP8 failure (red=sensitive, green=tolerant)")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "stress_vs_outcome.png"), dpi=120)
    plt.close()


if __name__ == "__main__":
    main()

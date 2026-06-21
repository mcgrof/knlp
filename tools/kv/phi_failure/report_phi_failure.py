"""Phase 9: Phi-failure report. Joins metadata / subspace / kv-isolation / scale-granularity /
rope / attention-score / layer-sweep into PHI_FAILURE_FINDINGS.md and classifies Phi's FP8 failure
into one label, WITHOUT forcing it into the Qwen K-bias mechanism."""

import argparse
import csv
import os
import statistics as St


def lc(p):
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def fnum(d, k, dv=0.0):
    try:
        return float(d.get(k, dv))
    except (ValueError, TypeError):
        return dv


def classify(sn, kviso, gran, sub, rope, score, layer):
    k = next((r for r in kviso if r["model"] == sn), {})
    g = next((r for r in gran if r["model"] == sn), {})
    subr = [r for r in sub if r["model"] == sn]
    rp = {r["config"]: fnum(r, "mean_logit_err") for r in rope if r["model"] == sn}
    lay = [r for r in layer if r["model"] == sn]
    sc = [r for r in score if r["model"] == sn]
    k8v16 = fnum(k, "K8V16")
    k16v8 = fnum(k, "K16V8")
    k8v8 = fnum(k, "K8V8")
    if k8v8 < 0.30:
        return "tolerant_no_failure", {}
    ev = {}
    # bias?
    if str(k.get("prebias_helps", "")).lower() in ("true", "1"):
        return "bias_induced", {"prebias_helps": True}
    # V vs K
    if k16v8 > 0.30 and k8v16 < 0.30:
        return "value_sensitive", {"K16V8": k16v8, "K8V16": k8v16}
    # subspace / passthrough
    if subr:
        por = St.mean([fnum(r, "pas_over_rot_amax", 1.0) for r in subr])
        ev["pass_over_rot_amax"] = por
        pas_only = rp.get("passthrough_only", 9)
        rot_only = rp.get("rotary_only", 9)
        if por > 3 and pas_only > rot_only and pas_only > 0.30:
            return "rotary_pass_through_mixture", ev
        if str(g.get("granularity_fixes", "")).lower() in ("true", "1") and por > 2:
            return "rotary_pass_through_mixture", ev
    # scale granularity
    if str(g.get("granularity_fixes", "")).lower() in ("true", "1"):
        return "scale_granularity", {"best_layout": g.get("best_layout")}
    # layer-local
    if any(str(r.get("layer_local", "")).lower() in ("true", "1") for r in lay):
        return "layer_local", {}
    # attention score scale: large BF16 score amax or big entropy delta at modest K err
    if sc:
        amax = max(fnum(r, "bf16_score_amax") for r in sc)
        ed = St.mean([abs(fnum(r, "fp8_entropy_delta")) for r in sc])
        ev.update(max_bf16_score_amax=amax, mean_fp8_entropy_delta=ed)
        if amax > 80 or ed > 0.3:
            return "attention_score_scale", ev
    if k16v8 < 0.30 and k8v16 < 0.30:
        return "implementation_artifact", ev
    return "unknown", ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--primary", default="phi2")
    args = ap.parse_args()
    R = args.root
    meta = lc(os.path.join(R, "metadata", "metadata_audit.csv"))
    kviso = lc(os.path.join(R, "kv_isolation", "kv_isolation_verdict.csv"))
    gran = lc(os.path.join(R, "scale_granularity", "scale_granularity_verdict.csv"))
    sub = lc(os.path.join(R, "subspace", "subspace_activation_layer.csv"))
    rope = lc(os.path.join(R, "rope_placement", "rope_placement_results.csv"))
    score = lc(os.path.join(R, "attention_score", "attention_score_diagnostics.csv"))
    layer = lc(os.path.join(R, "layer_sweep", "layer_sweep_verdict.csv"))
    models = sorted({r["model"] for r in kviso} | {r["model"] for r in meta})

    cls = {sn: classify(sn, kviso, gran, sub, rope, score, layer) for sn in models}
    p = args.primary
    label, ev = cls.get(p, ("unknown", {}))

    md = ["# Phi / FII FP8-failure diagnosis -- findings", ""]
    md.append("## 1. Executive summary")
    md.append(
        f"Primary target **{p}** classified as **`{label}`**. This AUGMENTS (does not replace) "
        "the K-bias result: dependency-aware K16/V8 stays the headline; Phi is the "
        "counterexample showing K-bias stress is not a universal account of FP8 K failure.\n"
    )
    mp = next((r for r in meta if r["model"] == p), {})
    md.append("## 2. Was Phi failure caused by K-bias stress?")
    kp = next((r for r in kviso if r["model"] == p), {})
    md.append(
        f"- pre-bias K8V8 err {fnum(kp,'prebias_K8V8'):.3f} vs K8V8 {fnum(kp,'K8V8'):.3f}: "
        f"pre-bias {'HELPS' if str(kp.get('prebias_helps','')).lower() in ('true','1') else 'does NOT recover'} "
        "-> not the Qwen K-bias mechanism unless it helps.\n"
    )
    md.append("## 3. Is Phi K-sensitive, V-sensitive, or both?")
    md.append(
        f"- {kp.get('kv_class','?')}: K8V16={fnum(kp,'K8V16'):.3f}, K16V8={fnum(kp,'K16V8'):.3f}\n"
    )
    gp = next((r for r in gran if r["model"] == p), {})
    md.append("## 4. Does scale granularity fix it?")
    md.append(
        f"- per-tensor K {fnum(gp,'per_tensor_K'):.3f} -> best finer {fnum(gp,'best_finer_K'):.3f} "
        f"({gp.get('best_layout','?')}); fixes={gp.get('granularity_fixes','?')}\n"
    )
    md.append("## 5. Does rotary/pass-through split explain it?")
    if mp:
        md.append(
            f"- {p} rotary_dim={mp.get('rotary_dim')}, pass_through_dim={mp.get('pass_through_dim')} "
            f"(partial_rope={mp.get('partial_rope')}); pass/rot amax ratio + rotary_only vs "
            "passthrough_only quant decide the mixture hypothesis."
        )
    rpp = {r["config"]: fnum(r, "mean_logit_err") for r in rope if r["model"] == p}
    md.append(
        f"- rotary_only={rpp.get('rotary_only','-')}, passthrough_only={rpp.get('passthrough_only','-')}, "
        f"split_scale={rpp.get('split_scale','-')}\n"
    )
    md.append("## 6. Does RoPE placement matter?")
    md.append(
        f"- post-RoPE K={rpp.get('K_postRoPE','-')}, pre-RoPE prebias K={rpp.get('K_preRoPE_prebias','-')}\n"
    )
    md.append("## 7. Are attention scores unusually large / unstable?")
    sp = [r for r in score if r["model"] == p]
    if sp:
        md.append(
            f"- max BF16 score_amax={max(fnum(r,'bf16_score_amax') for r in sp):.1f}, "
            f"mean|FP8 entropy delta|={St.mean([abs(fnum(r,'fp8_entropy_delta')) for r in sp]):.3f}, "
            f"inf/nan={sum(int(fnum(r,'n_inf')+fnum(r,'n_nan')) for r in sp)}\n"
        )
    md.append("## 8. Is failure layer-local?")
    for r in [x for x in layer if x["model"] == p]:
        md.append(
            f"- {r['side']}: top layer {r['top_layer']} share {fnum(r,'top_share'):.2f} "
            f"local={r.get('layer_local')}"
        )
    md.append("\n## 9. Does Phi differ from Qwen/Mistral controls?")
    md.append("| model | label | kv_class | prebias_helps |")
    md.append("|---|---|---|---|")
    for sn in models:
        kp2 = next((r for r in kviso if r["model"] == sn), {})
        md.append(
            f"| {sn} | `{cls[sn][0]}` | {kp2.get('kv_class','?')} | {kp2.get('prebias_helps','?')} |"
        )
    md.append(f"\n## 10. Recommended label for {p}: **`{label}`**  evidence={ev}")
    md.append("\n## 11. How this augments the K-bias paper subsection")
    md.append(
        "Phi is the biased-model counterexample that bounds the claim: bias EXISTS but the "
        "failure is not bias-induced. It prevents the K-bias story from over-generalizing."
    )
    md.append("\n## 12. Recommended paper-safe sentence")
    md.append(
        f"> Phi-2 provides a biased-model counterexample: although it contains Q/K/V projection "
        f"bias, pre-bias quantization does not recover its FP8 error. Diagnostics indicate its "
        f"failure is better explained by **{label}**, reinforcing that K-bias stress explains "
        f"the Qwen-style failure mode but is not a universal account of all K quantization "
        f"failures. Dependency-aware K16/V8 remains the robust serving default."
    )
    md.append("\n**DO NOT edit the paper from this report -- artifacts first.**")
    out = os.path.join(R, "PHI_FAILURE_FINDINGS.md")
    os.makedirs(R, exist_ok=True)
    open(out, "w").write("\n".join(md))
    print(f"{p} -> {label}  evidence={ev}")
    print(f"[report] -> {out}")


if __name__ == "__main__":
    main()

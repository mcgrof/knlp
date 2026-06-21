"""Reconstruct the verdict CSVs that the per-model-loop CSV-write bug dropped (kv_isolation,
scale_granularity, rope_placement) by parsing phi.log -- the source of truth (its per-model
[marker] lines carry the real numbers). Then report_phi_failure can regenerate a correct artifact
without a GPU re-run. subspace / attention_score / layer_sweep CSVs were written and are untouched.
"""

import argparse
import csv
import os
import re


def reconstruct(R):
    """Rebuild the verdict CSVs that a mid-loop CSV-write crash dropped, from phi.log (the source
    of truth). Idempotent + safe to call from the report so the artifact always matches the log.
    """
    lp = os.path.join(R, "phi.log")
    if not os.path.exists(lp):
        return
    log = open(lp).read()

    def wcsv(sub, name, rows):
        if not rows:
            return
        d = os.path.join(R, sub)
        if os.path.exists(os.path.join(d, name)):
            return  # a real CSV is present -- do not clobber; only fill gaps from the log
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            [w.writerow(r) for r in rows]
        print(f"  wrote {sub}/{name}: {len(rows)} rows")

    # [kviso] MODEL: K8V16=.. K16V8=.. K8V8=.. prebias=.. -> CLASS
    kv = []
    for m in re.finditer(
        r"\[kviso\] (\S+): K8V16=([\d.]+) K16V8=([\d.]+) K8V8=([\d.]+) "
        r"prebias=([\d.]+) -> (\w+)",
        log,
    ):
        mod, k8v16, k16v8, k8v8, preb, cls = m.groups()
        kv.append(
            dict(
                model=mod,
                K8V16=float(k8v16),
                K16V8=float(k16v8),
                K8V8=float(k8v8),
                prebias_K8V8=float(preb),
                kv_class=cls,
                prebias_helps=bool(float(preb) < 0.5 * float(k8v8)),
            )
        )
    wcsv("kv_isolation", "kv_isolation_verdict.csv", kv)

    # [gran] MODEL: per_tensor=.. best_finer=..(layout) fixes=..
    gr = []
    for m in re.finditer(
        r"\[gran\] (\S+): per_tensor=([\d.]+) best_finer=([\d.]+)\((\w+(?::\d+)?)\) "
        r"fixes=(\w+)",
        log,
    ):
        mod, pt, bf, lay, fx = m.groups()
        gr.append(
            dict(
                model=mod,
                per_tensor_K=float(pt),
                best_finer_K=float(bf),
                best_layout=lay,
                granularity_fixes=(fx == "True"),
            )
        )
    wcsv("scale_granularity", "scale_granularity_verdict.csv", gr)

    # [rope] MODEL: postRoPE=.. preRoPE=.. rotary_only=.. passthru_only=..
    rp = []
    for m in re.finditer(
        r"\[rope\] (\S+): postRoPE=([\d.]+) preRoPE=([\d.]+) "
        r"rotary_only=([\d.eE+-]+) passthru_only=([\d.eE+-]+)",
        log,
    ):
        mod, post, pre, ro, po = m.groups()
        for cfg, val in [
            ("K_postRoPE", post),
            ("K_preRoPE_prebias", pre),
            ("rotary_only", ro),
            ("passthrough_only", po),
        ]:
            if val in ("-", ""):  # full-RoPE controls have no pass-through subspace
                continue
            rp.append(dict(model=mod, config=cfg, mean_logit_err=float(val)))
    wcsv("rope_placement", "rope_placement_results.csv", rp)
    print(
        f"[parse] reconstructed {len(kv)} kviso, {len(gr)} gran, "
        f"{len(set(r['model'] for r in rp))} rope models from {R}/phi.log"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    reconstruct(ap.parse_args().root)


if __name__ == "__main__":
    main()

"""Phase 3: K/V isolation -- is Phi's FP8 failure K-side, V-side, both, or neither?"""

import argparse, csv, os
import torch
import _phicommon as pc
from _phicommon import kbc, t2

CELLS = {
    "K16V16": ("bf16", "bf16", False),
    "K16V8": ("bf16", "fp8:per_tensor", False),
    "K8V16": ("fp8:per_tensor", "bf16", False),
    "K8V8": ("fp8:per_tensor", "fp8:per_tensor", False),
    "K16V4": ("bf16", "int4:per_token", False),
    "K4V16": ("int4:per_token", "bf16", False),
    "K4V8": ("int4:per_token", "fp8:per_tensor", False),
    "prebias_K8V8": ("fp8:per_tensor", "fp8:per_tensor", True),
    "prebias_K8V16": ("fp8:per_tensor", "bf16", True),
}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--models-file")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--num-prompts", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()
    specs = t2.load_models_spec(a.models_file, a.models, a.only)
    os.makedirs(a.output_dir, exist_ok=True)
    rows = []
    verdicts = []
    for m in specs:
        mid, sn = m["model_id"], m["short_name"]
        try:
            model, tok = kbc.load_model(mid, a.dtype, a.device, t2.trc_for(mid))
            infos = kbc.discover_attention(model)
            ids = kbc.calib_prompts(tok, n=a.num_prompts, seq_len=a.seq_len)
            base = t2.logits_list(model, ids, a.device)
            r = {}
            for name, (k, v, pb) in CELLS.items():
                h = kbc.FlexKVHarness(
                    model, infos, kbc.parse_spec(k), kbc.parse_spec(v), prebias=pb
                )
                h.install()
                try:
                    mt = t2.metrics(base, t2.logits_list(model, ids, a.device))
                finally:
                    h.remove()
                r[name] = mt["mean_logit_err"]
                rows.append(dict(model=sn, cell=name, **mt))
            kside = r["K8V16"] > 0.30
            vside = r["K16V8"] > 0.30
            verdicts.append(
                dict(
                    model=sn,
                    K8V16=r["K8V16"],
                    K16V8=r["K16V8"],
                    K8V8=r["K8V8"],
                    prebias_K8V8=r["prebias_K8V8"],
                    kv_class=(
                        "K_sensitive"
                        if kside and not vside
                        else (
                            "V_sensitive"
                            if vside and not kside
                            else "both" if kside and vside else "neither"
                        )
                    ),
                    prebias_helps=bool(r["prebias_K8V8"] < 0.5 * max(r["K8V8"], 1e-9)),
                )
            )
            print(
                f"[kviso] {sn}: K8V16={r['K8V16']:.3f} K16V8={r['K16V8']:.3f} K8V8={r['K8V8']:.3f} prebias={r['prebias_K8V8']:.3f} -> {verdicts[-1]['kv_class']}"
            )
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[FAIL] {sn}: {type(e).__name__}: {str(e)[:120]}")
    for fn, data in [("kv_isolation_results.jsonl", rows)]:
        import json

        open(os.path.join(a.output_dir, fn), "w").write(
            "\n".join(json.dumps(x) for x in data)
        )
    if rows:
        with open(
            os.path.join(a.output_dir, "kv_isolation_summary.csv"), "w", newline=""
        ) as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            [w.writerow(x) for x in rows]
    if verdicts:
        with open(
            os.path.join(a.output_dir, "kv_isolation_verdict.csv"), "w", newline=""
        ) as f:
            w = csv.DictWriter(f, fieldnames=list(verdicts[0].keys()))
            w.writeheader()
            [w.writerow(x) for x in verdicts]
    print(f"[kviso] -> {a.output_dir}")


if __name__ == "__main__":
    main()

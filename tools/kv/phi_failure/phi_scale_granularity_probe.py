"""Phase 4: K-FP8 scale granularity (+ rotary/passthrough subspace). Does a finer K scale fix Phi?
V16 first to isolate K, then V8. If per-channel/per-head or rotary/passthrough-split fixes it ->
scale-granularity / mixed-subspace; if nothing finer fixes it but K16 works -> score-scale.
"""

import argparse, csv, os
import torch
import _phicommon as pc
from _phicommon import kbc, t2

LAYOUTS = [
    "per_tensor",
    "per_head",
    "per_channel",
    "per_token",
    "per_group:16",
    "per_group:32",
    "per_group:64",
    "per_group:128",
]


@torch.no_grad()
def err(model, infos, base, ids, dev, kspec, vspec):
    h = kbc.FlexKVHarness(model, infos, kbc.parse_spec(kspec), kbc.parse_spec(vspec))
    h.install()
    try:
        return t2.metrics(base, t2.logits_list(model, ids, dev))["mean_logit_err"]
    finally:
        h.remove()


@torch.no_grad()
def sub_err(model, infos, rd, base, ids, dev, mode, vspec):
    h = pc.SubspaceKHarness(model, infos, rd, mode, v_spec=vspec)
    h.install()
    try:
        return t2.metrics(base, t2.logits_list(model, ids, dev))["mean_logit_err"]
    finally:
        h.remove()


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
            rd, hd = pc.rotary_split(model)
            ids = kbc.calib_prompts(tok, n=a.num_prompts, seq_len=a.seq_len)
            base = t2.logits_list(model, ids, a.device)
            for vlabel, vspec in [("V16", "bf16"), ("V8", "fp8:per_tensor")]:
                for lay in LAYOUTS:
                    e = err(model, infos, base, ids, a.device, f"fp8:{lay}", vspec)
                    rows.append(
                        dict(model=sn, vside=vlabel, k_layout=lay, mean_logit_err=e)
                    )
                if rd < hd:
                    for mode in ("split_scale", "rotary_only", "passthrough_only"):
                        e = sub_err(model, infos, rd, base, ids, a.device, mode, vspec)
                        rows.append(
                            dict(
                                model=sn, vside=vlabel, k_layout=mode, mean_logit_err=e
                            )
                        )
            v16 = [r for r in rows if r["model"] == sn and r["vside"] == "V16"]
            per_tensor = next(
                r["mean_logit_err"] for r in v16 if r["k_layout"] == "per_tensor"
            )
            best_finer = min(
                r["mean_logit_err"] for r in v16 if r["k_layout"] != "per_tensor"
            )
            best_lay = min(
                (r for r in v16 if r["k_layout"] != "per_tensor"),
                key=lambda r: r["mean_logit_err"],
            )["k_layout"]
            verdicts.append(
                dict(
                    model=sn,
                    per_tensor_K=per_tensor,
                    best_finer_K=best_finer,
                    best_layout=best_lay,
                    granularity_fixes=bool(
                        best_finer < 0.34 * max(per_tensor, 1e-9) and per_tensor > 0.30
                    ),
                )
            )
            print(
                f"[gran] {sn}: per_tensor={per_tensor:.3f} best_finer={best_finer:.3f}({best_lay}) fixes={verdicts[-1]['granularity_fixes']}"
            )
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[FAIL] {sn}: {type(e).__name__}: {str(e)[:120]}")
    if rows:
        with open(
            os.path.join(a.output_dir, "scale_granularity_results.csv"), "w", newline=""
        ) as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            [w.writerow(x) for x in rows]
    if verdicts:
        with open(
            os.path.join(a.output_dir, "scale_granularity_verdict.csv"), "w", newline=""
        ) as f:
            w = csv.DictWriter(f, fieldnames=list(verdicts[0].keys()))
            w.writeheader()
            [w.writerow(x) for x in verdicts]
    print(f"[gran] -> {a.output_dir}")


if __name__ == "__main__":
    main()

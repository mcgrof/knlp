"""Phase 7: layer-local sweep -- quantize one layer's K (then V) at a time; find dominant layers."""

import argparse, csv, os
import torch
import _phicommon as pc
from _phicommon import kbc, t2


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--models-file")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--num-prompts", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()
    specs = t2.load_models_spec(a.models_file, a.models, a.only)
    os.makedirs(a.output_dir, exist_ok=True)
    rows = []
    verd = []
    for m in specs:
        mid, sn = m["model_id"], m["short_name"]
        try:
            model, tok = kbc.load_model(mid, a.dtype, a.device, t2.trc_for(mid))
            infos = kbc.discover_attention(model)
            ids = kbc.calib_prompts(tok, n=a.num_prompts, seq_len=a.seq_len)
            base = t2.logits_list(model, ids, a.device)
            nL = len(infos)
            for side, (k, v) in {
                "K": ("fp8:per_tensor", "bf16"),
                "V": ("bf16", "fp8:per_tensor"),
            }.items():
                per = []
                for L in range(nL):
                    h = kbc.FlexKVHarness(
                        model, infos, kbc.parse_spec(k), kbc.parse_spec(v), layers={L}
                    )
                    h.install()
                    try:
                        e = t2.metrics(base, t2.logits_list(model, ids, a.device))[
                            "mean_logit_err"
                        ]
                    finally:
                        h.remove()
                    rows.append(dict(model=sn, side=side, layer=L, only_err=e))
                    per.append((L, e))
                per.sort(key=lambda x: -x[1])
                tot = sum(e for _, e in per)
                verd.append(
                    dict(
                        model=sn,
                        side=side,
                        top_layer=per[0][0],
                        top_err=per[0][1],
                        top_share=per[0][1] / max(tot, 1e-9),
                        layer_local=bool(per[0][1] / max(tot, 1e-9) > 0.4),
                    )
                )
                print(
                    f"[layer] {sn} {side}: top L{per[0][0]} err={per[0][1]:.3f} share={verd[-1]['top_share']:.2f} local={verd[-1]['layer_local']}"
                )
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[FAIL] {sn}: {type(e).__name__}: {str(e)[:120]}")
    if rows:
        with open(
            os.path.join(a.output_dir, "layer_sweep_results.csv"), "w", newline=""
        ) as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            [w.writerow(x) for x in rows]
    if verd:
        with open(
            os.path.join(a.output_dir, "layer_sweep_verdict.csv"), "w", newline=""
        ) as f:
            w = csv.DictWriter(f, fieldnames=list(verd[0].keys()))
            w.writeheader()
            [w.writerow(x) for x in verd]
    print(f"[layer] -> {a.output_dir}")


if __name__ == "__main__":
    main()

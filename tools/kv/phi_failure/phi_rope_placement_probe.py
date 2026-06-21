"""Phase 5: RoPE placement -- pre vs post-RoPE K quant, and rotary-only vs passthrough-only."""

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
    ap.add_argument("--num-prompts", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()
    specs = t2.load_models_spec(a.models_file, a.models, a.only)
    os.makedirs(a.output_dir, exist_ok=True)
    rows = []
    for m in specs:
        mid, sn = m["model_id"], m["short_name"]
        try:
            model, tok = kbc.load_model(mid, a.dtype, a.device, t2.trc_for(mid))
            infos = kbc.discover_attention(model)
            rd, hd = pc.rotary_split(model)
            ids = kbc.calib_prompts(tok, n=a.num_prompts, seq_len=a.seq_len)
            base = t2.logits_list(model, ids, a.device)
            # post-RoPE K (normal) and pre-RoPE residual K (prebias) -- V16 to isolate K
            for name, (kspec, pb) in {
                "K_postRoPE": ("fp8:per_tensor", False),
                "K_preRoPE_prebias": ("fp8:per_tensor", True),
            }.items():
                h = kbc.FlexKVHarness(
                    model,
                    infos,
                    kbc.parse_spec(kspec),
                    kbc.parse_spec("bf16"),
                    prebias=pb,
                )
                h.install()
                try:
                    e = t2.metrics(base, t2.logits_list(model, ids, a.device))[
                        "mean_logit_err"
                    ]
                finally:
                    h.remove()
                rows.append(dict(model=sn, config=name, mean_logit_err=e))
            if rd < hd:
                for mode in ("split_scale", "rotary_only", "passthrough_only"):
                    h = pc.SubspaceKHarness(model, infos, rd, mode, v_spec="bf16")
                    h.install()
                    try:
                        e = t2.metrics(base, t2.logits_list(model, ids, a.device))[
                            "mean_logit_err"
                        ]
                    finally:
                        h.remove()
                    rows.append(dict(model=sn, config=mode, mean_logit_err=e))
            d = {r["config"]: r["mean_logit_err"] for r in rows if r["model"] == sn}
            print(
                f"[rope] {sn}: postRoPE={d.get('K_postRoPE',0):.3f} preRoPE={d.get('K_preRoPE_prebias',0):.3f} "
                f"rotary_only={d.get('rotary_only','-')} passthru_only={d.get('passthrough_only','-')}"
            )
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[FAIL] {sn}: {type(e).__name__}: {str(e)[:120]}")
    if rows:
        with open(
            os.path.join(a.output_dir, "rope_placement_results.csv"), "w", newline=""
        ) as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            [w.writerow(x) for x in rows]
    print(f"[rope] -> {a.output_dir}")


if __name__ == "__main__":
    main()

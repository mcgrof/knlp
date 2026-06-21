"""Phase 3: long-context check. Does the bias-stress story hold at the contexts where Qwen FP8 K
failure worsens? Runs bf16 / normal-FP8 / pre-bias / K16-V8 at several seq lens, reports logit err
+ stress per length. measurement_level=fake_quant.
"""

import argparse
import csv
import os

import torch

import _t2common as t2
from _t2common import kbc

CELLS = {
    "normal_fp8": ("fp8:per_tensor", "fp8:per_tensor", False),
    "prebias_fp8": ("fp8:per_tensor", "fp8:per_tensor", True),
    "k16v8": ("bf16", "fp8:per_tensor", False),
}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", required=True)
    ap.add_argument("--models-file")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--seq-lens", default="2048,8192,16384")
    ap.add_argument("--num-prompts", type=int, default=16)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    specs = t2.load_models_spec(args.models_file, args.models, args.only)
    seq_lens = [int(s) for s in args.seq_lens.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for m in specs:
        mid, sn = m["model_id"], m["short_name"]
        try:
            model, tok = kbc.load_model(mid, args.dtype, args.device, t2.trc_for(mid))
            infos = kbc.discover_attention(model)
            for sl in seq_lens:
                try:
                    ids = kbc.calib_prompts(tok, n=args.num_prompts, seq_len=sl)
                    base = t2.logits_list(model, ids, args.device)
                    row = dict(model=sn, seq_len=sl, n_prompts=len(ids))
                    for name, (k, v, pb) in CELLS.items():
                        h = kbc.FlexKVHarness(
                            model,
                            infos,
                            kbc.parse_spec(k),
                            kbc.parse_spec(v),
                            prebias=pb,
                        )
                        h.install()
                        try:
                            row[f"{name}_err"] = t2.metrics(
                                base, t2.logits_list(model, ids, args.device)
                            )["mean_logit_err"]
                        finally:
                            h.remove()
                    rows.append(row)
                    print(
                        f"[longctx] {sn} T={sl}: normal={row['normal_fp8_err']:.3f} "
                        f"prebias={row['prebias_fp8_err']:.3f} k16v8={row['k16v8_err']:.3f}"
                    )
                    torch.cuda.empty_cache()
                except torch.cuda.OutOfMemoryError:
                    print(
                        f"[longctx] {sn} T={sl}: OOM (W7900) -> skip (tier-2 large lane)"
                    )
                    torch.cuda.empty_cache()
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[FAIL] {sn}: {type(e).__name__}: {str(e)[:140]}")
    if rows:
        with open(
            os.path.join(args.output_dir, "long_context_summary.csv"), "a", newline=""
        ) as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if f.tell() == 0:
                w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"[longctx] -> {args.output_dir}")


if __name__ == "__main__":
    main()

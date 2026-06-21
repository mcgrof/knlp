"""Phase 4 (near-serving): real next-token PPL with a fake-quantized KV cache.

Reviewer-proofing for "fake-quant probes are an artifact": this measures REAL WikiText PPL (NLL on
the dataset's own next-token targets, not a vs-BF16 logit diff) with the KV cache quantized inside
the forward, across BF16 / normal-FP8 / pre-bias / K16-V8. measurement_level=hf_dynamic_cache_fake_quant
-- a close-to-serving HF path, NOT full vLLM serving; no throughput claims.
"""

import argparse
import csv
import os

import torch

import _t2common as t2
from _t2common import kbc

CELLS = {
    "bf16": ("bf16", "bf16", False),
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
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--num-prompts", type=int, default=24)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    specs = t2.load_models_spec(args.models_file, args.models, args.only)
    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for m in specs:
        mid, sn = m["model_id"], m["short_name"]
        try:
            model, tok = kbc.load_model(mid, args.dtype, args.device, t2.trc_for(mid))
            infos = kbc.discover_attention(model)
            row = dict(
                model=sn,
                seq_len=args.seq_len,
                measurement_level="hf_dynamic_cache_fake_quant",
            )
            for name, (k, v, pb) in CELLS.items():
                har = (
                    None
                    if name == "bf16"
                    else kbc.FlexKVHarness(
                        model, infos, kbc.parse_spec(k), kbc.parse_spec(v), prebias=pb
                    )
                )
                ppl, ntok = kbc.real_ppl(
                    model,
                    tok,
                    args.device,
                    n=args.num_prompts,
                    seq_len=args.seq_len,
                    harness=har,
                )
                row[f"{name}_ppl"] = ppl
            rows.append(row)
            print(
                f"[serving] {sn}: bf16={row['bf16_ppl']:.3f} normal_fp8={row['normal_fp8_ppl']:.3f} "
                f"prebias={row['prebias_fp8_ppl']:.3f} k16v8={row['k16v8_ppl']:.3f}"
            )
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[FAIL] {sn}: {type(e).__name__}: {str(e)[:140]}")
    if rows:
        cp = os.path.join(args.output_dir, "serving_prebias_summary.csv")
        with open(cp, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if f.tell() == 0:
                w.writeheader()
            for r in rows:
                w.writerow(r)
        md = [
            "# Near-serving pre-bias FP8 validation (real WikiText PPL)",
            "measurement_level = hf_dynamic_cache_fake_quant (NOT full vLLM serving; no throughput claims)",
            "",
            "| model | BF16 PPL | normal-FP8 | pre-bias FP8 | K16/V8 |",
            "|---|---|---|---|---|",
        ]
        for r in rows:
            md.append(
                f"| {r['model']} | {r['bf16_ppl']:.3f} | {r['normal_fp8_ppl']:.3f} | "
                f"{r['prebias_fp8_ppl']:.3f} | {r['k16v8_ppl']:.3f} |"
            )
        open(os.path.join(args.output_dir, "serving_prebias_table.md"), "w").write(
            "\n".join(md)
        )
    print(f"[serving] -> {args.output_dir}")


if __name__ == "__main__":
    main()

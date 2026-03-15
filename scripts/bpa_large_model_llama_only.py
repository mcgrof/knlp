#!/usr/bin/env python3
# BPA Large Model Validation: Llama-3.1-70B-Instruct only
#
# Reuses the RunPod harness from bpa_large_model_gemma_llama.py
# but targets only meta-llama/Llama-3.1-70B-Instruct (v4 plan).
#
# Usage:
#     source ~/.enhance-bash && source ~/envs/runpod/bin/activate
#     python3 scripts/bpa_large_model_llama_only.py

import json
import time

from bpa_large_model_gemma_llama import JSON_DIR, run_model_on_pod


def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    models = [
        {
            "name": "meta-llama/Llama-3.1-70B-Instruct",
            "dtype": "bfloat16",
            "gpu": "NVIDIA H200",
            "tag": f"llama_3_1_70b_instruct_{ts}",
        },
    ]

    all_results = {}
    for m in models:
        print(f"\n{'#'*70}")
        print(f"# MODEL: {m['name']}")
        print(f"# GPU: {m['gpu']}")
        print(f"{'#'*70}")

        result = run_model_on_pod(m["name"], m["dtype"], m["gpu"], m["tag"])
        all_results[m["name"]] = result

        if "error" in result:
            print(f"\n*** {m['name']} FAILED: {result['error']}")
        else:
            info = result["model_info"]
            print(
                f"\n*** {m['name']} COMPLETE: "
                f"{info['n_params']/1e9:.1f}B, D={info['n_layers']}"
            )

    # Save combined
    combined_path = f"{JSON_DIR}/large_model_llama_only.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nCombined results: {combined_path}")

    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    for mname, res in all_results.items():
        short = mname.split("/")[-1]
        if "error" in res:
            print(f"\n{short}: FAILED - {res['error'][:100]}")
            continue
        info = res["model_info"]
        print(f"\n{short} ({info['n_params']/1e9:.1f}B, D={info['n_layers']}):")
        for r in res.get("kv_asymmetry", []):
            print(
                f"  {r['label']:20s} err={r['avg_logit_error']:.4f} "
                f"agree={r['avg_token_agreement']:.4f} "
                f"ppl_d={r['avg_ppl_delta_pct']:+.2f}%"
            )
        rc = res.get("ratio_classifier", {})
        if rc:
            print(
                f"  Ratio: {rc['ratio_INT6_INT8']:.2f} "
                f"(needs_fp16={rc['needs_fp16_keys']})"
            )

    print("\nALL DONE")


if __name__ == "__main__":
    main()

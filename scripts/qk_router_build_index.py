#!/usr/bin/env python3
"""Phase 4: Build K-summary index for all summary modes."""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.qk_router.k_summary import build_k_summaries
from lib.qk_router.utils import save_json, Timer


def main():
    run_root = os.environ.get("RUN_ROOT", "/mnt/tmpfs/knlp/results/qk_router_01")
    cache_dir = os.environ.get("HF_CACHE", "/mnt/SFS-hugging/hub")
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B")

    prefix_length = int(os.environ.get("PREFIX_LENGTH", "4096"))
    block_size = int(os.environ.get("BLOCK_SIZE", "128"))
    seed = int(os.environ.get("SEED", "42"))

    modes = [
        "direct_centroid",
        "random_summary",
        "first_k_real",
        "sampled_real_geometry",
    ]

    print("=" * 60)
    print("QK Router Phase 4: Build K-Summary Index")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    # Load representative input (first request from trace collection)
    from scripts.qk_router_collect_traces import load_workload_texts

    inputs = load_workload_texts(
        tokenizer, 1, prefix_length, 64, cache_dir=cache_dir, seed=seed
    )
    input_ids = inputs[0]

    index_dir = os.path.join(run_root, "k_summaries")
    os.makedirs(index_dir, exist_ok=True)

    summary_info = {}
    for mode in modes:
        print(f"\nBuilding K-summary: {mode}...")
        with Timer(mode):
            summaries = build_k_summaries(
                model,
                input_ids,
                prefix_length,
                block_size,
                mode=mode,
                device="cuda",
                seed=seed,
            )

        # Save
        out_path = os.path.join(index_dir, f"{mode}.npy")
        np.save(out_path, summaries)

        summary_info[mode] = {
            "shape": list(summaries.shape),
            "path": out_path,
            "mean_norm": float(np.linalg.norm(summaries, axis=-1).mean()),
        }
        print(
            f"  Shape: {summaries.shape}, "
            f"mean norm: {summary_info[mode]['mean_norm']:.4f}"
        )

        torch.cuda.empty_cache()

    save_json(summary_info, os.path.join(run_root, "k_summary_info.json"))
    print(f"\nPhase 4 complete. Summaries saved to {index_dir}")


if __name__ == "__main__":
    main()

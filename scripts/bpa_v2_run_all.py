#!/usr/bin/env python
"""
BPA v2: Run all phases end-to-end.

Phase 0: Collect oracle labels + local features
Phase 1: Train learned gate
Phase 2: Speculative local-first pipeline evaluation
Phase 3: Coarse gating evaluation
Phase 4: End-to-end frontier evaluation

Usage:
    python scripts/bpa_v2_run_all.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    t0 = time.time()

    print("=" * 70)
    print("BPA v2: Full Pipeline Run")
    print("=" * 70)

    data_dir = "bpa_v2_gate_dataset"
    seeds = [1, 2, 3]

    # Phase 0: Collect data
    print("\n" + "=" * 70)
    print("PHASE 0: Data Collection")
    print("=" * 70)
    from scripts.bpa_v2_collect import run_collection

    if not os.path.exists(os.path.join(data_dir, "manifest.json")):
        run_collection(
            n_samples=200,
            batch_size=4,
            seq_len=256,
            n_positions=8,
            local_window=64,
            seed=1,
            output_dir=data_dir,
            shard_size=100000,
        )
    else:
        print(f"Dataset already exists in {data_dir}, skipping.")

    # Phase 1: Train gate
    print("\n" + "=" * 70)
    print("PHASE 1: Train Learned Gate")
    print("=" * 70)
    from scripts.bpa_v2_train_gate import run_multi_seed

    gate_results = run_multi_seed(
        data_dir=data_dir,
        seeds=seeds,
        hidden=128,
        n_layers=2,
        epochs=20,
        lr=1e-3,
        output_dir="bpa_v2_gate_results",
    )

    # Phase 2: Speculative pipeline
    print("\n" + "=" * 70)
    print("PHASE 2: Speculative Local-First Pipeline")
    print("=" * 70)
    from scripts.bpa_v2_speculative import run_evaluation as run_speculative

    run_speculative(
        n_eval=30,
        batch_size=2,
        seq_len=256,
        local_window=64,
        gate_threshold=0.5,
        seed=1,
        data_dir=data_dir,
        gate_results_dir="bpa_v2_gate_results",
        output_dir="bpa_v2_speculative_results",
    )

    # Phase 3: Coarse gating
    print("\n" + "=" * 70)
    print("PHASE 3: Coarse Gating")
    print("=" * 70)
    from scripts.bpa_v2_coarse_gating import run_evaluation as run_coarse

    run_coarse(
        data_dir=data_dir,
        seeds=seeds,
        output_dir="bpa_v2_coarse_results",
        seq_len=256,
        local_window=64,
    )

    # Phase 4: Frontier evaluation
    print("\n" + "=" * 70)
    print("PHASE 4: End-to-End Frontier")
    print("=" * 70)
    from scripts.bpa_v2_frontier import run_frontier

    run_frontier(
        seeds=seeds,
        seq_lens=[256],
        n_eval=30,
        batch_size=2,
        local_window=64,
        data_dir=data_dir,
        output_dir="bpa_v2_frontier_results",
    )

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"ALL PHASES COMPLETE ({elapsed:.1f}s)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compare two training runs: magnitude pruning vs bitter7 state pruning.
Visualizes validation perplexity and sparsity progression over time.

This script can be run repeatedly to monitor ongoing training runs.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

def parse_log_file(log_path):
    """Extract eval metrics and iteration stats from training log."""
    evals = []
    iters = []

    with open(log_path) as f:
        for line in f:
            # Parse eval lines: "Eval @ iter 500: train 5.5020, val 5.4741, ppl 238.44"
            eval_match = re.match(r'Eval @ iter (\d+): train ([\d.]+), val ([\d.]+), ppl ([\d.]+)', line)
            if eval_match:
                iteration = int(eval_match.group(1))
                train_loss = float(eval_match.group(2))
                val_loss = float(eval_match.group(3))
                val_ppl = float(eval_match.group(4))
                evals.append({
                    'iter': iteration,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_ppl': val_ppl
                })

            # Parse iter lines: "Iter   500 | loss 5.5020 | ppl  238.44 | lr 1.50e-04 | sparsity 0.0% | 1200.0ms/iter"
            iter_match = re.match(r'Iter\s+(\d+) \| loss ([\d.]+) \| ppl\s+([\d.]+) \| lr ([\de.-]+) \| sparsity ([\d.]+)% \|', line)
            if iter_match:
                iteration = int(iter_match.group(1))
                loss = float(iter_match.group(2))
                ppl = float(iter_match.group(3))
                lr = float(iter_match.group(4))
                sparsity = float(iter_match.group(5))
                iters.append({
                    'iter': iteration,
                    'loss': loss,
                    'ppl': ppl,
                    'lr': lr,
                    'sparsity': sparsity
                })

    return evals, iters

def main():
    # Parse both runs
    mag_log = Path("test_matrix_results_20251114_170753/gpt2_adamwspam_magnitude_50/output.log")
    bitter7_log = Path("test_matrix_results_20251115_020707/gpt2_adamwprune_bitter7_state_50/output.log")

    print("="*80)
    print(f"TRAINING COMPARISON REPORT - Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    print("\nParsing magnitude pruning run (COMPLETED)...")
    mag_evals, mag_iters = parse_log_file(mag_log)

    print("Parsing bitter7 state pruning run (IN PROGRESS)...")
    bitter7_evals, bitter7_iters = parse_log_file(bitter7_log)

    # Check if bitter7 is still running
    b7_running = bitter7_iters[-1]['iter'] < 10000
    if b7_running:
        print(f"  → Bitter7 currently at iteration {bitter7_iters[-1]['iter']}/10000 ({bitter7_iters[-1]['iter']/100:.1f}%)")
        print(f"  → Current sparsity: {bitter7_iters[-1]['sparsity']:.1f}%")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    status_text = " [IN PROGRESS]" if b7_running else ""
    fig.suptitle(f'Training Comparison: Magnitude Pruning (50%) vs Bitter7 State Pruning (50%){status_text}',
                 fontsize=14, fontweight='bold')

    # Plot 1: Validation Perplexity over time
    ax1 = axes[0, 0]
    mag_eval_iters = [e['iter'] for e in mag_evals]
    mag_val_ppl = [e['val_ppl'] for e in mag_evals]
    bitter7_eval_iters = [e['iter'] for e in bitter7_evals]
    bitter7_val_ppl = [e['val_ppl'] for e in bitter7_evals]

    ax1.plot(mag_eval_iters, mag_val_ppl, 'o-', label='Magnitude (AdamWSPAM)', linewidth=2, markersize=6)
    ax1.plot(bitter7_eval_iters, bitter7_val_ppl, 's-', label='Bitter7 (AdamWPrune)', linewidth=2, markersize=6)
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Validation Perplexity', fontsize=11)
    ax1.set_title('Validation Perplexity (lower is better)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: Validation Loss over time
    ax2 = axes[0, 1]
    mag_val_loss = [e['val_loss'] for e in mag_evals]
    bitter7_val_loss = [e['val_loss'] for e in bitter7_evals]

    ax2.plot(mag_eval_iters, mag_val_loss, 'o-', label='Magnitude (AdamWSPAM)', linewidth=2, markersize=6)
    ax2.plot(bitter7_eval_iters, bitter7_val_loss, 's-', label='Bitter7 (AdamWPrune)', linewidth=2, markersize=6)
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Validation Loss', fontsize=11)
    ax2.set_title('Validation Loss (lower is better)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Sparsity progression
    ax3 = axes[1, 0]
    mag_iter_nums = [it['iter'] for it in mag_iters]
    mag_sparsity = [it['sparsity'] for it in mag_iters]
    bitter7_iter_nums = [it['iter'] for it in bitter7_iters]
    bitter7_sparsity = [it['sparsity'] for it in bitter7_iters]

    # Sample every 10th point to avoid overcrowding
    ax3.plot(mag_iter_nums[::10], mag_sparsity[::10], '-', label='Magnitude (AdamWSPAM)',
             linewidth=1.5, alpha=0.8)
    ax3.plot(bitter7_iter_nums[::10], bitter7_sparsity[::10], '-', label='Bitter7 (AdamWPrune)',
             linewidth=1.5, alpha=0.8)
    ax3.set_xlabel('Iteration', fontsize=11)
    ax3.set_ylabel('Sparsity (%)', fontsize=11)
    ax3.set_title('Sparsity Progression', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-1, 55)

    # Plot 4: Training efficiency (val ppl vs sparsity)
    ax4 = axes[1, 1]

    # Get sparsity at each eval point
    mag_eval_sparsity = []
    for eval_point in mag_evals:
        iter_num = eval_point['iter']
        # Find closest iteration record
        closest = min(mag_iters, key=lambda x: abs(x['iter'] - iter_num))
        mag_eval_sparsity.append(closest['sparsity'])

    bitter7_eval_sparsity = []
    for eval_point in bitter7_evals:
        iter_num = eval_point['iter']
        closest = min(bitter7_iters, key=lambda x: abs(x['iter'] - iter_num))
        bitter7_eval_sparsity.append(closest['sparsity'])

    ax4.plot(mag_eval_sparsity, mag_val_ppl, 'o-', label='Magnitude (AdamWSPAM)',
             linewidth=2, markersize=6)
    ax4.plot(bitter7_eval_sparsity, bitter7_val_ppl, 's-', label='Bitter7 (AdamWPrune)',
             linewidth=2, markersize=6)
    ax4.set_xlabel('Sparsity (%)', fontsize=11)
    ax4.set_ylabel('Validation Perplexity', fontsize=11)
    ax4.set_title('Validation Perplexity vs Sparsity', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')

    plt.tight_layout()
    plt.savefig('training_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to training_comparison.png")

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    print("\nMagnitude Pruning (AdamWSPAM):")
    print(f"  Total iterations: {mag_iters[-1]['iter']}")
    print(f"  Final sparsity: {mag_iters[-1]['sparsity']:.1f}%")
    print(f"  Final val perplexity: {mag_evals[-1]['val_ppl']:.2f}")
    print(f"  Final val loss: {mag_evals[-1]['val_loss']:.4f}")
    print(f"  Best val perplexity: {min(e['val_ppl'] for e in mag_evals):.2f} @ iter {min(mag_evals, key=lambda x: x['val_ppl'])['iter']}")

    print("\nBitter7 State Pruning (AdamWPrune):")
    print(f"  Total iterations: {bitter7_iters[-1]['iter']}")
    print(f"  Final sparsity: {bitter7_iters[-1]['sparsity']:.1f}%")
    print(f"  Final val perplexity: {bitter7_evals[-1]['val_ppl']:.2f}")
    print(f"  Final val loss: {bitter7_evals[-1]['val_loss']:.4f}")
    print(f"  Best val perplexity: {min(e['val_ppl'] for e in bitter7_evals):.2f} @ iter {min(bitter7_evals, key=lambda x: x['val_ppl'])['iter']}")

    # Compare at common iteration points
    print("\n" + "="*80)
    print("COMPARISON AT EVALUATION POINTS")
    print("="*80)
    print(f"{'Iter':<8} {'Mag Val PPL':<15} {'B7 Val PPL':<15} {'Mag Sparsity':<15} {'B7 Sparsity':<15} {'PPL Diff':<10}")
    print("-"*80)

    # Find common eval points
    mag_eval_dict = {e['iter']: e for e in mag_evals}
    bitter7_eval_dict = {e['iter']: e for e in bitter7_evals}
    common_iters = sorted(set(mag_eval_dict.keys()) & set(bitter7_eval_dict.keys()))

    for iter_num in common_iters:
        mag_e = mag_eval_dict[iter_num]
        b7_e = bitter7_eval_dict[iter_num]

        # Get sparsity at this point
        mag_sp = min(mag_iters, key=lambda x: abs(x['iter'] - iter_num))['sparsity']
        b7_sp = min(bitter7_iters, key=lambda x: abs(x['iter'] - iter_num))['sparsity']

        ppl_diff = mag_e['val_ppl'] - b7_e['val_ppl']

        print(f"{iter_num:<8} {mag_e['val_ppl']:<15.2f} {b7_e['val_ppl']:<15.2f} "
              f"{mag_sp:<15.1f} {b7_sp:<15.1f} {ppl_diff:+.2f}")

    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    # Calculate final comparison
    if len(common_iters) > 0:
        final_common = common_iters[-1]
        mag_final = mag_eval_dict[final_common]
        b7_final = bitter7_eval_dict[final_common]
        ppl_improvement = ((mag_final['val_ppl'] - b7_final['val_ppl']) / mag_final['val_ppl']) * 100

        print(f"\n1. At iteration {final_common} (latest common checkpoint):")
        print(f"   Magnitude: {mag_final['val_ppl']:.2f} perplexity, {mag_final['val_loss']:.4f} loss")
        print(f"   Bitter7:   {b7_final['val_ppl']:.2f} perplexity, {b7_final['val_loss']:.4f} loss")
        if ppl_improvement > 0:
            print(f"   → Bitter7 is {ppl_improvement:.1f}% BETTER (lower perplexity)")
        else:
            print(f"   → Magnitude is {-ppl_improvement:.1f}% BETTER (lower perplexity)")

    # Sparsity progression comparison
    mag_50_iter = next((it['iter'] for it in mag_iters if it['sparsity'] >= 49.5), None)
    b7_target_sparsity = bitter7_iters[-1]['sparsity']
    b7_50_iter = next((it['iter'] for it in bitter7_iters if it['sparsity'] >= 49.5), None)

    print(f"\n2. Sparsity Progression:")
    print(f"   Magnitude reached 50% sparsity at iteration {mag_50_iter if mag_50_iter else 'N/A'}")
    print(f"   Bitter7 current sparsity: {b7_target_sparsity:.1f}% at iteration {bitter7_iters[-1]['iter']}")
    if b7_50_iter:
        print(f"   Bitter7 reached 50% sparsity at iteration {b7_50_iter}")
        print(f"   → Sparsity ramp comparison: Bitter7 took {b7_50_iter - mag_50_iter:+d} iterations")
    elif b7_target_sparsity < 50:
        print(f"   → Bitter7 still ramping up sparsity (only {b7_target_sparsity:.1f}% so far)")
        remaining = 50 - b7_target_sparsity
        if bitter7_iters[-1]['iter'] > 100:
            # Estimate sparsity rate
            rate = b7_target_sparsity / bitter7_iters[-1]['iter']
            est_iters = remaining / rate if rate > 0 else float('inf')
            print(f"   → Estimated {int(est_iters)} more iterations to reach 50%")

    print(f"\n3. Training Progress:")
    print(f"   Magnitude: {mag_iters[-1]['iter']:,} iterations (COMPLETED)")
    print(f"   Bitter7:   {bitter7_iters[-1]['iter']:,} iterations", end="")
    if b7_running:
        print(f" ({bitter7_iters[-1]['iter']/100:.1f}% of target 10k)")
    else:
        print(" (COMPLETED)")

    if b7_running:
        print(f"\n4. Current Status:")
        print(f"   ⏳ Bitter7 is still training - run this script again for updated comparison")
        print(f"   📊 Latest bitter7: iter {bitter7_iters[-1]['iter']}, loss {bitter7_iters[-1]['loss']:.4f}, ppl {bitter7_iters[-1]['ppl']:.2f}")
    else:
        # Fair comparison only when both complete
        if b7_50_iter and mag_50_iter:
            # Find eval closest to when both hit 50% sparsity
            mag_eval_at_50 = min(mag_evals, key=lambda x: abs(x['iter'] - mag_50_iter))
            b7_eval_at_50 = min(bitter7_evals, key=lambda x: abs(x['iter'] - b7_50_iter))

            print(f"\n4. Fair Comparison (both at 50% sparsity):")
            print(f"   Magnitude @ iter {mag_eval_at_50['iter']}: ppl {mag_eval_at_50['val_ppl']:.2f}")
            print(f"   Bitter7   @ iter {b7_eval_at_50['iter']}: ppl {b7_eval_at_50['val_ppl']:.2f}")

            winner_ppl = ((mag_eval_at_50['val_ppl'] - b7_eval_at_50['val_ppl']) / mag_eval_at_50['val_ppl']) * 100
            if winner_ppl > 0:
                print(f"   → Bitter7 WINS by {winner_ppl:.1f}% at 50% sparsity")
            else:
                print(f"   → Magnitude WINS by {-winner_ppl:.1f}% at 50% sparsity")

if __name__ == "__main__":
    main()

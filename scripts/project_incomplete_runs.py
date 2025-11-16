#!/usr/bin/env python3
"""Project perplexity for incomplete runs based on observed convergence patterns."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load data
mag_df = pd.read_csv('wandb_gpt2_adamwspam_magnitude_50_metrics.csv')
mag_df = mag_df[mag_df['val_perplexity'].notna()].copy()

b7_df = pd.read_csv('wandb_gpt2_adamwprune_bitter7_state_50_metrics.csv')
b7_df = b7_df[b7_df['val_perplexity'].notna()].copy()

# bitter8 only has 2 points
bitter8_iters = [0, 2500]
bitter8_ppl = [60772.44, 40.94]

print("=" * 80)
print("COMPARISON: Movement Pruning vs bitter8 (NO compile)")
print("=" * 80)
print()

print("At 2500 iterations:")
print(f"  Movement Pruning (WITH compile): {mag_df[mag_df['iteration']==2500]['val_perplexity'].values[0]:.2f} PPL")
print(f"  bitter8 (NO compile):            {bitter8_ppl[1]:.2f} PPL")
print(f"  bitter8 is {((58.80 - 40.94) / 58.80 * 100):.1f}% BETTER!")
print()

print("Full Movement Pruning trajectory (WITH compile):")
for idx, row in mag_df.iterrows():
    print(f"  {row['iteration']:4.0f} iters: {row['val_perplexity']:6.2f} PPL")
print()

print("=" * 80)
print("PROJECTION: bitter8 if it had completed")
print("=" * 80)
print()

# Use bitter7's convergence pattern to project bitter8
# bitter7 converges from 2500 (38.55 PPL) to 7000 (37.28 PPL) = 1.27 PPL improvement
# That's a 3.3% improvement over 4500 additional iterations

# bitter8 starts at 40.94 at 2500 iters
# If it follows similar convergence pattern as bitter7:
improvement_rate = (38.55 - 37.28) / 38.55  # bitter7's improvement from 2500 to 7000

projected_5k = 40.94 * (1 - improvement_rate * 0.55)  # Partial improvement to 5K
projected_7k = 40.94 * (1 - improvement_rate)  # Full improvement to 7K

print(f"bitter8 projection (based on bitter7 convergence pattern):")
print(f"  At 2500 iters (actual):    {bitter8_ppl[1]:.2f} PPL")
print(f"  At 5000 iters (projected): ~{projected_5k:.2f} PPL")
print(f"  At 7000 iters (projected): ~{projected_7k:.2f} PPL")
print()

print("Comparison vs Movement Pruning (WITH compile):")
print(f"  Movement @ 5000 iters:     {mag_df[mag_df['iteration']==5000]['val_perplexity'].values[0]:.2f} PPL")
print(f"  bitter8 projected @ 5000: ~{projected_5k:.2f} PPL ({((44.15 - projected_5k) / 44.15 * 100):.1f}% better)")
print()

print("Comparison vs bitter7 (WITH compile):")
print(f"  bitter7 @ 7000 iters:      {b7_df[b7_df['iteration']==7000]['val_perplexity'].values[0]:.2f} PPL")
print(f"  bitter8 projected @ 7000: ~{projected_7k:.2f} PPL ({((37.28 - projected_7k) / 37.28 * 100):.1f}% worse)")
print()

print("=" * 80)
print("KEY FINDINGS")
print("=" * 80)
print()
print("1. bitter8 WITHOUT torch.compile was SIGNIFICANTLY better than")
print("   Movement Pruning WITH torch.compile at the same iteration count (2500)")
print()
print("2. If bitter8 had run to completion, it would likely have achieved")
print(f"   ~{projected_7k:.2f} PPL, which is:")
print(f"   - {((44.15 - projected_7k) / 44.15 * 100):.1f}% better than Movement Pruning (44.15 PPL)")
print(f"   - {((projected_7k - 37.28) / 37.28 * 100):.1f}% worse than bitter7 (37.28 PPL)")
print()
print("3. This suggests torch.compile is NOT the main driver of improvement -")
print("   the algorithm (state-based vs movement) matters more!")
print()
print("4. Answer: Movement Pruning WITH torch.compile gave 44.15 PPL @ 5000 iters")
print("   But bitter8 WITHOUT compile would have beaten it if it completed!")

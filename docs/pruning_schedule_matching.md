# Matching Pruning Schedules for Fair A/B Testing

## Problem

Magnitude pruning and Bitter7 state pruning have different sparsity progression rates:

- **Magnitude**: Reached 50% sparsity at iteration ~2,990
- **Bitter7**: At iteration 2,270, only 0.5% sparsity

This makes comparison unfair - we're comparing a sparse model vs essentially dense model.

## Root Cause

Both use the same **cubic schedule** formula, but different `ramp_end_step`:

### Magnitude Pruning Schedule
```python
# lib/magnitude_pruning.py, line 28-29
ramp_end_step=3000,  # DEFAULT
schedule="cubic",
```

Cubic formula (line 86-88):
```python
ramp_progress = (step - warmup) / (ramp_end - warmup)
ramp_progress = ramp_progress ** 3  # cubic
sparsity = target_sparsity * ramp_progress
```

With defaults:
- warmup_steps = 100 (from command line)
- ramp_end_step = 3000 (hardcoded default)
- Progress at 2990: (2990-100)/(3000-100) = 0.996 → 0.996³ = 0.988 → 49.4% sparsity ✓

### Bitter7 (AdamWPrune) Schedule
```python
# lib/optimizers.py, line 786
ramp_end_step = adamprune_state.get("ramp_end_step", 10000)
```

From vanilla.py line 114:
```python
self.args.adamwprune_ramp_end_step = self.args.max_iters  # 10000!
```

With bitter7 config:
- warmup_steps = 100 (from command line: --pruning-warmup 100)
- ramp_end_step = 10000 (set to max_iters)
- Progress at 2270: (2270-100)/(10000-100) = 0.219 → 0.219³ = 0.0105 → 0.5% sparsity ✓

**The issue**: Bitter7 ramps to 50% over 10,000 iterations while magnitude does it in 3,000!

## Solution Options

### Option 1: Match ramp_end_step (Simplest)

Make both methods use the same `ramp_end_step = 3000`:

**For Magnitude** - Already using 3000 by default in code, but check command line:
```bash
--pruning-method magnitude --target-sparsity 0.5 --pruning-warmup 100
# (ramp_end_step defaults to 3000 in MagnitudePruning.__init__)
```

**For Bitter7** - Override the ramp_end_step:

In `gpt2/trainers/vanilla.py` line 114, change:
```python
# OLD:
self.args.adamwprune_ramp_end_step = self.args.max_iters

# NEW:
self.args.adamwprune_ramp_end_step = 3000  # Match magnitude pruning
```

OR add a command-line argument:
```bash
--adamwprune-ramp-end-step 3000
```

### Option 2: Shared Sparsity Schedule Config

Create a common configuration that both inherit:

```python
# In defconfig or config system
CONFIG_PRUNING_WARMUP=100
CONFIG_PRUNING_RAMP_END=3000
CONFIG_TARGET_SPARSITY=0.5
CONFIG_PRUNING_SCHEDULE=cubic

# Both magnitude and bitter7 read these
```

### Option 3: Match Magnitude's Observed Schedule

Extract magnitude's actual schedule and make bitter7 follow it exactly via interpolation table.

### Option 4: Explicit Milestone-Based Schedule

Define discrete pruning milestones both methods must hit:

```python
PRUNING_SCHEDULE = {
    100: 0.00,   # warmup done
    500: 0.01,   # 1%
    1000: 0.08,  # 8%
    1500: 0.19,  # 19%
    2000: 0.35,  # 35%
    2500: 0.58,  # 58% (overshoots to settle at 50%)
    3000: 0.50,  # 50% target
}
```

## Recommended Approach

**Use Option 1** - it's the simplest and preserves the cubic schedule:

1. Set `adamwprune_ramp_end_step = 3000` to match magnitude
2. Both reach 50% at ~iteration 3000
3. Both use cubic schedule (slow start, fast finish)
4. Fair comparison of pruning **criteria** (magnitude vs optimizer state)

## Implementation

### Quick Fix (no code changes)

Currently **NOT POSSIBLE** - no command-line arg exists for `--adamwprune-ramp-end-step`.

### Proper Fix (minimal code change)

Edit `gpt2/trainers/vanilla.py` line 114:

```python
# Change from:
self.args.adamwprune_ramp_end_step = self.args.max_iters

# To:
self.args.adamwprune_ramp_end_step = getattr(
    self.args, 'adamwprune_ramp_end_step', 3000
)
```

Then add command-line argument in `gpt2/train.py`:

```python
parser.add_argument(
    "--adamwprune-ramp-end-step",
    type=int,
    default=3000,
    help="Iteration at which pruning reaches target sparsity (default: 3000)"
)
```

### For Your Current Runs

Since bitter7 is already running with ramp_end=10000, you have two choices:

1. **Let it complete** - it will eventually reach 50% around iter 10,000, then compare
2. **Restart with matched schedule** - waste the current run but get fair comparison sooner

## Verification

After matching schedules, both should show similar sparsity progression:

| Iteration | Target Sparsity (cubic) | Magnitude Actual | Bitter7 Actual |
|-----------|------------------------|------------------|----------------|
| 100       | 0.0%                   | 0.0%            | 0.0%          |
| 500       | 0.6%                   | 0.1%            | ~0.6%         |
| 1000      | 4.6%                   | 1.5%            | ~4.6%         |
| 1500      | 11.6%                  | 5.6%            | ~11.6%        |
| 2000      | 21.1%                  | 14.1%           | ~21.1%        |
| 2500      | 32.9%                  | ?               | ~32.9%        |
| 3000      | 50.0%                  | 50.0%           | ~50.0%        |

Note: Magnitude's actual values lag behind theoretical because it updates masks every 100 steps (pruning_frequency=100).

## Expected Outcome

With matched schedules:
- Both reach 50% sparsity around iteration 3000
- Both train to 10,000 iterations at 50% sparsity
- Valid comparison of **perplexity at same sparsity level**
- Identifies which **pruning criterion** (magnitude vs optimizer state) is superior

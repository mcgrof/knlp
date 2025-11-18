# Claude AI Assistant Preferences

## Git Commit Practices

### Commit Structure
- Make small, atomic commits - one logical change per commit
- Each commit should be functional and not break the build
- Run code formatter (black for Python) after each change
- Run scripts/fix_whitespace_issues.py always on all files
- Test that code runs successfully before committing

### Commit Messages
- **MANDATORY**: Always use this exact format for ALL commits:
  ```
  file.py: brief description of change

  Detailed explanation of what was changed and why.
  Include technical details about the implementation.

  Generated-by: Claude AI
  Signed-off-by: Luis Chamberlain <mcgrof@kernel.org>
  ```

- **LINE LENGTH**: Maximum 70 characters per line in commit messages
  - Subject line (first line): 70 characters max
  - Body paragraphs: 70 characters max per line
  - Ensures proper display in git log, email patches, and terminal output
- **CRITICAL**: Never use "🤖 Generated with [Claude Code]" or "Co-Authored-By: Claude"
- **REQUIRED**: Every commit MUST have both "Generated-by: Claude AI" and "Signed-off-by: Luis Chamberlain <mcgrof@kernel.org>"
- **NO EXCEPTIONS**: This format is mandatory for ALL commits, no matter how small
- **STYLE**: Be terse and to the point. NO shopping-list style bullet points. Write in paragraphs explaining the change, rationale, and technical details concisely. Avoid verbose enumeration unless absolutely necessary for clarity.

## Cross-Agent Access

Some automation looks for agent-specific instruction files (e.g., `CODEX.md`)
instead of `CLAUDE.md`. To avoid future assistants missing these guidelines,
ensure every agent entrypoint symlinks back to this document. For Codex runs,
`CODEX.md` must always be a symlink to `CLAUDE.md`; add additional symlinks if
new agent names are introduced.

### Development Workflow
1. Make a single focused change
2. Run `black` formatter on Python files
3. Test that the code runs without errors
4. **If architectural changes**: Run `make check` to validate
5. Commit with detailed message
6. Repeat for next change

Architectural changes include:
- New attention or MLP mechanisms
- Modified forward/backward pass logic
- Changes to model patching or wrapper classes
- New ablation steps or configurations
- Updates to reciprocity/context flow

## Code Style

### Python
- Use `black` formatter for all Python code
- Follow PEP 8 conventions (handled by black)
- No manual formatting - always use black

### Defconfig Files
- **CRITICAL**: Defconfig files must use exact Kconfig syntax: `CONFIG_XXX=y` (no spaces around `=`)
- **CRITICAL**: NO inline comments allowed - comments MUST be on separate lines starting with `#`
  - ✅ CORRECT:
    ```
    # This is a comment
    CONFIG_SOMETHING=y
    ```
  - ❌ WRONG (breaks Kconfig parser):
    ```
    CONFIG_SOMETHING=y  # This breaks everything
    CONFIG_SOMETHING=y # This also breaks
    ```
- **DO NOT** apply `black` formatter to defconfig files or `.config` files
- Kconfig parser silently ignores lines with spaces around equals signs
- After any edit to defconfigs, verify syntax: `grep " = " defconfigs/*` should return nothing

## GPU Optimization Preferences

### Training Optimizations
When optimizing PyTorch training for AMD GPUs:
- Increase batch size to utilize GPU memory
- Enable cuDNN benchmark mode
- Use mixed precision training (AMP)
- Add multiple data loader workers with pinned memory
- Include GPU warmup routine
- Use torch.compile() for graph optimization
- Enable TensorFloat32 for matrix operations
- Add comprehensive timing and metrics
- Save trained models after completion

### Performance Monitoring
- Display GPU info at startup
- Show per-epoch timing
- Track test accuracy after each epoch
- Report total training time and average per epoch

## Hardware
- Primary GPU: AMD Radeon Pro W7900 (48GB)
- Optimize for maximum GPU utilization

## Testing Requirements
- Always verify code runs before committing
- Check for linting/formatting issues
- Ensure no syntax errors

## Experiment Workflow

The standard workflow for running experiments:

1. **Load configuration**: `make defconfig-<name>`
   - Example: `make defconfig-gpt2-ratio-ablation`
   - This loads the defconfig and generates config.py

2. **Build and run**: `make`
   - The build system automatically runs the configured
     experiments
   - For test matrix mode, this runs all ablation steps
   - Results are saved to the configured output directory

3. **NEVER manually invoke make targets or scripts**
   - Never run `make train` directly
   - Never run `scripts/run_test_matrix.py` directly
   - The default `make` target handles everything automatically
   - Manual target/script invocation is for debugging only

Example complete workflow:
```bash
make defconfig-gpt2-ratio-ablation
make
# Results appear in test_matrix_results_ratio_ablation/
```

**CRITICAL**: Always use `make` (not `make train`) to run
experiments. The default target adapts automatically based on
configuration (single training vs test matrix mode).

## Configuration System Internals

### Type Handling
- `.config` files use string values: `"y"`, `"n"`, `"value"`
- `config.py` converts to Python types: `True`, `False`, integers, floats
- When checking config values in Python code, handle both types:
  ```python
  # Good - handles both string and boolean
  if value in ("y", True):

  # Bad - only works with one type
  if value == "y":
  ```

### Test Matrix vs Ablation Mode
- **Mutually exclusive**: Cannot enable both `CONFIG_TEST_MATRIX_MODE` and `CONFIG_RA_MLA_ABLATION_MODE`
- Test matrix mode: Tests optimizer/pruning combinations
- Ablation mode: Tests architectural variations (RA, MLA, RA-CT, etc.)
- Always verify which mode is active when debugging unexpected test counts

## Ablation Study Requirements

### Multi-File Synchronization
When extending ablation studies with new steps, **THREE** files must be updated in sync:

1. **defconfigs/gpt2-ratio-ablation**: Add step descriptions in comments
2. **gpt2/train_ra_mla.py**: Add step configurations (elif step == "N" blocks)
3. **scripts/run_test_matrix.py**: Update `step_descriptions` dictionary

Missing any of these causes:
- Defconfig only: Steps run but have no description
- train_ra_mla.py only: Steps fail to execute
- run_test_matrix.py only: Descriptions show but steps don't run

### Ablation Step Checklist
When adding a new ablation step:
- [ ] Add step config block to train_ra_mla.py (around line 500+)
- [ ] Update step_descriptions dict in run_test_matrix.py (around line 2095)
- [ ] Document step in defconfig comments
- [ ] Update CONFIG_RA_MLA_ABLATION_STEPS string to include new step number
- [ ] **REQUIRED**: Validate with dry-run: `./scripts/validate_ablation_steps.sh`

## Dry-Run Validation

### Architecture Validation Before GPU Training
**CRITICAL**: Always validate architectural changes with dry-run
before committing GPU resources. Recent bugs wasted 7+ hours of
GPU time that dry-run would have caught in 60 seconds.

### When to Use Dry-Run
Run dry-run validation before:
- Committing architectural changes (new attention mechanisms,
  MLP modifications)
- Adding new ablation steps
- Modifying forward/backward pass logic
- Changing wrapper classes or patching code
- After fixing bugs that affected multiple configurations

### Dry-Run Tools

#### Quick Check (Recommended)
```bash
# Run full architecture validation via Makefile
make check

# Completes in ~97 seconds (19 steps @ ~5s each)
# Loads gpt2-ratio-ablation config with DRY_RUN=1
# Tests all ablation steps automatically
# Exit code 0: all pass, 1: failures detected
```

**ALWAYS run `make check` before committing architectural changes
that may affect runtime behavior.**

#### Single Step Validation
```bash
# Test specific ablation step
python3 gpt2/train_ra_mla.py --ra-mla-ablation-step N \
  --optimizer adamwspam --dataset finewebedu --dry-run

# Exit code 0: architecture valid
# Exit code 1: error (prints stack trace)
```

#### Manual All Steps Validation
```bash
# Test all 19 RATIO ablation steps (manual script)
./scripts/validate_ablation_steps.sh

# Completes in ~60 seconds
# Reports which steps pass/fail
# Provides commands to debug failures
```

### What Dry-Run Catches
- Configuration errors (wrong test mode, invalid parameters)
- Architecture errors (TypeError from wrong arguments)
- Assertion failures (missing required data)
- Forward pass failures (dimension mismatches)
- Backward pass failures (gradient computation errors)
- Optimizer step failures (parameter update errors)

### What Dry-Run Misses
- OOM errors (uses small batch on CPU)
- Multi-GPU/DDP issues (runs single CPU)
- Data loading errors (uses dummy data)
- Long-term training instabilities
- Performance regressions

### Recent Bugs Caught by Dry-Run
1. **RA_MLA_Block argument passing**: 17/19 steps failed with
   TypeError when MLP received unexpected kwargs
2. **Assertion strictness**: 6/19 steps failed when first block
   had no context from previous block

Both would have been caught before GPU training with dry-run.

## Defensive Programming

### Assertions for Optional Features
When implementing optional/conditional features that depend on
data flow:

- Add assertions for data that MUST be present (e.g., within
  a single component)
- Avoid assertions for data that may legitimately be None
  (e.g., first block in sequence)
- Silent failures waste GPU time - better to fail fast with
  clear error messages
- Pattern for required data within component:
  ```python
  if self.cfg.feature_enabled:
      assert required_data is not None, "feature_enabled but no required_data"
  ```
- Pattern for optional data from other blocks:
  ```python
  if self.cfg.feature_enabled and data_from_prev_block is not None:
      # use the data
  ```

Examples from RA+MLA:
- ReciprocalMLP asserts `attn_weights`/`attn_latent` are
  provided by RA_MLA_Block (same component, always required)
- RA_MLA_Attention handles None `mlp_gate_context` gracefully
  (from previous block, None for first block)
- Use dry-run validation to catch assertion failures before
  GPU training

### Context Flow for Multi-Block Architectures
When implementing bidirectional information flow between transformer blocks:

- Use wrapper classes (e.g., `RA_MLA_Block`) to manage context state across blocks
- Store contexts in instance variable (e.g., `self._ctx = {}`)
- Pass contexts as keyword arguments (enables detection of missing connections)
- Produce contexts for the **next** block at the end of forward pass
- Never assume contexts exist - always check with assertions when used

### Wrapper Class Adaptability
When creating wrapper classes for mixed configurations:

- **Check wrapped component type at runtime**: Use `hasattr()` or `isinstance()` to detect capabilities
- **Conditionally pass arguments**: Standard components may not accept extended keyword arguments
- **Graceful degradation**: Support both enhanced and standard components in same wrapper
- Pattern:
  ```python
  # Good - adapts to component type
  is_enhanced = hasattr(self.component, "enhanced_method")
  if is_enhanced:
      out = self.component(x, extra_arg=value)
  else:
      out = self.component(x)

  # Bad - assumes all components are enhanced
  out = self.component(x, extra_arg=value)  # crashes on standard components
  ```

Example: `RA_MLA_Block` wraps either `ReciprocalMLP` (accepts attn_weights/attn_latent) or standard `MLP` (does not). Runtime check prevents TypeError when ablation steps disable reciprocity mechanisms.

## Architectural Pattern Guidelines

### Feature Independence and Composability
When adding new attention/MLP mechanisms:

- **Keep features orthogonal**: RA-CT (attention-only gating) vs MLP mechanisms (cross-layer flow)
- **Use clear naming**: `ra_cross_token` for attention features, `mlp_attn_gate` for MLP features
- **Enable ablation**: Each feature should be independently testable
- **Avoid coupling**: RA-CT doesn't require MLA/RA, can be tested on baseline GPT-2

### Per-Head Learnable Parameters
For per-head gating mechanisms:

- Initialize to near-identity: `bias ≈ 2.0` for sigmoid gates (pass-through initially)
- Use affine transforms: `sigmoid(stat * scale + bias)` for numerical stability
- Shape: `[n_head]` for per-head parameters, expandable to `[B,H,T]` when needed
- Consider `head_average=True` option for cheaper computation

### Statistics-Based Gating
When implementing gating based on attention statistics:

- Support multiple modes: `topk`, `max`, `entropy`, `rms`
- Provide `detach_stats` option to compute under `no_grad()` for memory savings
- Apply gate at multiple points: `weights` (pre-softmax) or `output` (post-aggregation)
- Use `alpha` mixing parameter for smooth interpolation: `(1-α)·x + α·(x⊙gate)`

## GPU Memory Management

### Memory Optimization Strategies
- Enable expandable segments: `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`
- Disable expensive metrics logging during training (e.g., entropy computation on attention weights)
- Use `@torch.no_grad()` for statistics computation that doesn't need gradients
- Monitor for OOM errors in attention mechanisms - often caused by extra allocations for metrics
- Batch size × gradient accumulation = effective batch size (keep constant when adjusting for memory)

### A10G-Specific Considerations
- 24GB VRAM per GPU requires careful batch size tuning
- For GPT-2 124M with RA+MLA: batch_size=8, gradient_accumulation=8 (effective=64)
- Tensor dimensions should be multiples of 64 for optimal tensor core utilization
- Disable metrics logging for attention mechanisms to prevent OOM during entropy computation

## WandB Helper Scripts

When analyzing experiment results or comparing GPU performance across
runs, use the W&B query scripts in the scripts/ directory. These
require the micromamba environment.

### Environment Setup

Before running any W&B query scripts:

```bash
source ~/bin/wl700-ml  # Activates w7900-ml micromamba environment
```

This provides wandb, pandas, and other dependencies needed for
querying experiment data.

### Available Scripts

**scripts/inspect_wandb_keys.py**: Discover available metrics in a run

Usage for inspecting what data is available:
```bash
python scripts/inspect_wandb_keys.py \
  --entity mcgrof-citizen \
  --project gpt2-bitter9-compiled-b200x4 \
  --run-name gpt2_adamwprune_bitter9_state_50
```

**scripts/query_wandb_gpu.py**: Query GPU metrics from training history

Usage for checking GPU memory and compute utilization:
```bash
python scripts/query_wandb_gpu.py \
  --entity mcgrof-citizen \
  --project gpt2-bitter9-compiled-b200x4 \
  --run-name gpt2_adamwprune_bitter9_state_50
```

**scripts/query_wandb_gpu_full.py**: Query detailed GPU metrics from
system events

Usage for detailed system metrics including power and temperature:
```bash
python scripts/query_wandb_gpu_full.py \
  --entity mcgrof-citizen \
  --project gpt2-bitter9-compiled-b200x4 \
  --run-name gpt2_adamwprune_bitter9_state_50
```

**scripts/plot_torch_compile_impact.py**: Generate publication-quality
visualizations comparing GPU performance across runs

This is a reusable visualization script that queries W&B and
generates four graphs showing performance comparisons. Used to
prove torch.compile() was the bottleneck.

Usage:
```bash
source ~/bin/wl700-ml
python scripts/plot_torch_compile_impact.py
```

The script is hardcoded to query
`mcgrof-citizen/gpt2-bitter8-nocompile-w7900` but can be easily
adapted for other projects by editing the `project` variable in
`main()`.

Generated graphs (300 DPI, publication quality):
- `torch_compile_comparison.png`: Side-by-side memory and compute
  comparison
- `torch_compile_grouped.png`: All runs in grouped bar chart with
  color coding
- `torch_compile_before_after.png`: Dramatic before/after horizontal
  bars with annotations
- `bitter8_vs_baseline.png`: Spotlight showing minimal overhead of
  state-based pruning

The script demonstrates the pattern for:
1. Querying W&B API for multiple runs
2. Extracting system.gpu.* metrics from event stream
3. Computing averages across runs
4. Creating matplotlib visualizations with annotations
5. Using color coding (red=bad, green=good) for clarity

When to use this script:
- After GPU profiling reveals performance differences
- To prove bottleneck hypotheses with visual evidence
- To compare optimization variants systematically
- To generate graphs for documentation or papers

Customization tips:
- Edit `project` variable to query different W&B project
- Modify `fetch_wandb_data()` to extract different metrics
- Update graph functions to change visual style
- Add new graph types by creating new functions following existing
  patterns

### Comparing Runs

To compare GPU performance across multiple runs (baseline vs
optimizations), write a custom Python script using the W&B API.
See docs/tracker.md for detailed examples.

Pattern for comparing runs:
```python
import wandb

api = wandb.Api()
project = "mcgrof-citizen/gpt2-bitter9-compiled-b200x4"

run_names = ["baseline", "bitter8", "bitter9"]

for name in run_names:
    runs = api.runs(project, filters={"config.run_name": name})
    if runs:
        run = runs[0]
        history = run.history(
            keys=["gpu/memory_util_avg", "gpu/compute_util_avg"],
            samples=1000
        )
        if not history.empty:
            print(f"{name}:")
            print(f"  Memory: {history['gpu/memory_util_avg'].mean():.2f}%")
            print(f"  Compute: {history['gpu/compute_util_avg'].mean():.2f}%")
```

### Key Metrics to Check

When analyzing GPU performance issues:

- `gpu/memory_util_avg`: Memory bandwidth utilization (%)
- `gpu/compute_util_avg`: Compute utilization (%)
- `gpu/memory_used_avg_gb`: Average memory per GPU (GB)

Low memory utilization (<20%) indicates memory bandwidth bottleneck.
Low compute utilization (<50%) indicates compute bottleneck.
Compare optimization runs to baseline to verify improvements.

## Publishing Results

Before publishing experimental results in documentation, papers, or
public communications, perform rigorous verification to ensure
reproducibility and fairness.

### Verification Checklist

When publishing statistics or performance comparisons:

1. **Use W&B API to verify hyperparameters**: Query all runs via
   W&B API to confirm consistent hyperparameters across comparisons.
   Verify batch size, gradient accumulation, learning rate, warmup
   steps, and all optimizer-specific settings match exactly.

2. **Verify git commit exists and is public**: Confirm the exact
   git commit SHA used for training exists in the public repository.
   Document the commit ID in published results so others can
   reproduce experiments with identical code.

3. **Perform apples-to-apples sanity checks**: Before claiming
   performance differences, verify:
   - Equal training time (CONFIG_GPT2_MAX_TIME) across all methods
   - Same effective batch size (batch × grad_acc × num_gpus)
   - Same hardware configuration (GPU type, count, memory)
   - Same torch.compile status (all enabled or all disabled)
   - Same dataset and preprocessing
   - Same evaluation protocol (samples, intervals)

4. **Check for confounding variables**: Verify no unintended
   differences like:
   - Different torch.compile status (one compiled, one not)
   - Different batch sizes due to GPU-specific configs
   - Different stopping conditions (time vs iterations)
   - Different random seeds causing outlier results
   - Different CUDA/PyTorch/GPU driver versions

### W&B Verification Script Pattern

Use this pattern to verify hyperparameter consistency:

```python
import wandb

api = wandb.Api()
project = "mcgrof-citizen/your-project"
run_names = ["baseline", "method_a", "method_b"]

configs = {}
for name in run_names:
    runs = api.runs(project, filters={"display_name": name})
    if runs:
        run = runs[0]
        configs[name] = {
            "batch_size": run.config.get("batch_size"),
            "gradient_accumulation": run.config.get("gradient_accumulation"),
            "learning_rate": run.config.get("learning_rate"),
            "max_time": run.config.get("max_time"),
            "compile": run.config.get("compile_model"),
            "commit": run.config.get("git_commit"),
        }

# Verify all configs match on critical hyperparameters
for key in ["batch_size", "gradient_accumulation", "learning_rate"]:
    values = [c[key] for c in configs.values()]
    if len(set(values)) > 1:
        print(f"WARNING: {key} differs across runs: {configs}")
```

### Publication Requirements

Published results MUST include:

- Git commit SHA for exact code version
- W&B project and run names for verification
- Hardware specification (GPU model, count, memory)
- Training time allocation per method
- Effective batch size calculation
- torch.compile status
- Dataset and preprocessing details

This enables independent verification and reproduction of published
claims. Do not publish results without completing verification
checklist.

## Documentation
- Keep changes well-documented in commit messages
- Explain technical rationale for optimizations
- Include performance impact where applicable

## Avoid silly language

You are not allowed to use the word "comprehensive". It is overused
and does not explain anything. We prefer to be terse and to the point.

# Memory

I want you to remember most of our conversations about this project.

"""
Training script for GPT-2 with Reciprocal Attention (RA) + MLA.

This script extends the base GPT-2 training with:
- RA+MLA attention mechanism
- Attention entropy and reciprocity logging
- Empirical complexity measurement (actual FLOPs and memory)
- Integration with AdamWPrune pruning hooks

Usage:
    python train_ra_mla.py --dataset finewebedu \
                           --latent-dim 64 \
                           --ra-window 64 \
                           --ra-alpha 0.5 \
                           --max-iters 10000

Ablation studies:
    # Pure MLA (no reciprocal)
    python train_ra_mla.py --ra-alpha 0.0 --latent-dim 64

    # Different latent dimensions
    python train_ra_mla.py --latent-dim 32  # more compression
    python train_ra_mla.py --latent-dim 128  # less compression

    # Different reciprocal windows
    python train_ra_mla.py --ra-window 32  # narrow local context
    python train_ra_mla.py --ra-window 128  # wider local context
"""

import os
import sys

# CRITICAL: Set environment variables before importing torch
# Read from config.py if available, otherwise use defaults
try:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    from config import Config

    config = Config()
    if hasattr(config, "PYTORCH_CUDA_ALLOC_CONF"):
        # Set both old and new variable names for compatibility
        alloc_conf = config.PYTORCH_CUDA_ALLOC_CONF
        os.environ.setdefault("PYTORCH_ALLOC_CONF", alloc_conf)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", alloc_conf)

    # Detect GPU type and clear ROCm variables if on NVIDIA
    # This prevents spurious "No ROCm runtime found" messages on NVIDIA GPUs
    try:
        import subprocess

        # Try nvidia-smi first (fast check for NVIDIA GPUs)
        nvidia_result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if nvidia_result.returncode == 0:
            # NVIDIA GPU detected - clear ROCm environment variables
            for var in ["ROCM_PATH", "ROCM_HOME", "HIP_PATH", "HIP_PLATFORM"]:
                if var in os.environ:
                    del os.environ[var]
            # Also remove ROCm paths from LD_LIBRARY_PATH
            if "LD_LIBRARY_PATH" in os.environ:
                ld_paths = os.environ["LD_LIBRARY_PATH"].split(":")
                ld_paths = [
                    p for p in ld_paths if "/opt/rocm" not in p and "/hip/" not in p
                ]
                os.environ["LD_LIBRARY_PATH"] = ":".join(ld_paths)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass  # nvidia-smi not available

    # Enable experimental flash attention for Navi31/32/33 when explicitly requested
    # Only set environment variable if CONFIG_GPT2_FLASH_ATTENTION=y AND we detect Navi3x
    flash_attention_enabled = (
        hasattr(config, "GPT2_FLASH_ATTENTION")
        and config.GPT2_FLASH_ATTENTION
        and config.GPT2_FLASH_ATTENTION != "n"
    )

    if flash_attention_enabled:
        # Detect Navi31/32/33 before importing torch using rocm-smi
        try:
            import subprocess

            result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                gpu_name = result.stdout.lower()
                is_navi3x = any(
                    x in gpu_name for x in ["navi31", "navi32", "navi33", "w7900"]
                )
                if is_navi3x:
                    os.environ.setdefault(
                        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1"
                    )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass  # rocm-smi not available or timed out
except (ImportError, AttributeError):
    # Fallback to safe defaults if config.py doesn't exist or doesn't have the settings
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
import math
import argparse
import json
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import numpy as np

# Suppress wandb weave warning
try:
    import weave
except ImportError:
    pass

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPT, GPTConfig
from ra_mla_gpt2 import patch_gpt2_with_ra_mla, score_heads_for_prune_gpt2
from ra_lens_gpt2 import (
    patch_gpt2_with_lens_attention,
    apply_route_annealing,
    get_mean_route_gate,
    analyze_lens_gates,
)

# Import training utilities from base train.py
try:
    from lib.optimizers import create_optimizer
    from lib.scaling_curves import show_scaling_curves
except ImportError:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from lib.optimizers import create_optimizer
    from lib.scaling_curves import show_scaling_curves


# -----------------------------------------------------------------------------
# Helper functions
def supports_tensorcore_fp32():
    """
    Detect if the current device supports tensor cores (NVIDIA) or WMMA (AMD).
    Works on both CUDA and ROCm backends.
    """
    if not torch.cuda.is_available():
        return False

    try:
        # Check device capability for NVIDIA GPUs
        cap = torch.cuda.get_device_capability(0)
        # Tensor cores require SM >= 7.0 (Volta and newer)
        if cap and cap[0] >= 7:
            return True
    except Exception:
        pass

    # ROCm heuristic: look for RDNA3 / gfx11+ which have WMMA ISA
    if torch.version.hip:
        try:
            arch = torch.cuda.get_device_name(0).lower()
            # Check for RDNA3 (gfx110x, Navi3x, W7900) or MI series
            return any(
                x in arch
                for x in [
                    "gfx110",
                    "navi31",
                    "navi32",
                    "navi33",
                    "w7900",
                    "mi300",
                    "mi250",
                ]
            )
        except Exception:
            pass

    return False


# ============================================================================
# Argument Parsing
# ============================================================================

parser = argparse.ArgumentParser(description="GPT-2 RA+MLA training")

# Model configuration
parser.add_argument("--model-name", type=str, default="gpt2", help="GPT-2 model size")
parser.add_argument("--block-size", type=int, default=1024, help="Context length")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

# RA+MLA configuration
parser.add_argument(
    "--latent-dim", type=int, default=64, help="Latent dimension for K/V compression"
)
parser.add_argument(
    "--ra-window", type=int, default=64, help="Reciprocal attention band width"
)
parser.add_argument(
    "--ra-alpha",
    type=float,
    default=0.5,
    help="Reciprocal attention weight (0.0 = pure MLA)",
)
parser.add_argument(
    "--per-head-q-latent",
    action="store_true",
    default=True,
    help="Per-head Q-to-latent projection",
)
parser.add_argument(
    "--per-head-v-up",
    action="store_true",
    default=True,
    help="Per-head V up-projection",
)
parser.add_argument(
    "--use-flash",
    action="store_true",
    default=True,
    help="Use FlashAttention if available",
)

# Lens-gated architecture option (simplified, parameter-efficient)
parser.add_argument(
    "--use-lens",
    action="store_true",
    default=False,
    help="Use lens-gated architecture (reciprocity, discoverability, route gate) instead of complex RA+MLA",
)

# RWR (Random Walk with Restart) attention option
parser.add_argument(
    "--use-rwr",
    action="store_true",
    default=False,
    help="Use RWR attention (LOCAL+RWR factorization with random walk long-range)",
)

parser.add_argument(
    "--lens-kv-compression",
    action="store_true",
    default=False,
    help="Enable K/V compression for parameter-neutral design (lens mode only)",
)
parser.add_argument(
    "--lens-kv-latent-dim",
    type=int,
    default=128,
    help="Latent dimension for K/V compression (lens mode only)",
)
parser.add_argument(
    "--lens-ctx-rank",
    type=int,
    default=128,
    help="Low-rank bottleneck for MLP context projection (lens mode only)",
)
parser.add_argument(
    "--lens-ctx-conductor",
    action="store_true",
    default=False,
    help="Conductor mode: only use MLP context when route_gate < 0.5 (lens mode only)",
)

# Reciprocal MLP configuration (legacy RA+MLA)
parser.add_argument(
    "--mlp-attn-gate",
    action="store_true",
    default=False,
    help="Enable mechanism 1: MLP gates attention heads (legacy RA+MLA)",
)
parser.add_argument(
    "--no-mlp-attn-gate",
    dest="mlp_attn_gate",
    action="store_false",
    help="Disable MLP attention gating",
)
parser.add_argument(
    "--mlp-cross-token",
    action="store_true",
    default=False,
    help="Enable mechanism 2: Cross-token MLP aggregation",
)
parser.add_argument(
    "--no-mlp-cross-token",
    dest="mlp_cross_token",
    action="store_false",
    help="Disable cross-token MLP",
)
parser.add_argument(
    "--mlp-latent-recip",
    action="store_true",
    default=False,
    help="Enable mechanism 3: MLP-attention latent reciprocity",
)
parser.add_argument(
    "--no-mlp-latent-recip",
    dest="mlp_latent_recip",
    action="store_false",
    help="Disable MLP latent reciprocity",
)
parser.add_argument(
    "--mlp-gate-alpha",
    type=float,
    default=0.1,
    help="Mixing weight for MLP attention gating",
)
parser.add_argument(
    "--mlp-cross-alpha",
    type=float,
    default=0.3,
    help="Mixing weight for cross-token MLP",
)
parser.add_argument(
    "--mlp-recip-alpha",
    type=float,
    default=0.2,
    help="Mixing weight for MLP latent reciprocity",
)
parser.add_argument(
    "--mlp-gate-dim",
    type=int,
    default=64,
    help="Context vector dimension for gating",
)
parser.add_argument(
    "--mlp-latent-dim",
    type=int,
    default=128,
    help="Dimension for MLP latent space",
)
parser.add_argument(
    "--mlp-expansion-ratio",
    type=float,
    default=4.0,
    help="MLP expansion ratio (mlp_dim = expansion_ratio * n_embd)",
)
parser.add_argument(
    "--mlp-tying-mode",
    type=str,
    default="tied_transpose",
    choices=["untied", "tied_transpose", "per_head_scalar"],
    help="Parameter tying mode for MLP-Attention coupling",
)
parser.add_argument(
    "--mlp-sparse-mode",
    type=str,
    default="topk",
    choices=["none", "topk", "rms"],
    help="Sparsification mode for cross-token MLP aggregation",
)
parser.add_argument(
    "--mlp-sparse-k",
    type=int,
    default=8,
    help="Top-k value for sparse cross-token aggregation",
)
parser.add_argument(
    "--mlp-sparse-tau",
    type=float,
    default=0.5,
    help="RMS threshold for sparse aggregation",
)
parser.add_argument(
    "--mlp-sparse-normalize",
    action="store_true",
    default=True,
    help="Normalize sparsified weights",
)
parser.add_argument(
    "--no-mlp-sparse-normalize",
    dest="mlp_sparse_normalize",
    action="store_false",
    help="Disable weight normalization after sparsification",
)
parser.add_argument(
    "--mlp-sparse-head-average",
    action="store_true",
    default=True,
    help="Average attention weights across heads",
)
parser.add_argument(
    "--no-mlp-sparse-head-average",
    dest="mlp_sparse_head_average",
    action="store_false",
    help="Disable head averaging for cross-token MLP",
)
parser.add_argument(
    "--ra-mla-ablation-step",
    type=str,
    default=None,
    help="Ablation study step (0-5). Overrides mechanism flags to enable specific combinations.",
)

# Dataset
parser.add_argument("--dataset", type=str, default="shakespeare", help="Dataset to use")
parser.add_argument("--data-dir", type=str, default="data", help="Data directory")

# Training
parser.add_argument("--batch-size", type=int, default=12, help="Batch size")
parser.add_argument(
    "--gradient-accumulation", type=int, default=4, help="Gradient accumulation steps"
)
parser.add_argument("--max-iters", type=int, default=10000, help="Maximum iterations")
parser.add_argument(
    "--max-time",
    type=int,
    default=None,
    help="Maximum training time in seconds (alternative to max-iters)",
)
parser.add_argument("--learning-rate", type=float, default=6e-4, help="Learning rate")
parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
parser.add_argument("--min-lr", type=float, default=6e-5, help="Minimum learning rate")

# Optimizer
parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer to use")

# SPAM configuration (for AdamWSPAM optimizer)
parser.add_argument("--spam-theta", type=float, default=50.0, help="SPAM theta")
parser.add_argument(
    "--spam-interval", type=int, default=1000, help="SPAM reset interval"
)
parser.add_argument(
    "--spam-warmup-steps", type=int, default=100, help="SPAM warmup steps"
)
parser.add_argument(
    "--spam-enable-clip", action="store_true", help="Enable SPAM clipping"
)

# SinkGD configuration (for SinkGD optimizer)
parser.add_argument(
    "--sinkgd-lr",
    type=float,
    default=None,
    help="SinkGD learning rate (defaults to --learning-rate)",
)
parser.add_argument(
    "--sinkgd-weight-decay",
    type=float,
    default=None,
    help="SinkGD weight decay (defaults to --weight-decay)",
)
parser.add_argument(
    "--sinkgd-tau",
    type=float,
    default=0.1,
    help="SinkGD temperature for Sinkhorn smoothing",
)
parser.add_argument(
    "--sinkgd-iters",
    type=int,
    default=5,
    help="SinkGD number of normalization iterations",
)
parser.add_argument(
    "--sinkgd-eps", type=float, default=1e-8, help="SinkGD numerical stability epsilon"
)

# RWR (Random Walk with Restart) Attention configuration
parser.add_argument(
    "--rwr-alpha", type=float, default=0.2, help="RWR restart probability"
)
parser.add_argument(
    "--rwr-steps", type=int, default=4, help="RWR number of walk iterations"
)
parser.add_argument(
    "--rwr-topk", type=int, default=32, help="RWR top-k neighbors per query"
)
parser.add_argument(
    "--rwr-threshold", type=float, default=0.0, help="RWR minimum similarity threshold"
)
parser.add_argument(
    "--rwr-reversible", action="store_true", help="Enable RWR reversible chain"
)
parser.add_argument(
    "--rwr-reciprocal-beta",
    type=float,
    default=0.5,
    help="RWR reciprocal mixing (forward/backward saliency)",
)
parser.add_argument(
    "--rwr-lens-strength",
    type=float,
    default=0.3,
    help="RWR lens blending factor γ",
)
parser.add_argument(
    "--rwr-window", type=int, default=128, help="RWR local attention window"
)
parser.add_argument(
    "--rwr-block-size", type=int, default=128, help="RWR SRAM tile size"
)
parser.add_argument(
    "--rwr-head-dim-pad",
    type=int,
    default=64,
    help="RWR head dimension padding multiple",
)
parser.add_argument(
    "--rwr-use-discoverability",
    action="store_true",
    help="Enable RWR lens column bias",
)

# Test matrix compatibility arguments (accepted but ignored for RA+MLA)
parser.add_argument(
    "--decay-lr",
    action="store_true",
    default=True,
    help="Use LR decay (always enabled for RA+MLA)",
)
parser.add_argument(
    "--pruning-method",
    type=str,
    default="none",
    help="Pruning method (ignored for RA+MLA)",
)

# Evaluation and logging
parser.add_argument(
    "--eval-interval", type=int, default=100, help="Evaluation interval"
)
parser.add_argument(
    "--eval-samples", type=int, default=200, help="Number of evaluation samples"
)
parser.add_argument("--log-interval", type=int, default=10, help="Logging interval")
parser.add_argument(
    "--log-metrics", action="store_true", default=True, help="Log attention metrics"
)

# Experiment tracking
parser.add_argument(
    "--tracker",
    type=str,
    default="none",
    help="Experiment tracker(s): none, trackio, wandb, or comma-separated",
)
parser.add_argument(
    "--tracker-project",
    type=str,
    default=None,
    help="Project name for experiment tracking",
)
parser.add_argument(
    "--tracker-run-name",
    type=str,
    default=None,
    help="Run name for experiment tracking",
)

# Output
parser.add_argument(
    "--json-output", type=str, default=None, help="Path to save metrics JSON"
)
parser.add_argument(
    "--checkpoint-dir",
    type=str,
    default="checkpoints_ra_mla",
    help="Checkpoint directory",
)

# System
parser.add_argument("--device", type=str, default="cuda", help="Device to use")
parser.add_argument(
    "--dtype",
    type=str,
    default="bfloat16",
    help="Data type (bfloat16, float16, float32)",
)
parser.add_argument(
    "--compile", action="store_true", default=False, help="Use torch.compile"
)
parser.add_argument(
    "--dry-run",
    action="store_true",
    default=False,
    help="Quick architecture validation: create model, run single "
    "forward/backward pass with dummy data on CPU, exit. "
    "Catches config/architecture errors without GPU time.",
)

args = parser.parse_args()

# Override dry-run from config if set (CLI --dry-run takes precedence)
if not args.dry_run:
    try:
        if hasattr(config, "DRY_RUN") and (
            config.DRY_RUN is True or config.DRY_RUN == "y"
        ):
            args.dry_run = True
    except (NameError, AttributeError):
        pass

# Override max_iters from environment if set (takes precedence over config and command-line)
if os.environ.get("GPT2_MAX_ITERS"):
    args.max_iters = int(os.environ.get("GPT2_MAX_ITERS"))

# Override max_time from environment if set (time-based training limit in seconds)
# If both MAX_ITERS and MAX_TIME are set, training stops when EITHER is reached
# Environment variable takes precedence over command line argument
if os.environ.get("GPT2_MAX_TIME"):
    args.max_time = int(os.environ.get("GPT2_MAX_TIME"))

if args.max_time is not None:
    print(
        f"Time-based training enabled: max {args.max_time} seconds ({args.max_time/3600:.2f} hours)"
    )

# Handle ablation study step if specified
# This overrides configuration for RATIO ablation study (15 steps)
if args.ra_mla_ablation_step is not None:
    step = args.ra_mla_ablation_step
    if step == "0":
        # Step 0: Baseline GPT-2 (ratio 1:2.0, standard attention)
        args.enable_mla = False
        args.ra_alpha = 0.0
        args.mlp_attn_gate = False
        args.mlp_cross_token = False
        args.mlp_latent_recip = False
        args.mlp_expansion_ratio = 4.0  # Standard 3072
        args.ra_cross_token = False
    elif step == "1":
        # Step 1: Baseline + SPAM pruning 50% (pruning baseline)
        args.enable_mla = False
        args.ra_alpha = 0.0
        args.mlp_attn_gate = False
        args.mlp_cross_token = False
        args.mlp_latent_recip = False
        args.mlp_expansion_ratio = 4.0  # Standard 3072
        args.ra_cross_token = False
        # Pruning enabled via config (OPTIMIZER=adamwspam, TARGET_SPARSITY=0.5)
    elif step == "2":
        # Step 2: Golden ratio 1:2.5 via MLP resize (mlp_dim=3840)
        args.enable_mla = False
        args.ra_alpha = 0.0
        args.mlp_attn_gate = False
        args.mlp_cross_token = False
        args.mlp_latent_recip = False
        args.mlp_expansion_ratio = 5.0  # 3840 for golden ratio
        args.ra_cross_token = False
    elif step == "3":
        # Step 3: Step 2 + MLP gating 15%
        args.enable_mla = False
        args.ra_alpha = 0.0
        args.mlp_attn_gate = True
        args.mlp_cross_token = False
        args.mlp_latent_recip = False
        args.mlp_expansion_ratio = 4.25  # 3264 (85% of 3840), gating takes 15%
        args.ra_cross_token = False
    elif step == "4":
        # Step 4: Step 3 + cross-token 10%
        args.enable_mla = False
        args.ra_alpha = 0.0
        args.mlp_attn_gate = True
        args.mlp_cross_token = True
        args.mlp_latent_recip = False
        args.mlp_expansion_ratio = 4.0  # 3072 (80%), mechanisms take 20%
        args.ra_cross_token = False
    elif step == "5":
        # Step 5: Baseline + RA (ra_alpha=0.3, ratio 1:2.0)
        args.enable_mla = False
        args.ra_alpha = 0.3
        args.mlp_attn_gate = False
        args.mlp_cross_token = False
        args.mlp_latent_recip = False
        args.mlp_expansion_ratio = 4.0  # Standard 3072
        args.ra_cross_token = False
    elif step == "6":
        # Step 6: RA + golden ratio (ra_alpha=0.3, mlp_dim=3840, ratio 1:2.5)
        args.enable_mla = False
        args.ra_alpha = 0.3
        args.mlp_attn_gate = False
        args.mlp_cross_token = False
        args.mlp_latent_recip = False
        args.mlp_expansion_ratio = 5.0  # 3840 for golden ratio
        args.ra_cross_token = False
    elif step == "7":
        # Step 7: Step 6 + mechanisms (RA + ratio + gating + cross-token)
        args.enable_mla = False
        args.ra_alpha = 0.3
        args.mlp_attn_gate = True
        args.mlp_cross_token = True
        args.mlp_latent_recip = False
        args.mlp_expansion_ratio = 4.0  # 3072 (80%), mechanisms take 20%
        args.ra_cross_token = False
    elif step == "8":
        # Step 8: Baseline + MLA (latent_dim=128, ratio 1:3.0)
        args.enable_mla = True
        args.ra_alpha = 0.0
        args.mlp_attn_gate = False
        args.mlp_cross_token = False
        args.mlp_latent_recip = False
        args.mlp_expansion_ratio = 4.0  # 3072 (creates ratio 1:3.0 with MLA)
        args.ra_cross_token = False
    elif step == "9":
        # Step 9: MLA + golden ratio (latent_dim=128, mlp_dim=2560, ratio 1:2.5)
        args.enable_mla = True
        args.ra_alpha = 0.0
        args.mlp_attn_gate = False
        args.mlp_cross_token = False
        args.mlp_latent_recip = False
        args.mlp_expansion_ratio = 3.3333333333  # 2560 = 40×64 (GPU-aligned)
    elif step == "10":
        # Step 10: Step 9 + mechanisms (MLA + ratio + gating + cross-token)
        args.enable_mla = True
        args.ra_alpha = 0.0
        args.mlp_attn_gate = True
        args.mlp_cross_token = True
        args.mlp_latent_recip = False
        args.mlp_expansion_ratio = 2.6666666667  # 2048 = 32×64 (GPU-aligned)
    elif step == "11":
        # Step 11: RA + MLA + golden ratio (ra_alpha=0.3, latent_dim=128, ratio 1:2.5)
        args.enable_mla = True
        args.ra_alpha = 0.3
        args.mlp_attn_gate = False
        args.mlp_cross_token = False
        args.mlp_latent_recip = False
        args.mlp_expansion_ratio = 3.3333333333  # 2560 = 40×64 (GPU-aligned)
    elif step == "12":
        # Step 12: Step 11 + mechanisms (RA + MLA + ratio + mechanisms)
        args.enable_mla = True
        args.ra_alpha = 0.3
        args.mlp_attn_gate = True
        args.mlp_cross_token = True
        args.mlp_latent_recip = False
        args.mlp_expansion_ratio = 2.6666666667  # 2048 = 32×64 (GPU-aligned)
    elif step == "13":
        # Step 13: Baseline + RA-CT (topk, output mode, alpha=0.2)
        args.enable_mla = False
        args.ra_alpha = 0.0
        args.mlp_attn_gate = False
        args.mlp_cross_token = False
        args.mlp_latent_recip = False
        args.mlp_expansion_ratio = 4.0  # Standard 3072
        args.ra_cross_token = True
        args.ra_ct_mode = "topk"
        args.ra_ct_apply = "output"
        args.ra_ct_alpha = 0.2
        args.ra_ct_k = 8
    elif step == "14":
        # Step 14: MLA + RA-CT (compression + cross-token gating)
        args.enable_mla = True
        args.ra_alpha = 0.0
        args.mlp_attn_gate = False
        args.mlp_cross_token = False
        args.mlp_latent_recip = False
        args.mlp_expansion_ratio = 3.3333333333  # 2560 = 40×64 (GPU-aligned)
        args.ra_cross_token = True
        args.ra_ct_mode = "topk"
        args.ra_ct_apply = "output"
        args.ra_ct_alpha = 0.2
        args.ra_ct_k = 8
    elif step == "15":
        # Step 15: MLA + RA + RA-CT (full attention stack)
        args.enable_mla = True
        args.ra_alpha = 0.3
        args.mlp_attn_gate = False
        args.mlp_cross_token = False
        args.mlp_latent_recip = False
        args.mlp_expansion_ratio = 3.3333333333  # 2560 = 40×64 (GPU-aligned)
        args.ra_cross_token = True
        args.ra_ct_mode = "topk"
        args.ra_ct_apply = "output"
        args.ra_ct_alpha = 0.2
        args.ra_ct_k = 8
    elif step == "16":
        # Step 16: Full RATIO (step 12 + RA-CT, all mechanisms)
        args.enable_mla = True
        args.ra_alpha = 0.3
        args.mlp_attn_gate = True
        args.mlp_cross_token = True
        args.mlp_latent_recip = False
        args.mlp_expansion_ratio = 2.6666666667  # 2048 = 32×64 (GPU-aligned)
        args.ra_cross_token = True
        args.ra_ct_mode = "topk"
        args.ra_ct_apply = "output"
        args.ra_ct_alpha = 0.2
        args.ra_ct_k = 8
    elif step == "17":
        # Step 17: RA-CT weights mode (vs step 13 output mode)
        args.enable_mla = False
        args.ra_alpha = 0.0
        args.mlp_attn_gate = False
        args.mlp_cross_token = False
        args.mlp_latent_recip = False
        args.mlp_expansion_ratio = 4.0  # Standard 3072
        args.ra_cross_token = True
        args.ra_ct_mode = "topk"
        args.ra_ct_apply = "weights"  # Different from step 13
        args.ra_ct_alpha = 0.2
        args.ra_ct_k = 8
    elif step == "18":
        # Step 18: RA-CT entropy mode (adaptive gating)
        args.enable_mla = False
        args.ra_alpha = 0.0
        args.mlp_attn_gate = False
        args.mlp_cross_token = False
        args.mlp_latent_recip = False
        args.mlp_expansion_ratio = 4.0  # Standard 3072
        args.ra_cross_token = True
        args.ra_ct_mode = "entropy"  # Different from step 13
        args.ra_ct_apply = "output"
        args.ra_ct_alpha = 0.2
        args.ra_ct_k = 8  # Not used in entropy mode
    # ==== Lens-Gated Ablation Steps (Simplified Architecture) ====
    elif step == "L0":
        # L0: Baseline (no enhancements)
        args.use_lens = True
        args.ra_alpha = 0.0  # Disable reciprocity
        args.mlp_expansion_ratio = 4.0
        # Will set use_reciprocity=False, use_discoverability=False in patching
    elif step == "L1":
        # L1: Reciprocity only (S^T mixing)
        args.use_lens = True
        args.ra_alpha = 0.3  # Enable reciprocity
        args.mlp_expansion_ratio = 4.0
        # Will set use_reciprocity=True, use_discoverability=False
    elif step == "L2":
        # L2: Discoverability only (column bias)
        args.use_lens = True
        args.ra_alpha = 0.0
        args.mlp_expansion_ratio = 4.0
        # Will set use_reciprocity=False, use_discoverability=True
    elif step == "L3":
        # L3: Reciprocity + Discoverability (lens gates)
        args.use_lens = True
        args.ra_alpha = 0.3
        args.mlp_expansion_ratio = 4.0
        # Will set use_reciprocity=True, use_discoverability=True
    elif step == "L4":
        # L4: Attention-only (MLP disabled)
        args.use_lens = True
        args.ra_alpha = 0.3
        args.mlp_expansion_ratio = 4.0
        # Will set mlp_disabled=True
    elif step == "L5":
        # L5: Full lens (no MLP context)
        args.use_lens = True
        args.ra_alpha = 0.3
        args.mlp_expansion_ratio = 4.0
        # use_reciprocity=True, use_discoverability=True, use_route_gate=True
        # mlp_use_ctx_summary=False
    elif step == "L6":
        # L6: Full lens + low-rank MLP context (R=128) + K/V compression (parameter-neutral)
        args.use_lens = True
        args.ra_alpha = 0.3
        args.mlp_expansion_ratio = 4.0
        args.lens_ctx_rank = 128
        args.lens_kv_compression = True  # Parameter-neutral design
        args.lens_kv_latent_dim = 128
        # use_reciprocity=True, use_discoverability=True, use_route_gate=True
        # mlp_use_ctx_summary=True, mlp_ctx_rank=128, use_kv_compression=True
    elif step == "L7":
        # L7: Full lens + conductor mode + K/V compression (parameter-neutral)
        args.use_lens = True
        args.ra_alpha = 0.3
        args.mlp_expansion_ratio = 4.0
        args.lens_ctx_rank = 128
        args.lens_kv_compression = True  # Parameter-neutral design
        args.lens_kv_latent_dim = 128
        args.lens_ctx_conductor = True
        # mlp_ctx_conductor=True (context only when route_gate < 0.5)
    elif step == "S0":
        # S0: Lens L6 baseline with AdamWSPAM (control for SinkGD experiments)
        # Same architecture as L6, but using AdamWSPAM optimizer for comparison
        args.use_lens = True
        args.ra_alpha = 0.3
        args.mlp_expansion_ratio = 4.0
        args.lens_ctx_rank = 128
        args.lens_kv_compression = True
        args.lens_kv_latent_dim = 128
        args.optimizer = "adamwspam"
        # Control: L6 architecture with AdamWSPAM for SinkGD comparison
    elif step == "S1":
        # S1: Lens L6 + SinkGD default (tau=0.1, n_iter=5)
        # Test SinkGD with balanced entropic smoothing
        args.use_lens = True
        args.ra_alpha = 0.3
        args.mlp_expansion_ratio = 4.0
        args.lens_ctx_rank = 128
        args.lens_kv_compression = True
        args.lens_kv_latent_dim = 128
        args.optimizer = "sinkgd"
        args.sinkgd_tau = 0.1
        args.sinkgd_iters = 5
        # SinkGD: balanced temperature and iterations
    elif step == "S2":
        # S2: Lens L6 + SinkGD sharper (tau=0.05, n_iter=10)
        # Test sharper Sinkhorn smoothing with more iterations
        args.use_lens = True
        args.ra_alpha = 0.3
        args.mlp_expansion_ratio = 4.0
        args.lens_ctx_rank = 128
        args.lens_kv_compression = True
        args.lens_kv_latent_dim = 128
        args.optimizer = "sinkgd"
        args.sinkgd_tau = 0.05
        args.sinkgd_iters = 10
        # SinkGD: sharper (lower tau), more iterations
    elif step == "S3":
        # S3: Lens L6 + SinkGD softer (tau=0.2, n_iter=3)
        # Test softer Sinkhorn smoothing with fewer iterations
        args.use_lens = True
        args.ra_alpha = 0.3
        args.mlp_expansion_ratio = 4.0
        args.lens_ctx_rank = 128
        args.lens_kv_compression = True
        args.lens_kv_latent_dim = 128
        args.optimizer = "sinkgd"
        args.sinkgd_tau = 0.2
        args.sinkgd_iters = 3
        # SinkGD: softer (higher tau), fewer iterations
    elif step == "R0":
        # R0: Standard GPT-2 baseline (control for RWR experiments)
        # No RWR, no RA, no MLA - pure baseline
        args.use_lens = False
        args.use_rwr = False
        args.enable_mla = False
        args.ra_alpha = 0.0
        args.mlp_attn_gate = False
        args.mlp_cross_token = False
        args.ra_cross_token = False
        args.mlp_expansion_ratio = 4.0
        # Control: standard attention for RWR comparison
    elif step == "R1":
        # R1: RWR attention default (α=0.2, T=4, topk=32)
        # Test basic RWR with local + long-range factorization
        args.use_lens = False
        args.use_rwr = True
        args.enable_mla = False
        args.ra_alpha = 0.0
        args.mlp_attn_gate = False
        args.mlp_cross_token = False
        args.ra_cross_token = False
        args.mlp_expansion_ratio = 4.0
        args.rwr_alpha = 0.2
        args.rwr_steps = 4
        args.rwr_topk = 32
        args.rwr_window = 128
        args.rwr_reversible = False
        args.rwr_reciprocal_beta = 0.5  # Not used unless reversible
        args.rwr_lens_strength = 0.3
        # RWR: basic random walk with restart
    elif step == "R2":
        # R2: RWR + reversible chain (detailed balance)
        # Enable P_rev for symmetric walks
        args.use_lens = False
        args.use_rwr = True
        args.enable_mla = False
        args.ra_alpha = 0.0
        args.mlp_attn_gate = False
        args.mlp_cross_token = False
        args.ra_cross_token = False
        args.mlp_expansion_ratio = 4.0
        args.rwr_alpha = 0.2
        args.rwr_steps = 4
        args.rwr_topk = 32
        args.rwr_window = 128
        args.rwr_reversible = True
        args.rwr_reciprocal_beta = 0.5
        args.rwr_lens_strength = 0.3
        # RWR: reversible Markov chain (detailed balance)
    elif step == "R3":
        # R3: RWR + reversible + reciprocal (full bidirectional)
        # Enable P_rev + forward/backward saliency mixing
        args.use_lens = False
        args.use_rwr = True
        args.enable_mla = False
        args.ra_alpha = 0.0
        args.mlp_attn_gate = False
        args.mlp_cross_token = False
        args.ra_cross_token = False
        args.mlp_expansion_ratio = 4.0
        args.rwr_alpha = 0.2
        args.rwr_steps = 4
        args.rwr_topk = 32
        args.rwr_window = 128
        args.rwr_reversible = True
        args.rwr_reciprocal_beta = 0.7  # Higher weight on forward
        args.rwr_lens_strength = 0.3
        args.rwr_use_discoverability = True  # Enable column bias
        # RWR: full lens (reversible + reciprocal + discoverability)
    # ==== Unified RA Ablation Steps ====
    elif step == "V0":
        # V0: Baseline GPT-2 (standard SDPA, for Unified RA comparison)
        args.use_ra_v5 = False
        args.enable_mla = False
        args.ra_alpha = 0.0
        args.mlp_expansion_ratio = 4.0
        # Control baseline for Unified RA experiments
    elif step == "V1":
        # V1: Unified RA (direct folded layout, R=4, matches baseline speed)
        args.use_ra_v5 = True
        args.ra_v5_R = 4  # Validated optimal R value
        args.ra_v5_use_self_restart = False  # Disable self-restart for V1
        args.enable_mla = False
        args.ra_alpha = 0.0  # Unified RA doesn't use legacy RA
        args.mlp_expansion_ratio = 4.0
        # Unified RA: direct folded layout, learned per-head gates, zero overhead
    elif step == "V2":
        # V2: Unified RA + Self-Restart (identity residual path)
        args.use_ra_v5 = True
        args.ra_v5_R = 4
        args.ra_v5_use_self_restart = True  # Enable self-restart mechanism
        args.enable_mla = False
        args.ra_alpha = 0.0
        args.mlp_expansion_ratio = 4.0
        # Unified RA + Self-Restart: adds (1-α)*attn + α*V identity path
        # Per-head learnable α (init 0.05, clamped [0, 0.5])
        # Tests: Does identity residual improve training stability/quality?
    elif step == "V3":
        # V3: Unified RA + R-MLP (basic)
        # Tests reciprocal MLP folding without optional features
        args.use_ra_v5 = True
        args.ra_v5_R = 4
        args.ra_v5_use_self_restart = False
        args.use_rmlp = True  # Enable R-MLP
        args.rmlp_R_ff = 64  # Low-rank reciprocal dimension
        args.rmlp_use_mixer = False
        args.rmlp_use_gates = False
        args.rmlp_tie_up_low = False
        args.enable_mla = False
        args.ra_alpha = 0.0
        args.mlp_expansion_ratio = 4.0
    elif step == "V4":
        # V4: Unified RA + R-MLP + Mixer
        # Tests 1x1 linear mixer on h_low for enhanced expressivity
        args.use_ra_v5 = True
        args.ra_v5_R = 4
        args.ra_v5_use_self_restart = False
        args.use_rmlp = True
        args.rmlp_R_ff = 64
        args.rmlp_use_mixer = True  # Enable mixer
        args.rmlp_use_gates = False
        args.rmlp_tie_up_low = False
        args.enable_mla = False
        args.ra_alpha = 0.0
        args.mlp_expansion_ratio = 4.0
    elif step == "V5":
        # V5: Unified RA + R-MLP + Per-token gates
        # Tests learned discoverability gates
        args.use_ra_v5 = True
        args.ra_v5_R = 4
        args.ra_v5_use_self_restart = False
        args.use_rmlp = True
        args.rmlp_R_ff = 64
        args.rmlp_use_mixer = False
        args.rmlp_use_gates = True  # Enable per-token gates
        args.rmlp_tie_up_low = False
        args.enable_mla = False
        args.ra_alpha = 0.0
        args.mlp_expansion_ratio = 4.0
    elif step == "V6":
        # V6: Unified RA + R-MLP + Mixer + Gates
        # Tests composition of all R-MLP mechanisms
        args.use_ra_v5 = True
        args.ra_v5_R = 4
        args.ra_v5_use_self_restart = False
        args.use_rmlp = True
        args.rmlp_R_ff = 64
        args.rmlp_use_mixer = True  # Enable mixer
        args.rmlp_use_gates = True  # Enable per-token gates
        args.rmlp_tie_up_low = False
        args.enable_mla = False
        args.ra_alpha = 0.0
        args.mlp_expansion_ratio = 4.0
    elif step == "V7":
        # V7: Unified RA with R=8 (higher reciprocal rank)
        # Tests if doubling reciprocal capacity improves quality
        args.use_ra_v5 = True
        args.ra_v5_R = 8  # Double the reciprocal rank
        args.ra_v5_use_self_restart = False
        args.use_rmlp = False
        args.enable_mla = False
        args.ra_alpha = 0.0
        args.mlp_expansion_ratio = 4.0
    elif step == "V8":
        # V8: Unified RA with R=8 + Self-Restart
        # Tests high reciprocal capacity with identity stabilization
        args.use_ra_v5 = True
        args.ra_v5_R = 8
        args.ra_v5_use_self_restart = True  # Enable self-restart
        args.use_rmlp = False
        args.enable_mla = False
        args.ra_alpha = 0.0
        args.mlp_expansion_ratio = 4.0
    elif step == "V9":
        # V9: Unified RA with R=2 (minimal reciprocal rank)
        # Tests if minimal reciprocal capacity is sufficient
        args.use_ra_v5 = True
        args.ra_v5_R = 2  # Minimal reciprocal rank
        args.ra_v5_use_self_restart = False
        args.use_rmlp = False
        args.enable_mla = False
        args.ra_alpha = 0.0
        args.mlp_expansion_ratio = 4.0
    elif step == "V10":
        # V10: Unified RA (R=4) + Self-Restart + 6x MLP expansion
        # Tests if wider MLP compensates for attention modifications
        args.use_ra_v5 = True
        args.ra_v5_R = 4
        args.ra_v5_use_self_restart = True
        args.use_rmlp = False
        args.enable_mla = False
        args.ra_alpha = 0.0
        args.mlp_expansion_ratio = 6.0  # 50% wider MLP
    else:
        raise ValueError(
            f"Invalid ablation step: {step}. Must be 0-18, L0-L7, S0-S3, R0-R3, or V0-V10."
        )

# Override RA+MLA config from config.py if available (for test matrix integration)
# IMPORTANT: Skip config overrides when running ablation study - ablation step has full control
if args.ra_mla_ablation_step is None:
    try:
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(parent_dir, "config.py")
        if os.path.exists(config_path):
            import importlib.util

            spec = importlib.util.spec_from_file_location("config", config_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)

            if hasattr(config_module, "config"):
                cfg = config_module.config
                # Override RA+MLA parameters from config if they exist
                if hasattr(cfg, "RA_MLA_LATENT_DIM"):
                    args.latent_dim = int(cfg.RA_MLA_LATENT_DIM)
                if hasattr(cfg, "RA_MLA_RA_WINDOW"):
                    args.ra_window = int(cfg.RA_MLA_RA_WINDOW)
                if hasattr(cfg, "RA_MLA_RA_ALPHA"):
                    args.ra_alpha = float(cfg.RA_MLA_RA_ALPHA)
                if hasattr(cfg, "RA_MLA_PER_HEAD_Q_LATENT"):
                    args.per_head_q_latent = (
                        cfg.RA_MLA_PER_HEAD_Q_LATENT == "y"
                        or cfg.RA_MLA_PER_HEAD_Q_LATENT is True
                    )
                if hasattr(cfg, "RA_MLA_PER_HEAD_V_UP"):
                    args.per_head_v_up = (
                        cfg.RA_MLA_PER_HEAD_V_UP == "y"
                        or cfg.RA_MLA_PER_HEAD_V_UP is True
                    )
                if hasattr(cfg, "RA_MLA_USE_FLASH"):
                    args.use_flash = (
                        cfg.RA_MLA_USE_FLASH == "y" or cfg.RA_MLA_USE_FLASH is True
                    )
                if hasattr(cfg, "RA_MLA_LOG_METRICS"):
                    args.log_metrics = (
                        cfg.RA_MLA_LOG_METRICS == "y" or cfg.RA_MLA_LOG_METRICS is True
                    )
                # Override training parameters only if not set via environment variable
                # (test matrix passes GPT2_MAX_ITERS via environment which overrides config)
                if hasattr(cfg, "GPT2_MAX_ITERS") and not os.environ.get(
                    "GPT2_MAX_ITERS"
                ):
                    args.max_iters = int(cfg.GPT2_MAX_ITERS)
    except Exception as e:
        # If config.py doesn't exist or can't be loaded, just use command line args
        pass


# ============================================================================
# Data Loading
# ============================================================================


def get_batch(
    split: str, batch_size: int, block_size: int, device: str, data_dir: str = "data"
):
    """Load a batch of data."""
    # This is a placeholder - you should replace with actual data loading
    # For now, generate random data for testing
    data_path = os.path.join(data_dir, args.dataset)

    if not os.path.exists(data_path):
        print(
            f"Warning: Data path {data_path} not found, using random data for testing"
        )
        # Generate random tokens
        x = torch.randint(0, 50257, (batch_size, block_size), device=device)
        y = torch.randint(0, 50257, (batch_size, block_size), device=device)
        return x, y

    # Load actual data (implement based on your data format)
    # For FineWebEdu, you'd load from preprocessed .bin files
    train_data = np.memmap(
        os.path.join(data_path, "train.bin"), dtype=np.uint16, mode="r"
    )
    val_data = np.memmap(os.path.join(data_path, "val.bin"), dtype=np.uint16, mode="r")

    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64))
            for i in ix
        ]
    )

    if device != "cpu":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )

    return x, y


# ============================================================================
# Metrics Logging
# ============================================================================


def set_metrics_computation(model: GPT, enabled: bool):
    """Enable/disable expensive metrics computation (entropy, reciprocity)."""
    actual_model = model.module if hasattr(model, "module") else model
    for block in actual_model.transformer.h:
        if hasattr(block.attn, "core"):  # RA_MLA_Attention
            block.attn.core.enable_metrics_computation = enabled


class RAMLAMetrics:
    """Track and log RA+MLA specific metrics."""

    def __init__(self):
        self.attention_entropy = []
        self.reciprocity_score = []
        self.forward_time = []
        self.memory_allocated = []
        self.memory_reserved = []
        self.iteration = []
        # Evaluation metrics (logged at eval_interval)
        self.eval_iteration = []
        self.train_loss = []
        self.val_loss = []
        self.train_perplexity = []
        self.val_perplexity = []
        self.learning_rate = []

    def log(self, iter: int, model: GPT, forward_time_ms: float):
        """Log metrics from the model."""
        # Collect attention entropy and reciprocity from all layers
        entropy_vals = []
        reciprocity_vals = []

        # Unwrap DDP model if necessary
        actual_model = model.module if hasattr(model, "module") else model

        for block in actual_model.transformer.h:
            if hasattr(block.attn, "core"):  # RA_MLA_Attention
                attn = block.attn.core
                if attn.attention_entropy is not None:
                    entropy_vals.append(attn.attention_entropy)
                if attn.reciprocity_score is not None:
                    reciprocity_vals.append(attn.reciprocity_score)

        # Average across layers
        if entropy_vals:
            self.attention_entropy.append(np.mean(entropy_vals))
        if reciprocity_vals:
            self.reciprocity_score.append(np.mean(reciprocity_vals))

        self.forward_time.append(forward_time_ms)
        if torch.cuda.is_available():
            # Sum memory across all GPUs for DDP
            total_allocated = 0
            total_reserved = 0
            for i in range(torch.cuda.device_count()):
                total_allocated += torch.cuda.memory_allocated(i) / 1024**2
                total_reserved += torch.cuda.memory_reserved(i) / 1024**2
            self.memory_allocated.append(total_allocated)
            self.memory_reserved.append(total_reserved)
        else:
            self.memory_allocated.append(0)
            self.memory_reserved.append(0)
        self.iteration.append(iter)

    def log_eval(
        self,
        iter: int,
        train_loss: float,
        val_loss: float,
        train_perplexity: float,
        val_perplexity: float,
        learning_rate: float,
    ):
        """Log evaluation metrics (called at eval_interval)."""
        self.eval_iteration.append(iter)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_perplexity.append(train_perplexity)
        self.val_perplexity.append(val_perplexity)
        self.learning_rate.append(learning_rate)

    def save(self, path: str):
        """Save metrics to JSON."""
        data = {
            "iteration": self.iteration,
            "attention_entropy": self.attention_entropy,
            "reciprocity_score": self.reciprocity_score,
            "forward_time_ms": self.forward_time,
            "memory_allocated_mb": self.memory_allocated,
            "memory_reserved_mb": self.memory_reserved,
        }
        # Add evaluation metrics if available
        if self.eval_iteration:
            data["eval_iteration"] = self.eval_iteration
            data["train_loss"] = self.train_loss
            data["val_loss"] = self.val_loss
            data["train_perplexity"] = self.train_perplexity
            data["val_perplexity"] = self.val_perplexity
            data["learning_rate"] = self.learning_rate
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved metrics to {path}")


# ============================================================================
# Learning Rate Schedule
# ============================================================================


def get_lr(
    iter: int, warmup_iters: int, max_iters: int, lr: float, min_lr: float
) -> float:
    """Cosine learning rate schedule with warmup."""
    # Linear warmup
    if iter < warmup_iters:
        return lr * iter / warmup_iters
    # Cosine decay
    if iter > max_iters:
        return min_lr
    decay_ratio = (iter - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


# ============================================================================
# Evaluation
# ============================================================================


@torch.no_grad()
def estimate_loss(
    model, eval_samples: int, batch_size: int, block_size: int, device: str
):
    """Estimate loss on train and val sets."""
    model.eval()
    losses = {}

    for split in ["train", "val"]:
        batch_losses = []
        for _ in range(eval_samples):
            x, y = get_batch(split, batch_size, block_size, device, args.data_dir)
            logits, loss = model(x, y)
            batch_losses.append(loss.item())
        losses[split] = np.mean(batch_losses)

    model.train()
    return losses


def analyze_unified_ra_gates(model) -> Dict[str, float]:
    """
    Analyze Unified RA gate values (w_std, w_rec, rwr_alpha) across all layers.

    Unified RA gates control per-head blending of standard and
    reciprocal attention:
    - w_std: Standard attention weight (Q_std @ K_std^T)
    - w_rec: Reciprocal attention weight (K_low @ Q_low^T)
    - rwr_alpha: Self-restart weight (optional, if use_self_restart=True)

    Returns:
        dict with gate statistics (mean across all heads and layers)
    """
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from unified_ra import UnifiedRAttention

    w_std_list = []
    w_rec_list = []
    rwr_alpha_list = []

    for name, module in model.named_modules():
        if isinstance(module, UnifiedRAttention):
            # Gates are learnable parameters or buffers
            with torch.no_grad():
                # Get raw gate values (not necessarily normalized)
                w_std = module.w_std.cpu()
                w_rec = module.w_rec.cpu()
                w_std_list.extend(w_std.tolist())
                w_rec_list.extend(w_rec.tolist())

                # Get rwr_alpha if self-restart is enabled
                if (
                    hasattr(module, "use_self_restart")
                    and module.use_self_restart
                    and module.rwr_alpha is not None
                ):
                    # Clamp to [0, 0.5] like in forward pass
                    alpha = torch.clamp(module.rwr_alpha, 0.0, 0.5).cpu()
                    rwr_alpha_list.extend(alpha.tolist())

    if not w_std_list:
        return {}

    # Compute statistics
    w_std_tensor = torch.tensor(w_std_list)
    w_rec_tensor = torch.tensor(w_rec_list)

    stats = {
        # Standard attention weights
        "unified_ra_w_std_mean": w_std_tensor.mean().item(),
        "unified_ra_w_std_std": w_std_tensor.std().item(),
        "unified_ra_w_std_min": w_std_tensor.min().item(),
        "unified_ra_w_std_max": w_std_tensor.max().item(),
        # Reciprocal attention weights
        "unified_ra_w_rec_mean": w_rec_tensor.mean().item(),
        "unified_ra_w_rec_std": w_rec_tensor.std().item(),
        "unified_ra_w_rec_min": w_rec_tensor.min().item(),
        "unified_ra_w_rec_max": w_rec_tensor.max().item(),
    }

    # Add self-restart statistics if enabled
    if rwr_alpha_list:
        rwr_alpha_tensor = torch.tensor(rwr_alpha_list)
        stats.update(
            {
                "unified_ra_rwr_alpha_mean": rwr_alpha_tensor.mean().item(),
                "unified_ra_rwr_alpha_std": rwr_alpha_tensor.std().item(),
                "unified_ra_rwr_alpha_min": rwr_alpha_tensor.min().item(),
                "unified_ra_rwr_alpha_max": rwr_alpha_tensor.max().item(),
            }
        )

    return stats


# ============================================================================
# Main Training Loop
# ============================================================================


def main():
    # Setup
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Dry-run mode: force CPU to avoid GPU allocation
    if args.dry_run:
        print("=" * 60)
        print("DRY-RUN MODE: Architecture Validation")
        print("=" * 60)
        print("Testing: model creation, forward pass, backward pass")
        print("Device: CPU (forced)")
        print("=" * 60)
        args.device = "cpu"

    # Validate device - handle CUDA/ROCm availability
    device = args.device
    if device == "cuda":
        if not torch.cuda.is_available():
            print("WARNING: CUDA/ROCm not available - falling back to CPU")
            print("For AMD GPUs (W7900), install PyTorch with ROCm support:")
            print(
                "  pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1"
            )
            device = "cpu"
        else:
            print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

            # Enable TensorFloat32 for matmul if supported (WMMA on AMD, Tensor Cores on NVIDIA)
            if supports_tensorcore_fp32():
                torch.set_float32_matmul_precision("high")
                print("  Enabled TensorFloat32 matmul precision (WMMA/Tensor Cores)")
            else:
                print("  Tensor cores/WMMA not detected, using default precision")
    else:
        print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # DDP setup (if enabled)
    ddp = False
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    seed_offset = 0

    # Check if DDP is enabled via config
    try:
        import config as cfg

        if hasattr(cfg, "config") and hasattr(cfg.config, "GPT2_USE_DDP"):
            gpt2_use_ddp = getattr(cfg.config, "GPT2_USE_DDP", False)
            use_ddp = gpt2_use_ddp is True or gpt2_use_ddp == "y"
            ddp_backend = getattr(cfg.config, "GPT2_DDP_BACKEND", "nccl")
            ddp_find_unused_param = getattr(
                cfg.config, "GPT2_DDP_FIND_UNUSED_PARAMS", "y"
            )
            ddp_find_unused = (
                ddp_find_unused_param is True or ddp_find_unused_param == "y"
            )
        else:
            use_ddp = False
            ddp_backend = "nccl"
            ddp_find_unused = True
    except (ImportError, AttributeError):
        use_ddp = False
        ddp_backend = "nccl"
        ddp_find_unused = True

    # Initialize DDP if enabled and environment variables are set
    if use_ddp and "RANK" in os.environ:
        ddp = True
        init_process_group(backend=ddp_backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        print(
            f"DDP initialized: rank {ddp_rank}/{ddp_world_size}, local rank {ddp_local_rank}, device {device}",
            flush=True,
        )
    else:
        master_process = True
        seed_offset = 0
        if use_ddp:
            print(
                "DDP enabled in config but RANK environment variable not set. Running in single GPU mode.",
                flush=True,
            )

    # -------------------------------------------------------------------------
    # Configure torch.compile() for better performance
    if args.compile:
        # Allow .item() calls in compiled graphs (for metrics logging)
        torch._dynamo.config.capture_scalar_outputs = True
        # Increase recompilation limit to handle attention variations
        torch._dynamo.config.cache_size_limit = 128
        print("  torch.compile enabled with scalar output capture")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # Create model
    print(f"Creating GPT-2 model: {args.model_name}")
    model_config = GPTConfig.from_name(args.model_name)
    model_config.block_size = args.block_size
    model_config.dropout = args.dropout
    model = GPT(model_config)

    # Determine which architecture to use
    use_lens = getattr(args, "use_lens", False)

    # Apply lens-gated OR legacy RA+MLA patch
    if use_lens:
        # === Lens-Gated Architecture (Simplified, Parameter-Efficient) ===
        # Determine mechanism configuration based on ablation step
        step = args.ra_mla_ablation_step
        if step == "L0":
            # Baseline (no enhancements)
            use_reciprocity = False
            use_discoverability = False
            use_route_gate = False
            mlp_use_ctx_summary = False
            mlp_disabled = False
        elif step == "L1":
            # Reciprocity only
            use_reciprocity = True
            use_discoverability = False
            use_route_gate = False
            mlp_use_ctx_summary = False
            mlp_disabled = False
        elif step == "L2":
            # Discoverability only
            use_reciprocity = False
            use_discoverability = True
            use_route_gate = False
            mlp_use_ctx_summary = False
            mlp_disabled = False
        elif step == "L3":
            # Reciprocity + Discoverability
            use_reciprocity = True
            use_discoverability = True
            use_route_gate = False
            mlp_use_ctx_summary = False
            mlp_disabled = False
        elif step == "L4":
            # Attention-only (MLP disabled)
            use_reciprocity = True
            use_discoverability = True
            use_route_gate = False
            mlp_use_ctx_summary = False
            mlp_disabled = True
        elif step in ("L5", "L6", "L7"):
            # Full lens with route gate
            use_reciprocity = True
            use_discoverability = True
            use_route_gate = True
            mlp_use_ctx_summary = step in ("L6", "L7")  # Context for L6, L7
            mlp_disabled = False
        else:
            # Default: all features
            use_reciprocity = args.ra_alpha > 0.0
            use_discoverability = True
            use_route_gate = True
            mlp_use_ctx_summary = True
            mlp_disabled = False

        print("=" * 70)
        print("Applying Lens-Gated Architecture:")
        print(f"  Reciprocity (S^T):     {use_reciprocity}")
        print(f"  Discoverability (d):   {use_discoverability}")
        print(f"  K/V Compression:       {args.lens_kv_compression}")
        if args.lens_kv_compression:
            print(f"    Latent dimension:    R={args.lens_kv_latent_dim}")
        print(f"  Route Gate:            {use_route_gate}")
        print(f"  MLP Context Summary:   {mlp_use_ctx_summary}")
        if mlp_use_ctx_summary:
            print(f"    Low-rank bottleneck: R={args.lens_ctx_rank}")
            print(f"    Conductor mode:      {args.lens_ctx_conductor}")
        print(f"  MLP Disabled:          {mlp_disabled}")
        print(f"  MLP Expansion Ratio:   {args.mlp_expansion_ratio}")
        print("=" * 70)

        model, lens_cfg = patch_gpt2_with_lens_attention(
            model,
            use_reciprocity=use_reciprocity,
            use_discoverability=use_discoverability,
            use_kv_compression=args.lens_kv_compression,
            kv_latent_dim=args.lens_kv_latent_dim,
            use_route_gate=use_route_gate,
            mlp_use_ctx_summary=mlp_use_ctx_summary,
            mlp_ctx_detach=True,
            mlp_expansion_ratio=args.mlp_expansion_ratio,
            mlp_ctx_rank=args.lens_ctx_rank,
            mlp_ctx_conductor=args.lens_ctx_conductor,
            mlp_disabled=mlp_disabled,
            log_attention_entropy=args.log_metrics,
        )
        ra_cfg = lens_cfg  # Alias for compatibility

    elif getattr(args, "use_ra_v5", False) and (
        args.enable_mla
        or args.mlp_attn_gate
        or args.mlp_cross_token
        or args.mlp_latent_recip
    ):
        # === Unified RA + R-MLP (Combined) ===
        use_self_restart = getattr(args, "ra_v5_use_self_restart", False)
        restart_str = " + Self-Restart" if use_self_restart else ""

        print("=" * 70)
        print(f"Applying Unified RA{restart_str} + R-MLP:")
        print(f"  R value:              {getattr(args, 'ra_v5_R', 4)}")
        print(f"  Architecture:         Direct folded Q/K emission")
        print(f"  Learned gates:        Per-head w_std, w_rec")
        if use_self_restart:
            print(f"  Self-restart:         Enabled (α init=0.05, clamped [0, 0.5])")
        print(f"  Performance:          Matches baseline SDPA (1.33ms)")
        if args.mlp_attn_gate or args.mlp_cross_token or args.mlp_latent_recip:
            print("R-MLP mechanisms:")
            if args.mlp_attn_gate:
                print(f"  [1] MLP-to-Attention Gating: α={args.mlp_gate_alpha}")
            if args.mlp_cross_token:
                print(f"  [2] Cross-Token MLP Aggregation: α={args.mlp_cross_alpha}")
            if args.mlp_latent_recip:
                print(f"  [3] MLP Latent Reciprocity: α={args.mlp_recip_alpha}")
        print("=" * 70)

        # Import Unified RA patching function
        from ra_v5_patch import patch_gpt2_with_ra_v5

        # First patch with Unified RA (attention only)
        model = patch_gpt2_with_ra_v5(
            model,
            R=getattr(args, "ra_v5_R", 4),
            dropout=args.dropout,
            use_self_restart=use_self_restart,
        )

        # Then patch MLP with R-MLP mechanisms
        from ra_mla_gpt2 import ReciprocalMLP, RA_MLA_Block, RA_MLA_Config

        # Create R-MLP config
        mlp_cfg = RA_MLA_Config(
            mlp_attn_gate=args.mlp_attn_gate,
            mlp_cross_token=args.mlp_cross_token,
            mlp_latent_recip=args.mlp_latent_recip,
            mlp_gate_alpha=args.mlp_gate_alpha,
            mlp_cross_alpha=args.mlp_cross_alpha,
            mlp_recip_alpha=args.mlp_recip_alpha,
            mlp_gate_dim=args.mlp_gate_dim,
            mlp_latent_dim=args.mlp_latent_dim,
            mlp_expansion_ratio=args.mlp_expansion_ratio,
            mlp_tying_mode=args.mlp_tying_mode,
            mlp_sparse_mode=args.mlp_sparse_mode,
            mlp_sparse_k=args.mlp_sparse_k,
            mlp_sparse_tau=args.mlp_sparse_tau,
            mlp_sparse_normalize=args.mlp_sparse_normalize,
            mlp_sparse_head_average=args.mlp_sparse_head_average,
            latent_dim=args.latent_dim,
            resid_dropout=args.dropout,
        )

        # Replace MLPs with ReciprocalMLP and wrap blocks for context flow
        n_embd = model.config.n_embd
        n_head = model.config.n_head
        for i, block in enumerate(model.transformer.h):
            # Create ReciprocalMLP to replace standard MLP
            reciprocal_mlp = ReciprocalMLP(n_embd=n_embd, n_head=n_head, cfg=mlp_cfg)

            # Wrap block with RA_MLA_Block for cross-layer context flow
            # (MLP→Attention in next layer)
            block_wrapper = RA_MLA_Block(block.attn, reciprocal_mlp, block)
            model.transformer.h[i] = block_wrapper

            print(f"  Layer {i}: Standard MLP → R-MLP")

        print(f"Successfully patched {len(model.transformer.h)} layers with R-MLP")
        ra_cfg = mlp_cfg

    elif getattr(args, "use_ra_v5", False) and getattr(args, "use_rmlp", False):
        # === Unified RA + R-MLP ===
        use_self_restart = getattr(args, "ra_v5_use_self_restart", False)
        restart_str = " + Self-Restart" if use_self_restart else ""

        print("=" * 70)
        print(f"Applying Unified RA{restart_str} + R-MLP:")
        print(f"  Attention R value:    {getattr(args, 'ra_v5_R', 4)}")
        print(f"  Architecture:         Direct folded Q/K emission")
        print(f"  Learned gates:        Per-head w_std, w_rec")
        if use_self_restart:
            print(f"  Self-restart:         Enabled (α init=0.05, clamped [0, 0.5])")
        print(f"  MLP R_ff value:       {getattr(args, 'rmlp_R_ff', 64)}")
        print(f"  MLP expansion:        {args.mlp_expansion_ratio}x")

        rmlp_features = []
        if getattr(args, "rmlp_use_mixer", False):
            rmlp_features.append("mixer")
        if getattr(args, "rmlp_use_gates", False):
            rmlp_features.append("per-token gates")
        if getattr(args, "rmlp_tie_up_low", False):
            rmlp_features.append("weight tying")

        if rmlp_features:
            print(f"  R-MLP features:       {', '.join(rmlp_features)}")
        print("=" * 70)

        # Import patching functions
        from ra_v5_patch import patch_gpt2_with_unified_ra_and_rmlp

        model = patch_gpt2_with_unified_ra_and_rmlp(
            model,
            R=getattr(args, "ra_v5_R", 4),
            attn_dropout=args.dropout,
            use_self_restart=use_self_restart,
            mlp_expansion=args.mlp_expansion_ratio,
            R_ff=getattr(args, "rmlp_R_ff", 64),
            mlp_dropout=args.dropout,
            use_mixer=getattr(args, "rmlp_use_mixer", False),
            use_gates=getattr(args, "rmlp_use_gates", False),
            tie_up_low=getattr(args, "rmlp_tie_up_low", False),
        )

        ra_cfg = {
            "R": getattr(args, "ra_v5_R", 4),
            "use_self_restart": use_self_restart,
            "R_ff": getattr(args, "rmlp_R_ff", 64),
            "use_mixer": getattr(args, "rmlp_use_mixer", False),
            "use_gates": getattr(args, "rmlp_use_gates", False),
            "tie_up_low": getattr(args, "rmlp_tie_up_low", False),
        }

    elif getattr(args, "use_ra_v5", False):
        # === Unified RA only (no R-MLP) ===
        use_self_restart = getattr(args, "ra_v5_use_self_restart", False)
        restart_str = " + Self-Restart" if use_self_restart else ""

        print("=" * 70)
        print(f"Applying Unified RA{restart_str}:")
        print(f"  R value:              {getattr(args, 'ra_v5_R', 4)}")
        print(f"  Architecture:         Direct folded Q/K emission")
        print(f"  Learned gates:        Per-head w_std, w_rec")
        if use_self_restart:
            print(f"  Self-restart:         Enabled (α init=0.05, clamped [0, 0.5])")
        print(f"  Performance:          Matches baseline SDPA (1.33ms)")
        print("=" * 70)

        # Import Unified RA patching function
        from ra_v5_patch import patch_gpt2_with_ra_v5

        model = patch_gpt2_with_ra_v5(
            model,
            R=getattr(args, "ra_v5_R", 4),
            dropout=args.dropout,
            use_self_restart=use_self_restart,
        )
        ra_cfg = None  # Unified RA has own gate parameters

    elif getattr(args, "use_rwr", False):
        # === RWR (Random Walk with Restart) Attention ===
        from rwr_attention import patch_gpt2_with_rwr

        print("=" * 70)
        print("Applying RWR Attention:")
        print(f"  Restart probability (α):  {args.rwr_alpha}")
        print(f"  Walk steps (T):           {args.rwr_steps}")
        print(f"  Top-k neighbors:          {args.rwr_topk}")
        print(f"  Local window:             {args.rwr_window}")
        print(f"  Reversible chain:         {args.rwr_reversible}")
        print(f"  Reciprocal beta:          {args.rwr_reciprocal_beta}")
        print(f"  Lens strength (γ):        {args.rwr_lens_strength}")
        print(f"  Discoverability:          {args.rwr_use_discoverability}")
        print("=" * 70)

        model = patch_gpt2_with_rwr(
            model,
            rwr_alpha=args.rwr_alpha,
            rwr_steps=args.rwr_steps,
            rwr_topk=args.rwr_topk,
            rwr_threshold=args.rwr_threshold,
            reversible=args.rwr_reversible,
            reciprocal_beta=args.rwr_reciprocal_beta,
            lens_strength=args.rwr_lens_strength,
            window=args.rwr_window,
            block_size=args.rwr_block_size,
            head_dim_pad=args.rwr_head_dim_pad,
            use_discoverability=args.rwr_use_discoverability,
        )
        ra_cfg = None  # RWR doesn't use RA config

    elif (
        args.enable_mla
        or args.ra_alpha > 0.0
        or args.mlp_attn_gate
        or args.mlp_cross_token
        or args.mlp_latent_recip
        or getattr(args, "ra_cross_token", False)
        or args.mlp_expansion_ratio != 4.0
    ):
        # === Legacy RA+MLA Architecture (Complex) ===
        print(f"Applying RA+MLA patch:")
        print(
            f"  latent_dim={args.latent_dim}, ra_window={args.ra_window}, ra_alpha={args.ra_alpha}"
        )
        print(
            f"  per_head_q_latent={args.per_head_q_latent}, per_head_v_up={args.per_head_v_up}"
        )
        if args.mlp_attn_gate or args.mlp_cross_token or args.mlp_latent_recip:
            print("Reciprocal MLP mechanisms:")
            if args.mlp_attn_gate:
                print(f"  [1] MLP-to-Attention Gating: α={args.mlp_gate_alpha}")
            if args.mlp_cross_token:
                print(f"  [2] Cross-Token MLP Aggregation: α={args.mlp_cross_alpha}")
            if args.mlp_latent_recip:
                print(f"  [3] MLP Latent Reciprocity: α={args.mlp_recip_alpha}")
        if getattr(args, "ra_cross_token", False):
            print("Cross-Token RA (RA-CT) gating:")
            print(
                f"  mode={args.ra_ct_mode}, apply={args.ra_ct_apply}, α={args.ra_ct_alpha}, k={args.ra_ct_k}"
            )

        model, ra_cfg = patch_gpt2_with_ra_mla(
            model,
            latent_dim=args.latent_dim,
            ra_window=args.ra_window,
            ra_alpha=args.ra_alpha,
            per_head_q_latent=args.per_head_q_latent,
            per_head_v_up=args.per_head_v_up,
            use_flash=args.use_flash,
            log_metrics=args.log_metrics,
            # Reciprocal MLP parameters
            mlp_attn_gate=args.mlp_attn_gate,
            mlp_cross_token=args.mlp_cross_token,
            mlp_latent_recip=args.mlp_latent_recip,
            mlp_gate_alpha=args.mlp_gate_alpha,
            mlp_cross_alpha=args.mlp_cross_alpha,
            mlp_recip_alpha=args.mlp_recip_alpha,
            mlp_gate_dim=args.mlp_gate_dim,
            mlp_latent_dim=args.mlp_latent_dim,
            mlp_expansion_ratio=args.mlp_expansion_ratio,
            # Parameter tying and sparsification
            mlp_tying_mode=args.mlp_tying_mode,
            mlp_sparse_mode=args.mlp_sparse_mode,
            mlp_sparse_k=args.mlp_sparse_k,
            mlp_sparse_tau=args.mlp_sparse_tau,
            mlp_sparse_normalize=args.mlp_sparse_normalize,
            mlp_sparse_head_average=args.mlp_sparse_head_average,
            # Cross-Token RA (RA-CT) parameters
            ra_cross_token=getattr(args, "ra_cross_token", False),
            ra_ct_mode=getattr(args, "ra_ct_mode", "topk"),
            ra_ct_apply=getattr(args, "ra_ct_apply", "output"),
            ra_ct_alpha=getattr(args, "ra_ct_alpha", 0.2),
            ra_ct_k=getattr(args, "ra_ct_k", 8),
        )
    else:
        print("Using standard GPT-2 (no RA/MLA patching needed)")
        ra_cfg = None

    model = model.to(device)

    # Wrap model in DDP if enabled
    if ddp:
        model = DDP(
            model, device_ids=[ddp_local_rank], find_unused_parameters=ddp_find_unused
        )

    # Compile model if requested (only compile the base model, not DDP wrapper)
    if args.compile and hasattr(torch, "compile") and not ddp:
        print("Compiling model with torch.compile()...", flush=True)
        model = torch.compile(model)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params/1e6:.2f}M")

    # Create optimizer
    optimizer, scheduler, gradient_clip_norm, spam_state, adamprune_state = (
        create_optimizer(
            model,
            args.optimizer,
            args.learning_rate,
            num_epochs=None,
            args=args,
            model_type="gpt2",
        )
    )

    print(
        f"Optimizer: {args.optimizer}, LR: {args.learning_rate}, weight_decay: {args.weight_decay}"
    )

    # =========================================================================
    # DRY-RUN MODE: Quick architecture validation
    # =========================================================================
    if args.dry_run:
        print("\nRunning architecture validation...")
        print(f"  Model parameters: {model.get_num_params() / 1e6:.2f}M")

        # Create minimal dummy batch (batch_size=2, seq_len=32)
        batch_size = 2
        seq_len = 32
        x = torch.randint(0, 50257, (batch_size, seq_len), device=device)
        y = torch.randint(0, 50257, (batch_size, seq_len), device=device)

        print(f"  Dummy batch: {batch_size}x{seq_len}")

        try:
            # Forward pass
            print("  ✓ Testing forward pass...")
            logits, loss = model(x, y)
            print(f"    Output shape: {logits.shape}, Loss: {loss.item():.4f}")

            # Backward pass
            print("  ✓ Testing backward pass...")
            loss.backward()
            print("    Gradients computed")

            # Optimizer step
            print("  ✓ Testing optimizer step...")
            optimizer.step()
            optimizer.zero_grad()
            print("    Parameters updated")

            # Success
            print("\n" + "=" * 60)
            print("✓ DRY-RUN PASSED: Architecture is valid")
            print("=" * 60)
            sys.exit(0)

        except Exception as e:
            print("\n" + "=" * 60)
            print("✗ DRY-RUN FAILED: Architecture validation error")
            print("=" * 60)
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    # Metrics tracking
    metrics = RAMLAMetrics()

    # Initialize experiment tracking (only on master process for DDP)
    tracker_names = [
        t.strip()
        for t in args.tracker.split(",")
        if t.strip() and t.strip().lower() != "none"
    ]

    if tracker_names and master_process:
        # Generate run name if not provided
        run_name = args.tracker_run_name
        if not run_name:
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"ra_mla_{args.model_name}_{args.optimizer}_L{args.latent_dim}_a{args.ra_alpha}_{timestamp}"

        # Default project name
        project_name = args.tracker_project if args.tracker_project else "gpt2-ra-mla"

        print(f"\nInitializing experiment tracking: {', '.join(tracker_names)}")
        print(f"  Project: {project_name}")
        print(f"  Run: {run_name}")

        if "trackio" in tracker_names:
            try:
                import trackio

                trackio.init(project=project_name, config=vars(args), name=run_name)
                print("  ✓ Trackio initialized")

                # Upload .config file if it exists
                config_file = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    ".config",
                )
                if os.path.exists(config_file):
                    try:
                        # Trackio may have a save method, but if not, we'll just log the path
                        if hasattr(trackio, "save"):
                            trackio.save(config_file)
                            print(f"  ✓ Uploaded .config to Trackio")
                        else:
                            trackio.log({"config_file": config_file})
                            print(f"  ✓ Logged .config path to Trackio")
                    except Exception as e:
                        print(f"  ⚠ Could not upload .config to Trackio: {e}")
            except ImportError:
                print("  ✗ Trackio not available (install with: pip install trackio)")
                tracker_names.remove("trackio")

        if "wandb" in tracker_names:
            try:
                import wandb

                wandb.init(
                    project=project_name,
                    config=vars(args),
                    name=run_name,
                )
                print("  ✓ WandB initialized")

                # Upload .config file if it exists
                config_file = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    ".config",
                )
                if os.path.exists(config_file):
                    wandb.save(config_file, base_path=os.path.dirname(config_file))
                    print(f"  ✓ Uploaded .config to WandB")
            except ImportError:
                print("  ✗ WandB not available (install with: pip install wandb)")
                tracker_names.remove("wandb")

    # Training loop - Print detailed configuration
    print(f"\nStarting training for {args.max_iters} iterations...")
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(
        f"Batch size: {args.batch_size}, gradient accumulation: {args.gradient_accumulation}"
    )
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"Optimizer: {args.optimizer}")

    # Show pruning configuration
    if (
        hasattr(args, "pruning_method")
        and args.pruning_method
        and args.pruning_method != "none"
    ):
        print(
            f"Pruning: {args.pruning_method} @ {float(args.target_sparsity)*100:.0f}% target sparsity"
        )
    else:
        print("Pruning: None")

    # Show AdamWPrune variant if applicable
    if args.optimizer == "adamwprune" and hasattr(args, "adamwprune_variant"):
        variant = getattr(args, "adamwprune_variant", "bitter0")
        print(f"AdamWPrune variant: {variant}")
        if hasattr(args, "adamwprune_base_optimizer_name"):
            base_opt = getattr(args, "adamwprune_base_optimizer_name", "adamw")
            print(f"Base optimizer: {base_opt}")

    # Show RATIO ablation step if specified
    if args.ra_mla_ablation_step is not None:
        step_names = {
            "0": "Baseline GPT-2 (ratio 1:2.0, standard attention)",
            "1": "Baseline + SPAM pruning 50%",
            "2": "Golden ratio 1:2.5 via MLP resize",
            "3": "Step 2 + MLP gating 15%",
            "4": "Step 3 + cross-token 10%",
            "5": "Baseline + RA (ra_alpha=0.3)",
            "6": "RA + golden ratio 1:2.5",
            "7": "Step 6 + mechanisms (RA + ratio + gating + cross-token)",
            "8": "Baseline + MLA (ratio 1:3.0)",
            "9": "MLA + golden ratio 1:2.5",
            "10": "Step 9 + mechanisms (MLA + ratio + mechanisms)",
            "11": "RA + MLA + golden ratio",
            "12": "Step 11 + mechanisms (RA + MLA + ratio + mechanisms)",
            "13": "Step 10 + AdamWStructure",
            "14": "Step 13 + ratio-preserving pruning (Full RATIO)",
        }
        step_desc = step_names.get(
            args.ra_mla_ablation_step, f"Step {args.ra_mla_ablation_step}"
        )
        print(f"RATIO Ablation: Step {args.ra_mla_ablation_step} - {step_desc}")

    # Show reciprocal MLP mechanisms
    if args.mlp_attn_gate or args.mlp_cross_token or args.mlp_latent_recip:
        mechanisms = []
        if args.mlp_attn_gate:
            mechanisms.append("MLP-Attn-Gate")
        if args.mlp_cross_token:
            mechanisms.append("Cross-Token")
        if args.mlp_latent_recip:
            mechanisms.append("Latent-Recip")
        print(f"Reciprocal MLP: {', '.join(mechanisms)}")

    print("=" * 60 + "\n")

    # Show inference scaling law U-curves
    try:
        # Get model configuration parameters from model_config
        n_layers = model_config.n_layer
        n_embd = model_config.n_embd

        # Calculate MLP dimension
        # Standard GPT-2 uses 4x expansion (n_embd -> 4*n_embd)
        mlp_dim = 4 * n_embd

        # Count total parameters
        param_count = sum(p.numel() for p in model.parameters())

        # Display the U-curves
        print(show_scaling_curves(n_layers, n_embd, mlp_dim, param_count))
    except Exception as e:
        # If scaling curves fail, just continue - this is informational only
        print(f"Note: Could not display scaling curves: {e}\n")

    model.train()
    iter_num = 0
    best_val_loss = float("inf")

    # Track training start time for MAX_TIME support
    training_start_time = time.time()

    # Training loop - stops when EITHER max_iters OR max_time is reached
    while iter_num < args.max_iters:
        # Check time limit if MAX_TIME is set
        if args.max_time is not None:
            elapsed_time = time.time() - training_start_time
            if elapsed_time >= args.max_time:
                if master_process:
                    print(f"\nReached time limit: {elapsed_time/3600:.2f} hours")
                    print(f"Completed {iter_num} iterations")
                break
        # Update learning rate
        lr = get_lr(
            iter_num, args.warmup_steps, args.max_iters, args.learning_rate, args.min_lr
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Evaluation
        if iter_num % args.eval_interval == 0 or iter_num == args.max_iters - 1:
            losses = estimate_loss(
                model, args.eval_samples, args.batch_size, args.block_size, device
            )
            if master_process:
                # Calculate perplexity (exp of loss)
                train_perplexity = math.exp(losses["train"])
                val_perplexity = math.exp(losses["val"])

                # Log to RAMLAMetrics
                if args.log_metrics:
                    metrics.log_eval(
                        iter_num,
                        losses["train"],
                        losses["val"],
                        train_perplexity,
                        val_perplexity,
                        lr,
                    )

                # Build validation output string
                val_str = f"Iter {iter_num:6d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | lr {lr:.2e}"

                # Add route gate if lens-gated
                if use_lens and use_route_gate:
                    mean_g = get_mean_route_gate(model.module if ddp else model)
                    val_str += f" | route_gate {mean_g:.3f}"

                # Add lens gates if using lens architecture
                if use_lens:
                    lens_stats = analyze_lens_gates(model.module if ddp else model)
                    if lens_stats:
                        val_str += f" | w_std {lens_stats['w_std_mean']:.3f}"
                        val_str += f" | w_rec {lens_stats['w_rec_mean']:.3f}"
                        val_str += f" | w_disc {lens_stats['w_disc_mean']:.3f}"

                # Add Unified RA gates if using Unified RA
                if getattr(args, "use_ra_v5", False):
                    unified_ra_stats = analyze_unified_ra_gates(
                        model.module if ddp else model
                    )
                    if unified_ra_stats:
                        val_str += f" | UA_w_std {unified_ra_stats['unified_ra_w_std_mean']:.3f}"
                        val_str += f" | UA_w_rec {unified_ra_stats['unified_ra_w_rec_mean']:.3f}"

                print(val_str)

            # Log evaluation metrics to trackers (only master process)
            if tracker_names and master_process:
                # Perplexity already calculated above (reuse those values)

                eval_metrics = {
                    "iteration": iter_num,
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                    "train_perplexity": train_perplexity,
                    "val_perplexity": val_perplexity,
                    "best_val_loss": best_val_loss,
                    "learning_rate": lr,
                }

                # Add gate metrics to evaluation logging
                if use_lens and use_route_gate:
                    mean_g = get_mean_route_gate(model.module if ddp else model)
                    eval_metrics["route_gate_mean"] = mean_g

                if use_lens:
                    lens_stats = analyze_lens_gates(model.module if ddp else model)
                    if lens_stats:
                        eval_metrics.update(lens_stats)

                # Add Unified RA gate stats if using Unified RA
                if getattr(args, "use_ra_v5", False):
                    unified_ra_stats = analyze_unified_ra_gates(
                        model.module if ddp else model
                    )
                    if unified_ra_stats:
                        eval_metrics.update(unified_ra_stats)

                if "trackio" in tracker_names:
                    import trackio

                    trackio.log(eval_metrics)

                if "wandb" in tracker_names:
                    import wandb

                    wandb.log(eval_metrics)

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                if master_process:
                    checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pt")
                    torch.save(
                        {
                            "iteration": iter_num,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_loss": best_val_loss,
                            "config": vars(args),
                        },
                        checkpoint_path,
                    )
                    print(f"Saved best checkpoint (val_loss={best_val_loss:.4f})")

        # Forward-backward
        t0 = time.time()
        total_loss = 0.0

        # Enable expensive metrics computation only when logging
        should_log_metrics = args.log_metrics and iter_num % args.log_interval == 0
        if should_log_metrics:
            set_metrics_computation(model, True)

        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(args.gradient_accumulation):
            x, y = get_batch(
                "train", args.batch_size, args.block_size, device, args.data_dir
            )

            with torch.amp.autocast(
                device_type="cuda" if "cuda" in device else "cpu", dtype=dtype
            ):
                logits, loss = model(x, y)
                loss = loss / args.gradient_accumulation

            total_loss += loss.item()
            loss.backward()

        # Disable metrics computation after forward pass
        if should_log_metrics:
            set_metrics_computation(model, False)

        # Gradient clipping
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

        # Optimizer step
        optimizer.step()

        # Timing
        t1 = time.time()
        dt = (t1 - t0) * 1000  # ms

        # Log metrics
        if args.log_metrics and iter_num % args.log_interval == 0:
            metrics.log(iter_num, model, dt)

        # Log to trackers (only master process)
        if tracker_names and master_process and iter_num % args.log_interval == 0:
            # Build metrics dictionary
            ra_metrics = {
                "iteration": iter_num,
                "train_loss_step": total_loss,
                "learning_rate": lr,
                "forward_time_ms": dt,
            }

            # Add RA-specific metrics if available
            if args.log_metrics and len(metrics.attention_entropy) > 0:
                ra_metrics["memory_mb"] = (
                    metrics.memory_allocated[-1] if metrics.memory_allocated else 0
                )
                if metrics.attention_entropy:
                    ra_metrics["attention_entropy"] = metrics.attention_entropy[-1]
                if metrics.reciprocity_score:
                    ra_metrics["reciprocity_score"] = metrics.reciprocity_score[-1]

            # Add route gate if lens-gated architecture (always track this!)
            if use_lens and use_route_gate:
                mean_g = get_mean_route_gate(model.module if ddp else model)
                ra_metrics["route_gate_mean"] = mean_g

                # Also log per-layer gates for detailed analysis
                gate_stats = analyze_route_gates(model.module if ddp else model)
                if gate_stats:
                    ra_metrics["route_gate_min"] = gate_stats["min_route_gate"]
                    ra_metrics["route_gate_max"] = gate_stats["max_route_gate"]
                    ra_metrics["route_gate_std"] = gate_stats["std_route_gate"]

            # Add lens gates (w_std, w_rec, w_disc) if using lens architecture
            if use_lens:
                lens_stats = analyze_lens_gates(model.module if ddp else model)
                if lens_stats:
                    # Log all lens gate statistics to tracker
                    ra_metrics.update(lens_stats)

            # Add Unified RA gate stats if using Unified RA
            if getattr(args, "use_ra_v5", False):
                unified_ra_stats = analyze_unified_ra_gates(
                    model.module if ddp else model
                )
                if unified_ra_stats:
                    ra_metrics.update(unified_ra_stats)

            if "trackio" in tracker_names:
                import trackio

                trackio.log(ra_metrics)

            if "wandb" in tracker_names:
                import wandb

                wandb.log(ra_metrics)

        # Print progress (only master process)
        if master_process and iter_num % args.log_interval == 0:
            # Build progress string
            progress_str = f"Iter {iter_num:6d} | loss {total_loss:.4f} | time {dt:.1f}ms | lr {lr:.2e}"

            # Add route gate if lens-gated
            if use_lens and use_route_gate:
                mean_g = get_mean_route_gate(model.module if ddp else model)
                progress_str += f" | route_gate {mean_g:.3f}"

            # Add lens gates if using lens architecture
            if use_lens:
                lens_stats = analyze_lens_gates(model.module if ddp else model)
                if lens_stats:
                    progress_str += f" | w_std {lens_stats['w_std_mean']:.3f}"
                    progress_str += f" | w_rec {lens_stats['w_rec_mean']:.3f}"
                    progress_str += f" | w_disc {lens_stats['w_disc_mean']:.3f}"

            # Add Unified RA gates if using Unified RA
            if getattr(args, "use_ra_v5", False):
                unified_ra_stats = analyze_unified_ra_gates(
                    model.module if ddp else model
                )
                if unified_ra_stats:
                    progress_str += (
                        f" | UA_w_std {unified_ra_stats['unified_ra_w_std_mean']:.3f}"
                    )
                    progress_str += (
                        f" | UA_w_rec {unified_ra_stats['unified_ra_w_rec_mean']:.3f}"
                    )

            print(progress_str)

        iter_num += 1

        # Apply route gate annealing (lens-gated architecture only)
        if use_lens and use_route_gate:
            apply_route_annealing(model.module if ddp else model, iter_num, lens_cfg)

    # Save final checkpoint (only master process in DDP mode)
    if master_process:
        final_checkpoint_path = os.path.join(args.checkpoint_dir, "final_model.pt")
        torch.save(
            {
                "iteration": iter_num,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": vars(args),
            },
            final_checkpoint_path,
        )
        print(f"\nSaved final checkpoint to {final_checkpoint_path}")

        # Save metrics
        if args.json_output:
            metrics.save(args.json_output)
        else:
            default_metrics_path = os.path.join(args.checkpoint_dir, "metrics.json")
            metrics.save(default_metrics_path)

    # Final evaluation (only master process prints)
    if master_process:
        print("\nFinal evaluation:")
    final_losses = estimate_loss(
        model, args.eval_samples, args.batch_size, args.block_size, device
    )
    if master_process:
        print(f"Train loss: {final_losses['train']:.4f}")
        print(f"Val loss: {final_losses['val']:.4f}")
        print(f"Best val loss: {best_val_loss:.4f}")

        # Print metrics summary
        if metrics.attention_entropy:
            print(f"\nAttention Metrics:")
            print(
                f"  Entropy: {np.mean(metrics.attention_entropy):.3f} ± {np.std(metrics.attention_entropy):.3f}"
            )
        if metrics.reciprocity_score:
            print(
                f"  Reciprocity: {np.mean(metrics.reciprocity_score):.3f} ± {np.std(metrics.reciprocity_score):.3f}"
            )
        if metrics.forward_time:
            print(f"  Avg iteration time: {np.mean(metrics.forward_time):.1f}ms")

    # Log final summary to trackers (only master process)
    if tracker_names and master_process:
        final_summary = {
            "final_train_loss": final_losses["train"],
            "final_val_loss": final_losses["val"],
            "best_val_loss": best_val_loss,
        }

        if metrics.attention_entropy:
            final_summary["avg_attention_entropy"] = float(
                np.mean(metrics.attention_entropy)
            )
            final_summary["std_attention_entropy"] = float(
                np.std(metrics.attention_entropy)
            )

        if metrics.reciprocity_score:
            final_summary["avg_reciprocity_score"] = float(
                np.mean(metrics.reciprocity_score)
            )
            final_summary["std_reciprocity_score"] = float(
                np.std(metrics.reciprocity_score)
            )

        if metrics.forward_time:
            final_summary["avg_forward_time_ms"] = float(np.mean(metrics.forward_time))

        if "trackio" in tracker_names:
            import trackio

            trackio.log(final_summary)
            trackio.finish()

        if "wandb" in tracker_names:
            import wandb

            wandb.log(final_summary)
            wandb.finish()

    # Cleanup DDP
    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()

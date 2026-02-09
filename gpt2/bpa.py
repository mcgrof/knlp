"""
GPT-2 with Boundary-Pressure Attention (BPA)

BPA is the renamed successor to RGSA (Route-Gated Sparse Attention).
The key insight from v19 research:

  Importance is query-conditional, not head-static.
  boundary_pressure (attention mass at local boundary) predicts
  when far-context access matters.

BPA is attention where:
- Local window is always available
- Far-context access is conditionally enabled per (layer, head, token)
- Gating is based on boundary_pressure signal
- Default policy is threshold-based

This module provides BPA-named aliases for the underlying implementation
in rgsa.py. The implementation is shared; only the naming differs.

See docs/bp_v1_migration_note.md for migration details.
"""

# Import all classes from rgsa.py and alias with BPA naming
from gpt2.rgsa import (
    # Config
    RGSAConfig,
    # Metrics
    RGSAMetrics,
    HeadMetrics,
    ImpactMetrics,
    ImpactKLTracker,
    ConditionalImpactMetrics,
    ConditionalImpactTracker,
    ConditionalSignals,
    ConditionalSignalComputer,
    # Model components
    ChunkRouter,
    RetrievalGate,
    RGSACausalSelfAttention,
    RGSABlock,
    GPT2_RGSA,
    # Utilities
    LayerNorm,
    MLP,
)

# BPA-named aliases (preferred going forward)
BPAConfig = RGSAConfig
BPAMetrics = RGSAMetrics
BPACausalSelfAttention = RGSACausalSelfAttention
BPABlock = RGSABlock
GPT2_BPA = GPT2_RGSA

# Conditional signal infrastructure (no rename needed, already signal-based)
# ConditionalImpactMetrics, ConditionalImpactTracker,
# ConditionalSignals, ConditionalSignalComputer are used as-is

__all__ = [
    # Primary BPA names
    "BPAConfig",
    "BPAMetrics",
    "BPACausalSelfAttention",
    "BPABlock",
    "GPT2_BPA",
    # Conditional signal infrastructure
    "ConditionalImpactMetrics",
    "ConditionalImpactTracker",
    "ConditionalSignals",
    "ConditionalSignalComputer",
    # Supporting classes
    "HeadMetrics",
    "ImpactMetrics",
    "ImpactKLTracker",
    "ChunkRouter",
    "RetrievalGate",
    "LayerNorm",
    "MLP",
    # Legacy aliases (still available)
    "RGSAConfig",
    "RGSAMetrics",
    "RGSACausalSelfAttention",
    "RGSABlock",
    "GPT2_RGSA",
]

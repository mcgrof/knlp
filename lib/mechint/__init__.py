# SPDX-License-Identifier: MIT
"""
Mechanistic Interpretability Tools

This module provides tools for mechanistic interpretability analysis,
including:
- KV feature-circuit discovery via binary channel masking
- Post-training sparse circuit optimization
- Visualization and logging to W&B
- Circuit faithfulness evaluation

Based on approaches from "Scaling Sparse Feature Circuit Finding to Gemma 9B"
and related mechanistic interpretability research.
"""

from .kv_circuit import (
    AnalysisConfig,
    KVFeatureMask,
    KVCircuitAnalyzer,
    SparsitySchedule,
    run_kv_circuit_analysis,
)
from .visualize import (
    visualize_kv_masks,
    plot_sparsity_curves,
    log_circuit_to_wandb,
    create_circuit_summary_report,
)

__all__ = [
    "AnalysisConfig",
    "KVFeatureMask",
    "KVCircuitAnalyzer",
    "SparsitySchedule",
    "run_kv_circuit_analysis",
    "visualize_kv_masks",
    "plot_sparsity_curves",
    "log_circuit_to_wandb",
    "create_circuit_summary_report",
]

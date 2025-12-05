"""
KV Cache Compression Plugin: Base Interface

Pluggable KV-cache compression system for open-weight causal LMs.
Supports multiple compression algorithms with minimal model surgery.

Design principles from ChatGPT's pluggable architecture:
- Post-hoc plugin for existing trained models
- Minimal surgery to original model code
- Inference-focused deployment
- Per-layer/head heterogeneous compression
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class KVCompressorBase(ABC, nn.Module):
    """
    Base class for KV cache compression algorithms.

    Supports:
    - Multiple compression backends (KVSplice, PCA, PCA+Spline)
    - Per-layer/head compression policies
    - Calibration workflow (reconstruction-only or task-aware)
    - State serialization for deployment

    Config format:
    {
        "global": {
            "target_memory_reduction": 0.5,
            "d_head": 64,
            "algo_default": "kvsplice"
        },
        "per_layer_head": {
            "(layer, head)": {
                "enabled": True,
                "algo": "kvsplice" | "pca" | "pca_spline",
                "rank": 32,
                "extra": {...}  # Algorithm-specific options
            }
        }
    }
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.calibration_mode = False
        self.calibration_data: Dict[Tuple[int, int], list] = {}

        # Extract global config
        global_cfg = config.get("global", {})
        self.target_memory_reduction = global_cfg.get("target_memory_reduction", 0.5)
        self.d_head = global_cfg.get("d_head", 64)
        self.algo_default = global_cfg.get("algo_default", "kvsplice")

        # Parse per-layer/head config
        self.layer_head_configs = {}
        for key, cfg in config.get("per_layer_head", {}).items():
            # Key format: "layerX/headY" or tuple (layer_idx, head_idx)
            if isinstance(key, str):
                parts = key.split("/")
                if len(parts) == 2:
                    layer_str, head_str = parts
                    layer_idx = int(layer_str.replace("layer", ""))
                    head_idx = int(head_str.replace("head", ""))
                else:
                    raise ValueError(f"Invalid key format: {key}")
            else:
                layer_idx, head_idx = key

            self.layer_head_configs[(layer_idx, head_idx)] = cfg

    def get_config(self, layer_idx: int, head_idx: int) -> Optional[Dict]:
        """Get compression config for specific layer/head."""
        return self.layer_head_configs.get((layer_idx, head_idx), None)

    def is_enabled(self, layer_idx: int, head_idx: int) -> bool:
        """Check if compression is enabled for this layer/head."""
        cfg = self.get_config(layer_idx, head_idx)
        if cfg is None:
            return False
        return cfg.get("enabled", False)

    @abstractmethod
    def start_calibration(self):
        """
        Prepare internal buffers for calibration.

        Called once before calibration data collection begins.
        Subclasses should initialize per-layer/head sample buffers.
        """
        pass

    @abstractmethod
    def observe_kv(
        self,
        layer_idx: int,
        head_idx: int,
        K: torch.Tensor,
        V: torch.Tensor,
    ):
        """
        Observe KV pairs during calibration.

        Args:
            layer_idx: Layer index
            head_idx: Head index within layer
            K: Keys [batch, seq, d_k]
            V: Values [batch, seq, d_v]

        Subclasses should downsample if necessary to keep memory bounded.
        Typically stores snapshots for later parameter fitting.
        """
        pass

    @abstractmethod
    def end_calibration(self):
        """
        Compute compression parameters from calibration data.

        Called once after all calibration data has been observed.
        Subclasses should:
        - Fit projection matrices (KVSplice) or PCA bases
        - Fit scale/shift parameters or spline coefficients
        - Populate internal state for compress/decompress
        - Clear calibration buffers to free memory
        """
        pass

    @abstractmethod
    def compress(
        self,
        layer_idx: int,
        head_idx: int,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runtime compression of KV pairs.

        Args:
            layer_idx: Layer index
            head_idx: Head index
            K: Keys [batch, seq, d_k]
            V: Values [batch, seq, d_v]

        Returns:
            Z_k, Z_v: Compressed representations
                Shape depends on algorithm (typically [batch, seq, rank])
        """
        pass

    @abstractmethod
    def decompress(
        self,
        layer_idx: int,
        head_idx: int,
        Z_k: torch.Tensor,
        Z_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runtime decompression of compressed KV.

        Args:
            layer_idx: Layer index
            head_idx: Head index
            Z_k, Z_v: Compressed representations

        Returns:
            K_hat, V_hat: Reconstructed KV with original shapes
        """
        pass

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Return serializable parameters for saving.

        Includes:
        - Config metadata
        - Per-layer/head compression parameters
        - Algorithm-specific state (projection matrices, PCA bases, etc.)
        """
        state = super().state_dict(*args, **kwargs)
        state["config"] = self.config
        return state

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True):
        """
        Restore parameters from saved state.

        Args:
            state: State dict from state_dict()
            strict: Whether to strictly enforce key matching
        """
        if "config" in state:
            self.config = state["config"]
            # Re-parse config
            self.__init__(self.config)

        super().load_state_dict(state, strict=strict)

    def memory_stats(self) -> Dict[str, Any]:
        """
        Compute memory statistics for this compressor.

        Returns:
            stats: Dict with keys:
                - total_original_params: Uncompressed KV parameters
                - total_compressed_params: Compressed parameters
                - compression_ratio: Ratio of compressed/original
                - memory_savings_pct: Percentage memory saved
        """
        total_orig = 0
        total_comp = 0

        for (layer_idx, head_idx), cfg in self.layer_head_configs.items():
            if not cfg.get("enabled", False):
                total_orig += self.d_head
                total_comp += self.d_head
                continue

            rank = cfg.get("rank", self.d_head)
            total_orig += self.d_head
            total_comp += rank

        if total_orig == 0:
            return {
                "total_original_params": 0,
                "total_compressed_params": 0,
                "compression_ratio": 1.0,
                "memory_savings_pct": 0.0,
            }

        return {
            "total_original_params": total_orig,
            "total_compressed_params": total_comp,
            "compression_ratio": total_comp / total_orig,
            "memory_savings_pct": (1.0 - total_comp / total_orig) * 100,
        }

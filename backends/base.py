"""
Base class for v14 KV compression backends.

All backends implement the same interface:
  1. calibrate(model, token_data, L, device) — one-time calibration
  2. run_decode(model, prefix_ids, continuation_ids, ...) — decode loop

The harness keeps routing fixed:
  - Near window (last W_min tokens) = full KV always
  - Far tokens (before W_min) = compressed by backend
  - No eviction, no dropping
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import numpy as np


@dataclass
class V14StepStats:
    """Per-step statistics for v14 compression backends."""

    kv_kept: int = 0
    kv_bytes_full: int = 0
    kv_bytes_compressed: int = 0
    kv_bytes_total: int = 0
    compress_ms: float = 0.0
    decompress_ms: float = 0.0
    gate_ms: float = 0.0
    n_full: int = 0
    n_compressed: int = 0


class CompressionBackend(ABC):
    """Abstract compression backend for KV cache."""

    W_min: int = 1024  # near window: always full KV
    W_sink: int = 4  # sink tokens: never compress

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def configure(self, L: int, model_config: dict, **kwargs):
        """Configure backend parameters for a given L."""
        pass

    def calibrate(self, model, token_data, L, device_str, model_config):
        """Optional: offline calibration on sample data."""
        pass

    @abstractmethod
    def run_decode(
        self,
        model,
        prefix_ids: torch.Tensor,
        continuation_ids: torch.Tensor,
        device_str: str,
        max_ctx: int,
    ) -> Tuple[torch.Tensor, List[V14StepStats]]:
        """Run decode with compression.

        Returns:
            all_logits: [B, decode_steps+1, V]
            step_stats: per-step statistics
        """
        pass

    def compression_ratio(self) -> float:
        """Nominal compression ratio (compressed/dense bytes)."""
        return 1.0

    def description(self) -> str:
        """Short description of backend config."""
        return self.name


class DenseBackend(CompressionBackend):
    """Dense baseline — no compression."""

    @property
    def name(self):
        return "dense"

    def configure(self, L, model_config, **kwargs):
        self.L = L
        self.mc = model_config

    @torch.no_grad()
    def run_decode(self, model, prefix_ids, continuation_ids, device_str, max_ctx):
        decode_steps = continuation_ids.shape[1]
        n_layers = self.mc["n_layers"]
        n_kv_heads = self.mc["n_kv_heads"]
        head_dim = self.mc["head_dim"]
        elem = 2  # fp16

        out = model(prefix_ids, use_cache=True)
        past = out.past_key_values
        all_logits = [out.logits[:, -1:, :]]
        step_stats = []

        for step in range(decode_steps):
            next_token = continuation_ids[:, step : step + 1]
            out = model(next_token, past_key_values=past, use_cache=True)
            past = out.past_key_values
            all_logits.append(out.logits)

            cache_len = past[0][0].shape[2]
            bpt = 2 * n_kv_heads * head_dim * elem
            total_bytes = cache_len * bpt * n_layers
            step_stats.append(
                V14StepStats(
                    kv_kept=cache_len,
                    kv_bytes_full=total_bytes,
                    kv_bytes_compressed=0,
                    kv_bytes_total=total_bytes,
                    n_full=cache_len,
                    n_compressed=0,
                )
            )

        return torch.cat(all_logits, dim=1), step_stats

    def compression_ratio(self):
        return 1.0

    def description(self):
        return "dense (no compression)"

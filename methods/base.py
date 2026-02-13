"""
Base interface for all bitter methods.

Every bitter method must implement:
- name: str property
- configure(L, model_config, **kwargs): set up for a given context length
- decode_step(model, next_token, past, step, **kwargs) -> (past, logits, stats)
- summary() -> dict with tier counts, overhead stats, etc.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from transformers.cache_utils import DynamicCache


@dataclass
class StepStats:
    """Per-step statistics from a bitter method."""

    kv_kept: int = 0
    tier_full: int = 0
    tier_mla: int = 0
    tier_splice: int = 0
    tier_dropped: int = 0
    gate_ms: float = 0.0
    kv_bytes_proxy: float = 0.0


class BitterMethod(ABC):
    """Base class for all bitter KV compression methods."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Method identifier (e.g., 'bitter0', 'dense')."""
        pass

    @abstractmethod
    def configure(self, L: int, model_config: dict, **kwargs):
        """Configure for a specific context length and model.

        Args:
            L: context/prefix length
            model_config: dict with n_layers, n_heads, n_kv_heads,
                         head_dim, hidden_size, etc.
        """
        pass

    @abstractmethod
    def run_decode(
        self,
        model,
        prefix_ids,
        continuation_ids,
        device_str: str,
        max_ctx: int,
    ) -> Tuple[torch.Tensor, list]:
        """Run full decode loop.

        Args:
            model: HF CausalLM model
            prefix_ids: [B, prefix_len] token ids for prefill
            continuation_ids: [B, decode_steps] token ids to decode
            device_str: 'cuda' or 'cpu'
            max_ctx: model's max context length

        Returns:
            all_logits: [B, decode_steps+1, vocab] concatenated logits
            step_stats: list of StepStats per decode step
        """
        pass

    def n_thresholds(self) -> int:
        """Number of hand-tuned thresholds/knobs in this method."""
        return 0

    def learned_fraction(self) -> float:
        """Fraction of routing decisions that are learned (0-1)."""
        return 0.0


class DenseMethod(BitterMethod):
    """Dense baseline — no compression, keeps full KV cache."""

    @property
    def name(self):
        return "dense"

    def configure(self, L, model_config, **kwargs):
        self.L = L
        self.model_config = model_config

    @torch.no_grad()
    def run_decode(self, model, prefix_ids, continuation_ids, device_str, max_ctx):
        decode_steps = continuation_ids.shape[1]

        # Prefill
        out = model(prefix_ids, use_cache=True)
        past = out.past_key_values
        all_logits = [out.logits[:, -1:, :]]
        step_stats = []

        n_kv_heads = self.model_config["n_kv_heads"]
        head_dim = self.model_config["head_dim"]
        elem = 2  # fp16/bf16

        for step in range(decode_steps):
            next_token = continuation_ids[:, step : step + 1]
            out = model(next_token, past_key_values=past, use_cache=True)
            past = out.past_key_values
            all_logits.append(out.logits)

            cache_len = past[0][0].shape[2]
            bpt = 2 * n_kv_heads * head_dim * elem
            n_layers = self.model_config["n_layers"]
            step_stats.append(
                StepStats(
                    kv_kept=cache_len,
                    tier_full=cache_len,
                    kv_bytes_proxy=cache_len * bpt * n_layers,
                )
            )

        return torch.cat(all_logits, dim=1), step_stats

    def n_thresholds(self):
        return 0

    def learned_fraction(self):
        return 0.0

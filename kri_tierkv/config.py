# SPDX-License-Identifier: GPL-2.0
"""Configuration for KRI-TierKV.

Maps the Kconfig symbols (CONFIG_KNLP_KRI_TIERKV*) and the experiment CLI flags
onto one dataclass. Defaults are the safe operating point: KRI-D-sum eviction,
kri_topk retrieval, no quantization; when quant is enabled the default is the
lossless-leaning K16/V8.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EvictionPolicy(str, Enum):
    FIFO = "fifo"
    RECENCY = "recency"
    KRI_D_SUM = "kri_d_sum"


class RetrievalPolicy(str, Enum):
    DENSE_REFERENCE = "dense_reference"  # all blocks used
    RECENT_ONLY = "recent_only"  # fast tier only
    KRI_TOPK = "kri_topk"  # fast + top-K slow by KRI-D-sum
    ORACLE_TOPK = "oracle_topk"  # top-K slow by measured attention mass (offline)


class QuantPolicy(str, Enum):
    NONE = "none"
    V8_ONLY = "v8_only"  # K16 / V8  (safe default)
    V4_ONLY = "v4_only"  # K16 / V4
    K8V8 = "k8v8"
    K8V4 = "k8v4"  # experimental / unsafe


# Quant policies whose keys drop below 16 bits are flagged unsafe: on
# fragile-key models (e.g. Qwen) sub-16-bit keys collapse quality.
UNSAFE_QUANT = {QuantPolicy.K8V4}


@dataclass
class TierKVConfig:
    # tiering geometry
    block_size: int = 128
    fast_window_tokens: int = 8192
    protect_prefix_tokens: int = 1024
    slow_topk_blocks: int = 16
    decode_neighborhood_blocks: int = 1  # blocks around the current decode point

    # policies
    eviction_policy: EvictionPolicy = EvictionPolicy.KRI_D_SUM
    retrieval_policy: RetrievalPolicy = RetrievalPolicy.KRI_TOPK
    quant_policy: QuantPolicy = QuantPolicy.NONE

    # quant bit-widths (override the policy shorthand if set)
    k_bits: int = 16
    v_bits: int = 16

    # run
    emulation: bool = True  # milestone 1: record decisions, do not alter attention

    @property
    def fast_window_blocks(self) -> int:
        return max(1, self.fast_window_tokens // self.block_size)

    @property
    def protect_prefix_blocks(self) -> int:
        return max(0, self.protect_prefix_tokens // self.block_size)

    def resolved_bits(self) -> tuple:
        """Return (k_bits, v_bits) from the quant policy, honoring explicit
        overrides. NONE keeps full precision.
        """
        p = self.quant_policy
        if p == QuantPolicy.NONE:
            return (16, 16)
        table = {
            QuantPolicy.V8_ONLY: (16, 8),
            QuantPolicy.V4_ONLY: (16, 4),
            QuantPolicy.K8V8: (8, 8),
            QuantPolicy.K8V4: (8, 4),
        }
        kb, vb = table[p]
        # explicit CLI overrides win if the caller changed them from defaults
        if self.k_bits != 16:
            kb = self.k_bits
        if self.v_bits != 16:
            vb = self.v_bits
        return (kb, vb)

    def is_unsafe_quant(self) -> bool:
        return self.quant_policy in UNSAFE_QUANT

    @classmethod
    def from_kconfig(cls, config: dict) -> "TierKVConfig":
        """Build from a parsed knlp .config dict (string/bool values)."""

        def on(key):
            v = config.get(key)
            return v in ("y", True, "true", "1", 1)

        cfg = cls()
        cfg.emulation = on("CONFIG_KNLP_KRI_TIERKV_EMU")
        if on("CONFIG_KNLP_KRI_TIERKV_KRI_D_SUM"):
            cfg.eviction_policy = EvictionPolicy.KRI_D_SUM
        if on("CONFIG_KNLP_KRI_TIERKV_ASYM_QUANT"):
            cfg.quant_policy = (
                QuantPolicy.V8_ONLY
                if on("CONFIG_KNLP_KRI_TIERKV_V_ONLY")
                else QuantPolicy.K8V8
            )
        return cfg

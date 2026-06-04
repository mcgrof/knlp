"""TrellisConfig — configuration for the Trellis bounded-memory LM."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class TrellisConfig:
    # --- model dimensions ---
    vocab_size: int = 50257
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_head: int = 64
    n_slots: int = 64           # M: bounded memory slots per head
    max_seq_len: int = 1024

    # --- Trellis memory knobs ---
    conv_kernel: int = 4
    use_short_conv_qk: bool = True
    # phi: intermediate activation applied to z = M @ write (over the slot dim)
    activation: str = "ln_silu"          # ["ln_silu","l2_silu","softmax"]
    # alpha: the learned write target / code
    alpha_mode: str = "linear"           # ["linear","softmax","ln_silu","l2_silu"]
    # beta: forget gate granularity
    beta_mode: str = "scalar_per_head"   # ["scalar_per_head","per_slot"]
    gamma_init: float = 1e-2             # learning-rate of the inner OGD step
    exact_inner: bool = True             # exact sequential VJP (Phase 0)
    chunk_size: int = 1                  # 1 = pure sequential
    chunk_refine: int = 1                # intra-chunk z refinement passes (faithful chunkwise)
    post_gate: bool = True               # SwiGLU-style post gate on mixer output
    forget_gate: bool = True             # if False, beta is forced to 1 (no decay)

    # --- training / misc ---
    dropout: float = 0.0
    dtype: str = "bf16"                  # ["bf16","fp16","fp32"]
    mlp_ratio: float = 4.0
    tie_embeddings: bool = True

    def __post_init__(self):
        assert self.activation in ("ln_silu", "l2_silu", "softmax"), self.activation
        assert self.alpha_mode in ("linear", "softmax", "ln_silu", "l2_silu"), self.alpha_mode
        assert self.beta_mode in ("scalar_per_head", "per_slot"), self.beta_mode
        assert self.dtype in ("bf16", "fp16", "fp32"), self.dtype

    @property
    def inner_dim(self) -> int:
        return self.n_heads * self.d_head

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrellisConfig":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})

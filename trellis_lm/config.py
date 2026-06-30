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
    n_slots: int = 64  # M: bounded memory slots per head
    max_seq_len: int = 1024

    # --- Trellis memory knobs ---
    conv_kernel: int = 4
    use_short_conv_qk: bool = True
    use_short_conv_v: bool = False
    # L2-normalize the write vector (and the read query) over head_dim before the
    # memory update -- DeltaNet's contraction stabilizer (the published paper's
    # Trellis equations use raw k; this is exploratory). Bounds gamma*||w||^2 so
    # the ungated/aggressive-LR linear write stops detonating; may also help the
    # nonlinear write disproportionately (variable key norm x nonlinear curvature).
    write_l2norm: bool = False
    # phi: intermediate activation applied to z = M @ write (over the slot dim).
    # "identity" reduces the nonlinear write to the (gated) delta rule -- the
    # same-shell control for "does the nonlinear write help?" (paper ablation).
    activation: str = "ln_silu"  # ["ln_silu","l2_silu","softmax","identity"]
    # alpha: the learned write target / code
    alpha_mode: str = "linear"  # ["linear","softmax","ln_silu","l2_silu"]
    # beta: forget gate granularity
    beta_mode: str = "scalar_per_head"  # ["scalar_per_head","per_slot"]
    beta_init: float = 0.5  # init MEAN of the forget gate; the
    # beta_proj bias is set to logit(beta_init). 0.5 = zero-bias (legacy). The
    # paper's retention semantics want beta near 1; sweep {0.8..0.995}.
    # trellis_retention_mode controls the retention source. "token_proj" is the
    # historical behavior: beta is projected from each token through beta_proj.
    # The explicit branch modes below test static fixed/learned timescales.
    trellis_retention_mode: str = "token_proj"
    trellis_beta_init: float = 0.99
    trellis_beta_min: float = 0.90
    trellis_beta_max: float = 0.9995
    trellis_beta_param: str = "sigmoid_logit"
    trellis_beta_lr_mult: float = 1.0
    trellis_beta_weight_decay: float = 0.0
    trellis_beta_init_schedule: str = "flat_099"
    gamma_init: float = 1e-2  # learning-rate of the inner OGD step
    # Stabilize the memory update without changing the default Trellis math.
    # "innovation_rms_cap" applies a one-sided RMS cap to phi(z)-alpha before
    # the VJP/update. "layerwise_gamma" only scales gamma in layer 0 through
    # trellis_layer0_gamma_mult. The combo enables both. "delta_ratio_cap" is a
    # reference-path safety cap on aggregate update/state ratio.
    trellis_update_stabilizer: str = "none"
    trellis_innovation_rms_cap: float = 0.0
    trellis_delta_ratio_cap: float = 0.0
    trellis_state_rms_floor: float = 1e-3
    trellis_layer0_gamma_mult: float = 1.0
    trellis_stabilizer_detach_scale: bool = True
    update_gate_mode: str = "none"  # ["none","scalar","channel"]
    update_gate_init: float = 0.95
    residual_update_mix: float = 0.0
    # output_path: "current" = out_proj(y) then *SiLU(gate) AFTER out_proj
    # (legacy). "paper" = PostNorm(y) -> *GeLU(gate) -> out_proj (Fig 1 order:
    # Trellis -> Norm -> gated branch -> Linear), gate in the inner_dim space.
    output_path: str = "current"  # ["current","paper"]
    # value_readout_act: final phi on the value-pass readout y = phi(M^T r).
    # Paper applies normalized-SiLU; "none" = legacy (no activation).
    value_readout_act: str = "none"  # ["none","ln_silu","l2_silu"]
    exact_inner: bool = True  # exact sequential VJP (Phase 0)
    chunk_size: int = 1  # 1 = pure sequential
    chunk_refine: int = 0  # intra-chunk z refinement passes (faithful chunkwise)
    post_gate: bool = True  # SwiGLU-style post gate on mixer output
    forget_gate: bool = True  # if False, beta is forced to 1 (no decay)

    # --- training / misc ---
    dropout: float = 0.0
    dtype: str = "bf16"  # ["bf16","fp16","fp32"]
    mlp_ratio: float = 4.0
    tie_embeddings: bool = True

    def __post_init__(self):
        assert self.activation in (
            "silu",
            "ln_silu",
            "norm_silu",
            "l2_silu",
            "softmax",
            "identity",
            "scaled_identity",
        ), self.activation
        assert self.alpha_mode in (
            "linear",
            "softmax",
            "ln_silu",
            "norm_silu",
            "l2_silu",
        ), self.alpha_mode
        assert self.beta_mode in ("scalar_per_head", "per_slot"), self.beta_mode
        assert self.trellis_retention_mode in (
            "token_proj",
            "fixed_beta",
            "learned_per_head",
            "learned_per_channel",
            "learned_per_head_channel",
        ), self.trellis_retention_mode
        assert self.trellis_beta_param == "sigmoid_logit", self.trellis_beta_param
        assert 0.0 < self.trellis_beta_min < self.trellis_beta_max < 1.0, (
            self.trellis_beta_min,
            self.trellis_beta_max,
        )
        assert self.trellis_beta_min < self.trellis_beta_init < (
            self.trellis_beta_max
        ), self.trellis_beta_init
        assert self.trellis_beta_lr_mult >= 0.0, self.trellis_beta_lr_mult
        assert self.trellis_beta_weight_decay >= 0.0, (
            self.trellis_beta_weight_decay
        )
        assert self.trellis_beta_init_schedule in (
            "flat_099",
            "layer_short_to_long",
            "head_logspace",
            "layer_head_logspace",
        ), self.trellis_beta_init_schedule
        assert self.trellis_update_stabilizer in (
            "none",
            "innovation_rms_cap",
            "delta_ratio_cap",
            "layerwise_gamma",
            "innovation_rms_cap_plus_layer0_gamma",
            "innovation_rms_cap_plus_layerwise_gamma",
        ), self.trellis_update_stabilizer
        assert self.trellis_innovation_rms_cap >= 0.0, (
            self.trellis_innovation_rms_cap
        )
        assert self.trellis_delta_ratio_cap >= 0.0, self.trellis_delta_ratio_cap
        assert self.trellis_state_rms_floor >= 0.0, self.trellis_state_rms_floor
        assert self.trellis_layer0_gamma_mult >= 0.0, (
            self.trellis_layer0_gamma_mult
        )
        assert self.update_gate_mode in (
            "none",
            "scalar",
            "channel",
        ), self.update_gate_mode
        assert self.output_path in ("current", "paper"), self.output_path
        assert self.value_readout_act in (
            "none",
            "silu",
            "identity",
            "ln_silu",
            "norm_silu",
            "l2_silu",
        ), self.value_readout_act
        assert 0.0 < self.beta_init < 1.0, self.beta_init
        assert 0.0 < self.update_gate_init < 1.0, self.update_gate_init
        assert self.residual_update_mix >= 0.0, self.residual_update_mix
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

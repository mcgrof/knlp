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
    # trellis_write_mode selects the innovation rule. "nonlinear_phi" (default)
    # is the paper's state-dependent write u_t = J_phi(z)^T(phi(z)-alpha), whose
    # state-dependent Jacobian forfeits the delta-rule free lunch (no cheap exact
    # chunk) and couples staleness<->stability. "input_conditioned" replaces it
    # with u_t = a(x_t) ⊙ z_t − alpha_t, where a(x_t) ∈ R^M is a per-slot gate
    # computed from the token INPUT x_t (not the state code z_t=M@w). This keeps
    # the update AFFINE in M -> the WY/UT chunk transform survives (exact and
    # parallelizable), and a≡1 recovers the (gated) delta rule exactly. It is the
    # salvageable Trellis: input-conditioned nonlinear expressivity without the
    # state-dependent Jacobian that breaks exact chunking (Pro consult 2026-07-05).
    trellis_write_mode: str = "nonlinear_phi"
    # activation for the input-conditioned per-slot gate a(x_t). "softplus" with
    # a bias init of log(e-1) starts a≡1 (exact delta rule) then learns.
    trellis_input_gate_act: str = "softplus"
    # scope of the input-conditioned gate. "per_slot": a(x_t) in R^M, one gain
    # per memory slot -- more expressive, needs n per-slot chunk solves.
    # "scalar": a(x_t) in R, one gain per token/head broadcast to all slots --
    # cheaper (the n slot-solves collapse to one shared solve, GDN-cost) and the
    # ablation for whether per-slot expressivity is actually needed to bind.
    trellis_input_gate_scope: str = "per_slot"
    # rank of the SLOT-MIXING part of the input-conditioned gain. 0 (default) is
    # the diagonal gate G(x)=diag(a(x)) -- slots do not mix; the write to slot m
    # uses only its own readout z_m. rank>0 adds a token-conditioned low-rank
    # term: G(x) = diag(a(x)) + U(x) V(x)^T with U,V in R^{M x r}, so each slot's
    # write can read the whole memory readout z (content-addressed cross-slot
    # routing). It stays AFFINE in M (U,V from the token, not the state), so the
    # recurrence is still exact-chunkable; it is the chunkable analog of the
    # paper's dense state-dependent Jacobian mixing. rank=0 is bit-identical to
    # the diagonal path (no low-rank projection is built). This is the
    # Slot-Mixing Delta rank ladder (scalar -> diagonal -> rank1 -> rank2).
    trellis_input_gate_rank: int = 0
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
    # Where to apply the token-conditioned update gate. The historical scalar
    # and channel gates used "both"; overwrite probes can gate only value writes.
    trellis_update_gate_target: str = "both"  # ["both","key","value"]
    trellis_update_gate_layer_mode: str = "all"
    trellis_update_gate_context_mode: str = "current"
    trellis_update_gate_floor: float = 0.0
    residual_update_mix: float = 0.0
    # output_path: "current" = out_proj(y) then *SiLU(gate) AFTER out_proj
    # (legacy). "paper" = PostNorm(y) -> *GeLU(gate) -> out_proj (Fig 1 order:
    # Trellis -> Norm -> gated branch -> Linear), gate in the inner_dim space.
    output_path: str = "current"  # ["current","paper"]
    # value_readout_act: final phi on the value-pass readout y = phi(M^T r).
    # Paper applies normalized-SiLU; "none" = legacy (no activation).
    value_readout_act: str = "none"  # ["none","ln_silu","l2_silu"]
    # value alpha controls the target code used when the value memory writes.
    # "shared" is the historical Trellis path: key and value memories both use
    # alpha_proj(h). "key_readout" tests explicit key-code binding by using the
    # key-pass readout code r as the value-pass write target.
    # The prev-* correction modes preserve shared alpha and add a bounded
    # learned correction toward the previous token's code. They are overwrite
    # diagnostics for grammars where a value immediately follows its key.
    # The local-key correction mode uses a separate key-address projection:
    # value writes target the previous token's projected address, and paired
    # read-query modes can read with the current token's projected address.
    trellis_value_alpha_mode: str = "shared"
    trellis_value_alpha_mix: float = 1.0
    trellis_value_alpha_correction_init: float = 1e-3
    trellis_value_alpha_correction_max: float = 0.25
    # value_read_query controls the code used to read from the value memory.
    # "key_readout" is the historical path: use r = phi(M_key q). The gated
    # alpha-residual modes test whether overwrite failures come from unstable
    # alignment between key-pass read codes and the value write addresses.
    trellis_value_read_query_mode: str = "key_readout"
    trellis_value_read_query_gate_init: float = 0.05
    trellis_value_read_query_gate_max: float = 0.75
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
        assert self.trellis_write_mode in (
            "nonlinear_phi",
            "input_conditioned",
        ), self.trellis_write_mode
        assert self.trellis_input_gate_act in (
            "softplus",
            "sigmoid",
            "identity",
        ), self.trellis_input_gate_act
        assert self.trellis_input_gate_scope in (
            "per_slot",
            "scalar",
            "identity",
        ), self.trellis_input_gate_scope
        assert (
            isinstance(self.trellis_input_gate_rank, int)
            and self.trellis_input_gate_rank >= 0
        ), self.trellis_input_gate_rank
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
        assert (
            self.trellis_beta_min < self.trellis_beta_init < (self.trellis_beta_max)
        ), self.trellis_beta_init
        assert self.trellis_beta_lr_mult >= 0.0, self.trellis_beta_lr_mult
        assert self.trellis_beta_weight_decay >= 0.0, self.trellis_beta_weight_decay
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
        assert self.trellis_innovation_rms_cap >= 0.0, self.trellis_innovation_rms_cap
        assert self.trellis_delta_ratio_cap >= 0.0, self.trellis_delta_ratio_cap
        assert self.trellis_state_rms_floor >= 0.0, self.trellis_state_rms_floor
        assert self.trellis_layer0_gamma_mult >= 0.0, self.trellis_layer0_gamma_mult
        assert self.update_gate_mode in (
            "none",
            "scalar",
            "channel",
        ), self.update_gate_mode
        assert self.trellis_update_gate_target in (
            "both",
            "key",
            "value",
        ), self.trellis_update_gate_target
        assert self.trellis_update_gate_layer_mode in (
            "all",
            "layer0",
            "lower_half",
            "upper_half",
            "not_layer0",
        ), self.trellis_update_gate_layer_mode
        assert self.trellis_update_gate_context_mode in (
            "current",
            "prev",
            "current_prev",
        ), self.trellis_update_gate_context_mode
        assert (
            0.0 <= self.trellis_update_gate_floor < 1.0
        ), self.trellis_update_gate_floor
        assert self.output_path in ("current", "paper"), self.output_path
        assert self.value_readout_act in (
            "none",
            "silu",
            "identity",
            "ln_silu",
            "norm_silu",
            "l2_silu",
        ), self.value_readout_act
        assert self.trellis_value_alpha_mode in (
            "shared",
            "key_readout",
            "key_readout_detached",
            "shared_plus_key_correction",
            "shared_plus_key_correction_detached",
            "shared_plus_local_key_correction",
            "shared_plus_local_key_correction_detached",
            "shared_plus_prev_alpha_correction",
            "shared_plus_prev_alpha_correction_detached",
            "shared_plus_prev_key_correction",
            "shared_plus_prev_key_correction_detached",
        ), self.trellis_value_alpha_mode
        assert 0.0 <= self.trellis_value_alpha_mix <= 1.0, self.trellis_value_alpha_mix
        assert (
            self.trellis_value_alpha_correction_init >= 0.0
        ), self.trellis_value_alpha_correction_init
        assert (
            self.trellis_value_alpha_correction_max > 0.0
        ), self.trellis_value_alpha_correction_max
        assert self.trellis_value_alpha_correction_init <= (
            self.trellis_value_alpha_correction_max
        ), (
            self.trellis_value_alpha_correction_init,
            self.trellis_value_alpha_correction_max,
        )
        assert self.trellis_value_read_query_mode in (
            "key_readout",
            "local_key_address",
            "local_key_address_detached",
            "alpha_residual_gate",
            "alpha_residual_gate_detached",
        ), self.trellis_value_read_query_mode
        assert (
            0.0
            <= self.trellis_value_read_query_gate_init
            <= (self.trellis_value_read_query_gate_max)
        ), (
            self.trellis_value_read_query_gate_init,
            self.trellis_value_read_query_gate_max,
        )
        assert (
            0.0 < self.trellis_value_read_query_gate_max <= 1.0
        ), self.trellis_value_read_query_gate_max
        assert 0.0 < self.beta_init < 1.0, self.beta_init
        assert 0.0 < self.update_gate_init < 1.0, self.update_gate_init
        if self.update_gate_mode != "none":
            assert self.update_gate_init > self.trellis_update_gate_floor, (
                self.update_gate_init,
                self.trellis_update_gate_floor,
            )
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

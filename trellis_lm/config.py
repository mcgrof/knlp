"""TrellisConfig — configuration for the Trellis bounded-memory LM."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional


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
    # phi and f are DISTINCT functions in the paper and must be decouplable:
    #   phi -- the write nonlinearity inside the compression loss
    #          ||phi(M w) - alpha||^2 (Eq. 4). The paper never states the
    #          baseline phi; ln_silu here is a documented reconstruction, not a
    #          paper value. phi = identity is a paper ablation (reduces the
    #          two passes to the gated delta rule).
    #   f   -- the inter-pass map applied to the key readout before the value
    #          pass. The method text defines f = normalized SiLU (Eq. 14), but
    #          the reported experimental baseline uses LN-SiLU (L2-SiLU is a
    #          listed modification: 10.98 vs 10.87). So ln_silu is the
    #          better-supported f for reproducing the tables.
    # The paper varies f independently and ablates phi separately, so a single
    # shared knob cannot express those. phi_activation / f_activation override
    # `activation` independently; None (default) ties both to `activation` for
    # backward compatibility.
    phi_activation: Optional[str] = None
    f_activation: Optional[str] = None
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
    # solver for the input-conditioned affine chunk kernel. "solve": per-slot
    # cuSOLVER unit-triangular trsm (default, reference). "neumann": all-slots
    # nilpotent fixed-point (C-1 matmul sweeps, shared coupling) -- bit-exact,
    # graph-break-free so torch.compile can fuse it (the speed path).
    trellis_ic_solver: str = "solve"
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
    # --- training semantics: the two independent axes the deprecated
    # exact_inner flag conflated. State staleness (which state feeds z_t) and
    # outer-gradient order (whether the outer loss differentiates through the
    # inner correction, du/dz) are DIFFERENT approximations.
    #   trellis_state_mode:
    #     "sequential_current"  z_t = M_{t-1} @ w_t from the live state.
    #                           Requires chunk_size=1 (per-head beta), or any
    #                           chunk_size with per-slot beta (which always
    #                           dispatches sequentially).
    #     "chunk_start_stale"   z_t = M_0 @ w_t from the chunk-start state --
    #                           the paper's chunked training scheme. Requires
    #                           chunk_refine=0 and per-head beta; chunk_size=1
    #                           coincides with sequential_current.
    #   trellis_outer_gradient_mode:
    #     "full_bilevel"          keep z in the graph: the outer loss retains
    #                             du/dz (the paper's stated bilevel objective).
    #     "first_order_detached"  detach z before the inner VJP (the historical
    #                             fast mode; u still carries the alpha grad).
    # None (default) resolves through resolve_training_semantics(). A legacy
    # explicit exact_inner maps True->full_bilevel / False->first_order_detached
    # ONLY on the sequential path where the flag was ever honored; any chunked
    # configuration resolves to first_order_detached regardless of exact_inner,
    # because the chunked backends never received the flag (they detached
    # unconditionally).
    trellis_state_mode: Optional[str] = None
    trellis_outer_gradient_mode: Optional[str] = None
    # DEPRECATED: use trellis_state_mode / trellis_outer_gradient_mode. Kept
    # only for checkpoint/config compatibility and read only by the resolver.
    exact_inner: Optional[bool] = None
    # chunk_size = 1 is the pure sequential recurrence. In paper terms this is
    # the "fully non-linear recurrence" ablation (chunk B=1), which the paper
    # reports at slightly BETTER perplexity (10.75 vs 10.87) than its headline
    # result -- it is NOT the reported baseline. The reported baseline uses the
    # stale-gradient chunk approximation with C>1, but the paper never states
    # the baseline C, so an exact numerical reproduction of it is not possible
    # from the published text (see reports/trellis_paper_fidelity.md).
    chunk_size: int = 1  # 1 = pure sequential (paper's B=1 ablation)
    chunk_refine: int = 0  # intra-chunk z refinement passes (faithful chunkwise)
    post_gate: bool = True  # SwiGLU-style post gate on mixer output
    forget_gate: bool = True  # if False, beta is forced to 1 (no decay)

    # --- training / misc ---
    dropout: float = 0.0
    dtype: str = "bf16"  # ["bf16","fp16","fp32"]
    mlp_ratio: float = 4.0
    tie_embeddings: bool = True

    def __post_init__(self):
        _acts = (
            "silu",
            "ln_silu",
            "norm_silu",
            "l2_silu",
            "softmax",
            "identity",
            "scaled_identity",
        )
        assert self.activation in _acts, self.activation
        assert (
            self.phi_activation is None or self.phi_activation in _acts
        ), self.phi_activation
        assert (
            self.f_activation is None or self.f_activation in _acts
        ), self.f_activation
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
        assert self.trellis_state_mode in (
            None,
            "sequential_current",
            "chunk_start_stale",
        ), self.trellis_state_mode
        assert self.trellis_outer_gradient_mode in (
            None,
            "full_bilevel",
            "first_order_detached",
        ), self.trellis_outer_gradient_mode
        assert self.exact_inner in (None, True, False), self.exact_inner
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

    def resolve_training_semantics(self, warn: bool = True) -> dict:
        """Resolve the training semantics this config actually selects.

        Returns a manifest-ready dict with keys write_path, state_mode,
        outer_gradient_mode, legacy_exact_inner and notes. Raises ValueError
        for combinations no backend can honor -- a request must fail loudly,
        never silently downgrade. Legacy configs (exact_inner set, new axes
        unset) resolve to what the code ACTUALLY did: the chunked backends
        never received exact_inner, so any chunked legacy config is
        first_order_detached even if it stored exact_inner=True.
        """
        import warnings as _warnings

        notes: List[str] = []

        def _note(msg: str) -> None:
            notes.append(msg)
            if warn:
                _warnings.warn(msg, stacklevel=3)

        if self.trellis_write_mode == "input_conditioned":
            # affine-in-M write: exact forward recurrence reconstruction and
            # exact outer gradient by construction (no inner VJP exists), for
            # both the sequential and the chunked kernel.
            return {
                "write_path": "input_conditioned_affine",
                "state_mode": "sequential_current",
                "outer_gradient_mode": "exact_affine",
                "legacy_exact_inner": self.exact_inner,
                "notes": notes,
            }

        per_head = self.beta_mode == "scalar_per_head"
        chunk_dispatch = self.chunk_size > 1 and per_head

        state = self.trellis_state_mode
        if state == "sequential_current":
            if chunk_dispatch:
                raise ValueError(
                    "trellis_state_mode='sequential_current' requires "
                    f"chunk_size=1 with per-head beta (got chunk_size="
                    f"{self.chunk_size}); use 'chunk_start_stale' for "
                    "chunked training"
                )
        elif state == "chunk_start_stale":
            if self.chunk_refine != 0:
                raise ValueError(
                    "trellis_state_mode='chunk_start_stale' requires "
                    f"chunk_refine=0 (got {self.chunk_refine})"
                )
            if self.chunk_size > 1 and not per_head:
                raise ValueError(
                    "the chunked path supports per-head beta only "
                    f"(beta_mode={self.beta_mode!r})"
                )
        else:  # derive from the legacy knobs
            if not chunk_dispatch:
                state = "sequential_current"
                if self.chunk_size > 1:
                    _note(
                        "chunk_size>1 with per-slot beta dispatches to the "
                        "sequential path; resolved state_mode="
                        "'sequential_current'"
                    )
            elif self.chunk_refine == 0:
                state = "chunk_start_stale"
            else:
                # legacy diagnostic path: refined (possibly exact) forward,
                # unconditionally first-order backward
                state = "chunk_start_refined"

        grad = self.trellis_outer_gradient_mode
        if grad is None:
            if self.exact_inner is not None:
                _note(
                    "TrellisConfig.exact_inner is deprecated; set "
                    "trellis_outer_gradient_mode explicitly"
                )
                if state == "sequential_current":
                    grad = (
                        "full_bilevel" if self.exact_inner else "first_order_detached"
                    )
                else:
                    grad = "first_order_detached"
                    if self.exact_inner:
                        _note(
                            "exact_inner=True never reached the chunked "
                            "backends (they detach z unconditionally); "
                            "resolved to first_order_detached. Set "
                            "trellis_outer_gradient_mode='full_bilevel' to "
                            "request the bilevel chunk reference."
                        )
            elif state == "sequential_current":
                grad = "full_bilevel"
            else:
                grad = "first_order_detached"
                _note(
                    "chunked Trellis without an explicit "
                    "trellis_outer_gradient_mode resolves to "
                    "first_order_detached (the historical behavior); set "
                    "the mode explicitly to silence this warning"
                )
        if grad == "full_bilevel" and state == "chunk_start_refined":
            raise ValueError(
                "full_bilevel is not implemented for the chunk_refine path; "
                "use chunk_refine=0 (chunk_start_stale) or chunk_size=1"
            )
        return {
            "write_path": "nonlinear_phi",
            "state_mode": state,
            "outer_gradient_mode": grad,
            "legacy_exact_inner": self.exact_inner,
            "notes": notes,
        }

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrellisConfig":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})

    # Named research profiles. Kept as plain class attributes (no field
    # annotation) so the dataclass machinery ignores them.
    _PROFILES = {
        # the correctness default: live sequential state, full bilevel
        # gradient, reference PyTorch backend
        "correctness_oracle": dict(
            trellis_state_mode="sequential_current",
            trellis_outer_gradient_mode="full_bilevel",
            chunk_size=1,
        ),
        # the paper-style chunk approximation trained under its literal
        # end-to-end bilevel reading; chunk size and phi must be explicit
        # because the paper states neither
        "paper_chunk_full": dict(
            trellis_state_mode="chunk_start_stale",
            trellis_outer_gradient_mode="full_bilevel",
        ),
        # the historical fast mode, now explicitly labeled
        "fast_first_order": dict(
            trellis_state_mode="chunk_start_stale",
            trellis_outer_gradient_mode="first_order_detached",
        ),
        # the repaired SiLU candidate recipe under bilevel semantics. The
        # archived first-order runs of this candidate TIED f to SiLU through
        # the single activation knob; this profile pins the paper-specified
        # f = LN-SiLU and makes phi explicit, so the tie cannot recur.
        "repaired_silu_full": dict(
            trellis_state_mode="chunk_start_stale",
            trellis_outer_gradient_mode="full_bilevel",
            phi_activation="silu",
            f_activation="ln_silu",
            alpha_mode="linear",
            value_readout_act="none",
            output_path="paper",
            use_short_conv_v=True,
            n_slots=48,
            beta_init=0.99,
            gamma_init=0.005,
        ),
    }
    _PROFILE_REQUIRED = {
        "paper_chunk_full": ("chunk_size", "phi_activation"),
        "fast_first_order": ("chunk_size",),
        "repaired_silu_full": ("chunk_size",),
    }

    @classmethod
    def profile(cls, name: str, **overrides) -> "TrellisConfig":
        """Build a config from a named research profile.

        Profiles pin the training-semantics axes explicitly; experiment
        drivers should require one instead of inheriting silent defaults.
        Knobs listed in _PROFILE_REQUIRED must be passed by the caller.
        """
        if name not in cls._PROFILES:
            raise ValueError(
                f"unknown profile {name!r}; known: {sorted(cls._PROFILES)}"
            )
        missing = [k for k in cls._PROFILE_REQUIRED.get(name, ()) if k not in overrides]
        if missing:
            raise ValueError(f"profile {name!r} requires explicit {missing}")
        kwargs = dict(cls._PROFILES[name])
        kwargs.update(overrides)
        return cls(**kwargs)

    @classmethod
    def faithful_baseline(cls, **overrides) -> "TrellisConfig":
        """Config set to the paper's *reported-baseline* choices, to the extent
        the text pins them down. It does NOT claim bit-exact reproduction --
        several details (the write phi, the baseline chunk size C, the gamma
        source/granularity) are simply unspecified by the paper and are marked
        here as reconstructions. See reports/trellis_paper_fidelity.md.

        Set relative to the defaults:
          - f_activation = ln_silu       : the reported baseline's inter-pass f
                                           (the method text's L2-SiLU is a listed
                                           modification, not the baseline).
          - value_readout_act = ln_silu  : the paper's final phi on the value
                                           read, y = phi(M^T r) (off by default).
          - output_path = "paper"        : Fig. 1 shell (Trellis -> Norm ->
                                           GeLU-gated branch -> Linear).
          - trellis_beta_min = 1e-3      : the paper's forget gate spans (0,1)
                                           and can erase memory; do not floor it
                                           near 1.
          - phi_activation left None (ties to `activation`=ln_silu) because the
            paper never defines the baseline phi -- override to sweep it.
        chunk_size is left at the default (1 = the B=1 ablation); the reported
        baseline's stale C>1 is unspecified, so pick C explicitly if reproducing
        the throughput regime.
        """
        cfg = dict(
            f_activation="ln_silu",
            value_readout_act="ln_silu",
            output_path="paper",
            trellis_beta_min=1e-3,
        )
        cfg.update(overrides)
        return cls(**cfg)

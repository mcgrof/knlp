"""TrellisMixer — two-pass bounded-memory sequence mixing sublayer.

Replaces self-attention. Produces its output directly from the compressed
memory state (no [B,H,T,T] mask). Returns the sublayer delta; the block adds
the residual.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TrellisConfig
from .activations import get_activation
from .trellis_memory import (
    run_trellis_memory,
    run_trellis_memory_chunked,
    trellis_chunk_decay,
    run_trellis_memory_chunked_state_evolution,
    run_trellis_memory_chunked_batched_readout,
)

try:
    from .trellis_triton import TrellisStateEvolutionTriton, HAS_TRITON
except Exception:  # pragma: no cover
    HAS_TRITON = False

# Head-on bf16 test (Codex-advised recipe): round the write/read/alpha inputs to
# bf16 while the decay (beta->P/rmat), gamma, the resident Mstate and all the
# LN-SiLU reductions stay fp32 (the kernel accumulates in fp32 internally). This
# is the autocast-bf16 regime for Trellis without a fully-bf16 operator. Off by
# default; the ladder/test flips it to compare PPL vs fp32.
BF16_INPUTS = False


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return n * self.w


class CausalDWConv1d(nn.Module):
    """Depthwise causal conv over time on a [B,H,T,D] tensor (per-channel)."""

    def __init__(self, channels, kernel):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(channels, channels, kernel, groups=channels, bias=True)

    def forward(self, x):  # x: [B,H,T,D]
        B, H, T, D = x.shape
        xt = x.permute(0, 1, 3, 2).reshape(B, H * D, T)  # [B, C=H*D, T]
        xt = F.pad(xt, (self.kernel - 1, 0))  # left pad = causal
        out = self.conv(xt)  # [B, C, T]
        out = out.reshape(B, H, D, T).permute(0, 1, 3, 2)  # [B,H,T,D]
        return out


class TrellisMixer(nn.Module):
    def __init__(self, cfg: TrellisConfig, layer_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        H, D, M, d = cfg.n_heads, cfg.d_head, cfg.n_slots, cfg.d_model
        self.H, self.D, self.M = H, D, M
        self.norm = RMSNorm(d)
        self.q_proj = nn.Linear(d, H * D, bias=False)
        self.k_proj = nn.Linear(d, H * D, bias=False)
        self.v_proj = nn.Linear(d, H * D, bias=False)
        self.alpha_proj = nn.Linear(d, H * M, bias=False)
        if cfg.trellis_retention_mode == "token_proj":
            beta_out = H if cfg.beta_mode == "scalar_per_head" else H * M
            self.beta_proj = nn.Linear(d, beta_out, bias=True)
        else:
            self.beta_proj = None
        self.retention_theta = None
        self.register_buffer("retention_beta_fixed", None)
        self.reset_beta_parameters()
        gate_dim = d
        if cfg.trellis_update_gate_context_mode == "current_prev":
            gate_dim = 2 * d
        if cfg.update_gate_mode == "scalar":
            self.update_gate_proj = nn.Linear(gate_dim, H, bias=True)
        elif cfg.update_gate_mode == "channel":
            self.update_gate_proj = nn.Linear(gate_dim, H * M, bias=True)
        else:
            self.update_gate_proj = None
        self.reset_update_gate_bias()
        # input-conditioned write gate a(x_t) in R^{H*M}: per-slot gain from the
        # token input, used only when trellis_write_mode == "input_conditioned".
        # Bias init makes a≡1 at start (exact gated delta rule) via the chosen
        # gate activation; weights zeroed so the recurrence starts linear.
        self.write_gate_proj = None
        if cfg.trellis_write_mode == "input_conditioned":
            gate_out = H if cfg.trellis_input_gate_scope == "scalar" else H * M
            self.write_gate_proj = nn.Linear(d, gate_out, bias=True)
            self.reset_write_gate_bias()
        # gamma positive per head via softplus(raw); init so softplus(raw)=gamma_init
        raw0 = math.log(math.expm1(cfg.gamma_init))
        self.gamma_raw = nn.Parameter(torch.full((H,), raw0))
        self.out_proj = nn.Linear(H * D, d, bias=False)
        self.output_path = cfg.output_path
        self.post_gate = cfg.post_gate
        if cfg.output_path == "paper":
            # Fig 1 order: Trellis -> PostNorm -> GeLU gate -> out_proj. The gate
            # lives in the inner_dim (H*D) space, applied BEFORE out_proj.
            self.post_norm = RMSNorm(H * D)
            self.gate_in = nn.Linear(d, H * D, bias=False)
        elif cfg.post_gate:
            self.gate_proj = nn.Linear(d, d, bias=False)
        # final phi on the value-pass readout y = phi(M^T r) (paper); None=legacy
        self.value_readout_act = (
            get_activation(cfg.value_readout_act)
            if cfg.value_readout_act != "none"
            else None
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.use_conv = cfg.use_short_conv_qk
        if self.use_conv:
            self.q_conv = CausalDWConv1d(H * D, cfg.conv_kernel)
            self.k_conv = CausalDWConv1d(H * D, cfg.conv_kernel)
        self.use_v_conv = cfg.use_short_conv_v
        if self.use_v_conv:
            self.v_conv = CausalDWConv1d(H * D, cfg.conv_kernel)
        self.phi = get_activation(cfg.activation)
        self.f = get_activation(cfg.activation)
        self.alpha_act = get_activation(cfg.alpha_mode)
        self.value_alpha_correction_raw = None
        self.value_read_query_gate_proj = None
        self.value_address_proj = None
        if cfg.trellis_value_alpha_mode in (
            "shared_plus_key_correction",
            "shared_plus_key_correction_detached",
            "shared_plus_local_key_correction",
            "shared_plus_local_key_correction_detached",
            "shared_plus_prev_alpha_correction",
            "shared_plus_prev_alpha_correction_detached",
            "shared_plus_prev_key_correction",
            "shared_plus_prev_key_correction_detached",
        ):
            raw = self._value_alpha_correction_init_raw()
            self.value_alpha_correction_raw = nn.Parameter(torch.full((H,), raw))
        if self._needs_value_address():
            self.value_address_proj = nn.Linear(d, H * M, bias=False)
        if cfg.trellis_value_read_query_mode in (
            "alpha_residual_gate",
            "alpha_residual_gate_detached",
        ):
            self.value_read_query_gate_proj = nn.Linear(d, H, bias=True)

    def _needs_value_address(self) -> bool:
        cfg = self.cfg
        return cfg.trellis_value_alpha_mode in (
            "shared_plus_local_key_correction",
            "shared_plus_local_key_correction_detached",
        ) or cfg.trellis_value_read_query_mode in (
            "local_key_address",
            "local_key_address_detached",
        )

    def _value_alpha_correction_init_raw(self) -> float:
        cfg = self.cfg
        max_scale = float(cfg.trellis_value_alpha_correction_max)
        init = float(cfg.trellis_value_alpha_correction_init)
        # Use a near-zero finite logit for exact-zero requests so gradients can
        # still move the correction scale if the branch has signal.
        unit = max(1e-8, min(1.0 - 1e-8, init / max_scale))
        return math.log(unit / (1.0 - unit))

    def effective_gamma(self, gamma: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if (
            cfg.trellis_update_stabilizer
            in (
                "layerwise_gamma",
                "innovation_rms_cap_plus_layer0_gamma",
                "innovation_rms_cap_plus_layerwise_gamma",
            )
            and self.layer_idx == 0
            and cfg.trellis_layer0_gamma_mult != 1.0
        ):
            return gamma * float(cfg.trellis_layer0_gamma_mult)
        return gamma

    def reset_beta_bias(self):
        """Set beta_proj bias = logit(beta_init) so the forget gate STARTS near
        beta_init, not 0.5. beta~0.5 (zero bias) is ~1-token memory half-life;
        the paper wants beta near 1 (ChatGPT-Pro suspect #2). Must be called
        AFTER the model-wide _init_weights (which zeros all Linear biases),
        else it is silently clobbered."""
        if self.beta_proj is None:
            return
        with torch.no_grad():
            b0 = math.log(self.cfg.beta_init / (1.0 - self.cfg.beta_init))
            self.beta_proj.bias.fill_(b0)

    def reset_beta_parameters(self):
        cfg = self.cfg
        if cfg.trellis_retention_mode == "token_proj":
            self.reset_beta_bias()
            return
        init = self._retention_init_values(cfg.trellis_retention_mode)
        if cfg.trellis_retention_mode == "fixed_beta":
            self.retention_beta_fixed = init
            return
        self.retention_theta = nn.Parameter(self._beta_to_theta(init))

    def _layer_position(self) -> float:
        denom = max(1, self.cfg.n_layers - 1)
        return float(self.layer_idx) / float(denom)

    @staticmethod
    def _beta_to_tau(beta: float) -> float:
        return 1.0 / max(1e-9, 1.0 - float(beta))

    @classmethod
    def _logspace_betas(cls, low: float, high: float, count: int) -> torch.Tensor:
        if count <= 1:
            return torch.tensor([float(high)], dtype=torch.float32)
        low_tau = cls._beta_to_tau(low)
        high_tau = cls._beta_to_tau(high)
        taus = torch.logspace(
            math.log10(low_tau),
            math.log10(high_tau),
            steps=count,
            dtype=torch.float32,
        )
        return 1.0 - 1.0 / taus

    def _layer_short_to_long_beta(self) -> float:
        s = self._layer_position()
        low_tau = self._beta_to_tau(0.95)
        high_tau = self._beta_to_tau(0.995)
        tau = math.exp(math.log(low_tau) * (1.0 - s) + math.log(high_tau) * s)
        return 1.0 - 1.0 / tau

    def _layer_head_logspace_betas(self, count: int) -> torch.Tensor:
        s = self._layer_position()
        low_tau0 = self._beta_to_tau(0.95)
        low_tau1 = self._beta_to_tau(0.975)
        high_tau0 = self._beta_to_tau(0.98)
        high_tau1 = self._beta_to_tau(0.999)
        low_tau = math.exp(math.log(low_tau0) * (1.0 - s) + math.log(low_tau1) * s)
        high_tau = math.exp(math.log(high_tau0) * (1.0 - s) + math.log(high_tau1) * s)
        low_beta = 1.0 - 1.0 / low_tau
        high_beta = 1.0 - 1.0 / high_tau
        return self._logspace_betas(low_beta, high_beta, count)

    def _scheduled_betas(self, count: int) -> torch.Tensor:
        cfg = self.cfg
        if cfg.trellis_beta_init_schedule == "flat_099":
            values = torch.full(
                (count,),
                float(cfg.trellis_beta_init),
                dtype=torch.float32,
            )
        elif cfg.trellis_beta_init_schedule == "layer_short_to_long":
            values = torch.full(
                (count,),
                self._layer_short_to_long_beta(),
                dtype=torch.float32,
            )
        elif cfg.trellis_beta_init_schedule == "head_logspace":
            values = self._logspace_betas(0.95, 0.999, count)
        elif cfg.trellis_beta_init_schedule == "layer_head_logspace":
            values = self._layer_head_logspace_betas(count)
        else:  # pragma: no cover - guarded by config validation
            raise ValueError(cfg.trellis_beta_init_schedule)
        return values.clamp(cfg.trellis_beta_min + 1e-6, cfg.trellis_beta_max - 1e-6)

    def _retention_init_values(self, mode: str) -> torch.Tensor:
        if mode in ("fixed_beta", "learned_per_head"):
            return self._scheduled_betas(self.H)
        if mode == "learned_per_channel":
            return self._scheduled_betas(self.M)
        if mode == "learned_per_head_channel":
            return self._scheduled_betas(self.H).view(self.H, 1).expand(self.H, self.M)
        raise ValueError(mode)

    def _beta_to_theta(self, beta: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        unit = (beta.float() - cfg.trellis_beta_min) / (
            cfg.trellis_beta_max - cfg.trellis_beta_min
        )
        unit = unit.clamp(1e-6, 1.0 - 1e-6)
        return torch.log(unit / (1.0 - unit))

    def retention_beta_values(self) -> torch.Tensor | None:
        cfg = self.cfg
        mode = cfg.trellis_retention_mode
        if mode == "token_proj":
            return None
        if mode == "fixed_beta":
            return self.retention_beta_fixed
        if self.retention_theta is None:  # pragma: no cover - defensive
            raise RuntimeError("learned retention requested without theta")
        unit = torch.sigmoid(self.retention_theta.float())
        beta_range = cfg.trellis_beta_max - cfg.trellis_beta_min
        return cfg.trellis_beta_min + beta_range * unit

    def _expand_retention_beta(
        self,
        beta_values: torch.Tensor,
        batch: int,
        seq_len: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        values = beta_values.to(dtype=dtype)
        if values.dim() == 1 and values.numel() == self.H:
            return values.view(1, self.H, 1, 1).expand(batch, self.H, seq_len, 1)
        if values.dim() == 1 and values.numel() == self.M:
            return values.view(1, 1, 1, self.M).expand(batch, self.H, seq_len, self.M)
        if values.shape == (self.H, self.M):
            return values.view(1, self.H, 1, self.M).expand(
                batch,
                self.H,
                seq_len,
                self.M,
            )
        raise ValueError(f"unsupported retention beta shape {tuple(values.shape)}")

    def compute_beta(self, h_float: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        B, T, _ = h_float.shape
        if cfg.trellis_retention_mode == "token_proj":
            if self.beta_proj is None:  # pragma: no cover - defensive
                raise RuntimeError("token_proj retention requires beta_proj")
            if cfg.beta_mode == "scalar_per_head":
                beta = (
                    torch.sigmoid(self.beta_proj(h_float))
                    .view(B, T, self.H, 1)
                    .permute(0, 2, 1, 3)
                )
            else:
                beta = torch.sigmoid(self._heads(self.beta_proj(h_float), self.M))
        else:
            beta_values = self.retention_beta_values()
            if beta_values is None:  # pragma: no cover - defensive
                raise RuntimeError("static retention values are unavailable")
            beta = self._expand_retention_beta(beta_values, B, T, h_float.dtype)
        if not cfg.forget_gate:
            beta = torch.ones_like(beta)
        return beta

    def reset_update_gate_bias(self):
        if self.update_gate_proj is None:
            return
        with torch.no_grad():
            floor = float(self.cfg.trellis_update_gate_floor)
            p = (float(self.cfg.update_gate_init) - floor) / (1.0 - floor)
            b0 = math.log(p / (1.0 - p))
            self.update_gate_proj.weight.zero_()
            self.update_gate_proj.bias.fill_(b0)

    def reset_write_gate_bias(self):
        """Init the input-conditioned write gate so a(x_t) ≡ 1 at start, which
        makes u_t = 1⊙z − alpha = M@w − alpha — the exact (gated) delta rule.
        Weights zeroed; bias solves act(bias)=1 for the chosen gate activation."""
        if self.write_gate_proj is None:
            return
        act = self.cfg.trellis_input_gate_act
        if act == "softplus":
            b0 = math.log(math.expm1(1.0))  # softplus(b0) = 1
        elif act == "sigmoid":
            b0 = 0.0  # sigmoid(0) = 0.5; readout scaled x2 below -> a≈1
        else:  # identity
            b0 = 1.0
        with torch.no_grad():
            self.write_gate_proj.weight.zero_()
            self.write_gate_proj.bias.fill_(b0)

    def _write_gate(self, h_float: torch.Tensor) -> torch.Tensor | None:
        """Input-conditioned gate a(x_t) in [B,H,T,M], fp32. "scalar" scope
        projects one gain per head/token and broadcasts to all M slots."""
        if self.write_gate_proj is None:
            return None
        if self.cfg.trellis_input_gate_scope == "scalar":
            logits = self._heads(self.write_gate_proj(h_float), 1)  # [B,H,T,1]
        else:
            logits = self._heads(self.write_gate_proj(h_float), self.M)  # [B,H,T,M]
        act = self.cfg.trellis_input_gate_act
        if act == "softplus":
            gate = F.softplus(logits)
        elif act == "sigmoid":
            gate = 2.0 * torch.sigmoid(logits)  # range (0,2), init a≈1
        else:
            gate = logits  # identity
        if self.cfg.trellis_input_gate_scope == "scalar":
            gate = gate.expand(-1, -1, -1, self.M)  # broadcast to all slots
        return gate

    def reset_value_read_query_gate_bias(self):
        if self.value_read_query_gate_proj is None:
            return
        with torch.no_grad():
            max_gate = float(self.cfg.trellis_value_read_query_gate_max)
            p = float(self.cfg.trellis_value_read_query_gate_init) / max_gate
            p = max(1e-8, min(1.0 - 1e-8, p))
            b0 = math.log(p / (1.0 - p))
            self.value_read_query_gate_proj.weight.zero_()
            self.value_read_query_gate_proj.bias.fill_(b0)

    def _update_gate_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(logits)
        floor = float(self.cfg.trellis_update_gate_floor)
        if floor != 0.0:
            gate = floor + (1.0 - floor) * gate
        return gate

    def _gate_for_memory(
        self,
        update_gate: torch.Tensor | None,
        memory: str,
    ) -> torch.Tensor | None:
        if update_gate is None:
            return None
        layer_mode = self.cfg.trellis_update_gate_layer_mode
        if layer_mode == "layer0" and self.layer_idx != 0:
            return None
        if layer_mode == "not_layer0" and self.layer_idx == 0:
            return None
        lower_half_cut = max(1, self.cfg.n_layers // 2)
        if layer_mode == "lower_half" and self.layer_idx >= lower_half_cut:
            return None
        if layer_mode == "upper_half" and self.layer_idx < lower_half_cut:
            return None
        target = self.cfg.trellis_update_gate_target
        if target == "both" or target == memory:
            return update_gate
        return None

    @staticmethod
    def _diag_tensor_stats(t: torch.Tensor | None) -> dict[str, float] | None:
        if t is None:
            return None
        with torch.no_grad():
            x = t.detach().float()
            return {
                "mean": float(x.mean().item()),
                "std": float(x.std(unbiased=False).item()) if x.numel() > 1 else 0.0,
                "rms": float(x.pow(2).mean().sqrt().item()),
                "absmax": float(x.abs().max().item()),
                "min": float(x.min().item()),
                "max": float(x.max().item()),
            }

    def _diag_gate_stats(
        self,
        update_gate: torch.Tensor | None,
        query_index: int,
    ) -> dict[str, object] | None:
        if update_gate is None:
            return None
        stats: dict[str, object] = self._diag_tensor_stats(update_gate) or {}
        with torch.no_grad():
            g = update_gate.detach().float()
            stats["frac_lt_0_2"] = float((g < 0.2).float().mean().item())
            stats["frac_gt_0_95"] = float((g > 0.95).float().mean().item())
            stats["query"] = self._diag_tensor_stats(g[:, :, query_index, :])
        return stats

    def _base_forward_diag(
        self,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        update_gate: torch.Tensor | None,
        key_update_gate: torch.Tensor | None,
        value_update_gate: torch.Tensor | None,
        seq_len: int,
        backend: str,
    ) -> dict[str, object]:
        query_index = max(0, seq_len - 2)
        return {
            "layer": self.layer_idx,
            "backend": backend,
            "update_gate_layer_mode": self.cfg.trellis_update_gate_layer_mode,
            "update_gate_target": self.cfg.trellis_update_gate_target,
            "update_gate_context_mode": self.cfg.trellis_update_gate_context_mode,
            "beta": self._diag_tensor_stats(beta),
            "effective_gamma": self._diag_tensor_stats(gamma),
            "update_gate": self._diag_gate_stats(update_gate, query_index),
            "key_update_gate": self._diag_gate_stats(key_update_gate, query_index),
            "value_update_gate": self._diag_gate_stats(value_update_gate, query_index),
        }

    def _set_forward_diag(self, diag: dict[str, object]) -> None:
        self.last_trellis_diag = diag

    def _stack_key_value_gates(self, update_gate: torch.Tensor | None):
        if update_gate is None:
            return None
        key_gate = self._gate_for_memory(update_gate, "key")
        value_gate = self._gate_for_memory(update_gate, "value")
        if key_gate is None and value_gate is None:
            return None
        if key_gate is None:
            key_gate = torch.ones_like(update_gate)
        if value_gate is None:
            value_gate = torch.ones_like(update_gate)
        B, H, T, W = update_gate.shape
        return (
            torch.stack((key_gate, value_gate), dim=0)
            .reshape(2 * B, H, T, W)
            .contiguous()
        )

    def _heads(self, x, last):  # [B,T,H*last] -> [B,H,T,last]
        B, T, _ = x.shape
        return x.view(B, T, self.H, last).permute(0, 2, 1, 3)

    @staticmethod
    def _previous_code(x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] <= 1:
            return x
        return torch.cat((x[:, :, :1], x[:, :, :-1]), dim=2)

    def _update_gate_context(self, h_float: torch.Tensor) -> torch.Tensor:
        mode = self.cfg.trellis_update_gate_context_mode
        if mode == "current":
            return h_float
        prev = torch.cat((h_float[:, :1], h_float[:, :-1]), dim=1)
        if mode == "prev":
            return prev
        if mode == "current_prev":
            return torch.cat((h_float, prev), dim=-1)
        raise ValueError(mode)

    def _value_address(self, h_float: torch.Tensor) -> torch.Tensor | None:
        if self.value_address_proj is None:
            return None
        address = self._heads(self.value_address_proj(h_float), self.M)
        return self.alpha_act(address)

    def _value_alpha(
        self,
        shared_alpha: torch.Tensor,
        key_code: torch.Tensor,
        local_key_address: torch.Tensor | None = None,
    ):
        cfg = self.cfg
        mix = float(cfg.trellis_value_alpha_mix)
        if cfg.trellis_value_alpha_mode == "shared" or mix == 0.0:
            return shared_alpha
        if cfg.trellis_value_alpha_mode in (
            "shared_plus_key_correction",
            "shared_plus_key_correction_detached",
            "shared_plus_local_key_correction",
            "shared_plus_local_key_correction_detached",
            "shared_plus_prev_alpha_correction",
            "shared_plus_prev_alpha_correction_detached",
            "shared_plus_prev_key_correction",
            "shared_plus_prev_key_correction_detached",
        ):
            mode = cfg.trellis_value_alpha_mode
            if mode.startswith("shared_plus_prev_alpha"):
                target = self._previous_code(shared_alpha)
            elif mode.startswith("shared_plus_prev_key"):
                target = self._previous_code(key_code)
            elif mode.startswith("shared_plus_local_key"):
                if local_key_address is None:  # pragma: no cover - guarded
                    raise RuntimeError("local key correction missing address")
                target = self._previous_code(local_key_address)
            else:
                target = key_code
            if mode.endswith("_detached"):
                target = target.detach()
            if self.value_alpha_correction_raw is None:  # pragma: no cover
                raise RuntimeError("value alpha correction mode missing scale")
            scale = float(cfg.trellis_value_alpha_correction_max) * torch.sigmoid(
                self.value_alpha_correction_raw.float()
            )
            scale = scale.to(device=shared_alpha.device, dtype=shared_alpha.dtype)
            scale = scale.view(1, self.H, 1, 1) * mix
            return shared_alpha + scale * (target - shared_alpha)
        target = key_code
        if cfg.trellis_value_alpha_mode == "key_readout_detached":
            target = target.detach()
        if mix == 1.0:
            return target
        return shared_alpha * (1.0 - mix) + target * mix

    def _value_read_query(
        self,
        key_readout: torch.Tensor,
        shared_alpha: torch.Tensor,
        h_float: torch.Tensor,
        local_key_address: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cfg = self.cfg
        if cfg.trellis_value_read_query_mode == "key_readout":
            return key_readout, None
        if cfg.trellis_value_read_query_mode in (
            "local_key_address",
            "local_key_address_detached",
        ):
            if local_key_address is None:  # pragma: no cover - guarded
                raise RuntimeError("local key read query missing address")
            target = local_key_address
            if cfg.trellis_value_read_query_mode.endswith("_detached"):
                target = target.detach()
            return target, None
        if self.value_read_query_gate_proj is None:  # pragma: no cover
            raise RuntimeError("value read query gate mode missing projection")
        B, T, _ = h_float.shape
        gate = torch.sigmoid(self.value_read_query_gate_proj(h_float)).view(
            B, T, self.H, 1
        ).permute(0, 2, 1, 3) * float(cfg.trellis_value_read_query_gate_max)
        gate = gate.to(device=key_readout.device, dtype=key_readout.dtype)
        target = shared_alpha
        if cfg.trellis_value_read_query_mode.endswith("_detached"):
            target = target.detach()
        return key_readout + gate * (target - key_readout), gate

    def forward(self, x, training: bool = True):
        cfg = self.cfg
        B, T, d = x.shape
        h = self.norm(x)
        q = self._heads(self.q_proj(h), self.D)  # [B,H,T,D]
        k = self._heads(self.k_proj(h), self.D)
        v = self._heads(self.v_proj(h), self.D)
        if self.use_conv:
            q = self.q_conv(q)
            k = self.k_conv(k)
        if self.use_v_conv:
            v = self.v_conv(v)
        if cfg.write_l2norm:
            # DeltaNet-style L2 normalization of the write vectors (keys) and the
            # key-pass query, over head_dim, to bound gamma*||w||^2.
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            v = F.normalize(v, dim=-1)
        alpha = self._heads(self.alpha_proj(h), self.M)  # [B,H,T,M]
        alpha = self.alpha_act(alpha)
        # Decay (beta) and inner-LR (gamma) stay fp32 regardless of any outer
        # autocast: bf16 decay precision is poor (Codex) and the head-on bf16
        # test kept these fp32 (q/k/v/alpha may be bf16, which it validated).
        with torch.autocast(device_type=x.device.type, enabled=False):
            hf = h.float()
            beta = self.compute_beta(hf)
            gamma = self.effective_gamma(F.softplus(self.gamma_raw.float()))
            update_gate = None
            if cfg.update_gate_mode == "scalar":
                gate_context = self._update_gate_context(hf)
                update_gate = (
                    self._update_gate_from_logits(self.update_gate_proj(gate_context))
                    .view(B, T, self.H, 1)
                    .permute(0, 2, 1, 3)
                )
            elif cfg.update_gate_mode == "channel":
                gate_context = self._update_gate_context(hf)
                update_gate = self._update_gate_from_logits(
                    self._heads(self.update_gate_proj(gate_context), self.M)
                )
            key_update_gate = self._gate_for_memory(update_gate, "key")
            value_update_gate = self._gate_for_memory(update_gate, "value")
            local_key_address = self._value_address(hf)
            write_gate = self._write_gate(hf)  # [B,H,T,M] or None

        # key pass -> yhat; value pass -> y. The Trellis memory runs in fp32 (the
        # chunk recurrence state, LN-SiLU reductions and decay are fp32; bf16
        # inputs are fine -- the head-on test showed bf16 PPL ~ fp32). Output
        # rejoins any outer autocast at out_proj. Chunked stale path when
        # chunk_size>1 and beta is per-head.
        # The fused state-evolution / triton path computes the nonlinear-phi VJP,
        # so the input-conditioned write routes around it. It has its own EXACT
        # chunk kernel (run_trellis_memory_chunked with input_gate, affine
        # forward-substitution): use it when chunk_size>1 and beta is per-head,
        # else the exact sequential path.
        use_chunk = cfg.chunk_size > 1 and beta.shape[-1] == 1 and write_gate is None
        ic_chunk = write_gate is not None and cfg.chunk_size > 1 and beta.shape[-1] == 1
        qf, kf, vf = q.float(), k.float(), v.float()
        af, bf, gf = alpha.float(), beta.float(), gamma.float()
        if BF16_INPUTS:
            # round inputs to bf16 precision (decay bf/gf stay fp32)
            qf, kf, vf, af = (t.bfloat16().float() for t in (qf, kf, vf, af))
        with torch.autocast(device_type=x.device.type, enabled=False):
            if use_chunk and cfg.chunk_refine == 0:
                # Phase-1 fast path: evolve key+value states stacked on a 2x
                # batch axis (neither depends on r), then batched readouts.
                cs = cfg.chunk_size
                P, rmat, _ = trellis_chunk_decay(bf, cs)
                nC = P.shape[2]
                # The fused Triton kernel handles LN-SiLU only in the square
                # memory case because LN reductions are over the true slot count.
                # Pointwise SiLU/identity can pad slots internally, so the
                # Stage-2 winner/control at n_slots=48,d_head=64 can stay fused.
                triton_pointwise = cfg.activation in ("silu", "identity")
                triton_ln_silu = (
                    cfg.activation == "ln_silu" and cfg.n_slots == cfg.d_head
                )
                nvidia_cuda = kf.is_cuda and (
                    getattr(torch.version, "hip", None) is None
                )
                triton_fused_update = cfg.trellis_update_stabilizer != (
                    "delta_ratio_cap"
                )
                fused_backend = (
                    HAS_TRITON
                    and nvidia_cuda
                    and triton_fused_update
                    and (triton_pointwise or triton_ln_silu)
                )
                forward_diag = self._base_forward_diag(
                    beta=bf,
                    gamma=gf,
                    update_gate=update_gate,
                    key_update_gate=key_update_gate,
                    value_update_gate=value_update_gate,
                    seq_len=T,
                    backend=(
                        "triton_state_evolution"
                        if fused_backend
                        else "pytorch_state_evolution"
                    ),
                )
                if local_key_address is not None:
                    forward_diag["value_address"] = self._diag_tensor_stats(
                        local_key_address
                    )
                    forward_diag["value_address_prev"] = self._diag_tensor_stats(
                        self._previous_code(local_key_address)
                    )

                def evolve_state(write_in, alpha_in, P_in, rmat_in, gate_in):
                    if fused_backend:
                        # Fused Triton state-evolution: collapses the nC-chunk
                        # loop into one kernel, gradient-equivalent to the
                        # PyTorch path (z detached -> true-stale).
                        return TrellisStateEvolutionTriton.apply(
                            write_in.contiguous(),
                            alpha_in.contiguous(),
                            P_in.contiguous(),
                            rmat_in.contiguous(),
                            gf,
                            cs,
                            cfg.activation,
                            cfg.trellis_update_stabilizer,
                            cfg.trellis_innovation_rms_cap,
                            gate_in,
                            cfg.residual_update_mix,
                        )
                    M0, u, _, _, _ = run_trellis_memory_chunked_state_evolution(
                        write_in,
                        alpha_in,
                        None,
                        gf,
                        self.phi,
                        cs,
                        P=P_in,
                        rmat=rmat_in,
                        update_gate=gate_in,
                        residual_update_mix=cfg.residual_update_mix,
                        trellis_update_stabilizer=cfg.trellis_update_stabilizer,
                        trellis_innovation_rms_cap=(cfg.trellis_innovation_rms_cap),
                        trellis_delta_ratio_cap=cfg.trellis_delta_ratio_cap,
                        trellis_state_rms_floor=cfg.trellis_state_rms_floor,
                        trellis_stabilizer_detach_scale=(
                            cfg.trellis_stabilizer_detach_scale
                        ),
                    )
                    return M0, u

                keyed_value_alpha = (
                    cfg.trellis_value_alpha_mode != "shared"
                    and cfg.trellis_value_alpha_mix != 0.0
                )
                if not keyed_value_alpha:
                    write2 = (
                        torch.stack((kf, vf), dim=0)
                        .reshape(2 * B, self.H, T, self.D)
                        .contiguous()
                    )
                    alpha2 = (
                        af.unsqueeze(0)
                        .expand(2, B, self.H, T, self.M)
                        .reshape(2 * B, self.H, T, self.M)
                        .contiguous()
                    )
                    P2 = (
                        P.unsqueeze(0)
                        .expand(2, B, self.H, nC, cs, 1)
                        .reshape(2 * B, self.H, nC, cs, 1)
                        .contiguous()
                    )
                    rmat2 = (
                        rmat.unsqueeze(0)
                        .expand(2, B, self.H, nC, cs, cs)
                        .reshape(2 * B, self.H, nC, cs, cs)
                        .contiguous()
                    )
                    update_gate2 = None
                    if update_gate is not None:
                        update_gate2 = self._stack_key_value_gates(update_gate)
                    M0_2, u_2 = evolve_state(write2, alpha2, P2, rmat2, update_gate2)
                    M0_2 = M0_2.view(2, B, self.H, nC, self.M, self.D)
                    u_2 = u_2.view(2, B, self.H, nC, cs, self.M)
                    forward_diag["key_state"] = self._diag_tensor_stats(M0_2[0])
                    forward_diag["value_state"] = self._diag_tensor_stats(M0_2[1])
                    forward_diag["key_update"] = self._diag_tensor_stats(u_2[0])
                    forward_diag["value_update"] = self._diag_tensor_stats(u_2[1])
                    yhat = run_trellis_memory_chunked_batched_readout(
                        kf, qf, M0_2[0], u_2[0], P, rmat, gf, "M_q", cs, T_out=T
                    )
                    r = self.f(yhat)
                    value_read_query, value_read_query_gate = self._value_read_query(
                        r, af, hf, local_key_address
                    )
                    forward_diag["value_read_query_gate"] = self._diag_gate_stats(
                        value_read_query_gate, max(0, T - 2)
                    )
                    forward_diag["value_read_query"] = self._diag_tensor_stats(
                        value_read_query
                    )
                    self._set_forward_diag(forward_diag)
                    y = run_trellis_memory_chunked_batched_readout(
                        vf,
                        value_read_query,
                        M0_2[1],
                        u_2[1],
                        P,
                        rmat,
                        gf,
                        "M_T_r",
                        cs,
                        T_out=T,
                    )
                else:
                    M0_k, u_k = evolve_state(kf, af, P, rmat, key_update_gate)
                    yhat = run_trellis_memory_chunked_batched_readout(
                        kf, qf, M0_k, u_k, P, rmat, gf, "M_q", cs, T_out=T
                    )
                    r = self.f(yhat)
                    value_alpha = self._value_alpha(af, r, local_key_address)
                    M0_v, u_v = evolve_state(
                        vf,
                        value_alpha,
                        P,
                        rmat,
                        value_update_gate,
                    )
                    forward_diag["key_state"] = self._diag_tensor_stats(M0_k)
                    forward_diag["value_state"] = self._diag_tensor_stats(M0_v)
                    forward_diag["key_update"] = self._diag_tensor_stats(u_k)
                    forward_diag["value_update"] = self._diag_tensor_stats(u_v)
                    forward_diag["value_alpha"] = self._diag_tensor_stats(value_alpha)
                    value_read_query, value_read_query_gate = self._value_read_query(
                        r, af, hf, local_key_address
                    )
                    forward_diag["value_read_query_gate"] = self._diag_gate_stats(
                        value_read_query_gate, max(0, T - 2)
                    )
                    forward_diag["value_read_query"] = self._diag_tensor_stats(
                        value_read_query
                    )
                    self._set_forward_diag(forward_diag)
                    y = run_trellis_memory_chunked_batched_readout(
                        vf,
                        value_read_query,
                        M0_v,
                        u_v,
                        P,
                        rmat,
                        gf,
                        "M_T_r",
                        cs,
                        T_out=T,
                    )
            elif use_chunk:
                forward_diag = self._base_forward_diag(
                    beta=bf,
                    gamma=gf,
                    update_gate=update_gate,
                    key_update_gate=key_update_gate,
                    value_update_gate=value_update_gate,
                    seq_len=T,
                    backend="pytorch_chunked_refine",
                )
                cs = cfg.chunk_size
                yhat = run_trellis_memory_chunked(
                    kf,
                    qf,
                    af,
                    bf,
                    gf,
                    self.phi,
                    "M_q",
                    cs,
                    cfg.chunk_refine,
                    update_gate=key_update_gate,
                    residual_update_mix=cfg.residual_update_mix,
                    trellis_update_stabilizer=cfg.trellis_update_stabilizer,
                    trellis_innovation_rms_cap=cfg.trellis_innovation_rms_cap,
                    trellis_delta_ratio_cap=cfg.trellis_delta_ratio_cap,
                    trellis_state_rms_floor=cfg.trellis_state_rms_floor,
                    trellis_stabilizer_detach_scale=(
                        cfg.trellis_stabilizer_detach_scale
                    ),
                )
                r = self.f(yhat)
                value_alpha = self._value_alpha(af, r, local_key_address)
                value_read_query, value_read_query_gate = self._value_read_query(
                    r, af, hf, local_key_address
                )
                forward_diag["value_read_query_gate"] = self._diag_gate_stats(
                    value_read_query_gate, max(0, T - 2)
                )
                if local_key_address is not None:
                    forward_diag["value_address"] = self._diag_tensor_stats(
                        local_key_address
                    )
                    forward_diag["value_address_prev"] = self._diag_tensor_stats(
                        self._previous_code(local_key_address)
                    )
                forward_diag["value_alpha"] = self._diag_tensor_stats(value_alpha)
                forward_diag["value_read_query"] = self._diag_tensor_stats(
                    value_read_query
                )
                self._set_forward_diag(forward_diag)
                y = run_trellis_memory_chunked(
                    vf,
                    value_read_query,
                    value_alpha,
                    bf,
                    gf,
                    self.phi,
                    "M_T_r",
                    cs,
                    cfg.chunk_refine,
                    update_gate=value_update_gate,
                    residual_update_mix=cfg.residual_update_mix,
                    trellis_update_stabilizer=cfg.trellis_update_stabilizer,
                    trellis_innovation_rms_cap=cfg.trellis_innovation_rms_cap,
                    trellis_delta_ratio_cap=cfg.trellis_delta_ratio_cap,
                    trellis_state_rms_floor=cfg.trellis_state_rms_floor,
                    trellis_stabilizer_detach_scale=(
                        cfg.trellis_stabilizer_detach_scale
                    ),
                )
            else:
                forward_diag = self._base_forward_diag(
                    beta=bf,
                    gamma=gf,
                    update_gate=update_gate,
                    key_update_gate=key_update_gate,
                    value_update_gate=value_update_gate,
                    seq_len=T,
                    backend="pytorch_sequential",
                )
                ex = cfg.exact_inner

                def _ic_or_seq(write_in, read_in, alpha_in, read_mode, ugate):
                    # input-conditioned: exact affine chunk kernel when
                    # chunk_size>1 (matmul throughput), else exact sequential.
                    if ic_chunk:
                        return run_trellis_memory_chunked(
                            write_in,
                            read_in,
                            alpha_in,
                            bf,
                            gf,
                            self.phi,
                            read_mode,
                            cfg.chunk_size,
                            update_gate=ugate,
                            input_gate=write_gate,
                        )
                    return run_trellis_memory(
                        write_in,
                        read_in,
                        alpha_in,
                        bf,
                        gf,
                        self.phi,
                        read_mode,
                        training,
                        exact_inner=ex,
                        update_gate=ugate,
                        residual_update_mix=cfg.residual_update_mix,
                        trellis_update_stabilizer=cfg.trellis_update_stabilizer,
                        trellis_innovation_rms_cap=cfg.trellis_innovation_rms_cap,
                        trellis_delta_ratio_cap=cfg.trellis_delta_ratio_cap,
                        trellis_state_rms_floor=cfg.trellis_state_rms_floor,
                        trellis_stabilizer_detach_scale=(
                            cfg.trellis_stabilizer_detach_scale
                        ),
                        input_gate=write_gate,
                    )

                yhat = _ic_or_seq(kf, qf, af, "M_q", key_update_gate)
                r = self.f(yhat)  # [B,H,T,M]
                value_alpha = self._value_alpha(af, r, local_key_address)
                value_read_query, value_read_query_gate = self._value_read_query(
                    r, af, hf, local_key_address
                )
                forward_diag["value_read_query_gate"] = self._diag_gate_stats(
                    value_read_query_gate, max(0, T - 2)
                )
                if local_key_address is not None:
                    forward_diag["value_address"] = self._diag_tensor_stats(
                        local_key_address
                    )
                    forward_diag["value_address_prev"] = self._diag_tensor_stats(
                        self._previous_code(local_key_address)
                    )
                forward_diag["value_alpha"] = self._diag_tensor_stats(value_alpha)
                forward_diag["value_read_query"] = self._diag_tensor_stats(
                    value_read_query
                )
                self._set_forward_diag(forward_diag)
                y = _ic_or_seq(
                    vf, value_read_query, value_alpha, "M_T_r", value_update_gate
                )

        # final phi on the value-pass readout (paper: y = phi(M^T r)). Applied
        # over the head dim D, on the fp32 memory output before head-merge.
        if self.value_readout_act is not None:
            y = self.value_readout_act(y)

        y = y.permute(0, 2, 1, 3).reshape(B, T, self.H * self.D)  # merge heads
        if self.output_path == "paper":
            yn = self.post_norm(y)
            g = F.gelu(self.gate_in(h))
            out = self.out_proj(yn * g)
        else:
            out = self.out_proj(y)
            if self.post_gate:
                out = out * F.silu(self.gate_proj(h))
        return self.drop(out)

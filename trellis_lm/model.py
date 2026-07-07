"""TrellisLM and a matched DenseTransformerTiny baseline.

Both expose forward(idx, labels=None) -> (logits, loss) and get_num_params(),
so the eval/train harness treats them uniformly. The dense baseline is the
fair from-scratch control (same d_model/layers/heads/d_head, SwiGLU MLP,
RMSNorm, causal MHA).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TrellisConfig
from .trellis_mixer import TrellisMixer, RMSNorm


class SwiGLU(nn.Module):
    def __init__(self, d, ratio, dropout=0.0):
        super().__init__()
        hidden = int(d * ratio)
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d, bias=False)
        self.norm = RMSNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        return self.drop(self.w3(F.silu(self.w1(h)) * self.w2(h)))


class TrellisBlock(nn.Module):
    def __init__(self, cfg: TrellisConfig, layer_idx: int = 0):
        super().__init__()
        self.mixer = TrellisMixer(cfg, layer_idx=layer_idx)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_ratio, cfg.dropout)

    def forward(self, x, training=True):
        x = x + self.mixer(x, training=training)
        x = x + self.mlp(x)
        return x


class _LMBase(nn.Module):
    cfg: TrellisConfig

    def _init_head(self):
        self.wte = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
        self.norm_f = RMSNorm(self.cfg.d_model)
        self.lm_head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)
        if self.cfg.tie_embeddings:
            self.lm_head.weight = self.wte.weight

    def get_num_params(self):
        n = sum(p.numel() for p in self.parameters())
        if self.cfg.tie_embeddings:
            n -= self.wte.weight.numel()  # counted once
        return n

    def memory_state_bytes(self, batch_size: int) -> int:
        """Bytes of bounded memory state (Trellis) per the config; 0 for dense
        (dense uses a growing KV cache instead)."""
        return 0


class TrellisLM(_LMBase):
    def __init__(self, cfg: TrellisConfig):
        super().__init__()
        self.cfg = cfg
        self._init_head()
        self.blocks = nn.ModuleList(
            [TrellisBlock(cfg, layer_idx=i) for i in range(cfg.n_layers)]
        )
        self.apply(self._init_weights)
        # _init_weights zeros every Linear bias; restore the forget-gate
        # retention bias afterwards so beta_init actually takes effect.
        for m in self.modules():
            if isinstance(m, TrellisMixer):
                m.reset_beta_bias()
                m.reset_update_gate_bias()
                m.reset_value_read_query_gate_bias()
                m.reset_write_gate_bias()
                m.reset_write_lowrank()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, labels=None, training=None):
        if training is None:
            training = self.training
        x = self.wte(idx)
        for blk in self.blocks:
            x = blk(x, training=training)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    def memory_state_bytes(self, batch_size: int) -> int:
        c = self.cfg
        elem = 2 if c.dtype in ("bf16", "fp16") else 4
        # 2 passes (key + value memory) per layer
        return 2 * c.n_layers * batch_size * c.n_heads * c.n_slots * c.d_head * elem


# --- matched dense baseline ---


class CausalMHA(nn.Module):
    def __init__(self, cfg: TrellisConfig):
        super().__init__()
        self.H, self.D = cfg.n_heads, cfg.d_head
        self.norm = RMSNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * self.H * self.D, bias=False)
        self.o = nn.Linear(self.H * self.D, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, d = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).view(B, T, 3, self.H, self.D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,H,T,D]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.permute(0, 2, 1, 3).reshape(B, T, self.H * self.D)
        return self.drop(self.o(y))


class DenseBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = CausalMHA(cfg)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_ratio, cfg.dropout)

    def forward(self, x, training=True):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class DenseTransformerTiny(_LMBase):
    def __init__(self, cfg: TrellisConfig):
        super().__init__()
        self.cfg = cfg
        self._init_head()
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([DenseBlock(cfg) for _ in range(cfg.n_layers)])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, labels=None, training=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.pos(pos)[None]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return logits, loss


def build_model(cfg: TrellisConfig, kind: str):
    if kind == "trellis":
        return TrellisLM(cfg)
    if kind == "dense":
        return DenseTransformerTiny(cfg)
    if kind in ("delta", "gated_delta"):
        from .linear_baselines import build_linear_baseline

        return build_linear_baseline(cfg, gated=(kind == "gated_delta"))
    if kind in ("delta_ref", "gated_delta_ref"):
        # Kernel-fair headline baselines: fla's REFERENCE DeltaNet/GatedDeltaNet
        # (short conv + qk-norm + output gate), matched width to the ladder.
        from .linear_baselines_fla_ref import build_linear_baseline_ref

        return build_linear_baseline_ref(cfg, gated=(kind == "gated_delta_ref"))
    if kind.startswith("gated_delta_product_ref"):
        # FLA GatedDeltaProduct reference; kind carries n_h ("..._nh2"/"_nh3").
        from .linear_baselines_fla_ref import build_linear_baseline_ref

        nh = int(kind.rsplit("nh", 1)[-1]) if "nh" in kind else 2
        return build_linear_baseline_ref(cfg, gated=True, num_householder=nh)
    raise ValueError(kind)

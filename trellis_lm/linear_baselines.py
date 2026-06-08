"""DeltaNet and Gated-DeltaNet baselines — the linear-inner-step cousins of
Trellis, in the same harness for a matched-size scratch comparison.

The point of these baselines: Trellis = gated delta rule with a *nonlinear*
inner gradient step (normalised-SiLU phi). DeltaNet / Gated DeltaNet are the
same delta-rule fast-weight memory with a *linear* inner step. Beating a matched
dense transformer is necessary but weak; the scientifically pointed question is
whether Trellis's nonlinearity buys anything over its linear cousins at matched
size. These mixers plug into the same block/LM/trainer/eval as TrellisMixer.

State recurrence (per head, write-before-read), with the standard Gated-DeltaNet
form S_t = alpha_t S_{t-1} + beta_t (v_t - alpha_t S_{t-1} k_t) k_t^T:
  pred_t = alpha_t * (S_{t-1} @ k_t)        # decayed memory's guess for k_t
  S_t    = alpha_t S_{t-1} + beta_t (v_t - pred_t) (x) k_t
  o_t    = S_t @ q_t
Plain DeltaNet is the alpha_t = 1 special case. Keys are L2-normalised (the
DeltaNet convention that keeps the (I - beta k k^T) factor a contraction).
Sequential per-token loop, matching the Trellis `seq` operator for fairness.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TrellisConfig
from .trellis_mixer import RMSNorm


class DeltaNetMixer(nn.Module):
    def __init__(self, cfg: TrellisConfig, gated: bool):
        super().__init__()
        self.cfg = cfg
        self.gated = gated
        H, D, d = cfg.n_heads, cfg.d_head, cfg.d_model
        self.H, self.D = H, D
        self.norm = RMSNorm(d)
        self.q_proj = nn.Linear(d, H * D, bias=False)
        self.k_proj = nn.Linear(d, H * D, bias=False)
        self.v_proj = nn.Linear(d, H * D, bias=False)
        self.beta_proj = nn.Linear(d, H, bias=True)
        if gated:
            self.alpha_proj = nn.Linear(d, H, bias=True)
        self.out_proj = nn.Linear(H * D, d, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def _heads(self, x):  # [B,T,H*D] -> [B,H,T,D]
        B, T, _ = x.shape
        return x.view(B, T, self.H, self.D).permute(0, 2, 1, 3)

    def forward(self, x, training: bool = True):
        B, T, d = x.shape
        h = self.norm(x)
        q = self._heads(self.q_proj(h))
        k = self._heads(self.k_proj(h))
        v = self._heads(self.v_proj(h))
        k = F.normalize(k, dim=-1)
        beta = torch.sigmoid(self.beta_proj(h)).permute(0, 2, 1)  # [B,H,T]
        if self.gated:
            alpha = torch.sigmoid(self.alpha_proj(h)).permute(0, 2, 1)  # [B,H,T]
        S = torch.zeros(B, self.H, self.D, self.D, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(T):
            kt, vt, qt = k[:, :, t, :], v[:, :, t, :], q[:, :, t, :]
            bt = beta[:, :, t][..., None, None]
            if self.gated:
                at = alpha[:, :, t][..., None, None]
                S = at * S
            pred = torch.einsum("bhij,bhj->bhi", S, kt)  # (decayed) S @ k
            S = S + bt * torch.einsum("bhi,bhj->bhij", vt - pred, kt)
            outs.append(torch.einsum("bhij,bhj->bhi", S, qt))  # S @ q (updated)
        y = torch.stack(outs, dim=2).permute(0, 2, 1, 3).reshape(B, T, self.H * self.D)
        return self.drop(self.out_proj(y))


def build_linear_baseline(cfg: TrellisConfig, gated: bool):
    """Build a tiny LM whose mixer is DeltaNet / Gated-DeltaNet. Imports the
    shared block/LM scaffolding lazily to avoid a circular import with model.py.
    """
    from .model import SwiGLU, _LMBase
    from .linear_baselines_chunked import DeltaNetMixerChunked
    from .linear_baselines_fla import FLADeltaNetMixer, HAS_FLA

    def _make_mixer():
        # fla's Triton chunked-scan is ~76-107x faster than our bmm kernel and
        # actually faster than dense attention at T=2048 (linear O(T) beats the
        # O(T^2) softmax). Use it on the CUDA pods; fall back to the bmm kernel
        # on monster/prune (no fla) so the same code runs everywhere.
        if HAS_FLA:
            return FLADeltaNetMixer(cfg, gated)
        return DeltaNetMixerChunked(cfg, gated)

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mixer = _make_mixer()
            self.mlp = SwiGLU(cfg.d_model, cfg.mlp_ratio, cfg.dropout)

        def forward(self, x, training=True):
            x = x + self.mixer(x, training=training)
            x = x + self.mlp(x)
            return x

    class LinearBaselineLM(_LMBase):
        def __init__(self):
            super().__init__()
            self.cfg = cfg
            self._init_head()
            self.blocks = nn.ModuleList([_Block() for _ in range(cfg.n_layers)])
            self.apply(self._init_weights)

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
            return c.n_layers * batch_size * c.n_heads * c.d_head * c.d_head * elem

    return LinearBaselineLM()

# SPDX-License-Identifier: MIT

"""
KV Cache Pruning strategies for attention mechanisms.

Implements:
1. V-only pruning: Keep K full, prune V only (minimal semantic drift)
2. KV pruning with score reuse: Prune both K&V but reuse scores (no double GEMM)
3. V-reconstruction: MLP-based reconstruction of pruned V information
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class KVPruneCfg:
    """Configuration for KV cache pruning strategies."""

    keep_ratio: float = 0.382  # start near 1/(1+phi)
    k_min: int = 64  # minimum tokens to keep
    recency: int = 64  # force-keep last N tokens (0 disables)
    exposure_correct: bool = True  # correct for causal attention bias
    ema_momentum: float = 0.9  # EMA smoothing (0 disables)
    ema_update_interval: int = 8  # steps between EMA updates
    v_recon_hidden_dim: int = 0  # V-reconstruction hidden dim (0 disables)
    mode: str = "v_only"  # "v_only" or "kv_scores_reuse"


class VOnlyPruner(nn.Module):
    """
    V-only pruning: Keep K full, prune V based on importance.

    Benefits:
    - Single softmax (no distribution shift)
    - No second GEMM (reuses full attention)
    - Minimal semantic drift from baseline
    - ~31% memory savings on V cache

    Strategy:
    1. Compute full attention: softmax(Q @ K^T / sqrt(d))
    2. Measure V importance: mean attention each V receives
    3. Optional: correct for causal exposure bias
    4. Select top-k V indices
    5. Slice attention columns and multiply by pruned V
    """

    def __init__(self, cfg: KVPruneCfg, n_heads: int, head_dim: int):
        super().__init__()
        self.cfg = cfg
        self.n_heads = n_heads
        self.head_dim = head_dim

        # EMA state for importance smoothing
        self.register_buffer("_ema_importance", torch.tensor([]), persistent=False)
        self.register_buffer(
            "_ema_step", torch.tensor(0, dtype=torch.long), persistent=False
        )

    def _compute_exposure(self, T, device):
        """
        Compute causal exposure counts.
        exposure[j] = number of queries that can attend to key j = T - j
        """
        exposure = torch.arange(T, 0, -1, device=device, dtype=torch.float32)
        return exposure.clamp_min_(1.0)  # avoid div-by-zero

    def _maybe_exposure_correct(self, attn, scores):
        """
        Compute importance with optional exposure correction.

        Without correction: importance[j] = mean over queries of attn[i,j]
        With correction: importance[j] = (sum over queries of attn[i,j]) / exposure[j]

        This prevents early tokens from being over-selected just because
        they're visible to more queries in causal attention.
        """
        if not self.cfg.exposure_correct:
            # Simple mean over queries (original approach)
            return attn.sum(dim=2) / attn.size(2)  # [B,H,T]

        # Exposure-correct: divide column mass by visibility count
        B, H, T, _ = attn.shape
        exposure = self._compute_exposure(T, attn.device)  # [T]
        col_mass = attn.sum(dim=2)  # [B,H,T]
        return col_mass / exposure  # broadcast over [B,H,*]

    def _apply_recency_force(self, importance):
        """Force last recency tokens to maximum importance."""
        if self.cfg.recency <= 0:
            return importance

        B, H, T = importance.shape
        if T <= self.cfg.recency:
            return importance

        importance = importance.clone()
        importance[:, :, -self.cfg.recency :] = 1.0
        return importance

    def _topk(self, importance, T):
        """Select top-k indices based on importance."""
        k_keep = max(self.cfg.k_min, int(T * float(self.cfg.keep_ratio)))
        k_keep = min(k_keep, T)  # safety
        vals, idx = torch.topk(importance, k_keep, dim=-1)  # [B,H,k]
        return vals, idx

    def _maybe_ema(self, importance):
        """
        Optional EMA over batches/steps to stabilize selection.
        Resets if shape changes (different sequence length).
        """
        if self.cfg.ema_momentum <= 0:
            return importance

        # Reset EMA if shape changed
        if self._ema_importance.numel() == 0 or tuple(
            self._ema_importance.shape
        ) != tuple(importance.shape):
            self._ema_importance = importance.detach()
            self._ema_step.zero_()
            return importance

        step = int(self._ema_step.item())
        if (step % max(1, self.cfg.ema_update_interval)) == 0:
            m = self.cfg.ema_momentum
            self._ema_importance = (
                m * self._ema_importance + (1 - m) * importance.detach()
            )

        self._ema_step += 1
        return self._ema_importance

    @torch.no_grad()
    def compute_indices(self, scores, attn):
        """
        Compute V indices to keep based on importance.

        Args:
            scores: [B,H,T,T] (not used directly but available for alt metrics)
            attn: [B,H,T,T] attention weights (post-softmax)

        Returns:
            idx: [B,H,k] indices of V tokens to keep
        """
        importance = self._maybe_exposure_correct(attn, scores)  # [B,H,T]
        importance = self._apply_recency_force(importance)  # force last N
        importance = self._maybe_ema(importance)  # smooth over time
        _, idx = self._topk(importance, importance.size(-1))  # [B,H,k]
        return idx

    def forward(self, q, k, v, inv_sqrt_d):
        """
        V-only pruning forward pass.

        Args:
            q, k, v: [B,H,T,D]
            inv_sqrt_d: scalar (1/sqrt(head_dim))

        Returns:
            out: [B,H,T,D] attention output
            idx: [B,H,k] indices of kept V tokens
        """
        # 1) Full scores -> softmax once (no recompute)
        scores = (q @ k.transpose(-2, -1)) * inv_sqrt_d  # [B,H,T,T]
        attn = F.softmax(scores, dim=-1)  # [B,H,T,T]

        # 2) Pick V indices (per head)
        idx = self.compute_indices(scores, attn)  # [B,H,k]

        # 3) Slice attention columns and V
        B, H, T, _ = attn.shape
        k_keep = idx.size(-1)

        # Gather attention weights for selected V positions
        idx_T = idx.unsqueeze(-2).expand(B, H, T, k_keep)  # [B,H,T,k]
        attn_pruned = torch.gather(attn, dim=-1, index=idx_T)  # [B,H,T,k]

        # Renormalize attention weights after pruning to sum to 1
        # Without this, pruned weights don't form a valid distribution
        attn_pruned = attn_pruned / (attn_pruned.sum(dim=-1, keepdim=True) + 1e-10)

        # Gather V for selected positions
        idx_D = idx.unsqueeze(-1).expand(B, H, k_keep, v.size(-1))  # [B,H,k,D]
        V_keep = torch.gather(v, dim=2, index=idx_D)  # [B,H,k,D]

        # 4) Output (no re-GEMM, no distribution shift in K)
        out = attn_pruned @ V_keep  # [B,H,T,D]

        return out, idx, attn_pruned, V_keep


class KVScoresReusePruner(nn.Module):
    """
    KV pruning with score reuse (no double GEMM).

    Prunes both K and V but reuses the full scores matrix instead of
    recomputing q @ K_keep. Still has semantic drift from changing the
    key set, but avoids the expensive second matmul.
    """

    def __init__(self, cfg: KVPruneCfg, n_heads: int, head_dim: int):
        super().__init__()
        # Reuse VOnlyPruner's importance logic
        self.importance_computer = VOnlyPruner(cfg, n_heads, head_dim)

    def forward(self, q, k, v, inv_sqrt_d):
        """
        KV pruning with score reuse.

        Args:
            q, k, v: [B,H,T,D]
            inv_sqrt_d: scalar

        Returns:
            out: [B,H,T,D]
            idx: [B,H,k]
        """
        # 1) Compute full scores once
        scores_full = (q @ k.transpose(-2, -1)) * inv_sqrt_d  # [B,H,T,T]
        attn_full = F.softmax(scores_full, dim=-1)  # [B,H,T,T]

        # 2) Get importance-based indices
        idx = self.importance_computer.compute_indices(scores_full, attn_full)
        B, H, T, _ = scores_full.shape
        k_keep = idx.size(-1)

        # 3) Gather scores (not recompute!)
        idx_T = idx.unsqueeze(-2).expand(B, H, T, k_keep)
        scores_kept = torch.gather(scores_full, -1, idx_T)  # [B,H,T,k]

        # 4) Renormalize (semantic drift happens here)
        attn_kept = F.softmax(scores_kept, dim=-1)  # [B,H,T,k]

        # 5) Gather V
        idx_D = idx.unsqueeze(-1).expand(B, H, k_keep, v.size(-1))
        V_keep = torch.gather(v, 2, idx_D)  # [B,H,k,D]

        # 6) Output
        out = attn_kept @ V_keep  # [B,H,T,D]

        return out, idx


class VReconstructor(nn.Module):
    """
    MLP-based V reconstruction with learnable gating.

    Reconstructs pruned V information from MLP hidden states and blends
    with actual V via a learned gate. Tests hypothesis: "MLP can replace
    attention detail".

    Architecture:
        V_hat = Linear(mlp_hidden) -> [B,H,T,D]
        gate = sigmoid(Linear(mlp_hidden)) -> [B,H,T,1]
        V_eff = gate * V_keep + (1 - gate) * V_hat_keep
    """

    def __init__(
        self, cfg: KVPruneCfg, n_heads: int, head_dim: int, mlp_hidden_dim: int
    ):
        super().__init__()
        assert cfg.v_recon_hidden_dim >= 0
        self.cfg = cfg
        self.n_heads = n_heads
        self.head_dim = head_dim

        # Map MLP hidden -> all heads * D, then reshape to [B,H,T,D]
        self.v_recon_proj = nn.Linear(mlp_hidden_dim, n_heads * head_dim, bias=True)

        # Gate per head per token (broadcast across D)
        self.v_gate_proj = nn.Linear(mlp_hidden_dim, n_heads, bias=True)

    def forward(self, mlp_hidden, idx, V_keep):
        """
        Reconstruct and blend V.

        Args:
            mlp_hidden: [B,T,M] MLP hidden states
            idx: [B,H,k] indices of kept V positions
            V_keep: [B,H,k,D] actual kept V values

        Returns:
            V_eff_keep: [B,H,k,D] blended V (gate * V_keep + (1-gate) * V_hat)
        """
        B, T, M = mlp_hidden.shape
        H, D = self.n_heads, self.head_dim

        # 1) Predict V_hat for all tokens/heads
        V_hat_all = self.v_recon_proj(mlp_hidden)  # [B,T,H*D]
        V_hat_all = V_hat_all.view(B, T, H, D).permute(0, 2, 1, 3)  # [B,H,T,D]

        # 2) Gate per head/token (broadcast over D)
        gate_logits = self.v_gate_proj(mlp_hidden)  # [B,T,H]
        gate = torch.sigmoid(gate_logits).permute(0, 2, 1)  # [B,H,T]
        gate = gate.unsqueeze(-1)  # [B,H,T,1]

        # 3) Gather to kept columns (match V_keep's indexing)
        B, H, k = idx.shape
        idx_D = idx.unsqueeze(-1).expand(B, H, k, D)
        V_hat_keep = torch.gather(V_hat_all, dim=2, index=idx_D)  # [B,H,k,D]
        gate_keep = torch.gather(gate, dim=2, index=idx.unsqueeze(-1))  # [B,H,k,1]

        # 4) Blend: gate * V_kept + (1-gate) * V_hat
        V_eff_keep = gate_keep * V_keep + (1.0 - gate_keep) * V_hat_keep

        return V_eff_keep, gate_keep.squeeze(-1)  # also return gate for logging

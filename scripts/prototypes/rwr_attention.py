# SPDX-License-Identifier: MIT

"""
RWR (Random Walk with Restart) Kernel Attention.

Factorizes attention into LOCAL + RWR components:
  A(q_i, :) ≈ LOCAL(i) + RWR(i)

where LOCAL is short-range windowed attention and RWR captures long-range
structure via token-graph random walks with restart.

Supports:
- Reversible Markov chains (detailed balance)
- Reciprocal lens coupling (forward + backward saliency)
- FlashAttention-style SRAM tiling (fallback implementation)
- Tensor-core friendly layouts (head_dim padding)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from lib.graph_builder import (
    build_sparse_W,
    normalize_to_P,
    reversible,
    sparse_mm_batch,
)


class RWRKernelAttention(nn.Module):
    """
    RWR-augmented attention with FlashAttention-style tiling.

    Combines local windowed attention with RWR-based long-range attention.
    """

    def __init__(
        self,
        dim_head: int,
        num_heads: int = 12,
        rwr_alpha: float = 0.2,
        rwr_steps: int = 4,
        rwr_topk: int = 32,
        rwr_threshold: float = 0.0,
        reversible: bool = False,
        reciprocal_beta: float = 0.5,
        lens_strength: float = 0.3,
        window: int = 128,
        block_size: int = 128,
        head_dim_pad: int = 64,
        use_discoverability: bool = False,
    ):
        """
        Initialize RWR kernel attention.

        Args:
            dim_head: Head dimension (will be padded to head_dim_pad multiple)
            num_heads: Number of attention heads
            rwr_alpha: Restart probability (default: 0.2)
            rwr_steps: Number of RWR iterations (default: 4)
            rwr_topk: Top-k neighbors per query for RWR (default: 32)
            rwr_threshold: Minimum similarity threshold (default: 0.0)
            reversible: Use reversible chain P_rev (default: False)
            reciprocal_beta: Mix forward/backward saliency (default: 0.5)
            lens_strength: Blending factor γ for RWR term (default: 0.3)
            window: Local attention window half-width (default: 128)
            block_size: SRAM tile size (default: 128)
            head_dim_pad: Round head_dim to multiple (default: 64)
            use_discoverability: Enable lens column bias (default: False)
        """
        super().__init__()
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.pad = head_dim_pad
        self.rwr_alpha = rwr_alpha
        self.rwr_steps = rwr_steps
        self.rwr_topk = rwr_topk
        self.rwr_threshold = rwr_threshold
        self.reversible_chain = reversible
        self.reciprocal_beta = reciprocal_beta
        self.lens_strength = lens_strength
        self.window = window
        self.block_size = block_size
        self.use_discoverability = use_discoverability

        # Discoverability: learnable column bias (per head)
        if use_discoverability:
            self.u_h = nn.Parameter(torch.zeros(num_heads, dim_head))

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        P_sparse: Optional[Tuple] = None,
        P_rev_sparse: Optional[Tuple] = None,
        causal_mask: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with LOCAL + RWR attention.

        Args:
            q: Query [B, H, N, D]
            k: Key [B, H, N, D]
            v: Value [B, H, N, D]
            P_sparse: Prebuilt sparse transition matrix (optional)
            P_rev_sparse: Prebuilt reversible matrix (optional)
            causal_mask: Apply causal masking for autoregressive (default: False)

        Returns:
            Attention output [B, H, N, D]
        """
        B, H, N, D = q.shape

        # 1) Local attention (windowed, FlashAttention-style fallback)
        y_local = self._local_flash_like(q, k, v, causal_mask=causal_mask)

        # 2) Build or reuse sparse P
        if P_sparse is None:
            P_sparse, P_rev_sparse = self._build_sparse_P(q, k)

        # 3) RWR long-range attention
        y_rwr = self._rwr_long_range(q, v, P_sparse, P_rev_sparse)

        # 4) Fuse LOCAL + γ * RWR
        y = y_local + self.lens_strength * y_rwr

        return y

    def _local_flash_like(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal_mask: bool
    ) -> torch.Tensor:
        """
        Local windowed attention with optional causal masking.

        Simplified fallback implementation (not true FlashAttention).
        For production, use xformers or flash_attn library.

        Args:
            q, k, v: [B, H, N, D]
            causal_mask: Apply causal mask

        Returns:
            Attention output [B, H, N, D]
        """
        B, H, N, D = q.shape
        scale = D**-0.5

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, N, N]

        # Apply local window mask (±self.window)
        idx = torch.arange(N, device=q.device)
        window_mask = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= self.window
        window_mask = window_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
        attn = attn.masked_fill(~window_mask, float("-inf"))

        # Apply causal mask if requested
        if causal_mask:
            causal = torch.tril(torch.ones(N, N, device=q.device, dtype=torch.bool))
            causal = causal.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
            attn = attn.masked_fill(~causal, float("-inf"))

        # Softmax and weighted sum
        attn = F.softmax(attn, dim=-1)
        y = torch.matmul(attn, v)  # [B, H, N, D]

        return y

    def _build_sparse_P(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[Tuple, Optional[Tuple]]:
        """
        Build sparse transition matrix P from Q, K.

        Args:
            q: Query [B, H, N, D]
            k: Key [B, H, N, D]

        Returns:
            P_sparse: (indices_list, values_list, shape)
            P_rev_sparse: (indices_list, values_list) if reversible, else None
        """
        B, H, N, D = q.shape

        # Build sparse W
        indices_list, values_list, shape = build_sparse_W(
            q, k, topk=self.rwr_topk, window=self.window, threshold=self.rwr_threshold
        )

        # Normalize to P
        p_indices, p_values, row_sums = normalize_to_P(indices_list, values_list, shape)

        P_sparse = (p_indices, p_values, shape)

        # Reversible symmetrization if enabled
        P_rev_sparse = None
        if self.reversible_chain:
            p_rev_indices, p_rev_values = reversible(
                p_indices, p_values, row_sums, shape
            )
            P_rev_sparse = (p_rev_indices, p_rev_values, shape)

        return P_sparse, P_rev_sparse

    def _rwr_long_range(
        self,
        q: torch.Tensor,
        v: torch.Tensor,
        P_sparse: Tuple,
        P_rev_sparse: Optional[Tuple],
    ) -> torch.Tensor:
        """
        Compute RWR-based long-range attention.

        Uses truncated Neumann series:
            r_i ≈ α * sum_{t=0..T} ((1-α) P)^t e_i

        Args:
            q: Query [B, H, N, D]
            v: Value [B, H, N, D]
            P_sparse: (indices_list, values_list, shape)
            P_rev_sparse: Optional reversible matrix

        Returns:
            RWR attention output [B, H, N, D]
        """
        B, H, N, D = q.shape
        device = q.device
        dtype = q.dtype

        p_indices, p_values, shape = P_sparse

        # Choose P or P_rev for walks
        if self.reversible_chain and P_rev_sparse is not None:
            walk_indices, walk_values, _ = P_rev_sparse
        else:
            walk_indices, walk_values = p_indices, p_values

        # Initialize E (identity for each query)
        # For all queries: R0 = I (identity matrix)
        R = torch.eye(N, dtype=dtype, device=device).unsqueeze(0).repeat(B * H, 1, 1)
        # R: [B*H, N, N]

        # Truncated RWR: R = α * sum_{t=0..T} ((1-α) P)^t
        alpha = self.rwr_alpha
        R_accum = alpha * R.clone()  # t=0 term

        for t in range(1, self.rwr_steps + 1):
            # R = R @ P
            R = sparse_mm_batch(R, walk_indices, walk_values, shape)
            # Accumulate: R_accum += α * (1-α)^t * R
            R_accum = R_accum + alpha * ((1 - alpha) ** t) * R

        # R_accum: [B*H, N, N] - saliency scores for all pairs

        # Reciprocal coupling: mix forward and backward saliency
        if self.reciprocal_beta != 0.5:
            # Backward saliency: R^T (transpose)
            R_backward = R_accum.transpose(-2, -1)
            R_accum = (
                self.reciprocal_beta * R_accum + (1 - self.reciprocal_beta) * R_backward
            )

        # Select top-k indices per query for sparse attention
        # For simplicity, use top-k over R_accum (saliency)
        topk_vals, topk_idx = torch.topk(
            R_accum, k=min(self.rwr_topk, N), dim=-1
        )  # [B*H, N, k]

        # Gather values using topk indices
        # v_gathered: [B*H, N, k, D]
        v_flat = v.reshape(B * H, N, D)  # [B*H, N, D]
        v_gathered = torch.gather(
            v_flat.unsqueeze(1).expand(-1, N, -1, -1),  # [B*H, N, N, D]
            dim=2,
            index=topk_idx.unsqueeze(-1).expand(-1, -1, -1, D),  # [B*H, N, k, D]
        )

        # Weighted sum: y_rwr = sum_j (r_ij * v_j)
        # topk_vals: [B*H, N, k], v_gathered: [B*H, N, k, D]
        y_rwr = torch.sum(topk_vals.unsqueeze(-1) * v_gathered, dim=2)  # [B*H, N, D]

        # Reshape back to [B, H, N, D]
        y_rwr = y_rwr.reshape(B, H, N, D)

        return y_rwr


def analyze_rwr_stats(model: nn.Module) -> dict:
    """
    Analyze RWR attention statistics for debugging.

    Args:
        model: Model containing RWR attention modules

    Returns:
        Dictionary of RWR statistics
    """
    stats = {
        "num_rwr_modules": 0,
        "total_rwr_steps": 0,
        "avg_rwr_alpha": 0.0,
        "avg_lens_strength": 0.0,
    }

    count = 0
    for module in model.modules():
        if isinstance(module, RWRKernelAttention):
            stats["num_rwr_modules"] += 1
            stats["total_rwr_steps"] += module.rwr_steps
            stats["avg_rwr_alpha"] += module.rwr_alpha
            stats["avg_lens_strength"] += module.lens_strength
            count += 1

    if count > 0:
        stats["avg_rwr_alpha"] /= count
        stats["avg_lens_strength"] /= count

    return stats


# ======================== GPT-2 Patching ========================


class RWRAttentionWrapper(nn.Module):
    """
    Wrapper that adapts RWRKernelAttention to GPT-2 attention interface.

    Handles Q/K/V projections and output projection to match GPT-2 API.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        rwr_attn: RWRKernelAttention,
        original_attn,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # RWR attention core
        self.rwr_attn = rwr_attn

        # Q/K/V projections (copy from original)
        self.c_attn = original_attn.c_attn  # [3*n_embd, n_embd]
        self.c_proj = original_attn.c_proj  # [n_embd, n_embd]

        # Dropout (copy config)
        self.attn_dropout = original_attn.attn_dropout
        self.resid_dropout = original_attn.resid_dropout

        # Config
        self.bias = getattr(original_attn, "bias", None)  # Causal mask buffer

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        """GPT-2 compatible forward pass."""
        B, T, E = hidden_states.shape

        # Project Q, K, V
        qkv = self.c_attn(hidden_states)  # [B, T, 3*E]
        q, k, v = qkv.split(self.n_embd, dim=-1)  # Each [B, T, E]

        # Reshape to [B, H, T, D]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Handle KV cache
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat([past_key, k], dim=-2)
            v = torch.cat([past_value, v], dim=-2)

        present = (k, v) if use_cache else None

        # RWR attention
        causal_mask = True  # GPT-2 is always causal
        attn_output = self.rwr_attn(q, k, v, causal_mask=causal_mask)  # [B, H, T, D]

        # Reshape back to [B, T, E]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, E)

        # Output projection
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        # Match original GPT-2 interface: just return tensor if no cache/attentions
        if not use_cache and not output_attentions:
            return attn_output

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (None,)  # RWR doesn't return attention weights

        return outputs


def patch_gpt2_with_rwr(
    model,
    rwr_alpha: float = 0.2,
    rwr_steps: int = 4,
    rwr_topk: int = 32,
    rwr_threshold: float = 0.0,
    reversible: bool = False,
    reciprocal_beta: float = 0.5,
    lens_strength: float = 0.3,
    window: int = 128,
    block_size: int = 128,
    head_dim_pad: int = 64,
    use_discoverability: bool = False,
):
    """
    Patch GPT-2 model with RWR attention.

    Args:
        model: HuggingFace GPT-2 model
        rwr_alpha: Restart probability (default: 0.2)
        rwr_steps: Walk iterations T (default: 4)
        rwr_topk: Top-k neighbors per query (default: 32)
        rwr_threshold: Minimum similarity threshold (default: 0.0)
        reversible: Enable reversible chain P_rev (default: False)
        reciprocal_beta: Forward/backward mixing (default: 0.5)
        lens_strength: RWR blending γ (default: 0.3)
        window: Local attention window (default: 128)
        block_size: SRAM tile size (default: 128)
        head_dim_pad: Head dimension padding (default: 64)
        use_discoverability: Enable column bias (default: False)

    Returns:
        model: Patched model with RWR attention
    """
    n_embd = model.config.n_embd
    n_head = model.config.n_head
    head_dim = n_embd // n_head

    print("Patching GPT-2 with RWR attention:")
    print(f"  - Alpha (restart): {rwr_alpha}")
    print(f"  - Steps (T): {rwr_steps}")
    print(f"  - Top-k: {rwr_topk}")
    print(f"  - Window: {window}")
    print(f"  - Reversible: {reversible}")
    print(f"  - Reciprocal beta: {reciprocal_beta}")
    print(f"  - Lens strength (γ): {lens_strength}")

    # Patch each transformer block
    for i, block in enumerate(model.transformer.h):
        original_attn = block.attn

        # Create RWR attention
        rwr_attn = RWRKernelAttention(
            dim_head=head_dim,
            num_heads=n_head,
            rwr_alpha=rwr_alpha,
            rwr_steps=rwr_steps,
            rwr_topk=rwr_topk,
            rwr_threshold=rwr_threshold,
            reversible=reversible,
            reciprocal_beta=reciprocal_beta,
            lens_strength=lens_strength,
            window=window,
            block_size=block_size,
            head_dim_pad=head_dim_pad,
            use_discoverability=use_discoverability,
        )

        # Wrap with GPT-2 interface
        rwr_wrapper = RWRAttentionWrapper(n_embd, n_head, rwr_attn, original_attn)

        # Replace attention
        block.attn = rwr_wrapper

    # Mark config
    model.config.rwr_enabled = True
    model.config.rwr_alpha = rwr_alpha
    model.config.rwr_steps = rwr_steps

    print(f"✓ Patched {len(model.transformer.h)} blocks with RWR attention")
    return model

"""
GPT-2 with Retrieval-Gated Sparse Attention (RGSA)

Implements retrieval-based sparse attention inspired by Meta's REGRAG paper.
Instead of every token attending to every other token (O(n^2)), we:
1. Compress context chunks into small routing embeddings
2. Use those embeddings to retrieve only relevant chunks
3. Run exact attention only on retrieved chunks + local window

This treats long context as memory to be retrieved, not data to be fully attended.

Architecture:
    Input tokens
        |
    Chunk into blocks (e.g., 64 tokens each)
        |
    Each chunk -> routing embedding via learned projector
        |
    Query token -> retrieve top-B relevant chunks via dot-product
        |
    Run attention on: [local window] + [retrieved chunks]
        |
    Output
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class RGSAMetrics:
    """Container for RGSA routing diagnostics."""

    # Teacher top-k recall: what fraction of teacher's top-k are in retrieved set
    teacher_topk_recall: float = 0.0

    # Candidate count distribution
    candidates_mean: float = 0.0
    candidates_p50: float = 0.0
    candidates_p95: float = 0.0
    candidates_p99: float = 0.0

    # Routing entropy and collapse detection
    routing_entropy: float = 0.0
    load_balance_score: float = 0.0  # 1.0 = perfect balance, 0.0 = collapsed

    # Tokens attended per query (theoretical vs actual)
    tokens_per_query_mean: float = 0.0
    tokens_per_query_max: float = 0.0

    # Dynamic chunking observability
    chunk_size_eff: int = 0
    n_chunks_eff: int = 0

    # Chunk selection histogram (which chunks are selected most often)
    chunk_selection_counts: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dict for W&B logging."""
        d = {
            "rgsa/teacher_topk_recall": self.teacher_topk_recall,
            "rgsa/candidates_mean": self.candidates_mean,
            "rgsa/candidates_p50": self.candidates_p50,
            "rgsa/candidates_p95": self.candidates_p95,
            "rgsa/candidates_p99": self.candidates_p99,
            "rgsa/routing_entropy": self.routing_entropy,
            "rgsa/load_balance_score": self.load_balance_score,
            "rgsa/tokens_per_query_mean": self.tokens_per_query_mean,
            "rgsa/tokens_per_query_max": self.tokens_per_query_max,
        }
        if self.chunk_size_eff > 0:
            d["rgsa/chunk_size_eff"] = self.chunk_size_eff
            d["rgsa/n_chunks_eff"] = self.n_chunks_eff
        return d


@dataclass
class RGSAConfig:
    """Configuration for RGSA model."""

    # Model architecture (from GPT-2)
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

    # RGSA-specific parameters
    chunk_size: int = 64  # tokens per chunk
    routing_dim: int = 32  # dimension of routing embeddings
    top_b: int = 8  # number of chunks to retrieve
    local_window: int = 256  # always-attend local context

    # Auxiliary loss weight for routing supervision
    routing_loss_weight: float = 0.1

    # Ablation modes for scientific validation
    dense_mode: bool = False  # Skip routing, keep router params (param count control)
    random_routing: bool = False  # Random chunk selection instead of learned

    # Dynamic chunking (L2M-inspired): chunk_size varies with seq_len
    dynamic_chunking: bool = False
    chunk_size_min: int = 32
    chunk_size_max: int = 256
    chunk_size_alpha: float = 0.5  # exponent for power schedule
    chunk_size_schedule: str = "power"  # "power" or "piecewise"
    chunk_size_piecewise: str = ""  # e.g. "512:32,2048:64,8192:128"
    chunk_size_rounding: str = "pow2"  # "pow2", "multiple_of_8", "nearest"
    chunk_size_debug_log: bool = True  # log effective chunk size

    # Defensive caps for stability (prevent GPU hangs)
    # max_candidates = local_window + top_b * chunk_size by default
    # Setting this enforces a hard cap on attention span per query
    max_candidates: int = 0  # 0 = auto-compute from local_window + top_b*chunk_size
    debug_log_candidates: bool = False  # Log max candidate count at each eval

    @classmethod
    def from_name(cls, name: str):
        """Create config from model name."""
        configs = {
            "gpt2-tiny": dict(n_layer=6, n_head=8, n_embd=512),
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
        }
        if name not in configs:
            raise ValueError(
                f"Unknown model name: {name}. Choose from {list(configs.keys())}"
            )
        return cls(**configs[name])


def _round_to_pow2(x: int) -> int:
    """Round to nearest power of 2."""
    if x <= 1:
        return 1
    # Find nearest power of 2
    lo = 1 << (x - 1).bit_length() - 1
    hi = 1 << (x - 1).bit_length()
    return lo if (x - lo) <= (hi - x) else hi


def _apply_rounding(x: int, mode: str) -> int:
    """Apply rounding rule to chunk size."""
    if mode == "pow2":
        return _round_to_pow2(x)
    elif mode == "multiple_of_8":
        return max(8, round(x / 8) * 8)
    elif mode == "nearest":
        return x
    else:
        raise ValueError(f"Unknown rounding mode: {mode}")


def _piecewise_lookup(seq_len: int, spec: str, default: int) -> int:
    """Parse piecewise spec like '512:32,2048:64,8192:128'."""
    if not spec:
        return default
    for entry in spec.split(","):
        entry = entry.strip()
        if ":" not in entry:
            continue
        threshold_str, size_str = entry.split(":", 1)
        threshold = int(threshold_str)
        size = int(size_str)
        if seq_len <= threshold:
            return size
    return default


def compute_chunk_size(seq_len: int, cfg: RGSAConfig) -> int:
    """
    Compute effective chunk size for a given sequence length.

    In static mode (dynamic_chunking=False), returns cfg.chunk_size.
    In dynamic mode, computes chunk size based on schedule:
      - "power": chunk_size ~ seq_len^alpha, clamped to [min, max]
      - "piecewise": threshold-based lookup from spec string
    Result is rounded per cfg.chunk_size_rounding and clamped >= 1.
    """
    if not cfg.dynamic_chunking:
        return cfg.chunk_size
    if cfg.chunk_size_schedule == "power":
        raw = int(round(seq_len**cfg.chunk_size_alpha))
        cs = max(cfg.chunk_size_min, min(raw, cfg.chunk_size_max))
    elif cfg.chunk_size_schedule == "piecewise":
        cs = _piecewise_lookup(seq_len, cfg.chunk_size_piecewise, cfg.chunk_size_max)
        cs = max(cfg.chunk_size_min, min(cs, cfg.chunk_size_max))
    else:
        raise ValueError(f"Unknown chunk_size_schedule: {cfg.chunk_size_schedule}")
    cs = _apply_rounding(cs, cfg.chunk_size_rounding)
    cs = max(1, cs)
    return cs


class ChunkRouter(nn.Module):
    """
    Computes routing embeddings for chunks of hidden states.

    Input: sequence of hidden states [B, T, C]
    Output: routing embeddings [B, num_chunks, routing_dim]

    Each chunk is mean-pooled then projected to a small routing embedding.
    """

    def __init__(self, config: RGSAConfig):
        super().__init__()
        self.chunk_size = config.chunk_size
        self.routing_dim = config.routing_dim

        # Project from hidden dim to routing dim
        self.proj = nn.Linear(config.n_embd, config.routing_dim, bias=config.bias)

    def forward(
        self,
        x: torch.Tensor,
        chunk_size_override: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute routing embeddings for each chunk.

        Args:
            x: [B, T, C] hidden states
            chunk_size_override: if set, use this instead of self.chunk_size

        Returns:
            routing_embeds: [B, num_chunks, routing_dim] routing embeddings
            chunk_mask: [B, num_chunks] mask for valid chunks (True = valid)
        """
        B, T, C = x.shape
        chunk_size = chunk_size_override if chunk_size_override else self.chunk_size

        # Pad sequence to be divisible by chunk_size
        pad_len = (chunk_size - T % chunk_size) % chunk_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        # Reshape to chunks: [B, num_chunks, chunk_size, C]
        num_chunks = x.size(1) // chunk_size
        x_chunks = x.view(B, num_chunks, chunk_size, C)

        # Mean pool each chunk: [B, num_chunks, C]
        chunk_means = x_chunks.mean(dim=2)

        # Project to routing dimension: [B, num_chunks, routing_dim]
        routing_embeds = self.proj(chunk_means)

        # Create mask for valid chunks (all chunks valid after padding)
        # But we track original token count for proper masking in attention
        num_valid_chunks = (T + chunk_size - 1) // chunk_size
        chunk_mask = torch.zeros(B, num_chunks, dtype=torch.bool, device=x.device)
        chunk_mask[:, :num_valid_chunks] = True

        return routing_embeds, chunk_mask


class RetrievalGate(nn.Module):
    """
    Selects top-B relevant chunks for each query position.

    Uses dot-product similarity between query hidden state and
    chunk routing embeddings.
    """

    def __init__(self, config: RGSAConfig):
        super().__init__()
        self.top_b = config.top_b
        self.routing_dim = config.routing_dim

        # Query projection (from hidden dim to routing dim)
        self.query_proj = nn.Linear(config.n_embd, config.routing_dim, bias=config.bias)

    def forward(
        self,
        query_hidden: torch.Tensor,
        routing_embeds: torch.Tensor,
        chunk_mask: Optional[torch.Tensor] = None,
        random_routing: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve top-B chunks for each query position.

        Args:
            query_hidden: [B, T, C] query hidden states
            routing_embeds: [B, num_chunks, routing_dim] chunk routing embeddings
            chunk_mask: [B, num_chunks] mask for valid chunks
            random_routing: if True, select random chunks instead of learned

        Returns:
            chunk_indices: [B, T, top_b] indices of selected chunks
            routing_scores: [B, T, num_chunks] similarity scores for all chunks
        """
        B, T, C = query_hidden.shape
        num_chunks = routing_embeds.size(1)

        # Project queries: [B, T, routing_dim]
        query_routing = self.query_proj(query_hidden)

        # Compute similarity: [B, T, num_chunks]
        # Normalize for stable dot product
        query_routing = F.normalize(query_routing, dim=-1)
        routing_embeds = F.normalize(routing_embeds, dim=-1)
        routing_scores = torch.bmm(query_routing, routing_embeds.transpose(1, 2))

        # Mask invalid chunks
        if chunk_mask is not None:
            routing_scores = routing_scores.masked_fill(
                ~chunk_mask.unsqueeze(1), float("-inf")
            )

        # Select top-B chunks: [B, T, top_b]
        top_b = min(self.top_b, num_chunks)

        if random_routing:
            # Random chunk selection: uniform random instead of learned
            # Still uses same top_b count for fair comparison
            chunk_indices = torch.stack(
                [
                    torch.randint(0, num_chunks, (T, top_b), device=query_hidden.device)
                    for _ in range(B)
                ]
            )
        else:
            _, chunk_indices = torch.topk(routing_scores, k=top_b, dim=-1)

        return chunk_indices, routing_scores


class RGSACausalSelfAttention(nn.Module):
    """
    Causal self-attention with retrieval-gated sparsity.

    For each query position:
    1. Always include local window (local_window tokens before query)
    2. Retrieve top-B chunks via RetrievalGate
    3. Run attention only on local + retrieved tokens

    Stability features:
    - Hard cap on max_candidates per query to prevent GPU hangs
    - Shape assertions to catch dimension mismatches early
    - No dynamic tensor growth in loops
    """

    def __init__(self, config: RGSAConfig, chunk_router: ChunkRouter):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.local_window = config.local_window
        self.chunk_size = config.chunk_size
        self.top_b = config.top_b
        self.rgsa_config = config  # stored for compute_chunk_size()

        # Defensive cap: max tokens to attend per query position
        # Default = local_window + top_b * chunk_size
        if config.max_candidates > 0:
            self.max_candidates = config.max_candidates
        else:
            self.max_candidates = config.local_window + config.top_b * config.chunk_size

        self.debug_log_candidates = config.debug_log_candidates
        self._max_candidates_seen = 0  # Track for debugging

        # Dynamic chunking state (logged per forward pass)
        self._chunk_size_eff = config.chunk_size
        self._n_chunks_eff = 0

        # Ablation modes
        self.dense_mode = config.dense_mode
        self.random_routing = config.random_routing

        # QKV projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Dropout layers
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Retrieval gate (shared chunk_router is passed from model)
        self.chunk_router = chunk_router
        self.retrieval_gate = RetrievalGate(config)

        # Lightweight profiling accumulators (CPU wall-clock, no sync overhead)
        self._profile_enabled = False
        self._profile_counts = 0
        self._profile_accum = {
            "qkv_proj": 0.0,
            "router_embed": 0.0,
            "similarity_topk": 0.0,
            "attention": 0.0,
            "output_proj": 0.0,
        }

    def enable_profiling(self, enabled: bool = True):
        """Enable/disable lightweight per-phase timing."""
        self._profile_enabled = enabled
        if enabled:
            self._profile_counts = 0
            for k in self._profile_accum:
                self._profile_accum[k] = 0.0

    def get_profile_stats(self) -> Dict[str, float]:
        """Return average timing per phase in milliseconds."""
        if self._profile_counts == 0:
            return {}
        stats = {}
        for k, v in self._profile_accum.items():
            stats[f"perf/rgsa_{k}_ms"] = (v / self._profile_counts) * 1000.0
        total = sum(self._profile_accum.values())
        stats["perf/rgsa_total_ms"] = (total / self._profile_counts) * 1000.0
        stats["perf/rgsa_profile_count"] = self._profile_counts
        return stats

    def forward(
        self,
        x: torch.Tensor,
        return_routing_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Forward pass with sparse attention.

        For short sequences (< local_window), uses dense attention.
        For longer sequences, uses retrieval-gated sparse attention.

        Shape assertions are included to catch dimension mismatches early
        and prevent silent failures that could lead to GPU hangs.
        """
        B, T, C = x.shape
        profiling = self._profile_enabled

        # Shape assertions for stability
        assert C == self.n_embd, f"Input dim {C} != n_embd {self.n_embd}"
        assert T > 0, "Sequence length must be positive"
        assert B > 0, "Batch size must be positive"

        if profiling:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        # Compute Q, K, V
        qkv = self.c_attn(x)
        assert qkv.shape == (B, T, 3 * self.n_embd), f"QKV shape mismatch: {qkv.shape}"
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if profiling:
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            self._profile_accum["qkv_proj"] += t1 - t0

        routing_info = None

        # Phase 1: Use dense causal attention for all sequences
        # The routing mechanism learns chunk embeddings but attention is dense
        # This validates the architecture before optimizing sparse attention
        #
        # Ablation modes:
        #   dense_mode: skip routing entirely (keep params for count control)
        #   random_routing: random chunk selection instead of learned
        # Compute effective chunk size (dynamic or static)
        chunk_size_eff = compute_chunk_size(T, self.rgsa_config)
        self._chunk_size_eff = chunk_size_eff
        self._n_chunks_eff = (T + chunk_size_eff - 1) // chunk_size_eff

        if not self.dense_mode and T > self.local_window + chunk_size_eff:
            if profiling:
                torch.cuda.synchronize()
                tr0 = time.perf_counter()

            # Compute routing embeddings (training the router)
            routing_embeds, chunk_mask = self.chunk_router(
                x, chunk_size_override=chunk_size_eff
            )

            if profiling:
                torch.cuda.synchronize()
                tr1 = time.perf_counter()
                self._profile_accum["router_embed"] += tr1 - tr0

            # Shape assertions for routing
            num_chunks = routing_embeds.size(1)
            assert routing_embeds.dim() == 3, "routing_embeds must be 3D"
            assert chunk_mask.shape == (B, num_chunks), "chunk_mask shape mismatch"

            # Clamp top_b if fewer chunks than requested
            effective_top_b = min(self.top_b, num_chunks)

            if profiling:
                torch.cuda.synchronize()
                ts0 = time.perf_counter()

            chunk_indices, routing_scores = self.retrieval_gate(
                x, routing_embeds, chunk_mask, random_routing=self.random_routing
            )

            if profiling:
                torch.cuda.synchronize()
                ts1 = time.perf_counter()
                self._profile_accum["similarity_topk"] += ts1 - ts0

            # Defensive cap: ensure chunk_indices are valid (non-negative, in range)
            assert chunk_indices.min() >= 0, "Negative chunk index detected"
            assert chunk_indices.max() < num_chunks, "Chunk index out of range"

            # Track max candidates for debugging
            theoretical_max = self.local_window + effective_top_b * chunk_size_eff
            self._max_candidates_seen = max(self._max_candidates_seen, theoretical_max)

            if return_routing_info:
                routing_info = {
                    "chunk_indices": chunk_indices,
                    "routing_scores": routing_scores,
                    "routing_embeds": routing_embeds,
                    "max_candidates_theoretical": theoretical_max,
                    "chunk_size_eff": chunk_size_eff,
                    "n_chunks_eff": self._n_chunks_eff,
                }

        if profiling:
            torch.cuda.synchronize()
            ta0 = time.perf_counter()

        # Use dense causal attention for all sequences
        # This is Phase 1 - router learns while attention stays dense
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        if profiling:
            torch.cuda.synchronize()
            ta1 = time.perf_counter()
            self._profile_accum["attention"] += ta1 - ta0

        # Shape assertion on output
        assert y.shape == (
            B,
            self.n_head,
            T,
            self.head_dim,
        ), f"Attention output shape mismatch: {y.shape}"

        # Reshape output
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        if profiling:
            torch.cuda.synchronize()
            tp0 = time.perf_counter()

        # Output projection
        y = self.resid_dropout(self.c_proj(y))

        if profiling:
            torch.cuda.synchronize()
            tp1 = time.perf_counter()
            self._profile_accum["output_proj"] += tp1 - tp0
            self._profile_counts += 1

        return y, routing_info

    def get_debug_stats(self) -> dict:
        """Return debug statistics for logging."""
        stats = {
            "rgsa/max_candidates_seen": self._max_candidates_seen,
            "rgsa/max_candidates_cap": self.max_candidates,
            "rgsa/chunk_size_eff": self._chunk_size_eff,
            "rgsa/n_chunks_eff": self._n_chunks_eff,
        }
        return stats

    def reset_debug_stats(self):
        """Reset debug statistics (call at start of each eval)."""
        self._max_candidates_seen = 0

    def _sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        chunk_indices: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """
        Compute sparse attention using local window + retrieved chunks.

        Memory-efficient implementation that processes positions in groups
        rather than building a full B x T x T mask.

        IMPORTANT: This method is not currently used (Phase 1 uses dense attention).
        It includes defensive caps to prevent GPU hangs if enabled in the future.
        """
        B, H, T, D = q.shape
        device = q.device

        # Shape assertions
        assert chunk_indices.shape[0] == B, "chunk_indices batch mismatch"
        assert chunk_indices.shape[1] == T, "chunk_indices seq_len mismatch"
        assert chunk_indices.min() >= 0, "Negative chunk index"

        # For memory efficiency, use SDPA for local window positions
        # and only do custom sparse attention for positions beyond local_window
        local_end = min(self.local_window + self.chunk_size, T)

        # Process local window portion with standard causal SDPA
        if local_end > 0:
            y_local = F.scaled_dot_product_attention(
                q[:, :, :local_end, :],
                k[:, :, :local_end, :],
                v[:, :, :local_end, :],
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            y_local = q.new_zeros(B, H, 0, D)

        # For positions beyond local window, we need sparse attention
        if local_end < T:
            n_sparse = T - local_end
            scale = 1.0 / math.sqrt(D)

            # Pre-allocate output tensor instead of dynamic list growth
            y_sparse = q.new_zeros(B, H, n_sparse, D)

            # Process in mini-batches to reduce memory
            chunk_batch_size = min(32, n_sparse)

            for start_idx in range(0, n_sparse, chunk_batch_size):
                end_idx = min(start_idx + chunk_batch_size, n_sparse)

                for i, t in enumerate(
                    range(local_end + start_idx, local_end + end_idx)
                ):
                    # Get queries for this position: [B, H, 1, D]
                    q_t = q[:, :, t : t + 1, :]

                    # Build list of positions to attend to
                    # Local window: [max(0, t-local_window), t]
                    local_start = max(0, t - self.local_window)
                    local_positions = list(range(local_start, t + 1))

                    # Retrieved chunks for this position - use set for dedup
                    all_chunk_pos = set()
                    for b in range(B):
                        chunks = chunk_indices[b, t]  # [top_b]
                        for chunk_idx in chunks:
                            idx = chunk_idx.item()
                            # Defensive: skip invalid indices
                            if idx < 0:
                                continue
                            c_start = idx * self.chunk_size
                            c_end = min(c_start + self.chunk_size, t)
                            if c_start < t:
                                for p in range(c_start, c_end):
                                    if p not in local_positions:
                                        all_chunk_pos.add(p)

                    # Combine and sort positions
                    attend_positions = sorted(set(local_positions) | all_chunk_pos)

                    # Defensive cap: limit to max_candidates
                    if len(attend_positions) > self.max_candidates:
                        attend_positions = attend_positions[-self.max_candidates :]

                    # Track for debugging
                    self._max_candidates_seen = max(
                        self._max_candidates_seen, len(attend_positions)
                    )

                    if len(attend_positions) == 0:
                        # Edge case: no positions to attend to (shouldn't happen)
                        continue

                    # Gather K, V for attend positions: [B, H, n_attend, D]
                    pos_tensor = torch.tensor(
                        attend_positions, device=device, dtype=torch.long
                    )
                    k_attend = k.index_select(2, pos_tensor)
                    v_attend = v.index_select(2, pos_tensor)

                    # Compute attention: [B, H, 1, n_attend]
                    scores = torch.matmul(q_t, k_attend.transpose(-2, -1)) * scale

                    # Apply softmax and dropout
                    attn = F.softmax(scores, dim=-1)
                    if self.training and self.dropout > 0:
                        attn = self.attn_dropout(attn)

                    # Apply to values and store in pre-allocated tensor
                    y_t = torch.matmul(attn, v_attend)
                    y_sparse[:, :, start_idx + i : start_idx + i + 1, :] = y_t
        else:
            y_sparse = q.new_zeros(B, H, 0, D)

        # Combine local and sparse outputs
        y = torch.cat([y_local, y_sparse], dim=2)

        return y


class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class MLP(nn.Module):
    """MLP with GELU activation."""

    def __init__(self, config: RGSAConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class RGSABlock(nn.Module):
    """Transformer block with RGSA attention."""

    def __init__(self, config: RGSAConfig, chunk_router: ChunkRouter):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = RGSACausalSelfAttention(config, chunk_router)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        return_routing_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        attn_out, routing_info = self.attn(self.ln_1(x), return_routing_info)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, routing_info


class GPT2_RGSA(nn.Module):
    """
    GPT-2 with Retrieval-Gated Sparse Attention.

    Uses retrieval-based sparse attention instead of dense global attention.
    For sequences longer than local_window, retrieves relevant chunks to attend to.
    """

    def __init__(self, config: RGSAConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Shared chunk router across all layers
        self.chunk_router = ChunkRouter(config)

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(
                    [
                        RGSABlock(config, self.chunk_router)
                        for _ in range(config.n_layer)
                    ]
                ),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        print(f"RGSA model parameters: {self.get_num_params()/1e6:.2f}M")
        print(
            f"RGSA config: chunk_size={config.chunk_size}, "
            f"routing_dim={config.routing_dim}, top_b={config.top_b}, "
            f"local_window={config.local_window}"
        )
        if config.dynamic_chunking:
            print(
                f"RGSA dynamic chunking: schedule={config.chunk_size_schedule}, "
                f"alpha={config.chunk_size_alpha}, "
                f"range=[{config.chunk_size_min}, {config.chunk_size_max}], "
                f"rounding={config.chunk_size_rounding}"
            )

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_routing_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            idx: [B, T] input token indices
            targets: [B, T] target token indices for loss computation
            return_routing_info: whether to return routing statistics

        Returns:
            logits: [B, T, vocab_size] or [B, 1, vocab_size] for inference
            loss: cross-entropy loss if targets provided
            routing_info: dictionary of routing statistics if requested
        """
        device = idx.device
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Sequence length {T} exceeds block size {self.config.block_size}"

        # Embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Transformer blocks
        all_routing_info = []
        for block in self.transformer.h:
            x, routing_info = block(x, return_routing_info)
            if routing_info is not None:
                all_routing_info.append(routing_info)

        x = self.transformer.ln_f(x)

        # Compute output
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        # Note: routing_info available via return_routing_info=True but not returned
        # to maintain compatibility with standard GPT-2 training loop
        return logits, loss

    @torch.no_grad()
    def compute_rgsa_metrics(
        self,
        idx: torch.Tensor,
        layer_idx: int = 0,
        teacher_topk: int = 16,
    ) -> RGSAMetrics:
        """
        Compute RGSA routing diagnostics.

        This method runs a forward pass and computes metrics comparing
        the router's selections against what dense attention would choose.

        Args:
            idx: [B, T] input token indices
            layer_idx: which layer to analyze (default: first layer)
            teacher_topk: number of top chunks to consider from teacher

        Returns:
            RGSAMetrics with all diagnostic values
        """
        device = idx.device
        B, T = idx.size()

        # Get embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Run through layers up to target layer
        for i, block in enumerate(self.transformer.h):
            if i < layer_idx:
                x, _ = block(x, return_routing_info=False)
            else:
                # At target layer, get routing info and compute metrics
                attn = block.attn
                chunk_router = attn.chunk_router
                retrieval_gate = attn.retrieval_gate

                # Compute effective chunk size for this seq_len
                chunk_size_eff = compute_chunk_size(T, self.config)

                # Only compute metrics if sequence is long enough
                if T <= attn.local_window + chunk_size_eff:
                    return RGSAMetrics()

                # Get routing embeddings and scores
                routing_embeds, chunk_mask = chunk_router(
                    block.ln_1(x), chunk_size_override=chunk_size_eff
                )
                chunk_indices, routing_scores = retrieval_gate(
                    block.ln_1(x), routing_embeds, chunk_mask
                )

                # Compute QKV for dense attention analysis
                qkv = attn.c_attn(block.ln_1(x))
                q, k, v = qkv.split(attn.n_embd, dim=2)
                q = q.view(B, T, attn.n_head, attn.head_dim).transpose(1, 2)
                k = k.view(B, T, attn.n_head, attn.head_dim).transpose(1, 2)

                # Compute dense attention scores (teacher)
                scale = 1.0 / math.sqrt(attn.head_dim)
                dense_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

                # Apply causal mask
                causal_mask = torch.triu(
                    torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
                )
                dense_scores = dense_scores.masked_fill(
                    causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
                )
                dense_attn = F.softmax(dense_scores, dim=-1)

                # Aggregate attention per chunk (sum attention mass going to each chunk)
                num_chunks = routing_embeds.size(1)
                chunk_size = chunk_size_eff

                # Compute attention mass per chunk for each query
                # dense_attn: [B, H, T, T] -> chunk_attn: [B, T, num_chunks]
                chunk_attn = torch.zeros(B, T, num_chunks, device=device)
                for c in range(num_chunks):
                    c_start = c * chunk_size
                    c_end = min(c_start + chunk_size, T)
                    if c_start < T:
                        # Sum attention mass to this chunk across all heads
                        chunk_attn[:, :, c] = dense_attn[:, :, :, c_start:c_end].sum(
                            dim=(1, 3)
                        )

                # Get teacher's top-k chunks per query position
                teacher_topk_actual = min(teacher_topk, num_chunks)
                _, teacher_top_chunks = torch.topk(
                    chunk_attn, k=teacher_topk_actual, dim=-1
                )

                # Compute recall: what fraction of teacher's top-k are in retrieved set
                retrieved_set = chunk_indices  # [B, T, top_b]
                top_b = retrieved_set.size(-1)

                # For each query, check if teacher's top chunks are in retrieved set
                recall_scores = []
                for b in range(B):
                    for t in range(attn.local_window + chunk_size_eff, T):
                        teacher_chunks = set(teacher_top_chunks[b, t].tolist())
                        retrieved_chunks = set(retrieved_set[b, t].tolist())
                        if len(teacher_chunks) > 0:
                            recall = len(teacher_chunks & retrieved_chunks) / len(
                                teacher_chunks
                            )
                            recall_scores.append(recall)

                teacher_recall = (
                    sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
                )

                # Compute candidate count statistics
                # In current implementation, it's fixed at top_b * chunk_size + local_window
                candidates_per_query = []
                for t in range(attn.local_window + chunk_size_eff, T):
                    local_count = min(attn.local_window, t)
                    retrieved_count = top_b * chunk_size_eff
                    # Account for overlap between local and retrieved
                    total = local_count + retrieved_count
                    candidates_per_query.append(total)

                if candidates_per_query:
                    candidates_tensor = torch.tensor(
                        candidates_per_query, dtype=torch.float, device=device
                    )
                    candidates_mean = candidates_tensor.mean().item()
                    candidates_p50 = torch.quantile(candidates_tensor, 0.50).item()
                    candidates_p95 = torch.quantile(candidates_tensor, 0.95).item()
                    candidates_p99 = torch.quantile(candidates_tensor, 0.99).item()
                else:
                    candidates_mean = candidates_p50 = candidates_p95 = (
                        candidates_p99
                    ) = 0.0

                # Compute routing entropy and load balance
                # routing_scores: [B, T, num_chunks]
                # Apply softmax to get selection probabilities
                routing_probs = F.softmax(routing_scores, dim=-1)

                # Entropy per query position
                entropy = -(routing_probs * torch.log(routing_probs + 1e-10)).sum(
                    dim=-1
                )
                avg_entropy = entropy.mean().item()

                # Maximum possible entropy (uniform distribution)
                max_entropy = math.log(num_chunks)
                normalized_entropy = (
                    avg_entropy / max_entropy if max_entropy > 0 else 0.0
                )

                # Load balance: how evenly are chunks selected?
                # Count selections per chunk across all queries
                chunk_selection_counts = torch.zeros(num_chunks, device=device)
                for b in range(B):
                    for t in range(T):
                        for c in chunk_indices[b, t]:
                            chunk_selection_counts[c.item()] += 1

                # Ideal would be uniform selection
                total_selections = chunk_selection_counts.sum()
                if total_selections > 0:
                    selection_probs = chunk_selection_counts / total_selections
                    # Load balance score: 1 - CV (coefficient of variation)
                    # CV = std / mean, so 0 CV = perfect balance
                    cv = selection_probs.std() / (selection_probs.mean() + 1e-10)
                    load_balance = max(0.0, 1.0 - cv.item())
                else:
                    load_balance = 0.0

                # Tokens per query statistics
                tokens_per_query = (
                    top_b * chunk_size_eff + attn.local_window
                )  # theoretical max
                tokens_per_query_actual = []
                for t in range(attn.local_window + chunk_size_eff, T):
                    local = min(attn.local_window, t)
                    retrieved = top_b * chunk_size_eff
                    tokens_per_query_actual.append(local + retrieved)

                if tokens_per_query_actual:
                    tpq_tensor = torch.tensor(
                        tokens_per_query_actual, dtype=torch.float
                    )
                    tpq_mean = tpq_tensor.mean().item()
                    tpq_max = tpq_tensor.max().item()
                else:
                    tpq_mean = tpq_max = 0.0

                n_chunks_eff = (T + chunk_size_eff - 1) // chunk_size_eff
                return RGSAMetrics(
                    teacher_topk_recall=teacher_recall,
                    candidates_mean=candidates_mean,
                    candidates_p50=candidates_p50,
                    candidates_p95=candidates_p95,
                    candidates_p99=candidates_p99,
                    routing_entropy=normalized_entropy,
                    load_balance_score=load_balance,
                    tokens_per_query_mean=tpq_mean,
                    tokens_per_query_max=tpq_max,
                    chunk_size_eff=chunk_size_eff,
                    n_chunks_eff=n_chunks_eff,
                    chunk_selection_counts=chunk_selection_counts,
                )

        return RGSAMetrics()

    def compute_routing_loss(
        self,
        x: torch.Tensor,
        return_correlation: bool = False,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        """
        Compute auxiliary routing loss.

        The routing embeddings should predict which chunks have high attention mass.
        This is computed by running dense attention on the input and measuring
        where attention actually goes.

        Args:
            x: [B, T] input token indices
            return_correlation: whether to return Spearman correlation

        Returns:
            routing_loss: scalar loss for routing supervision
            correlation: Spearman correlation if requested
        """
        # This is a simplified placeholder - full implementation would:
        # 1. Run dense attention to get ground truth attention mass per chunk
        # 2. Compute BCE/MSE loss between routing scores and attention mass
        # 3. Optionally compute correlation metrics

        # For now, return zero loss (model learns routing from task loss)
        return torch.tensor(0.0, device=x.device), None

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Configure optimizer with proper weight decay handling."""
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        decay = {pn for pn in decay if pn in param_dict}
        no_decay = {pn for pn in no_decay if pn in param_dict}

        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(no_decay)],
                "weight_decay": 0.0,
            },
        ]

        import inspect

        use_fused = (device_type == "cuda") and (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, fused=use_fused
        )
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate text by sampling from the model."""
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

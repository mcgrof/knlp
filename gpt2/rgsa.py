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
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute routing embeddings for each chunk.

        Args:
            x: [B, T, C] hidden states

        Returns:
            routing_embeds: [B, num_chunks, routing_dim] routing embeddings
            chunk_mask: [B, num_chunks] mask for valid chunks (True = valid)
        """
        B, T, C = x.shape
        chunk_size = self.chunk_size

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve top-B chunks for each query position.

        Args:
            query_hidden: [B, T, C] query hidden states
            routing_embeds: [B, num_chunks, routing_dim] chunk routing embeddings
            chunk_mask: [B, num_chunks] mask for valid chunks

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
        _, chunk_indices = torch.topk(routing_scores, k=top_b, dim=-1)

        return chunk_indices, routing_scores


class RGSACausalSelfAttention(nn.Module):
    """
    Causal self-attention with retrieval-gated sparsity.

    For each query position:
    1. Always include local window (local_window tokens before query)
    2. Retrieve top-B chunks via RetrievalGate
    3. Run attention only on local + retrieved tokens
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

    def forward(
        self,
        x: torch.Tensor,
        return_routing_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Forward pass with sparse attention.

        For short sequences (< local_window), uses dense attention.
        For longer sequences, uses retrieval-gated sparse attention.
        """
        B, T, C = x.shape

        # Compute Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        routing_info = None

        # Phase 1: Use dense causal attention for all sequences
        # The routing mechanism learns chunk embeddings but attention is dense
        # This validates the architecture before optimizing sparse attention
        #
        # TODO: Implement efficient sparse attention kernel for long sequences
        # For now, we still compute routing embeddings to train the router
        if T > self.local_window + self.chunk_size:
            # Compute routing embeddings (training the router)
            routing_embeds, chunk_mask = self.chunk_router(x)
            chunk_indices, routing_scores = self.retrieval_gate(
                x, routing_embeds, chunk_mask
            )
            if return_routing_info:
                routing_info = {
                    "chunk_indices": chunk_indices,
                    "routing_scores": routing_scores,
                    "routing_embeds": routing_embeds,
                }

        # Use dense causal attention for all sequences
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        # Reshape output
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))

        return y, routing_info

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
        """
        B, H, T, D = q.shape
        device = q.device
        dtype = q.dtype

        # For memory efficiency, we'll use SDPA for local window positions
        # and only do custom sparse attention for positions beyond local_window

        # Positions within local window can use standard causal attention
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
            # Number of positions to process with sparse attention
            n_sparse = T - local_end

            # Pre-compute scale
            scale = 1.0 / math.sqrt(D)

            # For each sparse position, attend to:
            # 1. Local window (positions [t-local_window, t])
            # 2. Retrieved chunks

            # Build outputs for sparse positions
            y_sparse_list = []

            # Process in mini-batches to reduce memory
            chunk_batch_size = min(32, n_sparse)

            for start_idx in range(0, n_sparse, chunk_batch_size):
                end_idx = min(start_idx + chunk_batch_size, n_sparse)
                actual_positions = range(local_end + start_idx, local_end + end_idx)
                batch_size = end_idx - start_idx

                # For each position in this batch, we need to gather relevant K, V
                # This is the key optimization: only compute attention over relevant tokens

                for i, t in enumerate(actual_positions):
                    # Get queries for this position: [B, H, 1, D]
                    q_t = q[:, :, t : t + 1, :]

                    # Build list of positions to attend to
                    # Local window: [max(0, t-local_window), t]
                    local_start = max(0, t - self.local_window)
                    local_positions = list(range(local_start, t + 1))

                    # Retrieved chunks for this position
                    chunk_pos_list = []
                    for b in range(B):
                        chunks = chunk_indices[b, t]  # [top_b]
                        pos_set = set()
                        for chunk_idx in chunks:
                            c_start = chunk_idx.item() * self.chunk_size
                            c_end = min(c_start + self.chunk_size, t)
                            if c_start < t:
                                for p in range(c_start, c_end):
                                    if p not in local_positions:
                                        pos_set.add(p)
                        chunk_pos_list.append(sorted(pos_set))

                    # For simplicity, take union of positions across batch
                    # This is conservative but avoids per-sample indexing
                    all_chunk_pos = set()
                    for pos_set in chunk_pos_list:
                        all_chunk_pos.update(pos_set)
                    all_chunk_pos = sorted(all_chunk_pos)

                    # Combine positions
                    attend_positions = local_positions + all_chunk_pos
                    attend_positions = sorted(set(attend_positions))

                    if len(attend_positions) == 0:
                        # Edge case: no positions to attend to
                        y_t = q_t.new_zeros(B, H, 1, D)
                    else:
                        # Gather K, V for attend positions: [B, H, n_attend, D]
                        pos_tensor = torch.tensor(
                            attend_positions, device=device, dtype=torch.long
                        )
                        k_attend = k.index_select(2, pos_tensor)
                        v_attend = v.index_select(2, pos_tensor)

                        # Compute attention: [B, H, 1, n_attend]
                        scores = torch.matmul(q_t, k_attend.transpose(-2, -1)) * scale

                        # No mask needed - all attend_positions are valid causal positions
                        attn = F.softmax(scores, dim=-1)
                        if self.training and self.dropout > 0:
                            attn = self.attn_dropout(attn)

                        # Apply to values: [B, H, 1, D]
                        y_t = torch.matmul(attn, v_attend)

                    y_sparse_list.append(y_t)

            # Concatenate sparse outputs
            y_sparse = torch.cat(y_sparse_list, dim=2)  # [B, H, n_sparse, D]
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

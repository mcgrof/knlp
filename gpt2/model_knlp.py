"""
GPT-2 KNLP: Experimental GPT-2 variants with custom attention features.

This module extends the baseline GPT-2 with experimental features while
keeping the baseline pristine. Features include:

- SDPA Output Gating (Qwen3-style): Adds non-linearity after attention via
  sigmoid-modulated gating. Based on "Gated Attention for Large Language
  Models" (NeurIPS 2025 Oral).

These features are motivated by empirical observations that trained models
use far fewer effective attention dimensions than available. KV compression
experiments (2-4x cache reduction with <1% PPL increase) demonstrate this
redundancy. See https://github.com/mcgrof/attention-low-rank for analysis.

Usage:
    from gpt2.model_knlp import GPT2_KNLP, GPT2_KNLP_Config

    config = GPT2_KNLP_Config(use_sdpa_gate=True)
    model = GPT2_KNLP(config)
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from gpt2.model import LayerNorm, MLP, GPTConfig


@dataclass
class GPT2_KNLP_Config(GPTConfig):
    """
    Configuration for GPT2_KNLP with experimental features.

    Inherits all GPTConfig parameters plus:
    - use_sdpa_gate: Enable SDPA output gating (Qwen3-style)
    """

    # SDPA Output Gating (Qwen3-style)
    use_sdpa_gate: bool = False


class CausalSelfAttention_KNLP(nn.Module):
    """
    Multi-head causal self-attention with optional experimental features.

    Features:
    - SDPA output gating: y = attn_output * sigmoid(W_gate @ x)
    """

    def __init__(
        self,
        config: GPT2_KNLP_Config,
        use_sdpa_gate: bool = False,
    ):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Standard attention projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = config.n_embd // config.n_head

        # SDPA Output Gating (Qwen3-style)
        # Based on "Gated Attention for Large Language Models" (NeurIPS 2025 Oral)
        # Gate is computed from input x, applied element-wise to attention output
        self.use_sdpa_gate = use_sdpa_gate
        if use_sdpa_gate:
            self.sdpa_gate = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        else:
            self.register_parameter("sdpa_gate", None)

    def forward(self, x, mechint_kv_mask=None):
        B, T, C = x.size()

        # Compute Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Apply mechint KV masks if provided (for circuit discovery)
        if mechint_kv_mask is not None:
            k, v, _ = mechint_kv_mask(k, v)

        # Base attention: standard SDPA
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # SDPA output gating: apply sigmoid gate to attention output
        if self.use_sdpa_gate:
            gate = torch.sigmoid(self.sdpa_gate(x))
            y = y * gate

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block_KNLP(nn.Module):
    """Transformer block with optional KNLP features."""

    def __init__(
        self,
        config: GPT2_KNLP_Config,
        use_sdpa_gate: bool = False,
    ):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention_KNLP(
            config,
            use_sdpa_gate=use_sdpa_gate,
        )
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2_KNLP(nn.Module):
    """
    GPT-2 with KNLP experimental features.

    This keeps the baseline GPT-2 pristine while allowing experimentation
    with attention mechanisms. Features are controlled via config flags.
    """

    def __init__(self, config: GPT2_KNLP_Config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(
                    [
                        Block_KNLP(
                            config,
                            use_sdpa_gate=config.use_sdpa_gate,
                        )
                        for i in range(config.n_layer)
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

        print(f"GPT2_KNLP parameters: {self.get_num_params()/1e6:.2f}M")
        print(f"  SDPA gating: {config.use_sdpa_gate}")

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

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Sequence length {t} exceeds block size {self.config.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate tokens autoregressively."""
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

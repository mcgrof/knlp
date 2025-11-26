"""
MLA + RALATE: MLA variants with RALATE (late layers reciprocal) pattern.

Adds fixed reciprocal attention pattern to MLA compression mechanisms.
RALATE uses standard attention in layers 0-5, reciprocal in layers 6-11.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import sys
import os

# Add parent to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from ra import (
    RA_MLA_Config,
    RotaryEmbedding,
    apply_rope,
    MLA_Flash,
    MLA_KVSplice,
    MLA_KV2_Attention,
    MLPSplice,
    LearnedKVSplice,
)


class MLA_Flash_RALATE(nn.Module):
    """
    MLA with RALATE pattern (late layers reciprocal).

    Layers 0-5: Standard Q @ K.T attention
    Layers 6-11: Reciprocal K @ Q.T attention

    Same MLA compression as base class (KV-latent cache).
    """

    def __init__(self, cfg: RA_MLA_Config, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx

        # RALATE decision: late layers (6-11) use reciprocal
        self.use_reciprocal = layer_idx >= (cfg.n_layers // 2)

        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d_latent = cfg.d_latent

        # Q path - direct projection
        q_dim = cfg.n_heads * cfg.head_dim
        self.W_q = nn.Linear(cfg.d_model, q_dim)

        # KV path - compressed latent
        self.to_kv_latent = nn.Linear(cfg.d_model, cfg.d_latent)
        kv_dim = 2 * cfg.n_heads * cfg.head_dim
        self.from_kv_latent = nn.Linear(cfg.d_latent, kv_dim)

        # Output projection
        self.out_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model)

        # RoPE
        self.rope = RotaryEmbedding(
            cfg.head_dim, max_seq_len=cfg.block_size, theta=cfg.rope_theta
        )

        self.scale = 1.0 / (cfg.head_dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward with RALATE pattern."""
        B, T, D = x.shape

        # Q computed directly
        q = self.W_q(x)
        q = q.view(B, T, self.n_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # [B, H, T, head_dim]

        # KV from compressed latent
        kv_latent = self.to_kv_latent(x)

        if cache is not None:
            full_kv_latent = torch.cat([cache, kv_latent], dim=1)
            T_total = full_kv_latent.shape[1]
        else:
            full_kv_latent = kv_latent
            T_total = T

        # Decompress to K, V
        kv = self.from_kv_latent(full_kv_latent)
        kv = kv.view(B, T_total, 2, self.n_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Apply RoPE
        cos, sin = self.rope(x, T_total)
        if cache is not None:
            q_cos, q_sin = cos[-T:], sin[-T:]
            q, _ = apply_rope(q, q, q_cos, q_sin)
            k, _ = apply_rope(k, k, cos, sin)
        else:
            q, k = apply_rope(q, k, cos, sin)

        # RALATE pattern: fixed at initialization, no runtime branching
        use_causal = True
        if self.use_reciprocal:
            # Reciprocal: K @ Q.T (late layers)
            attn_out = F.scaled_dot_product_attention(
                k[:, :, -T:, :] if cache is not None else k,
                q,
                v,
                is_causal=use_causal,
                dropout_p=self.cfg.dropout if self.training else 0.0,
            )
        else:
            # Standard: Q @ K.T (early layers)
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=use_causal,
                dropout_p=self.cfg.dropout if self.training else 0.0,
            )

        # Merge heads and project
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(B, T, self.n_heads * self.head_dim)
        out = self.out_proj(attn_out)

        new_cache = full_kv_latent if use_cache else None
        return out, new_cache


class MLABlock_RALATE(nn.Module):
    """Transformer block with MLA_Flash_RALATE attention + MLP."""

    def __init__(self, cfg: RA_MLA_Config, layer_idx: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.attn = MLA_Flash_RALATE(cfg, layer_idx)
        self.ln_2 = nn.LayerNorm(cfg.d_model)

        # Standard MLP
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x, cache=None, use_cache=False):
        attn_out, new_cache = self.attn(self.ln_1(x), cache, use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


class GPT2_MLA_RALATE(nn.Module):
    """Full GPT-2 with MLA + RALATE pattern."""

    def __init__(self, cfg: RA_MLA_Config, vocab_size: int = 50257):
        super().__init__()
        self.cfg = cfg

        # Embeddings
        self.wte = nn.Embedding(vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        # Transformer blocks with RALATE
        self.blocks = nn.ModuleList(
            [MLABlock_RALATE(cfg, i) for i in range(cfg.n_layers)]
        )

        # Output
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)

        # Weight tying
        self.wte.weight = self.lm_head.weight
        assert self.wte.weight is self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(torch.arange(T, device=idx.device))
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x, _ = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_pattern_stats(self):
        """Get RALATE pattern statistics."""
        n_layers = self.cfg.n_layers
        n_late = n_layers // 2
        return {
            "pattern": "ralate",
            "n_reciprocal": n_late,
            "n_standard": n_layers - n_late,
            "reciprocal_layers": list(range(n_late, n_layers)),
            "standard_layers": list(range(n_late)),
        }


class MLA_KVSplice_RALATE(nn.Module):
    """MLA with KVSplice compression and RALATE pattern."""

    def __init__(
        self,
        cfg: RA_MLA_Config,
        layer_idx: int,
        compression_ratio: float = 0.5,
    ):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.use_reciprocal = layer_idx >= (cfg.n_layers // 2)

        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d_latent = cfg.d_latent

        # KVSplice compression
        d_compressed = int(cfg.d_latent * compression_ratio)
        self.kvsplice = LearnedKVSplice(cfg.d_latent, d_compressed)
        self.d_compressed = d_compressed

        # Q path - direct
        self.W_q = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim)

        # KV path - with compression
        self.to_latent = nn.Linear(cfg.d_model, cfg.d_latent)
        qkv_dim = 3 * cfg.n_heads * cfg.head_dim
        self.from_latent = nn.Linear(cfg.d_latent, qkv_dim)

        # Output projection
        self.out_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model)

        # RoPE
        self.rope = RotaryEmbedding(
            cfg.head_dim, max_seq_len=cfg.block_size, theta=cfg.rope_theta
        )

        self.scale = 1.0 / (cfg.head_dim**0.5)
        self._last_reconstruction_error = None

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward with KVSplice compression and RALATE pattern."""
        B, T, D = x.shape

        # Q computed directly
        q = self.W_q(x)
        q = q.view(B, T, self.n_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)

        # KV from compressed latent
        latent_orig = self.to_latent(x)
        latent = self.kvsplice(latent_orig)

        # Track reconstruction error occasionally
        if self.training and torch.rand(1).item() < 0.01:
            self._last_reconstruction_error = self.kvsplice.get_reconstruction_error(
                latent_orig
            ).item()

        # Handle cache
        if cache is not None:
            cache_decompressed = self.kvsplice.decompress_only(cache)
            full_latent = torch.cat([cache_decompressed, latent], dim=1)
            T_total = full_latent.shape[1]
        else:
            full_latent = latent
            T_total = T

        # Decompress to Q, K, V (we only use K, V from latent)
        qkv = self.from_latent(full_latent)
        qkv = qkv.view(B, T_total, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        _, k, v = qkv[0], qkv[1], qkv[2]  # Ignore Q from latent, use direct Q

        # Apply RoPE
        cos, sin = self.rope(x, T_total)
        if cache is not None:
            q_cos, q_sin = cos[-T:], sin[-T:]
            q, _ = apply_rope(q, q, q_cos, q_sin)
            k, _ = apply_rope(k, k, cos, sin)
        else:
            q, k = apply_rope(q, k, cos, sin)

        # RALATE pattern
        use_causal = True
        if self.use_reciprocal:
            # Reciprocal: K @ Q.T
            attn_out = F.scaled_dot_product_attention(
                k[:, :, -T:, :] if cache is not None else k,
                q,
                v,
                is_causal=use_causal,
                dropout_p=self.cfg.dropout if self.training else 0.0,
            )
        else:
            # Standard: Q @ K.T
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=use_causal,
                dropout_p=self.cfg.dropout if self.training else 0.0,
            )

        # Merge heads and project
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(B, T, self.n_heads * self.head_dim)
        out = self.out_proj(attn_out)

        new_cache = self.kvsplice.compress_only(full_latent) if use_cache else None
        return out, new_cache


class MLAKVBlock_RALATE(nn.Module):
    """Block with MLA_KVSplice_RALATE attention + MLP."""

    def __init__(
        self,
        cfg: RA_MLA_Config,
        layer_idx: int,
        compression_ratio: float,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.attn = MLA_KVSplice_RALATE(cfg, layer_idx, compression_ratio)
        self.ln_2 = nn.LayerNorm(cfg.d_model)

        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x, cache=None, use_cache=False):
        attn_out, new_cache = self.attn(self.ln_1(x), cache, use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


class GPT2_MLA_KV_RALATE(nn.Module):
    """Full GPT-2 with MLA + KVSplice + RALATE pattern."""

    def __init__(
        self,
        cfg: RA_MLA_Config,
        vocab_size: int = 50257,
        compression_ratio: float = 0.5,
    ):
        super().__init__()
        self.cfg = cfg
        self.compression_ratio = compression_ratio

        # Embeddings
        self.wte = nn.Embedding(vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        # Transformer blocks with RALATE
        self.blocks = nn.ModuleList(
            [
                MLAKVBlock_RALATE(cfg, i, compression_ratio)
                for i in range(cfg.n_layers)
            ]
        )

        # Output
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)

        # Weight tying
        self.wte.weight = self.lm_head.weight
        assert self.wte.weight is self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(torch.arange(T, device=idx.device))
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x, _ = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_compression_stats(self):
        d_compressed = int(self.cfg.d_latent * self.compression_ratio)
        return {
            "d_latent": self.cfg.d_latent,
            "d_compressed": d_compressed,
            "compression_ratio": self.compression_ratio,
            "cache_reduction": f"{(1 - self.compression_ratio) * 100:.1f}%",
            "pattern": "ralate",
        }

    def get_pattern_stats(self):
        """Get RALATE pattern statistics."""
        n_layers = self.cfg.n_layers
        n_late = n_layers // 2
        return {
            "pattern": "ralate",
            "n_reciprocal": n_late,
            "n_standard": n_layers - n_late,
            "reciprocal_layers": list(range(n_late, n_layers)),
            "standard_layers": list(range(n_late)),
        }


class MLA_KV2_Attention_RALATE(nn.Module):
    """MLA with 2 separate K/V latents and RALATE pattern."""

    def __init__(
        self,
        cfg: RA_MLA_Config,
        layer_idx: int,
        compression_ratio: float = 0.5,
    ):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.use_reciprocal = layer_idx >= (cfg.n_layers // 2)

        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d_latent = cfg.d_latent

        # Q path - direct
        q_dim = cfg.n_heads * cfg.head_dim
        self.W_q = nn.Linear(cfg.d_model, q_dim)

        # K path - separate compressed latent
        d_k_compressed = int(cfg.d_latent * compression_ratio)
        self.to_k_latent = nn.Linear(cfg.d_model, cfg.d_latent)
        self.k_splice = LearnedKVSplice(cfg.d_latent, d_k_compressed)
        k_dim = cfg.n_heads * cfg.head_dim
        self.from_k_latent = nn.Linear(cfg.d_latent, k_dim)

        # V path - separate compressed latent
        d_v_compressed = int(cfg.d_latent * compression_ratio)
        self.to_v_latent = nn.Linear(cfg.d_model, cfg.d_latent)
        self.v_splice = LearnedKVSplice(cfg.d_latent, d_v_compressed)
        v_dim = cfg.n_heads * cfg.head_dim
        self.from_v_latent = nn.Linear(cfg.d_latent, v_dim)

        # Output projection
        self.out_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model)

        # RoPE
        self.rope = RotaryEmbedding(
            cfg.head_dim, max_seq_len=cfg.block_size, theta=cfg.rope_theta
        )

        self.scale = 1.0 / (cfg.head_dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[tuple] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[tuple]]:
        """Forward with 2-latent caching and RALATE pattern."""
        B, T, D = x.shape

        # Q: Direct computation
        q = self.W_q(x)
        q = q.view(B, T, self.n_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)

        # K: Compress latent
        k_latent_orig = self.to_k_latent(x)
        k_latent = self.k_splice(k_latent_orig)

        # V: Compress latent
        v_latent_orig = self.to_v_latent(x)
        v_latent = self.v_splice(v_latent_orig)

        # Handle cache
        if cache is not None:
            k_cache, v_cache = cache
            k_cache_decompressed = self.k_splice.decompress_only(k_cache)
            v_cache_decompressed = self.v_splice.decompress_only(v_cache)
            full_k_latent = torch.cat([k_cache_decompressed, k_latent], dim=1)
            full_v_latent = torch.cat([v_cache_decompressed, v_latent], dim=1)
            T_total = full_k_latent.shape[1]
        else:
            full_k_latent = k_latent
            full_v_latent = v_latent
            T_total = T

        # Decompress to K, V
        k = self.from_k_latent(full_k_latent)
        k = k.view(B, T_total, self.n_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3)

        v = self.from_v_latent(full_v_latent)
        v = v.view(B, T_total, self.n_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3)

        # Apply RoPE
        cos, sin = self.rope(x, T_total)
        if cache is not None:
            q_cos, q_sin = cos[-T:], sin[-T:]
            q, _ = apply_rope(q, q, q_cos, q_sin)
            k, _ = apply_rope(k, k, cos, sin)
        else:
            q, k = apply_rope(q, k, cos, sin)

        # RALATE pattern
        use_causal = True
        if self.use_reciprocal:
            # Reciprocal: K @ Q.T
            attn_out = F.scaled_dot_product_attention(
                k[:, :, -T:, :] if cache is not None else k,
                q,
                v,
                is_causal=use_causal,
                dropout_p=self.cfg.dropout if self.training else 0.0,
            )
        else:
            # Standard: Q @ K.T
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=use_causal,
                dropout_p=self.cfg.dropout if self.training else 0.0,
            )

        # Output
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(B, T, self.n_heads * self.head_dim)
        out = self.out_proj(attn_out)

        # New cache (compressed K and V latents)
        new_cache = None
        if use_cache:
            k_compressed = self.k_splice.compress_only(k_latent_orig)
            v_compressed = self.v_splice.compress_only(v_latent_orig)
            if cache is not None:
                k_cache, v_cache = cache
                k_compressed = torch.cat([k_cache, k_compressed], dim=1)
                v_compressed = torch.cat([v_cache, v_compressed], dim=1)
            new_cache = (k_compressed, v_compressed)

        return out, new_cache


class MLA_KV2_Block_RALATE(nn.Module):
    """Block with MLA_KV2_Attention_RALATE + MLP."""

    def __init__(
        self,
        cfg: RA_MLA_Config,
        layer_idx: int,
        compression_ratio: float,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.attn = MLA_KV2_Attention_RALATE(cfg, layer_idx, compression_ratio)
        self.ln_2 = nn.LayerNorm(cfg.d_model)

        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x, cache=None, use_cache=False):
        attn_out, new_cache = self.attn(self.ln_1(x), cache, use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


class GPT2_MLA_KV2_RALATE(nn.Module):
    """Full GPT-2 with MLA 2-latent + RALATE pattern."""

    def __init__(
        self,
        cfg: RA_MLA_Config,
        vocab_size: int = 50257,
        compression_ratio: float = 0.5,
    ):
        super().__init__()
        self.cfg = cfg
        self.compression_ratio = compression_ratio

        # Embeddings
        self.wte = nn.Embedding(vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        # Transformer blocks with RALATE
        self.blocks = nn.ModuleList(
            [
                MLA_KV2_Block_RALATE(cfg, i, compression_ratio)
                for i in range(cfg.n_layers)
            ]
        )

        # Output
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)

        # Weight tying
        self.wte.weight = self.lm_head.weight
        assert self.wte.weight is self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(torch.arange(T, device=idx.device))
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x, _ = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_compression_stats(self):
        d_compressed = int(self.cfg.d_latent * self.compression_ratio)
        return {
            "d_latent": self.cfg.d_latent,
            "d_compressed": d_compressed,
            "compression_ratio": self.compression_ratio,
            "cache_reduction": f"{(1 - self.compression_ratio) * 100:.1f}%",
            "pattern": "ralate",
        }

    def get_pattern_stats(self):
        """Get RALATE pattern statistics."""
        n_layers = self.cfg.n_layers
        n_late = n_layers // 2
        return {
            "pattern": "ralate",
            "n_reciprocal": n_late,
            "n_standard": n_layers - n_late,
            "reciprocal_layers": list(range(n_late, n_layers)),
            "standard_layers": list(range(n_late)),
        }

"""
Compressed Attention Wrapper (FlashBias Architecture)

Wraps HuggingFace attention modules to integrate KV compression
BEFORE SDPA, not via hooks that run after.

This version uses a *wrapper* around the existing GPT2Attention module
instead of re-instantiating and reloading weights, to avoid any
state_dict / init mismatch that could corrupt the model.

KV compression is applied to (K, V) right before SDPA; when the
compressor is full-rank (rank >= d_head) we structurally bypass it.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Union
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

from .kv_compressor_plugin import KVCompressor, KVCompressionConfig, VResidualAdapter


class KVCompressedGPT2Attention(nn.Module):
    """
    Wrapper around an existing GPT2Attention that injects KV compression
    BEFORE the attention kernel.

    All original parameters, buffers, and behavior of GPT2Attention are
    preserved inside `inner_attn`. The wrapper only changes how K/V are
    transformed before `_attn()` is called.

    This design guarantees that full-rank mode (rank >= d_head) is
    structurally identical to baseline, since the compressor path is
    never taken.
    """

    def __init__(
        self,
        inner_attn: GPT2Attention,
        kv_compressor: Optional[KVCompressor] = None,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.inner_attn = inner_attn
        self.kv_compressor = kv_compressor
        self.layer_idx = layer_idx

    @staticmethod
    def _split_heads(tensor, num_heads: int, head_dim: int):
        """[B, T, H*D] -> [B, H, T, D]"""
        new_shape = tensor.size()[:-1] + (num_heads, head_dim)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)

    @staticmethod
    def _merge_heads(tensor, num_heads: int, head_dim: int):
        """[B, H, T, D] -> [B, T, H*D]"""
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * head_dim,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass that mirrors GPT2Attention.forward, but compresses
        K,V BEFORE calling the attention kernel.

        All projections, scaling, masks, dropout, etc. are delegated to
        `inner_attn` where possible.
        """
        # If no compression needed, just pass through to inner attention
        if self.kv_compressor is None or (
            self.kv_compressor.rank >= self.kv_compressor.d_head
        ):
            # Structural bypass: use original attention unchanged
            return self.inner_attn(
                hidden_states=hidden_states,
                past_key_value=past_key_value or layer_past,
                cache_position=cache_position,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                **kwargs,
            )

        # === Compression path: manually compute attention with compressed K/V ===
        attn = self.inner_attn

        # HF compat: some versions pass `past_key_value` instead of `layer_past`
        if past_key_value is not None:
            layer_past = past_key_value

        # === Standard Q,K,V projections (reuse inner_attn params) ===
        if encoder_hidden_states is not None:
            # Cross-attention path
            if not hasattr(attn, "q_attn"):
                raise ValueError(
                    "Cross-attention requested but inner_attn.q_attn is missing"
                )
            query = attn.q_attn(hidden_states)
            key, value = attn.c_attn(encoder_hidden_states).split(
                attn.split_size, dim=2
            )
            attention_mask = encoder_attention_mask
        else:
            # Self-attention
            query, key, value = attn.c_attn(hidden_states).split(attn.split_size, dim=2)

        # [B, T, H*D] -> [B, H, T, D]
        num_heads = attn.num_heads
        head_dim = attn.head_dim
        query = self._split_heads(query, num_heads, head_dim)
        key = self._split_heads(key, num_heads, head_dim)
        value = self._split_heads(value, num_heads, head_dim)

        # === KV cache handling (unchanged) ===
        if layer_past is not None and len(layer_past) > 0:
            past_key, past_value = layer_past
            # seq dim is -2 (B, H, S, D)
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache:
            # NOTE: cache stores *full* K,V; compression is only for SDPA compute
            present = (key, value)
        else:
            present = None

        # === KV compression BEFORE SDPA ===
        # Apply compression (we already checked rank < d_head above)
        #
        # Special handling for V-only residual adapter mode:
        # VResidualAdapter needs the original V for residual connection
        if isinstance(self.kv_compressor, VResidualAdapter):
            # V-only residual: K stays uncompressed, V uses residual connection
            key_for_attn = key
            value_latent = self.kv_compressor.compress_v(value)
            value_for_attn = self.kv_compressor.expand_v(value_latent, v_original=value)
        else:
            # Standard KV compression
            key_latent, value_latent = self.kv_compressor(key, value)
            key_for_attn = self.kv_compressor.expand_k(key_latent)
            value_for_attn = self.kv_compressor.expand_v(value_latent)

        # === Compute attention with compressed K/V ===
        # Use SDPA directly for simplicity
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key_for_attn,
            value_for_attn,
            attn_mask=attention_mask,
            dropout_p=attn.attn_dropout.p if self.training else 0.0,
            is_causal=attention_mask is None and query.size(-2) > 1,
        )

        # === Merge heads and project output (unchanged) ===
        attn_output = self._merge_heads(attn_output, num_heads, head_dim)
        attn_output = attn.c_proj(attn_output)
        attn_output = attn.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            # Note: we don't have attn_weights when using SDPA
            outputs += (None,)

        return outputs


def wrap_model_with_compression(
    model: nn.Module,
    kv_compressors: Dict[int, KVCompressor],
    model_type: str = "gpt2",
) -> nn.Module:
    """
    Wrap a HuggingFace model's attention layers with KV compression.

    Critically, this does NOT re-instantiate GPT2Attention or reload
    weights. It wraps the existing attention modules in a
    KVCompressedGPT2Attention that holds the original module in
    `inner_attn`.

    This design eliminates the state_dict corruption bug that occurred
    when re-instantiating attention with random weights and then
    loading pretrained weights with strict=False.

    Args:
        model: HuggingFace model (e.g., GPT2LMHeadModel)
        kv_compressors: Dict[layer_idx -> KVCompressor] (may share objects)
        model_type: "gpt2" for now

    Returns:
        The same model instance with wrapped attention layers.

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> from gpt2.compression.kv_compressor_plugin import create_compressor
        >>>
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> compressor = create_compressor("pca", d_head=64, rank=16)
        >>> kv_compressors = {i: compressor for i in range(12)}
        >>> model = wrap_model_with_compression(model, kv_compressors)
    """
    if model_type != "gpt2":
        raise ValueError(f"Unsupported model_type: {model_type}")

    transformer = model.transformer

    for layer_idx, block in enumerate(transformer.h):
        if layer_idx not in kv_compressors:
            continue

        original_attn = block.attn
        kv_comp = kv_compressors[layer_idx]

        # Preserve device/dtype by reusing the original module directly
        # No .to() call needed - original module is already on correct device
        wrapped_attn = KVCompressedGPT2Attention(
            inner_attn=original_attn,
            kv_compressor=kv_comp,
            layer_idx=layer_idx,
        )
        block.attn = wrapped_attn

    return model


def unwrap_model_compression(model: nn.Module, model_type: str = "gpt2") -> nn.Module:
    """
    Remove compression wrappers and restore plain GPT2Attention modules.

    This simply unwraps `KVCompressedGPT2Attention` by replacing it with
    its `inner_attn` module.

    Args:
        model: Model with wrapped attention layers.
        model_type: Model architecture ("gpt2" only).

    Returns:
        The same model with standard attention restored.
    """
    if model_type != "gpt2":
        raise ValueError(f"Unsupported model_type: {model_type}")

    transformer = model.transformer

    for layer_idx, block in enumerate(transformer.h):
        if isinstance(block.attn, KVCompressedGPT2Attention):
            block.attn = block.attn.inner_attn

    return model

#!/usr/bin/env python3
"""
KVSplice plug-in for DeepSeek models.

This module provides a plug-and-play KVSplice compression layer that can be
inserted into pretrained DeepSeek models to reduce KV cache memory usage
during inference.

DeepSeek-V2 already uses MLA (Multi-head Latent Attention) which compresses
the KV cache into a latent space. KVSplice adds an additional compression
layer on top of this latent representation, achieving further memory savings.

Usage:
    from transformers import AutoModelForCausalLM
    from deepseek_kvsplice_plugin import patch_model_with_kvsplice

    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V2-Lite")
    patch_model_with_kvsplice(model, compression_ratio=0.5)

    # Model now uses KVSplice for reduced KV cache memory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LearnedKVSplice(nn.Module):
    """
    Learned compression layer for KV cache reduction.

    Applies a learned monotonic transform followed by low-rank projection:
        1. Transform: x_t = x * softplus(scale) + shift
        2. Compress: z = W_c @ x_t
        3. Expand: x_reconstructed = W_e @ z

    Args:
        d_in: Input dimension (MLA latent dimension)
        compression_ratio: Compression ratio (0.5 = 2x compression)
        use_layernorm: Whether to add LayerNorm in latent space
    """

    def __init__(
        self, d_in: int, compression_ratio: float = 0.5, use_layernorm: bool = True
    ):
        super().__init__()
        self.d_in = d_in
        self.d_compressed = max(1, int(d_in * compression_ratio))
        self.compression_ratio = compression_ratio

        # Learned monotonic transform parameters
        self.transform_scale = nn.Parameter(torch.ones(d_in))
        self.transform_shift = nn.Parameter(torch.zeros(d_in))

        # Low-rank projection
        self.compress = nn.Linear(d_in, self.d_compressed, bias=False)
        self.expand = nn.Linear(self.d_compressed, d_in, bias=False)

        # Optional LayerNorm for latent space stabilization
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.latent_ln = nn.LayerNorm(self.d_compressed)

        # Initialize compress/expand as approximate identity
        with torch.no_grad():
            # SVD-based initialization for better reconstruction
            U, S, Vh = torch.linalg.svd(torch.eye(d_in), full_matrices=False)
            self.compress.weight.copy_(U[: self.d_compressed, :])
            self.expand.weight.copy_(Vh[: self.d_compressed, :].T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compress and expand input tensor.

        Args:
            x: Input tensor [batch, seq_len, d_in]

        Returns:
            Reconstructed tensor [batch, seq_len, d_in]
        """
        # Apply learned monotonic transform
        scale = F.softplus(self.transform_scale)
        x_transformed = x * scale + self.transform_shift

        # Compress to lower dimension
        z = self.compress(x_transformed)

        # Optional LayerNorm
        if self.use_layernorm:
            z = self.latent_ln(z)

        # Expand back to original dimension
        x_reconstructed = self.expand(z)

        return x_reconstructed

    def compress_cache(self, kv_latent: torch.Tensor) -> torch.Tensor:
        """
        Compress KV cache latent representation.

        Args:
            kv_latent: KV cache in latent space [batch, seq_len, d_in]

        Returns:
            Compressed cache [batch, seq_len, d_compressed]
        """
        scale = F.softplus(self.transform_scale)
        x_transformed = kv_latent * scale + self.transform_shift
        z = self.compress(x_transformed)

        if self.use_layernorm:
            z = self.latent_ln(z)

        return z

    def expand_cache(self, compressed_cache: torch.Tensor) -> torch.Tensor:
        """
        Expand compressed cache back to original dimension.

        Args:
            compressed_cache: Compressed cache [batch, seq_len, d_compressed]

        Returns:
            Reconstructed cache [batch, seq_len, d_in]
        """
        return self.expand(compressed_cache)


class KVSpliceWrapper(nn.Module):
    """
    Wrapper that adds KVSplice compression to an existing attention layer.

    Intercepts the KV cache and applies compression/expansion transparently.
    """

    def __init__(
        self,
        original_layer: nn.Module,
        d_latent: int,
        compression_ratio: float = 0.5,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.kvsplice = LearnedKVSplice(d_latent, compression_ratio, use_layernorm)

        # Store compressed cache separately
        self._compressed_cache = None

    def forward(self, *args, **kwargs):
        """
        Forward pass with KVSplice compression.

        Intercepts past_key_values, compresses them, and expands before use.
        """
        # Check if past_key_values is provided
        past_key_values = kwargs.get("past_key_values", None)

        if past_key_values is not None and self._compressed_cache is not None:
            # Expand compressed cache before passing to attention
            expanded_cache = self.kvsplice.expand_cache(self._compressed_cache)
            kwargs["past_key_values"] = expanded_cache

        # Run original attention layer
        outputs = self.original_layer(*args, **kwargs)

        # If the layer returns updated cache, compress it
        if isinstance(outputs, tuple) and len(outputs) > 1:
            output, new_cache = outputs[0], outputs[1]
            if new_cache is not None:
                # Compress the new cache
                self._compressed_cache = self.kvsplice.compress_cache(new_cache)
            return (output, new_cache)
        else:
            return outputs

    def reset_cache(self):
        """Clear compressed cache."""
        self._compressed_cache = None


def patch_model_with_kvsplice(
    model: nn.Module,
    compression_ratio: float = 0.5,
    use_layernorm: bool = True,
    layer_pattern: str = "layers",
) -> nn.Module:
    """
    Patch a pretrained model with KVSplice compression.

    Args:
        model: Pretrained model (e.g., DeepSeek-V2)
        compression_ratio: KVSplice compression ratio (0.5 = 2x compression)
        use_layernorm: Whether to use LayerNorm in latent space
        layer_pattern: Attribute name for transformer layers

    Returns:
        Modified model with KVSplice compression
    """
    # Detect model architecture
    if hasattr(model, "model") and hasattr(model.model, layer_pattern):
        layers = getattr(model.model, layer_pattern)
    elif hasattr(model, layer_pattern):
        layers = getattr(model, layer_pattern)
    # GPT-2 style: model.transformer.h
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        raise ValueError(
            f"Could not find layers with pattern '{layer_pattern}' in model. "
            f"Try passing layer_pattern='h' for GPT-2 models."
        )

    # Detect latent dimension from first layer
    # DeepSeek-V2 typically has config.kv_lora_rank for MLA latent dimension
    if hasattr(model.config, "kv_lora_rank"):
        d_latent = model.config.kv_lora_rank
    elif hasattr(model.config, "hidden_size"):
        # Fallback: assume MLA uses hidden_size / num_heads as latent
        d_latent = model.config.hidden_size // model.config.num_attention_heads
    else:
        raise ValueError("Could not detect latent dimension from model config")

    print(f"Detected latent dimension: {d_latent}")
    print(f"Applying KVSplice with compression ratio: {compression_ratio}")
    print(
        f"Compressed dimension: {max(1, int(d_latent * compression_ratio))} (was {d_latent})"
    )

    # Wrap each attention layer with KVSplice
    patched_count = 0
    for i, layer in enumerate(layers):
        # Handle different attention attribute names
        if hasattr(layer, "self_attn"):
            attn_attr = "self_attn"
        elif hasattr(layer, "attn"):
            attn_attr = "attn"
        else:
            continue

        original_attn = getattr(layer, attn_attr)
        wrapped_attn = KVSpliceWrapper(
            original_attn, d_latent, compression_ratio, use_layernorm
        )
        setattr(layer, attn_attr, wrapped_attn)
        patched_count += 1
        if i < 3 or i >= len(layers) - 1:  # Print first 3 and last layer
            print(f"  Patched layer {i}")

    print(f"\nSuccessfully patched {patched_count} layers with KVSplice")
    return model


def get_kv_cache_size(model: nn.Module) -> Tuple[int, int]:
    """
    Estimate KV cache memory usage.

    Returns:
        (original_size_mb, compressed_size_mb)
    """
    # Detect model architecture
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        return (0, 0)

    original_size = 0
    compressed_size = 0

    for layer in layers:
        # Check for wrapped attention (handle both self_attn and attn)
        attn = None
        if hasattr(layer, "self_attn") and isinstance(layer.self_attn, KVSpliceWrapper):
            attn = layer.self_attn
        elif hasattr(layer, "attn") and isinstance(layer.attn, KVSpliceWrapper):
            attn = layer.attn

        if attn is not None:
            kvsplice = attn.kvsplice
            # Assuming batch=1, seq_len=2048, fp16
            seq_len = 2048
            bytes_per_param = 2  # fp16

            original_size += kvsplice.d_in * seq_len * bytes_per_param
            compressed_size += kvsplice.d_compressed * seq_len * bytes_per_param

    original_mb = original_size / (1024 * 1024)
    compressed_mb = compressed_size / (1024 * 1024)

    return (original_mb, compressed_mb)


if __name__ == "__main__":
    print("KVSplice plug-in for DeepSeek models")
    print("=" * 50)
    print()
    print("This module provides KVSplice compression for pretrained models.")
    print("Import this module and use patch_model_with_kvsplice() to add")
    print("compression to your model.")
    print()
    print("Example:")
    print("  from transformers import AutoModelForCausalLM")
    print("  from deepseek_kvsplice_plugin import patch_model_with_kvsplice")
    print()
    print(
        '  model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V2-Lite")'
    )
    print("  patch_model_with_kvsplice(model, compression_ratio=0.5)")

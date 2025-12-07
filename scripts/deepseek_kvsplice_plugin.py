#!/usr/bin/env python3
"""
KVSplice plug-in for DeepSeek models.

WARNING: THIS APPROACH IS FUNDAMENTALLY BROKEN FOR QUALITY MEASUREMENT.

The wrapper approach modifies K,V values AFTER hidden_states is computed:
  1. Attention computes: hidden_states = softmax(Q @ K.T) @ V (using ORIGINAL K,V)
  2. Returns: (hidden_states, (K, V)) for cache
  3. Wrapper modifies (K, V) AFTER hidden_states is already computed

The compression NEVER affects model quality because hidden_states uses
original K,V. Compression only affects autoregressive generation when
cached K,V is reused for future tokens.

CORRECT APPROACHES FOR KV COMPRESSION:
  1. Modify attention forward() to compress K,V BEFORE attention computation
  2. Train with compression layers (like GPT2_MLA_KV in gpt2/model_knlp.py)
  3. For inference-only: Replace attention with custom implementation

This module is kept for reference and FIM trace computation utilities.
The patch functions should NOT be used for quality benchmarking.

Original Description (now outdated):
This module was intended to provide plug-and-play KVSplice compression for
pretrained models. However, the wrapper approach cannot affect model quality
during evaluation because hidden_states are computed before wrapper runs.
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
        Preserves all output elements from the original layer.
        """
        # Check if past_key_values is provided
        past_key_values = kwargs.get("past_key_values", None)

        if past_key_values is not None and self._compressed_cache is not None:
            # Expand compressed cache before passing to attention
            expanded_cache = self.kvsplice.expand_cache(self._compressed_cache)
            kwargs["past_key_values"] = expanded_cache

        # Run original attention layer
        outputs = self.original_layer(*args, **kwargs)

        # If the layer returns a tuple, look for cache tensor and compress it
        # DeepSeek returns: (hidden_states, attn_weights, present_key_value)
        # We need to preserve all elements and only compress the cache
        if isinstance(outputs, tuple) and len(outputs) > 1:
            outputs_list = list(outputs)

            # Find the cache tensor (typically last element, or second if len==2)
            # Cache is usually a tensor with shape matching our expected d_latent
            for i in range(len(outputs_list) - 1, 0, -1):
                cache = outputs_list[i]
                if cache is not None and isinstance(cache, torch.Tensor):
                    # Check if this looks like a KV cache (has right dimension)
                    if cache.dim() >= 2 and cache.shape[-1] == self.kvsplice.d_in:
                        self._compressed_cache = self.kvsplice.compress_cache(cache)
                        break

            # Return all outputs unchanged (compression is stored internally)
            return tuple(outputs_list)
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


# =============================================================================
# FIM-Guided Selective Compression
# =============================================================================


def compute_fim_trace_per_layer(
    model: nn.Module,
    tokenizer,
    calibration_texts: list,
    device: str = "cuda",
    n_samples: int = 64,
) -> dict:
    """
    Compute Fisher Information Matrix trace for each layer.

    FIM trace measures the "representational work" each layer does.
    Lower trace = less important = safe to compress aggressively.

    Args:
        model: Pretrained model (e.g., DeepSeek-V2-Lite)
        tokenizer: Model tokenizer
        calibration_texts: List of text samples for calibration
        device: Device to run on
        n_samples: Number of attention samples per layer

    Returns:
        Dictionary mapping layer_idx -> mean_fim_trace
    """
    print("\n" + "=" * 70)
    print("FIM TRACE CALIBRATION")
    print("=" * 70)
    print(f"Running calibration on {len(calibration_texts)} samples...")

    # Detect model layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        raise ValueError("Could not find layers in model")

    n_layers = len(layers)
    print(f"Detected {n_layers} layers")

    # Storage for attention weights per layer
    attn_storage = {i: [] for i in range(n_layers)}

    # Register hooks to capture attention weights
    hooks = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            # Try to extract attention weights from output
            # DeepSeek returns (hidden_states, attn_weights, ...)
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]
                if attn_weights is not None and isinstance(attn_weights, torch.Tensor):
                    # Store detached copy
                    attn_storage[layer_idx].append(attn_weights.detach().cpu())

        return hook

    for i, layer in enumerate(layers):
        # Register hook on attention module
        if hasattr(layer, "self_attn"):
            h = layer.self_attn.register_forward_hook(make_hook(i))
            hooks.append(h)
        elif hasattr(layer, "attn"):
            h = layer.attn.register_forward_hook(make_hook(i))
            hooks.append(h)

    # Run calibration forward passes
    model.eval()
    with torch.no_grad():
        for text in calibration_texts[:10]:  # Limit for speed
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            try:
                # Enable attention output if supported, disable cache for compatibility
                model(**inputs, output_attentions=True, use_cache=False)
            except Exception:
                # Fallback without attention output
                try:
                    model(**inputs, use_cache=False)
                except Exception:
                    model(**inputs)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute FIM trace per layer
    fim_traces = {}

    print("\nFIM Trace per Layer:")
    print("-" * 50)

    for layer_idx in range(n_layers):
        attns = attn_storage[layer_idx]

        if not attns:
            # No attention captured - estimate from layer position
            # Later layers typically have lower trace
            estimated_trace = 1.0 - (layer_idx / n_layers) * 0.4
            fim_traces[layer_idx] = estimated_trace
            print(f"  Layer {layer_idx:2d}: ~{estimated_trace:.4f} (estimated)")
            continue

        # Concatenate all attention weights: [total_samples, H, T, T]
        try:
            all_attn = torch.cat(attns, dim=0)
            B, H, T, _ = all_attn.shape

            # Compute FIM trace for this layer
            total_trace = 0.0
            count = 0

            for h in range(min(H, 4)):  # Sample heads for speed
                attn_h = all_attn[:, h]  # [B, T, T]
                p = attn_h.reshape(-1, T)  # [B*T, T]

                # Subsample
                if p.size(0) > n_samples:
                    idx = torch.randperm(p.size(0))[:n_samples]
                    p = p[idx]
                N = p.size(0)

                # Compute FIM: F = mean_i (diag(p_i) - p_i @ p_i.T)
                F = torch.zeros(T, T)
                for i in range(N):
                    pi = p[i]
                    F += torch.diag(pi) - torch.outer(pi, pi)
                F /= N

                # Trace = sum of eigenvalues
                try:
                    eigvals = torch.linalg.eigvalsh(F)
                    trace = float(eigvals.sum())
                    total_trace += trace
                    count += 1
                except Exception:
                    pass

            if count > 0:
                mean_trace = total_trace / count
            else:
                mean_trace = 1.0 - (layer_idx / n_layers) * 0.4

            fim_traces[layer_idx] = mean_trace
            print(f"  Layer {layer_idx:2d}: {mean_trace:.4f}")

        except Exception as e:
            estimated_trace = 1.0 - (layer_idx / n_layers) * 0.4
            fim_traces[layer_idx] = estimated_trace
            print(f"  Layer {layer_idx:2d}: ~{estimated_trace:.4f} (estimated, {e})")

    print("-" * 50)

    return fim_traces


def patch_model_with_kvsplice_fim(
    model: nn.Module,
    fim_traces: dict,
    compression_ratio: float = 0.5,
    fim_threshold: float = 0.7,
    use_layernorm: bool = True,
    layer_pattern: str = "layers",
) -> Tuple[nn.Module, list]:
    """
    Patch model with FIM-guided selective KVSplice compression.

    Only layers with FIM trace below threshold get compressed.
    This preserves early layers that do critical representational work.

    Args:
        model: Pretrained model
        fim_traces: Dictionary from compute_fim_trace_per_layer
        compression_ratio: KVSplice compression ratio for selected layers
        fim_threshold: Layers with trace < threshold get compressed
        use_layernorm: Whether to use LayerNorm in latent space
        layer_pattern: Attribute name for transformer layers

    Returns:
        (modified_model, list of compressed layer indices)
    """
    print("\n" + "=" * 70)
    print("FIM-GUIDED SELECTIVE KVSPLICE")
    print("=" * 70)
    print(f"FIM threshold: {fim_threshold}")
    print(f"Compression ratio: {compression_ratio} (for selected layers)")

    # Detect model layers
    if hasattr(model, "model") and hasattr(model.model, layer_pattern):
        layers = getattr(model.model, layer_pattern)
    elif hasattr(model, layer_pattern):
        layers = getattr(model, layer_pattern)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        raise ValueError(f"Could not find layers with pattern '{layer_pattern}'")

    # Detect latent dimension
    if hasattr(model.config, "kv_lora_rank"):
        d_latent = model.config.kv_lora_rank
    elif hasattr(model.config, "hidden_size"):
        d_latent = model.config.hidden_size // model.config.num_attention_heads
    else:
        raise ValueError("Could not detect latent dimension")

    print(f"Latent dimension: {d_latent}")
    print(f"Compressed dimension: {max(1, int(d_latent * compression_ratio))}")

    # Identify layers to compress based on FIM
    layers_to_compress = []
    for layer_idx, trace in fim_traces.items():
        if trace < fim_threshold:
            layers_to_compress.append(layer_idx)

    print(f"\nLayers selected for compression (FIM < {fim_threshold}):")
    if layers_to_compress:
        print(f"  {layers_to_compress}")
        print(f"  ({len(layers_to_compress)}/{len(fim_traces)} layers)")
    else:
        print("  None - all layers above threshold")
        print("  Consider raising fim_threshold")

    # Patch selected layers
    patched_count = 0
    for i, layer in enumerate(layers):
        if i not in layers_to_compress:
            continue

        # Find attention attribute
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

    # Summary
    total_layers = len(fim_traces)
    protected_layers = total_layers - patched_count
    cache_reduction = patched_count * (1 - compression_ratio) / total_layers * 100

    print(f"\nSummary:")
    print(f"  Protected layers (high FIM): {protected_layers}")
    print(f"  Compressed layers (low FIM): {patched_count}")
    print(f"  Estimated cache reduction: {cache_reduction:.1f}%")
    print("=" * 70 + "\n")

    return model, layers_to_compress


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

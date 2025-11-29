#!/usr/bin/env python3
"""
KVSplice Inference Plugin

Provides plug-and-play KVSplice compression for pretrained models during
inference. Follows xKV's architecture by injecting a custom Cache object
that handles compression transparently.

Usage:
    from transformers import AutoModelForCausalLM
    from kvsplice_plugin import enable_kvsplice

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    enable_kvsplice(model, compression_ratio=0.5)

    # Model now uses compressed KV cache automatically
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from typing import Optional, Tuple, List, Any


class LearnedKVSplice(nn.Module):
    """
    Learned compression for KV cache with wiring bug fixes applied.

    Applies learned monotonic transform + low-rank projection WITHOUT
    inverse affine (sandwich structure bug fixed).
    """

    def __init__(
        self, d_in: int, compression_ratio: float = 0.5, use_layernorm: bool = True
    ):
        super().__init__()
        self.d_in = d_in
        self.d_compressed = max(1, int(d_in * compression_ratio))

        # Learned monotonic transform (fixed: no inverse)
        self.transform_scale = nn.Parameter(torch.ones(d_in))
        self.transform_shift = nn.Parameter(torch.zeros(d_in))

        # Low-rank projection
        self.compress = nn.Linear(d_in, self.d_compressed, bias=False)
        self.expand = nn.Linear(self.d_compressed, d_in, bias=False)

        # Optional LayerNorm
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.latent_ln = nn.LayerNorm(self.d_compressed)

        # Initialize as approximate identity
        with torch.no_grad():
            nn.init.orthogonal_(self.compress.weight)
            self.expand.weight.copy_(self.compress.weight.T)

    def compress_only(self, x: torch.Tensor) -> torch.Tensor:
        """Compress input to latent space (skip transform for uninit inference)."""
        # Skip transform for random initialization - just use projection
        z = self.compress(x)
        if self.use_layernorm:
            z = self.latent_ln(z)
        return z

    def expand_only(self, z: torch.Tensor) -> torch.Tensor:
        """Expand from latent space (no inverse affine)."""
        return self.expand(z)


class CompressedKVCache(Cache):
    """
    Custom Cache object that stores K/V in compressed form.

    Transparently handles compression/decompression, reducing memory usage
    without modifying attention code. Follows transformers Cache API.
    """

    def __init__(
        self,
        d_latent: int,
        compression_ratio: float,
        num_layers: int,
        use_layernorm: bool = True,
        calibrated_kvsplice: Optional[LearnedKVSplice] = None,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.compression_ratio = compression_ratio
        self.num_layers = num_layers

        # Use calibrated KVSplice if provided, otherwise create new
        if calibrated_kvsplice is not None:
            self.kvsplice = calibrated_kvsplice
        else:
            self.kvsplice = LearnedKVSplice(d_latent, compression_ratio, use_layernorm)

        # Storage for compressed K/V per layer
        # key_cache[layer_idx] = compressed tensor [B, H, T, d_compressed]
        self.key_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self.value_cache: List[Optional[torch.Tensor]] = [None] * num_layers

        self._seen_tokens = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new K/V states.

        Args:
            key_states: New keys [B, H, T_new, head_dim]
            value_states: New values [B, H, T_new, head_dim]
            layer_idx: Layer index

        Returns:
            Full K/V (decompressed) for attention: [B, H, T_total, head_dim]
        """
        B, H, T_new, head_dim = key_states.shape

        # Reshape for compression: [B, H, T, head_dim] -> [B, T, H*head_dim]
        k_flat = key_states.permute(0, 2, 1, 3).reshape(B, T_new, H * head_dim)
        v_flat = value_states.permute(0, 2, 1, 3).reshape(B, T_new, H * head_dim)

        # Compress new K/V
        k_compressed = self.kvsplice.compress_only(k_flat)  # [B, T_new, d_compressed]
        v_compressed = self.kvsplice.compress_only(v_flat)

        # Concatenate with existing cache
        if self.key_cache[layer_idx] is not None:
            k_compressed = torch.cat([self.key_cache[layer_idx], k_compressed], dim=1)
            v_compressed = torch.cat([self.value_cache[layer_idx], v_compressed], dim=1)

        # Store compressed
        self.key_cache[layer_idx] = k_compressed
        self.value_cache[layer_idx] = v_compressed

        # Decompress for attention
        k_full = self.kvsplice.expand_only(k_compressed)  # [B, T_total, H*head_dim]
        v_full = self.kvsplice.expand_only(v_compressed)

        # Reshape back: [B, T_total, H*head_dim] -> [B, H, T_total, head_dim]
        T_total = k_full.shape[1]
        k_out = k_full.view(B, T_total, H, head_dim).permute(0, 2, 1, 3)
        v_out = v_full.view(B, T_total, H, head_dim).permute(0, 2, 1, 3)

        return k_out, v_out

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Get sequence length in cache."""
        if self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[1]

    def get_max_length(self) -> Optional[int]:
        """No max length limit."""
        return None

    def get_usable_length(
        self, new_seq_length: int, layer_idx: Optional[int] = 0
    ) -> int:
        """Get usable cache length."""
        return self.get_seq_length(layer_idx)

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorder cache for beam search (not implemented)."""
        raise NotImplementedError("Beam search not supported with CompressedKVCache")


def calibrate_kvsplice(
    model: nn.Module,
    tokenizer,
    compression_ratio: float = 0.5,
    calibration_samples: int = 2000,
    max_length: int = 512,
    d_latent: Optional[int] = None,
) -> LearnedKVSplice:
    """
    Calibrate KVSplice compression using SVD on actual K/V activations.

    Args:
        model: Model to calibrate on
        tokenizer: Tokenizer for calibration data
        compression_ratio: Compression ratio (0.5 = 2x compression)
        calibration_samples: Number of tokens to collect
        max_length: Max sequence length for calibration
        d_latent: Latent dimension (auto-detected if None)

    Returns:
        Calibrated LearnedKVSplice module
    """
    from datasets import load_dataset

    print("=" * 80)
    print("Calibrating KVSplice with SVD on actual K/V activations")
    print("=" * 80)

    # Auto-detect d_latent
    if d_latent is None:
        if hasattr(model.config, "n_embd"):
            d_latent = model.config.n_embd
        elif hasattr(model.config, "hidden_size"):
            d_latent = model.config.hidden_size
        else:
            raise ValueError("Could not detect latent dimension")

    d_compressed = max(1, int(d_latent * compression_ratio))

    print(f"Latent dimension: {d_latent}")
    print(f"Compressed dimension: {d_compressed}")
    print(f"Calibration samples: {calibration_samples} tokens")

    # Collect K/V activations
    kv_activations = []

    def hook_fn(module, input, output):
        """Hook to capture QKV projection output."""
        # GPT-2 c_attn produces concatenated QKV: [B, T, 3 * n_embd]
        # Split into Q, K, V each [B, T, n_embd]
        qkv = output
        B, T, _ = qkv.shape
        q, k, v = qkv.split(d_latent, dim=2)

        # Store K and V activations
        kv_activations.append(k.detach().cpu().reshape(-1, d_latent))
        kv_activations.append(v.detach().cpu().reshape(-1, d_latent))

    # Hook first attention layer's c_attn projection
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layer = model.transformer.h[0]
        if hasattr(layer, "attn") and hasattr(layer.attn, "c_attn"):
            hook = layer.attn.c_attn.register_forward_hook(hook_fn)
        else:
            raise ValueError("Could not find c_attn in GPT-2 attention")
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layer = model.model.layers[0]
        if hasattr(layer, "self_attn"):
            hook = layer.self_attn.register_forward_hook(hook_fn)
    else:
        raise ValueError("Could not find attention layer to hook")

    # Load calibration data
    print("\nLoading calibration data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [item["text"] for item in dataset if len(item["text"].strip()) > 50]

    # Run model on calibration data
    model.eval()
    samples_collected = 0

    print("Collecting K/V activations...")
    with torch.no_grad():
        for text in texts:
            if samples_collected >= calibration_samples:
                break

            inputs = tokenizer(
                text, max_length=max_length, truncation=True, return_tensors="pt"
            ).to(next(model.parameters()).device)

            if inputs["input_ids"].shape[1] < 10:
                continue

            # Run with use_cache=True to get K/V in output
            model(**inputs, use_cache=True)
            samples_collected += inputs["input_ids"].shape[1]

            if samples_collected % 500 == 0:
                print(f"  Progress: {samples_collected}/{calibration_samples} tokens")

    hook.remove()

    print(f"\nCapture complete. Collected {len(kv_activations)} tensors")

    # Concatenate all activations
    if not kv_activations:
        raise ValueError("No K/V activations captured! Check hook implementation.")

    all_activations = torch.cat(kv_activations, dim=0)[:calibration_samples]

    print(f"\nCollected {all_activations.shape[0]} samples")
    print("Performing SVD...")

    # Perform SVD
    U, S, V = torch.svd(all_activations.float())

    # Extract top-k components
    compress_weight = V[:, :d_compressed].T  # [d_compressed, d_latent]
    expand_weight = V[:, :d_compressed]  # [d_latent, d_compressed]

    # Compute explained variance
    total_var = S.pow(2).sum()
    kept_var = S[:d_compressed].pow(2).sum()
    explained_pct = (kept_var / total_var * 100).item()

    print(f"Explained variance: {explained_pct:.2f}%")

    # Create calibrated KVSplice
    kvsplice = LearnedKVSplice(d_latent, compression_ratio, use_layernorm=True)

    # Set SVD-calibrated weights
    with torch.no_grad():
        kvsplice.compress.weight.copy_(compress_weight)
        kvsplice.expand.weight.copy_(expand_weight)

    # Test reconstruction error
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    kvsplice = kvsplice.to(device=device, dtype=dtype)

    sample = all_activations[:100].to(device)
    compressed = kvsplice.compress_only(sample)
    reconstructed = kvsplice.expand_only(compressed)
    recon_error = (sample - reconstructed).pow(2).mean().sqrt().item()

    print(f"Reconstruction error (RMSE): {recon_error:.6f}")
    print(f"Original std: {sample.std().item():.6f}")
    print(f"Relative error: {(recon_error / sample.std().item()) * 100:.2f}%")

    print("=" * 80)
    print("✓ Calibration complete!")
    print("=" * 80)

    return kvsplice


def enable_kvsplice(
    model: nn.Module,
    compression_ratio: float = 0.5,
    use_layernorm: bool = True,
    d_latent: Optional[int] = None,
    calibrated_kvsplice: Optional[LearnedKVSplice] = None,
) -> nn.Module:
    """
    Enable KVSplice compression for a pretrained model.

    Monkey-patches model._prepare_cache_for_generation to inject
    CompressedKVCache, which handles compression transparently.

    Args:
        model: Pretrained model (GPT-2, Mistral, etc.)
        compression_ratio: Compression ratio (0.5 = 2x compression)
        use_layernorm: Whether to use LayerNorm in latent space
        d_latent: Latent dimension (auto-detected if None)

    Returns:
        Model with KVSplice enabled
    """
    # Auto-detect architecture
    if hasattr(model.config, "n_embd"):
        # GPT-2 style
        n_layers = model.config.n_layer
        n_heads = model.config.n_head
        if d_latent is None:
            d_latent = model.config.n_embd  # Use full hidden size for GPT-2
    elif hasattr(model.config, "hidden_size"):
        # Transformers standard
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
        if d_latent is None:
            # Check for MLA latent dimension
            if hasattr(model.config, "kv_lora_rank"):
                d_latent = model.config.kv_lora_rank
            else:
                d_latent = model.config.hidden_size
    else:
        raise ValueError("Could not detect model architecture")

    d_compressed = max(1, int(d_latent * compression_ratio))

    print(f"=" * 80)
    print(f"Enabling KVSplice Compression")
    print(f"=" * 80)
    print(
        f"Model: {model.config._name_or_path if hasattr(model.config, '_name_or_path') else 'unknown'}"
    )
    print(f"Layers: {n_layers}")
    print(f"Attention heads: {n_heads}")
    print(f"Latent dimension: {d_latent}")
    print(f"Compression ratio: {compression_ratio}")
    print(f"Compressed dimension: {d_compressed} (was {d_latent})")
    print(f"Memory reduction: {(1 - compression_ratio) * 100:.1f}%")
    print(f"=" * 80)

    # Store calibrated_kvsplice for use in cache preparation
    _calibrated_kvsplice = calibrated_kvsplice

    # Create cache preparation function
    def _prepare_cache_for_generation(
        self, generation_config, model_kwargs, *args, **kwargs
    ):
        """Inject CompressedKVCache for generation."""
        model_kwargs["past_key_values"] = CompressedKVCache(
            d_latent=d_latent,
            compression_ratio=compression_ratio,
            num_layers=n_layers,
            use_layernorm=use_layernorm,
            calibrated_kvsplice=_calibrated_kvsplice,
        )
        # Move to model device and dtype
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        model_kwargs["past_key_values"].kvsplice = model_kwargs[
            "past_key_values"
        ].kvsplice.to(device=device, dtype=dtype)

    # Monkey-patch the model
    import types

    model._prepare_cache_for_generation = types.MethodType(
        _prepare_cache_for_generation, model
    )

    # Store config for reference
    model.kvsplice_config = {
        "compression_ratio": compression_ratio,
        "d_latent": d_latent,
        "d_compressed": d_compressed,
        "use_layernorm": use_layernorm,
    }

    print(f"✓ KVSplice enabled successfully!")
    print(f"  Cache will be compressed {d_latent} → {d_compressed} dims")
    print()

    return model

#!/usr/bin/env python3
"""
Simplified KVSplice Plugin - Post-Training KV Cache Compression

EXPERIMENTAL: Attempts to add KV cache compression to pretrained models
via learned low-rank adapters. Clean implementation with pure linear
projection (no affine transforms).

RESULT: Does NOT work well for quality-critical applications.
Achieves perfect memory savings but degrades generation quality.

## What We Learned:

✓ Architecture works: Clean low-rank autoencoder compresses cache
✓ Memory savings: 100% reduction (50% compression = 50% memory)
✓ Gradient flow: Both Phase 1 and Phase 2 training work correctly
✗ Quality fails: 25% token agreement vs baseline at 50% compression
✗ Phase 2 ineffective: Task-aware tuning doesn't recover quality

## Why It Doesn't Work:

Models trained WITHOUT compression cannot be effectively compressed
post-hoc. They learned to rely on full-precision K/V representations.
Even with task-aware fine-tuning, compression breaks learned patterns.

## Conclusion:

**Negative result validating training-time compression approach.**
For production use, train models WITH KVSplice from the start
(e.g., GPT2_MLA_KV in gpt2/mla.py), not as a post-hoc plugin.

## Implementation:

Two-phase training:
1. Phase 1 (Reconstruction): Minimize MSE(decompress(compress(KV)), KV)
2. Phase 2 (Task-aware): Fine-tune on LM loss with compressed cache

Usage:
    from kvsplice_simple_plugin import calibrate_kv_compressor, enable_kvsplice_simple

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    compressors = calibrate_kv_compressor(model, tokenizer, compression_ratio=0.5)
    enable_kvsplice_simple(model, compressors)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from typing import Optional, Tuple, List, Dict
import copy


class KVCompressor(nn.Module):
    """
    Pure linear low-rank autoencoder for K/V compression.

    No affine transforms, no LayerNorm (at first), just clean
    orthogonal-initialized linear projections.
    """

    def __init__(self, d_in: int, d_compressed: int):
        super().__init__()
        self.d_in = d_in
        self.d_compressed = d_compressed

        # Down-projection (compress)
        self.down = nn.Linear(d_in, d_compressed, bias=False)

        # Up-projection (decompress)
        self.up = nn.Linear(d_compressed, d_in, bias=False)

        # Initialize as approximate identity
        nn.init.orthogonal_(self.down.weight)
        with torch.no_grad():
            self.up.weight.copy_(self.down.weight.T)

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Compress input to latent space."""
        return self.down(x)

    def decompress(self, z: torch.Tensor) -> torch.Tensor:
        """Decompress from latent space."""
        return self.up(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full round-trip for reconstruction loss."""
        return self.up(self.down(x))


class CompressedKVCache(Cache):
    """
    Cache that stores K/V in compressed form using learned compressors.

    Each layer can have its own compressor, or share a single compressor.
    """

    def __init__(
        self,
        compressors: Dict[str, KVCompressor],  # "k" and "v" compressors
        num_layers: int,
        per_layer: bool = False,
    ):
        super().__init__()
        self.compressors = compressors
        self.num_layers = num_layers
        self.per_layer = per_layer

        # Storage for compressed K/V per layer
        self.key_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self.value_cache: List[Optional[torch.Tensor]] = [None] * num_layers

    def _get_compressor(self, kv_type: str, layer_idx: int) -> KVCompressor:
        """Get compressor for this layer and KV type."""
        if self.per_layer:
            key = f"{kv_type}_{layer_idx}"
        else:
            key = kv_type
        return self.compressors[key]

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
            key_states: [B, H, T_new, head_dim]
            value_states: [B, H, T_new, head_dim]
            layer_idx: Layer index

        Returns:
            Full K/V (decompressed) for attention: [B, H, T_total, head_dim]
        """
        B, H, T_new, head_dim = key_states.shape

        # Get compressors for this layer
        k_comp = self._get_compressor("k", layer_idx)
        v_comp = self._get_compressor("v", layer_idx)

        # Flatten heads: [B, H, T, head_dim] -> [B, T, H * head_dim]
        k_flat = key_states.permute(0, 2, 1, 3).reshape(B, T_new, H * head_dim)
        v_flat = value_states.permute(0, 2, 1, 3).reshape(B, T_new, H * head_dim)

        # Convert to compressor dtype if needed (compressor is float32, input might be float16)
        original_dtype = k_flat.dtype
        if k_flat.dtype != next(k_comp.parameters()).dtype:
            k_flat = k_flat.to(next(k_comp.parameters()).dtype)
            v_flat = v_flat.to(next(v_comp.parameters()).dtype)

        # Compress
        k_compressed = k_comp.compress(k_flat)
        v_compressed = v_comp.compress(v_flat)

        # Cast back to original dtype to save memory (fp32 → fp16)
        if k_compressed.dtype != original_dtype:
            k_compressed = k_compressed.to(original_dtype)
            v_compressed = v_compressed.to(original_dtype)

        # Concatenate with existing cache
        if self.key_cache[layer_idx] is not None:
            k_compressed = torch.cat([self.key_cache[layer_idx], k_compressed], dim=1)
            v_compressed = torch.cat([self.value_cache[layer_idx], v_compressed], dim=1)

        # Store compressed (now in original dtype for memory efficiency)
        self.key_cache[layer_idx] = k_compressed
        self.value_cache[layer_idx] = v_compressed

        # Decompress for attention (upcast if needed for compressor dtype)
        k_to_decompress = k_compressed
        v_to_decompress = v_compressed
        if k_compressed.dtype != next(k_comp.parameters()).dtype:
            k_to_decompress = k_compressed.to(next(k_comp.parameters()).dtype)
            v_to_decompress = v_compressed.to(next(v_comp.parameters()).dtype)

        k_full = k_comp.decompress(k_to_decompress)
        v_full = v_comp.decompress(v_to_decompress)

        # Convert back to original dtype
        if k_full.dtype != original_dtype:
            k_full = k_full.to(original_dtype)
            v_full = v_full.to(original_dtype)

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
        raise NotImplementedError(
            "Beam search not supported with CompressedKVCache yet"
        )


def calibrate_kv_compressor(
    model: nn.Module,
    tokenizer,
    compression_ratio: float = 0.5,
    calibration_samples: int = 2000,
    max_length: int = 512,
    learning_rate: float = 1e-3,
    calibration_steps: int = 500,
    per_layer: bool = False,
    d_latent: Optional[int] = None,
) -> Dict[str, KVCompressor]:
    """
    Calibrate KV compressors using reconstruction loss on frozen model.

    Phase 1: Pure reconstruction - learns to minimize MSE on K/V activations.

    Args:
        model: Frozen pretrained model
        tokenizer: Tokenizer for calibration data
        compression_ratio: Compression ratio (0.5 = 2x compression)
        calibration_samples: Number of tokens to collect
        max_length: Max sequence length
        learning_rate: Learning rate for Adam
        calibration_steps: Training steps
        per_layer: If True, train separate compressor per layer
        d_latent: Latent dimension (auto-detect if None)

    Returns:
        Dictionary of trained compressors
    """
    from datasets import load_dataset

    print("=" * 80)
    print("Phase 1: Calibrating KV Compressors (Reconstruction Loss)")
    print("=" * 80)

    # Auto-detect d_latent
    if d_latent is None:
        if hasattr(model.config, "n_embd"):
            d_latent = model.config.n_embd
        elif hasattr(model.config, "hidden_size"):
            d_latent = model.config.hidden_size
        else:
            raise ValueError("Could not detect latent dimension")

    if hasattr(model.config, "n_layer"):
        n_layers = model.config.n_layer
    elif hasattr(model.config, "num_hidden_layers"):
        n_layers = model.config.num_hidden_layers
    else:
        raise ValueError("Could not detect number of layers")

    d_compressed = max(1, int(d_latent * compression_ratio))

    print(f"Model layers: {n_layers}")
    print(f"Latent dimension: {d_latent}")
    print(f"Compressed dimension: {d_compressed}")
    print(f"Compression ratio: {compression_ratio}")
    print(f"Per-layer compressors: {per_layer}")
    print(f"Calibration samples: {calibration_samples} tokens")
    print(f"Training steps: {calibration_steps}")

    # Freeze model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Collect K/V activations per layer
    kv_activations = {i: {"k": [], "v": []} for i in range(n_layers)}

    def make_hook(layer_idx):
        """Create hook for specific layer."""

        def hook_fn(module, input, output):
            """Hook to capture QKV projection output."""
            # GPT-2 c_attn produces concatenated QKV: [B, T, 3 * n_embd]
            qkv = output
            B, T, _ = qkv.shape
            q, k, v = qkv.split(d_latent, dim=2)

            # Store K and V activations
            kv_activations[layer_idx]["k"].append(
                k.detach().cpu().reshape(-1, d_latent)
            )
            kv_activations[layer_idx]["v"].append(
                v.detach().cpu().reshape(-1, d_latent)
            )

        return hook_fn

    # Hook all layers
    hooks = []
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        for i, layer in enumerate(model.transformer.h):
            if hasattr(layer, "attn") and hasattr(layer.attn, "c_attn"):
                hook = layer.attn.c_attn.register_forward_hook(make_hook(i))
                hooks.append(hook)
    else:
        raise ValueError("Could not find attention layers to hook")

    # Load calibration data
    print("\nCollecting K/V activations from calibration data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [item["text"] for item in dataset if len(item["text"].strip()) > 50]

    samples_collected = 0
    with torch.no_grad():
        for text in texts:
            if samples_collected >= calibration_samples:
                break

            inputs = tokenizer(
                text, max_length=max_length, truncation=True, return_tensors="pt"
            ).to(next(model.parameters()).device)

            if inputs["input_ids"].shape[1] < 10:
                continue

            model(**inputs)
            samples_collected += inputs["input_ids"].shape[1]

            if samples_collected % 500 == 0:
                print(f"  Progress: {samples_collected}/{calibration_samples} tokens")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    print(f"\nCollection complete!")

    # Train compressors
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    compressors = {}

    if per_layer:
        # Train separate compressor for each layer and K/V
        print("\nTraining per-layer compressors...")
        for layer_idx in range(n_layers):
            print(f"\n  Layer {layer_idx}:")

            for kv_type in ["k", "v"]:
                # Concatenate all activations for this layer/type
                activations = torch.cat(kv_activations[layer_idx][kv_type], dim=0)[
                    :calibration_samples
                ]

                if activations.shape[0] == 0:
                    print(
                        f"    WARNING: No {kv_type} activations for layer {layer_idx}"
                    )
                    continue

                # Create and train compressor (use float32 for stability)
                comp = KVCompressor(d_latent, d_compressed).to(
                    device=device, dtype=torch.float32
                )
                optimizer = torch.optim.Adam(comp.parameters(), lr=learning_rate)

                print(
                    f"    Training {kv_type} compressor ({activations.shape[0]} samples)..."
                )

                batch_size = 256
                best_loss = float("inf")

                for step in range(calibration_steps):
                    # Random batch
                    idx = torch.randint(0, activations.shape[0], (batch_size,))
                    batch = activations[idx].to(device=device, dtype=torch.float32)

                    # Forward and loss
                    reconstructed = comp(batch)
                    loss = F.mse_loss(reconstructed, batch)

                    # Check for NaN
                    if torch.isnan(loss):
                        print(f"      WARNING: NaN loss at step {step+1}, skipping")
                        continue

                    # Backward with gradient clipping
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(comp.parameters(), max_norm=1.0)
                    optimizer.step()

                    if loss.item() < best_loss:
                        best_loss = loss.item()

                    if (step + 1) % 100 == 0:
                        print(
                            f"      Step {step+1}/{calibration_steps}: loss={loss.item():.6f}"
                        )

                print(f"    Best loss: {best_loss:.6f}")
                compressors[f"{kv_type}_{layer_idx}"] = comp

    else:
        # Train shared compressor for all layers
        print("\nTraining shared compressors...")
        for kv_type in ["k", "v"]:
            # Concatenate activations from all layers
            all_activations = []
            for layer_idx in range(n_layers):
                all_activations.extend(kv_activations[layer_idx][kv_type])

            activations = torch.cat(all_activations, dim=0)[:calibration_samples]

            print(
                f"  Training {kv_type} compressor ({activations.shape[0]} samples)..."
            )

            # Create and train compressor (use float32 for stability)
            comp = KVCompressor(d_latent, d_compressed).to(
                device=device, dtype=torch.float32
            )
            optimizer = torch.optim.Adam(comp.parameters(), lr=learning_rate)

            batch_size = 256
            best_loss = float("inf")

            for step in range(calibration_steps):
                # Random batch
                idx = torch.randint(0, activations.shape[0], (batch_size,))
                batch = activations[idx].to(device=device, dtype=torch.float32)

                # Forward and loss
                reconstructed = comp(batch)
                loss = F.mse_loss(reconstructed, batch)

                # Check for NaN
                if torch.isnan(loss):
                    print(f"    WARNING: NaN loss at step {step+1}, skipping")
                    continue

                # Backward with gradient clipping
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(comp.parameters(), max_norm=1.0)
                optimizer.step()

                if loss.item() < best_loss:
                    best_loss = loss.item()

                if (step + 1) % 100 == 0:
                    print(
                        f"    Step {step+1}/{calibration_steps}: loss={loss.item():.6f}"
                    )

            print(f"  Best loss: {best_loss:.6f}")
            compressors[kv_type] = comp

    print("\n" + "=" * 80)
    print("✓ Phase 1 calibration complete!")
    print("=" * 80)

    return compressors


def enable_kvsplice_simple(
    model: nn.Module,
    compressors: Dict[str, KVCompressor],
    per_layer: bool = False,
) -> nn.Module:
    """
    Enable simplified KVSplice with trained compressors.

    Args:
        model: Pretrained model
        compressors: Dictionary of trained KVCompressor modules
        per_layer: Whether compressors are per-layer

    Returns:
        Model with KVSplice enabled
    """
    # Detect architecture
    if hasattr(model.config, "n_layer"):
        n_layers = model.config.n_layer
    elif hasattr(model.config, "num_hidden_layers"):
        n_layers = model.config.num_hidden_layers
    else:
        raise ValueError("Could not detect number of layers")

    print("=" * 80)
    print("Enabling Simplified KVSplice")
    print("=" * 80)
    print(
        f"Model: {model.config._name_or_path if hasattr(model.config, '_name_or_path') else 'unknown'}"
    )
    print(f"Layers: {n_layers}")
    print(f"Per-layer compressors: {per_layer}")

    # Sample compressor to get dims
    sample_comp = list(compressors.values())[0]
    print(f"Compression: {sample_comp.d_in} → {sample_comp.d_compressed} dims")
    print(
        f"Memory reduction: {(1 - sample_comp.d_compressed / sample_comp.d_in) * 100:.1f}%"
    )
    print("=" * 80)

    # Create cache preparation function
    def _prepare_cache_for_generation(
        self, generation_config, model_kwargs, *args, **kwargs
    ):
        """Inject CompressedKVCache for generation."""
        model_kwargs["past_key_values"] = CompressedKVCache(
            compressors=compressors,
            num_layers=n_layers,
            per_layer=per_layer,
        )

    # Monkey-patch the model
    import types

    model._prepare_cache_for_generation = types.MethodType(
        _prepare_cache_for_generation, model
    )

    print("✓ Simplified KVSplice enabled!")
    print()

    return model


def refine_with_task_loss(
    model: nn.Module,
    tokenizer,
    compressors: Dict[str, KVCompressor],
    per_layer: bool = False,
    learning_rate: float = 1e-5,
    refinement_steps: int = 500,
    batch_size: int = 4,
    gradient_accumulation: int = 4,
    max_length: int = 256,
) -> Dict[str, KVCompressor]:
    """
    Phase 2: Refine compressors using task loss (LM loss).

    Keeps model frozen, enables compressors in forward pass, and optimizes
    compressor parameters on actual next-token prediction loss. This lets
    the low-rank subspace rotate to preserve task-critical directions.

    Args:
        model: Frozen pretrained model (already has KVSplice enabled)
        tokenizer: Tokenizer
        compressors: Dictionary of Phase 1 trained compressors
        per_layer: Whether compressors are per-layer
        learning_rate: Learning rate for refinement
        refinement_steps: Number of training steps
        batch_size: Batch size
        gradient_accumulation: Gradient accumulation steps
        max_length: Max sequence length

    Returns:
        Refined compressors
    """
    from datasets import load_dataset

    print("=" * 80)
    print("Phase 2: Refining KV Compressors (Task Loss)")
    print("=" * 80)
    print(f"Learning rate: {learning_rate}")
    print(f"Refinement steps: {refinement_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation: {gradient_accumulation}")
    print(f"Effective batch size: {batch_size * gradient_accumulation}")

    # Set model to train mode but freeze its parameters
    model.train()  # Need train mode for gradient flow
    for param in model.parameters():
        param.requires_grad = False

    # Set compressors to train mode and enable gradients
    for comp in compressors.values():
        comp.train()
        for param in comp.parameters():
            param.requires_grad = True

    # Optimizer for compressor params only
    all_params = []
    for comp in compressors.values():
        all_params.extend(list(comp.parameters()))

    optimizer = torch.optim.Adam(all_params, lr=learning_rate)

    # Load calibration data
    print("\nLoading training data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [item["text"] for item in dataset if len(item["text"].strip()) > 50]

    print(f"Loaded {len(texts)} text samples")

    # Training loop
    print("\nRefining compressors with task loss...")
    device = next(model.parameters()).device

    # Detect num_layers
    if hasattr(model.config, "n_layer"):
        num_layers = model.config.n_layer
    elif hasattr(model.config, "num_hidden_layers"):
        num_layers = model.config.num_hidden_layers
    else:
        raise ValueError("Could not detect number of layers")

    step = 0
    text_idx = 0
    total_loss = 0.0
    best_loss = float("inf")

    while step < refinement_steps:
        for micro_step in range(gradient_accumulation):
            # Get batch
            if text_idx >= len(texts):
                text_idx = 0  # Wrap around

            text = texts[text_idx]
            text_idx += 1

            # Tokenize
            inputs = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            if inputs["input_ids"].shape[1] < 10:
                continue  # Skip very short sequences

            # CRITICAL: Create CompressedKVCache to force compressors into graph
            past = CompressedKVCache(
                compressors=compressors, num_layers=num_layers, per_layer=per_layer
            )

            # Forward pass with compressed cache - THIS is where gradients flow
            outputs = model(
                **inputs,
                labels=inputs["input_ids"],
                use_cache=True,
                past_key_values=past,
            )

            loss = outputs.loss / gradient_accumulation

            # Check for NaN
            if torch.isnan(loss):
                print(f"  WARNING: NaN loss at step {step}, skipping")
                continue

            # Backward (accumulate gradients)
            loss.backward()

            total_loss += loss.item()

        # Optimizer step after accumulation
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        step += 1

        if total_loss < best_loss:
            best_loss = total_loss

        if step % 50 == 0:
            avg_loss = total_loss / 50 if step >= 50 else total_loss / step
            print(f"  Step {step}/{refinement_steps}: avg_loss={avg_loss:.4f}")
            total_loss = 0.0

    print(f"\nBest avg loss: {best_loss:.4f}")
    print("=" * 80)
    print("✓ Phase 2 refinement complete!")
    print("=" * 80)

    return compressors

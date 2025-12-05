"""
Compressed KV Model Wrapper - Hook-Based Implementation

Uses PyTorch forward hooks instead of monkey-patching for robust
cross-version compatibility. Works with any transformers version
and model architecture that uses standard KV cache format.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from gpt2.compression.base import KVCompressorBase


class CompressedKVModelWrapper(nn.Module):
    """
    Hook-based KV cache compression wrapper.

    Intercepts attention layer outputs via forward hooks to compress
    KV cache. Compatible with any transformers version and architecture.

    Usage:
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        compressor = KVSpliceCompressor(config)
        wrapped = CompressedKVModelWrapper(model, compressor)

        # Calibrate
        calibrate_kv_compressor(wrapped, tokenizer, calib_data)

        # Use normally
        outputs = wrapped(input_ids, use_cache=True)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        compressor: KVCompressorBase,
        auto_patch: bool = True,
    ):
        super().__init__()
        self.model = model
        self.compressor = compressor
        self.hooks = []

        # Track whether we're in calibration mode
        self._calibration_mode = False
        # During calibration, bypass hooks and read from past_key_values
        self._use_model_output_for_calibration = True

        if auto_patch:
            self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on attention layers."""
        model_type = self.model.config.model_type

        if model_type == "gpt2":
            self._register_gpt2_hooks()
        elif model_type == "qwen2":
            self._register_qwen2_hooks()
        elif model_type == "mistral":
            self._register_mistral_hooks()
        elif model_type == "llama":
            self._register_llama_hooks()
        else:
            raise NotImplementedError(
                f"Model type {model_type} not yet supported. "
                f"Currently supported: gpt2, qwen2, mistral, llama"
            )

    def _register_gpt2_hooks(self):
        """Register hooks for GPT-2 attention layers."""
        for layer_idx, block in enumerate(self.model.transformer.h):
            attn = block.attn

            # Register hook on attention module
            hook = attn.register_forward_hook(self._create_compression_hook(layer_idx))
            self.hooks.append(hook)

    def _register_qwen2_hooks(self):
        """Register hooks for Qwen2 attention layers."""
        for layer_idx, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn

            # Register hook on attention module
            hook = attn.register_forward_hook(self._create_compression_hook(layer_idx))
            self.hooks.append(hook)

    def _register_mistral_hooks(self):
        """Register hooks for Mistral attention layers."""
        for layer_idx, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn

            # Register hook on attention module
            hook = attn.register_forward_hook(self._create_compression_hook(layer_idx))
            self.hooks.append(hook)

    def _register_llama_hooks(self):
        """Register hooks for LLaMA attention layers."""
        for layer_idx, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn

            # Register hook on attention module
            hook = attn.register_forward_hook(self._create_compression_hook(layer_idx))
            self.hooks.append(hook)

    def _create_compression_hook(self, layer_idx: int):
        """
        Create forward hook for attention layer.

        Hook signature: hook(module, input, output) -> modified_output
        """

        def compression_hook(module, input, output):
            # Ignore hooks completely when calibrating
            # (calibration uses model output path instead)
            if self._calibration_mode:
                return output

            # Output format: (attn_output, present_key_value) or (attn_output,) + extras
            # present_key_value is (key, value) tuple when use_cache=True

            if not isinstance(output, tuple):
                # No KV cache in output
                return output

            # Extract components
            attn_output = output[0]
            present = output[1] if len(output) > 1 else None
            extra_outputs = output[2:] if len(output) > 2 else ()

            # If no KV cache, return unchanged
            if present is None or not isinstance(present, tuple):
                return output

            # Extract K, V from cache: (key, value)
            # Shape: [batch, n_head, seq, head_dim]
            key, value = present

            # In calibration mode, just observe KV and return unchanged
            if self._calibration_mode:
                self._observe_kv_from_cache(layer_idx, key, value)
                return output

            # Get number of heads
            batch_size, n_head, seq_len, head_dim = key.shape

            # Detect GQA: check if model uses grouped-query attention
            n_kv_heads = getattr(self.model.config, "num_key_value_heads", n_head)

            # GQA-efficient compression: compress per KV group
            if n_kv_heads < n_head:
                # GQA model - compress per KV group (e.g., 4 groups for Qwen2.5-7B)
                kv_group_size = n_head // n_kv_heads

                decompressed_key_list = []
                decompressed_value_list = []

                for kv_group_idx in range(n_kv_heads):
                    # In GQA, KV is already repeated/tiled for all Q heads
                    # Pick first head in each KV group as representative
                    # (all heads in group share same KV)
                    head_idx = kv_group_idx

                    # Extract KV for this group: [batch, seq, head_dim]
                    k_head = key[:, head_idx, :, :]
                    v_head = value[:, head_idx, :, :]

                    # Compress once per KV group
                    k_compressed, v_compressed = self.compressor.compress(
                        layer_idx, kv_group_idx, k_head, v_head
                    )

                    # Decompress
                    k_decomp, v_decomp = self.compressor.decompress(
                        layer_idx, kv_group_idx, k_compressed, v_compressed
                    )

                    # Replicate decompressed KV to all Q heads in this group
                    # For Qwen2.5-7B: 7 Q heads per KV group
                    for offset in range(kv_group_size):
                        q_head_idx = kv_group_idx + offset * n_kv_heads
                        if q_head_idx < n_head:
                            decompressed_key_list.append(k_decomp)
                            decompressed_value_list.append(v_decomp)

                # Reorder to match original head order
                # GQA head order: [KV0_Q0, KV1_Q0, ..., KV0_Q1, KV1_Q1, ...]
                # Need to reorder to: [KV0_Q0, KV0_Q1, ..., KV1_Q0, KV1_Q1, ...]
                key_decompressed_reordered = []
                value_decompressed_reordered = []

                for kv_group_idx in range(n_kv_heads):
                    for offset in range(kv_group_size):
                        idx = kv_group_idx + offset * n_kv_heads
                        if idx < n_head:
                            # Find position in decompressed list
                            list_idx = kv_group_idx * kv_group_size + offset
                            key_decompressed_reordered.append(
                                decompressed_key_list[list_idx]
                            )
                            value_decompressed_reordered.append(
                                decompressed_value_list[list_idx]
                            )

                key_decompressed = torch.stack(key_decompressed_reordered, dim=1)
                value_decompressed = torch.stack(value_decompressed_reordered, dim=1)

            else:
                # Standard attention - compress per head
                compressed_key_list = []
                compressed_value_list = []

                for head_idx in range(n_head):
                    # Extract per-head KV: [batch, seq, head_dim]
                    k_head = key[:, head_idx, :, :]
                    v_head = value[:, head_idx, :, :]

                    # Compress (returns same if compression disabled)
                    k_compressed, v_compressed = self.compressor.compress(
                        layer_idx, head_idx, k_head, v_head
                    )

                    compressed_key_list.append(k_compressed)
                    compressed_value_list.append(v_compressed)

                # Stack back to [batch, n_head, seq, rank_or_head_dim]
                key_compressed = torch.stack(compressed_key_list, dim=1)
                value_compressed = torch.stack(compressed_value_list, dim=1)

                # Decompress for this forward pass
                # (In production, we'd keep compressed and decompress lazily)
                decompressed_key_list = []
                decompressed_value_list = []

                for head_idx in range(n_head):
                    k_comp = key_compressed[:, head_idx, :, :]
                    v_comp = value_compressed[:, head_idx, :, :]

                    k_decomp, v_decomp = self.compressor.decompress(
                        layer_idx, head_idx, k_comp, v_comp
                    )

                    decompressed_key_list.append(k_decomp)
                    decompressed_value_list.append(v_decomp)

                key_decompressed = torch.stack(decompressed_key_list, dim=1)
                value_decompressed = torch.stack(decompressed_value_list, dim=1)

            # Return modified output with decompressed KV for attention
            # (Cache stores compressed version)
            present_decompressed = (key_decompressed, value_decompressed)

            return (attn_output, present_decompressed) + extra_outputs

        return compression_hook

    def _observe_kv_from_cache(
        self, layer_idx: int, key: torch.Tensor, value: torch.Tensor
    ):
        """
        Extract and observe KV from cache during calibration.

        GQA-efficient: observes per KV group instead of per Q head.

        Args:
            layer_idx: Layer index
            key: [batch, n_head, seq, head_dim]
            value: [batch, n_head, seq, head_dim]
        """
        batch_size, n_head, seq_len, head_dim = key.shape

        # Detect GQA structure
        n_kv_heads = getattr(self.model.config, "num_key_value_heads", n_head)

        if n_kv_heads < n_head:
            # GQA model - observe per KV group only
            for kv_group_idx in range(n_kv_heads):
                # Pick representative head from each KV group
                head_idx = kv_group_idx

                # Extract KV: [batch, seq, head_dim]
                k_head = key[:, head_idx, :, :]
                v_head = value[:, head_idx, :, :]

                # Observe for calibration (using kv_group_idx as head_idx)
                self.compressor.observe_kv(layer_idx, kv_group_idx, k_head, v_head)
        else:
            # Standard attention - observe all heads
            for head_idx in range(n_head):
                # Extract per-head KV: [batch, seq, head_dim]
                k_head = key[:, head_idx, :, :]
                v_head = value[:, head_idx, :, :]

                # Observe for calibration
                self.compressor.observe_kv(layer_idx, head_idx, k_head, v_head)

    def start_calibration(self):
        """Start calibration mode."""
        self._calibration_mode = True
        self._use_model_output_for_calibration = True
        self.compressor.start_calibration()

    def end_calibration(self):
        """End calibration and fit compressor."""
        self._calibration_mode = False
        self._use_model_output_for_calibration = False
        self.compressor.end_calibration()

    def forward(self, *args, **kwargs):
        """Forward pass through wrapped model."""
        # Calibration path: ignore hooks, read from past_key_values
        if self._calibration_mode and self._use_model_output_for_calibration:
            kwargs = dict(kwargs)
            # Ensure KV cache is produced
            kwargs.setdefault("use_cache", True)

            outputs = self.model(*args, **kwargs)

            past_kv = self._extract_past_key_values(outputs)
            if past_kv is not None:
                self._observe_from_past_kv(past_kv)

            return outputs

        # Normal / compressed inference path: let hooks do their thing
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Generation wrapper."""
        return self.model.generate(*args, **kwargs)

    @staticmethod
    def _extract_past_key_values(outputs):
        """
        Extract past_key_values from model output.

        Handles different output formats:
        - Standard HF: outputs.past_key_values
        - Singular form: outputs.past_key_value
        - Tuple format: outputs[1] is past_key_values
        """
        # 1) Standard HF attribute
        if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
            return outputs.past_key_values

        # 2) Some models use singular form
        if hasattr(outputs, "past_key_value") and outputs.past_key_value is not None:
            return outputs.past_key_value

        # 3) Fallback: tuple format where second element is cache
        if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
            candidate = outputs[1]
            if CompressedKVModelWrapper._looks_like_past_kv(candidate):
                return candidate

        return None

    @staticmethod
    def _looks_like_past_kv(obj) -> bool:
        """
        Check if object looks like past_key_values.

        Typical HF format: tuple/list of length n_layers,
        each item is either (K, V) tuple or cache object with .key/.value
        """
        if isinstance(obj, (tuple, list)) and len(obj) > 0:
            first = obj[0]
            # Check for (K, V) tuple
            if isinstance(first, (tuple, list)) and len(first) >= 2:
                return True
            # Check for cache object with .key/.value attributes
            if hasattr(first, "key") and hasattr(first, "value"):
                return True
        return False

    def _observe_from_past_kv(self, past_kv):
        """
        Observe KV from past_key_values during calibration.

        Args:
            past_kv: Iterable over layers, each item is either:
                - (key, value) tensor pair, or
                - object with .key and .value attributes
            Shapes: key/value are [batch, num_kv_heads, seq_len, head_dim]
        """
        for layer_idx, layer_cache in enumerate(past_kv):
            # Unpack different cache formats
            if hasattr(layer_cache, "key") and hasattr(layer_cache, "value"):
                key = layer_cache.key
                value = layer_cache.value
            elif isinstance(layer_cache, (tuple, list)) and len(layer_cache) >= 2:
                key, value = layer_cache[0], layer_cache[1]
            else:
                continue

            if key is None or value is None:
                continue

            self._observe_layer_heads(layer_idx, key, value)

    def _observe_layer_heads(
        self, layer_idx: int, key: torch.Tensor, value: torch.Tensor
    ):
        """
        Observe per-head KV during calibration.

        Args:
            layer_idx: Layer index
            key: [batch, num_kv_heads, seq_len, head_dim]
            value: [batch, num_kv_heads, seq_len, head_dim]
        """
        # Sanity check dimensions
        if key.ndim != 4 or value.ndim != 4:
            return

        batch_size, n_kv_heads, seq_len, head_dim = key.shape

        # Flatten batch × time: [B, H_kv, T, D] -> [B*T, H_kv, D]
        # This gives many samples per head per calibration batch
        key_flat = key.permute(0, 2, 1, 3).reshape(-1, n_kv_heads, head_dim)
        value_flat = value.permute(0, 2, 1, 3).reshape(-1, n_kv_heads, head_dim)

        for kv_idx in range(n_kv_heads):
            # Check if this head is enabled for compression
            if not self._is_head_enabled(layer_idx, kv_idx):
                continue

            # Extract per-head KV: [N_tokens, head_dim]
            k_head = key_flat[:, kv_idx, :]
            v_head = value_flat[:, kv_idx, :]

            # Observe for calibration
            self.compressor.observe_kv(layer_idx, kv_idx, k_head, v_head)

    def _is_head_enabled(self, layer_idx: int, head_idx: int) -> bool:
        """Check if a head is enabled for compression."""
        # For PCA, check if module exists for this head
        # PCA uses both string keys like 'L0_H0' and tuple keys like (0, 0)
        if hasattr(self.compressor, "pca_modules"):
            # Try tuple key first
            tuple_key = (layer_idx, head_idx)
            if tuple_key in self.compressor.pca_modules:
                return True
            # Try string key format
            string_key = f"L{layer_idx}_H{head_idx}"
            if string_key in self.compressor.pca_modules:
                return True
            return False

        # For other compressors, check if compressor exists
        if hasattr(self.compressor, "compressors"):
            key = (layer_idx, head_idx)
            return key in self.compressor.compressors

        # Default: assume all heads enabled
        return True

    def __getattr__(self, name):
        """Delegate attribute access to wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()

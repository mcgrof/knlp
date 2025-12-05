"""
Compressed KV Model Wrapper

Wraps HuggingFace transformers models to use compressed KV cache.
Minimal surgery approach: monkey-patches attention layers at runtime.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from gpt2.compression.base import KVCompressorBase


class CompressedKVModelWrapper(nn.Module):
    """
    Wrapper for HF models with compressed KV cache.

    Intercepts attention forward pass to compress/decompress KV cache.
    Drop-in replacement preserving HF API.

    Usage:
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        compressor = KVSpliceCompressor(config)
        wrapped = CompressedKVModelWrapper(model, compressor)

        # Calibrate compressor
        calibrate_kv_compressor(wrapped, tokenizer, calib_data)

        # Use normally
        outputs = wrapped(input_ids, use_cache=True)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        compressor: KVCompressorBase,
        patch_attention: bool = True,
    ):
        super().__init__()
        self.model = model
        self.compressor = compressor

        # Store original attention forward methods
        self._original_attention_forwards = {}

        if patch_attention:
            self._patch_attention_layers()

    def _patch_attention_layers(self):
        """
        Monkey-patch attention layers to use compressed KV cache.

        For GPT-2: model.transformer.h[i].attn
        """
        model_type = self.model.config.model_type

        if model_type == "gpt2":
            self._patch_gpt2_attention()
        else:
            raise NotImplementedError(
                f"Model type {model_type} not yet supported. "
                f"Currently supported: gpt2"
            )

    def _patch_gpt2_attention(self):
        """Patch GPT-2 attention layers."""
        for layer_idx, block in enumerate(self.model.transformer.h):
            attn = block.attn

            # Store original forward
            self._original_attention_forwards[(layer_idx, "attn")] = attn.forward

            # Create patched forward with compressor
            attn.forward = self._create_compressed_attn_forward(
                layer_idx, attn, attn.forward
            )

    def _create_compressed_attn_forward(
        self, layer_idx: int, attn_module: nn.Module, original_forward
    ):
        """
        Create compressed attention forward function.

        Wraps original attention to compress/decompress per-head KV cache.
        """

        def compressed_forward(
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
            past_key_value=None,  # Newer transformers use this name
            **kwargs,  # Handle additional parameters (cache_position, etc.)
        ):
            # Get attention config
            n_head = attn_module.num_heads
            head_dim = attn_module.head_dim

            # Handle both parameter names (layer_past vs past_key_value)
            past = past_key_value if past_key_value is not None else layer_past

            # Run original attention up to KV computation
            batch_size = hidden_states.size(0)

            # Compute Q, K, V
            # GPT-2: uses split_heads after c_attn projection
            query, key, value = attn_module.c_attn(hidden_states).split(
                attn_module.split_size, dim=2
            )

            # Split heads: [batch, seq, n_head * head_dim] -> [batch, n_head, seq, head_dim]
            query = attn_module._split_heads(
                query, attn_module.num_heads, attn_module.head_dim
            )
            key = attn_module._split_heads(
                key, attn_module.num_heads, attn_module.head_dim
            )
            value = attn_module._split_heads(
                value, attn_module.num_heads, attn_module.head_dim
            )

            # Handle past KV if provided
            if past is not None:
                past_key, past_value = past
                key = torch.cat((past_key, key), dim=-2)
                value = torch.cat((past_value, value), dim=-2)

            # Compress per-head KV if enabled
            if use_cache:
                compressed_key_heads = []
                compressed_value_heads = []

                for head_idx in range(n_head):
                    # Extract per-head KV: [batch, seq, head_dim]
                    k_head = key[:, head_idx, :, :]
                    v_head = value[:, head_idx, :, :]

                    # Compress (may be no-op if compression disabled for this head)
                    k_compressed, v_compressed = self.compressor.compress(
                        layer_idx, head_idx, k_head, v_head
                    )

                    compressed_key_heads.append(k_compressed)
                    compressed_value_heads.append(v_compressed)

                # Stack back: [batch, n_head, seq, rank_or_head_dim]
                key_compressed = torch.stack(compressed_key_heads, dim=1)
                value_compressed = torch.stack(compressed_value_heads, dim=1)

                # For attention computation, decompress
                decompressed_key_heads = []
                decompressed_value_heads = []

                for head_idx in range(n_head):
                    k_comp = key_compressed[:, head_idx, :, :]
                    v_comp = value_compressed[:, head_idx, :, :]

                    k_decomp, v_decomp = self.compressor.decompress(
                        layer_idx, head_idx, k_comp, v_comp
                    )

                    decompressed_key_heads.append(k_decomp)
                    decompressed_value_heads.append(v_decomp)

                key_for_attn = torch.stack(decompressed_key_heads, dim=1)
                value_for_attn = torch.stack(decompressed_value_heads, dim=1)

                # Cache compressed version
                present = (key_compressed, value_compressed) if use_cache else None
            else:
                # No caching, use original KV
                key_for_attn = key
                value_for_attn = value
                present = None

            # Compute attention with decompressed KV
            attn_output, attn_weights = attn_module._attn(
                query, key_for_attn, value_for_attn, attention_mask, head_mask
            )

            # Merge heads and project output
            attn_output = attn_module._merge_heads(
                attn_output, attn_module.num_heads, attn_module.head_dim
            )
            attn_output = attn_module.c_proj(attn_output)
            attn_output = attn_module.resid_dropout(attn_output)

            outputs = (attn_output, present)
            if output_attentions:
                outputs += (attn_weights,)

            return outputs

        return compressed_forward

    def forward(self, *args, **kwargs):
        """Forward pass through wrapped model."""
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Generation wrapper."""
        return self.model.generate(*args, **kwargs)

    def __getattr__(self, name):
        """Delegate attribute access to wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

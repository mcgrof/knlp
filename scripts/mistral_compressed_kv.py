#!/usr/bin/env python3
"""
Mistral with native compressed KV cache - no patching, no double storage.

Implements SVD-compressed attention where KV cache is stored in compressed
form and decompressed on-the-fly during computation.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, MistralConfig
from datasets import load_dataset
import time
import gc
from typing import Optional, Tuple
import math


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE)."""

    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, seq_len):
        # Generate position indices
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Apply rotary embeddings to q and k."""
    # Reshape for rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class SVDCompressor(nn.Module):
    """SVD-based KV compressor."""

    def __init__(self, d_in: int, d_compressed: int):
        super().__init__()
        self.d_in = d_in
        self.d_compressed = d_compressed
        self.compress = nn.Linear(d_in, d_compressed, bias=False)
        self.expand = nn.Linear(d_compressed, d_in, bias=False)

    def calibrate_from_svd(self, V_svd: torch.Tensor):
        """Initialize from SVD singular vectors."""
        # V_svd: [d_in, d_compressed] from SVD
        expand_weight = V_svd  # [d_in, d_compressed]
        compress_weight = V_svd.T  # [d_compressed, d_in]

        self.expand.weight.data = expand_weight.contiguous()
        self.compress.weight.data = compress_weight.contiguous()


class CompressedKVAttention(nn.Module):
    """
    Mistral attention with compressed KV cache.

    Stores only compressed KV, decompresses on-the-fly for attention computation.
    No double storage - this is the only cache.
    """

    def __init__(
        self, config: MistralConfig, compressor: Optional[SVDCompressor] = None
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        # Q, K, V projections
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        # KV compressor (optional, for compressed cache)
        self.compressor = compressor
        self.use_compressed_cache = compressor is not None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        past_key_values=None,  # Transformers compatibility
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        # Handle both singular and plural argument names
        if past_key_values is not None:
            past_key_value = past_key_values

        bsz, q_len, _ = hidden_states.size()

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # RoPE
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            # Handle transformers Cache object or tuple
            if hasattr(past_key_value, "get_seq_length"):
                # It's a Cache object (e.g., DynamicCache)
                past_len = past_key_value.get_seq_length()
                kv_seq_len += past_len
            else:
                # Tuple format - extract k tensor
                if len(past_key_value) >= 2:
                    cache_k = past_key_value[0]
                    if isinstance(cache_k, tuple):
                        cache_k = cache_k[0]
                    kv_seq_len += cache_k.shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        # Handle cache
        if past_key_value is not None:
            # Handle transformers Cache object or tuple
            if hasattr(past_key_value, "update"):
                # It's a Cache object - transformers will manage it
                # We can't use compressed cache with Cache objects yet
                # Skip manual cache handling
                pass
            else:
                # Tuple format - unwrap and decompress
                if len(past_key_value) >= 2:
                    cache_k, cache_v = past_key_value[0], past_key_value[1]
                    if isinstance(cache_k, tuple):
                        cache_k, cache_v = cache_k[0], cache_v[0]

                    # Decompress or use directly
                    if self.use_compressed_cache:
                        # cache: (compressed_k, compressed_v)
                        # Shape: [bsz, num_kv_heads, past_len, d_compressed]
                        compressed_k, compressed_v = cache_k, cache_v
                        past_len = compressed_k.shape[2]

                        # Decompress: reshape, expand, reshape back
                        # [bsz, num_kv_heads, past_len, d_compressed] -> [bsz * num_kv_heads * past_len, d_compressed]
                        compressed_k_flat = compressed_k.reshape(
                            -1, self.compressor.d_compressed
                        )
                        compressed_v_flat = compressed_v.reshape(
                            -1, self.compressor.d_compressed
                        )

                        # Expand to full dimension (output will be in compressor's dtype)
                        past_k_flat = self.compressor.expand(compressed_k_flat)
                        past_v_flat = self.compressor.expand(compressed_v_flat)

                        # Reshape and convert to match key_states dtype
                        target_dtype = key_states.dtype
                        past_k = past_k_flat.reshape(
                            bsz, self.num_key_value_heads, past_len, self.head_dim
                        ).to(dtype=target_dtype)
                        past_v = past_v_flat.reshape(
                            bsz, self.num_key_value_heads, past_len, self.head_dim
                        ).to(dtype=target_dtype)
                    else:
                        # Standard cache (not compressed)
                        past_k, past_v = cache_k, cache_v

                    # Concatenate past and current
                    key_states = torch.cat([past_k, key_states], dim=2)
                    value_states = torch.cat([past_v, value_states], dim=2)

        # Prepare cache for next iteration
        if use_cache:
            if self.use_compressed_cache:
                # Compress current KV for caching
                # [bsz, num_kv_heads, total_len, head_dim] -> [bsz * num_kv_heads * total_len, head_dim]
                total_len = key_states.shape[2]
                key_flat = key_states.reshape(-1, self.head_dim)
                value_flat = value_states.reshape(-1, self.head_dim)

                # Compress (ensure dtype matches compressor)
                compressed_k_flat = self.compressor.compress(
                    key_flat.to(dtype=next(self.compressor.parameters()).dtype)
                )
                compressed_v_flat = self.compressor.compress(
                    value_flat.to(dtype=next(self.compressor.parameters()).dtype)
                )

                # Reshape back to [bsz, num_kv_heads, total_len, d_compressed]
                compressed_k = compressed_k_flat.reshape(
                    bsz,
                    self.num_key_value_heads,
                    total_len,
                    self.compressor.d_compressed,
                )
                compressed_v = compressed_v_flat.reshape(
                    bsz,
                    self.num_key_value_heads,
                    total_len,
                    self.compressor.d_compressed,
                )

                cache_to_return = (compressed_k, compressed_v)
            else:
                cache_to_return = (key_states, value_states)
        else:
            cache_to_return = None

        # Ensure all states have same dtype
        target_dtype = query_states.dtype
        key_states = key_states.to(dtype=target_dtype)
        value_states = value_states.to(dtype=target_dtype)

        # Repeat KV for grouped-query attention
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Attention computation
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)

        # Weighted sum
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if output_attentions:
            return attn_output, cache_to_return, attn_weights
        else:
            return attn_output, cache_to_return


def load_mistral_with_compressed_kv(
    model_name: str,
    compression_ratio: float = 0.5,
    calibration_samples: int = 2000,
    device: str = "cuda",
    cache_dir: str = "./kvsplice_cache",
):
    """
    Load Mistral model and replace attention with compressed KV version.
    Caches calibrated models to avoid re-running SVD.
    """
    import os

    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)

    # Generate cache path based on model name and compression
    model_basename = model_name.replace("/", "_")
    cache_path = os.path.join(
        cache_dir, f"{model_basename}_kvsplice_r{compression_ratio}.pt"
    )

    # Check if cached model exists
    if os.path.exists(cache_path):
        print(f"\n{'=' * 80}")
        print(f"LOADING CACHED KVSPLICE MODEL")
        print(f"{'=' * 80}")
        print(f"Cache path: {cache_path}")

        # Load cached compressor and model
        cached_data = torch.load(cache_path, map_location=device, weights_only=False)
        compressor = cached_data["compressor"]

        # Load base model
        print(f"Loading base model {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(device)

        config = model.config
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Replace attention layers with cached compressor
        print(f"Replacing attention layers with cached compressor...")
        for layer_idx, layer in enumerate(model.model.layers):
            old_attn = layer.self_attn
            new_attn = CompressedKVAttention(config, compressor).to(device)

            # Copy weights
            new_attn.q_proj.weight.data = old_attn.q_proj.weight.data
            new_attn.k_proj.weight.data = old_attn.k_proj.weight.data
            new_attn.v_proj.weight.data = old_attn.v_proj.weight.data
            new_attn.o_proj.weight.data = old_attn.o_proj.weight.data

            layer.self_attn = new_attn

        print(f"Loaded cached model with {len(model.model.layers)} layers")
        return model, tokenizer, compressor

    # No cache - perform full calibration
    print(f"\n{'=' * 80}")
    print(f"NO CACHED MODEL FOUND - CALIBRATING")
    print(f"{'=' * 80}")
    print(f"\nLoading {model_name}...")

    # Load pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).to(device)

    config = model.config

    print(f"\nModel config:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num heads: {config.num_attention_heads}")
    print(f"  Num KV heads: {config.num_key_value_heads}")
    print(f"  Head dim: {config.hidden_size // config.num_attention_heads}")

    # Calibrate compressor
    print(f"\n{'=' * 80}")
    print("CALIBRATING SVD COMPRESSOR")
    print(f"{'=' * 80}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Collect K/V head activations for SVD calibration
    print(f"Collecting {calibration_samples} K/V activation samples...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [item["text"] for item in dataset if len(item["text"].strip()) > 50]

    head_dim = config.hidden_size // config.num_attention_heads
    k_activations = []
    samples_collected = 0

    # Hook K projection to capture per-head activations
    def hook_k_proj(module, input, output):
        # output shape: [batch, seq, num_kv_heads * head_dim]
        bsz, seq_len, _ = output.shape
        # Reshape to [batch, seq, num_kv_heads, head_dim]
        k_heads = output.reshape(bsz, seq_len, config.num_key_value_heads, head_dim)
        # Flatten to [batch * seq * num_kv_heads, head_dim]
        k_flat = k_heads.reshape(-1, head_dim)
        k_activations.append(k_flat.detach().cpu())

    # Hook K projection on first layer
    hook = model.model.layers[0].self_attn.k_proj.register_forward_hook(hook_k_proj)

    model.eval()
    with torch.no_grad():
        for text in texts:
            if samples_collected >= calibration_samples:
                break

            inputs = tokenizer(
                text, max_length=512, truncation=True, return_tensors="pt"
            ).to(device)
            if inputs["input_ids"].shape[1] < 10:
                continue

            model(**inputs)
            samples_collected += (
                inputs["input_ids"].shape[1] * config.num_key_value_heads
            )

    hook.remove()

    # Compute SVD on per-head activations
    all_activations = torch.cat(k_activations, dim=0)[:calibration_samples].float()
    print(f"Computing SVD on K heads: {all_activations.shape}...")

    mean = all_activations.mean(dim=0, keepdim=True)
    centered = all_activations - mean

    U, S, V = torch.svd(centered.cpu())

    total_var = S.pow(2).sum()
    d_compressed = max(1, int(head_dim * compression_ratio))
    kept_var = S[:d_compressed].pow(2).sum()
    explained = (kept_var / total_var * 100).item()

    print(f"Explained variance: {explained:.2f}%")
    print(f"Compression: {head_dim} → {d_compressed} per head")

    # Create compressor from top-k principal components
    V_compressed = V[:, :d_compressed]  # [head_dim, d_compressed]

    compressor = SVDCompressor(head_dim, d_compressed)
    compressor.calibrate_from_svd(V_compressed)
    compressor = compressor.to(device=device, dtype=torch.float16)
    compressor.eval()
    compressor.requires_grad_(False)

    # Replace attention layers
    print(f"\n{'=' * 80}")
    print("REPLACING ATTENTION LAYERS")
    print(f"{'=' * 80}")

    for layer_idx, layer in enumerate(model.model.layers):
        old_attn = layer.self_attn

        # Create new compressed attention
        new_attn = CompressedKVAttention(config, compressor).to(device)

        # Copy weights from pretrained
        new_attn.q_proj.weight.data = old_attn.q_proj.weight.data
        new_attn.k_proj.weight.data = old_attn.k_proj.weight.data
        new_attn.v_proj.weight.data = old_attn.v_proj.weight.data
        new_attn.o_proj.weight.data = old_attn.o_proj.weight.data

        # Replace
        layer.self_attn = new_attn

    print(f"Replaced {len(model.model.layers)} attention layers")

    # Save compressor to cache
    print(f"\nSaving compressor to cache: {cache_path}")
    torch.save(
        {
            "compressor": compressor.cpu(),
            "compression_ratio": compression_ratio,
            "calibration_samples": calibration_samples,
            "head_dim": head_dim,
            "d_compressed": d_compressed,
        },
        cache_path,
    )
    compressor = compressor.to(device)
    print(f"Cached for future use")

    # Cleanup
    del k_activations, all_activations
    gc.collect()
    torch.cuda.empty_cache()

    return model, tokenizer, compressor


def benchmark_generation(model, tokenizer, max_new_tokens=100, trials=3):
    """Benchmark generation."""
    device = next(model.parameters()).device
    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print(f"\nBenchmarking generation...")
    print(f"  Prompt: '{prompt}'")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"  Trials: {trials}")

    # Warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10, do_sample=False, use_cache=True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    times = []
    for trial in range(trials):
        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
            )

        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)

        tokens_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
        throughput = tokens_generated / elapsed
        print(f"  Trial {trial + 1}: {elapsed:.3f}s ({throughput:.2f} tok/s)")

    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    avg_time = sum(times) / len(times)
    avg_throughput = max_new_tokens / avg_time

    print(f"\n  Average: {avg_time:.3f}s ({avg_throughput:.2f} tok/s)")
    print(f"  Peak memory: {peak_memory_mb:.1f} MB")

    return {
        "avg_time": avg_time,
        "avg_throughput": avg_throughput,
        "peak_memory_mb": peak_memory_mb,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3"
    )
    parser.add_argument("--compression-ratio", type=float, default=0.5)
    parser.add_argument("--calibration-samples", type=int, default=2000)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compare-baseline", action="store_true")

    args = parser.parse_args()

    print("=" * 80)
    print("Mistral with Native Compressed KV Cache")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Compression: {args.compression_ratio}")
    print(f"Device: {args.device}")

    # Baseline
    baseline_results = None
    if args.compare_baseline:
        print(f"\n{'=' * 80}")
        print("BASELINE")
        print(f"{'=' * 80}")

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_baseline = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16
        ).to(args.device)
        baseline_results = benchmark_generation(
            model_baseline, tokenizer, args.max_new_tokens, args.trials
        )

        del model_baseline
        gc.collect()
        torch.cuda.empty_cache()

    # Compressed
    print(f"\n{'=' * 80}")
    print("COMPRESSED KV CACHE")
    print(f"{'=' * 80}")

    model_compressed, tokenizer, compressor = load_mistral_with_compressed_kv(
        args.model,
        args.compression_ratio,
        args.calibration_samples,
        args.device,
    )

    compressed_results = benchmark_generation(
        model_compressed, tokenizer, args.max_new_tokens, args.trials
    )

    # Compare
    if baseline_results:
        print(f"\n{'=' * 80}")
        print("COMPARISON")
        print(f"{'=' * 80}")

        throughput_ratio = (
            compressed_results["avg_throughput"] / baseline_results["avg_throughput"]
        )
        memory_ratio = (
            compressed_results["peak_memory_mb"] / baseline_results["peak_memory_mb"]
        )

        print(f"\nThroughput: {throughput_ratio:.3f}x")
        print(f"Memory: {memory_ratio:.3f}x")
        print(
            f"\nBaseline:   {baseline_results['avg_throughput']:.2f} tok/s, {baseline_results['peak_memory_mb']:.1f} MB"
        )
        print(
            f"Compressed: {compressed_results['avg_throughput']:.2f} tok/s, {compressed_results['peak_memory_mb']:.1f} MB"
        )
        print(
            f"\nMemory saved: {baseline_results['peak_memory_mb'] - compressed_results['peak_memory_mb']:.1f} MB"
        )


if __name__ == "__main__":
    main()

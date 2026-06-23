#!/usr/bin/env python3
"""
Routing prior extraction + fused kernel accuracy sweep.

Phase 1+2 of the routing-accuracy-prior execution plan:
1. Load a GQA model (Llama-like)
2. Run dense prefill on real text, extract per-head block affinities
3. Feed prefill-derived priors into the fused Triton kernel
4. Measure cosine similarity at BS={128,256} x K={4,8,16}
5. Compare against random priors as a control

Prior extraction method (from capture_prefill):
- Bulk prefill tokens [0..N-2] with SDPA (no attention matrix)
- Forward last token with output_attentions=True (eager fallback)
- Sum attention weights per block → block affinities [n_layers, n_heads, n_blocks]
- Select top-K blocks per KV head as routing indices

Usage:
  python3 routing_prior_extraction_sweep.py --model <model_path_or_hf_id>
"""

import torch
import time
import json
import sys
import os
import gc
import argparse
import math
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

# Add routing kernel path
sys.path.insert(0, "/workspace/knlp-routing-dev/routing")
from fused_routed_attention import (
    fused_routed_decode,
    select_top_k_blocks,
)


def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] {msg}", flush=True)


# ─── Phase 1: Prefill-derived prior extraction ─────────────────────────

def extract_prefill_priors(model, input_ids, block_size):
    """Extract per-head block affinities during prefill.

    Strategy (same as capture_prefill from allhead_8k_cartridge_experiment.py):
    1. Prefill tokens [0..N-2] with SDPA (no attention matrix) → KV cache
    2. Forward last token with output_attentions=True → attention [1, n_heads, 1, N-1]
    3. Sum attention weights per block to get block affinities

    Returns:
        block_affinities: [n_layers, n_q_heads, n_blocks] float32
        kv_cache: past_key_values from the model
    """
    n_layers = model.config.num_hidden_layers
    n_q_heads = model.config.num_attention_heads
    seq_len = input_ids.shape[1]
    n_blocks = (seq_len + block_size - 1) // block_size

    block_affinities = torch.zeros(n_layers, n_q_heads, n_blocks)

    # Step 1: Bulk prefill (all but last token) with SDPA — no OOM
    log(f"  Bulk prefill {seq_len - 1} tokens...")
    with torch.no_grad():
        bulk_out = model(
            input_ids[:, :-1],
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
    past_kv = bulk_out.past_key_values

    # Step 2: Last token with eager fallback to get attention weights
    log("  Last-token forward with attention capture...")
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            # output can be (attn_output, attn_weights, ...) or just attn_output
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn_weights = output[1]
                # attn_weights: [batch, n_heads, 1, cache_len+1]
                last_row = attn_weights[0, :, 0, :].detach().cpu().float()
                for b in range(n_blocks):
                    s = b * block_size
                    e = min((b + 1) * block_size, seq_len)
                    if e <= last_row.shape[-1]:
                        block_affinities[layer_idx, :, b] = last_row[:, s:e].sum(dim=-1)
                return (output[0], None) + output[2:] if len(output) > 2 else (output[0], None)
            return output
        return hook_fn

    # Find attention modules - works for Llama, Qwen, Mistral, etc.
    for i, layer in enumerate(model.model.layers):
        h = layer.self_attn.register_forward_hook(make_hook(i), with_kwargs=True)
        hooks.append(h)

    # Force eager attention for this single-token step
    orig_impl = getattr(model.config, '_attn_implementation', None)
    model.config._attn_implementation = "eager"

    with torch.no_grad():
        last_out = model(
            input_ids[:, -1:],
            past_key_values=past_kv,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )

    if orig_impl is not None:
        model.config._attn_implementation = orig_impl

    for h in hooks:
        h.remove()

    past_kv = last_out.past_key_values
    return block_affinities, past_kv


def affinities_to_block_tables(block_affinities, n_kv_heads, K, device='cuda'):
    """Convert per-Q-head block affinities to per-KV-head block tables.

    With GQA, multiple Q heads share one KV head. We average affinities
    across the Q heads in each group, then select top-K blocks.

    Args:
        block_affinities: [n_layers, n_q_heads, n_blocks]
        n_kv_heads: number of KV heads
        K: number of blocks to retain per KV head

    Returns:
        block_tables_per_layer: list of [1, n_kv_heads, K] int32 tensors (batch=1)
        block_counts_per_layer: list of [1, n_kv_heads] int32 tensors (batch=1)
    """
    n_layers, n_q_heads, n_blocks = block_affinities.shape
    gqa_ratio = n_q_heads // n_kv_heads

    block_tables_list = []
    block_counts_list = []

    for l in range(n_layers):
        bt = torch.zeros(1, n_kv_heads, K, device=device, dtype=torch.int32)
        bc = torch.full((1, n_kv_heads), K, device=device, dtype=torch.int32)

        for kv_h in range(n_kv_heads):
            # Average affinities across Q heads in this KV group
            q_start = kv_h * gqa_ratio
            q_end = q_start + gqa_ratio
            avg_aff = block_affinities[l, q_start:q_end, :].mean(dim=0)  # [n_blocks]

            # Select top-K blocks
            actual_k = min(K, n_blocks)
            top_k_indices = avg_aff.topk(actual_k).indices
            bt[0, kv_h, :actual_k] = top_k_indices.int().to(device)
            bc[0, kv_h] = actual_k

        block_tables_list.append(bt)
        block_counts_list.append(bc)

    return block_tables_list, block_counts_list


def generate_random_block_tables(n_layers, n_kv_heads, total_blocks, K, device='cuda'):
    """Generate random block tables as a control (batch=1)."""
    block_tables_list = []
    block_counts_list = []

    for _ in range(n_layers):
        bt = torch.zeros(1, n_kv_heads, K, device=device, dtype=torch.int32)
        bc = torch.full((1, n_kv_heads), K, device=device, dtype=torch.int32)
        for h in range(n_kv_heads):
            perm = torch.randperm(total_blocks, device=device)[:K]
            bt[0, h] = perm.int()
        block_tables_list.append(bt)
        block_counts_list.append(bc)

    return block_tables_list, block_counts_list


# ─── Phase 2: Accuracy sweep using fused kernel ────────────────────────

def build_kv_cache_from_model(past_kv, block_size, n_kv_heads, head_dim, device='cuda'):
    """Reshape model KV cache into blocked layout for the fused kernel.

    Model KV cache layout (per layer): key/value = [batch, n_kv_heads, seq_len, head_dim]
    Kernel KV cache layout: [total_blocks, block_size, n_kv_heads, head_dim]

    Returns k_cache, v_cache tensors.
    """
    # Extract from the first layer to get seq_len
    # Different transformers versions have different cache formats
    if hasattr(past_kv, 'key_cache'):
        # transformers >= 4.36 DynamicCache
        k0 = past_kv.key_cache[0]  # [batch, n_kv_heads, seq_len, head_dim]
        v0 = past_kv.value_cache[0]
    elif hasattr(past_kv, 'layers'):
        # Newer DynamicCache with layers
        k0 = past_kv.layers[0].keys if hasattr(past_kv.layers[0], 'keys') else past_kv[0][0]
        v0 = past_kv.layers[0].values if hasattr(past_kv.layers[0], 'values') else past_kv[0][1]
    else:
        # Legacy tuple format
        k0 = past_kv[0][0]
        v0 = past_kv[0][1]

    batch, actual_kv_heads, seq_len, actual_head_dim = k0.shape
    n_layers = len(past_kv.key_cache) if hasattr(past_kv, 'key_cache') else len(past_kv)
    total_blocks = (seq_len + block_size - 1) // block_size

    # Pad sequence to multiple of block_size if needed
    pad_len = total_blocks * block_size - seq_len

    k_caches = []
    v_caches = []

    for l in range(n_layers):
        if hasattr(past_kv, 'key_cache'):
            k = past_kv.key_cache[l][0]  # [n_kv_heads, seq_len, head_dim]
            v = past_kv.value_cache[l][0]
        elif hasattr(past_kv, 'layers'):
            k = past_kv.layers[l].keys[0] if hasattr(past_kv.layers[l], 'keys') else past_kv[l][0][0]
            v = past_kv.layers[l].values[0] if hasattr(past_kv.layers[l], 'values') else past_kv[l][1][0]
        else:
            k = past_kv[l][0][0]
            v = past_kv[l][1][0]

        # k, v: [n_kv_heads, seq_len, head_dim]
        if pad_len > 0:
            k = torch.cat([k, torch.zeros(actual_kv_heads, pad_len, actual_head_dim,
                                          device=k.device, dtype=k.dtype)], dim=1)
            v = torch.cat([v, torch.zeros(actual_kv_heads, pad_len, actual_head_dim,
                                          device=v.device, dtype=v.dtype)], dim=1)

        # Reshape: [n_kv_heads, total_blocks*block_size, head_dim]
        #       → [total_blocks, block_size, n_kv_heads, head_dim]
        k = k.reshape(actual_kv_heads, total_blocks, block_size, actual_head_dim)
        k = k.permute(1, 2, 0, 3).contiguous()  # [total_blocks, block_size, n_kv_heads, head_dim]
        v = v.reshape(actual_kv_heads, total_blocks, block_size, actual_head_dim)
        v = v.permute(1, 2, 0, 3).contiguous()

        k_caches.append(k)
        v_caches.append(v)

    return k_caches, v_caches, total_blocks, seq_len


def compute_dense_output(q, k_cache, v_cache):
    """Compute dense attention output using the fused kernel (all blocks, batch=1)."""
    total_blocks = k_cache.shape[0]
    n_kv_heads = k_cache.shape[2]
    # All blocks selected — shape [1, n_kv_heads, total_blocks]
    block_tables = torch.arange(total_blocks, device=q.device, dtype=torch.int32).unsqueeze(0).unsqueeze(0).expand(1, n_kv_heads, -1).contiguous()
    block_counts = torch.full((1, n_kv_heads), total_blocks, device=q.device, dtype=torch.int32)
    return fused_routed_decode(q, k_cache, v_cache, block_tables, block_counts)


def run_accuracy_sweep(model, tokenizer, block_sizes, k_values, device='cuda'):
    """Run the full accuracy sweep with prefill-derived vs random priors.

    For each block_size:
    1. Extract prefill-derived priors at that block size
    2. For each K:
       a. Build block tables from prefill priors
       b. Build random block tables
       c. Run fused kernel with both
       d. Compare to dense output (cosine similarity)
       e. Measure latency
    """
    n_layers = model.config.num_hidden_layers
    n_q_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // n_q_heads

    log(f"Model config: {n_layers}L, {n_q_heads}Q/{n_kv_heads}KV, d={head_dim}")

    # Generate a real text sequence for prefill
    # Use a diverse prompt to get realistic attention patterns
    text = (
        "The theory of distributed systems rests on several fundamental impossibility results "
        "that constrain what can be achieved in asynchronous networks. The FLP impossibility "
        "result shows that consensus is impossible in an asynchronous system if even one process "
        "may fail. The CAP theorem demonstrates that a distributed data store cannot simultaneously "
        "provide more than two of three guarantees: consistency, availability, and partition tolerance. "
        "These results have profound implications for the design of modern cloud systems, databases, "
        "and microservice architectures. In practice, engineers must make explicit tradeoffs between "
        "consistency models, choosing between strong consistency (linearizability), eventual consistency, "
        "or causal consistency based on application requirements. Modern consensus protocols like Raft "
        "and Paxos provide crash fault tolerance, while Byzantine fault tolerant protocols like PBFT "
        "handle arbitrary failures at higher cost. The evolution from monolithic databases to "
        "distributed NewSQL systems like CockroachDB and TiDB represents a practical response to "
        "these theoretical constraints, offering both horizontal scalability and strong consistency "
        "through careful protocol design. Key innovations include multi-version concurrency control, "
        "hybrid logical clocks, and optimistic transaction processing that reduces coordination overhead. "
        "The rise of serverless computing introduces additional challenges for state management, as "
        "ephemeral function instances must coordinate through external storage or messaging systems."
    )
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    seq_len = input_ids.shape[1]
    log(f"Input sequence: {seq_len} tokens")

    # Extend to 4096 tokens if the model supports it and text is short
    max_len = min(4096, getattr(model.config, 'max_position_embeddings', 4096))
    if seq_len < max_len:
        log(f"Extending to {max_len} tokens via generation...")
        with torch.no_grad():
            extended = model.generate(
                input_ids,
                max_new_tokens=max_len - seq_len,
                min_new_tokens=max(1, max_len - seq_len - 50),
                do_sample=True, temperature=0.8, top_p=0.95,
            )
        input_ids = extended[:, :max_len]
        seq_len = input_ids.shape[1]
        log(f"Extended to {seq_len} tokens")

    results = {}

    for block_size in block_sizes:
        total_blocks = (seq_len + block_size - 1) // block_size
        log(f"\n=== Block Size = {block_size}, {total_blocks} blocks, {seq_len} tokens ===")

        # Extract prefill-derived priors
        log("Extracting prefill-derived priors...")
        t0 = time.time()
        block_affinities, past_kv = extract_prefill_priors(model, input_ids, block_size)
        extraction_time = time.time() - t0
        log(f"  Prior extraction: {extraction_time:.1f}s")

        # Audit: sparsity of affinities
        total_entries = block_affinities.numel()
        near_zero = (block_affinities.abs() < 0.01).sum().item()
        log(f"  Affinity sparsity: {near_zero}/{total_entries} ({100*near_zero/total_entries:.1f}%) near-zero")

        # Build blocked KV cache from model output
        log("Building blocked KV cache...")
        k_caches, v_caches, actual_total_blocks, actual_seq_len = build_kv_cache_from_model(
            past_kv, block_size, n_kv_heads, head_dim, device
        )

        # Generate a decode query token (next token prediction)
        # Use the model's last hidden state to get a realistic query
        with torch.no_grad():
            # Simple: use the embedding of the last generated token
            last_token_id = input_ids[0, -1].unsqueeze(0).unsqueeze(0)  # [1, 1]
            # Get hidden states for the query
            q_out = model(last_token_id, past_key_values=past_kv, use_cache=False, return_dict=True)

        # Extract Q from the query layer outputs — we need the actual Q projection
        # Simpler approach: generate a random Q with the right distribution
        # But more accurately: run a forward hook to capture Q
        q_vectors = []
        q_hooks = []

        def make_q_capture_hook(layer_idx):
            def hook_fn(module, args, kwargs, output):
                # For Llama/Qwen style: module.q_proj produces Q
                # We need to capture the actual query vector
                pass  # We'll use a different approach
            return hook_fn

        # Actually, let's compute Q directly from the model's Q projection
        log("Computing decode Q vectors per layer...")
        last_hidden = None

        # Get the hidden state at the last position
        hidden_hooks = []
        layer_hidden_states = {}

        def make_hidden_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    layer_hidden_states[layer_idx] = output[0][:, -1:, :].detach()
                else:
                    layer_hidden_states[layer_idx] = output[:, -1:, :].detach()
            return hook_fn

        for i, layer in enumerate(model.model.layers):
            if i == 0:
                # Capture input to first layer
                def capture_input(module, input):
                    layer_hidden_states[-1] = input[0][:, -1:, :].detach() if isinstance(input, tuple) else input[:, -1:, :].detach()
                h = layer.register_forward_pre_hook(capture_input)
                hidden_hooks.append(h)
            h = layer.register_forward_hook(make_hidden_hook(i))
            hidden_hooks.append(h)

        with torch.no_grad():
            model(last_token_id, past_key_values=past_kv, use_cache=False, return_dict=True)

        for h in hidden_hooks:
            h.remove()

        # Now compute Q for each layer using its input hidden state and Q projection
        q_per_layer = []
        for l in range(n_layers):
            input_hidden = layer_hidden_states.get(l - 1, layer_hidden_states.get(-1))
            if input_hidden is None:
                log(f"  WARNING: no hidden state for layer {l}, using random Q")
                q = torch.randn(1, n_q_heads, head_dim, device=device, dtype=torch.float16)
            else:
                # Apply the layer's Q projection
                attn = model.model.layers[l].self_attn
                q_proj = attn.q_proj(input_hidden.to(attn.q_proj.weight.dtype))
                # Reshape: [1, 1, n_q_heads * head_dim] → [1, n_q_heads, head_dim]
                q = q_proj.view(1, n_q_heads, head_dim).to(torch.float16)
            q_per_layer.append(q)

        # Free model KV cache to save GPU memory
        del past_kv
        gc.collect()
        torch.cuda.empty_cache()

        for K in k_values:
            if K >= actual_total_blocks:
                log(f"  K={K} >= total_blocks={actual_total_blocks}, skipping")
                continue

            kv_reduction = 1.0 - K / actual_total_blocks
            tokens_per_kvh = K * block_size
            log(f"\n  --- BS={block_size}, K={K} ({tokens_per_kvh} tokens/KVH, {kv_reduction*100:.1f}% KV reduction) ---")

            # Build block tables: prefill-derived and random
            prefill_tables, prefill_counts = affinities_to_block_tables(
                block_affinities, n_kv_heads, K, device
            )
            random_tables, random_counts = generate_random_block_tables(
                n_layers, n_kv_heads, actual_total_blocks, K, device
            )

            # Per-layer cosine similarity: routed vs dense
            cos_prefill_layers = []
            cos_random_layers = []
            latency_prefill_layers = []
            latency_dense_layers = []

            for l in range(n_layers):
                q = q_per_layer[l]
                k_cache = k_caches[l]
                v_cache = v_caches[l]

                # Dense output (reference)
                dense_out = compute_dense_output(q, k_cache, v_cache)

                # Prefill-prior routed output
                routed_prefill = fused_routed_decode(
                    q, k_cache, v_cache, prefill_tables[l], prefill_counts[l]
                )

                # Random-prior routed output
                routed_random = fused_routed_decode(
                    q, k_cache, v_cache, random_tables[l], random_counts[l]
                )

                # Cosine similarity
                dense_flat = dense_out.float().flatten()
                cos_p = torch.nn.functional.cosine_similarity(
                    routed_prefill.float().flatten().unsqueeze(0),
                    dense_flat.unsqueeze(0)
                ).item()
                cos_r = torch.nn.functional.cosine_similarity(
                    routed_random.float().flatten().unsqueeze(0),
                    dense_flat.unsqueeze(0)
                ).item()

                cos_prefill_layers.append(cos_p)
                cos_random_layers.append(cos_r)

            # Aggregate metrics
            avg_cos_prefill = np.mean(cos_prefill_layers)
            avg_cos_random = np.mean(cos_random_layers)
            std_cos_prefill = np.std(cos_prefill_layers)
            min_cos_prefill = np.min(cos_prefill_layers)
            max_cos_prefill = np.max(cos_prefill_layers)

            # Latency measurement (use layer 16 as representative)
            mid_layer = n_layers // 2
            q_bench = q_per_layer[mid_layer]
            k_bench = k_caches[mid_layer]
            v_bench = v_caches[mid_layer]

            # Warmup
            for _ in range(10):
                fused_routed_decode(q_bench, k_bench, v_bench, prefill_tables[mid_layer], prefill_counts[mid_layer])
                compute_dense_output(q_bench, k_bench, v_bench)
            torch.cuda.synchronize()

            # Timed runs
            n_runs = 100
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n_runs):
                fused_routed_decode(q_bench, k_bench, v_bench, prefill_tables[mid_layer], prefill_counts[mid_layer])
            torch.cuda.synchronize()
            routed_latency = (time.time() - t0) / n_runs * 1000  # ms

            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n_runs):
                compute_dense_output(q_bench, k_bench, v_bench)
            torch.cuda.synchronize()
            dense_latency = (time.time() - t0) / n_runs * 1000  # ms

            speedup = dense_latency / routed_latency if routed_latency > 0 else float('inf')

            result_key = f"BS={block_size}_K={K}"
            results[result_key] = {
                "block_size": block_size,
                "K": K,
                "tokens_per_kvh": tokens_per_kvh,
                "kv_reduction_pct": kv_reduction * 100,
                "prefill_cos_mean": float(avg_cos_prefill),
                "prefill_cos_std": float(std_cos_prefill),
                "prefill_cos_min": float(min_cos_prefill),
                "prefill_cos_max": float(max_cos_prefill),
                "random_cos_mean": float(avg_cos_random),
                "random_cos_std": float(np.std(cos_random_layers)),
                "prefill_cos_per_layer": [float(c) for c in cos_prefill_layers],
                "random_cos_per_layer": [float(c) for c in cos_random_layers],
                "routed_latency_ms": float(routed_latency),
                "dense_latency_ms": float(dense_latency),
                "speedup_vs_dense": float(speedup),
            }

            log(f"  PREFILL cos={avg_cos_prefill:.4f} (std={std_cos_prefill:.4f}, "
                f"min={min_cos_prefill:.4f}, max={max_cos_prefill:.4f})")
            log(f"  RANDOM  cos={avg_cos_random:.4f}")
            log(f"  Δcos = {avg_cos_prefill - avg_cos_random:+.4f} (prefill - random)")
            log(f"  Latency: routed={routed_latency:.3f}ms, dense={dense_latency:.3f}ms, "
                f"speedup={speedup:.1f}x")

        # Free per-block-size resources
        del k_caches, v_caches, q_per_layer, block_affinities
        gc.collect()
        torch.cuda.empty_cache()

    return results


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       help="HuggingFace model ID or local path")
    parser.add_argument("--block-sizes", type=str, default="128,256",
                       help="Comma-separated block sizes")
    parser.add_argument("--k-values", type=str, default="4,8,16",
                       help="Comma-separated K values")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON path (default: auto-generated)")
    args = parser.parse_args()

    block_sizes = [int(x) for x in args.block_sizes.split(",")]
    k_values = [int(x) for x in args.k_values.split(",")]

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output or f"/workspace/routing_prior_sweep_{ts}.json"

    log("=== ROUTING PRIOR EXTRACTION + ACCURACY SWEEP ===")
    log(f"Model: {args.model}")
    log(f"Block sizes: {block_sizes}")
    log(f"K values: {k_values}")
    log(f"Output: {output_path}")

    # Load model
    log("Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    n_q_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // n_q_heads

    log(f"Model loaded: {n_layers}L, {n_q_heads}Q/{n_kv_heads}KV, d={head_dim}")
    log(f"Max position embeddings: {getattr(model.config, 'max_position_embeddings', 'unknown')}")

    # Run sweep
    results = run_accuracy_sweep(model, tokenizer, block_sizes, k_values, device='cuda')

    # Add metadata
    output_data = {
        "metadata": {
            "timestamp": ts,
            "model": args.model,
            "n_layers": n_layers,
            "n_q_heads": n_q_heads,
            "n_kv_heads": n_kv_heads,
            "head_dim": head_dim,
            "block_sizes": block_sizes,
            "k_values": k_values,
            "gpu": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
        },
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    log(f"\nResults written to: {output_path}")

    # Print summary table
    log("\n=== SUMMARY TABLE ===")
    log(f"{'Config':<20} {'Prefill cos':>12} {'Random cos':>12} {'Δcos':>8} {'KV red%':>8} {'Speedup':>8}")
    log("-" * 78)
    for key in sorted(results.keys()):
        r = results[key]
        delta = r['prefill_cos_mean'] - r['random_cos_mean']
        log(f"BS={r['block_size']:>3} K={r['K']:>2}       "
            f"{r['prefill_cos_mean']:>12.4f} {r['random_cos_mean']:>12.4f} "
            f"{delta:>+8.4f} {r['kv_reduction_pct']:>7.1f}% {r['speedup_vs_dense']:>7.1f}x")

    log("\n=== DONE ===")


if __name__ == "__main__":
    main()

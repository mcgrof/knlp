#!/usr/bin/env python3
"""Phase 3: Collect exact ground-truth decode traces."""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.qk_router.trace_collect import (
    collect_traces,
    collect_q_vectors,
    traces_to_serializable,
)
from lib.qk_router.blocking import build_block_map
from lib.qk_router.utils import save_json, Timer


def load_workload_texts(
    tokenizer,
    num_requests: int,
    prefix_length: int,
    max_new_tokens: int,
    dataset: str = "wikitext-2-raw-v1",
    seed: int = 42,
    cache_dir: str = None,
):
    """Load workload texts from dataset, return list of input_ids tensors."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", dataset, split="train", cache_dir=cache_dir)
    # Concatenate all text
    all_text = "\n".join([t for t in ds["text"] if len(t) > 100])
    all_tokens = tokenizer.encode(all_text)
    total_len = prefix_length + max_new_tokens

    rng = np.random.RandomState(seed)
    inputs = []
    for i in range(num_requests):
        # Sample a random starting position
        max_start = len(all_tokens) - total_len - 1
        if max_start <= 0:
            start = 0
        else:
            start = rng.randint(0, max_start)
        chunk = all_tokens[start : start + total_len]
        inputs.append(torch.tensor([chunk], dtype=torch.long))

    return inputs


def main():
    run_root = os.environ.get("RUN_ROOT", "/mnt/tmpfs/knlp/results/qk_router_01")
    cache_dir = os.environ.get("HF_CACHE", "/mnt/SFS-hugging/hub")
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B")

    prefix_length = int(os.environ.get("PREFIX_LENGTH", "4096"))
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "64"))
    num_requests = int(os.environ.get("NUM_REQUESTS", "64"))
    block_size = int(os.environ.get("BLOCK_SIZE", "128"))
    seed = int(os.environ.get("SEED", "42"))

    print("=" * 60)
    print("QK Router Phase 3: Trace Collection")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Prefix length: {prefix_length}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Num requests: {num_requests}")
    print(f"Block size: {block_size}")
    print(f"Num prefix blocks: {prefix_length // block_size}")
    print()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    print("Loading workload texts...")
    inputs = load_workload_texts(
        tokenizer,
        num_requests,
        prefix_length,
        max_new_tokens,
        cache_dir=cache_dir,
        seed=seed,
    )

    # Build block map
    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    block_map = build_block_map(
        prefix_length, block_size, num_layers, num_kv_heads, head_dim
    )
    print(f"Block map: {len(block_map)} blocks, {block_map[0].total_bytes} bytes each")

    # Save block map
    block_map_data = [
        {
            "block_id": b.block_id,
            "token_start": b.token_start,
            "token_end": b.token_end,
            "bytes_per_layer": b.bytes_per_layer,
            "total_bytes": b.total_bytes,
        }
        for b in block_map
    ]
    save_json(block_map_data, os.path.join(run_root, "block_map.json"))

    # Collect traces and Q vectors for each request
    trace_dir = os.path.join(run_root, "traces")
    q_dir = os.path.join(run_root, "q_vectors")
    os.makedirs(trace_dir, exist_ok=True)
    os.makedirs(q_dir, exist_ok=True)

    all_trace_summaries = []

    for req_idx in range(num_requests):
        print(f"\n--- Request {req_idx + 1}/{num_requests} ---")

        input_ids = inputs[req_idx]

        # Collect attention traces
        with Timer(f"traces_req{req_idx}"):
            traces = collect_traces(
                model,
                tokenizer,
                input_ids,
                prefix_length,
                max_new_tokens,
                block_size,
                request_id=req_idx,
                device="cuda",
            )

        # Save traces
        trace_data = traces_to_serializable(traces)
        save_json(trace_data, os.path.join(trace_dir, f"trace_req{req_idx:03d}.json"))

        # Collect Q vectors (separate pass with SDPA for speed)
        # Reset attention implementation to sdpa
        model.config._attn_implementation = "sdpa"
        for layer in model.model.layers:
            layer.self_attn.config._attn_implementation = "sdpa"

        with Timer(f"q_vectors_req{req_idx}"):
            q_vecs = collect_q_vectors(
                model, tokenizer, input_ids, prefix_length, max_new_tokens
            )

        # Save Q vectors
        np.save(
            os.path.join(q_dir, f"q_req{req_idx:03d}.npy"),
            np.stack(q_vecs, axis=0),  # [num_steps, num_layers, head_dim]
        )

        # Summary
        needed_counts = [len(t["needed_blocks_mass"]) for t in trace_data]
        all_trace_summaries.append(
            {
                "request_id": req_idx,
                "num_steps": len(trace_data),
                "avg_needed_blocks": float(np.mean(needed_counts)),
                "max_needed_blocks": int(max(needed_counts)),
            }
        )

        torch.cuda.empty_cache()

    # Save summary
    save_json(
        {
            "num_requests": num_requests,
            "prefix_length": prefix_length,
            "max_new_tokens": max_new_tokens,
            "block_size": block_size,
            "num_prefix_blocks": prefix_length // block_size,
            "per_request": all_trace_summaries,
        },
        os.path.join(run_root, "trace_summary.json"),
    )

    print(f"\nPhase 3 complete. Traces saved to {trace_dir}")
    print(f"Q vectors saved to {q_dir}")


if __name__ == "__main__":
    main()

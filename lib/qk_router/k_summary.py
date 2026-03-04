"""K-summary construction: build always-resident routing index."""

import numpy as np
import torch


def build_k_summaries(
    model,
    input_ids: torch.Tensor,
    prefix_length: int,
    block_size: int,
    mode: str = "direct_centroid",
    device: str = "cuda",
    seed: int = 42,
) -> np.ndarray:
    """Build K-summary vectors for all prefix blocks.

    Returns: [num_blocks, num_layers, summary_dim] array.

    Modes:
      direct_centroid: per-layer centroid of real block keys, normalized
      random_summary: random normalized vectors (dumb baseline)
      first_k_real: summaries from first few blocks only
      sampled_real_geometry: sampled across full workload span
    """
    num_prefix_blocks = prefix_length // block_size
    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    if mode == "random_summary":
        rng = np.random.RandomState(seed)
        summaries = rng.randn(num_prefix_blocks, num_layers, head_dim).astype(
            np.float32
        )
        # Normalize
        norms = np.linalg.norm(summaries, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        summaries = summaries / norms
        return summaries

    # For real-key modes, we need to extract K values from the model
    k_captures = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            k_captures[layer_idx] = output.detach()

        return hook_fn

    hooks = []
    for i, layer in enumerate(model.model.layers):
        h = layer.self_attn.k_proj.register_forward_hook(make_hook(i))
        hooks.append(h)

    try:
        ids = input_ids[:, :prefix_length].to(device)
        with torch.no_grad():
            model(ids, use_cache=False)

        # Build summaries from captured K projections
        summaries = np.zeros(
            (num_prefix_blocks, num_layers, head_dim), dtype=np.float32
        )

        for layer_idx in range(num_layers):
            if layer_idx not in k_captures:
                continue
            k_proj = k_captures[layer_idx]  # [B, T, num_kv_heads * head_dim]
            k_proj = k_proj[0]  # [T, num_kv_heads * head_dim]

            # Reshape to [T, num_kv_heads, head_dim]
            k_heads = k_proj.view(prefix_length, num_kv_heads, head_dim)

            # Mean across KV heads
            k_mean = k_heads.mean(dim=1)  # [T, head_dim]

            for b_idx in range(num_prefix_blocks):
                t_start = b_idx * block_size
                t_end = t_start + block_size
                block_keys = k_mean[t_start:t_end]  # [block_size, head_dim]

                if mode == "direct_centroid":
                    centroid = block_keys.mean(dim=0).cpu().float().numpy()
                elif mode == "first_k_real":
                    # Use first 4 blocks' keys for all blocks
                    if b_idx < 4:
                        centroid = block_keys.mean(dim=0).cpu().float().numpy()
                    else:
                        # Copy from block b_idx % 4
                        centroid = summaries[b_idx % 4, layer_idx].copy()
                elif mode == "sampled_real_geometry":
                    # Sample keys from across the full prefix
                    rng = np.random.RandomState(seed + b_idx + layer_idx)
                    sample_indices = rng.choice(
                        prefix_length, size=min(32, block_size), replace=False
                    )
                    sampled = k_mean[sample_indices]
                    centroid = sampled.mean(dim=0).cpu().float().numpy()
                else:
                    centroid = block_keys.mean(dim=0).cpu().float().numpy()

                # Normalize
                norm = np.linalg.norm(centroid)
                if norm > 1e-8:
                    centroid = centroid / norm
                summaries[b_idx, layer_idx] = centroid

    finally:
        for h in hooks:
            h.remove()

    return summaries

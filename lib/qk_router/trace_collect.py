"""Trace collection: run decode with all KV hot, record Q_t and attention mass."""

import torch
import numpy as np
from dataclasses import dataclass, field


@dataclass
class DecodeStep:
    """One decode step's trace data."""

    request_id: int
    decode_step: int
    generated_token_id: int
    baseline_latency_us: float
    q_per_layer: list  # [num_layers, num_heads, head_dim] as numpy
    attn_mass_per_block: list  # [num_blocks] aggregated attention mass
    top_k_blocks: list  # top-k block indices by attention mass
    needed_blocks_mass: list  # blocks with mass > threshold
    needed_blocks_topk: list  # blocks in top-k
    reuse_distance: dict  # {block_id: steps_since_last_needed}
    prefix_block_count: int


def collect_traces(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    prefix_length: int,
    max_new_tokens: int,
    block_size: int,
    request_id: int = 0,
    attention_mass_threshold: float = 0.01,
    top_k_blocks: int = 8,
    device: str = "cuda",
) -> list[DecodeStep]:
    """Collect exact decode traces with attention mass per prefix block.

    Runs model with eager attention to get attention weights,
    then aggregates per block.
    """
    model.eval()
    num_layers = model.config.num_hidden_layers
    num_prefix_blocks = prefix_length // block_size

    # Set eager attention to get weights
    model.config._attn_implementation = "eager"
    for layer in model.model.layers:
        layer.self_attn.config._attn_implementation = "eager"

    # Prefill
    ids = input_ids[:, :prefix_length].to(device)
    with torch.no_grad():
        out = model(
            ids,
            output_attentions=True,
            use_cache=True,
        )
    past = out.past_key_values
    attentions_prefill = out.attentions  # tuple of [B, H, T, T]

    traces = []
    last_needed = {}  # block_id -> last step it was needed
    generated_ids = ids.clone()

    for step in range(max_new_tokens):
        # Get the next token position
        cur_pos = prefix_length + step

        if step == 0:
            # Use prefill attention for step 0: last query row
            attn_weights_list = attentions_prefill
            next_token_logits = out.logits[:, -1, :]
        else:
            # Decode step: single token
            last_tok = generated_ids[:, -1:].to(device)
            position_ids = torch.tensor(
                [[cur_pos - 1]], device=device, dtype=torch.long
            )
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            with torch.no_grad():
                out = model(
                    last_tok,
                    past_key_values=past,
                    output_attentions=True,
                    use_cache=True,
                    position_ids=position_ids,
                )
            t1.record()
            torch.cuda.synchronize()
            latency_us = t0.elapsed_time(t1) * 1000
            past = out.past_key_values
            attn_weights_list = out.attentions
            next_token_logits = out.logits[:, -1, :]

        # Greedy decode
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        # Extract Q vectors and attention mass per prefix block
        q_per_layer = []
        block_mass_per_layer = []

        for layer_idx, attn_w in enumerate(attn_weights_list):
            # attn_w: [B, H, T_q, T_kv]
            # For decode steps after 0, T_q=1
            # Take the last query row
            last_row = attn_w[0, :, -1, :]  # [H, T_kv]

            # Extract Q from the layer (approximate from attention pattern)
            # Actually we need the raw Q vector. Let's get it from the cache.
            # For now, use attention mass as the primary signal.

            # Aggregate attention mass per prefix block
            # last_row covers positions 0..T_kv-1
            # Prefix blocks cover positions 0..prefix_length-1
            kv_len = last_row.shape[-1]
            num_heads = last_row.shape[0]

            block_mass = torch.zeros(num_prefix_blocks, device=device)
            for b_idx in range(num_prefix_blocks):
                t_start = b_idx * block_size
                t_end = min(t_start + block_size, kv_len)
                if t_start < kv_len:
                    # Sum attention mass across all heads for this block
                    block_mass[b_idx] = last_row[:, t_start:t_end].sum()

            # Normalize: divide by num_heads so mass sums to ~1 per head
            block_mass = block_mass / num_heads
            block_mass_per_layer.append(block_mass.cpu().numpy())

        # Aggregate across layers: mean
        block_mass_all = np.stack(block_mass_per_layer, axis=0)
        block_mass_agg = block_mass_all.mean(axis=0)

        # Needed blocks
        needed_mass = [
            int(i)
            for i in range(num_prefix_blocks)
            if block_mass_agg[i] > attention_mass_threshold
        ]
        top_k_idx = np.argsort(block_mass_agg)[::-1][:top_k_blocks].tolist()
        needed_topk = [int(i) for i in top_k_idx]

        # Reuse distance
        reuse_dist = {}
        for bid in range(num_prefix_blocks):
            if bid in last_needed:
                reuse_dist[bid] = step - last_needed[bid]
            else:
                reuse_dist[bid] = step + 1  # never needed before
        for bid in needed_mass:
            last_needed[bid] = step

        latency = 0.0 if step == 0 else latency_us

        traces.append(
            DecodeStep(
                request_id=request_id,
                decode_step=step,
                generated_token_id=int(next_token[0, 0].item()),
                baseline_latency_us=latency,
                q_per_layer=[],  # filled separately if needed
                attn_mass_per_block=block_mass_agg.tolist(),
                top_k_blocks=needed_topk,
                needed_blocks_mass=needed_mass,
                needed_blocks_topk=needed_topk,
                reuse_distance=reuse_dist,
                prefix_block_count=num_prefix_blocks,
            )
        )

    return traces


def collect_q_vectors(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    prefix_length: int,
    max_new_tokens: int,
    device: str = "cuda",
) -> list[np.ndarray]:
    """Collect per-layer mean Q vectors for each decode step.

    Returns list of [num_layers, head_dim] arrays.
    Uses hooks on q_proj to capture raw Q.
    """
    model.eval()
    num_layers = model.config.num_hidden_layers
    q_captures = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output is [B, T, num_heads * head_dim]
            q_captures[layer_idx] = output.detach()

        return hook_fn

    hooks = []
    for i, layer in enumerate(model.model.layers):
        h = layer.self_attn.q_proj.register_forward_hook(make_hook(i))
        hooks.append(h)

    try:
        # Prefill
        ids = input_ids[:, :prefix_length].to(device)
        with torch.no_grad():
            out = model(ids, use_cache=True)
        past = out.past_key_values

        q_vectors = []
        generated_ids = ids.clone()

        for step in range(max_new_tokens):
            cur_pos = prefix_length + step
            q_captures.clear()

            if step == 0:
                # Prefill Q was already captured; take last token Q
                pass
            else:
                last_tok = generated_ids[:, -1:].to(device)
                position_ids = torch.tensor(
                    [[cur_pos - 1]], device=device, dtype=torch.long
                )
                with torch.no_grad():
                    out = model(
                        last_tok,
                        past_key_values=past,
                        use_cache=True,
                        position_ids=position_ids,
                    )
                past = out.past_key_values

            # Greedy decode
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Collect Q: per-layer, take mean over heads for the last token
            head_dim = model.config.hidden_size // model.config.num_attention_heads
            num_heads = model.config.num_attention_heads
            layer_qs = []
            for l_idx in range(num_layers):
                if l_idx in q_captures:
                    q = q_captures[l_idx]  # [B, T, hidden]
                    q_last = q[0, -1, :]  # [hidden]
                    # Reshape to [num_heads, head_dim] and mean
                    q_heads = q_last.view(num_heads, head_dim)
                    q_mean = q_heads.mean(dim=0)  # [head_dim]
                    layer_qs.append(q_mean.cpu().float().numpy())
                else:
                    layer_qs.append(np.zeros(head_dim))

            q_vectors.append(np.stack(layer_qs, axis=0))  # [num_layers, head_dim]
    finally:
        for h in hooks:
            h.remove()

    return q_vectors


def traces_to_serializable(traces: list[DecodeStep]) -> list[dict]:
    """Convert trace list to JSON-serializable format."""
    result = []
    for t in traces:
        result.append(
            {
                "request_id": t.request_id,
                "decode_step": t.decode_step,
                "generated_token_id": t.generated_token_id,
                "baseline_latency_us": t.baseline_latency_us,
                "attn_mass_per_block": t.attn_mass_per_block,
                "top_k_blocks": t.top_k_blocks,
                "needed_blocks_mass": t.needed_blocks_mass,
                "needed_blocks_topk": t.needed_blocks_topk,
                "reuse_distance": t.reuse_distance,
                "prefix_block_count": t.prefix_block_count,
            }
        )
    return result

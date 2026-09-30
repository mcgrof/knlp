# SPDX-License-Identifier: GPL-2.0
"""Measure the leave-one-block-out cost of a KRI-D-kv prefix prior.

The prior removes one prefix block at a time, replays the final token of a
fixed probe against the remaining mapped cache, and sums its hidden-state L2
change over receiver layers.  This is a cost and convention probe: callers
can time a small, fixed block sample and forecast the full bake before deciding
whether to compute every block.
"""

from __future__ import annotations

import time

import torch

from .published.capture import cache_layers, make_cache


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def measure_kri_d_kv_blocks(
    model,
    keys,
    values,
    probe_ids,
    *,
    prefix_tokens,
    block_size,
    blocks,
):
    """Measure sampled KRI-D-kv ablations on an already mapped cache.

    ``keys`` and ``values`` remain unchanged.  ``probe_ids`` is independent of
    evaluation answers.  Each sample includes building the ablated cache and
    the one-token receiver forward; model loading, mapping and the reference
    probe are reported separately.
    """
    if len(keys) != len(values) or not keys:
        raise ValueError("keys and values must contain the same nonzero layers")
    total_prefix = int(keys[0].shape[2])
    if total_prefix != int(prefix_tokens):
        raise ValueError(
            f"cache has {total_prefix} prefix tokens, expected {prefix_tokens}"
        )
    if probe_ids.dim() != 2 or probe_ids.shape[0] != 1 or probe_ids.shape[1] < 1:
        raise ValueError("probe_ids must have shape [1, one_or_more_tokens]")
    n_blocks = prefix_tokens // block_size
    blocks = [int(block) for block in blocks]
    if len(set(blocks)) != len(blocks):
        raise ValueError("sample block indices repeat")
    if any(block < 0 or block >= n_blocks for block in blocks):
        raise ValueError(f"sample block is outside 0..{n_blocks - 1}")

    device = keys[0].device
    probe_ids = probe_ids.to(device)
    _sync(device)
    reference_start = time.perf_counter()
    probe = model(
        input_ids=probe_ids,
        past_key_values=make_cache(zip(keys, values)),
        use_cache=True,
        output_hidden_states=True,
        logits_to_keep=1,
    )
    extended = cache_layers(probe.past_key_values)
    total = prefix_tokens + probe_ids.shape[1]
    last_position = total - 1
    last_id = probe_ids[:, -1:]

    # Re-run the last probe token against the full prefix+earlier-probe cache,
    # so the reference and every ablation execute the same one-token path.
    full_pairs = [(k[:, :, :-1], v[:, :, :-1]) for k, v in extended]
    reference = model(
        input_ids=last_id,
        past_key_values=make_cache(full_pairs),
        position_ids=torch.tensor([[last_position]], device=device),
        cache_position=torch.tensor([total - 1], device=device),
        use_cache=True,
        output_hidden_states=True,
        logits_to_keep=1,
    )
    hidden_reference = [h[:, -1].float() for h in reference.hidden_states]
    _sync(device)
    reference_seconds = time.perf_counter() - reference_start

    all_positions = torch.arange(total, device=device)
    samples = []
    for block in blocks:
        keep = torch.ones(total, dtype=torch.bool, device=device)
        lo = block * block_size
        keep[lo : lo + block_size] = False
        keep[-1] = False
        indices = all_positions[keep]
        _sync(device)
        start = time.perf_counter()
        ablated_pairs = [
            (
                k.index_select(2, indices).contiguous(),
                v.index_select(2, indices).contiguous(),
            )
            for k, v in extended
        ]
        out = model(
            input_ids=last_id,
            past_key_values=make_cache(ablated_pairs),
            position_ids=torch.tensor([[last_position]], device=device),
            cache_position=torch.tensor([indices.numel()], device=device),
            use_cache=True,
            output_hidden_states=True,
            logits_to_keep=1,
        )
        score = float(
            torch.stack(
                [
                    (base - changed[:, -1].float()).norm()
                    for base, changed in zip(hidden_reference, out.hidden_states)
                ]
            ).sum()
        )
        _sync(device)
        samples.append(
            {
                "block": block,
                "seconds": time.perf_counter() - start,
                "hidden_l2_sum": score,
                "retained_cache_tokens": int(indices.numel()),
            }
        )
        del ablated_pairs, out

    return {
        "contract": "kri_d_kv_sum_leave_one_prefix_block_out_v1",
        "prefix_tokens": int(prefix_tokens),
        "probe_tokens": int(probe_ids.shape[1]),
        "block_size": int(block_size),
        "total_blocks": int(n_blocks),
        "sampled_blocks": blocks,
        "reference_seconds": reference_seconds,
        "samples": samples,
    }

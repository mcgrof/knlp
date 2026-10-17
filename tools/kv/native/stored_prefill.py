# SPDX-License-Identifier: MIT
"""Sampled independent causal attention over stored K16/V8 values.

FlashInfer emits base-2 LSE. Preserve the serving path's FP16 output
rounding before applying V scale. No production attention is replaced.
"""

import math

import torch


def reference(query, key, value, positions, *, sm_scale, k_scale, v_scale):
    """One fixed KV head, with its grouped query heads at causal positions."""
    logits = torch.einsum("qhd,td->qht", query.float(), key.float())
    logits *= sm_scale * k_scale
    allowed = torch.arange(len(key), device=query.device)[None, :] <= positions[:, None]
    logits.masked_fill_(~allowed[:, None, :], -torch.inf)
    output = torch.einsum("qht,td->qhd", logits.softmax(-1), value.float())
    output = (output.to(query.dtype).float() * v_scale).to(query.dtype)
    lse = torch.logsumexp(logits, -1) / math.log(2)
    return output, lse


def check(
    workspace, query, cache, source_indices, page_plan, actual, actual_lse, kwargs, path
):
    """Retain every sampled operand before checking the unchanged 1e-3 limits."""
    from .compact import full_comparison

    rows = []
    page_start = query_start = 0
    group = query.shape[1] // cache[0].shape[2]
    # Force an independent FP32 matmul, without changing production settings.
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for length, page_end in zip(page_plan.query_lengths, page_plan.page_indptr[1:]):
            ids = source_indices[page_start:page_end].long()
            key = cache[0].index_select(0, ids)[:, :, 0].flatten(0, 1)[:length]
            # Keep genuine FP8 bytes in the retained reference input.
            value = (
                cache[1]
                .view(torch.uint8)
                .index_select(0, ids)
                .view(cache[1].dtype)[:, :, 0]
                .flatten(0, 1)[:length]
            )
            positions = torch.tensor(
                sorted({0, length // 2, length - 1}), device=query.device
            )
            selected = query.index_select(0, query_start + positions)[:, :group]
            observed = actual.index_select(0, query_start + positions)[:, :group]
            observed_lse = actual_lse.index_select(0, query_start + positions)[
                :, :group
            ]
            scales = dict(
                sm_scale=workspace.wrapper._sm_scale,
                k_scale=kwargs.get("k_scale", 1.0),
                v_scale=kwargs.get("v_scale", 1.0),
            )
            expected, lse = reference(selected, key, value, positions, **scales)
            rows.append(
                dict(
                    query=selected.cpu(),
                    key=key.cpu(),
                    value=value.cpu(),
                    positions=positions.cpu(),
                    observed=observed.cpu(),
                    observed_lse=observed_lse.cpu(),
                    reference=expected.cpu(),
                    reference_lse=lse.cpu(),
                    source_ids=ids.cpu(),
                    scales=scales,
                )
            )
            page_start, query_start = page_end, query_start + length
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    torch.save(rows, path)
    outputs = [full_comparison(r["observed"], r["reference"]) for r in rows]
    lses = [full_comparison(r["observed_lse"], r["reference_lse"]) for r in rows]
    return dict(
        passed=all(x["passed"] for x in outputs + lses),
        output=outputs,
        lse=lses,
        scope="KV head zero and its grouped Q heads; first/middle/last causal positions; every routed layer",
        operands=path.name,
    )

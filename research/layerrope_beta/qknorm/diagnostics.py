# SPDX-License-Identifier: MIT
"""Attention diagnostics run offline on a frozen validation batch."""

import math
import torch
from research.tinystories_depth.kv_cache import CachePolicy, install


@torch.no_grad()
def measure(model, ids, device):
    records = []

    def observe(layer):
        def collect(raw_q, raw_k, q, k, gain, offset):
            with torch.autocast(device_type=device, enabled=False):
                scale = gain.float() if gain is not None else 1 / math.sqrt(q.shape[-1])
                scores = (q.float() @ k.float().transpose(-2, -1)) * scale
                allowed = torch.arange(k.shape[-2], device=q.device)[None, :] <= (
                    offset + torch.arange(q.shape[-2], device=q.device)[:, None]
                )
                minimum = scores.masked_fill(~allowed, float("inf")).amin(-1)
                maximum = scores.masked_fill(~allowed, -float("inf")).amax(-1)
                probability = scores.masked_fill(~allowed, -float("inf")).softmax(-1)
                entropy = -(probability * probability.clamp_min(1e-30).log()).sum(-1)
                count = allowed.sum(-1)
                normalized = entropy / count.clamp_min(2).float().log()
                eligible = count >= 16
                for head in range(q.shape[1]):
                    records.append(
                        dict(
                            layer=layer,
                            head=head,
                            raw_q_norm_mean=raw_q.float()
                            .norm(dim=-1)[:, head]
                            .mean()
                            .item(),
                            raw_k_norm_mean=raw_k.float()
                            .norm(dim=-1)[:, head]
                            .mean()
                            .item(),
                            raw_q_norm_max=raw_q.float()
                            .norm(dim=-1)[:, head]
                            .max()
                            .item(),
                            raw_k_norm_max=raw_k.float()
                            .norm(dim=-1)[:, head]
                            .max()
                            .item(),
                            causal_logit_spread_max=(maximum - minimum)[:, head]
                            .max()
                            .item(),
                            normalized_entropy_mean=normalized[:, head, eligible]
                            .mean()
                            .item(),
                            maximum_probability_gt_099=int(
                                (probability[:, head].amax(-1)[:, eligible] > 0.99)
                                .sum()
                                .item()
                            ),
                            normalized_entropy_lt_01=int(
                                (normalized[:, head, eligible] < 0.1).sum().item()
                            ),
                            eligible_queries=int(eligible.sum()) * q.shape[0],
                            gain=None if gain is None else gain.item(),
                            gain_sign=None if gain is None else int(gain.sign().item()),
                            gain_magnitude=None if gain is None else gain.abs().item(),
                        )
                    )

        return collect

    blocks = model.lm.model.layers
    for i, block in enumerate(blocks):
        block.self_attn._qk_observer = observe(i)
    try:
        with install(model, CachePolicy()), torch.autocast(
            device_type=device, dtype=torch.bfloat16
        ):
            model(ids[:2])
    finally:
        for block in blocks:
            del block.self_attn._qk_observer
    rope = model.lm.model.layerrope
    gains = (
        {}
        if rope is None
        else {n: p.detach().float().cpu().tolist() for n, p in rope.named_parameters()}
    )
    return dict(attention=records, layerrope_parameters=gains, optimizer_updates=0)

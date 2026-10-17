# SPDX-License-Identifier: MIT
"""Matched clean/cache/weight endpoint paths with explicit restoration checks."""

import time
import numpy as np
import torch
from research.layerrope_beta.measure import intervention, linears
from research.layerrope_beta.time_budget.protocol import freeze
from research.layerrope_beta.time_budget.routing import RoutingObservation
from research.mo_variants.stage_c_common import sha
from research.tinystories_depth.kv_cache import (
    evaluate,
    CachePolicy,
    install,
    bytes_per_token,
)
from research.tinystories_depth.train import state_hash


@torch.no_grad()
def qualify_paths(model, tokens, device, window=256, batch_size=32):
    if len(tokens) < batch_size * window + 1:
        raise ValueError("Parity requires the declared full evaluation batch")
    ids = torch.as_tensor(
        np.array(tokens[: batch_size * window], dtype=np.int64).reshape(
            batch_size, window
        ),
        device=device,
    )
    before = state_hash(model.state_dict())
    model.eval()

    def forward():
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            return model(ids)

    ref = forward()
    model.observer = lambda *args: None
    try:
        manual = forward()
    finally:
        model.observer = None
    policy = CachePolicy(16, 16, "symmetric")
    with install(model, policy):
        adapter = forward()
    observer = RoutingObservation()
    model.routing_observation = observer
    try:
        observed = forward()
    finally:
        model.routing_observation = None
    routing = observer.report(2 * model.depth if model.routes is not None else 0)
    if (
        not torch.equal(ref, manual)
        or not torch.equal(ref, adapter)
        or not torch.equal(ref, observed)
    ):
        raise RuntimeError("Same-shape observer/manual/cache identity path mismatch")
    if before != state_hash(model.state_dict()):
        raise RuntimeError("Qualification changed checkpoint")
    return dict(
        batch_size=batch_size,
        window=window,
        manual_logits_equal=True,
        identity_cache_logits_equal=True,
        routing_logits_equal=True,
        routing=routing,
        model_unchanged=True,
    )


@torch.no_grad()
def endpoint(model, tokens, device, *, supporting=False, window=256, batch_size=32):
    protocol = freeze()
    before = state_hash(model.state_dict())
    model.eval()
    parity = qualify_paths(model, tokens, device, window, batch_size)
    menu = protocol["cache_cases"]
    if supporting:
        menu = [x for x in menu if x["name"] in protocol["supporting_cache_names"]]
    cache = []
    weights = []
    clean = None
    for case in menu:
        result = evaluate(
            model, tokens, case, device, window=window, batch_size=batch_size
        )
        if case["name"] == "clean":
            clean = result
        result.update(
            case,
            storage=bytes_per_token(case, depth=model.depth),
            delta_ce=result["ce"] - clean["ce"],
        )
        cache.append(result)
    if not supporting:
        for policy in protocol["weight_policies"]:
            # evaluate() enters/exits autocast per batch; in-place weight changes
            # cannot reuse casts cached by a long-lived outer autocast scope.
            with intervention(
                model, **{k: v for k, v in policy.items() if k != "name"}
            ):
                result = evaluate(
                    model,
                    tokens,
                    protocol["cache_cases"][0],
                    device,
                    window=window,
                    batch_size=batch_size,
                )
            result.update(policy, delta_ce=result["ce"] - clean["ce"])
            if policy.get("weight_bits"):
                chosen = linears(model, policy.get("scope", "blocks"))
                n = sum(m.weight.numel() for _, m in chosen)
                rows = sum(m.weight.numel() // m.weight.shape[-1] for _, m in chosen)
                total = sum(p.numel() for p in model.parameters())
                result.update(
                    covered_parameters=n,
                    total_parameters=total,
                    ideal_weight_bytes=n * policy["weight_bits"] / 8
                    + rows * 4
                    + (total - n) * 4,
                    unquantized_parameter_bytes=(total - n) * 4,
                    scale_metadata_bytes=rows * 4,
                )
            if state_hash(model.state_dict()) != before:
                raise RuntimeError("Intervention did not restore weights")
            weights.append(result)
    restored = evaluate(
        model,
        tokens,
        protocol["cache_cases"][0],
        device,
        window=window,
        batch_size=batch_size,
    )
    if (
        restored["window_ce"] != clean["window_ce"]
        or state_hash(model.state_dict()) != before
    ):
        raise RuntimeError("Clean restoration changed checkpoint or same-shape CE")
    return dict(
        clean=clean,
        cache=cache,
        weight_activation_residual=weights,
        parity=parity,
        restored_clean=restored,
        weights_unchanged=True,
        evaluation_shape=dict(
            window=window,
            batch_size=batch_size,
            order="adjacent fixed windows",
            tail="retain partial batch",
            arithmetic="BF16 matrices, FP32 CE, FP64 aggregation",
        ),
        scope="Numerical Q/DQ only; no packed-memory or hardware-speed claim",
    )

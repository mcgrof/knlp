# SPDX-License-Identifier: MIT
"""Count backing allocations once and reconcile their occupied byte intervals."""


def reconcile(tensors):
    allocations = {}
    payload = 0
    for tensor in tensors:
        key = (tensor["device"], tensor["storage_pointer"])
        entry = allocations.setdefault(
            key, {"bytes": tensor["storage_bytes"], "intervals": []}
        )
        if entry["bytes"] != tensor["storage_bytes"] or not tensor["contiguous"]:
            raise ValueError(
                "inconsistent storage identity or noncontiguous cache view"
            )
        begin = tensor["tensor_pointer"] - tensor["storage_pointer"]
        end = begin + tensor["logical_bytes"]
        if begin < 0 or end > entry["bytes"]:
            raise ValueError("cache view escapes its backing allocation")
        entry["intervals"].append((begin, end))
        payload += tensor["logical_bytes"]
    padding = 0
    for entry in allocations.values():
        previous = 0
        for begin, end in sorted(entry["intervals"]):
            if begin < previous:
                raise ValueError("cache planes/layers overlap occupied bytes")
            padding += begin - previous
            previous = end
        padding += entry["bytes"] - previous
    return {
        "unique_allocations": len(allocations),
        "backing_bytes": sum(e["bytes"] for e in allocations.values()),
        "occupied_payload_bytes": payload,
        "unoccupied_backing_bytes": padding,
    }


def verify_attention_reads(layers, events, expected_types):
    """Require real prefill/decode reads from every recorded model layer."""
    backing = {
        layer["name"]: {
            (t["device"], t["storage_pointer"]): t["storage_bytes"]
            for t in layer["cache"]
        }
        for layer in layers
    }
    wrappers = {
        "BatchPrefillWithPagedKVCacheWrapper",
        "BatchDecodeWithPagedKVCacheWrapper",
    }
    wanted = {(wrapper, name) for wrapper in wrappers for name in backing}
    seen = set()
    for event in events:
        key = (event["wrapper"], event["layer"])
        if key not in wanted or key in seen:
            raise ValueError("unexpected or duplicate attention read")
        if {t["dtype"] for t in event["cache"]} != expected_types:
            raise ValueError("attention read physical dtype mismatch")
        for tensor in event["cache"]:
            identity = (tensor["device"], tensor["storage_pointer"])
            if backing[event["layer"]].get(identity) != tensor["storage_bytes"]:
                raise ValueError("attention read storage differs from model cache")
        seen.add(key)
    if seen != wanted:
        raise ValueError("attention read layer/phase coverage is incomplete")
    return {"layers": len(backing), "layer_phase_reads": len(seen)}

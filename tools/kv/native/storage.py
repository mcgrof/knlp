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

# SPDX-License-Identifier: MIT
"""Deterministic request schedules and comparison shared by storage backends."""

import hashlib
import json
import math
import random


def schedule(rows, requests, seed, pattern):
    if rows < 1 or requests < 1:
        raise ValueError("rows and requests must be positive")
    rng = random.Random(seed)
    if pattern == "sequential":
        return [i % rows for i in range(requests)]
    if pattern == "random":
        return [rng.randrange(rows) for _ in range(requests)]
    if pattern == "hot":
        hot = max(1, rows // 10)
        return [
            rng.randrange(hot if rng.random() < 0.9 else rows) for _ in range(requests)
        ]
    if pattern == "shuffle":
        result = []
        while len(result) < requests:
            epoch = list(range(rows))
            rng.shuffle(epoch)
            result.extend(epoch)
        return result[:requests]
    raise ValueError(f"unknown request pattern: {pattern}")


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def request_record(question, bank, row, bank_row, request_id, dependencies):
    """Bind a request without requiring raw documents in the evaluation rows."""
    collection = int(bank["collection_id"])
    return {
        "request_id": request_id,
        "row": row,
        "bank_row": bank_row,
        "collection_id": collection,
        "input_digest": fingerprint(
            {
                "question": question,
                "collection": collection,
                "dependencies": dependencies,
            }
        ),
    }


def validate_trace(records, rows):
    if not isinstance(records, list) or not records:
        raise ValueError("trace must be a nonempty request list")
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("invalid trace request")
        key, row = record.get("request_id"), record.get("row")
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError("missing or duplicate trace request ID")
        if type(row) is not int or not 0 <= row < rows:
            raise ValueError("trace row is outside the selected split")
        seen.add(key)


def percentile(values, fraction):
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(fraction * len(values)) - 1)]


def compare(reference, observed):
    """Compare by immutable request ID, rejecting missing and duplicate results."""

    def indexed(records):
        result = {}
        for record in records:
            key = record["request_id"]
            if key in result:
                raise ValueError(f"duplicate request ID: {key}")
            result[key] = record
        return result

    left, right = indexed(reference), indexed(observed)
    common = left.keys() & right.keys()
    changed = [
        key for key in common if left[key]["prediction"] != right[key]["prediction"]
    ]
    bad_inputs = [
        key for key in common if left[key]["input_digest"] != right[key]["input_digest"]
    ]
    return {
        "missing": sorted(left.keys() - right.keys()),
        "unexpected": sorted(right.keys() - left.keys()),
        "changed_predictions": sorted(changed),
        "changed_inputs": sorted(bad_inputs),
        "exact": left.keys() == right.keys() and not changed and not bad_inputs,
    }

# SPDX-License-Identifier: MIT
"""Bounded post-client cleanup, excluded from the fixed goodput denominator."""

import time

from tools.kv.native.serving_pilot import rpc


def verified(rows):
    if not rows:
        raise ValueError("Missing post-client scheduler observations")
    previous = None
    for row in rows:
        if len(row["results"]) != 1:
            raise ValueError("Expected one scheduler in the uni executor")
        item = row["results"][0]
        if previous is not None and item["at_ns"] <= previous:
            raise ValueError("Scheduler observation clock did not advance")
        previous = item["at_ns"]
        if any(
            type(item[k]) is not int or item[k] < 0
            for k in ("running", "waiting", "requests")
        ):
            raise ValueError("Invalid scheduler occupancy")
        if not item["scheduler_seen"] and any(
            item[k] for k in ("running", "waiting", "requests")
        ):
            raise ValueError("Unseen scheduler cannot report pending work")
    return (
        len(rows) >= 2
        and all(
            not any(row["results"][0][k] for k in ("running", "waiting", "requests"))
            for row in rows[-2:]
        )
        and rows[-1]["results"][0]["at_ns"] - rows[-2]["results"][0]["at_ns"]
        >= 200_000_000
    )


def wait(rows):
    """Append every observation even if cleanup fails; never retry a request."""
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        rows.append(rpc("study_pending"))
        if verified(rows):
            return
        time.sleep(0.25)
    raise TimeoutError("Worker still has pending requests after post-client cleanup")

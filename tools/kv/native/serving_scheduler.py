# SPDX-License-Identifier: MIT
"""Separate first admissions from both pinned vLLM resumption encodings."""

if not __debug__:
    raise RuntimeError("Evidence audits require Python assertions enabled")


def admissions(rows, input_tokens=None):
    """Yield fresh IDs and resumed IDs, requiring prior recorded preemption.

    The pinned scheduler's V2 runner re-sends resumed requests through
    scheduled_new_reqs; the older runner uses cached resumed_req_ids.
    A repeated ID alone is not evidence of a valid resumption.
    """
    seen, suspended, steps = set(), set(), set()
    for row in rows:
        assert row["batch_id"] not in steps
        steps.add(row["batch_id"])
        new = row["new_request_ids"]
        cached = row["resumed_request_ids"]
        preempted = row["preempted_request_ids"]
        assert len(new) == len(set(new)) and len(cached) == len(set(cached))
        assert len(preempted) == len(set(preempted))
        assert not set(new).intersection(cached)
        for rid in preempted:
            assert rid in seen and rid not in suspended
            assert rid not in row["scheduled_tokens"]
            suspended.add(rid)
        fresh, resent = [], []
        for rid in new:
            assert rid in row["scheduled_tokens"]
            if rid in seen:
                assert rid in suspended, "Repeated scheduler-new ID without preemption"
                suspended.remove(rid)
                resent.append(rid)
            else:
                assert rid not in suspended
                seen.add(rid)
                fresh.append(rid)
                if input_tokens is not None:
                    assert row["scheduled_tokens"][rid] == input_tokens
        for rid in cached:
            assert rid in seen and rid in suspended
            assert rid in row["scheduled_tokens"]
            suspended.remove(rid)
        if input_tokens is not None:
            assert all(
                input_tokens <= row["scheduled_tokens"][rid] < input_tokens + 128
                for rid in [*resent, *cached]
            )
        yield {"fresh": fresh, "resumed_via_new": resent, "resumed_via_cached": cached}

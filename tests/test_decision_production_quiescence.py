# SPDX-License-Identifier: MIT
"""No next cell while an aborted client's worker request is still active."""

import pytest

from tools.kv.decision.production_quiescence import verified


def observation(at, count=0):
    return {
        "results": [
            {
                "at_ns": at,
                "scheduler_seen": True,
                "running": count,
                "waiting": 0,
                "requests": count,
            }
        ]
    }


def test_two_separated_empty_observations_are_required():
    rows = [observation(1, 1), observation(250_000_001)]
    assert not verified(rows)
    rows.append(observation(500_000_001))
    assert verified(rows)


@pytest.mark.parametrize("defect", ["backwards", "negative", "multiple", "unseen"])
def test_corrupt_scheduler_receipt_is_not_idle(defect):
    rows = [observation(1), observation(250_000_001)]
    row = rows[-1]["results"][0]
    if defect == "backwards":
        row["at_ns"] = 1
    elif defect == "negative":
        row["running"] = -1
    elif defect == "multiple":
        rows[-1]["results"].append(dict(row))
    else:
        row.update(scheduler_seen=False, requests=1)
    with pytest.raises(ValueError):
        verified(rows)

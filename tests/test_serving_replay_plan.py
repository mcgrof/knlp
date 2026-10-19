"""A historical design cannot admit changed or incomplete qualification."""

import copy
import pytest
from tools.kv.native.serving_replay_plan import ARMS, qualified


def records():
    d = dict(
        format="fp16",
        length=8192,
        blocks=10,
        arms=list(ARMS),
        output_tokens=128,
        source_sha="a" * 40,
        persistent_cache_bytes=40 * 2**30,
    )
    p = {
        arm: dict(
            production_passed=True,
            source_sha=d["source_sha"],
            format="fp16",
            arm=arm,
            cache_bytes=40 * 2**30,
            protocol_sha256="b" * 64,
        )
        for arm in ARMS
    }
    q = dict(
        format="fp16",
        length=8192,
        quality_passed=True,
        controlled_parity_passed=True,
        audit={arm: dict(protocol_sha256="b" * 64) for arm in ARMS},
    )
    return d, p, q


def test_reject_failed_quality_or_missing_arm():
    d, p, q = records()
    qualified(d, p, q)
    for key in ("quality_passed", "controlled_parity_passed"):
        with pytest.raises(ValueError, match="must both pass"):
            qualified(d, p, {**q, key: False})
    for arm in ARMS:
        with pytest.raises(ValueError, match="all three arms"):
            qualified(d, {k: v for k, v in p.items() if k != arm}, q)


@pytest.mark.parametrize(
    "key,value",
    [
        ("source_sha", "c" * 40),
        ("cache_bytes", 32 * 2**30),
        ("protocol_sha256", "c" * 64),
        ("production_passed", False),
        ("format", "awq"),
        ("arm", "candidate"),
    ],
)
def test_reject_changed_production_identity(key, value):
    d, p, q = records()
    p = copy.deepcopy(p)
    p["native_fa3"][key] = value
    with pytest.raises(ValueError, match="identities differ"):
        qualified(d, p, q)

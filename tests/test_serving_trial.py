"""Fail closed when a prospective trial drifts from audited qualification."""

import copy
import hashlib
import json

import pytest

from tools.kv.native.serving_trial import CHECKS, read_frozen, validate


def records():
    scope = dict(
        format="awq",
        arm="candidate",
        length=8192,
        cache_bytes=48 * 2**30,
        backend_sources={"vllm": "a" * 40},
    )
    plan = dict(
        **scope,
        stage="confirmation",
        output_tokens=128,
        retry_count=0,
        warmup_repetitions=2,
        settle_seconds=2,
        cells=[{"label": "near"}]
    )
    qualified = dict(
        **scope,
        passed=True,
        checks=dict.fromkeys(CHECKS, True),
        implementation_sha256={"server.py": "b" * 64}
    )
    return plan, qualified


def test_qualification_and_scope_must_all_match():
    plan, qualified = records()
    implementation = qualified["implementation_sha256"]
    validate(plan, qualified, implementation)
    for check in CHECKS:
        broken = copy.deepcopy(qualified)
        broken["checks"][check] = False
        with pytest.raises(ValueError, match="incomplete"):
            validate(plan, broken, implementation)
    for name, value in [
        ("format", "fp16"),
        ("arm", "native_fa3"),
        ("length", 1024),
        ("cache_bytes", 32 * 2**30),
        ("backend_sources", {"vllm": "c" * 40}),
    ]:
        with pytest.raises(ValueError, match="scope"):
            validate({**plan, name: value}, qualified, implementation)
    with pytest.raises(ValueError, match="implementation changed"):
        validate(plan, qualified, {"server.py": "d" * 64})
    with pytest.raises(ValueError, match="retried"):
        validate({**plan, "retry_count": 1}, qualified, implementation)


def test_frozen_inputs_cannot_change_or_escape(tmp_path):
    raw = json.dumps({"requests": [1, 2]}).encode()
    (tmp_path / "trace.json").write_bytes(raw)
    item = {"file": "trace.json", "sha256": hashlib.sha256(raw).hexdigest()}
    assert read_frozen(tmp_path, item) == {"requests": [1, 2]}
    (tmp_path / "trace.json").write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="changed"):
        read_frozen(tmp_path, item)
    with pytest.raises(ValueError, match="escapes"):
        read_frozen(tmp_path, {**item, "file": "../trace.json"})

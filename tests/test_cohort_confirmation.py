"""A failed or absent block must never become a reduced-set gain claim."""

import json

import pytest

from tools.kv.native import cohort_confirmation as audit


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def study(tmp_path, monkeypatch):
    root, plan = tmp_path / "captures", tmp_path / "frozen"
    arms = ["candidate", "native_fi", "native_fa3"]
    save(
        plan / "PLAN.json",
        {
            "stage": "confirmation",
            "blocks": 10,
            "orders": [arms] * 10,
            "primary_family": [
                {"baseline": arm, "median_gain_screen": 0.01} for arm in arms[1:]
            ],
            "performance_scope": "synthetic gate check",
        },
    )
    for block in range(10):
        for index, arm in enumerate(arms):
            name = f"test-b{block:02}-{arm}"
            save(root / name / "DONE.json", {})
            job = {"argv": ["capture", name]}
            save(plan / "jobs" / ("190-" + name + ".json"), job)
            save(root / "jobs" / ("190-" + name + ".json"), job)
            save(
                root / "commands" / ("000-" + name + ".json"),
                {**job, "returncode": 0},
            )

    def capture(base, planpath, arm):
        block = int(base.name.split("-")[1][1:])
        start = (block * 3 + arms.index(arm)) * 10 + 1
        return {
            "start_monotonic_ns": start,
            "end_monotonic_ns": start + 1,
            "output_tokens_per_second": 103 if arm == "candidate" else 100,
        }

    monkeypatch.setattr(audit, "arm_capture", capture)
    return root, plan


def test_full_fixed_family(study):
    result = audit.run(*study, "test")
    assert result["all_blocks_valid_and_complete"] and result["complete"]
    assert len(result["primary_family"]) == 2
    assert not result["failed"] and not result["pending"]


def test_failed_capture_has_no_reduced_set_inference(study):
    root, plan = study
    name = "test-b04-native_fi"
    (root / name / "DONE.json").unlink()
    save(root / "jobs" / ("190-" + name + "-result.json"), {"passed": False})
    result = audit.run(root, plan, "test")
    assert result["complete"] and not result["all_blocks_valid_and_complete"]
    assert result["primary_family"] == [] and len(result["failed"]) == 1
    assert result["failed"][0]["block"] == 4


def test_pending_capture_cannot_close(study):
    root, plan = study
    (root / "test-b09-native_fa3/DONE.json").unlink()
    result = audit.run(root, plan, "test")
    assert not result["complete"] and not result["all_blocks_valid_and_complete"]
    assert result["primary_family"] == []
    assert result["pending"] == [{"block": 9, "arm": "native_fa3"}]


def test_missing_native_control_rejected(study):
    root, plan = study
    value = audit.read(plan / "PLAN.json")
    value["orders"][3] = ["candidate", "native_fi", "native_fi"]
    save(plan / "PLAN.json", value)
    with pytest.raises(AssertionError):
        audit.run(root, plan, "test")

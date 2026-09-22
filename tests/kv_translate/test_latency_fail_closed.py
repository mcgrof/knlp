# SPDX-License-Identifier: GPL-2.0
"""The timing run must refuse to start when its prerequisite is not met.

An exit code of zero and a line reading ALL_OK demonstrate that a driver
finished, not that it would have stopped had something failed. These tests
inject the failures instead of assuming they are handled: a missing receipt, a
malformed one, one from the wrong contract, and one that records its own
failure. Each must prevent timing rather than be stepped over.

They run on the CPU. The prerequisite gate is deliberately placed before the
device check so that it can be exercised without a GPU, which is also what
lets it be tested at all.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "research" / "kv_translate" / "time_first_next_token.py"
DRIVER = ROOT / "research" / "kv_translate" / "drive_latency.sh"


def _run(receipt, tmp_path, extra=()):
    return subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--artifact",
            str(tmp_path / "nonexistent.pt"),
            "--fixtures",
            str(tmp_path / "nonexistent_fixtures.pt"),
            "--qualification",
            str(receipt),
            "--out",
            str(tmp_path / "out.json"),
            *extra,
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )


def test_missing_receipt_prevents_timing(tmp_path):
    r = _run(tmp_path / "absent.json", tmp_path)
    assert r.returncode != 0
    assert "PREREQUISITE" in r.stdout + r.stderr
    assert not (tmp_path / "out.json").exists()


def test_malformed_receipt_prevents_timing(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    r = _run(p, tmp_path)
    assert r.returncode != 0
    assert "unreadable" in r.stdout + r.stderr
    assert not (tmp_path / "out.json").exists()


def _full_receipt(sha="de" * 32, **over):
    r = {
        "contract": "saved_operator_v1",
        "passed": True,
        "gpu": "cpu",
        "operator_dtype": "float32",
        "serving_dtype": "bfloat16",
        "tf32_allowed": False,
        "contract_limits": {
            "operator_matches_reference": 1e-6,
            "operator_is_deterministic": 0.0,
            "serving_cast_within_one_step": 1.001,
        },
        "checks": {
            "artifact": {"joint_weight_sha256": sha},
            "operator_matches_reference": {"measured": 1e-7, "passed": True},
            "operator_is_deterministic": {"measured": 0.0, "passed": True},
            "serving_cast_within_one_step": {"measured": 0.5, "passed": True},
        },
    }
    r.update(over)
    return r


def test_receipt_without_required_fields_prevents_timing(tmp_path):
    p = tmp_path / "thin.json"
    p.write_text(json.dumps({"contract": "saved_operator_v1"}))
    r = _run(p, tmp_path)
    assert r.returncode != 0
    assert "has no" in r.stdout + r.stderr
    assert not (tmp_path / "out.json").exists()


def test_receipt_with_empty_checks_is_rejected_before_any_model_loads(tmp_path):
    """The expensive version of this failure is the one worth preventing.

    A receipt that passes a shallow gate and only breaks when the artifact
    hash is dereferenced does so after both models are resident on a rented
    card, spending allocation time to learn something knowable beforehand.
    """
    p = tmp_path / "empty_checks.json"
    p.write_text(
        json.dumps(
            {
                "contract": "saved_operator_v1",
                "passed": True,
                "checks": {},
                "contract_limits": {
                    "operator_matches_reference": 1e-6,
                    "operator_is_deterministic": 0.0,
                    "serving_cast_within_one_step": 1.001,
                },
            }
        )
    )
    r = _run(p, tmp_path)
    out = r.stdout + r.stderr
    assert r.returncode != 0
    assert "joint_weight_sha256" in out
    assert "Traceback" not in out, "failed by exception rather than by refusing"


def test_receipt_earned_under_looser_limits_is_rejected(tmp_path):
    p = tmp_path / "loose.json"
    rec = _full_receipt()
    rec["contract_limits"]["operator_matches_reference"] = 1e-3
    p.write_text(json.dumps(rec))
    r = _run(p, tmp_path)
    assert r.returncode != 0
    assert "looser than" in r.stdout + r.stderr


def test_receipt_with_a_failing_subcheck_is_rejected(tmp_path):
    """`passed: true` beside a check that did not pass is a contradiction."""
    p = tmp_path / "subfail.json"
    rec = _full_receipt()
    rec["checks"]["operator_is_deterministic"]["passed"] = False
    p.write_text(json.dumps(rec))
    r = _run(p, tmp_path)
    assert r.returncode != 0
    assert "did not pass" in r.stdout + r.stderr


def test_receipt_earned_with_tf32_is_rejected(tmp_path):
    p = tmp_path / "tf32.json"
    p.write_text(json.dumps(_full_receipt(tf32_allowed=True)))
    r = _run(p, tmp_path)
    assert r.returncode != 0
    assert "TF32" in r.stdout + r.stderr


def test_wrong_contract_prevents_timing(tmp_path):
    p = tmp_path / "other.json"
    p.write_text(
        json.dumps({"contract": "something_else", "passed": True, "checks": {}})
    )
    r = _run(p, tmp_path)
    assert r.returncode != 0
    assert "wrong contract" in r.stdout + r.stderr


def test_failed_qualification_prevents_timing(tmp_path):
    """The case that actually happened once, in the other direction."""
    p = tmp_path / "failed.json"
    p.write_text(
        json.dumps({"contract": "saved_operator_v1", "passed": False, "checks": {}})
    )
    r = _run(p, tmp_path)
    assert r.returncode != 0
    assert "did not pass" in r.stdout + r.stderr
    assert not (tmp_path / "out.json").exists()


def test_passing_receipt_gets_past_the_gate(tmp_path):
    """A good receipt must not be rejected by the gate itself.

    The run still fails afterwards, on the absent artifact or the absent GPU,
    but it must fail for that reason and not at the prerequisite. Otherwise
    the gate would pass every test above for the wrong reason.
    """
    p = tmp_path / "ok.json"
    p.write_text(json.dumps(_full_receipt()))
    r = _run(p, tmp_path)
    out = r.stdout + r.stderr
    assert "qualification receipt accepted" in out
    assert "PREREQUISITE: qualification" not in out


def test_driver_stops_at_the_first_failing_stage(tmp_path):
    script = tmp_path / "run.sh"
    script.write_text(
        f'O="{tmp_path}"\nsource "{DRIVER}"\n'
        "stage first true\n"
        "stage second false\n"
        "stage third true\n"
        'echo "ALL_OK" >> "$O/drive.log"\n'
    )
    r = subprocess.run(["bash", str(script)], capture_output=True, text=True)
    log = (tmp_path / "drive.log").read_text()
    assert r.returncode != 0
    assert "STOP: second failed" in log
    assert "third" not in log, "a stage after the failure ran"
    assert "ALL_OK" not in log
    assert (tmp_path / "FAILED").read_text().strip() == "second"


@pytest.mark.parametrize("failing", ["first", "second"])
def test_driver_failure_is_recorded_wherever_it_lands(tmp_path, failing):
    script = tmp_path / "run.sh"
    body = "\n".join(
        f"stage {n} {'false' if n == failing else 'true'}" for n in ("first", "second")
    )
    script.write_text(f'O="{tmp_path}"\nsource "{DRIVER}"\n{body}\n')
    subprocess.run(["bash", str(script)], capture_output=True, text=True)
    assert (tmp_path / "FAILED").read_text().strip() == failing


def test_the_runner_completes_end_to_end_on_tiny_models(tmp_path):
    """The whole path runs, and its exit status is asserted here.

    This exists because the check it replaces was performed by eye. The dry
    run was invoked in the same shell command as the test suite, its output
    was piped through `tail`, and the reader saw the suite's "81 passed" and
    concluded the run had also succeeded. It had not: a helper the writer had
    deleted raised on the first flush, and the defect reached a rented A100,
    where it cost an allocation after the first of three lengths had already
    been measured.

    A failure that has to be noticed in a log is a failure that will be
    missed. Asserting the exit status is the whole point.
    """
    import shutil
    import sys

    if shutil.which("nvidia-smi") is None and not _has_torch():
        pytest.skip("needs torch")
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "research" / "kv_translate" / "dry_run_latency.py"),
            "--out-dir",
            str(tmp_path / "dry"),
            "--ctx",
            "32",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""},
    )
    assert r.returncode == 0, f"dry run failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
    d = json.loads((tmp_path / "dry" / "tiny_timing.json").read_text())
    assert d["complete"] is True
    assert d["contract"].endswith("_DRY_RUN")
    # The record every partial flush must also carry.
    for key in ("identity", "code", "pinning", "artifact", "fixtures", "boundary"):
        assert key in d, f"{key} missing from the record"


def _has_torch():
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False

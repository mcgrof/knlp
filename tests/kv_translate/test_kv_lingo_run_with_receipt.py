import json
from pathlib import Path
import sys

from research.kv_translate.published.run_with_receipt import run_with_receipt


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_success_writes_complete_receipt(tmp_path):
    path = tmp_path / "receipt.json"
    result = run_with_receipt([sys.executable, "-c", "pass"], path)
    assert result["state"] == "COMPLETE"
    assert result["exit_code"] == 0
    assert load(path) == result


def test_failure_cannot_leave_running_receipt(tmp_path):
    path = tmp_path / "receipt.json"
    result = run_with_receipt([sys.executable, "-c", "raise SystemExit(7)"], path)
    assert result["state"] == "FAILED"
    assert result["exit_code"] == 7
    assert load(path)["state"] == "FAILED"


def test_timeout_writes_terminal_receipt(tmp_path):
    path = tmp_path / "receipt.json"
    result = run_with_receipt(
        [sys.executable, "-c", "import time; time.sleep(1)"],
        path,
        deadline_seconds=0.01,
    )
    assert result["state"] == "TIMED_OUT"
    assert result["exit_code"] == 124
    assert load(path)["state"] == "TIMED_OUT"

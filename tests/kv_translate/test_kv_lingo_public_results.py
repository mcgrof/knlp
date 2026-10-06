import json
from pathlib import Path

from research.kv_translate.published.verify_public_results import summarize


def test_released_results_match_the_frozen_closeout_counts():
    root = Path(__file__).resolve().parents[2]
    value = json.loads((root / "docs/data/kv-lingo-results.json").read_text())
    assert summarize(value) == {
        "checkpoint_pairs": 4,
        "comparisons": 16,
        "passing_comparisons": 7,
        "passing_full_pairs": 0,
        "failed_4b_start_comparisons": 8,
        "passing_8b_start_comparisons": 7,
        "all_late_turn_gates_pass": True,
        "all_unhealthy_excess_gates_pass": True,
        "absolute_unhealthy_rows": 0,
    }

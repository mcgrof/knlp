# SPDX-License-Identifier: GPL-2.0
"""The rule that carries one screen condition forward, checked before it runs.

Written against constructed screens whose right answer is known by
construction. A preference order is only worth having if it is fixed before
the numbers arrive, and only demonstrably fixed if it is tested.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from research.kv_translate.pick_candidate import (
    choose,
    passes,
    r2_conditions,
    r3_conditions,
)
from research.kv_translate.recovery_runner import check_conditions

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFS = {
    "uniform_25": {"repair": "oracle", "selector": "uniform", "blocks": 25},
    "uniform_64": {"repair": "oracle", "selector": "uniform", "blocks": 64},
    "mapped_attn_25": {"repair": "oracle", "selector": "mapped_attn", "blocks": 25},
    "mapped_attn_64": {"repair": "oracle", "selector": "mapped_attn", "blocks": 64},
    "native_attn_25": {"repair": "oracle", "selector": "native_attn", "blocks": 25},
    "native_attn_64": {"repair": "oracle", "selector": "native_attn", "blocks": 64},
}


def screen(cond, net, drop=0.0, unhealthy=0.0, gross=0.0):
    return {
        "condition": cond,
        "net_retrieval_over_unrepaired": net,
        "cloze_drop": drop,
        "unhealthy": unhealthy,
        "gross_repetition_failures": gross,
        "meets_net_retrieval": net >= 2,
        "meets_cloze": drop <= 0.03,
        "unhealthy_not_worse": True,
        "gross_repetition_not_worse": True,
    }


def test_nothing_passing_is_no_signal_not_a_least_bad_choice():
    w, v, _ = choose([screen("uniform_25", 1), screen("mapped_attn_25", 0)], DEFS)
    assert w is None and v == "NO_LOCAL_REPAIR_SIGNAL"


def test_even_coverage_is_carried_when_it_ties_the_best_ranked_selector():
    """A selector that cannot beat free coverage has not earned its cost."""
    w, v, _ = choose([screen("uniform_25", 5), screen("mapped_attn_25", 5)], DEFS)
    assert w["condition"] == "uniform_25" and v == "SCREEN_SIGNAL_UNIFORM"


def test_a_ranked_selector_that_genuinely_wins_is_carried():
    w, v, _ = choose([screen("uniform_25", 3), screen("mapped_attn_25", 7)], DEFS)
    assert w["condition"] == "mapped_attn_25" and v == "SCREEN_SIGNAL_MAPPED_RANKING"


def test_the_smaller_budget_breaks_a_tie_within_a_family():
    w, _, _ = choose([screen("uniform_25", 4), screen("uniform_64", 4)], DEFS)
    assert w["condition"] == "uniform_25"


def test_a_larger_budget_is_carried_only_when_it_actually_scores_more():
    w, _, _ = choose([screen("uniform_25", 4), screen("uniform_64", 9)], DEFS)
    assert w["condition"] == "uniform_64"


def test_an_oracle_only_signal_is_flagged_and_does_not_advance():
    """Ranking that needs the receiver's own cache cannot be carried into a
    cost claim, however well it scores."""
    w, v, why = choose(
        [
            screen("uniform_25", 1),
            screen("mapped_attn_25", 1),
            screen("native_attn_64", 9),
        ],
        DEFS,
    )
    assert w["condition"] == "native_attn_64"
    assert v == "ORACLE_RANKING_ONLY"
    assert "not available at serving time" in why


def test_a_failing_health_check_disqualifies_a_high_scoring_condition():
    s = screen("mapped_attn_64", 20)
    s["unhealthy_not_worse"] = False
    assert not passes(s)
    w, v, _ = choose([s, screen("uniform_25", 3)], DEFS)
    assert w["condition"] == "uniform_25"


def test_a_cloze_regression_disqualifies_it_too():
    s = screen("mapped_attn_64", 20, drop=0.11)
    assert not passes(s)
    w, _, _ = choose([s, screen("uniform_64", 2)], DEFS)
    assert w["condition"] == "uniform_64"


@pytest.mark.parametrize("cond", ["uniform_25", "mapped_attn_64"])
def test_the_emitted_condition_files_are_ones_the_runner_will_accept(cond):
    """The point of emitting them is that nobody hand-edits a matrix on a
    billing card, which only holds if the runner accepts what comes out."""
    w = screen(cond, 5)
    for fn in (r2_conditions, r3_conditions):
        cfg = fn(w, DEFS, 256)
        assert check_conditions(cfg["definitions"], cfg["matrix"]) == []
        assert cfg["carried_from_R1"] == cond
        c = cfg["definitions"]["candidate"]
        assert c["repair"] == "replay", "the carried stage must recompute, not donate"
        assert c["selector"] == DEFS[cond]["selector"]
        assert c["blocks"] == DEFS[cond]["blocks"]


def test_the_gate_stage_carries_a_wrong_source_control_at_the_same_budget():
    cfg = r3_conditions(screen("uniform_25", 5), DEFS, 256)
    w = cfg["definitions"]["candidate|wrong"]
    assert w["source"] == "wrong" and w["repair"] == "replay" and w["blocks"] == 25
    assert "candidate|wrong" in cfg["matrix"]["retrieval"]


def test_the_replay_stage_keeps_the_donor_bound_for_the_same_mask():
    """Without it, a replay that underperforms cannot be told apart from a
    mask that was never worth much."""
    cfg = r2_conditions(screen("mapped_attn_25", 5), DEFS, 256)
    b = cfg["definitions"]["donor_bound"]
    assert b["repair"] == "oracle" and b["blocks"] == 25
    assert b["selector"] == "mapped_attn"


def test_refusing_invalid_screen_evidence_writes_a_verdict_and_no_conditions(tmp_path):
    man = {"documents": [{"doc": "a"}, {"doc": "b"}]}
    cfg = {
        "total_blocks": 256,
        "definitions": {
            "mapped_unrepaired": {"repair": "none"},
            "uniform_25": {"repair": "oracle", "selector": "uniform", "blocks": 25},
        },
        "matrix": {"retrieval": ["mapped_unrepaired", "uniform_25"]},
    }
    mp, cp = str(tmp_path / "m.json"), str(tmp_path / "c.json")
    json.dump(man, open(mp, "w"))
    json.dump(cfg, open(cp, "w"))
    rows = str(tmp_path / "rows.jsonl")
    with open(rows, "w") as fh:  # document b is missing entirely
        for c in ("mapped_unrepaired", "uniform_25"):
            fh.write(
                json.dumps(
                    {"kind": "retrieval", "condition": c, "doc": "a", "primary": 1.0}
                )
                + "\n"
            )
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "research.kv_translate.pick_candidate",
            "--rows",
            rows,
            "--manifest",
            mp,
            "--conditions",
            cp,
            "--out-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=dict(os.environ, PYTHONPATH=ROOT),
    )
    assert r.returncode == 2
    out = json.load(open(tmp_path / "CANDIDATE.json"))
    assert out["verdict"] == "INVALID_EVIDENCE" and not out["advances"]
    assert not os.path.exists(tmp_path / "conditions_r2.json")

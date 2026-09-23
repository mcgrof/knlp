# SPDX-License-Identifier: GPL-2.0
"""The screen's release rule, on evidence whose right answer is known.

The rule is small and the consequences of getting it wrong are not: it decides
whether a four-hour reserve is spent on a receiver pilot. Each case here is a
shape the real run could produce.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location(
    "c1", os.path.join(ROOT, "research/cartridges_cas/scripts/c1_source_utility.py")
)
c1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c1)


def rows(n_correct, n=20, *, unresolved=0, capped=0, arm="x"):
    out = []
    for i in range(n):
        resolved = i >= unresolved
        out.append(
            {
                "arm": arm,
                "correct": 1.0 if i < n_correct else 0.0,
                "outcome": "exact" if resolved else "abstain",
                "cap_hit": i < capped,
                "faithful_correct": 1.0 if i < n_correct else 0.0,
                "faithful_fell_back_to_option_a": False,
            }
        )
    return out


def test_a_source_arm_that_clears_the_floor_advances():
    g = c1.gate(
        {
            "nocart": rows(6),
            "source": rows(9),
            "fulldoc": rows(15),
        }
    )
    assert g["verdict"] == "SOURCE_UTILITY_ESTABLISHED"
    assert g["advances"] is True
    assert g["arms"]["source"]["net_gain_over_floor"] == 3


def test_a_source_arm_one_short_of_the_bar_does_not_advance():
    g = c1.gate({"nocart": rows(6), "source": rows(7), "fulldoc": rows(15)})
    assert g["arms"]["source"]["net_gain_over_floor"] == 1
    assert g["verdict"] == "SOURCE_UTILITY_UNESTABLISHED"
    assert g["advances"] is False


def test_a_source_arm_that_scores_by_failing_to_answer_does_not_advance():
    """Gaining questions while producing more unreadable output is not utility."""
    g = c1.gate(
        {
            "nocart": rows(6),
            "source": rows(10, unresolved=4),
            "fulldoc": rows(15),
        }
    )
    s = g["arms"]["source"]
    assert s["meets_net_gain"] is True
    assert s["meets_parser_excess"] is False
    assert s["passes"] is False
    assert g["advances"] is False


def test_hitting_the_token_cap_counts_against_an_arm():
    g = c1.gate(
        {
            "nocart": rows(6),
            "source": rows(9, capped=5),
            "fulldoc": rows(15),
        }
    )
    assert g["arms"]["source"]["parser_excess_over_floor"] == 5
    assert g["arms"]["source"]["passes"] is False


def test_one_generation_that_is_both_unreadable_and_capped_counts_once():
    g = c1.gate(
        {
            "nocart": rows(6),
            "source": rows(9, unresolved=3, capped=3),
            "fulldoc": rows(15),
        }
    )
    assert g["arms"]["source"]["unparseable_or_capped"] == 3


def test_a_ceiling_that_does_not_beat_the_floor_makes_the_screen_uninformative():
    """If having the record outright does not help, the questions cannot
    separate having it from not, and nothing about the cartridge follows."""
    g = c1.gate({"nocart": rows(6), "source": rows(9), "fulldoc": rows(6)})
    assert g.get("benchmark_uninformative") is True
    assert "no verdict about the cartridge can be read" in g["note"]


def test_an_informative_ceiling_raises_no_such_flag():
    g = c1.gate({"nocart": rows(6), "source": rows(9), "fulldoc": rows(16)})
    assert "benchmark_uninformative" not in g


def test_a_missing_floor_arm_decides_nothing():
    g = c1.gate({"source": rows(9), "fulldoc": rows(15)})
    assert g["verdict"] == "NO_FLOOR_ARM" and g["advances"] is False


def test_a_missing_source_arm_decides_nothing():
    g = c1.gate({"nocart": rows(6), "fulldoc": rows(15)})
    assert g["verdict"] == "SOURCE_ARM_ABSENT" and g["advances"] is False


def test_the_verdict_carries_its_own_resolution_caveat():
    g = c1.gate({"nocart": rows(6), "source": rows(9), "fulldoc": rows(15)})
    assert "cannot resolve" in g["caveat"]
    assert str(c1.MIN_NET_GAIN) in g["rule"]


@pytest.mark.parametrize(
    "floor,src,expect", [(0, 2, True), (0, 1, False), (18, 20, True)]
)
def test_the_bar_is_an_absolute_count_not_a_ratio(floor, src, expect):
    g = c1.gate({"nocart": rows(floor), "source": rows(src), "fulldoc": rows(20)})
    assert g["arms"]["source"]["meets_net_gain"] is expect

# SPDX-License-Identifier: MIT
"""Check tied magnitude intercepts and the frozen grid selection."""

import math
import os

import pytest
import torch

from research.layerrope_beta.intercept.plan import (
    AUTHOR_POINT,
    confirm_rows,
    primary,
    screen_rows,
    select,
)
from research.layerrope_beta.model import Model

DEVICE = os.environ.get("KNLP_BETA_TEST_DEVICE", "cpu")


@pytest.mark.parametrize("a,beta", [(-0.25, -0.2), (-0.5, 0.0), (-0.1, -0.5)])
def test_input_and_output_magnitudes_follow_the_tied_intercepts(a, beta):
    model = Model("L", 4, 77390, beta=beta, vocab=97, alpha_out=a).to(DEVICE)
    rope = model.lm.model.layerrope
    for site in rope.sites:
        sign = 1.0 if site.startswith("out_") else -1.0
        gain = getattr(rope, site + "_gamma").detach().cpu().double()
        pairs = torch.view_as_complex(gain.reshape(64, 2)).abs()
        for layer in range(4):
            u = torch.tensor(math.log(layer + 1.0), device=DEVICE)
            actual = torch.view_as_complex(
                rope(site, u).detach().cpu().double().reshape(64, 2)
            ).abs()
            expected = pairs * math.exp(sign * a + beta * math.log(layer + 1.0))
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("arm", ["R", "L", "M", "ML"])
def test_zero_intercept_keeps_the_earlier_model(arm):
    old = Model(arm, 3, 77390, beta=0.0, vocab=97)
    new = Model(arm, 3, 77390, beta=0.0, vocab=97, alpha_out=0.0)
    for (name, p), q in zip(old.state_dict().items(), new.state_dict().values()):
        assert torch.equal(p, q), name


def test_grid_rows_and_selection():
    rows = screen_rows()
    assert len(rows) == 2 + 2 * 4 * 5
    assert len({r["id"] for r in rows}) == len(rows)
    curve = [dict(step=s, ce=1.0) for s in range(1000, 2001, 100)]
    done = dict(status="complete", curve=curve)
    assert primary(done) == 1.0
    assert primary(dict(done, status="invalid_numerical")) is None
    gain_rows = [r for r in rows if r["arm"] in ("L", "ML")]
    scores = [(r, 2.0 + abs(r["alpha_out"] + 0.1) + abs(r["beta"])) for r in gain_rows]
    chosen = select(scores)
    assert chosen == {"L": "L-a-minus0p1-beta-0p0", "ML": "ML-a-minus0p1-beta-0p0"}
    confirmed = confirm_rows(chosen)
    ids = [r["id"] for r in confirmed]
    assert ids[:2] == ["R", "M"] and len(ids) == len(set(ids)) == 8
    a, beta = AUTHOR_POINT
    assert any(r["alpha_out"] == a and r["beta"] == beta for r in confirmed)


def test_analysis_applies_the_frozen_rule(tmp_path):
    from research.layerrope_beta.intercept.analyze import analyze
    from research.layerrope_beta.intercept.plan import CONFIRM_SEEDS, job
    from research.mo_variants.stage_c_common import write

    chosen = {"L": "L-a-minus0p5-beta-0p0", "ML": "ML-a-minus0p1-beta-0p0"}
    write(tmp_path / "SELECTED.json", dict(chosen=chosen))
    shift = {"L-a-minus0p5-beta-0p0": -0.02, "ML-a-minus0p1-beta-0p0": -0.005}
    for r in confirm_rows(chosen):
        for seed in CONFIRM_SEEDS:
            ce = 2.6 + shift.get(r["id"], 0.0)
            folder = tmp_path / "language/runs" / job(r, "final", seed)["id"]
            curve = [dict(step=s, ce=ce) for s in range(0, 2001, 100)]
            steps = [dict(step=1, ce=ce, seconds=1.0)]
            write(
                folder / "RESULT.json",
                dict(status="complete", completed_steps=2000, curve=curve, steps=steps),
            )
            cache = [
                dict(name=n, ce=ce + d, delta_ce=d, storage=dict(ideal_total_bytes=b))
                for n, d, b in (
                    ("symmetric-K3V3", 0.02, 3584),
                    ("symmetric-K3V2", 0.05, 3072),
                    ("symmetric-K2V2", 0.2, 2560),
                )
            ]
            other = [dict(name="all-FP4-weights", delta_ce=0.06)]
            write(
                folder / "QUANTIZATION.json",
                dict(clean=dict(ce=ce), cache=cache, weight_activation_residual=other),
            )
    report = analyze(tmp_path)
    assert report["decisions"]["L"]["winner_vs_anchor"]["helps"]
    assert not report["decisions"]["ML"]["winner_vs_anchor"]["helps"]
    row = report["rows"]["L-a-minus0p5-beta-0p0"]
    assert row["minimum_passing_bytes_per_token"] == 3072
    assert abs(row["k3v2_increment"] - 0.03) < 1e-12

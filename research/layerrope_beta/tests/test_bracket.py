# SPDX-License-Identifier: MIT
"""Check the learning-rate ladder, bracketing rule and frozen analysis."""

from research.layerrope_beta.intercept import bracket as b
from research.layerrope_beta.intercept.plan import job, row
from research.mo_variants.stage_c_common import write


def test_ladder_and_ids():
    assert [b.rate("L", k) for k in (-1, 0, 1, 2)] == [0.005, 0.01, 0.02, 0.04]
    assert [b.rate("M", k) for k in (-2, -1, 0, 1)] == [0.00075, 0.0015, 0.003, 0.006]
    r = b.at(row("L", -0.5, 0.0), 1)
    assert r["id"] == "L-a-minus0p5-beta-0p0-lr-0p02" and r["lr"] == 0.02
    assert job(r, "screen", 77911)["lr"] == 0.02
    assert len({x["id"] for x in b.bases()}) == 6


def test_bracketing_rule():
    assert b.step({-1: 2.7, 0: 2.6, 1: 2.65}) == (0, True, None)
    assert b.step({-1: 2.7, 0: 2.65, 1: 2.6}) == (1, False, 2)
    assert b.step({-1: 2.6, 0: 2.65, 1: 2.7}) == (-1, False, -2)
    # A diverged rate counts as worse, so it can close a bracket.
    assert b.step({-1: 2.7, 0: 2.65, 1: 2.6, 2: None}) == (1, True, None)
    assert b.step({-1: None, 0: None, 1: None}) == (-1, False, -2)
    edge = {k: 3.0 - 0.01 * k for k in range(-1, 5)}
    assert b.step(edge) == (4, False, None)


def test_analysis_applies_the_rule(tmp_path):
    table = {}
    for base in b.bases():
        k = 1 if base["arm"] in ("L", "ML") else 0
        table[base["id"]] = dict(best_k=k, bracketed=True)
    write(tmp_path / "BRACKET.json", dict(rows=table))
    shift = {"L-a-minus0p5-beta-0p0": -0.02, "M": 0.05, "R": 0.1}
    for base in b.bases():
        r = b.at(base, table[base["id"]]["best_k"])
        for seed in b.CONFIRM_SEEDS:
            ce = 2.6 + shift.get(base["id"], 0.0)
            folder = tmp_path / "language/runs" / job(r, "final", seed)["id"]
            write(
                folder / "RESULT.json",
                dict(
                    status="complete",
                    completed_steps=2000,
                    curve=[dict(step=s, ce=ce) for s in range(0, 2001, 100)],
                    steps=[dict(step=1, ce=ce, seconds=1.0)],
                ),
            )
            cache = [
                dict(
                    name=n, ce=ce + d, delta_ce=d, storage=dict(ideal_total_bytes=3072)
                )
                for n, d in (("symmetric-K3V3", 0.02), ("symmetric-K3V2", 0.05))
            ]
            write(
                folder / "QUANTIZATION.json",
                dict(
                    clean=dict(ce=ce),
                    cache=cache,
                    weight_activation_residual=[
                        dict(name="all-FP4-weights", delta_ce=0.06)
                    ],
                ),
            )
    for base, ce in ((row("R"), 2.69), (row("L", 0.0, 0.0), 2.61)):
        for seed in b.PREDECESSOR_SEEDS:
            folder = (
                tmp_path / "language/runs" / job(b.at(base, 0), "screen", seed)["id"]
            )
            write(
                folder / "RESULT.json",
                dict(
                    status="complete",
                    curve=[dict(step=s, ce=ce) for s in range(1000, 2001, 100)],
                ),
            )
    report = b.analyze(tmp_path)
    assert report["rows"]["L-a-0p0-beta-0p0"]["lr"] == 0.02
    assert report["decisions"]["L"]["winner_vs_anchor"]["helps"]
    assert not report["decisions"]["ML"]["winner_vs_anchor"]["helps"]
    assert report["decisions"]["ML"]["anchor_vs_mhc"]["ordered"]
    assert report["predecessor"]["R"]["reproduces"]
    assert not report["predecessor"]["L-a-0p0-beta-0p0"]["reproduces"]

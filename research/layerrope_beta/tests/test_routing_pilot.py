# SPDX-License-Identifier: MIT
"""Keep numerical success separate from paired equal-time learning improvement."""

import copy
import pytest
from research.layerrope_beta.time_budget.routing_pilot import analyze, ARMS, pilot_card
from research.layerrope_beta.time_budget.train import source_identity
from research.mo_variants.stage_c_common import write, sha


def records():
    result = []
    for seed in (77711, 77712):
        for label, arm, lr, changed in ARMS:
            ce = 3.0 - (0.02 if label == "M-candidate" else 0.0)
            result.append(
                dict(
                    label=label,
                    seed=seed,
                    receipt={},
                    result=dict(
                        status="complete",
                        identity=dict(
                            job=dict(solver_iterations=80 if changed else 20)
                        ),
                        history=dict(
                            routing=[dict(constraint_pass=True)],
                            landmarks=[
                                dict(
                                    validation=dict(ce=ce),
                                    routing=dict(constraint_pass=True),
                                )
                            ],
                            common_update_landmarks=[
                                dict(
                                    step=n,
                                    validation=dict(ce=ce),
                                    routing=dict(constraint_pass=True),
                                )
                                for n in (25, 100)
                            ],
                        ),
                    ),
                )
            )
    return result


def test_pilot_requires_both_seed_pairs_and_every_observed_constraint():
    card = dict(seeds=[77711, 77712])
    values = records()
    assert analyze(values, card)["continuation"]
    for kind in ("small_effect", "constraint_miss", "missing_common_update", "invalid"):
        modified = copy.deepcopy(values)
        row = next(
            x["result"]
            for x in modified
            if x["seed"] == 77712 and x["label"] == "M-candidate"
        )
        if kind == "small_effect":
            row["history"]["landmarks"][-1]["validation"]["ce"] = 2.995
        elif kind == "constraint_miss":
            row["history"]["routing"][0]["constraint_pass"] = False
        elif kind == "missing_common_update":
            row["history"]["common_update_landmarks"].pop()
        else:
            row["status"] = "invalid_numerical"
        assert not analyze(modified, card)["continuation"], kind
    with pytest.raises(ValueError, match="ten distinct"):
        analyze(values[:-1], card)


def test_pilot_admission_requires_qualified_recipe_and_nonempty_fresh_seed_provenance(
    tmp_path,
):
    inputs = tmp_path / "inputs.json"
    write(inputs, dict(frozen=True))
    audit = tmp_path / "audit.json"
    value = dict(status="stop", candidate_repeats=80, input_sha256=sha(inputs))
    write(audit, value)
    with pytest.raises(ValueError, match="did not qualify"):
        pilot_card({}, audit)
    value["status"] = "candidate_qualified"
    write(audit, value)
    proposal = dict(
        audit_inputs=dict(path=str(inputs), sha256=sha(inputs)),
        source=source_identity(),
        seed_audit=[],
        seeds=[77711, 77712],
        training={},
    )
    with pytest.raises(ValueError, match="Prior seed provenance"):
        pilot_card(proposal, audit)
    prior = tmp_path / "prior.json"
    write(prior, dict(seed=77611))
    proposal["seed_audit"] = [dict(path=str(prior), sha256=sha(prior))]
    assert pilot_card(proposal, audit)["seeds"] == [77711, 77712]
    proposal["seeds"] = [77611, 77712]
    with pytest.raises(ValueError, match="fresh paired"):
        pilot_card(proposal, audit)
    value["candidate_repeats"] = 640
    write(audit, value)
    with pytest.raises(ValueError, match="outside the declared"):
        pilot_card(proposal, audit)

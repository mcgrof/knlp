# SPDX-License-Identifier: MIT
"""Check protocol boundaries, real routing observation and exact-state resume."""

import random
import time
import numpy as np
import pytest
import torch
from research.layerrope_beta.model import Model
from research.layerrope_beta.time_budget.protocol import (
    freeze,
    median_claim,
    choose_lrs,
    edge_extensions,
    LR_GRID,
    ROWS,
)
from research.layerrope_beta.time_budget.routing import RoutingObservation
from research.layerrope_beta.time_budget.state import (
    TrainingClock,
    DataCursor,
    landmark_valid,
    save,
    restore,
    json_hash,
)
from research.layerrope_beta.time_budget.train import train
from research.mo_variants.stage_c_common import write, sha
from research.tinystories_depth.train import state_hash


def test_reduced_menu_keeps_the_declared_floor_and_inference_family():
    p = freeze()
    assert (
        len(p["cache_cases"]) == 8
        and len(p["weight_policies"]) == 8
        and len(p["primary_contrasts"]) == 10
    )
    low = {
        c["name"]
        for c in p["declared_cache_menu"]
        if c["storage"]["ideal_total_bytes"] <= 3072
    }
    assert low == {"symmetric-K2V2", "symmetric-K3V2", "symmetric-K2V3", "affine-K2V2"}
    assert low <= {c["name"] for c in p["cache_cases"]}
    assert p["inference"]["family_lower_coverage"] == 0.9609375


def test_median_bounds_require_all_pairs_and_strict_margins():
    assert median_claim([-0.02] * 8)["statement"] == "practical_median_improvement"
    assert median_claim([-0.01] * 8)["statement"] == "directional_median_improvement"
    assert median_claim([0.0] * 8)["statement"] == "median_noninferiority"
    assert median_claim([0.01] * 8)["statement"] == "unresolved"
    with pytest.raises(ValueError):
        median_claim([-0.1] * 7)
    with pytest.raises(ValueError):
        median_claim([float("nan")] * 8)


def test_pilot_ties_invalid_trials_and_bounded_common_extensions():
    trials = [
        dict(row=r, lr=lr, status="complete", validation_ce=2.0 + lr)
        for r in ROWS
        for lr in LR_GRID
    ]
    selected = choose_lrs(trials)
    assert edge_extensions(selected) == [0.0001]
    selected["R"] = 0.03
    assert edge_extensions(selected) == [0.0001, 0.1]
    trials[0].update(status="invalid_numerical", validation_ce=-10.0)
    assert choose_lrs(trials)["R"] == 0.001
    with pytest.raises(ValueError):
        choose_lrs(trials[:-1])


def test_clock_excludes_offline_time_and_landmark_gate_is_not_timeout():
    values = iter([10.0, 12.0, 20.0, 23.0])
    clock = TrainingClock("cpu", now=lambda: next(values))
    with clock.update():
        pass
    assert clock.seconds == 2.0
    with clock.update():
        pass
    assert clock.seconds == 5.0
    assert landmark_valid(100.0, 100.5, 99.0, 1.5)
    assert not landmark_valid(100.0, 101.01, 99.0, 2.01)
    assert not landmark_valid(100.0, 100.1, 100.0, 0.1)


def test_cursor_keeps_partial_epoch_without_empty_or_dropped_windows():
    tokens = np.arange(13, dtype=np.uint16)
    stream = DataCursor(tokens, 7, window=4, batch_size=2)
    seen = []
    for _ in range(3):
        x, y = stream.batch("cpu")
        assert x.shape == (2, 4) and torch.equal(x[:, 1:], y[:, :-1])
        seen += x[:, 0].tolist()
        stream.advance()
    assert sorted(seen[:3]) == [0, 4, 8] and sorted(seen[3:]) == [0, 4, 8]
    restored = DataCursor(tokens, 7, window=4, batch_size=2, cursor=stream.cursor)
    assert restored.identity() == stream.identity()


@pytest.mark.parametrize("arm", ["R", "L", "M", "ML"])
def test_opt_in_routing_does_not_change_outputs_or_gradients(arm):
    torch.set_num_threads(2)
    a = Model(arm, 2, 77690, beta=0.0, vocab=97)
    b = Model(arm, 2, 77690, beta=0.0, vocab=97)
    b.load_state_dict(a.state_dict())
    observer = RoutingObservation()
    b.routing_observation = observer
    ids = torch.tensor([[2, 3, 5, 7], [4, 6, 8, 9]])
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        x, y = a(ids), b(ids)
    assert torch.equal(x, y)
    x.float().square().mean().backward()
    y.float().square().mean().backward()
    for p, q in zip(a.parameters(), b.parameters()):
        if p.grad is not None:
            assert torch.equal(p.grad, q.grad)
    row = observer.report(4 if arm in ("M", "ML") else 0)
    assert row["finite"]
    assert b.routing_observation is observer and a.routing_observation is None


def test_trained_routing_violation_is_reported_without_operator_repair():
    o = RoutingObservation()
    carry = torch.tensor([[[0.8, 0.8], [0.2, 0.2]]])
    o.add(torch.ones(1), torch.ones(2), torch.ones(2), carry)
    report = o.report(1)
    assert report["row_sum_error"] > 0.5 and not report["constraint_pass"]
    assert carry[0, 0, 0].item() == pytest.approx(0.8)


def test_training_resume_uses_no_final_array_and_rejects_changed_recipe(
    tmp_path, monkeypatch
):
    torch.set_num_threads(2)
    data = tmp_path / "data"
    data.mkdir()
    tokens = np.random.default_rng(2).integers(0, 97, 513, dtype=np.uint16)
    for name in ["train", "validation"]:
        np.save(data / (name + ".npy"), tokens[:97] if name == "train" else tokens)
    write(
        data / "TRAINING.json",
        dict(files={n: sha(data / n) for n in ["train.npy", "validation.npy"]}),
    )
    load = np.load

    def sealed(path, *a, **kw):
        assert "final" not in str(path)
        return load(path, *a, **kw)

    monkeypatch.setattr(np, "load", sealed)
    job = dict(
        arm="ML",
        stage="qualification",
        seed=77690,
        lr=0.001,
        depth=2,
        vocab=97,
        window=16,
        batch_size=2,
    )
    whole = train(
        data, job, tmp_path / "whole", device="cpu", stop_after_updates=4, smoke=True
    )
    partial = train(
        data, job, tmp_path / "part", device="cpu", stop_after_updates=2, smoke=True
    )
    rec = partial["last_checkpoint"]
    resumed = train(
        data,
        job,
        tmp_path / "resumed",
        device="cpu",
        resume=rec["path"],
        resume_sha=rec["sha256"],
        stop_after_updates=4,
        smoke=True,
    )
    a = torch.load(whole["last_checkpoint"]["path"], weights_only=False)
    b = torch.load(resumed["last_checkpoint"]["path"], weights_only=False)
    for field in ["model", "optimizer", "rng"]:
        assert state_hash(a[field]) == state_hash(b[field])
    assert a["cursor"] == b["cursor"] and a["step"] == b["step"] == 4
    assert b["training_seconds"] >= partial["training_seconds"] > 0
    initial = whole["history"]["landmarks"][0]["checkpoint"]
    from_landmark = train(
        data,
        job,
        tmp_path / "landmark-resumed",
        device="cpu",
        resume=initial["path"],
        resume_sha=initial["sha256"],
        stop_after_updates=4,
        smoke=True,
    )
    assert (
        from_landmark["history"]["landmarks"][0]["checkpoint"]["sha256"]
        == initial["sha256"]
    )
    restored = torch.load(from_landmark["last_checkpoint"]["path"], weights_only=False)
    assert state_hash(restored["model"]) == state_hash(a["model"])
    with pytest.raises(ValueError, match="identity mismatch"):
        train(
            data,
            dict(job, lr=0.003),
            tmp_path / "bad",
            device="cpu",
            resume=rec["path"],
            resume_sha=rec["sha256"],
            stop_after_updates=4,
            smoke=True,
        )

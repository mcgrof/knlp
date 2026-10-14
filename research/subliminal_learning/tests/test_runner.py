# SPDX-License-Identifier: MIT
"""End-to-end CPU runs on synthetic data: isolation, controls, receipts."""

import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from research.subliminal_learning import run as runner
from research.subliminal_learning.recipe import REPO_ROOT, Recipe, resolve
from research.subliminal_learning.data import epoch_order, make_noise
from research.subliminal_learning.models import PaperMLP, build, init_common


def synthetic(n_train=192, n_test=64):
    g = torch.Generator().manual_seed(0)
    return [
        torch.rand(n_train, 784, generator=g) * 2 - 1,
        torch.randint(0, 10, (n_train,), generator=g),
        torch.rand(n_test, 784, generator=g) * 2 - 1,
        torch.randint(0, 10, (n_test,), generator=g),
    ]


@pytest.fixture
def fake_data(monkeypatch):
    monkeypatch.setattr(runner, "load_data", lambda recipe, device: synthetic())


@pytest.mark.parametrize(
    "arch", ["paper_mlp", "residual_mlp", "noskip_mlp", "hc", "mhc", "gr", "altup"]
)
@pytest.mark.parametrize("protocol", ["upstream_batched_v1", "per_example_v1"])
def test_end_to_end_receipts(tmp_path, fake_data, arch, protocol):
    r = Recipe(
        arch=arch,
        protocol=protocol,
        seeds=[1, 2, 3],
        pack_size=2,
        teacher_epochs=2,
        student_epochs=2,
        batch_size=80,
        noise_samples=150,
        width=16,
        blocks=2,
        streams=2,
        device="cpu",
        smoke=True,
        output_dir=str(tmp_path),
        checkpoint_dir=str(tmp_path / "ckpt"),
        run_name="t",
    )
    assert runner.cmd_run(r, {}, ["test"]) == 0
    d = tmp_path / "t"
    for name in (
        "recipe.json",
        "environment.json",
        "metrics.jsonl",
        "seeds.csv",
        "manifest.json",
        "status.json",
        "results.json",
    ):
        assert (d / name).exists(), name
    assert not (d / "checkpoints").exists()
    manifest = json.load(open(d / "manifest.json"))
    assert "pack00_teacher.pt" in manifest
    for entry in manifest.values():
        assert entry["path"].startswith(str(tmp_path / "ckpt" / "t"))
        assert os.path.getsize(entry["path"]) == entry["bytes"]
    rows = list(csv.DictReader(open(d / "seeds.csv")))
    assert [r_["model_key"] for r_ in rows] == ["seed-1", "seed-2", "seed-3"]
    for row in rows:
        assert row["teacher_aux_rows_unchanged"] == "True"
        assert row["aux_shared_digit_rows_unchanged"] == "True"
        assert row["aux_other_digit_rows_unchanged"] == "True"
        assert row["w0_digest"] != row["w0_other_digest"]
    results = json.load(open(d / "results.json"))
    assert all(p["results"]["teacher_frozen"] for p in results.values())
    st = json.load(open(d / "status.json"))
    assert st["complete"] and st["logical_cohort"] == 3 and st["packs"] == 2
    # every stage covered every example once per epoch: ceil(192/80)=3 steps
    rows = [json.loads(l) for l in open(d / "metrics.jsonl")]
    steps = [x["steps"] for x in rows if "steps" in x]
    assert set(steps) == {3, 2}  # teacher 192/80 -> 3, students 150/80 -> 2


def test_packing_invariance_end_to_end(tmp_path, fake_data):
    outs = {}
    for pack in (0, 1):
        r = Recipe(
            arch="paper_mlp",
            protocol="upstream_batched_v1",
            seeds=[5, 6],
            pack_size=pack,
            teacher_epochs=1,
            student_epochs=1,
            batch_size=64,
            noise_samples=100,
            device="cpu",
            smoke=True,
            output_dir=str(tmp_path),
            checkpoint_dir="",
            run_name=f"p{pack}",
        )
        runner.cmd_run(r, {}, ["test"])
        outs[pack] = list(csv.DictReader(open(tmp_path / f"p{pack}" / "seeds.csv")))
    for a, b in zip(outs[0], outs[1]):
        assert a["teacher_digest"] != "" and a["w0_digest"] == b["w0_digest"]
        for k in ("acc_teacher", "acc_aux_shared", "acc_full_other"):
            assert abs(float(a[k]) - float(b[k])) < 1e-9, k


def test_budget_stop_and_resume(tmp_path, fake_data, monkeypatch):
    kw = dict(
        arch="paper_mlp",
        seeds=[1, 2],
        teacher_epochs=2,
        student_epochs=2,
        batch_size=64,
        noise_samples=100,
        device="cpu",
        smoke=True,
        output_dir=str(tmp_path),
        checkpoint_dir="",
    )
    ref = Recipe(run_name="ref", **kw)
    assert runner.cmd_run(ref, {}, ["t"]) == 0
    stop = Recipe(run_name="stop", **kw)
    calls = {"n": 0}

    def over(self):
        calls["n"] += 1
        return calls["n"] == 3  # stop inside the first student stage

    monkeypatch.setattr(runner.Run, "over_budget", over)
    assert runner.cmd_run(stop, {}, ["t"]) == runner.EXIT_INCOMPLETE
    assert not json.load(open(tmp_path / "stop" / "status.json"))["complete"]
    monkeypatch.setattr(runner.Run, "over_budget", lambda self: False)
    assert runner.cmd_run(stop, {}, ["t"], resume_dir=str(tmp_path / "stop")) == 0
    a = list(csv.DictReader(open(tmp_path / "ref" / "seeds.csv")))
    b = list(csv.DictReader(open(tmp_path / "stop" / "seeds.csv")))
    for x, y in zip(a, b):
        for k in x:
            assert x[k] == y[k], k


def test_copy_isolation_and_hidden_gradients():
    keys = ["seed-1"]
    w0 = PaperMLP(1, 13)
    init_common(w0, keys, "init")
    student = copy.deepcopy(w0)
    ptrs = {p.data_ptr() for p in w0.parameters()}
    assert not ptrs & {p.data_ptr() for p in student.parameters()}
    snapshot = {k: v.clone() for k, v in w0.state_dict().items()}
    teacher = copy.deepcopy(w0)
    with torch.no_grad():
        for p in teacher.parameters():
            p.add_(0.1 * torch.randn_like(p))
    x = torch.rand(1, 32, 784) * 2 - 1
    with torch.no_grad():
        tgt = teacher(x)[..., 10:]
    loss, _, _ = runner.student_loss(student(x)[..., 10:], tgt, "per_example_v1", 1)
    loss.backward()
    assert student.l1.weight.grad.abs().sum() > 0
    assert student.head.weight.grad[:, :10].abs().sum() == 0
    torch.optim.Adam(student.parameters(), lr=3e-4).step()
    for k, v in w0.state_dict().items():
        assert torch.equal(v, snapshot[k])
    assert torch.equal(student.head.weight[:, :10], w0.head.weight[:, :10])
    assert not torch.equal(student.l1.weight, w0.l1.weight)


def test_named_streams_are_deterministic_and_independent():
    a, b = make_noise("seed-1", 50), make_noise("seed-1", 50)
    assert torch.equal(a, b) and a.min() >= -1 and a.max() < 1
    assert not torch.equal(a, make_noise("seed-2", 50))
    o1 = epoch_order("seed-1", "student_order", 0, 100)
    assert torch.equal(o1, epoch_order("seed-1", "student_order", 0, 100))
    assert not torch.equal(o1, epoch_order("seed-1", "student_order", 1, 100))
    assert sorted(o1.tolist()) == list(range(100))
    m1, m2 = build(Recipe(), 1), build(Recipe(), 1)
    init_common(m1, ["seed-1"], "init")
    init_common(m2, ["seed-1"], "init_other")
    assert not torch.equal(m1.l1.weight, m2.l1.weight)


def test_architectures_share_common_weights():
    r = build(Recipe(arch="residual_mlp", width=16, blocks=2), 1)
    h = build(Recipe(arch="mhc", width=16, blocks=2, streams=4), 1)
    for m in (r, h):
        init_common(m, ["seed-9"], "init")
    for a, b in zip(r.common_linears(), h.common_linears()):
        assert torch.equal(a.weight, b.weight)


def test_config_rejects_drift():
    with pytest.raises(ValueError):
        resolve(None, ["precision=bf16"])
    with pytest.raises(ValueError):
        resolve(None, ["temperature=2"])
    with pytest.raises(ValueError):
        resolve(None, ["reference_models=100"])  # seeds are also set
    with pytest.raises(ValueError):
        resolve(None, ["train_limit=100"])  # limits need smoke
    with pytest.raises(ValueError):
        resolve(None, ["student_loss_mode=not-a-mode"])
    r, src = resolve(None, ["seeds=", "reference_models=100"])
    assert len(r.model_keys) == 100 and r.cohort == 100 and src["seeds"] == "override"


def test_raw_defconfig_resolves_exact_cancellation_choice(tmp_path):
    config = tmp_path / "repair_defconfig"
    config.write_text(
        "CONFIG_SUBLIMINAL_LEARNING=y\n"
        "CONFIG_SUBLIMINAL_STUDENT_LOSS_EXACT_CANCEL=y\n"
    )
    recipe, source = resolve(str(config))
    assert recipe.student_loss_mode == "exact_cancel_v1"
    assert source["student_loss_mode"].startswith("kconfig")


DEFCONFIGS = sorted(
    f
    for f in os.listdir(os.path.join(REPO_ROOT, "defconfigs"))
    if f.startswith("subliminal-mnist-")
    and "CONFIG_SUBLIMINAL_LEARNING=y" in Path(REPO_ROOT, "defconfigs", f).read_text()
)


@pytest.mark.parametrize("name", DEFCONFIGS)
def test_defconfigs_resolve_through_kconfig(tmp_path, name):
    kconfiglib = pytest.importorskip("kconfiglib")
    os.environ.setdefault("srctree", REPO_ROOT)
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        kc = kconfiglib.Kconfig("Kconfig", warn=False)
        kc.load_config(os.path.join("defconfigs", name))
        out = tmp_path / ".config"
        kc.write_config(str(out))
    finally:
        os.chdir(cwd)
    recipe, source = resolve(str(out))
    raw_recipe, _ = resolve(os.path.join(REPO_ROOT, "defconfigs", name))
    assert recipe.to_json() == raw_recipe.to_json()
    assert source["arch"].startswith("kconfig")
    if "smoke" not in name:
        assert not recipe.smoke and recipe.precision == "fp32"


def test_milestones_and_heldout_diagnostics(tmp_path, fake_data):
    r = Recipe(
        arch="altup",
        seeds=[1, 2],
        teacher_epochs=1,
        student_epochs=3,
        student_eval_epochs=[1, 3],
        batch_size=64,
        noise_samples=100,
        heldout_noise=50,
        width=16,
        blocks=2,
        device="cpu",
        smoke=True,
        output_dir=str(tmp_path),
        checkpoint_dir="",
        run_name="m",
    )
    assert runner.cmd_run(r, {}, ["t"]) == 0
    d = tmp_path / "m"
    final = list(csv.DictReader(open(d / "seeds.csv")))
    e3 = list(csv.DictReader(open(d / "seeds_e3.csv")))
    e1 = list(csv.DictReader(open(d / "seeds_e1.csv")))
    assert final == e3
    rows = [json.loads(l) for l in open(d / "metrics.jsonl")]
    ep1 = {x["stage"]: x["test_acc"] for x in rows if x.get("epoch") == 1}
    for m, row in enumerate(e1):
        for kind in ("aux_shared", "full_shared", "aux_other", "full_other"):
            assert float(row[f"acc_{kind}"]) == ep1[kind][m]
    zero = [x for x in rows if x.get("epoch") == 0]
    assert len(zero) == 4 and all("heldout_kl" in x for x in zero)
    res = json.load(open(d / "results.json"))["0"]["results"]
    assert set(res["heldout_targets"]) == {"aux", "full"}
    assert res["common_digest"][0]["stem"] != res["common_digest_other"][0]["stem"]


def test_zero_target_scale_means_no_learning(tmp_path, fake_data):
    r = Recipe(
        arch="residual_mlp",
        seeds=[1, 2],
        teacher_epochs=1,
        student_epochs=2,
        batch_size=64,
        noise_samples=100,
        heldout_noise=50,
        width=16,
        blocks=2,
        target_scale=0.0,
        device="cpu",
        smoke=True,
        output_dir=str(tmp_path),
        checkpoint_dir="",
        run_name="z",
    )
    assert runner.cmd_run(r, {}, ["t"]) == 0
    rows = list(csv.DictReader(open(tmp_path / "z" / "seeds.csv")))
    for row in rows:
        assert float(row["gain_aux_shared"]) == 0.0
        assert float(row["gain_full_shared"]) == 0.0


def test_target_match_kl_calibrates(tmp_path, fake_data):
    r = Recipe(
        arch="residual_mlp",
        seeds=[1, 2],
        teacher_epochs=2,
        student_epochs=1,
        batch_size=64,
        noise_samples=100,
        heldout_noise=50,
        width=16,
        blocks=2,
        target_match_kl=1e-4,
        device="cpu",
        smoke=True,
        output_dir=str(tmp_path),
        checkpoint_dir="",
        run_name="c",
    )
    assert runner.cmd_run(r, {}, ["t"]) == 0
    res = json.load(open(tmp_path / "c" / "results.json"))["0"]["results"]
    for a, kl, ok in zip(res["target_a"], res["target_a_kl"], res["target_a_reached"]):
        assert 0 <= a <= 1
        if ok:
            assert abs(kl - 1e-4) < 1e-6


def test_fresh_noise_runs_and_analyzer_validates(tmp_path, fake_data):
    from research.subliminal_learning import analyze

    kw = dict(
        arch="residual_mlp",
        seeds=[1, 2],
        teacher_epochs=1,
        student_epochs=3,
        student_eval_epochs=[1, 3],
        batch_size=64,
        noise_samples=100,
        heldout_noise=50,
        width=16,
        blocks=2,
        device="cpu",
        smoke=True,
        output_dir=str(tmp_path),
        checkpoint_dir="",
    )
    for name, extra in (("a", {}), ("b", {"fresh_noise": True})):
        assert runner.cmd_run(Recipe(run_name=name, **kw, **extra), {}, ["t"]) == 0
    for name in ("a", "b"):
        p = tmp_path / name / "recipe.json"
        rec = json.load(open(p))
        rec["git"] = {"commit": "test", "workload_dirty": False}
        json.dump(rec, open(p, "w"))
    a, b = analyze.load(str(tmp_path / "a")), analyze.load(str(tmp_path / "b"))
    out = analyze.compare(a, b, 0.95, 5.0, 200)
    assert set(out["milestones"]) == {"1", "3"} and "promote" in out
    assert out["speed"]["full_shared"]["status"] in ("ok", "fail", "unresolved")
    # a run missing one seed must be rejected
    rows = list(csv.DictReader(open(tmp_path / "a" / "seeds.csv")))
    with open(tmp_path / "a" / "seeds.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows[:1])
    with pytest.raises(analyze.RunError):
        analyze.load(str(tmp_path / "a"))

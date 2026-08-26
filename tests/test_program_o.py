"""Offline tests for the Program O prune-eval lane.

Everything runs on CPU with a tiny randomly-initialized GPT-NeoX
and an injected fake batch source — no network, no downloads.
"""

import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fim.fisher_pruning.program_o import (  # noqa: E402
    build_scores,
    calibrate,
    cmd_prune_eval,
    discover_target_linears,
)

transformers = pytest.importorskip("transformers")
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM  # noqa: E402

VOCAB = 128
SEQ_LEN = 16
BATCH = 2

EXPECTED_TARGETS = [
    f"gpt_neox.layers.{i}.{suffix}"
    for i in range(2)
    for suffix in (
        "attention.query_key_value",
        "attention.dense",
        "mlp.dense_h_to_4h",
        "mlp.dense_4h_to_h",
    )
]


def tiny_model() -> GPTNeoXForCausalLM:
    cfg = GPTNeoXConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        vocab_size=VOCAB,
    )
    torch.manual_seed(0)
    return GPTNeoXForCausalLM(cfg)


def fake_batch_fn(skip_docs: int, n_batches: int):
    """Deterministic token blocks; disjoint per skip_docs offset."""
    gen = torch.Generator().manual_seed(9000 + skip_docs)
    return [
        torch.randint(0, VOCAB, (BATCH, SEQ_LEN), generator=gen)
        for _ in range(n_batches)
    ]


@pytest.fixture(scope="module")
def model():
    return tiny_model()


@pytest.fixture(scope="module")
def calibrated(model):
    targets = discover_target_linears(model)
    factors, n_tokens = calibrate(model, targets, fake_batch_fn(0, 3))
    return targets, factors, n_tokens


def test_discover_targets_gptneox(model):
    targets = discover_target_linears(model)
    assert sorted(targets) == sorted(EXPECTED_TARGETS)
    for name, mod in targets.items():
        assert isinstance(mod, nn.Linear), name
    assert "gpt_neox.embed_in" not in targets
    assert "embed_out" not in targets


def test_discover_targets_generic_fallback():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc_up = nn.Linear(8, 16)
            self.fc_down = nn.Linear(16, 8)

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(10, 8)
            self.layers = nn.ModuleList([Block(), Block()])
            self.lm_head = nn.Linear(8, 10)

    targets = discover_target_linears(Toy())
    assert sorted(targets) == [
        "layers.0.fc_down",
        "layers.0.fc_up",
        "layers.1.fc_down",
        "layers.1.fc_up",
    ]


def test_calibration_factor_shapes(calibrated):
    _, factors, n_tokens = calibrated
    assert n_tokens == 3 * BATCH * SEQ_LEN
    # Rectangular MLP layers pin the [out, in] orientation.
    up = factors["gpt_neox.layers.0.mlp.dense_h_to_4h"]  # 32 -> 64
    assert up["A"].shape == (32, 32)
    assert up["G"].shape == (64, 64)
    assert up["D"].shape == (64, 32)
    down = factors["gpt_neox.layers.0.mlp.dense_4h_to_h"]  # 64 -> 32
    assert down["A"].shape == (64, 64)
    assert down["G"].shape == (32, 32)
    assert down["D"].shape == (32, 64)
    for f in (up, down):
        assert torch.allclose(f["A"], f["A"].T, atol=1e-5)
        assert (torch.diagonal(f["A"]) >= 0).all()
        assert (f["D"] >= 0).all()


def test_wanda_score_matches_input_norms(model, calibrated):
    """wanda == |w| * sqrt(E[x_j^2]) broadcast over output rows,
    verified against activations captured independently of the
    accumulator."""
    targets, factors, _ = calibrated
    name = "gpt_neox.layers.0.mlp.dense_h_to_4h"
    mod = targets[name]
    captured = []
    handle = mod.register_forward_hook(
        lambda m, inp, out: captured.append(inp[0].detach().reshape(-1, 32).float())
    )
    try:
        with torch.no_grad():
            for x in fake_batch_fn(0, 3):
                model(input_ids=x)
    finally:
        handle.remove()
    x_all = torch.cat(captured)
    diag_a = (x_all**2).mean(dim=0)
    assert torch.allclose(
        torch.diagonal(factors[name]["A"]).float(), diag_a, rtol=1e-4, atol=1e-6
    )
    weights = {name: mod.weight.detach().float().cpu()}
    scores = build_scores(weights, {name: factors[name]})
    expected = weights[name].abs() * diag_a.sqrt().unsqueeze(0)
    assert scores["wanda"][name].shape == (64, 32)
    assert torch.allclose(scores["wanda"][name], expected, rtol=1e-4, atol=1e-6)


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = torch.argsort(torch.argsort(a.reshape(-1))).float()
    rb = torch.argsort(torch.argsort(b.reshape(-1))).float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return (ra @ rb / (ra.norm() * rb.norm())).item()


def test_q050_rank_agrees_with_w2_fisher(model, calibrated):
    """fresh_q050 = |w| * (D+eps)^0.5 = sqrt(w^2 * (D+eps)) must
    rank-agree with the w^2 * F saliency (F = diagonal Fisher D)."""
    targets, factors, _ = calibrated
    weights = {n: m.weight.detach().float().cpu() for n, m in targets.items()}
    scores = build_scores(weights, factors)
    for n, w in weights.items():
        rho = _spearman(scores["fresh_q050"][n], w.pow(2) * factors[n]["D"].float())
        assert rho > 0.999, (n, rho)


def test_prune_eval_smoke(model, tmp_path):
    cfg = {
        "model_id": "tiny-gptneox-random-init",
        "out_dir": str(tmp_path),
        "data": {"seq_len": SEQ_LEN, "batch_size": BATCH, "seed": 0},
        "calibration": {"n_batches": 3, "skip_docs": 0},
        "eval": {"n_batches": 2, "skip_docs": [100, 200]},
        "sparsities": [0.3, 0.5, 0.7],
    }
    before = {
        n: m.weight.detach().clone() for n, m in discover_target_linears(model).items()
    }
    cmd_prune_eval(cfg, device="cpu", model=model, batch_fn=fake_batch_fn)
    log = tmp_path / "results" / "programO.jsonl"
    events = [json.loads(line) for line in log.read_text().splitlines()]

    start = [e for e in events if e["event"] == "start"]
    assert len(start) == 1
    assert start[0]["model_id"] == cfg["model_id"]
    assert "config_hash" in start[0] and "code_commit" in start[0]
    arms = start[0]["arms"]
    assert sorted(arms) == sorted(
        [
            "random",
            "magnitude",
            "fresh_q025",
            "fresh_q050",
            "kron_q025",
            "kron_q050",
            "kfac_obs",
            "wanda",
        ]
    )

    evals = [e for e in events if e["event"] == "eval"]
    # dense (2 eval sets) + 8 arms * 3 sparsities * 2 eval sets
    assert len(evals) == 2 + 8 * 3 * 2
    dense = [e for e in evals if e["arm"] == "dense"]
    assert {e["eval_set"] for e in dense} == {0, 1}
    for e in evals:
        assert e["val_ppl"] == pytest.approx(
            torch.tensor(e["val_loss"]).exp().item(), rel=1e-6
        )
        if e["arm"] != "dense":
            assert abs(e["actual_sparsity"] - e["sparsity"]) < 0.02

    assert events[-1]["event"] == "done"

    # Pristine weights restored after the run.
    after = discover_target_linears(model)
    for n, w in before.items():
        assert torch.equal(after[n].weight.detach(), w), n


def test_stale_state_arms(tmp_path):
    import collections

    from fim.fisher_pruning.program_o import build_scores

    weights = {"blk.fc": torch.randn(4, 3).abs() + 0.1}
    factors = {
        "blk.fc": {
            "D": torch.rand(4, 3),
            "G": torch.eye(4),
            "A": torch.eye(3),
        }
    }
    stale = {"blk.fc": torch.rand(4, 3)}
    scores = build_scores(weights, factors, stale_v=stale)
    assert "stale_q025" in scores and "stale_q050" in scores
    w = weights["blk.fc"]
    expected = w.abs() * (stale["blk.fc"].float() + 1e-8) ** 0.25
    torch.testing.assert_close(scores["stale_q025"]["blk.fc"], expected)
    # without stale state the arms are absent
    assert "stale_q025" not in build_scores(weights, factors)
    # shape mismatch is rejected
    bad = {"blk.fc": torch.rand(3, 3)}
    try:
        build_scores(weights, factors, stale_v=bad)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_arm_view_builds_one_arm_at_a_time():
    """Score maps are model-sized; holding every arm at once is what
    the host out-of-memory killer reacted to on a 1B model."""
    from fim.fisher_pruning.program_o import _ArmView

    built = []
    view = _ArmView(["a", "b"], lambda n: (built.append(n), {"w": n})[1])
    assert len(view) == 2 and "a" in view and "z" not in view
    assert built == [], "listing names must not build anything"
    assert view["a"] == {"w": "a"} and built == ["a"]
    names = [n for n, _ in view.items()]
    assert names == ["a", "b"]
    try:
        view["zzz"]
        assert False, "unknown arm must raise"
    except KeyError:
        pass


def test_build_scores_only_filter():
    import torch

    from fim.fisher_pruning.program_o import build_scores

    weights = {"l": torch.randn(4, 3)}
    factors = {"l": {"D": torch.rand(4, 3), "G": torch.eye(4), "A": torch.eye(3)}}
    one = build_scores(weights, factors, only="magnitude")
    assert set(one) == {"magnitude"}
    torch.testing.assert_close(one["magnitude"]["l"], weights["l"].abs())


def test_build_scores_names_only_builds_nothing():
    """Naming the arms must not touch the tensors: the probe that did
    sliced weights but not the replay state and raised on shape."""
    import torch

    from fim.fisher_pruning.program_o import build_scores

    weights = {"l": torch.randn(4, 3)}
    factors = {"l": {"D": torch.rand(4, 3), "G": torch.eye(4), "A": torch.eye(3)}}
    replay = {16: {"l": torch.rand(4, 3)}}
    names = build_scores(weights, factors, replay_v=replay, names_only=True)
    assert "replay_b16" in names and "magnitude" in names
    assert all(v == {} for v in names.values()), "names_only must build no tensors"
    full = build_scores(weights, factors, replay_v=replay)
    assert sorted(full) == sorted(names), "name list must match the real build"

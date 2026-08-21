"""Offline tests for the state-synthesis signal builders.

Everything runs on CPU with a tiny randomly-initialized GPT-NeoX
and injected batch sources — no network, no downloads (the
trajectory checkpoint loader is monkeypatched in the smoke test).
"""

import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import fim.fisher_pruning.program_o as program_o  # noqa: E402
from fim.fisher_pruning.program_o import (  # noqa: E402
    EPS,
    build_scores,
    calibrate,
    cmd_prune_eval,
    discover_target_linears,
)
from fim.fisher_pruning.state_synthesis import (  # noqa: E402
    frozen_adam_replay,
    trajectory_scores,
)

transformers = pytest.importorskip("transformers")
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM  # noqa: E402

VOCAB = 128
SEQ_LEN = 16
BATCH = 2


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


def per_batch_grads(model, targets, batches, name):
    """Mean-loss gradient of one target weight, per batch."""
    model.eval()
    grads = []
    for x in batches:
        model.zero_grad(set_to_none=True)
        out = model(input_ids=x, labels=x)
        out.loss.backward()
        grads.append(targets[name].weight.grad.detach().float().clone())
    model.zero_grad(set_to_none=True)
    return grads


def test_replay_ema_matches_hand_computed(model):
    """Two-batch replay EMA is exactly v1 = (1-b2) g1^2 and
    v2 = b2 v1 + (1-b2) g2^2, computed from independently captured
    minibatch gradients (weights frozen, so grads are identical)."""
    targets = discover_target_linears(model)
    name = "gpt_neox.layers.0.mlp.dense_h_to_4h"
    batches = fake_batch_fn(0, 2)
    g1, g2 = per_batch_grads(model, targets, batches, name)
    beta2 = 0.999
    snaps = frozen_adam_replay(
        model, targets, batches, betas=(0.9, beta2), checkpoints=(1, 2)
    )
    assert sorted(snaps) == [1, 2]
    v1 = (1.0 - beta2) * g1**2
    v2 = beta2 * v1 + (1.0 - beta2) * g2**2
    assert torch.equal(snaps[1][name], v1)
    assert torch.equal(snaps[2][name], v2)
    for snap in snaps.values():
        assert sorted(snap) == sorted(targets)
        for t in snap.values():
            assert t.dtype == torch.float32
            assert t.device.type == "cpu"


def test_replay_snapshots_are_distinct_objects(model):
    targets = discover_target_linears(model)
    snaps = frozen_adam_replay(model, targets, fake_batch_fn(0, 2), checkpoints=(1, 2))
    for n in targets:
        assert snaps[1][n].data_ptr() != snaps[2][n].data_ptr()
        # An aliased live accumulator would make both snapshots
        # equal to the final v.
        assert not torch.equal(snaps[1][n], snaps[2][n])


def test_replay_leaves_weights_and_grads_untouched(model):
    targets = discover_target_linears(model)
    before = {n: m.weight.detach().clone() for n, m in targets.items()}
    frozen_adam_replay(model, targets, fake_batch_fn(0, 2), checkpoints=(2,))
    for n, m in targets.items():
        assert torch.equal(m.weight.detach(), before[n]), n
        assert m.weight.grad is None, n


def test_replay_exhausted_or_bad_checkpoints(model):
    targets = discover_target_linears(model)
    with pytest.raises(RuntimeError):
        frozen_adam_replay(model, targets, fake_batch_fn(0, 2), checkpoints=(5,))
    with pytest.raises(ValueError):
        frozen_adam_replay(model, targets, fake_batch_fn(0, 2), checkpoints=())
    with pytest.raises(ValueError):
        frozen_adam_replay(model, targets, fake_batch_fn(0, 2), checkpoints=(0,))


def test_trajectory_score_formula():
    gen = torch.Generator().manual_seed(7)
    w_now = {
        "a": torch.randn(4, 3, generator=gen),
        "b": torch.randn(2, 5, generator=gen),
    }
    w_prev = {n: torch.randn(w.shape, generator=gen) for n, w in w_now.items()}
    scores = trajectory_scores(w_now, w_prev)
    for n, w in w_now.items():
        expected = (w * (w - w_prev[n])).abs()
        assert torch.equal(scores[n], expected)
        assert scores[n].dtype == torch.float32
    # extra prev entries are tolerated (full state dicts)
    extra = dict(w_prev, unrelated=torch.randn(2, 2, generator=gen))
    assert sorted(trajectory_scores(w_now, extra)) == ["a", "b"]
    with pytest.raises(ValueError):
        trajectory_scores(w_now, {"a": torch.randn(3, 4), "b": w_prev["b"]})
    with pytest.raises(KeyError):
        trajectory_scores(w_now, {"a": w_prev["a"]})


def test_build_scores_wired_arms(model):
    targets = discover_target_linears(model)
    factors, _ = calibrate(model, targets, fake_batch_fn(0, 2))
    weights = {n: m.weight.detach().float().cpu() for n, m in targets.items()}
    replay = frozen_adam_replay(model, targets, fake_batch_fn(0, 2), checkpoints=(1, 2))
    gen = torch.Generator().manual_seed(11)
    w_prev = {
        n: w + 0.01 * torch.randn(w.shape, generator=gen) for n, w in weights.items()
    }
    scores = build_scores(weights, factors, replay_v=replay, w_prev=w_prev)
    assert {"replay_b1", "replay_b2", "trajectory"} <= set(scores)
    for n, w in weights.items():
        expected = w.abs() * (replay[2][n].float() + EPS) ** 0.25
        torch.testing.assert_close(scores["replay_b2"][n], expected)
        torch.testing.assert_close(scores["trajectory"][n], (w * (w - w_prev[n])).abs())
    # absent when not configured
    base = build_scores(weights, factors)
    assert not any(a.startswith("replay_b") for a in base)
    assert "trajectory" not in base
    # replay shape mismatch is rejected
    bad = {1: {n: torch.rand(2, 2) for n in weights}}
    with pytest.raises(ValueError):
        build_scores(weights, factors, replay_v=bad)


def test_prune_eval_smoke_replay_trajectory(model, tmp_path, monkeypatch):
    seen = {}

    def fake_loader(model_id, revision, device="cpu"):
        seen["args"] = (model_id, revision)
        gen = torch.Generator().manual_seed(21)
        return {
            n: m.weight.detach().float().cpu()
            + 0.01 * torch.randn(m.weight.shape, generator=gen)
            for n, m in discover_target_linears(model).items()
        }

    monkeypatch.setattr(program_o, "load_hf_target_weights", fake_loader)
    cfg = {
        "model_id": "tiny-gptneox-random-init",
        "out_dir": str(tmp_path),
        "data": {"seq_len": SEQ_LEN, "batch_size": BATCH, "seed": 0},
        "calibration": {"n_batches": 3, "skip_docs": 0},
        "eval": {"n_batches": 2, "skip_docs": [100, 200]},
        "sparsities": [0.5],
        "replay": {"checkpoints": [1, 3]},
        "trajectory": {"prev_revision": "step142000"},
    }
    before = {
        n: m.weight.detach().clone() for n, m in discover_target_linears(model).items()
    }
    cmd_prune_eval(cfg, device="cpu", model=model, batch_fn=fake_batch_fn)
    assert seen["args"] == (cfg["model_id"], "step142000")

    log = tmp_path / "results" / "programO.jsonl"
    events = [json.loads(line) for line in log.read_text().splitlines()]
    start = [e for e in events if e["event"] == "start"]
    assert len(start) == 1
    assert {"replay_b1", "replay_b3", "trajectory"} <= set(start[0]["arms"])

    evals = [e for e in events if e["event"] == "eval"]
    # dense (2 eval sets) + (8 base + 2 replay + 1 trajectory)
    # arms * 1 sparsity * 2 eval sets
    assert len(evals) == 2 + 11 * 1 * 2
    for arm in ("replay_b1", "replay_b3", "trajectory"):
        cells = [e for e in evals if e["arm"] == arm]
        assert {e["eval_set"] for e in cells} == {0, 1}
        for e in cells:
            assert e["sparsity"] == 0.5
            assert abs(e["actual_sparsity"] - 0.5) < 0.02
    assert events[-1]["event"] == "done"

    # Pristine weights restored after the run.
    after = discover_target_linears(model)
    for n, w in before.items():
        assert torch.equal(after[n].weight.detach(), w), n

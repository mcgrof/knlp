"""Probe-only role-oracle overwrite binding tests."""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PROBE_PATH = ROOT / "scripts" / "trellis_overwrite_probe.py"
SPEC = importlib.util.spec_from_file_location("trellis_overwrite_probe", PROBE_PATH)
probe = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def _args(**overrides):
    base = dict(
        contexts="32",
        unique_keys="4",
        overwrites="1",
        latest_gaps="8",
        n_keys=16,
        n_vals=16,
        distractor_vocab=4,
        batch=2,
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_head=8,
        n_slots=8,
        chunk_size=4,
        dtype="fp32",
        binding_residual_init=0.10,
        binding_residual_max=1.0,
        binding_eta_init=1.0,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _cfg(args):
    return probe.make_cfg(
        args,
        "none",
        "shared",
        1.0,
        1e-3,
        0.25,
        "key_readout",
        0.05,
        0.75,
        "scalar",
        0.80,
        "value",
        "all",
        "current",
        0.0,
    )


def test_make_batch_emits_roles_and_key_ids():
    args = _args()
    cell = probe.Cell(seq_len=32, n_unique=4, overwrites=1, latest_gap=8)
    idx, labels, meta = probe.make_batch(
        cell,
        args,
        random.Random(0),
        torch.device("cpu"),
    )
    assert idx.shape == labels.shape == meta.roles.shape == meta.key_ids.shape
    assert meta.roles.shape == (args.batch, cell.seq_len)
    for bi, pos in enumerate(meta.query_pos):
        assert int(meta.roles[bi, pos]) == probe.ROLE_QUERY_KEY
        assert int(meta.roles[bi, pos + 1]) == probe.ROLE_ANSWER
        assert int(meta.key_ids[bi, pos]) >= 0


def test_role_oracle_main_forward_backward_is_finite():
    torch.manual_seed(0)
    args = _args()
    cell = probe.Cell(seq_len=32, n_unique=4, overwrites=1, latest_gap=8)
    idx, labels, meta = probe.make_batch(
        cell,
        args,
        random.Random(1),
        torch.device("cpu"),
    )
    model = probe.build_role_oracle_probe_model(
        _cfg(args),
        "trellis_role_main",
        args,
    )
    logits, loss = probe.forward_model(model, idx, labels, True, meta)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    loss.backward()
    assert model.last_binding_diag["backend"] == "role_oracle_main_update_probe"


def test_role_oracle_separate_kv_learned_forward_backward_is_finite():
    torch.manual_seed(1)
    args = _args()
    cell = probe.Cell(seq_len=32, n_unique=4, overwrites=1, latest_gap=8)
    idx, labels, meta = probe.make_batch(
        cell,
        args,
        random.Random(2),
        torch.device("cpu"),
    )
    model = probe.build_role_oracle_probe_model(
        _cfg(args),
        "trellis_role_kv_learned",
        args,
    )
    logits, loss = probe.forward_model(model, idx, labels, True, meta)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    loss.backward()
    diag = model.last_binding_diag
    assert diag["backend"] == "role_oracle_binding_probe"
    assert diag["address_mode"] == "learned"
    assert diag["write_count_mean"] > 0
    assert diag["read_count_mean"] > 0


def test_role_oracle_separate_kv_oracle_address_is_frozen():
    torch.manual_seed(2)
    args = _args()
    cell = probe.Cell(seq_len=32, n_unique=4, overwrites=1, latest_gap=8)
    idx, labels, meta = probe.make_batch(
        cell,
        args,
        random.Random(3),
        torch.device("cpu"),
    )
    model = probe.build_role_oracle_probe_model(
        _cfg(args),
        "trellis_role_kv_oracle",
        args,
    )
    assert model.oracle_address.requires_grad is False
    logits, loss = probe.forward_model(model, idx, labels, True, meta)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    loss.backward()
    diag = model.last_binding_diag
    assert diag["address_mode"] == "oracle"
    assert diag["alignment_accuracy"] == 1.0

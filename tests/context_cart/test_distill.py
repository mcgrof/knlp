# SPDX-License-Identifier: MIT
"""Distillation objective, shared-prefix gradients and the tiny loop."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from research.context_cart import cas_bridge as B
from research.context_cart import distill as DS
from research.context_cart import model as M
from research.context_cart import runs as R
from research.context_cart import selfstudy as S

pytest.importorskip("peft")

SPEC = dict(
    name="tiny", revision="0" * 40, dtype="float32", memory_slots=5, lora_rank=2
)


def _element(seed: int, length: int = 9, k: int = 3) -> S.Element:
    rng = np.random.default_rng(seed)
    ids = rng.integers(2, 500, size=length).tolist()
    positions = list(range(4, length))
    t = len(positions)
    target_ids = rng.integers(2, 500, size=(t, k))
    target_ids[:, 0] = [ids[p] for p in positions]
    logprobs = np.log(rng.dirichlet(np.ones(k) * 2, size=t) * 0.9).astype(np.float32)
    return S.Element(
        row_index=seed,
        prompt=f"prompt {seed}",
        input_ids=ids,
        target_positions=positions,
        target_ids=target_ids.astype(np.int64),
        target_logprobs=logprobs,
        truncated=False,
        kind="question",
        note_id="t",
    )


def _models():
    spec = M.ModelSpec(**SPEC)
    torch.manual_seed(0)
    reader_base = M.load_base(spec, "cpu")
    writer_base = M.load_base(spec, "cpu")
    writer = M.Writer(writer_base, spec, init_rows=[5, 6, 7, 8, 9])
    reader = M.Reader(reader_base, eos_ids=(1,), pad_id=0)
    return spec, writer, reader


def test_topk_loss_matches_direct_computation():
    _, _, reader = _models()
    element = _element(1)
    prefix = [(k.detach(), v.detach()) for k, v in B.sink_rows(reader.model, 3)]
    out = DS.topk_loss(reader, prefix, element, 1)
    ids = torch.tensor([element.input_ids])
    hidden, _ = reader.forward(prefix, ids, torch.ones_like(ids), 1)
    logq = torch.log_softmax(reader.model.lm_head(hidden[0]).float(), -1)
    expected = 0.0
    for t, pos in enumerate(element.target_positions):
        for j in range(element.target_ids.shape[1]):
            p = float(np.exp(element.target_logprobs[t, j]))
            expected -= p * float(logq[pos - 1, int(element.target_ids[t, j])])
    assert abs(float(out["loss_sum"]) - expected) < 1e-4
    assert out["tokens"] == len(element.target_positions) and out["k"] == 3
    assert 0.8 < out["mass"] < 0.95


def test_shared_prefix_step_equals_separate_backward_passes():
    _, writer, reader = _models()
    source = DS.Source("p", "r", "d" * 64, [11, 12, 13, 14, 15, 16, 17], rows=4)
    sink = [(k.detach(), v.detach()) for k, v in B.sink_rows(reader.model, 3)]
    elements = [_element(2), _element(3)]
    stats = DS.shared_prefix_step(
        writer, reader, source, elements, sink, torch.float32, True
    )
    shared = {
        n: p.grad.clone() for n, p in writer.named_parameters() if p.grad is not None
    }
    writer.zero_grad(set_to_none=True)
    total = sum(e.scored_tokens for e in elements)
    for element in elements:
        framed, start = DS.framed_prefix(
            writer, reader, source, sink, torch.float32, False
        )
        out = DS.topk_loss(reader, framed, element, start)
        (out["loss_sum"] / total).backward()
    separate = {
        n: p.grad.clone() for n, p in writer.named_parameters() if p.grad is not None
    }
    assert set(shared) == set(separate) and shared
    for name in shared:
        assert torch.allclose(shared[name], separate[name], atol=1e-5), name
    assert stats["tokens"] == total and 0 <= stats["teacher_agreement"] <= 1
    assert all(p.grad is None for p in reader.model.parameters())


def _prepared(tmp_path: Path, spec: M.ModelSpec, reader_model) -> Path:
    out = tmp_path / "prepared"
    out.mkdir()
    # a fake native cartridge of four rows with the model's own sink first
    sink = B.sink_rows(reader_model, 3)
    gen = torch.Generator().manual_seed(9)
    g = M.Geometry.from_config(reader_model.config)
    prefix = []
    for sk, sv in sink:
        k = torch.randn(1, g.kv_heads, 3, g.head_dim, generator=gen)
        v = torch.randn(1, g.kv_heads, 3, g.head_dim, generator=gen)
        prefix.append((torch.cat([sk, k], 2), torch.cat([sv, v], 2)))
    cart = out / "native_a.pt"
    B.save_native_cartridge(cart, prefix, frozen_rows=1)
    loaded = B.load_native_cartridge(cart, dtype=torch.float32)
    sources = {
        "a": DS.Source("a", "ra", "d" * 64, [11, 12, 13, 14, 15, 16], rows=4),
        "b": DS.Source("b", "rb", "e" * 64, [21, 22, 23, 24, 25, 26, 27, 28], rows=6),
    }
    DS.save_elements(out / "a_train.pt", [_element(i) for i in range(6)])
    DS.save_elements(out / "a_question.pt", [_element(20), _element(21)])
    DS.save_elements(out / "b_holdout.pt", [_element(30), _element(31)])
    R.dump_json(
        out / "prepared.json",
        {
            "train_patients": ["a"],
            "selection_patient": "b",
            "test_patient": "b",
            "wrong_source_patient": "a",
            "selection_panel": "b_holdout",
            "panels": {"a_question": "a_question.pt", "b_holdout": "b_holdout.pt"},
            "panel_patients": {"a_question": "a", "b_holdout": "b"},
            "controls": {"native_b": str(cart), "fixed_a": str(cart)},
            "sink": {
                "patient": "a",
                "path": str(cart),
                "native_tensor_digest": B.digest_native(loaded),
            },
            "sources": {p: dataclasses.asdict(s) for p, s in sources.items()},
            "memory_bank_rows": 5,
        },
    )
    return out


def test_tiny_and_profile_loops_train_only_the_writer(tmp_path: Path):
    spec, writer, reader = _models()
    prepared = _prepared(tmp_path, spec, reader.model)
    cfg = DS.DistillConfig(
        run_dir=str(tmp_path / "run"),
        prepared_dir=str(prepared),
        mode="tiny",
        tiny_patient="a",
        tiny_groups=4,
        updates=3,
        examples_per_update=2,
        validate_every=3,
        device="cpu",
        model=SPEC,
        seed=1,
    )
    summary = DS.train(cfg)
    assert summary["reader_unchanged"] and summary["updates_run"] == 3
    assert set(summary["validations"]) == {"0", "3"}
    v0, v3 = summary["validations"]["0"], summary["validations"]["3"]
    assert (
        v0["tiny_fit"]["generated"]["tokens"]
        == v3["tiny_fit"]["generated"]["tokens"]
        > 0
    )
    assert summary["selected_checkpoint"] in {"0", "3"}
    assert summary["exposure"]["unique_examples_seen"] <= 4
    rows = [
        json.loads(l)
        for l in (tmp_path / "run" / "train_log.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 3 and all(np.isfinite(r["loss_per_token"]) for r in rows)
    ckpt = torch.load(tmp_path / "run" / "ckpt_3.pt", weights_only=False)
    assert "memory" in ckpt["trainable"] and ckpt["update"] == 3
    # pilot-style validation with controls and the wrong-source arm
    cfg = dataclasses.replace(
        cfg, mode="pilot", run_dir=str(tmp_path / "pilot"), updates=2, validate_every=2
    )
    summary = DS.train(cfg)
    val = summary["validations"]["2"]["b_holdout"]
    assert {"generated", "native_b", "fixed_a", "wrong_source_generated"} <= set(val)
    assert val["native_b"]["rows"] == 4 and val["generated"]["tokens"] > 0
    # profile mode reports timing only
    cfg = dataclasses.replace(
        cfg,
        mode="profile",
        run_dir=str(tmp_path / "prof"),
        profile_warmup=1,
        profile_timed=2,
    )
    summary = DS.train(cfg)
    assert (
        summary["profile"]["timed_updates"] == 2
        and "selected_checkpoint" not in summary
    )
    # export puts the sink first and the generated rows behind it
    source = DS.Source(**R.load_json(prepared / "prepared.json")["sources"]["b"])
    sink = [
        (k[:, :, :1], v[:, :, :1])
        for k, v in B.load_native_cartridge(
            prepared / "native_a.pt", dtype=torch.float32
        ).prefix("cpu")
    ]
    prefix, start = DS.export_generated(writer, reader, source, sink, torch.float32)
    assert start == 6 and prefix[0][0].shape[2] == 6
    assert torch.equal(prefix[0][0][:, :, :1], sink[0][0])
    other, start2 = DS.export_generated(
        writer, reader, source, sink, torch.float32, rows_override=4
    )
    assert start2 == 4 and other[0][0].shape[2] == 4

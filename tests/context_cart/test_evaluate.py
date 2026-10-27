# SPDX-License-Identifier: MIT
"""The stock library generation control must reproduce the cached text-arm
path on a tiny random model that shares the real tokenizer."""

from __future__ import annotations

import pytest
import torch

from research.context_cart import dataset as D
from research.context_cart import evaluate as E
from research.context_cart import model as M


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B",
            revision="c1899de289a04d12100db370d81485cdf75e47ca",
            local_files_only=True,
        )
    except Exception as error:  # pragma: no cover - environment dependent
        pytest.skip(f"pinned tokenizer unavailable: {error}")


def test_stock_control_matches_cached_text_arm(tok):
    spec = M.ModelSpec(name="tiny:151936", revision="0" * 40, dtype="float32")
    base = M.load_base(spec, "cpu")
    eos = (int(tok.eos_token_id), int(tok.pad_token_id))
    reader = M.Reader(base, eos_ids=eos, pad_id=int(tok.pad_token_id))
    worlds = [
        w.to_dict()
        for w in D.generate_pool(
            "calibration", D.Tokenizer(tok), D.NameRegistry(), count=2
        )
    ]
    items = E.items_from_worlds(worlds)[:4]
    batched, _ = E.answer_text_arm(reader, tok, "full", worlds, items, 2, 4)
    summary, rows = E.stock_control(reader, tok, "full", worlds, items, 4, limit=4)
    assert summary["items"] == 4
    assert summary["ids_equal"] == 4, rows
    by_key = {(r["world_id"], r["qid"]): r for r in batched}
    for row in rows:
        assert by_key[(row["world_id"], row["qid"])]["pred_ids"] == row["cached_ids"]
    assert all(
        torch.isfinite(torch.tensor([float(r["prompt_tokens"])])).all() for r in rows
    )

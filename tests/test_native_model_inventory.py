"""Frozen inventory counts admit explicit shapes and still reject drift."""

from types import SimpleNamespace

import pytest
import torch

from tools.kv.native.instrument import configure


class UnquantizedLinearMethod:
    pass


def model():
    value = torch.nn.Module()
    value.model = torch.nn.Module()
    value.model.layers = torch.nn.ModuleList()
    for _ in range(2):
        layer = torch.nn.Module()
        for index in range(4):
            linear = torch.nn.Linear(2, 2, bias=False, dtype=torch.float16)
            linear.quant_method = UnquantizedLinearMethod()
            layer.add_module("linear" + str(index), linear)
        layer._k_scale = torch.ones((), dtype=torch.float32)
        layer._v_scale = torch.ones((), dtype=torch.float32)
        layer._q_scale = torch.ones((), dtype=torch.float32)
        layer.kv_cache = torch.zeros(2, 2, 16, 2, 64, dtype=torch.float16)
        layer.impl = SimpleNamespace()
        value.model.layers.append(layer)
    return value


def test_explicit_inventory_and_mismatched_count(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)
    expectation = {
        "format": "fp16",
        "attention_layers": 2,
        "linear_modules": 8,
        "embedding_modules": 0,
    }
    result = configure(model(), {}, "k16_v16", tmp_path, expectation)
    assert result["layers"] == 2 and result["cache_dtypes"] == ["torch.float16"]
    with pytest.raises(RuntimeError, match="inventory is incomplete"):
        configure(
            model(), {}, "k16_v16", tmp_path, {**expectation, "attention_layers": 3}
        )
    with pytest.raises(RuntimeError, match="inventory is incomplete"):
        configure(
            model(), {}, "k16_v16", tmp_path, {**expectation, "linear_modules": 9}
        )

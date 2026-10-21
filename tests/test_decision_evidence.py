# SPDX-License-Identifier: MIT
"""Catch corrupted serialized evidence without depending on a model result."""

import copy

import pytest
import torch
from transformers.cache_utils import Cache

from tools.kv.decision.cache import Layer, inventory
from tools.kv.decision.evidence import generation, storage


CONFIG = {
    "num_hidden_layers": 1,
    "num_key_value_heads": 2,
    "hidden_size": 16,
    "num_attention_heads": 2,
}


def physical(arm):
    layer = Layer(arm, 0, {"K": 0.01, "V": 0.02})
    c = Cache(layers=[layer])
    x = (
        torch.arange(112 * 2 * 8, dtype=torch.float32).reshape(1, 2, 112, 8).half()
        / 1000
    )
    layer.update(x, x)
    x = (
        torch.arange(128 * 2 * 8, dtype=torch.float32).reshape(1, 2, 128, 8).half()
        / 1000
    )
    layer.update(x, x)
    return inventory(c)


def test_accounting_from_real_tensor_storages_and_reverse_allocation():
    for policy in ["F", "D"]:
        a = storage(physical("A" + policy), "A" + policy, CONFIG, "float16", 112)
        r = storage(physical("R" + policy), "R" + policy, CONFIG, "float16", 112)
        s = storage(physical("S" + policy), "S" + policy, CONFIG, "float16", 112)
        assert a == r
        assert a["allocation_count"] == 3 and s["allocation_count"] == 4
        assert (
            s["persistent_requested_backing_bytes"]
            < a["persistent_requested_backing_bytes"]
        )
    assert (
        storage(physical("N"), "N", CONFIG, "float16", 112)["scale_backing_bytes"] == 0
    )


@pytest.mark.parametrize(
    "corruption", ["alias", "dtype", "shape", "bytes", "total", "scale", "overlap"]
)
def test_corrupted_cache_metadata_is_rejected(corruption):
    value = copy.deepcopy(physical("AD"))
    row = value["tensors"][0]
    if corruption == "alias":
        value["tensors"][1]["storage_id"] = row["storage_id"]
    elif corruption == "dtype":
        row["dtype"] = "torch.float8_e4m3fn"
    elif corruption == "shape":
        row["shape"][2] += 1
    elif corruption == "bytes":
        row["backing_bytes"] += 512
    elif corruption == "total":
        value["unique_backing_bytes"] -= 1
    elif corruption == "scale":
        value["scale_logical_bytes"] = 0
    else:
        row["stride"][2] = 0
    with pytest.raises(ValueError):
        storage(value, "AD", CONFIG, "float16", 112)


class Decoder:
    def decode(self, tokens, *, skip_special_tokens):
        assert skip_special_tokens
        return "".join(map(str, tokens))


def test_decoding_and_original_termination_flags_are_bound_to_tokens():
    value = {
        "tokens": [1, 2],
        "text": "12",
        "output_tokens": 2,
        "eos": True,
        "truncated": False,
        "error": None,
    }
    generation(value, Decoder(), 2048, [9])
    for changed in [
        dict(text="13"),
        dict(output_tokens=3),
        dict(tokens=[1, 9]),
        dict(eos=False),
        dict(truncated=True),
    ]:
        with pytest.raises(ValueError):
            generation(value | changed, Decoder(), 2048, [9])

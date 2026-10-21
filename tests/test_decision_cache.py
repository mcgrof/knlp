# SPDX-License-Identifier: MIT
import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import Cache
from tools.kv.decision.oracle import oracle
from tools.kv.decision.cache import ARMS, Layer, decode, encode, inventory, make_cache


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fixed", [False, True])
def test_independent_codebook(dtype, fixed):
    torch.manual_seed(42)
    x = (torch.randn(2, 3, 9, 32) * 4).to(dtype).transpose(1, 2)  # noncontiguous
    scale = torch.tensor(0.01, dtype=torch.float32) if fixed else None
    q, s = encode(x, scale)
    np.testing.assert_array_equal(q.view(torch.uint8).numpy(), oracle(x, s.numpy()))
    assert q.element_size() == 1 and s.dtype == torch.float32
    assert torch.equal(decode(q, s, dtype), (q.float() * s).to(dtype))
    z, zs = encode(torch.zeros_like(x), scale)
    assert torch.isfinite(decode(z, zs, dtype)).all() and torch.all(z.float() == 0)


@pytest.mark.parametrize("arm", ARMS)
def test_incremental_and_physical_storage(arm):
    torch.manual_seed(7)
    k = torch.randn(2, 3, 11, 32).half()
    v = torch.randn_like(k) * 2
    fixed = {"K": 0.02, "V": 0.03} if arm.endswith("F") else None
    whole = Layer(arm, 0, fixed)
    chunk = Layer(arm, 0, fixed)
    wk, wv = whole.update(k, v)
    chunk.update(k[:, :, :4], v[:, :, :4])
    oldk = chunk.keys.view(torch.uint8).clone()
    oldv = chunk.values.view(torch.uint8).clone()
    ck, cv = chunk.update(k[:, :, 4:7], v[:, :, 4:7])
    assert torch.equal(chunk.keys[:, :, :4].view(torch.uint8), oldk)
    assert torch.equal(chunk.values[:, :, :4].view(torch.uint8), oldv)
    ck, cv = chunk.update(k[:, :, 7:], v[:, :, 7:])
    assert torch.equal(wk, ck) and torch.equal(wv, cv)
    assert torch.equal(whole.keys.view(torch.uint8), chunk.keys.view(torch.uint8))
    assert torch.equal(whole.values.view(torch.uint8), chunk.values.view(torch.uint8))
    assert chunk.get_seq_length() == 11 and chunk.get_mask_sizes(3) == (14, 0)
    for attr, component in [("keys", "K"), ("values", "V")]:
        quantized = arm[0] == "S" or arm[0] == ("R" if component == "K" else "A")
        assert getattr(chunk, attr).dtype == (
            torch.float8_e4m3fn if quantized else torch.float16
        )
    inv = inventory(Cache(layers=[chunk]))
    assert inv["unique_backing_bytes"] == inv["logical_bytes"]
    expected_tensors = (
        {"keys", "values"}
        | ({"k_scale"} if arm[0] in "RS" else set())
        | ({"v_scale"} if arm[0] in "AS" else set())
    )
    assert {
        name for name, x in vars(chunk).items() if isinstance(x, torch.Tensor)
    } == expected_tensors
    scalars = int(arm[0] in "RS") + int(arm[0] in "AS")
    assert inv["scale_logical_bytes"] == scalars * 4 * (
        2 * 3 * 11 if arm.endswith("D") else 1
    )


@pytest.mark.parametrize("policy", ["F", "D"])
def test_reverse_equal_bytes(policy):
    x = torch.randn(2, 3, 5, 32).half()
    inventories = []
    for arm in ["A" + policy, "R" + policy]:
        l = Layer(arm, 0, {"K": 0.1, "V": 0.2} if policy == "F" else None)
        l.update(x, x)
        inventories.append(inventory(Cache(layers=[l])))
    assert (
        inventories[0]["unique_backing_bytes"] == inventories[1]["unique_backing_bytes"]
    )
    assert (
        inventories[0]["scale_logical_bytes"] == inventories[1]["scale_logical_bytes"]
    )


@pytest.mark.parametrize(
    "config_cls,model_cls",
    [(Qwen3Config, Qwen3ForCausalLM), (Qwen2Config, Qwen2ForCausalLM)],
)
def test_native_model_exact_multitoken_parity(config_cls, model_cls):
    torch.manual_seed(5)
    torch.set_num_threads(1)
    cfg = config_cls(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["full_attention"] * 2,
    )
    model = model_cls(cfg).half().eval()
    native = make_cache(cfg, "N")
    library = make_cache(cfg, "N", library=True)
    with torch.no_grad():
        for start, length in [(0, 7), (7, 3), (10, 1)]:
            x = torch.randint(0, 64, (2, length))
            position = torch.arange(start, start + length)[None, :]
            a = model(
                input_ids=x,
                past_key_values=native,
                position_ids=position,
                use_cache=True,
            ).logits
            b = model(
                input_ids=x,
                past_key_values=library,
                position_ids=position,
                use_cache=True,
            ).logits
            assert torch.equal(a, b)
            assert (
                inventory(native)["unique_backing_bytes"]
                == inventory(library)["unique_backing_bytes"]
            )


def test_sliding_trims_backing_and_keeps_old_codes():
    l = Layer("RD", 0, window=4)
    x = torch.randn(1, 2, 7, 8).half()
    a, b = l.update(x, x)
    assert a.shape[-2] == 7 and l.keys.shape[-2] == 3
    assert l.keys.untyped_storage().nbytes() == l.keys.numel()
    assert l.get_mask_sizes(2) == (5, 4)
    a, b = l.update(x[:, :, :2], x[:, :, :2])
    assert a.shape[-2] == 5


def test_nonfinite_and_missing_scale_rejected():
    with pytest.raises(RuntimeError):
        encode(torch.full((1, 1, 1, 8), float("nan")).half())
    with pytest.raises(ValueError):
        Layer("SF", 0)
    with pytest.raises(RuntimeError):
        encode(torch.ones(1, 1, 1, 8).half(), torch.tensor(0.0))


def test_calibration_maxima_and_full_operand_audit():
    from tools.kv.decision.calibration import Observer

    k = torch.tensor([[[[0.0, -2.0, 4.0, 1.0]]]], dtype=torch.float16)
    v = torch.tensor([[[[8.0, 0.0, -1.0, 2.0]]]], dtype=torch.float16)
    collect = Observer(1, capture=True)
    collect.begin({"id": "cal", "length": 1024})
    collect(0, k, v)
    collect(0, k * 2, v * 0.5)
    scales = collect.scales()
    assert scales == [{"K": 8 / 448, "V": 8 / 448}]
    assert len(collect.fixtures) == 4
    audit = Observer(1, fixed=scales)
    audit.begin({"id": "cal", "length": 1024})
    audit(0, k, v)
    audit(0, k * 2, v * 0.5)
    assert audit.scales() == scales
    rows = audit.summary()
    assert len(rows) == 4
    assert all(r["elements"] == 8 and r["clipped"] == 0 for r in rows)
    # Both arms use the identical component quantizer and calibrated scale.
    a = Layer("AF", 0, scales[0])
    s = Layer("SF", 0, scales[0])
    r = Layer("RF", 0, scales[0])
    a.update(k, v)
    s.update(k, v)
    r.update(k, v)
    assert torch.equal(a.values.view(torch.uint8), s.values.view(torch.uint8))
    assert torch.equal(r.keys.view(torch.uint8), s.keys.view(torch.uint8))

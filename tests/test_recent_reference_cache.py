"""Independent quantization, rolling ownership, and hybrid cache checks."""

import pytest
import torch
from transformers import Gemma4TextConfig, Qwen3_5TextConfig
from tools.kv.recent.cache import (
    encode,
    decode,
    make_cache,
    native_library_cache,
    inventory,
)


@pytest.mark.parametrize("window", [None, 5])
@pytest.mark.parametrize("mode", ["native", "k16v8", "k8v8"])
def test_chunked_cache_matches_independent_reference(mode, window):
    torch.manual_seed(143)
    config = Gemma4TextConfig(
        num_hidden_layers=2,
        layer_types=[
            "sliding_attention" if window else "full_attention",
            "full_attention",
        ],
        sliding_window=window or 5,
        num_kv_shared_layers=0,
    )
    cache = make_cache(config, mode)
    keys = torch.randn(2, 3, 13, 16, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    end = 0
    for width in [7, 1, 5]:
        start = end
        end += width
        lo = max(0, start - window + 1) if window else 0
        actual = cache.update(keys[..., start:end, :], values[..., start:end, :], 0)
        for side, x in enumerate([keys[..., lo:end, :], values[..., lo:end, :]]):
            quant = mode == "k8v8" or (side == 1 and mode == "k16v8")
            if quant:
                scale = (
                    torch.clamp(x.float().abs().amax(-1, keepdim=True), min=1e-12) / 448
                )
                expected = (
                    (x.float() / scale).to(torch.float8_e4m3fn).float() * scale
                ).bfloat16()
            else:
                expected = x
            torch.testing.assert_close(actual[side], expected, atol=0, rtol=0)
        assert cache.get_seq_length() == end
        audit = inventory(cache)
        assert audit["unique_backing_bytes"] == audit["logical_bytes"]
        if window:
            assert cache.layers[0].keys.shape[-2] == window - 1


def test_native_matches_library_and_preserves_hybrid_state_type():
    config = Qwen3_5TextConfig(
        num_hidden_layers=4, layer_types=["linear_attention"] * 3 + ["full_attention"]
    )
    ours, stock = make_cache(config, "native"), native_library_cache(config)
    assert type(ours.layers[0]) is type(stock.layers[0])
    x = torch.randn(1, 2, 8, 16, dtype=torch.bfloat16)
    for _ in range(3):
        a, b = ours.update(x, x, 3), stock.update(x, x, 3)
        for left, right in zip(a, b):
            torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_zero_and_memory_ratio_with_scale_overhead():
    config = Qwen3_5TextConfig(num_hidden_layers=1, layer_types=["full_attention"])
    x = torch.zeros(2, 4, 32, 256, dtype=torch.bfloat16)
    sizes = []
    for mode in ("native", "k16v8", "k8v8"):
        c = make_cache(config, mode)
        k, v = c.update(x, x, 0)
        assert torch.isfinite(k).all() and torch.isfinite(v).all()
        sizes.append(inventory(c)["unique_backing_bytes"])
    assert sizes[1] / sizes[0] == (3 * 256 + 4) / (4 * 256)
    assert sizes[2] / sizes[0] == (2 * 256 + 8) / (4 * 256)


@pytest.mark.parametrize("family", ["qwen", "gemma"])
def test_tiny_model_native_logits_and_quantized_decode(family, monkeypatch):
    from transformers.models.qwen3_5 import modeling_qwen3_5
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    # These logits checks run on CPU even when optional GPU kernels are
    # installed. Full checkpoint GPU qualification uses the official module
    # without these test-local patches.
    for name in (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
        "FusedRMSNormGated",
    ):
        monkeypatch.setattr(modeling_qwen3_5, name, None)
    monkeypatch.setattr(modeling_qwen3_5, "is_fast_path_available", False)
    torch.manual_seed(77)
    common = dict(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
    )
    if family == "qwen":
        config = Qwen3_5TextConfig(
            **common,
            layer_types=["linear_attention", "full_attention"],
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_num_key_heads=1,
            linear_num_value_heads=2,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "mrope_section": [2, 3, 3],
            }
        )
        model = Qwen3_5ForCausalLM(config).bfloat16().eval()
    else:
        config = Gemma4TextConfig(
            **common,
            layer_types=["sliding_attention", "full_attention"],
            global_head_dim=32,
            num_global_key_value_heads=1,
            sliding_window=8,
            hidden_size_per_layer_input=0,
            attention_k_eq_v=True,
            num_kv_shared_layers=0
        )
        model = Gemma4ForCausalLM(config).bfloat16().eval()
    ids = torch.randint(3, 128, (2, 13))
    continuations = [torch.randint(3, 128, (2, width)) for width in (1, 11, 3)]
    with torch.inference_mode():
        library_cache = native_library_cache(config)
        expected = model(
            input_ids=ids, past_key_values=library_cache, use_cache=True
        ).logits
        expected_continuations = [
            model(input_ids=part, past_key_values=library_cache, use_cache=True).logits
            for part in continuations
        ]
        for mode in ("native", "k16v8", "k8v8"):
            cache = make_cache(config, mode)
            actual = model(input_ids=ids, past_key_values=cache, use_cache=True).logits
            if mode == "native":
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for part, continuation in zip(continuations, expected_continuations):
                actual = model(
                    input_ids=part, past_key_values=cache, use_cache=True
                ).logits
                if mode == "native":
                    torch.testing.assert_close(actual, continuation, rtol=0, atol=0)
                assert torch.isfinite(actual).all()
            assert cache.get_seq_length() == 28


@pytest.mark.parametrize("mode", ["native", "k16v8", "k8v8"])
def test_text_nll_after_different_generation_batch(mode, monkeypatch):
    from transformers import Qwen3_5Config, Qwen3_5VisionConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForConditionalGeneration,
    )
    from tools.kv.recent.run import nll_position_kwargs

    for name in (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
        "FusedRMSNormGated",
    ):
        monkeypatch.setattr(modeling_qwen3_5, name, None)
    monkeypatch.setattr(modeling_qwen3_5, "is_fast_path_available", False)
    torch.manual_seed(231)
    text = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 1, 2],
        },
    )
    vision = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        out_hidden_size=32,
    )
    config = Qwen3_5Config(text_config=text.to_dict(), vision_config=vision.to_dict())
    model = Qwen3_5ForConditionalGeneration(config).bfloat16().eval()
    ids = torch.randint(3, 128, (1, 17))

    def continuation(explicit):
        cache = make_cache(config, mode)
        for start, end in [(0, 13), (13, 17)]:
            part = ids[:, start:end]
            kwargs = (
                nll_position_kwargs("Qwen/Qwen3.8-27B", part, start) if explicit else {}
            )
            logits = model(
                input_ids=part, past_key_values=cache, use_cache=True, **kwargs
            ).logits
        return logits

    with torch.inference_mode():
        # A clean model has no model-level offsets from a previous request.
        expected = continuation(False)
        model.generate(
            input_ids=ids[:, :13].expand(4, -1),
            attention_mask=torch.ones(4, 13, dtype=torch.long),
            past_key_values=make_cache(config, "native"),
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=0,
        )
        assert model.model.rope_deltas.shape == (4, 1)
        actual = continuation(True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.isfinite(actual).all()
        assert nll_position_kwargs("google/gemma-4-26B-A4B-it", ids, 0) == {}

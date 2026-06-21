"""The pre-bias FP8 path quantizes the residual (K - b_K) then adds b_K back. On a biasless model
b_K == 0, so the subtract/add must be an exact no-op: quant(K - 0) + 0 == quant(K). If this ever
diverges, the pre-bias 'fix' is mutating biasless models -- a silent confound across the atlas."""

import torch

import _fixtures as fx

import k_bias_common as kbc
from tools.kv.fp8_failure import architecture_discovery as AD


def test_biasless_kbias_is_zero_vector():
    infos = AD.discover(fx.tiny_llama())
    info = infos[0]
    assert info["k_bias"] is None
    v = kbc.k_bias_vector(info)
    assert v.shape[0] == info["n_kv_heads"] * info["head_dim"]
    assert torch.count_nonzero(v) == 0


def test_prebias_subtract_addback_is_exact_noop():
    torch.manual_seed(0)
    k = torch.randn(1, 2, 5, 8) * 3.0  # [B, n_kv, T, head_dim]
    b = torch.zeros(8)  # biasless => zero per-channel bias
    spec = kbc.parse_spec("fp8:per_tensor")
    direct = kbc._quant_lastdims(k, spec["fmt"], spec["bits"], spec["layout"], spec["group"], False)
    prebias = (
        kbc._quant_lastdims(k - b, spec["fmt"], spec["bits"], spec["layout"], spec["group"], False)
        + b
    )
    assert torch.equal(direct, prebias)


def test_flexkv_prebias_on_biasless_matches_clean_bf16():
    # Documented harness behavior: with prebias=True and no bias, no k-proj hook is installed and
    # the post-RoPE K-quant branch is skipped, so the model is bit-identical to clean bf16.
    m = fx.tiny_llama()
    infos = AD.discover(m)
    ids = torch.randint(0, 64, (1, 7))
    with torch.no_grad():
        clean = m(ids).logits
    k_spec, v_spec = kbc.parse_spec("fp8:per_tensor"), kbc.parse_spec("bf16")
    with kbc.FlexKVHarness(m, infos, k_spec, v_spec, prebias=True):
        with torch.no_grad():
            got = m(ids).logits
    assert torch.equal(clean, got)

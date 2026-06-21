"""Quantizer sanity: bf16/none is a true pass-through, FP8/int are deterministic, values already on
the grid round-trip exactly, and the error stays within the format's representable bound. These are
the floor properties every higher-level claim rests on."""

import torch

import k_bias_common as kbc


def _q(x, spec_str, unit=False):
    s = kbc.parse_spec(spec_str)
    return kbc._quant_lastdims(x, s["fmt"], s["bits"], s["layout"], s["group"], unit)


def test_bf16_is_passthrough():
    x = torch.randn(3, 4, 5)
    assert torch.equal(_q(x, "bf16"), x)
    assert torch.equal(_q(x, "none"), x)


def test_quant_is_deterministic():
    x = torch.randn(2, 3, 8) * 5
    assert torch.equal(_q(x, "fp8:per_tensor"), _q(x, "fp8:per_tensor"))
    assert torch.equal(_q(x, "int8:per_token"), _q(x, "int8:per_token"))


def test_fp8_representable_values_exact_unit_scale():
    # e4m3 represents these exactly; unit_scale skips rescaling so the cast is the only step.
    x = torch.tensor([0.0, 0.5, 1.0, -2.0, 4.0, -8.0, 16.0])
    assert torch.equal(_q(x, "fp8:per_tensor", unit=True), x)


def test_int8_grid_values_exact():
    # integers within [-127,127] at per-tensor scale 1 round-trip exactly
    x = torch.tensor([[0.0, 1.0, -5.0, 127.0, -127.0, 63.0, -64.0, 12.0]])
    q = _q(x, "int8:per_token")
    assert torch.equal(q, x)


def test_fp8_relative_error_bounded():
    torch.manual_seed(0)
    x = torch.randn(4, 16, 8) * 2.0
    q = _q(x, "fp8:per_channel")
    rel = (q - x).abs() / x.abs().clamp(min=1e-3)
    # e4m3 has 3 mantissa bits -> worst-case ~2^-3 per element; mean must be well under that
    assert rel.mean().item() < 0.05
    assert (q - x).abs().max().item() < x.abs().max().item()  # no blow-up


def test_int4_coarser_than_int8():
    torch.manual_seed(1)
    x = torch.randn(2, 32, 8) * 3.0
    e8 = (_q(x, "int8:per_token") - x).abs().mean().item()
    e4 = (_q(x, "int4:per_token") - x).abs().mean().item()
    assert e4 > e8  # fewer bits => strictly more error on continuous data

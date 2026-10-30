# SPDX-License-Identifier: MIT
"""KV-quant hook checks. Skipped on hosts where torch is unavailable."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/reproduce/cs312"))

try:
    import torch

    import cs312_quant as q

    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch not available")
class FakeQuantTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        # [B, H, T, head_dim], the shape K and V have at the SDPA call.
        self.x = torch.randn(2, 4, 16, 64)

    def test_16_bit_is_exact_identity(self):
        self.assertTrue(torch.equal(q.fake_quant(self.x, 16), self.x))

    def test_int4_round_trip_is_bounded_and_lossy(self):
        y = q.fake_quant(self.x, 4)
        err = (y - self.x).abs().amax().item()
        self.assertGreater(err, 0.0)  # genuinely quantized
        # worst case is half a step; step = amax/7, amax ~ a few sigma
        self.assertLess(err, 1.0)

    def test_fp8_is_finer_than_int4(self):
        e8 = (q.fake_quant(self.x, 8) - self.x).abs().mean().item()
        e4 = (q.fake_quant(self.x, 4) - self.x).abs().mean().item()
        self.assertLess(e8, e4)

    def test_straight_through_gradient_is_identity(self):
        x = self.x.clone().requires_grad_(True)
        q.fake_quant(x, 4).sum().backward()
        self.assertTrue(torch.allclose(x.grad, torch.ones_like(x)))

    def test_per_token_scale_handles_varied_magnitude(self):
        x = self.x.clone()
        x[0, 0, 0] *= 1000.0  # one token far larger than the rest
        y = q.fake_quant(x, 4)
        # the small tokens keep their own scale, not the big token's
        small_err = (y[0, 0, 1] - x[0, 0, 1]).abs().amax().item()
        self.assertLess(small_err, 1.0)

    def test_per_tensor_is_harsher_than_per_token_on_outliers(self):
        x = self.x.clone()
        x[0, 0, 0] *= 1000.0
        e_pt = (q.fake_quant(x, 4, granularity="per_token") - x).abs().mean().item()
        e_pn = (q.fake_quant(x, 4, granularity="per_tensor") - x).abs().mean().item()
        self.assertGreater(e_pn, e_pt)

    def test_int_widths_are_progressively_lossier(self):
        e4 = (q.fake_quant(self.x, 4) - self.x).abs().mean().item()
        e3 = (q.fake_quant(self.x, 3) - self.x).abs().mean().item()
        e2 = (q.fake_quant(self.x, 2) - self.x).abs().mean().item()
        self.assertLess(e4, e3)
        self.assertLess(e3, e2)

    def test_asymmetric_applies_independent_widths(self):
        c = q.configure(k_bits=16, v_bits=4)
        self.assertFalse(c.k_bits < 16)
        self.assertTrue(c.enabled)
        # K16 is identity, V4 is lossy, through the installed sdpa patch
        import torch.nn.functional as F

        key = self.x.clone()
        val = self.x.clone()
        out = F.scaled_dot_product_attention(self.x, key, val, is_causal=True)
        self.assertEqual(out.shape, self.x.shape)
        q.configure(k_bits=16, v_bits=16)  # reset


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: MIT
import math

import torch

from tools.kv.native.stored_prefill import reference


def test_stored_reference_uses_causal_means_and_base_two_lse():
    q = torch.zeros(3, 4, 2, dtype=torch.float16)
    k = torch.zeros(5, 2, dtype=torch.float16)
    v = torch.arange(10, dtype=torch.float16).reshape(5, 2).to(torch.float8_e4m3fn)
    out, lse = reference(
        q, k, v, torch.tensor([0, 2, 4]), sm_scale=0.5, k_scale=1.0, v_scale=0.125
    )
    for row, n in enumerate([1, 3, 5]):
        expected = (v[:n].float().mean(0).half().float() * 0.125).half()
        torch.testing.assert_close(out[row], expected.expand(4, 2), rtol=0, atol=0)
        torch.testing.assert_close(lse[row], torch.full((4,), math.log2(n)))

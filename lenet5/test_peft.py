# SPDX-License-Identifier: MIT
"""Smoke test for the LeNet-5 pedagogical adaptation modes.

Instantiates each of full / lora / prefix on CPU, runs a forward and backward
pass on a tiny batch, and asserts the trainable-parameter ordering that makes
the teaching point: full >> lora >> prefix.

Run directly:
    python3 lenet5/test_peft.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from lenet5.peft import build_lenet5_adaptation, count_params


def _forward_backward(model: nn.Module) -> torch.Tensor:
    x = torch.randn(2, 1, 32, 32)
    out = model(x)
    assert out.shape == (2, 10), f"expected [2, 10], got {tuple(out.shape)}"
    loss = out.pow(2).mean()
    loss.backward()
    return out


def main() -> int:
    torch.manual_seed(0)

    counts = {}
    for mode in ("full", "lora", "prefix"):
        model = build_lenet5_adaptation(
            num_classes=10,
            adaptation_mode=mode,
            lora_rank=4,
            prefix_rows=4,
            train_head=True,
        )
        _forward_backward(model)
        trainable, total = count_params(model)
        assert trainable > 0, f"{mode}: no trainable params (inert model)"
        counts[mode] = (trainable, total)
        pct = 100.0 * trainable / total
        print(f"{mode:6s}  trainable={trainable:6d}  total={total:6d}  ({pct:5.2f}%)")

    # The pedagogical punchline: same skeleton, shrinking trainable budgets.
    assert (
        counts["full"][0] > counts["lora"][0]
    ), "full must train more params than lora"
    assert (
        counts["lora"][0] > counts["prefix"][0]
    ), "lora must train more params than prefix"

    # A bad mode must fail loudly, not silently fall through.
    try:
        build_lenet5_adaptation(adaptation_mode="banana")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown adaptation_mode should raise ValueError")

    print("OK: full > lora > prefix, all modes forward/backward on CPU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

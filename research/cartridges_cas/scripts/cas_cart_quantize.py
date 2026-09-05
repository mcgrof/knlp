#!/usr/bin/env python3
"""Round-trip a cartridge's keys and values through an 8-bit float grid.

A serving runtime that stores values in FP8 does not give the model
something different to attend to; it gives it the same tensor snapped to a
coarser grid. So the question "does an 8-bit cache cost a learned cartridge
anything" can be asked without any kernel at all: quantize the trained
tensors to the grid, expand them back to bfloat16, and score the result.
What comes back is exactly what an asymmetric or symmetric runtime would
compute with, up to the accumulation order inside the kernel.

This makes that artifact. It is deliberately paired: one training, two
readings, so nothing here carries training-to-training variance, which on
this recipe is the largest term in any comparison.

The frozen sink tokens are quantized alongside the trainable ones, because
a runtime quantizes whatever is in the cache and does not know which rows
were trained.

Grids:
    v8      values on the grid, keys untouched -- the asymmetric layout
    k8v8    both on the grid -- the symmetric layout
    k8      keys only, as the control that isolates which half pays

Env:
    IN       cartridge to read
    OUT      cartridge to write
    GRID     v8 (default), k8v8, or k8
    FORMAT   e4m3 (default) or e5m2
    SCALE    per-tensor (default), per-head, or none
    REPORT   1 to print the relative error per layer

Prints ``CART_QUANT,<grid>,<format>,<scale>,<median rel err>,<max rel err>``
and ``CART_QUANT_DONE``.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cas_cart_init import load_cart, split_cart  # noqa: E402

IN = os.environ["IN"]
OUT = os.environ["OUT"]
GRID = os.environ.get("GRID", "v8")
FORMAT = os.environ.get("FORMAT", "e4m3")
SCALE = os.environ.get("SCALE", "per-tensor")
REPORT = os.environ.get("REPORT", "1") == "1"

DTYPE = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[FORMAT]
# The 8-bit float types map anything above their finite range to NaN rather
# than to an infinity, so the clamp below is load-bearing rather than tidy.
FMAX = {"e4m3": 448.0, "e5m2": 57344.0}[FORMAT]


def _quantize(t):
    """bfloat16 -> 8-bit grid -> bfloat16, at the requested scale."""
    x = t.float()
    if SCALE == "none":
        scale = torch.ones((), device=x.device)
    elif SCALE == "per-tensor":
        scale = x.abs().amax().clamp(min=1e-12) / FMAX
    elif SCALE == "per-head":
        # [1, H, T, D] -> one scale per key/value head
        scale = x.abs().amax(dim=(0, 2, 3), keepdim=True).clamp(min=1e-12) / FMAX
    else:
        raise SystemExit(f"unknown SCALE {SCALE}")
    q = (x / scale).clamp(-FMAX, FMAX).to(DTYPE)
    return (q.float() * scale).to(t.dtype)


def _relerr(a, b):
    d = (a.float() - b.float()).norm()
    n = a.float().norm().clamp(min=1e-12)
    return float(d / n)


def main():
    keys, values, nfrozen = load_cart(IN)
    do_k = GRID in ("k8", "k8v8")
    do_v = GRID in ("v8", "k8v8")
    errs = []
    for i in range(len(keys)):
        if do_k:
            q = _quantize(keys[i])
            errs.append(_relerr(keys[i], q))
            keys[i] = q
        if do_v:
            q = _quantize(values[i])
            errs.append(_relerr(values[i], q))
            values[i] = q
    torch.save(split_cart(keys, values, nfrozen), OUT)
    errs.sort()
    med = errs[len(errs) // 2] if errs else 0.0
    mx = errs[-1] if errs else 0.0
    if REPORT:
        print(f"CART_QUANT,{GRID},{FORMAT},{SCALE},{med:.6g},{mx:.6g}", flush=True)
    print("CART_QUANT_DONE", flush=True)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: MIT
"""Observe the native cache-write operands; fixed scales never use holdout."""
import math
import torch
from .cache import encode, decode


class Observer:
    def __init__(self, layers, fixed=None, capture=False):
        self.maxima = [{} for _ in range(layers)]
        self.fixed = fixed
        self.capture = capture
        self.metrics = {}
        self.fixtures = {}
        self.sample = None
        self.calls = {}
        self.captured_lengths = set()
        self.capture_sample = False

    def begin(self, sample):
        self.sample = sample
        self.calls = {}
        self.capture_sample = (
            self.capture and sample["length"] not in self.captured_lengths
        )
        self.captured_lengths.add(sample["length"])

    def __call__(self, index, k, v):
        step = self.calls.get(index, 0)
        self.calls[index] = step + 1
        for component, x in [("K", k), ("V", v)]:
            torch._assert_async(
                torch.isfinite(x).all(), "Nonfinite calibration operand"
            )
            maximum = x.float().abs().amax()
            old = self.maxima[index].get(component)
            self.maxima[index][component] = (
                maximum if old is None else torch.maximum(old, maximum)
            )
            if self.capture_sample and step in (0, 1, 128):
                key = f'{self.sample["id"]}/layer-{index}/{component}/step-{step}'
                self.fixtures[key] = x[:, :2, :4, :].detach().cpu().clone()
            if self.fixed is None:
                continue
            for policy in ["fixed", "dynamic"]:
                scale = (
                    torch.tensor(
                        self.fixed[index][component],
                        device=x.device,
                        dtype=torch.float32,
                    )
                    if policy == "fixed"
                    else None
                )
                codes, scale = encode(x, scale)
                y = decode(codes, scale, x.dtype)
                error = (y.float() - x.float()).double()
                normalized = x.float().abs() / scale
                row = torch.stack(
                    [
                        torch.tensor(x.numel(), device=x.device, dtype=torch.float64),
                        (normalized > 448).sum().double(),
                        (codes.float().abs() == 448).sum().double(),
                        error.abs().sum(),
                        error.square().sum(),
                        error.abs().max(),
                        maximum.double(),
                    ]
                )
                key = (index, component, policy)
                if key in self.metrics:
                    old = self.metrics[key]
                    row = torch.cat(
                        [old[:5] + row[:5], torch.maximum(old[5:], row[5:])]
                    )
                self.metrics[key] = row

    def scales(self):
        return [
            {c: max(float(row[c]), 1e-12) / 448 for c in ["K", "V"]}
            for row in self.maxima
        ]

    def summary(self):
        out = []
        for (index, component, policy), tensor in sorted(self.metrics.items()):
            n, sat, maxcodes, absolute, square, maximum, maxx = tensor.cpu().tolist()
            out.append(
                dict(
                    layer=index,
                    component=component,
                    policy=policy,
                    elements=int(n),
                    clipped=int(sat),
                    max_magnitude_codes=int(maxcodes),
                    mean_absolute_error=absolute / n,
                    rms_error=math.sqrt(square / n),
                    max_absolute_error=maximum,
                    max_operand=maxx,
                )
            )
        return out

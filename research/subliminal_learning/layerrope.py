# SPDX-License-Identifier: MIT
"""Depth-conditioned normalization gains from arXiv:2610.09179, Eq. 1.

Rotate the shared gain vector, not the activations. Adjacent real gain
entries form complex pairs. This MNIST adaptation follows the paper's
Vision Transformer setting: condition existing pre-LayerNorm gains and
retain each layer's centering and bias. The same gain generator can also
gate a block output without output normalization. Each site and packed
model has its own gain and four schedule parameters, shared across depth.
"""

import torch
from torch import nn


class LayerRoPE(nn.Module):
    def __init__(
        self,
        n_models,
        width,
        depth,
        base=100.0,
        mag_slope=0.0,
        rot_slope=-0.5,
        mag_intercept=0.0,
        rot_intercept=0.0,
    ):
        super().__init__()
        if width % 2 or width < 2 or depth < 1 or base <= 0:
            raise ValueError("LayerRoPE needs even width, positive depth and base")
        self.depth, self.base = depth, base
        self.weight = nn.Parameter(torch.ones(n_models, width))
        for name, value in (
            ("alpha_mag", mag_intercept),
            ("beta_mag", mag_slope),
            ("alpha_rot", rot_intercept),
            ("beta_rot", rot_slope),
        ):
            setattr(self, name, nn.Parameter(torch.full((n_models,), float(value))))

    def forward(self):
        """Return [models, depth, width] real gains, computed once per forward."""
        width = self.weight.shape[-1]
        opts = {"device": self.weight.device, "dtype": self.weight.dtype}
        log_depth = torch.arange(1, self.depth + 1, **opts).log()[None, :]
        freq = self.base ** (-2 * torch.arange(width // 2, **opts) / width)
        magnitude = (self.alpha_mag[:, None] + self.beta_mag[:, None] * log_depth).exp()
        angle = (self.alpha_rot[:, None] + self.beta_rot[:, None] * log_depth).exp()[
            ..., None
        ] * freq
        real, imag = self.weight[:, None, 0::2], self.weight[:, None, 1::2]
        cosine, sine = angle.cos(), angle.sin()
        rotated = torch.stack(
            (real * cosine - imag * sine, real * sine + imag * cosine), -1
        ).flatten(-2)
        return magnitude[..., None] * rotated

# SPDX-License-Identifier: GPL-2.0
"""Cross-model KV cache translation: map a source model's cache into a target's.

Both models stay frozen.  The question is whether a prefix cached by a small
model can be handed to a larger one cheaply enough, and faithfully enough, to
beat re-prefilling that prefix on the target.

This is deliberately a separate lane from the receiver-metric screen.  That
screen asked whether a Jacobian second moment ranks cache damage better than
cheap incumbents and concluded no; it never tested a source-to-target mapper,
which is what this does.  The ordering here is structured affine first, with
attention-local weighting and nonlinear refinement behind it, and no global
receiver metric attached.

The metric that decides anything is target *behaviour* -- held-out continuation
divergence and task retention -- never cache reconstruction error, which the
prior lane showed ranks damage poorly on its own.
"""

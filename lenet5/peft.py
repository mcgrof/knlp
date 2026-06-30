# SPDX-License-Identifier: MIT
"""Pedagogical adaptation (PEFT) mechanisms for LeNet-5.

LeNet-5 is small enough to be a teaching lab for the three families of
parameter-efficient adaptation, so the same skeleton can be trained three
ways and the trainable-parameter count tells the story:

    full    train every weight (ordinary fine-tuning)
    lora    freeze the base, train low-rank adapters on the FC layers
    prefix  freeze the base, train a learned strip added to the input image

Honesty tax: LeNet-5 has no attention, no keys/values, no KV cache, so it
cannot demonstrate literal transformer prefix tuning. The "prefix" mode here
is a CNN analogy -- a learned soft image prefix / visual prompt. Real prefix
tuning belongs in a tiny transformer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lenet5.model import LeNet5


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True


def count_params(module: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return trainable, total


class LoRALinear(nn.Module):
    """Minimal LoRA wrapper for nn.Linear.

    Original Linear:
        y = x W^T + b
    LoRA version:
        y = frozen_linear(x) + scale * B(A(x))

    Only A and B train; B starts at zero so the adapter is a no-op at init.
    """

    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()

        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")

        self.base = base
        freeze_module(self.base)

        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        self.A = nn.Linear(base.in_features, rank, bias=False)
        self.B = nn.Linear(rank, base.out_features, bias=False)

        # Start with no adapter effect: A random, B zero.
        nn.init.kaiming_uniform_(self.A.weight, a=5**0.5)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.base(x) + self.scale * self.B(self.A(x))


def replace_linear_with_lora(
    model: nn.Module,
    module_names: tuple[str, ...] = ("fc", "fc1"),
    rank: int = 4,
    alpha: float = 1.0,
) -> None:
    """Replace selected top-level Linear modules on LeNet-5 with LoRA wrappers.

    For the current LeNet5, the useful names are:
        fc   : 400 -> 120
        fc1  : 120 -> 84
        fc2  :  84 -> num_classes

    Convs are left alone for the first pedagogical pass -- keep the example
    brutally obvious.
    """

    for name in module_names:
        child = getattr(model, name)
        setattr(model, name, LoRALinear(child, rank=rank, alpha=alpha))


class ImagePrefixLeNet5(nn.Module):
    """CNN-friendly analogy to prefix / prompt tuning.

    Freezes LeNet-5 and trains only a small learned strip added to the top
    rows of the input image. This is NOT transformer KV prefix tuning -- it is
    a visual soft prompt / image-prefix teaching example.
    """

    def __init__(self, base: LeNet5, prefix_rows: int = 4):
        super().__init__()

        if prefix_rows <= 0 or prefix_rows > 32:
            raise ValueError("prefix_rows must be in [1, 32] for 32x32 MNIST input")

        self.base = base
        freeze_module(self.base)

        self.prefix_rows = prefix_rows
        self.prefix = nn.Parameter(torch.zeros(1, 1, prefix_rows, 32))

    def forward(self, x):
        # x is [batch, 1, 32, 32]; add the learned prefix to the top rows.
        x = x.clone()
        x[:, :, : self.prefix_rows, :] = x[:, :, : self.prefix_rows, :] + self.prefix
        return self.base(x)


def build_lenet5_adaptation(
    num_classes: int = 10,
    adaptation_mode: str = "full",
    lora_rank: int = 4,
    lora_alpha: float = 1.0,
    lora_modules: tuple[str, ...] = ("fc", "fc1"),
    prefix_rows: int = 4,
    train_head: bool = True,
    base_checkpoint: str | None = None,
) -> nn.Module:
    """Build LeNet-5 in one of three pedagogical adaptation modes.

        full    train all parameters
        lora    freeze base, train LoRA adapters, optionally train the head
        prefix  freeze base, train a learned image prefix, optionally the head

    For lora/prefix a base_checkpoint is recommended: freezing a random base
    and training only tiny adapters is educational but not informative.
    """

    base = LeNet5(num_classes=num_classes)

    if base_checkpoint:
        state = torch.load(base_checkpoint, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        base.load_state_dict(state, strict=False)

    if adaptation_mode == "full":
        unfreeze_module(base)
        return base

    if adaptation_mode == "lora":
        freeze_module(base)
        replace_linear_with_lora(
            base,
            module_names=lora_modules,
            rank=lora_rank,
            alpha=lora_alpha,
        )
        if train_head:
            unfreeze_module(base.fc2)
        return base

    if adaptation_mode == "prefix":
        # ImagePrefixLeNet5 freezes the whole base in __init__, so unfreeze
        # the head AFTER wrapping or the train_head request is silently lost.
        wrapped = ImagePrefixLeNet5(base, prefix_rows=prefix_rows)
        if train_head:
            unfreeze_module(wrapped.base.fc2)
        return wrapped

    raise ValueError(f"unknown adaptation_mode: {adaptation_mode}")

#!/usr/bin/env python3
"""Core primitives for an independent KV-Lingo reproduction.

The paper's default Qwen path translates keys before Qwen3's per-head key
normalisation and rotary embedding.  Those tensors do not survive in a normal
Transformers cache, so a source span is captured at ``k_proj`` while it is
being written.  Only the newly written span needs that auxiliary form: after a
switch it can be discarded, while each model retains its ordinary decode-ready
cache line.

This module deliberately contains no data or training policy.  It defines the
paper-level unit that the mechanics and cost pilot exercises: one token-wise,
head-mixing linear map per target layer and side, with gradients flowing from a
frozen receiver into the map alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CacheGeometry:
    layers: int
    kv_heads: int
    head_dim: int

    @property
    def width(self) -> int:
        return self.kv_heads * self.head_dim


@dataclass
class PreNormSpan:
    """A just-written source span at the paper's pre-normalisation cut."""

    keys: list[torch.Tensor]
    values: list[torch.Tensor]

    @property
    def tokens(self) -> int:
        return int(self.keys[0].shape[2])


def geometry(model) -> CacheGeometry:
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        head_dim = cfg.hidden_size // cfg.num_attention_heads
    return CacheGeometry(
        layers=int(cfg.num_hidden_layers),
        kv_heads=int(cfg.num_key_value_heads),
        head_dim=int(head_dim),
    )


def attention_modules(model):
    return [layer.self_attn for layer in model.model.layers]


def freeze_model(model) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


class PreNormTap:
    """Capture Qwen3 keys before ``k_norm`` and values at ``v_proj``."""

    def __init__(self, model):
        self.geom = geometry(model)
        self.keys: dict[int, torch.Tensor] = {}
        self.values: dict[int, torch.Tensor] = {}
        self.handles = []
        for layer, attention in enumerate(attention_modules(model)):
            if not all(
                hasattr(attention, name) for name in ("k_proj", "v_proj", "k_norm")
            ):
                raise TypeError(
                    "pre-normalisation capture requires Qwen3-style attention"
                )
            self.handles.append(
                attention.k_proj.register_forward_hook(self._hook(self.keys, layer))
            )
            self.handles.append(
                attention.v_proj.register_forward_hook(self._hook(self.values, layer))
            )

    def _hook(self, destination, layer):
        def save(_module, _inputs, output):
            batch, tokens = output.shape[:2]
            destination[layer] = (
                output.detach()
                .reshape(batch, tokens, self.geom.kv_heads, self.geom.head_dim)
                .transpose(1, 2)
                .contiguous()
            )

        return save

    def span(self) -> PreNormSpan:
        expected = set(range(self.geom.layers))
        if set(self.keys) != expected or set(self.values) != expected:
            raise RuntimeError("not every model layer was captured")
        return PreNormSpan(
            keys=[self.keys[layer] for layer in range(self.geom.layers)],
            values=[self.values[layer] for layer in range(self.geom.layers)],
        )

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def capture_pre_norm_span(model, input_ids: torch.Tensor) -> PreNormSpan:
    """Run one source span and retain only its translatable pre-norm tensors."""

    tap = PreNormTap(model)
    try:
        # Call the base model so a long calibration prefix does not also
        # materialise an unused [batch, tokens, vocabulary] logits tensor.
        model.model(input_ids=input_ids, use_cache=False)
        return tap.span()
    finally:
        tap.close()


@torch.no_grad()
def capture_and_append_pre_norm_span(
    model,
    input_ids: torch.Tensor,
    *,
    past_key_values=None,
):
    """Append ``input_ids`` and return both decode cache and auxiliary span.

    Retained-span switching needs the ordinary post-RoPE cache line and the
    pre-normalisation representation of exactly the newly written tokens.  The
    latter is temporary and may be discarded after the other model consumes
    the translated span.
    """

    tap = PreNormTap(model)
    try:
        output = model.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return output.past_key_values, tap.span()
    finally:
        tap.close()


@torch.no_grad()
def forward_and_capture_pre_norm_span(
    model,
    input_ids: torch.Tensor,
    *,
    past_key_values=None,
):
    """Run the language-model head while capturing exactly the appended span."""

    tap = PreNormTap(model)
    try:
        output = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return output, tap.span()
    finally:
        tap.close()


def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    first, second = tensor.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def rotate_keys(model, keys: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    """Apply the receiver's exact configured RoPE to ``[B,H,T,D]`` keys."""

    cos, sin = model.model.rotary_emb(keys, position_ids)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return keys * cos + _rotate_half(keys) * sin


class LinearTranslator(nn.Module):
    """One head-mixing linear K map and V map per target layer."""

    def __init__(self, source: CacheGeometry, target: CacheGeometry):
        super().__init__()
        if source.layers != target.layers:
            raise ValueError("this shape-preserving reproduction requires equal depth")
        self.source = source
        self.target = target
        self.key_weight = nn.Parameter(
            torch.empty(target.layers, target.width, source.width, dtype=torch.float32)
        )
        self.value_weight = nn.Parameter(
            torch.empty(target.layers, target.width, source.width, dtype=torch.float32)
        )
        self.reset_identity()

    @torch.no_grad()
    def reset_identity(self):
        """A finite mechanics initialisation, replaced by the stage-1 solve."""

        self.key_weight.zero_()
        self.value_weight.zero_()
        diagonal = min(self.source.width, self.target.width)
        indices = torch.arange(diagonal)
        self.key_weight[:, indices, indices] = 1.0
        self.value_weight[:, indices, indices] = 1.0

    @property
    def trainable_parameters(self) -> int:
        return translator_parameter_count(self.source, self.target)

    @property
    def parameter_bytes(self) -> int:
        return sum(
            parameter.numel() * parameter.element_size()
            for parameter in self.parameters()
        )

    def forward(
        self,
        source_span: PreNormSpan,
        target_model,
        position_ids: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if len(source_span.keys) != self.source.layers:
            raise ValueError("source span layer count does not match the translator")
        if position_ids.shape[-1] != source_span.tokens:
            raise ValueError("position ledger does not cover the source span")

        target_keys, target_values = [], []
        for layer, attention in enumerate(attention_modules(target_model)):
            source_key = source_span.keys[layer]
            source_value = source_span.values[layer]
            batch, _heads, tokens, _dim = source_key.shape
            key_rows = source_key.transpose(1, 2).reshape(batch, tokens, -1)
            value_rows = source_value.transpose(1, 2).reshape(batch, tokens, -1)
            mapped_key = F.linear(key_rows, self.key_weight[layer])
            mapped_value = F.linear(value_rows, self.value_weight[layer])
            mapped_key = mapped_key.reshape(
                batch, tokens, self.target.kv_heads, self.target.head_dim
            )
            mapped_key = attention.k_norm(mapped_key).transpose(1, 2)
            mapped_key = rotate_keys(target_model, mapped_key, position_ids)
            mapped_value = mapped_value.reshape(
                batch, tokens, self.target.kv_heads, self.target.head_dim
            ).transpose(1, 2)
            target_keys.append(mapped_key)
            target_values.append(mapped_value)
        return target_keys, target_values


def forward_kl(teacher_logprobs: torch.Tensor, student_logits: torch.Tensor):
    """Mean token-wise ``KL(teacher native || translated-cache student)``."""

    if teacher_logprobs.shape != student_logits.shape:
        raise ValueError(
            f"teacher/student shapes differ: {teacher_logprobs.shape} vs "
            f"{student_logits.shape}"
        )
    teacher = teacher_logprobs.float()
    student = torch.log_softmax(student_logits.float(), dim=-1)
    return (teacher.exp() * (teacher - student)).sum(-1).mean()


def translator_parameter_count(source: CacheGeometry, target: CacheGeometry) -> int:
    """Return the split-K/V dense-map ledger without allocating the maps."""

    if source.layers != target.layers:
        raise ValueError("this shape-preserving reproduction requires equal depth")
    return target.layers * 2 * target.width * source.width


def tensor_bytes(tensors: Sequence[torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def span_bytes(span: PreNormSpan) -> int:
    return tensor_bytes([*span.keys, *span.values])

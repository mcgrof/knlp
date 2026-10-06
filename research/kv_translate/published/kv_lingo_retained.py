#!/usr/bin/env python3
"""Ownership and cache-line primitives for retained-span KV translation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .capture import cache_layers, make_cache


@dataclass(frozen=True)
class WrittenSpan:
    writer: str
    start: int
    end: int
    serial: int


class SpanOwnershipLedger:
    """Prove that every new token is captured and translated at most once.

    ``line_covered`` is the number of complete decode-ready positions held by
    each model.  Only the active writer may extend the global token ledger.
    The inactive model catches up by consuming a suffix of immutable written
    spans.  Re-translating an old prefix is rejected.
    """

    def __init__(self, models: tuple[str, str], initial_tokens: int):
        if len(set(models)) != 2:
            raise ValueError("retained switching requires two distinct models")
        if initial_tokens < 0:
            raise ValueError("initial token count cannot be negative")
        self.models = models
        self.total_tokens = int(initial_tokens)
        self.line_covered = {model: int(initial_tokens) for model in models}
        self.spans: list[WrittenSpan] = []
        self.translations: set[tuple[int, str]] = set()

    def write(self, writer: str, tokens: int) -> WrittenSpan:
        if writer not in self.line_covered:
            raise KeyError(writer)
        if tokens <= 0:
            raise ValueError("a written span must contain at least one token")
        if self.line_covered[writer] != self.total_tokens:
            raise RuntimeError("writer cache is not caught up to the global ledger")
        span = WrittenSpan(
            writer=writer,
            start=self.total_tokens,
            end=self.total_tokens + int(tokens),
            serial=len(self.spans),
        )
        self.spans.append(span)
        self.total_tokens = span.end
        self.line_covered[writer] = span.end
        return span

    def missing(self, receiver: str) -> list[WrittenSpan]:
        if receiver not in self.line_covered:
            raise KeyError(receiver)
        start = self.line_covered[receiver]
        missing = [span for span in self.spans if span.end > start]
        if missing and missing[0].start != start:
            raise RuntimeError("missing spans do not begin at the receiver frontier")
        expected = start
        for span in missing:
            if span.start != expected:
                raise RuntimeError("missing spans are not a contiguous suffix")
            if span.writer == receiver:
                raise RuntimeError(
                    "receiver is missing a span it claims to have written"
                )
            if (span.serial, receiver) in self.translations:
                raise RuntimeError("an already translated span re-entered the suffix")
            expected = span.end
        if expected != self.total_tokens:
            raise RuntimeError("missing spans do not reach the global frontier")
        return missing

    def translated(self, receiver: str, spans: list[WrittenSpan]) -> None:
        expected = self.line_covered[receiver]
        for span in spans:
            if span.start != expected:
                raise RuntimeError(
                    "translation is not contiguous at the receiver frontier"
                )
            key = (span.serial, receiver)
            if key in self.translations:
                raise RuntimeError("span was translated to this receiver twice")
            self.translations.add(key)
            expected = span.end
        self.line_covered[receiver] = expected

    def receipt(self) -> dict:
        return {
            "models": list(self.models),
            "total_tokens": self.total_tokens,
            "line_covered": dict(self.line_covered),
            "spans": [span.__dict__ for span in self.spans],
            "translations": [
                {"span_serial": serial, "receiver": receiver}
                for serial, receiver in sorted(self.translations)
            ],
        }


def append_translated_span(cache, keys, values):
    """Append mapped ``[B,H,T,D]`` tensors to an ordinary receiver cache."""

    existing = cache_layers(cache)
    if len(existing) != len(keys) or len(keys) != len(values):
        raise ValueError("translated and resident cache layer counts differ")
    pairs = []
    for layer, ((old_key, old_value), new_key, new_value) in enumerate(
        zip(existing, keys, values, strict=True)
    ):
        if (
            old_key.shape[:2] != new_key.shape[:2]
            or old_key.shape[3:] != new_key.shape[3:]
        ):
            raise ValueError(f"key geometry differs at layer {layer}")
        if (
            old_value.shape[:2] != new_value.shape[:2]
            or old_value.shape[3:] != new_value.shape[3:]
        ):
            raise ValueError(f"value geometry differs at layer {layer}")
        pairs.append(
            (
                torch.cat((old_key, new_key.to(old_key.dtype)), dim=2),
                torch.cat((old_value, new_value.to(old_value.dtype)), dim=2),
            )
        )
    return make_cache(pairs)

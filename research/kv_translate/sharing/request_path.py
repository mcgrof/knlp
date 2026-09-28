#!/usr/bin/env python3
"""Three ways for a receiver to reach its first token after a long prefix.

``recompute``  the receiver is given the tokens and prefills the prefix.
``restore``    the receiver's own cache for the prefix is read from storage.
``translate``  the source's cache is read from storage and mapped into the
               receiver's cache space. The source model is not loaded.

All three end the same way: the suffix is processed on top of the prefix
cache, the vocabulary projection is applied to the last position alone, and
the first token is taken. A request is timed from before its storage read to
the moment that token is on the host, with the card synchronised at both
ends, and its stages are timed inside it.

None of this uses the likelihood scorer's prefix builder, which projects
every prefix position onto the vocabulary. At 32,768 tokens that projection
is a tensor of ten gigabytes that no server would compute.
"""

from __future__ import annotations

import time

import torch

from ..published import fit_pair
from ..published.capture import cache_layers, geometry, make_cache
from . import cache_store

ARMS = ("recompute", "restore", "translate")


class Clock:
    """Wall time with the card drained at every mark."""

    def __init__(self, device):
        self.cuda = torch.device(device).type == "cuda"
        self.marks = []
        self.t0 = self.now()

    def now(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    def mark(self, name):
        t = self.now()
        last = self.marks[-1][2] if self.marks else self.t0
        self.marks.append((name, t - last, t))

    def report(self):
        out = {f"seconds_{n}": round(d, 6) for n, d, _ in self.marks}
        out["seconds_total"] = round(self.marks[-1][2] - self.t0, 6)
        return out


class Receiver:
    """One warm receiver, its staging buffer and, for translation, a mapper."""

    def __init__(self, model, *, largest_payload, mapper=None, source_theta=None):
        self.model = model
        self.trunk = model.model
        self.head = model.lm_head
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.geom = geometry(model)
        self.reader = cache_store.Reader(largest_payload, self.device)
        self.mapper = None
        self.source_theta = source_theta
        if mapper is not None:
            self.mapper = {
                **mapper,
                "W": mapper["W"].to(self.device),
                "b": mapper["b"].to(self.device),
            }

    def ids(self, tokens):
        return torch.as_tensor(tokens, dtype=torch.long, device=self.device)[None]

    @torch.no_grad()
    def _finish(self, clock, suffix, cache):
        """Suffix on top of the cache, then the first token."""
        out = self.trunk(
            input_ids=self.ids(suffix), past_key_values=cache, use_cache=True
        )
        clock.mark("suffix")
        logits = self.head(out.last_hidden_state[:, -1:]).float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite logits at the first token")
        token = int(logits[0, -1].argmax().item())
        clock.mark("first_token")
        return token, out.past_key_values, logits[0, -1]

    @torch.no_grad()
    def prefix_cache(self, prefix):
        """The receiver's own cache for a prefix, with no projection at all."""
        out = self.trunk(input_ids=self.ids(prefix), use_cache=True)
        return [(k, v) for k, v in cache_layers(out.past_key_values)]

    @torch.no_grad()
    def recompute(self, prefix, suffix):
        clock = Clock(self.device)
        out = self.trunk(input_ids=self.ids(prefix), use_cache=True)
        clock.mark("prefill")
        token, cache, logits = self._finish(clock, suffix, out.past_key_values)
        return {"token": token, "cache": cache, "logits": logits, **clock.report()}

    @torch.no_grad()
    def restore(self, path, suffix, *, direct, expect):
        clock = Clock(self.device)
        head, io = self.reader.read(path, direct=direct, expect=expect)
        clock.mark("read")
        pairs = cache_store.as_pairs(self.reader.tensor(head))
        clock.mark("to_card")
        cache = make_cache(pairs)
        clock.mark("install")
        token, cache, logits = self._finish(clock, suffix, cache)
        return {
            "token": token,
            "cache": cache,
            "logits": logits,
            "io": io,
            **clock.report(),
        }

    @torch.no_grad()
    def translate(self, path, suffix, *, direct, expect):
        if self.mapper is None:
            raise ValueError("translation needs a fitted mapper")
        clock = Clock(self.device)
        head, io = self.reader.read(path, direct=direct, expect=expect)
        clock.mark("read")
        pairs = cache_store.as_pairs(self.reader.tensor(head))
        clock.mark("to_card")
        moved = fit_pair.transfer(
            self.mapper,
            pairs,
            self.source_theta,
            self.geom["rope_theta"],
            dtype=self.dtype,
        )
        clock.mark("map")
        cache = make_cache(moved)
        clock.mark("install")
        token, cache, logits = self._finish(clock, suffix, cache)
        return {
            "token": token,
            "cache": cache,
            "logits": logits,
            "io": io,
            **clock.report(),
        }

    @torch.no_grad()
    def closed_book(self, suffix_with_head):
        """The question with no prefix at all: how much the prefix matters."""
        clock = Clock(self.device)
        out = self.trunk(input_ids=self.ids(suffix_with_head), use_cache=True)
        clock.mark("suffix")
        logits = self.head(out.last_hidden_state[:, -1:]).float()
        token = int(logits[0, -1].argmax().item())
        clock.mark("first_token")
        return {
            "token": token,
            "cache": out.past_key_values,
            "logits": logits[0, -1],
            **clock.report(),
        }

    @torch.no_grad()
    def generate(self, first, cache, *, max_new_tokens, stop_ids):
        """Greedy continuation from a first token that is already chosen."""
        tokens, stopped = [first], first in stop_ids
        t0 = time.perf_counter()
        while not stopped and len(tokens) < max_new_tokens:
            out = self.trunk(
                input_ids=self.ids([tokens[-1]]), past_key_values=cache, use_cache=True
            )
            cache = out.past_key_values
            nxt = int(self.head(out.last_hidden_state[:, -1:])[0, -1].argmax().item())
            tokens.append(nxt)
            stopped = nxt in stop_ids
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        body = tokens[:-1] if stopped else tokens
        return {
            "tokens": body,
            "stopped": stopped,
            "new_tokens": len(tokens),
            "seconds_generating": round(time.perf_counter() - t0, 4),
        }

#!/usr/bin/env python3
"""Likelihood scoring of benchmark continuations through a supplied prefix cache.

One scorer, four ways of supplying the prefix, so every arm travels the same
code after the cache is in hand:

``direct``       the receiver scores context and continuation in one pass with
                 no cache at all. This is the receiver standing alone.
``native``       the receiver prefills the context itself and scores from its
                 own cache. It must agree with ``direct``, and the gap between
                 them is the harness's own error.
``full_head``    the source prefills, the baseline mapper transfers.
``cache_bridge`` the source prefills, the flagship mapper transfers.

The context is split one token before its end. A cache for the first T-1
context tokens is installed at positions 0..T-2, and the receiver is fed the
last context token followed by the continuation. The papers do not say where
they split; this is the latest split that still leaves the receiver a token
to predict the first continuation token from, so it hands the transferred
cache as much of the context as any split could.

Continuations of one context are scored in one batch, right-padded. Padding
sits after the real tokens, and attention is causal, so no real position can
see it and no mask is needed.
"""

from __future__ import annotations

import time

import torch

from . import fit_pair
from .capture import cache_layers, geometry, make_cache

MODES = ("direct", "native", "full_head", "cache_bridge")


class PrefixScorer:
    def __init__(self, target, mode, *, source=None, mapper=None):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose one of {MODES}")
        if mode in ("full_head", "cache_bridge"):
            if source is None or mapper is None:
                raise ValueError(f"{mode} needs a source model and a fitted mapper")
            if mapper["method"].replace("_mapping", "") != mode:
                raise ValueError(
                    f"mode is {mode} but the mapper was fitted as {mapper['method']}"
                )
        self.target, self.source, self.mode = target, source, mode
        self.tgeom = geometry(target)
        self.sgeom = geometry(source) if source is not None else None
        self.device = next(target.parameters()).device
        self.dtype = next(target.parameters()).dtype
        self.mapper = mapper
        if mapper is not None:
            # resident beside the source, or every prefix pays to move it again
            where = next(source.parameters()).device
            self.mapper = {
                **mapper,
                "W": mapper["W"].to(where),
                "b": mapper["b"].to(where),
            }
        self.prefix_tokens = 0
        self.prefixes = 0
        self.seconds_prefix = 0.0
        self.seconds_scoring = 0.0

    def _clock(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    def _scored_logits(self, ids, first, count, kwargs):
        """Logits for the scored rows only.

        The vocabulary projection is the widest tensor in the pass, and only
        the rows that predict a continuation token are read, so the projection
        is applied to those rows alone when the model exposes its trunk.
        """
        trunk = getattr(self.target, "model", None)
        head = getattr(self.target, "lm_head", None)
        if trunk is None or head is None:
            return self.target(input_ids=ids, **kwargs).logits[:, first : first + count]
        hidden = trunk(input_ids=ids, **kwargs).last_hidden_state
        return head(hidden[:, first : first + count])

    @torch.no_grad()
    def _prefix(self, ctx):
        """Per-layer (keys, values) for the context, by whichever route."""
        ids = torch.tensor([ctx], device=self.device)
        if self.mode == "native":
            out = self.target(input_ids=ids, use_cache=True)
            return [(k, v) for k, v in cache_layers(out.past_key_values)]
        where = next(self.source.parameters()).device
        out = self.source(input_ids=ids.to(where), use_cache=True)
        src = [(k, v) for k, v in cache_layers(out.past_key_values)]
        moved = fit_pair.transfer(
            self.mapper,
            src,
            self.sgeom["rope_theta"],
            self.tgeom["rope_theta"],
            dtype=self.dtype,
        )
        return [(k.to(self.device), v.to(self.device)) for k, v in moved]

    @torch.no_grad()
    def score(self, context, continuations):
        """Sum of log-probabilities and greedy agreement for each continuation."""
        if not context:
            raise ValueError("an empty context leaves nothing to predict from")
        n = len(continuations)
        longest = max(len(c) for c in continuations)
        if self.mode == "direct" or len(context) == 1:
            head = context
            pairs = None
        else:
            head = context[-1:]
            t0 = self._clock()
            pairs = self._prefix(context[:-1])
            self.seconds_prefix += self._clock() - t0
            self.prefix_tokens += len(context) - 1
            self.prefixes += 1
        t0 = self._clock()
        width = len(head) + longest - 1
        ids = torch.zeros(n, width, dtype=torch.long, device=self.device)
        for i, c in enumerate(continuations):
            row = list(head) + list(c[:-1])
            ids[i, : len(row)] = torch.tensor(row, device=self.device)
        kwargs = {}
        if pairs is not None:
            kwargs["past_key_values"] = make_cache(
                [(k.expand(n, -1, -1, -1), v.expand(n, -1, -1, -1)) for k, v in pairs]
            )
        first = len(head) - 1
        logits = self._scored_logits(ids, first, longest, kwargs).float()
        logp = torch.log_softmax(logits, dim=-1)
        if not torch.isfinite(logp).all():
            raise FloatingPointError(
                "nonfinite scored logits; this is a corrupt run, not a low score"
            )
        out = []
        for i, c in enumerate(continuations):
            tgt = torch.tensor(c, device=self.device)
            rows = logp[i, : len(c)]
            ll = rows.gather(1, tgt[:, None]).sum().item()
            greedy = bool((rows.argmax(-1) == tgt).all().item())
            out.append((ll, greedy))
        self.seconds_scoring += self._clock() - t0
        return out


def make_lm(scorer, tokenizer):
    """Wrap a scorer as an lm-evaluation-harness model.

    Task formatting, few-shot construction and the accuracy definitions all
    stay the harness's own, which is what the baseline paper says it used.
    Only the question of where the prefix cache comes from is ours.
    """
    from lm_eval.api.model import TemplateLM

    class TransferLM(TemplateLM):
        def __init__(self):
            super().__init__()
            self.scorer = scorer
            self.tok = tokenizer

        @property
        def eot_token_id(self):
            return self.tok.eos_token_id

        def tok_encode(self, string, **kwargs):
            return self.tok.encode(string, add_special_tokens=False)

        def _loglikelihood_tokens(self, requests, disable_tqdm=False, **kwargs):
            # group by context so each prefix is built once
            order = {}
            for i, (_pair, ctx, cont) in enumerate(requests):
                order.setdefault(tuple(ctx), []).append((i, cont))
            res = [None] * len(requests)
            for ctx, items in order.items():
                scored = self.scorer.score(list(ctx), [c for _, c in items])
                for (i, _c), r in zip(items, scored):
                    res[i] = r
            return res

        def loglikelihood_rolling(self, requests, disable_tqdm=False):
            raise NotImplementedError("rolling likelihood is not part of this protocol")

        def generate_until(self, requests, disable_tqdm=False):
            raise NotImplementedError("no generation occurs in this protocol")

    return TransferLM()

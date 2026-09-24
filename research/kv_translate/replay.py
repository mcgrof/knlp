# SPDX-License-Identifier: GPL-2.0
"""Recompute selected positions of an approximate cache, in place.

This is the difference between a repair and an appendix. The receiver's cache
already holds a value for every position; a repair replaces some of them with
values the receiver itself computed, leaving the cache exactly as long as it
was and every other position exactly as it was. Appending recomputed copies to
the end would leave the original wrong values still present and still
attended to, and would lengthen the prefix the question reads.

Two things make that awkward with an ordinary implementation. The library's
cache appends on update, so writing into fixed slots needs a cache that
scatters instead. And a token recomputed out of order must carry its original
absolute position, both for the rotation applied to its key and for deciding
what it may attend to: a token from the middle of the document must see the
document before it and must not see the document after it, regardless of where
it sits in the batch of tokens being recomputed.

The alternative that looks equivalent and is not: run a full receiver prefill
and scatter a fraction of its output. That computes everything, so it measures
nothing about the cost of computing a fraction, and every recomputed token
would have read correct context rather than the mixed cache a real repair
reads.
"""

from __future__ import annotations

import torch


class ScatterCache:
    """A fixed-length cache whose updates land in chosen slots.

    Presented to the model in place of the appending cache. On each layer's
    update it writes the freshly computed keys and values into the slots the
    selection named, and returns the whole cache so attention reads repaired
    and unrepaired positions together, which is what a real repair would face.
    """

    def __init__(self, keys, values, slots, in_place=False):
        if len(keys) != len(values):
            raise ValueError("key and value caches differ in layer count")
        # The copy is not ceremony by default: one mapped cache is read by
        # several conditions for the same document, so repairing it in place
        # for one would hand the next a cache that had already been repaired.
        # It is waste only where the caller owns the cache and discards it,
        # which is the timing path, and that is what the flag is for.
        self.keys = list(keys) if in_place else [k.clone() for k in keys]
        self.values = list(values) if in_place else [v.clone() for v in values]
        self.slots = slots
        self._seen = set()

    def __len__(self):
        return len(self.keys)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return int(self.keys[layer_idx].shape[2])

    def get_max_cache_shape(self):
        return int(self.keys[0].shape[2])

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if key_states.shape[2] != self.slots.shape[0]:
            raise ValueError(
                f"layer {layer_idx}: {key_states.shape[2]} fresh positions for "
                f"{self.slots.shape[0]} slots"
            )
        self.keys[layer_idx][:, :, self.slots, :] = key_states.to(
            self.keys[layer_idx].dtype
        )
        self.values[layer_idx][:, :, self.slots, :] = value_states.to(
            self.values[layer_idx].dtype
        )
        self._seen.add(int(layer_idx))
        return self.keys[layer_idx], self.values[layer_idx]

    def layers_written(self):
        return sorted(self._seen)


def absolute_causal_mask(positions, n_positions, dtype, device):
    """What each recomputed token may attend to, by absolute position.

    A token at absolute position p may read every cache position up to and
    including p, and nothing after it. The ordinary triangular mask over the
    compact selected rows is not this: it would let the first selected token
    see only itself even when it sits late in the document, and would let a
    late one see an early one only if the early one happened to be selected.
    """
    q = positions.reshape(-1, 1)
    k = torch.arange(n_positions, device=device).reshape(1, -1)
    allowed = k <= q
    m = torch.zeros(allowed.shape, dtype=dtype, device=device)
    m = m.masked_fill(~allowed, torch.finfo(dtype).min)
    return m.reshape(1, 1, positions.shape[0], n_positions)


@torch.no_grad()
def replay_selected(model, input_ids, mask, keys, values, dtype=None, in_place=False):
    """Recompute the masked positions and scatter them into the cache.

    ``input_ids`` is the whole prompt; only the masked positions are fed. The
    cache handed in is the approximate one, and the cache handed back differs
    from it at exactly those positions.
    """
    device = keys[0].device
    T = keys[0].shape[2]
    if mask.shape[0] != T:
        raise ValueError(f"mask covers {mask.shape[0]} of {T} positions")
    slots = torch.nonzero(mask, as_tuple=False).flatten().to(device)
    if slots.numel() == 0:
        # A true no-op: nothing is computed and nothing is copied. Returning
        # clones here would cost the work of a repair while performing none.
        return keys, values, {"replayed": 0, "layers_written": []}

    sel_ids = input_ids[0, slots].reshape(1, -1).to(device)
    dt = dtype or keys[0].dtype
    cache = ScatterCache(keys, values, slots, in_place=in_place)
    attn = absolute_causal_mask(slots, T, dt, device)
    # A repair wants keys and values, never a prediction, so the vocabulary
    # projection over every recomputed position is work whose output is
    # discarded. At a four-thousand-token context and a vocabulary of a
    # hundred and fifty thousand that tensor is gigabytes. Asking for one
    # position is the smallest request the interface allows.
    model(
        input_ids=sel_ids,
        attention_mask=attn,
        position_ids=slots.reshape(1, -1),
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )
    return (
        cache.keys,
        cache.values,
        {"replayed": int(slots.numel()), "layers_written": cache.layers_written()},
    )

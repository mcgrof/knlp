#!/usr/bin/env python3
"""Train the pieces of one split document in a single cartridge run.

``cas_split.py`` cuts a record into contiguous pieces, each with its own
cartridge and its own Self-Study rows. Trained one run per piece, every
run is a full pass of the reader per packed sequence, so pieces only
share hardware by running side by side. This module lets the native
trainer (``cas_train_isolated.py``) hold all pieces in one cache and
train them in one run, in one of two modes:

- ``isolated``: each conversation sees only its own piece, at the
  positions it has when that piece is loaded alone (keys rotated at
  0 .. P_j - 1, prompt from P_j). A piece's slots get gradient only from
  its own conversations, so this is the separate runs, sharing passes:
  with ``MICRO_SEQS`` > 1 one forward pass carries conversations of
  several pieces.
- ``joint``: the pieces sit at their bundle positions (keys of piece j
  rotated on by the lengths of the pieces before it, as ``cas_split.py
  compose`` does) and the prompt follows the whole bundle, as at
  inference. Each conversation sees its own piece and, with probability
  1 - P_ISO, one or more of the others (Cartridges at Scale, section
  2.1); P_ISO = 0 (default) shows the whole bundle every time.

Each piece keeps its first slot (its frozen sink) frozen: the first
piece's through the cache's frozen prefix, the others by zeroing their
gradient, so Adam never moves them.

The pieces' Self-Study parquets are passed as separate data sources, in
piece order; the library keeps elements in source order, so an
element's index gives its piece. Each packed sequence's element ids are
rewritten as ``e * 256 + bits * 16 + piece``, where ``bits`` is the set
of pieces the element may see. Ids stay unique per sequence and keep
their order, and ``cas_microbatch.py``'s offset (a multiple of 256)
leaves the code intact. The mask reads the code from the ids:

    cartridge slot of piece c: visible iff bit c of the query's bits
    prompt token:              same element id and causal (the library's)

Usage (as a module): ``install(...)`` before the library builds its
dataset and cache. CLI:

    python cas_pieces.py concat   # PIECE_CARTS, MODE, OUT, THETA
    python cas_pieces.py split    # CART, LENGTHS, OUT_PREFIX
    python cas_pieces.py selftest
"""

import bisect
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PIECE_MOD = 16
ID_STRIDE = 256
MAX_PIECES = 4  # bits must fit 0 .. 15

_S = {}


def encode(e, bits, piece):
    return e * ID_STRIDE + bits * PIECE_MOD + piece


def decode(code):
    return code // ID_STRIDE, (code // PIECE_MOD) % PIECE_MOD, code % PIECE_MOD


def reveal_bits(piece, n, p_iso, rng):
    """Own piece always; with prob 1 - p_iso also k ~ U(1, n-1) others."""
    bits = 1 << piece
    if n > 1 and rng.random() >= p_iso:
        others = [c for c in range(n) if c != piece]
        for c in rng.sample(others, rng.randint(1, n - 1)):
            bits |= 1 << c
    return bits


def concat(parts, shift, theta=1e6):
    """parts: [(keys, values, nfrozen)] per piece. ``shift`` rotates piece
    j's keys on by the earlier pieces' lengths (bundle positions);
    otherwise each piece keeps its standalone rotation."""
    if shift:
        from cas_split import compose_tensors

        return compose_tensors(parts, theta)
    L = len(parts[0][0])
    keys = [torch.cat([p[0][i] for p in parts], dim=2).contiguous() for i in range(L)]
    vals = [torch.cat([p[1][i] for p in parts], dim=2).contiguous() for i in range(L)]
    return keys, vals, parts[0][2]


def piece_masks(lengths):
    """kv_piece over the cartridge (frozen prefix first, as the cache lays
    it out) and the trainable-slot indices of the later pieces' sinks."""
    kv_piece = torch.cat(
        [torch.full((n,), j, dtype=torch.long) for j, n in enumerate(lengths)]
    )
    starts = [sum(lengths[:j]) for j in range(len(lengths))]
    sinks = [s - 1 for s in starts[1:]]  # trainable index = slot - 1 frozen
    return kv_piece, sinks


def mask_fn(seq_ids, cl, kv_piece_full, kv_seq_ids):
    """The visibility rule as a FlexAttention mask_mod (also used by the
    selftest on dense grids)."""

    def mask(_b, _h, q_idx, kv_idx):
        is_cart = kv_idx < cl
        bits = (seq_ids[q_idx] // PIECE_MOD) % PIECE_MOD
        cart_ok = ((bits >> kv_piece_full[kv_idx]) & 1) == 1
        req_ok = (seq_ids[q_idx] == kv_seq_ids[kv_idx]) & (q_idx + cl >= kv_idx)
        return torch.where(is_cart, cart_ok, req_ok)

    return mask


def install(train_dataset_cls, models_module, cache_cls, lengths, mode, p_iso, seed):
    """Patch the library: element codes and positions in TrainDataset
    batches, the piece-aware block mask, and frozen sinks of later pieces."""
    from torch.nn.attention.flex_attention import create_block_mask

    n = len(lengths)
    assert 1 <= n <= MAX_PIECES and mode in ("isolated", "joint")
    total = sum(lengths)
    kv_piece, sinks = piece_masks(lengths)
    if mode == "isolated":
        p_iso = 1.0  # own piece only
    _S.update(lengths=lengths, mode=mode, p_iso=p_iso, seed=seed, total=total)

    orig_prep = train_dataset_cls._prepare_elements

    def _prep(self):
        from cartridges.datasets import _prepare_data_source

        sizes = [len(_prepare_data_source(s)) for s in self.config.data_sources]
        if type(self) is train_dataset_cls:
            assert len(sizes) == n, f"{len(sizes)} data sources for {n} pieces"
            self._cr_bounds = [sum(sizes[: j + 1]) for j in range(n)]
        return orig_prep(self)

    orig_get = train_dataset_cls._get_batch

    def _get(self, batch_idx):
        b = orig_get(self, batch_idx)
        if type(self) is not train_dataset_cls:
            return b
        idxs = self.batches[batch_idx]
        pieces = [bisect.bisect_right(self._cr_bounds, i) for i in idxs]
        rng = random.Random(seed * 1_000_003 + batch_idx)
        bits = [reveal_bits(p, n, p_iso, rng) for p in pieces]
        e = b.element_ids
        piece_t = torch.tensor(pieces, dtype=torch.long)[e]
        bits_t = torch.tensor(bits, dtype=torch.long)[e]
        b.element_ids = encode(e, bits_t, piece_t)
        if mode == "isolated":
            shift = torch.tensor([lengths[p] - total for p in range(n)])
            b.position_ids = b.position_ids + shift[piece_t]
        return b

    train_dataset_cls._prepare_elements = _prep
    train_dataset_cls._get_batch = _get

    orig_mask = models_module.create_block_mask_w_cache

    def _mask(cache, seq_ids, device):
        cl = cache.num_tokens() if cache is not None else 0
        if cl != total:  # generation or a foreign cache: library mask
            return orig_mask(cache, seq_ids, device)
        R = seq_ids.shape[0]
        kvp = torch.cat(
            [kv_piece.to(device), torch.zeros(R, dtype=torch.long, device=device)]
        )
        kvs = torch.cat(
            [torch.full((cl,), -1, dtype=seq_ids.dtype, device=device), seq_ids]
        )
        return create_block_mask(
            mask_fn(seq_ids, cl, kvp, kvs),
            B=1,
            H=1,
            Q_LEN=R,
            KV_LEN=R + cl,
            device=device,
        )

    models_module.create_block_mask_w_cache = _mask

    orig_init = cache_cls.__init__

    def _init(self, *a, **kw):
        orig_init(self, *a, **kw)
        if self.num_cartridge_tokens() != total or not sinks:
            return
        assert self._num_frozen_tokens == 1, "each piece expects one frozen sink"

        def _hook(g):
            g = g.clone()
            g[:, :, sinks] = 0
            return g

        for p in list(self.trainable_keys) + list(self.trainable_values):
            p.register_hook(_hook)
        print(
            f"[pieces] {n} pieces {lengths} mode={mode} p_iso={_S['p_iso']} "
            f"frozen later sinks at trainable {sinks}",
            flush=True,
        )

    cache_cls.__init__ = _init


def split_cart(keys, vals, nfrozen, lengths):
    """Cut a combined cartridge into per-piece (keys, values, 1)."""
    assert nfrozen == 1 and sum(lengths) == keys[0].shape[2]
    out, s = [], 0
    for n in lengths:
        out.append(
            ([k[:, :, s : s + n] for k in keys], [v[:, :, s : s + n] for v in vals], 1)
        )
        s += n
    return out


def selftest():
    from cas_kv_rope import rerot

    assert decode(encode(5, 0b101, 2)) == (5, 5, 2)
    assert decode(encode(7, 1, 0) + 3 * (1 << 16)) == (7 + 3 * 256, 1, 0)
    rng = random.Random(0)
    assert reveal_bits(1, 3, 1.0, rng) == 2
    seen = {reveal_bits(0, 3, 0.0, rng) for _ in range(200)}
    assert seen <= {3, 5, 7} and len(seen) == 3
    # dense mask grid: two pieces of 3 and 2 slots, three elements
    lengths = [3, 2]
    kv_piece, sinks = piece_masks(lengths)
    assert kv_piece.tolist() == [0, 0, 0, 1, 1] and sinks == [2]
    cl = 5
    e = torch.tensor([0, 0, 1, 1, 2])
    piece = torch.tensor([0, 0, 1, 1, 0])
    bits = torch.tensor([1, 1, 2, 2, 3])
    sid = encode(e, bits, piece)
    kvp = torch.cat([kv_piece, torch.zeros(5, dtype=torch.long)])
    kvs = torch.cat([torch.full((cl,), -1), sid])
    m = mask_fn(sid, cl, kvp, kvs)
    q = torch.arange(5)[:, None]
    kv = torch.arange(10)[None, :]
    grid = m(0, 0, q, kv).int()
    want = torch.tensor(
        [
            [1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 1, 1, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
        ]
    )
    assert torch.equal(grid, want), grid
    # concat without shift keeps standalone rotations; with shift equals
    # rotating the pre-rotary keys at bundle positions
    torch.manual_seed(0)
    raw = [torch.randn(1, 2, 5, 8)]
    a = ([rerot(raw[0][:, :, :3])], [torch.randn(1, 2, 3, 8)], 1)
    b = ([rerot(raw[0][:, :, 3:])], [torch.randn(1, 2, 2, 8)], 1)
    k0, _, _ = concat([a, b], shift=False)
    assert torch.equal(k0[0][:, :, 3:], b[0][0])
    k1, _, _ = concat([a, b], shift=True)
    assert torch.allclose(k1[0], rerot(raw[0]), atol=1e-5)
    parts = split_cart(*concat([a, b], shift=False), lengths=[3, 2])
    assert torch.equal(parts[1][0][0], b[0][0])
    print("CAS_PIECES_SELFTEST PASS")


def main():
    cmd = sys.argv[1]
    if cmd == "selftest":
        return selftest()
    from cas_cart_init import load_cart, save_cart

    if cmd == "concat":
        parts = [load_cart(p) for p in os.environ["PIECE_CARTS"].split(",")]
        shift = os.environ.get("MODE", "isolated") == "joint"
        k, v, nf = concat(parts, shift, float(os.environ.get("THETA", "1e6")))
        save_cart(k, v, nf, os.environ["OUT"])
        print(
            f"PIECES_CONCAT {os.environ['OUT']} lengths={[p[0][0].shape[2] for p in parts]} shift={shift}"
        )
    elif cmd == "split":
        lengths = [int(x) for x in os.environ["LENGTHS"].split(",")]
        for j, (k, v, nf) in enumerate(
            split_cart(*load_cart(os.environ["CART"]), lengths)
        ):
            save_cart(k, v, nf, f"{os.environ['OUT_PREFIX']}{j}.pt")
        print(f"PIECES_SPLIT {os.environ['CART']} -> {len(lengths)} pieces")


if __name__ == "__main__":
    main()

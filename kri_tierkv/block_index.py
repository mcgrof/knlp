# SPDX-License-Identifier: GPL-2.0
"""Block bookkeeping for the tiered KV cache.

The cache is a sequence of fixed-size token blocks. Each block records where it
sits, which tier it is in, its current KRI-D-sum score, when it was evicted, and
how often it has been fetched back. The index also knows which blocks are
protected -- the system/prefix region, the recent fast window, and the
neighborhood of the current decode point -- so an eviction policy never drops
them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config import TierKVConfig


FAST = "fast"
SLOW = "slow"


@dataclass
class Block:
    block_id: int
    start_token: int
    end_token: int  # exclusive
    tier: str = FAST
    score: float = 0.0
    evicted_at_step: Optional[int] = None
    fetched_count: int = 0

    @property
    def num_tokens(self) -> int:
        return self.end_token - self.start_token


class BlockIndex:
    """Blocks over a prefix of `seq_len` tokens, tiered by a fast recent window."""

    def __init__(self, seq_len: int, cfg: TierKVConfig):
        self.cfg = cfg
        self.seq_len = seq_len
        bs = cfg.block_size
        self.blocks = []
        b = 0
        for start in range(0, seq_len, bs):
            self.blocks.append(
                Block(block_id=b, start_token=start, end_token=min(seq_len, start + bs))
            )
            b += 1
        self._assign_tiers()

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    def _assign_tiers(self):
        """Recent `fast_window_blocks` are fast; the rest are slow."""
        n = self.num_blocks
        fast_start = max(0, n - self.cfg.fast_window_blocks)
        for blk in self.blocks:
            blk.tier = FAST if blk.block_id >= fast_start else SLOW

    def protected_ids(self, decode_block: Optional[int] = None) -> set:
        """Blocks an eviction policy must never drop: the prefix region, the
        fast recent window, and the decode neighborhood.
        """
        n = self.num_blocks
        prot = set(range(min(self.cfg.protect_prefix_blocks, n)))
        prot |= {b.block_id for b in self.blocks if b.tier == FAST}
        if decode_block is not None:
            r = self.cfg.decode_neighborhood_blocks
            prot |= {
                b for b in range(decode_block - r, decode_block + r + 1) if 0 <= b < n
            }
        return prot

    def slow_ids(self) -> list:
        return [b.block_id for b in self.blocks if b.tier == SLOW]

    def fast_ids(self) -> list:
        return [b.block_id for b in self.blocks if b.tier == FAST]

    def evictable_ids(self, decode_block: Optional[int] = None) -> list:
        """Slow, non-protected blocks -- the candidates for eviction."""
        prot = self.protected_ids(decode_block)
        return [
            b.block_id for b in self.blocks if b.tier == SLOW and b.block_id not in prot
        ]

    def set_scores(self, scores):
        """Attach per-block scores (list or dict block_id->score)."""
        if isinstance(scores, dict):
            for bid, s in scores.items():
                self.blocks[bid].score = float(s)
        else:
            for bid, s in enumerate(scores):
                if bid < self.num_blocks:
                    self.blocks[bid].score = float(s)

    def bytes_per_block(self, n_layers, n_kv_heads, head_dim, k_bits, v_bits) -> float:
        """Bytes one block occupies given K/V bit-widths (fake-quant storage)."""
        toks = self.cfg.block_size
        kbytes = n_layers * n_kv_heads * head_dim * toks * (k_bits / 8.0)
        vbytes = n_layers * n_kv_heads * head_dim * toks * (v_bits / 8.0)
        return kbytes + vbytes

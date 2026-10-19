# SPDX-License-Identifier: MIT
"""Bound extra references to existing, completed full-attention cache blocks.

This experiment controller runs on the synchronous scheduler thread. The caller
must acquire after model execution completes and before releasing the producing
request, and close all leases before cache reset or model invalidation. It does
not provide a production scheduler, admission policy or device synchronization.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LeaseToken:
    family_id: str
    generation: int


class FamilyLeases:
    """Charge overlapping families by unique physical blocks and fence releases."""

    def __init__(self, pool, block_size, max_blocks, max_families):
        if (
            any(type(v) is not int for v in (block_size, max_blocks, max_families))
            or block_size <= 0
            or not 0 < max_blocks < len(pool.blocks) - 1
            or max_families <= 0
        ):
            raise ValueError("lease budget must leave request capacity")
        self.pool = pool
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.max_families = max_families
        self.active = {}
        self.generation = 0
        self.rejected = 0
        self.released = {key: 0 for key in ("complete", "cancel", "expire", "close")}

    def block_ids(self):
        return {b.block_id for _, blocks, _ in self.active.values() for b in blocks}

    def acquire_completed(self, family_id, blocks, *, now_ns, ttl_ns):
        """Hold already computed blocks while the producing request owns them."""
        if not isinstance(family_id, str) or not family_id:
            raise ValueError("missing family identity")
        if (
            type(now_ns) is not int
            or now_ns < 0
            or type(ttl_ns) is not int
            or ttl_ns <= 0
        ):
            raise ValueError("invalid lease clock or lifetime")
        self.expire(now_ns)
        if family_id in self.active:
            raise ValueError("family already holds a lease")
        blocks = tuple(blocks)
        ids = [b.block_id for b in blocks]
        if not blocks or len(ids) != len(set(ids)):
            raise ValueError("lease needs distinct full blocks")
        for index, block in enumerate(blocks):
            if (
                block.is_null
                or not 0 <= block.block_id < len(self.pool.blocks)
                or self.pool.blocks[block.block_id] is not block
                or block.ref_cnt <= 0
                or block.block_hash is None
                or block.block_hash_num_tokens != (index + 1) * self.block_size
            ):
                raise ValueError(
                    "lease requires owned, hashed full blocks in this pool"
                )
        if (
            len(self.active) >= self.max_families
            or len(self.block_ids() | set(ids)) > self.max_blocks
        ):
            self.rejected += 1
            return None
        self.generation += 1
        token = LeaseToken(family_id, self.generation)
        self.pool.touch(blocks)
        self.active[family_id] = (token, blocks, now_ns + ttl_ns)
        return token

    def release(self, token, reason="complete"):
        """Drop this generation's references once, preserving request owners."""
        if reason not in self.released:
            raise ValueError("unknown release reason")
        entry = self.active.get(token.family_id)
        if entry is None or entry[0] != token:
            return False
        _, blocks, _ = entry
        if any(block.ref_cnt <= 0 for block in blocks):
            raise RuntimeError("lease lost its allocator reference")
        self.pool.free_blocks(reversed(blocks))
        del self.active[token.family_id]
        self.released[reason] += 1
        return True

    def expire(self, now_ns):
        """Release elapsed holds using the caller's monotonic clock domain."""
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("invalid lease clock")
        for token, _, deadline in list(self.active.values()):
            if now_ns >= deadline:
                self.release(token, "expire")

    def close(self):
        for token, _, _ in list(self.active.values()):
            self.release(token, "close")

    def snapshot(self):
        return {
            "families": len(self.active),
            "unique_held_blocks": len(self.block_ids()),
            "family_block_references": sum(len(v[1]) for v in self.active.values()),
            "rejected": self.rejected,
            "released": dict(self.released),
        }

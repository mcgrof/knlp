# SPDX-License-Identifier: MIT
"""Feed a complete layer-major model group from private native span restores."""

from dataclasses import dataclass
import functools
import time
from typing import Any

if __package__:
    from .bridge import RegisteredReplay, identity
else:
    from bridge import RegisteredReplay, identity


@dataclass(frozen=True)
class PreparedMember:
    request: Any
    tokens: tuple[int, ...]
    blocks: tuple[int, ...]
    spans: Any


def prepare_members(worker, requests):
    """Check every actual destination table before any native restoration."""
    if type(requests) is not tuple or len(requests) != 4:
        raise ValueError("the frozen model group requires exactly four members")
    prepared, names, reserved = [], set(), set()
    generations = set()
    for request in requests:
        tokens = request.tokens
        if (
            type(request.request_id) is not str
            or not request.request_id
            or request.request_id in names
            or type(request.generation) is not int
            or request.generation < 0
            or type(tokens) is not list
            or len(tokens) != 512
            or any(type(t) is not int or not 0 <= t < 2**63 for t in tokens)
        ):
            raise ValueError("model group identity or token window changed")
        names.add(request.request_id)
        generations.add(request.generation)
        spans = request.retrieve_kwargs["explicit_spans"]
        spans.validate_token_count(len(tokens))
        if len(spans.spans) != 4 or sum(s.end - s.start for s in spans.spans) != 416:
            raise ValueError("model group no longer names the frozen shared extents")
        slots = tuple(request.retrieve_kwargs["slot_mapping"].cpu().tolist())
        blocks = worker.block_table(slots)
        if len(slots) != len(tokens) or reserved.intersection(blocks):
            raise ValueError(
                "model group does not own private complete destination tables"
            )
        reserved.update(blocks)
        prepared.append(PreparedMember(request, tuple(tokens), blocks, spans))
    if len(generations) != 1:
        raise ValueError("model group spans different scheduling generations")
    return tuple(prepared)


class RegisteredGroupReplay(RegisteredReplay):
    """Retain one registered owner for the complete consuming model group."""

    def consume_group(self, blender, requests, kwargs, original):
        from lmcache.integration.vllm.kmesh_span_group import (
            RestoredSpanGroup,
            RestoredSpanGroupMember,
        )
        from lmcache.integration.vllm.kmesh_span_layers import RestoredSpanBatch

        if self.disabled or self.worker is None:
            raise RuntimeError("registered group replay has no ready source owner")
        if kwargs.get("collective_position_recovery") or kwargs.get("collective_qkv"):
            raise ValueError(
                "registered group replay requires request-local model operations"
            )
        requests = tuple(requests)
        prepared = prepare_members(self.worker, requests)
        start = time.perf_counter_ns()
        members = []
        for member in prepared:
            placements = []
            for span in member.spans.spans:
                key = self._key(
                    member.tokens[span.start : span.end],
                    member.request.retrieve_kwargs["request_configs"],
                )
                handle = self.handles.get(key)
                if handle is None or key in self.omitted_sources:
                    continue
                placed = self.worker.restore(
                    handle,
                    span.start,
                    identity(member.tokens[: span.start]),
                    member.blocks,
                )
                placements.append(placed)
                self.ledger.emit(
                    "registered.span.restored",
                    request_id=member.request.request_id,
                    source_start=handle.source_start,
                    destination_start=span.start,
                    token_count=handle.token_count,
                    receipt=placed.receipt,
                )
            if not placements:
                raise RuntimeError("registered group member has no completed source")
            batch = RestoredSpanBatch(
                self.worker.fence,
                self.worker.runtime.context,
                member.tokens,
                member.blocks,
                member.spans,
                tuple(placements),
            )
            members.append(
                RestoredSpanGroupMember(
                    member.request.request_id, member.request.generation, batch
                )
            )
        owner = self.worker.layer_source_group(RestoredSpanGroup(tuple(members)))
        self.sources.append(owner)
        ready = time.perf_counter_ns()
        try:
            result = original(blender, requests, layer_source_group=owner, **kwargs)
        finally:
            owner.close()
        finished = time.perf_counter_ns()
        self.ledger.emit(
            "registered.model.group.consumed",
            request_order=[m.request.request_id for m in prepared],
            span_counts=[len(m.batch.placements) for m in members],
            restored_tokens=[
                sum(p.receipt.span.token_count for p in m.batch.placements)
                for m in members
            ],
            outcomes=[m.outcome.value for m in result.members],
            source_closed=owner.is_closed,
            pending=owner.pending,
            restore_ns=ready - start,
            processing_ns=finished - ready,
        )
        return result

    def install(self):
        from lmcache.v1.compute.blend.blender import LMCBlender
        from lmcache.v1.compute.blend.layer_source import UnsettledLayerSourceError

        super().install()
        original = LMCBlender.blend_group

        @functools.wraps(original)
        def consume(blender, requests, **kwargs):
            try:
                return self.consume_group(blender, requests, kwargs, original)
            except Exception as error:
                self.disabled = True
                self.ledger.emit("registered.group.disabled", error=str(error))
                if isinstance(error, UnsettledLayerSourceError):
                    raise
                if self.worker is not None and self.worker.pending:
                    raise UnsettledLayerSourceError(
                        "registered group restore still owes physical completion",
                        self.worker.drain_pending,
                    ) from error
                raise

        self.originals.append((LMCBlender, "blend_group", original))
        LMCBlender.blend_group = consume

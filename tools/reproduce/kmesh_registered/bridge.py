# SPDX-License-Identifier: MIT
"""Connect real serial CacheBlend inputs to the managed native span service.

The original LocalCPU store and lookup remain scheduling controls. They never
provide the model's retrieved K/V in this arm. This duplicates storage and
therefore does not establish serving latency or storage capacity savings.
"""

from __future__ import annotations

import functools
import hashlib
import json
import time
from typing import Any


def identity(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class RegisteredReplay:
    """Own one worker and the first canonical source in each logical domain."""

    def __init__(
        self,
        client: Any,
        artifacts: dict[str, str],
        ledger: Any,
        abort_importer: Any,
        *,
        before_consume: Any = None,
        after_consume: Any = None,
    ):
        self.client = client
        self.artifacts = artifacts
        self.ledger = ledger
        self.abort_importer = abort_importer
        self.before_consume, self.after_consume = before_consume, after_consume
        self.worker = None
        self.handles: dict[tuple[Any, ...], Any] = {}
        self.sources = []
        self.disabled = False
        self.omitted_sources = set()
        self.originals = []

    def _worker(self, engine: Any) -> Any:
        if self.worker is None:
            from lmcache.integration.vllm.kmesh_span_client import (
                RegisteredSpanWorker,
                SpanWorkerArtifacts,
            )
            from lmcache.integration.vllm.utils import ENGINE_NAME
            from lmcache.v1.compute.blend.utils import LMCBlenderBuilder

            blender = LMCBlenderBuilder.get(ENGINE_NAME)
            model = blender.layerwise_model.vllm_model
            rotary = model.model.layers[0].self_attn.rotary_emb
            caches = tuple(engine.gpu_connector.kvcaches)
            # The frozen model uses token-major fused K/V content. These are
            # the connector's actual normalized views sharing engine storage.
            if len(caches) != 24 or any(
                tuple(cache.shape[1:]) != (32, 2, 128) or cache.ndim != 4
                for cache in caches
            ):
                raise ValueError(
                    "model replay requires actual 24-layer fused NHD views"
                )
            artifacts = SpanWorkerArtifacts(
                self.artifacts["weights"],
                identity(model.config.to_dict()),
                self.artifacts["tokenizer"],
                self.artifacts["execution"],
            )
            self.worker = RegisteredSpanWorker.connect(
                self.client,
                caches,
                rotary.cos_sin_cache,
                layout="NHD",
                model_name=str(engine.metadata.model_name),
                is_neox_style=rotary.is_neox_style,
                artifacts=artifacts,
            )
            self.ledger.emit(
                "registered.worker.ready",
                artifacts=vars(artifacts),
                layout="NHD",
                physical_format="NL_X_NB_BS_NH_CS",
                layer_count=len(caches),
            )
        return self.worker

    @staticmethod
    def _key(tokens: tuple[int, ...], configs: dict[str, Any]) -> tuple[Any, ...]:
        from lmcache.integration.vllm.kmesh import (
            KMESH_DOMAIN_TAG_KEY,
            KMESH_RUNTIME_TAG_KEY,
        )

        domain, runtime = configs[KMESH_DOMAIN_TAG_KEY], configs[KMESH_RUNTIME_TAG_KEY]
        if (
            type(domain) is not str
            or type(runtime) is not str
            or not domain
            or not runtime
        ):
            raise ValueError("actual validated cache-domain tags are required")
        return domain, runtime, identity(tokens), len(tokens)

    def publish(self, engine: Any, tokens: Any, spans: Any, kwargs: dict) -> None:
        if self.disabled:
            if self.worker is not None and self.worker.pending:
                self.worker.drain_pending()
            return
        worker = self._worker(engine)
        token_ids = tuple(int(t) for t in tokens)
        blocks = worker.block_table(tuple(kwargs["slot_mapping"].cpu().tolist()))
        for span in spans.spans:
            content = token_ids[span.start : span.end]
            key = self._key(content, kwargs["request_configs"])
            if key in self.handles:
                continue
            try:
                handle = worker.store(
                    content, span.start, identity(token_ids[: span.start]), blocks
                )
            except BaseException:
                if worker.pending:
                    # The reference store session knows only its own cleanup.
                    # Stop and join the owned native importer before its error
                    # can unwind into scheduler block reuse.
                    self.abort_importer()
                raise
            self.handles[key] = handle
            self.ledger.emit(
                "registered.span.stored",
                request_id=kwargs.get("req_id"),
                source_start=span.start,
                token_count=len(content),
                content_digest=key[2],
                handle=handle,
            )

    def consume(
        self, blender: Any, tokens: Any, mask: Any, kwargs: dict, original: Any
    ):
        import torch

        if self.disabled:
            if self.worker is not None and self.worker.pending:
                from lmcache.v1.compute.blend.layer_source import (
                    UnsettledLayerSourceError,
                )

                raise UnsettledLayerSourceError(
                    "disabled replay still owns physical work",
                    self.worker.drain_pending,
                )
            self.ledger.emit(
                "registered.model.disabled", request_id=kwargs.get("request_id")
            )
            return torch.zeros(len(tokens), dtype=torch.bool)
        token_ids = tuple(int(t) for t in tokens)
        spans = kwargs["explicit_spans"]
        if self.worker is None:
            raise RuntimeError("scheduler hit has no registered worker/source")
        start = time.perf_counter_ns()
        if self.before_consume is not None:
            self.before_consume(kwargs.get("request_id"))
        ready = time.perf_counter_ns()
        blocks = self.worker.block_table(tuple(kwargs["slot_mapping"].cpu().tolist()))
        placements = []
        for span in spans.spans:
            key = self._key(token_ids[span.start : span.end], kwargs["request_configs"])
            handle = self.handles.get(key)
            if handle is None or key in self.omitted_sources:
                continue
            placed = self.worker.restore(
                handle, span.start, identity(token_ids[: span.start]), blocks
            )
            placements.append(placed)
            self.ledger.emit(
                "registered.span.restored",
                request_id=kwargs.get("request_id"),
                source_start=handle.source_start,
                destination_start=span.start,
                token_count=handle.token_count,
                receipt=placed.receipt,
            )
        if not placements:
            self.ledger.emit(
                "registered.model.miss", request_id=kwargs.get("request_id")
            )
            return torch.zeros(len(tokens), dtype=torch.bool)
        source = self.worker.layer_source(token_ids, blocks, spans, tuple(placements))
        self.sources.append(source)
        try:
            result = original(blender, tokens, mask, layer_source=source, **kwargs)
        finally:
            # This refuses a still-live generator and preserves pending physical
            # debt through the typed cleanup gate. It never releases by fiat.
            source.close()
        finished = time.perf_counter_ns()
        elapsed, processing = finished - start, finished - ready
        if self.after_consume is not None:
            self.after_consume(kwargs.get("request_id"))
        self.ledger.emit(
            "registered.model.consumed",
            request_id=kwargs.get("request_id"),
            span_count=len(placements),
            restored_tokens=sum(p.receipt.span.token_count for p in placements),
            returned_mask=[bool(x) for x in result.cpu().tolist()],
            source_closed=source.is_closed,
            pending=source.pending,
            elapsed_ns=elapsed,
            processing_ns=processing,
        )
        return result

    def install(self) -> None:
        from lmcache.v1.cache_engine import LMCacheEngine
        from lmcache.v1.compute.blend.blender import LMCBlender

        original_store, original_blend = LMCacheEngine.store_layer, LMCBlender.blend

        @functools.wraps(original_store)
        def store(engine, tokens, mask=None, explicit_spans=None, **kwargs):
            stream = original_store(
                engine, tokens, mask=mask, explicit_spans=explicit_spans, **kwargs
            )
            try:
                for index, value in enumerate(stream):
                    if explicit_spans is not None and index == engine.num_layers:
                        # Final admission is observed before yielding back to
                        # wait_for_save, while the real worker owns source blocks.
                        self.publish(engine, tokens, explicit_spans, kwargs)
                    yield value
            finally:
                stream.close()

        @functools.wraps(original_blend)
        def blend(blender, tokens, mask=None, **kwargs):
            try:
                return self.consume(blender, tokens, mask, kwargs, original_blend)
            except Exception:
                self.disabled = True
                self.ledger.emit(
                    "registered.backend.disabled", request_id=kwargs.get("request_id")
                )
                raise

        self.originals = [
            (LMCacheEngine, "store_layer", original_store),
            (LMCBlender, "blend", original_blend),
        ]
        LMCacheEngine.store_layer, LMCBlender.blend = store, blend

    def close(self) -> None:
        if self.worker is not None:
            release = self.worker.close()
            self.ledger.emit("registered.worker.retired", receipt=release)
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)
        self.originals.clear()

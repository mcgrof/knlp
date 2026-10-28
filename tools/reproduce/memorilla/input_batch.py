# SPDX-License-Identifier: MIT
"""Assemble owned document batches while each external bank stays leased."""

from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import threading

import torch
from memorilla.data import MemoryCollator

from .tensor_copy import DIRECT_ELEMENTS, copy_fp16_cpu


class _BankLoan:
    def __init__(self, context, tensor, released):
        self.context = context
        self.tensor = tensor
        self.released = released
        self.lock = threading.Lock()

    def close(self):
        with self.lock:
            if self.context is None:
                return
            context, released = self.context, self.released
            self.context = self.tensor = self.released = None
            try:
                context.__exit__(None, None, None)
            finally:
                released()


class LeasedMemoryCollator(MemoryCollator):
    """Copy document banks directly into the author's owned padded batch.

    External stores provide collection_shape() and copy_documents(). The
    latter must finish its CPU copy before releasing the source lease.
    Ordinary stores may return mmap views through documents(); these views
    remain referenced until the batch copies finish. The optional bfloat16
    destination casts only valid rows and zeros padding directly, avoiding
    a padded float16 intermediate. Prompt, mask and question processing
    otherwise follow the original collator.

    An immutable mapped store can expose borrowed_documents(). Its returned
    tensors own their mapping lifetime without a separate per-copy lease.

    Calls are serialized. Each copy worker holds at most one bank lease and
    releases it before accepting another task, so aggregate batch bytes may
    exceed the storage arena. A failed batch drains every accepted copy and
    is never returned to the reader.
    Stores and their collection identities must remain immutable for this
    collator's lifetime. Optional serial read-ahead uses one FIFO reader and
    at most two bank loans, overlapping verification with foreground copying.
    """

    def __init__(
        self,
        *args,
        clients=1,
        final_dtype=torch.float16,
        prefetch=False,
        max_geometries=1024,
        **kwargs,
    ):
        if type(clients) is not int or clients < 1:
            raise ValueError("clients must be a positive integer")
        if final_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("batch dtype must be float16 or bfloat16")
        if prefetch and final_dtype != torch.bfloat16:
            raise ValueError("document prefetch requires bfloat16 assembly")
        if type(max_geometries) is not int or max_geometries < 1:
            raise ValueError("geometry entry limit must be positive")
        super().__init__(*args, **kwargs)
        self._final_dtype = final_dtype
        self._prefetch = prefetch and clients == 1
        self._reader = None
        self._geometries = OrderedDict()
        self._max_geometries = max_geometries
        self._stat_lock = threading.Lock()
        self._live_loans = self._peak_loans = self._prefetch_batches = 0
        self._executor = (
            ThreadPoolExecutor(max_workers=clients) if clients > 1 else None
        )
        self._batch_lock = threading.RLock()
        self._closed = False

    def _shape(self, store, key):
        identity = (id(store), getattr(store, "set_id", None), key)
        shape = self._geometries.get(identity)
        if shape is None:
            shape = store.collection_shape(key)
            if len(shape) != 2 or any(type(n) is not int or n < 1 for n in shape):
                raise ValueError("invalid document bank geometry")
            shape = tuple(shape)
            self._geometries[identity] = shape
            if len(self._geometries) > self._max_geometries:
                self._geometries.popitem(last=False)
        self._geometries.move_to_end(identity)
        return shape

    def _open_bank(self, source, shape):
        store, key = source
        context = store.lease_documents(key)
        tensor = context.__enter__()
        counted = False
        try:
            if (
                tensor.device.type != "cpu"
                or tensor.dtype != torch.float16
                or tuple(tensor.shape) != shape
            ):
                raise ValueError(
                    "borrowed document bank differs from admitted geometry"
                )
            with self._stat_lock:
                self._live_loans += 1
                self._peak_loans = max(self._peak_loans, self._live_loans)
            counted = True

            def released():
                with self._stat_lock:
                    self._live_loans -= 1

            return _BankLoan(context, tensor, released)
        except BaseException:
            try:
                context.__exit__(None, None, None)
            finally:
                if counted:
                    released()
            raise

    def _copy_prefetched(self, sources, embeds, mask):
        if self._reader is None:
            self._reader = ThreadPoolExecutor(max_workers=1)
        futures = []

        def submit(index):
            source, shape = sources[index]
            futures.append(self._reader.submit(self._open_bank, source, shape))

        try:
            for index in range(min(2, len(sources))):
                submit(index)
            for index, (_, shape) in enumerate(sources):
                loan = futures[index].result()
                try:
                    copy_fp16_cpu(loan.tensor, embeds[index, : shape[0]])
                    embeds[index, shape[0] :].zero_()
                    mask[index, : shape[0]] = False
                finally:
                    loan.close()
                # Release the consumed bank before adding another request.
                # One FIFO reader plus a two-bank window avoids an ordered
                # consumer waiting behind later leases that fill the arena.
                if index + 2 < len(sources):
                    submit(index + 2)
            self._prefetch_batches += 1
        except BaseException as error:
            cleanup_errors = []
            for future in futures:
                future.cancel()
            for future in futures:
                try:
                    loan = future.result()
                except BaseException:
                    continue
                try:
                    loan.close()
                except BaseException as cleanup_error:
                    cleanup_errors.append(repr(cleanup_error))
            if cleanup_errors:
                raise RuntimeError(
                    f"document lease cleanup failed: {cleanup_errors}"
                ) from error
            raise

    def __call__(self, batch):
        with self._batch_lock:
            if self._closed:
                raise RuntimeError("document collator is closed")
            return super().__call__(batch)

    def _documents(self, batch):
        with self._batch_lock:
            if self._closed:
                raise RuntimeError("document collator is closed")
            sources = []
            for item in batch:
                store = self.stores[(item["config"], item["split"])]
                key = item["collection_id"]
                if callable(getattr(store, "borrowed_documents", None)):
                    source = store.borrowed_documents(key)
                    shape = tuple(source.shape)
                    if len(shape) != 2 or any(
                        type(n) is not int or n < 1 for n in shape
                    ):
                        raise ValueError("invalid document bank geometry")
                elif callable(getattr(store, "copy_documents", None)):
                    shape = self._shape(store, key)
                    source = (store, key)
                else:
                    source = store.documents(key)
                    shape = tuple(source.shape)
                    if len(shape) != 2 or any(
                        type(n) is not int or n < 1 for n in shape
                    ):
                        raise ValueError("invalid document bank geometry")
                sources.append((source, shape))
            if not sources or len({shape[1] for _, shape in sources}) != 1:
                raise ValueError("document banks need a shared positive width")
            max_docs = max(shape[0] for _, shape in sources)
            width = sources[0][1][1]
            if self._final_dtype == torch.float16:
                embeds = torch.zeros(len(sources), max_docs, width, dtype=torch.float16)
            else:
                # Every valid row is overwritten before exposure. Initialize
                # only padding, whose conversion would otherwise do no useful
                # work. An incomplete destination never leaves this method.
                embeds = torch.empty(
                    len(sources), max_docs, width, dtype=torch.bfloat16
                )
            mask = torch.ones(len(sources), max_docs, dtype=torch.bool)

            def copy_row(index):
                source, shape = sources[index]
                destination = embeds[index, : shape[0]]
                if isinstance(source, tuple):
                    store, key = source
                    store.copy_documents(key, destination)
                else:
                    if source.dtype != torch.float16 or source.device.type != "cpu":
                        raise ValueError(
                            "document banks must contain CPU float16 values"
                        )
                    if (
                        self._final_dtype == torch.float16
                        or source.numel() <= DIRECT_ELEMENTS
                    ):
                        destination.copy_(source)
                    else:
                        copy_fp16_cpu(source, destination)
                if self._final_dtype == torch.bfloat16:
                    embeds[index, shape[0] :].zero_()
                mask[index, : shape[0]] = False

            pipeline = (
                self._prefetch
                and all(
                    isinstance(source, tuple)
                    and callable(getattr(source[0], "lease_documents", None))
                    and getattr(source[0], "prefetch_useful", True)
                    for source, _ in sources
                )
                and any(shape[0] * shape[1] > DIRECT_ELEMENTS for _, shape in sources)
            )
            if pipeline:
                self._copy_prefetched(sources, embeds, mask)
            elif self._executor is None:
                for index in range(len(sources)):
                    copy_row(index)
            else:
                futures = []
                try:
                    for index in range(len(sources)):
                        futures.append(self._executor.submit(copy_row, index))
                    for future in futures:
                        future.result()
                except BaseException:
                    for future in futures:
                        future.cancel()
                    # A cancelled queued task has no lease. Running copies
                    # must finish before destination buffers can be released.
                    for future in futures:
                        try:
                            future.result()
                        except BaseException:
                            pass
                    raise
            return embeds, mask

    def stats(self):
        with self._stat_lock:
            return {
                "prefetch_batches": self._prefetch_batches,
                "active_prefetch_loans": self._live_loans,
                "peak_prefetch_loans": self._peak_loans,
                "geometry_entries": len(self._geometries),
            }

    def close(self):
        """Drain copies before dropping the executor; safe to call twice."""
        with self._batch_lock:
            if self._closed:
                return
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
            if self._reader is not None:
                self._reader.shutdown(wait=True, cancel_futures=True)
            self._geometries.clear()
            self._closed = True

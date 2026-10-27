# SPDX-License-Identifier: MIT
"""Assemble owned document batches while each external bank stays leased."""

from concurrent.futures import ThreadPoolExecutor
import threading

import torch
from memorilla.data import MemoryCollator


class LeasedMemoryCollator(MemoryCollator):
    """Copy document banks directly into the author's owned padded batch.

    External stores provide collection_shape() and copy_documents(). The
    latter must finish its CPU copy before releasing the source lease.
    Ordinary stores may return mmap views through documents(); these views
    remain referenced until the batch copies finish. Prompt, mask, question
    and dtype processing otherwise follow the original collator.

    Calls are serialized. Each copy worker holds at most one bank lease and
    releases it before accepting another task, so aggregate batch bytes may
    exceed the storage arena. A failed batch drains every accepted copy and
    is never returned to the reader.
    """

    def __init__(self, *args, clients=1, final_dtype=torch.float16, **kwargs):
        if type(clients) is not int or clients < 1:
            raise ValueError("clients must be a positive integer")
        if final_dtype != torch.float16:
            raise ValueError("only float16 batch assembly is supported")
        super().__init__(*args, **kwargs)
        self._executor = (
            ThreadPoolExecutor(max_workers=clients) if clients > 1 else None
        )
        self._batch_lock = threading.Lock()
        self._closed = False

    def _documents(self, batch):
        with self._batch_lock:
            if self._closed:
                raise RuntimeError("document collator is closed")
            sources = []
            for item in batch:
                store = self.stores[(item["config"], item["split"])]
                key = item["collection_id"]
                if callable(getattr(store, "copy_documents", None)):
                    shape = store.collection_shape(key)
                    source = (store, key)
                else:
                    source = store.documents(key)
                    shape = source.shape
                if len(shape) != 2 or any(type(n) is not int or n < 1 for n in shape):
                    raise ValueError("invalid document bank geometry")
                sources.append((source, tuple(int(n) for n in shape)))
            if not sources or len({shape[1] for _, shape in sources}) != 1:
                raise ValueError("document banks need a shared positive width")
            max_docs = max(shape[0] for _, shape in sources)
            width = sources[0][1][1]
            embeds = torch.zeros(len(sources), max_docs, width, dtype=torch.float16)
            mask = torch.ones(len(sources), max_docs, dtype=torch.bool)

            def copy_row(index):
                source, shape = sources[index]
                destination = embeds[index, : shape[0]]
                if isinstance(source, tuple):
                    store, key = source
                    store.copy_documents(key, destination)
                else:
                    if source.dtype != torch.float16:
                        raise ValueError("document banks must contain float16 values")
                    destination.copy_(source)
                mask[index, : shape[0]] = False

            if self._executor is None:
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

    def close(self):
        """Drain copies before dropping the executor; safe to call twice."""
        with self._batch_lock:
            if self._closed:
                return
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
            self._closed = True

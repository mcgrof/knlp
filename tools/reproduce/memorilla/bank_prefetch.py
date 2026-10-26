# SPDX-License-Identifier: MIT
"""Concurrent bank fetches feeding an unchanged, ordered decoder batch."""

import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor


class ConcurrentBanks:
    """Own prefetched tensors until the original collator consumes them.

    The engine and learned reader stay on the calling thread. Completion
    order never changes request order or decoder batch composition. Native
    mmap clients may supply a factory to keep their handles thread-local.
    """

    def __init__(self, store, clients, factory=None, timeout=120):
        self.store = store
        self.factory = factory
        self.local = threading.local()
        self.executor = ThreadPoolExecutor(max_workers=clients)
        self.prepared = defaultdict(deque)
        self.timeout = timeout
        self.retries = 0
        self.counter_lock = threading.Lock()

    def _load(self, collection):
        if self.factory and not hasattr(self.local, "store"):
            self.local.store = self.factory()
        store = self.local.store if self.factory else self.store
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                return store.documents(collection)
            except MemoryError:
                with self.counter_lock:
                    self.retries += 1
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "concurrent bank admission exceeded its deadline"
                    )
                time.sleep(0.001)

    def prepare(self, collections):
        self.finish()
        futures = [self.executor.submit(self._load, key) for key in collections]
        for key, future in zip(collections, futures, strict=True):
            self.prepared[key].append(future.result())

    def documents(self, collection):
        rows = self.prepared.get(collection)
        if not rows:
            raise RuntimeError("collator requested an unprepared bank")
        return rows.popleft()

    def question(self, row):
        return self.store.question(row)

    def finish(self):
        if any(self.prepared.values()):
            raise RuntimeError("collator did not consume its complete bank batch")
        self.prepared.clear()

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.finish()

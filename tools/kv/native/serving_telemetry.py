# SPDX-License-Identifier: MIT
"""Retain raw server queue/cache metrics on the client's monotonic clock."""

import threading
import time
import urllib.request


class MetricSampler:
    def __init__(self, url, interval=1.0):
        self.url = url
        self.interval = interval
        self.rows = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            row = {"start_ns": time.monotonic_ns()}
            try:
                with urllib.request.urlopen(
                    self.url + "/metrics", timeout=1
                ) as response:
                    row["http_status"] = response.status
                    row["prometheus_text"] = response.read().decode()
            except Exception as exc:
                row["error"] = repr(exc)
            row["end_ns"] = time.monotonic_ns()
            self.rows.append(row)
            self.stop.wait(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            raise RuntimeError("server metric sampler did not stop")

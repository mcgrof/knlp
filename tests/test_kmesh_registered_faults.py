# SPDX-License-Identifier: MIT
"""Keep replay fault injection ordered after the real operation it observes."""

from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from tools.reproduce.kmesh_registered.faults import install_service, install_worker


@contextmanager
def modules(values):
    """Expose narrow CPU stand-ins without importing a CUDA runtime."""
    entries = {}
    for name in values:
        parts = name.split(".")
        for count in range(1, len(parts) + 1):
            path = ".".join(parts[:count])
            if path not in entries:
                entries[path] = ModuleType(path)
                entries[path].__path__ = []
            if count > 1:
                setattr(
                    entries[".".join(parts[: count - 1])],
                    parts[count - 1],
                    entries[path],
                )
    for name, attrs in values.items():
        vars(entries[name]).update(attrs)
    with patch.dict("sys.modules", entries):
        yield entries


class InjectionOrdering(unittest.TestCase):
    def service(self, mode):
        calls = []

        class Plan:
            def enqueue(self):
                calls.append("native enqueue")

        class Storage:
            def unsafe_read(self):
                return None, []

            def submit_prefetch_task(self, spec):
                return None

        values = {
            "lmcache.v1.distributed.storage_manager": {"StorageManager": Storage},
            "lmcache.v1.multiprocess.kmesh_span_store": {
                "synchronize_device": lambda _: calls.append("physical drain")
            },
            "lmcache.v1.multiprocess.modules.blend.retrieve": {
                "RegisteredRetrievePlan": Plan
            },
        }
        return calls, Plan, values

    def test_post_enqueue_refusal_keeps_the_genuine_submission(self):
        calls, plan, values = self.service("post_enqueue")
        with modules(values):
            state = install_service("post_enqueue")
            state.armed = True
            with self.assertRaisesRegex(RuntimeError, "after actual native enqueue"):
                plan().enqueue()
        self.assertEqual(calls, ["native enqueue"])
        self.assertEqual(state.observed["enqueues"], 1)

    def test_transient_refusal_requires_a_later_physical_drain(self):
        calls, _, values = self.service("transient_drain")
        with modules(values) as loaded:
            state = install_service("transient_drain")
            state.armed = True
            synchronize = loaded[
                "lmcache.v1.multiprocess.kmesh_span_store"
            ].synchronize_device
            with self.assertRaisesRegex(RuntimeError, "before actual service"):
                synchronize("device")
            self.assertEqual(calls, [])
            synchronize("device")
        self.assertEqual(calls, ["physical drain"])
        self.assertEqual(state.observed["refused_drains"], 1)
        self.assertEqual(state.observed["drains"], 1)

    def test_worker_refusal_starts_only_after_a_genuine_gather(self):
        calls = []
        native = {"TransferDirection": SimpleNamespace(D2H="read")}
        ops = SimpleNamespace(
            multi_layer_kv_transfer=lambda *args: calls.append("gather")
        )
        values = {
            "lmcache.lmcache_native": native,
            "lmcache.integration.vllm.kmesh_span_layers": {
                "device_ops": ops,
                "synchronize_device": lambda _: calls.append("drain"),
            },
        }
        with modules(values) as loaded:
            state = install_worker("worker_drain")
            state.armed = True
            layer = loaded["lmcache.integration.vllm.kmesh_span_layers"]
            layer.synchronize_device("device")
            ops.multi_layer_kv_transfer(None, None, None, None, None, "read")
            with self.assertRaisesRegex(RuntimeError, "after actual gather"):
                layer.synchronize_device("device")
        self.assertEqual(calls, ["drain", "gather"])
        self.assertEqual(state.observed["worker_gathers"], 1)


if __name__ == "__main__":
    unittest.main()

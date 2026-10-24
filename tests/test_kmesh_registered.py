# SPDX-License-Identifier: MIT
"""Protect source-block ownership at the real layer-store generator boundary."""

from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from tools.reproduce.kmesh_registered.bridge import RegisteredReplay


class PublicationBoundary(unittest.TestCase):
    def exercise(self, stop_early=False, reject=False):
        state = {"alive": False, "admitted": False, "published": False}

        class Engine:
            num_layers = 2

            def store_layer(self, tokens, **kwargs):
                state["alive"] = True
                try:
                    yield
                    yield
                    state["admitted"] = True
                    yield
                finally:
                    state["alive"] = False

        class Blender:
            def blend(self, *args, **kwargs):
                raise AssertionError("model is outside this CPU ownership test")

        cache = ModuleType("lmcache.v1.cache_engine")
        cache.LMCacheEngine = Engine
        blend = ModuleType("lmcache.v1.compute.blend.blender")
        blend.LMCBlender = Blender
        replay = RegisteredReplay(None, {}, None, lambda: None)

        def publish(*args):
            self.assertTrue(state["alive"])
            self.assertTrue(state["admitted"])
            state["published"] = True
            if reject:
                raise RuntimeError("native admission refused")

        replay.publish = publish
        with patch.dict("sys.modules", {cache.__name__: cache, blend.__name__: blend}):
            replay.install()
            stream = Engine().store_layer([1], explicit_spans=SimpleNamespace())
            try:
                next(stream)
                self.assertFalse(state["published"])
                next(stream)
                self.assertFalse(state["published"])
                if not stop_early:
                    if reject:
                        with self.assertRaisesRegex(RuntimeError, "admission refused"):
                            next(stream)
                    else:
                        next(stream)
                        self.assertTrue(state["published"])
                        self.assertTrue(state["alive"])
            finally:
                stream.close()
                replay.close()
        self.assertFalse(state["alive"])
        return state

    def test_publish_before_final_yield_while_blocks_are_owned(self):
        self.assertTrue(self.exercise()["published"])

    def test_early_close_never_publishes_incomplete_layers(self):
        self.assertFalse(self.exercise(stop_early=True)["published"])

    def test_failed_admission_still_closes_the_reference_generator(self):
        self.assertTrue(self.exercise(reject=True)["published"])

    def test_uncertain_native_store_stops_importer_before_error_unwinds(self):
        state = {"importer_alive": True}

        class Worker:
            pending = True

            def block_table(self, slots):
                return (1,)

            def store(self, *args):
                raise RuntimeError("store reply was lost")

        def stop_importer():
            state["importer_alive"] = False

        replay = RegisteredReplay(None, {}, None, stop_importer)
        replay.worker = Worker()
        replay._key = lambda *args: ("domain", "runtime", "content", 1)
        mapping = SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: [32]))
        with self.assertRaisesRegex(RuntimeError, "reply was lost"):
            replay.publish(
                None,
                [1],
                SimpleNamespace(spans=(SimpleNamespace(start=0, end=1),)),
                {"slot_mapping": mapping, "request_configs": {}},
            )
        self.assertFalse(state["importer_alive"])


if __name__ == "__main__":
    unittest.main()

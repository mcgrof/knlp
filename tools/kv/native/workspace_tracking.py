# SPDX-License-Identifier: MIT
"""Observe allocations without relying on GC generation membership."""

import weakref


class InstanceTracker:
    def __init__(self, cls):
        self.cls = cls
        self.live = weakref.WeakValueDictionary()
        self.original = cls.__init__

    def start(self):
        tracker = self

        def initialize(instance, *args, **kwargs):
            tracker.original(instance, *args, **kwargs)
            tracker.live[id(instance)] = instance

        self.cls.__init__ = initialize

    def stop(self):
        self.cls.__init__ = self.original

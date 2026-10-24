# SPDX-License-Identifier: MIT
"""Observe actual model repair selection through the public Q/K/V method."""

import functools
import inspect


def install(ledger):
    from lmcache.v1.compute.blend.blender import LMCBlender

    original = LMCBlender.process_qkv
    signature = inspect.signature(original)

    @functools.wraps(original)
    def process(*args, **kwargs):
        bound = signature.bind(*args, **kwargs).arguments
        result = original(*args, **kwargs)
        session = bound["session"]
        selected = session.metadata.imp_indices
        ledger.emit(
            "registered.model.selection",
            request_id=session.request_id,
            layer_id=bound["layer_id"],
            selected_indices=None if selected is None else selected.cpu().tolist(),
        )
        return result

    LMCBlender.process_qkv = process

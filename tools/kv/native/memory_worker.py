# SPDX-License-Identifier: MIT
"""Observe cache materialization while delegating every operation to vLLM."""

import json
import os
import time
from pathlib import Path

import torch
from vllm.v1.worker.gpu_worker import Worker

from .instrument import tensor_inventory


def snapshot(phase, result=None):
    name = os.environ.get("KNLP_NATIVE_ALLOC_TRACE")
    if not name:
        return
    free, total = torch.cuda.mem_get_info()
    if isinstance(result, dict):
        result = list(result.values())
    row = {
        "phase": phase,
        "pid": os.getpid(),
        "monotonic": time.monotonic(),
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "peak_allocated_since_start": torch.cuda.max_memory_allocated(),
        "driver_device_used_bytes": total - free,
        "returned_tensors": tensor_inventory(result),
    }
    path = Path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")


class ObservedWorker(Worker):
    def init_device(self):
        result = super().init_device()
        limit = os.environ.get("KNLP_NATIVE_TORCH_LIMIT_BYTES")
        if limit:
            torch.cuda.set_per_process_memory_fraction(
                int(limit) / torch.cuda.get_device_properties(0).total_memory
            )
        snapshot("device_initialized_before_model_load")
        return result

    def initialize_from_config(self, kv_cache_config):
        from vllm.v1.worker.gpu import attn_utils

        originals = {
            name: getattr(attn_utils, name)
            for name in ("_allocate_kv_cache", "_reshape_kv_cache")
        }
        for name, original in originals.items():

            def observed(*args, _name=name, _original=original, **kwargs):
                snapshot("before" + _name)
                result = _original(*args, **kwargs)
                snapshot("after" + _name, result)
                return result

            setattr(attn_utils, name, observed)
        snapshot("before_cache_initialization")
        try:
            return super().initialize_from_config(kv_cache_config)
        finally:
            for name, original in originals.items():
                setattr(attn_utils, name, original)
            snapshot("after_cache_initialization")

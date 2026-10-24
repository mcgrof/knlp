# SPDX-License-Identifier: MIT
"""Bounded fault injection in the replay's owned process and worker only.

These hooks either mutate an actual stored source, delete its actual L1 key,
or refuse a completion after genuine native work. They never fabricate a
successful receipt, device completion, or scheduler reservation.
"""

from dataclasses import dataclass, field
from functools import wraps


@dataclass
class FaultState:
    mode: str
    armed: bool = False
    observed: dict = field(
        default_factory=lambda: {
            "enqueues": 0,
            "drains": 0,
            "refused_drains": 0,
            "corruptions": 0,
            "evictions": 0,
            "worker_gathers": 0,
        }
    )


def install_service(mode):
    from lmcache.v1.distributed.storage_manager import StorageManager
    from lmcache.v1.multiprocess import kmesh_span_store as spans
    from lmcache.v1.multiprocess.modules.blend.retrieve import RegisteredRetrievePlan

    state = FaultState(mode)
    enqueue, drain = RegisteredRetrievePlan.enqueue, spans.synchronize_device
    read, prefetch = StorageManager.unsafe_read, StorageManager.submit_prefetch_task

    @wraps(enqueue)
    def submit(plan):
        enqueue(plan)
        state.observed["enqueues"] += 1
        if state.armed and mode in (
            "post_enqueue",
            "transient_drain",
            "persistent_drain",
        ):
            raise RuntimeError("injected refusal after actual native enqueue")

    @wraps(drain)
    def synchronize(device):
        if state.armed and (
            mode == "persistent_drain"
            or (mode == "transient_drain" and state.observed["refused_drains"] == 0)
        ):
            state.observed["refused_drains"] += 1
            raise RuntimeError("injected refusal before actual service device drain")
        drain(device)
        state.observed["drains"] += 1

    @wraps(read)
    def read_source(manager, *args, **kwargs):
        result = read(manager, *args, **kwargs)
        if state.armed and mode == "corrupt" and result[1]:
            tensor = result[1][0].get_tensor(0)
            if tensor is None:
                raise RuntimeError("corruption control has no actual source tensor")
            tensor.view(-1)[0].neg_()
            state.observed["corruptions"] += 1
        return result

    @wraps(prefetch)
    def prefetch_source(manager, spec, *args, **kwargs):
        if state.armed and mode == "evict":
            keys = spec.key_groups[0].keys
            deleted, failed = manager.delete_l1_keys(keys)
            if failed or deleted != len(keys):
                raise RuntimeError("eviction control did not delete actual L1 keys")
            state.observed["evictions"] += deleted
        return prefetch(manager, spec, *args, **kwargs)

    RegisteredRetrievePlan.enqueue, spans.synchronize_device = submit, synchronize
    StorageManager.unsafe_read, StorageManager.submit_prefetch_task = (
        read_source,
        prefetch_source,
    )
    return state


def install_worker(mode):
    from lmcache.integration.vllm import kmesh_span_layers as layers
    import lmcache.lmcache_native as native

    state = FaultState(mode)
    gather, drain = layers.device_ops.multi_layer_kv_transfer, layers.synchronize_device

    @wraps(gather)
    def copy(*args, **kwargs):
        result = gather(*args, **kwargs)
        if state.armed and args[5] == native.TransferDirection.D2H:
            state.observed["worker_gathers"] += 1
        return result

    @wraps(drain)
    def synchronize(device):
        if state.armed and mode == "worker_drain" and state.observed["worker_gathers"]:
            state.observed["refused_drains"] += 1
            raise RuntimeError("injected worker drain refusal after actual gather")
        drain(device)
        state.observed["drains"] += 1

    layers.device_ops.multi_layer_kv_transfer, layers.synchronize_device = (
        copy,
        synchronize,
    )
    return state

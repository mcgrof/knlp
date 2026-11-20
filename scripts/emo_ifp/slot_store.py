#!/usr/bin/env python3
"""Expert slots in GPU memory, filled from one copy of each expert (GPU).

A pool server that switches by reading per-pair delta files builds each
new layer in a fresh buffer: it copies the experts both pools keep from
the resident layer and writes the loaded ones beside them, then frees
the old layer. This module keeps instead, per layer, one fixed GPU
buffer of S expert slots, allocated and registered for io_uring
DMA-BUF reads once. S is the pool size plus the largest number of
experts any switch between the given pools brings into that layer, so
a switch always finds enough free slots while the resident pool keeps
running on its own.

A switch from pool A to pool B reads only the experts of B that are not
in a slot, from the single-copy layer files (pack_single_copy.py),
straight into free slots; experts B shares with A stay where they are,
so nothing is copied on the GPU. The layer then runs on B through a
slot map: the router picks among B's experts, and the fused kernel
reads expert k of B from slot map[k] (fused_moe.FusedEmoMoe.slot_map).
Once the switch is done, the slots of experts B does not keep become
free. Reads into free slots never touch a slot the resident pool uses,
so a switch can run while requests are served on the resident pool.

Each switch's experts are read in file order; consecutive experts that
land in consecutive slots share one read, so a fresh store reads a pool
in its file runs, and later switches split reads where free slots are
not contiguous.
"""

import json
import os
import threading
import time

import torch

from fastsafetensors.dlpack import from_cuda_buffer
from fastsafetensors.io_uring_dmabuf import (
    IoUring,
    dedicated_malloc,
    register_destination,
    release_dedicated,
    release_regions,
)
from fastsafetensors.st_types import Device, DType

ALIGN = 4096
IS_HIP = torch.version.hip is not None


def max_entering(pools, li):
    """Largest number of experts any switch between pools brings in."""
    sets = {n: set(p["layers"][str(li)]["routed"]) for n, p in pools.items()}
    return max(
        (len(sets[b] - sets[a]) for a in sets for b in sets if a != b), default=0
    )


class SlotStore:
    """Per layer, S registered expert slots and the pool that fills them."""

    def __init__(self, single_dir, pools, shape13, shape2, dtype=torch.bfloat16):
        with open(os.path.join(single_dir, "LAYOUT.json")) as f:
            self.layout = json.load(f)
        missing = set(pools) - set(self.layout["pools"])
        assert not missing, f"pools not in the single-copy layout: {missing}"
        self.pools = pools
        self.nl = len(self.layout["layers"])
        # DLPack tags device memory as ROCm or CUDA from the runtime the
        # extension loaded; load it now, as the copiers do, so the tag is
        # right even before any copier has run in this process.
        from fastsafetensors import cpp as fstcpp
        from fastsafetensors.copier.nogds import load_library_func

        load_library_func()
        assert bool(fstcpp.is_hip_found()) == IS_HIP, "runtime mismatch"
        self.ring = IoUring(entries=1024, buffer_slots=self.nl * 2 + 8)
        dev = Device.from_str("cuda:0")
        self.layers = []
        for li in range(self.nl):
            lay = self.layout["layers"][str(li)]
            n = {len(p["layers"][str(li)]["routed"]) for p in pools.values()}
            assert len(n) == 1, f"pools differ in size at layer {li}: {n}"
            S = n.pop() + max_entering(pools, li)
            b13, b2 = lay["w13_bytes"], lay["w2_bytes"]
            nbytes = S * (b13 + b2)
            ptr = dedicated_malloc(nbytes + ALIGN, IS_HIP)
            base = ptr + (-ptr) % ALIGN
            path = os.path.join(single_dir, lay["file"])
            fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
            regions = register_destination(self.ring, base, nbytes, fd, True, IS_HIP)
            u8 = torch.from_dlpack(from_cuda_buffer(base, [nbytes], [1], DType.U8, dev))
            w13 = u8[: S * b13].view(dtype).view(S, *shape13)
            w2 = u8[S * b13 :].view(dtype).view(S, *shape2)
            assert w13[0].numel() * w13.element_size() == b13
            assert w2[0].numel() * w2.element_size() == b2
            pos = {e: i for i, e in enumerate(lay["order"])}
            self.layers.append(
                {
                    "S": S,
                    "ptr": ptr,
                    "base": base,
                    "fd": fd,
                    "regions": regions,
                    "u8": u8,
                    "w13": w13,
                    "w2": w2,
                    "lay": lay,
                    "pos": pos,
                    "slot_of": {},
                }
            )
        self.resident = None
        self.bytes = sum(
            L["S"] * (L["lay"]["w13_bytes"] + L["lay"]["w2_bytes"]) for L in self.layers
        )

    def slots(self):
        return [L["S"] for L in self.layers]

    def plan(self, name):
        """Per layer: the new pool's slot of each expert and the reads."""
        out = []
        for li, L in enumerate(self.layers):
            routed = self.pools[name]["layers"][str(li)]["routed"]
            held = L["slot_of"]
            used = set(held.values())
            free = [s for s in range(L["S"]) if s not in used]
            enter = sorted((e for e in routed if e not in held), key=L["pos"].get)
            assert len(enter) <= len(free), (li, len(enter), len(free))
            slot_of = {e: held[e] for e in routed if e in held}
            for e, s in zip(enter, free):
                slot_of[e] = s
            # Coalesce experts adjacent in the file and in the slots.
            reads = []
            for e in enter:
                p, s = L["pos"][e], slot_of[e]
                if (
                    reads
                    and reads[-1][0] + reads[-1][2] == p
                    and reads[-1][1] + reads[-1][2] == s
                ):
                    reads[-1][2] += 1
                else:
                    reads.append([p, s, 1])
            out.append(
                {
                    "slot_of": slot_of,
                    "map": [slot_of[e] for e in routed],
                    "reads": reads,
                    "entering": len(enter),
                }
            )
        return out

    def _queue(self, li, reads):
        """Queue one layer's reads; return [(user_data, length)]."""
        L = self.layers[li]
        lay = L["lay"]
        S, b13, b2 = L["S"], lay["w13_bytes"], lay["w2_bytes"]
        uds = []
        for p, s, c in reads:
            for file_off, dest_off, length in (
                (lay["w13_offset"] + p * b13, s * b13, c * b13),
                (lay["w2_offset"] + p * b2, S * b13 + s * b2, c * b2),
            ):
                while length:
                    addr = L["base"] + dest_off
                    reg = next(
                        r for r in L["regions"] if r.dev_start <= addr < r.dev_end
                    )
                    n = min(length, reg.dev_end - addr)
                    ud = self.ring.read_fixed(
                        L["fd"],
                        file_off,
                        reg.slot,
                        reg.buf_off + addr - reg.dev_start,
                        n,
                    )
                    uds.append((ud, n))
                    file_off, dest_off, length = file_off + n, dest_off + n, length - n
        return uds

    def commit(self, name, plan):
        for L, p in zip(self.layers, plan):
            L["slot_of"] = dict(p["slot_of"])
        self.resident = name

    def reset(self):
        for L in self.layers:
            L["slot_of"] = {}
        self.resident = None

    def close(self):
        for L in self.layers:
            err = release_regions(self.ring, L["regions"])
            if err is not None:
                raise err
            os.close(L["fd"])
            L["w13"] = L["w2"] = L["u8"] = None
        torch.cuda.synchronize()
        for L in self.layers:
            release_dedicated(L["ptr"])
        self.layers = []
        self.ring.close()


class SlotLoader:
    """PoolLoader's interface for a switch that reads into free slots.

    start() plans the switch and starts a thread that queues every
    layer's reads at once and reaps them, layer by layer, in order.
    tensors[li] holds the layer's slot map once it is loaded.
    """

    def __init__(self, store, name):
        self.store, self.name = store, name
        n = store.nl
        self.events = [threading.Event() for _ in range(n)]
        self.tensors = [None] * n
        self.t_ready = [None] * n
        self.load_info = [{"fallback": False, "copier_class": "slots"}] * n
        self.loads = [None] * n
        self.post = None
        self.err = None

    def start(self):
        self.t0 = time.perf_counter()
        self.plan = self.store.plan(self.name)
        self.maps = [
            torch.tensor(p["map"], dtype=torch.int64, device="cuda") for p in self.plan
        ]
        self.entering = [p["entering"] for p in self.plan]
        self.reads = [len(p["reads"]) for p in self.plan]
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            ring = self.store.ring
            uds = []
            for li, p in enumerate(self.plan):
                uds.append(self.store._queue(li, p["reads"]))
                ring.submit()
            for li in range(self.store.nl):
                for ud, n in uds[li]:
                    res = ring.wait(ud)
                    if res < 0:
                        raise OSError(-res, os.strerror(-res))
                    if res != n:
                        raise IOError(f"short read: {res} of {n} bytes")
                self.tensors[li] = {"map": self.maps[li]}
                self.t_ready[li] = time.perf_counter()
                self.events[li].set()
        except Exception as exc:  # surfaced to the waiting request
            self.err = exc
            for e in self.events:
                e.set()

    def wait_layer(self, i):
        self.events[i].wait()
        if self.err is not None:
            raise RuntimeError(f"slot load failed: {self.err!r}")

    def join(self):
        self.thread.join()
        if self.err is not None:
            raise RuntimeError(f"slot load failed: {self.err!r}")

    def close(self):
        pass


def set_slot_layer(model, store, li, gate, slot_map):
    """Point layer li at the slot buffer, the new router rows and map."""
    m = model.model.layers[li].mlp
    L = store.layers[li]
    m.gate.weight.data = gate
    m.w13.data = L["w13"]
    m.w2.data = L["w2"]
    m.slot_map = slot_map


def slot_wait_hooks(model, store, loader, gates):
    """Pre-hooks: each MoE layer waits for its reads, then switches."""

    def install():
        hooks = []
        for li, layer in enumerate(model.model.layers):

            def pre(mod, inp, li=li):
                loader.wait_layer(li)
                set_slot_layer(model, store, li, gates[li], loader.maps[li])

            hooks.append(layer.mlp.register_forward_pre_hook(pre))
        return hooks

    return install


def clear_slots(model):
    for layer in model.model.layers:
        layer.mlp.slot_map = None

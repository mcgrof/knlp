# SPDX-License-Identifier: MIT
"""Complete-update clocks, deterministic epoch cursors and atomic resume states."""

import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from research.mo_variants.stage_c_common import sha
from research.tinystories_depth.train import state_hash


def json_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


class TrainingClock:
    def __init__(self, device, seconds=0.0, now=time.monotonic):
        self.device, self.seconds, self.now = device, float(seconds), now
        self.last_update_seconds = 0.0
        self.active = False

    @contextlib.contextmanager
    def update(self):
        if self.active:
            raise RuntimeError("Nested training clock")
        synchronize(self.device)
        started = self.now()
        self.active = True
        try:
            yield
        finally:
            synchronize(self.device)
            self.last_update_seconds = self.now() - started
            self.seconds += self.last_update_seconds
            self.active = False


def landmark_valid(target, elapsed, preceding, last_update, fraction=0.01):
    if target == 0:
        return elapsed == 0
    return (
        preceding < target <= elapsed
        and elapsed - target <= last_update + 1e-9
        and elapsed - target <= fraction * target + 1e-9
    )


class DataCursor:
    """Keep every window, including a partial epoch at a batch boundary."""

    def __init__(self, tokens, seed, window=256, batch_size=8, cursor=0):
        self.tokens, self.seed, self.window, self.batch_size = (
            tokens,
            seed,
            window,
            batch_size,
        )
        self.count = (len(tokens) - 1) // window
        if self.count < 1 or cursor < 0:
            raise ValueError("Invalid training stream")
        self.cursor, self.orders = cursor, {}

    def order(self, epoch):
        if epoch not in self.orders:
            seed = int.from_bytes(
                hashlib.sha256(
                    f"equal-time-order:{self.seed}:{epoch}".encode()
                ).digest()[:8],
                "little",
            )
            self.orders[epoch] = np.random.default_rng(seed).permutation(self.count)
            self.orders = {k: v for k, v in self.orders.items() if k >= epoch - 1}
        return self.orders[epoch]

    def identity(self):
        epoch, offset = divmod(self.cursor, self.count)
        return dict(
            cursor=self.cursor,
            epoch=epoch,
            epoch_offset=offset,
            windows=self.count,
            order_sha256=hashlib.sha256(
                self.order(epoch).astype("<i8").tobytes()
            ).hexdigest(),
        )

    def batch(self, device):
        indices = [
            self.order(i // self.count)[i % self.count]
            for i in range(self.cursor, self.cursor + self.batch_size)
        ]
        offsets = (
            np.asarray(indices)[:, None] * self.window
            + np.arange(self.window + 1)[None, :]
        )
        source = torch.as_tensor(
            np.array(self.tokens[offsets], dtype=np.int64), device=device
        )
        return source[:, :-1], source[:, 1:]

    def advance(self):
        self.cursor += self.batch_size


def runtime(device):
    cuda = str(device).startswith("cuda")
    props = torch.cuda.get_device_properties(device) if cuda else None
    return dict(
        python=platform.python_version(),
        torch=torch.__version__,
        numpy=np.__version__,
        transformers=importlib.metadata.version("transformers"),
        device=str(device),
        cuda=torch.version.cuda,
        hip=torch.version.hip,
        gpu=props.name if props else None,
        gpu_memory_bytes=props.total_memory if props else None,
        gpu_multiprocessors=props.multi_processor_count if props else None,
        cuda_capability=(
            list(torch.cuda.get_device_capability(device)) if cuda else None
        ),
        cudnn=torch.backends.cudnn.version() if cuda else None,
    )


def rng_state(device):
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        cpu=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else [],
    )


def save(path, model, optimizer, clock, stream, step, identity, device, history):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    synchronize(device)
    if clock.active:
        raise RuntimeError("Checkpoint inside an unfinished update")
    payload = dict(
        schema="layerrope-time-state.v1",
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        rng=rng_state(device),
        step=step,
        training_seconds=clock.seconds,
        cursor=stream.identity(),
        identity=identity,
        runtime=runtime(device),
        history=history,
    )
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return dict(
        path=str(path),
        sha256=sha(path),
        step=step,
        training_seconds=clock.seconds,
        model_sha256=state_hash(payload["model"]),
        optimizer_sha256=state_hash(payload["optimizer"]),
        cursor=payload["cursor"],
        bytes=path.stat().st_size,
    )


def restore(path, model, optimizer, stream, identity, device, expected_sha256):
    if sha(path) != expected_sha256:
        raise ValueError("Checkpoint digest mismatch")
    state = torch.load(path, map_location=device, weights_only=False)
    if (
        state["schema"] != "layerrope-time-state.v1"
        or state["identity"] != identity
        or state["runtime"] != runtime(device)
    ):
        raise ValueError("Resume source/data/recipe/runtime identity mismatch")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    stream.cursor = state["cursor"]["cursor"]
    if stream.identity() != state["cursor"]:
        raise ValueError("Resume epoch/order/cursor mismatch")
    if state_hash(model.state_dict()) != state_hash(state["model"]) or state_hash(
        optimizer.state_dict()
    ) != state_hash(state["optimizer"]):
        raise ValueError("Resume model/optimizer mismatch")
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["cpu"].cpu())
    if state["rng"]["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["rng"]["cuda"]])
    return state

#!/usr/bin/env python3
"""Store a prefix cache in a file, and read it back the way a server would.

One file holds one model's keys and values for one prefix. A fixed header of
4096 bytes says whose cache it is: the model and revision, the geometry, the
number of tokens and the hash of the tokens it was computed from. The payload
follows, keys then values for each layer in turn, in the precision the model
runs in. Nothing is compressed, so the file's size is the cache's size.

A cache is read either through the page cache or around it. Reading around
it, with direct I/O, is what makes a timing a storage timing: a file this
process has just written is otherwise served from memory, and its read
measures nothing about the device. Every read reports how many bytes the
kernel says it fetched from storage, so a read that was meant to be cold and
was not can be seen.

A reader refuses a file that is not the cache it was asked for.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess

import torch

MAGIC = b"KVPREFIX1\n"
HEADER = 4096
BLOCK = 4096
CHUNK = 64 * 2**20
SERIALIZER = "kv-prefix-raw-1"


class WrongCache(ValueError):
    """The file is not the cache that was asked for."""


def payload_bytes(layers, kv_heads, tokens, head_dim, itemsize=2):
    return layers * 2 * kv_heads * tokens * head_dim * itemsize


def storage_counters():
    """Bytes this process has asked for, and bytes the kernel fetched for it."""
    out = {}
    try:
        with open("/proc/self/io") as f:
            for line in f:
                k, v = line.split(":")
                out[k.strip()] = int(v)
    except OSError:
        pass
    return {"requested": out.get("rchar"), "from_storage": out.get("read_bytes")}


def storage_facts(path):
    """What can be learned about the place the caches are stored."""
    out = {"path": path}
    st = os.statvfs(path)
    out["free_bytes"] = st.f_bavail * st.f_frsize
    out["total_bytes"] = st.f_blocks * st.f_frsize
    for name, cmd in (
        ("mount", ["findmnt", "-n", "-o", "SOURCE,FSTYPE,OPTIONS", "-T", path]),
        ("block_devices", ["lsblk", "-d", "-n", "-o", "NAME,ROTA,SIZE,TYPE,TRAN"]),
    ):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            out[name] = r.stdout.strip()[:2000]
        except (OSError, subprocess.SubprocessError) as e:
            out[name] = f"unavailable: {e}"
    return out


def write(path, pairs, meta):
    """Write per-layer (keys, values), each [1, heads, tokens, width].

    Returns the receipt a byte ledger needs: logical size, allocated size and
    the hash of the payload.
    """
    k0 = pairs[0][0]
    _, heads, tokens, width = k0.shape
    head = {
        "serializer": SERIALIZER,
        "layers": len(pairs),
        "kv_heads": int(heads),
        "tokens": int(tokens),
        "head_dim": int(width),
        "dtype": str(k0.dtype).replace("torch.", ""),
        **meta,
    }
    size = payload_bytes(len(pairs), heads, tokens, width, k0.element_size())
    head["payload_bytes"] = size
    h = hashlib.sha256()
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(b"\0" * HEADER)
        for k, v in pairs:
            for t in (k, v):
                if t.shape != k0.shape or t.dtype != k0.dtype:
                    raise ValueError("every layer must share one shape and precision")
                # numpy has no bf16; the bytes are what is stored either way
                raw = t[0].contiguous().cpu().view(torch.uint8).numpy().tobytes()
                h.update(raw)
                f.write(raw)
        head["payload_sha256"] = h.hexdigest()
        blob = MAGIC + json.dumps(head, sort_keys=True).encode()
        if len(blob) > HEADER:
            raise ValueError("the header does not fit its block")
        f.seek(0)
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    st = os.stat(path)
    return {
        "path": path,
        "logical_bytes": st.st_size,
        "allocated_bytes": st.st_blocks * 512,
        "payload_bytes": size,
        "payload_sha256": head["payload_sha256"],
        "serializer": SERIALIZER,
        **meta,
    }


def header(path):
    with open(path, "rb") as f:
        blob = f.read(HEADER)
    if not blob.startswith(MAGIC):
        raise WrongCache(f"{path} is not a prefix cache file")
    return json.loads(blob[len(MAGIC) :].rstrip(b"\0"))


def check(head, expect):
    for k, want in expect.items():
        if head.get(k) != want:
            raise WrongCache(f"{k} is {head.get(k)!r}, and {want!r} was asked for")


class Reader:
    """Reads caches into one pinned buffer that is kept between requests.

    Pinning memory is slow and a server does it once, so the buffer is part
    of the warm state and not of any request.
    """

    def __init__(self, largest_payload, device="cuda"):
        self.device = torch.device(device)
        pin = self.device.type == "cuda"
        size = -(-largest_payload // BLOCK) * BLOCK
        self.buf = torch.empty(size, dtype=torch.uint8, pin_memory=pin)
        self.view = memoryview(self.buf.numpy())

    def read(self, path, *, direct, expect=None):
        """Payload bytes into the buffer. Returns the header and the I/O record."""
        head = header(path)
        if expect:
            check(head, expect)
        size = head["payload_bytes"]
        if size > len(self.view):
            raise ValueError("the cache is larger than the buffer kept for it")
        before = storage_counters()
        flags = os.O_RDONLY | (os.O_DIRECT if direct else 0)
        fd = os.open(path, flags)
        try:
            got, off = 0, HEADER
            while got < size:
                want = min(
                    CHUNK, -(-(size - got) // BLOCK) * BLOCK if direct else size - got
                )
                n = os.preadv(fd, [self.view[got : got + want]], off + got)
                if n <= 0:
                    raise OSError(f"{path} ended at {got} of {size} bytes")
                got += n
        finally:
            os.close(fd)
        after = storage_counters()
        io = {
            "direct": bool(direct),
            "bytes": size,
            "requested": (
                None
                if before["requested"] is None
                else after["requested"] - before["requested"]
            ),
            "from_storage": (
                None
                if before["from_storage"] is None
                else after["from_storage"] - before["from_storage"]
            ),
        }
        return head, io

    def tensor(self, head):
        """The buffer's payload on the device, as [layers, 2, heads, tokens, width]."""
        size = head["payload_bytes"]
        dtype = getattr(torch, head["dtype"])
        flat = self.buf[:size].to(self.device, non_blocking=False)
        return flat.view(dtype).view(
            head["layers"], 2, head["kv_heads"], head["tokens"], head["head_dim"]
        )

    def verify(self, head):
        return (
            hashlib.sha256(self.view[: head["payload_bytes"]]).hexdigest()
            == head["payload_sha256"]
        )


def as_pairs(stacked):
    """[layers, 2, heads, tokens, width] as the per-layer pairs a model takes."""
    return [
        (stacked[i, 0].unsqueeze(0), stacked[i, 1].unsqueeze(0))
        for i in range(len(stacked))
    ]

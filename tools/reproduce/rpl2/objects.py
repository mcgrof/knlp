# SPDX-License-Identifier: MIT
"""Immutable test objects on a user-supplied directory, and reads of them.

Objects are written once from a deterministic seed, synced, made read-only
and listed with their SHA-256 digests in a manifest. A later run reuses an
object set only when its manifest matches the requested seed, size and
count; anything else is refused rather than overwritten.
"""

import hashlib
import json
import mmap
import os
import random
from pathlib import Path

MANIFEST = "manifest.json"
MANIFEST_SCHEMA = "knlp.rpl2.objects.v1"
ALIGN = 4096


class ObjectError(RuntimeError):
    """The object set is missing, changed or unusable."""


def payload(seed, index, size):
    return random.Random(f"{seed}:{index}").randbytes(size)


def set_name(seed, size, count):
    return f"rpl2-objects-s{seed}-b{size}-n{count}"


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _expected(seed, size, count):
    return {
        "schema": MANIFEST_SCHEMA,
        "seed": seed,
        "object_bytes": size,
        "count": count,
    }


def prepare(target, seed, size, count):
    """Create or reuse the object set; return its directory and manifest."""
    directory = Path(target) / set_name(seed, size, count)
    manifest_path = directory / MANIFEST
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, value in _expected(seed, size, count).items():
            if manifest.get(name) != value:
                raise ObjectError(
                    f"{manifest_path}: {name} is {manifest.get(name)!r}, "
                    f"expected {value!r}; refusing to reuse or overwrite it"
                )
        for item in manifest["objects"]:
            path = directory / item["file"]
            if not path.is_file() or path.stat().st_size != size:
                raise ObjectError(f"{path}: missing or wrong size")
        return directory, manifest
    if directory.exists() and any(directory.iterdir()):
        raise ObjectError(f"{directory}: exists without a manifest; refusing")
    directory.mkdir(parents=True, exist_ok=True)
    objects = []
    for index in range(count):
        key = f"obj-{index:06d}"
        data = payload(seed, index, size)
        path = directory / f"{key}.bin"
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        path.chmod(0o444)
        objects.append(
            {
                "key": key,
                "generation": 1,
                "file": path.name,
                "bytes": size,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    manifest = _expected(seed, size, count) | {"objects": objects}
    temporary = directory / (MANIFEST + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.rename(manifest_path)
    _fsync_dir(directory)
    return directory, manifest


def mount_of(path):
    """Filesystem type, and NFS version when present, for the mount of path.

    Only the type and protocol version are returned. Mount sources and
    server addresses identify a host and are deliberately left out.
    """
    target = os.path.realpath(path)
    best = None
    with open("/proc/self/mountinfo") as stream:
        for line in stream:
            left, _, right = line.partition(" - ")
            fields = left.split()
            point = fields[4].replace("\\040", " ")
            if target == point or target.startswith(point.rstrip("/") + "/"):
                if best is None or len(point) > len(best[0]):
                    rest = right.split()
                    best = (point, rest[0], rest[2] if len(rest) > 2 else "")
    if best is None:
        return {"type": "unknown"}
    info = {"type": best[1]}
    for option in best[2].split(","):
        if option.startswith("vers="):
            info["nfs_version"] = option.removeprefix("vers=")
    return info


def free_bytes(path):
    stats = os.statvfs(path)
    return stats.f_bavail * stats.f_frsize


def probe_direct(directory):
    """Return (supported, reason) for an aligned O_DIRECT read in directory."""
    flag = getattr(os, "O_DIRECT", None)
    if flag is None:
        return False, "this platform has no O_DIRECT"
    path = Path(directory) / ".rpl2-direct-probe"
    try:
        path.write_bytes(b"\0" * ALIGN)
        fd = os.open(path, os.O_RDONLY | flag)
    except OSError as exc:
        path.unlink(missing_ok=True)
        return False, f"open with O_DIRECT failed: {os.strerror(exc.errno)}"
    buffer = mmap.mmap(-1, ALIGN)
    try:
        count = os.preadv(fd, [buffer], 0)
    except OSError as exc:
        return False, f"aligned O_DIRECT read failed: {os.strerror(exc.errno)}"
    finally:
        os.close(fd)
        buffer.close()
        path.unlink(missing_ok=True)
    if count != ALIGN:
        return False, f"aligned O_DIRECT read returned {count} bytes"
    return True, "aligned O_DIRECT read succeeded"


def read_object(path, size, mode, buffer):
    """Open, read and close one object; return the byte count read.

    buffer is a page-aligned mmap of at least size bytes, reused across
    requests so that allocation stays outside the timed region.
    """
    flags = os.O_RDONLY
    if mode == "direct":
        flags |= os.O_DIRECT
    fd = os.open(path, flags)
    try:
        view = memoryview(buffer)
        done = 0
        while done < size:
            count = os.preadv(fd, [view[done:size]], done)
            if count == 0:
                break
            done += count
        view.release()
        return done
    finally:
        os.close(fd)


def check_size(size, mode):
    if size <= 0:
        raise ObjectError("object size must be positive")
    if mode == "direct" and size % ALIGN:
        raise ObjectError(f"direct I/O needs a size that is a multiple of {ALIGN}")

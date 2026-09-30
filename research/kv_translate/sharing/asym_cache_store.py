#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""Shape-aware native K16/V8 prefix-cache storage.

The on-disk payload is produced by LMCache's existing asymmetric codec:
keys remain BF16 and values use FP8 e4m3fn with one FP32 scale per layer.
Each independently checksummed layer blob is block padded so direct reads
remain possible.  Values are transferred as FP8 and expanded on the target
device before the model's ordinary attention path sees the restored cache.
"""

from __future__ import annotations

import json
import importlib.util
import mmap
import os
import struct
import subprocess
import sys
import time
import types
import zlib

import torch

from .cache_store import BLOCK, HEADER, WrongCache, storage_counters

MAGIC = b"KVASYM1\n"
OUTER_HEADER = 8 * BLOCK
SERIALIZER = "kv-prefix-asym-k16-v8-1"
FP8_FORMAT = "float8_e4m3fn"
SCALE_DTYPE = "float32"
SCALE_SCOPE = "per_layer_tensor"


def _align(value, alignment=BLOCK):
    return -(-int(value) // alignment) * alignment


def _codec_api(checkout):
    """Load the externally pinned LMCache codec without vendoring it."""
    root = os.path.abspath(checkout)
    package = os.path.join(root, "lmcache")
    codec_dir = os.path.join(package, "v1", "kv_codec")
    if not os.path.isfile(os.path.join(codec_dir, "asym_k16_v8.py")):
        raise FileNotFoundError(f"LMCache codec source is absent under {root}")
    loaded = sys.modules.get("lmcache.v1.kv_codec.asym_k16_v8")
    if loaded is not None and getattr(loaded, "_knlp_codec_root", None) == root:
        encoded = sys.modules["lmcache.v1.kv_codec.encoded_kv"]
        return loaded.AsymK16V8Codec, encoded.CodecHashes, encoded.ScaleScope

    # The codec files are self-contained, but importing LMCache's top-level
    # package also initializes unrelated runtime backends.  Load only the
    # pinned codec modules and provide the dtype table that their header code
    # normally obtains from lmcache.v1.protocol.
    for name in tuple(sys.modules):
        if name == "lmcache" or name.startswith("lmcache."):
            del sys.modules[name]
    for name, path in (
        ("lmcache", package),
        ("lmcache.v1", os.path.join(package, "v1")),
        ("lmcache.v1.kv_codec", codec_dir),
    ):
        module = types.ModuleType(name)
        module.__path__ = [path]
        sys.modules[name] = module
    protocol = types.ModuleType("lmcache.v1.protocol")
    protocol.DTYPE_TO_INT = {
        None: 0,
        torch.half: 1,
        torch.float16: 2,
        torch.bfloat16: 3,
        torch.float: 4,
        torch.float32: 4,
        torch.float64: 5,
        torch.double: 5,
        torch.uint8: 6,
        torch.float8_e4m3fn: 7,
        torch.float8_e5m2: 8,
    }
    protocol.INT_TO_DTYPE = {
        0: None,
        1: torch.half,
        2: torch.float16,
        3: torch.bfloat16,
        4: torch.float,
        5: torch.float64,
        6: torch.uint8,
        7: torch.float8_e4m3fn,
        8: torch.float8_e5m2,
    }
    sys.modules[protocol.__name__] = protocol

    def load(name, filename):
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(codec_dir, filename)
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load pinned codec module {name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    load("lmcache.v1.kv_codec.errors", "errors.py")
    encoded = load("lmcache.v1.kv_codec.encoded_kv", "encoded_kv.py")
    implementation = load("lmcache.v1.kv_codec.asym_k16_v8", "asym_k16_v8.py")
    implementation._knlp_codec_root = root
    return implementation.AsymK16V8Codec, encoded.CodecHashes, encoded.ScaleScope


def _codec_modules(checkout):
    """Return the pinned codec API together with its two source modules."""
    api = _codec_api(checkout)
    return (
        api,
        sys.modules["lmcache.v1.kv_codec.asym_k16_v8"],
        sys.modules["lmcache.v1.kv_codec.encoded_kv"],
    )


def _hashes(api, meta):
    _, CodecHashes, _ = api
    return CodecHashes(
        model_id=meta["model"],
        model_revision_hash=meta["revision"],
        tokenizer_hash=meta.get("tokenizer_hash", ""),
        rope_config_hash=meta.get("rope_config_hash", ""),
        attention_backend=meta.get("attention_backend", "sdpa"),
        kv_layout="hf_dynamic_cache_1htd",
    )


def write(path, pairs, meta, *, codec_checkout, codec_revision):
    """Encode per-layer ``[1, heads, tokens, width]`` BF16 K/V pairs."""
    path = os.fspath(path)
    if not pairs:
        raise ValueError("at least one K/V layer is required")
    k0 = pairs[0][0]
    if k0.ndim != 4 or k0.shape[0] != 1:
        raise ValueError("K/V tensors must have shape [1, heads, tokens, width]")
    if k0.dtype != torch.bfloat16:
        raise ValueError(f"K must remain BF16, got {k0.dtype}")
    _, heads, tokens, width = k0.shape
    api = _codec_api(codec_checkout)
    AsymK16V8Codec, _, ScaleScope = api
    codec = AsymK16V8Codec(
        fp8_dtype=torch.float8_e4m3fn,
        scale_dtype=torch.float32,
        scale_scope=ScaleScope.PER_TENSOR,
    )
    hashes = _hashes(api, meta)
    segments = []
    diagnostics = {"v_nonfinite": 0, "layers": []}
    tmp = path + ".tmp"
    started = time.perf_counter()
    with open(tmp, "wb") as f:
        f.write(b"\0" * OUTER_HEADER)
        for layer, (k, v) in enumerate(pairs):
            if k.shape != k0.shape or v.shape != k0.shape:
                raise ValueError("every layer must share the declared K/V shape")
            if k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
                raise ValueError("K and V must both enter the codec as BF16")
            nonfinite = int((~torch.isfinite(v)).sum().item())
            diagnostics["v_nonfinite"] += nonfinite
            if nonfinite:
                raise FloatingPointError(f"layer {layer} V contains nonfinite values")
            amax = float(v.float().abs().amax().item())
            encoded = codec.encode(
                k,
                v,
                hashes=hashes,
                layer_id=layer,
                chunk_id=0,
                chunk_size=int(tokens),
                page_size=-1,
                kv_head_count=int(heads),
                head_dim=int(width),
            )
            blob = codec.to_bytes(encoded)
            offset = f.tell()
            padded = _align(len(blob))
            if offset % BLOCK:
                raise RuntimeError("asymmetric segment offset is not block aligned")
            f.write(blob)
            f.write(b"\0" * (padded - len(blob)))
            segments.append(
                {
                    "layer": layer,
                    "offset": offset,
                    "length": len(blob),
                    "padded_length": padded,
                    "k_payload_bytes": encoded.k_payload_len,
                    "v_payload_bytes": encoded.v_payload_len,
                    "scale_payload_bytes": encoded.scale_payload_len,
                }
            )
            diagnostics["layers"].append(
                {"layer": layer, "v_amax": amax, "nonfinite": nonfinite}
            )
        head = {
            "serializer": SERIALIZER,
            "layers": len(pairs),
            "kv_heads": int(heads),
            "tokens": int(tokens),
            "head_dim": int(width),
            "k_dtype": "bfloat16",
            "v_storage_dtype": FP8_FORMAT,
            "v_runtime_dtype": "bfloat16",
            "scale_dtype": SCALE_DTYPE,
            "scale_scope": SCALE_SCOPE,
            "chunk_tokens": int(tokens),
            "head_axis": None,
            "page_axis": None,
            "outer_header_bytes": OUTER_HEADER,
            "padding_bytes": BLOCK,
            "rounding": "torch float8_e4m3fn cast after symmetric clamp",
            "clipping": "[-torch.finfo(float8_e4m3fn).max, +max]",
            "expansion": "target device before ordinary attention",
            "codec_revision": codec_revision,
            "segments": segments,
            "bf16_payload_bytes": sum(
                2 * k.numel() * k.element_size() for k, _ in pairs
            ),
            "asym_payload_bytes": sum(
                s["k_payload_bytes"] + s["v_payload_bytes"] + s["scale_payload_bytes"]
                for s in segments
            ),
            "diagnostics": diagnostics,
            **meta,
        }
        blob = MAGIC + json.dumps(head, sort_keys=True).encode()
        if len(blob) > OUTER_HEADER:
            raise ValueError("the asymmetric cache header does not fit its block")
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
        "bf16_payload_bytes": head["bf16_payload_bytes"],
        "asym_payload_bytes": head["asym_payload_bytes"],
        "serialized_reduction": 1 - st.st_size / (HEADER + head["bf16_payload_bytes"]),
        "seconds_encode_write": round(time.perf_counter() - started, 6),
        "serializer": SERIALIZER,
        "codec_revision": codec_revision,
        "outer_header_bytes": OUTER_HEADER,
        "diagnostics": diagnostics,
        **meta,
    }


def header(path):
    path = os.fspath(path)
    with open(path, "rb") as f:
        blob = f.read(OUTER_HEADER)
    if not blob.startswith(MAGIC):
        raise WrongCache(f"{path} is not an asymmetric prefix cache")
    return json.loads(blob[len(MAGIC) :].rstrip(b"\0"))


def check(head, expect):
    for key, wanted in expect.items():
        if head.get(key) != wanted:
            raise WrongCache(f"{key} is {head.get(key)!r}, wanted {wanted!r}")


class Reader:
    """Restore codec blobs without materializing pageable payload copies.

    ``payload_crc`` preserves the codec's original per-read verification.
    ``prevalidated`` verifies an immutable file once, then retains format,
    shape, and identity checks while omitting the repeated payload CRC.  That
    is the measured BF16 reader's integrity policy made explicit.
    """

    INTEGRITY_POLICIES = ("payload_crc", "prevalidated")

    def __init__(
        self,
        largest_segment,
        *,
        device,
        codec_checkout,
        integrity_policy="payload_crc",
        staging_buffers=2,
    ):
        self.device = torch.device(device)
        if integrity_policy not in self.INTEGRITY_POLICIES:
            raise ValueError(
                f"integrity_policy must be one of {self.INTEGRITY_POLICIES}"
            )
        if int(staging_buffers) < 1:
            raise ValueError("at least one staging buffer is required")
        self.integrity_policy = integrity_policy
        self.api, self.implementation, self.encoded_module = _codec_modules(
            codec_checkout
        )
        AsymK16V8Codec, _, ScaleScope = self.api
        self.codec = AsymK16V8Codec(
            fp8_dtype=torch.float8_e4m3fn,
            scale_dtype=torch.float32,
            scale_scope=ScaleScope.PER_TENSOR,
        )
        size = _align(largest_segment)
        self.buffers = []
        self._maps = []
        for _ in range(int(staging_buffers)):
            if self.device.type == "cuda":
                buf = torch.empty(size, dtype=torch.uint8, pin_memory=True)
                view = memoryview(buf.numpy())
            else:
                mapping = mmap.mmap(-1, size)
                self._maps.append(mapping)
                buf = torch.frombuffer(mapping, dtype=torch.uint8)
                view = memoryview(mapping)
            if buf.data_ptr() % BLOCK:
                raise OSError("the asymmetric staging buffer is not block aligned")
            self.buffers.append((buf, view))
        self._events = [None] * len(self.buffers)
        self._validated = {}

    @staticmethod
    def _file_identity(path):
        status = os.stat(path)
        return (
            status.st_dev,
            status.st_ino,
            status.st_size,
            status.st_mtime_ns,
            status.st_ctime_ns,
        )

    @staticmethod
    def _unpack_string(view, offset):
        if offset + 2 > len(view):
            raise WrongCache("truncated codec string length")
        (length,) = struct.unpack_from("<H", view, offset)
        offset += 2
        if offset + length > len(view):
            raise WrongCache("truncated codec string")
        try:
            value = bytes(view[offset : offset + length]).decode("utf-8")
        except UnicodeDecodeError as error:
            raise WrongCache("non-UTF-8 codec string") from error
        return value, offset + length

    def _parse(self, source, *, expected_hashes, verify_payload):
        """Parse the pinned V1 format while retaining a payload view."""
        view = memoryview(source)
        module = self.encoded_module
        fixed_format = "<8sHHHHHH" + "qqqqqqq"
        fixed_length = struct.calcsize(fixed_format)
        if len(view) < fixed_length:
            raise WrongCache("codec segment is shorter than its fixed header")
        (
            magic,
            version,
            scale_scope,
            k_dtype_id,
            v_dtype_id,
            scale_dtype_id,
            _reserved,
            layer_id,
            chunk_id,
            chunk_size,
            page_size,
            kv_head_count,
            head_dim,
            scale_shape_n,
        ) = struct.unpack_from(fixed_format, view, 0)
        if magic != module.CODEC_MAGIC:
            raise WrongCache(f"bad codec magic {magic!r}")
        if version != int(module.CodecVersion.V1):
            raise WrongCache(f"unsupported codec version {version}")
        if scale_shape_n < 0 or scale_shape_n > 8:
            raise WrongCache(f"implausible scale rank {scale_shape_n}")
        offset = fixed_length
        shape_bytes = 8 * scale_shape_n
        if offset + shape_bytes > len(view):
            raise WrongCache("truncated codec scale shape")
        scale_shape = tuple(struct.unpack_from(f"<{scale_shape_n}q", view, offset))
        offset += shape_bytes
        if any(dimension < 0 for dimension in scale_shape):
            raise WrongCache("negative codec scale dimension")
        if offset + 24 > len(view):
            raise WrongCache("truncated codec payload lengths")
        k_length, v_length, scale_length = struct.unpack_from("<qqq", view, offset)
        offset += 24
        if min(k_length, v_length, scale_length) < 0:
            raise WrongCache("negative codec payload length")
        if offset + 2 > len(view):
            raise WrongCache("truncated codec hash count")
        (hash_count,) = struct.unpack_from("<H", view, offset)
        offset += 2
        hashes = module.CodecHashes()
        known_hashes = set(module.CodecHashes._CHECK_ORDER)
        for _ in range(hash_count):
            key, offset = self._unpack_string(view, offset)
            value, offset = self._unpack_string(view, offset)
            if key in known_hashes:
                setattr(hashes, key, value)
        if offset + 4 > len(view):
            raise WrongCache("truncated codec checksum")
        (declared_crc,) = struct.unpack_from("<I", view, offset)
        offset += 4
        payload_length = k_length + v_length + scale_length
        if offset + payload_length != len(view):
            raise WrongCache("codec segment length does not match its declared payload")
        payload = view[offset:]
        if verify_payload:
            computed_crc = zlib.crc32(payload) & 0xFFFFFFFF
            if computed_crc != declared_crc:
                raise WrongCache(
                    f"payload CRC mismatch: declared {declared_crc:#x}, "
                    f"computed {computed_crc:#x}"
                )
        encoded = module.EncodedKV(
            k_dtype=module._int_to_dtype(k_dtype_id),
            v_dtype=module._int_to_dtype(v_dtype_id),
            scale_dtype=module._int_to_dtype(scale_dtype_id),
            scale_scope=module.ScaleScope(scale_scope),
            hashes=hashes,
            layer_id=layer_id,
            chunk_id=chunk_id,
            chunk_size=chunk_size,
            page_size=page_size,
            kv_head_count=kv_head_count,
            head_dim=head_dim,
            scale_shape=scale_shape,
            k_payload_len=k_length,
            v_payload_len=v_length,
            scale_payload_len=scale_length,
            payload=payload,
            header_bytes=bytes(view[:offset]),
        )
        self.codec._check_hash_match(encoded.hashes, expected_hashes)
        return encoded, offset

    def _check_encoded(self, encoded, head, layer):
        elements = head["kv_heads"] * head["tokens"] * head["head_dim"]
        expected = {
            "layer_id": layer,
            "chunk_id": 0,
            "chunk_size": head["tokens"],
            "page_size": -1,
            "kv_head_count": head["kv_heads"],
            "head_dim": head["head_dim"],
            "k_dtype": torch.bfloat16,
            "v_dtype": torch.float8_e4m3fn,
            "scale_dtype": torch.float32,
            "scale_scope": self.api[2].PER_TENSOR,
            "scale_shape": (),
            "k_payload_len": elements * 2,
            "v_payload_len": elements,
            "scale_payload_len": 4,
        }
        for field, wanted in expected.items():
            got = getattr(encoded, field)
            if got != wanted:
                raise WrongCache(f"codec {field} is {got!r}, expected {wanted!r}")

    def _wait_for_buffer(self, index):
        event = self._events[index]
        if event is not None:
            event.synchronize()
            self._events[index] = None

    def _read_segment(self, fd, segment, index):
        self._wait_for_buffer(index)
        _, view = self.buffers[index]
        padded = segment["padded_length"]
        if padded > len(view):
            raise ValueError("codec segment exceeds the warm staging buffer")
        got = os.preadv(fd, [view[:padded]], segment["offset"])
        if got != padded:
            raise OSError(f"short codec read: {got} of {padded} bytes")
        return view[: segment["length"]]

    def _decode_view(self, encoded, payload_offset, index):
        _, view = self.buffers[index]
        k_offset = payload_offset
        v_offset = k_offset + encoded.k_payload_len
        scale_offset = v_offset + encoded.v_payload_len
        k_host = torch.frombuffer(
            view,
            dtype=encoded.k_dtype,
            count=encoded.k_payload_len // 2,
            offset=k_offset,
        )
        v_host = torch.frombuffer(
            view,
            dtype=encoded.v_dtype,
            count=encoded.v_payload_len,
            offset=v_offset,
        )
        scales_host = torch.frombuffer(
            view,
            dtype=encoded.scale_dtype,
            count=encoded.scale_payload_len // 4,
            offset=scale_offset,
        ).reshape(encoded.scale_shape)
        if self.device.type == "cuda":
            k = k_host.to(self.device, non_blocking=True)
            v_quant = v_host.to(self.device, non_blocking=True)
            scales = scales_host.to(self.device, non_blocking=True)
            v = self.implementation.dequantize_v_fp8(
                v_quant,
                scales,
                encoded.scale_scope,
                out_dtype=torch.bfloat16,
            )
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            self._events[index] = event
        else:
            # CPU results must own their bytes before this buffer is reused.
            # Dequantization already gives V independent storage.
            k = k_host.clone()
            v = self.implementation.dequantize_v_fp8(
                v_host,
                scales_host,
                encoded.scale_scope,
                out_dtype=torch.bfloat16,
            )
        return k, v

    def validate(self, path, *, expect=None):
        """Verify every payload CRC and attest the file for fast reads."""
        path = os.path.abspath(os.fspath(path))
        head = header(path)
        if expect:
            check(head, expect)
        if head["serializer"] != SERIALIZER:
            raise WrongCache(f"unsupported serializer {head['serializer']!r}")
        hashes = _hashes(self.api, head)
        fd = os.open(path, os.O_RDONLY)
        try:
            for layer, segment in enumerate(head["segments"]):
                index = layer % len(self.buffers)
                source = self._read_segment(fd, segment, index)
                encoded, _ = self._parse(
                    source,
                    expected_hashes=hashes,
                    verify_payload=True,
                )
                self._check_encoded(encoded, head, layer)
        finally:
            os.close(fd)
        identity = self._file_identity(path)
        self._validated[path] = identity
        return head, {"payload_crc": "verified", "file_identity": identity}

    def _require_prevalidated(self, path):
        expected = self._validated.get(path)
        if expected is None:
            raise WrongCache(f"{path} has not been payload-validated")
        if self._file_identity(path) != expected:
            raise WrongCache(f"{path} changed after payload validation")
        return expected

    def read(self, path, *, direct, expect=None):
        path = os.path.abspath(os.fspath(path))
        head = header(path)
        if expect:
            check(head, expect)
        if head["serializer"] != SERIALIZER:
            raise WrongCache(f"unsupported serializer {head['serializer']!r}")
        attested = None
        if self.integrity_policy == "prevalidated":
            attested = self._require_prevalidated(path)
        hashes = _hashes(self.api, head)
        before = storage_counters()
        flags = os.O_RDONLY | (os.O_DIRECT if direct else 0)
        fd = os.open(path, flags)
        pairs = []
        h2d_bytes = 0
        try:
            for layer, segment in enumerate(head["segments"]):
                index = layer % len(self.buffers)
                source = self._read_segment(fd, segment, index)
                encoded, payload_offset = self._parse(
                    source,
                    expected_hashes=hashes,
                    verify_payload=self.integrity_policy == "payload_crc",
                )
                self._check_encoded(encoded, head, layer)
                k, v = self._decode_view(encoded, payload_offset, index)
                shape = (
                    1,
                    head["kv_heads"],
                    head["tokens"],
                    head["head_dim"],
                )
                pairs.append((k.reshape(shape), v.reshape(shape)))
                h2d_bytes += (
                    encoded.k_payload_len
                    + encoded.v_payload_len
                    + encoded.scale_payload_len
                )
        finally:
            os.close(fd)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self._events = [None] * len(self.buffers)
        if attested is not None and self._file_identity(path) != attested:
            raise WrongCache(f"{path} changed during restore")
        after = storage_counters()
        io = {
            "direct": bool(direct),
            "bytes": sum(s["padded_length"] for s in head["segments"]),
            "serialized_bytes": os.stat(path).st_size,
            "codec_payload_bytes": head["asym_payload_bytes"],
            "h2d_bytes": h2d_bytes,
            "expansion_device": self.device.type,
            "integrity_policy": self.integrity_policy,
            "payload_crc": (
                "per_read"
                if self.integrity_policy == "payload_crc"
                else "prevalidated_immutable"
            ),
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
        return head, pairs, io


def codec_source_revision(checkout):
    """Return a Git revision when the configured codec tree is a checkout."""
    result = subprocess.run(
        ["git", "-C", checkout, "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip() if result.returncode == 0 else None

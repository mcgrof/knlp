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
import subprocess
import sys
import time
import types

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
    implementation = load(
        "lmcache.v1.kv_codec.asym_k16_v8", "asym_k16_v8.py"
    )
    implementation._knlp_codec_root = root
    return implementation.AsymK16V8Codec, encoded.CodecHashes, encoded.ScaleScope


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
                s["k_payload_bytes"]
                + s["v_payload_bytes"]
                + s["scale_payload_bytes"]
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
    """Read block-aligned codec blobs and expand V on the target device."""

    def __init__(self, largest_segment, *, device, codec_checkout):
        self.device = torch.device(device)
        self.api = _codec_api(codec_checkout)
        AsymK16V8Codec, _, ScaleScope = self.api
        self.codec = AsymK16V8Codec(
            fp8_dtype=torch.float8_e4m3fn,
            scale_dtype=torch.float32,
            scale_scope=ScaleScope.PER_TENSOR,
        )
        size = _align(largest_segment)
        if self.device.type == "cuda":
            self.buf = torch.empty(size, dtype=torch.uint8, pin_memory=True)
            self.view = memoryview(self.buf.numpy())
        else:
            self.map = mmap.mmap(-1, size)
            self.buf = torch.frombuffer(self.map, dtype=torch.uint8)
            self.view = memoryview(self.map)
        if self.buf.data_ptr() % BLOCK:
            raise OSError("the asymmetric staging buffer is not block aligned")

    def read(self, path, *, direct, expect=None):
        head = header(path)
        if expect:
            check(head, expect)
        if head["serializer"] != SERIALIZER:
            raise WrongCache(f"unsupported serializer {head['serializer']!r}")
        hashes = _hashes(self.api, head)
        before = storage_counters()
        flags = os.O_RDONLY | (os.O_DIRECT if direct else 0)
        fd = os.open(path, flags)
        pairs = []
        h2d_bytes = 0
        try:
            for layer, segment in enumerate(head["segments"]):
                padded = segment["padded_length"]
                if padded > len(self.view):
                    raise ValueError("codec segment exceeds the warm staging buffer")
                got = os.preadv(fd, [self.view[:padded]], segment["offset"])
                if got != padded:
                    raise OSError(f"short codec read: {got} of {padded} bytes")
                encoded = self.codec.from_bytes(
                    bytes(self.view[: segment["length"]]),
                    expected_hashes=hashes,
                )
                if encoded.layer_id != layer:
                    raise WrongCache(
                        f"codec layer is {encoded.layer_id}, expected {layer}"
                    )
                if (
                    encoded.chunk_size != head["tokens"]
                    or encoded.kv_head_count != head["kv_heads"]
                    or encoded.head_dim != head["head_dim"]
                ):
                    raise WrongCache("codec shape metadata does not match outer header")
                k, v, scales = self.codec.decode(
                    encoded,
                    out_k_dtype=torch.bfloat16,
                    out_v_dtype=torch.bfloat16,
                    device=self.device,
                )
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
                del scales
        finally:
            os.close(fd)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        after = storage_counters()
        io = {
            "direct": bool(direct),
            "bytes": sum(s["padded_length"] for s in head["segments"]),
            "serialized_bytes": os.stat(path).st_size,
            "codec_payload_bytes": head["asym_payload_bytes"],
            "h2d_bytes": h2d_bytes,
            "expansion_device": self.device.type,
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

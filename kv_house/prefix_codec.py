# SPDX-License-Identifier: GPL-2.0
"""Sealed-block KV codec with prefix-cache semantics.

The contract, which the tests enforce and Prefix Integrity Analysis
polices end to end:

- a context of T tokens seals floor(T / B) blocks; the partial tail
  stays raw and is never encoded;
- block j depends only on tokens [j*B, (j+1)*B) and fixed codec
  configuration — never on a suffix, a future decode token, or any
  query;
- encoding is deterministic: the same block content and configuration
  produce byte-identical artifacts (same digest), so a prefix shared
  by different requests maps to identical cached blocks;
- blocks decode independently (random access, no delta chain);
- all codec configuration that affects bytes is part of the cache
  key.

Artifacts here store quantized values unpacked (one int8 per value)
for simplicity; the rate-distortion byte accounting lives in
``quant.payload_bytes`` and is what experiments report. What this
module guarantees is semantics: determinism, immutability, and
random access.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import torch

from kv_house.transforms import make_transform

CODEC_VERSION = 1


@dataclass(frozen=True)
class CodecConfig:
    block_size: int
    transform: str
    quant_alloc: tuple
    transform_version: int = 1
    calibration_id: str = ""
    codec_version: int = CODEC_VERSION

    def cache_key(self):
        blob = json.dumps(
            {
                "codec_version": self.codec_version,
                "block_size": self.block_size,
                "transform": self.transform,
                "transform_version": self.transform_version,
                "quant_alloc": list(self.quant_alloc),
                "calibration_id": self.calibration_id,
            },
            sort_keys=True,
        ).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class BlockArtifact:
    config_key: str
    block_index: int
    payload: bytes
    digest: str = field(init=False)

    def __post_init__(self):
        h = hashlib.sha256()
        h.update(self.config_key.encode())
        h.update(self.block_index.to_bytes(8, "little"))
        h.update(self.payload)
        self.digest = h.hexdigest()


def _quantize_row(row, bits):
    """Return (stored_bytes, scale) for one fp32 row; deterministic."""
    if bits <= 0:
        return b"", 0.0
    if bits >= 16:
        return row.astype("<f2").tobytes(), 0.0
    qmax = float(2 ** (bits - 1) - 1)
    scale = np.float16(max(float(np.abs(row).max()), 1e-8) / qmax)
    q = np.clip(np.rint(row / float(scale)), -qmax - 1, qmax).astype(np.int8)
    return q.tobytes(), float(scale)


def _dequantize_row(blob, bits, scale, d):
    if bits <= 0:
        return np.zeros(d, dtype=np.float32)
    if bits >= 16:
        return np.frombuffer(blob, dtype="<f2").astype(np.float32)
    return np.frombuffer(blob, dtype=np.int8).astype(np.float32) * scale


class SealedBlockCodec:
    """Encode/decode fixed blocks of one layer/head K or V stream."""

    def __init__(self, config):
        self.config = config

    def _transform(self, x32, metadata=None):
        """Build the transform, deterministically. Data-adaptive
        transforms are fitted, their metadata rounded to fp16, and
        the transform rebuilt from the rounded metadata so encode
        and decode use bit-identical bases."""
        t = make_transform(self.config.transform, self.config.block_size)
        if t.name == "poweriter_householder":
            if metadata is None:
                t.fit(x32)
                metadata = t.v.numpy().astype("<f2")
            t.v = torch.from_numpy(metadata.astype(np.float32))
        elif t.name == "pca_oracle":
            if metadata is None:
                t.fit(x32)
                metadata = t._mat.to(torch.float32).numpy().astype("<f2")
            m = torch.from_numpy(
                metadata.astype(np.float64).reshape(
                    self.config.block_size, self.config.block_size
                )
            )
            # fp16 rounding breaks exact orthogonality; canonical QR
            # re-orthonormalization (positive diag(R)) restores it
            # deterministically from the stored bytes, so encode and
            # decode share a bit-identical orthogonal basis.
            q, r = torch.linalg.qr(m.transpose(-1, -2))
            s = torch.sign(torch.diagonal(r))
            s[s == 0] = 1.0
            t._mat = (q * s).transpose(-1, -2).contiguous()
        elif metadata is None:
            metadata = np.zeros(0, dtype="<f2")
        return t, metadata

    def encode_block(self, x_block, block_index):
        b = self.config.block_size
        assert x_block.shape[0] == b, "sealed blocks are exactly B tokens"
        x32 = x_block.detach().to(torch.float32).cpu()
        t, metadata = self._transform(x32)
        y = t.forward(x32).numpy()
        rows, scales = [], []
        for i, bits in enumerate(self.config.quant_alloc):
            blob, scale = _quantize_row(y[i], bits)
            rows.append(blob)
            scales.append(scale)
        header = json.dumps(
            {
                "d": int(y.shape[1]),
                "scales": [float(np.float16(s)) for s in scales],
                "meta_len": int(metadata.size),
            },
            sort_keys=True,
        ).encode()
        payload = (
            len(header).to_bytes(4, "little")
            + header
            + metadata.tobytes()
            + b"".join(rows)
        )
        return BlockArtifact(self.config.cache_key(), block_index, payload)

    def decode_block(self, artifact):
        assert artifact.config_key == self.config.cache_key(), "config mismatch"
        buf = artifact.payload
        hlen = int.from_bytes(buf[:4], "little")
        header = json.loads(buf[4 : 4 + hlen].decode())
        d = header["d"]
        off = 4 + hlen
        metadata = np.frombuffer(
            buf[off : off + 2 * header["meta_len"]], dtype="<f2"
        ).copy()
        off += 2 * header["meta_len"]
        y = np.zeros((self.config.block_size, d), dtype=np.float32)
        for i, bits in enumerate(self.config.quant_alloc):
            n = 0 if bits <= 0 else (2 * d if bits >= 16 else d)
            y[i] = _dequantize_row(buf[off : off + n], bits, header["scales"][i], d)
            off += n
        t, _ = self._transform(None, metadata=metadata if metadata.size else None)
        return t.inverse(torch.from_numpy(y))

    def seal_context(self, x_ctx):
        """Split a [T, d] stream into sealed artifacts plus raw tail.

        Returns (artifacts, tail) where tail is the trailing T % B
        tokens, untouched.
        """
        b = self.config.block_size
        n_full = x_ctx.shape[0] // b
        artifacts = [
            self.encode_block(x_ctx[j * b : (j + 1) * b], j) for j in range(n_full)
        ]
        return artifacts, x_ctx[n_full * b :]

# SPDX-License-Identifier: MIT
"""Python adapter to Google's compiled FDE generator; no algorithm port."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile

import numpy as np

DEFAULT_BUILD_DIR = Path(__file__).resolve().parents[3] / ".cache/google-muvera"


@dataclass(frozen=True)
class FDEConfig:
    repetitions: int = 5
    k_sim: int = 4
    projection_dim: int | None = None
    seed: int = 1
    fill_empty: bool = True
    final_dim: int | None = None

    def __post_init__(self):
        for name in ("repetitions", "k_sim", "seed"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if self.repetitions < 1 or not 0 <= self.k_sim <= 20:
            raise ValueError("repetitions must be positive and k_sim within [0, 20]")
        if not 0 <= self.seed <= (2**31 - 1 - self.repetitions):
            raise ValueError("seed plus repetitions must fit a signed int32")
        for name in ("projection_dim", "final_dim"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                or value > 100_000_000
            ):
                raise ValueError(f"{name} must be a nonnegative integer or None")
        if not isinstance(self.fill_empty, bool):
            raise ValueError("fill_empty must be boolean")

    def output_dim(self, input_dim):
        raw = self.repetitions * 2**self.k_sim * (self.projection_dim or input_dim)
        if raw > 100_000_000:
            raise ValueError("intermediate FDE exceeds 100 million coordinates")
        return self.final_dim or raw


class GoogleFDE:
    def __init__(self, binary=None, build_dir=None):
        build_dir = Path(build_dir or DEFAULT_BUILD_DIR).expanduser().resolve()
        self.binary = (
            Path(
                binary
                or os.environ.get("GOOGLE_FDE_BINARY")
                or build_dir / "google_fde"
            )
            .expanduser()
            .resolve()
        )
        if not self.binary.is_file():
            raise FileNotFoundError(
                f"Missing official FDE binary: {self.binary}. "
                "Run build_google_fde.py --build-dir PATH first."
            )
        provenance_file = self.binary.parent / "provenance.json"
        if not provenance_file.is_file():
            raise FileNotFoundError(
                f"Missing build provenance: {provenance_file}. Rebuild with "
                "build_google_fde.py before benchmarking."
            )
        self.provenance = json.loads(provenance_file.read_text())
        with self.binary.open("rb") as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != self.provenance.get("binary_sha256"):
            raise RuntimeError("FDE binary checksum differs from build provenance")

    def encode(self, clouds, role, config=None):
        """Encode nonempty float32 token clouds with the same config for both roles.

        `fill_empty` applies only to documents. The output is unnormalized.
        With identity projection, divide dot products by repetitions when
        comparing score approximation error against unnormalized Chamfer.
        """
        config = (
            FDEConfig(**config) if isinstance(config, dict) else config or FDEConfig()
        )
        if not isinstance(config, FDEConfig):
            raise TypeError("config must be FDEConfig or a dict")
        if role not in ("query", "document"):
            raise ValueError("role must be query or document")
        clouds = list(clouds)
        if not clouds:
            raise ValueError("at least one point cloud is required")
        arrays = []
        dimension = None
        for cloud in clouds:
            array = np.asarray(cloud, dtype="<f4", order="C")
            if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
                raise ValueError("each cloud must be nonempty [tokens, dimensions]")
            if not np.isfinite(array).all():
                raise ValueError("cloud contains nonfinite values")
            if dimension is None:
                dimension = array.shape[1]
            if array.shape[1] != dimension:
                raise ValueError("all clouds must have the same embedding dimension")
            if array.size > 100_000_000 or dimension > 1_000_000:
                raise ValueError("cloud exceeds adapter size limits")
            arrays.append(array)
        output_dim = config.output_dim(dimension)
        options = asdict(config)
        command = [str(self.binary), "--role", role]
        for field, flag in (
            ("repetitions", "--repetitions"),
            ("k_sim", "--k-sim"),
            ("projection_dim", "--projection-dim"),
            ("seed", "--seed"),
            ("fill_empty", "--fill-empty"),
            ("final_dim", "--final-dim"),
        ):
            value = options[field]
            command.extend([flag, str(int(value or 0))])
        # Files avoid holding serialized inputs and outputs as extra RAM copies.
        with tempfile.TemporaryDirectory(prefix="google-fde-") as temporary:
            incoming = Path(temporary) / "input.bin"
            outgoing = Path(temporary) / "output.bin"
            with incoming.open("wb") as stream:
                stream.write(
                    struct.pack("<8sII", b"GFDEv1\0\0", len(arrays), dimension)
                )
                for array in arrays:
                    stream.write(struct.pack("<I", len(array)))
                    array.tofile(stream)
            with incoming.open("rb") as source, outgoing.open("wb") as target:
                completed = subprocess.run(
                    command, stdin=source, stdout=target, stderr=subprocess.PIPE
                )
            if completed.returncode:
                raise RuntimeError(completed.stderr.decode("utf-8", errors="replace"))
            expected_size = 16 + len(arrays) * output_dim * 4
            if outgoing.stat().st_size != expected_size:
                raise RuntimeError("official FDE output has an unexpected size")
            with outgoing.open("rb") as stream:
                header = struct.unpack("<8sII", stream.read(16))
                if header != (b"GFDEo1\0\0", len(arrays), output_dim):
                    raise RuntimeError("official FDE output header mismatch")
                result = np.fromfile(stream, dtype="<f4").reshape(
                    len(arrays), output_dim
                )
        if not np.isfinite(result).all():
            raise RuntimeError("official FDE returned nonfinite coordinates")
        return result

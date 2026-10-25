# SPDX-License-Identifier: MIT
"""Compute ordinary SHA-256 on contiguous CUDA bytes without a payload download."""

import ctypes
from pathlib import Path


class CudaSha256:
    """Load the caller-built SHA-256 kernel; allocate nothing until hashing.

    The caller must complete external I/O and producer writes before use,
    retain source ownership through completion, and drain device work before
    releasing buffers if a call fails. Exceptions keep the operation's frames
    available to the caller's existing quarantine and drain handling.
    """

    def __init__(self, library: Path):
        self.library = ctypes.CDLL(str(library.resolve(strict=True)))
        self.launch = self.library.kmesh_sha256
        self.launch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_void_p,
        ]
        self.launch.restype = ctypes.c_int

    def hexdigest(self, tensor) -> str:
        """Hash all contiguous bytes, including padding, and return one digest.

        Accept a contiguous CUDA tensor of any dtype. Return the ordinary
        64-character SHA-256 hex string. Raise ValueError for unsupported
        tensors, or RuntimeError on launch, completion or download failure.
        """
        import torch

        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError("GPU SHA-256 requires a contiguous CUDA tensor")
        raw = tensor.view(torch.uint8).reshape(1, -1)
        return self.hexdigests(raw)[0]

    def hexdigests(self, rows) -> tuple[str, ...]:
        """Hash each row of a contiguous two-dimensional CUDA uint8 tensor.

        Every row is a complete independent message. Only 32 bytes per row
        return to the host. The call waits for digest availability. The caller
        owns I/O visibility, source lifetime and error-drain handling.
        """
        import torch

        if (
            not rows.is_cuda
            or rows.dtype != torch.uint8
            or rows.ndim != 2
            or not rows.is_contiguous()
            or not 1 <= rows.shape[0] <= 65535
        ):
            raise ValueError("hash rows must be contiguous CUDA uint8 messages")
        count, size = rows.shape
        with torch.cuda.device(rows.device):
            output = torch.empty((count, 32), dtype=torch.uint8, device=rows.device)
            stream = torch.cuda.current_stream(rows.device)
            status = self.launch(
                rows.data_ptr(),
                size,
                size,
                output.data_ptr(),
                count,
                stream.cuda_stream,
            )
            if status:
                raise RuntimeError(
                    f"GPU SHA-256 launch failed with CUDA error {status}"
                )
            rows.record_stream(stream)
            output.record_stream(stream)
            raw = output.cpu().numpy().tobytes()
        return tuple(raw[i * 32 : (i + 1) * 32].hex() for i in range(count))

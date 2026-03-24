"""Compatibility shim for the historical GPT-2 path.

The reusable Triton decode-path kernels now live under `kernels/`.
Keep this module so older imports continue to work while the neutral path
becomes the documented home.
"""

from kernels.triton_decode_kernels import *  # noqa: F401,F403

"""Paper-grade fused INT4 decode kernels with explicit provenance.

This module exists to stop future confusion between:

1. the generic fused expand helpers in `gpt2/compression/triton_kernels.py`, and
2. the actual AMD/W7900 decode-kernel optimization ladder used in the
   fused-KV paper benchmarks.

Provenance:
- Source benchmark/ablation: `scripts/v31_kernel_bench.py`
- Paper context: `docs/fused_kv_quantization.md`
- Key W7900 path: v31 pipelines C/D/E, where
  * C = Delta1 (scale broadcast reuse)
  * D = Delta2 (RDNA3 wavefront-aware tiling, BLOCK_N=128)
  * E = Delta1 + Delta2 (paper-grade W7900 production path)

Important:
- `gpt2/compression/triton_kernels.py` provides generic fused expand kernels.
- This module provides the reusable decode-path entry points that match the
  paper's W7900 fused attention experiments more closely.
"""

from __future__ import annotations

from scripts.v31_kernel_bench import (
    _kernel_b,
    _kernel_c,
    _kernel_d,
    _kernel_e,
    _launch_kernel,
)


def pipeline_v30_baseline(Q, K_packed, V_packed, scale_k, scale_v, cfg):
    """Pipeline B from v31: baseline fused INT4 decode kernel (BLOCK_N=64)."""
    return _launch_kernel(_kernel_b, Q, K_packed, V_packed, scale_k, scale_v, cfg, 64)


def pipeline_w7900_delta1(Q, K_packed, V_packed, scale_k, scale_v, cfg):
    """Pipeline C from v31: scale broadcast reuse / reduced redundant loads."""
    return _launch_kernel(_kernel_c, Q, K_packed, V_packed, scale_k, scale_v, cfg, 64)


def pipeline_w7900_delta2(Q, K_packed, V_packed, scale_k, scale_v, cfg):
    """Pipeline D from v31: RDNA3 wavefront-aware tiling (BLOCK_N=128)."""
    return _launch_kernel(_kernel_d, Q, K_packed, V_packed, scale_k, scale_v, cfg, 128)


def pipeline_w7900_production(Q, K_packed, V_packed, scale_k, scale_v, cfg):
    """Pipeline E from v31: Delta1 + Delta2 combined, recommended W7900 path."""
    return _launch_kernel(_kernel_e, Q, K_packed, V_packed, scale_k, scale_v, cfg, 128)


PIPELINES = {
    "v30_baseline": pipeline_v30_baseline,
    "w7900_delta1": pipeline_w7900_delta1,
    "w7900_delta2": pipeline_w7900_delta2,
    "w7900_production": pipeline_w7900_production,
}

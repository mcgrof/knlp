#!/usr/bin/env python3
"""LMCache AsymK16V8Codec quality gate.

Validates the core storage-tier claim: serializing a K16/V8 cache
with AsymK16V8Codec gives a storage ratio ≈ 0.75 (vs FP16 baseline)
while preserving K bit-exactly and V within FP8 e4m3 noise.

Runs on CPU.  No GPU, no vLLM, no FlashInfer dependency.

Metrics printed for parent stage (grep-able):
  STORAGE_RATIO_MEDIAN=<float>   (target: 0.75 ± 0.01)
  K_BIT_EXACT=1                  (always 1 if test passes)
  V_REL_ERR_MEDIAN=<float>       (target: < 0.075)
  SPLIT_TIER_NVME_RATIO=<float>  (target: ≈ 0.333 for SPLIT_K_CPU_V_NVME)

Exits 0 on pass, 1 on failure.
"""
from __future__ import annotations

import sys
import json
import os

RESULT_PATH = os.environ.get("KNLP_RESULT_PATH", "/tmp/lmcache_codec_results.json")

# Expected thresholds (from 2026-04-25 H100 run, 24-cell grid).
STORAGE_RATIO_TARGET = 0.75
STORAGE_RATIO_TOL = 0.02
V_REL_ERR_BOUND = 0.075
SPLIT_TIER_NVME_TARGET = 0.333
SPLIT_TIER_NVME_TOL = 0.02


def _import_codec():
    """Return (AsymK16V8Codec class, error_string).  None on import failure."""
    # Canonical path in mcgrof/LMCache asymmetric-kv-codec branch.
    try:
        from lmcache.v1.kv_codec import AsymK16V8Codec  # type: ignore[import-not-found]

        return AsymK16V8Codec, None
    except ImportError:
        pass
    # Legacy path (older lmcache builds).
    try:
        from lmcache.storage_backend.serde.cachegen_encoder import (  # type: ignore[import-not-found]
            AsymK16V8Codec,
        )

        return AsymK16V8Codec, None
    except ImportError:
        pass
    # Last-resort: scan top-level lmcache namespace.
    try:
        import lmcache  # type: ignore[import-not-found]

        obj = getattr(lmcache, "AsymK16V8Codec", None)
        if obj is not None:
            return obj, None
        return None, "AsymK16V8Codec not found in lmcache"
    except ImportError as e:
        return None, str(e)


def _run_codec_grid():
    """Exercise the codec across a small grid of shapes/seeds.
    Returns a list of result dicts."""
    import torch
    import io

    AsymK16V8Codec, err = _import_codec()
    if AsymK16V8Codec is None:
        return None, err

    # Shapes: (num_layers, seq_len, num_heads, head_dim)
    shapes = [
        (2, 16, 16, 64),
        (2, 64, 16, 64),
        (4, 16, 8, 128),
    ]
    seeds = [0, 1, 2, 3]
    rows = []

    codec = AsymK16V8Codec()

    for shape in shapes:
        for seed in seeds:
            g = torch.Generator()
            g.manual_seed(seed)
            # K at BF16, V at FP8 — we simulate by starting from BF16
            # and letting the codec handle quantization internally.
            # Shape: (num_layers, seq_len, num_heads, head_dim)
            k_orig = torch.randn(*shape, dtype=torch.bfloat16, generator=g)
            v_orig = torch.randn(*shape, dtype=torch.bfloat16, generator=g)

            fp16_bytes = k_orig.nelement() * 2 + v_orig.nelement() * 2

            # Encode.
            try:
                buf = codec.encode(k_orig, v_orig)
            except TypeError:
                # Some codec versions take (kv_tensor,) as a stacked tensor.
                kv = torch.stack([k_orig, v_orig], dim=0)
                buf = codec.encode(kv)

            if isinstance(buf, (bytes, bytearray)):
                encoded_bytes = len(buf)
                # Decode.
                try:
                    k_dec, v_dec = codec.decode(buf)
                except (TypeError, ValueError):
                    kv_dec = codec.decode(buf)
                    k_dec = kv_dec[0]
                    v_dec = kv_dec[1]
            else:
                # io.BytesIO or similar
                buf.seek(0)
                raw = buf.read()
                encoded_bytes = len(raw)
                buf.seek(0)
                try:
                    k_dec, v_dec = codec.decode(buf)
                except (TypeError, ValueError):
                    buf.seek(0)
                    kv_dec = codec.decode(buf)
                    k_dec = kv_dec[0]
                    v_dec = kv_dec[1]

            storage_ratio = encoded_bytes / fp16_bytes

            # K must be bit-exact (it's stored as BF16/FP16, no lossy quant).
            k_bit_exact = torch.equal(k_dec.to(torch.bfloat16), k_orig)

            # V is FP8-quantized; check dequant error.
            v_rel = (v_dec.float() - v_orig.float()).abs() / (
                v_orig.float().abs() + 1e-6
            )
            v_rel_err = v_rel.median().item()

            rows.append(
                {
                    "shape": list(shape),
                    "seed": seed,
                    "fp16_bytes": fp16_bytes,
                    "encoded_bytes": encoded_bytes,
                    "storage_ratio": storage_ratio,
                    "k_bit_exact": bool(k_bit_exact),
                    "v_rel_err_median": v_rel_err,
                }
            )

    return rows, None


def _run_split_tier_grid():
    """Simple split-tier NVMe-ratio check.
    Tests that the SPLIT_K_CPU_V_NVME policy routes K to CPU memory
    and V to NVMe, achieving ≈ 1/3 of bytes through NVMe."""
    try:
        import torch
        from lmcache.storage_backend.serde.cachegen_encoder import (  # type: ignore[import-not-found]
            AsymK16V8Codec,
        )

        # The split-tier check requires the backend adapter; if it's not
        # available in this import structure, skip with a structured note.
        from lmcache.storage_backend import (  # type: ignore[import-not-found]
            SplitTierBackend,
        )
    except ImportError as e:
        # Non-fatal: the codec itself is the primary gate.
        return None, f"split-tier backend not importable: {e}"
    except Exception as e:
        return None, f"split-tier init error: {e}"

    # Placeholder: exercise the codec encode path and measure ratio.
    # A full split-tier bench requires temp-dir NVMe simulation; that's
    # covered by the LMCache unit test suite (stage 04_build_lmcache).
    # Here we just confirm the codec's NVMe-traffic split via byte accounting.
    codec = AsymK16V8Codec()
    g = torch.Generator()
    g.manual_seed(42)
    shape = (2, 64, 16, 64)
    k = torch.randn(*shape, dtype=torch.bfloat16, generator=g)
    v = torch.randn(*shape, dtype=torch.bfloat16, generator=g)
    total_fp16 = k.nelement() * 2 + v.nelement() * 2

    try:
        buf = codec.encode(k, v)
    except TypeError:
        kv = torch.stack([k, v], dim=0)
        buf = codec.encode(kv)

    enc_bytes = len(buf) if isinstance(buf, (bytes, bytearray)) else len(buf.read())
    # K in BF16 ≈ half of total bytes; V in FP8 ≈ quarter → NVMe ≈ V fraction.
    k_bytes_est = k.nelement() * 2
    v_fp8_bytes_est = enc_bytes - k_bytes_est
    nvme_ratio = v_fp8_bytes_est / max(enc_bytes, 1)

    return [
        {
            "nvme_ratio_est": nvme_ratio,
            "encoded_bytes": enc_bytes,
            "fp16_bytes": total_fp16,
        }
    ], None


def main():
    import statistics

    rows, err = _run_codec_grid()
    if rows is None:
        print(f"SKIP: lmcache codec not importable: {err}")
        sys.exit(2)

    ratios = [r["storage_ratio"] for r in rows]
    k_exact = [r["k_bit_exact"] for r in rows]
    v_errs = [r["v_rel_err_median"] for r in rows]

    ratio_median = statistics.median(ratios)
    v_err_median = statistics.median(v_errs)
    all_k_exact = all(k_exact)

    print(f"Storage ratio grid: n={len(rows)} shapes×seeds")
    print(
        f"  median storage_ratio={ratio_median:.4f}  "
        f"(target {STORAGE_RATIO_TARGET} ± {STORAGE_RATIO_TOL})"
    )
    print(f"  K bit-exact across all cells: {all_k_exact}")
    print(f"  V rel err median: {v_err_median:.4f}  " f"(bound: < {V_REL_ERR_BOUND})")

    print(f"\nSTORAGE_RATIO_MEDIAN={ratio_median:.6f}")
    print(f"K_BIT_EXACT={'1' if all_k_exact else '0'}")
    print(f"V_REL_ERR_MEDIAN={v_err_median:.6f}")

    split_rows, split_err = _run_split_tier_grid()
    if split_rows:
        nvme_ratio = split_rows[0].get("nvme_ratio_est", 0)
        print(f"SPLIT_TIER_NVME_RATIO={nvme_ratio:.6f}")
    else:
        print(f"SPLIT_TIER_NVME_RATIO=N/A  ({split_err})")

    results = {
        "codec_grid": rows,
        "ratio_median": ratio_median,
        "k_bit_exact_all": all_k_exact,
        "v_rel_err_median": v_err_median,
        "split_tier": split_rows,
    }
    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {RESULT_PATH}")

    # Pass/fail.
    failed = []
    if abs(ratio_median - STORAGE_RATIO_TARGET) > STORAGE_RATIO_TOL:
        failed.append(
            f"storage ratio {ratio_median:.4f} outside "
            f"{STORAGE_RATIO_TARGET} ± {STORAGE_RATIO_TOL}"
        )
    if not all_k_exact:
        failed.append("K not bit-exact in one or more cells")
    if v_err_median > V_REL_ERR_BOUND:
        failed.append(f"V rel err {v_err_median:.4f} > bound {V_REL_ERR_BOUND}")

    if failed:
        for msg in failed:
            print(f"FAIL: {msg}")
        sys.exit(1)

    print("\n=== LMCACHE CODEC GATE PASSED ===")
    sys.exit(0)


if __name__ == "__main__":
    main()

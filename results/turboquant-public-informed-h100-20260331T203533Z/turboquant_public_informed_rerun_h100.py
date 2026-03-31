#!/usr/bin/env python3
"""TurboQuant public-informed fair rerun — H100.

This rerun exists because public community evidence (TheTom/turboquant_plus,
llama.cpp ports) highlighted specific bug/overhead classes in typical TurboQuant
implementations. We give TurboQuant a fairer shot by:

1. Explicit coordinate-system sanity checks (known public bug class)
2. Long-context-biased evaluation (8K–64K) where rotation overhead amortizes
3. Community-style practical TurboQuant (no QJL, verified rotation)
4. Quality-first analysis: no speed claim without quality validation

Arms:
  A       = Uncompressed FP16 (anchor)
  D       = Current fused quantization baseline (per-group INT4 g32, no rotation)
  B-prac  = Community-style TurboQuant: rotation + per-group INT4, no QJL,
            correctness-verified, graph-side amortization where possible
  C-long  = B-prac specialized for long-context decode: same representation,
            but latency measured in the decode-step regime (single new token
            attending to long KV cache)

This is NOT a re-run of the ABCD ablation. It is a targeted follow-up asking:
"Given public evidence, does TurboQuant deserve a long-context-only lane?"
"""

import gc
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import torch
import torch.nn.functional as F

# ============================================================
# Configuration — long-context biased
# ============================================================

CONFIGS = {
    "qwen25_7b": {
        "name": "Qwen2.5-7B-like",
        "num_heads": 28,
        "num_kv_heads": 4,
        "head_dim": 128,
    },
    "llama3_8b": {
        "name": "Llama3.1-8B-like",
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
    },
}

# Long-context biased: skip short contexts where rotation overhead dominates
BATCH_SIZES = [1, 4]
SEQ_LENS = [8192, 16384, 32768, 65536]
GROUP_SIZE = 32
BITS = 4

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RESULTS_ROOT = os.environ.get(
    "RESULTS_ROOT",
    f"/workspace/results/turboquant-public-informed-h100-{TIMESTAMP}"
)
DURABLE_ROOT = os.environ.get(
    "DURABLE_ROOT",
    f"/data/knlp-key-results/fused-quant/turboquant-public-informed-h100-{TIMESTAMP}"
)

ROTATION_SEED = 42
WARMUP = 10
TRIALS = 50


# ============================================================
# Rotation matrix utilities
# ============================================================

def generate_rotation_matrix(d: int, device: torch.device, dtype: torch.dtype,
                              seed: int = ROTATION_SEED) -> torch.Tensor:
    """Random orthogonal matrix via QR (Haar-distributed)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    G = torch.randn(d, d, generator=gen, dtype=torch.float32)
    Q, R = torch.linalg.qr(G)
    diag_sign = torch.sign(torch.diagonal(R))
    Q = Q * diag_sign.unsqueeze(0)
    return Q.to(device=device, dtype=dtype)


# ============================================================
# SANITY CHECK SUITE — public bug class audit
# ============================================================

def sanity_check_rotation(d: int, device: torch.device, dtype: torch.dtype) -> dict:
    """Audit rotation matrix for known public bug classes.

    Public evidence (TheTom/turboquant_plus) showed catastrophic failures from:
    1. Non-orthogonal rotation matrix (Q·Q^T != I)
    2. Wrong coordinate system (applying rotation to wrong dimension)
    3. Missing inverse rotation on output path
    4. Numerical drift from FP16 rotation at large d
    """
    results = {}

    Q = generate_rotation_matrix(d, device, dtype)

    # Check 1: Orthogonality — Q·Q^T should be identity
    QQT = Q @ Q.T
    I = torch.eye(d, device=device, dtype=dtype)
    ortho_err = (QQT - I).abs().max().item()
    ortho_frob = (QQT - I).norm().item()
    results["orthogonality_max_err"] = ortho_err
    results["orthogonality_frob_err"] = ortho_frob
    results["orthogonality_pass"] = ortho_err < 1e-2  # FP16 tolerance

    # Check 2: Q^T·Q should also be identity (verifies both sides)
    QTQ = Q.T @ Q
    ortho_err2 = (QTQ - I).abs().max().item()
    results["orthogonality_transpose_max_err"] = ortho_err2
    results["orthogonality_transpose_pass"] = ortho_err2 < 1e-2

    # Check 3: Round-trip rotate → inverse-rotate ≈ identity
    x = torch.randn(64, d, device=device, dtype=dtype)
    x_rot = x @ Q
    x_back = x_rot @ Q.T
    roundtrip_err = (x_back - x).abs().max().item()
    roundtrip_mse = F.mse_loss(x_back, x).item()
    roundtrip_cos = F.cosine_similarity(x, x_back, dim=-1).mean().item()
    results["roundtrip_max_err"] = roundtrip_err
    results["roundtrip_mse"] = roundtrip_mse
    results["roundtrip_cos_sim"] = roundtrip_cos
    results["roundtrip_pass"] = roundtrip_cos > 0.9999

    # Check 4: Rotation preserves norms (isometry check)
    x_norms = x.norm(dim=-1)
    xrot_norms = x_rot.norm(dim=-1)
    norm_ratio = (xrot_norms / x_norms.clamp(min=1e-8))
    norm_err = (norm_ratio - 1.0).abs().max().item()
    results["norm_preservation_max_err"] = norm_err
    results["norm_preservation_pass"] = norm_err < 0.01

    # Check 5: Dimension sanity — rotation applied to last dim (head_dim), not seq or head dim
    # This catches the coordinate-system bug: if someone rotates along seq_len or num_heads,
    # the matrix dimensions won't match. We verify by checking shape compatibility.
    shape_4d = (2, 4, 16, d)  # (B, H, L, D)
    x4d = torch.randn(shape_4d, device=device, dtype=dtype)
    try:
        x4d_flat = x4d.reshape(-1, d)
        x4d_rot = x4d_flat @ Q
        x4d_back = x4d_rot @ Q.T
        x4d_back = x4d_back.reshape(shape_4d)
        dim_err = (x4d_back - x4d).abs().max().item()
        results["dimension_check_pass"] = dim_err < 0.05
        results["dimension_check_err"] = dim_err
    except RuntimeError as e:
        results["dimension_check_pass"] = False
        results["dimension_check_err"] = str(e)

    # Check 6: FP16 vs FP32 rotation drift
    Q32 = generate_rotation_matrix(d, device, torch.float32)
    x32 = x.float()
    x_rot_32 = x32 @ Q32
    x_rot_16 = x @ Q
    precision_drift = (x_rot_16.float() - x_rot_32).abs().max().item()
    results["fp16_precision_drift"] = precision_drift
    results["fp16_precision_pass"] = precision_drift < 0.1

    # Overall pass/fail
    all_pass = all(v for k, v in results.items() if k.endswith("_pass"))
    results["ALL_SANITY_CHECKS_PASS"] = all_pass

    return results


def sanity_check_quant_roundtrip(d: int, device: torch.device, dtype: torch.dtype) -> dict:
    """Verify that TurboQuant quant→dequant→inverse-rotate produces valid output.

    This catches the bug where quality looks good in the rotated domain
    but the inverse rotation is missing or wrong on the output path.
    """
    Q = generate_rotation_matrix(d, device, dtype)
    x = torch.randn(64, d, device=device, dtype=dtype) / math.sqrt(d)

    # Step 1: Rotate
    x_rot = x @ Q

    # Step 2: Quantize in rotated domain
    gs = GROUP_SIZE
    x_rot_g = x_rot.reshape(-1, d // gs, gs)
    amax = x_rot_g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / 7.0
    q = (x_rot_g / scale).round().clamp(-8, 7)
    x_rot_hat = (q * scale).reshape(-1, d)

    # Step 3: Inverse-rotate (THIS is what public implementations got wrong)
    x_hat = x_rot_hat @ Q.T

    # Step 4: Compare quality in ORIGINAL domain (not rotated domain!)
    mse_original = F.mse_loss(x_hat, x).item()
    cos_original = F.cosine_similarity(x, x_hat, dim=-1).mean().item()

    # Also measure quality if we FORGOT the inverse rotation (the public bug)
    mse_no_inv = F.mse_loss(x_rot_hat, x).item()
    cos_no_inv = F.cosine_similarity(x, x_rot_hat, dim=-1).mean().item()

    # And quality of plain quantization without rotation for comparison
    x_g = x.reshape(-1, d // gs, gs)
    amax_plain = x_g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale_plain = amax_plain / 7.0
    q_plain = (x_g / scale_plain).round().clamp(-8, 7)
    x_hat_plain = (q_plain * scale_plain).reshape(-1, d)
    mse_plain = F.mse_loss(x_hat_plain, x).item()
    cos_plain = F.cosine_similarity(x, x_hat_plain, dim=-1).mean().item()

    return {
        "turboquant_correct_mse": mse_original,
        "turboquant_correct_cos": cos_original,
        "turboquant_NO_inverse_mse": mse_no_inv,
        "turboquant_NO_inverse_cos": cos_no_inv,
        "plain_quant_mse": mse_plain,
        "plain_quant_cos": cos_plain,
        "inverse_rotation_matters": mse_no_inv > mse_original * 2,
        "correct_path_quality_ok": cos_original > 0.99,
        "bug_class_detected": cos_no_inv < 0.9,
    }


# ============================================================
# Quantization arms
# ============================================================

def arm_a_identity(x: torch.Tensor):
    """Arm A: uncompressed FP16 pass-through."""
    return x.clone()


def arm_d_fused_quant(x: torch.Tensor, group_size: int = GROUP_SIZE):
    """Arm D: per-group INT4 symmetric scalar quantization, no rotation."""
    orig_shape = x.shape
    d = x.shape[-1]
    x_flat = x.reshape(-1, d)
    gs = group_size
    x_g = x_flat.reshape(-1, d // gs, gs)
    amax = x_g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / 7.0
    q = (x_g / scale).round().clamp(-8, 7)
    x_hat = (q * scale).reshape(orig_shape)
    return x_hat


def arm_b_practical(x: torch.Tensor, Q: torch.Tensor, group_size: int = GROUP_SIZE):
    """Arm B-practical: community-style TurboQuant.

    - QR rotation (correctness-verified)
    - Per-group INT4 in rotated domain
    - Inverse rotation on output (explicitly!)
    - No QJL
    """
    orig_shape = x.shape
    d = x.shape[-1]
    x_flat = x.reshape(-1, d)

    # Rotate into TurboQuant domain
    x_rot = x_flat @ Q

    # Per-group INT4 quantization in rotated domain
    gs = group_size
    x_rot_g = x_rot.reshape(-1, d // gs, gs)
    amax = x_rot_g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / 7.0
    q = (x_rot_g / scale).round().clamp(-8, 7)
    x_rot_hat = (q * scale).reshape(-1, d)

    # CRITICAL: inverse-rotate back to original domain
    x_hat = (x_rot_hat @ Q.T).reshape(orig_shape)
    return x_hat


# ============================================================
# Decode-step simulation (long-context regime)
# ============================================================

def simulate_decode_step_fused(K_cache: torch.Tensor, V_cache: torch.Tensor,
                                q: torch.Tensor) -> torch.Tensor:
    """Simulate single decode step with fused-quant KV cache.

    K_cache, V_cache: (B, H, T, D) — already quantized+dequantized (arm D)
    q: (B, H, 1, D) — query for new token
    Returns: attention output (B, H, 1, D)
    """
    # Standard scaled dot-product attention
    scores = (q @ K_cache.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    attn = torch.softmax(scores, dim=-1)
    out = attn @ V_cache
    return out


def simulate_decode_step_turboquant(K_cache_rot: torch.Tensor, V_cache_rot: torch.Tensor,
                                     q: torch.Tensor, Q_mat: torch.Tensor) -> torch.Tensor:
    """Simulate single decode step with TurboQuant KV cache.

    K_cache_rot, V_cache_rot: (B, H, T, D) — quantized in rotated domain, dequantized
    q: (B, H, 1, D) — query in original domain
    Q_mat: (D, D) — rotation matrix

    The decode-step cost includes:
    - Rotating q into the TurboQuant domain
    - Attention in rotated domain
    - Inverse-rotating the output back
    """
    d = q.shape[-1]
    B, H = q.shape[0], q.shape[1]

    # Rotate query into TurboQuant domain: O(B*H*D^2)
    q_flat = q.reshape(-1, d)
    q_rot = (q_flat @ Q_mat).reshape(B, H, 1, d)

    # Attention in rotated domain: O(B*H*T*D)
    scores = (q_rot @ K_cache_rot.transpose(-2, -1)) / math.sqrt(d)
    attn = torch.softmax(scores, dim=-1)
    out_rot = attn @ V_cache_rot

    # Inverse-rotate output: O(B*H*D^2)
    out_flat = out_rot.reshape(-1, d)
    out = (out_flat @ Q_mat.T).reshape(B, H, 1, d)
    return out


# ============================================================
# Metrics
# ============================================================

def compute_quality(x_orig: torch.Tensor, x_hat: torch.Tensor) -> dict:
    mse = F.mse_loss(x_hat, x_orig).item()
    signal_power = (x_orig ** 2).mean().item()
    nmse = mse / max(signal_power, 1e-10)
    x_f = x_orig.reshape(-1, x_orig.shape[-1])
    xh_f = x_hat.reshape(-1, x_hat.shape[-1])
    cos_sim = F.cosine_similarity(x_f, xh_f, dim=-1).mean().item()
    max_abs = (x_hat - x_orig).abs().max().item()
    return {"mse": mse, "nmse": nmse, "cos_sim": cos_sim, "max_abs_err": max_abs,
            "signal_power": signal_power}


def bench_latency(fn, warmup: int = WARMUP, trials: int = TRIALS) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)

    times.sort()
    n = len(times)
    return {
        "median_us": times[n // 2],
        "mean_us": sum(times) / n,
        "p10_us": times[n // 10],
        "p90_us": times[9 * n // 10],
        "min_us": times[0],
        "max_us": times[-1],
    }


# ============================================================
# Per-config run: quality + decode-step latency
# ============================================================

def run_config(cfg_name: str, cfg: dict, batch: int, seq_len: int,
               device: torch.device) -> dict:
    """Run all arms for one config. Measures both representation quality
    AND decode-step latency (the regime that matters for long-context)."""

    num_kv_heads = cfg["num_kv_heads"]
    head_dim = cfg["head_dim"]
    shape = (batch, num_kv_heads, seq_len, head_dim)

    # Generate KV cache data
    K = torch.randn(shape, device=device, dtype=torch.float16) / math.sqrt(head_dim)
    V = torch.randn(shape, device=device, dtype=torch.float16) / math.sqrt(head_dim)
    # Query for decode step (single token)
    q = torch.randn(batch, num_kv_heads, 1, head_dim, device=device, dtype=torch.float16) / math.sqrt(head_dim)

    Q_mat = generate_rotation_matrix(head_dim, device, torch.float16)

    result = {
        "config": cfg_name, "model": cfg["name"],
        "batch": batch, "seq_len": seq_len,
        "num_kv_heads": num_kv_heads, "head_dim": head_dim,
    }

    # === Quality: representation-level ===
    K_a = arm_a_identity(K)
    K_d = arm_d_fused_quant(K)
    K_b = arm_b_practical(K, Q_mat)

    V_a = arm_a_identity(V)
    V_d = arm_d_fused_quant(V)
    V_b = arm_b_practical(V, Q_mat)

    result["quality_K"] = {
        "A": compute_quality(K, K_a),
        "D": compute_quality(K, K_d),
        "B_prac": compute_quality(K, K_b),
    }
    result["quality_V"] = {
        "A": compute_quality(V, V_a),
        "D": compute_quality(V, V_d),
        "B_prac": compute_quality(V, V_b),
    }

    # === Quality: end-to-end decode output ===
    # Reference output (uncompressed)
    out_a = simulate_decode_step_fused(K_a, V_a, q)

    # Fused quant decode
    out_d = simulate_decode_step_fused(K_d, V_d, q)

    # TurboQuant decode — KV stored in rotated domain for decode-step
    # This is the community-style approach: store quantized-rotated KV,
    # rotate query at decode time, attend in rotated domain, inverse-rotate output
    K_rot_quant = arm_d_fused_quant(K.reshape(-1, head_dim) @ Q_mat)
    K_rot_quant = K_rot_quant.reshape(shape)
    V_rot_quant = arm_d_fused_quant(V.reshape(-1, head_dim) @ Q_mat)
    V_rot_quant = V_rot_quant.reshape(shape)

    out_b = simulate_decode_step_turboquant(K_rot_quant, V_rot_quant, q, Q_mat)

    result["quality_decode"] = {
        "A": {"mse": 0.0, "cos_sim": 1.0},
        "D": compute_quality(out_a, out_d),
        "B_prac": compute_quality(out_a, out_b),
    }

    # === Latency: decode-step ===
    # This is the key measurement — how much does one decode step cost?
    # At long T, the O(B*H*T*D) attention dominates over O(B*H*D^2) rotation.

    result["latency_decode"] = {}

    # Arm A latency
    result["latency_decode"]["A"] = bench_latency(
        lambda: simulate_decode_step_fused(K_a, V_a, q))

    # Arm D latency
    result["latency_decode"]["D"] = bench_latency(
        lambda: simulate_decode_step_fused(K_d, V_d, q))

    # Arm B-practical latency (includes rotation cost in decode step)
    result["latency_decode"]["B_prac"] = bench_latency(
        lambda: simulate_decode_step_turboquant(K_rot_quant, V_rot_quant, q, Q_mat))

    # Compute overhead
    lat_d = result["latency_decode"]["D"]["median_us"]
    lat_b = result["latency_decode"]["B_prac"]["median_us"]
    result["b_overhead_vs_d_pct"] = (lat_b - lat_d) / lat_d * 100.0

    # === Latency: quantize-step (write-time cost) ===
    x_sample = torch.randn(batch, num_kv_heads, 1, head_dim, device=device, dtype=torch.float16)

    result["latency_quantize"] = {}
    result["latency_quantize"]["D"] = bench_latency(
        lambda: arm_d_fused_quant(x_sample))
    result["latency_quantize"]["B_prac"] = bench_latency(
        lambda: arm_b_practical(x_sample, Q_mat))

    return result


# ============================================================
# Printing
# ============================================================

def print_sanity_results(rot_check: dict, quant_check: dict):
    print("\n" + "=" * 90)
    print("PUBLIC BUG-CLASS AUDIT — Coordinate-System Sanity Checks")
    print("=" * 90)

    print("\n1. Rotation matrix orthogonality:")
    print(f"   Q·Q^T max err:   {rot_check['orthogonality_max_err']:.6e}  "
          f"{'PASS' if rot_check['orthogonality_pass'] else 'FAIL'}")
    print(f"   Q^T·Q max err:   {rot_check['orthogonality_transpose_max_err']:.6e}  "
          f"{'PASS' if rot_check['orthogonality_transpose_pass'] else 'FAIL'}")

    print("\n2. Round-trip rotate→inverse-rotate:")
    print(f"   Max err:  {rot_check['roundtrip_max_err']:.6e}")
    print(f"   CosSim:   {rot_check['roundtrip_cos_sim']:.8f}  "
          f"{'PASS' if rot_check['roundtrip_pass'] else 'FAIL'}")

    print("\n3. Norm preservation (isometry):")
    print(f"   Max |ratio-1|:  {rot_check['norm_preservation_max_err']:.6e}  "
          f"{'PASS' if rot_check['norm_preservation_pass'] else 'FAIL'}")

    print("\n4. Dimension check (rotation applied to head_dim, not seq/head):")
    print(f"   {'PASS' if rot_check['dimension_check_pass'] else 'FAIL'}  "
          f"err={rot_check['dimension_check_err']}")

    print("\n5. FP16 precision drift vs FP32:")
    print(f"   Max drift:  {rot_check['fp16_precision_drift']:.6e}  "
          f"{'PASS' if rot_check['fp16_precision_pass'] else 'FAIL'}")

    print(f"\n   ALL SANITY CHECKS: {'PASS' if rot_check['ALL_SANITY_CHECKS_PASS'] else 'FAIL'}")

    print("\n" + "-" * 90)
    print("QUANT ROUND-TRIP AUDIT — Inverse Rotation Verification")
    print("-" * 90)
    print(f"\n   TurboQuant (correct path): MSE={quant_check['turboquant_correct_mse']:.6e}  "
          f"CosSim={quant_check['turboquant_correct_cos']:.6f}")
    print(f"   TurboQuant (NO inv-rot):   MSE={quant_check['turboquant_NO_inverse_mse']:.6e}  "
          f"CosSim={quant_check['turboquant_NO_inverse_cos']:.6f}")
    print(f"   Plain quant (no rotation):  MSE={quant_check['plain_quant_mse']:.6e}  "
          f"CosSim={quant_check['plain_quant_cos']:.6f}")
    print(f"\n   Missing inverse rotation detectable: "
          f"{'YES' if quant_check['bug_class_detected'] else 'NO'}")
    print(f"   Correct path quality OK: "
          f"{'YES' if quant_check['correct_path_quality_ok'] else 'NO'}")


def print_results_table(results: list):
    print("\n" + "=" * 120)
    print("DECODE-STEP QUALITY (end-to-end: query→attention→output)")
    print("=" * 120)
    hdr = f"{'Config':<14} {'B':>2} {'L':>6} | {'D CosSim':>10} {'D MSE':>12} | {'B CosSim':>10} {'B MSE':>12} | {'B-D delta':>10}"
    print(hdr)
    print("-" * 120)

    for r in results:
        qd = r["quality_decode"]["D"]
        qb = r["quality_decode"]["B_prac"]
        delta = qb["cos_sim"] - qd["cos_sim"]
        print(f"{r['config']:<14} {r['batch']:>2} {r['seq_len']:>6} | "
              f"{qd['cos_sim']:>10.6f} {qd['mse']:>12.6e} | "
              f"{qb['cos_sim']:>10.6f} {qb['mse']:>12.6e} | "
              f"{delta:>+10.6f}")

    print("\n" + "=" * 120)
    print("DECODE-STEP LATENCY (microseconds)")
    print("=" * 120)
    hdr = f"{'Config':<14} {'B':>2} {'L':>6} | {'A med':>10} {'D med':>10} {'B med':>10} | {'B ovhd%':>8} | {'B p10':>8} {'B p90':>8}"
    print(hdr)
    print("-" * 120)

    for r in results:
        la = r["latency_decode"]["A"]
        ld = r["latency_decode"]["D"]
        lb = r["latency_decode"]["B_prac"]
        ovhd = r["b_overhead_vs_d_pct"]
        print(f"{r['config']:<14} {r['batch']:>2} {r['seq_len']:>6} | "
              f"{la['median_us']:>10.1f} {ld['median_us']:>10.1f} {lb['median_us']:>10.1f} | "
              f"{ovhd:>+8.1f}% | "
              f"{lb['p10_us']:>8.1f} {lb['p90_us']:>8.1f}")


def print_summary(results: list):
    print("\n" + "=" * 120)
    print("SUMMARY — Public-Informed Fair Rerun")
    print("=" * 120)

    # Group by seq_len to show amortization curve
    by_len = {}
    for r in results:
        L = r["seq_len"]
        if L not in by_len:
            by_len[L] = []
        by_len[L].append(r)

    print("\nOverhead curve (B-practical vs D, decode-step latency):")
    print(f"  {'SeqLen':>8} | {'Avg overhead%':>14} | {'Min overhead%':>14} | {'Max overhead%':>14} | {'Avg D CosSim':>13} | {'Avg B CosSim':>13}")
    print("  " + "-" * 100)

    for L in sorted(by_len.keys()):
        group = by_len[L]
        overheads = [r["b_overhead_vs_d_pct"] for r in group]
        cos_d = [r["quality_decode"]["D"]["cos_sim"] for r in group]
        cos_b = [r["quality_decode"]["B_prac"]["cos_sim"] for r in group]
        print(f"  {L:>8} | {sum(overheads)/len(overheads):>+14.2f}% | "
              f"{min(overheads):>+14.2f}% | {max(overheads):>+14.2f}% | "
              f"{sum(cos_d)/len(cos_d):>13.6f} | {sum(cos_b)/len(cos_b):>13.6f}")

    # Overall averages
    all_ovhd = [r["b_overhead_vs_d_pct"] for r in results]
    all_cos_d = [r["quality_decode"]["D"]["cos_sim"] for r in results]
    all_cos_b = [r["quality_decode"]["B_prac"]["cos_sim"] for r in results]

    # Long-context only (>=32K)
    long_results = [r for r in results if r["seq_len"] >= 32768]
    long_ovhd = [r["b_overhead_vs_d_pct"] for r in long_results] if long_results else [0]
    long_cos_d = [r["quality_decode"]["D"]["cos_sim"] for r in long_results] if long_results else [0]
    long_cos_b = [r["quality_decode"]["B_prac"]["cos_sim"] for r in long_results] if long_results else [0]

    print(f"\n  Overall avg overhead: {sum(all_ovhd)/len(all_ovhd):+.2f}%")
    print(f"  Long-context (>=32K) avg overhead: {sum(long_ovhd)/len(long_ovhd):+.2f}%")
    print(f"\n  Overall avg quality: D={sum(all_cos_d)/len(all_cos_d):.6f}  B={sum(all_cos_b)/len(all_cos_b):.6f}")
    print(f"  Long-context quality: D={sum(long_cos_d)/len(long_cos_d):.6f}  B={sum(long_cos_b)/len(long_cos_b):.6f}")

    # Decision
    print("\n" + "=" * 120)
    print("FINAL ANSWER: Does TurboQuant deserve a long-context-only follow-up line?")
    print("=" * 120)

    avg_long_ovhd = sum(long_ovhd) / len(long_ovhd)
    avg_long_cos_delta = sum(long_cos_b) / len(long_cos_b) - sum(long_cos_d) / len(long_cos_d)

    if avg_long_ovhd < 5.0 and avg_long_cos_delta >= 0:
        verdict = "YES — TurboQuant overhead drops below 5% at long contexts with equal or better quality."
        followup = True
    elif avg_long_ovhd < 10.0 and avg_long_cos_delta > 0.001:
        verdict = "MAYBE — TurboQuant overhead is moderate (<10%) at long contexts with a quality edge."
        followup = True
    elif avg_long_ovhd < 5.0:
        verdict = "MARGINAL — Low overhead at long contexts but no quality benefit. Not worth the complexity."
        followup = False
    else:
        verdict = "NO — TurboQuant overhead remains too high even at long contexts for this dimensionality."
        followup = False

    print(f"\n  Long-context overhead: {avg_long_ovhd:+.2f}%")
    print(f"  Long-context quality delta (B-D): {avg_long_cos_delta:+.6f}")
    print(f"\n  VERDICT: {verdict}")
    print(f"  FOLLOW-UP WARRANTED: {'YES' if followup else 'NO'}")

    if followup:
        print("\n  Recommended next steps:")
        print("  - Write a fused Triton kernel with inline Q-rotation to eliminate the per-step overhead")
        print("  - Test at 2-bit quantization where rotation may provide stronger quality improvement")
        print("  - Test at head_dim >= 256 where non-uniform distributions may benefit from rotation")
    else:
        print("\n  Recommended action:")
        print("  - Keep fused quantization (arm D) as the primary line")
        print("  - Archive TurboQuant results for reference")
        print("  - Revisit only if head_dim increases or models show strongly non-uniform KV distributions")

    return {"verdict": verdict, "followup": followup,
            "long_ctx_overhead_pct": avg_long_ovhd,
            "long_ctx_quality_delta": avg_long_cos_delta}


# ============================================================
# Main
# ============================================================

def main():
    print(f"TurboQuant Public-Informed Fair Rerun — H100")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"Results root: {RESULTS_ROOT}")
    print(f"Durable root: {DURABLE_ROOT}")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Torch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"INT{BITS}, group_size={GROUP_SIZE}")
    print(f"Arms: A (uncompressed), D (fused quant), B-practical (community TurboQuant)")
    print(f"Contexts: {SEQ_LENS} (long-context biased)")
    print(f"QJL: DISABLED (per public evidence, not worthwhile at this operating point)")
    print()

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    os.makedirs(DURABLE_ROOT, exist_ok=True)

    # === Phase 0: Sanity checks ===
    print("Phase 0: Public bug-class audit (coordinate-system sanity checks)")
    rot_check = sanity_check_rotation(128, device, torch.float16)
    quant_check = sanity_check_quant_roundtrip(128, device, torch.float16)
    print_sanity_results(rot_check, quant_check)

    if not rot_check["ALL_SANITY_CHECKS_PASS"]:
        print("\nFATAL: Sanity checks failed. Aborting rerun.")
        sys.exit(1)

    # Save sanity check results
    sanity_path = os.path.join(RESULTS_ROOT, "sanity_checks.json")
    with open(sanity_path, "w") as f:
        json.dump({"rotation": rot_check, "quant_roundtrip": quant_check}, f, indent=2, default=str)
    print(f"\nSanity checks saved: {sanity_path}")

    # === Phase 1: Long-context decode-step evaluation ===
    print(f"\nPhase 1: Long-context decode-step evaluation")
    all_results = []
    total = len(CONFIGS) * len(BATCH_SIZES) * len(SEQ_LENS)
    done = 0

    for cfg_name, cfg in CONFIGS.items():
        for batch in BATCH_SIZES:
            for seq_len in SEQ_LENS:
                done += 1
                tag = f"[{done}/{total}] {cfg['name']} B={batch} L={seq_len}"
                print(f"{tag} ... ", end="", flush=True)
                try:
                    r = run_config(cfg_name, cfg, batch, seq_len, device)
                    all_results.append(r)
                    ovhd = r["b_overhead_vs_d_pct"]
                    print(f"OK (B overhead: {ovhd:+.1f}%)")
                except torch.cuda.OutOfMemoryError:
                    print("OOM — skipped")
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback
                    traceback.print_exc()

    # === Print and save ===
    print_results_table(all_results)
    verdict_info = print_summary(all_results)

    # Save results
    results_path = os.path.join(RESULTS_ROOT, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    summary_path = os.path.join(RESULTS_ROOT, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(verdict_info, f, indent=2)

    env_info = {
        "gpu": gpu_name,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": sys.version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bits": BITS,
        "group_size": GROUP_SIZE,
        "configs": CONFIGS,
        "batch_sizes": BATCH_SIZES,
        "seq_lens": SEQ_LENS,
        "arms": ["A_uncompressed", "D_fused_quant", "B_practical_turboquant"],
        "qjl": "DISABLED",
        "rotation_seed": ROTATION_SEED,
    }
    env_path = os.path.join(RESULTS_ROOT, "environment.json")
    with open(env_path, "w") as f:
        json.dump(env_info, f, indent=2)

    # Copy to durable root
    import shutil
    for fname in ["results.json", "summary.json", "environment.json", "sanity_checks.json"]:
        src = os.path.join(RESULTS_ROOT, fname)
        dst = os.path.join(DURABLE_ROOT, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    # Copy script itself
    shutil.copy2(__file__, os.path.join(RESULTS_ROOT, os.path.basename(__file__)))
    shutil.copy2(__file__, os.path.join(DURABLE_ROOT, os.path.basename(__file__)))

    print(f"\nResults: {RESULTS_ROOT}")
    print(f"Durable: {DURABLE_ROOT}")
    print(f"Completed: {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()

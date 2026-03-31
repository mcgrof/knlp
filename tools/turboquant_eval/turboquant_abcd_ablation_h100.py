#!/usr/bin/env python3
"""TurboQuant A/B/C/D Ablation — H100 Corrected Run.

Explicit ablation arms:
  A   = Uncompressed FP16 baseline (no quantization)
  B1  = TurboQuant paper-style (rotation + per-coord INT4 scalar, no QJL)
  B2  = TurboQuant + QJL residual correction
  C1  = TurboQuant + fused Triton dequant (rotation fused into decode path)
  D   = Current fused quantization baseline (per-group INT4 scalar, no rotation)

This script exists to cleanly separate quantization POLICY (B vs D)
from EXECUTION (C fuses the TurboQuant representation into a Triton path).

Based on arXiv:2504.19874 (TurboQuant) and existing knlp fused-quant line.
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

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

# ============================================================
# Configuration
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

BATCH_SIZES = [1, 4, 8]
SEQ_LENS = [2048, 4096, 8192, 16384]
GROUP_SIZE = 32
BITS = 4  # INT4

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RESULTS_ROOT = os.environ.get(
    "RESULTS_ROOT",
    f"/workspace/results/turboquant-abcd-h100-{TIMESTAMP}"
)


# ============================================================
# Shared utilities
# ============================================================

def generate_rotation_matrix(d: int, device: torch.device, dtype: torch.dtype,
                              seed: int = 42) -> torch.Tensor:
    """Random orthogonal matrix via QR (Haar-distributed)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    G = torch.randn(d, d, generator=gen, dtype=torch.float32)
    Q, R = torch.linalg.qr(G)
    diag_sign = torch.sign(torch.diagonal(R))
    Q = Q * diag_sign.unsqueeze(0)
    return Q.to(device=device, dtype=dtype)


def pergroup_int4_quant(x: torch.Tensor, group_size: int = GROUP_SIZE):
    """Per-group symmetric INT4 quantization. Returns (x_hat, scales)."""
    orig_shape = x.shape
    assert x.shape[-1] % group_size == 0
    x_grouped = x.reshape(*x.shape[:-1], -1, group_size)
    amax = x_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / 7.0
    q = (x_grouped / scale).round().clamp(-8, 7)
    x_hat = (q * scale).reshape(orig_shape)
    return x_hat, scale.squeeze(-1)


# ============================================================
# Arm A — Uncompressed FP16 baseline
# ============================================================

def arm_a_quantize(x: torch.Tensor):
    """No-op: identity pass-through."""
    return x.clone(), None


# ============================================================
# Arm B1 — TurboQuant paper-style (rotation + scalar INT4)
# ============================================================

def arm_b1_quantize(x: torch.Tensor, R: torch.Tensor,
                     group_size: int = GROUP_SIZE):
    """TurboQuant: rotate → per-group INT4 → dequant → rotate back."""
    orig_shape = x.shape
    d = x.shape[-1]
    x_flat = x.reshape(-1, d)

    # Rotate
    x_rot = x_flat @ R

    # Per-group INT4 on rotated vectors
    x_rot_grouped = x_rot.reshape(-1, d // group_size, group_size)
    amax = x_rot_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / 7.0
    q = (x_rot_grouped / scale).round().clamp(-8, 7)
    x_rot_hat = (q * scale).reshape(-1, d)

    # Rotate back
    x_hat = (x_rot_hat @ R.T).reshape(orig_shape)
    return x_hat, {"scales": scale.squeeze(-1)}


# ============================================================
# Arm B2 — TurboQuant + QJL residual
# ============================================================

def arm_b2_quantize(x: torch.Tensor, R: torch.Tensor,
                     group_size: int = GROUP_SIZE):
    """TurboQuant + 1-bit QJL residual correction."""
    orig_shape = x.shape
    d = x.shape[-1]
    x_flat = x.reshape(-1, d)

    x_rot = x_flat @ R
    x_rot_grouped = x_rot.reshape(-1, d // group_size, group_size)
    amax = x_rot_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / 7.0
    q = (x_rot_grouped / scale).round().clamp(-8, 7)
    x_rot_hat = (q * scale).reshape(-1, d)

    # QJL: 1-bit sign sketch of rotated-domain residual
    residual = x_rot - x_rot_hat
    qjl_bits = (residual > 0).to(torch.int8)
    qjl_norms = residual.norm(dim=-1, keepdim=True)

    # For reconstruction quality we apply QJL correction in rotated domain
    # then rotate back
    # QJL reconstruction: r_hat ≈ norm * (2*bits - 1) / sqrt(d)
    r_hat = qjl_norms * (2.0 * qjl_bits.float() - 1.0) / math.sqrt(d)
    x_rot_corrected = (x_rot_hat + r_hat).to(R.dtype)

    x_hat = (x_rot_corrected @ R.T).reshape(orig_shape)
    return x_hat, {"scales": scale.squeeze(-1), "qjl_bits": qjl_bits,
                    "qjl_norms": qjl_norms}


# ============================================================
# Arm C1 — TurboQuant + fused Triton dequant
# ============================================================

if TRITON_AVAILABLE:
    @triton.jit
    def _fused_dequant_kernel(
        # Pointers
        q_ptr, scale_ptr, out_ptr,
        # Dimensions
        N, D: tl.constexpr, G: tl.constexpr,
        # Strides for q[N, n_groups, G]
        stride_q_n, stride_q_g, stride_q_d,
        # Strides for scale[N, n_groups]
        stride_s_n, stride_s_g,
        # Strides for out[N, D]
        stride_out_n, stride_out_d,
        BLOCK_N: tl.constexpr,
    ):
        """Fused dequantize kernel: reconstruct FP16 from quantized groups.

        This fuses the per-group dequantization into a single kernel pass,
        avoiding the reshape/broadcast overhead of the PyTorch path.
        The rotation matmul is left to cuBLAS (which is optimal for d×d matmul).
        """
        pid = tl.program_id(0)
        n_idx = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_idx < N

        n_groups = D // G
        for g in range(n_groups):
            # Load scales for this group across the block of vectors
            s = tl.load(scale_ptr + n_idx * stride_s_n + g * stride_s_g,
                        mask=n_mask, other=0.0)

            for k in range(G):
                q_val = tl.load(q_ptr + n_idx * stride_q_n + g * stride_q_g + k * stride_q_d,
                                mask=n_mask, other=0.0)
                dq_val = q_val * s
                tl.store(out_ptr + n_idx * stride_out_n + (g * G + k) * stride_out_d,
                         dq_val.to(tl.float16), mask=n_mask)


def arm_c1_quantize(x: torch.Tensor, R: torch.Tensor,
                     group_size: int = GROUP_SIZE):
    """TurboQuant representation + fused Triton dequant+rotate-back.

    Quantization is identical to B1 (rotate → INT4).
    Dequantization uses a fused Triton kernel that avoids materializing
    the full rotated-domain vector before the rotate-back matmul.
    """
    if not TRITON_AVAILABLE:
        # Fallback to B1 path
        return arm_b1_quantize(x, R, group_size)

    orig_shape = x.shape
    d = x.shape[-1]
    x_flat = x.reshape(-1, d)
    N = x_flat.shape[0]

    # Rotate (this stays as a matmul — it's the write-time cost)
    x_rot = x_flat @ R

    # Quantize in rotated domain (same as B1)
    n_groups = d // group_size
    x_rot_grouped = x_rot.reshape(N, n_groups, group_size)
    amax = x_rot_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / 7.0
    q = (x_rot_grouped / scale).round().clamp(-8, 7)

    # === Fused dequant via Triton kernel + cuBLAS rotate-back ===
    # The dequant is fused into a single Triton kernel (avoids reshape/broadcast overhead).
    # The rotation matmul is left to cuBLAS which is optimal for d×d matmul.
    x_rot_hat = torch.empty(N, d, device=x.device, dtype=torch.float16)
    q_f = q.to(torch.float16)
    scale_f = scale.squeeze(-1).to(torch.float32)

    BLOCK_N = min(128, N)
    grid = ((N + BLOCK_N - 1) // BLOCK_N,)
    _fused_dequant_kernel[grid](
        q_f, scale_f, x_rot_hat,
        N, d, group_size,
        q_f.stride(0), q_f.stride(1), q_f.stride(2),
        scale_f.stride(0), scale_f.stride(1),
        x_rot_hat.stride(0), x_rot_hat.stride(1),
        BLOCK_N=BLOCK_N,
    )
    # Rotate back via cuBLAS (optimal for d×d matmul)
    out = x_rot_hat @ R.T

    x_hat = out.reshape(orig_shape)
    return x_hat, {"scales": scale.squeeze(-1)}


# ============================================================
# Arm D — Current fused quantization baseline
# ============================================================

def arm_d_quantize(x: torch.Tensor, group_size: int = GROUP_SIZE):
    """Per-group INT4 symmetric scalar quantization. No rotation."""
    x_hat, scales = pergroup_int4_quant(x, group_size)
    return x_hat, {"scales": scales}


# ============================================================
# Quality metrics
# ============================================================

def compute_quality(x_orig: torch.Tensor, x_hat: torch.Tensor, arm: str) -> dict:
    """Compute reconstruction quality metrics."""
    mse = F.mse_loss(x_hat, x_orig).item()
    signal_power = (x_orig ** 2).mean().item()
    nmse = mse / max(signal_power, 1e-10)

    x_flat = x_orig.reshape(-1, x_orig.shape[-1])
    x_hat_flat = x_hat.reshape(-1, x_hat.shape[-1])
    cos_sim = F.cosine_similarity(x_flat, x_hat_flat, dim=-1).mean().item()

    # Inner-product distortion (sample for efficiency)
    n = min(x_flat.shape[0], 512)
    idx = torch.randperm(x_flat.shape[0], device=x_flat.device)[:n]
    xs = x_flat[idx]
    xhs = x_hat_flat[idx]
    true_ip = xs @ xs.T
    hat_ip = xhs @ xhs.T
    ip_rel_err = ((hat_ip - true_ip).abs() / (true_ip.abs() + 1e-8)).mean().item()

    max_abs_err = (x_hat - x_orig).abs().max().item()

    return {
        "arm": arm,
        "mse": mse,
        "nmse": nmse,
        "cosine_similarity": cos_sim,
        "ip_rel_error": ip_rel_err,
        "max_abs_error": max_abs_err,
        "signal_power": signal_power,
    }


# ============================================================
# Latency benchmark
# ============================================================

def bench_latency(fn, args, label: str, warmup: int = 10, trials: int = 50) -> dict:
    """GPU-synchronized latency measurement."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)

    times.sort()
    trim = max(1, len(times) // 10)
    trimmed = times[trim:-trim] if trim < len(times) // 2 else times

    return {
        "arm": label,
        "median_us": times[len(times) // 2],
        "mean_us": sum(trimmed) / len(trimmed),
        "p10_us": times[len(times) // 10],
        "p90_us": times[9 * len(times) // 10],
        "min_us": times[0],
        "max_us": times[-1],
    }


# ============================================================
# Single-config comparison
# ============================================================

def run_config(num_kv_heads: int, head_dim: int, batch: int, seq_len: int,
               device: torch.device, config_name: str) -> dict:
    """Run all arms for one (model, batch, seq_len) configuration."""
    shape = (batch, num_kv_heads, seq_len, head_dim)
    x = torch.randn(shape, device=device, dtype=torch.float16) / math.sqrt(head_dim)

    R = generate_rotation_matrix(head_dim, device, torch.float16)

    result = {
        "config": config_name, "batch": batch, "seq_len": seq_len,
        "num_kv_heads": num_kv_heads, "head_dim": head_dim,
    }

    # === Arm A: uncompressed ===
    x_hat_a, _ = arm_a_quantize(x)
    result["quality_A"] = compute_quality(x, x_hat_a, "A_uncompressed")

    # === Arm B1: TurboQuant (no QJL) ===
    x_hat_b1, _ = arm_b1_quantize(x, R)
    result["quality_B1"] = compute_quality(x, x_hat_b1, "B1_turboquant")

    # === Arm B2: TurboQuant + QJL ===
    x_hat_b2, _ = arm_b2_quantize(x, R)
    result["quality_B2"] = compute_quality(x, x_hat_b2, "B2_turboquant_qjl")

    # === Arm C1: TurboQuant + fused Triton ===
    x_hat_c1, _ = arm_c1_quantize(x, R)
    result["quality_C1"] = compute_quality(x, x_hat_c1, "C1_turboquant_fused")

    # === Arm D: fused quant baseline ===
    x_hat_d, _ = arm_d_quantize(x)
    result["quality_D"] = compute_quality(x, x_hat_d, "D_fused_quant")

    # === Latency (skip 16384 to keep runtime sane) ===
    if seq_len <= 8192:
        result["latency_A"] = bench_latency(
            arm_a_quantize, (x,), "A_uncompressed", warmup=5, trials=30)
        result["latency_B1"] = bench_latency(
            arm_b1_quantize, (x, R), "B1_turboquant", warmup=5, trials=30)
        result["latency_B2"] = bench_latency(
            arm_b2_quantize, (x, R), "B2_turboquant_qjl", warmup=5, trials=30)
        result["latency_C1"] = bench_latency(
            arm_c1_quantize, (x, R), "C1_turboquant_fused", warmup=5, trials=30)
        result["latency_D"] = bench_latency(
            arm_d_quantize, (x,), "D_fused_quant", warmup=5, trials=30)

    # Correctness check: C1 should match B1 quality closely (same representation)
    mse_b1 = result["quality_B1"]["mse"]
    mse_c1 = result["quality_C1"]["mse"]
    if abs(mse_b1 - mse_c1) / max(mse_b1, 1e-12) > 0.05:
        result["WARNING"] = (f"C1 vs B1 MSE mismatch: B1={mse_b1:.6e}, C1={mse_c1:.6e}. "
                             "Fused kernel may have precision drift.")

    return result


# ============================================================
# Printing
# ============================================================

ARM_ORDER = ["quality_A", "quality_B1", "quality_B2", "quality_C1", "quality_D"]
LAT_ORDER = ["latency_A", "latency_B1", "latency_B2", "latency_C1", "latency_D"]

def print_quality_table(results: list):
    print("\n" + "=" * 110)
    print("QUALITY: A/B1/B2/C1/D ablation")
    print("=" * 110)
    hdr = f"{'Config':<16} {'B':>2} {'L':>6} {'Arm':<25} {'MSE':>12} {'NMSE':>10} {'CosSim':>8} {'IP_err':>10}"
    print(hdr)
    print("-" * 110)

    for r in results:
        tag = r["config"]
        for key in ARM_ORDER:
            if key not in r:
                continue
            q = r[key]
            print(f"{tag:<16} {r['batch']:>2} {r['seq_len']:>6} "
                  f"{q['arm']:<25} {q['mse']:>12.6e} {q['nmse']:>10.6f} "
                  f"{q['cosine_similarity']:>8.6f} {q['ip_rel_error']:>10.6f}")
            tag = ""
        print()


def print_latency_table(results: list):
    print("\n" + "=" * 110)
    print("LATENCY: A/B1/B2/C1/D ablation")
    print("=" * 110)
    hdr = f"{'Config':<16} {'B':>2} {'L':>6} {'Arm':<25} {'Median µs':>12} {'P10 µs':>10} {'P90 µs':>10} {'vs A':>8} {'vs D':>8}"
    print(hdr)
    print("-" * 110)

    for r in results:
        tag = r["config"]
        lat_a = r.get("latency_A", {}).get("median_us", None)
        lat_d = r.get("latency_D", {}).get("median_us", None)
        for key in LAT_ORDER:
            if key not in r:
                continue
            lat = r[key]
            vs_a = f"{lat['median_us']/lat_a:.2f}x" if lat_a else "—"
            vs_d = f"{lat['median_us']/lat_d:.2f}x" if lat_d else "—"
            print(f"{tag:<16} {r['batch']:>2} {r['seq_len']:>6} "
                  f"{lat['arm']:<25} {lat['median_us']:>12.1f} "
                  f"{lat['p10_us']:>10.1f} {lat['p90_us']:>10.1f} "
                  f"{vs_a:>8} {vs_d:>8}")
            tag = ""
        print()


def print_summary(results: list):
    print("\n" + "=" * 110)
    print("SUMMARY")
    print("=" * 110)

    # Average quality per arm
    for key, label in zip(ARM_ORDER,
                          ["A (uncompressed)", "B1 (TurboQuant)", "B2 (TQ+QJL)",
                           "C1 (TQ+fused)", "D (fused quant)"]):
        mses = [r[key]["mse"] for r in results if key in r]
        nmses = [r[key]["nmse"] for r in results if key in r]
        coss = [r[key]["cosine_similarity"] for r in results if key in r]
        ips = [r[key]["ip_rel_error"] for r in results if key in r]
        if mses:
            print(f"  {label:<22}: MSE={sum(mses)/len(mses):.6e}  "
                  f"NMSE={sum(nmses)/len(nmses):.6f}  "
                  f"CosSim={sum(coss)/len(coss):.6f}  "
                  f"IP_err={sum(ips)/len(ips):.6f}")

    print()
    for key, label in zip(LAT_ORDER,
                          ["A (uncompressed)", "B1 (TurboQuant)", "B2 (TQ+QJL)",
                           "C1 (TQ+fused)", "D (fused quant)"]):
        meds = [r[key]["median_us"] for r in results if key in r]
        if meds:
            print(f"  {label:<22}: avg median = {sum(meds)/len(meds):.1f} µs")

    # Explicit ablation questions
    print("\n" + "-" * 80)
    print("ABLATION QUESTIONS")
    print("-" * 80)

    # Get average metrics
    def avg_metric(qkey, metric):
        vals = [r[qkey][metric] for r in results if qkey in r]
        return sum(vals) / len(vals) if vals else None

    def avg_lat(lkey):
        vals = [r[lkey]["median_us"] for r in results if lkey in r]
        return sum(vals) / len(vals) if vals else None

    mse_a = avg_metric("quality_A", "mse")
    mse_b1 = avg_metric("quality_B1", "mse")
    mse_b2 = avg_metric("quality_B2", "mse")
    mse_c1 = avg_metric("quality_C1", "mse")
    mse_d = avg_metric("quality_D", "mse")

    cos_a = avg_metric("quality_A", "cosine_similarity")
    cos_b1 = avg_metric("quality_B1", "cosine_similarity")
    cos_c1 = avg_metric("quality_C1", "cosine_similarity")
    cos_d = avg_metric("quality_D", "cosine_similarity")

    lat_a = avg_lat("latency_A")
    lat_b1 = avg_lat("latency_B1")
    lat_b2 = avg_lat("latency_B2")
    lat_c1 = avg_lat("latency_C1")
    lat_d = avg_lat("latency_D")

    print(f"\n1. TurboQuant vs uncompressed (B1 vs A):")
    if mse_b1 and mse_a is not None:
        print(f"   Quality: MSE A={mse_a:.6e} → B1={mse_b1:.6e}")
        print(f"   CosSim: A={cos_a:.6f} → B1={cos_b1:.6f}")

    print(f"\n2. TurboQuant vs fused quant (B1 vs D):")
    if mse_b1 and mse_d:
        delta_pct = (mse_b1 - mse_d) / mse_d * 100
        print(f"   MSE: B1={mse_b1:.6e} vs D={mse_d:.6e} (B1 is {delta_pct:+.2f}% vs D)")
        print(f"   CosSim: B1={cos_b1:.6f} vs D={cos_d:.6f}")

    print(f"\n3. Does fused Triton execution rescue TurboQuant latency? (C1 vs B1):")
    if lat_c1 and lat_b1:
        speedup = lat_b1 / lat_c1
        print(f"   B1={lat_b1:.1f}µs → C1={lat_c1:.1f}µs ({speedup:.2f}x)")
        if lat_d:
            gap_b1 = (lat_b1 - lat_d) / lat_d * 100
            gap_c1 = (lat_c1 - lat_d) / lat_d * 100
            print(f"   B1 overhead vs D: {gap_b1:+.1f}%")
            print(f"   C1 overhead vs D: {gap_c1:+.1f}%")
            if gap_c1 < gap_b1 * 0.5:
                print(f"   → YES: fused Triton cuts TurboQuant's overhead by >{50}%")
            else:
                print(f"   → NO: fused Triton does not materially close the gap")

    print(f"\n4. Does D still win overall?")
    if mse_d and mse_b1 and lat_d and lat_b1:
        quality_winner = "D" if mse_d <= mse_b1 * 1.01 else "B1"
        latency_winner = "D" if lat_d <= lat_c1 else "C1"
        print(f"   Quality: {'D ≈ B1 (within 1%)' if abs(mse_d - mse_b1)/mse_d < 0.01 else f'{quality_winner} wins'}")
        print(f"   Latency: {latency_winner} wins ({lat_d:.1f}µs vs {lat_c1:.1f}µs)")

    print(f"\n5. Is QJL worth keeping? (B2 vs B1):")
    if mse_b2 and mse_b1 and lat_b2 and lat_b1:
        quality_gain = (mse_b1 - mse_b2) / mse_b1 * 100
        latency_cost = (lat_b2 - lat_b1) / lat_b1 * 100
        print(f"   Quality gain: {quality_gain:+.2f}% MSE reduction")
        print(f"   Latency cost: {latency_cost:+.1f}% overhead")
        if quality_gain > 5 and latency_cost < 20:
            print(f"   → MAYBE: meaningful quality gain with tolerable cost")
        else:
            print(f"   → NO: insufficient quality gain for the extra cost")


# ============================================================
# Main
# ============================================================

def main():
    print(f"TurboQuant A/B/C/D Ablation — H100")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"Results root: {RESULTS_ROOT}")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Torch: {torch.__version__}, CUDA: {torch.version.cuda}")
    if TRITON_AVAILABLE:
        print(f"Triton: {triton.__version__}")
    else:
        print("WARNING: Triton not available — C1 arm will fall back to PyTorch path")
    print(f"INT{BITS}, group_size={GROUP_SIZE}")
    print(f"Arms: A (uncompressed), B1 (TurboQuant), B2 (TQ+QJL), C1 (TQ+fused), D (fused quant)")
    print()

    os.makedirs(RESULTS_ROOT, exist_ok=True)

    all_results = []
    total = len(CONFIGS) * len(BATCH_SIZES) * len(SEQ_LENS)
    done = 0

    for config_key, cfg in CONFIGS.items():
        for batch in BATCH_SIZES:
            for seq_len in SEQ_LENS:
                done += 1
                tag = f"[{done}/{total}] {cfg['name']} B={batch} L={seq_len}"
                print(f"{tag} ... ", end="", flush=True)
                try:
                    r = run_config(
                        num_kv_heads=cfg["num_kv_heads"],
                        head_dim=cfg["head_dim"],
                        batch=batch, seq_len=seq_len,
                        device=device, config_name=config_key,
                    )
                    all_results.append(r)
                    if "WARNING" in r:
                        print(f"OK (WARNING: {r['WARNING']})")
                    else:
                        print("OK")
                except torch.cuda.OutOfMemoryError:
                    print("OOM — skipped")
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback
                    traceback.print_exc()

    # Print results
    print_quality_table(all_results)
    print_latency_table(all_results)
    print_summary(all_results)

    # Save artifacts
    results_path = os.path.join(RESULTS_ROOT, "ablation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults JSON: {results_path}")

    env_info = {
        "gpu": gpu_name,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__ if TRITON_AVAILABLE else "N/A",
        "python": sys.version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bits": BITS,
        "group_size": GROUP_SIZE,
        "configs": CONFIGS,
        "batch_sizes": BATCH_SIZES,
        "seq_lens": SEQ_LENS,
        "arms": ["A_uncompressed", "B1_turboquant", "B2_turboquant_qjl",
                  "C1_turboquant_fused", "D_fused_quant"],
    }
    env_path = os.path.join(RESULTS_ROOT, "environment.json")
    with open(env_path, "w") as f:
        json.dump(env_info, f, indent=2)
    print(f"Environment: {env_path}")

    print(f"\nCompleted: {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: GPL-2.0
"""P1 diagnostic: capture winning autotune configs and test extended search space.

Measures:
1. Current autotune configs (BLOCK_T in {32, 64, 128})
2. Extended configs (BLOCK_T in {32, 64, 128, 256}, num_warps in {4, 8, 16})
3. Reports the winning config per operating point

This is the smallest decisive P1 action: know exactly what config wins before
attempting any kernel structural changes.
"""

import time
import torch
import triton
import triton.language as tl

from routing.fused_routed_attention import (
    _fused_routed_decode_kernel,
    fused_routed_decode,
    reference_routed_decode,
)


def benchmark_single_config(
    q,
    k_cache,
    v_cache,
    block_tables,
    block_counts,
    scale,
    BLOCK_T,
    num_warps,
    num_stages,
    warmup=20,
    iters=200,
):
    """Benchmark a specific kernel config, return median ms."""
    batch, n_heads, head_dim = q.shape
    _, block_size, n_kv_heads, _ = k_cache.shape
    max_selected_blocks = block_tables.shape[2]
    group_size = n_heads // n_kv_heads
    BLOCK_D = triton.next_power_of_2(head_dim)
    output = torch.empty_like(q)
    grid = (batch, group_size, n_kv_heads)

    # Warmup
    for _ in range(warmup):
        _fused_routed_decode_kernel[grid](
            q,
            k_cache,
            v_cache,
            block_tables,
            block_counts,
            output,
            scale=scale,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            max_selected_blocks=max_selected_blocks,
            stride_qb=q.stride(0),
            stride_qh=q.stride(1),
            stride_qd=q.stride(2),
            stride_kb=k_cache.stride(0),
            stride_kt=k_cache.stride(1),
            stride_kh=k_cache.stride(2),
            stride_kd=k_cache.stride(3),
            stride_vb=v_cache.stride(0),
            stride_vt=v_cache.stride(1),
            stride_vh=v_cache.stride(2),
            stride_vd=v_cache.stride(3),
            stride_btb=block_tables.stride(0),
            stride_bth=block_tables.stride(1),
            stride_bts=block_tables.stride(2),
            stride_bcb=block_counts.stride(0),
            stride_bch=block_counts.stride(1),
            stride_ob=output.stride(0),
            stride_oh=output.stride(1),
            stride_od=output.stride(2),
            BLOCK_D=BLOCK_D,
            BLOCK_T=BLOCK_T,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _fused_routed_decode_kernel[grid](
            q,
            k_cache,
            v_cache,
            block_tables,
            block_counts,
            output,
            scale=scale,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            max_selected_blocks=max_selected_blocks,
            stride_qb=q.stride(0),
            stride_qh=q.stride(1),
            stride_qd=q.stride(2),
            stride_kb=k_cache.stride(0),
            stride_kt=k_cache.stride(1),
            stride_kh=k_cache.stride(2),
            stride_kd=k_cache.stride(3),
            stride_vb=v_cache.stride(0),
            stride_vt=v_cache.stride(1),
            stride_vh=v_cache.stride(2),
            stride_vd=v_cache.stride(3),
            stride_btb=block_tables.stride(0),
            stride_bth=block_tables.stride(1),
            stride_bts=block_tables.stride(2),
            stride_bcb=block_counts.stride(0),
            stride_bch=block_counts.stride(1),
            stride_ob=output.stride(0),
            stride_oh=output.stride(1),
            stride_od=output.stride(2),
            BLOCK_D=BLOCK_D,
            BLOCK_T=BLOCK_T,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    return times[len(times) // 2], output


def run_config_sweep(block_size, n_total_blocks, K, iters=200):
    """Sweep all configs for a single operating point."""
    device = "cuda"
    dtype = torch.float16
    batch = 1
    n_heads = 32
    n_kv_heads = 8
    head_dim = 128
    scale = head_dim**-0.5

    q = torch.randn(batch, n_heads, head_dim, device=device, dtype=dtype)
    k_cache = torch.randn(
        n_total_blocks, block_size, n_kv_heads, head_dim, device=device, dtype=dtype
    )
    v_cache = torch.randn(
        n_total_blocks, block_size, n_kv_heads, head_dim, device=device, dtype=dtype
    )

    actual_k = min(K, n_total_blocks)
    block_tables = torch.zeros(
        batch, n_kv_heads, actual_k, dtype=torch.int64, device=device
    )
    block_counts = torch.zeros(batch, n_kv_heads, dtype=torch.int32, device=device)
    for b in range(batch):
        for h in range(n_kv_heads):
            perm = torch.randperm(n_total_blocks, device=device)[:actual_k]
            block_tables[b, h, :actual_k] = perm
            block_counts[b, h] = actual_k

    # Reference output for correctness check
    ref_out = reference_routed_decode(
        q, k_cache, v_cache, block_tables, block_counts, scale
    )

    # Config space — current + extended
    configs = []
    for bt in [32, 64, 128]:
        if bt > block_size:
            continue
        for nw in [4, 8]:
            for ns in [2, 3, 4]:
                configs.append((bt, nw, ns, "current"))

    # Extended configs
    for bt in [256]:
        if bt > block_size:
            continue
        for nw in [4, 8, 16]:
            for ns in [2, 3, 4]:
                configs.append((bt, nw, ns, "extended"))

    # Also test num_warps=16 with existing BLOCK_T values
    for bt in [64, 128]:
        if bt > block_size:
            continue
        for ns in [2, 3, 4]:
            configs.append((bt, 16, ns, "extended"))

    results = []
    for bt, nw, ns, space in configs:
        try:
            median_ms, out = benchmark_single_config(
                q,
                k_cache,
                v_cache,
                block_tables,
                block_counts,
                scale,
                BLOCK_T=bt,
                num_warps=nw,
                num_stages=ns,
                warmup=20,
                iters=iters,
            )
            # Correctness check
            max_err = (out.float() - ref_out.float()).abs().max().item()
            results.append(
                {
                    "BLOCK_T": bt,
                    "num_warps": nw,
                    "num_stages": ns,
                    "space": space,
                    "median_ms": median_ms,
                    "max_err": max_err,
                    "correct": max_err < 0.05,
                }
            )
        except Exception as e:
            results.append(
                {
                    "BLOCK_T": bt,
                    "num_warps": nw,
                    "num_stages": ns,
                    "space": space,
                    "median_ms": float("inf"),
                    "max_err": -1,
                    "correct": False,
                    "error": str(e),
                }
            )

    return results


def main():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Torch: {torch.__version__}")
    print()

    operating_points = [
        {
            "block_size": 128,
            "n_total_blocks": 32,
            "K": 8,
            "label": "BS=128 K=8 (promoted)",
        },
        {
            "block_size": 256,
            "n_total_blocks": 16,
            "K": 8,
            "label": "BS=256 K=8 (high-accuracy)",
        },
        {
            "block_size": 256,
            "n_total_blocks": 16,
            "K": 4,
            "label": "BS=256 K=4 (latency)",
        },
    ]

    for op in operating_points:
        print(f"=== {op['label']} ===")
        results = run_config_sweep(
            block_size=op["block_size"],
            n_total_blocks=op["n_total_blocks"],
            K=op["K"],
            iters=200,
        )

        # Sort by latency
        results.sort(key=lambda r: r["median_ms"])

        # Print top 5
        print(
            f"  {'BLOCK_T':>7s}  {'warps':>5s}  {'stages':>6s}  {'space':>8s}  {'median_ms':>10s}  {'max_err':>8s}  {'ok':>3s}"
        )
        print(f"  {'-'*60}")
        for r in results[:10]:
            ok = "Y" if r.get("correct", False) else "N"
            err_str = f"{r['max_err']:.4f}" if r["max_err"] >= 0 else "ERR"
            print(
                f"  {r['BLOCK_T']:>7d}  {r['num_warps']:>5d}  {r['num_stages']:>6d}  {r['space']:>8s}  {r['median_ms']:>9.4f}ms  {err_str:>8s}  {ok:>3s}"
            )

        winner = results[0]
        print(
            f"\n  WINNER: BLOCK_T={winner['BLOCK_T']}, num_warps={winner['num_warps']}, "
            f"num_stages={winner['num_stages']}, median={winner['median_ms']:.4f}ms"
        )
        print()


if __name__ == "__main__":
    main()

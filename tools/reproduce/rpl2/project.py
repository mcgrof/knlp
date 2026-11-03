#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Project restore bandwidth, NVMe count and cost per GPU, against GPU price.

Break-even restore bandwidth: the rate at which reading a prefix's KV from
storage matches recomputing it with prefill. Prefill is taken as linear
FLOPs only, 2 x parameters per token, at an assumed MFU; attention FLOPs
are ignored, which understates recompute cost at long context and so
overstates the bandwidth storage needs. KV is BF16 unless stated.

Prices are October 2026 market estimates (NVIDIA publishes no GPU list
price; drive prices are retail and reseller listings), and the host CPU
cost per GB is the range measured on a desktop CPU for page-cache reads
into pinned memory plus a host-to-device copy. Edit the tables below to
match your own hardware and prices.

  python3 tools/reproduce/rpl2/project.py
"""

MFU = 0.5  # assumed model FLOPs utilization during prefill
GiB = 1 << 30

models = {  # layers, kv_heads, head_dim, params
    "Llama-3.1-8B": (32, 8, 128, 8.0e9),
    "Llama-3.1-70B": (80, 8, 128, 70.6e9),
}
gpus = {  # dense TFLOPS by precision, price range USD
    "H100": ({"BF16": 989, "FP8": 1979}, (25_000, 35_000)),
    "B200": ({"BF16": 2250, "FP8": 4500, "FP4": 9000}, (30_000, 50_000)),
    "B300": ({"FP8": 4500, "FP4": 13500}, (37_500, 53_000)),
}
# drive: delivered GB/s (85% of rated sequential read), price USD
drives = {
    "consumer Gen5 2 TB": (0.85 * 14.0, 386),
    "datacenter Gen5 7.68 TB": (0.85 * 14.0, 5_198),
}
PCIE5_X16 = 63.0  # GB/s per direction, 32 GT/s x16, 128b/130b
# measured on a desktop host: page-cache hit + host-to-device copy, and DMA-BUF;
# core-seconds per GB
CPU_PER_GB = {"pool": (0.39, 0.94), "inline": (0.11, 0.21), "dmabuf": (0.004, 0.04)}

print(f"MFU {MFU}; KV BF16; break-even = restore as fast as prefill recompute\n")
print(
    "| GPU | prefill | model | KV/token | prefill tok/s | break-even GB/s | 4x faster GB/s "
    "| Gen5 drives for 4x | drive cost / GPU price (consumer; datacenter) "
    "| host cores at 4x, page cache (inline-pool) |"
)
print("|---" * 10 + "|")
for g, (peaks, (lo, hi)) in gpus.items():
    for prec, tf in peaks.items():
        for m, (L, H, D, P) in models.items():
            kv = L * H * D * 2 * 2  # bytes per token, K and V, BF16
            tok_s = MFU * tf * 1e12 / (2 * P)
            be = tok_s * kv / 1e9
            need = 4 * be
            capped = min(need, PCIE5_X16)
            n = -(-capped // drives["consumer Gen5 2 TB"][0])
            c_cons = n * drives["consumer Gen5 2 TB"][1]
            c_dc = n * drives["datacenter Gen5 7.68 TB"][1]
            cores = (CPU_PER_GB["inline"][0] * capped, CPU_PER_GB["pool"][1] * capped)
            cap = " (x16 cap)" if need > PCIE5_X16 else ""
            print(
                f"| {g} | {prec} | {m} | {kv // 1024} KiB | {tok_s:,.0f} | {be:.1f} | {need:.1f}{cap} "
                f"| {n:.0f} | {c_cons / hi:.1%}-{c_cons / lo:.1%}; {c_dc / hi:.0%}-{c_dc / lo:.0%} "
                f"| {cores[0]:.0f}-{cores[1]:.0f} |"
            )

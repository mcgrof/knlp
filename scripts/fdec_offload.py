"""Brick B: active-decode SSD-offload lower-bound, from MEASURED prune NVMe bandwidth.

The question before building any offload pipeline: can the I/O substrate even approach
the bytes/token that active decode needs? Per generated token, V-only offload must read
the WHOLE V cache from SSD. We compare that against the measured read bandwidth of
prune's Samsung 9100 PRO (PCIe5) and against H100 HBM, and compute the fetch fraction
that query-aware paging (Brick C) would need to be HBM-decode-competitive.

NVMe read bandwidth MEASURED on prune (/data, Samsung 9100 PRO, fio O_DIRECT, qd256):
"""

# measured GB/s (fio --minimal field 7, KB/s -> GB/s)
NVME = {
    "seq 1M": 14.44,
    "rand 4k": 8.79,
    "rand 64k": 11.59,
    "rand 256k": 12.32,
    "rand 1M": 13.71,
}
HBM_GBPS = 3350.0  # H100 SXM

# Qwen2.5-7B: 28 layers, 4 KV heads, head_dim 128. FP8 (1 byte) V side per context token:
V8_BYTES_PER_CTXTOK = 28 * 4 * 128 * 1  # = 14336 bytes per context token (V only)
KV8_BYTES_PER_CTXTOK = 2 * V8_BYTES_PER_CTXTOK  # K+V both FP8


def full_read_gb(ctx, both=False):
    b = (KV8_BYTES_PER_CTXTOK if both else V8_BYTES_PER_CTXTOK) * ctx
    return b / 1e9


def ms(gb, gbps):
    return gb / gbps * 1000.0  # GB / (GB/s) -> ms


def main():
    print("Brick B: active-decode SSD-offload lower bound (Qwen2.5-7B, FP8 V cache)")
    print("Measured prune NVMe (Samsung 9100 PRO, PCIe5, O_DIRECT qd256):")
    for k, v in NVME.items():
        print(f"  {k:<10} {v:>6.1f} GB/s")
    print(f"  {'HBM (H100)':<10} {HBM_GBPS:>6.0f} GB/s\n")

    # best (seq) and realistic (rand 64k page) SSD numbers
    ssd_best = NVME["seq 1M"]
    ssd_page = NVME["rand 64k"]
    print(
        f"{'ctx':>8}{'full V GB/tok':>15}{'HBM ms':>9}{'SSD-seq ms':>12}"
        f"{'SSD-64k ms':>12}{'slowdown':>10}"
    )
    print("-" * 66)
    for ctx in (32768, 131072, 262144):
        gb = full_read_gb(ctx)
        h = ms(gb, HBM_GBPS)
        s_best = ms(gb, ssd_best)
        s_page = ms(gb, ssd_page)
        print(
            f"{ctx:>8}{gb:>15.2f}{h:>9.2f}{s_best:>12.1f}{s_page:>12.1f}"
            f"{s_page/h:>9.0f}x"
        )

    print(
        "\nVerdict: reading the FULL V cache from SSD every token is "
        f"{ms(full_read_gb(131072), ssd_page)/ms(full_read_gb(131072), HBM_GBPS):.0f}x "
        "slower than HBM at 128K -- full-V active decode is infeasible (confirms the "
        "decode paper). The PCIe5 random penalty is modest (rand-64k 11.6 vs seq 14.4 "
        "GB/s), so the wall is ABSOLUTE bandwidth, not fragmentation."
    )

    # what fetch fraction makes query-aware paging (Brick C) viable?
    print("\nBrick C target: query-aware V-page fetch fraction to hit a latency budget")
    print(f"{'ctx':>8}{'budget ms':>11}{'bytes/tok':>12}{'frac of V':>11}")
    print("-" * 42)
    for ctx in (32768, 131072, 262144):
        gb = full_read_gb(ctx)
        for budget_ms in (3.0,):  # ~ the int8+K8V8 decode roofline at long ctx
            bytes_budget = ssd_page * budget_ms / 1000.0  # GB
            frac = bytes_budget / gb
            print(
                f"{ctx:>8}{budget_ms:>11.1f}{bytes_budget*1000:>10.1f}MB"
                f"{frac*100:>10.2f}%"
            )
    print(
        "\nSo query-aware paging is viable ONLY if attention mass concentrates in "
        "<~1-2% of V pages per token at long context (Quest-style). That sparsity is "
        "the Brick C experiment: measure what fraction of pages carries the mass."
    )


if __name__ == "__main__":
    main()

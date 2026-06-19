"""Deadline-aware event model: residency + prefetch prediction + measured SSD timing -> TPOT.

The byte-budget brick asked "do the fetched bytes fit?". This asks the question that actually
decides viability: with a persistent HBM residency, a real (imperfect) prefetcher, and the
MEASURED layer-serial NVMe timing, what is the per-token stall and the time-per-output-token
(TPOT)? It replays the per-(token,layer) V-page demand trace from fdec_residency --dump-trace
(no GPU) through a layer-serial decode pipeline:

  For each decode token, layers 0..L-1 run serially. Per layer:
   * residency miss set = demand pages not in the per-layer LRU cache (already measured).
   * a prefetcher tries to issue those reads ONE LAYER AHEAD (overlap with the prior layer's
     compute). Two signals, unioned: TEMPORAL (this layer's pages from the previous token) and
     SCOUT (this layer's pages predicted from the one-layer-stale residual; measured recall).
     A page the prefetcher flagged is HIDDEN behind compute; an unflagged miss is SYNCHRONOUS
     -> it stalls the layer for its read latency.
   * read time uses the measured 64KB-random LAYER-SERIAL burst curve (iodepth -> GB/s, us),
     because decode issues only ~budget reads per layer, not a deep queue.

It reports, across tokens, the p50/p90/p99 of: new SSD bytes/token, synchronous (unhidden)
stall/token, and TPOT = compute_floor + stall. compute_floor is the int8-weights + resident-KV
HBM roofline. The prefetch hit-rates come from fdec_predictor (temporal) and fdec_scout
(Scout layer-ahead); we sweep them as the achievable-vs-oracle band. No new GPU run.
"""

import argparse
import json

# Measured Samsung 9100 PRO PCIe5 layer-serial burst (64KB random, O_DIRECT), from
# BRICK-B: iodepth (reads in flight) -> (effective GB/s, mean per-IO latency us).
BURST = [
    (4, 2.85, 83),
    (8, 4.82, 101),
    (12, 6.22, 119),
    (16, 7.14, 138),
    (20, 8.29, 150),
    (32, 8.76, 229),
    (64, 10.45, 389),
]
HBM_GBPS = 3350.0  # H100 HBM3


def burst_bw(depth):
    # effective GB/s at a given number of in-flight reads (clamp + linear interp)
    if depth <= BURST[0][0]:
        return BURST[0][1]
    if depth >= BURST[-1][0]:
        return BURST[-1][1]
    for (d0, b0, _), (d1, b1, _) in zip(BURST, BURST[1:]):
        if d0 <= depth <= d1:
            return b0 + (b1 - b0) * (depth - d0) / (d1 - d0)
    return BURST[-1][1]


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def compute_floor_ms(ctx, n_layers, page_kb, cache_pages, kv_heads=4, head_dim=128):
    # int8-weights 7B + resident KV (full FP8 keys + resident V pages) read once/token / HBM
    w = 7.0e9  # int8 weights ~7 GB
    k_bytes = n_layers * kv_heads * head_dim * 1 * ctx  # full FP8 keys resident
    v_bytes = n_layers * cache_pages * page_kb * 1024  # resident V pages
    return (w + k_bytes + v_bytes) / (HBM_GBPS * 1e9) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trace", required=True, help="JSON from fdec_residency --dump-trace"
    )
    ap.add_argument(
        "--capacity", default="1", help="cache mult key in miss_matrix (1/2/4)"
    )
    ap.add_argument(
        "--prefetch",
        default="0.76,0.92,0.96",
        help="union prefetch hit-rates to sweep: 0.76=temporal-only (fdec_predictor), "
        "0.92=Scout layer-ahead @1.5x overfetch (fdec_scout), 0.96=union estimate",
    )
    ap.add_argument(
        "--overfetch",
        type=float,
        default=1.5,
        help="prefetch fetches overfetch x budget candidate pages (Scout recall cost)",
    )
    args = ap.parse_args()
    prefetch_sweep = [float(x) for x in args.prefetch.split(",")]
    T = json.load(open(args.trace))
    nL = T["n_layers"]
    kb = T["page_kb"]
    ctx = T["ctx"]
    B = T["B"]
    mult = int(args.capacity)
    cache_pages = mult * B
    miss_mat = T["miss_matrix"][args.capacity]  # per-token {layer: miss_count}
    layers = [str(L) for L in T["layers"]]

    floor = compute_floor_ms(ctx, nL, kb, cache_pages)
    compute_per_layer = floor / nL
    page_bytes = kb * 1024

    # SSD bytes/token is prefetch-independent (residency misses are the same set)
    ssd_mb = []
    for s in range(1, len(miss_mat)):
        ssd_mb.append(sum(miss_mat[s].values()) * page_bytes / 1024 / 1024)

    def sim(prefetch):
        stall_ms, tpot_ms = [], []
        for s in range(1, len(miss_mat)):  # skip cold first token
            row = miss_mat[s]
            tok_stall = 0.0
            for L in layers:
                miss = row.get(L, 0)
                if miss == 0:
                    continue
                # of the miss pages, the prefetcher flags a fraction one layer ahead
                # (HIDDEN behind the prior layer's compute) and the rest are SYNCHRONOUS.
                hidden = miss * prefetch
                sync = miss - hidden
                if sync > 0:  # shallow queue, full stall
                    tok_stall += (
                        sync * page_bytes / (burst_bw(max(1, sync)) * 1e9) * 1e3
                    )
                if hidden > 0:  # overlapped; stall only overflow past 1-layer compute
                    rt = hidden * page_bytes / (burst_bw(max(1, hidden)) * 1e9) * 1e3
                    tok_stall += max(0.0, rt - compute_per_layer)
            stall_ms.append(tok_stall)
            tpot_ms.append(floor + tok_stall)
        return stall_ms, tpot_ms

    budget_ms = 3.0
    print(
        f"[deadline] ctx={ctx} B={B} cache={mult}xB({cache_pages}pg/layer) page={kb}KB"
    )
    print(
        f"compute floor (int8 W + resident KV / HBM): {floor:.2f} ms/token "
        f"({compute_per_layer*1000:.0f} us/layer) | target budget {budget_ms:.1f} ms"
    )
    print(
        f"new SSD MB/token (prefetch-independent): "
        f"p50={pct(ssd_mb,50):.2f} p90={pct(ssd_mb,90):.2f} p99={pct(ssd_mb,99):.2f}\n"
    )
    print(
        f"{'prefetch':>9}{'stall p50':>11}{'stall p90':>11}{'stall p99':>11}"
        f"{'TPOT p50':>10}{'TPOT p99':>10}{'over3ms':>9}"
    )
    print("-" * 71)
    for pf in prefetch_sweep:
        st, tp = sim(pf)
        over = sum(1 for t in tp if t > budget_ms) / max(1, len(tp))
        print(
            f"{pf:>9.2f}{pct(st,50):>11.2f}{pct(st,90):>11.2f}{pct(st,99):>11.2f}"
            f"{pct(tp,50):>10.2f}{pct(tp,99):>10.2f}{over*100:>8.0f}%"
        )


if __name__ == "__main__":
    main()

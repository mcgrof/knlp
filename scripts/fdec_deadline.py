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
    # --- charged-model knobs (ChatGPT-Pro integration-gate audit) ---
    # The optimistic run ignores: overfetch false-positive bytes, the SSD->host->GPU PCIe
    # hop, the sparse-V gather/page-table kernel cost, the selector/q_hat-projection cost,
    # and prefetch/sync queue contention. --charged turns all of these on. They matter
    # because they are PER-LAYER and decode is 28 layers serial, so a fixed us/layer cost
    # is multiplied by nL on the critical path.
    ap.add_argument(
        "--charged", action="store_true", help="enable all overhead charges"
    )
    ap.add_argument(
        "--pcie-gbps",
        type=float,
        default=20.0,
        help="effective SSD->host->GPU PCIe bandwidth (Gen5 x16 host-bounce); read bw "
        "becomes min(nvme_burst, this)",
    )
    ap.add_argument(
        "--pcie-lat-us",
        type=float,
        default=12.0,
        help="fixed PCIe/host-bounce latency per SYNCHRONOUS read batch (per layer)",
    )
    ap.add_argument(
        "--kernel-tax-us",
        type=float,
        default=10.0,
        help="per-layer sparse-V gather + page-table lookup overhead (critical path)",
    )
    ap.add_argument(
        "--selector-tax-us",
        type=float,
        default=5.0,
        help="per-layer exact-QK page-score reduction + Scout q_hat projection",
    )
    args = ap.parse_args()
    prefetch_sweep = [float(x) for x in args.prefetch.split(",")]
    charged = args.charged
    pcie_bw = args.pcie_gbps if charged else 1e9
    pcie_lat = args.pcie_lat_us / 1e3 if charged else 0.0
    layer_tax = (args.kernel_tax_us + args.selector_tax_us) / 1e3 if charged else 0.0
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
    # charged compute floor: add per-layer kernel+selector tax (x nL, serial critical path)
    floor_c = floor + layer_tax * nL

    def eff_bw(depth):  # NVMe burst, capped by the PCIe host-bounce when charged
        return min(burst_bw(depth), pcie_bw)

    # SSD bytes/token: optimistic = residency misses; charged adds overfetch false positives
    fp_per_layer_factor = max(
        0.0, args.overfetch - 1.0
    )  # extra candidate pages vs budget

    def ssd_bytes_token(row, charge_fp):
        b = 0.0
        for L in layers:
            m = row.get(L, 0)
            if m == 0:
                continue
            fp = (
                fp_per_layer_factor * B * (m / B) if charge_fp else 0.0
            )  # FP reads scale with the per-layer demand miss fraction (m/B)
            b += (m + fp) * page_bytes
        return b / 1024 / 1024

    def sim(prefetch):
        cpl = compute_per_layer + layer_tax  # compute window per layer (incl tax)
        flr = floor_c
        stall_ms, tpot_ms = [], []
        for s in range(1, len(miss_mat)):  # skip cold first token
            row = miss_mat[s]
            tok_stall = 0.0
            for L in layers:
                miss = row.get(L, 0)
                if miss == 0:
                    continue
                # prefetcher flags a fraction one layer ahead (HIDDEN, overlapped); the rest
                # are SYNCHRONOUS (stall). Charged: hidden ALSO drags overfetch false
                # positives, and every read pays the PCIe bw cap + a sync-batch latency.
                hidden_useful = miss * prefetch
                sync = miss - hidden_useful
                fp = fp_per_layer_factor * miss if charged else 0.0
                hidden = hidden_useful + fp
                if sync > 0:  # shallow queue + fixed PCIe/host latency, full stall
                    rt = sync * page_bytes / (eff_bw(max(1, sync)) * 1e9) * 1e3
                    tok_stall += rt + pcie_lat
                if (
                    hidden > 0
                ):  # overlapped; only the overflow past the compute window stalls
                    rt = hidden * page_bytes / (eff_bw(max(1, hidden)) * 1e9) * 1e3
                    tok_stall += max(0.0, rt - cpl)
            stall_ms.append(tok_stall)
            tpot_ms.append(flr + tok_stall)
        return stall_ms, tpot_ms

    ssd_mb = [ssd_bytes_token(miss_mat[s], charged) for s in range(1, len(miss_mat))]
    budget_ms = 3.0
    mode = "CHARGED" if charged else "optimistic"
    print(
        f"[deadline:{mode}] ctx={ctx} B={B} cache={mult}xB({cache_pages}pg/layer) page={kb}KB"
    )
    if charged:
        print(
            f"charges: overfetch={args.overfetch}x FP bytes, PCIe cap {args.pcie_gbps}GB/s "
            f"+ {args.pcie_lat_us}us/sync-batch, kernel {args.kernel_tax_us}us + selector "
            f"{args.selector_tax_us}us per layer (x{nL} serial)"
        )
    print(
        f"compute floor: {floor_c:.2f} ms/token "
        f"({(compute_per_layer+layer_tax)*1000:.0f} us/layer) | target budget {budget_ms:.1f} ms"
    )
    print(
        f"new SSD MB/token: "
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

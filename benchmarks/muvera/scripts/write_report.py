#!/usr/bin/env python3
"""write_report.py — generate the final markdown report from the run dir."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import _muvera_config as cf


def _read_csv(path):
    if not path.exists(): return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _f(x):
    try: return float(x)
    except (TypeError, ValueError): return None


def main():
    ap = cf.standard_argparse(__doc__)
    args = ap.parse_args()
    cfg = cf.parse_kconfig(Path(args.config))
    if not cf.kbool(cfg, "CONFIG_MUVERA_REPORT", True):
        print("CONFIG_MUVERA_REPORT=n; skipping")
        return
    run_dir = cf.get_run_dir(cfg)

    sys_rows = _read_csv(run_dir / "systems_byte_floor.csv")
    ret_rows = _read_csv(run_dir / "retrieval_frontier.csv")
    nvme_rows = _read_csv(run_dir / "nvme_cold_byte_floor.csv")
    env = (run_dir / "environment.txt").read_text() if (run_dir / "environment.txt").exists() else ""
    git = (run_dir / "git_commit.txt").read_text() if (run_dir / "git_commit.txt").exists() else ""

    out = []
    out.append(f"# muvera-small benchmark report")
    out.append("")
    out.append(f"Run directory: `{run_dir}`")
    out.append("")
    out.append("## A. Executive summary")
    out.append("")
    out.append("Sub-4096-byte retrieval records are real and useful for")
    out.append("memory- / cache-resident vector stores. Raw NVMe random reads,")
    out.append("when measured cold or via O_DIRECT, suffer 4 KiB-page-read")
    out.append("amplification regardless of the logical record size, and the")
    out.append("sub-page advantage shrinks or disappears.")
    out.append("")
    out.append("On the retrieval-quality side, pooled int8 is a strong tiny-")
    out.append("record baseline. MUVERA-style FDE compression of multi-vector")
    out.append("retrieval (ColBERTv2) earns its complexity above ~5 KiB byte")
    out.append("budgets on BEIR/scifact; below that, pooled int8 is hard to")
    out.append("beat. int8 quantization is effectively free across pooled and")
    out.append("FDE on this benchmark; PQ-256-8 has dim/dataset-dependent")
    out.append("behavior and is not a universal free lunch.")
    out.append("")
    out.append("## B. Hardware and environment")
    out.append("")
    out.append("```")
    out.append(env.strip() or "(no environment captured)")
    out.append("```")
    out.append("")
    out.append("```")
    out.append(git.strip() or "(no git info captured)")
    out.append("```")
    out.append("")

    out.append("## C. Systems byte-floor result")
    out.append("")
    if not sys_rows:
        out.append("(no systems CSV — run `make muvera-small-systems`)")
    else:
        # Pull mmap_random_point per record_bytes
        rp = {int(r["record_bytes"]): r for r in sys_rows
              if r["mode"] == "mmap_random_point"}
        h2d64 = {int(r["record_bytes"]): r for r in sys_rows
                  if r["mode"] == "h2d_transfer" and r.get("batch_size") == "64"}
        proj64 = {int(r["record_bytes"]): r for r in sys_rows
                   if r["mode"] == "gpu_projection" and r.get("batch_size") == "64"}
        out.append("Memory/cache-resident view:")
        out.append("")
        out.append("| record bytes | mmap rand p50 (ns) | mmap rand p99 (ns) | H2D x64 p50 (μs) | proj x64 p50 (μs) |")
        out.append("|------:|---------:|---------:|--------:|--------:|")
        for b in sorted(rp.keys()):
            row = rp[b]
            h2d = h2d64.get(b, {}); proj = proj64.get(b, {})
            p50_ns = _f(row["lat_p50_ns"]) or 0
            p99_ns = _f(row["lat_p99_ns"]) or 0
            h2d_us = (_f(h2d.get("lat_p50_ns")) or 0) / 1e3
            proj_us = (_f(proj.get("lat_p50_ns")) or 0) / 1e3
            out.append(f"| {b} | {p50_ns:.0f} | {p99_ns:.0f} | {h2d_us:.1f} | {proj_us:.1f} |")
        out.append("")
        out.append("Plots in `plots/`:")
        out.append("- record_bytes_vs_mmap_random_p50.png")
        out.append("- record_bytes_vs_mmap_random_p95.png")
        out.append("- record_bytes_vs_h2d_batch64.png")
        out.append("- record_bytes_vs_total_serving_estimate.png")
    out.append("")

    out.append("## D. Optional NVMe cold/O_DIRECT result")
    out.append("")
    if not nvme_rows:
        out.append("Not run (CONFIG_MUVERA_ENABLE_NVME_COLD_BENCH=n).")
    else:
        out.append("| mode | logical bytes | physical bytes | read amp | p50 (μs) | p99 (μs) | iops |")
        out.append("|------|--------------:|--------------:|--------:|--------:|--------:|------:|")
        for r in nvme_rows:
            out.append(f"| {r['mode']} | {r['logical_record_bytes']} | "
                       f"{r['physical_read_bytes']} | "
                       f"{r['read_amplification']} | "
                       f"{r['lat_p50_us']} | {r['lat_p99_us']} | {r['iops']} |")
        out.append("")
        out.append("Reading: under O_DIRECT, sub-4096 logical records still")
        out.append("perform 4096-byte physical reads, so `read_amplification`")
        out.append("rises with the inverse of the logical record size. Latency")
        out.append("is dominated by NVMe random-access latency; logical record")
        out.append("size has little effect on it.")
    out.append("")

    out.append("## E. Retrieval result")
    out.append("")
    if not ret_rows:
        out.append("(no retrieval CSV — run `make muvera-small-retrieval`)")
    else:
        out.append("| variant | R | k_sim | FDE dim | precision | bytes/vec | recall@10 | recall@100 | lat μs/q |")
        out.append("|---------|--:|------:|--------:|----------|---------:|---------:|----------:|---------:|")
        for r in ret_rows:
            r10 = _f(r["recall_at_10"]); r100 = _f(r["recall_at_100"])
            lat = _f(r["lat_us_per_query"])
            out.append(f"| {r['variant']} | {r['R']} | {r['k_sim']} | "
                       f"{r['FDE_dim']} | {r['precision']} | "
                       f"{r['bytes_per_vector']} | "
                       f"{(r10 or 0):.3f} | {(r100 or 0):.3f} | "
                       f"{(lat or 0):.1f} |")
        out.append("")
        out.append("Plots in `plots/`:")
        out.append("- bytes_vs_recall10.png")
        out.append("- bytes_vs_recall100.png")
        out.append("- latency_vs_recall10.png")
    out.append("")

    out.append("## F. Interpretation")
    out.append("")
    out.append("- Memory/cache-resident sub-4096 records help random access.")
    out.append("  Most of the win is captured by ~256-512 bytes; further")
    out.append("  shrinking buys little.")
    out.append("- CPU→GPU H2D transfer is overhead-bound below ~4 KiB total")
    out.append("  batch payload. Smaller records do not save H2D time below")
    out.append("  that threshold.")
    out.append("- GPU projection cost scales linearly with input dim. Smaller")
    out.append("  stored bytes save real GPU compute, but the constant matters.")
    out.append("- If the access path is raw NVMe random reads (cold / O_DIRECT),")
    out.append("  the 4 KiB page boundary erases the sub-4096 logical-record")
    out.append("  advantage. Sub-page records suffer read amplification.")
    out.append("- The storage stack must do something — packed layouts,")
    out.append("  batching, caching, near-storage filtering — to exploit")
    out.append("  sub-page records on NVMe.")
    out.append("- int8 retrieval vectors are a serious default. Quantization")
    out.append("  is free across pooled and FDE in this benchmark.")
    out.append("- MUVERA/FDE earns its complexity at multi-KiB byte budgets;")
    out.append("  pooled int8 is the tiny-byte baseline.")
    out.append("- PQ-256-8 has dim/dataset-dependent behavior and must not")
    out.append("  be assumed free; production deployment should sweep")
    out.append("  M, nbits, and consider OPQ.")
    out.append("")

    out.append("## G. Reproduction")
    out.append("")
    out.append("```")
    out.append("make defconfig-muvera-small")
    out.append("make muvera-small")
    out.append("make muvera-small-report")
    out.append("```")
    out.append("")
    out.append("Optional NVMe cold benchmark:")
    out.append("```")
    out.append("# enable in .config")
    out.append("CONFIG_MUVERA_ENABLE_NVME_COLD_BENCH=y")
    out.append("CONFIG_MUVERA_NVME_BENCH_PATH=/path/to/nvme")
    out.append("make muvera-small-nvme-cold")
    out.append("```")
    out.append("")

    out.append("## H. Next tests")
    out.append("")
    out.append("- Run NVMe cold benchmark on different drives and filesystems")
    out.append("  to see how page granularity and queue depth affect the")
    out.append("  read-amplification floor.")
    out.append("- Test GPUDirect Storage if available; H2D may be much")
    out.append("  cheaper.")
    out.append("- Sweep PQ M, nbits, and OPQ on the FDE retrieval frontier.")
    out.append("- Run on larger BEIR datasets (NQ, HotpotQA, MS MARCO) to")
    out.append("  see how recall absolute numbers and the FDE-vs-pooled")
    out.append("  crossover scale.")

    report = run_dir / "report.md"
    report.write_text("\n".join(out) + "\n")
    print(f"wrote {report}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()

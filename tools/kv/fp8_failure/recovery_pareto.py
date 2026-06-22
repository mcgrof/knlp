"""Recovery Pareto (stop condition: recovery methods include storage + metadata cost). Each K-side
repair buys back some of the FP8-K failure at some cost in K-cache bytes per token and in side
metadata (stored bias / per-channel scales / gauge D). This puts the two on one axis so the playbook
can recommend the cheapest repair that clears the bar -- not just the one with the best quality.

CPU-only: the recovery fractions are the measured values from the atlas runs (cited in the row), and
the byte costs are arithmetic from the model's KV geometry. `--self-test` checks the byte math.
"""

import argparse
import csv
import json
import os

# repair -> (k_bits_per_element, metadata_floats_per_layer_per_kv_channel, note). Metadata is the
# side info a serving stack must store/ship to apply the repair, amortized per cached token.
REPAIRS = {
    "K8/V8 per-tensor (the failure)": (8, 0.0, "1 scale/tensor; negligible metadata"),
    "K8 per-channel": (8, 1.0, "one scale per K channel"),
    "K8 pre-bias": (8, 1.0, "stores the per-channel K-bias (subtracted pre-RoPE)"),
    "int8-K per-tensor": (
        8,
        0.0,
        "uniform int8; same cost as fp8 but rescues the bias mechanism",
    ),
    "QK-gauge": (8, 1.0, "stores the diagonal gauge D per channel"),
    "K16/V8 (asym)": (16, 0.0, "K stays bf16 -> 2x K bytes, zero metadata"),
    "native K (bf16)": (16, 0.0, "no quant; the upper bound"),
}


def k_bytes_per_token(k_bits, n_kv_heads, head_dim):
    return k_bits / 8.0 * n_kv_heads * head_dim


def metadata_overhead_pct(
    meta_floats_per_chan, n_kv_heads, head_dim, k_bits, context_len
):
    """Side metadata is per (layer, kv-channel) but amortized over `context_len` cached tokens; the
    K cache itself is per (token, layer, kv-channel). So the per-token overhead ratio is the metadata
    bytes (fp32 scales) divided by (context_len * K bytes), as a percent."""
    meta_bytes = meta_floats_per_chan * n_kv_heads * head_dim * 4.0  # fp32 scales/bias
    k_bytes = k_bytes_per_token(k_bits, n_kv_heads, head_dim) * context_len
    return 100.0 * meta_bytes / max(k_bytes, 1e-9)


def build_pareto(n_kv_heads, head_dim, context_len, recovery):
    rows = []
    for name, (kb, mf, note) in REPAIRS.items():
        rows.append(
            dict(
                repair=name,
                k_bits=kb,
                k_bytes_per_token=round(k_bytes_per_token(kb, n_kv_heads, head_dim), 1),
                metadata_overhead_pct=round(
                    metadata_overhead_pct(mf, n_kv_heads, head_dim, kb, context_len), 4
                ),
                recovery_fraction=recovery.get(name, ""),
                note=note,
            )
        )
    return rows


# Measured recovery on the catastrophic anchor (Qwen2.5-7B), from the atlas runs. 1.0 = fully
# rescued the K8/V8 failure; the value is the recovery fraction or "rescues"/"fails" for the
# coarse ones. Provenance in the SUMMARY docs.
QWEN7B_RECOVERY = {
    "K8/V8 per-tensor (the failure)": 0.0,
    "K8 per-channel": 0.0,  # phase-7: per-channel still catastrophic (top1 0.56)
    "K8 pre-bias": 0.97,  # controls: true-bias recovery 0.97
    "int8-K per-tensor": 0.95,  # phase-7: int8 rescues (top1 0.30->0.96)
    "QK-gauge": 0.0,  # loses to per-channel on Qwen
    "K16/V8 (asym)": 1.0,  # tolerant
    "native K (bf16)": 1.0,
}

FIELDS = [
    "repair",
    "k_bits",
    "k_bytes_per_token",
    "metadata_overhead_pct",
    "recovery_fraction",
    "note",
]


def write_pareto(out_dir, sn, rows):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "RECOVERY_PARETO.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    return p


def self_test():
    # Qwen2.5-7B GQA: 4 KV heads, head_dim 128.
    assert k_bytes_per_token(8, 4, 128) == 512.0  # fp8: 1 byte * 4 * 128
    assert k_bytes_per_token(16, 4, 128) == 1024.0  # bf16: 2x
    # per-channel metadata (1 fp32 scale per channel) over an 8K context is tiny vs the cache
    ov = metadata_overhead_pct(1.0, 4, 128, 8, 8192)
    assert 0.0 < ov < 0.1, ov  # well under 0.1%
    # zero-metadata repairs report 0 overhead
    assert metadata_overhead_pct(0.0, 4, 128, 8, 8192) == 0.0
    rows = build_pareto(4, 128, 8192, QWEN7B_RECOVERY)
    assert len(rows) == len(REPAIRS)
    assert any(
        r["repair"].startswith("int8") and r["recovery_fraction"] == 0.95 for r in rows
    )
    print("[self-test] recovery_pareto OK")
    print(json.dumps(rows, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--output-dir")
    ap.add_argument("--short-name", default="qwen2.5-7b")
    ap.add_argument("--n-kv-heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--context", type=int, default=8192)
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    rows = build_pareto(args.n_kv_heads, args.head_dim, args.context, QWEN7B_RECOVERY)
    p = write_pareto(args.output_dir, args.short_name, rows)
    print(f"[pareto] -> {p}")
    for r in rows:
        print(
            f"    {r['repair']:34} {r['k_bytes_per_token']:>7} B/tok  "
            f"meta {r['metadata_overhead_pct']}%  recovery {r['recovery_fraction']}"
        )


if __name__ == "__main__":
    main()

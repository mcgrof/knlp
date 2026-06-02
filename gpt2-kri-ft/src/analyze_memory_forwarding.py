"""Phase 2.7 — memory-forwarding diagnostics.

Three analyses on the existing eval outputs, no new model forwards
required:

1) NLL by position bin, broken down by (model, policy, budget).
   If KRI-FT has learned to forward information into recent hidden
   states, late-position NLL should improve disproportionately for
   KRI-FT under pruning — late positions are exactly where the
   pruned cache hurts vanilla and where the trained model's
   memory-forwarding pays off. Source: eval_l2m_scaling's
   nll_bin_0..3 columns.

2) The A/B/C/D condition table — for each model at a fixed budget,
   compare gap under recent (B) vs random_global (C) vs kri (A).
   D (oracle) comes from eval_oracle_kl_blocks; we cross-reference.
   The pattern "KRI-FT improves all of A, B, C similarly" supports
   the "training is the lever" reading; "KRI-FT improves A much
   more than B, C" supports "the score function helps too."

3) Selected-block frequency / overlap between KRI-FT and vanilla.
   For each query position bin, which prefix blocks does KRI most
   often select? Are they consistently far-prefix (suggesting the
   model relies on long-range info) or local (suggesting the
   training just teaches it to drop more aggressively)?

Outputs are JSON tables; the script does not produce plots.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--l2m_jsonl", type=str, required=True,
                   help="Path to eval_rows.jsonl from eval_l2m_scaling.py")
    p.add_argument("--oracle_jsonl", type=str, default=None,
                   help="Optional path to oracle_rows.jsonl for the A/B/C/D table")
    p.add_argument("--router_jsonl", type=str, default=None,
                   help="Optional path to router.jsonl from eval_router_ablation.py")
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def analyze_position_binned(l2m_rows):
    """Group nll_bin_0..3 by (model, policy, budget) and report."""
    bins = []
    for r in l2m_rows:
        if r.get("policy") == "full" or r.get("nll_bin_0") is None:
            continue
        bins.append({
            "model": r["model"],
            "policy": r["policy"],
            "context_length": r["context_length"],
            "budget_label": r["budget_label"],
            "actual_mean_retained_frac": r["actual_mean_retained_frac"],
            "nll_bin_0_to_25pct": r["nll_bin_0"],
            "nll_bin_25_to_50pct": r["nll_bin_1"],
            "nll_bin_50_to_75pct": r["nll_bin_2"],
            "nll_bin_75_to_100pct": r["nll_bin_3"],
            "nll_full": r["nll_full"],
            "nll_sparse": r["nll_sparse"],
            "nll_gap": r["nll_gap"],
            "late_vs_early_excess": (r["nll_bin_3"] or 0.0) - (r["nll_bin_0"] or 0.0),
        })
    return bins


def abcd_table(l2m_rows, oracle_rows):
    """For each (model, L, ~ matched budget), compare the four conditions:
       A = kri policy
       B = recent
       C = random_global
       D = oracle (from oracle_rows; only at the budget the oracle achieved)
    """
    rows = []
    by_key = defaultdict(dict)
    for r in l2m_rows:
        if r.get("policy") not in ("kri", "recent", "sink_recent", "random_global"):
            continue
        key = (r["model"], r["context_length"], r["budget_label"])
        by_key[key][r["policy"]] = r

    for (model, L, budget_label), entries in by_key.items():
        rows.append({
            "model": model,
            "context_length": L,
            "budget_label": budget_label,
            "A_kri_nll_gap": (entries.get("kri") or {}).get("nll_gap"),
            "A_kri_actual_frac": (entries.get("kri") or {}).get("actual_mean_retained_frac"),
            "B_recent_nll_gap": (entries.get("recent") or {}).get("nll_gap"),
            "B_sink_recent_nll_gap": (entries.get("sink_recent") or {}).get("nll_gap"),
            "C_random_nll_gap": (entries.get("random_global") or {}).get("nll_gap"),
            "C_random_actual_frac": (entries.get("random_global") or {}).get("actual_mean_retained_frac"),
        })

    if oracle_rows:
        oracle_by_model_L = defaultdict(list)
        for r in oracle_rows:
            oracle_by_model_L[(r["model"], r["context_length"])].append(r)
        # Find the minimum KL achieved per (model, L) at step=max(step)
        # (this is the loose "best the oracle could do given the
        # block count we ran")
        for k, lst in oracle_by_model_L.items():
            max_step = max(r["step"] for r in lst)
            kls = [r["kl_to_full_teacher"] for r in lst if r["step"] == max_step]
            oracle_by_model_L[k] = {"max_step": max_step,
                                    "mean_kl_at_max": sum(kls) / max(1, len(kls))}

        for row in rows:
            k = (row["model"], row["context_length"])
            if k in oracle_by_model_L:
                row["D_oracle_max_step"] = oracle_by_model_L[k]["max_step"]
                row["D_oracle_kl_at_max"] = oracle_by_model_L[k]["mean_kl_at_max"]
    return rows


def block_selection_summary(oracle_rows):
    """Aggregate which prefix blocks the oracle selects across batches
    and models. High-frequency early blocks suggest the model relies
    on long-prefix information; high-frequency late blocks suggest
    the boundary region between local window and full-attention
    matters most.
    """
    per_model = defaultdict(lambda: defaultdict(int))
    for r in oracle_rows:
        for b in r["selected_blocks"]:
            per_model[r["model"]][b] += 1
    # Convert to sorted lists
    out = {}
    for model, freqs in per_model.items():
        sorted_freqs = sorted(freqs.items(), key=lambda x: -x[1])[:20]
        out[model] = [{"block": b, "count": c} for b, c in sorted_freqs]
    return out


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    l2m_rows = [json.loads(l) for l in open(args.l2m_jsonl)]
    oracle_rows = [json.loads(l) for l in open(args.oracle_jsonl)] if args.oracle_jsonl else []

    # 1. Position-binned NLL
    pos = analyze_position_binned(l2m_rows)
    (out / "position_binned.jsonl").write_text(
        "\n".join(json.dumps(r) for r in pos) + "\n"
    )
    if pos:
        with (out / "position_binned.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(pos[0].keys()))
            w.writeheader()
            for r in pos:
                w.writerow(r)
    print(f"position_binned: {len(pos)} rows -> {out}/position_binned.csv")

    # 2. A/B/C/D table
    abcd = abcd_table(l2m_rows, oracle_rows)
    if abcd:
        (out / "abcd_table.jsonl").write_text(
            "\n".join(json.dumps(r) for r in abcd) + "\n"
        )
        # CSV needs the union of all keys (some rows have oracle
        # fields after abcd_table merged with oracle data).
        all_keys = set()
        for r in abcd:
            all_keys.update(r.keys())
        fieldnames = list(abcd[0].keys()) + [k for k in sorted(all_keys)
                                              if k not in abcd[0]]
        with (out / "abcd_table.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for r in abcd:
                w.writerow(r)
    print(f"abcd_table: {len(abcd)} rows -> {out}/abcd_table.csv")

    # 3. Block selection frequency (oracle)
    if oracle_rows:
        bs = block_selection_summary(oracle_rows)
        (out / "oracle_block_frequency.json").write_text(json.dumps(bs, indent=2))
        print(f"oracle_block_frequency: -> {out}/oracle_block_frequency.json")

    # Headline summary text
    summary_lines = [
        "=== Phase 2.7 memory-forwarding summary ===",
        "",
        "Position-binned NLL — late-bin minus early-bin excess at low budgets",
        "(higher = more degradation at late positions; KRI-FT should show LESS)",
    ]
    # rank rows by late_vs_early_excess at the smallest actual_mean_retained_frac
    pos_rows = [r for r in pos if r["budget_label"].startswith("fixedF_0.0625")]
    for r in sorted(pos_rows, key=lambda r: (r["context_length"], r["model"])):
        summary_lines.append(
            f"  L={r['context_length']:>4} model={r['model'][:30]:30s} "
            f"policy={r['policy']:13s} late−early={r['late_vs_early_excess']:+.4f}  "
            f"nll_gap={r['nll_gap']:+.4f}  actual={r['actual_mean_retained_frac']:.3f}"
        )
    summary_lines.append("")
    summary_lines.append("If KRI-FT consistently shows late−early excess close to or BELOW")
    summary_lines.append("vanilla / dense-FT at the same actual retention, that is the")
    summary_lines.append("memory-forwarding signature: late tokens cope without distant KV.")
    (out / "phase27_summary.txt").write_text("\n".join(summary_lines) + "\n")
    print(f"\nsummary -> {out}/phase27_summary.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

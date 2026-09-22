# SPDX-License-Identifier: GPL-2.0
"""Render the latency result from the measurement, not from recollection.

Every number in the prose below is read out of the timing artifact. Numbers
that are typed into a report by hand drift from the file they came from, and
this lane has already had to withdraw one figure that existed in no artifact
at all.

Run: python research/kv_translate/render_latency_result.py --timing <p> --out <p>
"""

from __future__ import annotations

import argparse
import json

NAMES = {
    "target_native": "native",
    "existing_gpu_source_kv_switch": "switch, source cache already resident",
    "source_prefill_inclusive_translation": "switch, source prefill included",
}


def verdict_line(e):
    if e["supported_saving"]:
        return "saving supported"
    iv = e["paired_interval"]
    if iv["lo"] < 1.0 < iv["hi"]:
        return "uncertain, interval spans 1"
    return "no saving"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", required=True)
    ap.add_argument("--receipt", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = json.load(open(args.timing))
    rec = json.load(open(args.receipt)) if args.receipt else {}

    L = sorted(d["results"], key=int)
    out = []
    w = out.append

    w("# First-next-token latency at 512, 2,048 and 4,096 tokens\n")
    w(
        "The remaining question from the replay was whether the translated "
        "path loses on latency everywhere, or only at the one context length "
        "it had been measured at. It was measured at 512 tokens, where the "
        "target's own prefill is mostly fixed overhead. This answers it at "
        "three lengths, under two different assumptions about whether the "
        "source cache already exists.\n"
    )
    w(
        "It answers it for this implementation, on this card, at these three "
        "lengths. Three points cannot establish a breakpoint, monotonicity, "
        "or anything about contexts longer than the longest one here. Quality "
        "at 2,048 and 4,096 was not measured and is not claimed.\n"
    )

    w("\n## The boundary\n")
    w(
        "Stated because the earlier gate's boundary was not. That gate "
        "computed the source prefix before starting its clock, so its "
        "translated figure is this page's *switch* condition and contains no "
        "source prefill. Nothing here was tuned to reproduce it.\n"
    )
    for k, v in d["boundary"].items():
        if k == "note":
            continue
        w(f"- **{NAMES.get(k, k)}** — {v}\n")

    w("\n## What was measured\n")
    dev = d["device"]
    pin = d["pinning"]
    pol = d["analysis_policy"]
    w(
        f"One {dev['gpu']}, {pin['model_and_cache_dtype']} weights and cache, "
        f"{pin['mapper_and_feature_dtype']} mapper and features, TF32 "
        f"{'on' if dev['tf32_allowed'] else 'off'}, batch one, "
        f"{pin['attn_implementation']} attention. Peak allocation "
        f"{dev['peak_alloc_bytes'] / 2**30:.1f} GiB of "
        f"{dev['total_bytes'] / 2**30:.1f} GiB, so both models and the mapper "
        f"were resident together.\n"
    )
    w(
        f"\n{pol['n_fixtures']} fixtures, {pol['warmups_per_fixture_path']} "
        f"warmups then {pol['measured_repetitions_per_fixture_path']} measured "
        f"repetitions per fixture and path, paired and interleaved in a fixed "
        f"order. Timing is {pol['primary_measure']}. The interval is "
        f"{pol['interval']}, resampled on the "
        f"{pol['resample_unit']}; it is {pol['interval_status']}.\n"
    )
    w(
        f"\nArtifact `{d['artifact']['joint_weight_sha256'][:16]}`"
        + (
            " — the frozen probe the plan names.\n"
            if d["artifact"].get("matches_required")
            else ".\n"
        )
    )

    w("\n## Results\n")
    w("| tokens | condition | p50 ms | p95 ms | ratio p50 | interval | verdict |\n")
    w("|---|---|---:|---:|---:|---|---|\n")
    for length in L:
        r = d["results"][length]
        nat = r["conditions"]["target_native"]
        w(
            f"| {length} | {NAMES['target_native']} | {nat['p50_ms']:.1f} | "
            f"{nat['p95_ms']:.1f} | — | — | control |\n"
        )
        for c in (
            "existing_gpu_source_kv_switch",
            "source_prefill_inclusive_translation",
        ):
            e = r["conditions"][c]
            iv = e["paired_interval"]
            w(
                f"| {length} | {NAMES[c]} | {e['p50_ms']:.1f} | "
                f"{e['p95_ms']:.1f} | {e['ratio_p50_over_native_p50']:.3f} | "
                f"[{iv['lo']:.3f}, {iv['hi']:.3f}] | {verdict_line(e)} |\n"
            )

    w(
        "\nA saving counts only where the interval's upper endpoint sits below "
        "one. The inherited deployment criteria are stricter and separate: a "
        "p50 ratio at or under 0.75, and translated p95 over native p50 below "
        "one — not a p95 against a p95.\n"
    )
    w("\n| tokens | condition | meets p50 0.75 | meets tail | blocks |\n")
    w("|---|---|---|---|---:|\n")
    for length in L:
        for c in (
            "existing_gpu_source_kv_switch",
            "source_prefill_inclusive_translation",
        ):
            e = d["results"][length]["conditions"][c]
            w(
                f"| {length} | {NAMES[c]} | {e['meets_inherited_p50']} | "
                f"{e['meets_inherited_tail']} | {e['blocks_taken']} |\n"
            )

    if rec:
        acct = rec["accounting"]
        w("\n## Cost and closure\n")
        w(
            f"Occupied {acct['occupied_seconds']} seconds of an allowance of "
            f"{acct['allowance_seconds']}, measured from the allocation request "
            f"to the confirmed release. Repairs used "
            f"{acct['repairs_used']} of {acct['repairs_allowed']}. "
            f"Allocation release: {rec['cleanup']['allocation_release']}. "
            f"Timing verdict **{rec['verdicts']['timing']}**.\n"
        )

    w("\n## Reproducing\n")
    w("```\n")
    w("python research/kv_translate/make_timing_fixtures.py \\\n")
    w("    --manifests manifests.json --out timing_fixtures.pt\n")
    w("W=<workspace> O=<out> T0=<unix> bash research/kv_translate/measure_latency.sh\n")
    w("```\n")
    w(
        f"\nSource archived at `{d['code']['revision'][:12]}`"
        f"{' with a dirty tree' if d['code']['tree_dirty'] else ', clean tree'}. "
        f"Raw samples: {d['outputs']['raw_sample_count']} rows, sha256 "
        f"`{d['outputs']['raw_sha256'][:16]}`.\n"
    )

    open(args.out, "w").write("".join(out))
    print(f"rendered {args.out} from {args.timing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

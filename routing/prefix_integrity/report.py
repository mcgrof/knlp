# SPDX-License-Identifier: GPL-2.0
"""Render a PIA result to JSON, a CSV of per-block survival, and a Markdown
report that classifies the candidate in plain language.

The report exists so the verdict is legible to a human who did not run the
harness: status, danger score, the metrics that drove it, and a one-paragraph
decision telling them where the algorithm is and is not safe to deploy.
"""

from __future__ import annotations

import csv
import json
import os

_DECISIONS = {
    "SAFE_FOR_PREFIX_OFFLOAD": (
        "Shape-preserving, deterministic, and prefix_hash-safe. Safe to store "
        "and share under prefix_hash in an ordinary prefix-cache / offload path."
    ),
    "SAFE_ONLY_WITH_EXTENDED_CACHE_KEY": (
        "Reloadable and shape-preserving, but the stored object depends on the "
        "query. It must NOT be keyed by prefix_hash alone -- include query_hash "
        "(or the policy hash) in the cache key, or treat it as routing-only."
    ),
    "SAFE_ONLY_WITH_CUSTOM_CONNECTOR": (
        "Uses partial blocks, merging, sub-block masks, or variable-shape "
        "tensors. Not reusable by vanilla block-hash prefix caching without a "
        "custom connector and extra metadata."
    ),
    "ROUTING_ONLY_NOT_PREFIX_CACHE_SAFE": (
        "Semantically acceptable as a query-aware sparse-attention / routing "
        "prior, but not compatible with existing prefix sharing: two requests "
        "with the same prefix no longer agree on the prefix object."
    ),
    "DANGEROUS_FOR_PREFIX_SHARING": (
        "The same prefix can map to different or non-reloadable KV objects. Do "
        "not deploy on a prefix-sharing / offload path as-is."
    ),
}


def write_reports(result: dict, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    paths = {}

    rj = os.path.join(out_dir, "result.json")
    with open(rj, "w") as f:
        json.dump(result, f, indent=2)
    paths["result_json"] = rj

    csv_path = os.path.join(out_dir, "block_survival.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["block", "status", "weight", "selected"])
        for row in result.get("block_survival", []):
            w.writerow([row["block"], row["status"], row["weight"], row["selected"]])
    paths["block_survival_csv"] = csv_path

    md_path = os.path.join(out_dir, "result.md")
    with open(md_path, "w") as f:
        f.write(render_markdown(result))
    paths["result_md"] = md_path

    return paths


def render_markdown(result: dict) -> str:
    m = result.get("metrics", {})
    cls = result.get("classification", "")
    lines = []
    lines.append(f"# Prefix Integrity Report: {result.get('algorithm', '?')}")
    lines.append("")
    lines.append(f"Status: {result.get('status')}")
    lines.append(f"Classification: {cls}")
    lines.append(f"Danger score: {result.get('danger_score')}")
    lines.append("")
    ident = result.get("identity", {})
    lines.append(
        f"Cartridge `{ident.get('cartridge_id')}` "
        f"(model {ident.get('model_id')}, {ident.get('num_blocks')} blocks of "
        f"{ident.get('block_size')} tokens, dtype {ident.get('dtype')}), "
        f"policy `{result.get('policy')}`, "
        f"cache key {result.get('cache_key_fields')}."
    )
    lines.append("")
    lines.append("## Core metrics")
    lines.append("")
    for key in (
        "pre_mean",
        "anchor_survival_mean",
        "recent_survival",
        "contiguous_prefix_survival",
        "partial_block_rate",
        "manifest_stability",
        "artifact_stability",
        "same_query_artifact_count",
        "cr_mean",
        "cr_cv",
        "read_ranges",
        "read_amplification",
    ):
        if key in m:
            lines.append(f"- {key}: {m[key]}")
    for key in ("kl", "top1"):
        if key in m:
            lines.append(f"- {key}: {m[key]}")
    lines.append("")

    if result.get("violations"):
        lines.append("## Violations (FAIL)")
        lines.append("")
        for v in result["violations"]:
            lines.append(f"- {v}")
        lines.append("")
    if result.get("warnings"):
        lines.append("## Warnings")
        lines.append("")
        for w in result["warnings"]:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("## Decision")
    lines.append("")
    lines.append(_DECISIONS.get(cls, cls))
    if result.get("recommendations"):
        lines.append("")
        for r in result["recommendations"]:
            lines.append(f"- {r}")
    lines.append("")
    return "\n".join(lines)

# SPDX-License-Identifier: MIT
"""Render complete reference aggregates, with certificate and memory scope explicit.

Raw preparation requires the complete evidence/analysis receipt. Published
aggregate bundles can be redrawn without access to raw request text. Redrawing
an aggregate bundle does not independently re-audit the underlying experiment.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import re


MODELS = {
    "qwen3": ("Qwen/Qwen3-8B", "FP16"),
    "qwen25": ("Qwen/Qwen2.5-7B-Instruct", "BF16"),
}
ARMS = ("N", "AF", "RF", "SF", "AD", "RD", "SD")
COLORS = {"N": "#64748b", "A": "#c56817", "R": "#287d8e", "S": "#7555a1"}
LABELS = {
    "N": "Native K16/V16",
    "AF": "K16/V8 · fixed",
    "RF": "K8/V16 · fixed",
    "SF": "K8/V8 · fixed",
    "AD": "K16/V8 · dynamic",
    "RD": "K8/V16 · dynamic",
    "SD": "K8/V8 · dynamic",
}
BINARY_FIELDS = (
    "n",
    "native_correct",
    "candidate_correct",
    "losses",
    "gains",
    "net_delta",
    "net_interval_95",
    "net_one_sided_bounds",
    "gross_loss_one_sided_bounds",
    "certificate_alpha",
)
NLL_FIELDS = (
    "n_documents",
    "native_mean",
    "candidate_mean",
    "absolute_delta",
    "relative_delta",
    "absolute_one_sided_bounds",
    "relative_one_sided_bounds",
    "alpha",
    "resamples",
    "seed",
)
ENDPOINTS = ("gsm8k", "retrieval-1024", "retrieval-8192", "nll-1024", "nll-8192")


def numeric(value):
    """Only finite numeric aggregates may cross the publication boundary."""
    if type(value) in (int, float) and math.isfinite(value):
        return value
    if isinstance(value, list) and len(value) == 2:
        if all(type(x) in (int, float) and math.isfinite(x) for x in value):
            return list(value)
    raise ValueError("Expected a finite numeric aggregate or two-endpoint interval")


def endpoint(row, name):
    if name.startswith("nll-"):
        if (
            row["n_documents"] != 64
            or row["alpha"] != 0.025 / 12
            or row["resamples"] != 100000
            or row["seed"] != 20261020
            or row["quantile"] != "linear"
        ):
            raise ValueError("NLL panel or inference settings differ")
        fields = NLL_FIELDS
    else:
        if row["n"] != (1319 if name == "gsm8k" else 200):
            raise ValueError("Incomplete binary panel")
        if row["certificate_alpha"] != 0.025 / 12:
            raise ValueError("Binary certificate alpha differs")
        fields = BINARY_FIELDS
    return {key: numeric(row[key]) for key in fields}


def direct_contrasts(rows):
    expected = {
        ("A" + policy, control + policy, name)
        for policy in ("F", "D")
        for control in ("R", "S")
        for name in ENDPOINTS
    }
    identities = [(r["candidate"], r["baseline"], r["endpoint"]) for r in rows]
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError("Require every predeclared direct comparison exactly once")
    result = []
    for row in rows:
        nll = row["endpoint"].startswith("nll-")
        interval = row["bonferroni8_nll_interval"]
        if not nll and interval is not None:
            raise ValueError("Binary contrast unexpectedly has an NLL interval")
        result.append(
            {key: row[key] for key in ("candidate", "baseline", "endpoint")}
            | {
                "result": endpoint(row["result"], row["endpoint"]),
                "p_two_sided": numeric(row["p_two_sided"]),
                "holm8_p": numeric(row["holm8_p"]),
                "paired_interval_95": numeric(row["paired_interval_95"]),
                "bonferroni8_nll_interval": numeric(interval) if nll else None,
                "test_method": (
                    "Paired-document centered bootstrap, 100000 seed20261020; approximate"
                    if nll
                    else "Exact two-sided McNemar; descriptive Tango interval"
                ),
            }
        )
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def aggregate(analysis_path, evidence_path, receipt_path):
    receipt = read(receipt_path)
    for path in (analysis_path, evidence_path):
        if receipt["outputs"].get(path.name) != sha(path):
            raise ValueError("Complete reader receipt does not bind the input")
    analysis, evidence = read(analysis_path), read(evidence_path)
    if (
        set(analysis["models"]) != set(MODELS)
        or set(evidence["models"]) != set(MODELS)
        or analysis["alpha"] != 0.025 / 12
    ):
        raise ValueError("Require both complete reference models and fixed alpha")
    result = {
        "schema": 1,
        "scope": "Physical reference with native attention arithmetic; no fused-kernel performance claim",
        "analysis_sha256": sha(analysis_path),
        "evidence_sha256": sha(evidence_path),
        "reader_receipt_sha256": sha(receipt_path),
        "certificate_alpha": analysis["alpha"],
        "models": {},
    }
    for name, (model_id, dtype) in MODELS.items():
        model, audit = analysis["models"][name], evidence["models"][name]
        if (
            model["model"] != model_id
            or not re.fullmatch(r"[0-9a-f]{40}", model["revision"])
            or not re.fullmatch(r"[0-9a-f]{40}", model["source"]["commit"])
            or not re.fullmatch(r"[0-9a-f]{64}", model["input_sha256"])
            or any(
                model[k] != audit[k]
                for k in ("model", "revision", "source", "input_sha256")
            )
            or set(model["candidates"]) != set(ARMS[1:])
            or audit["counts"]
            != {arm: {"nll": 128, "retrieval": 400, "gsm8k": 1319} for arm in ARMS}
            or audit["retained_tokens_decode_exactly"] is not True
        ):
            raise ValueError("Incomplete reference or inconsistent identities")
        candidates = {}
        for arm, row in model["candidates"].items():
            if any(
                type(row[key]) is not bool
                for key in ("preservation_certificate", "aggregate_math_certificate")
            ):
                raise ValueError("Certificate labels must be booleans")
            candidates[arm] = {
                key: row[key]
                for key in ("preservation_certificate", "aggregate_math_certificate")
            }
            candidates[arm].update(
                {name: endpoint(row[name], name) for name in ENDPOINTS}
            )
        # Explicit aggregate allowlist: no request text, operator receipts,
        # filesystem paths or cloud identities enter this public bundle.
        result["models"][name] = {
            "model": model_id,
            "dtype": dtype,
            "revision": model["revision"],
            "measured_source_commit": model["source"]["commit"],
            "input_sha256": model["input_sha256"],
            "candidates": candidates,
            "direct": direct_contrasts(model["direct"]),
            "direct_scope": "All deltas are candidate minus the named baseline. In direct result fields, native_correct/native_mean denote that compressed baseline. Holm adjustment is across eight contrasts per endpoint over both models; endpoint significance is not general quality superiority.",
            "memory": {
                arm: {
                    length: {
                        key: numeric(model["memory"][arm][length][key])
                        for key in ("persistent_bytes", "scale_bytes")
                    }
                    for length in ("1024", "8192")
                }
                for arm in ARMS
            },
        }
    return result


def draw(data, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if data["schema"] != 1 or set(data["models"]) != set(MODELS):
        raise ValueError("Unsupported reference aggregate bundle")
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update(
        {"svg.fonttype": "none", "svg.hashsalt": "decision-reference", "font.size": 10}
    )
    files = []

    def save(fig, name):
        if data.get("synthetic"):
            fig.text(
                0.5,
                0.5,
                "SYNTHETIC — NOT RESULTS",
                ha="center",
                fontsize=25,
                rotation=18,
                alpha=0.2,
            )
        for suffix in ("svg", "png"):
            path = output / (name + "." + suffix)
            options = {"metadata": {"Date": None}} if suffix == "svg" else {}
            fig.savefig(path, dpi=180, facecolor="white", **options)
            files.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
    for ax, (name, (_, dtype)) in zip(axes, MODELS.items()):
        model = data["models"][name]
        for y, arm in enumerate(ARMS[1:]):
            row = model["candidates"][arm]
            gsm = row["gsm8k"]
            point = 100 * gsm["net_delta"]
            lo, hi = [100 * x for x in gsm["net_interval_95"]]
            ax.errorbar(
                point,
                y,
                xerr=[[point - lo], [hi - point]],
                color=COLORS[arm[0]],
                marker="o" if arm.endswith("F") else "D",
                capsize=4,
                markersize=6,
            )
        ax.axvline(0, color="#94a3b8", linewidth=1, linestyle="--")
        ax.set_yticks(range(6), [LABELS[a] for a in ARMS[1:]])
        ax.invert_yaxis()
        ax.set_xlabel("Math accuracy change from native (percentage points)")
        native = model["candidates"]["AF"]["gsm8k"]["native_correct"]
        ax.set_title(
            f"{model['model'].split('/')[-1]} · {dtype}\nNative: {native}/1,319"
        )
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Paired math accuracy: native Q and controlled KV scale policy", fontsize=14
    )
    fig.text(
        0.5,
        0.045,
        "Descriptive paired 95% score intervals. Full certificates use separate family-adjusted bounds\nand also require both retrieval and raw-text loss gates; these intervals alone do not certify quality.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.01, 0.13, 0.99, 0.9))
    save(fig, "reference-math")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, (name, (_, dtype)) in zip(axes, MODELS.items()):
        model = data["models"][name]
        native = model["memory"]["N"]["8192"]["persistent_bytes"]
        fractions = [
            100 * model["memory"][a]["8192"]["persistent_bytes"] / native for a in ARMS
        ]
        ax.barh(range(7), fractions, color=[COLORS[a[0]] for a in ARMS])
        for y, value in enumerate(fractions):
            ax.text(value + 1, y, f"{value:.2f}%", va="center", fontsize=9)
        ax.set_yticks(range(7), [LABELS[a] for a in ARMS])
        ax.invert_yaxis()
        ax.set_xlim(0, 119)
        ax.set_xlabel("Persistent tensor backing bytes (% of native)")
        ax.set_title(f"{model['model'].split('/')[-1]} · {dtype}")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Physical reference cache at 8,320 consumed tokens", fontsize=14)
    fig.text(
        0.5,
        0.045,
        "Unique tensor backing bytes include FP32 scales. Weights, CUDA allocation rounding and temporary\nreconstruction are excluded. Storage measurements alone do not establish quality or serving capacity.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.01, 0.13, 0.99, 0.9))
    save(fig, "reference-memory")
    bundle = output / "reference-aggregates.json"
    bundle.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    files.append(bundle)
    (output / "figure-manifest.json").write_text(
        json.dumps(
            {
                "plotter_sha256": sha(__file__),
                "python": platform.python_version(),
                "matplotlib": matplotlib.__version__,
                "files": {p.name: sha(p) for p in files},
                "scope": "Figures reproduce the supplied aggregates; raw validation is a separate reader receipt",
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument(
        "--data", type=Path, help="Redraw a previously published aggregate bundle"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = (args.analysis, args.evidence, args.receipt)
    if (args.data is not None and any(raw)) or (args.data is None and not all(raw)):
        parser.error(
            "Supply analysis/evidence/receipt together, or one published data bundle"
        )
    draw(read(args.data) if args.data else aggregate(*raw), args.out)


if __name__ == "__main__":
    main()

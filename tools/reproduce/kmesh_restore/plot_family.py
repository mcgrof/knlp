#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plot measured preload arrivals and first-child latency from the public export."""

import argparse
import json
from pathlib import Path
import statistics


def plot(data: dict, output: Path) -> None:
    """Show every retained sample and the median within each requested-gap case."""
    import matplotlib.pyplot as plt

    maximum_gap = max(
        sample["actual_fork_gap_ms"]
        for name, case in data["cases"].items()
        if "-family-" in name
        for sample in case["retained_samples"]
    )
    with plt.rc_context({"font.size": 10, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(
            2, 2, figsize=(10, 6.5), sharex=True, constrained_layout=True
        )
        for row, mode in enumerate(("staged", "direct")):
            for column, fanout in enumerate((4, 8)):
                ax = axes[row, column]
                medians = []
                for gap in (0, 10, 25, 100):
                    samples = data["cases"][f"{mode}-n{fanout}-family-g{gap}"][
                        "retained_samples"
                    ]
                    xs = [s["actual_fork_gap_ms"] for s in samples]
                    ys = [s["first_submitted_child_ttft_ms"] for s in samples]
                    ax.scatter(xs, ys, s=23, color="#147b91", alpha=0.5)
                    medians.append((statistics.median(xs), statistics.median(ys)))
                ax.plot(
                    *zip(*medians),
                    color="#147b91",
                    marker="D",
                    markersize=5,
                    label="Declared preload: median",
                )
                for policy, color, label in (
                    ("reactive", "#c45b2c", "Reactive: median"),
                    ("oracle", "#646b78", "Fully warm: median"),
                ):
                    reference = data["cases"][f"{mode}-n{fanout}-{policy}-g0"][
                        "first_child_ttft"
                    ]["p50_ms"]
                    ax.axhline(reference, color=color, linestyle="--", label=label)
                ax.set_title(
                    f"{'Host staged' if mode == 'staged' else 'DMA-BUF'} · {fanout} children"
                )
                ax.set_xlim(-3, max(105, maximum_gap * 1.04))
                ax.set_ylim(bottom=0)
                ax.grid(alpha=0.18)
                ax.spines[["top", "right"]].set_visible(False)
                if row == 1:
                    ax.set_xlabel("Attained fork gap (ms)")
                if column == 0:
                    ax.set_ylabel("First submitted child's TTFT (ms)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False)
        count = data["configuration"]["retained_families_per_case"]
        fig.suptitle(
            f"Ancestor preload on the current storage stack\n{count} samples/case; points are measured trials, diamonds are medians",
            fontsize=13,
        )
        fig.savefig(output, metadata={"Date": None})
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(json.loads(args.input.read_text()), args.output)

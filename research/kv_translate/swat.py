# SPDX-License-Identifier: GPL-2.0
"""Run a paired cache-transfer evaluation using the authors' SwaT code."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from research.kv_translate.published.paired import compare

UPSTREAM = "34c7653c626c0c22ccf278b6628fa57dec4160d2"
BENCHMARKS = {
    "hellaswag": ("hellaswag:validation", "accuracy", 64),
    "hotpotqa": ("hotpotqa+terse:validation", "f1", 32),
    "squad": ("squad+terse:validation", "f1", 32),
    "musique": ("musique+terse:validation", "f1", 32),
}


def load_release(root):
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if revision != UPSTREAM:
        raise ValueError(f"Expected SwaT revision {UPSTREAM}, got {revision}")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root
    ):
        raise ValueError("SwaT checkout has modified tracked files")
    os.environ["MULTIKV_PYTHON"] = sys.executable
    spec = importlib.util.spec_from_file_location(
        "swat_exp1", root / "scripts/exp1/run.py"
    )
    release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release)
    return release


def recipe(root, args):
    config = json.loads((root / "scripts/exp1/config.json").read_text())
    config.update(
        sender=str(args.sender.resolve()), receiver=str(args.receiver.resolve())
    )
    shared = config["train_common"]["args"]
    shared["max-length"] = args.context
    shared["seed"] = args.seed
    for stage, steps in (
        ("stage1", args.reconstruction_steps),
        ("stage2", args.task_steps),
    ):
        training = config[stage]["args"]
        training.update(
            {
                "max-steps": steps,
                "max-trace-length": min(4096, args.context),
                "downstream-eval-every": 0,
                "save-every": -1,
            }
        )
    # The short recipe uses the released text mix and the named training sets.
    # The public release includes GSM8K gold solutions in its trace schema.
    config["stage2"]["args"]["datasets"][
        -1
    ] = "{repo}/multiagent/data/phase2_traces/gsm8k_train_gold_solutions.jsonl"
    config["stage2"]["args"]["dense-max-length"] = args.context
    if args.reconstruction_steps < 300:
        config["stage1"]["args"]["warmup-steps"] = max(
            1, args.reconstruction_steps // 10
        )
    if args.task_steps < 50:
        config["stage2"]["args"]["warmup-steps"] = max(1, args.task_steps // 10)

    evaluation = config["eval"]
    evaluation["benchmarks"] = []
    for name in args.tasks:
        task, metric, answer_length = BENCHMARKS[name]
        renderer = {"renderer": "{stage2_checkpoint}/renderer.pt"}
        if name == "hellaswag":
            renderer["renderer-decode-prefix"] = "generation"
        evaluation["benchmarks"].append(
            {
                "key": name,
                "domain": "out-of-domain" if name == "squad" else "in-domain",
                "metric": metric,
                "args": {
                    "tasks": task,
                    "limit-per-dataset": args.samples,
                    "max-new-tokens": answer_length,
                    "max-length": args.eval_context,
                },
                "renderer_args": renderer,
            }
        )
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--sender", required=True, type=Path)
    parser.add_argument("--receiver", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--reconstruction-steps", type=int, default=1000)
    parser.add_argument("--task-steps", type=int, default=2000)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--eval-context", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument(
        "--tasks", choices=BENCHMARKS, nargs="+", default=list(BENCHMARKS)
    )
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args(argv)
    if (
        min(
            args.reconstruction_steps,
            args.task_steps,
            args.context,
            args.eval_context,
            args.samples,
        )
        <= 0
    ):
        parser.error("step counts, context lengths and sample counts must be positive")

    root, run = args.upstream.resolve(), args.run.resolve()
    release = load_release(root)
    resolved = release.resolve(recipe(root, args), run)
    manifest = run / "config.json"
    if manifest.exists() and release.read(manifest) != resolved:
        parser.error("configuration differs from the existing run")
    release.write(manifest, resolved)
    release.check_blocks(resolved)
    jobs = [
        job
        for job in release.build_queue(resolved, run)
        if job["name"] not in {"traces", "filter_traces"}
    ]
    release.write(run / "queue.json", jobs)
    if args.prepare:
        return
    for job in jobs:
        print(f"Running {job['name']}", flush=True)
        # The upstream job can append its completion marker with &&.
        subprocess.run(
            ["/bin/bash", "-e", "-c", job["cmd"]],
            cwd=root,
            check=True,
        )
    results = release.read(run / "results.json")
    if set(results["benchmarks"]) != set(args.tasks):
        raise ValueError("Missing paired scores in the upstream summary")
    paired = {}
    for item in resolved["eval"]["benchmarks"]:
        name = item["key"]
        output = Path(resolved["eval"]["output_dir"])
        native = release.read_scores(output / f"{name}_vanilla.json", item["metric"])
        transferred = release.read_scores(
            output / f"{name}_renderer.json", item["metric"]
        )
        stats = compare(native, transferred)
        stats.pop("_resampled_retention")
        paired[name] = stats
    release.write(run / "paired.json", paired)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

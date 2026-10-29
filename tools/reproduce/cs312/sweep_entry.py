#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Drive the course launchers from a knlp plan.

This script runs inside the course environment with the course checkout
on PYTHONPATH and as the working directory. It builds TrainConfig objects
from plan.json, launches them on Modal or in sequence on the local GPU,
and later collects results from Weights & Biases and local receipts.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")
    tmp.replace(path)


def check_defaults(plan):
    from train import TrainConfig

    for axis, default in plan["axis_defaults"].items():
        actual = getattr(TrainConfig, axis)
        if actual != default:
            raise SystemExit(
                f"course default {axis}={actual} differs from the plan's "
                f"{default}; re-pin the course revision or the plan"
            )


def train_configs(plan):
    from train import TrainConfig

    configs = []
    for run in plan["runs"]:
        kwargs = dict(run["train_config"])
        kwargs["wandb_tags"] = tuple(kwargs.get("wandb_tags", ()))
        configs.append(TrainConfig(**kwargs))
    return configs


def run_names(configs):
    from train import checked_train_config, training_run_name

    return [training_run_name(checked_train_config(c)) for c in configs]


def wandb_entity():
    entity = os.environ.get("WANDB_ENTITY")
    if entity:
        return entity
    import wandb

    return wandb.Api().default_entity


def mode_modal(args, plan):
    from modal_train import launch_training_jobs

    check_defaults(plan)
    configs = train_configs(plan)
    names = run_names(configs)
    launched = utc_now()
    started = time.perf_counter()
    jobs = launch_training_jobs(
        configs,
        gpu=args.gpu,
        max_parallel_runs=args.max_parallel or None,
    )
    receipt = {
        "backend": "modal",
        "gpu": args.gpu,
        "environment": os.environ.get("DL_ALCHEMY_MODAL_ENVIRONMENT"),
        "wandb_entity": wandb_entity(),
        "wandb_project": os.environ.get("WANDB_PROJECT"),
        "launched_utc": launched,
        "launch_seconds": time.perf_counter() - started,
        "plan_digest": plan["digest"],
        "jobs": [{**job, "run_name": names[job["job_index"]]} for job in jobs],
    }
    write_json(args.output / "launch.json", receipt)


VAL_LOSS = re.compile(r"\bval Loss: ([0-9]+\.[0-9]+)")


def last_val_loss(log_path):
    value = None
    try:
        for line in Path(log_path).read_text(errors="replace").splitlines():
            match = VAL_LOSS.search(line)
            if match:
                value = float(match.group(1))
    except FileNotFoundError:
        return None
    return value


def device_name():
    import torch

    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def mode_local(args, plan):
    import torch

    check_defaults(plan)
    configs = train_configs(plan)
    names = run_names(configs)
    runs_dir = args.output / "runs"
    launch_path = args.output / "launch.json"
    receipt = read_json(launch_path) if launch_path.exists() else {}
    receipt.update(
        {
            "backend": "local",
            "gpu": device_name(),
            "torch": torch.__version__,
            "python": sys.version.split()[0],
            "parallel": args.parallel,
            "wandb_entity": wandb_entity() if configs[0].wandb_online else None,
            "wandb_project": os.environ.get("WANDB_PROJECT"),
            "plan_digest": plan["digest"],
        }
    )
    receipt.setdefault("started_utc", utc_now())
    receipt.setdefault("sessions", [])
    receipt["sessions"].append({"started_utc": utc_now()})
    receipt.pop("finished_utc", None)
    write_json(launch_path, receipt)

    def one(index):
        name = names[index]
        run_dir = runs_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        receipt_path = run_dir / "receipt.json"
        if receipt_path.exists() and read_json(receipt_path).get("returncode") == 0:
            print(f"[{index:2d}] skip finished {name}", flush=True)
            return
        log_path = run_dir / "train.log"
        print(f"[{index:2d}] start {name}", flush=True)
        started_utc = utc_now()
        started = time.perf_counter()
        with log_path.open("a") as stream:
            stream.write(f"# {started_utc} start\n")
            stream.flush()
            completed = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--mode",
                    "local-one",
                    "--plan",
                    str(args.plan),
                    "--output",
                    str(args.output),
                    "--index",
                    str(index),
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        total = time.perf_counter() - started
        write_json(
            receipt_path,
            {
                "index": index,
                "run_name": name,
                "returncode": completed.returncode,
                "started_utc": started_utc,
                "finished_utc": utc_now(),
                "total_seconds": total,
                "val_loss": last_val_loss(log_path),
                "gpu": receipt["gpu"],
                "log": str(log_path),
            },
        )
        print(
            f"[{index:2d}] done  {name} rc={completed.returncode} "
            f"{total / 60:.1f} min",
            flush=True,
        )

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        list(pool.map(one, range(len(configs))))
    receipt = read_json(launch_path)
    receipt["finished_utc"] = utc_now()
    receipt["sessions"][-1]["finished_utc"] = receipt["finished_utc"]
    write_json(launch_path, receipt)


def mode_local_one(args, plan):
    from train import train

    configs = train_configs(plan)
    train(configs[args.index])


def wandb_runs(plan, entity, project):
    import wandb

    api = wandb.Api()
    plan_tag = f"cs312-plan-{plan['digest']}"
    suffix_tag = f"cs312-{plan['run_suffix']}"
    found = {}
    for run in api.runs(f"{entity}/{project}", filters={"tags": {"$in": [plan_tag]}}):
        if suffix_tag not in run.tags:
            continue
        summary = run.summary
        created = run.created_at
        runtime = summary.get("_runtime")
        finished = None
        if created and runtime is not None:
            created_dt = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=timezone.utc)
            finished = (created_dt + timedelta(seconds=float(runtime))).isoformat(
                timespec="seconds"
            )
        found.setdefault(run.name, []).append(
            {
                "id": run.id,
                "url": run.url,
                "state": run.state,
                "val_loss": summary.get("val_loss"),
                "train_loss": summary.get("train_loss"),
                "total_seconds": summary.get("timing/total_seconds"),
                "training_seconds": summary.get("timing/training_seconds"),
                "startup_seconds": summary.get("timing/startup_seconds"),
                "completed_steps": summary.get("timing/completed_steps"),
                "runtime_seconds": runtime,
                "created_utc": str(created),
                "finished_utc": finished,
            }
        )
    return found


def modal_call_results(launch):
    results = {}
    try:
        import modal
    except ImportError:
        return results
    for job in launch.get("jobs", []):
        call_id = job.get("modal_call_id")
        if not call_id:
            continue
        try:
            value = modal.FunctionCall.from_id(call_id).get(timeout=0)
        except Exception as exc:  # pending, failed or expired call
            results[job["run_name"]] = {"pending_or_error": type(exc).__name__}
            continue
        results[job["run_name"]] = value
    return results


def mode_collect(args, plan):
    configs = train_configs(plan)
    names = run_names(configs)
    launch_path = args.output / "launch.json"
    launch = read_json(launch_path) if launch_path.exists() else {}
    entity = launch.get("wandb_entity") or wandb_entity()
    project = launch.get("wandb_project") or os.environ.get("WANDB_PROJECT")
    online = {}
    if configs[0].wandb_online:
        online = wandb_runs(plan, entity, project)
    modal_results = (
        modal_call_results(launch) if launch.get("backend") == "modal" else {}
    )
    runs = []
    for run, name in zip(plan["runs"], names):
        entry = {
            "index": run["index"],
            "axis": run["axis"],
            "exponent": run["exponent"],
            "overrides": run["overrides"],
            "run_name": name,
            "val_loss": None,
            "state": None,
            "total_seconds": None,
            "started_utc": None,
            "finished_utc": None,
        }
        receipt_path = args.output / "runs" / name / "receipt.json"
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            entry["local"] = receipt
            entry["val_loss"] = receipt.get("val_loss")
            entry["state"] = "finished" if receipt.get("returncode") == 0 else "failed"
            entry["total_seconds"] = receipt.get("total_seconds")
            entry["started_utc"] = receipt.get("started_utc")
            entry["finished_utc"] = receipt.get("finished_utc")
        candidates = online.get(name, [])
        if candidates:
            best = sorted(
                candidates,
                key=lambda c: (c["state"] != "finished", c["created_utc"]),
            )[0]
            entry["wandb"] = best
            if best.get("val_loss") is not None:
                entry["val_loss"] = best["val_loss"]
            if entry["state"] is None:
                entry["state"] = best["state"]
            if entry["total_seconds"] is None:
                entry["total_seconds"] = best.get("total_seconds") or best.get(
                    "runtime_seconds"
                )
            entry["started_utc"] = entry["started_utc"] or best.get("created_utc")
            entry["finished_utc"] = entry["finished_utc"] or best.get("finished_utc")
        if name in modal_results:
            entry["modal"] = modal_results[name]
            value = modal_results[name]
            if isinstance(value, dict) and value.get("elapsed_seconds"):
                entry["total_seconds"] = (
                    entry["total_seconds"] or value["elapsed_seconds"]
                )
        runs.append(entry)
    write_json(
        args.output / "results.json",
        {
            "collected_utc": utc_now(),
            "backend": launch.get("backend") or plan["backend"],
            "launch": launch,
            "plan": plan,
            "runs": runs,
        },
    )
    finished = sum(1 for r in runs if r["val_loss"] is not None)
    print(f"collected {finished}/{len(runs)} finished runs")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mode", required=True, choices=["modal", "local", "local-one", "collect"]
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", default="H100")
    parser.add_argument("--max-parallel", type=int, default=0)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    plan = read_json(args.plan)
    if args.mode == "modal":
        mode_modal(args, plan)
    elif args.mode == "local":
        mode_local(args, plan)
    elif args.mode == "local-one":
        mode_local_one(args, plan)
    else:
        mode_collect(args, plan)


if __name__ == "__main__":
    main()

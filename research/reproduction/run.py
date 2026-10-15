"""Run separate residual-path studies with saved recipes and bounded smoke tests."""

import argparse
import ast
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import uuid

REPO = Path(__file__).resolve().parents[2]
STUDIES = (
    "mnist",
    "null",
    "layerrope",
    "tinystories",
    "depth",
    "confirmation",
    "cache",
    "gsm8k",
)


def configuration(path, overrides=()):
    values = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("CONFIG_") and "=" in line:
            name, raw = line.split("=", 1)
            values[name] = ast.literal_eval(raw) if raw.startswith('"') else raw
    for override in overrides:
        name, raw = override.split("=", 1)
        allowed = {
            "STUDY",
            "SMOKE",
            "DEVICE",
            "DATA_DIR",
            "OUTPUT_DIR",
            "CHECKPOINT_DIR",
            "CPU_PYTHON",
            "SEEDS",
            "MAX_SECONDS",
            "CPU_THREADS",
            "KV_CACHE",
            "CACHE_DEPTH_RUN",
            "CACHE_CONFIRMATION_RUN",
        }
        if name not in allowed:
            raise ValueError("Unknown configuration override: " + name)
        values["CONFIG_RESIDUAL_" + name] = raw
    if values.get("CONFIG_RESIDUAL_REPRODUCTION") != "y":
        raise ValueError("Select a residual research configuration")
    get = lambda name, default: values.get("CONFIG_RESIDUAL_" + name, default)
    if get("SMOKE", "n") not in ("y", "n"):
        raise ValueError("SMOKE must be y or n")
    result = {
        "study": get("STUDY", "mnist"),
        "smoke": get("SMOKE", "n") == "y",
        "device": get("DEVICE", "cuda"),
        "data_dir": get("DATA_DIR", "~/.cache/knlp/residual-data"),
        "output_dir": get("OUTPUT_DIR", "~/knlp-residual-runs"),
        "checkpoint_dir": get("CHECKPOINT_DIR", ""),
        "cpu_python": get("CPU_PYTHON", sys.executable),
        "seeds": [int(s) for s in get("SEEDS", "").replace(",", " ").split()],
        "max_seconds": int(get("MAX_SECONDS", 86400)),
        "cpu_threads": int(get("CPU_THREADS", 8)),
        "kv_cache": get("KV_CACHE", "y") == "y",
        "cache_depth_run": get("CACHE_DEPTH_RUN", ""),
        "cache_confirmation_run": get("CACHE_CONFIRMATION_RUN", ""),
    }
    if result["study"] not in STUDIES or result["device"] not in ("cpu", "cuda"):
        raise ValueError("Unknown study or device")
    if result["max_seconds"] < 0 or not 1 <= result["cpu_threads"] <= 64:
        raise ValueError("Invalid resource limit")
    if result["seeds"] and result["study"] not in ("null", "layerrope"):
        raise ValueError("Aggregate and confirmation seed sets cannot be replaced")
    return result


def plan(config):
    from research.subliminal_learning.layerrope_study import ARMS
    from research.mo_variants.stage_c_common import ROWS, SEEDS

    return {
        "effective_config": config,
        "scope": "smoke" if config["smoke"] else "scientific reproduction",
        "scientific_confirmation_allowed": not config["smoke"],
        "mnist": {
            "reference_models": 100,
            "historical_cohort_seeds": list(range(2101, 2121)),
            "loss_modes": {
                "historical": "legacy_kl_v1",
                "repaired": "exact_cancel_v1",
            },
            "layerrope_seeds": config["seeds"] or list(range(74001, 74011)),
            "layerrope_arms": list(ARMS),
            "layerrope_student_epochs": 80,
            "null_student_epochs": 80,
            "repaired_control_seeds": list(range(73001, 73006)),
            "auxiliary_outputs": 3,
        },
        "tinystories": {
            "kv_cache": {
                "enabled": config["kv_cache"],
                "cases_per_checkpoint": 49,
                "depth_cohort": "21 saved endpoints, three seeds, 2000 updates; R/M depths 4/16/32 plus M-O at 32",
                "confirmation_cohort": "56 saved endpoints, eight seeds, 4000 updates; seven mHC-derived rows",
                "pass_rule": "Every seed adds <=0.1 nats over its own clean CE",
                "source_depth_run": config["cache_depth_run"],
                "source_confirmation_run": config["cache_confirmation_run"],
                "training": "none in the cache-only preset",
            },
            "depth_screen": json.loads(
                (REPO / "research/reproduction/plans/depth-screen.json").read_text()
            ),
            "confirmation": {
                "rows": [r for r, _, _ in ROWS],
                "seeds": list(SEEDS),
                "depth": 32,
                "width": 128,
                "steps": 4000,
                "predicted_tokens_per_update": 2048,
                "lr": 0.003,
                "primary_steps": list(range(2000, 4001, 100)),
                "checkpoints": [1000, 2000, 3000, 4000],
                "gpu_count": 2,
                "D_enabled": False,
            },
        },
    }


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def gpu_slots(environment, count=2):
    """Select workers within the caller's visible device allocation."""
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    slots = (
        [part.strip() for part in visible.split(",") if part.strip()]
        if visible is not None
        else [str(index) for index in range(count)]
    )
    if len(slots) < count:
        raise RuntimeError("The caller's GPU allocation is too small")
    return slots[:count]


class Runner:
    def __init__(self, config, root):
        self.config, self.root = config, Path(root)
        self.started = time.monotonic()
        self.operations = []
        self.lock = threading.Lock()
        self.env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                filter(None, (str(REPO), os.environ.get("PYTHONPATH", "")))
            ),
            "OMP_NUM_THREADS": str(config["cpu_threads"]),
            "MKL_NUM_THREADS": str(config["cpu_threads"]),
            "MPLBACKEND": "Agg",
            "RAYON_NUM_THREADS": str(config["cpu_threads"]),
        }

    def command(self, module, *args, cpu=False, env=None):
        elapsed = time.monotonic() - self.started
        limit = self.config["max_seconds"]
        if limit and elapsed >= limit:
            raise TimeoutError("The configured elapsed-time limit was reached")
        command = [
            self.config["cpu_python"] if cpu else sys.executable,
            "-m",
            module,
            *map(str, args),
        ]
        entry = {"command": command, "status": "running"}
        with self.lock:
            number = len(self.operations)
            self.operations.append(entry)
            write(self.root / "OPERATIONS.json", self.operations)
        before = time.monotonic()
        try:
            with (self.root / "logs" / f"operation-{number:03}.log").open("w") as log:
                subprocess.run(
                    command,
                    cwd=REPO,
                    env={**self.env, **(env or {})},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=max(1, limit - elapsed) if limit else None,
                )
            entry["status"] = "complete"
        except BaseException as error:
            entry.update(status="failed", error=str(error))
            raise
        finally:
            entry["seconds"] = time.monotonic() - before
            with self.lock:
                write(self.root / "OPERATIONS.json", self.operations)

    def mnist(self):
        study, smoke = self.config["study"], self.config["smoke"]
        if smoke:
            self.command(
                "research.reproduction.smoke",
                "--root",
                self.root,
                "--kind",
                "mnist",
                "--device",
                self.config["device"],
                "--data-dir",
                self.config["data_dir"],
            )
            return
        args = [
            "--root",
            self.root / "layerrope",
            "--device",
            self.config["device"],
            "--data-dir",
            self.config["data_dir"],
            "--checkpoint-root",
            self.config["checkpoint_dir"] or self.root / "checkpoints",
            "--seeds",
            ",".join(map(str, self.config["seeds"] or range(74001, 74011))),
        ]
        # Qualification uses separate disposable teachers and students.
        self.command("research.subliminal_learning.layerrope_study", "qualify", *args)
        if study in ("mnist", "null"):
            self.command(
                "research.reproduction.mnist",
                "--root",
                self.root,
                "--config-json",
                self.root / "EFFECTIVE.json",
            )
        if study in ("mnist", "layerrope"):
            self.command("research.subliminal_learning.layerrope_study", "run", *args)

    def tinystories(self):
        if self.config["smoke"]:
            self.command(
                "research.reproduction.smoke",
                "--root",
                self.root,
                "--kind",
                "tinystories",
                "--device",
                self.config["device"],
            )
            if self.config["kv_cache"]:
                self.cache("depth", smoke=True)
            return
        if self.config["device"] != "cuda":
            raise ValueError(
                "Full TinyStories trajectories require an explicit CUDA/ROCm device"
            )
        if self.config["study"] in ("confirmation", "tinystories"):
            self.command("research.reproduction.devices", "--count", 2, "--h100")
        self.command(
            "research.reproduction.data",
            "--root",
            self.root,
            "--cache",
            self.config["data_dir"],
            cpu=True,
        )
        if self.config["study"] in ("depth", "tinystories"):
            self.command(
                "research.reproduction.depth",
                "--root",
                self.root,
                "--config-json",
                self.root / "EFFECTIVE.json",
            )
            if self.config["kv_cache"]:
                self.cache("depth", self.root)
        if self.config["study"] in ("confirmation", "tinystories"):
            self.confirmation()
            if self.config["kv_cache"]:
                self.cache("confirmation", self.root)

    def cache(self, cohort, source=None, smoke=False):
        args = [
            "--root",
            self.root / ("cache-" + cohort),
            "--cohort",
            cohort,
            "--device",
            self.config["device"],
            "--max-seconds",
            self.config["max_seconds"],
        ]
        if smoke:
            args.append("--smoke")
        else:
            args.extend(("--input-run", source))
        self.command("research.reproduction.kv_cache", *args)

    def saved_cache(self):
        if self.config["smoke"]:
            self.cache("depth", smoke=True)
            return
        for cohort, field in (
            ("depth", "cache_depth_run"),
            ("confirmation", "cache_confirmation_run"),
        ):
            if self.config[field]:
                self.cache(cohort, Path(self.config[field]).expanduser().resolve())

    def gsm8k(self):
        if self.config["smoke"]:
            raise ValueError("Use residual-test for bounded CPU generation checks")
        self.command(
            "research.reproduction.gsm8k",
            "--root",
            self.root / "gsm8k",
            "--source-run",
            self.config["cache_depth_run"],
            "--cache",
            self.config["data_dir"],
            "--device",
            self.config["device"],
        )

    def confirmation(self):
        from research.mo_variants.stage_c_common import read, sha

        slots = gpu_slots(self.env)

        root = self.root / "confirmation"
        self.command(
            "research.reproduction.prepare",
            "--root",
            root,
            "--data",
            self.root / "data",
            *(
                ["--checkpoints", self.config["checkpoint_dir"]]
                if self.config["checkpoint_dir"]
                else []
            ),
            cpu=True,
        )
        self.command("research.mo_variants.stage_c_prepare", "--root", root, cpu=True)
        frozen = read(root / "records/STAGE-C-PLAN.json")
        write(
            root / "records/STAGE-C-RESOURCE-DECISION.json",
            {
                "status": "approved",
                "scope": "C",
                "invoked_by_user": True,
                "plan_sha256": sha(root / "records/STAGE-C-PLAN.json"),
                "effective_config": self.config,
            },
        )
        self.command(
            "research.mo_variants.stage_c_fixtures",
            "--root",
            root,
            "--device",
            "cpu",
            cpu=True,
        )
        self.command("research.reproduction.devices", "--count", 2, "--h100")
        self.command(
            "research.mo_variants.stage_c_fixtures",
            "--root",
            root,
            "--device",
            "cuda",
            env={"CUDA_VISIBLE_DEVICES": slots[0]},
        )
        cpu_futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as cpu_pool:

            def endpoints(jid):
                for module in ("stage_c_numerics", "stage_c_quantize"):
                    self.command(
                        "research.mo_variants." + module,
                        "--root",
                        root,
                        "--job-id",
                        jid,
                        cpu=True,
                        env={"CUDA_VISIBLE_DEVICES": ""},
                    )

            def train(index):
                for job in frozen["jobs"]:
                    if job["worker"] != index:
                        continue
                    jid = job["id"]
                    self.command(
                        "research.mo_variants.stage_c_job",
                        "--root",
                        root,
                        "--job-id",
                        jid,
                        env={"CUDA_VISIBLE_DEVICES": slots[index]},
                    )
                    result = read(root / "records/runs" / jid / "RESULT.json")
                    if result["status"] != "complete":
                        raise RuntimeError("Incomplete confirmation trajectory: " + jid)
                    for checkpoint in result["checkpoints"]:
                        if sha(checkpoint["path"]) != checkpoint["sha256"]:
                            raise RuntimeError("Checkpoint integrity failure: " + jid)
                    cpu_futures.append(cpu_pool.submit(endpoints, jid))

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as gpu_pool:
                tasks = [gpu_pool.submit(train, index) for index in range(2)]
                for task in tasks:
                    task.result()
            for task in cpu_futures:
                task.result()
        self.command("research.mo_variants.stage_c_analyze", "--root", root, cpu=True)


def execute(config):
    import torch

    if config["study"] == "gsm8k" and not config["cache_depth_run"]:
        raise ValueError("Set CACHE_DEPTH_RUN to a saved depth reproduction run")
    if (
        config["study"] == "cache"
        and not config["smoke"]
        and not (config["cache_depth_run"] or config["cache_confirmation_run"])
    ):
        raise ValueError(
            "Set CACHE_DEPTH_RUN and/or CACHE_CONFIRMATION_RUN to saved reproduction run directories"
        )
    parent = Path(config["output_dir"]).expanduser().resolve()
    if parent.is_relative_to(REPO):
        raise ValueError(
            "Set OUTPUT_DIR outside the source checkout; generated artifacts do not belong in knlp"
        )
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA/ROCm was requested but is unavailable; select cpu explicitly"
        )
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = parent / f"{config['study']}-{stamp}-{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    write(root / "EFFECTIVE.json", config)
    write(root / "PLAN.json", plan(config))
    runner = Runner(config, root)
    report = {
        "status": "running",
        "scope": "smoke" if config["smoke"] else "reproduction",
        "scientific_confirmation_allowed": False,
        "device_requested": config["device"],
        "output": str(root),
    }
    write(root / "REPORT.json", report)
    try:
        if config["study"] in ("mnist", "null", "layerrope"):
            runner.mnist()
        elif config["study"] == "cache":
            runner.saved_cache()
        elif config["study"] == "gsm8k":
            runner.gsm8k()
        else:
            runner.tinystories()
        report["status"] = "complete"
        report["scientific_confirmation_allowed"] = not config["smoke"] and config[
            "study"
        ] not in ("cache", "gsm8k")
    except BaseException as error:
        report.update(status="incomplete", error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - runner.started
        report["operations"] = runner.operations
        write(root / "REPORT.json", report)
        print(json.dumps(report), flush=True)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument("--config", type=Path, default=REPO / ".config")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = configuration(args.config, args.set)
    if args.action == "plan":
        print(json.dumps(plan(config), indent=2))
    else:
        execute(config)


if __name__ == "__main__":
    main()

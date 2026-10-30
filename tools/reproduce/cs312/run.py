#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan, launch and report the CS312 Assignment 1 sweeps.

The course repository is an external checkout pinned by commit. knlp
never copies it: this runner prepares the checkout and its environment,
writes a plan, and drives the course's own launchers through
environment variables and a small entry script executed inside the
course environment. Everything here is plain Python without torch so
that planning and reporting work on any machine.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PREFIX = "CONFIG_KNLP_CS312_"
ENV_PREFIX = "CS312_"

DEFAULTS = {
    "ASSIGNMENTS_DIR": "~/.cache/knlp/cs312/assignments",
    "ASSIGNMENTS_URL": "https://github.com/deep-learning-alchemy/assignments.git",
    "ASSIGNMENTS_REV": "4350166ac1cedba0222a3124a2a2a54b65fb635d",
    "PROBLEM": "p1a",
    "P1A_BASE": "3",
    "P1A_STEPS": "5",
    "P1A_BATCH_BASE": "2",
    "P1A_AXES": "warmup_percent,learning_rate,batch_size,weight_decay",
    "P1B_PAIRS": (
        "learning_rate:batch_size,learning_rate:weight_decay,"
        "learning_rate:warmup_percent,batch_size:warmup_percent,"
        "batch_size:weight_decay,weight_decay:warmup_percent"
    ),
    "P1B_FULL_CORNERS": "n",
    "P1B_SEED_REPEATS": "2",
    "REFERENCE_OUTPUT": "",
    "DEPTH": "8",
    "NUM_TRAIN_SEQUENCES": "600000",
    "BACKEND": "modal",
    "MODAL_GPU": "H100",
    "MODAL_ENVIRONMENT": "main",
    "MODAL_MAX_PARALLEL": "0",
    "MODAL_DATA_VOLUME": "hard-dl-dclm-v1",
    "MODAL_EXTRA_SECRETS": "",
    "LOCAL_PARALLEL": "1",
    "PYTHON": "",
    "SCRATCH_ROOT": "",
    "TORCH_COMPILE_MODE": "",
    "NUM_EVALS": "0",
    "WANDB_ONLINE": "y",
    "WANDB_ENTITY": "",
    "WANDB_PROJECT": "cs312",
    "RUN_SUFFIX": "",
    "OUTPUT": "results/cs312-problem-1",
    "ONLY": "",
    "PURGE": "n",
    "KVQ_FRONTIER_LEVELS": "K4V16,K4V4,K2V2",
    "KVQ_GRANS": "per_token,per_tensor",
    "KVQ_QKNORM": "on,off",
    "KVQ_STRESS": "",
    "KVQ_PHASE": "frontier",
    "KVQ_HP_WARMUP": "0.03,0.09",
    "KVQ_HP_LR": "0.001,0.009",
    "KVQ_HP_WD": "0.3,0.9",
    "KVQ_SEED_REPEATS": "2",
    "KVQ_POSTHOC": "",
}

# Course defaults for the swept axes at the pinned revision. The entry
# script checks them against the checkout before any launch.
AXIS_DEFAULTS = {
    "warmup_percent": 0.01,
    "learning_rate": 0.003,
    "batch_size": 64,
    "weight_decay": 0.1,
}
INTEGER_AXES = {"batch_size"}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def settings(config_path):
    """Merge defaults, the Kconfig .config and CS312_* overrides."""
    result = dict(DEFAULTS)
    path = Path(config_path)
    if path.is_file():
        for line in path.read_text().splitlines():
            if not line.startswith(PREFIX) or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key[len(PREFIX) :]
            value = value.strip().strip('"')
            if key == "BACKEND_MODAL" and value == "y":
                result["BACKEND"] = "modal"
            elif key == "BACKEND_LOCAL" and value == "y":
                result["BACKEND"] = "local"
            elif key.startswith("PROBLEM_") and value == "y":
                result["PROBLEM"] = key[len("PROBLEM_") :].lower()
            else:
                result[key] = value
    for key in list(result):
        if ENV_PREFIX + key in os.environ:
            result[key] = os.environ[ENV_PREFIX + key]
    if result["BACKEND"] not in ("modal", "local"):
        raise SystemExit("CS312_BACKEND must be modal or local")
    if not result["RUN_SUFFIX"]:
        result["RUN_SUFFIX"] = result["BACKEND"]
    return result


def log_grid(default, base, steps, integer=False):
    """Return steps values spaced by base around default, default included."""
    half = steps // 2
    values = []
    for exponent in range(-half, steps - half):
        value = default * base**exponent
        if integer:
            value = max(1, int(round(value)))
        else:
            value = float(f"{value:.6g}")
        values.append((exponent, value))
    return values


def plan_p1a(cfg):
    base = float(cfg["P1A_BASE"])
    batch_base = float(cfg["P1A_BATCH_BASE"])
    steps = int(cfg["P1A_STEPS"])
    axes = [axis.strip() for axis in cfg["P1A_AXES"].split(",") if axis.strip()]
    unknown = [axis for axis in axes if axis not in AXIS_DEFAULTS]
    if unknown:
        raise SystemExit(f"unknown sweep axes: {unknown}")
    grids = {}
    for axis in axes:
        grids[axis] = log_grid(
            AXIS_DEFAULTS[axis],
            batch_base if axis == "batch_size" else base,
            steps,
            integer=axis in INTEGER_AXES,
        )
    points = []
    for axis_index, axis in enumerate(axes):
        for exponent, value in grids[axis]:
            points.append((abs(exponent), axis_index, exponent, axis, value))
    points.sort()
    runs = []
    seen = {}
    for _, _, exponent, axis, value in points:
        overrides = {} if exponent == 0 else {axis: value}
        key = json.dumps(overrides, sort_keys=True)
        if key in seen:
            seen[key]["axes"].append(axis)
            continue
        entry = {
            "index": len(runs),
            "axis": "baseline" if exponent == 0 else axis,
            "axes": [axis],
            "exponent": exponent,
            "overrides": overrides,
        }
        seen[key] = entry
        runs.append(entry)
    return {
        "axes": axes,
        "axis_defaults": {axis: AXIS_DEFAULTS[axis] for axis in axes},
        "axis_values": {axis: [value for _, value in grids[axis]] for axis in axes},
        "runs": runs,
    }


# Problem 1(b): which pairs co-vary. Cells are offsets on the 1(a) grid
# (base 3 for the continuous axes, base 2 for batch size); the shared
# single-axis cells come from the 1(a) plan through REFERENCE_OUTPUT.
# The trimmed design puts the budget where theory predicts compensation
# and keeps two anti-corners per pair as controls; the expected weak
# pairs get two cells. Full corners is the plain 2x2 around the baseline.
FULL_CORNERS = [(1, 1), (-1, -1), (1, -1), (-1, 1)]
PAIR_DESIGN = {
    ("learning_rate", "batch_size"): [
        (1, 1),
        (-1, -1),
        (1, -1),
        (-1, 1),
        (1, 2),
        (2, 2),
    ],
    ("learning_rate", "weight_decay"): [(1, -1), (-1, 1), (1, 1), (-1, -1)],
    ("learning_rate", "warmup_percent"): [(1, 1), (1, 2), (2, 2), (-1, -1)],
    ("batch_size", "warmup_percent"): [(2, 1), (2, 2), (-2, -1)],
    ("batch_size", "weight_decay"): [(1, 1), (-1, -1)],
    ("weight_decay", "warmup_percent"): [(1, 1), (-1, -1)],
}
SEED_REPEAT_START = 43  # the course default seed is 42


def grid_value(cfg, axis, exponent):
    base = float(cfg["P1A_BATCH_BASE"] if axis == "batch_size" else cfg["P1A_BASE"])
    return log_grid(
        AXIS_DEFAULTS[axis], base, 1 + 2 * abs(exponent), integer=axis in INTEGER_AXES
    )[exponent + abs(exponent)][1]


def plan_p1b(cfg):
    pairs = []
    for item in cfg["P1B_PAIRS"].split(","):
        item = item.strip()
        if not item:
            continue
        first, second = item.split(":")
        pair = (first.strip(), second.strip())
        if pair not in PAIR_DESIGN and pair[::-1] in PAIR_DESIGN:
            pair = pair[::-1]
        if pair not in PAIR_DESIGN:
            raise SystemExit(f"unknown pair {item!r}")
        pairs.append(pair)
    full = cfg["P1B_FULL_CORNERS"] == "y"
    runs, seen = [], set()
    for pair in pairs:
        first, second = pair
        for ea, eb in FULL_CORNERS if full else PAIR_DESIGN[pair]:
            overrides = {
                first: grid_value(cfg, first, ea),
                second: grid_value(cfg, second, eb),
            }
            key = json.dumps(overrides, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            runs.append(
                {
                    "index": len(runs),
                    "axis": f"{first}x{second}",
                    "pair": [first, second],
                    "exponents": {first: ea, second: eb},
                    "exponent": 0,
                    "overrides": overrides,
                }
            )
    for i in range(int(cfg["P1B_SEED_REPEATS"])):
        seed = SEED_REPEAT_START + i
        runs.append(
            {
                "index": len(runs),
                "axis": "seed",
                "pair": None,
                "exponents": {},
                "exponent": 0,
                "overrides": {"model_seed": seed, "data_seed": seed},
            }
        )
    axes = sorted(
        {axis for pair in pairs for axis in pair}, key=list(AXIS_DEFAULTS).index
    )
    return {
        "axes": axes,
        "axis_defaults": {axis: AXIS_DEFAULTS[axis] for axis in axes},
        "axis_values": {
            axis: [grid_value(cfg, axis, e) for e in range(-2, 3)] for axis in axes
        },
        "pairs": [list(pair) for pair in pairs],
        "runs": runs,
    }


QUANT_BUILDER = "cs312_quant:build_quant_model"
_GRAN_SHORT = {"per_token": "pt", "per_tensor": "ptns"}
_QK = {"on": True, "off": False}


def _parse_level(label):
    m = re.fullmatch(r"K(\d+)V(\d+)", label.strip())
    if not m:
        raise SystemExit(f"bad quant level {label!r}; expected like K4V8")
    return int(m.group(1)), int(m.group(2))


def _kvq_overrides(level, gran, qk_norm, hp=None):
    k, v = _parse_level(level)
    ov = {
        "model_builder": QUANT_BUILDER,
        "model_builder_kwargs": {
            "k_bits": k,
            "v_bits": v,
            "k_format": "e4m3",
            "v_format": "e4m3",
            "granularity": gran,
        },
        "qk_norm": qk_norm,
    }
    if hp:
        ov.update(hp)
    return ov


def plan_kvq(cfg):
    """KV-cache quantization-aware training.

    Frontier: a K/V ladder crossed with granularity (per_token, per_tensor)
    and qk_norm (on, off), to find where quantization genuinely breaks d8.
    Sweep: at a chosen stress cell (KVQ_STRESS = level:gran:qk), vary
    warmup, learning rate and weight decay one at a time, plus baseline
    seed repeats for the noise floor.
    """
    grans = [g.strip() for g in cfg["KVQ_GRANS"].split(",") if g.strip()]
    qks = [q.strip() for q in cfg["KVQ_QKNORM"].split(",") if q.strip()]
    phase = cfg["KVQ_PHASE"]
    runs, seen = [], set()

    def add(axis, level, gran, qk_norm, hp=None):
        ov = _kvq_overrides(level, gran, qk_norm, hp)
        key = json.dumps(ov, sort_keys=True)
        if key in seen:
            return
        seen.add(key)
        runs.append(
            {
                "index": len(runs),
                "axis": axis,
                "level": level,
                "granularity": gran,
                "qk_norm": qk_norm,
                "exponent": 0,
                "overrides": ov,
                "suffix": f"{level}-{_GRAN_SHORT[gran]}-{cfg['RUN_SUFFIX']}",
            }
        )

    levels = [l.strip() for l in cfg["KVQ_FRONTIER_LEVELS"].split(",") if l.strip()]
    if phase in ("frontier", "both"):
        for level in levels:
            for gran in grans:
                for qk in qks:
                    add("frontier", level, gran, _QK[qk])

    stress = cfg["KVQ_STRESS"].strip()
    if phase in ("sweep", "both"):
        if not stress:
            raise SystemExit(
                "KVQ_STRESS must be set for the sweep phase (level:gran:qk)"
            )
        slevel, sgran, sqk = (part.strip() for part in stress.split(":"))
        sqk_b = _QK[sqk]
        add("stress-center", slevel, sgran, sqk_b)
        for w in cfg["KVQ_HP_WARMUP"].split(","):
            add("warmup", slevel, sgran, sqk_b, {"warmup_percent": float(w)})
        for lr in cfg["KVQ_HP_LR"].split(","):
            add("learning_rate", slevel, sgran, sqk_b, {"learning_rate": float(lr)})
        for wd in cfg["KVQ_HP_WD"].split(","):
            add("weight_decay", slevel, sgran, sqk_b, {"weight_decay": float(wd)})
        for i in range(int(cfg["KVQ_SEED_REPEATS"])):
            seed = 43 + i
            add("seed", slevel, sgran, sqk_b, {"model_seed": seed, "data_seed": seed})

    return {
        "grans": grans,
        "qknorms": [q.strip() for q in qks],
        "levels": levels,
        "stress": stress,
        "phase": phase,
        "runs": runs,
    }


def plan_digest(problem, depth, num_train_sequences, runs):
    payload = {
        "problem": problem,
        "depth": depth,
        "num_train_sequences": num_train_sequences,
        "runs": [run["overrides"] for run in runs],
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def build_plan(cfg):
    problem = cfg["PROBLEM"]
    planners = {"p1a": plan_p1a, "p1b": plan_p1b, "kvq": plan_kvq}
    if problem not in planners:
        raise SystemExit(f"problem {problem!r} is not implemented")
    depth = int(cfg["DEPTH"])
    num_train_sequences = int(cfg["NUM_TRAIN_SEQUENCES"])
    body = planners[problem](cfg)
    digest = plan_digest(problem, depth, num_train_sequences, body["runs"])
    tags = [
        "cs312",
        f"cs312-{problem}",
        f"cs312-plan-{digest}",
        f"cs312-{cfg['RUN_SUFFIX']}",
    ]
    for run in body["runs"]:
        train_config = {
            "model_name": f"d{depth}",
            "num_train_sequences": num_train_sequences,
            "wandb_project": cfg["WANDB_PROJECT"],
            "wandb_online": cfg["WANDB_ONLINE"] == "y",
            "wandb_tags": tags,
            "run_name_suffix": cfg["RUN_SUFFIX"],
        }
        if cfg["TORCH_COMPILE_MODE"]:
            train_config["torch_compile_mode"] = cfg["TORCH_COMPILE_MODE"]
        if int(cfg["NUM_EVALS"]) > 0:
            train_config["num_evals"] = int(cfg["NUM_EVALS"])
        train_config.update(run["overrides"])
        if run.get("suffix"):
            train_config["run_name_suffix"] = run["suffix"]
        run["train_config"] = train_config
    public = {key: cfg[key] for key in cfg if key not in ("WANDB_ENTITY",)}
    return {
        "problem": problem,
        "digest": digest,
        "created_utc": utc_now(),
        "depth": depth,
        "num_train_sequences": num_train_sequences,
        "backend": cfg["BACKEND"],
        "run_suffix": cfg["RUN_SUFFIX"],
        "reference_output": cfg["REFERENCE_OUTPUT"],
        "settings": public,
        **body,
    }


def write_json(path, payload, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with path.open(mode) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def output_dir(cfg):
    path = Path(os.path.expanduser(cfg["OUTPUT"]))
    return path if path.is_absolute() else ROOT / path


def assignments_dir(cfg):
    return Path(os.path.expanduser(cfg["ASSIGNMENTS_DIR"])).resolve()


def uv_binary():
    found = shutil.which("uv") or str(Path("~/.local/bin/uv").expanduser())
    if not Path(found).exists():
        raise SystemExit("uv is required; see https://docs.astral.sh/uv/")
    return found


def course_python(cfg):
    if cfg["PYTHON"]:
        return [os.path.expanduser(cfg["PYTHON"])]
    return [uv_binary(), "run", "--project", str(assignments_dir(cfg)), "python"]


def course_env(cfg):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(assignments_dir(cfg))
    env["DL_ALCHEMY_MODAL_ENVIRONMENT"] = cfg["MODAL_ENVIRONMENT"]
    env["DL_ALCHEMY_MODAL_SHARED_DATA_ENVIRONMENT"] = cfg["MODAL_ENVIRONMENT"]
    env["DL_ALCHEMY_MODAL_SHARED_DATA_VOLUME_NAME"] = cfg["MODAL_DATA_VOLUME"]
    env["DL_ALCHEMY_MODAL_APP_NAME"] = "cs312"
    if cfg["MODAL_EXTRA_SECRETS"]:
        env["DL_ALCHEMY_MODAL_SECRETS"] = cfg["MODAL_EXTRA_SECRETS"]
    env["WANDB_PROJECT"] = cfg["WANDB_PROJECT"]
    if cfg["WANDB_ENTITY"]:
        env["WANDB_ENTITY"] = cfg["WANDB_ENTITY"]
    if cfg["SCRATCH_ROOT"]:
        env["DL_ALCHEMY_SCRATCH_ROOT"] = os.path.expanduser(cfg["SCRATCH_ROOT"])
    env["CS312_ASSIGNMENTS_DIR"] = str(assignments_dir(cfg))
    env["CS312_MODAL_DATA_VOLUME"] = cfg["MODAL_DATA_VOLUME"]
    return env


def run_command(argv, cfg, log=None, check=True):
    printable = " ".join(argv)
    print(f"+ {printable}", flush=True)
    kwargs = {"cwd": str(assignments_dir(cfg)), "env": course_env(cfg)}
    if log:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        with Path(log).open("a") as stream:
            stream.write(f"# {utc_now()} {printable}\n")
            stream.flush()
            completed = subprocess.run(
                argv, stdout=stream, stderr=subprocess.STDOUT, **kwargs
            )
    else:
        completed = subprocess.run(argv, **kwargs)
    if check and completed.returncode != 0:
        raise SystemExit(f"command failed ({completed.returncode}): {printable}")
    return completed.returncode


def git(directory, *args, capture=False):
    argv = ["git", "-C", str(directory), *args]
    if capture:
        return subprocess.check_output(argv, text=True).strip()
    subprocess.run(argv, check=True)
    return None


def stage_quant_hook(cfg):
    """Drop the KV-quant hook into the course checkout as an untracked file.

    model_builder resolves cs312_quant inside the container and the local
    env; staging it this way keeps the pinned commit unchanged (untracked
    files do not move HEAD).
    """
    src = HERE / "cs312_quant.py"
    if not src.exists():
        return
    dst = assignments_dir(cfg) / "cs312_quant.py"
    import shutil as _sh

    _sh.copyfile(src, dst)
    print(f"staged KV-quant hook at {dst} (untracked)")


def cmd_fetch(cfg, _args):
    directory = assignments_dir(cfg)
    if not (directory / ".git").is_dir():
        directory.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", cfg["ASSIGNMENTS_URL"], str(directory)], check=True
        )
    git(directory, "fetch", "-q", "origin")
    git(directory, "checkout", "-q", "--detach", cfg["ASSIGNMENTS_REV"])
    head = git(directory, "rev-parse", "HEAD", capture=True)
    if head != cfg["ASSIGNMENTS_REV"]:
        raise SystemExit(f"checkout is at {head}, expected {cfg['ASSIGNMENTS_REV']}")
    print(f"course checkout {directory} at {head}")
    stage_quant_hook(cfg)
    if not cfg["PYTHON"]:
        subprocess.run([uv_binary(), "sync", "--project", str(directory)], check=True)
    run_command(
        course_python(cfg)
        + [
            "-c",
            "import torch; print('torch', torch.__version__, "
            "'cuda' if torch.cuda.is_available() else 'cpu')",
        ],
        cfg,
    )


def check_checkout(cfg):
    directory = assignments_dir(cfg)
    if not (directory / ".git").is_dir():
        raise SystemExit(f"{directory} is missing; run make cs312-fetch first")
    head = git(directory, "rev-parse", "HEAD", capture=True)
    if head != cfg["ASSIGNMENTS_REV"]:
        raise SystemExit(
            f"{directory} is at {head}, expected {cfg['ASSIGNMENTS_REV']}; "
            "run make cs312-fetch"
        )


def cmd_plan(cfg, _args):
    plan = build_plan(cfg)
    path = output_dir(cfg) / "plan.json"
    if path.exists():
        existing = read_json(path)
        if existing["digest"] != plan["digest"]:
            raise SystemExit(
                f"{path} holds plan {existing['digest']}, new plan is "
                f"{plan['digest']}; choose another output directory"
            )
        plan = existing
        print(f"plan {plan['digest']} already written to {path}")
    else:
        write_json(path, plan)
        print(f"plan {plan['digest']} written to {path}")
    print_plan(plan)
    return plan


def print_plan(plan):
    print(
        f"problem {plan['problem']} d{plan['depth']} "
        f"{plan['num_train_sequences']} sequences, {len(plan['runs'])} runs, "
        f"backend {plan['backend']}, suffix {plan['run_suffix']}"
    )
    for axis in plan.get("axes", []):
        print(f"  {axis}: {plan['axis_values'][axis]}")
    for run in plan["runs"]:
        print(f"  [{run['index']:2d}] {run['axis']:>28s} {run['overrides']}")
    if plan.get("reference_output"):
        print(f"  shared single-axis cells come from {plan['reference_output']}")


def ensure_plan(cfg):
    path = output_dir(cfg) / "plan.json"
    if path.exists():
        return read_json(path)
    return cmd_plan(cfg, None)


def entry_command(cfg, mode, plan_path, extra=()):
    return course_python(cfg) + [
        str(HERE / "sweep_entry.py"),
        "--mode",
        mode,
        "--plan",
        str(plan_path),
        "--output",
        str(output_dir(cfg)),
        *extra,
    ]


def cmd_data(cfg, args):
    check_checkout(cfg)
    log = output_dir(cfg) / f"data-{cfg['BACKEND']}.log"
    if cfg["BACKEND"] == "modal":
        modal = (
            course_python(cfg)[:-1] + ["modal"]
            if not cfg["PYTHON"]
            else (course_python(cfg) + ["-m", "modal"])
        )
        argv = modal + [
            "run",
            "--env",
            cfg["MODAL_ENVIRONMENT"],
            str(HERE / "modal_seed_data.py"),
        ]
        if args.force:
            argv.append("--force")
    else:
        argv = course_python(cfg) + ["-m", "download_data"]
        if args.force:
            argv.append("--force")
    run_command(argv, cfg, log=log)
    print(f"data step finished; log at {log}")


def cmd_launch(cfg, _args):
    check_checkout(cfg)
    plan = ensure_plan(cfg)
    out = output_dir(cfg)
    plan_path = out / "plan.json"
    if cfg["BACKEND"] == "modal":
        if (out / "launch.json").exists():
            raise SystemExit(
                f"{out / 'launch.json'} exists; collect it or choose a new output"
            )
        extra = [
            "--gpu",
            cfg["MODAL_GPU"],
            "--max-parallel",
            cfg["MODAL_MAX_PARALLEL"],
        ]
        if cfg["ONLY"]:
            # Relaunch only these plan indices, for cells that failed.
            extra += ["--only", cfg["ONLY"]]
            if cfg["PURGE"] == "y":
                extra.append("--purge")
        run_command(
            entry_command(cfg, "modal", plan_path, extra),
            cfg,
            log=out / "launch-modal.log",
        )
        receipt = read_json(out / "launch.json")
        launched = sum(1 for j in receipt["jobs"] if j.get("modal_call_id"))
        skipped = len(receipt["jobs"]) - launched
        print(
            f"launched {launched} Modal runs, skipped {skipped} with a saved "
            f"model; see {out / 'launch.json'}"
        )
    else:
        extra = ["--parallel", cfg["LOCAL_PARALLEL"]]
        run_command(
            entry_command(cfg, "local", plan_path, extra),
            cfg,
            log=out / "launch-local.log",
        )
        print(f"local sweep finished; receipts under {out / 'runs'}")


def cmd_collect(cfg, _args):
    check_checkout(cfg)
    plan_path = output_dir(cfg) / "plan.json"
    if not plan_path.exists():
        raise SystemExit(f"{plan_path} is missing; nothing to collect")
    run_command(entry_command(cfg, "collect", plan_path), cfg)
    results_path = output_dir(cfg) / "results.json"
    results = read_json(results_path)
    if cfg["REFERENCE_OUTPUT"]:
        ref_path = Path(os.path.expanduser(cfg["REFERENCE_OUTPUT"]))
        ref_path = (
            ref_path if ref_path.is_absolute() else ROOT / ref_path
        ) / "results.json"
        if not ref_path.exists():
            raise SystemExit(
                f"reference results {ref_path} are missing; run collect for that plan first"
            )
        reference = read_json(ref_path)
        results["reference_plan_digest"] = reference["plan"]["digest"]
        results["reference_runs"] = [
            {
                k: r.get(k)
                for k in (
                    "index",
                    "axis",
                    "overrides",
                    "run_name",
                    "val_loss",
                    "state",
                    "total_seconds",
                    "wandb",
                )
            }
            for r in reference["runs"]
        ]
        write_json(results_path, results, overwrite=True)
        print(
            f"merged {len(results['reference_runs'])} reference cells from {ref_path}"
        )
    print(f"results written to {results_path}")


def fmt(value, places=4):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{places}f}"
    return str(value)


def minutes(entry):
    seconds = entry.get("total_seconds")
    return None if seconds is None else seconds / 60.0


def result_rows(results):
    rows = {}
    for entry in results["runs"]:
        rows[entry["index"]] = entry
    return rows


def sweep_window(results):
    launch = results.get("launch") or {}
    starts, ends = [], []
    for entry in results["runs"]:
        if entry.get("started_utc"):
            starts.append(entry["started_utc"])
        if entry.get("finished_utc"):
            ends.append(entry["finished_utc"])
    start = (
        launch.get("launched_utc")
        or launch.get("started_utc")
        or (min(starts) if starts else None)
    )
    end = max(ends) if ends else None
    if start and end:
        delta = datetime.fromisoformat(end) - datetime.fromisoformat(start)
        return start, end, delta.total_seconds() / 60.0
    return start, end, None


def render_report(results, against=None):
    plan = results["plan"]
    lines = []
    lines.append(f"# CS312 problem {plan['problem']} report")
    lines.append("")
    lines.append(
        f"Plan `{plan['digest']}`: d{plan['depth']}, "
        f"{plan['num_train_sequences']:,} sequences per run, "
        f"{len(plan['runs'])} runs, backend `{results['backend']}`, "
        f"suffix `{plan['run_suffix']}`, collected {results['collected_utc']}."
    )
    device = results.get("launch", {}).get("gpu")
    if device:
        lines.append(f"Device: {device}.")
    start, end, span = sweep_window(results)
    done = sum(1 for e in results["runs"] if e.get("state") == "finished")
    scored = sum(1 for e in results["runs"] if e.get("val_loss") is not None)
    lines.append(
        f"Finished runs: {done}/{len(results['runs'])} "
        f"({scored} with a validation loss; unfinished rows show their latest evaluation). "
        f"Sweep window: {start or '-'} to {end or '-'}"
        + (f" ({span:.1f} min)." if span is not None else ".")
    )
    lines.append("")
    rows = result_rows(results)
    baseline = next(e for e in results["runs"] if e["axis"] == "baseline")
    for axis in plan["axes"]:
        lines.append(f"## {axis}")
        lines.append("")
        lines.append(
            "| value | x default | run | val loss | d vs default | minutes | state |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        entries = [baseline] + [e for e in results["runs"] if e["axis"] == axis]
        entries.sort(key=lambda e: e["exponent"])
        for e in entries:
            value = e["overrides"].get(axis, plan["axis_defaults"][axis])
            ratio = value / plan["axis_defaults"][axis]
            delta = (
                None
                if e.get("val_loss") is None or baseline.get("val_loss") is None
                else e["val_loss"] - baseline["val_loss"]
            )
            lines.append(
                f"| {value} | {ratio:g} | `{e['run_name']}` | {fmt(e.get('val_loss'))} | "
                f"{'' if delta is None or delta < 0 else '+'}{fmt(delta)} | "
                f"{fmt(minutes(e), 1)} | {e.get('state') or '-'} |"
            )
        lines.append("")
    if against:
        other_rows = result_rows(against)
        o_start, o_end, o_span = sweep_window(against)
        lines.append(
            f"## Comparison with backend `{against['backend']}` "
            f"(suffix `{against['plan']['run_suffix']}`)"
        )
        lines.append("")
        lines.append(
            f"Other device: {against.get('launch', {}).get('gpu') or '-'}; "
            f"other sweep window {o_start or '-'} to {o_end or '-'}"
            + (f" ({o_span:.1f} min)." if o_span is not None else ".")
        )
        lines.append("")
        lines.append(
            "| run | axis | this val loss | other val loss | diff | "
            "this min | other min | speedup |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for index in sorted(rows):
            mine = rows[index]
            other = other_rows.get(index)
            if other is None:
                continue
            diff = (
                None
                if mine.get("val_loss") is None or other.get("val_loss") is None
                else mine["val_loss"] - other["val_loss"]
            )
            m1, m2 = minutes(mine), minutes(other)
            speed = None if not m1 or not m2 else m2 / m1
            lines.append(
                f"| {index} | {mine['axis']} {mine['overrides']} | "
                f"{fmt(mine.get('val_loss'))} | {fmt(other.get('val_loss'))} | "
                f"{fmt(diff)} | {fmt(m1, 1)} | {fmt(m2, 1)} | "
                f"{fmt(speed, 2)} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _cell_key(overrides):
    return json.dumps(overrides, sort_keys=True)


def render_report_p1b(results):
    plan = results["plan"]
    defaults = plan["axis_defaults"]
    cells = {}
    for r in results.get("reference_runs", []):
        if r.get("val_loss") is not None:
            cells[_cell_key(r["overrides"])] = (r["val_loss"], "1a")
    for r in results["runs"]:
        if r.get("val_loss") is not None:
            cells[_cell_key(r["overrides"])] = (r["val_loss"], "1b")
    baseline = cells.get(_cell_key({}))
    base_loss = baseline[0] if baseline else None
    lines = [f"# CS312 problem {plan['problem']} report", ""]
    lines.append(
        f"Plan `{plan['digest']}`: d{plan['depth']}, {plan['num_train_sequences']:,} sequences per run, "
        f"{len(plan['runs'])} new runs, backend `{results['backend']}`, suffix `{plan['run_suffix']}`, "
        f"collected {results['collected_utc']}."
    )
    if results.get("reference_plan_digest"):
        lines.append(
            f"Single-axis cells and the baseline come from plan `{results['reference_plan_digest']}` (problem 1(a))."
        )
    done = sum(1 for e in results["runs"] if e.get("state") == "finished")
    start, end, span = sweep_window(results)
    lines.append(
        f"Finished runs: {done}/{len(results['runs'])}. Sweep window: {start or '-'} to {end or '-'}"
        + (f" ({span:.1f} min)." if span is not None else ".")
    )
    lines.append("")
    seeds = [
        r
        for r in results["runs"]
        if r["axis"] == "seed" and r.get("val_loss") is not None
    ]
    lines.append("## Noise floor: the baseline recipe at paired model and data seeds")
    lines.append("")
    lines.append("| seed | val loss | vs seed 42 |")
    lines.append("|---|---|---|")
    values = []
    if base_loss is not None:
        lines.append(f"| 42 | {base_loss:.4f} | 0 |")
        values.append(base_loss)
    for r in seeds:
        d = None if base_loss is None else r["val_loss"] - base_loss
        lines.append(
            f"| {r['overrides']['model_seed']} | {r['val_loss']:.4f} | {fmt(d)} |"
        )
        values.append(r["val_loss"])
    if len(values) >= 2:
        mean = sum(values) / len(values)
        sd = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
        lines.append("")
        lines.append(
            f"{len(values)} seeds: mean {mean:.4f}, sample SD {sd:.4f}, range {max(values) - min(values):.4f}. "
            "Pair differences inside about two SDs are not distinguishable from seed noise."
        )
    lines.append("")
    for first, second in plan["pairs"]:
        lines.append(f"## {first} x {second}")
        lines.append("")
        a_vals = plan["axis_values"][first]
        b_vals = plan["axis_values"][second]
        header = (
            "| "
            + f"{first} \\ {second}"
            + " | "
            + " | ".join(f"{b:g}" for b in b_vals)
            + " |"
        )
        lines.append(header)
        lines.append("|---|" + "---|" * len(b_vals))
        for a in a_vals:
            row = [f"**{a:g}**" if a == defaults[first] else f"{a:g}"]
            for b in b_vals:
                ov = {}
                if a != defaults[first]:
                    ov[first] = a
                if b != defaults[second]:
                    ov[second] = b
                hit = cells.get(_cell_key(ov))
                if hit is None:
                    row.append("-")
                else:
                    loss, src = hit
                    d = "" if base_loss is None else f" ({loss - base_loss:+.3f})"
                    mark = "" if src == "1b" else "*"
                    row.append(f"{loss:.4f}{d}{mark}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append(
            "Cells marked * are problem 1(a) runs; others are new. Parentheses give the difference from the baseline."
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def render_report_kvq(results, posthoc=None, baseline=2.9258):
    plan = results["plan"]
    posthoc = posthoc or {}
    cells = {r["index"]: r for r in results["runs"]}

    def loss(run):
        return run.get("val_loss")

    lines = [f"# CS312 KV-cache quantization report", ""]
    lines.append(
        f"Plan `{plan['digest']}`: d{plan['depth']}, "
        f"{plan['num_train_sequences']:,} sequences per run, "
        f"{len(plan['runs'])} runs, backend `{results['backend']}`, "
        f"phase `{plan.get('phase', '?')}`, collected {results['collected_utc']}."
    )
    done = sum(1 for r in results["runs"] if r.get("state") == "finished")
    start, end, span = sweep_window(results)
    lines.append(
        f"Quantization-aware training; straight-through fake-quant of the KV "
        f"cache. Anchor (no quant, default recipe) = {baseline:.4f}. "
        f"Finished {done}/{len(results['runs'])}. Window {start or '-'} to "
        f"{end or '-'}" + (f" ({span:.1f} min)." if span is not None else ".")
    )
    lines.append("")

    frontier = [r for r in results["runs"] if r["axis"] == "frontier"]
    if frontier:
        grans = sorted({r["granularity"] for r in frontier})
        qks = sorted({r["qk_norm"] for r in frontier}, reverse=True)
        levels = [lvl for lvl in plan["levels"]]
        have_posthoc = bool(posthoc)
        lines.append("## Frontier: where QAT holds under quantization")
        lines.append("")
        for gran in grans:
            for qk in qks:
                group = {
                    r["level"]: r
                    for r in frontier
                    if r["granularity"] == gran and r["qk_norm"] == qk
                }
                if not group:
                    continue
                qk_lbl = "qk_norm on" if qk else "qk_norm OFF"
                lines.append(f"### {gran}, {qk_lbl}")
                lines.append("")
                head = "| level | QAT val loss | d vs anchor |"
                sep = "|---|---|---|"
                if have_posthoc:
                    head += " post-hoc | QAT gain |"
                    sep += "---|---|"
                lines.append(head)
                lines.append(sep)
                for lvl in levels:
                    r = group.get(lvl)
                    if r is None:
                        continue
                    ql = loss(r)
                    qd = None if ql is None else ql - baseline
                    row = f"| {lvl} | {fmt(ql)} | {'' if qd is None or qd < 0 else '+'}{fmt(qd)} |"
                    if have_posthoc:
                        ph = posthoc.get(f"{lvl}-{gran}")
                        gain = None if (ph is None or ql is None) else ph - ql
                        row += f" {fmt(ph)} | {'' if gain is None or gain < 0 else '+'}{fmt(gain)} |"
                    lines.append(row)
                lines.append("")
        if have_posthoc:
            lines.append(
                "QAT gain = post-hoc loss minus QAT loss; positive means "
                "training-to-tolerate recovered that much. post-hoc is the "
                "eval-only curve (qk_norm on), so compare it only to the "
                "qk_norm-on rows."
            )
            lines.append("")

    sweep = [
        r
        for r in results["runs"]
        if r["axis"] in ("stress-center", "warmup", "learning_rate", "weight_decay")
    ]
    if sweep:
        center = next(
            (r for r in results["runs"] if r["axis"] == "stress-center"), None
        )
        c = center and loss(center)
        lines.append(f"## HP sweep at the stress cell `{plan.get('stress', '?')}`")
        lines.append("")
        lines.append("| axis | setting | val loss | d vs center |")
        lines.append("|---|---|---|---|")
        if center:
            lines.append(f"| center | default recipe | {fmt(c)} | 0 |")
        for r in sweep:
            if r["axis"] == "stress-center":
                continue
            ov = r["overrides"]
            setting = (
                ov.get("warmup_percent")
                or ov.get("learning_rate")
                or ov.get("weight_decay")
            )
            l = loss(r)
            d = None if (l is None or c is None) else l - c
            lines.append(
                f"| {r['axis']} | {setting} | {fmt(l)} | {'' if d is None or d < 0 else '+'}{fmt(d)} |"
            )
        lines.append("")

    seeds = [r for r in results["runs"] if r["axis"] == "seed" and loss(r) is not None]
    if seeds or any(r["axis"] == "stress-center" for r in results["runs"]):
        lines.append("## Noise floor at the stress cell")
        lines.append("")
        vals = []
        center = next(
            (r for r in results["runs"] if r["axis"] == "stress-center"), None
        )
        if center and loss(center) is not None:
            vals.append(loss(center))
        for r in seeds:
            vals.append(loss(r))
        if len(vals) >= 2:
            mean = sum(vals) / len(vals)
            sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
            lines.append(
                f"{len(vals)} seeds at the stress cell: mean {mean:.4f}, "
                f"sample SD {sd:.4f}. Sweep effects inside about two SDs are "
                f"not distinguishable from seed noise."
            )
        else:
            lines.append("Seed repeats pending.")
        lines.append("")
    return "\n".join(lines) + "\n"


def cmd_report(cfg, args):
    out = output_dir(cfg)
    results = read_json(out / "results.json")
    against = None
    if args.against:
        against = read_json(Path(args.against) / "results.json")
    problem = results["plan"].get("problem")
    if problem == "kvq":
        posthoc = {}
        if cfg["KVQ_POSTHOC"]:
            ph_path = Path(os.path.expanduser(cfg["KVQ_POSTHOC"]))
            ph_path = ph_path if ph_path.is_absolute() else ROOT / ph_path
            if ph_path.exists():
                posthoc = read_json(ph_path).get("results", {})
        text = render_report_kvq(results, posthoc)
    elif problem == "p1b":
        text = render_report_p1b(results)
    else:
        text = render_report(results, against)
    (out / "report.md").write_text(text)
    print(text)
    print(f"report written to {out / 'report.md'}")


def cmd_run(cfg, args):
    cmd_fetch(cfg, args)
    cmd_plan(cfg, args)
    cmd_launch(cfg, args)
    if cfg["BACKEND"] == "local":
        cmd_collect(cfg, args)
        cmd_report(cfg, args)


COMMANDS = {
    "fetch": cmd_fetch,
    "plan": cmd_plan,
    "data": cmd_data,
    "launch": cmd_launch,
    "collect": cmd_collect,
    "report": cmd_report,
    "run": cmd_run,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=".config")
    parser.add_argument("--against", help="another output directory to compare")
    parser.add_argument("--force", action="store_true", help="re-download data")
    parser.add_argument("command", choices=sorted(COMMANDS))
    args = parser.parse_args()
    cfg = settings(args.config)
    COMMANDS[args.command](cfg, args)


if __name__ == "__main__":
    main()

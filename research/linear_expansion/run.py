"""GPU phases: inspect, profile, warm start, tuning and paired confirmation.

Every trajectory starts from one ordinary warm checkpoint per seed, expands
the selected attention matrices for its arm, resets the optimizer and
continues on the same data cursor. Training-work time covers batch
preparation and transfer, forward, backward and the optimizer step, with a
device synchronization on both sides. Evaluation, diagnostics, saving and
export are timed separately and excluded from it.
"""

import argparse
import datetime
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from research.tinystories_depth.data import sha
from research.tinystories_depth.model import Model, loss
from research.tinystories_depth.probe import batch, finite
from research.tinystories_depth.probe import evaluate as gpu_evaluate

from .expand import (
    ARMS,
    counts,
    effective_weights,
    expand,
    folded_state,
    matrix_diagnostics,
    param_groups,
    target_names,
)

WARM_UPDATES = 512
WARM_LR = 3e-4
WARMUP = 32
LRS = (1e-4, 3e-4, 1e-3)
MULTIPLIERS = (0.5, 1.0, 2.0)
TUNE_SEED = 75201
CONFIRM_SEEDS = (75211, 75212, 75213)
DEVICE = os.environ.get("ABX_DEVICE", "cuda")
SMOKE = os.environ.get("ABX_SMOKE") == "1"


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def reset_peak():
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peaks():
    if DEVICE != "cuda":
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def rng_device():
    return torch.cuda.get_rng_state() if DEVICE == "cuda" else None


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def setup():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)


def optimizer(model, names, lr, multiplier):
    return torch.optim.AdamW(
        param_groups(model, names, lr, multiplier),
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0,
        foreach=True,
        fused=False,
    )


def fp32_forward(model, ids):
    """CPU smoke only: the CPU here lacks fast BF16, so skip Model's cast."""
    return model.lm(
        inputs_embeds=model.lm.model.embed_tokens(ids), use_cache=False
    ).logits


@torch.no_grad()
def evaluate(model, tokens, device):
    if device == "cuda":
        return gpu_evaluate(model, tokens, device)
    model.eval()
    count = (len(tokens) - 1) // 256
    total = 0.0
    for start in range(0, count, 8):
        ids, target = batch(tokens, np.arange(count), start // 8, device)
        total += loss(fp32_forward(model, ids), target).item() * target.numel()
    model.train()
    return total / (count * 256)


def environment():
    if DEVICE == "cuda":
        name = torch.cuda.get_device_properties(0).name
        free, total = torch.cuda.mem_get_info()
    else:
        name, free, total = "cpu-smoke", None, None
    return {
        "gpu": name,
        "gpu_total_bytes": total,
        "gpu_free_bytes_at_start": free,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "python": platform.python_version(),
        "kernel": platform.release(),
        "driver": _driver(),
        "sdpa": {
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
        },
        "tf32": False,
        "compile": False,
        "optimizer": "torch.optim.AdamW foreach=True fused=False",
        "run_id": os.environ.get("ABX_RUN_ID"),
        "source": os.environ.get("ABX_SOURCE_COMMIT"),
    }


def _driver():
    try:
        import subprocess

        return subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except OSError:
        return None


def verify_data(data):
    manifest = json.loads((data / "DATA.json").read_text())
    bad = [n for n, h in manifest["files"].items() if sha(data / n) != h]
    assert not bad, f"data hash mismatch: {bad}"
    return {
        "DATA.json_sha256": sha(data / "DATA.json"),
        "files": len(manifest["files"]),
    }


class Arrays:
    def __init__(self, data, seed):
        self.train = np.load(data / "train.npy", mmap_mode="r")
        self.curve = np.load(data / "curve.npy", mmap_mode="r")
        self.final = np.load(data / "final.npy", mmap_mode="r")
        self.order = np.load(data / f"order-{seed}.npy")
        assert self.order.size // 8 > 4608 + WARM_UPDATES
        if SMOKE:
            self.curve = self.curve[: 8 * 256 + 1]
            self.final = self.final[: 8 * 256 + 1]


def build(arm, seed, model_state=None):
    model = Model("R", 4, seed)
    if model_state is not None:
        model.load_state_dict(model_state)
    names = target_names(model)
    receipt = expand(model, arm, seed, names)
    if DEVICE == "cpu":
        import types

        model.forward = types.MethodType(fp32_forward, model)
    return model.to(DEVICE), names, receipt


def timed_step(model, opt, arrays, cursor, update):
    """One continuation update; returns (loss tensor, seconds)."""
    sync()
    started = time.perf_counter()
    x, y = batch(arrays.train, arrays.order, cursor, DEVICE)
    scale = min(update / WARMUP, 1.0)
    for g in opt.param_groups:
        g["lr"] = g["base_lr"] * scale
    opt.zero_grad(set_to_none=True)
    # BF16 matrix operations on the GPU; CPU smoke runs stay FP32.
    with torch.autocast(
        device_type=DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"
    ):
        value = loss(model(x), y)
    value.backward()
    opt.step()
    sync()
    return value, time.perf_counter() - started


def full_finite(model, opt):
    try:
        finite(model, opt)
        return True
    except AssertionError:
        return False


def grad_norms(model, names):
    sel = 0.0
    for n in names:
        for p in model.get_submodule(n).parameters():
            if p.grad is not None:
                sel += p.grad.float().square().sum().item()
    return {"selected_grad_norm": math.sqrt(sel)}


def relative_updates(before, after):
    return {
        n: ((after[n] - before[n]).norm() / before[n].norm()).item() for n in before
    }


@torch.no_grad()
def export_check(model, seed, arrays, fixed):
    """Fold to the ordinary model and compare functions before and after."""
    started = time.perf_counter()
    state = folded_state(model)
    plain, _, _ = build("D", seed, state)
    plain.eval()
    model.eval()
    ref_keys = set(Model("R", 4, seed).state_dict())
    h = lambda m: m.lm(
        inputs_embeds=m.lm.model.embed_tokens(fixed[0]), use_cache=False
    ).logits
    a, b = h(model), h(plain)
    fp32_rel = ((b - a).norm() / a.norm()).item()
    fp32_ce = abs(loss(a, fixed[1]).item() - loss(b, fixed[1]).item())
    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
        a16, b16 = model(fixed[0]), plain(fixed[0])
    bf16_rel = ((b16.float() - a16.float()).norm() / a16.float().norm()).item()
    bf16_ce = abs(loss(a16, fixed[1]).item() - loss(b16, fixed[1]).item())
    model.train()
    result = {
        "keys_equal": set(state) == ref_keys,
        "elements": sum(v.numel() for v in state.values()),
        "fp32_logits_rel_l2": fp32_rel,
        "fp32_ce_abs_diff": fp32_ce,
        "bf16_logits_rel_l2": bf16_rel,
        "bf16_ce_abs_diff": bf16_ce,
        "fold_seconds": time.perf_counter() - started,
    }
    return plain, state, result


def trajectory(job, arrays, warm, out, resume=None):
    """Run one arm until every requested endpoint is reached or the cap."""
    run = out / job["id"]
    run.mkdir(parents=True, exist_ok=resume is not None)
    torch.manual_seed(job["seed"])
    gc.collect()
    reset_peak()
    model, names, receipt = build(job["arm"], job["seed"], warm["model"])
    opt = optimizer(model, names, job["lr"], job["multiplier"])
    record = {
        "job": job,
        "receipt_targets": len(names),
        "counts": counts(model, names),
        "start_utc": now(),
        "train": [],
        "curve": [],
        "diagnostics": [],
        "endpoints": {},
        "training_seconds": 0.0,
        "eval_seconds": 0.0,
        "diagnostic_seconds": 0.0,
        "status": "running",
    }
    update = 0
    if resume is not None:
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["optimizer"])
        for g, base in zip(opt.param_groups, resume["base_lrs"]):
            g["base_lr"] = base
        torch.set_rng_state(resume["rng_cpu"])
        if resume["rng_gpu"] is not None:
            torch.cuda.set_rng_state(resume["rng_gpu"])
        update = resume["update"]
        prior = resume["record"]
        for k in ("train", "curve", "diagnostics"):
            record[k] = prior[k]
        for k in ("training_seconds", "eval_seconds", "diagnostic_seconds"):
            record[k] = prior[k]
        record["resumed_from_update"] = update
    fixed = batch(arrays.curve, np.arange(8), 0, DEVICE)
    eval_at = set(job["eval_at"])
    diag_at = set(job.get("diagnose_at", ()))
    T = job.get("time_endpoint_seconds")
    token_end = job.get("token_endpoint")
    log = (run / "TRAIN.jsonl").open("a")

    def curve_eval(tag=None):
        s = time.perf_counter()
        ce = evaluate(model, arrays.curve, DEVICE)
        dt = time.perf_counter() - s
        record["eval_seconds"] += dt
        record["curve"].append(
            {
                "update": update,
                "tokens": update * 2048,
                "ce": ce,
                "training_seconds": record["training_seconds"],
                "eval_seconds": dt,
                **({"endpoint": tag} if tag else {}),
            }
        )
        return ce

    def endpoint(tag):
        s = time.perf_counter()
        ce = curve_eval(tag)
        entry = {
            "update": update,
            "tokens": update * 2048,
            "training_seconds": record["training_seconds"],
            "curve_ce": ce,
        }
        if job.get("final_eval"):
            plain, state, check = export_check(model, job["seed"], arrays, fixed)
            entry["export"] = check
            t = time.perf_counter()
            entry["final_ce_exported"] = evaluate(plain, arrays.final, DEVICE)
            entry["final_ce_trained_form"] = evaluate(model, arrays.final, DEVICE)
            entry["final_eval_seconds"] = time.perf_counter() - t
            path = run / f"exported-{tag}.pt"
            t = time.perf_counter()
            torch.save(state, path)
            entry["exported_checkpoint"] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha(path),
                "save_seconds": time.perf_counter() - t,
            }
            del plain
        if job.get("save_endpoints"):
            path = run / f"state-{tag}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "update": update,
                },
                path,
            )
            entry["state_checkpoint"] = {
                "path": str(path),
                "bytes": path.stat().st_size,
            }
        entry["endpoint_seconds"] = time.perf_counter() - s
        record["endpoints"][tag] = entry

    def diagnose(before=None):
        s = time.perf_counter()
        row = {"update": update, "matrices": matrix_diagnostics(model, names)}
        if before is not None:
            row["relative_update"] = relative_updates(
                before, effective_weights(model, names)
            )
            row.update(grad_norms(model, names))
        record["diagnostics"].append(row)
        record["diagnostic_seconds"] += time.perf_counter() - s

    try:
        if update == 0:
            curve_eval()
            diagnose()
        while True:
            if update >= job["max_updates"]:
                break
            if token_end is not None or T is not None:
                done_tokens = token_end is None or update >= token_end
                done_time = T is None or record["training_seconds"] >= T
                if done_tokens and done_time:
                    break
            update += 1
            before = effective_weights(model, names) if update in diag_at else None
            value, dt = timed_step(
                model, opt, arrays, WARM_UPDATES + update - 1, update
            )
            record["training_seconds"] += dt
            ce = value.item()
            row = {"update": update, "ce": ce, "seconds": dt}
            record["train"].append(row)
            log.write(json.dumps(row) + "\n")
            if not math.isfinite(ce):
                record["status"] = "nonfinite"
                record["failure"] = {"update": update, "loss": str(ce)}
                break
            if before is not None:
                diagnose(before)
            reached = []
            if token_end is not None and update == token_end:
                reached.append("equal_tokens")
            if (
                T is not None
                and "equal_time" not in record["endpoints"]
                and record["training_seconds"] >= T
            ):
                record["time_overshoot_seconds"] = record["training_seconds"] - T
                reached.append("equal_time")
            if update in eval_at or reached:
                if not full_finite(model, opt):
                    record["status"] = "nonfinite"
                    record["failure"] = {"update": update, "where": "state"}
                    break
            for tag in reached:
                endpoint(tag)
            if update in eval_at and not reached:
                curve_eval()
            if job.get("save_at") == update:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "base_lrs": [g["base_lr"] for g in opt.param_groups],
                        "rng_cpu": torch.get_rng_state(),
                        "rng_gpu": rng_device(),
                        "update": update,
                        "record": {
                            k: record[k]
                            for k in (
                                "train",
                                "curve",
                                "diagnostics",
                                "training_seconds",
                                "eval_seconds",
                                "diagnostic_seconds",
                            )
                        },
                    },
                    run / f"resume-{update}.pt",
                )
        if record["status"] == "running":
            missing = [
                t
                for t, want in (("equal_tokens", token_end), ("equal_time", T))
                if want is not None and t not in record["endpoints"]
            ]
            record["status"] = "censored" if missing else "complete"
            if missing:
                record["censored"] = missing
    finally:
        log.close()
        record["updates"] = update
        record.update(peaks())
        record["optimizer_state_bytes"] = sum(
            v.numel() * v.element_size()
            for s in opt.state.values()
            for v in s.values()
            if torch.is_tensor(v)
        )
        record["end_utc"] = now()
        write(run / "RECORD.json", record)
    print(
        json.dumps(
            {
                "id": job["id"],
                "status": record["status"],
                "updates": update,
                "train_s": round(record["training_seconds"], 2),
                "last_curve": record["curve"][-1]["ce"] if record["curve"] else None,
            }
        ),
        flush=True,
    )
    return record


def load_warm(checkpoints, seed):
    path = checkpoints / f"warm-{seed}.pt"
    return torch.load(path, map_location="cpu", weights_only=False)


def warm_start(data, checkpoints, seed, out):
    """512 ordinary updates of the plain model; the shared start per seed."""
    path = checkpoints / f"warm-{seed}.pt"
    if path.exists():
        return json.loads((out / f"warm-{seed}.json").read_text())
    arrays = Arrays(data, seed)
    torch.manual_seed(seed)
    model, _, _ = build("D", seed)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=WARM_LR,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0,
        foreach=True,
        fused=False,
    )
    for g in opt.param_groups:
        g["base_lr"] = WARM_LR
    init_ce = evaluate(model, arrays.curve, DEVICE)
    seconds = 0.0
    for update in range(1, WARM_UPDATES + 1):
        value, dt = timed_step(model, opt, arrays, update - 1, update)
        seconds += dt
        assert math.isfinite(value.item()), f"warm start nonfinite at {update}"
    finite(model, opt)
    warm_ce = evaluate(model, arrays.curve, DEVICE)
    state = {
        "model": {k: v.cpu() for k, v in model.state_dict().items()},
        "optimizer_provenance": opt.state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_gpu": rng_device(),
        "cursor": WARM_UPDATES,
        "seed": seed,
    }
    torch.save(state, path)
    info = {
        "seed": seed,
        "updates": WARM_UPDATES,
        "lr": WARM_LR,
        "warmup": WARMUP,
        "initial_curve_ce": init_ce,
        "warm_curve_ce": warm_ce,
        "drop": init_ce - warm_ce,
        "gate_pass": init_ce - warm_ce >= 0.05,
        "training_seconds": seconds,
        "checkpoint": {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha(path),
        },
    }
    write(out / f"warm-{seed}.json", info)
    assert info["gate_pass"] or SMOKE, "warm start did not fall by 0.05 nat/token"
    return info


def profile(data, out):
    """Disposable 10 + 50 update timing per arm and the BF16 start gate."""
    arrays = Arrays(data, TUNE_SEED)
    base = Model("R", 4, TUNE_SEED).state_dict()
    fixed = [batch(arrays.curve, np.arange(16), i, DEVICE) for i in range(2)]
    result = {"environment": environment(), "arms": {}}
    reference = {}
    for arm in ARMS:
        gc.collect()
        reset_peak()
        model, names, _ = build(arm, TUNE_SEED, base)
        model.eval()
        with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            logits = [model(x).float() for x, _ in fixed]
            ces = [loss(l, y).item() for l, (_, y) in zip(logits, fixed)]
        if arm == "D":
            reference = {"logits": logits, "ce": ces}
        gate = {
            "bf16_logits_rel_l2": max(
                ((a - b).norm() / b.norm()).item()
                for a, b in zip(logits, reference["logits"])
            ),
            "bf16_ce_abs_diff": max(abs(a - b) for a, b in zip(ces, reference["ce"])),
        }
        gate["pass"] = (
            gate["bf16_logits_rel_l2"] <= 5e-3 and gate["bf16_ce_abs_diff"] <= 5e-3
        )
        model.train()
        opt = optimizer(model, names, 3e-4, 1.0)
        times = []
        for update in range(1, 61):
            _, dt = timed_step(model, opt, arrays, update - 1, update)
            if update > 10:
                times.append(dt)
        s = time.perf_counter()
        evaluate(model, arrays.curve, DEVICE)
        eval_s = time.perf_counter() - s
        s = time.perf_counter()
        _, state, _ = export_check(model, TUNE_SEED, arrays, fixed[0])
        export_s = time.perf_counter() - s
        times.sort()
        result["arms"][arm] = {
            "bf16_start_gate": gate,
            "step_median_seconds": statistics.median(times),
            "step_p90_seconds": times[int(0.9 * len(times)) - 1],
            "tokens_per_second": 2048 / statistics.median(times),
            **peaks(),
            "optimizer_state_bytes": sum(
                v.numel() * v.element_size()
                for st in opt.state.values()
                for v in st.values()
                if torch.is_tensor(v)
            ),
            "counts": counts(model, names),
            "curve_eval_seconds": eval_s,
            "export_seconds": export_s,
        }
        del model, opt
    d = result["arms"]["D"]["step_median_seconds"]
    result["time_endpoint_seconds"] = math.ceil(2048 * d / 30) * 30
    result["all_gates_pass"] = all(
        a["bf16_start_gate"]["pass"] for a in result["arms"].values()
    )
    write(out / "PROFILE.json", result)
    print(
        json.dumps(
            {
                a: (v["step_median_seconds"], v["bf16_start_gate"]["pass"])
                for a, v in result["arms"].items()
            }
        )
    )
    return result


def tune(data, checkpoints, out):
    """Same nine settings per arm; two survive 128 updates, run to 512."""
    warm_info = warm_start(data, checkpoints, TUNE_SEED, out)
    warm = load_warm(checkpoints, TUNE_SEED)
    arrays = Arrays(data, TUNE_SEED)
    tdir = out / "tuning"
    tdir.mkdir(exist_ok=True)
    summary = {"warm": warm_info, "trials": {}, "selected": {}}
    path = out / "TUNING.json"
    if path.exists():
        summary = json.loads(path.read_text())
    for arm in ARMS:
        for lr in LRS:
            for mult in MULTIPLIERS:
                jid = f"tune-{arm}-lr{lr:g}-m{mult:g}"
                if jid in summary["trials"]:
                    continue
                job = {
                    "id": jid,
                    "arm": arm,
                    "seed": TUNE_SEED,
                    "lr": lr,
                    "multiplier": mult,
                    "max_updates": 128,
                    "token_endpoint": None,
                    "eval_at": [32, 64, 128],
                    "diagnose_at": [1, 128],
                    "save_at": 128,
                }
                r = trajectory(job, arrays, warm, tdir)
                ce = {c["update"]: c["ce"] for c in r["curve"]}
                summary["trials"][jid] = {
                    "arm": arm,
                    "lr": lr,
                    "multiplier": mult,
                    "status": r["status"],
                    "curve": ce,
                    "training_seconds": r["training_seconds"],
                    "score128": (
                        (ce[64] + ce[128]) / 2 if r["status"] == "complete" else None
                    ),
                }
                write(path, summary)
        trials = [
            (t["score128"], t["training_seconds"], k)
            for k, t in summary["trials"].items()
            if t["arm"] == arm and t["score128"] is not None
        ]
        keep = [k for _, _, k in sorted(trials)[:2]]
        for jid in keep:
            cid = jid + "-to512"
            if cid in summary["trials"]:
                continue
            t = summary["trials"][jid]
            resume = torch.load(
                tdir / jid / "resume-128.pt", map_location="cpu", weights_only=False
            )
            job = {
                "id": jid,
                "arm": arm,
                "seed": TUNE_SEED,
                "lr": t["lr"],
                "multiplier": t["multiplier"],
                "max_updates": 512,
                "token_endpoint": None,
                "eval_at": [192, 256, 320, 384, 448, 512],
                "diagnose_at": [512],
            }
            r = trajectory(job, arrays, warm, tdir, resume=resume)
            ce = {c["update"]: c["ce"] for c in r["curve"]}
            summary["trials"][cid] = {
                "arm": arm,
                "lr": t["lr"],
                "multiplier": t["multiplier"],
                "status": r["status"],
                "curve": ce,
                "training_seconds": r["training_seconds"],
                "score512": (
                    (ce[384] + ce[512]) / 2 if r["status"] == "complete" else None
                ),
            }
            write(path, summary)
        finals = [
            (t["score512"], t["training_seconds"], k)
            for k, t in summary["trials"].items()
            if t["arm"] == arm and t.get("score512") is not None
        ]
        finals.sort()
        best = finals[0]
        for f in finals[1:]:
            if abs(f[0] - best[0]) < 1e-4 and f[1] < best[1]:
                best = f
        chosen = summary["trials"][best[2]]
        summary["selected"][arm] = {
            "trial": best[2],
            "lr": chosen["lr"],
            "multiplier": chosen["multiplier"],
            "score512": chosen["score512"],
            "boundary_limited": chosen["lr"] in (LRS[0], LRS[-1])
            or chosen["multiplier"] in (MULTIPLIERS[0], MULTIPLIERS[-1]),
        }
        write(path, summary)
    return summary


def confirm(data, checkpoints, out, frozen):
    """Paired trajectories on three seeds at the frozen settings."""
    spec = json.loads(frozen.read_text())
    cdir = out / "confirmation"
    cdir.mkdir(exist_ok=True)
    status_path = out / "CONFIRMATION.json"
    status = (
        json.loads(status_path.read_text())
        if status_path.exists()
        else {"runs": {}, "warm": {}}
    )
    for seed in CONFIRM_SEEDS:
        status["warm"][str(seed)] = warm_start(data, checkpoints, seed, out)
        write(status_path, status)
        warm = load_warm(checkpoints, seed)
        arrays = Arrays(data, seed)
        status.setdefault("first_update", {})[str(seed)] = first_update(
            arrays, warm, seed, spec
        )
        write(status_path, status)
        for arm in spec["order"][str(seed)]:
            jid = f"confirm-{arm}-{seed}"
            if jid in status["runs"]:
                continue
            sel = spec["selected"][arm]
            job = {
                "id": jid,
                "arm": arm,
                "seed": seed,
                "lr": sel["lr"],
                "multiplier": sel["multiplier"],
                "max_updates": spec["max_updates"],
                "token_endpoint": spec["token_endpoint"],
                "time_endpoint_seconds": spec["time_endpoint_seconds"],
                "eval_at": list(range(128, spec["max_updates"] + 1, 128)),
                "diagnose_at": [1, 512, spec["token_endpoint"]],
                "final_eval": True,
                "save_endpoints": True,
            }
            r = trajectory(job, arrays, warm, cdir)
            status["runs"][jid] = {
                "arm": arm,
                "seed": seed,
                "status": r["status"],
                "updates": r["updates"],
                "training_seconds": r["training_seconds"],
                "endpoints": r["endpoints"],
                "time_overshoot_seconds": r.get("time_overshoot_seconds"),
            }
            write(status_path, status)
    return status


def first_update(arrays, warm, seed, spec):
    """Effective first-update magnitude and cosine against D, fixed batch."""
    deltas = {}
    for arm in ARMS:
        sel = spec["selected"][arm]
        model, names, _ = build(arm, seed, warm["model"])
        opt = optimizer(model, names, sel["lr"], sel["multiplier"])
        before = effective_weights(model, names)
        timed_step(model, opt, arrays, WARM_UPDATES, 1)
        after = effective_weights(model, names)
        deltas[arm] = {n: after[n] - before[n] for n in names}
        rel = {n: (deltas[arm][n].norm() / before[n].norm()).item() for n in names}
        deltas[arm + ":rel"] = statistics.median(rel.values())
        del model, opt
    out = {}
    for arm in ARMS:
        cos = [
            torch.nn.functional.cosine_similarity(
                deltas[arm][n].flatten(), deltas["D"][n].flatten(), dim=0
            ).item()
            for n in deltas["D"]
        ]
        out[arm] = {
            "median_relative_update": deltas[arm + ":rel"],
            "median_cosine_vs_D": statistics.median(cos),
            "min_cosine_vs_D": min(cos),
        }
    return out


def smoke(data, checkpoints, out):
    """Tiny end-to-end pass over every code path; not a measurement."""
    global WARM_UPDATES
    WARM_UPDATES = 8
    warm_start(data, checkpoints, TUNE_SEED, out)
    warm = load_warm(checkpoints, TUNE_SEED)
    arrays = Arrays(data, TUNE_SEED)
    sdir = out / "smoke"
    sdir.mkdir(exist_ok=True)
    for arm in ARMS:
        job = {
            "id": f"smoke-{arm}",
            "arm": arm,
            "seed": TUNE_SEED,
            "lr": 3e-4,
            "multiplier": 1.0,
            "max_updates": 6,
            "token_endpoint": None,
            "eval_at": [3, 6],
            "diagnose_at": [1, 6],
            "save_at": 6,
        }
        trajectory(job, arrays, warm, sdir)
        resume = torch.load(
            sdir / job["id"] / "resume-6.pt", map_location="cpu", weights_only=False
        )
        job.update(
            max_updates=40,
            token_endpoint=10,
            time_endpoint_seconds=0.05,
            eval_at=[8, 10],
            diagnose_at=[10],
            final_eval=True,
            save_endpoints=True,
            save_at=None,
        )
        r = trajectory(job, arrays, warm, sdir, resume=resume)
        assert r["status"] == "complete", r["status"]
        for e in r["endpoints"].values():
            x = e["export"]
            assert x["keys_equal"] and x["fp32_logits_rel_l2"] <= 1e-5, x
    spec = {"selected": {a: {"lr": 3e-4, "multiplier": 1.0} for a in ARMS}}
    write(sdir / "FIRST_UPDATE.json", first_update(arrays, warm, TUNE_SEED, spec))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["inspect", "profile", "tune", "confirm", "smoke"])
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--checkpoints", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--frozen", type=Path)
    a = p.parse_args()
    setup()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.checkpoints:
        a.checkpoints.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    info = {"phase": a.phase, "start_utc": now(), "environment": environment()}
    info["data"] = verify_data(a.data)
    if a.phase == "profile":
        profile(a.data, a.out)
    elif a.phase == "tune":
        tune(a.data, a.checkpoints, a.out)
    elif a.phase == "confirm":
        confirm(a.data, a.checkpoints, a.out, a.frozen)
    elif a.phase == "smoke":
        smoke(a.data, a.checkpoints, a.out)
    info["seconds"] = time.perf_counter() - started
    info["end_utc"] = now()
    write(a.out / f"PHASE-{a.phase}-{os.environ.get('ABX_RUN_ID', 'local')}.json", info)


if __name__ == "__main__":
    main()

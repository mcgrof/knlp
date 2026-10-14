# SPDX-License-Identifier: MIT
"""Bounded supervised-only MNIST study: probe, qualify, freeze, run, analyze."""

import argparse
import contextlib
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import signal
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from research.subliminal_learning.data import epoch_order, load_mnist, tensor_digest
from .model import ARMS, SupervisedModel, phase
from .numerics import qualify_reference

SCHEMA = "supervised-mnist-depth.v1"
CAP = 8 * 3600
GRID = (1e-4, 3e-4, 1e-3, 3e-3)
CORE = ("R", "L", "M", "ML")
CHANNELS = ("ML-O", "ML-S", "ML-C", "M-O")


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source():
    identity = json.loads(os.environ["KNLP_SUPERVISED_SOURCE_JSON"])
    identity["module_hashes"] = {
        p.name: digest_file(p) for p in Path(__file__).parent.glob("*.py")
    }
    return identity


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def telemetry():
    readings = {}
    for pattern in (
        "/sys/class/drm/card*/device/pp_dpm_sclk",
        "/sys/class/drm/card*/device/pp_dpm_mclk",
        "/sys/class/drm/card*/device/hwmon/hwmon*/temp*_input",
        "/sys/class/drm/card*/device/gpu_busy_percent",
    ):
        import glob

        for name in glob.glob(pattern):
            try:
                readings[name] = Path(name).read_text().strip()
            except OSError as e:
                readings[name] = str(e)
    return readings


def charged(root):
    file = root / "BUDGET.json"
    return (
        json.loads(file.read_text())
        if file.exists()
        else {"cap_seconds": CAP, "operations": []}
    )


@contextlib.contextmanager
def operation(root, name):
    ledger = charged(root)
    if any(x["status"] == "running" for x in ledger["operations"]):
        raise RuntimeError(
            "unfinished operation; reconcile its elapsed time before continuing"
        )
    used = sum(x["charged_seconds"] for x in ledger["operations"])
    if used >= CAP:
        raise RuntimeError("study GPU-host time cap exhausted")
    receipt = {
        "name": name,
        "start_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "running",
        "charged_seconds": 0,
    }
    ledger["operations"].append(receipt)
    write(root / "BUDGET.json", ledger)
    t0 = time.monotonic()
    try:
        yield lambda: CAP - used - (time.monotonic() - t0)
        receipt["status"] = "complete"
    except BaseException as e:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        sync("cuda")
        receipt["charged_seconds"] = time.monotonic() - t0
        write(root / "BUDGET.json", ledger)


def split_indices(labels):
    g = torch.Generator().manual_seed(20261009)
    counts = torch.bincount(labels, minlength=10)
    exact = counts.double() * 5000 / len(labels)
    quota = exact.floor().long()
    for digit in torch.argsort(exact - quota, descending=True)[
        : 5000 - int(quota.sum())
    ]:
        quota[digit] += 1
    fit, val = [], []
    for digit in range(10):
        idx = (labels == digit).nonzero().flatten()
        idx = idx[torch.randperm(len(idx), generator=g)]
        val.append(idx[: quota[digit]])
        fit.append(idx[quota[digit] :])
    return torch.cat(fit), torch.cat(val)


def dataset(args, move=True):
    t0 = time.monotonic()
    x, y, tx, ty = load_mnist(args.data)
    fit, val = split_indices(y)
    payload = {
        "split_seed": 20261009,
        "fit": fit.tolist(),
        "validation": val.tolist(),
        "fit_hash": tensor_digest(fit),
        "validation_hash": tensor_digest(val),
        "fit_class_counts": torch.bincount(y[fit], minlength=10).tolist(),
        "validation_class_counts": torch.bincount(y[val], minlength=10).tolist(),
        "cache_sha256": digest_file(
            Path(args.data) / "subliminal_mnist_28x28_norm05.pt"
        ),
        "preprocessing": "fixed pixel/127.5-1; no fitted parameters; no augmentation",
    }
    previous = args.root / "SPLIT.json"
    if previous.exists():
        assert json.loads(previous.read_text()) == payload
    else:
        write(previous, payload)
    tensors = [x[fit], y[fit], x[val], y[val]]
    if move:
        tensors = [v.to(args.device) for v in tensors]
        sync(args.device)
    return tensors, (tx, ty), payload, time.monotonic() - t0


def supervised_loss(logits, labels):
    if logits.shape[-1] != 10:
        raise ValueError("supervised runner requires exactly ten logits")
    m, b, _ = logits.shape
    if labels.ndim == 1:
        labels = labels[None].expand(m, b)
    means = (
        F.cross_entropy(logits.reshape(-1, 10), labels.reshape(-1), reduction="none")
        .view(m, b)
        .mean(1)
    )
    return means.sum(), means


@torch.no_grad()
def evaluate(model, x, y, device):
    sync(device)
    t0 = time.monotonic()
    model.eval()
    ce, correct = torch.zeros((), device=device), torch.zeros((), device=device)
    for i in range(0, len(x), 1024):
        logits = model(x[i : i + 1024])
        ce += F.cross_entropy(logits[0], y[i : i + 1024], reduction="sum")
        correct += (logits[0].argmax(-1) == y[i : i + 1024]).sum()
    sync(device)
    out = {
        "ce": float(ce.cpu() / len(x)),
        "accuracy": float(correct.cpu() / len(x)),
        "seconds": time.monotonic() - t0,
    }
    model.train()
    return out


def monitoring(model, finite_loss_grad):
    inv = model.invariants
    if inv is not None:
        values = inv.cpu().tolist()
        if max(values[:2]) > 1e-4 or max(values[2:]) > 1e-6:
            raise FloatingPointError(f"mHC invariant violation: {values}")
    else:
        values = None
    if not bool(finite_loss_grad.cpu()):
        raise FloatingPointError("nonfinite output, loss, gradient or update")
    return values


def flops_per_example(model):
    # Multiply-add=2; backward dense GEMMs=2*forward. Elementwise and
    # log/exp/sigmoid count one each (algorithmic count, not hardware cost).
    w, d = model.width, model.depth
    forward_dense = 2 * (784 * w + d * 2 * w * w + w * 10)
    forward_norm = d * 12 * w + 12 * w
    routing = (
        d * (2 * 4 * w * 24 + 2 * 4 * 4 * w + 2 * 4 * w + 14 * 4 * w + 24)
        if model.conns is not None
        else d * w
    )
    projection = d * (model.conns[0].iters * 16 * 12) if model.conns is not None else 0
    gains = d * w * 10 if model.layerrope is not None else 0
    code = d * 4 * 16 if model.arm in CHANNELS else 0
    return 3 * (forward_dense + forward_norm + routing + projection + gains + code)


@torch.no_grad()
def diagnostics(model, x, device):
    sync(device)
    t0 = time.monotonic()
    model.capture = True
    model(x[:256])
    layers, product = [], None
    rms = lambda a: float(a.double().square().mean().sqrt().cpu())
    for l, t in enumerate(model.terms):
        row = {
            "layer": l,
            **{
                n + "_rms": rms(t[n])
                for n in (
                    "state",
                    "block_input",
                    "block_output",
                    "carried",
                    "carry_change",
                    "write",
                    "updated",
                )
            },
        }
        if t["matrix"] is not None:
            h, p = t["state"], t["matrix"][:, :8].double()
            centre = h.mean(2, keepdim=True)
            pair = F.normalize(h, dim=-1) @ F.normalize(h, dim=-1).transpose(-1, -2)
            product = p if product is None else p @ product
            row.update(
                stream_mean_rms=rms(centre),
                stream_difference_rms=rms(h - centre),
                pairwise_cosine=pair.mean((0, 1)).cpu().tolist(),
                read_sum_mean=float(t["pre"].sum(-1).mean().cpu()),
                write_sum_mean=float(t["beta"].sum(-1).mean().cpu()),
                gate_saturation_fraction=float(
                    ((t["base_beta"] < 0.01) | (t["base_beta"] > 1.99))
                    .float()
                    .mean()
                    .cpu()
                ),
                code_strength_mean=float(t["code_strength"].mean().cpu()),
                base_write_l2=float(
                    (t["base_beta"].norm(dim=-1) * t["block_output"].norm(dim=-1))
                    .mean()
                    .cpu()
                ),
                coded_write_l2=float(t["write"].flatten(-2).norm(dim=-1).mean().cpu()),
                immediate_mean_write_error=float(
                    (
                        (t["beta"] - t["base_beta"]).sum(-1)[..., None]
                        * t["block_output"]
                        / 4
                    )
                    .abs()
                    .max()
                    .cpu()
                ),
                carry_distance_from_identity=rms(p - torch.eye(4, device=device)),
                row_residual=float((p.sum(-1) - 1).abs().max().cpu()),
                column_residual=float((p.sum(-2) - 1).abs().max().cpu()),
                carry_singular_values=torch.linalg.svdvals(p)
                .mean((0, 1))
                .cpu()
                .tolist(),
                carry_product_singular_values=torch.linalg.svdvals(product)
                .mean((0, 1))
                .cpu()
                .tolist(),
            )
        grads = [p.grad for p in model.blocks[l].parameters() if p.grad is not None]
        row["last_training_gradient_rms"] = math.sqrt(
            sum(float(p.double().square().sum().cpu()) for p in grads)
            / max(1, sum(p.numel() for p in grads))
        )
        layers.append(row)
    schedule = model.angular_schedule
    sched = (
        {n: p.detach().cpu().tolist() for n, p in schedule.named_parameters()}
        if schedule is not None
        else {}
    )
    gain = model.layerrope().detach().cpu() if model.layerrope is not None else None
    angles = (
        phase(schedule, model.depth).detach().cpu() if schedule is not None else None
    )
    model.capture = False
    model.terms = []
    sync(device)
    return {
        "layers": layers,
        "schedule": sched,
        "gains": gain.tolist() if gain is not None else None,
        "angles": angles.tolist() if angles is not None else None,
        "minimum_adjacent_phase_separation": (
            float(angles.diff().abs().min())
            if angles is not None and model.depth > 1
            else None
        ),
        "seconds": time.monotonic() - t0,
    }


def crossing(curve, target="primary"):
    pred = {
        "primary": lambda r: r["ce"] <= 0.10 and r["accuracy"] >= 0.97,
        "acc95": lambda r: r["accuracy"] >= 0.95,
        "acc98": lambda r: r["accuracy"] >= 0.98,
        "ce08": lambda r: r["ce"] <= 0.08,
    }[target]
    for a, b in zip(curve, curve[1:]):
        if pred(a) and pred(b):
            return b
    return None


def selection(rows):
    eligible = []
    for lr in GRID:
        rr = [r for r in rows if r["lr"] == lr]
        if len(rr) != 2 or any(r["status"] != "complete" for r in rr):
            continue
        hits = [crossing(r["curve"]) for r in rr]
        auc = np.mean(
            [
                np.trapezoid(
                    [x["ce"] for x in r["curve"]], [x["step"] for x in r["curve"]]
                )
                / r["curve"][-1]["step"]
                for r in rr
            ]
        )
        end = np.mean([r["curve"][-1]["ce"] for r in rr])
        score = (
            (0, math.exp(np.mean([math.log(max(1, h["step"])) for h in hits])), end, lr)
            if all(hits)
            else (1, float(auc), end, lr)
        )
        eligible.append((score, lr))
    if not eligible:
        return {"lr": None, "status": "no_valid_rate"}
    score, lr = min(eligible)
    return {
        "lr": lr,
        "score": list(score),
        "boundary": lr in (GRID[0], GRID[-1]),
        "status": "selected",
    }


def trajectory(
    args,
    arm,
    depth,
    seed,
    lr,
    epochs,
    iters,
    tensors,
    test,
    split,
    output,
    remaining,
    final=False,
):
    output.mkdir(parents=True, exist_ok=False)
    fx, fy, vx, vy = tensors
    t0 = time.monotonic()
    model = SupervisedModel(arm, depth, [f"seed-{seed}"], iters).to(args.device)
    opt = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    )
    torch.cuda.reset_peak_memory_stats()
    identity = model.identities()
    result = {
        "arm": arm,
        "depth": depth,
        "seed": seed,
        "lr": lr,
        "epochs": epochs,
        "iters": iters,
        "identity": identity,
        "source": source(),
        "split_hash": split["fit_hash"],
        "status": "running",
        "curve": [],
        "timing": {
            "train": 0.0,
            "evaluation": 0.0,
            "diagnostics": 0.0,
            "checkpoint": 0.0,
        },
    }
    write(output / "IDENTITY.json", result)
    diag = {}
    step, examples = 0, 0
    finite = torch.ones((), dtype=torch.bool, device=args.device)
    log = (output / "training.jsonl").open("w")

    def val():
        v = evaluate(model, vx, vy, args.device)
        result["timing"]["evaluation"] += v.pop("seconds")
        if not math.isfinite(v["ce"]) or not math.isfinite(v["accuracy"]):
            raise FloatingPointError("nonfinite validation outputs")
        result["curve"].append(
            {
                "step": step,
                "examples": examples,
                "training_flops": examples * flops_per_example(model),
                "training_seconds": result["timing"]["train"],
                "elapsed_seconds": time.monotonic() - t0,
                **v,
            }
        )
        result["curve"][-1]["telemetry"] = telemetry()
        result["invariants"] = monitoring(model, finite)
        write(output / "RESULT.json", result)

    try:
        val()
        if final:
            diag["0"] = diagnostics(model, vx, args.device)
            result["timing"]["diagnostics"] += diag["0"]["seconds"]
        for epoch in range(epochs):
            order = epoch_order(
                f"seed-{seed}", f"supervised_d{depth}", epoch, len(fx)
            ).to(args.device)
            order_hash = tensor_digest(order)
            for start in range(0, len(fx), 1024):
                if remaining() <= 60:
                    raise TimeoutError("study cap reached during trajectory")
                ix = order[start : start + 1024]
                sync(args.device)
                tt = time.monotonic()
                opt.zero_grad(set_to_none=True)
                logits = model(fx[ix])
                loss, _ = supervised_loss(logits, fy[ix])
                loss.backward()
                finite &= torch.isfinite(loss) & torch.isfinite(logits).all()
                for p in model.parameters():
                    if p.grad is not None:
                        finite &= torch.isfinite(p.grad).all()
                opt.step()
                for p in model.parameters():
                    finite &= torch.isfinite(p).all()
                sync(args.device)
                seconds = time.monotonic() - tt
                result["timing"]["train"] += seconds
                step += 1
                examples += len(ix)
                log.write(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "step": step,
                            "batch_examples": len(ix),
                            "examples": examples,
                            "train_ce": float(loss.detach().cpu()),
                            "training_seconds": result["timing"]["train"],
                            "training_flops": examples * flops_per_example(model),
                            "order_hash": order_hash,
                        }
                    )
                    + "\n"
                )
                if step % 10 == 0:
                    val()
            if final and epoch + 1 in (1, 5, 20):
                diag[str(epoch + 1)] = diagnostics(model, vx, args.device)
                result["timing"]["diagnostics"] += diag[str(epoch + 1)]["seconds"]
                write(output / "DIAGNOSTICS.json", diag)
        if result["curve"][-1]["step"] != step:
            val()
        if final:
            # Official test is evaluated once, only after fixed final endpoint.
            tx, ty = (t.to(args.device) for t in test)
            result["official_test"] = evaluate(model, tx, ty, args.device)
            del tx, ty
        cp = args.checkpoints / output.name
        cp.parent.mkdir(parents=True, exist_ok=True)
        if cp.exists():
            raise RuntimeError("checkpoint already exists")
        tt = time.monotonic()
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "source": source(),
                "split": split["fit_hash"],
                "step": step,
                "rng": torch.get_rng_state(),
            },
            cp,
        )
        saved = torch.load(cp, map_location="cpu", weights_only=False)
        assert all(torch.isfinite(t).all() for t in saved["model"].values())
        result["checkpoint"] = {
            "path": str(cp),
            "bytes": cp.stat().st_size,
            "sha256": digest_file(cp),
            "reloaded_finite": True,
        }
        result["timing"]["checkpoint"] = time.monotonic() - tt
        result["status"] = "complete"
    except BaseException as e:
        result["status"] = (
            "interrupted"
            if isinstance(e, (TimeoutError, KeyboardInterrupt))
            else "invalid"
        )
        result["error"] = f"{type(e).__name__}: {e}"
        if isinstance(e, (KeyboardInterrupt, TimeoutError)):
            raise
    finally:
        log.close()
        result["end_to_end_seconds"] = time.monotonic() - t0
        result["peak_allocated"] = torch.cuda.max_memory_allocated()
        result["peak_reserved"] = torch.cuda.max_memory_reserved()
        write(output / "RESULT.json", result)
        del model, opt
        torch.cuda.empty_cache()
    return result


def enumerate_cells():
    return [
        {"arm": a, "depth": d, "stage": "core" if a in CORE else "channels"}
        for a in ARMS
        for d in ((4, 16, 64) if a in CORE else (16, 64))
    ]


def qualify(args, remaining):
    tensors, test, split, setup = dataset(args)
    report = {
        "schema": SCHEMA,
        "source": source(),
        "attempts": [],
        "selected_iters": None,
        "status": "unqualified",
        "thresholds": {
            "carry_residual": 1e-4,
            "reference_residual": 1e-10,
            "relative_vjp": 1e-3,
            "small_vjp_absolute": 1e-6,
        },
    }
    for iters in (20, 50, 100, 200):
        ref, raw = qualify_reference(iters, args.device)
        cp = args.checkpoints / f"qualification-reference-{iters}.pt"
        cp.parent.mkdir(parents=True, exist_ok=True)
        if cp.exists():
            raise RuntimeError("qualification checkpoint exists")
        torch.save(raw, cp)
        ref["reference_receipt"] = {
            "path": str(cp),
            "sha256": digest_file(cp),
            "bytes": cp.stat().st_size,
        }
        report["attempts"].append({"reference": ref, "training": []})
        write(args.root / "QUALIFICATION.json", report)
        print("reference", iters, "pass", ref["pass"], flush=True)
        # A failed mandatory synthetic gate rules out this count. Do not
        # spend deep training time on a count that cannot qualify.
        if not ref["pass"]:
            continue
        for depth in (4, 64):
            for arm in ("M", "ML", "ML-O"):
                for seed in (75091, 75092):
                    p = args.root / "qualification" / f"{arm}-d{depth}-s{seed}-i{iters}"
                    r = trajectory(
                        args,
                        arm,
                        depth,
                        seed,
                        3e-4,
                        1,
                        iters,
                        tensors,
                        test,
                        split,
                        p,
                        remaining,
                    )
                    report["attempts"][-1]["training"].append(
                        {
                            "arm": arm,
                            "depth": depth,
                            "seed": seed,
                            "status": r["status"],
                            "error": r.get("error"),
                            "invariants": r.get("invariants"),
                        }
                    )
                    write(args.root / "QUALIFICATION.json", report)
        if all(r["status"] == "complete" for r in report["attempts"][-1]["training"]):
            report["selected_iters"] = iters
            report["status"] = "qualified"
            break
    report["training_qualification_status"] = (
        "not_started_failed_reference"
        if not any(r["training"] for r in report["attempts"])
        else "attempted"
    )
    write(args.root / "QUALIFICATION.json", report)
    return report


def probe(args, remaining):
    qual = (
        json.loads((args.root / "QUALIFICATION.json").read_text())
        if (args.root / "QUALIFICATION.json").exists()
        else {}
    )
    if qual.get("status") not in ("qualified", "synthetic_pass_provisional"):
        report = {
            "source": source(),
            "status": "not_run_failed_qualification",
            "eligible_core": False,
            "eligible_channels": False,
            "probes": [],
            "reason": "Stop the mHC queue after failed mandatory qualification; no runtime forecast is measured.",
        }
        write(args.root / "PREFLIGHT.json", report)
        return report
    tensors, _, split, setup = dataset(args)
    fx, fy, vx, vy = tensors
    iters = qual.get("selected_iters") or 20
    report = {
        "source": source(),
        "status": "provisional" if qual.get("status") != "qualified" else "qualified",
        "iters": iters,
        "data_setup_seconds": setup,
        "probes": [],
        "contingency": 1.25,
        "steps_per_epoch": math.ceil(55000 / 1024),
        "split": split["fit_hash"],
    }
    for depth in (4, 16, 64):
        for arm in (("R", "ML") if depth == 4 else ("R", "ML", "ML-O")):
            if remaining() < 120:
                raise TimeoutError("preflight budget exhausted")
            sync(args.device)
            t0 = time.monotonic()
            model = SupervisedModel(arm, depth, ["seed-75091"], iters).to(args.device)
            opt = torch.optim.Adam(model.parameters(), lr=3e-4)
            sync(args.device)
            cold = time.monotonic() - t0
            times = []
            finite = torch.ones((), dtype=torch.bool, device=args.device)
            for s in range(60):
                sync(args.device)
                t0 = time.monotonic()
                opt.zero_grad(set_to_none=True)
                logits = model(fx[:1024])
                loss, _ = supervised_loss(logits, fy[:1024])
                loss.backward()
                finite &= torch.isfinite(loss) & torch.isfinite(logits).all()
                for p in model.parameters():
                    finite &= (
                        torch.isfinite(p.grad).all() if p.grad is not None else True
                    )
                opt.step()
                for p in model.parameters():
                    finite &= torch.isfinite(p).all()
                sync(args.device)
                times.append(time.monotonic() - t0)
                if (s + 1) % 10 == 0:
                    monitoring(model, finite)
            v = evaluate(model, vx, vy, args.device)
            inv = monitoring(model, finite)  # includes validation forwards
            sync(args.device)
            io_start = time.monotonic()
            checkpoint = args.checkpoints / f"provisional-probe-{arm}-d{depth}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            if checkpoint.exists():
                raise RuntimeError("probe checkpoint already exists")
            torch.save(
                {"model": model.state_dict(), "optimizer": opt.state_dict()}, checkpoint
            )
            restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
            all_finite = all(
                torch.isfinite(v).all() for v in restored["model"].values()
            )
            all_finite &= all(
                torch.isfinite(v).all()
                for state in restored["optimizer"]["state"].values()
                for v in state.values()
                if torch.is_tensor(v)
            )
            if not all_finite:
                raise FloatingPointError("nonfinite restored probe checkpoint")
            receipt = {
                "path": str(checkpoint),
                "bytes": checkpoint.stat().st_size,
                "sha256": digest_file(checkpoint),
                "reload_finite": bool(all_finite),
            }
            io_seconds = time.monotonic() - io_start
            del restored
            r = {
                "arm": arm,
                "depth": depth,
                "cold_seconds": cold,
                "warmup_steps": 10,
                "measured_steps": 50,
                "seconds_per_step": float(np.mean(times[10:])),
                "all_step_seconds": times,
                "checkpoint_io_seconds": io_seconds,
                "checkpoint_receipt": receipt,
                "telemetry": telemetry(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "validation_seconds": v["seconds"],
                "invariants": inv,
                "parameters": sum(p.numel() for p in model.parameters()),
                "training_flops_per_example": flops_per_example(model),
            }
            report["probes"].append(r)
            write(args.root / "PREFLIGHT.json", report)
            print("probe", arm, depth, r["seconds_per_step"], flush=True)
            del model, opt
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    estimates = {"core": 0.0, "channels": 0.0}
    for cell in enumerate_cells():
        a, d = cell["arm"], cell["depth"]
        proxy = "R" if a in ("R", "L") else ("ML" if a in ("M", "ML") else "ML-O")
        row = next(r for r in report["probes"] if r["arm"] == proxy and r["depth"] == d)
        # R proxies L; ML proxies M; ML-O proxies the coded controls. Pin
        # those assumptions and use the prescribed 25% contingency.
        final_n = 10 if d == 64 else 5
        for n, epochs in ((8, 10), (final_n, 20)):
            steps = epochs * 54
            estimates[cell["stage"]] += n * (
                steps * row["seconds_per_step"]
                + (math.ceil(steps / 10) + 1) * row["validation_seconds"]
                + row["cold_seconds"]
                + row["checkpoint_io_seconds"]
                + setup
            )
    report["estimate_proxy_rules"] = {"L": "R", "M": "ML", "ML-S/ML-C/M-O": "ML-O"}
    report["estimated_training_and_validation_seconds"] = estimates
    # Twelve fresh one-epoch qualification jobs, six at each depth.
    # ML proxies M, and the D=4 ML-O cost uses ML plus the contingency.
    qualification_cost = (
        sum(
            2
            * (
                54 * row["seconds_per_step"]
                + 7 * row["validation_seconds"]
                + row["cold_seconds"]
                + row["checkpoint_io_seconds"]
                + setup
            )
            for d in (4, 64)
            for a in ("M", "ML", "ML-O")
            for row in [
                next(
                    r
                    for r in report["probes"]
                    if r["depth"] == d
                    and r["arm"] == ("ML" if a == "M" or d == 4 else a)
                )
            ]
        )
        if qual.get("status") != "qualified"
        else 0.0
    )
    report["qualification_remaining_estimate_seconds"] = qualification_cost
    report["remaining_at_estimate_seconds"] = remaining()
    report["estimate_with_contingency_seconds"] = {
        "core_including_qualification": 1.25 * (estimates["core"] + qualification_cost),
        "full_including_qualification": 1.25
        * (sum(estimates.values()) + qualification_cost),
    }
    report["feasible_core"] = (
        report["estimate_with_contingency_seconds"]["core_including_qualification"]
        < remaining()
    )
    report["feasible_channels"] = (
        report["estimate_with_contingency_seconds"]["full_including_qualification"]
        < remaining()
    )
    report["eligible_core"] = (
        qual.get("status") == "qualified" and report["feasible_core"]
    )
    report["eligible_channels"] = (
        report["eligible_core"] and report["feasible_channels"]
    )
    report["forecast_basis"] = (
        "Measured steps, validation, model/optimizer setup, checkpoint save/reload/hash and data setup for exact cells; 25% contingency includes unmeasured scheduled diagnostics and orchestration. Single checkpoint per cell proxies two checkpoints conservatively through contingency; this is an estimate."
    )
    report["status"] = (
        "qualified" if qual.get("status") == "qualified" else "provisional"
    )
    write(args.root / "PREFLIGHT.json", report)
    return report


def plan(args):
    q = json.loads((args.root / "QUALIFICATION.json").read_text())
    pre = (
        json.loads((args.root / "PREFLIGHT.json").read_text())
        if (args.root / "PREFLIGHT.json").exists()
        else None
    )
    _, _, split, _ = dataset(args, move=False)
    cells = enumerate_cells()
    jobs = []
    identities = {}
    for c in cells:
        a, d = c["arm"], c["depth"]
        for stage, seeds, lrs, epochs in (
            ("selection", (75001, 75002), GRID, 10),
            ("final", range(75101, 75111 if d == 64 else 75106), (None,), 20),
        ):
            for seed in seeds:
                key = f"{a}-d{d}-s{seed}"
                model = SupervisedModel(
                    a, d, [f"seed-{seed}"], q.get("selected_iters") or 20
                )
                identities[key] = model.identities()
                del model
                for lr in lrs:
                    jobs.append(
                        {
                            **c,
                            "phase": stage,
                            "seed": seed,
                            "lr": lr,
                            "epochs": epochs,
                            "identity_key": key,
                        }
                    )
    for d in (4, 16, 64):
        for seed in range(75101, 75111 if d == 64 else 75106):
            pairs = [
                v["common_weights"]
                for k, v in identities.items()
                if f"-d{d}-s{seed}" in k
            ]
            assert all(p == pairs[0] for p in pairs)
    payload = {
        "schema": SCHEMA,
        "guidance_commit": "supervised-depth-comparison",
        "source": source(),
        "split": split,
        "cells": cells,
        "jobs": jobs,
        "initialization": identities,
        "qualification": q["status"],
        "qualification_source": q.get("source"),
        "sinkhorn_iters": q.get("selected_iters"),
        "preflight": pre,
        "batch_size": 1024,
        "pack_size": 1,
        "width": 256,
        "streams": 4,
        "outputs": 10,
        "optimizer": {
            "name": "Adam",
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 0.0,
        },
        "precision": "FP32",
        "compile": False,
        "lr_grid": list(GRID),
        "qualification_seeds": [75091, 75092],
        "primary_depth": 64,
        "primary_target": {
            "validation_ce_max": 0.10,
            "accuracy_min": 0.97,
            "consecutive_evaluations": 2,
            "every_steps": 10,
        },
        "bootstrap": {
            "resamples": 100000,
            "seed": 20261010,
            "primary_level": 0.99,
            "exploratory_level": 0.95,
        },
        "primary_contrasts": [
            ["ML", "M"],
            ["ML", "L"],
            ["ML-O", "ML"],
            ["ML-O", "ML-S"],
            ["ML-O", "ML-C"],
        ],
        "sample_gate": {
            "ratio_max": 0.9,
            "upper_ratio_max_exclusive": 1.0,
            "accuracy_lower_pp": -0.5,
            "ce_upper_nats": 0.02,
        },
        "cap_seconds": CAP,
        "no_resume": True,
        "test_use": "once per completed final trajectory, after fixed endpoint; never selection",
        "censoring": "retain non-attainment; any censored pair disables uncensored ratio CI",
        "timing": "synchronize boundaries; training includes invariant checks; eval, diagnostics, setup and I/O separate",
        "iteration_candidates": [20, 50, 100, 200],
        "monitoring_tolerance": 1e-4,
        "code": {
            "rho": 0.5,
            "ordered": "phi(l)=exp(alpha_rot+beta_rot*log(l+1))",
            "constant_site": "floor((D-1)/2)",
            "shuffle_stream": "depth_code_order",
            "amin_ties": "equal-split PyTorch subgradient",
            "shared_angular_parameters": True,
        },
        "lr_rule": "both seeds primary attainment, geometric crossing steps; else mean CE AUC; ties endpoint CE, then smaller LR",
        "stopping": "qualification failure stops mHC queue; core only if estimate fits; channels only if whole estimate fits; finish whole paired blocks within cap",
        "flops_convention": "multiply-add=2; 3x forward arithmetic for forward+backward; elementary/transcendentals=1; optimizer/eval excluded; analytical estimate, unfused latency not represented",
        "unmeasured_overheads": "diagnostics and checkpoint verification are not separately probed; prescribed 25% contingency covers them; live paired-block guard uses elapsed receipts",
    }
    write(args.root / "PLAN.json", payload)
    return payload


def run_stage(args, stage, remaining):
    p = json.loads((args.root / "PLAN.json").read_text())
    assert p["source"] == source(), "source changed after frozen plan"
    if (
        p["qualification"] != "qualified"
        or not p["preflight"]
        or not p["preflight"]["eligible_" + stage]
    ):
        raise RuntimeError(f"{stage} not eligible under frozen qualification/preflight")
    if not (args.root / "FROZEN.json").exists():
        raise RuntimeError("plan must be committed and frozen before selection")
    freeze = json.loads((args.root / "FROZEN.json").read_text())
    assert freeze["plan_sha256"] == digest_file(args.root / "PLAN.json")
    tensors, test, split, _ = dataset(args)
    arms = CORE if stage == "core" else CHANNELS
    decisions = {}
    rng = random.Random(20261010)
    index = []
    for phase_name, seeds in (("selection", (75001, 75002)), ("final", None)):
        for d in ((4, 16, 64) if stage == "core" else (16, 64)):
            current = (
                seeds
                if seeds is not None
                else range(75101, 75111 if d == 64 else 75106)
            )
            if phase_name == "final":
                for arm in arms:
                    rows = [
                        r
                        for r in index
                        if r["arm"] == arm
                        and r["depth"] == d
                        and r["phase"] == "selection"
                    ]
                    decisions[f"{arm}-d{d}"] = selection(rows)
                write(args.root / f"SELECTION-{stage}.json", decisions)
            for seed in current:
                jobs = [
                    j.copy()
                    for j in p["jobs"]
                    if j["stage"] == stage
                    and j["phase"] == phase_name
                    and j["depth"] == d
                    and j["seed"] == seed
                ]
                rng.shuffle(jobs)
                estimate = 0.0
                for j in jobs:
                    proxy = (
                        "R"
                        if j["arm"] in ("R", "L")
                        else ("ML" if j["arm"] in ("M", "ML") else "ML-O")
                    )
                    row = next(
                        r
                        for r in p["preflight"]["probes"]
                        if r["arm"] == proxy and r["depth"] == d
                    )
                    nsteps = j["epochs"] * 54
                    estimate += (
                        nsteps * row["seconds_per_step"]
                        + (math.ceil(nsteps / 10) + 1) * row["validation_seconds"]
                        + row["cold_seconds"]
                    )
                if estimate * 1.25 > remaining():
                    write(
                        args.root / f"STATUS-{stage}.json",
                        {
                            "status": "incomplete_budget",
                            "phase": phase_name,
                            "depth": d,
                            "seed": seed,
                            "required_block_seconds": estimate * 1.25,
                            "remaining_seconds": remaining(),
                        },
                    )
                    return
                for j in jobs:
                    lr = (
                        j["lr"]
                        if phase_name == "selection"
                        else decisions[f"{j['arm']}-d{d}"]["lr"]
                    )
                    if lr is None:
                        index.append({**j, "status": "missing_no_valid_rate"})
                        continue
                    name = f"{phase_name}-{j['arm']}-d{d}-s{seed}-lr{lr:g}.pt"
                    print("run", name, flush=True)
                    r = trajectory(
                        args,
                        j["arm"],
                        d,
                        seed,
                        lr,
                        j["epochs"],
                        p["sinkhorn_iters"],
                        tensors,
                        test,
                        split,
                        args.root / "runs" / name[:-3],
                        remaining,
                        phase_name == "final",
                    )
                    assert r["identity"] == p["initialization"][j["identity_key"]]
                    r["phase"] = phase_name
                    index.append(r)
                    write(args.root / f"RUNS-{stage}.json", index)
    write(
        args.root / f"STATUS-{stage}.json",
        {"status": "complete", "trajectories": len(index)},
    )


def paired_boot(values, level):
    v = np.asarray(values, float)
    rng = np.random.default_rng(20261010)
    means = v[rng.integers(0, len(v), (100000, len(v)))].mean(1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return [float(v.mean()), float(lo), float(hi)]


def analyze(args):
    q = json.loads((args.root / "QUALIFICATION.json").read_text())
    rows = []
    for stage in ("core", "channels"):
        file = args.root / f"RUNS-{stage}.json"
        if file.exists():
            rows += json.loads(file.read_text())
    final = [r for r in rows if r["phase"] == "final"]
    contrasts = []
    primary = [
        ("ML", "M"),
        ("ML", "L"),
        ("ML-O", "ML"),
        ("ML-O", "ML-S"),
        ("ML-O", "ML-C"),
    ]
    for d in (4, 16, 64):
        for cand, base in primary + [("M", "R"), ("ML-O", "M-O")]:
            c = {r["seed"]: r for r in final if r["arm"] == cand and r["depth"] == d}
            b = {r["seed"]: r for r in final if r["arm"] == base and r["depth"] == d}
            seeds = list(range(75101, 75111 if d == 64 else 75106))
            level = 0.99 if d == 64 and (cand, base) in primary else 0.95
            out = {
                "candidate": cand,
                "baseline": base,
                "depth": d,
                "level": level,
                "expected_seeds": seeds,
                "status": "missing_cells",
            }
            if all(
                s in c
                and s in b
                and c[s]["status"] == "complete"
                and b[s]["status"] == "complete"
                for s in seeds
            ):
                out["endpoint_ce_diff"] = paired_boot(
                    [c[s]["curve"][-1]["ce"] - b[s]["curve"][-1]["ce"] for s in seeds],
                    level,
                )
                out["endpoint_accuracy_diff_pp"] = paired_boot(
                    [
                        100
                        * (
                            c[s]["curve"][-1]["accuracy"]
                            - b[s]["curve"][-1]["accuracy"]
                        )
                        for s in seeds
                    ],
                    level,
                )
                hits = [
                    (crossing(c[s]["curve"]), crossing(b[s]["curve"])) for s in seeds
                ]
                out["attainment"] = {
                    "candidate": sum(a is not None for a, bh in hits),
                    "baseline": sum(bh is not None for a, bh in hits),
                }
                out["raw_crossings"] = [
                    {"seed": s, "candidate": a, "baseline": bh}
                    for s, (a, bh) in zip(seeds, hits)
                ]
                out["status"] = (
                    "censored"
                    if any(a is None or bh is None for a, bh in hits)
                    else "complete"
                )
                if out["status"] == "complete":
                    for metric in (
                        "step",
                        "examples",
                        "training_flops",
                        "training_seconds",
                        "elapsed_seconds",
                    ):
                        ratios = [
                            math.log(max(1e-12, a[metric]) / max(1e-12, bh[metric]))
                            for a, bh in hits
                        ]
                        out[metric + "_ratio"] = [
                            math.exp(x) for x in paired_boot(ratios, level)
                        ]
                    ratio = out["step_ratio"]
                    time_ratio = out["training_seconds_ratio"]
                    quality = (
                        out["endpoint_ce_diff"][2] <= 0.02
                        and out["endpoint_accuracy_diff_pp"][1] >= -0.5
                    )
                    out["beats_at_quality"] = ratio[2] < 1 and quality
                    out["step_nominee"] = ratio[0] <= 0.9 and ratio[2] < 1 and quality
                    out["practical_speedup"] = (
                        out["step_nominee"]
                        and time_ratio[0] <= 0.9
                        and time_ratio[2] < 1
                    )
            contrasts.append(out)
    interactions = []
    depth_changes = []
    for d in (4, 16, 64):
        cohort = list(range(75101, 75111 if d == 64 else 75106))
        table = {
            (r["arm"], r["seed"]): r
            for r in final
            if r["depth"] == d and r["status"] == "complete"
        }
        if all((a, s) in table for a in CORE for s in cohort):
            values = [
                table["ML", s]["curve"][-1]["ce"]
                - table["M", s]["curve"][-1]["ce"]
                - table["L", s]["curve"][-1]["ce"]
                + table["R", s]["curve"][-1]["ce"]
                for s in cohort
            ]
            result = {
                "depth": d,
                "ce_interaction_nats": paired_boot(values, 0.95),
                "raw_ce_interaction": values,
            }
            hits = {
                (a, s): crossing(table[a, s]["curve"]) for a in CORE for s in cohort
            }
            if all(h is not None for h in hits.values()):
                logs = [
                    math.log(hits["ML", s]["step"])
                    - math.log(hits["M", s]["step"])
                    - math.log(hits["L", s]["step"])
                    + math.log(hits["R", s]["step"])
                    for s in cohort
                ]
                result["log_step_interaction"] = paired_boot(logs, 0.95)
            else:
                result["log_step_interaction_status"] = "censored"
            interactions.append(result)
    for arm in ARMS:
        for shallow, deep in ((4, 16), (16, 64), (4, 64)):
            table = {
                (r["depth"], r["seed"]): r
                for r in final
                if r["arm"] == arm and r["status"] == "complete"
            }
            cohort = list(range(75101, 75106))
            if all((d, s) in table for d in (shallow, deep) for s in cohort):
                values = [
                    table[deep, s]["curve"][-1]["ce"]
                    - table[shallow, s]["curve"][-1]["ce"]
                    for s in cohort
                ]
                depth_changes.append(
                    {
                        "arm": arm,
                        "shallow": shallow,
                        "deep": deep,
                        "seeds": cohort,
                        "endpoint_ce_change": paired_boot(values, 0.95),
                    }
                )

    def contrast(candidate, baseline):
        return next(
            c
            for c in contrasts
            if c["depth"] == 64
            and c["candidate"] == candidate
            and c["baseline"] == baseline
        )

    nominees = {
        "uncoupled_combination": all(
            contrast("ML", a).get("step_nominee", False) for a in ("M", "L")
        ),
        "ordered_variant": contrast("ML-O", "ML").get("step_nominee", False),
        "ordered_depth": contrast("ML-O", "ML").get("step_nominee", False)
        and all(
            contrast("ML-O", a).get("beats_at_quality", False) for a in ("ML-S", "ML-C")
        ),
    }
    report = {
        "schema": SCHEMA,
        "qualification": q["status"],
        "completed_selection": sum(
            r["phase"] == "selection" and r["status"] == "complete" for r in rows
        ),
        "completed_final": sum(r["status"] == "complete" for r in final),
        "contrasts": contrasts,
        "factorial_interactions_exploratory": interactions,
        "within_arm_depth_changes_exploratory": depth_changes,
        "nominees": nominees,
        "budget": charged(args.root),
        "incomplete_reason": (
            "mandatory numerical qualification failed"
            if q["status"] != "qualified"
            else None
        ),
    }
    write(args.root / "ANALYSIS.json", report)
    if rows:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for field in ("step", "examples", "training_flops", "training_seconds"):
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for ax, d in zip(axes, (4, 16, 64)):
                for arm in ARMS:
                    rr = [
                        r
                        for r in final
                        if r["arm"] == arm
                        and r["depth"] == d
                        and r["status"] == "complete"
                    ]
                    for i, r in enumerate(rr):
                        ax.plot(
                            [x[field] for x in r["curve"]],
                            [x["ce"] for x in r["curve"]],
                            alpha=0.5,
                            label=arm if i == 0 else None,
                        )
                ax.set(title=f"{d} blocks", xlabel=field, ylabel="Validation CE (nats)")
                ax.axhline(0.1, ls=":", color="gray")
                if ax.lines:
                    ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(args.root / f"convergence-{field}.png")
            plt.close(fig)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for col, d in enumerate((4, 16, 64)):
            for arm in ARMS:
                tuning = [
                    r
                    for r in rows
                    if r["phase"] == "selection"
                    and r["arm"] == arm
                    and r["depth"] == d
                    and r["status"] == "complete"
                ]
                rates = sorted({r["lr"] for r in tuning})
                means = [
                    np.mean([r["curve"][-1]["ce"] for r in tuning if r["lr"] == lr])
                    for lr in rates
                ]
                if rates:
                    axes[0, col].plot(rates, means, "o-", label=arm)
                runs = [
                    r
                    for r in final
                    if r["arm"] == arm and r["depth"] == d and r["status"] == "complete"
                ]
                if runs:
                    axes[1, col].scatter(
                        [arm] * len(runs),
                        [r["peak_allocated"] / 2**30 for r in runs],
                        label=arm,
                    )
            axes[0, col].set(
                xscale="log",
                xlabel="Selection learning rate",
                ylabel="Endpoint validation CE",
                title=f"{d} blocks",
            )
            axes[1, col].set(xlabel="Architecture", ylabel="Peak allocated GiB")
            if axes[0, col].lines:
                axes[0, col].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.root / "lr-response-memory.png")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4))
        for arm in ARMS:
            ds = [
                d
                for d in (4, 16, 64)
                if any(
                    r["arm"] == arm and r["depth"] == d and r["status"] == "complete"
                    for r in final
                )
            ]
            means = [
                np.mean(
                    [
                        r["curve"][-1]["ce"]
                        for r in final
                        if r["arm"] == arm
                        and r["depth"] == d
                        and r["status"] == "complete"
                    ]
                )
                for d in ds
            ]
            if ds:
                ax.plot(ds, means, "o-", label=arm)
        ax.set(xlabel="MLP blocks", ylabel="Endpoint validation CE")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.root / "depth-quality.png")
        plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("probe", "qualify", "plan", "run-core", "run-channels", "analyze"),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.root = args.root.expanduser()
    args.data = args.data.expanduser()
    args.checkpoints = args.checkpoints.expanduser()
    args.root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(4, int(os.environ.get("OMP_NUM_THREADS", "4"))))
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("qualified GPU runtime unavailable")
    environment = {
        "source": source(),
        "python": sys.version,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "kernel": platform.release(),
        "host": platform.node(),
        "device": str(torch.cuda.get_device_properties(0)),
        "threads": torch.get_num_threads(),
        "deterministic": True,
        "matmul_precision": "highest",
        "argv": sys.argv,
        "telemetry": telemetry(),
    }
    write(args.root / f"ENVIRONMENT-{args.command}.json", environment)
    signal.signal(
        signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM"))
    )
    if args.command in ("plan", "analyze"):
        result = plan(args) if args.command == "plan" else analyze(args)
    else:
        with operation(args.root, args.command) as remaining:
            if args.command == "qualify":
                result = qualify(args, remaining)
            elif args.command == "probe":
                result = probe(args, remaining)
            else:
                result = run_stage(args, args.command[4:], remaining)
    print(
        json.dumps(
            {
                "command": args.command,
                "status": (
                    result.get("status") if isinstance(result, dict) else "complete"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

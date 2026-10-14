# SPDX-License-Identifier: MIT
"""Audit exact archived inputs without changing the finite projector."""

import argparse
import copy
import hashlib
import itertools
import os
import json
import platform
import time
from pathlib import Path

import torch

from research.subliminal_learning.connections import (
    ManifoldHyperConnection,
    sinkhorn_knopp_log,
)
from research.subliminal_learning.data import tensor_digest
from research.supervised_depth.numerics import (
    ConvergedSinkhorn,
    map_reference,
    relative_error,
)
from research.supervised_depth.study import source, telemetry, write


def walk(value, prefix=""):
    if torch.is_tensor(value):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from walk(item, f"{prefix}/{key}")
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield from walk(item, f"{prefix}/{i}")


def input_only(case):
    return {k: v for k, v in case.items() if k in ("logits", "h", "state", "cotangent")}


def verify_inputs(parent, root):
    qual = json.loads((parent / "QUALIFICATION.json").read_text())
    receipts, baseline = [], None
    for attempt in qual["attempts"]:
        receipt = attempt["reference"]["reference_receipt"]
        path = Path(receipt["path"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == receipt["sha256"] and path.stat().st_size == receipt["bytes"]
        payload = torch.load(path, map_location="cpu", weights_only=True)
        tensors = {
            k: {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "sha256": tensor_digest(v),
                "finite": bool(torch.isfinite(v).all()),
            }
            for k, v in walk(payload)
        }
        assert all(v["finite"] for v in tensors.values())
        inputs = {name: input_only(case) for name, case in payload.items()}
        if baseline is not None:
            assert {k: tensor_digest(v) for k, v in walk(inputs)} == {
                k: tensor_digest(v) for k, v in walk(baseline)
            }
        baseline = inputs
        receipts.append({**receipt, "verified": True, "tensors": tensors})
    # Raw payloads have no embedded source; the qualification manifest binds
    # their file hashes to the pinned source and its module hashes.
    expected = qual["source"]["module_hashes"]
    current = source()["module_hashes"]
    assert current["model.py"] == expected["model.py"]
    assert current["numerics.py"] == expected["numerics.py"]
    write(
        root / "INPUT_VERIFICATION.json",
        {
            "source": source(),
            "qualification_source": qual["source"],
            "source_binding": "external qualification manifest, not embedded in tensor payload",
            "all_counts_input_identity": True,
            "files": receipts,
        },
    )
    return payload


def rounded(value):
    if torch.is_tensor(value):
        return value.float().double()
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(v) for v in value]
    return value


def evaluate(case, count, dtype, device, converged=False, individual=False):
    """One batched VJP plus per-matrix parameter VJPs when requested."""
    t0 = time.monotonic()
    if device == "cuda":
        torch.cuda.synchronize()
    if "logits" in case:
        x = case["logits"].to(device=device, dtype=dtype).clone().requires_grad_()
        outputs = [
            ConvergedSinkhorn.apply(x) if converged else sinkhorn_knopp_log(x, count)
        ]
        targets, names = [x], ["logits"]
        cot = [case["cotangent"].to(x)]
    else:
        conn = ManifoldHyperConnection(
            1, 4, 8, 1, count, 1.0, 1e-5, log_domain=True
        ).to(device=device, dtype=dtype)
        conn.load_state_dict(case["state"])
        x = case["h"].to(device=device, dtype=dtype).clone().requires_grad_()
        outputs = list(map_reference(conn, x) if converged else conn.maps(x))
        targets = [x] + list(conn.parameters())
        names = ["h"] + [n for n, _ in conn.named_parameters()]
        cot = [v.to(x) for v in case["cotangent"]]
    objective = sum((a * b).sum() for a, b in zip(outputs, cot))
    grads = torch.autograd.grad(
        objective, targets, retain_graph=individual and len(outputs) > 1
    )
    per = []
    n = x.shape[0] if len(outputs) == 1 else x.shape[1]
    if individual:
        if len(outputs) == 1:
            per = [{"logits": grads[0][i].detach().cpu()} for i in range(n)]
        else:
            for i in range(n):
                obj = sum((a[:, i] * b[:, i]).sum() for a, b in zip(outputs, cot))
                gs = torch.autograd.grad(obj, targets, retain_graph=i < n - 1)
                per.append(
                    {
                        name: (v[:, i] if name == "h" else v).detach().cpu()
                        for name, v in zip(names, gs)
                    }
                )
    if device == "cuda":
        torch.cuda.synchronize()
    return {
        "outputs": [v.detach().cpu() for v in outputs],
        "grads": {name: v.detach().cpu() for name, v in zip(names, grads)},
        "per_matrix_grads": per,
        "elapsed_seconds": time.monotonic() - t0,
        "dtype": str(dtype),
        "device": device,
        "count": count,
        "finite": bool(all(torch.isfinite(v).all() for v in outputs + list(grads))),
    }


def matrices(result):
    return result["outputs"][-1].reshape(-1, 4, 4)


def compare(actual, reference):
    pa, pr = matrices(actual), matrices(reference)
    rows = []
    for i, (a, b) in enumerate(zip(pa, pr)):
        row = (a.double().sum(-1) - 1).abs()
        col = (a.double().sum(-2) - 1).abs()
        diff = (a.double() - b.double()).abs()
        errors = {
            k: relative_error(v, reference["per_matrix_grads"][i][k])
            for k, v in actual["per_matrix_grads"][i].items()
        }
        rows.append(
            {
                "matrix_index": i,
                "row_residual": row.max().item(),
                "row_index": row.argmax().item(),
                "column_residual": col.max().item(),
                "column_index": col.argmax().item(),
                "entry_max_error": diff.max().item(),
                "entry_index_flat": diff.argmax().item(),
                "vjp": errors,
                "finite": actual["finite"] and reference["finite"],
                "dtype": actual["dtype"],
                "device": actual["device"],
                "iterations": actual["count"],
                "elapsed_seconds_batched_evaluation_and_per_matrix_vjps": actual[
                    "elapsed_seconds"
                ],
            }
        )
    # The mandatory gate retains its original batched parameter VJP norm.
    # Matrix-specific parameter VJPs are additional diagnostics, not a new gate.
    aggregate = {
        k: relative_error(v, reference["grads"][k]) for k, v in actual["grads"].items()
    }
    res = max((pa.sum(-1) - 1).abs().max().item(), (pa.sum(-2) - 1).abs().max().item())
    ref_res = max(
        (pr.sum(-1) - 1).abs().max().item(), (pr.sum(-2) - 1).abs().max().item()
    )
    return {
        "per_matrix": rows,
        "aggregate_vjp": aggregate,
        "row_column_residual": res,
        "reference_residual": ref_res,
        "maxima_matrix_indices": {
            key: max(rows, key=lambda r: r[key])["matrix_index"]
            for key in ("row_residual", "column_residual", "entry_max_error")
        },
        "vjp_maximum": max(
            (
                dict(matrix_index=r["matrix_index"], parameter=k, **v)
                for r in rows
                for k, v in r["vjp"].items()
            ),
            key=lambda v: (
                v["relative_error"]
                if v["relative_error"] is not None
                else v["absolute_error"]
            ),
        ),
        "pass": actual["finite"]
        and reference["finite"]
        and res <= 1e-4
        and ref_res <= 1e-10
        and all(v["pass"] for v in aggregate.values()),
    }


def geometry(ref):
    eye = torch.eye(4, dtype=torch.double)
    perms = list(itertools.permutations(range(4)))
    return [
        {
            "matrix_index": i,
            "singular_values": torch.linalg.svdvals(p).tolist(),
            "nearest_permutation_frobenius": min(
                (p - eye[list(order)]).norm().item() for order in perms
            ),
            "descriptive_only": True,
        }
        for i, p in enumerate(matrices(ref))
    ]


def finite_differences(case, count):
    logits, cot = case["logits"], case["cotangent"]
    x = logits.clone().requires_grad_()
    g = torch.autograd.grad((sinkhorn_knopp_log(x, count) * cot).sum(), x)[0]
    generator = torch.Generator().manual_seed(75092)
    rows = []
    for i in range(3):
        direction = torch.randn(logits.shape, generator=generator, dtype=torch.double)
        direction /= direction.norm()
        analytic = (g * direction).sum().item()
        values = []
        for eps in (1e-4, 1e-5, 1e-6):
            with torch.no_grad():
                high = (sinkhorn_knopp_log(logits + eps * direction, count) * cot).sum()
                low = (sinkhorn_knopp_log(logits - eps * direction, count) * cot).sum()
            numeric = ((high - low) / (2 * eps)).item()
            tolerance = 1e-8 + 1e-5 * max(abs(analytic), abs(numeric))
            values.append(
                {
                    "epsilon": eps,
                    "analytic": analytic,
                    "numeric": numeric,
                    "absolute_difference": abs(analytic - numeric),
                    "tolerance": tolerance,
                    "pass": abs(analytic - numeric) <= tolerance,
                }
            )
        rows.append(
            {
                "direction": i,
                "direction_sha256": tensor_digest(direction),
                "values": values,
                "pass": any(
                    a["pass"] and b["pass"] for a, b in zip(values, values[1:])
                ),
            }
        )
    return {
        "count": count,
        "seed": 75092,
        "dtype": "torch.float64",
        "device": "cpu",
        "rows": rows,
        "pass": all(r["pass"] for r in rows),
    }


def cpu_audit(payload, root, checkpoints):
    report = {
        "source": source(),
        "thresholds": {
            "residual": 1e-4,
            "reference_residual": 1e-10,
            "relative_vjp": 1e-3,
            "small_norm": 1e-8,
            "small_absolute": 1e-6,
        },
        "cases": {},
        "finite_differences": [],
    }
    saved = {}
    for name, case in payload.items():
        full = evaluate(case, 0, torch.double, "cpu", converged=True, individual=True)
        round_case = rounded(input_only(case))
        ref = evaluate(
            round_case, 0, torch.double, "cpu", converged=True, individual=True
        )
        # Verify reproduction of the archived full-precision reference.
        old_outputs = (
            case["reference"]
            if isinstance(case["reference"], (list, tuple))
            else [case["reference"]]
        )
        old_grads = (
            case["reference_vjp"]
            if isinstance(case["reference_vjp"], (list, tuple))
            else [case["reference_vjp"]]
        )
        assert all(torch.equal(a, b) for a, b in zip(full["outputs"], old_outputs))
        assert all(torch.equal(a, b) for a, b in zip(full["grads"].values(), old_grads))
        rows = []
        count_results = {}
        for count in (20, 50, 100, 200):
            c32 = evaluate(case, count, torch.float, "cpu", individual=True)
            c64 = evaluate(round_case, count, torch.double, "cpu", individual=True)
            rows.append(
                {
                    "count": count,
                    "cpu_fp32_vs_full_reference": compare(c32, full),
                    "cpu_fp32_vs_rounded_reference": compare(c32, ref),
                    "cpu_fp64_rounded_vs_rounded_reference": compare(c64, ref),
                    "cpu_fp32_vs_matching_cpu_fp64": compare(c32, c64),
                }
            )
            count_results[count] = {"cpu32": c32, "cpu64": c64}
        report["cases"][name] = {
            "input_quantization": compare(ref, full),
            "geometry_full": geometry(full),
            "geometry_rounded": geometry(ref),
            "counts": rows,
        }
        saved[name] = {
            "full_reference": full,
            "rounded_reference": ref,
            "counts": count_results,
        }
        print("cpu audit", name, flush=True)
    for count in (20, 200):
        report["finite_differences"].append(
            finite_differences(payload["sharp_random"], count)
        )
    report["finite_loop_derivative_pass"] = all(
        v["pass"] for v in report["finite_differences"]
    )
    assert report["finite_loop_derivative_pass"]
    path = checkpoints / "cpu-audit.pt"
    assert not path.exists()
    torch.save(saved, path)
    report["cpu_receipt"] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    write(root / "ERROR_DECOMPOSITION.json", report)


def gpu_audit(payload, root, checkpoints):
    report = json.loads((root / "ERROR_DECOMPOSITION.json").read_text())
    receipt = report["cpu_receipt"]
    path = Path(receipt["path"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["sha256"]
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert report["finite_loop_derivative_pass"]
    qualification = {
        "source": source(),
        "status": "synthetic_in_progress",
        "selected_iters": None,
        "attempts": [],
        "training": [],
        "thresholds": report["thresholds"],
    }
    selected = None
    gpu_saved = {}
    for count in (20, 50, 100, 200, 400, 800, 1600, 3200):
        rows = []
        gpu_saved[count] = {}
        for name, case in payload.items():
            actual = evaluate(case, count, torch.float, "cuda", individual=True)
            cache = saved[name]
            if count not in cache["counts"]:
                cache["counts"][count] = {
                    "cpu32": evaluate(case, count, torch.float, "cpu", individual=True),
                    "cpu64": evaluate(
                        rounded(input_only(case)),
                        count,
                        torch.double,
                        "cpu",
                        individual=True,
                    ),
                }
            same = cache["counts"][count]
            checks = {
                "gpu_fp32_vs_full_reference": compare(actual, cache["full_reference"]),
                "gpu_fp32_vs_rounded_reference": compare(
                    actual, cache["rounded_reference"]
                ),
                "gpu_fp32_vs_cpu_fp32": compare(actual, same["cpu32"]),
                "gpu_fp32_vs_matching_cpu_fp64": compare(actual, same["cpu64"]),
            }
            matching_ok = all(
                v["pass"]
                for v in checks["gpu_fp32_vs_matching_cpu_fp64"][
                    "aggregate_vjp"
                ].values()
            )
            rows.append(
                {
                    "case": name,
                    **checks,
                    "matching_count_vjp_pass": matching_ok,
                    "pass": checks["gpu_fp32_vs_full_reference"]["pass"],
                }
            )
            gpu_saved[count][name] = actual
            if count > 200:
                report["cases"][name]["counts"].append(
                    {
                        "count": count,
                        "cpu_fp32_vs_full_reference": compare(
                            same["cpu32"], cache["full_reference"]
                        ),
                        "cpu_fp32_vs_rounded_reference": compare(
                            same["cpu32"], cache["rounded_reference"]
                        ),
                        "cpu_fp64_rounded_vs_rounded_reference": compare(
                            same["cpu64"], cache["rounded_reference"]
                        ),
                        "cpu_fp32_vs_matching_cpu_fp64": compare(
                            same["cpu32"], same["cpu64"]
                        ),
                    }
                )
        attempt = {
            "count": count,
            "cases": rows,
            "pass": all(r["pass"] for r in rows),
            "matching_count_vjp_pass": all(r["matching_count_vjp_pass"] for r in rows),
        }
        qualification["attempts"].append(attempt)
        write(root / "QUALIFICATION.json", qualification)
        print(
            "GPU synthetic",
            count,
            attempt["pass"],
            attempt["matching_count_vjp_pass"],
            flush=True,
        )
        if not attempt["matching_count_vjp_pass"]:
            qualification["status"] = "blocked_matching_count_vjp"
            break
        if attempt["pass"]:
            fd = finite_differences(payload["sharp_random"], count)
            report["finite_differences"].append(fd)
            if fd["pass"]:
                selected = count
            break
    report["gpu_comparisons"] = qualification["attempts"]
    report["source"] = source()
    qualification["selected_iters"] = selected
    qualification["status"] = (
        "synthetic_pass_provisional" if selected else "numerical_blocker"
    )
    qualification["training_qualification_status"] = (
        "unstarted_pending_feasibility" if selected else "unstarted_failed_synthetic"
    )
    for name, data in (("cpu-extended.pt", saved), ("gpu-audit.pt", gpu_saved)):
        path = checkpoints / name
        assert not path.exists()
        torch.save(data, path)
        qualification.setdefault("receipts", []).append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    write(root / "ERROR_DECOMPOSITION.json", report)
    write(root / "QUALIFICATION.json", qualification)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("cpu", "gpu", "probe"))
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(4, int(os.environ.get("OMP_NUM_THREADS", "4"))))
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    write(
        args.root / f"ENVIRONMENT-audit-{args.command}.json",
        {
            "source": source(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "platform": platform.platform(),
            "telemetry": telemetry(),
            "device": (
                str(torch.cuda.get_device_properties(0))
                if args.command != "cpu"
                else "cpu"
            ),
        },
    )
    if args.command == "probe":
        from research.supervised_depth.study import probe

        started = float(os.environ["KNLP_GPU_COMMAND_STARTED_MONOTONIC"])
        allowed = float(os.environ["KNLP_GPU_REMAINING_SECONDS"])
        probe(args, lambda: allowed - (time.monotonic() - started))
        return
    payload = verify_inputs(args.parent, args.root)
    if args.command == "cpu":
        cpu_audit(payload, args.root, args.checkpoints)
    else:
        gpu_audit(payload, args.root, args.checkpoints)


if __name__ == "__main__":
    main()

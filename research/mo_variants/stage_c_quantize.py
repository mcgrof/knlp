"""Fixed primary, attribution and BF16 storage screens on CPU only."""

import os
import argparse
import gc
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from research.tinystories_depth.model import digest
from research.tinystories_depth.probe import evaluate
from research.tinystories_depth.quantize import quantize_rows
from research.tinystories_depth.train import state_hash
from .model import Model
from .stage_c_common import read, sha, write


@torch.no_grad()
def groupwise(weight, bits):
    rows, cols = weight.shape
    padded = F.pad(weight, (0, (-cols) % 128)).reshape(rows, -1, 128)
    scale = padded.abs().amax(-1, keepdim=True) / (2 ** (bits - 1) - 1)
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    qmax = 2 ** (bits - 1) - 1
    result = (torch.round(padded / safe).clamp(-qmax, qmax) * scale).reshape(rows, -1)[
        :, :cols
    ]
    return result, scale


def gpu_fds():
    found = []
    for p in Path("/proc/self/fd").iterdir():
        try:
            target = str(p.resolve())
            if target.startswith(("/dev/nvidia", "/dev/kfd", "/dev/dri/render")):
                found.append(target)
        except OSError:
            pass
    return found


def code_parameter(name):
    return name.startswith("angular.") or name in (
        "strength_logits",
        "free_angles",
        "normalized_frequency_logits",
        "normalized_offsets",
    )


@torch.no_grad()
def schedule(model):
    if model.variant == "V00":
        return []
    output = []
    for site in range(2):
        for layer in range(model.depth):
            code = model.code(layer, site)
            strength = torch.as_tensor(model.strength(layer, site))
            beta = torch.tensor([0.3, 0.7, 1.2, 1.6])
            slack = torch.cat((beta, 2 - beta)).amin()
            # A fixed interior coefficient fixture records insertion sensitivity.
            write_delta = strength * slack * code
            carry_delta = strength * torch.outer(code, code) / 2
            base = torch.eye(4) * 2 - 1
            from research.tinystories_depth.model import mhc

            carry = mhc.sinkhorn_normalize_ref(base + carry_delta, repeat=20, eps=1e-6)
            output.append(
                {
                    "site": site,
                    "layer": layer,
                    "phase": model.phase(layer, site).item(),
                    "strength": strength.item(),
                    "code": code.tolist(),
                    "write_perturbation": write_delta.tolist(),
                    "carry_logit_perturbation": carry_delta.tolist(),
                    "carry_matrix_fixture": carry.tolist(),
                }
            )
    return output


@torch.no_grad()
def capture(model, tokens):
    model.capture = True
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        model(
            torch.from_numpy(np.asarray(tokens[:257], dtype=np.int64)).unsqueeze(0)[
                :, :-1
            ]
        )
    model.capture = False
    return [
        {
            "sublayer": s["sublayer"],
            "carry": s["carry"].clone(),
            "coded_write": s["coded_write"].clone(),
            "effective_coefficient_perturbation_rms": s[
                "effective_coefficient_perturbation_rms"
            ],
            "minimum_original_margin": s["margins"].min().item(),
        }
        for s in model.samples
    ]


def endpoint(root, jobid, benchmark=False):
    root = Path(root)
    torch.set_num_threads(min(8, int(os.environ.get("OMP_NUM_THREADS", "8"))))
    assert not torch.cuda.is_initialized()
    assert not gpu_fds()
    r = root / "records"
    result = read(r / "runs" / jobid / "RESULT.json")
    assert result["status"] == "complete"
    job = result["metadata"]["job"]
    destination = r / "quantization" / (jobid + ".json")
    assert not destination.exists()
    receipt = result["checkpoint"]
    path = root / "checkpoints" / jobid / Path(receipt["path"]).name
    assert sha(path) == receipt["sha256"]
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert state_hash(saved["model"]) == receipt["parameter_sha256"]
    model = Model(
        job["variant"], job["depth"], job["seed"], mapping=job.get("mapping", "ordered")
    )
    model.load_state_dict(saved["model"], strict=True)
    model.step = saved["step"]
    del saved
    model.requires_grad_(False)
    array = root / (
        "data/discovery-benchmark-final.npy"
        if benchmark
        else "data/confirmation/final.npy"
    )
    if not benchmark:
        assert (
            sha(array)
            == read(r / "STAGE-C-PLAN.json")["data"]["confirmation_files"]["final.npy"]
        )
    tokens = np.load(array, mmap_mode="r")
    assert len(tokens) == 131073
    originals = {n: p.clone() for n, p in model.named_parameters()}
    named = dict(model.named_parameters())
    weights = {
        "lm." + n + ".weight": module.weight
        for n, module in model.lm.named_modules()
        if isinstance(module, (nn.Linear, nn.Embedding))
    }
    blocks = {n: p for n, p in weights.items() if n.startswith("lm.model.layers.")}
    total = sum(p.numel() for p in named.values())
    buffer_bytes = sum(
        p.numel() * p.element_size()
        for n, p in model.state_dict().items()
        if n not in named
    )
    original_schedule = schedule(model)
    record = {
        "id": jobid,
        "row": job.get("row", job["variant"]),
        "seed": job["seed"],
        "status": "running",
        "scope": (
            "cost benchmark on four repeats of discovery tokens; no confirmation evaluated"
            if benchmark
            else "independent confirmation final array"
        ),
        "checkpoint_sha256": receipt["sha256"],
        "array_sha256": sha(array),
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "total_parameters": total,
        "persistent_buffer_bytes": buffer_bytes,
        "gpu_device_handles": [],
        "baseline_schedule": original_schedule,
        "cases": [],
        "weight_diagnostics": [],
    }
    began = time.perf_counter()
    record["clean_ce"] = evaluate(model, tokens, "cpu")
    record["clean_seconds"] = time.perf_counter() - began
    record["paired_gpu_clean_ce"] = result.get("final_gpu_clean_ce")
    record["cpu_minus_gpu_clean_ce"] = (
        record["clean_ce"] - result["final_gpu_clean_ce"]
        if result.get("final_gpu_clean_ce") is not None
        else None
    )
    write(destination, record)
    cases = [("rowwise" + str(b), b, "rows", False, None) for b in (8, 4, 3, 2)]
    cases += [("group128-" + str(b), b, "groups", False, None) for b in (4, 3)]
    cases += [("blocks" + str(b), b, "rows", True, None) for b in (4, 3)]
    cases += [
        ("rowwise4-scalars-bf16-storage", 4, "rows", False, "scalars"),
        ("rowwise4-routing-code-bf16-storage", 4, "rows", False, "routing-code"),
    ]
    int4 = None
    int4_samples = None
    for label, bits, method, block_only, rounding in cases:
        before = time.perf_counter()
        coverage = blocks if block_only else weights
        hashes, scales, count, errors = {}, 0, 0, []
        with torch.no_grad():
            for n, p in named.items():
                p.copy_(originals[n])
            for n, p in coverage.items():
                w = originals[n]
                q, scale = (
                    groupwise(w, bits) if method == "groups" else quantize_rows(w, bits)
                )
                p.copy_(q)
                hashes[n] = digest(q)
                scales += scale.numel()
                count += p.numel()
                row_rms = w.square().mean(-1).sqrt()
                max_to_rms = w.abs().amax(-1) / row_rms.clamp_min(1e-30)
                errors.append(
                    {
                        "weight": n,
                        "relative_l2": (q - w).norm().item()
                        / max(w.norm().item(), 1e-30),
                        "row_max_to_rms_mean": max_to_rms.mean().item(),
                        "row_max_to_rms_max": max_to_rms.max().item(),
                    }
                )
            rounded = []
            if rounding:
                for n, p in named.items():
                    eligible = code_parameter(n) or (
                        rounding == "routing-code"
                        and n.startswith("routes.")
                        and not n.endswith("norm_weight")
                    )
                    if eligible:
                        p.copy_(p.bfloat16().float())
                        rounded.append(n)
            protected = {
                n: digest(p)
                for n, p in named.items()
                if n not in coverage and n not in rounded
            }
            assert all(digest(originals[n]) == h for n, h in protected.items())
        ce = evaluate(model, tokens, "cpu")
        delta = ce - record["clean_ce"]
        phase = schedule(model) if rounding else []
        phase_errors = []
        for base, current in zip(original_schedule, phase):
            phase_errors.append(
                {
                    "site": base["site"],
                    "layer": base["layer"],
                    "phase_error": current["phase"] - base["phase"],
                    "strength_error": current["strength"] - base["strength"],
                    "write_perturbation_l2_error": (
                        torch.tensor(current["write_perturbation"])
                        - torch.tensor(base["write_perturbation"])
                    )
                    .norm()
                    .item(),
                    "carry_logit_perturbation_l2_error": (
                        torch.tensor(current["carry_logit_perturbation"])
                        - torch.tensor(base["carry_logit_perturbation"])
                    )
                    .norm()
                    .item(),
                    "carry_matrix_fixture_l2_error": (
                        torch.tensor(current["carry_matrix_fixture"])
                        - torch.tensor(base["carry_matrix_fixture"])
                    )
                    .norm()
                    .item(),
                }
            )
        rounded_elements = sum(named[n].numel() for n in rounded)
        case = {
            "name": label,
            "bits": bits,
            "ce": ce,
            "delta_ce": delta,
            "within_point1_nats": delta <= 0.1,
            "perplexity_ratio": math.exp(delta),
            "seconds": time.perf_counter() - before,
            "quantized_parameters": count,
            "coverage": count / total,
            "scales": scales,
            "scale_bytes": scales * 4,
            "unquantized_parameter_bytes": (total - count) * 4 - rounded_elements * 2,
            "rounded_control_elements": rounded_elements,
            "ideal_packed_total_bytes": (count * bits + 7) // 8
            + scales * 4
            + (total - count) * 4
            - rounded_elements * 2
            + buffer_bytes,
            "dequantized_weight_sha256": hashes,
            "protected_parameter_sha256": protected,
            "rounded_parameters": rounded,
            "rounded_parameter_sha256": {n: digest(named[n]) for n in rounded},
            "phase_and_perturbation_errors": phase_errors,
        }
        if label == "rowwise4":
            int4 = ce
            int4_samples = capture(model, tokens)
            record["int4_sampled_slack_and_perturbation"] = [
                {k: v for k, v in x.items() if k not in ("carry", "coded_write")}
                for x in int4_samples
            ]
        if rounding:
            case["delta_ce_vs_int4_fp32_controls"] = ce - int4
            rounded_samples = capture(model, tokens)
            case["actual_sampled_storage_rounding_errors"] = [
                {
                    "sublayer": a["sublayer"],
                    "carry_l2_error": (a["carry"] - b["carry"]).norm().item(),
                    "coded_write_l2_error": (a["coded_write"] - b["coded_write"])
                    .norm()
                    .item(),
                    "perturbation_rms_change": a[
                        "effective_coefficient_perturbation_rms"
                    ]
                    - b["effective_coefficient_perturbation_rms"],
                }
                for a, b in zip(rounded_samples, int4_samples)
            ]
            case["storage_rounding_sample_scope"] = (
                "Same fixed final-array inputs and INT4 backbone; full rounded model path, including later changed activations"
            )
        record["cases"].append(case)
        record["weight_diagnostics"].append({"case": label, "weights": errors})
        assert not gpu_fds()
        write(destination, record)
    with torch.no_grad():
        for n, p in named.items():
            p.copy_(originals[n])
    assert all(digest(p) == digest(originals[n]) for n, p in named.items())
    record.update(
        status="complete",
        total_cpu_seconds=time.perf_counter() - began,
        gpu_device_handles=gpu_fds(),
        memory_claim="ideal packed bytes only; dequantized evaluation does not measure packed allocation or kernels",
    )
    write(destination, record)
    del model, named, originals
    gc.collect()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--job-id", required=True)
    p.add_argument("--benchmark", action="store_true")
    a = p.parse_args()
    endpoint(a.root, a.job_id, a.benchmark)

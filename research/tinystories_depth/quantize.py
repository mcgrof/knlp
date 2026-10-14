"""Fixed-checkpoint, per-row symmetric weight precision screen on CPU."""

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .data import sha
from .model import Model, digest
from .probe import evaluate


@torch.no_grad()
def quantize_rows(weight, bits):
    qmax = 2 ** (bits - 1) - 1
    scale = weight.abs().amax(dim=1, keepdim=True) / qmax
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    integer = torch.round(weight / safe).clamp(-qmax, qmax)
    return integer * scale, scale


def run(root, data, threads=8):
    root, data = Path(root), Path(data)
    plan = json.loads((root / "PLAN.json").read_text())
    torch.set_num_threads(threads)
    final = np.load(data / "final.npy", mmap_mode="r")
    output = {
        "protocol": plan["quantization"],
        "plan_sha256": sha(root / "PLAN.json"),
        "final_tokens_sha256": sha(data / "final.npy"),
        "threads": threads,
        "device": "cpu",
        "runs": [],
        "missing": [],
    }
    directory = root / "quantization"
    directory.mkdir(exist_ok=True)
    for cell in plan["cells"]:
        for job in cell["finals"]:
            rp = root / "runs" / job["id"] / "RESULT.json"
            if not rp.exists():
                output["missing"].append(job["id"])
                continue
            result = json.loads(rp.read_text())
            if result["status"] != "complete":
                output["missing"].append(job["id"])
                continue
            checkpoint = Path(result["checkpoint"]["path"])
            assert sha(checkpoint) == result["checkpoint"]["sha256"]
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = Model(job["arm"], job["depth"], job["seed"])
            model.load_state_dict(payload["model"], strict=True)
            del payload
            model.requires_grad_(False)
            originals = {
                name + ".weight": module.weight.detach().clone()
                for name, module in model.lm.named_modules()
                if isinstance(module, (nn.Linear, nn.Embedding))
            }
            parameters = dict(model.lm.named_parameters())
            count = sum(p.numel() for p in originals.values())
            rows = sum(p.shape[0] for p in originals.values())
            total = sum(p.numel() for p in model.parameters())
            record = {
                "id": job["id"],
                "cell": cell["id"],
                "seed": job["seed"],
                "checkpoint_sha256": sha(checkpoint),
                "trained_gpu_ce": result["final_report_ce"],
                "quantized_parameters": count,
                "total_parameters": total,
                "coverage": count / total,
                "row_scales": rows,
                "bits": [],
            }
            started = time.perf_counter()
            record["cpu_unquantized_ce"] = evaluate(model, final, "cpu")
            record["cpu_unquantized_seconds"] = time.perf_counter() - started
            record["cpu_minus_gpu_unquantized_ce"] = (
                record["cpu_unquantized_ce"] - record["trained_gpu_ce"]
            )
            for bits in plan["quantization"]["bits"]:
                before = time.perf_counter()
                hashes = {}
                with torch.no_grad():
                    for name, weight in originals.items():
                        dequantized, _ = quantize_rows(weight, bits)
                        parameters[name].copy_(dequantized)
                        hashes[name] = digest(dequantized)
                ce = evaluate(model, final, "cpu")
                delta = ce - record["cpu_unquantized_ce"]
                record["bits"].append(
                    {
                        "bits": bits,
                        "ce": ce,
                        "delta_ce": delta,
                        "pass": delta <= plan["quantization"]["max_ce_increase"],
                        "seconds_including_dequantization": time.perf_counter()
                        - before,
                        "dequantized_sha256": hashes,
                        "ideal_storage_bytes": (count * bits + 7) // 8
                        + rows * 4
                        + (total - count) * 4,
                    }
                )
                with torch.no_grad():
                    for name, weight in originals.items():
                        parameters[name].copy_(weight)
                assert all(
                    digest(parameters[n]) == digest(w) for n, w in originals.items()
                )
            destination = directory / (job["id"] + ".json")
            destination.write_text(json.dumps(record, indent=2) + "\n")
            output["runs"].append(
                {"id": job["id"], "path": str(destination), "sha256": sha(destination)}
            )
            (root / "QUANTIZATION.json").write_text(json.dumps(output, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "id": job["id"],
                        "baseline_ce": record["cpu_unquantized_ce"],
                        "deltas": {b["bits"]: b["delta_ce"] for b in record["bits"]},
                    }
                ),
                flush=True,
            )
            del model, originals, parameters
            gc.collect()
    output["cells"] = {}
    for cell in plan["cells"]:
        records = [
            json.loads(Path(r["path"]).read_text())
            for r in output["runs"]
            if r["id"] in {j["id"] for j in cell["finals"]}
        ]
        if len(records) != 3:
            output["cells"][cell["id"]] = {
                "status": "incomplete",
                "lowest_passing_tested_bits": None,
            }
            continue
        passed = [
            bits
            for bits in plan["quantization"]["bits"]
            if all(
                next(b["pass"] for b in r["bits"] if b["bits"] == bits) for r in records
            )
        ]
        output["cells"][cell["id"]] = {
            "status": "three_seeds",
            "lowest_passing_tested_bits": min(passed) if passed else None,
            "all_seed_passing_bits": passed,
        }
    (root / "QUANTIZATION.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    run(args.root, args.data, args.threads)

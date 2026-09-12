"""Export a continuous UFO actor to the NumPy-only shadow runtime format."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from rl.flight.contracts import FlightContract


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_checkpoint(contract: FlightContract, checkpoint: Path, output: Path) -> None:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = saved.get("state", {}) if isinstance(saved, dict) else {}
    if state.get("action_kind", "continuous") != "continuous":
        raise ValueError("flight actor export requires a continuous checkpoint")
    weights = saved["agent"] if isinstance(saved, dict) and "agent" in saved else saved
    required = {
        "observation_mean",
        "observation_scale",
        "action_mid",
        "action_scale",
        "actor_mean.0.weight",
        "actor_mean.0.bias",
        "actor_mean.2.weight",
        "actor_mean.2.bias",
        "actor_mean.4.weight",
        "actor_mean.4.bias",
    }
    missing = required - set(weights)
    if missing:
        raise ValueError(f"checkpoint is missing actor parameters: {sorted(missing)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError(f"refusing to replace actor export: {output}")
    arrays = {
        "format_version": np.asarray(1, dtype=np.int64),
        "contract_hash": np.asarray(contract.digest),
        "checkpoint_sha256": np.asarray(sha256(checkpoint)),
        "observation_mean": weights["observation_mean"].detach().cpu().numpy(),
        "observation_scale": weights["observation_scale"].detach().cpu().numpy(),
        "action_mid": weights["action_mid"].detach().cpu().numpy(),
        "action_scale": weights["action_scale"].detach().cpu().numpy(),
    }
    for export_index, layer_index in enumerate((0, 2, 4)):
        for kind in ("weight", "bias"):
            arrays[f"layer_{export_index}_{kind}"] = (
                weights[f"actor_mean.{layer_index}.{kind}"].detach().cpu().numpy()
            )
    np.savez(output, **arrays)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    contract = FlightContract.from_json(args.contract)
    export_checkpoint(contract, args.checkpoint, args.output)
    report = {
        "schema_version": 1,
        "format": "ufo_numpy_actor",
        "contract_hash": contract.digest,
        "checkpoint_sha256": sha256(args.checkpoint),
        "model_sha256": sha256(args.output),
        "model_bytes": args.output.stat().st_size,
    }
    manifest = args.output.with_suffix(args.output.suffix + ".json")
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

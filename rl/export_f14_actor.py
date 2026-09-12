"""Export an F-14 motor checkpoint to the NumPy-only actor format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl.envs.fighter_env import DEFAULT_CONTRACT
from rl.export_ufo_actor import export_checkpoint, sha256
from rl.flight.contracts import FlightContract


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args(argv)
    contract = FlightContract.from_json(args.contract)
    export_checkpoint(contract, args.checkpoint, args.output)
    report = {
        "schema_version": 1,
        "format": "f14_numpy_actor",
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

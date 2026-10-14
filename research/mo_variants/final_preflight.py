"""Final source qualification, controls, and runner smoke receipts."""

import argparse
import json
import time
from pathlib import Path

from research.tinystories_depth.data import sha
from .preflight import write
from .qualify import run as qualify
from .matrix_manifest import inspect
from .job import resume_fixture, run_loop_fixture


def run(root):
    root = Path(root)
    started = time.perf_counter()
    record = {
        "status": "running",
        "scope": "fixtures only; no outcome jobs",
        "source_files": {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
    }
    path = root / "records/FINAL-PREFLIGHT.json"
    write(path, record)
    try:
        qualify(root / "records/variant-fixtures-final", "cuda")
        inspect(root / "records", "cuda")
        checkpoints = root / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        resume_fixture(checkpoints, "cuda")
        run_loop_fixture(root)
        record["status"] = "complete"
        record["resume_fixture"] = {
            "path": str(checkpoints / "RESUME-FIXTURE-cuda.json"),
            "sha256": sha(checkpoints / "RESUME-FIXTURE-cuda.json"),
        }
        record["run_loop_fixture"] = {
            "path": str(root / "runner-preflight/RUN-LOOP-FIXTURE.json"),
            "sha256": sha(root / "runner-preflight/RUN-LOOP-FIXTURE.json"),
        }
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = repr(exc)
        raise
    finally:
        record["elapsed_seconds"] = time.perf_counter() - started
        write(path, record)
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    run(p.parse_args().root)

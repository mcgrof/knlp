# SPDX-License-Identifier: MIT
"""Validate and import released document banks without model inference."""

import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.environ.update(
        CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES=""
    )
    from .run import configuration, doctor, LMCACHE
    from .worker import ArtifactEmbeddingStore
    from .artifact_set import install_bytes
    from .environment import verify_package, dependency_problems

    # Import creates the first receipt; it cannot require that receipt yet.
    # Normal inference configuration always resolves and validates its pins.
    cfg = configuration(args.config, resolve_artifact_sets=False)
    if cfg["profile"] != "lmcache":
        raise ValueError("document import requires the LMCache profile")
    started = time.monotonic()
    checked = doctor(cfg)
    verified = time.monotonic()
    inputs = checked["inputs"]
    sys.path.insert(0, inputs["source"])
    from memorilla.embeddings import EmbeddingStore

    root = Path(cfg["root"])
    import lmcache

    verify_package(lmcache, root / "source/lmcache", LMCACHE)
    problems = dependency_problems()
    if problems:
        raise ValueError(problems)
    target = args.output or root / "artifact-sets.json"
    cfg["source_revision"] = inputs["source_revision"]
    sets = {}
    for benchmark in cfg["benchmarks"]:
        native = EmbeddingStore(
            benchmark, "test", root=inputs["assets"]["data"]["path"]
        )
        folder = benchmark if benchmark in {"pmv2", "pv4"} else "stage3"
        checkpoint = Path(inputs["assets"]["reader"]["path"]) / folder
        store = ArtifactEmbeddingStore(
            native, cfg, checkpoint, root / "import" / benchmark, mode="import"
        )
        try:
            sets[benchmark] = store.set_id
        finally:
            store.close()
    receipt = {"schema": 1, "sets": sets}
    install_bytes(
        target, (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
    )
    print(
        json.dumps(
            {
                "manifest": str(target),
                "sets": sets,
                "doctor_seconds": verified - started,
                "import_seconds": time.monotonic() - verified,
                "whole_seconds": time.monotonic() - started,
                "training": False,
                "model_inference": False,
            }
        )
    )


if __name__ == "__main__":
    main()

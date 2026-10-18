# SPDX-License-Identifier: MIT
"""Run the original 8B cascade controls through the current pinned vLLM."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from verify import MODES, verify

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "kmesh_restore"))
from restore_common import sha256, source_identity, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "verify"))
    parser.add_argument("--config", type=Path, default=Path(".config"))
    args = parser.parse_args()
    cfg = {"OUTPUT": "results/kmesh-cascade", "MODEL_DIR": "", "VLLM_SOURCE": ""}
    if args.config.is_file():
        for line in args.config.read_text().splitlines():
            if line.startswith("CONFIG_KNLP_KMESH_CASCADE_"):
                key, value = line.split("=", 1)
                cfg[key.removeprefix("CONFIG_KNLP_KMESH_CASCADE_")] = json.loads(value)
    cfg = {key: os.environ.get("KMESH_CASCADE_" + key, val) for key, val in cfg.items()}
    profile = json.loads((HERE / "profile.json").read_text())
    if args.action == "plan":
        print(json.dumps(profile | {"modes": MODES, "settings": cfg}, indent=2))
        return
    output = Path(cfg["OUTPUT"]).resolve()
    if args.action == "verify":
        print(json.dumps(verify(output), indent=2))
        return
    if not cfg["MODEL_DIR"] or not cfg["VLLM_SOURCE"]:
        parser.error("set KMESH_CASCADE_MODEL_DIR and KMESH_CASCADE_VLLM_SOURCE")
    source = source_identity(Path(cfg["VLLM_SOURCE"]), profile["vllm_commit"])
    model = Path(cfg["MODEL_DIR"])
    for name, expected in profile["model_files_sha256"].items():
        if sha256(model / name) != expected:
            raise ValueError("model artifact mismatch: " + name)
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(Path(cfg["VLLM_SOURCE"]).resolve())
        + os.pathsep
        + env.get("PYTHONPATH", ""),
        VLLM_ENABLE_V1_MULTIPROCESSING="0",
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_BATCH_INVARIANT="0",
        VLLM_LOGGING_LEVEL="WARNING",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        PYTHONHASHSEED="0",
        VLLM_NO_USAGE_STATS="1",
        DO_NOT_TRACK="1",
    )
    output.mkdir(parents=True, exist_ok=False)
    config = {"model": str(model.resolve()), "output": str(output)}
    config["vllm_source"] = str(Path(cfg["VLLM_SOURCE"]).resolve())
    write_json(output / "config.json", config)
    write_json(
        output / "provenance.json",
        {
            "source": source,
            "profile": profile,
            "scripts_sha256": {p.name: sha256(p) for p in HERE.glob("*.py")},
            "python": sys.version,
            "packages": {
                name: importlib.metadata.version(name) for name in ("torch", "vllm")
            },
        },
    )
    for mode in MODES:
        with (output / (mode + ".log")).open("x") as stream:
            subprocess.run(
                [
                    sys.executable,
                    str(HERE / "cell.py"),
                    str(output / "config.json"),
                    mode,
                ],
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1800,
            )
        print("completed " + mode, flush=True)
    write_json(output / "verified.json", verify(output))


if __name__ == "__main__":
    main()

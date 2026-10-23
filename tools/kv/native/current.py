# SPDX-License-Identifier: MIT
"""Download and validate the pinned asymmetric Qwen3 serving recipe."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .october import VARIANTS, check_protocol, check_scales
from .october_setup import recipe_pins

FLAGS = {
    "VLLM_ASYM_SINGLE_WRITER": "1",
    "FLASHINFER_FUSE_V_SCALE": "1",
    "FLASHINFER_ASYM_FP8_PV": "0",
    "VLLM_COMPACT_PREFILL": "1",
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_KV_CACHE_LAYOUT": "NHD",
    "KNLP_NATIVE_OBSERVER": "off",
    "TOKENIZERS_PARALLELISM": "false",
}


def fixture_root():
    """Locate captured inputs supplied outside the source checkout."""
    value = os.environ.get("KNLP_KV_FIXTURES")
    if not value:
        raise ValueError(
            "set KNLP_KV_FIXTURES to the external experiment-input directory "
            "containing current/awq and current/fp16"
        )
    root = Path(value).expanduser().resolve()
    repository = Path(__file__).resolve().parents[3]
    if root.is_relative_to(repository):
        raise ValueError("captured experiment inputs must be outside the checkout")
    current = root / "current"
    if not current.is_dir():
        raise ValueError("KNLP_KV_FIXTURES must contain a current directory")
    return current


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def environment(base):
    env = dict(base)
    for key, value in FLAGS.items():
        if key in env and env[key] != value:
            raise ValueError("unsupported inherited setting: " + key)
        env[key] = value
    return env


def validate_install(record, protocol, env):
    expected = recipe_pins("current")
    if record.get("recipe") != "current" or record.get("sources") != {
        name: list(pin) for name, pin in expected.items()
    }:
        raise ValueError(
            "install --recipe current; historical compact is not this route"
        )
    environment(env)
    check_protocol(protocol, "mixed_prefill", env)


def prepare(install, output, weight_format):
    source = Path(__file__).resolve().parents[3]
    fixture = fixture_root() / weight_format
    protocol = json.loads((fixture / "measured-protocol.json").read_text())
    env = environment(os.environ)
    validate_install(json.loads((install / "setup.json").read_text()), protocol, env)
    VARIANTS["mixed_prefill"].verify_sources(
        {name: install / name for name in ("vllm", "flashinfer")}
    )
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--", "tools/kv"], text=True
    ).strip():
        raise ValueError("use a clean committed tools/kv checkout")
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    files = json.loads((fixture / "manifest.json").read_text())
    for name, item in files.items():
        archive = fixture / item["file"]
        if sha(archive) != item["archive_sha256"]:
            raise ValueError("fixture archive changed: " + name)
        raw = archive.read_bytes()
        (inputs / name).write_bytes(
            raw if name.endswith(".gz") else gzip.decompress(raw)
        )
        if sha(inputs / name) != item["sha256"]:
            raise ValueError("fixture changed: " + name)
    check_scales(json.loads((inputs / "scales.json").read_text()), protocol["geometry"])
    protocol["measured_harness_sha"] = protocol["source_sha"]
    protocol["source_sha"] = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    protocol["runner_files_sha256"] = {
        p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    protocol["inputs_sha256"] = {name: sha(inputs / name) for name in files}
    save(inputs / "protocol.json", protocol)
    for name in ("vllm", "flashinfer"):
        (inputs / name).symlink_to(install / name)
    save(output / "environment.json", {key: env[key] for key in FLAGS})
    return inputs, protocol, env


def download(weight_format, output):
    from huggingface_hub import snapshot_download

    raw = (fixture_root() / weight_format / "qwen.json.gz").read_bytes()
    manifest = json.loads(gzip.decompress(raw))
    started = time.perf_counter()
    path = snapshot_download(
        manifest["model"],
        revision=manifest["model_revision"],
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
    )
    save(
        output,
        {
            "model": manifest["model"],
            "revision": manifest["model_revision"],
            "snapshot": path,
            "seconds": time.perf_counter() - started,
        },
    )


def smoke(args):
    install, output = args.install.resolve(), args.output.resolve()
    inputs, protocol, env = prepare(install, output, args.format)
    budget = str(protocol["engine"]["kv_cache_memory_bytes"])
    events = []
    stages = [
        (
            "geometry-and-first-JIT",
            ["tools.kv.native.october", "geometry", "--variant", "mixed_prefill"],
        ),
        (
            "serving-validation",
            ["tools.kv.native.current_smoke", "--format", args.format],
        ),
    ]
    for name, arguments in stages:
        destination = output / name
        argv = [
            sys.executable,
            "-m",
            *arguments,
            "--inputs",
            str(inputs),
            "--output",
            str(destination),
            "--cache-bytes",
            budget,
        ]
        started = time.perf_counter()
        result = subprocess.run(argv, env=env)
        events.append(
            {
                "stage": name,
                "argv": argv,
                "seconds": time.perf_counter() - started,
                "returncode": result.returncode,
            }
        )
        save(output / "stages.json", events)
        result.check_returncode()
        if name == "geometry-and-first-JIT":
            shutil.copy2(destination / "geometry.json", inputs / "geometry.json")
    save(
        output / "receipt.json",
        {
            "passed": True,
            "format": args.format,
            "source": protocol["source_sha"],
            "measured_harness": protocol["measured_harness_sha"],
            "backend_sources": protocol["backend_sources"],
            "stages": events,
            "scope": "installation and route validation; not a speed or quality study",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "smoke"):
        p = sub.add_parser(name)
        p.add_argument("--format", choices=("awq", "fp16"), default="awq")
        p.add_argument("--output", type=Path, required=True)
        if name == "smoke":
            p.add_argument("--install", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "download":
        download(args.format, args.output)
    else:
        smoke(args)


if __name__ == "__main__":
    main()

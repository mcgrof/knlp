#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Provision and run the LMCache SDK SnapKV reproduction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def parse_config(path: str) -> dict[str, str]:
    config: dict[str, str] = {}
    with open(path, encoding="utf-8") as stream:
        for raw in stream:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            config[key] = value.strip().strip('"')
    return config


def setting(config: dict[str, str], name: str, default: str) -> str:
    return config.get(f"CONFIG_KNLP_SNAPKV_{name}", default)


def paths(config: dict[str, str]) -> tuple[Path, Path, Path]:
    work = Path(setting(config, "WORK_ROOT", "./build/lmcache-snapkv"))
    results = Path(setting(config, "RESULTS_ROOT", "./results/lmcache-snapkv"))
    return work, work / "LMCache", results


def command(argv: list[str], **kwargs) -> None:
    print("  $", " ".join(argv), flush=True)
    subprocess.run(argv, check=True, **kwargs)


def cmd_doctor(config: dict[str, str]) -> None:
    print("=== LMCache SnapKV doctor ===")
    missing = []
    for executable in ("git", "patch", "nvidia-smi", "nvcc"):
        location = shutil.which(executable)
        print(f"  {executable}: {location or 'MISSING'}")
        if not location:
            missing.append(executable)
    python_name = setting(config, "PYTHON", "python3.12")
    python_path = shutil.which(python_name)
    uv_path = shutil.which("uv")
    print(f"  {python_name}: {python_path or 'uv will provision it'}")
    if not python_path and not uv_path:
        missing.append(python_name)
    if Path("/dev/shm").exists():
        usage = shutil.disk_usage("/dev/shm")
        print(f"  /dev/shm available: {usage.free / 2**30:.2f} GiB")
    else:
        missing.append("/dev/shm")
    if missing:
        raise SystemExit("missing prerequisites: " + ", ".join(missing))


def cmd_fetch(config: dict[str, str]) -> None:
    work, lmcache, _ = paths(config)
    repo = setting(config, "LMCACHE_REPO", "https://github.com/LMCache/LMCache.git")
    ref = setting(config, "LMCACHE_REF", "dev")
    work.mkdir(parents=True, exist_ok=True)
    if not (lmcache / ".git").exists():
        command(
            ["git", "clone", "--branch", ref, "--single-branch", repo, str(lmcache)]
        )
        return
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=lmcache, text=True
    ).strip()
    if dirty:
        raise SystemExit(f"refusing to update dirty LMCache tree: {lmcache}")
    command(["git", "fetch", "origin", ref], cwd=lmcache)
    command(["git", "checkout", ref], cwd=lmcache)
    command(["git", "merge", "--ff-only", f"origin/{ref}"], cwd=lmcache)


def _pip(python: Path, *args: str, cwd: Path | None = None) -> None:
    command([str(python), "-m", "pip", "install", *args], cwd=cwd)


def _apply_vllm_patch(python: Path, lmcache: Path) -> None:
    patch_file = (
        lmcache / "examples/token_dropping/vllm-export-intermediate-tensors.diff"
    )
    site = subprocess.check_output(
        [
            str(python),
            "-c",
            "import os,vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))",
        ],
        text=True,
    ).strip()
    with patch_file.open("rb") as diff:
        dry = subprocess.run(
            ["patch", "-p1", "--forward", "--dry-run"],
            cwd=site,
            stdin=diff,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    if dry.returncode == 0:
        with patch_file.open("rb") as diff:
            command(["patch", "-p1", "--forward"], cwd=site, stdin=diff)
        return
    with patch_file.open("rb") as diff:
        reverse = subprocess.run(
            ["patch", "-p1", "-R", "--dry-run"],
            cwd=site,
            stdin=diff,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    if reverse.returncode != 0:
        raise SystemExit("the LMCache query-tensor patch does not match this vLLM")
    print("  vLLM query-tensor patch already applied")


def cmd_build(config: dict[str, str]) -> None:
    work, lmcache, _ = paths(config)
    if not (lmcache / ".git").exists():
        raise SystemExit("run lmcache-snapkv-fetch first")
    venv = work / ".venv"
    if not venv.exists():
        python_name = setting(config, "PYTHON", "python3.12")
        uv_path = shutil.which("uv")
        python_path = shutil.which(python_name)
        if uv_path:
            command([uv_path, "venv", "--python", python_name, str(venv)])
        elif python_path:
            command([python_path, "-m", "venv", str(venv)])
        else:
            raise SystemExit(f"configured Python interpreter is missing: {python_name}")
    python = venv / "bin/python"
    _pip(python, "--upgrade", "pip", "wheel", "setuptools")
    version = setting(config, "VLLM_VERSION", "0.23.0")
    _pip(python, f"vllm=={version}")
    _apply_vllm_patch(python, lmcache)
    _pip(python, "--editable", ".", "--no-build-isolation", "--no-deps", cwd=lmcache)
    _pip(
        python,
        "datasets",
        "requests",
        "transformers",
        "opentelemetry-exporter-prometheus==0.65b0",
    )


def cmd_run(config: dict[str, str]) -> None:
    work, lmcache, results = paths(config)
    python = work / ".venv/bin/python"
    if not python.exists():
        raise SystemExit("run lmcache-snapkv-build first")
    results.mkdir(parents=True, exist_ok=True)
    argv = [
        str(python),
        str(Path(__file__).with_name("experiment.py")),
        "--lmcache-dir",
        str(lmcache.resolve()),
        "--output",
        str((results / "results.json").resolve()),
        "--model",
        setting(config, "MODEL", "Qwen/Qwen3-1.7B"),
        "--dataset",
        setting(config, "DATASET", "raniayu/token-dropping-demo-short"),
        "--num-prompts",
        setting(config, "NUM_PROMPTS", "10"),
        "--max-tokens",
        setting(config, "MAX_TOKENS", "256"),
        "--drop-ratio",
        setting(config, "DROP_RATIO", "0.5"),
        "--window-size",
        setting(config, "WINDOW_SIZE", "64"),
        "--gpu-memory-utilization",
        setting(config, "GPU_MEMORY_UTILIZATION", "0.35"),
        "--l1-size-gb",
        setting(config, "L1_SIZE_GB", "3"),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = setting(config, "CUDA_VISIBLE_DEVICES", "0")
    command(argv, env=env)


def cmd_report(config: dict[str, str]) -> None:
    _, _, results = paths(config)
    result_file = results / "results.json"
    if not result_file.exists():
        raise SystemExit(f"result is missing: {result_file}")
    data = json.loads(result_file.read_text(encoding="utf-8"))
    base = data["baseline"]
    snap = data["snapkv"]
    delta = 100 * (snap["output_tput"] / base["output_tput"] - 1)
    print("=== LMCache SnapKV result ===")
    print(
        f"  baseline: {base['correct']}/{base['total']} correct, "
        f"{base['output_tput']:.2f} token/s"
    )
    print(
        f"  SnapKV:   {snap['correct']}/{snap['total']} correct, "
        f"{snap['output_tput']:.2f} token/s"
    )
    print(f"  throughput change: {delta:+.2f}%")
    print(f"  artifact: {result_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("doctor", "fetch", "build", "run", "report"))
    parser.add_argument("--config", default=".config")
    args = parser.parse_args()
    config = parse_config(args.config)
    globals()[f"cmd_{args.action}"](config)


if __name__ == "__main__":
    main()

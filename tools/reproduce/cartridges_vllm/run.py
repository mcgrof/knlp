"""CartridgeConnector validation orchestrator.

Subcommands: doctor, fetch, build, test, report, clean.

Usage:
    python3 -m tools.reproduce.cartridges_vllm.run doctor --config .config
    python3 -m tools.reproduce.cartridges_vllm.run test --config .config
    python3 -m tools.reproduce.cartridges_vllm.run test --config .config --tier 6
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _parse_config(config_path: str) -> dict:
    """Parse key=value .config file into a dict."""
    cfg = {}
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                v = v.strip().strip('"')
                cfg[k] = v
    return cfg


def _run(cmd: list[str], cwd: str = None, check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=check)


# ─────────────────────────────────────────────────────────────
# doctor
# ─────────────────────────────────────────────────────────────
def cmd_doctor(cfg: dict):
    """Check prerequisites."""
    print("=== CartridgeConnector doctor ===")
    ok = True

    # GPU
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"], text=True)
        gpus = out.strip().split("\n")
        print(f"  GPUs: {len(gpus)}")
        for g in gpus:
            print(f"    {g.strip()}")
    except Exception:
        print("  WARNING: nvidia-smi not found — GPU tiers will be skipped")

    # Python
    print(f"  Python: {sys.version.split()[0]}")

    # HF token
    hf_token = Path.home() / ".cache" / "huggingface" / "token"
    if hf_token.exists():
        print(f"  HF token: present ({hf_token})")
    else:
        print("  WARNING: no HF token — gated models will fail")
        ok = False

    # uv
    if shutil.which("uv"):
        print("  uv: found")
    else:
        print("  WARNING: uv not found — will use pip")

    print(f"  Doctor: {'PASS' if ok else 'WARN'}")
    return ok


# ─────────────────────────────────────────────────────────────
# fetch
# ─────────────────────────────────────────────────────────────
def cmd_fetch(cfg: dict):
    """Clone or update the vLLM repo."""
    print("=== CartridgeConnector fetch ===")
    worktree = Path(cfg.get("CONFIG_KNLP_WORKTREE_ROOT", ".."))
    vllm_dir = worktree / cfg.get("CONFIG_KNLP_VLLM_DIR", "vllm")
    vllm_repo = cfg.get("CONFIG_KNLP_VLLM_REPO",
                        "https://github.com/mcgrof/vllm.git")
    vllm_ref = cfg.get("CONFIG_KNLP_VLLM_REF",
                       "20260429-cartridges-code-only")

    if (vllm_dir / ".git").exists():
        print(f"  vLLM repo exists at {vllm_dir}")
        # Fetch the branch if not already on it
        current = subprocess.check_output(
            ["git", "branch", "--show-current"],
            cwd=str(vllm_dir), text=True).strip()
        if current != vllm_ref:
            print(f"  Switching from {current} to {vllm_ref}")
            _run(["git", "fetch", "origin", vllm_ref], cwd=str(vllm_dir))
            _run(["git", "checkout", vllm_ref], cwd=str(vllm_dir))
        else:
            print(f"  Already on {vllm_ref}")
            _run(["git", "pull", "--ff-only"], cwd=str(vllm_dir),
                 check=False)
    else:
        print(f"  Cloning {vllm_repo} branch {vllm_ref}")
        _run(["git", "clone", "--branch", vllm_ref, "--depth", "50",
              vllm_repo, str(vllm_dir)])

    print(f"  vLLM at: {vllm_dir}")


# ─────────────────────────────────────────────────────────────
# build
# ─────────────────────────────────────────────────────────────
def cmd_build(cfg: dict):
    """Install vLLM into a venv."""
    print("=== CartridgeConnector build ===")
    worktree = Path(cfg.get("CONFIG_KNLP_WORKTREE_ROOT", ".."))
    vllm_dir = worktree / cfg.get("CONFIG_KNLP_VLLM_DIR", "vllm")
    venv = vllm_dir / ".venv"
    tfm_ver = cfg.get("CONFIG_KNLP_TRANSFORMERS_VERSION", "4.55.4")

    if not venv.exists():
        print("  Creating venv")
        _run(["uv", "venv", "--python", "3.12", str(venv)])

    activate = venv / "bin" / "activate"
    # Use system uv (not venv uv) to install into the venv.
    # uv pip install --python <path> puts packages into that venv.
    uv_bin = shutil.which("uv")
    venv_python = str((venv / "bin" / "python").resolve())
    if uv_bin:
        pip_cmd = [uv_bin, "pip", "install", "--python", venv_python]
    else:
        pip_cmd = [venv_python, "-m", "pip", "install"]

    # Check if vllm already installed
    check = subprocess.run(
        [str(venv / "bin" / "python"), "-c", "import vllm; print(vllm.__version__)"],
        capture_output=True, text=True)
    if check.returncode == 0:
        print(f"  vLLM already installed: {check.stdout.strip()}")
    else:
        print("  Installing vLLM (precompiled)")
        env = os.environ.copy()
        env["VLLM_USE_PRECOMPILED"] = "1"
        subprocess.run(
            [*pip_cmd, "-e", ".", "--torch-backend=auto"],
            cwd=str(vllm_dir), env=env, check=True)

    # Ensure transformers version
    subprocess.run(
        [*pip_cmd, f"transformers=={tfm_ver}",
         "requests", "pytest", "pytest-timeout", "tblib"],
        cwd=str(vllm_dir), check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Patch tokenizer if needed (fork-point compat)
    tokenizer_py = vllm_dir / "vllm" / "transformers_utils" / "tokenizer.py"
    content = tokenizer_py.read_text()
    if "all_special_tokens_extended)" in content and "getattr" not in content.split("all_special_tokens_extended")[0].split("\n")[-1]:
        print("  Patching tokenizer for fork-point compatibility")
        content = content.replace(
            "tokenizer_all_special_tokens_extended = (\n"
            "        tokenizer.all_special_tokens_extended)",
            'tokenizer_all_special_tokens_extended = getattr('
            'tokenizer, "all_special_tokens_extended", '
            'getattr(tokenizer, "all_special_tokens", []))'
        )
        tokenizer_py.write_text(content)

    ver = subprocess.check_output(
        [str(venv / "bin" / "python"), "-c",
         "import vllm; print(vllm.__version__)"],
        text=True).strip()
    print(f"  vLLM ready: {ver}")


# ─────────────────────────────────────────────────────────────
# test
# ─────────────────────────────────────────────────────────────
def cmd_test(cfg: dict, tier: str = None):
    """Run validation tiers."""
    print("=== CartridgeConnector test ===")
    worktree = Path(cfg.get("CONFIG_KNLP_WORKTREE_ROOT", ".."))
    vllm_dir = worktree / cfg.get("CONFIG_KNLP_VLLM_DIR", "vllm")
    results_dir = Path(cfg.get("CONFIG_KNLP_RESULTS_ROOT",
                               "./results/cartridges"))
    results_dir.mkdir(parents=True, exist_ok=True)
    python = str(vllm_dir / ".venv" / "bin" / "python")

    tiers_to_run = [tier] if tier else [
        "-1", "0", "0.5", "1", "2", "3", "4", "5", "6"]
    results = {}

    for t in tiers_to_run:
        if t == "-1":
            print("\n--- Tier -1: Unit tests ---")
            # Use -k to select cartridge tests (glob doesn't expand
            # without shell=True, and pytest -k is more robust).
            import glob as _glob
            test_files = sorted(_glob.glob(
                str(vllm_dir / "tests" / "v1" / "core" / "test_cartridge*.py")))
            r = subprocess.run(
                [python, "-m", "pytest", *test_files,
                 "-v", "--timeout=120", "--tb=short"],
                cwd=str(vllm_dir), capture_output=True, text=True)
            # Count pass/fail from pytest output
            last_line = r.stdout.strip().split("\n")[-1] if r.stdout else ""
            passed = r.returncode == 0
            print(f"  {last_line}")
            results["tier_-1"] = {"pass": passed, "output": last_line}

        elif t in ("0", "0.5", "1", "2", "3", "4", "5"):
            print(f"\n--- Tier {t}: GPU test ---")
            # These need the tier_test_run.py or equivalent
            # Check if it exists in the repo's tools or results dir
            test_script = vllm_dir / "tools" / "tier_test_run.py"
            if not test_script.exists():
                # Use the one from knlp reproduce dir
                test_script = (Path(__file__).parent / "tier_test_run.py")
            if test_script.exists():
                env = os.environ.copy()
                env["HF_HUB_DISABLE_XET"] = "1"
                r = subprocess.run(
                    [python, str(test_script)],
                    cwd=str(vllm_dir), env=env,
                    capture_output=True, text=True, timeout=600)
                print(f"  {r.stdout[-200:] if r.stdout else 'no output'}")
                results[f"tier_{t}"] = {
                    "pass": r.returncode == 0,
                    "output": r.stdout[-500:] if r.stdout else "",
                }
            else:
                print(f"  SKIP: no test script for tier {t}")
                results[f"tier_{t}"] = {"pass": None, "skipped": True}

        elif t == "6":
            print("\n--- Tier 6: TP=2 ---")
            # Check GPU count
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name",
                     "--format=csv,noheader"], text=True)
                n_gpus = len(out.strip().split("\n"))
            except Exception:
                n_gpus = 0
            if n_gpus < 2:
                print(f"  SKIP: need 2 GPUs, have {n_gpus}")
                results["tier_6"] = {"pass": None, "skipped": True,
                                      "reason": f"need 2 GPUs, have {n_gpus}"}
            else:
                print(f"  {n_gpus} GPUs available — TP=2 test would run here")
                results["tier_6"] = {"pass": None, "note": "manual test required"}

    # Save results
    results_file = results_dir / "test_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_file}")


# ─────────────────────────────────────────────────────────────
# report
# ─────────────────────────────────────────────────────────────
def cmd_report(cfg: dict):
    """Generate validation report."""
    print("=== CartridgeConnector report ===")
    results_dir = Path(cfg.get("CONFIG_KNLP_RESULTS_ROOT",
                               "./results/cartridges"))
    results_file = results_dir / "test_results.json"
    if results_file.exists():
        results = json.loads(results_file.read_text())
        print("\n  PASS/FAIL TABLE:")
        for k, v in sorted(results.items()):
            status = "PASS" if v.get("pass") else (
                "SKIP" if v.get("skipped") else "FAIL")
            print(f"    {k:20s} {status}")
    else:
        print("  No results yet — run `make cartridges-test` first")


# ─────────────────────────────────────────────────────────────
# clean
# ─────────────────────────────────────────────────────────────
def cmd_clean(cfg: dict):
    """Clean build artifacts (keeps repo)."""
    print("=== CartridgeConnector clean ===")
    results_dir = Path(cfg.get("CONFIG_KNLP_RESULTS_ROOT",
                               "./results/cartridges"))
    if results_dir.exists():
        shutil.rmtree(results_dir)
        print(f"  Removed {results_dir}")


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=[
        "doctor", "fetch", "build", "test", "report", "clean"])
    ap.add_argument("--config", default=".config")
    ap.add_argument("--tier", default=None)
    args = ap.parse_args()

    cfg = _parse_config(args.config)
    if cfg.get("CONFIG_KNLP_REPRODUCE_CARTRIDGES") != "y":
        print("CONFIG_KNLP_REPRODUCE_CARTRIDGES not enabled. "
              "Run `make defconfig DEFCONFIG=cartridges-vllm-tests` first.")
        sys.exit(2)

    if args.command == "doctor":
        cmd_doctor(cfg)
    elif args.command == "fetch":
        cmd_fetch(cfg)
    elif args.command == "build":
        cmd_build(cfg)
    elif args.command == "test":
        cmd_test(cfg, tier=args.tier)
    elif args.command == "report":
        cmd_report(cfg)
    elif args.command == "clean":
        cmd_clean(cfg)


if __name__ == "__main__":
    main()

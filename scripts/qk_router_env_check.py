#!/usr/bin/env python3
"""Phase 0: Environment and repo discovery for QK Router."""

import json
import os
import subprocess
import sys

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    repo_root = os.environ.get("REPO_ROOT", "/mnt/tmpfs/knlp")
    run_root = os.path.join(repo_root, "results", "qk_router_01")

    print("=" * 60)
    print("QK Router Phase 0: Environment Check")
    print("=" * 60)

    env = {}

    # Git
    try:
        env["git_commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        env["git_commit"] = "unknown"
    print(f"Git commit: {env['git_commit']}")

    # Python
    env["python_version"] = sys.version.split()[0]
    print(f"Python: {env['python_version']}")

    # PyTorch
    try:
        import torch

        env["torch_version"] = torch.__version__
        env["cuda_version"] = torch.version.cuda or "N/A"
        env["hip_version"] = torch.version.hip or "N/A"
        env["gpu_count"] = torch.cuda.device_count()
        if env["gpu_count"] > 0:
            props = torch.cuda.get_device_properties(0)
            env["gpu_name"] = props.name
            env["gpu_memory_gb"] = round(props.total_memory / 1e9, 1)
        else:
            env["gpu_name"] = "N/A"
            env["gpu_memory_gb"] = 0
    except ImportError:
        env["torch_version"] = "NOT INSTALLED"
        env["cuda_version"] = "N/A"
        env["gpu_count"] = 0
    print(f"PyTorch: {env['torch_version']}")
    print(f"CUDA: {env['cuda_version']}")
    print(f"GPU: {env.get('gpu_name', 'N/A')} ({env.get('gpu_memory_gb', 0)}GB)")

    # Flash attention
    try:
        import flash_attn

        env["flash_attn"] = flash_attn.__version__
    except ImportError:
        env["flash_attn"] = False
    print(f"Flash-attn: {env['flash_attn']}")

    # Transformers
    try:
        import transformers

        env["transformers_version"] = transformers.__version__
    except ImportError:
        env["transformers_version"] = "NOT INSTALLED"
    print(f"Transformers: {env['transformers_version']}")

    # Check required scripts
    print("\nRequired scripts:")
    scripts = [
        "scripts/benchmark_tiered_inference.py",
        "scripts/benchmark_sequential_io.py",
        "scripts/eval_long_context_needle.py",
        "scripts/bench_generate.py",
        "scripts/benchmark_kvsplice_inference_simple.py",
        "scripts/compare_inference.py",
        "gpt2/test_kv_cache_size.py",
        "gpt2/model.py",
        "gpt2/model_knlp.py",
        "gpt2/mla.py",
        "gpt2/kvsplice.py",
    ]
    env["existing_scripts"] = {}
    for s in scripts:
        exists = os.path.exists(os.path.join(repo_root, s))
        env["existing_scripts"][s] = exists
        print(f"  {'OK' if exists else 'MISSING'}: {s}")

    # Check write access
    print("\nWrite access:")
    paths = {
        "run_root": run_root,
        "tmpfs_store": "/mnt/tmpfs/qk_router_01",
        "sfs_store": "/mnt/SFS-hugging/hub/qk_router_01",
    }
    env["write_access"] = {}
    for name, path in paths.items():
        try:
            os.makedirs(path, exist_ok=True)
            test_file = os.path.join(path, ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            env["write_access"][name] = True
            print(f"  OK: {path}")
        except Exception as e:
            env["write_access"][name] = False
            print(f"  FAIL: {path} ({e})")

    # Save environment info
    os.makedirs(run_root, exist_ok=True)
    env_path = os.path.join(run_root, "environment.json")
    with open(env_path, "w") as f:
        json.dump(env, f, indent=2)
    print(f"\nEnvironment saved to {env_path}")

    # Update manifest
    manifest_path = os.path.join(run_root, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["git_commit"] = env["git_commit"]
        manifest["hardware"]["flash_attn"] = env["flash_attn"]
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print("Manifest updated")

    print("\nPhase 0 complete.")
    return env


if __name__ == "__main__":
    main()

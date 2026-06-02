#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Ensure GNN training dependencies and the C++ page samplers are ready.

This is what makes ``make defconfig-gnn-dgraphfin && make`` work from a
bare checkout. It is importable (run_gnn.py calls ``ensure_all()``) and
runnable directly:

    python gnn/scripts/setup_gnn_deps.py            # install + build
    python gnn/scripts/setup_gnn_deps.py --check    # report only

PyTorch is intentionally NOT auto-installed: the correct wheel is
hardware-specific (ROCm vs CUDA, see requirements.txt), so we only
verify it is importable and tell the user how to fix it if not.
"""
import argparse
import importlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GNN_DIR = os.path.dirname(HERE)
CPP_DIR = os.path.join(GNN_DIR, "cpp_extension")

# import name -> pip name. Safe to add on demand. torch is excluded on
# purpose (see module docstring).
PIP_DEPS = {
    "torch_geometric": "torch_geometric",
    "pymetis": "pymetis",
    "huggingface_hub": "huggingface_hub",
}

# C++ extensions built in place by cpp_extension/setup.py.
CPP_MODULES = [
    "cpp_page_batch_sampler",
    "page_aware_sampler_cpp",
    "amex_sampler_cpp",
]


def _importable(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


def ensure_torch() -> bool:
    """Verify torch is importable; never auto-install it."""
    if _importable("torch"):
        return True
    print(
        "ERROR: PyTorch is not importable. Install the wheel for your\n"
        "       accelerator first (ROCm or CUDA; see requirements.txt).\n"
        "       Refusing to auto-install torch because the right build is\n"
        "       hardware-specific.",
        file=sys.stderr,
    )
    return False


def ensure_pip_deps(install: bool = True) -> bool:
    """Make the pip-installable deps importable."""
    missing = [pip for mod, pip in PIP_DEPS.items() if not _importable(mod)]
    if not missing:
        return True
    if not install:
        print(f"Missing Python deps (auto-deps disabled): {', '.join(missing)}")
        return False
    print(f"Installing Python deps: {', '.join(missing)}")
    rc = subprocess.call([sys.executable, "-m", "pip", "install", *missing])
    return rc == 0 and all(_importable(m) for m in PIP_DEPS)


def ext_built() -> bool:
    """True if the C++ samplers import from cpp_extension/."""
    sys.path.insert(0, CPP_DIR)
    try:
        return all(_importable(m) for m in CPP_MODULES)
    finally:
        sys.path.pop(0)


def build_ext(install: bool = True) -> bool:
    """Build the in-place C++ samplers if not already importable."""
    if ext_built():
        return True
    if not install:
        print("C++ page sampler not built (auto-deps disabled).")
        return False
    print(f"Building C++ page samplers in {CPP_DIR} ...")
    rc = subprocess.call(
        [sys.executable, "setup.py", "build_ext", "--inplace"], cwd=CPP_DIR
    )
    return rc == 0 and ext_built()


def ensure_all(install: bool = True) -> bool:
    """Verify torch, install pip deps, build the C++ ext. Returns readiness."""
    have_torch = ensure_torch()
    deps_ok = ensure_pip_deps(install)
    # The extension build needs torch headers, so skip it without torch.
    ext_ok = build_ext(install) if have_torch else False
    return have_torch and deps_ok and ext_ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare GNN deps and C++ ext")
    ap.add_argument(
        "--check",
        action="store_true",
        help="report only; do not install or build",
    )
    args = ap.parse_args()
    ok = ensure_all(install=not args.check)
    print("GNN dependencies:", "READY" if ok else "INCOMPLETE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

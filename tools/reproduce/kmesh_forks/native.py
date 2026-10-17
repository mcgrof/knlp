#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Load the pinned vLLM source over its matching compiled wheel on CPU."""

import importlib.util
import os
from pathlib import Path
import runpy
import sys


def main():
    source = Path(os.environ["KMESH_FORK_SOURCE"])
    wheel = Path(
        os.environ.get(
            "KMESH_FORK_WHEEL_PACKAGE",
            "/opt/kmesh-target/lib/python3.12/site-packages/vllm",
        )
    )
    spec = importlib.util.spec_from_file_location(
        "vllm",
        source / "vllm/__init__.py",
        submodule_search_locations=[str(source / "vllm"), str(wheel)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["vllm"] = module
    spec.loader.exec_module(module)
    import torch
    from vllm.v1.core import kv_cache_manager

    profile_name = sys.argv[2] if len(sys.argv) > 2 else "historical"
    from verify import load_profile

    profile = load_profile(profile_name)
    actual = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "vllm": module.__version__,
    }
    if actual != profile["runtime_versions"]:
        raise ValueError(f"runtime mismatch: {actual}")
    if Path(kv_cache_manager.__file__).resolve() != (
        source / "vllm/v1/core/kv_cache_manager.py"
    ):
        raise ValueError("allocator did not load from the pinned source")
    if torch.cuda.is_available():
        raise ValueError("CPU experiment requires GPUs hidden")
    mode = sys.argv[1]
    if mode == "run":
        script = source / "benchmarks/kmesh/fork_block_accounting.py"
        sys.argv = [
            str(script),
            "--output",
            os.environ.get("KMESH_FORK_RESULT", "/outputs/results.json"),
        ]
        runpy.run_path(str(script), run_name="__main__")
    elif mode == "contracts":
        import pytest

        sys.path.insert(0, str(source))
        raise SystemExit(
            pytest.main(
                [
                    str(source / "tests/v1/core/test_prefix_caching.py"),
                    "--confcutdir=" + str(source / "tests/v1/core"),
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "--junitxml="
                    + str(
                        Path(os.environ.get("KMESH_FORK_RESULTS_DIR", "/outputs"))
                        / "contracts.xml"
                    ),
                ]
            )
        )
    else:
        raise ValueError(f"unknown native action: {mode}")


if __name__ == "__main__":
    main()

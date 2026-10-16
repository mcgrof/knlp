#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Load the pinned Python source over the matching compiled vLLM wheel."""

import importlib.util
import os
from pathlib import Path
import runpy
import sys


def main():
    source = Path(os.environ["TOKENDANCE_VLLM_SOURCE"])
    wheel = Path(os.environ["TOKENDANCE_WHEEL_PACKAGE"])
    spec = importlib.util.spec_from_file_location(
        "vllm",
        source / "vllm/__init__.py",
        submodule_search_locations=[str(source / "vllm"), str(wheel)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["vllm"] = module
    spec.loader.exec_module(module)
    script = Path(__file__).with_name(sys.argv[1])
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()

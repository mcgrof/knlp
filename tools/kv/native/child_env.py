# SPDX-License-Identifier: MIT
"""Build the same explicit environment for Python and its build-tool children."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def child_environment(venv, *, cuda="/usr/local/cuda", base=None):
    env = dict(os.environ if base is None else base)
    env["PATH"] = os.pathsep.join(
        [str(Path(venv) / "bin"), str(Path(cuda) / "bin"), env.get("PATH", "")]
    )
    env["VIRTUAL_ENV"] = str(venv)
    env["CUDA_HOME"] = str(cuda)
    return env


def build_tool_receipt(env):
    """Execute resolved tools through precisely the environment used for JIT."""
    tools = {}
    for name in ("ninja", "nvcc"):
        path = shutil.which(name, path=env["PATH"])
        if path is None:
            raise FileNotFoundError(name)
        tools[name] = {
            "path": path,
            "version": subprocess.check_output(
                [path, "--version"], env=env, text=True, stderr=subprocess.STDOUT
            ).strip(),
        }
    return {"python": sys.executable, "tools": tools, "path": env["PATH"]}

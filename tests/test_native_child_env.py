# SPDX-License-Identifier: MIT
import os
import subprocess
import sys

from tools.kv.native.child_env import child_environment


def test_child_finds_venv_tool_with_unactivated_parent(tmp_path):
    bindir = tmp_path / "env/bin"
    bindir.mkdir(parents=True)
    tool = bindir / "isolated-ninja-probe"
    tool.write_text("#!/bin/sh\nprintf 'venv-tool-executed\\n'\n")
    tool.chmod(0o755)
    parent = {**os.environ, "PATH": "/usr/bin:/bin"}
    env = child_environment(tmp_path / "env", cuda=tmp_path / "cuda", base=parent)
    result = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import subprocess; subprocess.run(['isolated-ninja-probe'], check=True)",
        ],
        env=env,
        text=True,
    )
    assert result == "venv-tool-executed\n"
    assert parent["PATH"] == "/usr/bin:/bin"

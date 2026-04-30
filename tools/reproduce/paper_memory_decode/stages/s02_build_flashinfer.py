"""Stage 02: build FlashInfer (editable install).

Installs the mcgrof/flashinfer asym-prefill-refactor-stage branch as
an editable package.  This must happen BEFORE vLLM is built because
the vLLM pip install pulls flashinfer from PyPI and clobbers the
editable install; stage 03_build_vllm re-installs flashinfer editable
afterwards.

Requires:
  - pip / pip3 on PATH
  - cmake ≥ 4 (checked by doctor; pip install --upgrade cmake if absent)
  - CUDA toolkit (nvcc)
  - The FlashInfer repo already checked out with submodules (stage 01)
"""

from __future__ import annotations

import sys
import shutil
from pathlib import Path

from ..stages import StageContext, StageResult


def run(ctx: StageContext) -> StageResult:
    cfg = ctx.cfg

    fi_path = Path(cfg.worktree_root).resolve() / cfg.raw.get(
        "CONFIG_KNLP_FLASHINFER_DIR", "flashinfer"
    )

    if not fi_path.is_dir():
        return StageResult(
            name=ctx.name,
            status="failed",
            reason=f"FlashInfer source dir not found: {fi_path}; "
            "run stage 01_fetch_repos first",
        )

    pip = shutil.which("pip3") or shutil.which("pip")
    if not pip:
        return StageResult(
            name=ctx.name, status="failed", reason="pip/pip3 not found in PATH"
        )

    # Make sure cutlass submodule is initialised (may have been skipped
    # in s01 if network was flaky).
    ctx.run_subprocess(
        ["git", "submodule", "update", "--init", "--recursive"],
        cwd=str(fi_path),
        timeout=300,
    )

    rc = ctx.run_subprocess(
        [pip, "install", "--no-build-isolation", "-e", "."],
        cwd=str(fi_path),
        extra_env={"FLASHINFER_DISABLE_VERSION_CHECK": "1"},
        timeout=3600,  # CUDA compile can take 45-60 min cold
    )
    if rc != 0:
        return StageResult(
            name=ctx.name,
            status="failed",
            reason=f"flashinfer editable install failed (rc={rc})",
        )

    # Verify import.
    import subprocess

    try:
        result = subprocess.run(
            [sys.executable, "-c", "import flashinfer; print(flashinfer.__version__)"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        version = result.stdout.strip() or "unknown"
    except Exception:
        version = "unknown"

    ctx.log_metric("flashinfer_version", version)
    ctx.mark_done({"flashinfer_version": version, "path": str(fi_path)})
    return StageResult(name=ctx.name, status="passed")

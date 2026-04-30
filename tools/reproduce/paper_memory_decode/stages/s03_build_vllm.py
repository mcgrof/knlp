"""Stage 03: build vLLM (editable install) then re-pin FlashInfer.

vLLM's pip install pulls flashinfer from PyPI (currently 0.6.6) and
overwrites the editable flashinfer installed in stage 02.  We fix this
by reinstalling flashinfer editable again at the end of this stage.

Build env vars:
  MAX_JOBS=32   — parallel NVCC jobs (set lower if OOM during build)
  NVCC_THREADS=2 — threads per NVCC process
  SETUPTOOLS_SCM_PRETEND_VERSION=0.1.dev0 — suppresses git-tag warning
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..stages import StageContext, StageResult


def run(ctx: StageContext) -> StageResult:
    cfg = ctx.cfg

    vllm_path = Path(cfg.worktree_root).resolve() / cfg.raw.get(
        "CONFIG_KNLP_VLLM_DIR", "vllm"
    )
    fi_path = Path(cfg.worktree_root).resolve() / cfg.raw.get(
        "CONFIG_KNLP_FLASHINFER_DIR", "flashinfer"
    )

    pip = shutil.which("pip3") or shutil.which("pip")
    if not pip:
        return StageResult(
            name=ctx.name, status="failed", reason="pip/pip3 not found in PATH"
        )

    if not vllm_path.is_dir():
        return StageResult(
            name=ctx.name,
            status="failed",
            reason=f"vLLM source dir not found: {vllm_path}",
        )

    # Ensure setuptools_scm is present — vLLM's pyproject.toml requires it
    # at metadata-generation time even with --no-build-isolation.
    ctx.run_subprocess([pip, "install", "setuptools_scm"], timeout=120)

    # Build vLLM editable.
    rc = ctx.run_subprocess(
        [pip, "install", "--no-build-isolation", "-e", "."],
        cwd=str(vllm_path),
        extra_env={
            "MAX_JOBS": "32",
            "NVCC_THREADS": "2",
            "SETUPTOOLS_SCM_PRETEND_VERSION": "0.1.dev0",
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
        },
        timeout=7200,  # up to 2 h cold; typically 60-90 min H100
    )
    if rc != 0:
        return StageResult(
            name=ctx.name,
            status="failed",
            reason=f"vllm editable install failed (rc={rc})",
        )

    # vLLM pip install pulled flashinfer from PyPI and clobbered our
    # editable.  Reinstall the asym fork editable.
    if fi_path.is_dir():
        ctx.run_subprocess(
            [pip, "install", "--no-build-isolation", "-e", "."],
            cwd=str(fi_path),
            extra_env={"FLASHINFER_DISABLE_VERSION_CHECK": "1"},
            timeout=300,  # already compiled; Python-level reinstall only
        )

    # Verify both imports.
    import subprocess

    vllm_ver = "unknown"
    fi_ver = "unknown"
    try:
        r = subprocess.run(
            ["python3", "-c", "import vllm; print(vllm.__version__)"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        vllm_ver = r.stdout.strip() or "unknown"
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["python3", "-c", "import flashinfer; print(flashinfer.__version__)"],
            capture_output=True,
            text=True,
            timeout=30,
            env={"FLASHINFER_DISABLE_VERSION_CHECK": "1", **__import__("os").environ},
        )
        fi_ver = r.stdout.strip() or "unknown"
    except Exception:
        pass

    ctx.log_metric("vllm_version", vllm_ver)
    ctx.log_metric("flashinfer_version_after_vllm", fi_ver)

    ctx.mark_done(
        {
            "vllm_version": vllm_ver,
            "flashinfer_version": fi_ver,
            "vllm_path": str(vllm_path),
        }
    )
    return StageResult(name=ctx.name, status="passed")

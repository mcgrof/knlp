import sys
"""Stage 04: build LMCache (editable install).

Installs the mcgrof/LMCache asymmetric-kv-codec branch as an editable
package and runs its unit test suite (74 CPU tests) to verify the
AsymK16V8Codec and split-tier storage tier are intact.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..stages import StageContext, StageResult


def run(ctx: StageContext) -> StageResult:
    cfg = ctx.cfg

    lmc_path = Path(cfg.worktree_root).resolve() / cfg.raw.get(
        "CONFIG_KNLP_LMCACHE_DIR", "lmcache"
    )

    pip = shutil.which("pip3") or shutil.which("pip")
    if not pip:
        return StageResult(
            name=ctx.name, status="failed", reason="pip/pip3 not found in PATH"
        )

    if not lmc_path.is_dir():
        return StageResult(
            name=ctx.name,
            status="failed",
            reason=f"LMCache source dir not found: {lmc_path}",
        )

    # Install editable.
    rc = ctx.run_subprocess(
        [pip, "install", "-e", ".[dev]"],
        cwd=str(lmc_path),
        timeout=600,
    )
    if rc != 0:
        # Try without [dev] extras in case the extras spec differs.
        rc = ctx.run_subprocess(
            [pip, "install", "-e", "."],
            cwd=str(lmc_path),
            timeout=600,
        )
        if rc != 0:
            return StageResult(
                name=ctx.name,
                status="failed",
                reason=f"lmcache editable install failed (rc={rc})",
            )

    # Run the unit tests (CPU-only; no GPU needed).
    pytest = shutil.which("pytest")
    if pytest:
        rc = ctx.run_subprocess(
            [pytest, "-x", "-q", "tests/"],
            cwd=str(lmc_path),
            timeout=300,
        )
        if rc != 0:
            return StageResult(
                name=ctx.name,
                status="failed",
                reason=f"lmcache unit tests failed (rc={rc}); "
                "see stages/04_build_lmcache/stdout.log",
            )
        ctx.log_metric("unit_tests_passed", 1)
    else:
        ctx.stderr_path.open("a").write(
            "WARN: pytest not found; skipping unit test run\n"
        )
        ctx.log_metric("unit_tests_passed", 0)

    import subprocess

    lmc_ver = "unknown"
    try:
        r = subprocess.run(
            [
                sys.executable,
                "-c",
                "import lmcache; print(getattr(lmcache, '__version__', 'dev'))",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        lmc_ver = r.stdout.strip() or "unknown"
    except Exception:
        pass

    ctx.log_metric("lmcache_version", lmc_ver)
    ctx.mark_done({"lmcache_version": lmc_ver, "path": str(lmc_path)})
    return StageResult(name=ctx.name, status="passed")

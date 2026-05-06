"""Stage 04: build LMCache (editable install).

Installs the mcgrof/LMCache asymmetric-kv-codec branch as an editable
package and runs its unit test suite (74 CPU tests) to verify the
AsymK16V8Codec and split-tier storage tier are intact.
"""

from __future__ import annotations

import sys
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

    # Run the codec / serde unit tests (CPU-only; no GPU needed).
    # The exact set of tests that exist depends on which lmcache
    # branch is checked out:
    #
    #   asymmetric-kv-codec branch    has tests/v1/kv_codec/ (74 CPU tests)
    #   serde-multi-output-extensions has tests/v1/distributed/serde/
    #                                 multi + asym multi + fp8 + utils tests
    #   upstream/dev                  has tests/v1/distributed/serde/
    #                                 fp8 + factory + utils tests
    #
    # We do NOT run pytest tests/ wholesale: that pulls in CLI / xpu /
    # connector benchmark suites that require optional deps and that
    # can fail with collection errors unrelated to codec correctness.
    # Instead, run the union of {kv_codec, distributed/serde} that
    # actually exists, and require at least one to pass.
    pytest = shutil.which("pytest")
    candidate_dirs = [
        "tests/v1/kv_codec",
        "tests/v1/distributed/serde",
    ]
    present_dirs = [d for d in candidate_dirs if (lmc_path / d).is_dir()]
    if pytest and present_dirs:
        rc = ctx.run_subprocess(
            [pytest, "-x", "-q", "--no-header", *present_dirs],
            cwd=str(lmc_path),
            timeout=300,
        )
        if rc != 0:
            return StageResult(
                name=ctx.name,
                status="failed",
                reason=(
                    f"lmcache unit tests failed (rc={rc}) in "
                    f"{', '.join(present_dirs)}; "
                    "see stages/04_build_lmcache/stdout.log"
                ),
            )
        ctx.log_metric("unit_tests_passed", 1)
        ctx.log_metric("test_dirs", ",".join(present_dirs))
    elif pytest and not present_dirs:
        ctx.stderr_path.open("a").write(
            "WARN: neither tests/v1/kv_codec nor tests/v1/distributed/serde "
            "present on this branch; skipping unit-test gate\n"
        )
        ctx.log_metric("unit_tests_passed", 0)
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

    # Install Python packages needed by the gate scripts in later stages.
    # These are not pulled in by vLLM or lmcache directly.
    ctx.run_subprocess(
        [
            pip,
            "install",
            "datasets",
            "accelerate",
        ],
        timeout=300,
    )

    ctx.mark_done({"lmcache_version": lmc_ver, "path": str(lmc_path)})
    return StageResult(name=ctx.name, status="passed")

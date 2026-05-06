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

    # Match the build stage: install via ``sys.executable -m pip`` so
    # the editable lmcache lands in the same interpreter the
    # orchestrator runs in.  Avoids the ``python3 != pip3 python``
    # split observed on RunPod (system python3=3.10, pip3=python3.12).
    pip_cmd = [sys.executable, "-m", "pip"]

    if not lmc_path.is_dir():
        return StageResult(
            name=ctx.name,
            status="failed",
            reason=f"LMCache source dir not found: {lmc_path}",
        )

    # Install editable.
    rc = ctx.run_subprocess(
        pip_cmd + ["install", "-e", ".[dev]"],
        cwd=str(lmc_path),
        timeout=600,
    )
    if rc != 0:
        # Try without [dev] extras in case the extras spec differs.
        rc = ctx.run_subprocess(
            pip_cmd + ["install", "-e", "."],
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
    # The e2e and fs_e2e suites need a running lmcache MP server with
    # the C extension built (or the python_only fallback configured a
    # specific way) and the multiprocess l1 manager wired up.  They
    # are environment-sensitive and pass-or-fail differently across
    # pods depending on CUDA toolkit availability and pinned-memory
    # budget.  The codec / serde *unit* tests (multi.py, asym
    # multi.py, fp8.py, utils.py, factory.py, async_processor.py)
    # are pure-Python correctness checks that do not depend on any
    # of that, and they are what this stage should gate on.
    _IGNORE_PATHS = [
        "tests/v1/distributed/serde/test_serde_e2e.py",
        "tests/v1/distributed/serde/test_serde_fs_e2e.py",
    ]
    pytest = shutil.which("pytest")
    candidate_dirs = [
        "tests/v1/kv_codec",
        "tests/v1/distributed/serde",
    ]
    present_dirs = [d for d in candidate_dirs if (lmc_path / d).is_dir()]
    if pytest and present_dirs:
        ignore_args: list[str] = []
        for p in _IGNORE_PATHS:
            if (lmc_path / p).exists():
                ignore_args += [f"--ignore={p}"]
        rc = ctx.run_subprocess(
            [pytest, "-x", "-q", "--no-header", *ignore_args, *present_dirs],
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
        pip_cmd + [
            "install",
            "datasets",
            "accelerate",
        ],
        timeout=300,
    )

    ctx.mark_done({"lmcache_version": lmc_ver, "path": str(lmc_path)})
    return StageResult(name=ctx.name, status="passed")

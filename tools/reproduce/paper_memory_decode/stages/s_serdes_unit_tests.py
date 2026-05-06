"""Stage: pytest the multi-output serde tests on the
serde-multi-output-extensions branch of LMCache.

Runs the four CPU-only test files that cover the multi-output ABC
contract and the AsymK16V8 concrete consumer:

  tests/v1/distributed/serde/test_multi.py
      MultiSerializer / MultiDeserializer ABCs, MemoryObjGroup
      semantics, single-to-multi adapter equivalence, validate_group_size.

  tests/v1/distributed/serde/test_asym_k16_v8_multi.py
      AsymK16V8MultiSerializer / AsymK16V8MultiDeserializer
      storage-only-dequant round-trip on Llama-3.1-8B KV shape:
      K bit-exact, V within FP8 noise, V-only-read skip semantics.

  tests/v1/distributed/serde/test_fp8.py
      Existing single-tensor fp8 serde — verifies that the
      multi-output extension did not regress the single-tensor path.

  tests/v1/distributed/serde/test_utils.py
      make_temp_key invariants used by SerdeL2AdapterWrapper temps.

Stage passes when all four test files exit 0.  Reads exit code from
pytest -q and writes summary metrics.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..stages import StageContext, StageResult


_TEST_FILES = [
    "tests/v1/distributed/serde/test_multi.py",
    "tests/v1/distributed/serde/test_asym_k16_v8_multi.py",
    "tests/v1/distributed/serde/test_fp8.py",
    "tests/v1/distributed/serde/test_utils.py",
]


def run(ctx: StageContext) -> StageResult:
    cfg = ctx.cfg
    lmc_path = Path(cfg.worktree_root).resolve() / cfg.raw.get(
        "CONFIG_KNLP_LMCACHE_DIR", "lmcache"
    )

    if not lmc_path.is_dir():
        return StageResult(
            name=ctx.name,
            status="failed",
            reason=f"lmcache dir not found at {lmc_path}; "
            "run stage 01_fetch_repos first",
        )

    pytest = shutil.which("pytest")
    if not pytest:
        return StageResult(
            name=ctx.name,
            status="failed",
            reason="pytest not found in PATH; lmcache install incomplete",
        )

    # Verify the test files exist on the checked-out branch (they
    # only live on serde-multi-output-extensions; if we are on a
    # different branch this is the actionable error).
    missing = [t for t in _TEST_FILES if not (lmc_path / t).exists()]
    if missing:
        return StageResult(
            name=ctx.name,
            status="failed",
            reason=(
                "missing test files (wrong lmcache branch?): "
                + ", ".join(missing)
                + "; expected branch serde-multi-output-extensions"
            ),
        )

    rc = ctx.run_subprocess(
        [pytest, "-q", "--tb=line", *_TEST_FILES],
        cwd=str(lmc_path),
        timeout=300,
    )
    ctx.log_metric("pytest_rc", rc)
    if rc != 0:
        return StageResult(
            name=ctx.name,
            status="failed",
            reason=(
                f"serde unit tests failed (rc={rc}); "
                f"see stages/{ctx.name}/stdout.log"
            ),
        )

    # Parse the trailing summary from stdout for a row count.
    n_passed = -1
    try:
        text = ctx.stdout_path.read_text(errors="replace")
        for line in reversed(text.splitlines()):
            if " passed" in line:
                tok = line.strip().split()
                # tokens look like: 17 passed, 3 skipped, ... in 0.84s
                for i, t in enumerate(tok):
                    if t == "passed" and i > 0 and tok[i - 1].isdigit():
                        n_passed = int(tok[i - 1])
                        break
                if n_passed >= 0:
                    break
    except OSError:
        pass

    ctx.log_metric("n_passed", n_passed)
    summary_path = ctx.stage_dir / "serdes_unit_tests_summary.json"
    summary_path.write_text(
        json.dumps({"rc": rc, "n_passed": n_passed, "files": _TEST_FILES}, indent=2)
    )
    ctx.telemetry.log_artifact(summary_path, "serdes_unit_tests_summary")

    ctx.mark_done({"n_passed": n_passed})
    return StageResult(name=ctx.name, status="passed")

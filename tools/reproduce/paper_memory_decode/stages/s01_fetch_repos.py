"""Stage 01: clone / update companion repos.

For each repo configured in .config, either clone fresh or fetch and
check out the pinned ref.  Runs git submodule update --init
--recursive after checkout (FlashInfer needs the cutlass submodule).
Fails if CONFIG_KNLP_FAIL_ON_DIRTY_GIT=y and any repo has uncommitted
changes after checkout.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..stages import StageContext, StageResult


def run(ctx: StageContext) -> StageResult:
    cfg = ctx.cfg
    repos = list(cfg.repos())  # [(name, url, ref, abs_path), ...]

    if not repos:
        ctx.mark_skipped("no repos configured")
        return StageResult(
            name=ctx.name, status="skipped", reason="no repos configured"
        )

    has_git = shutil.which("git")
    if not has_git:
        return StageResult(
            name=ctx.name, status="failed", reason="git not found in PATH"
        )

    # Prevent git from trying to prompt for credentials when running
    # non-interactively.  All repos in the default config are public;
    # if a clone still fails the error message will be visible in the log.
    _git_env = {"GIT_TERMINAL_PROMPT": "0"}

    failures: list[str] = []

    for name, url, ref, abs_path in repos:
        p = Path(abs_path)
        ctx.stdout_path.open("a").write(
            f"\n=== repo: {name}  ref={ref}  path={abs_path} ===\n"
        )

        if p.is_dir() and (p / ".git").exists():
            # Already cloned — fetch and checkout.
            rc = ctx.run_subprocess(
                ["git", "fetch", "--tags", "origin"],
                cwd=str(p),
                extra_env=_git_env,
                timeout=300,
            )
            if rc != 0:
                failures.append(f"{name}: git fetch failed (rc={rc})")
                continue
        else:
            # Fresh clone.
            p.parent.mkdir(parents=True, exist_ok=True)
            rc = ctx.run_subprocess(
                ["git", "clone", url, str(p)],
                extra_env=_git_env,
                timeout=600,
            )
            if rc != 0:
                failures.append(f"{name}: git clone failed (rc={rc})")
                continue

        # Checkout the configured ref.
        rc = ctx.run_subprocess(["git", "checkout", ref], cwd=str(p), timeout=60)
        if rc != 0:
            # May already be on a branch with that name; try reset.
            rc = ctx.run_subprocess(
                ["git", "checkout", "-B", ref, f"origin/{ref}"], cwd=str(p), timeout=60
            )
            if rc != 0:
                failures.append(f"{name}: git checkout {ref!r} failed (rc={rc})")
                continue

        # Submodules (FlashInfer needs cutlass).
        rc = ctx.run_subprocess(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=str(p),
            timeout=600,
        )
        if rc != 0:
            # Non-fatal — warn but don't abort.
            ctx.stderr_path.open("a").write(
                f"WARN: {name}: submodule update failed (rc={rc}), " "build may fail\n"
            )

        # Dirty-repo check.
        if cfg.fail_on_dirty_git:
            status_out = []
            import subprocess

            try:
                result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=str(p),
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                status_out = result.stdout.strip().splitlines()
            except Exception:
                pass
            if status_out:
                failures.append(
                    f"{name}: dirty working tree "
                    f"({len(status_out)} modified files) and "
                    "CONFIG_KNLP_FAIL_ON_DIRTY_GIT=y"
                )
                continue

        # Log the actual HEAD commit for the manifest.
        import subprocess

        try:
            sha = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(p),
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
                .decode()
                .strip()
            )
        except Exception:
            sha = "unknown"
        ctx.log_metric(f"{name}_sha", sha)

    ctx.log_metric("repos_fetched", len(repos) - len(failures))
    ctx.log_metric("repos_failed", len(failures))

    if failures:
        msg = "; ".join(failures)
        with ctx.stderr_path.open("a") as f:
            f.write(f"FETCH FAILURES:\n")
            for fail in failures:
                f.write(f"  {fail}\n")
        return StageResult(name=ctx.name, status="failed", reason=msg)

    ctx.mark_done({"repos": [r[0] for r in repos]})
    return StageResult(name=ctx.name, status="passed")

"""Reproduction orchestrator entry point.

Invoked from Makefile.decode as
    python3 -m tools.reproduce.paper_memory_decode.run <subcommand> --config .config

Subcommands: doctor, estimate, fetch, build, run, report, upload,
provider-info, clean, clobber.
"""

from __future__ import annotations
import argparse
import dataclasses
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from . import doctor as _doctor
from . import estimates as _estimates
from . import hardware as _hardware
from . import manifest as _manifest
from . import report as _report
from . import stages as _stages
from . import telemetry as _telemetry
from .decode_config import DecodeConfig


def _require_decode_enabled(cfg: DecodeConfig) -> None:
    if not cfg.is_enabled():
        sys.stderr.write(
            "CONFIG_KNLP_REPRODUCE_DECODE not enabled.  "
            "Run `make defconfig-decode` (or -sat / -full) first.\n"
        )
        sys.exit(2)


def _existing_run_id(results_root: Path, profile: str) -> Optional[str]:
    """Return the most recent run_id under results_root for this profile,
    if any.  Used by `make decode-run` to resume."""
    if not results_root.exists():
        return None
    candidates = sorted(
        [
            p
            for p in results_root.iterdir()
            if p.is_dir() and p.name.startswith(profile + "-")
        ],
        reverse=True,
    )
    return candidates[0].name if candidates else None


# ── Subcommands ──────────────────────────────────────────────────────────


def cmd_doctor(args, cfg: DecodeConfig) -> int:
    host = _hardware.detect()
    issues, warnings = _doctor.run_checks(cfg, host)
    print(_doctor.render(cfg, host, issues, warnings))
    return 1 if issues else 0


def cmd_estimate(args, cfg: DecodeConfig) -> int:
    host = _hardware.detect()
    print(_estimates.render(_estimates.estimate(cfg, host)))
    return 0


def cmd_provider_info(args) -> int:
    host = _hardware.detect()
    out = {
        "hostname": host.hostname,
        "provider": host.provider,
        "gpu_names": host.gpu_names,
        "gpu_count": host.gpu_count,
    }
    print(json.dumps(out, indent=2))
    return 0


def cmd_fetch(args, cfg: DecodeConfig) -> int:
    """Clone or fetch the four companion repos at their pinned refs."""
    _require_decode_enabled(cfg)
    wt = Path(cfg.worktree_root).resolve()
    wt.mkdir(parents=True, exist_ok=True)

    rc_total = 0
    for name, url, ref, target in cfg.repos():
        target_p = Path(target)
        print(f"\n=== {name}: {url}@{ref} → {target} ===")
        if not target_p.exists():
            rc = subprocess.call(["git", "clone", url, str(target_p)])
            if rc:
                rc_total |= rc
                continue
        else:
            rc = subprocess.call(["git", "fetch", "--all", "--tags"], cwd=str(target_p))
            if rc:
                rc_total |= rc
        # Detect dirty tree
        dirty = (
            subprocess.check_output(["git", "status", "--porcelain"], cwd=str(target_p))
            .decode()
            .strip()
        )
        if dirty and cfg.fail_on_dirty_git:
            print(
                f"ERROR: {name} has uncommitted changes.  "
                f"Set CONFIG_KNLP_FAIL_ON_DIRTY_GIT=n to override."
            )
            return 3
        rc = subprocess.call(["git", "checkout", ref], cwd=str(target_p))
        rc_total |= rc
        # Submodules (FlashInfer needs cutlass)
        subprocess.call(
            ["git", "submodule", "update", "--init", "--recursive"], cwd=str(target_p)
        )
    return rc_total


def cmd_build(args, cfg: DecodeConfig) -> int:
    """Install editable copies of FlashInfer, vLLM, LMCache.  Order
    matters: vLLM pulls flashinfer-python from PyPI and overwrites the
    editable, so we reinstall flashinfer-src last."""
    _require_decode_enabled(cfg)
    wt = Path(cfg.worktree_root).resolve()
    flashinfer = wt / cfg.flashinfer_dir
    vllm = wt / cfg.vllm_dir
    lmcache = wt / cfg.lmcache_dir

    pip = shutil.which("pip3") or shutil.which("pip")
    if not pip:
        print("ERROR: pip not found in PATH")
        return 4

    def install(path: Path) -> int:
        if not path.exists():
            print(f"SKIP: {path} not present (run decode-fetch first)")
            return 5
        print(f"\n=== pip install -e {path} ===")
        env = dict(os.environ)
        env.setdefault("MAX_JOBS", "32")
        env.setdefault("NVCC_THREADS", "2")
        return subprocess.call(
            [pip, "install", "--no-build-isolation", "-e", str(path)],
            env=env,
        )

    rc = 0
    rc |= install(flashinfer)
    rc |= install(vllm)
    rc |= install(lmcache)
    # Reinstall flashinfer last because vllm may overwrite it from PyPI
    rc |= install(flashinfer)
    return rc


def _run_stage_pipeline(cfg: DecodeConfig, stage_filter: Optional[str] = None) -> int:
    host = _hardware.detect()
    results_root = Path(cfg.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    run_id = _existing_run_id(results_root, cfg.profile) or _manifest.make_run_id(
        cfg.profile
    )
    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    m = _manifest.build_manifest(cfg, host, run_id)
    manifest_path = _manifest.save_manifest(m, run_dir)
    _manifest.save_environment(run_dir)
    print(f"run_id={run_id}  manifest={manifest_path}")

    manifest_dict = dataclasses.asdict(m)
    tel = _telemetry.build(cfg, manifest_dict, run_dir)
    tel.start_run(manifest_dict)

    stage_list = _stages.PROFILE_STAGES.get(
        cfg.profile, _stages.PROFILE_STAGES["decode"]
    )
    if stage_filter:
        stage_list = [s for s in stage_list if s == stage_filter]
        if not stage_list:
            print(f"ERROR: stage '{stage_filter}' not in profile {cfg.profile}")
            return 6

    overall_rc = 0
    for stage_name in stage_list:
        ctx = _stages.StageContext(stage_name, run_dir, cfg, host, tel)
        if ctx.already_done():
            print(f"[skip] {stage_name} already complete")
            continue

        tel.start_stage(stage_name, {"profile": cfg.profile})
        print(f"[run]  {stage_name}")
        try:
            result = _stages.get_callable(stage_name)(ctx)
        except Exception as e:
            with ctx.stderr_path.open("a") as f:
                f.write(f"EXCEPTION: {type(e).__name__}: {e}\n")
            result = _stages.StageResult(
                name=stage_name, status="failed", reason=f"{type(e).__name__}: {e}"
            )
        if result.status == "passed":
            ctx.mark_done({"status": "passed", "reason": result.reason})
        elif result.status == "skipped":
            ctx.mark_skipped(result.reason)
        else:
            with ctx.stderr_path.open("a") as f:
                f.write(f"STAGE FAILED: {result.reason}\n")
            overall_rc = 1
            tel.finish_stage(result.status)
            break  # halt on failure; user fixes and reruns
        tel.finish_stage(result.status)

    tel.finish_run()
    return overall_rc


def cmd_run(args, cfg: DecodeConfig) -> int:
    _require_decode_enabled(cfg)
    return _run_stage_pipeline(cfg, getattr(args, "stage", None) or None)


def cmd_report(args, cfg: DecodeConfig) -> int:
    _require_decode_enabled(cfg)
    results_root = Path(cfg.results_root).resolve()
    run_id = _existing_run_id(results_root, cfg.profile)
    if not run_id:
        print("No run dir found; run `make decode-run` first.")
        return 7
    run_dir = results_root / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text())
    stage_results = _report.collect_stage_outputs(run_dir)
    md_path, json_path = _report.write_reports(run_dir, manifest, stage_results)
    print(f"report: {md_path}\nreport: {json_path}")
    return 0


def cmd_upload(args, cfg: DecodeConfig) -> int:
    _require_decode_enabled(cfg)
    if not cfg.upload_artifacts:
        print("CONFIG_KNLP_UPLOAD_ARTIFACTS=n; skipping upload")
        return 0
    print(
        "upload: telemetry sinks already mirrored each metric live.  "
        "Artifact upload at run-end is integrated into stage execution."
    )
    return 0


def cmd_clean(args, cfg: DecodeConfig) -> int:
    """Remove results dir for the active profile but keep cloned repos."""
    _require_decode_enabled(cfg)
    p = Path(cfg.results_root).resolve()
    if p.exists():
        shutil.rmtree(p)
        print(f"removed {p}")
    return 0


def cmd_clobber(args, cfg: DecodeConfig) -> int:
    """Remove results AND cloned repos.  Destructive."""
    _require_decode_enabled(cfg)
    cmd_clean(args, cfg)
    for name, _, _, target in cfg.repos():
        p = Path(target)
        if p.exists():
            shutil.rmtree(p)
            print(f"removed {p}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="paper_memory_decode")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in [
        "doctor",
        "estimate",
        "fetch",
        "build",
        "run",
        "report",
        "upload",
        "clean",
        "clobber",
        "provider-info",
    ]:
        sp = sub.add_parser(name)
        sp.add_argument("--config", default=".config")
        if name == "run":
            sp.add_argument("--stage", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg_path = getattr(args, "config", ".config")
    cfg = DecodeConfig.from_file(cfg_path)

    dispatch = {
        "doctor": lambda: cmd_doctor(args, cfg),
        "estimate": lambda: cmd_estimate(args, cfg),
        "fetch": lambda: cmd_fetch(args, cfg),
        "build": lambda: cmd_build(args, cfg),
        "run": lambda: cmd_run(args, cfg),
        "report": lambda: cmd_report(args, cfg),
        "upload": lambda: cmd_upload(args, cfg),
        "clean": lambda: cmd_clean(args, cfg),
        "clobber": lambda: cmd_clobber(args, cfg),
        "provider-info": lambda: cmd_provider_info(args),
    }
    return dispatch[args.cmd]()


if __name__ == "__main__":
    sys.exit(main())

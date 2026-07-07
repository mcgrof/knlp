# SPDX-License-Identifier: MIT
"""kvio storage-IO capture / replay orchestrator (standalone).

Subcommands: doctor, fetch, build, replay, report, clean.

Usage:
    python3 -m tools.reproduce.kvio.run doctor  --config .config
    python3 -m tools.reproduce.kvio.run fetch   --config .config
    python3 -m tools.reproduce.kvio.run build   --config .config
    python3 -m tools.reproduce.kvio.run replay  --config .config
    python3 -m tools.reproduce.kvio.run report  --config .config

The pure parse / tokenize / schedule paths of the replay stages always run
(they power the offline smoke gate + the unit tests); the live vLLM + LMCache
serving replay + KV capture runs on a GPU box.  Everything skips gracefully
(records an offline JSON) when vllm / lmcache / a GPU is absent.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .kvio_config import KvioConfig
from .stages import StageContext


def _parse_config(config_path: str) -> dict:
    """Parse a kernel-style key=value .config into a dict (quotes stripped)."""
    cfg: dict = {}
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip().strip('"')
    return cfg


def _run(cmd: list[str], cwd: str | None = None, check: bool = True,
         **kw) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=cwd, check=check, **kw)


# ─────────────────────────────────────────────────────────────
# doctor
# ─────────────────────────────────────────────────────────────
def cmd_doctor(cfg: KvioConfig) -> bool:
    print("=== kvio doctor ===")
    print(f"  profile: {cfg.profile}")
    print(f"  Python: {sys.version.split()[0]}")

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"], text=True)
        gpus = [g.strip() for g in out.strip().split("\n") if g.strip()]
        print(f"  GPUs: {len(gpus)}")
        for g in gpus:
            print(f"    {g}")
    except Exception:
        print("  NOTE: nvidia-smi not found -- the live replay will SKIP "
              "(offline smoke/build still runs)")

    for mod in ("vllm", "lmcache", "torch"):
        try:
            __import__(mod)
            print(f"  {mod}: importable")
        except Exception as e:  # noqa: BLE001
            print(f"  NOTE: {mod} not importable ({e.__class__.__name__}); "
                  "live replay will skip")

    if shutil.which("cargo") or shutil.which("rustc"):
        print("  rust: found (needed to build LMCache rust/raw_block)")
    else:
        print("  NOTE: rust toolchain not found -- raw_block build needs "
              "Rust >= 1.87 (use rustup; Ubuntu 24.04 apt rust 1.75 is too old)")
    return True


# ─────────────────────────────────────────────────────────────
# fetch
# ─────────────────────────────────────────────────────────────
def _fetch_repo(repo: str, ref: str, dest: Path) -> None:
    if (dest / ".git").exists():
        current = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=str(dest), text=True).strip()
        if current != ref:
            print(f"  {dest.name}: switching {current or '(detached)'} -> {ref}")
            _run(["git", "fetch", "origin", ref], cwd=str(dest), check=False)
            _run(["git", "checkout", ref], cwd=str(dest))
        else:
            print(f"  {dest.name}: already on {ref}")
            _run(["git", "pull", "--ff-only"], cwd=str(dest), check=False)
    else:
        print(f"  cloning {repo} @ {ref} -> {dest}")
        _run(["git", "clone", "--branch", ref, repo, str(dest)])


def cmd_fetch(cfg: KvioConfig) -> None:
    print("=== kvio fetch ===")
    worktree = Path(cfg.worktree_root)
    _fetch_repo(cfg.vllm_repo, cfg.vllm_ref, worktree / cfg.vllm_dir)
    _fetch_repo(cfg.lmcache_repo, cfg.lmcache_ref, worktree / cfg.lmcache_dir)


# ─────────────────────────────────────────────────────────────
# build
# ─────────────────────────────────────────────────────────────
def cmd_build(cfg: KvioConfig) -> None:
    """Install LMCache (editable) + its rust/raw_block extension, and vLLM.

    raw_block is the engine kvio instruments (NVMe uring_cmd KV offload); the
    editable install does NOT build it, so build a wheel (no activated-venv
    requirement, unlike ``maturin develop``) and install it into this
    interpreter.  Needs Rust >= 1.87.
    """
    print("=== kvio build ===")
    pip = [sys.executable, "-m", "pip"]
    worktree = Path(cfg.worktree_root)
    lmc_path = (worktree / cfg.lmcache_dir).resolve()
    vllm_path = (worktree / cfg.vllm_dir).resolve()

    if not lmc_path.is_dir():
        print(f"  ERROR: LMCache source not found at {lmc_path} "
              "(run `make kvio-fetch` first)")
        sys.exit(1)

    # LMCache editable.
    rc = _run(pip + ["install", "-e", ".[dev]"], cwd=str(lmc_path),
              check=False).returncode
    if rc != 0:
        _run(pip + ["install", "-e", "."], cwd=str(lmc_path))

    # rust/raw_block wheel.
    rb_path = lmc_path / "rust" / "raw_block"
    if rb_path.is_dir():
        _run(pip + ["install", "maturin"], check=False)
        maturin_bin = Path(sys.executable).with_name("maturin")
        maturin = str(maturin_bin) if maturin_bin.exists() else (
            shutil.which("maturin") or "maturin")
        rc = _run([maturin, "build", "--release"], cwd=str(rb_path),
                  check=False).returncode
        if rc != 0:
            print("  ERROR: rust/raw_block maturin build failed; needs "
                  "Rust >= 1.87 (use rustup)")
            sys.exit(1)
        wheels = sorted(rb_path.glob("target/wheels/*.whl")) or sorted(
            lmc_path.glob("**/wheels/lmcache_rust_raw_block_io-*.whl"))
        if not wheels:
            print("  ERROR: rust/raw_block build produced no wheel")
            sys.exit(1)
        _run(pip + ["install", "--force-reinstall", str(wheels[-1])])
    else:
        print("  NOTE: rust/raw_block not present on this LMCache ref "
              "(the kvio branch ships it) -- NVMe uring_cmd capture unavailable")

    # vLLM (precompiled) if a checkout is present.
    if (vllm_path / ".git").exists():
        check = subprocess.run(
            [sys.executable, "-c", "import vllm; print(vllm.__version__)"],
            capture_output=True, text=True)
        if check.returncode == 0:
            print(f"  vLLM already installed: {check.stdout.strip()}")
        else:
            env = os.environ.copy()
            env["VLLM_USE_PRECOMPILED"] = "1"
            _run(pip + ["install", "-e", ".", "--torch-backend=auto"],
                 cwd=str(vllm_path), check=False, env=env)

    # Packages the replay stages / datasets need.
    _run(pip + ["install", "datasets", "accelerate", "transformers", "requests"],
         check=False)
    print("  build done")


# ─────────────────────────────────────────────────────────────
# replay
# ─────────────────────────────────────────────────────────────
_PROFILES = {"mooncake", "content"}
_RECORD_MANIFEST = "kvio_record.json"


def _is_recorded_set(d: Path) -> bool:
    """A directory is a recorded set if it holds a kvio_record.json manifest."""
    return d.is_dir() and (d / _RECORD_MANIFEST).is_file()


def cmd_replay(cfg: KvioConfig) -> int:
    print("=== kvio replay ===")
    profile = cfg.profile
    if profile not in _PROFILES:
        print(f"  ERROR: unknown CONFIG_KNLP_KVIO_PROFILE={profile!r} "
              f"(expected one of {sorted(_PROFILES)})")
        return 2

    # Recorded-set directory: the RECORD_DIR override (CLI: KVIO_PATH) or the
    # default RESULTS_ROOT/record.  "Look for a recorded set" is the default.
    record_dir = Path(
        cfg.record_dir or (Path(cfg.results_root) / "record")
    ).expanduser()

    # Phase 2: a recorded set is present -> replay it GPU-free through the
    # configured backend instead of capturing (unless smoke forces offline).
    if not cfg.smoke and _is_recorded_set(record_dir):
        return _replay_recorded_set(cfg, record_dir, profile)

    # Phase 1 (or smoke): capture.  Smoke forces the stage's offline path even
    # if a GPU is present.  When RECORD_DIR is set, capture writes the recorded
    # set there so a later `make kvio-replay` replays it.
    if cfg.smoke:
        os.environ["KNLP_KVIO_SMOKE"] = "1"
        print("  smoke: forcing offline CPU-only path (no GPU/device touched)")

    if profile == "mooncake":
        from .stages import mooncake_replay as stage_mod
        stage_name = "mooncake_trace_replay"
    else:
        from .stages import content_replay as stage_mod
        stage_name = "content_trace_replay"

    stage_dir = (Path(cfg.results_root) / profile if cfg.smoke else record_dir)
    ctx = StageContext(name=stage_name, cfg=cfg, stage_dir=stage_dir)
    result = stage_mod.run(ctx)

    summary = {"profile": profile, "stage": result.name,
               "status": result.status, "reason": result.reason,
               "mode": "smoke" if cfg.smoke else "capture"}
    (stage_dir / "result.json").write_text(json.dumps(summary, indent=2))
    print(f"  {result.name}: {result.status}"
          f"{' -- ' + result.reason if result.reason else ''}")
    print(f"  artifacts in {stage_dir}")
    if (not cfg.smoke and result.status == "passed"
            and _is_recorded_set(record_dir)):
        print(f"  recorded set -> {record_dir}"
              " (next `make kvio-replay` will replay it GPU-free)")
    return 0 if result.status in ("passed", "skipped") else 1


def _replay_recorded_set(cfg: KvioConfig, record_dir: Path, profile: str) -> int:
    """Replay a recorded set through the configured backend, GPU-free."""
    manifest = json.loads((record_dir / _RECORD_MANIFEST).read_text())
    print(f"  recorded set at {record_dir}: "
          f"model={manifest.get('model')} dataset={manifest.get('dataset')} "
          f"backend={cfg.backend}")
    out = {"profile": profile, "mode": "replay", "backend": cfg.backend,
           "record_dir": str(record_dir), "manifest": manifest}
    stage_dir = Path(cfg.results_root) / profile
    stage_dir.mkdir(parents=True, exist_ok=True)

    if cfg.backend != "lmcache":
        out["status"] = "skipped"
        out["reason"] = f"backend {cfg.backend!r} not implemented (only 'lmcache')"
    elif not manifest.get("l2_image") and not manifest.get("l2_device"):
        out["status"] = "skipped"
        out["reason"] = ("recorded set has no raw_block corpus "
                         "(e.g. mooncake records the semantic trace only)")
    elif not cfg.replay_driver:
        out["status"] = "skipped"
        out["reason"] = ("recorded set present but CONFIG_KNLP_KVIO_REPLAY_DRIVER "
                         "is unset -- point it at ebpf-syscall "
                         "examples/lmcache/kvio_replay.py to replay through "
                         "LMCache raw_block GPU-free")
    else:
        rc = _run_replay_driver(cfg, record_dir, manifest, stage_dir)
        out["status"] = "passed" if rc == 0 else "failed"
        if rc != 0:
            out["reason"] = f"replay driver exited {rc}"

    (stage_dir / "result.json").write_text(json.dumps(out, indent=2))
    print(f"  replay: {out['status']}"
          f"{' -- ' + out['reason'] if out.get('reason') else ''}")
    return 0 if out["status"] in ("passed", "skipped") else 1


def _run_replay_driver(cfg: KvioConfig, record_dir: Path, manifest: dict,
                       stage_dir: Path) -> int:
    """Shell out to the GPU-free kvio_replay driver for the recorded geometry.

    Today the driver replays the recorded set's projected single-payload
    geometry (from the manifest), not the full captured size distribution --
    captured-distribution replay is a TODO.
    """
    driver = Path(cfg.replay_driver).expanduser()
    if not driver.is_file():
        print(f"  ERROR: replay driver not found: {driver}")
        return 1
    if manifest.get("l2_use_uring_cmd") and manifest.get("l2_device"):
        device = cfg.replay_device or manifest["l2_device"]
    else:
        device = cfg.replay_device or str(
            (record_dir / manifest.get("l2_image", "")).resolve())
    payload = str(int(manifest.get("payload_bytes", 0) or 0))
    cap_gb = str(max(1, int(manifest.get("capacity_bytes", 0) or 0) // (1 << 30)))
    env = os.environ.copy()
    if cfg.worktree_root and cfg.lmcache_dir:
        env.setdefault(
            "KVIO_SRC",
            str((Path(cfg.worktree_root) / cfg.lmcache_dir).resolve()))
    cmd = [sys.executable, str(driver),
           "--payload-bytes", payload,
           "--device", device,
           "--engine", cfg.replay_engine,
           "--capacity-gb", cap_gb]
    print(f"  $ {' '.join(cmd)}")
    log = stage_dir / "replay_driver.out"
    with open(log, "w") as f:
        return subprocess.run(cmd, env=env, stdout=f,
                              stderr=subprocess.STDOUT).returncode


# ─────────────────────────────────────────────────────────────
# report
# ─────────────────────────────────────────────────────────────
def cmd_report(cfg: KvioConfig) -> None:
    print("=== kvio report ===")
    stage_dir = Path(cfg.results_root) / cfg.profile
    result_path = stage_dir / "result.json"
    if not result_path.exists():
        print("  no result yet -- run `make kvio-replay` first")
        return
    summary = json.loads(result_path.read_text())
    print(f"  profile : {summary.get('profile')}")
    print(f"  stage   : {summary.get('stage')}")
    print(f"  status  : {summary.get('status')}")
    if summary.get("reason"):
        print(f"  reason  : {summary['reason']}")
    detail = stage_dir / f"{summary.get('stage')}.json"
    if detail.exists():
        print(f"  detail  : {detail}")


# ─────────────────────────────────────────────────────────────
# clean
# ─────────────────────────────────────────────────────────────
def cmd_clean(cfg: KvioConfig) -> None:
    print("=== kvio clean ===")
    root = Path(cfg.results_root)
    if root.exists():
        shutil.rmtree(root)
        print(f"  removed {root}")


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="kvio storage-IO capture/replay")
    ap.add_argument("command",
                    choices=["doctor", "fetch", "build", "replay",
                             "report", "clean"])
    ap.add_argument("--config", default=".config")
    args = ap.parse_args()

    raw = _parse_config(args.config)
    if raw.get("CONFIG_KNLP_REPRODUCE_KVIO") != "y":
        print("CONFIG_KNLP_REPRODUCE_KVIO not enabled. Run "
              "`make defconfig-kvio-lmsys` (or -kvio-mooncake) first.")
        sys.exit(2)
    cfg = KvioConfig.from_raw(raw)

    if args.command == "doctor":
        cmd_doctor(cfg)
    elif args.command == "fetch":
        cmd_fetch(cfg)
    elif args.command == "build":
        cmd_build(cfg)
    elif args.command == "replay":
        sys.exit(cmd_replay(cfg))
    elif args.command == "report":
        cmd_report(cfg)
    elif args.command == "clean":
        cmd_clean(cfg)


if __name__ == "__main__":
    main()

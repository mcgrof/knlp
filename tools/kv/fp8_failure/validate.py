"""Orchestrator for `make defconfig-quant-fp8-validate` -- the HF-only fake-quant reproducibility
lock for the FP8 KV-cache failure atlas.

This is the Phase-2 (reproducible Kconfig) entrypoint for the atlas. It deliberately does NOT touch
the serving stack (vLLM / FlashInfer / LMCache); that lives in the decode reproduction
(`make defconfig-decode`) and stays by-reference here. Everything this runs is the teacher-forced,
fake-quant harness under tools/kv/fp8_failure.

Two profiles:
  - "validate" (the headline lock): runs every atlas runner's `--self-test`. Those build tiny
    random-weight transformers in-process on CPU and assert their invariants -- no network, no GPU,
    no model download, deterministic. This proves the whole harness (all ten runners + the recovery
    arithmetic) is intact on any machine, including CI, without burning a pod. This is what
    `make` runs after `make defconfig-quant-fp8-validate`.
  - "atlas" (the heavy reproduction): runs the real fake-quant fleet (multi-seed CIs, long-context
    2K/8K/16K, the preflight scanner, the recovery Pareto) over the configured models on a GPU.

The orchestrator itself is pure stdlib so it runs under any Python; it subprocesses the runners
under the configured ML interpreter (CONFIG_KNLP_QUANT_FP8_PYTHON, default ~/envs/knlp-serdes/bin/
python -- the repo-default python3 has a broken torch). Subcommands mirror the decode orchestrator:
doctor / selftest / run / report / clean. Run as a module from the repo root:

    python3 -m tools.kv.fp8_failure.validate <subcommand> --config .config
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = (
    Path(__file__).resolve().parents[3]
)  # tools/kv/fp8_failure/validate.py -> repo root

# The deterministic reproducibility lock: each runner's --self-test, CPU-only and offline. {out} is
# replaced with a per-runner output dir. recovery_pareto is pure arithmetic (no torch, no output dir).
SELFTESTS = [
    ("run_smoke", ["--self-test", "--output-dir", "{out}"]),
    ("run_failure_classes", ["--self-test", "--output-dir", "{out}"]),
    ("run_mechanism", ["--self-test", "--output-dir", "{out}"]),
    ("run_v_residual", ["--self-test", "--output-dir", "{out}"]),
    ("run_controls", ["--self-test", "--output-dir", "{out}"]),
    ("run_multiseed", ["--self-test", "--output-dir", "{out}"]),
    ("run_longctx", ["--self-test", "--output-dir", "{out}"]),
    ("run_preflight", ["--self-test", "--output-dir", "{out}"]),
    ("run_gptj", ["--self-test", "--output-dir", "{out}"]),
    ("recovery_pareto", ["--self-test"]),
]

# Atlas fleet: short_name -> HF id. Only models cached on prune (no gated/uncached download) are
# named here so the default "core" reproduction runs offline. Mirrors configs/kv/fp8_failure_models
# .yaml (cached_on_prune allowlist). Use a space-separated CONFIG_KNLP_QUANT_FP8_MODELS to override.
SHORT2HF = {
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
    "Qwen2-7B": "Qwen/Qwen2-7B",
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B",
    "Qwen2.5-3B": "Qwen/Qwen2.5-3B",
    "Qwen2.5-14B": "Qwen/Qwen2.5-14B",
    "Qwen3-4B": "Qwen/Qwen3-4B",
    "DeepSeek-R1-Distill-Qwen-7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "phi-2": "microsoft/phi-2",
    "phi-4": "microsoft/phi-4",
    "Mistral-7B-v0.3": "mistralai/Mistral-7B-v0.3",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-6.9b": "EleutherAI/pythia-6.9b",
    "gpt-neox-20b": "EleutherAI/gpt-neox-20b",
}
# The mechanism-defining core: catastrophic anchor + V-residual + tolerant control + partial-RoPE.
CORE_MODELS = ["Qwen2.5-7B", "phi-2", "Mistral-7B-v0.3", "pythia-410m"]


def _parse_value(raw):
    raw = raw.strip()
    if raw == "y":
        return True
    if raw == "n":
        return False
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    try:
        return int(raw)
    except ValueError:
        return raw


def parse_config(path):
    """Read a Kconfig .config into a flat dict, keys keep the CONFIG_ prefix."""
    out = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, raw = line.partition("=")
        out[key.strip()] = _parse_value(raw)
    return out


@dataclass
class Fp8Config:
    profile: str = "validate"
    python: str = "~/envs/knlp-serdes/bin/python"
    results_root: str = "./results/quant-fp8"
    models: str = "core"
    dataset: str = "wikitext-103-raw-v1"
    seq_len: int = 1024
    num_prompts: int = 16
    seeds: str = "0 1 2"
    contexts: str = "2048 8192 16384"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    run_multiseed: bool = True
    run_longctx: bool = True
    run_preflight: bool = True
    run_recovery: bool = True
    allow_hardware_skips: bool = True
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_file(cls, path):
        raw = parse_config(path)
        g = lambda k, d: raw.get("CONFIG_KNLP_QUANT_FP8_" + k, d)
        return cls(
            profile=g("PROFILE", "validate"),
            python=g("PYTHON", "~/envs/knlp-serdes/bin/python"),
            results_root=g("RESULTS_ROOT", "./results/quant-fp8"),
            models=g("MODELS", "core"),
            dataset=g("DATASET", "wikitext-103-raw-v1"),
            seq_len=int(g("SEQ_LEN", 1024)),
            num_prompts=int(g("NUM_PROMPTS", 16)),
            seeds=str(g("SEEDS", "0 1 2")),
            contexts=str(g("CONTEXTS", "2048 8192 16384")),
            device=g("DEVICE", "cuda:0"),
            dtype=g("DTYPE", "bfloat16"),
            run_multiseed=bool(g("RUN_MULTISEED", True)),
            run_longctx=bool(g("RUN_LONGCTX", True)),
            run_preflight=bool(g("RUN_PREFLIGHT", True)),
            run_recovery=bool(g("RUN_RECOVERY", True)),
            allow_hardware_skips=bool(g("ALLOW_HARDWARE_SKIPS", True)),
            raw=raw,
        )

    def is_enabled(self):
        return bool(self.raw.get("CONFIG_KNLP_QUANT_FP8"))

    def py(self):
        return os.path.expanduser(self.python)

    def model_list(self):
        m = self.models.strip()
        if m in ("core", ""):
            return list(CORE_MODELS)
        if m == "all":
            return list(SHORT2HF)
        return m.split()


def _run_id(profile):
    return f"{profile}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"


def _probe_python(py):
    """Return (ok, cuda_available, device_count, detail) for the configured interpreter."""
    code = (
        "import json,sys\n"
        "d={'torch':False,'transformers':False,'datasets':False,'cuda':False,'ndev':0}\n"
        "try:\n import torch; d['torch']=True; d['cuda']=bool(torch.cuda.is_available());"
        " d['ndev']=torch.cuda.device_count()\n"
        "except Exception as e: d['torch_err']=str(e)[:200]\n"
        "try:\n import transformers; d['transformers']=True\n"
        "except Exception as e: d['tf_err']=str(e)[:200]\n"
        "try:\n import datasets; d['datasets']=True\n"
        "except Exception as e: d['ds_err']=str(e)[:200]\n"
        "print(json.dumps(d))\n"
    )
    try:
        r = subprocess.run(
            [py, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120
        )
    except Exception as e:  # interpreter missing / not executable
        return False, False, 0, {"error": f"cannot launch {py}: {e}"}
    if r.returncode != 0:
        return False, False, 0, {"error": (r.stderr or r.stdout).strip()[:300]}
    try:
        d = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return False, False, 0, {"error": r.stdout.strip()[:300]}
    return d.get("torch", False), d.get("cuda", False), int(d.get("ndev", 0)), d


def _stage_env():
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


# ── doctor ──────────────────────────────────────────────────────────────────


def cmd_doctor(args, cfg):
    issues, warnings = [], []
    ok_torch, cuda, ndev, detail = _probe_python(cfg.py())
    needs_gpu = cfg.profile == "atlas"

    if not detail.get("transformers"):
        issues.append(f"transformers not importable under {cfg.py()}")
    if not ok_torch:
        msg = f"torch not importable under {cfg.py()}: {detail.get('torch_err', detail.get('error', ''))}"
        # recovery_pareto self-test still works without torch; everything else needs it.
        (warnings if cfg.allow_hardware_skips else issues).append(msg)
    if cfg.profile == "atlas" and not detail.get("datasets"):
        issues.append(
            f"datasets not importable under {cfg.py()} (atlas needs wikitext)"
        )
    if needs_gpu and not cuda:
        (warnings if cfg.allow_hardware_skips else issues).append(
            "no CUDA GPU available; atlas fake-quant needs a GPU (set ALLOW_HARDWARE_SKIPS=y to skip)"
        )
    if cfg.profile not in ("validate", "smoke", "atlas"):
        issues.append(f"unknown profile {cfg.profile!r} (validate|smoke|atlas)")

    print("=" * 60)
    print(f"quant-fp8 doctor  profile={cfg.profile}  python={cfg.py()}")
    print(
        f"  torch={ok_torch} transformers={detail.get('transformers')} "
        f"datasets={detail.get('datasets')} cuda={cuda} gpus={ndev}"
    )
    print(f"  results_root={cfg.results_root}  models={cfg.models}")
    for w in warnings:
        print(f"  ~ {w}")
    if issues:
        print("  issues:")
        for i in issues:
            print(f"  - {i}")
        print("=" * 60)
        return 1
    print("  all checks passed.")
    print("=" * 60)
    return 0


# ── selftest (the reproducibility lock) ─────────────────────────────────────


def _run_selftests(cfg, run_dir):
    """Run every runner's --self-test under the ML python. Returns list of result dicts and writes a
    DONE/SKIPPED marker per runner so the report can read them."""
    py = cfg.py()
    ok_torch, _, _, _ = _probe_python(py)
    results = []
    st_dir = run_dir / "selftest"
    st_dir.mkdir(parents=True, exist_ok=True)
    for name, argv_tmpl in SELFTESTS:
        out = st_dir / name
        argv = [a.format(out=str(out)) for a in argv_tmpl]
        needs_torch = name != "recovery_pareto"
        marker = st_dir / f"{name}.json"
        if needs_torch and not ok_torch:
            rec = {"runner": name, "status": "skipped", "reason": "torch unavailable"}
            marker.write_text(json.dumps(rec, indent=2))
            results.append(rec)
            print(f"  ~ {name}: SKIPPED (torch unavailable)")
            continue
        t0 = time.time()
        r = subprocess.run(
            [py, "-m", f"tools.kv.fp8_failure.{name}", *argv],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=_stage_env(),
        )
        dt = round(time.time() - t0, 1)
        status = "passed" if r.returncode == 0 else "failed"
        rec = {"runner": name, "status": status, "seconds": dt, "rc": r.returncode}
        if status == "failed":
            rec["stderr_tail"] = (r.stderr or r.stdout).strip().splitlines()[-8:]
        marker.write_text(json.dumps(rec, indent=2))
        results.append(rec)
        glyph = "OK" if status == "passed" else "FAIL"
        print(f"  [{glyph}] {name}  ({dt}s)")
        if status == "failed":
            for ln in rec.get("stderr_tail", []):
                print(f"        {ln}")
    return results


def cmd_selftest(args, cfg):
    if not cfg.is_enabled():
        sys.stderr.write(
            "CONFIG_KNLP_QUANT_FP8 not enabled; load defconfig-quant-fp8-validate\n"
        )
        return 2
    run_dir = _ensure_run_dir(cfg)
    print(f"quant-fp8 selftest -> {run_dir}")
    results = _run_selftests(cfg, run_dir)
    _write_manifest(cfg, run_dir, results)
    failed = [r for r in results if r["status"] == "failed"]
    npass = sum(r["status"] == "passed" for r in results)
    nskip = sum(r["status"] == "skipped" for r in results)
    print(f"selftest: {npass} passed, {len(failed)} failed, {nskip} skipped")
    return 1 if failed else 0


# ── run (profile dispatch) ──────────────────────────────────────────────────


def _ensure_run_dir(cfg):
    root = Path(cfg.results_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    run_dir = root / _run_id(cfg.profile)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _git_commit():
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _write_manifest(cfg, run_dir, results):
    manifest = {
        "schema_version": 1,
        "profile": cfg.profile,
        "run_id": run_dir.name,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": cfg.py(),
        "git_commit": _git_commit(),
        "measurement_level": "fake_quant_teacher_forced",
        "serving_stack": "by-reference (see make defconfig-decode)",
        "config": {
            k: v for k, v in cfg.raw.items() if k.startswith("CONFIG_KNLP_QUANT_FP8")
        },
        "results": results,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def _run_atlas_model(cfg, run_dir, short):
    """Dispatch the real fake-quant runners for one model. Each writes its own CSVs under run_dir."""
    py, dev, dt = cfg.py(), cfg.device, cfg.dtype
    hf = SHORT2HF.get(short, short)
    env = _stage_env()
    results = []

    def _call(name, argv, sub):
        out = run_dir / sub
        full = [
            py,
            "-m",
            f"tools.kv.fp8_failure.{name}",
            "--model",
            hf,
            "--short-name",
            short,
            "--output-dir",
            str(out),
            "--device",
            dev,
            "--dtype",
            dt,
            *argv,
        ]
        t0 = time.time()
        r = subprocess.run(full, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
        dt_s = round(time.time() - t0, 1)
        status = "passed" if r.returncode == 0 else "failed"
        rec = {
            "runner": name,
            "model": short,
            "status": status,
            "seconds": dt_s,
            "rc": r.returncode,
        }
        if status == "failed":
            rec["stderr_tail"] = (r.stderr or r.stdout).strip().splitlines()[-8:]
        print(f"  [{ 'OK' if status=='passed' else 'FAIL'}] {name}:{short}  ({dt_s}s)")
        results.append(rec)

    if cfg.run_multiseed:
        _call(
            "run_multiseed",
            [
                "--num-prompts",
                str(cfg.num_prompts),
                "--seq-len",
                str(cfg.seq_len),
                "--seeds",
                *cfg.seeds.split(),
            ],
            "multiseed",
        )
    if cfg.run_longctx:
        _call("run_longctx", ["--contexts", *cfg.contexts.split()], "longctx")
    if cfg.run_preflight:
        _call("run_preflight", ["--seq-len", str(cfg.seq_len)], "preflight")
    return results


def cmd_run(args, cfg):
    if not cfg.is_enabled():
        sys.stderr.write(
            "CONFIG_KNLP_QUANT_FP8 not enabled; load defconfig-quant-fp8-validate\n"
        )
        return 2
    if cfg.profile in ("validate", "smoke"):
        # the lock: deterministic runner self-tests
        return cmd_selftest(args, cfg)
    if cfg.profile != "atlas":
        sys.stderr.write(f"unknown profile {cfg.profile!r}\n")
        return 3

    run_dir = _ensure_run_dir(cfg)
    print(f"quant-fp8 atlas -> {run_dir}  models={cfg.model_list()}")
    results = []
    for short in cfg.model_list():
        print(f"--- {short} ---")
        results += _run_atlas_model(cfg, run_dir, short)
    if cfg.run_recovery:
        out = run_dir / "recovery"
        r = subprocess.run(
            [
                cfg.py(),
                "-m",
                "tools.kv.fp8_failure.recovery_pareto",
                "--output-dir",
                str(out),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=_stage_env(),
        )
        results.append(
            {
                "runner": "recovery_pareto",
                "status": "passed" if r.returncode == 0 else "failed",
                "rc": r.returncode,
            }
        )
    _write_manifest(cfg, run_dir, results)
    failed = [r for r in results if r["status"] == "failed"]
    print(f"atlas: {len(results) - len(failed)} ok, {len(failed)} failed")
    return 1 if failed else 0


# ── report ──────────────────────────────────────────────────────────────────


def _latest_run_dir(cfg):
    root = Path(cfg.results_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    runs = sorted(root.glob(f"{cfg.profile}-*")) if root.exists() else []
    return runs[-1] if runs else None


def cmd_report(args, cfg):
    run_dir = _latest_run_dir(cfg)
    if run_dir is None:
        sys.stderr.write(
            f"no run dir under {cfg.results_root} for profile {cfg.profile}\n"
        )
        return 7
    mpath = run_dir / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {"results": []}
    results = manifest.get("results", [])
    npass = sum(r.get("status") == "passed" for r in results)
    nfail = sum(r.get("status") == "failed" for r in results)
    nskip = sum(r.get("status") == "skipped" for r in results)
    report = {
        "run_id": run_dir.name,
        "profile": cfg.profile,
        "git_commit": manifest.get("git_commit"),
        "pass_count": npass,
        "fail_count": nfail,
        "skip_count": nskip,
        "results": results,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    lines = [
        f"# quant-fp8 {cfg.profile} report",
        "",
        f"run: `{run_dir.name}`  git: `{manifest.get('git_commit', 'unknown')[:12]}`",
        f"measurement: {manifest.get('measurement_level', 'fake_quant_teacher_forced')}  "
        f"(serving stack by-reference: `make defconfig-decode`)",
        "",
        "| runner | model | status | seconds |",
        "|---|---|---|---|",
    ]
    glyph = {"passed": "PASS", "failed": "FAIL", "skipped": "skip"}
    for r in results:
        lines.append(
            f"| {r.get('runner','?')} | {r.get('model','-')} | "
            f"{glyph.get(r.get('status'), r.get('status'))} | {r.get('seconds','-')} |"
        )
    lines += ["", f"**{npass} passed, {nfail} failed, {nskip} skipped.**", ""]
    (run_dir / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"[report] -> {run_dir/'report.md'} , {run_dir/'report.json'}")
    return 1 if nfail else 0


def cmd_clean(args, cfg):
    root = Path(cfg.results_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    import shutil

    n = 0
    if root.exists():
        for d in root.glob(f"{cfg.profile}-*"):
            shutil.rmtree(d, ignore_errors=True)
            n += 1
    print(f"[clean] removed {n} {cfg.profile} run dir(s) under {root}")
    return 0


def _build_parser():
    p = argparse.ArgumentParser(prog="quant-fp8-validate")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ["doctor", "selftest", "run", "report", "clean"]:
        sp = sub.add_parser(name)
        sp.add_argument("--config", default=".config")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    cfg = Fp8Config.from_file(getattr(args, "config", ".config"))
    dispatch = {
        "doctor": lambda: cmd_doctor(args, cfg),
        "selftest": lambda: cmd_selftest(args, cfg),
        "run": lambda: cmd_run(args, cfg),
        "report": lambda: cmd_report(args, cfg),
        "clean": lambda: cmd_clean(args, cfg),
    }
    return dispatch[args.cmd]()


if __name__ == "__main__":
    sys.exit(main())

# SPDX-License-Identifier: MIT
"""Weight-loading I/O characterization orchestrator.

Subcommands: doctor, plan, inventory, fetch, tracers, matrix, gpu, report,
clean. Each reads the kernel-style ``.config`` written by a
``defconfig-weight-io-*`` target and drives the scripts in
``scripts/weight_io``. Every stage is resumable: outputs that exist are
not recomputed, and a traced run with a summary is not repeated.

    python3 -m tools.reproduce.weight_io.run doctor --config .config
    python3 -m tools.reproduce.weight_io.run inventory --config .config

What the stages measure, and the meaning of the counters they report,
is documented in scripts/weight_io/README.md.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

KNLP = Path(__file__).resolve().parents[3]
SCRIPTS = KNLP / "scripts" / "weight_io"
TRACER_BINARIES = ("nvme_tp_monitor", "syscall_monitor", "mmap_readamp")


def _as_bool(v) -> bool:
    return str(v).strip().lower() in ("y", "1", "true", "yes")


def _as_int(v, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


@dataclass
class Config:
    enabled: bool = False
    root: str = "./results/weight-io"
    repos: list = None
    control_repos: list = None
    fetch_compact: bool = False
    tracer_repo: str = "https://github.com/SamsungDS/ebpf-syscall.git"
    tracer_ref: str = "main"
    tracer_dir: str = ""
    disk: str = ""
    groups: list = None
    trials: int = 5
    intent: bool = True
    idle_probe: bool = True
    gpu: bool = False
    gpu_layer: int = 13
    python: str = "python3"
    owner: str = ""

    @classmethod
    def from_file(cls, path: str) -> "Config":
        raw: dict = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                raw[k.strip()] = v.strip().strip('"')

        def g(key, default=""):
            return raw.get("CONFIG_KNLP_WEIGHT_IO_" + key, default)

        return cls(
            enabled=_as_bool(raw.get("CONFIG_KNLP_WEIGHT_IO", "n")),
            root=g("ROOT", "./results/weight-io"),
            repos=g("REPOS").split(),
            control_repos=g("CONTROL_REPOS").split(),
            fetch_compact=_as_bool(g("FETCH_COMPACT", "n")),
            tracer_repo=g(
                "TRACER_REPO", "https://github.com/SamsungDS/ebpf-syscall.git"
            ),
            tracer_ref=g("TRACER_REF", "main"),
            tracer_dir=g("TRACER_DIR", ""),
            disk=g("DISK", ""),
            groups=g("GROUPS").split(),
            trials=_as_int(g("TRIALS", 5), 5),
            intent=_as_bool(g("INTENT", "y")),
            idle_probe=_as_bool(g("IDLE_PROBE", "y")),
            gpu=_as_bool(g("GPU", "n")),
            gpu_layer=_as_int(g("GPU_LAYER", 13), 13),
            python=g("PYTHON", "python3") or "python3",
            owner=g("OWNER", ""),
        )

    @property
    def root_path(self) -> Path:
        return Path(self.root).expanduser().resolve()

    def stem(self, repo: str) -> str:
        return repo.replace("/", "__")

    def full_dir(self, repo: str) -> Path:
        return self.root_path / "models" / self.stem(repo) / "full"


def _run(cmd: list, check: bool = True, **kw) -> subprocess.CompletedProcess:
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], check=check, **kw)


def _git_head(path: Path) -> str:
    try:
        return (
            subprocess.run(
                ["git", "-C", str(path), "rev-parse", "--short=12", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            or "unknown"
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ---------------------------------------------------------------------------
# device geometry: the mount that holds the data root
# ---------------------------------------------------------------------------
def _sysfs_int(path: str, default: int = 0) -> int:
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return default


def device_geometry(root: Path, disk_override: str = "") -> dict:
    """Whole-disk name, partition start (512 B sectors) and queue limits of
    the device holding `root`. Geometry only: nothing that names a part."""
    root.mkdir(parents=True, exist_ok=True)
    out = subprocess.run(
        ["findmnt", "-no", "SOURCE,FSTYPE", "--target", str(root)],
        capture_output=True,
        text=True,
    ).stdout.split()
    source = out[0] if out else ""
    fstype = out[1] if len(out) > 1 else ""
    part = Path(source).name if source.startswith("/dev/") else ""
    disk = disk_override
    start = 0
    if part:
        link = Path("/sys/class/block") / part
        if (link / "partition").exists():
            parent = link.resolve().parent.name
            disk = disk or parent
            start = _sysfs_int(str(link / "start"))
        else:
            disk = disk or part
    q = f"/sys/block/{disk}/queue" if disk else ""
    return {
        "mount_source_kind": "partition" if start else ("disk" if part else "unknown"),
        "fstype": fstype,
        "disk": disk,
        "partition": part,
        "part_start_sectors": start,
        "logical_block_bytes": _sysfs_int(f"{q}/logical_block_size") if q else 0,
        "physical_block_bytes": _sysfs_int(f"{q}/physical_block_size") if q else 0,
        "max_sectors_kb": _sysfs_int(f"{q}/max_sectors_kb") if q else 0,
        "max_hw_sectors_kb": _sysfs_int(f"{q}/max_hw_sectors_kb") if q else 0,
        "max_segments": _sysfs_int(f"{q}/max_segments") if q else 0,
        "read_ahead_kb": _sysfs_int(f"{q}/read_ahead_kb") if q else 0,
    }


def tracer_dir(cfg: Config) -> Path:
    if cfg.tracer_dir:
        return Path(cfg.tracer_dir).expanduser().resolve()
    return cfg.root_path / "tools" / "ebpf-syscall"


def tracer_bins(cfg: Config) -> dict:
    d = tracer_dir(cfg)
    return {b: d / b for b in TRACER_BINARIES}


def have_tracers(cfg: Config) -> bool:
    return all(p.exists() for p in tracer_bins(cfg).values())


# ---------------------------------------------------------------------------
# doctor and plan
# ---------------------------------------------------------------------------
def cmd_doctor(cfg: Config) -> bool:
    print("=== weight-io doctor ===")
    ok = True
    print(f"  python: {sys.version.split()[0]} ({sys.executable})")
    print(f"  data root: {cfg.root_path}")
    needs_device = bool(cfg.groups) or cfg.gpu
    if needs_device:
        geo = device_geometry(cfg.root_path, cfg.disk)
        print(f"  device: {json.dumps(geo)}")
        if not geo["disk"] or not geo["disk"].startswith("nvme"):
            print(
                "  !! the data root is not on an NVMe namespace; the tracer attaches to nvme tracepoints"
            )
            ok = False
        if not Path("/sys/kernel/btf/vmlinux").exists():
            print("  !! no /sys/kernel/btf/vmlinux: the tracers need a BTF kernel")
            ok = False
        sudo = (
            subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0
        )
        print(
            f"  passwordless sudo: {'yes' if sudo else 'no (the tracers will prompt)'}"
        )
        if have_tracers(cfg):
            print(f"  tracers: built in {tracer_dir(cfg)}")
        elif cfg.tracer_dir:
            print(
                f"  !! tracer dir {cfg.tracer_dir} lacks {', '.join(TRACER_BINARIES)}"
            )
            ok = False
        else:
            for tool in ("clang", "bpftool", "git", "make"):
                if shutil.which(tool) is None:
                    print(f"  !! {tool} not found; needed to build the tracers")
                    ok = False
            if (
                subprocess.run(
                    ["pkg-config", "--exists", "libbpf"], capture_output=True
                ).returncode
                != 0
            ):
                print("  (no system libbpf; the tracer build clones libbpf itself)")
    if cfg.gpu:
        probe = subprocess.run(
            [
                cfg.python,
                "-c",
                "import torch, transformers, safetensors; "
                "print(torch.__version__, torch.cuda.is_available(), transformers.__version__)",
            ],
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            print(
                f"  gpu interpreter: {cfg.python}: torch, cuda available, transformers = {probe.stdout.strip()}"
            )
            if " False " in " " + probe.stdout:
                print("  !! torch reports no GPU")
                ok = False
        else:
            print(
                f"  !! {cfg.python} lacks torch, transformers or safetensors:\n{probe.stderr.strip()[-400:]}"
            )
            ok = False
    tok = (
        os.environ.get("HF_TOKEN")
        or (
            Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
            / "token"
        ).exists()
    )
    print(
        f"  hub token: {'present' if tok else 'none (gated repositories will be skipped)'}"
    )
    print("  verdict: " + ("ready" if ok else "not ready"))
    return ok


def cmd_plan(cfg: Config) -> bool:
    print("=== weight-io plan ===")
    print(f"  inventory: {len(cfg.repos)} repositories by shard headers")
    print(
        f"  fetch: {len(cfg.control_repos)} controls in full"
        + (", selected ranges of the rest" if cfg.fetch_compact else "")
    )
    print(
        f"  tracers: {'existing ' + cfg.tracer_dir if cfg.tracer_dir else cfg.tracer_repo + ' @ ' + cfg.tracer_ref}"
    )
    print(
        f"  matrix: groups {cfg.groups or '(none)'}, {cfg.trials} trials, intent tracing {'on' if cfg.intent else 'off'}, idle probe {'on' if cfg.idle_probe else 'off'}"
    )
    print(
        f"  gpu: {'layer ' + str(cfg.gpu_layer) + ' with ' + cfg.python if cfg.gpu else 'off'}"
    )
    print(f"  root: {cfg.root_path}")
    return True


# ---------------------------------------------------------------------------
# stage A: inventory
# ---------------------------------------------------------------------------
def cmd_inventory(cfg: Config) -> bool:
    print("=== weight-io inventory ===")
    a = cfg.root_path / "stageA"
    a.mkdir(parents=True, exist_ok=True)
    pins = a / "pins"
    missing = [
        r for r in cfg.repos if not (pins / (cfg.stem(r) + ".pin.json")).exists()
    ]
    if missing:
        _run(
            [
                sys.executable,
                SCRIPTS / "hf_inventory.py",
                "resolve",
                *missing,
                "--out",
                pins,
            ]
        )
    inv = a / "inventory.jsonl"
    if not inv.exists():
        _run(
            [
                sys.executable,
                SCRIPTS / "hf_inventory.py",
                "headers",
                "--pins",
                pins,
                "--out",
                inv,
            ]
        )
    classes = a / "classes.json"
    if not classes.exists():
        _run(
            [
                sys.executable,
                SCRIPTS / "hf_inventory.py",
                "classify",
                "--inventory",
                inv,
                "--out",
                classes,
            ]
        )
    sel = a / "selection.jsonl"
    if not sel.exists():
        _run(
            [
                sys.executable,
                SCRIPTS / "hf_inventory.py",
                "select",
                "--inventory",
                inv,
                "--out",
                sel,
            ]
        )
    geo = device_geometry(cfg.root_path, cfg.disk) if (cfg.groups or cfg.gpu) else {}
    max_cmd = (geo.get("max_sectors_kb") or 128) * 1024
    (a / "device.json").write_text(json.dumps(geo, indent=1))
    proj = a / "projections.json"
    if not proj.exists():
        _run(
            [
                sys.executable,
                SCRIPTS / "projections.py",
                "--selection",
                sel,
                "--max-cmd",
                str(max_cmd),
                "--out",
                proj,
            ]
        )
    report = [
        sys.executable,
        SCRIPTS / "inventory_report.py",
        "--stage-a",
        a,
        "--out",
        a / "INVENTORY.md",
    ]
    fm = cfg.root_path / "models" / "fetch_manifest.jsonl"
    if fm.exists():
        report += ["--fetch-manifest", fm]
    _run(report)
    print(f"  tables: {a / 'INVENTORY.md'}")
    return True


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------
def cmd_fetch(cfg: Config) -> bool:
    print("=== weight-io fetch ===")
    a = cfg.root_path / "stageA"
    models = cfg.root_path / "models"
    for repo in cfg.control_repos:
        pin = a / "pins" / (cfg.stem(repo) + ".pin.json")
        if not pin.exists():
            print(f"  !! no pin for {repo}; run the inventory stage first")
            return False
        if (cfg.full_dir(repo) / "full_manifest.json").exists():
            print(f"  have {repo} in full")
            continue
        _run(
            [
                sys.executable,
                SCRIPTS / "hf_inventory.py",
                "fetch-full",
                "--pin",
                pin,
                "--out",
                models,
            ]
        )
    if cfg.fetch_compact:
        others = [r for r in cfg.repos if r not in cfg.control_repos]
        if others:
            _run(
                [
                    sys.executable,
                    SCRIPTS / "hf_inventory.py",
                    "fetch",
                    "--selection",
                    a / "selection.jsonl",
                    "--out",
                    models,
                    "--repos",
                    *others,
                ]
            )
            _run(
                [
                    sys.executable,
                    SCRIPTS / "hf_inventory.py",
                    "verify",
                    "--manifest",
                    models / "fetch_manifest.jsonl",
                    "--out",
                    models,
                ]
            )
    if not cfg.control_repos and not cfg.fetch_compact:
        print("  nothing to fetch (inventory-only configuration)")
    return True


# ---------------------------------------------------------------------------
# tracers
# ---------------------------------------------------------------------------
def cmd_tracers(cfg: Config) -> bool:
    print("=== weight-io tracers ===")
    if not (cfg.groups or cfg.gpu):
        print("  no traced stage configured; skipping")
        return True
    d = tracer_dir(cfg)
    if have_tracers(cfg):
        print(f"  built: {d} ({_git_head(d)})")
        return True
    if cfg.tracer_dir:
        print(f"  !! {d} lacks {', '.join(TRACER_BINARIES)}; build them there")
        return False
    if not d.exists():
        d.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", cfg.tracer_repo, d])
    _run(["git", "-C", d, "fetch", "--all", "--tags"], check=False)
    _run(["git", "-C", d, "checkout", "--quiet", cfg.tracer_ref])
    _run(["make", "-C", d, "vmlinux.h"], check=False)
    _run(["make", "-C", d, *TRACER_BINARIES])
    print(f"  built: {d} ({_git_head(d)})")
    return have_tracers(cfg)


# ---------------------------------------------------------------------------
# traced replay matrix (no GPU)
# ---------------------------------------------------------------------------
def _intent_args(cfg: Config) -> list:
    if not cfg.intent:
        return []
    b = tracer_bins(cfg)
    return [
        "--intent-tracer",
        b["syscall_monitor"],
        "--fault-tracer",
        b["mmap_readamp"],
    ]


def cmd_matrix(cfg: Config) -> bool:
    print("=== weight-io matrix ===")
    if not cfg.groups:
        print("  no replay groups configured; skipping")
        return True
    if not have_tracers(cfg):
        print("  !! tracers not built; run the tracers stage")
        return False
    geo = device_geometry(cfg.root_path, cfg.disk)
    (cfg.root_path / "stageA" / "device.json").write_text(json.dumps(geo, indent=1))
    if not geo["disk"]:
        print("  !! cannot resolve the block device of the data root")
        return False
    bins = tracer_bins(cfg)
    runs = cfg.root_path / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    if cfg.idle_probe and not (runs / "probe" / "idle_latency.json").exists():
        shards = []
        for repo in cfg.control_repos:
            shards += sorted(cfg.full_dir(repo).glob("*.safetensors"))
        if shards:
            target = max(shards, key=lambda p: p.stat().st_size)
            (runs / "probe").mkdir(exist_ok=True)
            _run(
                [
                    sys.executable,
                    SCRIPTS / "idle_latency.py",
                    "--file",
                    target,
                    "--reps",
                    "10",
                    "--out",
                    runs / "probe" / "idle_latency.json",
                ]
            )
    _run(
        [
            sys.executable,
            SCRIPTS / "stageB_matrix.py",
            "--root",
            cfg.root_path,
            "--tracer",
            bins["nvme_tp_monitor"],
            "--disk",
            geo["disk"],
            "--part-start-sectors",
            str(geo["part_start_sectors"]),
            "--trials",
            str(cfg.trials),
            "--groups",
            *cfg.groups,
            *_intent_args(cfg),
        ]
    )
    return True


# ---------------------------------------------------------------------------
# GPU consumption
# ---------------------------------------------------------------------------
def _traced(cfg: Config, geo: dict, run_id: str, child: str, child_args: list) -> None:
    rd = cfg.root_path / "runs" / run_id
    if (rd / "summary.json").exists():
        print(f"  have {run_id}")
        return
    bins = tracer_bins(cfg)
    _run(
        [
            cfg.python,
            SCRIPTS / "traced_replay.py",
            "--tracer",
            bins["nvme_tp_monitor"],
            "--disk",
            geo["disk"],
            "--run-dir",
            rd,
            "--child",
            child,
            *_intent_args(cfg),
            "--",
            *child_args,
            "--part-start-sectors",
            str(geo["part_start_sectors"]),
        ],
        check=False,
    )


def _quant_method(cfg: Config, repo: str):
    pin = cfg.root_path / "stageA" / "pins" / (cfg.stem(repo) + ".pin.json")
    if not pin.exists():
        return None
    conf = json.loads(pin.read_text()).get("config.json") or {}
    return (conf.get("quantization_config") or {}).get("quant_method")


def cmd_gpu(cfg: Config) -> bool:
    print("=== weight-io gpu ===")
    if not cfg.gpu:
        print("  GPU stage not configured; skipping")
        return True
    if not have_tracers(cfg):
        print("  !! tracers not built; run the tracers stage")
        return False
    geo = device_geometry(cfg.root_path, cfg.disk)
    local = [r for r in cfg.control_repos if cfg.full_dir(r).is_dir()]
    if not local:
        print("  !! no control checkpoint downloaded")
        return False

    def size_of(repo):
        return sum(p.stat().st_size for p in cfg.full_dir(repo).glob("*.safetensors"))

    small = min(local, key=size_of)
    dense = [r for r in local if _quant_method(cfg, r) is None and r != small]
    awq = [r for r in local if _quant_method(cfg, r) == "awq"]
    L = cfg.gpu_layer
    trials = str(cfg.trials)
    name = lambda r: r.split("/")[-1]
    for cache in ("cold", "warm"):
        _traced(
            cfg,
            geo,
            f"gpu/{name(small)}/native-{cache}-resident",
            "gpu_resident_infer.py",
            ["--model-dir", cfg.full_dir(small), "--cache", cache, "--trials", trials],
        )
    if dense:
        d = max(dense, key=size_of)
        base = ["--model-dir", cfg.full_dir(d), "--layer", str(L), "--trials", trials]
        _traced(
            cfg,
            geo,
            f"gpu/{name(d)}-layer{L}/resident",
            "gpu_layer_stream.py",
            base + ["--mode", "resident"],
        )
        for read in ("pread", "direct"):
            _traced(
                cfg,
                geo,
                f"gpu/{name(d)}-layer{L}/stream-{read}-cold",
                "gpu_layer_stream.py",
                base + ["--mode", "stream", "--read", read, "--cache", "cold"],
            )
        _traced(
            cfg,
            geo,
            f"gpu/{name(d)}-layer{L}/stream-pread-warm",
            "gpu_layer_stream.py",
            base + ["--mode", "stream", "--read", "pread", "--cache", "warm"],
        )
        for q in awq:
            qb = ["--model-dir", cfg.full_dir(q), "--layer", str(L), "--trials", trials]
            _traced(
                cfg,
                geo,
                f"gpu/{name(q)}-layer{L}/resident",
                "gpu_layer_stream.py",
                qb + ["--mode", "resident", "--awq-reference-dir", cfg.full_dir(d)],
            )
            for read in ("pread", "direct"):
                _traced(
                    cfg,
                    geo,
                    f"gpu/{name(q)}-layer{L}/stream-{read}-cold",
                    "gpu_layer_stream.py",
                    qb + ["--mode", "stream", "--read", read, "--cache", "cold"],
                )
    return True


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def cmd_report(cfg: Config) -> bool:
    print("=== weight-io report ===")
    root = cfg.root_path
    a = root / "stageA"
    if not (root / "runs").exists():
        print("  no runs; only the inventory tables exist")
        return True
    _run(
        [
            sys.executable,
            SCRIPTS / "aggregate_runs.py",
            "--root",
            root,
            "--out",
            root / "metrics.csv",
        ]
    )
    with open(root / "RESULTS_TABLES.md", "w") as f:
        subprocess.run(
            [sys.executable, str(SCRIPTS / "results_tables.py"), "--root", str(root)],
            stdout=f,
            check=True,
        )
    if (a / "projections.json").exists() and (a / "classes.json").exists():
        _run(
            [
                sys.executable,
                SCRIPTS / "model_projections.py",
                "--medians",
                root / "metrics_medians.csv",
                "--projections",
                a / "projections.json",
                "--classes",
                a / "classes.json",
                "--out",
                root / "model_projections.json",
            ]
        )
    geo = (
        json.loads((a / "device.json").read_text())
        if (a / "device.json").exists()
        else {}
    )
    tools_commit = _git_head(KNLP)
    tracer_commit = _git_head(tracer_dir(cfg)) if have_tracers(cfg) else "unknown"
    _run(
        [
            sys.executable,
            SCRIPTS / "run_manifests.py",
            "--root",
            root,
            "--pins-dir",
            a / "pins",
            "--disk",
            geo.get("disk", "unknown"),
            "--tools-commit",
            tools_commit,
            "--tracer-commit",
            tracer_commit,
        ],
        check=False,
    )
    owner = cfg.owner or getpass.getuser()
    _run(
        [
            sys.executable,
            SCRIPTS / "artifact_manifest.py",
            "--root",
            root,
            "--owner",
            owner,
            "--tools-commit",
            tools_commit,
            "--out",
            root / "ARTIFACT_MANIFEST.json",
        ],
        check=False,
    )
    print(f"  tables: {root / 'RESULTS_TABLES.md'}")
    return True


def cmd_clean(cfg: Config) -> bool:
    root = cfg.root_path
    for p in (
        "runs",
        "metrics.csv",
        "metrics_medians.csv",
        "RESULTS_TABLES.md",
        "model_projections.json",
        "ARTIFACT_MANIFEST.json",
    ):
        t = root / p
        if t.is_dir():
            shutil.rmtree(t)
        elif t.exists():
            t.unlink()
    print(f"  removed runs and tables under {root}; models and inventory kept")
    return True


COMMANDS = {
    "doctor": cmd_doctor,
    "plan": cmd_plan,
    "inventory": cmd_inventory,
    "fetch": cmd_fetch,
    "tracers": cmd_tracers,
    "matrix": cmd_matrix,
    "gpu": cmd_gpu,
    "report": cmd_report,
    "clean": cmd_clean,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("--config", default=".config")
    args = ap.parse_args(argv)
    cfg = Config.from_file(args.config)
    if not cfg.enabled:
        print("CONFIG_KNLP_WEIGHT_IO is not set; select a defconfig-weight-io-* first")
        return 1
    return 0 if COMMANDS[args.command](cfg) else 1


if __name__ == "__main__":
    sys.exit(main())

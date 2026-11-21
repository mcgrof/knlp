# SPDX-License-Identifier: MIT
"""Prompt-selected EMO expert pools served from NVMe: reproduction driver.

Subcommands: doctor, plan, fetch, data, convert, tune, pools, matrix,
pack, features, serve, report, run (all stages in order), clean. Each
reads the kernel-style ``.config`` written by ``make defconfig-emo-ifp``
(or ``defconfig-emo-ifp-smoke``) and drives the scripts in
``scripts/emo_ifp``. Every stage is resumable: a stage whose output
exists is not repeated.

    python3 -m tools.reproduce.emo_ifp.run doctor --config .config
    python3 -m tools.reproduce.emo_ifp.run run --config .config

The stages, in order:

    fetch     the FP32 EMO checkpoint at a pinned revision, and a build of
              the fastsafetensors io_uring DMA-BUF reader
    data      ARC-Challenge and four MMLU categories, at pinned dataset
              revisions; each file is checked against its reference hash
    convert   a bit-checked BF16 copy of the checkpoint and its manifest
    tune      the fused MoE kernel's tile table for this GPU
    pools     five 32-routed-plus-shared expert pools, each from eight
              calibration texts of its domain
    matrix    every pool and the full model scored on every domain
    pack      each pool as aligned per-layer files, the full model the
              same way, and the single-copy layout (one copy per expert)
    features  the mean input embedding of every selector question
    serve     the mixed five-domain stream: a saturated pass and random
              arrivals, each checked bitwise against all pools resident
    report    REPORT.md with the result tables

What each stage measures is documented in the scripts' own docstrings
and on docs/emo-ifp.html.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

KNLP = Path(__file__).resolve().parents[3]
SCRIPTS = KNLP / "scripts" / "emo_ifp"
PREFIX = "CONFIG_KNLP_EMO_IFP_"

# Pool order matters twice: the stream starts on the first pool, and the
# single-copy layout gives the first pool the lowest Gray-code bit.
POOLS = ("arc", "health", "psychology", "history", "politics")
LAYOUT_ORDER = ("politics", "arc", "health", "psychology", "history")
MMLU = ("health", "psychology", "history", "politics")
LAYERS = 16
NUM_EXPERTS = 128

# Reference SHA-256 of the data files the published results used.
DATA_SHA256 = {
    "arc/calibration.json": "952e041585dcd99abdeda7273341a685f77d7dd39913bf85a18679afdc05197f",
    "arc/train.json": "bdccf547f2f3c9ee45fb8284399963a35d2077866fe7f17f4b47cee0948cf419",
    "arc/screen.json": "e407cecb96c665a37227c16c0a2a41dc9cab5b80a71725961670944d73128e13",
    "mmlu-health/calibration.json": "919df8094cdc04514390dd460fd0040ea288ce5ad648a08faa90da59067accdb",
    "mmlu-health/train.json": "40dbd4696c39489e46edb87688fde9628ce52a2d102b3bd7e4035249ed96ab27",
    "mmlu-health/test.json": "f5bb13665327d9fe5bf821156a7476a482574923d9d7983b4aac0d897584a36f",
    "mmlu-psychology/calibration.json": "9cc2d4c42b32e39b430dc31bf228dfd5954e71c2ed621c1c2708ea98f60a9d12",
    "mmlu-psychology/train.json": "5a6afc7f3110d8d19d154891689ebcb48d5305f4a25fc5db39697525c251cd21",
    "mmlu-psychology/test.json": "fe37403e12e54a84855d562b1b45a420567992de8f104111226be31351d4c97b",
    "mmlu-history/calibration.json": "a0f3c4de91b34a279f80c83c5b48acd350eadc6c36fe4d4fcf64cbedd5e09dc8",
    "mmlu-history/train.json": "c64fff9d078c4fc7c00274de8b1eda23d2aa9393705f36f4232594c9e77f9d02",
    "mmlu-history/test.json": "51b315586e2def81c7437b183cdcdd236dcd0d678a8e4947bb7cbeab41a573d1",
    "mmlu-politics/calibration.json": "ca5acc031cb0e44ebe768dc20cab533197a434237cca0eb66a611d7ec46a164a",
    "mmlu-politics/train.json": "55f79ad87a0c341c8b1bc681a6763dc00cc31eed637bd62e9e3c175d80c75b8a",
    "mmlu-politics/test.json": "d2880672708ed731a9c6d610c273599134c201999f21e6cad0111ca2de68b959",
}

# Reference SHA-256 of each pool's expert membership (the routed and
# shared lists of every layer, as sorted JSON). Calibration runs on the
# GPU, so a near tie in router probability may flip on other hardware;
# a differing pool is reported, not refused.
POOL_SHA256 = {
    "arc": "f491b2bd8c161febc6774a5a85922d74998a5578afff08af38a57199f9952a95",
    "health": "bd85934e30dccd079704d04a6beaecfa4fd4d41493247cdffc2e143eb608ad30",
    "psychology": "d7aa5ceceb90aa23a91f615fc06b19036a127354adb506c83d2dee0fd906b442",
    "history": "b9b5d377d9ff50c6e115abd40c3caeaed60253813ed7aa9f17b2cc61310a0d8e",
    "politics": "f8dc6cc29d7ac3b7072db6360dce4c078b6dbbf5cafc0d50d9b996912686cf13",
}


def _as_bool(v) -> bool:
    return str(v).strip().lower() in ("y", "1", "true", "yes")


@dataclass
class Config:
    enabled: bool = False
    root: str = "./results/emo-ifp"
    python: str = "python3"
    model_repo: str = "allenai/Emo_1b14b_1T"
    model_revision: str = "007852b1e3d22222f8ee03c6a53fa30e47898dcb"
    model_fp32_dir: str = ""
    model_bf16_dir: str = ""
    arc_revision: str = "210d026faf9955653af8916fad021475a3f00453"
    mmlu_revision: str = "c30699e8356da336a370243923dbaf21066bb9fe"
    fst_repo: str = "https://github.com/mcgrof/fastsafetensors.git"
    fst_ref: str = "a174cd23e4265b7241359586b2a0169c7cfaa6f9"
    fst_dir: str = ""
    tuned_configs: str = ""
    routed: int = 32
    run_length: int = 16
    lookahead: int = 4
    selector: str = "emb-logreg"
    n_train: int = 32
    arrival_rule: str = "queue"
    arrival_loads: list = field(default_factory=lambda: ["0.3", "0.6", "0.9"])
    confident_precision: str = "0.85"
    limit: int = 0

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
        c = cls(enabled=_as_bool(raw.get("CONFIG_KNLP_EMO_IFP", "n")))
        for name, f_ in cls.__dataclass_fields__.items():
            if name == "enabled":
                continue
            key = PREFIX + name.upper()
            if key not in raw:
                continue
            v = raw[key]
            if f_.type == "int":
                v = int(v)
            elif name == "arrival_loads":
                v = v.split()
            setattr(c, name, v)
        return c


class Paths:
    def __init__(self, cfg: Config):
        self.root = Path(cfg.root).expanduser().resolve()
        self.fp32 = (
            Path(cfg.model_fp32_dir).expanduser()
            if cfg.model_fp32_dir
            else self.root / "models" / "fp32"
        )
        self.bf16 = (
            Path(cfg.model_bf16_dir).expanduser()
            if cfg.model_bf16_dir
            else self.root / "models" / "bf16"
        )
        self.manifest = self.root / "models" / "bf16-manifest.json"
        self.fst = (
            Path(cfg.fst_dir).expanduser()
            if cfg.fst_dir
            else self.root / "src" / "fastsafetensors"
        )
        self.data = self.root / "data"
        self.pools = self.root / "pools"
        self.tuned = (
            Path(cfg.tuned_configs).expanduser()
            if cfg.tuned_configs
            else self.root / "tuned-configs.json"
        )
        self.matrix = self.root / "matrix"
        self.packed = self.root / "packed"
        self.single = self.packed / "single-copy"
        self.features = self.root / "features.pt"
        # A smoke (LIMIT set) keeps its own results, so a later full run
        # under the same root does not mistake them for its own.
        tag = f"-limit{cfg.limit}" if cfg.limit else ""
        self.serve = self.root / f"serve{tag}"
        self.report = self.root / f"REPORT{tag}.md"

    def calibration(self, d):
        sub = "arc" if d == "arc" else f"mmlu-{d}"
        return self.data / sub / "calibration.json"

    def train(self, d):
        return self.data / ("arc" if d == "arc" else f"mmlu-{d}") / "train.json"

    def test(self, d):
        sub = "arc" if d == "arc" else f"mmlu-{d}"
        return self.data / sub / ("screen.json" if d == "arc" else "test.json")

    def pool(self, d):
        return self.pools / f"pool-{d}-r32.json"


def env_for(cfg: Config, p: Paths) -> dict:
    env = dict(os.environ)
    env.update(
        HF_HUB_OFFLINE=env.get("HF_HUB_OFFLINE", "0"),
        TOKENIZERS_PARALLELISM="false",
        TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="1",
        HSA_ENABLE_IPC_MODE_LEGACY="0",
        FASTSAFETENSORS_DMABUF_STRICT="1",
        TRITON_CACHE_DIR=str(p.root / "triton-cache"),
    )
    # fastsafetensors reads its own package metadata at import, so an
    # in-place build needs its egg-info (or a dist-info under meta/).
    paths = [str(p.fst)] + [str(p.fst / "meta")] * (p.fst / "meta").is_dir()
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def sh(cfg, p, args, log=None, cwd=None):
    """Run one command, echo it, tee its output to a log file."""
    args = [str(a) for a in args]
    print("+ " + " ".join(args), flush=True)
    if log is None:
        subprocess.run(args, check=True, env=env_for(cfg, p), cwd=cwd)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        proc = subprocess.Popen(
            args,
            env=env_for(cfg, p),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        if proc.wait():
            sys.exit(f"command failed ({proc.returncode}); log: {log}")


def py(cfg, p, script, *args, log=None):
    sh(cfg, p, [cfg.python, SCRIPTS / script, *args], log=log)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def membership_sha(pool_json: Path) -> str:
    layers = json.loads(pool_json.read_text())["layers"]
    s = {k: {"routed": v["routed"], "shared": v["shared"]} for k, v in layers.items()}
    return hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest()


# Stages -------------------------------------------------------------------


def doctor(cfg, p):
    ok = True
    if not cfg.enabled:
        print("CONFIG_KNLP_EMO_IFP is not set; run make defconfig-emo-ifp")
        return False
    probe = (
        "import torch, triton, transformers, safetensors, datasets;"
        "print(torch.__version__, torch.version.hip or 'cuda', "
        "torch.cuda.is_available() and torch.cuda.get_device_name(0), "
        "transformers.__version__)"
    )
    r = subprocess.run(
        [cfg.python, "-c", probe], capture_output=True, text=True, env=env_for(cfg, p)
    )
    if r.returncode:
        print("python environment: FAIL\n" + r.stderr.strip().splitlines()[-1])
        ok = False
    else:
        out = r.stdout.strip()
        print(f"python environment: {out}")
        if " False " in f" {out} ":
            print("  no GPU is visible to torch")
            ok = False
        elif "cuda" in out.split()[1]:
            print(
                "  note: the published results were measured on an AMD GPU with "
                "ROCm; the slot store needs a GPU whose memory can be exported "
                "as a DMA-BUF and read into by the NVMe driver"
            )
    try:
        rel = os.uname().release
        print(f"kernel: {rel}")
    except OSError:
        pass
    print(
        "  the slot store reads NVMe straight into GPU memory through io_uring "
        "DMA-BUF registration; see docs/io-uring-dmabuf.html for the kernel"
    )
    p.root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(p.root).free / 1e9
    need = 0 if p.single.exists() else 230
    print(f"free space under {p.root}: {free:.0f} GB (about {need} GB needed)")
    if free < need:
        ok = False
    print("doctor: " + ("OK" if ok else "FAIL"))
    return ok


def plan(cfg, p):
    print(f"root           {p.root}")
    print(f"model          {cfg.model_repo}@{cfg.model_revision}")
    print(f"fp32 / bf16    {p.fp32} / {p.bf16}")
    print(f"reader         {cfg.fst_repo}@{cfg.fst_ref} -> {p.fst}")
    print(f"pools          {' '.join(POOLS)} ({cfg.routed} routed + shared)")
    print(
        f"stream         R={cfg.run_length} Q={cfg.lookahead} selector "
        f"{cfg.selector}@{cfg.n_train}"
    )
    print(
        f"arrivals       rule {cfg.arrival_rule}, loads "
        f"{' '.join(cfg.arrival_loads)}, confident precision "
        f"{cfg.confident_precision}"
    )
    print(f"limit          {cfg.limit or 'all 640 requests'}")
    for name, done in STAGE_DONE.items():
        print(f"  {name:9s} {'done' if done(cfg, p) else 'to do'}")


def fetch(cfg, p):
    if not cfg.model_bf16_dir and not (p.fp32 / "config.json").exists():
        code = (
            "import sys; from huggingface_hub import snapshot_download as s;"
            "s(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])"
        )
        sh(cfg, p, [cfg.python, "-c", code, cfg.model_repo, cfg.model_revision, p.fp32])
    if not cfg.fst_dir and not (p.fst / "fastsafetensors").exists():
        p.fst.parent.mkdir(parents=True, exist_ok=True)
        sh(cfg, p, ["git", "clone", cfg.fst_repo, p.fst])
        sh(cfg, p, ["git", "-C", p.fst, "checkout", "--detach", cfg.fst_ref])
    if not list((p.fst / "fastsafetensors").glob("cpp*.so")):
        sh(cfg, p, [cfg.python, "setup.py", "build_ext", "--inplace"], cwd=p.fst)
    meta = list(p.fst.glob("*.egg-info")) + list(p.fst.glob("meta/*.dist-info"))
    if not meta:
        sh(cfg, p, [cfg.python, "setup.py", "egg_info"], cwd=p.fst)
    sh(
        cfg,
        p,
        [
            cfg.python,
            "-c",
            "import fastsafetensors.io_uring_dmabuf as m; print(m.__file__)",
        ],
    )


def data(cfg, p):
    tok = p.bf16 if (p.bf16 / "tokenizer.json").exists() else p.fp32
    if not (p.data / "arc" / "screen.json").exists():
        py(
            cfg,
            p,
            "arc_data.py",
            p.data / "arc",
            "--revision",
            cfg.arc_revision,
            "--n-screen",
            128,
            "--n-train",
            128,
        )
    missing = [d for d in MMLU if not p.test(d).exists()]
    if missing:
        doms = [x for d in missing for x in ("--domain", d)]
        py(cfg, p, "mmlu_data.py", tok, p.data, "--revision", cfg.mmlu_revision, *doms)
    bad = []
    for rel, ref in DATA_SHA256.items():
        got = sha256(p.data / rel)
        if got != ref:
            bad.append(rel)
            print(f"MISMATCH {rel}: {got}")
    if bad:
        sys.exit("data files differ from the published ones; see above")
    print(f"data: all {len(DATA_SHA256)} files match the published hashes")


def convert(cfg, p):
    if cfg.model_bf16_dir:
        if not p.manifest.exists():
            sys.exit(
                "MODEL_BF16_DIR is set but its manifest is missing at "
                f"{p.manifest}; copy the convert_bf16.py manifest there"
            )
        return
    if not p.manifest.exists():
        py(
            cfg,
            p,
            "convert_bf16.py",
            p.fp32,
            p.bf16,
            p.manifest,
            log=p.root / "logs" / "convert.log",
        )


def tune(cfg, p):
    if not p.tuned.exists():
        py(cfg, p, "tune_fused_moe.py", p.tuned, log=p.root / "logs" / "tune.log")


def pools(cfg, p):
    p.pools.mkdir(parents=True, exist_ok=True)
    for d in POOLS:
        if not p.pool(d).exists():
            py(
                cfg,
                p,
                "calibrate_pool.py",
                p.bf16,
                p.calibration(d),
                p.pool(d),
                "--routed",
                cfg.routed,
                log=p.root / "logs" / f"calibrate-{d}.log",
            )
        got = membership_sha(p.pool(d))
        same = got == POOL_SHA256[d] if cfg.routed == 32 else None
        print(
            f"pool {d}: membership {got[:16]} "
            f"{'matches' if same else 'DIFFERS from'} the published pool"
        )


def domain_args(p, with_train=False):
    out = []
    for d in POOLS:
        out += ["--domain", d] + ([p.train(d)] if with_train else []) + [p.test(d)]
    return out


def matrix(cfg, p):
    if (p.matrix / "pool-matrix.json").exists():
        return
    pa = [x for d in POOLS for x in ("--pool", d, p.pool(d))]
    py(
        cfg,
        p,
        "pool_matrix.py",
        p.bf16,
        p.matrix,
        "--configs",
        p.tuned,
        *domain_args(p),
        *pa,
        log=p.root / "logs" / "pool-matrix.log",
    )


def pack_dir(cfg, p, pool_json, out):
    out.mkdir(parents=True, exist_ok=True)
    for li in range(LAYERS):
        f = out / f"layer{li}.safetensors"
        if Path(str(f) + ".json").exists():
            continue
        py(
            cfg,
            p,
            "pack_pool_layer.py",
            p.bf16,
            pool_json,
            p.manifest,
            f,
            "--layer",
            li,
            "--stacked",
            "--align",
            4096,
            log=out / f"layer{li}.log",
        )


def pack(cfg, p):
    full = p.pools / "pool-full.json"
    if not full.exists():
        py(cfg, p, "full_pool.py", p.pool("arc"), NUM_EXPERTS, full)
    pack_dir(cfg, p, full, p.packed / "full")
    for d in POOLS:
        pack_dir(cfg, p, p.pool(d), p.packed / d)
    if not (p.single / "LAYOUT.json").exists():
        pa = [f"{d}={p.pool(d)}" for d in LAYOUT_ORDER]
        py(
            cfg,
            p,
            "pack_single_copy.py",
            p.packed / "full",
            p.single,
            *pa,
            log=p.root / "logs" / "pack-single-copy.log",
        )


def features(cfg, p):
    if not p.features.exists():
        py(
            cfg,
            p,
            "selector_features.py",
            p.bf16,
            p.features,
            "--configs",
            p.tuned,
            *domain_args(p, with_train=True),
            "--depths",
            4,
            log=p.root / "logs" / "features.log",
        )


def serve(cfg, p):
    if (p.serve / "mixed-stream.json").exists():
        return
    pa = [x for d in POOLS for x in ("--pool", d, p.pool(d), p.packed / d)]
    extra = ["--limit", cfg.limit] if cfg.limit else []
    if cfg.arrival_loads:
        extra += [
            "--arrival-loads",
            *cfg.arrival_loads,
            "--arrival-rule",
            cfg.arrival_rule,
            "--confident-precision",
            cfg.confident_precision,
        ]
    py(
        cfg,
        p,
        "mixed_stream.py",
        p.bf16,
        p.features,
        p.matrix / "pool-matrix.json",
        p.serve,
        "--configs",
        p.tuned,
        *pa,
        "--single-dir",
        p.single,
        *domain_args(p),
        "--run-lengths",
        cfg.run_length,
        "--lookahead",
        cfg.lookahead,
        "--modes",
        "slots",
        "--policies",
        "debounce",
        "--gc",
        "none",
        "--selector",
        cfg.selector,
        "--n-train",
        cfg.n_train,
        *extra,
        log=p.root / "logs" / "serve.log",
    )


def report(cfg, p):
    res = json.loads((p.serve / "mixed-stream.json").read_text())
    lines = ["# EMO expert pools served from NVMe: reproduction report", ""]
    lines.append(f"Runtime: `{json.dumps(res.get('runtime', {}), sort_keys=True)}`")
    lines.append("")
    for R, st in res["streams"].items():
        summ = st.get("summary", {}).get("debounce", {})
        lines += [
            f"## Saturated stream, mean run length {R}",
            "",
            "| Pass | Throughput loss | Switches | Correct | Keeps up |",
            "| --- | ---: | ---: | --- | --- |",
        ]
        for k, v in summ.items():
            if isinstance(v, dict) and "throughput_loss" in v:
                lines.append(
                    f"| {k} | {100 * v['throughput_loss']:.2f}% | "
                    f"{v['switched_n']} | {v['correct']} | {v['keeps_up']} |"
                )
        lines.append("")
        arr = st.get("arrivals", {}).get("loads", {})
        if not arr:
            continue
        lines += [
            f"## Random arrivals, rule {cfg.arrival_rule}",
            "",
            "| Load | Median ratio | p99 ratio | Switches | Served | "
            "Correct | Holds up |",
            "| ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
        for rho, a in arr.items():
            s = a["summary"]
            sw, rs = s["switch"]["all"], s["resident"]["all"]
            m = sw["latency_ms_median"] / rs["latency_ms_median"]
            q = sw["latency_ms_p99"] / rs["latency_ms_p99"]
            served = sum(v["correct"] for v in s["quality"].values())
            ok = s["correct"] and s["pick_mismatches"] == 0
            lines.append(
                f"| {rho} | {m:.2f}x | {q:.2f}x | {s['switches']} | "
                f"{served} | {ok} | {ok and m <= 1.5 and q <= 2.0} |"
            )
        lines += [
            "",
            "Bars: median at most 1.5x and p99 at most 2x the "
            "server with every pool resident, on the same arrivals.",
            "",
        ]
    out = p.report
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {out}")


def clean(cfg, p):
    for x in (p.serve, p.report):
        if x.is_dir():
            shutil.rmtree(x)
        elif x.exists():
            x.unlink()
    print("removed the serve results; models, pools and packed files stay")


STAGES = {
    "fetch": fetch,
    "data": data,
    "convert": convert,
    "tune": tune,
    "pools": pools,
    "matrix": matrix,
    "pack": pack,
    "features": features,
    "serve": serve,
    "report": report,
}

STAGE_DONE = {
    "fetch": lambda c, p: (p.fst / "fastsafetensors").exists()
    and (bool(c.model_bf16_dir) or (p.fp32 / "config.json").exists()),
    "data": lambda c, p: all((p.data / r).exists() for r in DATA_SHA256),
    "convert": lambda c, p: p.manifest.exists(),
    "tune": lambda c, p: p.tuned.exists(),
    "pools": lambda c, p: all(p.pool(d).exists() for d in POOLS),
    "matrix": lambda c, p: (p.matrix / "pool-matrix.json").exists(),
    "pack": lambda c, p: (p.single / "LAYOUT.json").exists(),
    "features": lambda c, p: p.features.exists(),
    "serve": lambda c, p: (p.serve / "mixed-stream.json").exists(),
    "report": lambda c, p: p.report.exists(),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("stage", choices=["doctor", "plan", "run", "clean", *STAGES])
    ap.add_argument("--config", default=".config")
    args = ap.parse_args()
    cfg = Config.from_file(args.config)
    p = Paths(cfg)
    if args.stage == "doctor":
        sys.exit(0 if doctor(cfg, p) else 1)
    if not cfg.enabled:
        sys.exit("CONFIG_KNLP_EMO_IFP is not set; run make defconfig-emo-ifp")
    if args.stage == "plan":
        plan(cfg, p)
    elif args.stage == "clean":
        clean(cfg, p)
    elif args.stage == "run":
        if not doctor(cfg, p):
            sys.exit(1)
        for fn in STAGES.values():
            fn(cfg, p)
    else:
        STAGES[args.stage](cfg, p)


if __name__ == "__main__":
    main()

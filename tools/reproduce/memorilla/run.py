# SPDX-License-Identifier: MIT
"""Fetch pinned evaluation inputs and run checkpoint inference, without training."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from .workloads import compare

AUTHOR = "005854d13ced2dfe56588fb721eb78db5ec13f6a"
READER = "f6b15af56aa687861c8d2d03b1cd8f7979881361"
CHECKPOINT_RELEASE_LABELS = {
    "stage3": "Stage 3 (epoch-00), continued from Stage 2",
    "pmv2": "PersonaMem-v2, fine-tuned from Stage 1",
    "pv4": "PersonalizationV4, fine-tuned from Stage 1",
}
DATA = "c6b0d7eea65659254115ed31dd26869eb6dc0586"
DECODER = "b968826d9c46dd6066d109eabc6255188de91218"
ENCODER = "5cf2132abc99cad020ac570b19d031efec650f2b"
VLLM = "198199bc44da513a6eceee206cec9c62d55d190a"
LMCACHE = "e438be9d81ce8b76b135b813a107d9b189abdd33"
LMCACHE_MAPPED = "b520066121268be0d8354e02977a490862f17984"
BENCHMARKS = {
    "pubmedqa": (100, "accuracy", 59.00),
    "factkg": (9041, "accuracy", 84.33),
    "triviaqa": (500, "exact_match", 57.00),
    "narrativeqa": (10557, "rougeL", 19.31),
    "lamp4": (1925, "rougeL", 18.15),
    "lamp7": (1500, "rougeL", 48.37),
    "pmv2": (5000, "accuracy", 44.60),
    "pv4": (4237, "accuracy", 66.65),
}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def configuration(path, *, resolve_artifact_sets=True):
    cfg = {}
    for line in path.read_text().splitlines():
        if line.startswith("CONFIG_KNLP_MEM_") and "=" in line:
            key, value = line.split("=", 1)
            cfg[key.removeprefix("CONFIG_KNLP_MEM_").lower()] = (
                json.loads(value) if value.startswith('"') else value
            )
    profiles = [p for p in ("author", "port_control", "lmcache") if cfg.get(p) == "y"]
    if (
        len(profiles) != 1
        or "CONFIG_KNLP_REPRODUCE_MEMORILLA=y" not in path.read_text().splitlines()
    ):
        raise ValueError("select one Memorilla defconfig and run olddefconfig")
    cfg["profile"] = profiles[0]
    cfg["root"] = str(Path(cfg.get("root", "build/memorilla")).expanduser().resolve())
    cfg["benchmarks"] = (
        list(BENCHMARKS)
        if cfg.get("benchmark", "pubmedqa") == "all"
        else cfg.get("benchmark", "pubmedqa").split(",")
    )
    if not set(cfg["benchmarks"]) <= BENCHMARKS.keys():
        raise ValueError("unknown benchmark")
    for key, default in [
        ("batch", 32),
        ("pool_bytes", 67108864),
        ("requests", 1000),
        ("clients", 1),
        ("seed", 23),
        ("rows", 0),
        ("evict_every", 0),
    ]:
        cfg[key] = int(cfg.get(key, default))
    for key, default in [("gpu_fraction", 0.6), ("scorer_fraction", 0.3)]:
        cfg[key] = float(cfg.get(key, default))
        if not 0 < cfg[key] < 1:
            raise ValueError(f"invalid {key}")
    cfg.setdefault("pattern", "shuffle")
    cfg.setdefault("input_pipeline", "author")
    if cfg["input_pipeline"] not in {"author", "leased-fp16", "leased-bf16"}:
        raise ValueError("unknown document input pipeline")
    if cfg["profile"] == "author" and cfg["input_pipeline"] != "author":
        raise ValueError("original author mode rejects collator overrides")
    cfg.setdefault("document_prefetch", "n")
    if cfg["document_prefetch"] not in {"y", "n"}:
        raise ValueError("document prefetch must be y or n")
    if cfg["document_prefetch"] == "y" and cfg["input_pipeline"] != "leased-bf16":
        raise ValueError("document prefetch requires leased-bf16 inputs")
    cfg.setdefault("artifact_mode", "import")
    cfg.setdefault("artifact_backing", "arena")
    cfg.setdefault("artifact_verification", "on-map")
    cfg["max_mappings"] = int(cfg.get("max_mappings", 512))
    if cfg["artifact_backing"] not in {"arena", "mmap"}:
        raise ValueError("unknown artifact backing")
    if cfg["artifact_verification"] not in {"on-map", "published"}:
        raise ValueError("unknown artifact verification")
    if cfg["max_mappings"] < 1:
        raise ValueError("max mappings must be positive")
    if cfg["artifact_backing"] == "arena" and cfg["artifact_verification"] != "on-map":
        raise ValueError("arena backing requires load verification")
    if cfg["artifact_backing"] == "mmap" and cfg["profile"] != "lmcache":
        raise ValueError("mapped artifact backing requires the LMCache profile")
    if cfg["artifact_mode"] not in {"import", "attach"}:
        raise ValueError("unknown artifact preparation mode")
    if cfg["artifact_mode"] == "attach":
        if cfg["profile"] != "lmcache":
            raise ValueError("artifact attach requires the LMCache profile")
    if cfg["artifact_mode"] == "attach" and resolve_artifact_sets:
        from tools.reproduce.context_artifacts.artifacts import (
            digest as validate_digest,
        )

        receipt_path = Path(
            cfg.get("artifact_set_manifest")
            or (Path(cfg["root"]) / "artifact-sets.json")
        )
        if not receipt_path.is_file():
            raise ValueError("attach requires a manifest from memorilla-import")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("schema") != 1 or not isinstance(receipt.get("sets"), dict):
            raise ValueError("invalid imported artifact-set manifest")
        cfg["artifact_sets"] = receipt["sets"]
        for benchmark in cfg["benchmarks"]:
            validate_digest(cfg["artifact_sets"].get(benchmark), "artifact set ID")
    cfg["engine_options"] = json.loads(cfg.get("engine_json", "{}"))
    allowed = {
        "enforce_eager",
        "attention_config",
        "max_num_seqs",
        "enable_prefix_caching",
    }
    if (
        not isinstance(cfg["engine_options"], dict)
        or set(cfg["engine_options"]) - allowed
    ):
        raise ValueError(
            "unsupported engine option; sampling/model overrides are forbidden"
        )
    if cfg["profile"] == "author" and cfg["engine_options"]:
        raise ValueError("original author mode rejects engine overrides")
    cfg.setdefault(
        "scheduling", "runtime" if cfg["profile"] == "author" else "inprocess"
    )
    if cfg["scheduling"] not in {"runtime", "inprocess", "batch"}:
        raise ValueError("scheduling must be runtime, inprocess or batch")
    if cfg["profile"] == "author" and cfg["scheduling"] != "runtime":
        raise ValueError("original author mode rejects scheduling overrides")
    if cfg["profile"] == "author" and cfg["clients"] != 1:
        raise ValueError("original author mode rejects concurrent bank clients")
    if cfg.get("singleton_reference") == "y" and (
        cfg["batch"] != 1
        or cfg["scheduling"] != "inprocess"
        or cfg["engine_options"].get("enable_prefix_caching") is not False
        or cfg["engine_options"].get("max_num_seqs") != 1
    ):
        raise ValueError(
            "singleton reference requires batch=1, inprocess scheduling, "
            "max_num_seqs=1 and enable_prefix_caching=false"
        )
    if (
        min(cfg["batch"], cfg["pool_bytes"], cfg["requests"], cfg["clients"]) < 1
        or cfg["clients"] > 128
        or min(cfg["rows"], cfg["evict_every"]) < 0
    ):
        raise ValueError("invalid workload bounds")
    return cfg


def source_check(path):
    revision = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != AUTHOR:
        raise ValueError("author source revision mismatch")
    subprocess.run(
        ["git", "-C", str(path), "diff", "--exit-code", "HEAD", "--"],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def fetch(cfg):
    from huggingface_hub import snapshot_download

    root = Path(cfg["root"])
    source = root / "source/memorilla-author"
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/snap-stanford/memorilla.git",
                str(source),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(source), "checkout", "--detach", AUTHOR], check=True
        )
    source_check(source)
    if cfg["profile"] != "author":
        lmcache_revision = (
            LMCACHE_MAPPED if cfg.get("artifact_backing") == "mmap" else LMCACHE
        )
        for name, revision in [("vllm", VLLM), ("LMCache", lmcache_revision)]:
            target = root / "source" / name.lower()
            if not target.exists():
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--single-branch",
                        "--branch",
                        (
                            "2026-10-28-feature-mapped-artifacts"
                            if name == "LMCache"
                            and cfg.get("artifact_backing") == "mmap"
                            else "2026-10-26-feature-context-artifacts"
                        ),
                        f"https://github.com/mcgrof/{name}.git",
                        str(target),
                    ],
                    check=True,
                )
                subprocess.run(
                    ["git", "-C", str(target), "checkout", "--detach", revision],
                    check=True,
                )
    hub = root / "hf/hub"
    data_patterns = []
    for name in cfg["benchmarks"]:
        data_patterns.extend(
            [f"data/{name}/test-*", f"embeddings/qwen3-embedding-4b/{name}/test/*"]
        )
    specs = [
        (
            "reader",
            "memorilla/Memorilla-Qwen3-8B",
            READER,
            None,
            ["stage3/*", "pmv2/*", "pv4/*"],
        ),
        ("data", "memorilla/Memorilla-Data", DATA, "dataset", data_patterns),
        (
            "decoder",
            "Qwen/Qwen3-8B",
            DECODER,
            None,
            ["*.json", "*.safetensors", "*.txt", "*.jinja"],
        ),
    ]
    if set(cfg["benchmarks"]) & {"pmv2", "pv4", "factkg"}:
        specs.append(
            (
                "encoder",
                "Qwen/Qwen3-Embedding-4B",
                ENCODER,
                None,
                ["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"],
            )
        )
    inputs = {
        "source": str(source),
        "source_revision": AUTHOR,
        "assets": {},
        "benchmarks": cfg["benchmarks"],
    }
    for key, repo, rev, kind, patterns in specs:
        path = Path(
            snapshot_download(
                repo,
                revision=rev,
                repo_type=kind,
                cache_dir=str(hub),
                allow_patterns=patterns,
            )
        )
        if kind is None:
            ref = path.parent.parent / "refs/main"
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text(rev)
        files = {
            str(p.relative_to(path)): {"size": p.stat().st_size, "sha256": digest(p)}
            for p in path.rglob("*")
            if p.is_file()
        }
        inputs["assets"][key] = {
            "path": str(path),
            "repo": repo,
            "revision": rev,
            "files": files,
        }
    write(root / "inputs.json", inputs)
    return inputs


def doctor(cfg):
    root = Path(cfg["root"])
    inputs = json.loads((root / "inputs.json").read_text())
    source_check(Path(inputs["source"]))
    if not set(cfg["benchmarks"]) <= set(inputs["benchmarks"]):
        raise ValueError("run fetch for the selected benchmarks")
    expected = {"reader": READER, "data": DATA, "decoder": DECODER, "encoder": ENCODER}
    for name, asset in inputs["assets"].items():
        if asset["revision"] != expected[name] or not asset["files"]:
            raise ValueError(f"invalid input pin: {name}")
        for relative, record in asset["files"].items():
            p = Path(asset["path"]) / relative
            if p.stat().st_size != record["size"] or digest(p) != record["sha256"]:
                raise ValueError(f"input integrity failure: {p}")
    versions = {
        name: metadata.version(name)
        for name in ["torch", "vllm", "transformers", "datasets", "safetensors"]
    }
    if cfg["profile"] == "author":
        for name, want in {
            "torch": "2.7.0",
            "vllm": "0.9.1",
            "transformers": "4.53.3",
            "datasets": "4.1.1",
        }.items():
            if versions[name].split("+")[0] != want:
                raise ValueError(
                    f"author runtime requires {name}=={want}, observed {versions[name]}; use port-control to qualify a different runtime"
                )
    return {"inputs": inputs, "versions": versions, "profile": cfg["profile"]}


def execute(cfg, command):
    checked = doctor(cfg)
    root = Path(cfg["root"])
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    out = root / "runs" / f"{stamp}-{cfg['profile']}-{command}"
    out.mkdir(parents=True)
    write(out / "resolved.json", cfg)
    write(out / "inputs.json", checked)
    write(out / "input-spec.json", checked["inputs"])
    summary = []
    for name in cfg["benchmarks"]:
        case = out / name
        case.mkdir()
        argv = [
            sys.executable,
            "-m",
            "tools.reproduce.memorilla.worker",
            "--config",
            str(out / "resolved.json"),
            "--inputs",
            str(out / "input-spec.json"),
            "--benchmark",
            name,
            "--output",
            str(case),
            "--mode",
            command,
        ]
        start = time.monotonic()
        with (case / "stdout.log").open("w") as stdout, (case / "stderr.log").open(
            "w"
        ) as stderr:
            result = subprocess.run(argv, stdout=stdout, stderr=stderr, check=False)
        record = {
            "benchmark": name,
            "argv": argv,
            "returncode": result.returncode,
            "elapsed_seconds": time.monotonic() - start,
        }
        summary.append(record)
        write(out / "summary.json", summary)
        print(json.dumps(record), flush=True)
        if result.returncode:
            raise RuntimeError(f"worker failed; inspect {case}")
    return {"run": str(out), "cases": summary}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "command", choices=["fetch", "doctor", "benchmark", "stress", "compare"]
    )
    p.add_argument("--config", type=Path)
    p.add_argument("--reference", type=Path)
    p.add_argument("--observed", type=Path)
    p.add_argument("--trace", type=Path)
    args = p.parse_args()
    try:
        if args.command == "compare":

            def records(path):
                # JSON strings can contain literal Unicode line separators.
                # JSON Lines uses physical newlines between records.
                with path.open() as stream:
                    return [json.loads(line) for line in stream]

            result = compare(records(args.reference), records(args.observed))
            print(json.dumps(result, indent=2))
            return 0 if result["exact"] else 1
        if args.config is None:
            p.error("--config is required for this command")
        cfg = configuration(args.config)
        if args.trace:
            cfg["trace"] = str(args.trace.resolve(strict=True))
            if len(cfg["benchmarks"]) != 1:
                raise ValueError("a saved trace selects exactly one benchmark")
        result = (
            fetch(cfg)
            if args.command == "fetch"
            else doctor(cfg) if args.command == "doctor" else execute(cfg, args.command)
        )
        print(json.dumps(result, indent=2))
        return 0
    except (
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
        subprocess.CalledProcessError,
        metadata.PackageNotFoundError,
    ) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

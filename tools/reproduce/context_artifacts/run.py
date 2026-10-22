# SPDX-License-Identifier: MIT
"""Context artifact import and byte-restore tools. Inference is a separate gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from .artifacts import (
    ArtifactError,
    ConsumerIdentity,
    ContextArtifactManifest,
    ContextArtifactRegistry,
    canonical,
)


class Blocked(ArtifactError):
    pass


def read_config(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text().splitlines():
        if line.startswith("CONFIG_") and "=" in line:
            key, value = line.split("=", 1)
            result[key.removeprefix("CONFIG_")] = (
                json.loads(value) if value.startswith('"') else value
            )
    if result.get("KNLP_CONTEXT_ARTIFACTS") != "y":
        raise ArtifactError("enable CONFIG_KNLP_CONTEXT_ARTIFACTS")
    return result


def persistent(path: Path) -> dict:
    """Check the resolved filesystem, including existing parents of new paths."""
    path = path.expanduser().resolve()
    parent = path
    while not parent.exists():
        parent = parent.parent
    entry = json.loads(
        subprocess.check_output(
            [
                "findmnt",
                "--json",
                "--target",
                str(parent),
                "--output",
                "TARGET,SOURCE,FSTYPE",
            ],
            text=True,
        )
    )["filesystems"][0]
    if entry["fstype"] in {"tmpfs", "ramfs", "overlay", "fuse.sshfs", "nfs", "nfs4"}:
        raise ArtifactError(
            f"local persistent backing is unverified: {path} ({entry['fstype']})"
        )
    return {"path": str(path), "mount": entry}


def source_check(lock: dict) -> dict:
    observed = {}
    for name, spec in lock["sources"].items():
        path = Path(spec["path"]).resolve()
        persistent(path)
        actual = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != spec["revision"]:
            raise ArtifactError(f"source revision mismatch: {name}")
        dirty = subprocess.run(
            ["git", "-C", str(path), "diff", "--quiet", "HEAD", "--"], check=False
        )
        if dirty.returncode:
            raise ArtifactError(f"source has tracked modifications: {name}")
        for option in ("--absolute-git-dir", "--git-common-dir"):
            git_dir = Path(
                subprocess.check_output(
                    ["git", "-C", str(path), "rev-parse", option], text=True
                ).strip()
            )
            if not git_dir.is_absolute():
                git_dir = path / git_dir
            persistent(git_dir)
            alternates = git_dir / "objects/info/alternates"
            if alternates.exists():
                for line in alternates.read_text().splitlines():
                    alternate = Path(line)
                    alternate = (
                        alternate
                        if alternate.is_absolute()
                        else git_dir / "objects" / alternate
                    )
                    if not alternate.is_dir():
                        raise ArtifactError(
                            f"missing Git object alternate: {alternate}"
                        )
                    persistent(alternate)
        observed[name] = {"revision": actual, "path": str(path)}
    return observed


def selected(config: dict, prefix: str) -> str:
    values = [
        k.removeprefix(prefix).lower().replace("_", "-")
        for k, v in config.items()
        if k.startswith(prefix) and v == "y"
    ]
    if len(values) != 1:
        raise ArtifactError(f"select exactly one {prefix} option")
    return values[0]


def run(args, config: dict) -> dict:
    def get(key):
        return config.get("KNLP_CA_" + key, "")

    backend = selected(config, "KNLP_CA_BACKEND_")
    profile = selected(config, "KNLP_CA_PROFILE_")
    method = selected(config, "KNLP_CA_METHOD_")
    operation = selected(config, "KNLP_CA_OP_")
    representation = selected(config, "KNLP_CA_REP_").replace("-", "_")
    if method == "lpc" and representation != "prefix_kv":
        raise ArtifactError("LPC requires prefix_kv")
    if method == "memorilla" and representation == "prefix_kv":
        raise ArtifactError("Memorilla prefix-KV conversion is not implemented")
    evidence = {
        "profile": profile,
        "method": method,
        "backend": backend,
        "operation": operation,
        "representation": representation,
        "inference_tested": False,
        "interface_pass": False,
    }
    if args.command == "report":
        return evidence | {
            "status": "INCOMPLETE",
            "detail": "No inference or LMCache acceptance result has been recorded.",
        }
    for key in ("WORK_ROOT", "BUILD_ROOT", "STORE_ROOT", "RESULTS_ROOT"):
        if not get(key):
            raise Blocked(f"configure CONFIG_KNLP_CA_{key}")
        evidence[key.lower()] = persistent(Path(get(key)))
    lock_path = Path(get("SOURCE_LOCK"))
    if not lock_path.is_file():
        raise Blocked(
            "SOURCE_LOCK.json is missing; recover and pin the actual baselines"
        )
    lock = json.loads(lock_path.read_text())
    runner_root = Path(__file__).resolve().parents[3]
    knlp_source = lock.get("sources", {}).get("knlp", {})
    if Path(knlp_source.get("path", "")).resolve() != runner_root:
        raise Blocked("runner checkout does not match the source lock")
    evidence["sources"] = source_check(lock)
    evidence["python"] = str(Path(sys.executable).absolute())
    if str(Path(lock["python"]).absolute()) != evidence["python"]:
        raise Blocked("run with the exact interpreter pinned in SOURCE_LOCK.json")
    if args.command == "doctor":
        return evidence | {
            "status": "CPU_PREFLIGHT_PASS",
            "detail": "Source and disk checks only; model fit remains untested.",
        }
    if args.command in {"fetch", "build", "baseline"}:
        raise Blocked(
            f"{args.command}: no qualified {profile} engine recipe is installed"
        )
    if profile != "artifact":
        raise Blocked(
            "artifact byte tests require the artifact profile; "
            "baseline recipes remain separate"
        )
    if backend != "files":
        raise Blocked(
            "LMCache typed tensor transport is not qualified; "
            "file transport is available"
        )
    consumer = ConsumerIdentity(get("CONSUMER"), get("CONSUMER_REVISION"))
    registry = ContextArtifactRegistry(get("STORE_ROOT"))
    try:
        artifact_id = get("ARTIFACT_ID")
        if args.command in {"ingest", "import"}:
            manifest = ContextArtifactManifest.from_dict(
                json.loads(Path(get("MANIFEST")).read_text())
            )
            if manifest.representation != representation:
                raise ArtifactError(
                    "manifest representation differs from configuration"
                )
            if consumer not in manifest.consumers:
                raise ArtifactError("configured consumer is incompatible")
            artifact_id = registry.register(manifest, get("PAYLOAD"))
        elif args.command == "load":
            registry.load(artifact_id, consumer)
        elif args.command == "restore":
            kind = args.restore_kind
            if not kind:
                raise ArtifactError(
                    "restore requires --restore-kind eviction or restart"
                )
            if kind == "eviction":
                before = registry.load(artifact_id, consumer)
                if not registry.evict(artifact_id) or artifact_id in registry.resident:
                    raise ArtifactError("working-copy eviction failed")
                if registry.restore(artifact_id, consumer) != before:
                    raise ArtifactError("restored bytes differ")
            else:
                registry.close()
                registry = None
                argv = [
                    sys.executable,
                    "-m",
                    "tools.reproduce.context_artifacts.run",
                    "load",
                    "--config",
                    str(args.config.resolve()),
                ]
                child = subprocess.run(
                    argv, cwd=runner_root, text=True, capture_output=True, check=False
                )
                if child.returncode:
                    raise ArtifactError(f"fresh-process restore failed: {child.stderr}")
                evidence["child_result"] = json.loads(child.stdout)
        else:
            raise Blocked(
                f"{args.command}: inference acceptance is not implemented; "
                "use explicit import/load/restore byte tests"
            )
        return evidence | {
            "artifact_id": artifact_id,
            "status": "BYTE_TRANSPORT_PASS",
            "cache_state": "page-cache-state-unknown",
            "gpu_tested": False,
        }
    finally:
        if registry is not None:
            registry.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "doctor",
            "fetch",
            "build",
            "ingest",
            "import",
            "baseline",
            "load",
            "restore",
            "test",
            "report",
        ),
    )
    parser.add_argument("--config", type=Path, default=Path(".config"))
    parser.add_argument("--restore-kind", choices=("eviction", "restart"))
    args = parser.parse_args()
    result = {
        "command": args.command,
        "argv": sys.argv,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    code, config = 0, {}
    try:
        config = read_config(args.config)
        result.update(run(args, config))
    except (
        ArtifactError,
        OSError,
        KeyError,
        ValueError,
        subprocess.SubprocessError,
    ) as error:
        code = 2
        result.update(
            status="BLOCKED" if isinstance(error, Blocked) else "FAIL",
            detail=str(error),
        )
    output_root = config.get("KNLP_CA_RESULTS_ROOT", "")
    if output_root:
        try:
            persistent(Path(output_root))
            Path(output_root).mkdir(parents=True, exist_ok=True)
            directory = Path(
                tempfile.mkdtemp(prefix=args.command + "-", dir=output_root)
            )
            (directory / "result.json").write_bytes(canonical(result))
            (directory / "resolved.config").write_bytes(args.config.read_bytes())
        except (ArtifactError, OSError, subprocess.SubprocessError) as error:
            code = 2
            result.update(status="FAIL", detail=f"cannot persist run evidence: {error}")
    print(json.dumps(result, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: MIT
"""Configuration loading for the Modular A/B reproducer."""

from __future__ import annotations

import ast
import os
import shlex
from dataclasses import dataclass
from pathlib import Path


def _parse_dot_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("CONFIG_") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if value.startswith('"'):
            value = ast.literal_eval(value)
        values[key] = str(value)
    return values


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "y", "yes", "true", "on"}


def _split_items(raw: str | None) -> list[str]:
    if raw is None:
        return []
    value = raw.strip()
    if not value:
        return []
    if "\n" in value:
        out: list[str] = []
        for line in value.splitlines():
            candidate = line.strip()
            if not candidate or candidate.startswith("#"):
                continue
            out.append(candidate)
        return out
    return shlex.split(value)


def _integer(value: str | None, default: int) -> int:
    return int(value) if value not in {None, ""} else default


def _artifact_from_target(target: str) -> str:
    if not target.startswith("//") or ":" not in target:
        raise RuntimeError(f"invalid Bazel label: {target}")
    package, name = target[2:].split(":", 1)
    return f"{package}/{name}"


def _environment(name: str, fallback: str) -> str:
    return os.environ.get(name, fallback)


@dataclass(frozen=True)
class ModularConfig:
    top: Path
    config_path: Path
    source_dir: Path
    source_commit: str
    targets: list[tuple[str, str]]
    build_args: list[str]
    jobs: int
    results_root: Path
    reference_builder: str
    alternative_builder: str
    bazel_bin: str
    capture_profile: bool
    fail_on_dirty_source: bool
    runtime_args: list[str]
    runtime_warmups: int
    runtime_runs: int
    runtime_timeout: int
    runtime_marker: str
    dry_run: bool
    mode: str

    @property
    def target(self) -> str:
        if not self.targets:
            return ""
        return self.targets[0][0]

    @property
    def artifact_relpath(self) -> str:
        if not self.targets:
            return ""
        return self.targets[0][1]

    @property
    def variants(self) -> list[str]:
        if self.mode == "reference":
            return ["reference"]
        if self.mode == "alternative":
            return ["alternative"]
        return ["reference", "alternative"]

    @classmethod
    def load(cls, config_path: str) -> "ModularConfig":
        top = Path(__file__).resolve().parents[3]
        path = Path(config_path).expanduser()
        if not path.is_absolute():
            path = top / path
        if not path.is_file():
            raise RuntimeError(f"missing Kconfig output: {path}")

        values = _parse_dot_config(path)
        if not _bool(values.get("CONFIG_KNLP_MODULAR_REPRO")):
            raise RuntimeError("CONFIG_KNLP_MODULAR_REPRO is not enabled")

        if _bool(values.get("CONFIG_KNLP_MODULAR_BUILDER_AB")):
            mode = "ab"
        elif _bool(values.get("CONFIG_KNLP_MODULAR_BUILDER_ALTERNATIVE")):
            mode = "alternative"
        else:
            mode = "reference"

        source_dir = Path(
            _environment(
                "KNLP_MODULAR_SOURCE",
                values.get("CONFIG_KNLP_MODULAR_SOURCE_DIR", "/data/modular"),
            )
        ).expanduser()
        results_root = Path(
            _environment(
                "KNLP_MODULAR_RESULTS",
                values.get(
                    "CONFIG_KNLP_MODULAR_RESULTS_ROOT",
                    "./results/modular-tiled-matmul",
                ),
            )
        ).expanduser()
        if not results_root.is_absolute():
            results_root = top / results_root

        allow_dirty = _bool(os.environ.get("KNLP_MODULAR_ALLOW_DIRTY"))
        jobs = _integer(
            _environment(
                "KNLP_MODULAR_JOBS",
                values.get("CONFIG_KNLP_MODULAR_JOBS", "0"),
            ),
            0,
        )
        if jobs == 0:
            jobs = os.cpu_count() or 1

        build_args = _environment(
            "KNLP_MODULAR_BUILD_ARGS",
            values.get("CONFIG_KNLP_MODULAR_BUILD_ARGS", ""),
        )
        runtime_args = _environment(
            "KNLP_MODULAR_RUNTIME_ARGS",
            values.get("CONFIG_KNLP_MODULAR_RUNTIME_ARGS", ""),
        )
        targets_file = Path(
            _environment(
                "KNLP_MODULAR_TARGETS_FILE",
                values.get("CONFIG_KNLP_MODULAR_TARGETS_FILE", ""),
            )
        ).expanduser()
        if not targets_file.is_absolute():
            targets_file = top / targets_file
        targets_file_specified = bool(
            _environment(
                "KNLP_MODULAR_TARGETS_FILE",
                values.get("CONFIG_KNLP_MODULAR_TARGETS_FILE", ""),
            ).strip()
        )
        if targets_file_specified and not targets_file.is_file():
            raise RuntimeError(
                f"CONFIG_KNLP_MODULAR_TARGETS_FILE does not exist: {targets_file}"
            )
        target_values = _split_items(
            _environment(
                "KNLP_MODULAR_TARGETS",
                values.get("CONFIG_KNLP_MODULAR_TARGETS", ""),
            )
        )
        if not target_values and targets_file.is_file():
            file_targets = _split_items(targets_file.read_text(encoding="utf-8"))
            target_values = file_targets
        artifact_relpath_values = _split_items(
            _environment(
                "KNLP_MODULAR_ARTIFACT_RELPATHS",
                values.get("CONFIG_KNLP_MODULAR_ARTIFACT_RELPATHS", ""),
            )
        )
        legacy_target = _environment(
            "KNLP_MODULAR_TARGET",
            values.get("CONFIG_KNLP_MODULAR_TARGET", ""),
        ).strip()
        legacy_artifact = values.get("CONFIG_KNLP_MODULAR_ARTIFACT_RELPATH", "").strip()

        if not target_values:
            if legacy_target:
                target_values = [legacy_target]
        if not target_values:
            raise RuntimeError(
                "No Modular targets configured (set CONFIG_KNLP_MODULAR_TARGETS or CONFIG_KNLP_MODULAR_TARGET)"
            )
        if not artifact_relpath_values:
            if legacy_artifact:
                artifact_relpath_values = [legacy_artifact]
        if artifact_relpath_values and len(artifact_relpath_values) != 1:
            if len(artifact_relpath_values) != len(target_values):
                raise RuntimeError(
                    "CONFIG_KNLP_MODULAR_ARTIFACT_RELPATHS must be empty, one, "
                    "or match CONFIG_KNLP_MODULAR_TARGETS length"
                )
        elif len(target_values) > len(artifact_relpath_values):
            if artifact_relpath_values:
                artifact_relpath_values = artifact_relpath_values * len(target_values)

        if not artifact_relpath_values:
            artifact_relpath_values = []
            for target in target_values:
                artifact_relpath_values.append(_artifact_from_target(target))

        if len(target_values) != len(artifact_relpath_values):
            raise RuntimeError(
                "target and artifact list lengths do not match for Modular target sweep"
            )

        target_specs: list[tuple[str, str]] = []
        for target, artifact_relpath in zip(target_values, artifact_relpath_values):
            target_specs.append((target, artifact_relpath))

        return cls(
            top=top,
            config_path=path.resolve(),
            source_dir=source_dir.resolve(),
            source_commit=_environment(
                "KNLP_MODULAR_SOURCE_COMMIT",
                values.get("CONFIG_KNLP_MODULAR_SOURCE_COMMIT", ""),
            ),
            targets=target_specs,
            build_args=shlex.split(build_args),
            jobs=jobs,
            results_root=results_root.resolve(),
            reference_builder=_environment(
                "KNLP_MODULAR_REFERENCE_BUILDER",
                values.get("CONFIG_KNLP_MODULAR_REFERENCE_BUILDER", ""),
            ),
            alternative_builder=_environment(
                "KNLP_MODULAR_ALTERNATIVE_BUILDER",
                values.get("CONFIG_KNLP_MODULAR_ALTERNATIVE_BUILDER", ""),
            ),
            bazel_bin=_environment(
                "KNLP_MODULAR_BAZEL",
                values.get("CONFIG_KNLP_MODULAR_BAZEL_BIN", "./bazelw"),
            ),
            capture_profile=_bool(values.get("CONFIG_KNLP_MODULAR_PROFILE"), True),
            fail_on_dirty_source=(
                _bool(
                    values.get("CONFIG_KNLP_MODULAR_FAIL_ON_DIRTY_SOURCE"),
                    True,
                )
                and not allow_dirty
            ),
            runtime_args=shlex.split(runtime_args),
            runtime_warmups=_integer(
                _environment(
                    "KNLP_MODULAR_RUNTIME_WARMUPS",
                    values.get("CONFIG_KNLP_MODULAR_RUNTIME_WARMUPS", "1"),
                ),
                1,
            ),
            runtime_runs=_integer(
                _environment(
                    "KNLP_MODULAR_RUNTIME_RUNS",
                    values.get("CONFIG_KNLP_MODULAR_RUNTIME_RUNS", "10"),
                ),
                10,
            ),
            runtime_timeout=_integer(
                values.get("CONFIG_KNLP_MODULAR_RUNTIME_TIMEOUT"), 300
            ),
            runtime_marker=values.get(
                "CONFIG_KNLP_MODULAR_RUNTIME_MARKER", "Validation PASSED"
            ),
            dry_run=_bool(values.get("CONFIG_KNLP_MODULAR_DRY_RUN")),
            mode=mode,
        )

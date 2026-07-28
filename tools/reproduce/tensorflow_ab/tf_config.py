# SPDX-License-Identifier: MIT
"""Configuration loading for TensorFlow build-system reproduction."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def parse_kconfig(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open() as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key] = value.strip().strip('"')
    return values


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "y", "yes", "true", "on"}


@dataclass(frozen=True)
class Variant:
    label: str
    adapter: str


@dataclass(frozen=True)
class TensorFlowConfig:
    top: Path
    config_path: Path
    source_dir: Path
    target: str
    artifact_relpath: str
    build_args: str
    jobs: int
    results_root: Path
    reference_builder: str
    alternative_builder: str
    bazel_bin: str
    profile: bool
    fail_on_dirty_source: bool
    dry_run: bool
    mode: str

    @classmethod
    def load(cls, config_path: str) -> "TensorFlowConfig":
        top = Path(__file__).resolve().parents[3]
        path = Path(config_path).expanduser().resolve()
        values = parse_kconfig(path)
        if values.get("CONFIG_KNLP_TF_REPRO") != "y":
            raise ValueError(
                "CONFIG_KNLP_TF_REPRO is not enabled; load a "
                "tensorflow-gemma2b-* defconfig"
            )

        if values.get("CONFIG_KNLP_TF_BUILDER_AB") == "y":
            mode = "ab"
        elif values.get("CONFIG_KNLP_TF_BUILDER_ALTERNATIVE") == "y":
            mode = "alternative"
        else:
            mode = "reference"

        def setting(env_name: str, config_name: str, default: str = "") -> str:
            return os.environ.get(env_name, values.get(config_name, default))

        source_dir = (
            Path(
                setting(
                    "KNLP_TF_SOURCE",
                    "CONFIG_KNLP_TF_SOURCE_DIR",
                    "/data/tensorflow",
                )
            )
            .expanduser()
            .resolve()
        )
        results_root = Path(
            setting(
                "KNLP_TF_RESULTS",
                "CONFIG_KNLP_TF_RESULTS_ROOT",
                "./results/tensorflow-ab",
            )
        ).expanduser()
        if not results_root.is_absolute():
            results_root = (top / results_root).resolve()

        jobs = int(setting("KNLP_TF_JOBS", "CONFIG_KNLP_TF_JOBS", "0"))
        fail_on_dirty = _bool(values.get("CONFIG_KNLP_TF_FAIL_ON_DIRTY_SOURCE"), True)
        if _bool(os.environ.get("KNLP_TF_ALLOW_DIRTY")):
            fail_on_dirty = False

        return cls(
            top=top,
            config_path=path,
            source_dir=source_dir,
            target=setting(
                "KNLP_TF_TARGET",
                "CONFIG_KNLP_TF_TARGET",
                "//knlp_tensorflow:gemma_savedmodel_runner",
            ),
            artifact_relpath=setting(
                "KNLP_TF_ARTIFACT_RELPATH",
                "CONFIG_KNLP_TF_ARTIFACT_RELPATH",
                "knlp_tensorflow/gemma_savedmodel_runner",
            ),
            build_args=setting(
                "KNLP_TF_BUILD_ARGS",
                "CONFIG_KNLP_TF_BUILD_ARGS",
                "--config=opt --config=cuda",
            ),
            jobs=jobs,
            results_root=results_root,
            reference_builder=setting(
                "KNLP_TF_REFERENCE_BUILDER",
                "CONFIG_KNLP_TF_REFERENCE_BUILDER",
                "tools/reproduce/tensorflow_ab/builders/bazel.py",
            ),
            alternative_builder=setting(
                "KNLP_TF_ALTERNATIVE_BUILDER",
                "CONFIG_KNLP_TF_ALTERNATIVE_BUILDER",
            ),
            bazel_bin=setting(
                "KNLP_TF_BAZEL",
                "CONFIG_KNLP_TF_BAZEL_BIN",
                "bazel",
            ),
            profile=_bool(values.get("CONFIG_KNLP_TF_PROFILE"), True),
            fail_on_dirty_source=fail_on_dirty,
            dry_run=_bool(values.get("CONFIG_KNLP_TF_DRY_RUN")),
            mode=mode,
        )

    def variants(self) -> tuple[Variant, ...]:
        variants: list[Variant] = []
        if self.mode in {"reference", "ab"}:
            variants.append(Variant("reference", self.reference_builder))
        if self.mode in {"alternative", "ab"}:
            variants.append(Variant("candidate", self.alternative_builder))
        return tuple(variants)

    def resolve_repo_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.top / path
        return path.resolve()

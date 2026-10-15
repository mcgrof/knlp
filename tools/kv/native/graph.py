# SPDX-License-Identifier: MIT
"""Explicit source identities and setup ordering for captured model runs."""

import os
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from types import MappingProxyType


@dataclass(frozen=True)
class ExecutionVariant:
    sources: Mapping[str, str]
    single_writer: bool
    fuse_scale: bool

    def __post_init__(self):
        object.__setattr__(self, "sources", MappingProxyType(dict(self.sources)))

    def environment(self, base=None):
        return {
            **(os.environ if base is None else base),
            "VLLM_ASYM_SINGLE_WRITER": str(int(self.single_writer)),
            "FLASHINFER_FUSE_V_SCALE": str(int(self.fuse_scale)),
            "FLASHINFER_ASYM_FP8_PV": "0",
        }

    def verify_environment(self, environment=None):
        actual = os.environ if environment is None else environment
        for name, value in self.environment({}).items():
            if actual.get(name) != value:
                raise RuntimeError(f"capture environment differs: {name}")

    def verify_sources(self, directories, loaded_modules=None):
        if set(directories) != set(self.sources):
            raise ValueError("source directory names differ from the variant")
        for name, expected in self.sources.items():
            if re.fullmatch(r"[0-9a-f]{40}", expected) is None:
                raise ValueError("source identity must be a full Git object ID")
            directory = Path(directories[name]).resolve()
            actual = subprocess.check_output(
                ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
            ).strip()
            dirty = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(directory),
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                ],
                text=True,
            ).strip()
            if actual != expected or dirty:
                raise RuntimeError(f"source checkout differs: {name}")
            if loaded_modules is not None:
                loaded = Path(loaded_modules[name]).resolve()
                if not loaded.is_relative_to(directory):
                    raise RuntimeError(f"loaded module is outside source: {name}")


def resolve_variant(name, definitions):
    """Require an exact registered identity; never infer flags from a name."""
    if name not in definitions:
        raise ValueError(f"unknown execution variant: {name}")
    definition = definitions[name]
    if not isinstance(definition, ExecutionVariant):
        raise TypeError("execution variant is not an explicit definition")
    return definition


def install_capture_hooks(worker_type, runner_type, configure, check, variant):
    """Configure before warmup, then validate immediately before capture.

    These hooks run during engine construction. They do not wrap graph replay
    or change the captured writer. Restore them after constructing the engine.
    """
    marker = "_native_capture_setup_installed"
    if getattr(worker_type, marker, False) or getattr(runner_type, marker, False):
        raise RuntimeError("capture setup is already installed")
    warmup = worker_type.compile_or_warm_up_model
    capture = runner_type.capture_model

    @wraps(warmup)
    def configured(worker, *args, **kwargs):
        variant.verify_environment()
        configure(worker)
        return warmup(worker, *args, **kwargs)

    @wraps(capture)
    def checked(runner, *args, **kwargs):
        variant.verify_environment()
        check(runner)
        return capture(runner, *args, **kwargs)

    worker_type.compile_or_warm_up_model = configured
    runner_type.capture_model = checked
    setattr(worker_type, marker, True)
    setattr(runner_type, marker, True)

    def restore():
        worker_type.compile_or_warm_up_model = warmup
        runner_type.capture_model = capture
        delattr(worker_type, marker)
        delattr(runner_type, marker)

    return restore

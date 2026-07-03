#!/usr/bin/env python3
"""Deterministic artifact writer for code-reason runs.

Owns the run directory layout and guarantees every run records config,
environment, git state, manifest, and per-(task, model, mode) prompt,
transcript, certificate, answer, metrics, and costs. Optional JSON-Schema
validation runs when `jsonschema` is installed; otherwise it soft-warns so
the harness never hard-depends on it.

Layout (see research/code_reason/README.md):
  <run_dir>/
    config.json  environment.json  git-state.json  manifest.jsonl
    tasks/<task_id>/<model>/<mode>/
      prompt.md  transcript.jsonl  certificate.{json,md}  answer.json
      metrics.json  costs.json
      addendums/<name>.json
      ab_vs_blb/{A.paper_only,blB.augmented,comparison}.json
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys

SCHEMA_DIR = os.path.join(os.path.dirname(__file__), "..", "schemas")

try:
    import jsonschema  # type: ignore

    _HAVE_JSONSCHEMA = True
except Exception:  # pragma: no cover - optional dependency
    _HAVE_JSONSCHEMA = False

_SCHEMA_CACHE: dict = {}


def _load_schema(name):
    if name not in _SCHEMA_CACHE:
        path = os.path.join(SCHEMA_DIR, f"{name}.schema.json")
        with open(path) as fh:
            _SCHEMA_CACHE[name] = json.load(fh)
    return _SCHEMA_CACHE[name]


def _validate(obj, schema_name):
    """Validate obj against a schema. Soft-warn if jsonschema is absent."""
    if not _HAVE_JSONSCHEMA:
        return
    try:
        jsonschema.validate(obj, _load_schema(schema_name))
    except jsonschema.ValidationError as exc:  # pragma: no cover
        print(
            f"[artifact_writer] WARN {schema_name}: {exc.message}",
            file=sys.stderr,
        )


def _dump(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _git(root, *args, default=""):
    try:
        return subprocess.check_output(
            ["git", "-C", root, *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return default


def capture_environment():
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    for mod in ("torch", "anthropic", "openai", "jsonschema"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    return env


def capture_git_state(repo_root):
    return {
        "repo": repo_root,
        "commit": _git(repo_root, "rev-parse", "HEAD"),
        "branch": _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git(repo_root, "status", "--porcelain")),
    }


class ArtifactWriter:
    def __init__(self, run_dir, config_path=None, repo_root="."):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.manifest_path = os.path.join(run_dir, "manifest.jsonl")
        # run-level records
        if config_path and os.path.exists(config_path):
            with open(config_path) as fh:
                cfg = json.load(fh)
            _validate(cfg, "run_config")
            _dump(cfg, os.path.join(run_dir, "config.json"))
        _dump(capture_environment(), os.path.join(run_dir, "environment.json"))
        _dump(
            capture_git_state(repo_root),
            os.path.join(run_dir, "git-state.json"),
        )

    # ---- run level -----------------------------------------------------
    def append_manifest(self, task_entry):
        _validate(task_entry, "task_manifest")
        with open(self.manifest_path, "a") as fh:
            fh.write(json.dumps(task_entry, sort_keys=True) + "\n")

    # ---- per (task, model, mode) --------------------------------------
    def task_dir(self, task_id, model, mode):
        d = os.path.join(self.run_dir, "tasks", task_id, model, mode)
        os.makedirs(d, exist_ok=True)
        return d

    def write_prompt(self, task_id, model, mode, text):
        p = os.path.join(self.task_dir(task_id, model, mode), "prompt.md")
        with open(p, "w") as fh:
            fh.write(text if text.endswith("\n") else text + "\n")

    def append_transcript(self, task_id, model, mode, step):
        _validate(step, "agent_step")
        p = os.path.join(self.task_dir(task_id, model, mode), "transcript.jsonl")
        with open(p, "a") as fh:
            fh.write(json.dumps(step, sort_keys=True) + "\n")

    def write_certificate(self, task_id, model, mode, cert, markdown=None):
        _validate(cert, "certificate")
        d = self.task_dir(task_id, model, mode)
        _dump(cert, os.path.join(d, "certificate.json"))
        if markdown is not None:
            with open(os.path.join(d, "certificate.md"), "w") as fh:
                fh.write(markdown)

    def write_result(self, task_id, model, mode, result):
        _validate(result, "result")
        _dump(
            result,
            os.path.join(self.task_dir(task_id, model, mode), "answer.json"),
        )

    def write_metrics(self, task_id, model, mode, metrics):
        _validate(metrics, "metrics")
        _dump(
            metrics,
            os.path.join(self.task_dir(task_id, model, mode), "metrics.json"),
        )

    def write_costs(self, task_id, model, mode, costs):
        _dump(
            costs,
            os.path.join(self.task_dir(task_id, model, mode), "costs.json"),
        )

    def write_addendum(self, task_id, model, mode, addendum):
        _validate(addendum, "addendum")
        d = os.path.join(self.task_dir(task_id, model, mode), "addendums")
        _dump(addendum, os.path.join(d, f"{addendum['name']}.json"))

    def write_ab(self, task_id, model, mode, a_paper, blb_aug, comparison):
        d = os.path.join(self.task_dir(task_id, model, mode), "ab_vs_blb")
        _dump(a_paper, os.path.join(d, "A.paper_only.json"))
        _dump(blb_aug, os.path.join(d, "blB.augmented.json"))
        _dump(comparison, os.path.join(d, "comparison.json"))

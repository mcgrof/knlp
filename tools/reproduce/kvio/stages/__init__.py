# SPDX-License-Identifier: MIT
"""Minimal stage framework for the kvio reproduce orchestrator.

kvio is self-contained -- it does NOT reuse another orchestrator's stage
machinery.  A stage is a ``run(ctx) -> StageResult`` callable; the
orchestrator (``run.py``) builds a StageContext, invokes the profile's
stage, and records the result.

This intentionally mirrors only the small surface the replay stages use
(``ctx.cfg`` / ``ctx.stage_dir`` / ``ctx.name`` / ``ctx.telemetry.log_artifact``
/ ``ctx.mark_done`` / ``ctx.mark_skipped`` + ``StageResult``), so the pure
replay code can be unit-tested and run without a heavyweight framework.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class StageResult:
    """Outcome of a stage run."""

    name: str
    status: str  # "passed" | "skipped" | "failed"
    reason: str = ""


class _Telemetry:
    """Append-only record of the artifacts a stage produced (JSONL sidecar)."""

    def __init__(self, stage_dir: Path) -> None:
        self._log = Path(stage_dir) / "artifacts.jsonl"

    def log_artifact(self, path, label: str = "") -> None:
        try:
            with open(self._log, "a") as f:
                f.write(json.dumps({"path": str(path), "label": label}) + "\n")
        except OSError:
            # Telemetry is best-effort; never fail a stage over a sidecar write.
            pass


class StageContext:
    """Everything a kvio stage needs: its config, output dir, and telemetry."""

    def __init__(self, name: str, cfg, stage_dir) -> None:
        self.name = name
        self.cfg = cfg
        self.stage_dir = Path(stage_dir)
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry = _Telemetry(self.stage_dir)
        self.status: Optional[str] = None
        self.info: dict = {}

    def mark_done(self, info: Optional[dict] = None) -> None:
        self.status = "done"
        if info:
            self.info.update(info)

    def mark_skipped(self, reason: str) -> None:
        self.status = "skipped"
        self.info["skip_reason"] = reason

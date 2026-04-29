"""Telemetry abstraction.  Local JSONL is canonical; W&B and trackerio
are optional mirrors."""

from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class Telemetry:
    """Base interface.  Subclasses override to emit elsewhere."""

    def start_run(self, manifest: dict) -> None: ...
    def start_stage(self, stage: str, config: dict) -> None: ...
    def log_metric(
        self, name: str, value: float | int | str, **labels: Any
    ) -> None: ...
    def log_artifact(self, path: str | Path, artifact_type: str = "result") -> None: ...
    def finish_stage(self, status: str) -> None: ...
    def finish_run(self) -> None: ...


class LocalTelemetry(Telemetry):
    """Mandatory canonical sink.  Writes JSONL to results/<run_id>/."""

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.results_dir / "metrics.jsonl"
        self.artifacts_path = self.results_dir / "artifacts.json"
        self._stage: str | None = None
        self._artifacts: list[dict] = []

    def _write(self, record: dict) -> None:
        record["t"] = time.time()
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def start_run(self, manifest: dict) -> None:
        self._write({"event": "run_start", "manifest_keys": list(manifest.keys())})

    def start_stage(self, stage: str, config: dict) -> None:
        self._stage = stage
        self._write({"event": "stage_start", "stage": stage, "config": config})

    def log_metric(self, name: str, value, **labels) -> None:
        self._write(
            {
                "event": "metric",
                "stage": self._stage,
                "name": name,
                "value": value,
                "labels": labels,
            }
        )

    def log_artifact(self, path, artifact_type: str = "result") -> None:
        rec = {"stage": self._stage, "path": str(path), "type": artifact_type}
        self._artifacts.append(rec)
        self._write({"event": "artifact", **rec})
        self.artifacts_path.write_text(json.dumps(self._artifacts, indent=2))

    def finish_stage(self, status: str) -> None:
        self._write({"event": "stage_end", "stage": self._stage, "status": status})
        self._stage = None

    def finish_run(self) -> None:
        self._write({"event": "run_end"})


class WandbTelemetry(Telemetry):
    """Optional W&B mirror.  Silent no-op if wandb isn't installed."""

    def __init__(self, project: str, entity: str | None, mode: str, manifest: dict):
        self.project = project
        self.entity = entity or None
        self.mode = mode
        self._wandb = None
        try:
            import wandb  # type: ignore[import-not-found]

            self._wandb = wandb
        except Exception:
            return
        try:
            self._wandb.init(
                project=project,
                entity=self.entity,
                group=manifest.get("run_id"),
                config=manifest,
                tags=["paper-memory-decode", manifest.get("profile", "decode")],
                mode=mode,
            )
        except Exception:
            self._wandb = None

    def start_stage(self, stage: str, config: dict) -> None:
        if self._wandb:
            self._wandb.run.tags = self._wandb.run.tags + (f"stage:{stage}",)

    def log_metric(self, name: str, value, **labels) -> None:
        if self._wandb is not None and isinstance(value, (int, float)):
            self._wandb.log(
                {name: value, **{f"label.{k}": v for k, v in labels.items()}}
            )

    def log_artifact(self, path, artifact_type: str = "result") -> None:
        if self._wandb is not None and Path(path).exists():
            try:
                art = self._wandb.Artifact(Path(path).stem, type=artifact_type)
                art.add_file(str(path))
                self._wandb.run.log_artifact(art)
            except Exception:
                pass

    def finish_run(self) -> None:
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass


class TrackerIOTelemetry(Telemetry):
    """Optional trackerio mirror.  Stub adapter — extend when the
    knlp trackerio client API stabilizes."""

    def __init__(self, project: str, url: str, manifest: dict):
        self.project = project
        self.url = url
        self._client = None
        try:
            import trackio  # type: ignore[import-not-found]

            self._client = trackio
        except Exception:
            return

    def log_metric(self, name: str, value, **labels) -> None:
        if self._client is None:
            return
        try:
            self._client.log({name: value, **labels})
        except Exception:
            pass


class CompositeTelemetry(Telemetry):
    """Fan-out to several backends.  Failure of a mirror never fails the run."""

    def __init__(self, sinks: list[Telemetry]):
        self.sinks = sinks

    def _fanout(self, method: str, *args, **kwargs):
        for s in self.sinks:
            try:
                getattr(s, method)(*args, **kwargs)
            except Exception:
                pass

    def start_run(self, manifest):
        self._fanout("start_run", manifest)

    def start_stage(self, stage, config):
        self._fanout("start_stage", stage, config)

    def log_metric(self, name, value, **labels):
        self._fanout("log_metric", name, value, **labels)

    def log_artifact(self, path, artifact_type="result"):
        self._fanout("log_artifact", path, artifact_type)

    def finish_stage(self, status):
        self._fanout("finish_stage", status)

    def finish_run(self):
        self._fanout("finish_run")


def build(cfg, manifest_dict: dict, results_dir: Path) -> Telemetry:
    sinks: list[Telemetry] = [LocalTelemetry(results_dir)]
    if cfg.enable_wandb and os.environ.get("WANDB_API_KEY"):
        sinks.append(
            WandbTelemetry(
                cfg.wandb_project,
                cfg.wandb_entity or None,
                cfg.wandb_mode,
                manifest_dict,
            )
        )
    if cfg.enable_trackerio and (
        os.environ.get("TRACKERIO_API_KEY") or os.environ.get("TRACKERIO_TOKEN")
    ):
        sinks.append(
            TrackerIOTelemetry(cfg.trackerio_project, cfg.trackerio_url, manifest_dict)
        )
    return CompositeTelemetry(sinks)

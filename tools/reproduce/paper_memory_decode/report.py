"""Generate report.md and report.json from stage results."""
from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any


def collect_stage_outputs(run_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    stages_dir = run_dir / "stages"
    if not stages_dir.exists():
        return out
    for stage in sorted(stages_dir.iterdir()):
        if not stage.is_dir():
            continue
        rec = {"name": stage.name, "status": "missing", "reason": ""}
        done = stage / "DONE"
        skipped = stage / "SKIPPED"
        if skipped.exists():
            try:
                payload = json.loads(skipped.read_text())
                rec.update(payload)
            except Exception:
                pass
            rec["status"] = "skipped"
        elif done.exists():
            try:
                payload = json.loads(done.read_text())
                rec.update(payload)
            except Exception:
                pass
        # Pull metric summary from metrics.jsonl
        m_path = stage / "metrics.jsonl"
        rec["metric_count"] = 0
        rec["metrics"] = []
        if m_path.exists():
            for line in m_path.read_text().splitlines():
                try:
                    j = json.loads(line)
                    if "metric" in j:
                        rec["metric_count"] += 1
                        rec["metrics"].append(j)
                except Exception:
                    pass
        out.append(rec)
    return out


def write_reports(run_dir: Path, manifest: dict, stage_results: list[dict]) -> tuple[Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "report.json"
    md_path = run_dir / "report.md"

    summary = {
        "run_id": manifest.get("run_id"),
        "profile": manifest.get("profile"),
        "started_at": manifest.get("started_at"),
        "host": manifest.get("host", {}),
        "hardware": manifest.get("hardware", {}),
        "git": manifest.get("git", {}),
        "stages": stage_results,
        "pass_count": sum(1 for s in stage_results if s["status"] == "passed"),
        "fail_count": sum(1 for s in stage_results if s["status"] == "failed"),
        "skip_count": sum(1 for s in stage_results if s["status"] == "skipped"),
    }
    json_path.write_text(json.dumps(summary, indent=2, default=str))

    md = []
    md.append(f"# Reproduction report: paper-memory-decode\n")
    md.append(f"- Profile: `{summary['profile']}`")
    md.append(f"- Run ID: `{summary['run_id']}`")
    md.append(f"- Started: {summary['started_at']}")
    hw = summary["hardware"]
    md.append(f"- Hardware: {hw.get('gpu_count', 0)}× "
              f"{', '.join(hw.get('gpu_names') or ['no GPU'])}")
    md.append(f"  · CUDA {hw.get('cuda_version', '?')}, "
              f"driver {hw.get('driver_version', '?')}")
    md.append(f"  · Disk free {hw.get('free_disk_gb', 0)} GB, "
              f"RAM {hw.get('total_ram_gb', 0)} GB\n")

    md.append("## Git refs")
    for name, info in summary["git"].items():
        dirty = " (dirty)" if info.get("dirty") else ""
        md.append(f"- **{name}** `{info.get('ref', '')}` "
                  f"{(info.get('commit') or '')[:12]}{dirty}")
    md.append("")

    md.append("## Stage results")
    md.append("| Stage | Status | Notes |")
    md.append("|---|---|---|")
    for s in stage_results:
        notes = s.get("reason") or f"{s.get('metric_count', 0)} metrics"
        emoji = {"passed": "✅", "failed": "❌", "skipped": "⏭",
                 "warned": "⚠", "missing": "·"}.get(s["status"], "?")
        md.append(f"| `{s['name']}` | {emoji} {s['status']} | {notes} |")
    md.append("")

    md.append(f"## Summary\n")
    md.append(f"- **Passed**: {summary['pass_count']}")
    md.append(f"- **Failed**: {summary['fail_count']}")
    md.append(f"- **Skipped**: {summary['skip_count']}\n")

    if summary["fail_count"]:
        md.append("**Run did not complete successfully.** Inspect "
                  "`results/<run_id>/stages/<stage>/stderr.log` for the "
                  "failed stages.\n")

    md.append("## Artifacts")
    md.append(f"- Manifest: `{run_dir}/manifest.json`")
    md.append(f"- Metrics: `{run_dir}/metrics.jsonl`")
    md.append(f"- Per-stage logs: `{run_dir}/stages/<stage>/{{stdout,stderr}}.log`")
    md_path.write_text("\n".join(md))

    return md_path, json_path

"""Stage 11: generate the run report.

Calls report.write_reports() to produce report.md and report.json
from the stage outputs.  Always passes (a failed earlier stage is
surfaced in the report, not here).
"""

from __future__ import annotations

from ..stages import StageContext, StageResult
from .. import report as _report


def run(ctx: StageContext) -> StageResult:
    # Re-collect stage outputs at report time in case some were written
    # after the pipeline started.
    stage_results = _report.collect_stage_outputs(ctx.run_dir)
    manifest_path = ctx.run_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        import json

        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            pass

    md_path, json_path = _report.write_reports(ctx.run_dir, manifest, stage_results)

    ctx.telemetry.log_artifact(md_path, "report")
    ctx.telemetry.log_artifact(json_path, "report_json")

    ctx.log_metric("stages_total", len(stage_results))
    passed = sum(1 for r in stage_results if r.get("status") == "passed")
    failed = sum(1 for r in stage_results if r.get("status") == "failed")
    skipped = sum(1 for r in stage_results if r.get("status") == "skipped")
    ctx.log_metric("stages_passed", passed)
    ctx.log_metric("stages_failed", failed)
    ctx.log_metric("stages_skipped", skipped)

    ctx.mark_done(
        {
            "report_md": str(md_path),
            "report_json": str(json_path),
            "stages_passed": passed,
            "stages_failed": failed,
            "stages_skipped": skipped,
        }
    )
    return StageResult(name=ctx.name, status="passed")

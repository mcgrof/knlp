# SPDX-License-Identifier: MIT
"""Recompute scores and gates from raw prediction files.

The report never trusts a metric written by another stage. It rereads every
prediction row, rescores exact match from the stored prediction and gold
strings, and applies the predeclared decision rules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import dataset as D
from . import evaluate as E
from . import runs as R

READER_GATE = {"aggregate": 0.90, "family": 0.80}
PILOT_GATE = {
    "within_full": 0.10,
    "family_within_full": 0.15,
    "over_empty_and_wrong": 0.20,
    "over_untrained": 0.10,
}
RESCUE_RULE = {"deficit_min": 0.10, "deficit_max": 0.20, "family_deficit_max": 0.25}


def rescore(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        row = dict(row)
        if row.get("gold") is not None:
            row["em"] = D.exact_match(row["pred"], row["gold"], row["attribute"])
        out.append(row)
    return out


def load_arm(path: Path) -> list[dict[str, Any]]:
    return rescore(D.read_jsonl(path)) if path.exists() else []


def arms_in(directory: Path) -> dict[str, list[dict[str, Any]]]:
    arms = {}
    for path in sorted(directory.glob("*_predictions.jsonl")):
        arms[path.name[: -len("_predictions.jsonl")]] = load_arm(path)
    return arms


def reader_gate(full_rows: list[dict[str, Any]]) -> dict[str, Any]:
    m = E.metrics(full_rows)
    families_ok = all(v >= READER_GATE["family"] for v in m["by_family"].values())
    return {
        "metrics": m,
        "pass": m["em"] >= READER_GATE["aggregate"] and families_ok,
        "rule": READER_GATE,
    }


def pilot_gate(
    arms: dict[str, list[dict[str, Any]]],
    learned: str,
    wrong: str,
    stratum: str | None = None,
) -> dict[str, Any]:
    def pick(name: str) -> list[dict[str, Any]]:
        rows = arms.get(name, [])
        return [r for r in rows if stratum is None or r["stratum"] == stratum]

    full, empty, untrained = (
        E.metrics(pick("full")),
        E.metrics(pick("empty")),
        E.metrics(pick("untrained")),
    )
    learned_m, wrong_m = E.metrics(pick(learned)), E.metrics(pick(wrong))
    summary_m = E.metrics(pick("summary")) if "summary" in arms else None
    checks = {
        "learned_within_full": full["em"] - learned_m["em"]
        <= PILOT_GATE["within_full"],
        "families_within_full": all(
            full["by_family"].get(f, 0) - learned_m["by_family"].get(f, 0)
            <= PILOT_GATE["family_within_full"]
            for f in learned_m["by_family"]
        ),
        "beats_empty": learned_m["em"] - empty["em"]
        >= PILOT_GATE["over_empty_and_wrong"],
        "beats_wrong": learned_m["em"] - wrong_m["em"]
        >= PILOT_GATE["over_empty_and_wrong"],
        "beats_untrained": learned_m["em"] - untrained["em"]
        >= PILOT_GATE["over_untrained"],
    }
    deficit = full["em"] - learned_m["em"]
    family_deficit = max(
        (
            full["by_family"].get(f, 0) - learned_m["by_family"].get(f, 0)
            for f in learned_m["by_family"]
        ),
        default=0.0,
    )
    rescue = (
        checks["beats_empty"]
        and checks["beats_wrong"]
        and checks["beats_untrained"]
        and RESCUE_RULE["deficit_min"] < deficit <= RESCUE_RULE["deficit_max"]
        and family_deficit <= RESCUE_RULE["family_deficit_max"]
    )
    out = {
        "arms": {
            "full": full,
            "empty": empty,
            "untrained": untrained,
            learned: learned_m,
            wrong: wrong_m,
        },
        "checks": checks,
        "quality_pass": all(checks.values()),
        "deficit_to_full": deficit,
        "max_family_deficit": family_deficit,
        "rescue_rule_met": bool(rescue and not all(checks.values())),
    }
    if summary_m is not None:
        out["arms"]["summary"] = summary_m
        out["learned_minus_summary"] = E.paired_bootstrap(
            pick(learned), pick("summary")
        )
        out["summary_family_losses"] = {
            f: summary_m["by_family"].get(f, 0) - learned_m["by_family"].get(f, 0)
            for f in learned_m["by_family"]
        }
        out["utility_supported"] = out["learned_minus_summary"]["ci95"][0] > 0 and all(
            v <= 0.05 for v in out["summary_family_losses"].values()
        )
    return out


def tiny_fit_gate(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    gain = after["em"] - before["em"]
    return {
        "untrained_em": before["em"],
        "final_em": after["em"],
        "gain": gain,
        "pass": after["em"] >= 0.90 or gain >= 0.20,
        "rule": "final >= 0.90 training exact match or >= 0.20 gain over untrained",
    }


def build(run_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"run_dir": str(run_dir), "sections": {}}
    evaluations = run_dir / "evaluations"
    if (evaluations / "calibration").exists():
        arms = arms_in(evaluations / "calibration")
        if "full" in arms:
            report["sections"]["reader_gate"] = reader_gate(arms["full"])
            report["sections"]["calibration_arms"] = {
                a: E.metrics(r) for a, r in arms.items()
            }
    tiny_dir = run_dir / "tiny"
    if (tiny_dir / "val_0_predictions.jsonl").exists():
        before = E.metrics(load_arm(tiny_dir / "val_0_predictions.jsonl"))
        finals = sorted(
            tiny_dir.glob("val_*_predictions.jsonl"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        after = E.metrics(load_arm(finals[-1]))
        report["sections"]["tiny_fit_gate"] = tiny_fit_gate(before, after)
    for pool in ("validation", "test_ordinary", "test_shifted", "test"):
        directory = evaluations / pool
        if not directory.exists():
            continue
        arms = arms_in(directory)
        section: dict[str, Any] = {"arms": {a: E.metrics(r) for a, r in arms.items()}}
        learned_arms = sorted(a for a in arms if a.startswith("learned"))
        for learned in learned_arms:
            wrong = learned.replace("learned", "wrong", 1)
            if (
                wrong in arms
                and "full" in arms
                and "empty" in arms
                and "untrained" in arms
            ):
                section[f"gate:{learned}"] = pilot_gate(arms, learned, wrong)
                strata = {r["stratum"] for r in arms[learned]}
                if len(strata) > 1:
                    for stratum in sorted(strata):
                        section[f"gate:{learned}:{stratum}"] = pilot_gate(
                            arms, learned, wrong, stratum
                        )
        report["sections"][pool] = section
    for seed_dir in sorted(run_dir.glob("train_*")):
        summary_path = seed_dir / "train_summary.json"
        if summary_path.exists():
            summary = R.load_json(summary_path)
            report["sections"].setdefault("training", {})[seed_dir.name] = {
                "selected_checkpoint": summary.get("selected_checkpoint"),
                "validations": {
                    k: {
                        "macro_em": v["macro_em"],
                        "em": v["em"],
                        "by_family": v["by_family"],
                    }
                    for k, v in summary.get("validations", {}).items()
                },
                "train_seconds": summary.get("train_seconds"),
                "peak_memory": summary.get("peak_memory"),
                "updates_run": summary.get("updates_run"),
            }
    replay = run_dir / "replay" / "replay_acceptance.json"
    if replay.exists():
        report["sections"]["replay"] = R.load_json(replay)["verdict"]
    fit = run_dir / "prefix_fit" / "fit_summary.json"
    if fit.exists():
        data = R.load_json(fit)
        rows = {
            arm: load_arm(run_dir / "prefix_fit" / f"{arm}_predictions.jsonl")
            for arm in ("generated", "direct", "refined")
        }
        report["sections"]["prefix_fit"] = {
            "metrics": {arm: E.metrics(r) for arm, r in rows.items()},
            "direct_qualified_worlds": data["direct_qualified_worlds"],
            "refined_qualified_worlds": data["refined_qualified_worlds"],
            "construction_seconds": data["construction_seconds"],
            "total_updates": data["total_updates"],
        }
    profile = run_dir / "profile" / "train_summary.json"
    if profile.exists():
        report["sections"]["profile"] = R.load_json(profile).get("profile")
    R.dump_json(run_dir / "report.json", report)
    (run_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def _pct(value: float | None) -> str:
    if value is None or value != value:
        return "n/a"
    return f"{100 * value:.1f}%"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# Recomputed results for `{report['run_dir']}`", ""]
    sections = report["sections"]
    if "reader_gate" in sections:
        g = sections["reader_gate"]
        lines += [
            "## Reader qualification (full context, calibration)",
            "",
            f"Pass: {g['pass']}. Aggregate {_pct(g['metrics']['em'])}.",
            "",
        ]
        lines += ["| family | exact match |", "| --- | ---: |"]
        lines += [f"| {f} | {_pct(v)} |" for f, v in g["metrics"]["by_family"].items()]
        lines.append("")
    if "tiny_fit_gate" in sections:
        g = sections["tiny_fit_gate"]
        lines += [
            "## Tiny-fit gate",
            "",
            f"Untrained {_pct(g['untrained_em'])}, final {_pct(g['final_em'])}, gain {_pct(g['gain'])}. Pass: {g['pass']}.",
            "",
        ]
    for pool in ("validation", "test_ordinary", "test_shifted", "test"):
        if pool not in sections:
            continue
        s = sections[pool]
        lines += [
            f"## {pool}",
            "",
            "| arm | n | exact match | macro | direct | updated | linked | identifier |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for arm, m in s["arms"].items():
            fam = m["by_family"]
            lines.append(
                f"| {arm} | {m['n']} | {_pct(m['em'])} | {_pct(m['macro_em'])} | "
                + " | ".join(_pct(fam.get(f)) for f in D.FAMILIES)
                + " |"
            )
        lines.append("")
        for key, gate in s.items():
            if not key.startswith("gate:"):
                continue
            lines += [f"### {key}", ""]
            for check, ok in gate["checks"].items():
                lines.append(f"- {check}: {ok}")
            lines.append(
                f"- quality_pass: {gate['quality_pass']}; deficit to full {_pct(gate['deficit_to_full'])}; rescue rule met: {gate['rescue_rule_met']}"
            )
            if "learned_minus_summary" in gate:
                b = gate["learned_minus_summary"]
                lines.append(
                    f"- learned minus summary: mean {_pct(b['mean_difference'])}, 95% CI [{_pct(b['ci95'][0])}, {_pct(b['ci95'][1])}] over {b['worlds']} worlds; utility supported: {gate.get('utility_supported')}"
                )
            lines.append("")
    if "replay" in sections:
        r = sections["replay"]
        lines += [
            "## Cold-start replay",
            "",
            f"Pass: {r['pass']}. {r['token_sequences_equal']}/{r['live_answers']} restored token sequences equal; max scored-token logit difference {r['restored_scored_tokens_max_abs_logit']:.3e} against an unmodified-reader envelope of {r['unmodified_reader_envelope_max_abs_logit']:.3e}.",
            "",
        ]
    if "prefix_fit" in sections:
        f = sections["prefix_fit"]
        lines += [
            "## Bounded direct-learning comparison (diagnostic worlds)",
            "",
            "| arm | exact match | construction seconds |",
            "| --- | ---: | ---: |",
        ]
        for arm in ("generated", "direct", "refined"):
            lines.append(
                f"| {arm} | {_pct(f['metrics'][arm]['em'])} | {f['construction_seconds'][arm]:.1f} |"
            )
        lines += [
            "",
            f"Qualified direct fits: {f['direct_qualified_worlds']}; qualified refinements: {f['refined_qualified_worlds']}; total updates {f['total_updates']}.",
            "",
        ]
    if "training" in sections:
        lines += ["## Training", ""]
        for name, t in sections["training"].items():
            vals = ", ".join(
                f"{k}: {_pct(v['macro_em'])}" for k, v in t["validations"].items()
            )
            lines.append(
                f"- {name}: selected {t['selected_checkpoint']}; validation macro EM {vals}; {t['train_seconds']:.0f} s"
            )
        lines.append("")
    if "profile" in sections and sections["profile"]:
        p = sections["profile"]
        lines += [
            "## Profile",
            "",
            f"{p['timed_updates']} timed updates: mean {p['seconds_per_update_mean']:.3f} s, p50 {p['seconds_per_update_p50']:.3f} s, projected 1,000 updates {p['projected_seconds_1000_updates'] / 60:.1f} min.",
            "",
        ]
    return "\n".join(lines)

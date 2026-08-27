#!/usr/bin/env python3
"""Fold a control-aware screen into one report.

Reads the parity record, each arm's run manifest, and the evaluator
output, and lays the arms side by side on the measurements the plan
declares decisive: strict generation behavior with thinking disabled,
the thinking-enabled stress pass, forced-choice content scoring, and
the control-state probe.

The interpretation section states which of the plan's predeclared
readings the numbers are consistent with.  It deliberately stops at
"consistent with" -- one seed and twenty questions cannot settle a
mechanism, and the promotion rule requires a difference in output
behavior or forced-choice margin across seeds, not one extra correct
answer.

Usage: control_aware_report.py --screen-dir DIR --eval JSON --out MD
"""

import argparse
import json
from pathlib import Path

ARM_ORDER = [
    "legacy_raw",
    "dedup_only",
    "dedup_scale_matched",
    "control_anchor",
    "content_anchor_matched",
]
ARM_MEANING = {
    "legacy_raw": "exact historical objective (reproduction control)",
    "dedup_only": "unique support, anchor term removed",
    "dedup_scale_matched": "unique support at legacy coefficient mass "
    "(weak control: AdamW absorbs most of a global loss scalar, leaving "
    "a few percent as extra step size)",
    "control_anchor": "unique + anchors on first-answer and end-of-turn rows",
    "content_anchor_matched": "unique + matched anchors on non-control rows",
}


def fmt(v, spec=".3f"):
    return format(v, spec) if isinstance(v, (int, float)) else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen-dir", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sd = Path(args.screen_dir)
    runs = {}
    for arm in ARM_ORDER:
        rj = sd / arm / "run.json"
        if rj.is_file():
            runs[arm] = json.loads(rj.read_text())
    ev = json.loads(Path(args.eval).read_text())
    res = ev["results"]
    parity = None
    if (sd / "parity.json").is_file():
        parity = json.loads((sd / "parity.json").read_text())

    ref = next(iter(runs.values()), {})
    L = [
        "# Control-aware cartridge screen",
        "",
        f"Model {ref.get('model')}, patient {ref.get('patient')}, "
        f"{ref.get('steps')} steps per arm, accumulation {ref.get('accum')}, "
        f"learning rate {ref.get('lr')}, seed {ref.get('seed')}. "
        f"Starting cartridge `{Path(str(ref.get('cart_init'))).name}` "
        f"(sha {str(ref.get('cart_sha256'))[:12]}), one saved zero-moment "
        f"optimizer state and one frozen example schedule "
        f"(sha {str(ref.get('schedule_sha256'))[:12]}) shared by every arm. "
        f"Target schema {ref.get('target_schema')}, stored targets flattened "
        f"at synthesis time ({ref.get('flatten_mode')} flattener installed). "
        f"Evaluator {ev.get('evaluator')}.",
        "",
    ]

    if parity:
        L += [
            "## Parity gate",
            "",
            "The legacy objective must equal unique support plus explicit "
            "anchors, in loss and in cartridge gradients, on the real model "
            "before any arm is allowed to train.",
            "",
            "| element | legacy loss | grouped loss | relative | max relative gradient |",
            "|---|---|---|---|---|",
        ]
        for r in parity["results"]:
            L.append(
                f"| {r['element']} | {r['loss_raw']:.6f} | {r['loss_grouped']:.6f} "
                f"| {r['loss_rel']:.2e} | {r['grad_max_rel']:.2e} |"
            )
        L += ["", f"Gate: **{'PASS' if parity['ok'] else 'FAIL'}**", ""]

    if runs:
        cal = ref.get("content_report", {})
        L += [
            "## Calibration",
            "",
            f"Scale-matched scalar {fmt(ref.get('a2_scale'), '.6f')}; "
            f"content-anchor scale {fmt(ref.get('content_scale'), '.6f')} "
            f"matching {cal.get('control_count')} control anchors "
            f"(coefficient mass {fmt(cal.get('control_mass'), '.6g')}) with "
            f"{cal.get('selected_count')} content anchors. Anchor mass is "
            "weighted by each element's own denominator, the way the "
            "objective consumes it.",
            "",
            "## Training",
            "",
            "| arm | first loss | final loss | mean gradient norm | steps clipped |",
            "|---|---|---|---|---|",
        ]
        for arm in ARM_ORDER:
            if arm not in runs:
                continue
            h = runs[arm]["history"]
            gn = [e["grad_norm"] for e in h]
            clipped = sum(1 for e in h if e.get("clipped"))
            L.append(
                f"| {arm} | {h[0]['loss']:.4f} | {h[-1]['loss']:.4f} | "
                f"{sum(gn) / len(gn):.3f} | {clipped}/{len(h)} |"
            )
        L += [
            "",
            "Losses from different objectives are not comparable to each "
            "other; they are listed to show each arm descended.",
            "",
        ]

    L += [
        "## Held-out generation, thinking disabled (primary endpoint)",
        "",
        "| condition | strict acc | invalid | cap hit | standalone | mean len | eot rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in sorted(res):
        p = res[name].get("primary") or {}
        L.append(
            f"| {name} | {fmt(p.get('strict_acc'))} | {fmt(p.get('parser_invalid'), '.2f')} "
            f"| {fmt(p.get('cap_hit'), '.2f')} | {fmt(p.get('standalone_rate'), '.2f')} "
            f"| {fmt(p.get('mean_len'), '.1f')} | {fmt(p.get('eot_rate'), '.2f')} |"
        )

    L += [
        "",
        "## Held-out generation, thinking enabled, cap 256 (stress)",
        "",
        "This is the column comparable to the historical numbers, which "
        "were measured with thinking enabled.",
        "",
        "| condition | strict acc | invalid | unclosed think | cap hit | mean len |",
        "|---|---|---|---|---|---|",
    ]
    for name in sorted(res):
        s = res[name].get("stress") or {}
        L.append(
            f"| {name} | {fmt(s.get('strict_acc'))} | {fmt(s.get('parser_invalid'), '.2f')} "
            f"| {fmt(s.get('unclosed_think'), '.2f')} | {fmt(s.get('cap_hit'), '.2f')} "
            f"| {fmt(s.get('mean_len'), '.1f')} |"
        )

    L += [
        "",
        "## Forced-choice content (does the cartridge know the answer)",
        "",
        "| condition | argmax acc | correct logprob | margin | entropy | eot after letter |",
        "|---|---|---|---|---|---|",
    ]
    for name in sorted(res):
        f_ = res[name].get("forced_choice") or {}
        L.append(
            f"| {name} | {fmt(f_.get('fc_acc'))} | {fmt(f_.get('correct_lp_mean'))} "
            f"| {fmt(f_.get('margin_mean'))} | {fmt(f_.get('entropy_mean'))} "
            f"| {fmt(f_.get('eot_after_letter_mean'))} |"
        )

    L += [
        "",
        "## Control-state probe (held-out elements the schedule never touched)",
        "",
        "| condition | unique loss | anchor loss | first-row loss | p(chosen) first | p(chosen) content | p(eot) natural | grad cos | K/V energy |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in sorted(res):
        pr = res[name].get("probe")
        if not pr:
            continue
        L.append(
            f"| {name} | {fmt(pr.get('loss_unique'), '.4f')} | {fmt(pr.get('loss_anchor'), '.4f')} "
            f"| {fmt(pr.get('loss_first_row'), '.4f')} | {fmt(pr.get('chosen_p_first_mean'))} "
            f"| {fmt(pr.get('chosen_p_content_mean'))} | {fmt(pr.get('eot_p_natural_mean'))} "
            f"| {fmt(pr.get('grad_cosine'))} | {fmt(pr.get('kv_grad_energy_ratio'), '.3f')} |"
        )

    # ---- interpretation -------------------------------------------------
    def final(arm, section="primary", key="strict_acc"):
        cands = [n for n in res if n.startswith(arm + "_step")]
        if not cands:
            return None
        last = max(cands, key=lambda n: int(n.rsplit("step", 1)[1]))
        return (res[last].get(section) or {}).get(key)

    L += ["", "## Reading", ""]
    a0 = final("legacy_raw")
    a0s = final("legacy_raw", "stress")
    start = (res.get("start_step0", {}).get("stress") or {}).get("strict_acc")
    lines = []
    if a0 is None:
        lines.append("The reproduction control did not produce a scored checkpoint.")
    else:
        lines.append(
            f"The reproduction control ends at {a0:.3f} strict with thinking "
            f"disabled and {fmt(a0s)} with thinking enabled; the shared "
            f"starting cartridge scores {fmt(start)} on the comparable "
            "thinking-enabled pass. The historical ten-step continuation "
            "reached 0.50 there, so that is the number the control has to "
            "recover before any arm below carries a mechanism claim."
        )
        for arm in ARM_ORDER[1:]:
            v = final(arm)
            if v is None:
                continue
            d = v - a0
            rel = (
                "matches" if abs(d) < 0.05 else ("beats" if d > 0 else "falls short of")
            )
            lines.append(
                f"`{arm}` ({ARM_MEANING[arm]}) ends at {v:.3f} and {rel} the "
                f"control by {d:+.3f}."
            )
    lines.append(
        "Ordering at twenty questions and one seed is noise unless it is "
        "backed by output behavior (length, invalid rate, end-of-turn) or "
        "the forced-choice margin. Promotion needs seeds 1 and 2."
    )
    L += [f"- {x}" for x in lines]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"CTRL_REPORT {out}")


if __name__ == "__main__":
    main()

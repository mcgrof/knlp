#!/usr/bin/env python3
"""Compare a locally-run tool-use ladder against the published corpus.

Both sides carry a bootstrap 95% interval per cell, and the honest test is
whether those intervals overlap. Asking instead whether the local point
estimate sits inside the published interval is only fair when the local run
used the same sampling; under a lighter sweep the local estimate is coarse
and will fall outside a narrow published interval for reasons that have
nothing to do with reproduction.

That distinction is not academic here. The upstream sweep ships in two
passes: pass one is a single prompt variant at one run per combination,
pass two is five variants at five runs, and the paper's numbers come from
pass two. A pass-one run gives five observations per tier against the
paper's one hundred and twenty five, so it is a smoke test rather than a
reproduction, and it should be read as one.

Overlap is reported first because it is the defensible check. The stricter
point-in-interval result is reported alongside it, because once the sampling
does match it is the sharper question.

Both inputs are the stdout of run_metrics.py. Cells look like

    qwen3_0.6b   80%[73,86]*   44%[35,53]   ...

where the trailing asterisk marks a pass that leaned on a stubbed check.
"""

from __future__ import annotations

import argparse
import re
import sys

CELL = re.compile(r"(\d+)%\[(\d+),(\d+)\]\*?")
TIERS = ("A0", "A", "B", "C", "D", "E", "overall")


def parse(path: str) -> dict[str, list[tuple[int, int, int]]]:
    """Map model name to its per-tier (point, lo, hi) triples."""
    out: dict[str, list[tuple[int, int, int]]] = {}
    in_matrix = False
    for line in open(path, encoding="utf-8", errors="replace"):
        if "TCR Matrix" in line:
            in_matrix = True
            continue
        if in_matrix and line.startswith("=") and out:
            break
        cells = CELL.findall(line)
        if not cells:
            continue
        name = line.split()[0]
        out[name] = [(int(p), int(lo), int(hi)) for p, lo, hi in cells]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("published", help="run_metrics.py output over the released corpus")
    ap.add_argument("local", help="run_metrics.py output over a local re-run")
    args = ap.parse_args()

    pub, loc = parse(args.published), parse(args.local)
    shared = [m for m in loc if m in pub]
    if not shared:
        print("no models in common", file=sys.stderr)
        return 2

    overlap = strict = total = 0
    print(f"{'model':22s} {'tier':>7s} {'local':>14s} {'published':>14s}  overlap  strict")
    print("-" * 78)
    for m in sorted(shared):
        for i, (lp, llo, lhi) in enumerate(loc[m]):
            if i >= len(pub[m]):
                break
            pp, plo, phi = pub[m][i]
            ov = llo <= phi and plo <= lhi
            st = plo <= lp <= phi
            overlap += ov
            strict += st
            total += 1
            tier = TIERS[i] if i < len(TIERS) else str(i)
            print(f"{m:22s} {tier:>7s} {lp:3d}%[{llo:3d},{lhi:3d}] "
                  f"{pp:3d}%[{plo:3d},{phi:3d}]  "
                  f"{'yes' if ov else 'NO ':>7s}  {'yes' if st else 'no':>6s}")

    print("-" * 78)
    print(f"intervals overlap:            {overlap}/{total} "
          f"({100*overlap/total:.0f}%)")
    print(f"local point in published CI:  {strict}/{total} "
          f"({100*strict/total:.0f}%)")
    if overlap < total:
        print("\nNon-overlapping cells are the ones worth explaining. If the local "
              "run\nused lighter sampling than the published sweep, read the strict "
              "column\nas uninformative rather than as a discrepancy.")
    return 0 if overlap == total else 1


if __name__ == "__main__":
    raise SystemExit(main())

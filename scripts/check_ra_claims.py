#!/usr/bin/env python3
"""Guard against stale Reciprocal Attention claims returning to docs.

The RA record was cleaned up once (see docs/ra-evidence.md): the
untraceable 282.1/223.7/50.5 comparison and its 82% headline were
removed, over-broad mechanism claims were relabeled as hypotheses, and
the validated GPT-2 numbers were scoped. This checker fails if public
files reintroduce the forbidden strings, or use the scoped GPT-2
numbers without their model scope and scaling caveat nearby.

Escape hatches:
- a line containing the literal marker "ra-claims-ok" is exempt;
- a forbidden string is allowed when adjacent lines mark it as
  invalidation history (e.g. "invalidated", "could not be traced").

Run from the repo root: python3 scripts/check_ra_claims.py
Exit code 0 = clean, 1 = violations found.
"""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Public documentation surface scanned for forbidden strings.
DOC_GLOBS = ["README.md", "docs/*.md", "docs/*.html"]

# Files allowed to make RA quality claims; the scoped-number rules
# apply only to these.
RA_SURFACE = [
    "README.md",
    "docs/ra.md",
    "docs/ra.html",
    "docs/ra-evidence.md",
    "docs/index.html",
    "docs/FIM.md",
    "docs/hierarchical-tiering.md",
    "docs/bitter7_math.md",
    "docs/tiering.html",
    "docs/fisher_adam.html",
    "docs/marin.html",
    "docs/experiments.md",
    "scripts/plot_ra_comparison.py",
    "fim/reciprocal_attention/LLAMA150M_AUDIT.md",
    "fim/reciprocal_attention/LLAMA1B_AUDIT.md",
]

# Never allowed outside an invalidation-history context.
FORBIDDEN = [
    r"82%\s*better",
    r"50\.5\s*(?:vs\.?|versus)\s*282",
    r"\b282\.1\b",
    r"\b223\.7\b",
    r"RA is a real win",
    r"safe for RA",
    r"best RA target",
    r"use RA when quality is critical",
    r"same FLOP count as standard attention",
    r"bidirectional information flow",
    r"better gradient flow",
    r"flatter optimization landscape",
    r"tops out (?:at|around) (?:~\s*)?150\s*M",
    r"neutral-to-negative",
    r"\bRA scales\b\s*(?:to\b|[.,;:]|$)",
    r"universal RA improvement",
    r"robust at scale",
    r"FIM trace is the universal signal",
]

# Allowed only with model scope nearby and a scaling caveat in range.
SCOPED = [
    r"\b68\.9\b",
    r"\b72\.5\b",
    r"(?<![\d.])5%\s*better",
    r"\+2\s*HellaSwag",
]

MARKER = "ra-claims-ok"
INVALID_CONTEXT = re.compile(
    r"invalid|not be traced|untraceable|removed|excluded|must not|"
    r"forbidden|history|earlier (?:version|public-page)|do not read|"
    r"not the same as|cannot support|not evidence",
    re.IGNORECASE,
)
EIGMAX_CONTEXT = re.compile(
    r"degenerate|invalid|Perron|row-stochastic|bug", re.IGNORECASE
)
SCOPE_NEAR = re.compile(r"GPT-2|small[- ]scale|FineWebEdu", re.IGNORECASE)
LIMIT_NEAR = re.compile(r"\b1\s*B\b|scal(?:e|ing)|noise", re.IGNORECASE)


def _window(lines, idx, radius):
    lo = max(0, idx - radius)
    hi = min(len(lines), idx + radius + 1)
    return "\n".join(lines[lo:hi])


def scan_file(path: Path) -> list:
    rel = path.relative_to(REPO).as_posix()
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    problems = []

    for idx, line in enumerate(lines):
        if MARKER in line:
            continue
        for pat in FORBIDDEN:
            if re.search(pat, line, re.IGNORECASE):
                if INVALID_CONTEXT.search(_window(lines, idx, 3)):
                    continue
                problems.append(
                    f"{rel}:{idx + 1}: forbidden RA claim ({pat!r}): "
                    f"{line.strip()[:100]}"
                )
        # exact_eigmax may appear in docs only inside selector-bug history.
        if rel.startswith("docs/") or rel == "README.md":
            if "exact_eigmax" in line and not EIGMAX_CONTEXT.search(
                _window(lines, idx, 3)
            ):
                problems.append(
                    f"{rel}:{idx + 1}: exact_eigmax outside selector-bug "
                    f"history: {line.strip()[:100]}"
                )

    if rel in RA_SURFACE:
        # A file header that states the scope covers the whole file
        # (e.g. the plot script's provenance docstring).
        header = "\n".join(lines[:30])
        header_scoped = bool(SCOPE_NEAR.search(header))
        header_limited = bool(LIMIT_NEAR.search(header))
        for idx, line in enumerate(lines):
            if MARKER in line:
                continue
            for pat in SCOPED:
                if not re.search(pat, line):
                    continue
                if not header_scoped and not SCOPE_NEAR.search(_window(lines, idx, 10)):
                    problems.append(
                        f"{rel}:{idx + 1}: scoped RA number ({pat!r}) "
                        f"without GPT-2/small-scale/FineWebEdu scope "
                        f"nearby: {line.strip()[:100]}"
                    )
                if not header_limited and not LIMIT_NEAR.search(
                    _window(lines, idx, 25)
                ):
                    problems.append(
                        f"{rel}:{idx + 1}: scoped RA number ({pat!r}) "
                        f"without a scaling/1B caveat in range: "
                        f"{line.strip()[:100]}"
                    )

    # The 1B audit must never regress to "results pending".
    if rel.endswith("LLAMA1B_AUDIT.md"):
        if re.search(r"(?m)^##\s*Results\b[^\n]*\n+\s*Pending", text):
            problems.append(f"{rel}: 1B audit Results section says Pending again")

    return problems


def iter_files():
    seen = set()
    for pattern in DOC_GLOBS:
        for path in sorted(REPO.glob(pattern)):
            if path.is_file() and not path.is_symlink():
                seen.add(path)
    for rel in RA_SURFACE:
        path = REPO / rel
        if path.is_file() and not path.is_symlink():
            seen.add(path)
    return sorted(seen)


def main() -> int:
    problems = []
    for path in iter_files():
        problems.extend(scan_file(path))
    if problems:
        print("Stale Reciprocal Attention claims detected:\n")
        for problem in problems:
            print(f"  {problem}")
        print(
            "\nFix the wording (see docs/ra-evidence.md for what each "
            "result supports), or mark deliberate invalidation-history "
            "mentions with 'ra-claims-ok'."
        )
        return 1
    print(f"check_ra_claims: {len(iter_files())} files clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())

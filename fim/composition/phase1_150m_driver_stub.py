"""Phase 1 150M driver stub.

This script is the launch-control stub for the 2026-03-19 150M FIM gap-closure
phase. It intentionally binds the experiment families to canonical
`knlp-key-results` output roots so paid GPU runs have reproducible paths before
launch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CANONICAL_ROOT = Path("/home/mcgrof/devel/knlp-key-results/paper-fim/150m")


@dataclass(frozen=True)
class RunFamily:
    name: str
    output_dir: Path


RUN_FAMILIES = {
    "pruning": RunFamily("pruning", CANONICAL_ROOT / "pruning"),
    "kvsplice": RunFamily("kvsplice", CANONICAL_ROOT / "kvsplice"),
    "ra": RunFamily("ra", CANONICAL_ROOT / "ra"),
    "composition": RunFamily("composition", CANONICAL_ROOT / "composition"),
    "perf": RunFamily("perf", CANONICAL_ROOT / "perf"),
}


def main() -> None:
    for family in RUN_FAMILIES.values():
        family.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"prepared: {family.name} -> {family.output_dir}")


if __name__ == "__main__":
    main()

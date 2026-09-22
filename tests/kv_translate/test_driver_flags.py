# SPDX-License-Identifier: GPL-2.0
"""Every flag a driver passes must be one the runner accepts.

An editing pass added the uses of three arguments and, because two
string anchors silently failed to match after reformatting, not their
definitions. The file then referenced `args.expect_examples` without
declaring `--expect-examples`, which no unit test noticed because the unit
tests exercise the loss and not the command line. It was noticed by a rented
A100, four seconds into the first training run, after the machine had been
provisioned and one and a half gigabytes had been uploaded to it.

This reads the drivers and the runners and compares them. It is cheap, it
runs on the processor, and it fails in the second it takes rather than in the
minute a machine costs.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
KV = ROOT / "research" / "kv_translate"
DRIVERS = ("measure_eos.sh", "measure_latency.sh")

FLAG = re.compile(r"(--[a-z0-9][a-z0-9-]*)")
RUNNER = re.compile(r"python3\s+(research/kv_translate/[a-z0-9_]+\.py)")


def accepted_flags(runner: Path) -> set[str]:
    src = runner.read_text()
    return set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', src))


def invocations(driver: Path):
    """Each (runner, flags) the driver invokes, following line continuations."""
    text = driver.read_text().replace("\\\n", " ")
    for line in text.splitlines():
        m = RUNNER.search(line)
        if not m:
            continue
        # Strip the runner path so its own hyphens are not read as flags.
        rest = line[m.end() :]
        yield m.group(1), set(FLAG.findall(rest))


@pytest.mark.parametrize("driver_name", DRIVERS)
def test_every_flag_the_driver_passes_is_accepted(driver_name):
    driver = KV / driver_name
    if not driver.exists():
        pytest.skip(f"{driver_name} absent")
    seen = 0
    for rel, flags in invocations(driver):
        runner = ROOT / rel
        assert runner.exists(), f"{driver_name} invokes missing {rel}"
        ok = accepted_flags(runner)
        unknown = sorted(f for f in flags if f not in ok)
        assert (
            not unknown
        ), f"{driver_name} passes {unknown} to {rel}, which does not accept them"
        seen += 1
    assert seen, f"{driver_name} invokes no python runner; the check is vacuous"


def test_no_runner_uses_an_argument_it_never_defines():
    """The other half of the same defect: a use without a definition.

    argparse turns an undefined `--flag` into an error only when someone
    passes it. An `args.something` that no add_argument declares fails later,
    at the line that reads it, which may be well into a run.
    """
    problems = []
    for runner in sorted(KV.glob("*.py")):
        src = runner.read_text()
        if "add_argument" not in src:
            continue
        defined = {f.lstrip("-").replace("-", "_") for f in accepted_flags(runner)}
        # argparse's own attributes, plus names bound from other sources.
        used = set(re.findall(r"\bargs\.([a-z_][a-z0-9_]*)", src))
        missing = sorted(u for u in used if u not in defined)
        if missing:
            problems.append(f"{runner.name}: uses {missing} with no add_argument")
    assert not problems, problems

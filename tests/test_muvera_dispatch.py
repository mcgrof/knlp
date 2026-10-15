# SPDX-License-Identifier: MIT
"""Run Make with recording runners so study selection cannot launch training."""

import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
PRESETS = ("google-muvera-small", "muvera-small")
OTHER_WORKLOADS = {
    "MODEL_GNN": "gnn-run",
    "SUBLIMINAL_LEARNING": "subliminal-run",
    "RESIDUAL_REPRODUCTION": "residual-run",
    "KNLP_REPRODUCE_DECODE": "decode",
    "KNLP_REPRODUCE_CARTRIDGES": "cartridges",
    "KNLP_LMCACHE_SNAPKV": "lmcache-snapkv",
    "CARTRIDGES_CAS": "cas",
    "KV_LINGO": "kv-lingo-check",
    "KNLP_REPRODUCE_KVIO": "kvio",
    "KNLP_REPRODUCE_PD": "pd-all",
    "KVTIDE": "kvtide",
    "KNLP_QUANT_FP8": "quant-fp8",
    "KNLP_TF_REPRO": "tensorflow-repro",
    "KNLP_MODULAR_REPRO": "modular-repro",
}


@pytest.fixture
def workspace(tmp_path):
    for path in REPO.iterdir():
        if path.name not in (".git", ".config", ".config.old", "config.py"):
            (tmp_path / path.name).symlink_to(path, target_is_directory=path.is_dir())
    recorder = tmp_path / "record.py"
    recorder.write_text(
        "import os, sys\n"
        "with open(os.environ['MUVERA_CALLS'], 'a') as output:\n"
        "    output.write(os.path.basename(sys.argv[1]) + '\\n')\n"
    )
    # Stop the generic training recipes if a MUVERA preset falls through.
    # The real MUVERA recipes invoke the recording Python.
    guards = tmp_path / "guards.mk"
    targets = {"check-config", "train", "test-matrix", *OTHER_WORKLOADS.values()}
    guards.write_text(
        "".join(f"{target}:\n\t@echo route:{target}\n" for target in sorted(targets))
    )

    def make(*args):
        return subprocess.run(
            [
                "make",
                "--no-print-directory",
                "-f",
                "Makefile",
                "-f",
                str(guards),
                f"PYTHON={sys.executable}",
                f"MUVERA_PY={sys.executable} {recorder}",
                f"GOOGLE_MUVERA_PY={sys.executable} {recorder}",
                *args,
            ],
            cwd=tmp_path,
            env={**os.environ, "MUVERA_CALLS": str(tmp_path / "calls")},
            capture_output=True,
            text=True,
            timeout=15,
        )

    return tmp_path, make


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("jobs", (1, 4))
@pytest.mark.parametrize("explicit", (False, True))
def test_selected_preset_runs_only_its_pipeline(workspace, preset, jobs, explicit):
    root, make = workspace
    selected = make("defconfig-" + preset)
    assert selected.returncode == 0, selected.stderr
    assert (root / ".config").read_bytes() == (
        REPO / "configs" / ("defconfig-" + preset)
    ).read_bytes()
    result = make(f"-j{jobs}", *([preset] if explicit else []))
    assert result.returncode == 0, result.stderr
    assert "route:" not in result.stdout
    calls = (root / "calls").read_text().splitlines()
    expected = (
        ["run.py"]
        if preset == "google-muvera-small"
        else [
            "fetch_scifact.py",
            "build_colbertv2_embeddings.py",
            "build_bge_embeddings.py",
            "run_systems_byte_bench.py",
            "run_retrieval_bench.py",
            "plot_stage5_results.py",
            "write_report.py",
        ]
    )
    assert sorted(calls) == sorted(expected)


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("other", (*OTHER_WORKLOADS, "MATCHED_MICRO", "MUVERA"))
def test_conflicting_studies_fail_before_any_runner(workspace, preset, other):
    root, make = workspace
    assert make("defconfig-" + preset).returncode == 0
    if other == "MUVERA":
        other = "MUVERA_SMALL" if preset.startswith("google") else "GOOGLE_MUVERA_SMALL"
    with (root / ".config").open("a") as config:
        config.write(f"\nCONFIG_{other}=y\n")
    result = make("-j4")
    assert result.returncode != 0
    assert "Conflicting workloads" in result.stderr
    assert not (root / "calls").exists()
    assert "route:" not in result.stdout


@pytest.mark.parametrize("selector,target", OTHER_WORKLOADS.items())
def test_other_workloads_keep_their_existing_routes(workspace, selector, target):
    root, make = workspace
    (root / ".config").write_text(f"CONFIG_{selector}=y\n")
    result = make("-qp")
    assert result.returncode in (0, 1), result.stderr
    default = re.search(r"^\.DEFAULT_GOAL := (.+)$", result.stdout, re.MULTILINE)[1]
    if default == "all":
        assert re.search(
            rf"^all:.*\b{re.escape(target)}\b", result.stdout, re.MULTILINE
        )
    else:
        assert default == target
    assert not (root / "calls").exists()


@pytest.mark.parametrize(
    "selection,target",
    (
        ("", "train"),
        ("OPTIMIZER_MODE_MULTIPLE", "test-matrix"),
        ("MODEL_MODE_MULTIPLE", "test-matrix"),
    ),
)
def test_generic_defaults_keep_their_existing_routes(workspace, selection, target):
    root, make = workspace
    (root / ".config").write_text(f"CONFIG_{selection}=y\n" if selection else "")
    result = make("-qp")
    assert result.returncode in (0, 1), result.stderr
    assert ".DEFAULT_GOAL := all" in result.stdout
    assert re.search(rf"^all:.*\b{re.escape(target)}\b", result.stdout, re.MULTILINE)
    assert not (root / "calls").exists()

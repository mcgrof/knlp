#!/usr/bin/env python3
"""Keep captured experiment data out of the source index."""

from pathlib import PurePosixPath
import re
import subprocess
import sys


def is_artifact(name):
    path = PurePosixPath(name)
    if name.startswith("docs/") and name.endswith(
        (".json", ".jsonl", ".ndjson", ".csv", ".tsv", ".json.gz")
    ):
        return True
    if name.startswith("tools/kv/native/fixtures/") and name.endswith(
        (".json", ".jsonl", ".gz")
    ):
        return True
    if path.suffix == ".py":
        return False
    return bool(
        name.startswith(
            ("results/", "artifacts/", "runs/", "wandb/", "kvsplice_trained/")
        )
        or re.match(
            r"(?:bpa_v\d+[^/]*(?:results|final)[^/]*|rgsa_[^/]*(?:results|compare)[^/]*)/",
            name,
        )
        or re.match(r"bpa_v\d+_scoreboard\.(?:json|csv|md)$", name)
        or name
        in (
            "fim_traces_marin8b_wikitext.json",
            "gnn/canonical_results_20251220_0335.json",
        )
    )


def main():
    names = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
    historical = (
        subprocess.check_output(
            ["git", "log", "HEAD", "--format=", "--name-only", "-z"]
        )
        .decode()
        .split("\0")
    )
    names = {name.strip("\n") for name in names + historical}
    rejected = sorted(name for name in names if name and is_artifact(name))
    if rejected:
        print("Experiment artifacts are present in the index or reachable history:")
        print("\n".join(rejected))
        print(
            "Archive outputs separately; rebase old branches instead of merging old history."
        )
        return 1
    print("Tracked-artifact check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

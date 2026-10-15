"""Pinned LongBench HotpotQA acquisition and deterministic sampling."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterable

LONGBENCH_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
LONGBENCH_DATA_URL = (
    "https://huggingface.co/datasets/zai-org/LongBench/resolve/"
    f"{LONGBENCH_REVISION}/data.zip"
)
HOTPOT_MEMBER = "data/hotpotqa.jsonl"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_hotpotqa(destination: Path, force: bool = False) -> Path:
    """Download the pinned LongBench archive and extract HotpotQA only."""

    destination = destination.expanduser().resolve()
    if destination.exists() and not force:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive = destination.parent / "longbench-data.zip.part"
    with urllib.request.urlopen(LONGBENCH_DATA_URL) as response:
        with archive.open("wb") as output:
            shutil.copyfileobj(response, output)
    try:
        with zipfile.ZipFile(archive) as bundle:
            with bundle.open(HOTPOT_MEMBER) as source:
                with destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
    finally:
        archive.unlink(missing_ok=True)
    return destination


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = {"context", "input", "answers"} - record.keys()
            if missing:
                raise ValueError(
                    f"{path}:{line_number}: missing keys {sorted(missing)}"
                )
            records.append(record)
    return records


def deterministic_split(
    records: Iterable[dict[str, Any]],
    profile_count: int,
    eval_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return disjoint profile and held-out sets with recorded randomness."""

    if profile_count <= 0 or eval_count <= 0:
        raise ValueError("profile_count and eval_count must be positive")
    materialized = list(records)
    if profile_count + eval_count > len(materialized):
        raise ValueError("requested split is larger than the dataset")
    indices = list(range(len(materialized)))
    random.Random(seed).shuffle(indices)
    profile_indices = indices[:profile_count]
    eval_indices = indices[profile_count : profile_count + eval_count]
    return (
        [materialized[index] for index in profile_indices],
        [materialized[index] for index in eval_indices],
    )

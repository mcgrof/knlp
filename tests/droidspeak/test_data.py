import json
from pathlib import Path

import pytest

from research.droidspeak.data import deterministic_split, load_jsonl


def test_load_and_deterministic_split(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    records = [
        {"context": f"context-{index}", "input": "question", "answers": ["x"]}
        for index in range(6)
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    loaded = load_jsonl(path)
    first = deterministic_split(loaded, profile_count=2, eval_count=3, seed=7)
    second = deterministic_split(loaded, profile_count=2, eval_count=3, seed=7)

    assert first == second
    assert {item["context"] for item in first[0]}.isdisjoint(
        item["context"] for item in first[1]
    )


def test_load_rejects_missing_contract(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"context": "only"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="missing keys"):
        load_jsonl(path)

import json

from research.kv_translate.published import coqa
from research.kv_translate.published.kv_lingo_migration_control import prepare, verify


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def fixtures(tmp_path):
    conversations = []
    raw = []
    native = []
    ownership = []
    for domain in coqa.DOMAINS:
        for suffix in ("b", "a"):
            conversation_id = f"{domain}-{suffix}"
            conversations.append({"id": conversation_id, "source": domain})
            for start in ("4B", "8B"):
                ownership.append(
                    {"conversation_id": conversation_id, "starting_model": start}
                )
                for turn in range(1, 11):
                    base = {
                        "conversation_id": conversation_id,
                        "starting_model": start,
                        "turn": turn,
                    }
                    raw.append({**base, "answer": "raw"})
                    native.append({**base, "answer": "native"})
    conversations_path = tmp_path / "conversations.json"
    conversations_path.write_text(json.dumps(conversations))
    paths = {}
    for name, rows in (
        ("raw", raw),
        ("native", native),
        ("ownership", ownership),
    ):
        paths[name] = tmp_path / f"{name}.jsonl"
        write_jsonl(paths[name], rows)
    return conversations_path, paths


def test_prepare_selects_smallest_id_per_domain_and_verify_is_exact(tmp_path):
    conversations, paths = fixtures(tmp_path)
    expected = tmp_path / "expected"
    manifest = prepare(
        conversations,
        paths["raw"],
        paths["native"],
        paths["ownership"],
        expected,
    )
    assert set(manifest["domains"].values()) == {
        f"{domain}-a" for domain in coqa.DOMAINS
    }
    assert verify(expected, expected)["passed"]


def test_verify_detects_changed_answer(tmp_path):
    conversations, paths = fixtures(tmp_path)
    expected = tmp_path / "expected"
    prepare(
        conversations,
        paths["raw"],
        paths["native"],
        paths["ownership"],
        expected,
    )
    replay = tmp_path / "replay"
    replay.mkdir()
    for name in ("RAW.jsonl", "NATIVE_TRAJECTORY.jsonl", "OWNERSHIP.jsonl"):
        (replay / name).write_bytes((expected / name).read_bytes())
    rows = [
        json.loads(line) for line in (replay / "RAW.jsonl").read_text().splitlines()
    ]
    rows[0]["answer"] = "changed"
    write_jsonl(replay / "RAW.jsonl", rows)
    result = verify(expected, replay)
    assert not result["passed"]
    assert len(result["comparisons"]["RAW.jsonl"]["changed"]) == 1


def test_verify_ignores_hardware_dependent_catch_up_seconds(tmp_path):
    conversations, paths = fixtures(tmp_path)
    expected = tmp_path / "expected"
    prepare(
        conversations,
        paths["raw"],
        paths["native"],
        paths["ownership"],
        expected,
    )
    replay = tmp_path / "replay"
    replay.mkdir()
    for name in ("RAW.jsonl", "NATIVE_TRAJECTORY.jsonl", "OWNERSHIP.jsonl"):
        (replay / name).write_bytes((expected / name).read_bytes())
    rows = [
        json.loads(line) for line in (replay / "RAW.jsonl").read_text().splitlines()
    ]
    rows[0]["catch_up"] = {"spans": 1, "tokens": 12, "seconds": 9.5}
    write_jsonl(replay / "RAW.jsonl", rows)
    expected_rows = [
        json.loads(line) for line in (expected / "RAW.jsonl").read_text().splitlines()
    ]
    expected_rows[0]["catch_up"] = {"spans": 1, "tokens": 12, "seconds": 0.25}
    write_jsonl(expected / "RAW.jsonl", expected_rows)
    assert verify(expected, replay)["passed"]


def test_verify_detects_changed_catch_up_tokens(tmp_path):
    conversations, paths = fixtures(tmp_path)
    expected = tmp_path / "expected"
    prepare(
        conversations,
        paths["raw"],
        paths["native"],
        paths["ownership"],
        expected,
    )
    replay = tmp_path / "replay"
    replay.mkdir()
    for name in ("RAW.jsonl", "NATIVE_TRAJECTORY.jsonl", "OWNERSHIP.jsonl"):
        (replay / name).write_bytes((expected / name).read_bytes())
    rows = [
        json.loads(line) for line in (replay / "RAW.jsonl").read_text().splitlines()
    ]
    rows[0]["catch_up"] = {"spans": 1, "tokens": 13, "seconds": 9.5}
    write_jsonl(replay / "RAW.jsonl", rows)
    expected_rows = [
        json.loads(line) for line in (expected / "RAW.jsonl").read_text().splitlines()
    ]
    expected_rows[0]["catch_up"] = {"spans": 1, "tokens": 12, "seconds": 0.25}
    write_jsonl(expected / "RAW.jsonl", expected_rows)
    assert not verify(expected, replay)["passed"]

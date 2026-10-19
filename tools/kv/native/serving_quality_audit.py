# SPDX-License-Identifier: MIT
"""Audit raw HTTP answers, teacher targets and serial parity without a GPU.

Recompute parser decisions and paired uncertainty from retained response bodies
and worker losses. A quality pass alone never qualifies performance.
"""
import argparse
import collections
import gzip
import hashlib
import json
import math
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tools.kv.prequantized.contract import paired_nll, paired_retrieval
from .statistics import paired


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(text):
    try:
        x = Decimal(text.replace(",", "").replace("$", "").strip())
        return str(x.normalize()) if x.is_finite() else None
    except (InvalidOperation, AttributeError):
        return None


def response_choice(row):
    require(row["error"] is None and row["http_status"] == 200, "HTTP request failed")
    raw = json.loads(row["raw_response"])
    require(len(raw["choices"]) == 1, "Expected one HTTP completion")
    choice = raw["choices"][0]
    require(
        choice["text"] == row["text"]
        and choice["token_ids"] == row["generated_token_ids"]
        and (choice["finish_reason"] == row["finish_reason"]),
        "Saved completion differs from its raw HTTP response",
    )
    require(
        all((type(token) is int and token >= 0 for token in choice["token_ids"])),
        "Invalid output token IDs",
    )
    require(choice["finish_reason"] in ("stop", "length"), "Unexpected finish reason")
    return choice


def natural(root, kind, fixture):
    require(kind in ("gsm8k", "retrieval"), "Unknown scored HTTP task")
    path = root / (kind + "-raw.jsonl")
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    expected = {s["id"]: s for s in fixture}
    require(
        len(rows) == len(expected),
        "HTTP quality evidence failed: len(rows) == len(expected)",
    )
    result = {}
    failures = []
    for row in rows:
        sid = row["id"]
        require(
            sid in expected and sid not in result,
            "HTTP quality evidence failed: sid in expected and sid not in result",
        )
        s = expected[sid]
        require(
            row["prompt_hash"] == s["prompt_hash"]
            and row["gold"] == s.get("gold", s.get("answer")),
            "HTTP quality evidence failed: row['prompt_hash'] == s['prompt_hash'] and row['gold'] == s.get('gold', s.get('answer'))",
        )
        c = response_choice(row)
        if kind == "gsm8k":
            m = re.search("The answer is (\\-?[0-9\\.\\,]+).", c["text"])
            answer = number(m.group(1)) if m else None
        else:
            text = re.split("<\\|(?:im_end|end)\\|>", c["text"])[0]
            answer = re.sub("<\\|[^>]+\\|>", "", text).strip().strip(" \n\t.\"'`")
        correct = answer == row["gold"] and c["finish_reason"] == "stop"
        result[sid] = correct
        if not correct:
            failures.append(
                {
                    "id": sid,
                    "answer": answer,
                    "gold": row["gold"],
                    "finish_reason": c["finish_reason"],
                }
            )
    scored_rows = json.loads((root / (kind + "-scored.json")).read_text())
    require(len(scored_rows) == len(result), "Scored case count differs")
    scored = {r["id"]: r for r in scored_rows}
    require(
        set(scored) == set(result),
        "HTTP quality evidence failed: set(scored) == set(result)",
    )
    for sid in result:
        require(
            scored[sid]["correct"] == result[sid],
            "HTTP quality evidence failed: scored[sid]['correct'] == result[sid]",
        )
    return (
        result,
        {
            "cases": len(result),
            "correct": sum(result.values()),
            "raw_sha256": digest(path),
            "failures": failures,
        },
    )


def teacher(root, samples):
    receipt = root / "server/nll-teacher.json"
    meta = json.load(open(receipt))
    raw = root / "server/nll-teacher.jsonl"
    rows = [json.loads(l) for l in raw.read_text().splitlines()]
    expected = {s["id"]: s for s in samples}
    require(
        len(expected) == 32 and len(rows) == 32 * 129,
        "HTTP quality evidence failed: len(expected) == 32 and len(rows) == 32 * 129",
    )
    registered = {item["sample_id"]: item for item in meta["requests"].values()}
    require(
        len(meta["requests"]) == 32 and set(registered) == set(expected),
        "Teacher registration differs from frozen samples",
    )
    for sid, item in registered.items():
        sample = expected[sid]
        length = sample["length"]
        require(
            item["teacher"] == sample["token_ids"][length : length + 129]
            and item["prompt_token_ids"] == sample["token_ids"][:length],
            "Teacher prompt or targets differ from frozen IDs",
        )
    groups = collections.defaultdict(list)
    receipt_hash = digest(receipt)
    for r in rows:
        require(
            r["provenance_sha256"] == receipt_hash,
            "HTTP quality evidence failed: r['provenance_sha256'] == receipt_hash",
        )
        require(
            math.isfinite(r["nll"]) and r["nll"] >= 0,
            "HTTP quality evidence failed: math.isfinite(r['nll']) and r['nll'] >= 0",
        )
        require(
            r["sample_id"] in expected,
            "HTTP quality evidence failed: r['sample_id'] in expected",
        )
        groups[r["sample_id"]].append(r)
    require(
        set(groups) == set(expected),
        "HTTP quality evidence failed: set(groups) == set(expected)",
    )
    means = {}
    requestids = set()
    for sid, rs in groups.items():
        require(
            len(rs) == 129 and [r["step"] for r in rs] == list(range(129)),
            "HTTP quality evidence failed: len(rs) == 129 and [r['step'] for r in rs] == list(range(129))",
        )
        s = expected[sid]
        require(
            [r["target"] for r in rs]
            == s["token_ids"][s["length"] : s["length"] + 129],
            "HTTP quality evidence failed: [r['target'] for r in rs] == s['token_ids'][s['length']:s['length'] + 129]",
        )
        ids = {r["request_id"] for r in rs}
        require(
            len(ids) == 1 and (not ids & requestids),
            "HTTP quality evidence failed: len(ids) == 1 and (not ids & requestids)",
        )
        requestids |= ids
        means[sid] = sum((r["nll"] for r in rs[1:])) / 128
    http = [json.loads(l) for l in (root / "nll.jsonl").read_text().splitlines()]
    require(
        len(http) == 32 and len({r["id"] for r in http}) == 32,
        "HTTP quality evidence failed: len(http) == 32 and len({r['id'] for r in http}) == 32",
    )
    for r in http:
        response_choice(r)
        s = expected[r["id"]]
        require(
            r["error"] is None
            and r["generated_token_ids"]
            == s["token_ids"][s["length"] : s["length"] + 129],
            "HTTP quality evidence failed: r['error'] is None and r['generated_token_ids'] == s['token_ids'][s['length']:s['length'] + 129]",
        )
    end = json.load(open(root / "nll-teacher-end.json"))["results"]
    require(
        len(end) == 1 and end[0]["complete"] and (set(end[0]["counts"]) == requestids),
        "HTTP quality evidence failed: len(end) == 1 and end[0]['complete'] and (set(end[0]['counts']) == requestids)",
    )
    require(
        all((count == 129 for count in end[0]["counts"].values())),
        "Teacher completion counts differ from raw coverage",
    )
    return (
        means,
        {
            "rows": len(rows),
            "scored_cached_decode_steps_per_document": 128,
            "prefill_step_zero_excluded": True,
            "documents": len(means),
            "receipt_sha256": digest(receipt),
            "raw_sha256": digest(raw),
        },
    )


def parity(root, reference, samples):
    rows = [json.loads(l) for l in (root / "parity.jsonl").read_text().splitlines()]
    require(
        len(rows) == 32 and len({r["id"] for r in rows}) == 32,
        "HTTP quality evidence failed: len(rows) == 32 and len({r['id'] for r in rows}) == 32",
    )
    expected = {s["id"]: s for s in samples}
    ref = json.load(open(reference / "generation/generation-0.json"))
    ref2 = json.load(open(reference / "generation/generation-1.json"))
    a = {tuple(r["prompt_token_ids"]): r["output_token_ids"] for r in ref}
    b = {tuple(r["prompt_token_ids"]): r["output_token_ids"] for r in ref2}
    require(len(ref) == len(ref2) == 32, "Reference case count differs")
    require(
        len(a) == 32 and set(a) == set(b),
        "HTTP quality evidence failed: len(a) == 32 and set(a) == set(b)",
    )
    bad = []
    for r in rows:
        response_choice(r)
        s = expected[r["id"]]
        key = tuple(s["token_ids"][: s["length"]])
        tokens = r["generated_token_ids"]
        require(len(tokens) == 128, "HTTP quality evidence failed: len(tokens) == 128")
        if tokens != a[key] or a[key] != b[key]:
            bad.append(r["id"])
    return {
        "cases": 32,
        "matches": 32 - len(bad),
        "mismatched_sample_ids": bad,
        "passed": not bad,
        "http_sha256": digest(root / "parity.jsonl"),
        "references_sha256": [
            digest(reference / "generation" / f"generation-{i}.json") for i in range(2)
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--format", choices=("awq", "fp16"), required=True)
    p.add_argument("--prefix", help="Artifact name prefix; defaults to weight format")
    p.add_argument("--length", type=int, choices=(1024, 8192), default=8192)
    p.add_argument("--cache-bytes", type=int)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    arms = {}
    audit = {}
    for arm in ["candidate", "native_fi", "native_fa3"]:
        prefix = a.prefix or a.format
        root = a.root / f"{prefix}-{arm}-http-quality"
        command = json.load(open(root / "command.json"))
        require(command["length"] == a.length, "Captured quality length differs")
        require(
            json.load(open(root / "DONE.json"))["capture_complete"],
            "HTTP quality evidence failed: json.load(open(root / 'DONE.json'))['capture_complete']",
        )
        inputs = root / "server/inputs"
        protocol = json.load(open(inputs / "protocol.json"))
        if a.cache_bytes is not None:
            require(
                protocol["engine"]["kv_cache_memory_bytes"] == a.cache_bytes,
                "Captured quality cache budget differs",
            )
        manifest = json.loads(gzip.decompress((inputs / "qwen.json.gz").read_bytes()))
        task = json.load(open(inputs / "gsm8k-frozen.json"))
        nll = [
            s
            for s in manifest["samples"]
            if s["kind"] == "nll" and s["length"] == a.length and (not s.get("reserve"))
        ]
        ret = [
            s
            for s in manifest["samples"]
            if s["kind"] == "retrieval"
            and s["length"] == a.length
            and (not s.get("reserve"))
        ]
        require(
            len(ret) == 100 and len(task["samples"]) == 1319,
            "HTTP quality evidence failed: len(ret) == 100 and len(task['samples']) == 1319",
        )
        require(
            task["strict_regex"] == "The answer is (\\-?[0-9\\.\\,]+).",
            "HTTP quality evidence failed: task['strict_regex'] == 'The answer is (\\\\-?[0-9\\\\.\\\\,]+).'",
        )
        nn, na = teacher(root, nll)
        rr, ra = natural(root, "retrieval", ret)
        gg, ga = natural(root, "gsm8k", task["samples"])
        pa = parity(root, a.root / f"{prefix}-{arm}-sequential-reference", nll)
        arms[arm] = {"nll": nn, "retrieval": rr, "gsm8k": gg}
        audit[arm] = {
            "nll": na,
            "retrieval": ra,
            "gsm8k": ga,
            "parity": pa,
            "manifest_sha256": digest(inputs / "qwen.json.gz"),
            "scales_sha256": digest(inputs / "scales.json"),
            "protocol_sha256": digest(inputs / "protocol.json"),
        }
    comparisons = {}
    for baseline in ["native_fi", "native_fa3"]:
        nn = paired_nll(arms["candidate"]["nll"], arms[baseline]["nll"])
        rr = paired_retrieval(
            arms["candidate"]["retrieval"], arms[baseline]["retrieval"]
        )
        gg = paired(arms["candidate"]["gsm8k"], arms[baseline]["gsm8k"])
        floors = (
            sum(arms[baseline]["retrieval"].values()) / 100 >= 0.8
            and sum(arms[baseline]["gsm8k"].values()) / 1319 >= 0.5
        )
        comparisons[baseline] = {
            "nll": nn,
            "retrieval": rr,
            "gsm8k": gg,
            "native_accuracy_floors_passed": floors,
            "preservation_passed": floors
            and nn["supported_margin"]
            and rr["supported_margin"]
            and gg["supported_margin"],
        }
    result = {
        "format": a.format,
        "length": a.length,
        "audit": audit,
        "comparisons": comparisons,
        "quality_passed": all((x["preservation_passed"] for x in comparisons.values())),
        "controlled_parity_passed": all(
            (x["parity"]["passed"] for x in audit.values())
        ),
        "performance_qualified": False,
        "reader_sha256": digest(Path(__file__)),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "audit"}, indent=2))


if __name__ == "__main__":
    main()

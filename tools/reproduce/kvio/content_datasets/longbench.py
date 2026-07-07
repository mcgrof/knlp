# SPDX-License-Identifier: MIT
"""LongBench loader + normalizer (long-context single-turn).

LongBench (``THUDM/LongBench``) records carry a long ``context`` and an
``input`` question about it.  Because the goal is a shared long-document
prefix reused across several questions, ``load_longbench`` GROUPS records by
their context so each normalized record is ``{"doc_id", "document",
"questions": [...]}`` -- one document, many questions.  The pure
``normalize_longbench`` / grouping helpers need no network and are what the
unit tests exercise on tiny fixtures.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

HF_NAME = "THUDM/LongBench"
CACHE_REL = "longbench_normalized.json"


def _doc_id(document: str) -> str:
    return hashlib.sha256(document.encode("utf-8")).hexdigest()[:16]


def normalize_longbench(record: dict) -> dict:
    """Normalize one raw LongBench record to a single-document record.

    A raw record has ``context`` (long document) and ``input`` (a question).
    Returns ``{"doc_id", "document", "questions": [question]}`` -- a
    one-question record; ``group_longbench`` merges records sharing a document.
    """
    document = str(record.get("context") or record.get("document") or "")
    question = str(record.get("input") or record.get("question") or "")
    questions = [question] if question else []
    return {
        "doc_id": str(record.get("doc_id") or _doc_id(document)),
        "document": document,
        "questions": questions,
    }


def group_longbench(records: list[dict]) -> list[dict]:
    """Merge normalized single-question records that share a document.

    Preserves first-seen document order and appends questions in encounter
    order, so each output record is ``{"doc_id", "document", "questions"}``
    with the shared long-context prefix reused across its questions.
    """
    order: list[str] = []
    by_doc: dict[str, dict] = {}
    for r in records:
        did = r["doc_id"]
        if did not in by_doc:
            by_doc[did] = {
                "doc_id": did,
                "document": r["document"],
                "questions": [],
            }
            order.append(did)
        by_doc[did]["questions"].extend(r.get("questions", []))
    return [by_doc[d] for d in order]


def load_longbench(
    run_dir: str | Path,
    *,
    config: str = "narrativeqa",
    split: str = "test",
    max_documents: Optional[int] = None,
    revision: str = "",
) -> tuple[list[dict], str]:
    """Load + normalize + group LongBench records with a run-dir cache.

    Returns ``(records, reason)``.  ``reason`` is empty on success; on a
    graceful skip (dataset not accessible / offline) ``records`` is ``[]`` and
    ``reason`` explains why.  Never raises for a missing dataset.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cache = run_dir / CACHE_REL
    if cache.exists():
        try:
            data = json.loads(cache.read_text())
            return data, ""
        except (json.JSONDecodeError, OSError) as e:
            log.warning("ignoring unreadable LongBench cache %s: %s", cache, e)

    try:
        from datasets import load_dataset
    except ImportError as e:
        return [], f"datasets not importable: {e}"

    try:
        kw = {"split": split}
        if revision:
            kw["revision"] = revision
        ds = load_dataset(HF_NAME, config, **kw)
    except Exception as e:  # not accessible / offline -> graceful skip
        return [], f"LongBench not accessible ({HF_NAME}/{config}): {e}"

    raw = [normalize_longbench(row) for row in ds]
    grouped = group_longbench(raw)
    if max_documents is not None:
        grouped = grouped[:max_documents]

    try:
        cache.write_text(json.dumps(grouped))
    except OSError as e:
        log.warning("could not write LongBench cache %s: %s", cache, e)
    return grouped, ""

# SPDX-License-Identifier: MIT
"""LMSYS-Chat-1M loader + normalizer (multi-turn conversations).

LMSYS-Chat-1M (``lmsys/lmsys-chat-1m``) is HF-GATED (requires accepting terms
+ an authenticated token).  This loader therefore GRACEFULLY SKIPS -- returning
an empty list and a reason -- when the dataset is not accessible, and never
attempts to force a download.  The pure ``normalize_lmsys`` function needs no
network and is what the unit tests exercise on tiny fixtures.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

HF_NAME = "lmsys/lmsys-chat-1m"
CACHE_REL = "lmsys_normalized.json"


def normalize_lmsys(record: dict) -> dict:
    """Normalize one LMSYS record to ``{"conversation_id", "turns"}``.

    LMSYS records carry a ``conversation`` (list of ``{"role", "content"}``)
    and a ``conversation_id``.  We keep the role/content of each turn verbatim
    and preserve turn order.  Unknown / missing fields degrade to empty.
    """
    conv = record.get("conversation") or record.get("turns") or []
    turns = []
    for t in conv:
        if not isinstance(t, dict):
            continue
        turns.append(
            {"role": str(t.get("role", "")), "content": str(t.get("content", ""))}
        )
    return {
        "conversation_id": str(record.get("conversation_id", "")),
        "turns": turns,
    }


def load_lmsys(
    run_dir: str | Path,
    *,
    split: str = "train",
    max_conversations: Optional[int] = None,
    revision: str = "",
) -> tuple[list[dict], str]:
    """Load + normalize LMSYS conversations with an idempotent run-dir cache.

    Returns ``(records, reason)``.  ``reason`` is empty on success; on a
    graceful skip (gated dataset, no auth, offline) ``records`` is ``[]`` and
    ``reason`` explains why.  Never raises for a missing/gated dataset.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cache = run_dir / CACHE_REL
    if cache.exists():
        try:
            data = json.loads(cache.read_text())
            return data, ""
        except (json.JSONDecodeError, OSError) as e:
            log.warning("ignoring unreadable LMSYS cache %s: %s", cache, e)

    try:
        from datasets import load_dataset
    except ImportError as e:
        return [], f"datasets not importable: {e}"

    try:
        kw = {"split": split}
        if revision:
            kw["revision"] = revision
        ds = load_dataset(HF_NAME, **kw)
    except Exception as e:  # gated/no-auth/offline -> graceful skip
        return [], f"LMSYS not accessible ({HF_NAME}, likely HF-gated): {e}"

    records: list[dict] = []
    for i, row in enumerate(ds):
        if max_conversations is not None and i >= max_conversations:
            break
        records.append(normalize_lmsys(row))

    try:
        cache.write_text(json.dumps(records))
    except OSError as e:
        log.warning("could not write LMSYS cache %s: %s", cache, e)
    return records, ""

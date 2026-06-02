#!/usr/bin/env python3
"""fetch_scifact.py — load BEIR/scifact via HF datasets and cache it.

This is purely a fetch/cache step; the actual encoding happens in the
build_*_embeddings.py scripts. Idempotent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import _muvera_config as cf

CACHE_REL = "scifact_raw.json"


def main():
    ap = cf.standard_argparse(__doc__)
    args = ap.parse_args()
    cfg = cf.parse_kconfig(Path(args.config))
    if not cf.kbool(cfg, "CONFIG_MUVERA_DATASET_SCIFACT", True):
        print("CONFIG_MUVERA_DATASET_SCIFACT=n; skipping fetch")
        return

    run_dir = cf.get_run_dir(cfg)
    cf.stamp_environment(cfg, run_dir)
    cache = run_dir / CACHE_REL
    if cache.exists():
        print(f"already cached: {cache}")
        return

    print("loading BEIR/scifact via HF datasets…")
    from datasets import load_dataset
    corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")
    queries = load_dataset("BeIR/scifact", "queries", split="queries")
    qrels = load_dataset("BeIR/scifact-qrels", split="test")

    out = {
        "corpus": [{"id": r["_id"], "title": r["title"], "text": r["text"]}
                    for r in corpus],
        "queries": [{"id": r["_id"], "text": r["text"]} for r in queries],
        "qrels_test": [{"query-id": str(r["query-id"]),
                          "corpus-id": str(r["corpus-id"]),
                          "score": int(r["score"])} for r in qrels],
    }
    import json
    cache.write_text(json.dumps(out))
    print(f"  corpus N = {len(out['corpus'])}")
    print(f"  queries N = {len(out['queries'])}")
    print(f"  qrels test N = {len(out['qrels_test'])}")
    print(f"  wrote {cache}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()

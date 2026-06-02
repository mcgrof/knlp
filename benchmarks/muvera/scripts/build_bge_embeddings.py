#!/usr/bin/env python3
"""build_bge_embeddings.py — encode scifact corpus + queries with BGE-small
(single-vector pooled, normalized). Used as the tiny-vector baseline."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import _muvera_config as cf

CACHE_REL = "bge_embeddings.pt"


def main():
    ap = cf.standard_argparse(__doc__)
    args = ap.parse_args()
    cfg = cf.parse_kconfig(Path(args.config))
    if not cf.kbool(cfg, "CONFIG_MUVERA_ENABLE_BGE_SMALL", True):
        print("CONFIG_MUVERA_ENABLE_BGE_SMALL=n; skipping BGE encode")
        return

    run_dir = cf.get_run_dir(cfg)
    cf.stamp_environment(cfg, run_dir)
    out_path = run_dir / CACHE_REL
    if out_path.exists():
        print(f"already encoded: {out_path}")
        return

    raw_path = run_dir / "scifact_raw.json"
    if not raw_path.exists():
        print(f"missing {raw_path}; run fetch_scifact.py first", file=sys.stderr)
        sys.exit(1)
    raw = json.loads(raw_path.read_text())

    corpus_texts = [(r["title"] + " " if r["title"] else "") + r["text"]
                     for r in raw["corpus"]]
    cid_list = [r["id"] for r in raw["corpus"]]
    qid_to_text = {r["id"]: r["text"] for r in raw["queries"]}
    test_qids = sorted(set(r["query-id"] for r in raw["qrels_test"]))
    test_query_texts = [qid_to_text[qid] for qid in test_qids if qid in qid_to_text]
    test_qids_used = [qid for qid in test_qids if qid in qid_to_text]

    device = cfg.get("CONFIG_MUVERA_DEVICE") or "cuda:0"
    model_id = cfg.get("CONFIG_MUVERA_BGE_MODEL") or "BAAI/bge-small-en-v1.5"

    print(f"loading BGE ({model_id}) on {device}…")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_id, device=device)

    print(f"encoding {len(corpus_texts)} corpus + {len(test_query_texts)} queries…")
    t0 = time.perf_counter()
    doc_emb = model.encode(corpus_texts, batch_size=128,
                            convert_to_numpy=True, normalize_embeddings=True,
                            show_progress_bar=False)
    q_emb = model.encode(test_query_texts, batch_size=128,
                          convert_to_numpy=True, normalize_embeddings=True,
                          show_progress_bar=False)
    print(f"  {time.perf_counter()-t0:.0f}s")

    torch.save({
        "doc_emb": doc_emb, "q_emb": q_emb,
        "cid_list": cid_list, "test_qids": test_qids_used,
        "qrels_test": raw["qrels_test"], "model": model_id,
    }, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()

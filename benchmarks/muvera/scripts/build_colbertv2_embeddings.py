#!/usr/bin/env python3
"""build_colbertv2_embeddings.py — encode scifact corpus + queries with
ColBERTv2 (via pylate). Caches multi-vector tensors to disk."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

import _muvera_config as cf

CACHE_REL = "colbertv2_embeddings.pt"


def main():
    ap = cf.standard_argparse(__doc__)
    args = ap.parse_args()
    cfg = cf.parse_kconfig(Path(args.config))
    if not cf.kbool(cfg, "CONFIG_MUVERA_ENABLE_COLBERTV2", True):
        print("CONFIG_MUVERA_ENABLE_COLBERTV2=n; skipping ColBERTv2 encode")
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
    cid_to_idx = {c: i for i, c in enumerate(cid_list)}
    qid_to_text = {r["id"]: r["text"] for r in raw["queries"]}
    test_qids = sorted(set(r["query-id"] for r in raw["qrels_test"]))
    test_query_texts = [qid_to_text[qid] for qid in test_qids if qid in qid_to_text]
    test_qids_used = [qid for qid in test_qids if qid in qid_to_text]

    device = cfg.get("CONFIG_MUVERA_DEVICE") or "cuda:0"
    model_id = cfg.get("CONFIG_MUVERA_COLBERTV2_MODEL") or "lightonai/colbertv2.0"

    print(f"loading ColBERTv2 ({model_id}) on {device}…")
    from pylate import models
    model = models.ColBERT(model_name_or_path=model_id, device=device)

    print(f"encoding {len(corpus_texts)} corpus passages…")
    t0 = time.perf_counter()
    doc_embs = model.encode(corpus_texts, batch_size=32, is_query=False,
                              show_progress_bar=False, convert_to_tensor=True)
    doc_embs = [t.cpu() for t in doc_embs]
    print(f"  {time.perf_counter()-t0:.0f}s")

    print(f"encoding {len(test_query_texts)} test queries…")
    t0 = time.perf_counter()
    q_embs = model.encode(test_query_texts, batch_size=32, is_query=True,
                            show_progress_bar=False, convert_to_tensor=True)
    q_embs = [t.cpu() for t in q_embs]
    print(f"  {time.perf_counter()-t0:.0f}s")

    torch.save({
        "doc_embs": doc_embs, "q_embs": q_embs,
        "cid_list": cid_list, "cid_to_idx": cid_to_idx,
        "test_qids": test_qids_used,
        "qrels_test": raw["qrels_test"],
        "model": model_id,
    }, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()

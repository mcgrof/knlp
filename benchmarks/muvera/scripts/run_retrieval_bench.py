#!/usr/bin/env python3
"""run_retrieval_bench.py — MUVERA FDE + pooled retrieval benchmark.

Reads the cached ColBERTv2 / BGE embeddings produced by build_*_embeddings.py.
Computes:
  - exact Chamfer multi-vector oracle (ColBERTv2 if enabled)
  - pooled mean baseline at fp32/fp16/int8 (BGE if enabled, else
    pooled-mean-of-ColBERT)
  - FDE sweep (R, k_sim) × {fp32, fp16, int8, PQ}

Writes retrieval_frontier.csv into the run dir.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import _muvera_config as cf


def quantize_fp16(x): return x.astype(np.float16).astype(np.float32)


def quantize_int8(x):
    qmax = 127
    s = max(np.abs(x).max(), 1e-8) / qmax
    return ((x / s).round().clip(-qmax, qmax) * s).astype(np.float32)


def quantize_pq(docs, m_subq=256, nbits=8):
    import faiss
    n, d = docs.shape
    if d % m_subq != 0:
        for m_try in [m_subq, 128, 64, 32, 16, 8, 4]:
            if d % m_try == 0: m_subq = m_try; break
    pq = faiss.ProductQuantizer(d, m_subq, nbits)
    n_train = max(int(n * 0.5), 100)
    rng = np.random.default_rng(0)
    train_idx = rng.choice(n, size=min(n_train, n), replace=False)
    pq.train(docs[train_idx].astype(np.float32))
    codes = pq.compute_codes(docs.astype(np.float32))
    decoded = pq.decode(codes)
    bytes_per_vec = m_subq * (nbits // 8) if nbits % 8 == 0 else (m_subq * nbits + 7) // 8
    return decoded, bytes_per_vec, m_subq


def build_fde_torch(token_emb, hyperplanes, R, k_sim, dim):
    """token_emb: [n_tokens, dim]; hyperplanes: [R, k_sim, dim]."""
    n_buckets = 2 ** k_sim
    fde = torch.zeros((R, n_buckets, dim), dtype=torch.float32, device=token_emb.device)
    norms = token_emb.norm(dim=-1)
    valid = token_emb[norms > 1e-6]
    if valid.size(0) == 0:
        return fde.reshape(-1)
    for r in range(R):
        signs = (valid @ hyperplanes[r].T) > 0
        bits = signs.long()
        bucket = torch.zeros(valid.size(0), dtype=torch.long, device=valid.device)
        for k in range(k_sim):
            bucket = bucket | (bits[:, k] << k)
        fde[r].index_add_(0, bucket, valid)
    return fde.reshape(-1)


def recall_at_k(topk_lists, qrels_pos, ks=(10, 100)):
    out = {}
    for k in ks:
        n = sum(1 for p in qrels_pos if p)
        hits = 0
        for top, pos in zip(topk_lists, qrels_pos):
            if not pos: continue
            if any(int(idx) in pos for idx in top[:k]): hits += 1
        out[k] = hits / max(1, n)
    return out


def topk_via_argpartition(scores, k=100):
    n = scores.shape[1]
    k_eff = min(k, n - 1)
    top = np.argpartition(-scores, k_eff, axis=1)[:, :k]
    # Sort each row's topk
    return np.take_along_axis(top,
              np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1), axis=1)


def main():
    ap = cf.standard_argparse(__doc__)
    args = ap.parse_args()
    cfg = cf.parse_kconfig(Path(args.config))
    if not cf.kbool(cfg, "CONFIG_MUVERA_RETRIEVAL_BENCH", True):
        print("CONFIG_MUVERA_RETRIEVAL_BENCH=n; skipping")
        return

    run_dir = cf.get_run_dir(cfg)
    cf.stamp_environment(cfg, run_dir)
    device = cfg.get("CONFIG_MUVERA_DEVICE") or "cuda:0"
    use_pq = cf.kbool(cfg, "CONFIG_MUVERA_ENABLE_FAISS_PQ", True)

    rows = []

    # Build qrels positives
    colbert_path = run_dir / "colbertv2_embeddings.pt"
    bge_path = run_dir / "bge_embeddings.pt"
    cb = torch.load(colbert_path, map_location="cpu", weights_only=False) if colbert_path.exists() else None
    bg = torch.load(bge_path, map_location="cpu", weights_only=False) if bge_path.exists() else None
    if cb is None and bg is None:
        print("no embeddings found; run build_*_embeddings.py first", file=sys.stderr)
        sys.exit(1)

    src = cb if cb is not None else bg
    cid_to_idx = {c: i for i, c in enumerate(src["cid_list"])}
    test_qids = src["test_qids"]
    qid_to_pos = defaultdict(set)
    for r in src["qrels_test"]:
        if int(r["score"]) > 0:
            cid = r["corpus-id"]
            if cid in cid_to_idx:
                qid_to_pos[r["query-id"]].add(cid_to_idx[cid])
    qrels_pos = [qid_to_pos[qid] for qid in test_qids]

    # 1. Chamfer oracle (ColBERTv2 if available)
    if cb is not None:
        doc_embs = [t.to(device) for t in cb["doc_embs"]]
        q_embs = [t.to(device) for t in cb["q_embs"]]
        dim = doc_embs[0].size(-1)
        print(f"\n[1] Chamfer oracle ({len(doc_embs)} docs × {len(q_embs)} queries, dim={dim})…")
        t0 = time.perf_counter()
        topks = []
        for qi, q in enumerate(q_embs):
            q_norms = q.norm(dim=-1)
            qv = q[q_norms > 1e-6]
            if qv.size(0) == 0:
                topks.append(np.arange(min(100, len(doc_embs)), dtype=np.int64)); continue
            scores = np.zeros(len(doc_embs), dtype=np.float32)
            for di, d in enumerate(doc_embs):
                sim = qv @ d.T
                scores[di] = sim.max(dim=-1).values.sum().item()
            top = np.argpartition(-scores, min(100, len(scores)-1))[:100]
            top = top[np.argsort(-scores[top])]
            topks.append(top)
        lat = (time.perf_counter() - t0) / len(q_embs) * 1e6
        r = recall_at_k(topks, qrels_pos)
        bytes_avg = sum(d.numel() for d in cb["doc_embs"]) * 4 / len(cb["doc_embs"])
        print(f"  recall@10={r[10]:.3f}  recall@100={r[100]:.3f}  lat={lat:.1f} μs/q  "
              f"bytes/vec≈{bytes_avg:.0f}")
        rows.append({"variant": "chamfer_oracle", "model": cb["model"],
                     "dataset": "BeIR/scifact", "R": "-", "k_sim": "-",
                     "FDE_dim": "-", "precision": "fp32",
                     "bytes_per_vector": int(bytes_avg),
                     "recall_at_10": r[10], "recall_at_100": r[100],
                     "lat_us_per_query": lat, "notes": "ColBERTv2 multi-vector late interaction"})

    # 2. Pooled mean baseline
    pool_doc_emb = pool_q_emb = None
    pool_model_id = pool_dim = None
    if bg is not None:
        pool_doc_emb = bg["doc_emb"]; pool_q_emb = bg["q_emb"]
        pool_model_id = bg["model"]; pool_dim = pool_doc_emb.shape[1]
    elif cb is not None:
        # Pool ColBERTv2 token embeddings as a fallback baseline
        pool_doc = []
        for d in cb["doc_embs"]:
            n = d.norm(dim=-1)
            v = d[n > 1e-6]
            p = v.mean(0) if v.size(0) > 0 else torch.zeros(d.size(-1))
            pool_doc.append((p / p.norm().clamp_min(1e-8)).numpy())
        pool_doc_emb = np.stack(pool_doc)
        pool_q = []
        for q in cb["q_embs"]:
            n = q.norm(dim=-1)
            v = q[n > 1e-6]
            p = v.mean(0) if v.size(0) > 0 else torch.zeros(q.size(-1))
            pool_q.append((p / p.norm().clamp_min(1e-8)).numpy())
        pool_q_emb = np.stack(pool_q)
        pool_model_id = "ColBERTv2 token-mean"
        pool_dim = pool_doc_emb.shape[1]

    print(f"\n[2] pooled mean baseline (model={pool_model_id}, dim={pool_dim})…")
    for prec in ("fp32", "fp16", "int8"):
        if prec == "fp16": d_q = quantize_fp16(pool_doc_emb)
        elif prec == "int8": d_q = quantize_int8(pool_doc_emb)
        else: d_q = pool_doc_emb
        t0 = time.perf_counter()
        scores = pool_q_emb @ d_q.T
        topks = topk_via_argpartition(scores, 100)
        lat = (time.perf_counter() - t0) / len(pool_q_emb) * 1e6
        r = recall_at_k(topks.tolist(), qrels_pos)
        bpv = pool_dim * (4 if prec == "fp32" else 2 if prec == "fp16" else 1)
        print(f"  pooled-{prec}: recall@10={r[10]:.3f} recall@100={r[100]:.3f} "
              f"bytes={bpv} lat={lat:.1f} μs/q")
        rows.append({"variant": f"pooled-{prec}", "model": pool_model_id,
                     "dataset": "BeIR/scifact", "R": "-", "k_sim": "-",
                     "FDE_dim": pool_dim, "precision": prec,
                     "bytes_per_vector": bpv,
                     "recall_at_10": r[10], "recall_at_100": r[100],
                     "lat_us_per_query": lat,
                     "notes": "single-vector mean-pool"})

    # 3. FDE sweep — only when ColBERTv2 multi-vector is available
    if cb is not None:
        Rs = cf.parse_int_list(cfg.get("CONFIG_MUVERA_FDE_R_VALUES", "1 5 10"))
        ks = cf.parse_int_list(cfg.get("CONFIG_MUVERA_FDE_KSIM_VALUES", "2 3 4"))
        formats = cf.parse_str_list(cfg.get("CONFIG_MUVERA_STORAGE_FORMATS",
            "fp32 fp16 int8"))
        if use_pq: formats = list(formats) + ["pq"]
        rng = np.random.default_rng(20260502)
        print(f"\n[3] FDE sweep R={Rs} k_sim={ks} formats={formats}…")
        for R in Rs:
            for k_sim in ks:
                n_buckets = 2 ** k_sim
                fde_dim = R * n_buckets * dim
                hp = torch.from_numpy(rng.standard_normal(size=(R, k_sim, dim)).astype(np.float32))
                hp = hp / hp.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                hp = hp.to(device)
                doc_fdes = np.zeros((len(doc_embs), fde_dim), dtype=np.float32)
                for i, d in enumerate(doc_embs):
                    doc_fdes[i] = build_fde_torch(d, hp, R, k_sim, dim).cpu().numpy()
                q_fdes = np.zeros((len(q_embs), fde_dim), dtype=np.float32)
                for i, q in enumerate(q_embs):
                    q_fdes[i] = build_fde_torch(q, hp, R, k_sim, dim).cpu().numpy()
                doc_n = doc_fdes / np.linalg.norm(doc_fdes, axis=-1, keepdims=True).clip(1e-8)
                q_n = q_fdes / np.linalg.norm(q_fdes, axis=-1, keepdims=True).clip(1e-8)

                for prec in formats:
                    if prec == "fp16": d_q = quantize_fp16(doc_n); bpv = fde_dim * 2
                    elif prec == "int8": d_q = quantize_int8(doc_n); bpv = fde_dim
                    elif prec == "pq":
                        try:
                            d_q, bpv, _ = quantize_pq(doc_n, m_subq=256, nbits=8)
                        except Exception as e:
                            print(f"  PQ skip ({e}) for dim={fde_dim}")
                            continue
                    else: d_q = doc_n; bpv = fde_dim * 4
                    t0 = time.perf_counter()
                    scores = q_n @ d_q.T
                    topks = topk_via_argpartition(scores, 100)
                    lat = (time.perf_counter() - t0) / len(q_n) * 1e6
                    rk = recall_at_k(topks.tolist(), qrels_pos)
                    print(f"  FDE R={R} k={k_sim} dim={fde_dim:>6} {prec:>4}: "
                          f"r@10={rk[10]:.3f} r@100={rk[100]:.3f} "
                          f"bytes={bpv:>6} lat={lat:.1f} μs/q")
                    rows.append({"variant": "FDE", "model": cb["model"],
                                 "dataset": "BeIR/scifact",
                                 "R": R, "k_sim": k_sim, "FDE_dim": fde_dim,
                                 "precision": prec, "bytes_per_vector": bpv,
                                 "recall_at_10": rk[10], "recall_at_100": rk[100],
                                 "lat_us_per_query": lat,
                                 "notes": "MUVERA-style FDE, multi-vec query"})

    # CSV
    cols = ["variant", "model", "dataset", "R", "k_sim", "FDE_dim", "precision",
             "bytes_per_vector", "recall_at_10", "recall_at_100",
             "lat_us_per_query", "notes"]
    csv_path = run_dir / "retrieval_frontier.csv"
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"\nwrote {csv_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()

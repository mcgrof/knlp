#!/usr/bin/env python3
"""Build Memorilla document collections for LongHealth patients.

Memorilla (arXiv 2610.21375) reads a collection of document vectors:
each entry of a collection is one passage embedded with
Qwen3-Embedding-4B, and a trained reader turns the whole collection
plus the question into 16 memory tokens for the frozen Qwen3-8B. This
script writes each patient's record as such a collection, in the
authors' data layout, and then runs the authors' own embedding builder
on it, so the vectors are made exactly as for their benchmarks.

A patient's record is its clinical notes in stored order (the order the
full-record panel arm concatenates them). Each note is split into
passages of at most CHUNK reader tokens: lines are packed greedily, and
a line longer than CHUNK is cut by tokens. One row is written per panel
question with ``collection_id`` the patient number, ``question`` the
question text alone (the authors' benchmarks embed the question without
its answer choices), ``answer`` the correct option and ``choices`` the
five options.

The ``check`` mode re-embeds the 100 PubMedQA test questions with the
same builder and compares them with the vectors the authors published,
so the encoder path is known to reproduce theirs before it is trusted.

Env:
    OUT_ROOT    data root to write (data/ and embeddings/ below it)
    CHUNKS      passage sizes in reader tokens (default "256")
    PATIENTS    patient ids (default all 20)
    MEM_SRC     checkout of the Memorilla source
    MEM_DATA    the authors' data root (check mode)
    ENCODER     default Qwen/Qwen3-Embedding-4B
    TOKENIZER   reader tokenizer for passage sizes (default Qwen/Qwen3-8B)

Usage:
    python cas_memorilla_build.py write
    python cas_memorilla_build.py embed
    python cas_memorilla_build.py check
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

DATASET_URL = (
    "https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/"
    "data/benchmark_v5.json"
)
OUT_ROOT = Path(os.environ.get("OUT_ROOT", "memorilla_longhealth"))
CHUNKS = [int(c) for c in os.environ.get("CHUNKS", "256").split()]
PATIENTS = os.environ.get(
    "PATIENTS", " ".join(f"patient_{i:02d}" for i in range(1, 21))
).split()
ENCODER = os.environ.get("ENCODER", "Qwen/Qwen3-Embedding-4B")
TOKENIZER = os.environ.get("TOKENIZER", "Qwen/Qwen3-8B")


def config_name(chunk):
    return f"longhealth_c{chunk}"


def passages(tok, text, chunk):
    """Split one note into passages of at most ``chunk`` tokens."""
    out, cur, n = [], [], 0
    for line in text.split("\n"):
        ids = tok.encode(line + "\n", add_special_tokens=False)
        if len(ids) > chunk:
            if cur:
                out.append("".join(cur))
                cur, n = [], 0
            for i in range(0, len(ids), chunk):
                out.append(tok.decode(ids[i : i + chunk]))
            continue
        if n + len(ids) > chunk and cur:
            out.append("".join(cur))
            cur, n = [], 0
        cur.append(line + "\n")
        n += len(ids)
    if cur:
        out.append("".join(cur))
    return [p.strip() for p in out if p.strip()]


def write():
    import pandas as pd
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    data = json.loads(urllib.request.urlopen(DATASET_URL).read())
    stats = {}
    for chunk in CHUNKS:
        rows = []
        for pid in PATIENTS:
            pat = data[pid]
            docs = []
            for text in pat["texts"].values():
                docs.extend(passages(tok, text, chunk))
            for q in pat["questions"]:
                opts = [q[f"answer_{c}"] for c in "abcde"]
                rows.append(
                    dict(
                        collection_id=int(pid.split("_")[1]),
                        question=q["question"],
                        answer=q["correct"],
                        documents=docs,
                        choices=opts,
                        question_id=f"{pid}_{q['No']}",
                    )
                )
            stats[f"{config_name(chunk)}/{pid}"] = dict(
                passages=len(docs),
                tokens=sum(len(tok.encode(d, add_special_tokens=False)) for d in docs),
            )
        d = OUT_ROOT / "data" / config_name(chunk)
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(d / "test-00000-of-00001.parquet")
        print(f"wrote {d} ({len(rows)} rows)", flush=True)
    (OUT_ROOT / "collections.json").write_text(json.dumps(stats, indent=1))
    for k, v in stats.items():
        print(f"COLLECTION,{k},{v['passages']},{v['tokens']}", flush=True)


def modern_encoder():
    """Replace the authors' encoder loader for current vLLM.

    Their ``load_encoder`` passes ``task="embed"``, which vLLM 0.9.1 took
    and current releases reject. The current spelling is the pooling
    runner with the embed conversion; the pooling itself still comes
    from the encoder's own configuration. ``check`` verifies the result
    against the authors' published vectors.
    """
    sys.path.insert(0, os.environ["MEM_SRC"])
    import memorilla.embeddings as me
    from memorilla.utils import local_model_path

    def load_encoder(encoder=ENCODER, cache_dir=None, gpu_memory_utilization=0.8):
        from vllm import LLM

        return LLM(
            model=local_model_path(encoder, cache_dir),
            runner="pooling",
            convert="embed",
            dtype="bfloat16",
            download_dir=cache_dir,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    me.load_encoder = load_encoder
    return me


def embed():
    me = modern_encoder()
    compute_embeddings = me.compute_embeddings

    for chunk in CHUNKS:
        compute_embeddings(config_name(chunk), ["test"], OUT_ROOT, ENCODER, None)
        print(f"embedded {config_name(chunk)}", flush=True)


def check():
    import torch
    from safetensors.torch import load_file

    me = modern_encoder()
    from memorilla.data import load_split

    embed_texts, embedding_dir = me.embed_texts, me.embedding_dir
    load_encoder = me.load_encoder

    root = Path(os.environ["MEM_DATA"])
    ds = load_split("pubmedqa", "test", root, ["question"])
    qs = [str(q).strip() for q in ds["question"]]
    ref = load_file(
        str(embedding_dir("pubmedqa", "test", root, ENCODER) / "questions.safetensors")
    )["embeddings"].float()
    eng = load_encoder(ENCODER, gpu_memory_utilization=0.8)
    mine = torch.from_numpy(embed_texts(eng, qs)).float()
    cos = torch.nn.functional.cosine_similarity(mine, ref, dim=-1)
    print(
        f"ENCODER_CHECK,n={len(qs)},cos_min={cos.min():.6f},"
        f"cos_mean={cos.mean():.6f}",
        flush=True,
    )


if __name__ == "__main__":
    {"write": write, "embed": embed, "check": check}[sys.argv[1]]()

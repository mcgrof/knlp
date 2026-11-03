#!/usr/bin/env python3
"""Finish Self-Study answers that stopped at the teacher's length cap.

The self-study answers were generated greedily (temperature 0) by the
reader with the note context as the system turn, thinking on, and
top-20 log-probabilities, with a 2,048-token cap; an answer that hit
the cap has no end-of-turn token and usually no closing think tag, so
its ending is never trained on. This script continues each such answer
from its exact stored tokens, with the same model, chat template,
greedy decoding and top-k, until it ends or the whole answer reaches
CAP_TOTAL tokens. The stored tokens and their targets are kept byte
for byte; only the continuation is new.

The continuation runs on a vLLM OpenAI server through the completions
endpoint with the prompt given as token ids (the chat template with
the generation prompt, then the stored answer tokens) and tokens
returned as ids, so no text is re-tokenized. Each new position must
carry exactly TOP_K distinct alternatives with the chosen token first.

Engine agreement: for the first FIDELITY_N queued rows, the same
prompt is scored once with prompt log-probabilities, and the stored
targets of the last FIDELITY_TAIL answer positions are compared with
the engine's: how often its top token is the stored chosen token, the
top-k id overlap, and the log-probability differences. A continuation
is only as faithful as this agreement.

Each queued row ends as complete (ends with the end-of-turn token
within CAP_TOTAL), still_capped, or failed, and every outcome is
listed. A still-capped row also records the share of distinct
20-token windows in its last 1,000 tokens, which separates a greedy
repetition loop (which no cap ends) from a long answer. Rows are written in queue order to <stem>_repairs.parquet with
the source schema; the receipt <stem>_repairs.json records, per row,
the original row index, old and new row hashes (the split manifest's
definition), lengths, finish reason, closing think tag and status.
Progress is kept in <stem>_repairs.jsonl so a rerun resumes.

Env: PARQUET (source), QUEUE (repair TSV with a row column), OUT_DIR,
VLLM_URL (http://localhost:8000/v1), MODEL (Qwen/Qwen3-8B),
CAP_TOTAL (4096), TOP_K (20), CONCURRENCY (8), FIDELITY_N (8),
FIDELITY_TAIL (256), LIMIT (optional, first N queued rows).
"""

import asyncio
import csv
import hashlib
import json
import os
from pathlib import Path

import httpx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

PARQUET = os.environ["PARQUET"]
QUEUE = os.environ["QUEUE"]
OUT_DIR = Path(os.environ["OUT_DIR"])
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
CAP_TOTAL = int(os.environ.get("CAP_TOTAL", "4096"))
TOP_K = int(os.environ.get("TOP_K", "20"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "8"))
FIDELITY_N = int(os.environ.get("FIDELITY_N", "8"))
FIDELITY_TAIL = int(os.environ.get("FIDELITY_TAIL", "256"))
LIMIT = int(os.environ.get("LIMIT", "0"))
EOT, THINK_END = 151645, 151668


def row_sha(user, ids):
    h = hashlib.sha256(user.encode() + b"\0")
    h.update(np.asarray(ids, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def distinct_20grams(ids, tail=1000):
    """Share of distinct 20-token windows in the last `tail` tokens:
    near 1 for ordinary text, far below for a repetition loop (completed
    answers of 1,000+ tokens have a 1st percentile of about 0.82)."""
    t = ids[-tail:]
    g = [tuple(t[i : i + 20]) for i in range(len(t) - 20)]
    return len(set(g)) / max(len(g), 1)


def tid(s):
    assert s.startswith("token_id:"), s
    return int(s.split(":", 1)[1])


def prompt_ids(tok, row):
    sys_p = row["system_prompt"]
    user = row["messages"][0]["content"]
    text = tok.apply_chat_template(
        [{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return tok.encode(text, add_special_tokens=False)


async def continue_one(client, tok, row):
    old = list(row["messages"][1]["token_ids"])
    body = dict(
        model=MODEL,
        prompt=prompt_ids(tok, row) + old,
        max_tokens=CAP_TOTAL - len(old),
        temperature=0.0,
        logprobs=TOP_K,
        return_tokens_as_token_ids=True,
        skip_special_tokens=False,
    )
    r = await client.post("/completions", json=body)
    r.raise_for_status()
    ch = r.json()["choices"][0]
    lp = ch["logprobs"]
    new_ids = [tid(t) for t in lp["tokens"]]
    idx, ids, lps = [], [], []
    for j, (chosen, alts) in enumerate(zip(new_ids, lp["top_logprobs"])):
        top = sorted(((tid(k), v) for k, v in alts.items()), key=lambda x: -x[1])
        top = top[:TOP_K]
        assert len(top) == TOP_K and len({a for a, _ in top}) == TOP_K
        assert top[0][0] == chosen, "greedy token is not the top alternative"
        for a, v in top:
            idx.append(len(old) + j)
            ids.append(a)
            lps.append(float(v))
    return dict(
        new_ids=new_ids,
        idx=idx,
        ids=ids,
        lps=lps,
        finish_reason=ch.get("finish_reason"),
        stop_reason=ch.get("stop_reason"),
    )


async def fidelity_one(client, tok, row):
    old = list(row["messages"][1]["token_ids"])
    p = prompt_ids(tok, row)
    body = dict(
        model=MODEL,
        prompt=p + old,
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=TOP_K,
        return_tokens_as_token_ids=True,
    )
    r = await client.post("/completions", json=body)
    r.raise_for_status()
    plp = r.json()["choices"][0]["prompt_logprobs"]
    tl = row["messages"][1]["top_logprobs"]
    n = len(old)
    sid = np.asarray(tl["token_id"]).reshape(n, -1)
    slp = np.asarray(tl["logprobs"], dtype=np.float64).reshape(n, -1)
    top1_same = overlap = 0
    d_top1, d_chosen = [], []
    for j in range(max(0, n - FIDELITY_TAIL), n):
        eng = {
            int(str(k).rsplit(":", 1)[-1]): v["logprob"]
            for k, v in plp[len(p) + j].items()
        }
        top = sorted(eng, key=lambda k: -eng[k])[:TOP_K]
        top1_same += int(top[0] == old[j])
        overlap += len(set(top) & set(sid[j].tolist()))
        d_top1.append(abs(eng[top[0]] - float(slp[j].max())))
        if old[j] in eng:
            d_chosen.append(abs(eng[old[j]] - float(slp[j][sid[j] == old[j]][0])))
    m = min(FIDELITY_TAIL, n)
    return dict(
        positions=m,
        engine_top1_is_stored_token=top1_same / m,
        topk_id_overlap=overlap / (m * TOP_K),
        top1_logprob_absdiff_mean=float(np.mean(d_top1)),
        top1_logprob_absdiff_max=float(np.max(d_top1)),
        chosen_logprob_absdiff_mean=float(np.mean(d_chosen)) if d_chosen else None,
    )


def build_row(row, res):
    a = row["messages"][1]
    tl = a["top_logprobs"]
    n_new = len(res["new_ids"])
    new_ids = list(a["token_ids"]) + res["new_ids"]
    a2 = dict(a)
    a2["token_ids"] = new_ids
    a2["content"] = None  # filled by the caller from the tokenizer
    a2["top_logprobs"] = dict(
        token_idx=list(tl["token_idx"]) + res["idx"],
        token_id=list(tl["token_id"]) + res["ids"],
        logprobs=list(tl["logprobs"]) + res["lps"],
        shape=[len(new_ids), TOP_K],
    )
    assert len(a2["top_logprobs"]["token_idx"]) == len(new_ids) * TOP_K
    out = dict(row)
    out["messages"] = [row["messages"][0], a2]
    return out, n_new


async def run_all(tok, rows, todo, log):
    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient(base_url=VLLM_URL, timeout=3600) as client:

        async def one(i):
            async with sem:
                try:
                    res = await continue_one(client, tok, rows[i])
                except Exception as e:
                    res = dict(error=f"{type(e).__name__}: {e}")
                res["row"] = i
                log.write(json.dumps(res) + "\n")
                log.flush()
                done = sum(1 for _ in open(log.name))
                if done % 25 == 0:
                    print(f"[repair] {done} rows written", flush=True)

        await asyncio.gather(*(one(i) for i in todo))


async def run_fidelity(tok, rows, sel):
    async with httpx.AsyncClient(base_url=VLLM_URL, timeout=3600) as client:
        return [await fidelity_one(client, tok, rows[i]) for i in sel]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(PARQUET).stem
    tok = AutoTokenizer.from_pretrained(MODEL)
    table = pq.read_table(PARQUET)
    with open(QUEUE) as f:
        queue = [dict(r) for r in csv.DictReader(f, delimiter="\t")]
    if LIMIT:
        queue = queue[:LIMIT]
    rows = {}
    for q in queue:
        i = int(q["row"])
        r = table.slice(i, 1).to_pylist()[0]
        user, a = r["messages"][0]["content"], r["messages"][1]
        assert row_sha(user, a["token_ids"]) == q["row_sha"], f"row {i} differs"
        assert len(a["token_ids"]) < CAP_TOTAL and a["token_ids"][-1] != EOT
        rows[i] = r
    order = [int(q["row"]) for q in queue]

    eng = httpx.get(VLLM_URL.rsplit("/v1", 1)[0] + "/version", timeout=30).json()
    fid_path = OUT_DIR / f"{stem}_fidelity.json"
    if FIDELITY_N and not fid_path.exists():
        fid = asyncio.run(run_fidelity(tok, rows, order[:FIDELITY_N]))
        fid_path.write_text(
            json.dumps(dict(rows=order[:FIDELITY_N], each=fid), indent=1)
        )
        print("[repair] fidelity " + json.dumps(fid), flush=True)

    log_path = OUT_DIR / f"{stem}_repairs.jsonl"
    seen = set()
    if log_path.exists():
        for line in open(log_path):
            r = json.loads(line)
            if "error" not in r:
                seen.add(r["row"])
    todo = [i for i in order if i not in seen]
    print(f"[repair] {len(order)} queued, {len(todo)} to do", flush=True)
    with open(log_path, "a") as log:
        asyncio.run(run_all(tok, rows, todo, log))

    results = {}
    for line in open(log_path):
        r = json.loads(line)
        if r["row"] not in results or "error" in results[r["row"]]:
            results[r["row"]] = r
    out_rows, receipt = [], []
    for i in order:
        r, res = rows[i], results.get(i, dict(error="no result"))
        user, old = r["messages"][0]["content"], r["messages"][1]["token_ids"]
        rec = dict(row=i, old_row_sha=row_sha(user, old), old_len=len(old))
        if "error" in res:
            rec.update(status="failed", error=res["error"])
            receipt.append(rec)
            continue
        new, n_new = build_row(r, res)
        ids = new["messages"][1]["token_ids"]
        new["messages"][1]["content"] = r["messages"][1]["content"] + tok.decode(
            res["new_ids"], skip_special_tokens=True
        )
        ends = ids[-1] == EOT
        rec.update(
            new_row_sha=row_sha(user, ids),
            new_len=len(ids),
            added=n_new,
            finish_reason=res["finish_reason"],
            stop_reason=res["stop_reason"],
            ends_eot=ends,
            think_closed=THINK_END in ids,
            tail_distinct_20grams=round(distinct_20grams(ids), 4),
            status="complete" if ends else "still_capped",
        )
        receipt.append(rec)
        out_rows.append(new)
    if out_rows:
        pq.write_table(
            pa.Table.from_pylist(out_rows, schema=table.schema),
            OUT_DIR / f"{stem}_repairs.parquet",
        )
    counts = {
        s: sum(1 for x in receipt if x["status"] == s)
        for s in ("complete", "still_capped", "failed")
    }
    rec = dict(
        source=os.path.abspath(PARQUET),
        source_sha256=hashlib.sha256(open(PARQUET, "rb").read()).hexdigest(),
        queue=os.path.abspath(QUEUE),
        engine=dict(url=VLLM_URL, version=eng, model=MODEL),
        settings=dict(
            CAP_TOTAL=CAP_TOTAL,
            TOP_K=TOP_K,
            temperature=0.0,
            enable_thinking=True,
            continuation="from the stored answer tokens",
        ),
        counts=counts,
        rows=receipt,
    )
    (OUT_DIR / f"{stem}_repairs.json").write_text(json.dumps(rec, indent=1))
    print("REPAIR " + json.dumps(dict(parquet=stem, **counts)), flush=True)


if __name__ == "__main__":
    main()

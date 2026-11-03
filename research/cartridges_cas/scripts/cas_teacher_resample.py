#!/usr/bin/env python3
"""Sample new self-study answers for rows whose answer failed review.

For each listed row, the teacher (the reader the cartridge is trained
to imitate) answers its question again under the original conditioning:
the row's note as the system turn, the question as the user turn, the
chat template with thinking on. It samples N candidates per row
(temperature 0.6, top-p 0.95, top-k 20, a fixed seed per row) up to
CAP answer tokens, with top-20 log-probabilities at every position.
The completions endpoint takes the prompt as token ids and returns
tokens as ids, so nothing is re-tokenized. Top-k 20 sampling keeps the
sampled token inside the top 20, and a candidate whose positions do
not carry exactly 20 distinct alternatives including the sampled token
is marked invalid.

Writes, per row, OUT_DIR/cand/<patient>_<row>.npz with each candidate's
token ids and top-20 ids and log-probabilities, and a candidate item
per candidate to OUT_DIR/cand_items.jsonl in the judge's item format
(key patient:row:c), with the final answer after the closing think tag
when the candidate ended with the end-of-turn token. Resumable by row.

Env: ITEMS (items JSONL), NOTES (notes JSON), ROWS (JSON list of item
keys to resample), OUT_DIR, URL (http://localhost:8001/v1), MODEL
(Qwen/Qwen3-8B), N (4), CAP (4096), CONCURRENCY (32), SEED (0).
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

EOT, THINK_END = 151645, 151668
TOP_K = 20


def tid(s):
    return int(str(s).rsplit(":", 1)[-1])


def main():
    import httpx

    out = Path(os.environ["OUT_DIR"])
    (out / "cand").mkdir(parents=True, exist_ok=True)
    model = os.environ.get("MODEL", "Qwen/Qwen3-8B")
    url = os.environ.get("URL", "http://localhost:8001/v1")
    n = int(os.environ.get("N", "4"))
    cap = int(os.environ.get("CAP", "4096"))
    conc = int(os.environ.get("CONCURRENCY", "32"))
    seed0 = int(os.environ.get("SEED", "0"))
    tok = AutoTokenizer.from_pretrained(model)
    notes = {}
    for p in os.environ["NOTES"].split(","):
        notes.update(json.load(open(p)))
    items = {
        (it := json.loads(line))["key"]: it
        for p in os.environ["ITEMS"].split(",")
        for line in open(p)
    }
    keys = json.load(open(os.environ["ROWS"]))
    todo = [
        k for k in keys if not (out / "cand" / (k.replace(":", "_") + ".npz")).exists()
    ]
    print(f"[resample] {len(keys)} rows, {len(todo)} to do, N={n}", flush=True)
    log = open(out / "cand_items.jsonl", "a")
    sem = asyncio.Semaphore(conc)

    def prompt(it):
        text = tok.apply_chat_template(
            [
                {"role": "system", "content": notes[it["note"]]},
                {"role": "user", "content": it["question"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        return tok.encode(text, add_special_tokens=False)

    async def one(client, k):
        it = items[k]
        seed = seed0 + int(hashlib.sha256(k.encode()).hexdigest()[:8], 16) % 100000
        body = dict(
            model=model,
            prompt=prompt(it),
            n=n,
            max_tokens=cap,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            seed=seed,
            logprobs=TOP_K,
            return_tokens_as_token_ids=True,
            skip_special_tokens=False,
        )
        async with sem:
            r = await client.post("/completions", json=body)
            r.raise_for_status()
            choices = sorted(r.json()["choices"], key=lambda c: c["index"])
        arrays, recs = {}, []
        for c, ch in enumerate(choices):
            lp = ch["logprobs"]
            ids = [tid(t) for t in lp["tokens"]]
            top_ids = np.zeros((len(ids), TOP_K), dtype=np.int64)
            top_lps = np.zeros((len(ids), TOP_K), dtype=np.float32)
            valid = True
            for j, (chosen, alts) in enumerate(zip(ids, lp["top_logprobs"])):
                top = sorted(
                    ((tid(a), v) for a, v in alts.items()), key=lambda x: -x[1]
                )
                top = top[:TOP_K]
                if len(top) != TOP_K or chosen not in {a for a, _ in top}:
                    valid = False
                    break
                top_ids[j] = [a for a, _ in top]
                top_lps[j] = [v for _, v in top]
            ends = bool(ids) and ids[-1] == EOT
            final = ""
            if ends and THINK_END in ids:
                kk = ids.index(THINK_END)
                final = tok.decode(ids[kk + 1 : -1], skip_special_tokens=True).strip()
            arrays[f"ids_{c}"] = np.asarray(ids, dtype=np.int64)
            arrays[f"top_ids_{c}"] = top_ids
            arrays[f"top_lps_{c}"] = top_lps
            status = "judge" if valid and ends and final else "regenerate"
            recs.append(
                dict(
                    key=f"{k}:{c}",
                    patient=it["patient"],
                    row=it["row"],
                    split=it["split"],
                    note=it["note"],
                    question=it["question"],
                    answer=final,
                    status=status,
                    candidate=c,
                    seed=seed,
                    valid_topk=valid,
                    ends_eot=ends,
                    answer_tokens=len(ids),
                    finish_reason=ch.get("finish_reason"),
                )
            )
        np.savez_compressed(out / "cand" / (k.replace(":", "_") + ".npz"), **arrays)
        for rec in recs:
            log.write(json.dumps(rec) + "\n")
        log.flush()

    async def run():
        async with httpx.AsyncClient(base_url=url, timeout=7200) as client:
            done = [0]

            async def wrapped(k):
                try:
                    await one(client, k)
                except Exception as e:
                    print(f"[resample] {k} failed: {type(e).__name__}: {e}", flush=True)
                done[0] += 1
                if done[0] % 100 == 0:
                    print(f"[resample] {done[0]}/{len(todo)}", flush=True)

            await asyncio.gather(*(wrapped(k) for k in todo))

    asyncio.run(run())
    have = sum(
        1 for k in keys if (out / "cand" / (k.replace(":", "_") + ".npz")).exists()
    )
    print(f"RESAMPLE_DONE rows={len(keys)} written={have}", flush=True)


if __name__ == "__main__":
    main()

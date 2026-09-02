#!/usr/bin/env python3
"""CAS self-study phase 2: answer the phase-1 prompts with the target model.

Partner to cas_synth_qgen.py, which the paper's Appendix I splits out: a
stronger cross-family model M_Q writes the questions, and the model whose
KV cache we are learning, M_A, answers them with the document in context.
Its token log-probabilities are the distillation targets.

Phase 1 already exists and already does proportional-to-length sampling,
20 questions per call, temperature 0.6 / top-p 0.95 / 4096 tokens. This
script was referenced by its docstring and never written, which is why
that whole path has never run and why every dataset we have was built by
the library's single-model loop instead.

Three differences from that loop, all of them things the paper does and
we did not:

  * the questions come from M_Q, a different and larger model, rather
    than from the target model talking to itself;
  * answers are generated with thinking ENABLED (the library rolls a die,
    `prob_thinking`, defaulting to 0.2 in our runs, so four fifths of our
    targets came from non-thinking answers);
  * answers are greedy (temperature 0), so the distillation targets are
    deterministic.

Targets are built DIRECTLY from vLLM's chat/completions response rather
than through the library's OpenAI client. That client, in the public
cartridges tree, does not recover vocabulary ids from a logprobs reply
("simplified since we don't have token IDs") and collapses every position
to a single token — a silent hard-label dataset. The July token-id patch
that fixed this for vLLM 0.28 is not in the public tree and does not fit
vLLM's newer response shape anyway. So we parse the reply here: each
position carries the chosen token and its top-k alternatives with real
byte-level-BPE token strings, which the model's own tokenizer maps back
to ids with `convert_tokens_to_ids`. The result is the genuine top-k the
distillation objective is meant to see, written through the library's own
`write_conversations` so the on-disk COO layout matches every other
dataset the trainer consumes.

The run is resumable and shardable, because the first full run lost 6,300
finished answers when a rented pod was reclaimed under it: prompts are
processed in chunks of FLUSH_EVERY, each finished chunk is written as its
own parquet part immediately with a sidecar of the indices it holds, and
restarting with the same OUT_PARQUET skips them. A prompt whose request
failed is left out of the sidecar, so a restart retries it, and part
names are never reused, so a retry cannot overwrite an earlier part.
NSHARDS/SHARD split the prompt list across processes. `--merge`
concatenates all parts.

Env: IN_JSON (phase-1 output), VLLM_URL, MODEL, TOKENIZER (default MODEL),
     OUT_PARQUET, NUM_TOP_LOGPROBS (default 20),
     MAX_COMPLETION (default 2048), CONCURRENCY (default 32),
     FLUSH_EVERY (default 500), SHARD/NSHARDS (default 0/1),
     REQUEST_TIMEOUT (default 300), LIMIT (optional cap for smoke runs).
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CARTRIDGES_DIR", os.path.expanduser("~/cartridges"))
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", os.path.expanduser("~/cas_out"))
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import numpy as np

from cartridges.structs import Conversation, FlatTopLogprobs, write_conversations

IN_JSON = os.environ.get("IN_JSON", "")
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
TOKENIZER = os.environ.get("TOKENIZER", MODEL)
OUT_PARQUET = os.environ["OUT_PARQUET"]
NUM_TOP_LOGPROBS = int(os.environ.get("NUM_TOP_LOGPROBS", "20"))
MAX_COMPLETION = int(os.environ.get("MAX_COMPLETION", "2048"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "32"))
FLUSH_EVERY = int(os.environ.get("FLUSH_EVERY", "500"))
SHARD = int(os.environ.get("SHARD", "0"))
NSHARDS = int(os.environ.get("NSHARDS", "1"))
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "300"))
LIMIT = int(os.environ.get("LIMIT", "0"))

PARTS_DIR = Path(OUT_PARQUET + ".parts")

_TOK = None


def tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer

        _TOK = AutoTokenizer.from_pretrained(TOKENIZER)
    return _TOK


def parse_answer(content):
    """vLLM logprobs.content -> (assistant token_ids, FlatTopLogprobs).

    content is the per-position list: each entry has the chosen token, its
    logprob, and up to NUM_TOP_LOGPROBS alternatives, all as byte-level-BPE
    token strings. convert_tokens_to_ids maps those strings to ids; a value
    the vocab does not know comes back as None and fails the run loudly
    rather than silently poisoning a target row."""
    tok = tokenizer()
    token_ids = []
    idx, ids, logps = [], [], []
    for i, pos in enumerate(content):
        cid = tok.convert_tokens_to_ids(pos["token"])
        assert cid is not None and cid >= 0, f"chosen token {pos['token']!r} has no id"
        token_ids.append(int(cid))
        alts = pos.get("top_logprobs") or [
            {"token": pos["token"], "logprob": pos["logprob"]}
        ]
        for alt in alts:
            aid = tok.convert_tokens_to_ids(alt["token"])
            assert aid is not None and aid >= 0, f"alt token {alt['token']!r} has no id"
            idx.append(i)
            ids.append(int(aid))
            logps.append(float(alt["logprob"]))
    flat = FlatTopLogprobs(
        token_idx=np.asarray(idx, dtype=np.int64),
        token_id=np.asarray(ids, dtype=np.int64),
        logprobs=np.asarray(logps, dtype=np.float32),
        shape=(len(content), NUM_TOP_LOGPROBS),
    )
    return token_ids, flat


async def answer_one(client, rec):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": rec["note_context"]},
            {"role": "user", "content": rec["prompt"]},
        ],
        "temperature": 0.0,  # deterministic targets, Appendix I
        "max_tokens": MAX_COMPLETION,
        "logprobs": True,
        "top_logprobs": NUM_TOP_LOGPROBS,
        # the paper thinks; our old runs mostly did not
        "chat_template_kwargs": {"enable_thinking": True},
    }
    resp = await client.post("/chat/completions", json=body)
    resp.raise_for_status()
    choice = resp.json()["choices"][0]
    text = choice["message"]["content"]
    content = (choice.get("logprobs") or {}).get("content")
    if not text or not content:
        return None
    token_ids, flat = parse_answer(content)
    return text, token_ids, flat


def to_conversation(rec, text, token_ids, flat):
    return Conversation(
        messages=[
            Conversation.Message(
                content=rec["prompt"], role="user", token_ids=None, top_logprobs=None
            ),
            Conversation.Message(
                content=text, role="assistant", token_ids=token_ids, top_logprobs=flat
            ),
        ],
        system_prompt=rec["note_context"],
        metadata=dict(
            patient_id=rec.get("patient_id"),
            note_id=rec.get("note_id"),
            round=rec.get("round"),
            source="cas_synth_answer",
        ),
        type=rec.get("round", "question"),
    )


def check_targets(convos):
    """Each position should carry the teacher's top-k, not a single token,
    and no position should list the same id twice."""
    rows = mult = dup = tokens = 0
    for c in convos[:200]:
        for m in c.messages:
            tl = m.top_logprobs
            if tl is None:
                continue
            idx = np.asarray(tl.token_idx)
            ids = np.asarray(tl.token_id)
            order = np.argsort(idx, kind="stable")
            idx, ids = idx[order], ids[order]
            for g in np.split(ids, np.flatnonzero(np.diff(idx)) + 1):
                rows += 1
                tokens += g.size
                if g.size > 1:
                    mult += 1
                if len(set(g.tolist())) != g.size:
                    dup += 1
    print(
        f"[answer] target check: {rows} positions, {tokens / max(rows, 1):.2f} "
        f"tokens/position, {mult / max(rows, 1):.2f} with >1 token, {dup} with a duplicate",
        flush=True,
    )
    assert dup == 0, f"{dup} positions carry a duplicate id"
    assert (
        mult > 0
    ), "every position is a single token -- targets collapsed to hard labels"


def done_indices():
    done = set()
    for side in PARTS_DIR.glob("*.done.json"):
        done.update(json.loads(side.read_text()))
    return done


def merge_parts():
    import pyarrow as pa
    import pyarrow.parquet as pq

    parts = sorted(PARTS_DIR.glob("*.parquet"))
    assert parts, f"no parts under {PARTS_DIR}"
    table = pa.concat_tables([pq.read_table(p) for p in parts])
    pq.write_table(table, OUT_PARQUET)
    print(
        f"CAS_ANSWER_MERGED {table.num_rows} conversations from "
        f"{len(parts)} parts -> {OUT_PARQUET}",
        flush=True,
    )


def main():
    if "--merge" in sys.argv:
        merge_parts()
        return

    import httpx

    records = json.loads(Path(IN_JSON).read_text())
    if LIMIT:
        records = records[:LIMIT]
    mine = list(range(SHARD, len(records), NSHARDS))
    already = done_indices()
    todo = [i for i in mine if i not in already]
    print(
        f"[answer] shard {SHARD}/{NSHARDS}: {len(mine)} prompts, "
        f"{len(mine) - len(todo)} already written, {len(todo)} to do",
        flush=True,
    )
    if not todo:
        print(f"CAS_ANSWER_DONE shard {SHARD} (nothing left)", flush=True)
        return

    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer()  # load once, in the main task, before fanning out
    t0 = time.time()
    total_written = failures = 0

    async def run_chunk(idxs):
        sem = asyncio.Semaphore(CONCURRENCY)
        limits = httpx.Limits(max_connections=CONCURRENCY + 8)
        async with httpx.AsyncClient(
            base_url=VLLM_URL, timeout=REQUEST_TIMEOUT, limits=limits
        ) as client:

            async def one(i):
                async with sem:
                    try:
                        res = await answer_one(client, records[i])
                    except Exception as e:
                        print(
                            f"[answer] prompt {i} failed: {type(e).__name__}: {e}",
                            flush=True,
                        )
                        return i, None
                    return i, res

            return await asyncio.gather(*(one(i) for i in idxs))

    for c0 in range(0, len(todo), FLUSH_EVERY):
        idxs = todo[c0 : c0 + FLUSH_EVERY]
        convos, covered = [], []
        for i, res in asyncio.run(run_chunk(idxs)):
            if res is None:
                failures += 1
                continue
            covered.append(i)
            text, token_ids, flat = res
            convos.append(to_conversation(records[i], text, token_ids, flat))
        if convos:
            check_targets(convos)
            # a retried chunk can start at the same index as an earlier
            # part, so take a fresh name rather than overwrite it
            stem = f"s{SHARD}_{idxs[0]:06d}"
            n = 0
            while (PARTS_DIR / f"{stem}.parquet").exists():
                n += 1
                stem = f"s{SHARD}_{idxs[0]:06d}_r{n}"
            write_conversations(convos, str(PARTS_DIR / f"{stem}.parquet"))
            (PARTS_DIR / f"{stem}.done.json").write_text(json.dumps(covered))
            total_written += len(convos)
        print(
            f"[answer] shard {SHARD}: {c0 + len(idxs)}/{len(todo)} done, "
            f"{total_written} written, {failures} failed, {time.time() - t0:.0f}s",
            flush=True,
        )

    assert total_written, "no conversations produced; refusing to declare success"
    print(f"CAS_ANSWER_DONE shard {SHARD}: {total_written} written, {failures} failed")


if __name__ == "__main__":
    main()

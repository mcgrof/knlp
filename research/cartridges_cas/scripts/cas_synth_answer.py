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

Serialization goes through the library's own `write_conversations`. Do
not hand-roll the parquet: the target rows are a sparse COO layout and
the last person to construct them by hand introduced a duplicate-token
defect that survived for weeks.

Env: IN_JSON (phase-1 output), VLLM_URL, MODEL, OUT_PARQUET,
     NUM_TOP_LOGPROBS (default 20), MIN_PROB_MASS (default 0.998),
     MAX_COMPLETION (default 2048), CONCURRENCY (default 32),
     LIMIT (optional cap for smoke runs).
"""

import asyncio
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CARTRIDGES_DIR", os.path.expanduser("~/cartridges"))
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", os.path.expanduser("~/cas_out"))
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import numpy as np

from cartridges.structs import Conversation, write_conversations

IN_JSON = os.environ["IN_JSON"]
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
OUT_PARQUET = os.environ["OUT_PARQUET"]
NUM_TOP_LOGPROBS = int(os.environ.get("NUM_TOP_LOGPROBS", "20"))
MIN_PROB_MASS = float(os.environ.get("MIN_PROB_MASS", "0.998"))
MAX_COMPLETION = int(os.environ.get("MAX_COMPLETION", "2048"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "32"))
LIMIT = int(os.environ.get("LIMIT", "0"))


def build_client():
    """The library's OpenAI client, which our patch teaches to recover real
    vocabulary ids from vLLM and to write one canonical row per position."""
    from cartridges.clients.openai import OpenAIClient

    cfg = OpenAIClient.Config(model_name=MODEL, base_url=VLLM_URL)
    return cfg.instantiate()


def to_flat(sample):
    """ClientSample.top_logprobs -> FlatTopLogprobs, via the library's own
    flatten so the mass threshold and COO layout match every other
    dataset the trainer consumes."""
    if sample.top_logprobs is None:
        return None
    return sample.top_logprobs.flatten(threshold=MIN_PROB_MASS)


def main():
    records = json.loads(Path(IN_JSON).read_text())
    if LIMIT:
        records = records[:LIMIT]
    print(f"[answer] {len(records)} prompts from {IN_JSON}", flush=True)

    client = build_client()
    convos, failures = [], 0
    t0 = time.time()

    # client.chat is a coroutine; a semaphore bounds in-flight requests
    # rather than a thread pool.
    async def run_all():
        sem = asyncio.Semaphore(CONCURRENCY)
        done = 0

        async def one(rec):
            nonlocal done
            async with sem:
                msgs = [
                    {"role": "system", "content": rec["note_context"]},
                    {"role": "user", "content": rec["prompt"]},
                ]
                try:
                    resp = await client.chat(
                        [msgs],
                        temperature=0.0,  # deterministic targets, Appendix I
                        top_logprobs=NUM_TOP_LOGPROBS,
                        max_completion_tokens=MAX_COMPLETION,
                        enable_thinking=True,  # the paper thinks; our runs mostly did not
                    )
                    sample = resp.samples[0]
                except Exception as e:
                    print(f"[answer] request failed: {type(e).__name__}: {e}", flush=True)
                    return None
                finally:
                    done += 1
                    if done % 200 == 0:
                        print(
                            f"[answer] {done}/{len(records)} ({time.time() - t0:.0f}s)",
                            flush=True,
                        )
                return rec, sample

        return await asyncio.gather(*(one(r) for r in records))

    for res in asyncio.run(run_all()):
        if res is None:
            failures += 1
            continue
        rec, sample = res
        if sample is None or not sample.text:
            failures += 1
            continue
        flat = to_flat(sample)
        if flat is None:
            failures += 1
            continue
        convos.append(
            Conversation(
                messages=[
                    Conversation.Message(
                        content=rec["prompt"],
                        role="user",
                        token_ids=None,
                        top_logprobs=None,
                    ),
                    Conversation.Message(
                        content=sample.text,
                        role="assistant",
                        token_ids=(
                            sample.token_ids.tolist()
                            if isinstance(sample.token_ids, np.ndarray)
                            else sample.token_ids
                        ),
                        top_logprobs=flat,
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
        )

    assert convos, "no conversations produced; refusing to write an empty dataset"
    Path(OUT_PARQUET).parent.mkdir(parents=True, exist_ok=True)
    write_conversations(convos, OUT_PARQUET)

    # A duplicate id in a target row is the defect this pipeline shipped
    # once already. Fail loudly here rather than let it reach training.
    rows = uniq = dup = 0
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
                uniq += len(set(g.tolist()))
                if len(set(g.tolist())) != g.size:
                    dup += 1
    print(
        f"[answer] wrote {len(convos)} conversations ({failures} failed) -> {OUT_PARQUET}\n"
        f"[answer] target check: {rows} rows sampled, {uniq / max(rows, 1):.3f} "
        f"unique ids/row, {dup} rows with a duplicate",
        flush=True,
    )
    assert (
        dup == 0
    ), f"{dup} target rows carry a duplicate id; the generator patch is not active"
    print(f"CAS_ANSWER_DONE {len(convos)} -> {OUT_PARQUET}")


if __name__ == "__main__":
    main()

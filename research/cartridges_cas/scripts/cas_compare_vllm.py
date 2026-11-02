#!/usr/bin/env python3
"""Score cartridges, Memorilla and text controls on the LongHealth panel
through one vLLM engine.

Every arm answers the same panel (``cas_eval_table15.py``: its system
prompt, user prompt with the five options, thinking on, temperature 0.6,
top-k 20, top-p 0.95, the completion cap, and scoring by closest option)
with the same frozen Qwen3-8B on the same vLLM build, so the arms differ
only in how the patient's record reaches the reader:

  none       no record
  full       the whole record in the user turn (the panel's full-record
             mode: notes joined in stored order)
  rag        the K passages of the Memorilla collection most similar to
             the question, most similar first, in the user turn (the
             Memorilla authors' retrieval baseline, same vectors)
  cart       a trained cartridge served by the CartridgeConnector: the
             prompt follows one placeholder token per cartridge position,
             and the request names its patient's cartridge
  memorilla  16 memory tokens from the Memorilla reader over the whole
             collection, in the system turn as the authors place them,
             sent as prompt embeddings

Each request gets its own sampling seed (SEED, run, question), so two
arms see the same seeds. Prefix caching is off so no arm reuses another
request's work. The output JSON holds per-patient summaries in the panel
format, every answer, and the arm's wall time and generated tokens.

Env:
    ARM         none | full | rag | cart | memorilla
    PATIENTS    patient ids (default "patient_01 patient_03 patient_05
                patient_06")
    MAX_Q, RUNS, MAX_COMPLETION, SEED   panel settings (20, 3, 4096, 0)
    THINK       1 (default) thinking on, as the panel; 0 off
    CARTS       cart arm: "patient_01=/path.pt patient_03=/path.pt ..."
    MEM_SRC     Memorilla source checkout (rag, memorilla)
    MEM_ROOT    data root with the LongHealth collections (rag, memorilla)
    MEM_CONFIG  collection config (default longhealth_c256)
    MEM_CKPT    Memorilla checkpoint directory (memorilla)
    TOPK_DOCS   passages for rag (default 5)
    OUT_JSON    results
    MODEL, BLOCK_SIZE, BACKEND, GPU_UTIL, MAX_LEN   engine settings
                (Qwen/Qwen3-8B, 1, FLASHINFER, 0.85, 20480)
    KV_CACHE_DTYPE  paged cache dtype (default auto, the model dtype;
                fp8 stores keys and values in FP8)

Usage:
    python cas_compare_vllm.py
"""

import json
import os
import sys
import time
import urllib.request
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cas_eval_table15 as ev  # noqa: E402

ARM = os.environ["ARM"]
PATIENTS = os.environ.get(
    "PATIENTS", "patient_01 patient_03 patient_05 patient_06"
).split()
MAX_Q = int(os.environ.get("MAX_Q", "20"))
RUNS = int(os.environ.get("RUNS", "3"))
CAP = int(os.environ.get("MAX_COMPLETION", "4096"))
SEED = int(os.environ.get("SEED", "0"))
THINK = os.environ.get("THINK", "1") == "1"
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
MEM_CONFIG = os.environ.get("MEM_CONFIG", "longhealth_c256")
TOPK_DOCS = int(os.environ.get("TOPK_DOCS", "5"))
DATASET_URL = (
    "https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/"
    "data/benchmark_v5.json"
)


def load_patients():
    data = json.loads(urllib.request.urlopen(DATASET_URL).read())
    out = []
    for pid in PATIENTS:
        row = data[pid]
        qs = [
            SimpleNamespace(question_id=f"{pid}_{q['No']}", **q)
            for q in row["questions"][:MAX_Q]
        ]
        out.append(SimpleNamespace(patient_id=pid, texts=row["texts"], questions=qs))
    return out


def chat_ids(tok, system, user):
    ids = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=THINK,
    )
    if not isinstance(ids, list):
        ids = list(ids["input_ids"])
    return [int(t) for t in ids]


def collection_store():
    sys.path.insert(0, os.environ["MEM_SRC"])
    from memorilla.data import load_split
    from memorilla.embeddings import EmbeddingStore

    root = os.environ["MEM_ROOT"]
    ds = load_split(MEM_CONFIG, "test", root)
    store = EmbeddingStore(MEM_CONFIG, "test", root)
    rows = {r["question_id"]: (i, r) for i, r in enumerate(ds)}
    return store, rows


def engine(extra):
    from vllm import LLM

    kw = dict(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=int(os.environ.get("MAX_LEN", "20480")),
        gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.85")),
        enable_prefix_caching=False,
        block_size=int(os.environ.get("BLOCK_SIZE", "1")),
        attention_config={"backend": os.environ.get("BACKEND", "FLASHINFER")},
        kv_cache_dtype=os.environ.get("KV_CACHE_DTYPE", "auto"),
    )
    kw.update(extra)
    return LLM(**kw)


def build_requests(tok, patients):
    """(prompt, patient, question, extra sampling args) per question."""
    from vllm.inputs import TokensPrompt

    reqs, extra = [], {}
    if ARM in ("none", "full"):
        for p in patients:
            doc = "\n\n".join(p.texts.values()) if ARM == "full" else None
            for q in p.questions:
                ids = chat_ids(tok, ev.SYSTEM_PROMPT, ev.user_prompt(q, doc))
                reqs.append((TokensPrompt(prompt_token_ids=ids), p, q, {}))
    elif ARM == "rag":
        import torch

        store, rows = collection_store()
        for p in patients:
            for q in p.questions:
                i, r = rows[q.question_id]
                docs = store.documents(r["collection_id"]).float()
                qv = store.question(i).float()
                sim = torch.nn.functional.cosine_similarity(docs, qv[None], dim=-1)
                top = sim.topk(min(TOPK_DOCS, len(sim))).indices.tolist()
                doc = "\n\n".join(r["documents"][j] for j in top)
                ids = chat_ids(tok, ev.SYSTEM_PROMPT, ev.user_prompt(q, doc))
                reqs.append((TokensPrompt(prompt_token_ids=ids), p, q, {}))
    elif ARM == "cart":
        from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector import (
            load_cartridge,
        )

        carts = dict(kv.split("=", 1) for kv in os.environ["CARTS"].split())
        lens = {
            pid: int(load_cartridge(path)["num_tokens"]) for pid, path in carts.items()
        }
        extra["kv_transfer_config"] = {
            "kv_connector": "CartridgeConnector",
            "kv_connector_module_path": (
                "vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector"
            ),
            "kv_connector_extra_config": {
                "cartridges": [
                    {"cartridge_id": pid, "path": path} for pid, path in carts.items()
                ],
                "router": {"type": "explicit", "key": "cartridge_id"},
            },
            "kv_role": "kv_both",
        }
        for p in patients:
            ids = [tok.pad_token_id] * lens[p.patient_id]
            for q in p.questions:
                ids_q = ids + chat_ids(tok, ev.SYSTEM_PROMPT, ev.user_prompt(q))
                reqs.append(
                    (
                        TokensPrompt(prompt_token_ids=ids_q),
                        p,
                        q,
                        {"cartridge_id": p.patient_id},
                    )
                )
    elif ARM == "memorilla":
        extra["enable_prompt_embeds"] = True
    else:
        raise ValueError(ARM)
    return reqs, extra


def memorilla_requests(tok, patients):
    """Prompt embeddings with the reader's memory tokens in the system turn."""
    import torch
    from safetensors import safe_open

    sys.path.insert(0, os.environ["MEM_SRC"])
    from memorilla.data import memory_placeholder
    from memorilla.llm import inject_memory
    from memorilla.memory import MemoryModule
    from memorilla.utils import prepare_tokenizer
    from vllm.inputs import EmbedsPrompt

    mem_id = prepare_tokenizer(tok)
    memory = MemoryModule.from_pretrained(os.environ["MEM_CKPT"])
    memory = memory.to("cuda", dtype=torch.bfloat16).eval()
    n_mem = memory.num_memories
    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL, allow_patterns=["*.safetensors", "*.json"])
    weight = None
    for shard in sorted(os.listdir(path)):
        if not shard.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(path, shard), framework="pt", device="cuda") as h:
            if "model.embed_tokens.weight" in h.keys():
                weight = h.get_tensor("model.embed_tokens.weight").to(torch.bfloat16)
                break
    store, rows = collection_store()
    reqs = []
    with torch.inference_mode():
        for p in patients:
            for q in p.questions:
                i, r = rows[q.question_id]
                docs = store.documents(r["collection_id"])[None].to(
                    "cuda", torch.bfloat16
                )
                qv = store.question(i)[None].to("cuda", torch.bfloat16)
                mem = memory(docs, None, qv)
                system = ev.SYSTEM_PROMPT + memory_placeholder(n_mem)
                ids = torch.tensor(
                    [chat_ids(tok, system, ev.user_prompt(q))], device="cuda"
                )
                emb = inject_memory(weight[ids], mem, ids, mem_id)[0]
                reqs.append((EmbedsPrompt(prompt_embeds=emb.cpu()), p, q, {}))
    del memory, weight
    torch.cuda.empty_cache()
    return reqs


def _memorilla_worker(path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    reqs = memorilla_requests(tok, load_patients())
    import torch

    torch.save([(r[0]["prompt_embeds"], r[2].question_id) for r in reqs], path)


def memorilla_requests_isolated(patients):
    """Build the memory tokens in a child process on the GPU.

    The reader and the embedding matrix leave CUDA state in whatever
    process runs them, which the vLLM engine then cannot use; a spawned
    child returns all of it when it exits.
    """
    import multiprocessing
    import tempfile

    import torch
    from vllm.inputs import EmbedsPrompt

    path = tempfile.mktemp(suffix=".pt")
    proc = multiprocessing.get_context("spawn").Process(
        target=_memorilla_worker, args=(path,)
    )
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f"memory-token worker failed ({proc.exitcode})")
    saved = torch.load(path)
    os.unlink(path)
    byq = {qid: emb for emb, qid in saved}
    return [
        (EmbedsPrompt(prompt_embeds=byq[q.question_id]), p, q, {})
        for p in patients
        for q in p.questions
    ]


def main():
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    tok = AutoTokenizer.from_pretrained(MODEL)
    patients = load_patients()
    t_prep = time.time()
    if ARM == "memorilla":
        reqs = memorilla_requests_isolated(patients)
        extra = {"enable_prompt_embeds": True}
    else:
        reqs, extra = build_requests(tok, patients)
    prep_s = time.time() - t_prep
    llm = engine(extra)
    prompts, params, meta = [], [], []
    for run in range(RUNS):
        for k, (prompt, p, q, args) in enumerate(reqs):
            prompts.append(prompt)
            params.append(
                SamplingParams(
                    temperature=ev.TEMPERATURE,
                    top_k=ev.TOP_K,
                    top_p=ev.TOP_P,
                    max_tokens=CAP,
                    seed=SEED * 1_000_000 + run * 10_000 + k,
                    extra_args=args or None,
                )
            )
            meta.append((run, p.patient_id, q))
    t0 = time.time()
    outs = llm.generate(prompts, params, use_tqdm=False)
    gen_s = time.time() - t0
    per = {}
    raw = []
    ntok = 0
    for (run, pid, q), o in zip(meta, outs):
        text = o.outputs[0].text
        ntok += len(o.outputs[0].token_ids)
        correct, tag = ev.score_response(text, q)
        per.setdefault(pid, {}).setdefault(run, []).append((correct, tag))
        raw.append(
            dict(
                run=run,
                patient=pid,
                qid=q.question_id,
                correct=correct,
                tag_found=tag,
                n_tokens=len(o.outputs[0].token_ids),
                prompt_len=len(o.prompt_token_ids or []),
                text=text,
            )
        )
    summary = {}
    for pid, runs in per.items():
        summary[pid] = ev.aggregate(
            [
                (sum(c for c, _ in r), len(r), sum(1 for _, t in r if not t))
                for _, r in sorted(runs.items())
            ]
        )
        print(
            f"COMPARE,{ARM},{pid},acc={summary[pid]['acc_mean']:.4f}"
            f"+/-{summary[pid]['acc_std']:.4f}",
            flush=True,
        )
    res = dict(
        arm=ARM,
        model=MODEL,
        think=THINK,
        cap=CAP,
        seed=SEED,
        runs=RUNS,
        mem_config=MEM_CONFIG if ARM in ("rag", "memorilla") else None,
        mem_ckpt=os.environ.get("MEM_CKPT") if ARM == "memorilla" else None,
        kv_cache_dtype=os.environ.get("KV_CACHE_DTYPE", "auto"),
        carts=os.environ.get("CARTS") if ARM == "cart" else None,
        prep_s=round(prep_s, 2),
        generate_s=round(gen_s, 2),
        generated_tokens=ntok,
        summary=summary,
    )
    with open(os.environ["OUT_JSON"], "w") as f:
        json.dump(res, f, indent=1)
    with open(os.environ["OUT_JSON"] + ".raw.json", "w") as f:
        json.dump(raw, f)
    print(
        f"COMPARE_DONE,{ARM},generate_s={gen_s:.1f},tokens={ntok},prep_s={prep_s:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

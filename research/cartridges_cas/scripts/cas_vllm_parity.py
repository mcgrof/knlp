#!/usr/bin/env python3
"""Check that vLLM's CartridgeConnector serves a cartridge as the panel
evaluator does.

The panel evaluator (``cas_eval_batched.py``) runs the reader in Hugging
Face Transformers with the cartridge preloaded as a key-value prefix:
keys rotated at positions 0 .. S-1, the prompt at positions S + j. The
connector instead receives a prompt that starts with S placeholder
tokens and claims them as externally computed, writing the cartridge
into the request's pages. Both paths must give the same next-token
distribution after each panel prompt. Each mode below writes the top-K
log-probabilities of the first answer token for the first N panel
prompts; ``compare`` reads two dumps and prints the agreement.

Run ``hf`` and ``vllm`` once with the cartridge and once without
(``CART=none``): the no-cartridge pair measures how far the two engines
differ on their own (bf16 kernels, attention backend), so the cartridge
pair is judged against that floor, not against zero.

Env:
    CART        cartridge file, or ``none`` for the bare reader
    PATIENT     patient id (default patient_01)
    N           prompts (default 5)
    TOPK        log-probabilities kept per prompt (default 20)
    OUT         dump path (hf, vllm)
    A, B        dump paths (compare)
    MODEL       reader (default Qwen/Qwen3-8B)
    BLOCK_SIZE  vLLM page size (default 1, so no cartridge position is
                dropped to page alignment)
    BACKEND     vLLM attention backend (default FLASHINFER)
    GPU_UTIL    vLLM gpu_memory_utilization (default 0.85)

Usage:
    python cas_vllm_parity.py hf
    python cas_vllm_parity.py vllm
    python cas_vllm_parity.py compare
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
PATIENT = os.environ.get("PATIENT", "patient_01")
N = int(os.environ.get("N", "5"))
TOPK = int(os.environ.get("TOPK", "20"))
CART = os.environ.get("CART", "none")
GEN = int(os.environ.get("GEN", "16"))
# Close an empty thinking block after the prompt so that the next token
# begins the answer; with thinking on, the first token is "<think>" with
# probability near 1 whatever the cartridge holds.
SKIP_THINK = os.environ.get("SKIP_THINK", "1") == "1"


def panel_prompts(tok):
    from cas_eval_batched import prompt_ids
    from cartridges.data.longhealth.utils import load_longhealth_dataset

    pat = load_longhealth_dataset([PATIENT])[0]
    qs = pat.questions[:N]
    prompts = []
    for q in qs:
        ids = prompt_ids(tok, q)
        # Transformers 5 returns a mapping from apply_chat_template.
        if not isinstance(ids, list):
            ids = list(ids["input_ids"])
        ids = [int(t) for t in ids]
        if SKIP_THINK:
            ids += tok.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
        prompts.append(ids)
    return [q.question_id for q in qs], prompts


def cart_len():
    if CART == "none":
        return 0
    # The connector's own loader, so the placeholder count is the length
    # the connector will claim.
    from vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector import (
        load_cartridge,
    )

    return int(load_cartridge(CART)["num_tokens"])


def save(rows, S):
    with open(os.environ["OUT"], "w") as f:
        json.dump(dict(cart=CART, model=MODEL, patient=PATIENT, S=S, rows=rows), f)
    print(f"wrote {os.environ['OUT']} ({len(rows)} prompts, S={S})", flush=True)


def hf():
    import cas_eval_batched as eb

    eb.DEVICE = "cuda"
    tok, model = eb.load()
    qids, prompts = panel_prompts(tok)
    S = 0
    cache = None
    if CART != "none":
        keys, values = eb.cart_kv(CART)
        S = keys[0].shape[2]
        cache = eb.prefix_cache(keys, values, len(prompts))
    from transformers import DynamicCache

    if cache is None:
        cache = DynamicCache()
    ids, mask = eb.batch_inputs(tok, prompts, S)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    inp, step_pos = ids[:, S:], pos[:, S:]
    gen, first = [], None
    with torch.no_grad():
        for _ in range(GEN):
            out = model(
                input_ids=inp,
                attention_mask=mask,
                position_ids=step_pos,
                past_key_values=cache,
                use_cache=True,
            )
            logits = out.logits[:, -1].float()
            if first is None:
                first = torch.log_softmax(logits, -1)
            nxt = logits.argmax(-1, keepdim=True)
            gen.append(nxt)
            inp = nxt
            step_pos = step_pos[:, -1:] + 1
            mask = torch.cat([mask, torch.ones_like(nxt)], -1)
    gen = torch.cat(gen, -1).tolist()
    rows = []
    for i, qid in enumerate(qids):
        v, t = first[i].topk(TOPK)
        rows.append(
            dict(
                qid=qid,
                prompt=prompts[i],
                ids=t.tolist(),
                logprobs=v.tolist(),
                greedy=gen[i],
            )
        )
    save(rows, S)


def vllm():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = AutoTokenizer.from_pretrained(MODEL)
    qids, prompts = panel_prompts(tok)
    S = cart_len()
    kw = dict(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.85")),
        enable_prefix_caching=False,
        block_size=int(os.environ.get("BLOCK_SIZE", "1")),
        attention_config={"backend": os.environ.get("BACKEND", "FLASHINFER")},
        max_logprobs=TOPK,
    )
    if S:
        kw["kv_transfer_config"] = {
            "kv_connector": "CartridgeConnector",
            "kv_connector_module_path": (
                "vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector"
            ),
            "kv_connector_extra_config": {"cartridge_path": CART},
            "kv_role": "kv_both",
        }
    llm = LLM(**kw)
    dummy = tok.pad_token_id
    reqs = [TokensPrompt(prompt_token_ids=[dummy] * S + list(p)) for p in prompts]
    sp = SamplingParams(max_tokens=GEN, temperature=0.0, logprobs=TOPK)
    outs = llm.generate(reqs, sp, use_tqdm=False)
    rows = []
    for qid, p, o in zip(qids, prompts, outs):
        lp = o.outputs[0].logprobs[0]
        top = sorted(lp.items(), key=lambda kv: -kv[1].logprob)[:TOPK]
        rows.append(
            dict(
                qid=qid,
                prompt=p,
                ids=[t for t, _ in top],
                logprobs=[x.logprob for _, x in top],
                greedy=list(o.outputs[0].token_ids),
            )
        )
    save(rows, S)


def compare():
    a = json.load(open(os.environ["A"]))
    b = json.load(open(os.environ["B"]))
    assert a["S"] == b["S"], (a["S"], b["S"])
    worst = 0.0
    for ra, rb in zip(a["rows"], b["rows"]):
        assert ra["qid"] == rb["qid"]
        assert ra["prompt"] == rb["prompt"], f"prompt ids differ for {ra['qid']}"
        da = dict(zip(ra["ids"], ra["logprobs"]))
        db = dict(zip(rb["ids"], rb["logprobs"]))
        shared = set(da) & set(db)
        diff = max(abs(da[t] - db[t]) for t in shared)
        # Tokens A gives at least 1% probability: the ones that matter.
        big = [t for t in shared if da[t] > -4.6052]
        diff_big = max(abs(da[t] - db[t]) for t in big)
        # KL(A || B) over the shared top tokens, renormalized on both sides.
        pa = torch.tensor([da[t] for t in shared]).exp()
        pb = torch.tensor([db[t] for t in shared]).exp()
        pa, pb = pa / pa.sum(), pb / pb.sum()
        kl = float((pa * (pa.log() - pb.log())).sum())
        mass = float(torch.tensor([da[t] for t in shared]).exp().sum())
        worst = max(worst, diff_big)
        ga, gb = ra["greedy"], rb["greedy"]
        agree = next((k for k, (x, y) in enumerate(zip(ga, gb)) if x != y), len(ga))
        print(
            f"PARITY,{ra['qid']},top1={int(ra['ids'][0] == rb['ids'][0])},"
            f"shared={len(shared)}/{len(da)},mass={mass:.4f},kl={kl:.2e},"
            f"maxabs_p01={diff_big:.4f},maxabs_all={diff:.4f},"
            f"greedy_agree={agree}/{len(ga)}",
            flush=True,
        )
    print(f"PARITY_WORST_MAXABS_P01,{worst:.4f}", flush=True)


if __name__ == "__main__":
    {"hf": hf, "vllm": vllm, "compare": compare}[sys.argv[1]]()

#!/usr/bin/env python3
"""Score a Latent Compaction memory on the 20-question cartridge panel.

Latent Compaction (arXiv 2610.20582, code at github.com/thunlp/
latent-compaction) stores a compacted document as, per layer, selected
and learned keys and values, a scalar logit bias per entry, and a mask
for uneven per-head counts. Its attention adapter treats the memory as a
logical document of ``document_tokens`` positions: the prompt that
follows starts at that position, and attention over the memory and over
the prompt are merged through their log-normalizers. This script loads
such a memory through the authors' own adapter and runs the panel
protocol of ``cas_eval_batched.py`` (system prompt, user prompt, chat
template with thinking on, temperature 0.6, top-k 20, top-p 0.95, the
same completion cap and scoring), so its scores compare with native
cartridges scored by the batched evaluator.

``check`` verifies the adapter path before any panel is trusted: it
wraps a native cartridge as a memory with zero biases and a full mask,
whose logical length is the cartridge length (the native convention),
and compares next-token logits after the first N panel prompts with
the batched evaluator's path on the same cartridge, in one process.

Env:
    LC_SRC         checkout of thunlp/latent-compaction (put on sys.path)
    MEMORY         (panel) memory file written by the authors' save_memory
    CART           (check) native cartridge file
    PATIENT, MAX_Q, RUNS, MAX_COMPLETION, BATCH, SEED, OUT_JSON, MODEL,
    DEVICE         as in cas_eval_batched.py
    N              (check) prompts (default 5)

Usage:
    python cas_lc_panel.py check
    python cas_lc_panel.py panel
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ["LC_SRC"])

import cas_eval_batched as cb  # noqa: E402
from latent_compaction.memory import selected_memory  # noqa: E402
from latent_compaction.methods.latent import load_memory  # noqa: E402
from latent_compaction.models import (  # noqa: E402
    DocumentCache,
    install_attention_adapter,
)

DEVICE = cb.DEVICE
PATIENT = cb.PATIENT


def load():
    tok, model = cb.load()
    for p in model.parameters():
        p.requires_grad_(False)
    install_attention_adapter(model)
    return tok, model


def prompt_ids(tok, q):
    """Panel prompt token ids; transformers 5 returns a mapping from
    apply_chat_template(tokenize=True), 4.x a list."""
    ids = cb.prompt_ids(tok, q)
    return list(ids["input_ids"]) if hasattr(ids, "keys") else list(ids)


def doc_cache(model, memories, doc_tokens):
    cache = DocumentCache(model.config)
    cache.document_tokens = doc_tokens
    cache.memories = memories
    return cache


def run_prompts(model, ids, mask, S, cache):
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return model(
        input_ids=ids[:, S:],
        attention_mask=mask,
        position_ids=pos[:, S:],
        past_key_values=cache,
        use_cache=True,
    )


def check():
    from cartridges.data.longhealth.utils import load_longhealth_dataset

    n = int(os.environ.get("N", "5"))
    tok, model = load()
    keys, values = cb.cart_kv(os.environ["CART"])
    S = keys[0].shape[2]
    qs = load_longhealth_dataset([PATIENT])[0].questions[:n]
    prompts = [prompt_ids(tok, q) for q in qs]
    ids, mask = cb.batch_inputs(tok, prompts, S)
    with torch.no_grad():
        ref = run_prompts(model, ids, mask, S, cb.prefix_cache(keys, values, n))
        mems = {
            i: selected_memory(model, i, k.float(), v.float())
            for i, (k, v) in enumerate(zip(keys, values))
        }
        out = run_prompts(model, ids, mask, S, doc_cache(model, mems, S))
    worst = 0.0
    for i, q in enumerate(qs):
        la, lb = out.logits[i, -1].float(), ref.logits[i, -1].float()
        pa, pb = torch.log_softmax(la, -1), torch.log_softmax(lb, -1)
        kl = float((pb.exp() * (pb - pa)).sum())
        worst = max(worst, kl)
        print(
            f"LC_CHECK,{q.question_id},maxabs={float((la - lb).abs().max()):.4f},"
            f"kl={kl:.2e},top1={int(la.argmax() == lb.argmax())}",
            flush=True,
        )
    print(f"LC_CHECK_DONE worst_kl={worst:.2e}", flush=True)


def panel():
    import cas_eval_table15 as ev
    from cartridges.data.longhealth.utils import load_longhealth_dataset

    mem_path = os.environ["MEMORY"]
    max_q = int(os.environ.get("MAX_Q", "20"))
    runs = int(os.environ.get("RUNS", "3"))
    cap = int(os.environ.get("MAX_COMPLETION", "4096"))
    B = int(os.environ.get("BATCH", "20"))
    seed = int(os.environ.get("SEED", "0"))
    out_json = os.environ["OUT_JSON"]
    t0 = time.time()
    tok, model = load()
    memories, D = load_memory(mem_path, DEVICE)
    for m in memories.values():
        m.requires_grad_(False)
    slots = sum(m.logical_slots_per_head for m in memories.values()) / len(memories)
    eos = model.generation_config.eos_token_id
    eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
    qs = load_longhealth_dataset([PATIENT])[0].questions[:max_q]
    items = [(r, q) for r in range(runs) for q in qs]
    rows, n_gen = [], 0
    for b in range(0, len(items), B):
        chunk = items[b : b + B]
        prompts = [prompt_ids(tok, q) for _, q in chunk]
        # D placeholder columns stand for the logical document: the
        # prompt starts at position D, as in the authors' adapter
        ids, mask = cb.batch_inputs(tok, prompts, D)
        torch.manual_seed(seed + b // B)
        with torch.no_grad():
            gen = model.generate(
                input_ids=ids,
                attention_mask=mask,
                past_key_values=doc_cache(model, memories, D),
                do_sample=True,
                temperature=0.6,
                top_k=ev.TOP_K,
                top_p=ev.TOP_P,
                max_new_tokens=cap,
                eos_token_id=eos,
                pad_token_id=tok.pad_token_id,
            )
        new = gen[:, ids.shape[1] :]
        for (r, q), g in zip(chunk, new.tolist()):
            g = [t for t in g if t != tok.pad_token_id]
            cut = next((i for i, t in enumerate(g) if t in eos), len(g))
            n_gen += cut
            text = tok.decode(g[:cut], skip_special_tokens=True)
            corr, tag = ev.score_response(text, q)
            rows.append(
                dict(
                    run=r,
                    patient=PATIENT,
                    qid=q.question_id,
                    correct=int(corr),
                    tag_found=bool(tag),
                    n_tokens=cut,
                    truncated="</think>" not in text,
                    text=text,
                )
            )
        del gen
        torch.cuda.empty_cache()
        print(
            f"[lc] {min(b + B, len(items))}/{len(items)} answers "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
    per_run = []
    for r in range(runs):
        rr = [x for x in rows if x["run"] == r]
        per_run.append(
            (
                sum(x["correct"] for x in rr),
                len(rr),
                sum(1 for x in rr if not x["tag_found"]),
            )
        )
    done = [x for x in rows if not x["truncated"]]
    res = dict(
        protocol="Table 15 panel, batched HF generation through the Latent "
        f"Compaction adapter; temp 0.6, top-k {ev.TOP_K}, top-p {ev.TOP_P}, "
        f"max {cap}",
        evaluator="cas_lc_panel",
        model=cb.MODEL,
        memory=os.path.abspath(mem_path),
        document_tokens=D,
        logical_slots_per_head=slots,
        patient=PATIENT,
        max_q=max_q,
        batch=B,
        seed=seed,
        summary=ev.aggregate(per_run),
        completed_n=len(done),
        completed_acc=(
            round(sum(x["correct"] for x in done) / len(done), 4) if done else None
        ),
        generated_tokens=n_gen,
        seconds=time.time() - t0,
        device=torch.cuda.get_device_name(),
    )
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    raw = out_json.replace(".json", ".raw.json")
    json.dump(rows, open(raw, "w"), indent=1)
    res["raw_path"] = raw
    json.dump(res, open(out_json, "w"), indent=2)
    print(
        f"LC_PANEL_DONE acc={res['summary']['acc_mean']} "
        f"+/-{res['summary']['acc_std']} tokens={n_gen} "
        f"({res['seconds']:.0f}s) -> {out_json}",
        flush=True,
    )


if __name__ == "__main__":
    {"panel": panel, "check": check}[sys.argv[1]]()

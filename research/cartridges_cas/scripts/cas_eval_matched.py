#!/usr/bin/env python3
"""Decoder-parity CAS cartridge eval with a multi-scorer report.

Two confounds the plain Table-15 harness left open (both raised in the
2026-07-27 ChatGPT-Pro review of the CAS reproduction):

  1. DECODER MISMATCH. The cartridge flex loop sampled from the FULL softmax
     (logits/T only) while the vLLM baselines used Qwen3-8B's generation_config
     (top_k=20, top_p=0.95). Same weights, different output distribution. This
     harness applies the SAME explicit sampler (temp 0.6, top_k 20, top_p 0.95,
     min_p 0) so the cartridge decoder matches Qwen's recommended thinking-mode
     decoding -- removing the "the cart just samples worse" confound.

  2. ONE CONVENIENT SCORER. A missing <answer> tag is not self-evidently
     "wrong": the paper never specifies the rule, and the public HazyResearch
     scorer falls back to option A. We therefore report THREE scores side by
     side per the review -- never cherry-picking whichever lands near 0.736:
       strict : no complete <answer> tag  -> wrong
       hazy   : no tag -> choose option A  (the public HazyResearch behavior)
       tail   : no tag -> fuzzy-match the post-</think> tail (the local harness)
     plus the tag-compliance rate and the 2048-cap-hit rate.

Averages over RUNS seeds (the paper uses >=3; a single run has ~5pp SE at
n=100). Records per response are kept for the missing-answer audit.

Env: CART_DIR, PATIENTS, MAX_Q, MAX_COMPLETION (default 2048 = paper), RUNS
(default 3), TOP_K (20), TOP_P (0.95), MIN_P (0.0), DEVICE, SINK_MAX, OUT_JSON.
"""

import os
import sys
import json
import time
import difflib
import statistics

os.environ.setdefault("CARTRIDGES_DIR", "/home/mcgrof/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/home/mcgrof/cas_out")
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

CARTRIDGES = os.environ["CARTRIDGES_DIR"]
sys.path.insert(0, CARTRIDGES)

import torch
from transformers import AutoTokenizer

import cas_eval_table15 as E
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
from cartridges.data.longhealth.utils import load_longhealth_dataset

CART_DIR = os.environ["CART_DIR"]
PATIENTS = os.environ.get(
    "PATIENTS", "patient_01 patient_02 patient_03 patient_05 patient_06"
).split()
MAX_Q = int(os.environ.get("MAX_Q", "20"))
MAX_COMPLETION = int(os.environ.get("MAX_COMPLETION", "2048"))
RUNS = int(os.environ.get("RUNS", "3"))
TOP_K = int(os.environ.get("TOP_K", "20"))
TOP_P = float(os.environ.get("TOP_P", "0.95"))
MIN_P = float(os.environ.get("MIN_P", "0.0"))
DEVICE = os.environ.get("DEVICE", "cuda:0")
SINK_MAX = int(os.environ.get("SINK_MAX", "4"))
MODEL = E.MODEL
TEMPERATURE = E.TEMPERATURE
OUT_JSON = os.environ.get("OUT_JSON", "/home/mcgrof/cas_out/eval_matched.json")


# --- multi-scorer ------------------------------------------------------------
def _sim(a, b):
    return difflib.SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _closest(pred, options):
    if not pred:
        return -1
    ratios = [_sim(pred, o) for o in options]
    m = max(ratios)
    return ratios.index(m) if m > 0.0 else -1


def score_all(text, q):
    """Return the three scorer verdicts + tag flag for one response."""
    ci = E.correct_index(q)
    options = [q.answer_a, q.answer_b, q.answer_c, q.answer_d, q.answer_e]
    post = text.split("</think>")[-1] if "</think>" in text else text
    tags = E.ANSWER_RE.findall(post)
    tag_found = bool(tags)
    tagpred = tags[-1].strip() if tags else None

    if tag_found:
        hit = int(_closest(tagpred, options) == ci)
        strict = hazy = tail = hit
    else:
        strict = 0
        hazy = int(ci == 0)  # HazyResearch: no tag -> option A
        tailpred = post.strip()[-300:]
        tail = int(_closest(tailpred, options) == ci)
    return {"strict": strict, "hazy": hazy, "tail": tail, "tag_found": tag_found}


def main():
    di = int(DEVICE.split(":")[1]) if ":" in DEVICE else 0
    torch.cuda.set_device(di)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = FlexQwen3ForCausalLM.from_pretrained(MODEL).to(DEVICE).to(torch.bfloat16)
    model.eval()
    ac = AttnConfig(
        n_layers=model.config.num_hidden_layers,
        n_heads=model.config.num_key_value_heads,
        head_dim=getattr(
            model.config,
            "head_dim",
            model.config.hidden_size // model.config.num_attention_heads,
        ),
    )

    def load_cart_kv(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)

        def t(p):
            return torch.as_tensor(p.data if hasattr(p, "data") else p).to(
                torch.bfloat16
            )

        fk = ck.get("frozen_keys") or []
        nfrozen = t(fk[0]).shape[2] if fk else 0
        use_frozen = 0 < nfrozen <= SINK_MAX

        def cat(fro, tra):
            tt = [t(p) for p in tra]
            if fro and use_frozen:
                ff = [t(p) for p in fro]
                return [torch.cat([ff[i], tt[i]], dim=2) for i in range(len(tt))]
            return tt

        return (
            cat(ck.get("frozen_keys"), ck["trainable_keys"]),
            cat(ck.get("frozen_values"), ck["trainable_values"]),
        )

    def make_cache(path):
        k, v = load_cart_kv(path)
        return TrainableCache(config=ac, init_keys=k, init_values=v).to(DEVICE)

    eos_cfg = getattr(model.generation_config, "eos_token_id", None)
    if eos_cfg is None:
        eos_cfg = [tok.eos_token_id]
    stop_ids = set(eos_cfg if isinstance(eos_cfg, (list, tuple)) else [eos_cfg])

    def _filtered_sample(logits):
        """temp -> top_k -> top_p -> min_p -> multinomial. Matches Qwen3's
        recommended thinking-mode sampler so the cartridge decoder is not a
        free variable vs the vLLM baselines."""
        probs = torch.softmax(logits / TEMPERATURE, dim=-1)
        if TOP_K and TOP_K < probs.numel():
            kth = torch.topk(probs, TOP_K).values[-1]
            probs = torch.where(probs < kth, torch.zeros_like(probs), probs)
        if 0.0 < TOP_P < 1.0:
            sp, si = torch.sort(probs, descending=True)
            csum = torch.cumsum(sp, dim=-1)
            # keep the smallest prefix whose cumulative mass reaches TOP_P
            drop = csum - sp > TOP_P
            sp = sp.masked_fill(drop, 0.0)
            probs = torch.zeros_like(probs).scatter(0, si, sp)
        if MIN_P > 0.0:
            probs = torch.where(
                probs < MIN_P * probs.max(), torch.zeros_like(probs), probs
            )
        s = probs.sum()
        if s <= 0:
            return int(torch.argmax(logits))
        return int(torch.multinomial(probs / s, 1))

    @torch.no_grad()
    def generate(cache, ids, seed):
        torch.manual_seed(seed)
        cache.clear()
        cur_ids = ids
        pos = torch.arange(ids.shape[0], device=DEVICE)
        sids = torch.zeros(ids.shape[0], dtype=torch.long, device=DEVICE)
        out = []
        finish = "length"
        for _ in range(MAX_COMPLETION):
            o = model(
                input_ids=cur_ids,
                seq_ids=sids,
                position_ids=pos,
                past_key_values=cache,
                use_cache=True,
                mode="generate",
            )
            nxt = _filtered_sample(o.logits[0, -1].float())
            if nxt in stop_ids:
                finish = "eos"
                break
            out.append(nxt)
            cur_ids = torch.tensor([nxt], device=DEVICE)
            pos = pos[-1:] + 1
            sids = sids[-1:]
        cache.clear()
        return tok.decode(out, skip_special_tokens=True), finish, len(out)

    patients = load_longhealth_dataset(PATIENTS)
    present = [
        p
        for p in patients
        if os.path.exists(os.path.join(CART_DIR, f"{p.patient_id}.pt"))
    ]
    records = []
    t0 = time.time()
    for run in range(RUNS):
        idx = 0
        for patient in present:
            cache = make_cache(os.path.join(CART_DIR, f"{patient.patient_id}.pt"))
            for q in patient.questions[:MAX_Q]:
                ids = (
                    tok.apply_chat_template(
                        [
                            {"role": "system", "content": E.SYSTEM_PROMPT},
                            {"role": "user", "content": E.user_prompt(q)},
                        ],
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        enable_thinking=True,
                    )
                    .to(DEVICE)
                    .flatten()
                )
                text, finish, ntok = generate(cache, ids, seed=run * 100003 + idx)
                sc = score_all(text, q)
                records.append(
                    {
                        "run": run,
                        "patient": patient.patient_id,
                        "qid": q.question_id,
                        "finish": finish,
                        "n_tokens": ntok,
                        "closed_think": "</think>" in text,
                        **sc,
                        "tail_text": text.strip()[-200:],
                    }
                )
                idx += 1
            del cache
            torch.cuda.empty_cache()
        print(f"  run {run} done ({idx} q) t={time.time() - t0:.0f}s", flush=True)

    # Crash-safe: persist the raw (expensive) generation records BEFORE
    # aggregation, so a summary bug can never discard hours of GPU work again.
    os.makedirs(os.path.dirname(OUT_JSON) or ".", exist_ok=True)
    json.dump(
        {"cart_dir": CART_DIR, "patients": PATIENTS, "records": records},
        open(OUT_JSON + ".records.json", "w"),
    )

    # aggregate per run then mean over runs, for each scorer
    def per_run_means(field):
        vals = []
        for run in range(RUNS):
            rr = [r for r in records if r["run"] == run]
            vals.append(sum(int(r[field]) for r in rr) / max(len(rr), 1))
        return vals

    summary = {
        "runs": RUNS,
        "max_completion": MAX_COMPLETION,
        "sampler": {
            "temp": TEMPERATURE,
            "top_k": TOP_K,
            "top_p": TOP_P,
            "min_p": MIN_P,
        },
        "n_per_run": len(records) // RUNS if RUNS else 0,
    }
    for field in ("strict", "hazy", "tail"):
        v = per_run_means(field)
        summary[f"acc_{field}_mean"] = round(statistics.mean(v), 4)
        summary[f"acc_{field}_std"] = (
            round(statistics.stdev(v), 4) if len(v) > 1 else 0.0
        )
        summary[f"acc_{field}_per_run"] = [round(x, 4) for x in v]
    tag = per_run_means("tag_found")
    cap = [
        sum(1 for r in records if r["run"] == run and r["finish"] == "length")
        / max(sum(1 for r in records if r["run"] == run), 1)
        for run in range(RUNS)
    ]
    summary["tag_rate_mean"] = round(statistics.mean(tag), 4)
    summary["cap_hit_rate_mean"] = round(statistics.mean(cap), 4)
    summary["mean_tokens"] = round(
        sum(r["n_tokens"] for r in records) / max(len(records), 1), 1
    )

    out = {
        "cart_dir": CART_DIR,
        "patients": PATIENTS,
        "summary": summary,
        "records": records,
    }
    os.makedirs(os.path.dirname(OUT_JSON) or ".", exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"CAS_EVAL_MATCHED_DONE -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()

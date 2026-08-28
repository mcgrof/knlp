#!/usr/bin/env python3
"""Diagnostic pass over the CAS Table-15 cartridge eval: WHY are ~25% of
cartridge responses missing an <answer> tag, and is that truncation (the
model runs its thinking past the completion cap before it answers), format
drift (it finishes but never emits the tag), or genuine non-commit?

Reuses the canonical scorer / prompt from cas_eval_table15.py and re-runs the
cart-mode generate loop, but records per response: finish_reason (eos vs the
length cap), generated-token count, whether the thinking block closed
(</think> present), whether an <answer> tag was found, and the correctness the
canonical scorer assigns. It then breaks the tag-missing bucket down by
finish_reason and think-closure -- that is the measurement that separates a
budget artifact (raise the cap) from a real cartridge control failure.

MAX_COMPLETION is overridable so the SAME script serves the paper-faithful
2048 diagnostic and a raised-budget (e.g. 8192) recovery test.

Env:
  CART_DIR      dir of <patient>.pt cartridges (required)
  PATIENTS      space-separated patient ids (default the 5-patient pilot set)
  MAX_Q         questions per patient (default 20)
  MAX_COMPLETION generation cap in tokens (default 2048 = paper)
  DEVICE        cuda:N (default cuda:0)
  SINK_MAX      frozen-sink cap for cart reconstruction (default 4)
  OUT_JSON      diagnostic output path
"""

import os
import sys
import json
import time

os.environ.setdefault("CARTRIDGES_DIR", os.path.expanduser("~/cartridges"))
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", os.path.expanduser("~/cas_out"))
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

CARTRIDGES = os.environ["CARTRIDGES_DIR"]
sys.path.insert(0, CARTRIDGES)

import torch
from transformers import AutoTokenizer

# Reuse the canonical prompt + scorer so the diagnostic scores identically to
# the real eval; only the generate loop is re-implemented to expose finish_reason.
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
DEVICE = os.environ.get("DEVICE", "cuda:0")
SINK_MAX = int(os.environ.get("SINK_MAX", "4"))
MODEL = E.MODEL
TEMPERATURE = E.TEMPERATURE
OUT_JSON = os.environ.get(
    "OUT_JSON", fos.path.expanduser("~/cas_out/eval_diag_{MAX_COMPLETION}.json")
)


def load_cart_kv(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)

    def t(p):
        return torch.as_tensor(p.data if hasattr(p, "data") else p).to(torch.bfloat16)

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

    def make_cache(path):
        k, v = load_cart_kv(path)
        return TrainableCache(config=ac, init_keys=k, init_values=v).to(DEVICE)

    eos_cfg = getattr(model.generation_config, "eos_token_id", None)
    if eos_cfg is None:
        eos_cfg = [tok.eos_token_id]
    stop_ids = set(eos_cfg if isinstance(eos_cfg, (list, tuple)) else [eos_cfg])

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
            logits = o.logits[0, -1].float() / TEMPERATURE
            nxt = int(torch.multinomial(torch.softmax(logits, dim=-1), 1))
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
    records = []
    t0 = time.time()
    idx = 0
    for patient in patients:
        cart = os.path.join(CART_DIR, f"{patient.patient_id}.pt")
        if not os.path.exists(cart):
            print(f"  no cart for {patient.patient_id}, skip", flush=True)
            continue
        cache = make_cache(cart)
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
            text, finish, ntok = generate(cache, ids, seed=idx)
            corr, tag = E.score_response(text, q)
            closed_think = "</think>" in text
            records.append(
                {
                    "patient": patient.patient_id,
                    "qid": q.question_id,
                    "finish": finish,
                    "n_tokens": ntok,
                    "closed_think": closed_think,
                    "tag_found": tag,
                    "correct": int(corr),
                    "tail": text.strip()[-200:],
                }
            )
            idx += 1
        del cache
        torch.cuda.empty_cache()
        print(
            f"  {patient.patient_id} done ({idx} q) t={time.time() - t0:.0f}s",
            flush=True,
        )

    n = len(records)
    correct = sum(r["correct"] for r in records)
    miss = [r for r in records if not r["tag_found"]]
    length = [r for r in records if r["finish"] == "length"]
    miss_length = [r for r in miss if r["finish"] == "length"]
    miss_length_openthink = [r for r in miss_length if not r["closed_think"]]
    miss_eos = [r for r in miss if r["finish"] == "eos"]
    summary = {
        "max_completion": MAX_COMPLETION,
        "n": n,
        "correct": correct,
        "acc": round(correct / max(n, 1), 4),
        "tag_missing": len(miss),
        "finish_length_total": len(length),
        "tag_missing_and_length": len(miss_length),
        "tag_missing_length_open_think": len(miss_length_openthink),
        "tag_missing_but_eos": len(miss_eos),
        "correct_among_tag_missing": sum(r["correct"] for r in miss),
        "mean_tokens": round(sum(r["n_tokens"] for r in records) / max(n, 1), 1),
        "mean_tokens_tag_missing": round(
            sum(r["n_tokens"] for r in miss) / max(len(miss), 1), 1
        ),
    }
    out = {
        "cart_dir": CART_DIR,
        "patients": PATIENTS,
        "summary": summary,
        "records": records,
    }
    os.makedirs(os.path.dirname(OUT_JSON) or ".", exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"CAS_EVAL_DIAG_DONE max={MAX_COMPLETION} -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()

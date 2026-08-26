#!/usr/bin/env python3
"""Evaluation-only audit of saved OP-Cart cartridges (review Step A).

Re-evaluates every saved cartridge with no retraining, fixing the two
instrument defects the review identified: the first-letter-character
parser (which can score "The answer is C" as E) and the invisibility
of cap-hit behavior. Every raw response is persisted; scoring uses a
strict standalone-letter parser with parser-invalid as an explicit
outcome; cap-hit, unclosed-think, and length statistics are reported;
and no-cartridge plus full-context-teacher controls run under the
identical evaluator. Two generation caps are evaluated (the original
32, and 256 so long-form responses get the chance to close their
reasoning and answer).

Env: CARTS (comma list of name=path), RECORD, PATIENT, OUT_JSON,
MAX_Q, CAPS (comma list, default 32,256).
"""

import json
import os
import re
from pathlib import Path

os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/reeval_out")
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import torch
from transformers import AutoTokenizer

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.data.longhealth.utils import load_longhealth_dataset
from cartridges.generation import flex_generate
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
CARTS = os.environ["CARTS"]
RECORD = os.environ["RECORD"]
PATIENT = os.environ["PATIENT"]
OUT_JSON = os.environ.get("OUT_JSON", "/root/reeval_out/reeval.json")
MAX_Q = int(os.environ.get("MAX_Q", "20"))
CAPS = [int(c) for c in os.environ.get("CAPS", "32,256").split(",")]
SINK_MAX = 4
DEVICE = "cuda"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = FlexQwen3ForCausalLM.from_pretrained(MODEL).to(DEVICE).to(torch.bfloat16)
model.eval()
ac = AttnConfig(
    n_layers=model.config.num_hidden_layers,
    n_heads=model.config.num_key_value_heads,
    head_dim=model.config.head_dim,
)


def load_cart(path):
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

    ik = cat(ck.get("frozen_keys"), ck["trainable_keys"])
    iv = cat(ck.get("frozen_values"), ck["trainable_values"])
    return TrainableCache(config=ac, init_keys=ik, init_values=iv).to(DEVICE)


STANDALONE = re.compile(r"^[\s\*\_\#\-]*\(?([A-Ea-e])\)?[\.\:\)\s\*\_]*$")
STATED = re.compile(
    r"(?:answer|option|choice|letter)\s*(?:is|:|=)?\s*\**\(?([A-Ea-e])\)?\b",
    re.IGNORECASE,
)


def strict_parse(resp):
    """Return (letter or "", reason)."""
    body = resp.split("</think>")[-1] if "</think>" in resp else resp
    body = body.strip()
    if not body:
        return "", "empty"
    first_line = next((ln for ln in body.splitlines() if ln.strip()), "")
    m = STANDALONE.match(first_line.strip())
    if m:
        return m.group(1).upper(), "standalone"
    m = STATED.search(body)
    if m:
        return m.group(1).upper(), "stated"
    return "", "invalid"


def normalize(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", str(s).lower())).strip()


@torch.no_grad()
def evaluate(name, cache, cap, prefix_text=None):
    rows = []
    for patient in load_longhealth_dataset([PATIENT]):
        for q in patient.questions[:MAX_Q]:
            prompt = (
                f"Question: {q.question}\nA) {q.answer_a}\nB) {q.answer_b}\n"
                f"C) {q.answer_c}\nD) {q.answer_d}\nE) {q.answer_e}\n\n"
                "Answer with ONLY the letter (A, B, C, D, or E). Do not explain."
            )
            if prefix_text is not None:
                prompt = prefix_text + "\n\n" + prompt
            ids = (
                tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    enable_thinking=True,
                )
                .to(DEVICE)
                .flatten()
            )
            if cache is not None:
                cache.clear()
            out = flex_generate(
                model,
                tok,
                ids,
                seq_ids=torch.zeros(ids.shape[0], dtype=torch.long, device=DEVICE),
                position_ids=torch.arange(ids.shape[0], device=DEVICE),
                max_new_tokens=cap,
                cache=cache,
                temperature=0.0,
            )
            out_ids = list(out.get(0, []))
            resp = tok.decode(out_ids, skip_special_tokens=True)
            letter, reason = strict_parse(resp)
            amap = {
                "A": q.answer_a,
                "B": q.answer_b,
                "C": q.answer_c,
                "D": q.answer_d,
                "E": q.answer_e,
            }
            correct_letter = next((k for k, v in amap.items() if v == q.correct), "")
            rows.append(
                dict(
                    question=q.question,
                    correct_letter=correct_letter,
                    parsed=letter,
                    parse_reason=reason,
                    strict_correct=bool(letter) and amap.get(letter, "") == q.correct,
                    semantic_contains=normalize(q.correct) in normalize(resp),
                    n_tokens=len(out_ids),
                    cap_hit=len(out_ids) >= cap,
                    think_unclosed="</think>" not in resp
                    and "<think>" in resp
                    or ("</think>" not in resp and len(out_ids) >= cap),
                    raw=resp,
                )
            )
    n = len(rows)
    summary = dict(
        condition=name,
        cap=cap,
        n=n,
        strict_acc=sum(r["strict_correct"] for r in rows) / n,
        parser_invalid=sum(r["parse_reason"] == "invalid" for r in rows) / n,
        empty=sum(r["parse_reason"] == "empty" for r in rows) / n,
        cap_hit=sum(r["cap_hit"] for r in rows) / n,
        semantic_contains=sum(r["semantic_contains"] for r in rows) / n,
        mean_len=sum(r["n_tokens"] for r in rows) / n,
    )
    return summary, rows


def main():
    Path(os.path.dirname(OUT_JSON)).mkdir(parents=True, exist_ok=True)
    record_text = Path(RECORD).read_text()
    conditions = []
    for spec in CARTS.split(","):
        name, path = spec.split("=", 1)
        conditions.append((name, ("cart", path)))
    conditions.append(("no_cartridge", ("none", None)))
    conditions.append(("full_context", ("prefix", record_text)))

    report = dict(model=MODEL, patient=PATIENT, max_q=MAX_Q, caps=CAPS, results=[])
    raws = {}
    for name, (kind, arg) in conditions:
        cache = load_cart(arg) if kind == "cart" else None
        prefix = arg if kind == "prefix" else None
        for cap in CAPS:
            summary, rows = evaluate(name, cache, cap, prefix_text=prefix)
            report["results"].append(summary)
            raws[f"{name}_cap{cap}"] = rows
            print(
                f"[reeval] {name} cap={cap}: strict={summary['strict_acc']:.3f} "
                f"invalid={summary['parser_invalid']:.2f} cap_hit={summary['cap_hit']:.2f} "
                f"semantic={summary['semantic_contains']:.2f} len={summary['mean_len']:.1f}",
                flush=True,
            )
        if cache is not None:
            del cache
            torch.cuda.empty_cache()

    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=1)
    with open(OUT_JSON.replace(".json", "_raw.json"), "w") as f:
        json.dump(raws, f, indent=1)
    print(f"OPCART-REEVAL-DONE {OUT_JSON}")


if __name__ == "__main__":
    main()

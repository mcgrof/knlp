#!/usr/bin/env python3
"""Does a cartridge still answer its own document once its neighbours
are in the prefix, and does training longer make that worse?

Co-loading independently trained cartridges degrades them: the archived
reproduction has five isolated cartridges scoring 0.58 alone and 0.38
together, which is the no-context floor, while cartridges trained under
mixed visibility hold at 0.44 and 0.46.  The repair is usually read as
a property of composition.  There is another reading.  Training a
cartridge past the point where it stops gaining document knowledge
makes it a worse artifact -- measured on this document, specificity
falls to a trough and rebuilds only partway -- and a worse artifact
might simply be a more fragile neighbour.  If so, part of what mixed
visibility repairs is damage the schedule caused first.

That is a checkpoint question, so ask it with checkpoints: score every
cartridge alone and co-loaded, at an early step and at a late one, and
compare how much of each one's own advantage survives the neighbours.

Collapse is reported as the fraction of a cartridge's advantage over
the no-context floor that survives co-loading,
`(coload - floor) / (oracle - floor)`, because early and late
cartridges do not have the same advantage to lose and the raw drop
would reward the one that started lower.

Per-patient co-load numbers are kept, not just the mean: cartridges are
concatenated in a fixed order and position in the prefix is a plausible
confound, so it should be visible rather than averaged away.

Env: MODEL, CARTS (comma list patient=path), LONGHEALTH_JSON, OUT_JSON,
     ORDER (optional comma list of patients, default sorted).
"""

import json
import os
import statistics
from pathlib import Path

os.environ.setdefault("CARTRIDGES_DIR", "/data/mcgrof/cartridges")
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
CARTS = os.environ["CARTS"]
LONGHEALTH_JSON = os.environ["LONGHEALTH_JSON"]
OUT_JSON = os.environ["OUT_JSON"]
DEVICE = "cuda"
LETTERS = ["A", "B", "C", "D", "E"]

tok = AutoTokenizer.from_pretrained(MODEL)
model = FlexQwen3ForCausalLM.from_pretrained(MODEL).to(DEVICE).to(torch.bfloat16)
model.eval()
model.requires_grad_(False)
ac = AttnConfig(
    n_layers=model.config.num_hidden_layers,
    n_heads=model.config.num_key_value_heads,
    head_dim=model.config.head_dim,
)

LETTER_IDS = {}
for L in LETTERS:
    enc = tok(L, add_special_tokens=False).input_ids
    assert len(enc) == 1, f"letter {L} is not one token: {enc}"
    LETTER_IDS[L] = enc[0]


def cart_kv(path):
    """Per-layer (keys, values) with the attention sink reattached."""
    ck = torch.load(path, map_location="cpu", weights_only=False)

    def t(p):
        return torch.as_tensor(p.data if hasattr(p, "data") else p).to(torch.bfloat16)

    tk = [t(p) for p in ck["trainable_keys"]]
    tv = [t(p) for p in ck["trainable_values"]]
    fk = ck.get("frozen_keys") or []
    fv = ck.get("frozen_values") or []
    if fk:
        tk = [torch.cat([t(fk[i]), tk[i]], dim=2) for i in range(len(tk))]
        tv = [torch.cat([t(fv[i]), tv[i]], dim=2) for i in range(len(tv))]
    return tk, tv


def make_cache(paths):
    """One cache holding these cartridges end to end, in the given order."""
    kvs = [cart_kv(p) for p in paths]
    n_layers = len(kvs[0][0])
    ik = [torch.cat([kv[0][i] for kv in kvs], dim=2) for i in range(n_layers)]
    iv = [torch.cat([kv[1][i] for kv in kvs], dim=2) for i in range(n_layers)]
    return TrainableCache(
        config=ac, init_keys=ik, init_values=iv, num_frozen_tokens=0
    ).to(DEVICE)


data = json.loads(Path(LONGHEALTH_JSON).read_text())


def questions_for(pid):
    out = []
    for i, q in enumerate(data[pid]["questions"]):
        amap = {L: q["answer_" + L.lower()] for L in LETTERS}
        correct = next((L for L, v in amap.items() if v == q["correct"]), None)
        if correct is None:
            continue
        out.append(
            dict(
                qid=f"{pid}_{i}",
                patient=pid,
                text=q["question"],
                amap=amap,
                correct=correct,
            )
        )
    return out


def prompt_for(q):
    return (
        f"Question: {q['text']}\n"
        + "\n".join(f"{L}) {q['amap'][L]}" for L in LETTERS)
        + "\n\nAnswer with ONLY the letter (A, B, C, D, or E). Do not explain."
    )


@torch.no_grad()
def score(cache, qs):
    """cache=None is the no-context floor: a plain forward, no prefix."""
    rows = []
    for q in qs:
        ids = (
            tok.apply_chat_template(
                [{"role": "user", "content": prompt_for(q)}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False,
            )
            .to(DEVICE)
            .flatten()
        )
        kw = dict(
            input_ids=ids,
            seq_ids=torch.zeros(ids.shape[0], dtype=torch.long, device=DEVICE),
            position_ids=torch.arange(ids.shape[0], device=DEVICE),
        )
        if cache is not None:
            cache.clear()
            kw.update(use_cache=True, past_key_values=cache)
        else:
            kw.update(use_cache=False)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(**kw)
        lp = F.log_softmax(out.logits[0, -1].float(), dim=-1)
        lps = {L: float(lp[LETTER_IDS[L]]) for L in LETTERS}
        c = q["correct"]
        pred = max(lps, key=lps.get)
        rows.append(
            dict(
                qid=q["qid"],
                patient=q["patient"],
                correct=c,
                pred=pred,
                hit=pred == c,
                margin=lps[c] - max(v for L, v in lps.items() if L != c),
            )
        )
    return rows


def summarize(rows):
    if not rows:
        return None
    m = [r["margin"] for r in rows]
    return dict(
        n=len(rows),
        acc=sum(r["hit"] for r in rows) / len(rows),
        margin_mean=statistics.fmean(m),
    )


def main():
    carts = {}
    for spec in CARTS.split(","):
        pid, path = spec.split("=", 1)
        carts[pid] = path
    order = os.environ.get("ORDER")
    order = [p.strip() for p in order.split(",")] if order else sorted(carts)
    assert set(order) == set(carts), f"ORDER {order} does not match carts {list(carts)}"
    qs = {p: questions_for(p) for p in order}
    print(
        f"[coload] {len(order)} cartridges in prefix order {order}; "
        f"questions " + ", ".join(f"{p}:{len(qs[p])}" for p in order),
        flush=True,
    )

    Path(os.path.dirname(OUT_JSON)).mkdir(parents=True, exist_ok=True)
    report = dict(model=MODEL, order=order, carts=carts, per_patient={})

    def persist():
        with open(OUT_JSON, "w") as f:
            json.dump(report, f, indent=1)

    # The floor only normalizes the comparison; oracle against co-load is
    # the measurement and must not be lost to a prefix-free forward the
    # flex attention path may not accept.
    print("[coload] floor: no prefix at all", flush=True)
    for p in order:
        try:
            fr = summarize(score(None, qs[p]))
        except Exception as e:
            print(
                f"[coload]   {p} floor UNAVAILABLE: {type(e).__name__}: {e}", flush=True
            )
            fr = None
        report["per_patient"].setdefault(p, {})["floor"] = fr
        if fr:
            print(f"[coload]   {p} floor acc={fr['acc']:.3f}", flush=True)
    persist()

    print("[coload] oracle: each cartridge alone", flush=True)
    for p in order:
        cache = make_cache([carts[p]])
        report["per_patient"][p]["oracle"] = summarize(score(cache, qs[p]))
        print(
            f"[coload]   {p} oracle acc={report['per_patient'][p]['oracle']['acc']:.3f}",
            flush=True,
        )
        del cache
        torch.cuda.empty_cache()
    persist()

    print("[coload] co-load: every cartridge resident", flush=True)
    cache = make_cache([carts[p] for p in order])
    for p in order:
        report["per_patient"][p]["coload"] = summarize(score(cache, qs[p]))
        print(
            f"[coload]   {p} coload acc={report['per_patient'][p]['coload']['acc']:.3f}",
            flush=True,
        )
    del cache
    torch.cuda.empty_cache()

    fls = [report["per_patient"][p]["floor"] for p in order]
    fl = statistics.fmean(f["acc"] for f in fls) if all(fls) else None
    orc = statistics.fmean(report["per_patient"][p]["oracle"]["acc"] for p in order)
    col = statistics.fmean(report["per_patient"][p]["coload"]["acc"] for p in order)
    adv = None if fl is None else orc - fl
    report["aggregate"] = dict(
        floor=fl,
        oracle=orc,
        coload=col,
        drop=orc - col,
        advantage=adv,
        # undefined when the cartridges never got above the floor, and a
        # ratio against a near-zero denominator would be noise wearing a
        # number's clothes
        retained=((col - fl) / adv) if adv is not None and abs(adv) > 0.02 else None,
    )
    persist()
    r = report["aggregate"]["retained"]
    print(
        f"[coload] floor={'n/a' if fl is None else f'{fl:.3f}'} "
        f"oracle={orc:.3f} coload={col:.3f} drop={orc - col:+.3f} retained="
        + ("undefined" if r is None else f"{r:.3f}"),
        flush=True,
    )
    print(f"COLOAD_DONE {OUT_JSON}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare a cartridge reader's logits across two GPUs before mixing
their panel scores.

The panel (``cas_eval_table15.py`` MODE=cart) samples long answers, so
two devices whose arithmetic differs slightly can still drift apart
over a thousand tokens. Before a second device scores panels, this
script checks the parts that must agree: the next-token distribution
after each panel prompt with the cartridge loaded, and a short greedy
continuation. It also reports the decode speed, which sets how long a
panel takes on that device.

``dump`` runs the same prompts the panel uses (its system prompt, its
user prompt and chat template, thinking on) for the first N questions
of one patient, and saves per prompt the full last-position logits, a
greedy continuation of G tokens with each step's top-1 log-probability,
and the decode time per token. ``compare`` reads two dumps and prints,
per prompt, the largest absolute logit difference, the KL divergence of
the next-token distributions, whether the top-1 token agrees, and how
many greedy tokens agree before the first difference.

Env (dump):
    CART       cartridge file (frozen sink + trainable layout)
    PATIENT    patient id (default patient_01)
    N          questions (default 5)
    GREEDY     greedy tokens per prompt (default 64)
    OUT        output .pt
    MODEL, DEVICE  as in the panel

Usage:
    python cas_logit_parity.py dump
    python cas_logit_parity.py compare a.pt b.pt
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def dump():
    import cas_eval_table15 as ev
    from transformers import AutoTokenizer
    from cartridges.cache import AttnConfig, TrainableCache
    from cartridges.data.longhealth.utils import load_longhealth_dataset
    from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
    from cas_cart_init import load_cart

    cart = os.environ["CART"]
    patient = os.environ.get("PATIENT", "patient_01")
    n = int(os.environ.get("N", "5"))
    greedy = int(os.environ.get("GREEDY", "64"))
    out = os.environ["OUT"]
    dev = ev.DEVICE

    tok = AutoTokenizer.from_pretrained(ev.MODEL, trust_remote_code=True)
    model = FlexQwen3ForCausalLM.from_pretrained(ev.MODEL, dtype=torch.bfloat16)
    model = model.to(dev).eval()
    ac = AttnConfig(
        n_layers=model.config.num_hidden_layers,
        n_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
    )
    keys, values, _ = load_cart(cart, device=dev)
    cache = TrainableCache(config=ac, init_keys=keys, init_values=values).to(dev)
    pat = [p for p in load_longhealth_dataset([patient])][0]

    rows = []
    with torch.no_grad():
        for q in pat.questions[:n]:
            ids = (
                tok.apply_chat_template(
                    [
                        {"role": "system", "content": ev.SYSTEM_PROMPT},
                        {"role": "user", "content": ev.user_prompt(q)},
                    ],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    enable_thinking=True,
                )
                .to(dev)
                .flatten()
            )
            cache.clear()
            cur = ids
            pos = torch.arange(ids.shape[0], device=dev)
            sids = torch.zeros(ids.shape[0], dtype=torch.long, device=dev)
            first, toks, lps, t_dec = None, [], [], 0.0
            for i in range(greedy + 1):
                torch.cuda.synchronize()
                t0 = time.time()
                o = model(
                    input_ids=cur,
                    seq_ids=sids,
                    position_ids=pos,
                    past_key_values=cache,
                    use_cache=True,
                    mode="generate",
                )
                lg = o.logits[0, -1].float()
                torch.cuda.synchronize()
                if i > 0:
                    t_dec += time.time() - t0
                if first is None:
                    first = lg.cpu()
                if i == greedy:
                    break
                nxt = int(lg.argmax())
                toks.append(nxt)
                lps.append(float(torch.log_softmax(lg, -1)[nxt]))
                cur = torch.tensor([nxt], device=dev)
                pos = pos[-1:] + 1
                sids = sids[-1:]
            cache.clear()
            rows.append(
                dict(
                    qid=q.question_id,
                    prompt_len=int(ids.shape[0]),
                    logits=first,
                    greedy=toks,
                    greedy_lp=lps,
                    sec_per_token=t_dec / max(greedy, 1),
                )
            )
            print(
                f"[parity] {q.question_id} len={ids.shape[0]} "
                f"{rows[-1]['sec_per_token'] * 1000:.1f} ms/token",
                flush=True,
            )
    torch.save(
        dict(
            cart=os.path.abspath(cart),
            device=torch.cuda.get_device_name(),
            torch=torch.__version__,
            rows=rows,
        ),
        out,
    )
    print(f"PARITY_DUMP {out}", flush=True)


def compare(a_path, b_path):
    a = torch.load(a_path, weights_only=False)
    b = torch.load(b_path, weights_only=False)
    print(f"A: {a['device']} torch {a['torch']}")
    print(f"B: {b['device']} torch {b['torch']}")
    for ra, rb in zip(a["rows"], b["rows"]):
        assert ra["qid"] == rb["qid"] and ra["prompt_len"] == rb["prompt_len"]
        la, lb = ra["logits"], rb["logits"]
        pa, pb = torch.log_softmax(la, -1), torch.log_softmax(lb, -1)
        kl = float((pa.exp() * (pa - pb)).sum())
        same = 0
        for x, y in zip(ra["greedy"], rb["greedy"]):
            if x != y:
                break
            same += 1
        print(
            f"PARITY,{ra['qid']},maxabs={float((la - lb).abs().max()):.4f},"
            f"kl={kl:.2e},top1={int(la.argmax() == lb.argmax())},"
            f"greedy_agree={same}/{len(ra['greedy'])},"
            f"ms_a={ra['sec_per_token'] * 1000:.1f},"
            f"ms_b={rb['sec_per_token'] * 1000:.1f}"
        )


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump()
    else:
        compare(sys.argv[2], sys.argv[3])

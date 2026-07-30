#!/usr/bin/env python3
"""Disagreement-driven example selection (Pro's best-bet recipe).

Every same-family synthesis tweak plateaued because it kept feeding the
cartridge examples it already handles. This selects, from the existing
diverse self-study pool, the examples the CURRENT cartridge reproduces
WORST -- i.e. where the student most disagrees with the teacher -- so a fresh
cartridge trains on its own residual failures instead of re-learning what it
already knows.

For one patient: load the trained diverse cartridge and its 2000-question
self-study parquet. For each of CAND sampled candidates, teacher-force the
stored teacher answer through the cartridge and record the cart's mean
log-probability of that answer (LOW = high disagreement). Then write two
matched-size subset parquets, reusing the ORIGINAL rows so the training format
is byte-identical:
  * hi   : the DISAGREE lowest-logprob examples (the treatment)
  * rand : a random sample of the same size (the control)
Train a fresh cartridge on each and compare -- if hi beats rand, targeting the
cart's failures is the lever.

Env: PATIENT, CAND (candidates scored, default 1500), K (subset size, default
500), DEVICE, SINK_MAX, SYNTH_DIR (diverse parquets), CART_DIR (diverse carts),
OUT_HI, OUT_RAND (subset roots), SEED.
"""

import os
import sys
import glob
import random

os.environ.setdefault("CARTRIDGES_DIR", "/home/mcgrof/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/home/mcgrof/cas_out")
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
sys.path.insert(0, os.environ["CARTRIDGES_DIR"])

import torch
import pandas as pd
from transformers import AutoTokenizer

import cas_eval_table15 as E
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM

PATIENT = os.environ["PATIENT"]
CAND = int(os.environ.get("CAND", "1500"))
K = int(os.environ.get("K", "500"))
DEVICE = os.environ.get("DEVICE", "cuda:0")
SINK_MAX = int(os.environ.get("SINK_MAX", "4"))
MODEL = E.MODEL
SYNTH_DIR = os.environ.get("SYNTH_DIR", "/home/mcgrof/cas_out/synth_diverse")
CART_DIR = os.environ.get("CART_DIR", "/home/mcgrof/cas_out/iso_diverse/carts")
OUT_HI = os.environ.get("OUT_HI", "/home/mcgrof/cas_out/synth_disagree_hi")
OUT_RAND = os.environ.get("OUT_RAND", "/home/mcgrof/cas_out/synth_disagree_rand")
SEED = int(os.environ.get("SEED", "0"))
MAX_TGT = int(os.environ.get("MAX_TGT", "900"))

pstr = PATIENT.replace("patient_", "p")


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
    rng = random.Random(SEED)
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
    k, v = load_cart_kv(os.path.join(CART_DIR, f"{PATIENT}.pt"))
    cache = TrainableCache(config=ac, init_keys=k, init_values=v).to(DEVICE)

    pq = sorted(
        glob.glob(f"{SYNTH_DIR}/*/synth_*_14bq_8ba_{pstr}_n*/artifact/dataset.parquet")
    )[-1]
    df = pd.read_parquet(pq)
    idxs = list(range(len(df)))
    rng.shuffle(idxs)
    idxs = idxs[:CAND]

    @torch.no_grad()
    def disagreement(user_q, asst_a):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": user_q}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=True,
        ).flatten()
        full = tok.apply_chat_template(
            [
                {"role": "user", "content": user_q},
                {"role": "assistant", "content": asst_a},
            ],
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        ).flatten()
        tgt_start = prompt.shape[0]
        if full.shape[0] - tgt_start > MAX_TGT:
            full = full[: tgt_start + MAX_TGT]
        full = full.to(DEVICE)
        cache.clear()
        pos = torch.arange(full.shape[0], device=DEVICE)
        sids = torch.zeros(full.shape[0], dtype=torch.long, device=DEVICE)
        o = model(
            input_ids=full,
            seq_ids=sids,
            position_ids=pos,
            past_key_values=cache,
            use_cache=True,
            mode="generate",
        )
        cache.clear()
        lp = torch.log_softmax(o.logits[0].float(), dim=-1)
        total, n = 0.0, 0
        for j in range(tgt_start, full.shape[0]):
            total += float(lp[j - 1, int(full[j])])
            n += 1
        return total / max(n, 1)

    scored = []
    for c, i in enumerate(idxs):
        row = df.iloc[i]
        msgs = row["messages"]
        uq = next((m["content"] for m in msgs if m.get("role") == "user"), None)
        aa = next((m["content"] for m in msgs if m.get("role") == "assistant"), None)
        if not uq or not aa:
            continue
        scored.append((i, disagreement(uq, aa)))
        if (c + 1) % 250 == 0:
            print(f"  {PATIENT} scored {c+1}/{len(idxs)}", flush=True)

    scored.sort(key=lambda x: x[1])  # ascending logprob = descending disagreement
    hi_idx = [i for i, _ in scored[:K]]
    rand_idx = rng.sample([i for i, _ in scored], min(K, len(scored)))

    def write_subset(root, name, indices):
        d = os.path.join(root, "run", f"{name}_14bq_8ba_{pstr}_n{K}-0", "artifact")
        os.makedirs(d, exist_ok=True)
        df.iloc[indices].reset_index(drop=True).to_parquet(
            os.path.join(d, "dataset.parquet")
        )

    write_subset(OUT_HI, "synth_disagree", hi_idx)
    write_subset(OUT_RAND, "synth_control", rand_idx)
    lo = scored[0][1]
    mid = scored[len(scored) // 2][1]
    print(
        f"CAS_DISAGREE_DONE {PATIENT}: scored={len(scored)} "
        f"worst_logprob={lo:.3f} median={mid:.3f} -> hi/{K} rand/{K}",
        flush=True,
    )


if __name__ == "__main__":
    main()

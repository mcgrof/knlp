#!/usr/bin/env python3
"""Re-eval Tier-2 carts with the PROVEN letter-format eval (matched boundary), oracle +
collapse. cas_combine_eval used the <answer> format which overflows Qwen3 thinking and
depresses/flattens scores. This uses the gate's letter format on saved carts (no retrain).
Env: CART_DIR, PATIENTS, MAX_Q, OUT_JSON."""
import os, json, torch
os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/tier2")
os.environ["WANDB_DISABLED"] = "true"; os.environ["WANDB_MODE"] = "disabled"
from transformers import AutoTokenizer
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
from cartridges.data.longhealth.utils import load_longhealth_dataset
from cartridges.generation import flex_generate

MODEL = "Qwen/Qwen3-8B"
CART_DIR = os.environ["CART_DIR"]
TRAINABLE_ONLY = os.environ.get("TRAINABLE_ONLY","0")=="1"
PATIENTS = os.environ["PATIENTS"].split()
MAX_Q = int(os.environ.get("MAX_Q", "15"))
OUT_JSON = os.environ["OUT_JSON"]
DEVICE = "cuda"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = FlexQwen3ForCausalLM.from_pretrained(MODEL).to(DEVICE).to(torch.bfloat16); model.eval()
ac = AttnConfig(n_layers=model.config.num_hidden_layers, n_heads=model.config.num_key_value_heads,
                head_dim=model.config.hidden_size // model.config.num_attention_heads)


def load_full(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    def t(p): return torch.as_tensor(p.data if hasattr(p, "data") else p).to(torch.bfloat16)
    def cat(fro, tra):
        tt = [t(p) for p in tra]
        if fro:
            ff = [t(p) for p in fro]; return [torch.cat([ff[i], tt[i]], dim=2) for i in range(len(tt))]
        return tt
    if TRAINABLE_ONLY:
        return (cat(None, ck["trainable_keys"]), cat(None, ck["trainable_values"]))
    return (cat(ck.get("frozen_keys"), ck["trainable_keys"]), cat(ck.get("frozen_values"), ck["trainable_values"]))


def make_cache(cart_paths):
    per = [load_full(p) for p in cart_paths]
    nl = ac.n_layers
    ik = [torch.cat([per[c][0][li] for c in range(len(per))], dim=2) for li in range(nl)]
    iv = [torch.cat([per[c][1][li] for c in range(len(per))], dim=2) for li in range(nl)]
    return TrainableCache(config=ac, init_keys=ik, init_values=iv).to(DEVICE)


@torch.no_grad()
def letter_eval(cache, patient_ids):
    pats = load_longhealth_dataset(patient_ids)
    c = tot = degen = 0
    for patient in pats:
        for q in patient.questions[:MAX_Q]:
            prompt = (f"Question: {q.question}\nA) {q.answer_a}\nB) {q.answer_b}\nC) {q.answer_c}\n"
                      f"D) {q.answer_d}\nE) {q.answer_e}\n\nAnswer with ONLY the letter (A, B, C, D, or E). Do not explain.")
            ids = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                add_generation_prompt=True, return_tensors="pt", enable_thinking=True).to(DEVICE).flatten()
            if cache is not None: cache.clear()
            out = flex_generate(model, tok, ids, seq_ids=torch.zeros(ids.shape[0], dtype=torch.long, device=DEVICE),
                position_ids=torch.arange(ids.shape[0], device=DEVICE), max_new_tokens=32, cache=cache, temperature=0.0)
            resp = tok.decode(out.get(0, []), skip_special_tokens=True).strip()
            if resp.replace("</think>", "").strip() == "": degen += 1
            if "</think>" in resp: resp = resp.split("</think>")[-1].strip()
            letter = next((ch.upper() for ch in resp if ch.upper() in "ABCDE"), "")
            amap = {"A": q.answer_a, "B": q.answer_b, "C": q.answer_c, "D": q.answer_d, "E": q.answer_e}
            c += int(amap.get(letter, "") == q.correct); tot += 1
    return c, tot, degen


res = {"cart_dir": CART_DIR, "patients": PATIENTS, "modes": {}}
# oracle: each cart alone on its own patient
cp = {p: os.path.join(CART_DIR, f"{p}.pt") for p in PATIENTS}
oc = ot = od = 0; per_pt = {}
for p in PATIENTS:
    cache = make_cache([cp[p]])
    c, t, d = letter_eval(cache, [p]); per_pt[p] = {"correct": c, "total": t, "acc": c/max(t,1)}
    oc += c; ot += t; od += d
    print(f"[oracle] {p}: {c}/{t} = {c/max(t,1):.3f}")
    del cache; torch.cuda.empty_cache()
res["modes"]["oracle"] = {"per_patient": per_pt, "correct": oc, "total": ot, "acc": oc/max(ot,1), "degenerate": od}
print(f"[oracle] TOTAL {oc}/{ot} = {oc/max(ot,1):.3f}")
# collapse: all carts loaded, eval all patients
cache = make_cache([cp[p] for p in PATIENTS])
c, t, d = letter_eval(cache, PATIENTS)
res["modes"]["collapse"] = {"correct": c, "total": t, "acc": c/max(t,1), "degenerate": d}
print(f"[collapse] ALL-{len(PATIENTS)} {c}/{t} = {c/max(t,1):.3f}  degenerate={d}")
json.dump(res, open(OUT_JSON, "w"), indent=2)
print("REEVAL_DONE")

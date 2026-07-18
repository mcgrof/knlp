#!/usr/bin/env python3
"""Gate eval: does the trained oracle beat its no-prefix floor? Loads ONE patient's cart
reconstructed as [frozen|trainable] (the fix), evals letter-format, matched boundary
(enable_thinking=True) vs no-prefix. Env: PATIENT, CART_DIR, MODEL."""
import os, json, torch
os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cart_out")
os.environ["WANDB_DISABLED"] = "true"; os.environ["WANDB_MODE"] = "disabled"
from transformers import AutoTokenizer
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
from cartridges.data.longhealth.utils import load_longhealth_dataset
from cartridges.generation import flex_generate

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
PATIENT = os.environ.get("PATIENT", "patient_03")
CART_DIR = os.environ.get("CART_DIR", "/root/cart_out/carts")
OUT_JSON = os.environ.get("OUT_JSON", "/root/gate_eval.json")
DEVICE = "cuda"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = FlexQwen3ForCausalLM.from_pretrained(MODEL).to(DEVICE).to(torch.bfloat16); model.eval()
ac = AttnConfig(n_layers=model.config.num_hidden_layers, n_heads=model.config.num_key_value_heads,
                head_dim=model.config.hidden_size // model.config.num_attention_heads)
patient = load_longhealth_dataset([PATIENT])[0]


def full_cart(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    def t(p): return torch.as_tensor(p.data if hasattr(p, "data") else p).to(torch.bfloat16)
    def cat(fro, tra):
        tt = [t(p) for p in tra]
        if fro:
            ff = [t(p) for p in fro]; return [torch.cat([ff[i], tt[i]], dim=2) for i in range(len(tt))]
        return tt
    ik = cat(ck.get("frozen_keys"), ck["trainable_keys"]); iv = cat(ck.get("frozen_values"), ck["trainable_values"])
    return TrainableCache(config=ac, init_keys=ik, init_values=iv).to(DEVICE)


@torch.no_grad()
def evalc(cache, label):
    c = tot = degen = 0
    for q in patient.questions:
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
    acc = c / max(tot, 1)
    print(f"  [{label}] {c}/{tot} = {acc:.3f}  degenerate={degen}/{tot}")
    return {"correct": c, "total": tot, "acc": acc, "degenerate": degen}


res = {"patient": PATIENT, "model": MODEL}
res["no_prefix"] = evalc(None, "no-prefix (floor)")
cp = os.path.join(CART_DIR, f"{PATIENT}.pt")
res["oracle"] = evalc(full_cart(cp), "trained oracle")
res["delta_pp"] = round((res["oracle"]["acc"] - res["no_prefix"]["acc"]) * 100, 1)
gate = res["oracle"]["acc"] >= 0.60 and res["oracle"]["degenerate"] == 0
res["gate_pass"] = gate
with open(OUT_JSON, "w") as f: json.dump(res, f, indent=2)
print(f"  delta = {res['delta_pp']:+.1f}pp | GATE {'PASS' if gate else 'FAIL'} (oracle {res['oracle']['acc']:.1%} vs floor {res['no_prefix']['acc']:.1%})")
print("GATE_EVAL_DONE")

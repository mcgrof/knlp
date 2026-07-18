#!/usr/bin/env python3
"""Combine-at-inference eval: oracle (each cartridge alone on its patient) and collapse
(all cartridges loaded together, every patient). Also the rescue eval when pointed at
joint cartridges. Env: CART_DIR, PATIENTS, MAX_Q, OUT_JSON, MODES ("oracle collapse").

Two correctness points, both learned the hard way:
  * LETTER-format prompt + enable_thinking=True (matched to training). The <answer> format
    overflows Qwen3's thinking budget and flattens every score.
  * AUTO reconstruction. A cartridge .pt stores trainable_keys + frozen_keys. For an
    isolated cartridge frozen is a single attention-sink token (load [frozen|trainable],
    both are load-bearing). For a joint cartridge frozen is the distractor cartridges
    (load trainable-only -- the rescued target; the distractors must not be baked in).
    We auto-detect by frozen token count (a sink is tiny; distractors are large)."""
import os, json, torch
os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cas_out")
os.environ["WANDB_DISABLED"] = "true"; os.environ["WANDB_MODE"] = "disabled"
from transformers import AutoTokenizer
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
from cartridges.data.longhealth.utils import load_longhealth_dataset
from cartridges.generation import flex_generate

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
CART_DIR = os.environ["CART_DIR"]
PATIENTS = os.environ["PATIENTS"].split()
MAX_Q = int(os.environ.get("MAX_Q", "15"))
OUT_JSON = os.environ.get("OUT_JSON", "/root/cas_out/collapse.json")
MODES = os.environ.get("MODES", "oracle collapse").split()
SINK_MAX = int(os.environ.get("SINK_MAX", "4"))  # frozen<=this => attention sink, keep it
DEVICE = "cuda"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = FlexQwen3ForCausalLM.from_pretrained(MODEL).to(DEVICE).to(torch.bfloat16); model.eval()
ac = AttnConfig(n_layers=model.config.num_hidden_layers, n_heads=model.config.num_key_value_heads,
                head_dim=model.config.hidden_size // model.config.num_attention_heads)


def load_cart(path):
    """Return per-layer (K, V) for the cartridge's usable content: [frozen|trainable] when
    frozen is a sink, trainable-only when frozen is the distractor set."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    def t(p): return torch.as_tensor(p.data if hasattr(p, "data") else p).to(torch.bfloat16)
    fk = ck.get("frozen_keys") or []
    nfrozen = t(fk[0]).shape[2] if fk else 0
    use_frozen = 0 < nfrozen <= SINK_MAX
    def cat(fro, tra):
        tt = [t(p) for p in tra]
        if fro and use_frozen:
            ff = [t(p) for p in fro]; return [torch.cat([ff[i], tt[i]], dim=2) for i in range(len(tt))]
        return tt
    return (cat(ck.get("frozen_keys"), ck["trainable_keys"]), cat(ck.get("frozen_values"), ck["trainable_values"]))


def make_cache(paths):
    per = [load_cart(p) for p in paths]
    nl = ac.n_layers
    ik = [torch.cat([per[c][0][li] for c in range(len(per))], dim=2) for li in range(nl)]
    iv = [torch.cat([per[c][1][li] for c in range(len(per))], dim=2) for li in range(nl)]
    return TrainableCache(config=ac, init_keys=ik, init_values=iv).to(DEVICE)


@torch.no_grad()
def letter_eval(cache, patient_ids):
    c = tot = degen = 0
    for patient in load_longhealth_dataset(patient_ids):
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


def main():
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    cp = {p: os.path.join(CART_DIR, f"{p}.pt") for p in PATIENTS}
    for p in PATIENTS:
        assert os.path.exists(cp[p]), f"missing cartridge {cp[p]}"
    res = {"cart_dir": CART_DIR, "patients": PATIENTS, "max_q": MAX_Q, "modes": {}}

    if "oracle" in MODES:
        oc = ot = od = 0; per = {}
        for p in PATIENTS:
            cache = make_cache([cp[p]]); c, t, d = letter_eval(cache, [p])
            per[p] = {"correct": c, "total": t, "acc": c / max(t, 1)}; oc += c; ot += t; od += d
            print(f"[oracle] {p}: {c}/{t} = {c/max(t,1):.3f}")
            del cache; torch.cuda.empty_cache()
        res["modes"]["oracle"] = {"per_patient": per, "correct": oc, "total": ot, "acc": oc / max(ot, 1), "degenerate": od}
        print(f"[oracle] TOTAL {oc}/{ot} = {oc/max(ot,1):.3f}")

    if "collapse" in MODES:
        cache = make_cache([cp[p] for p in PATIENTS]); c, t, d = letter_eval(cache, PATIENTS)
        res["modes"]["collapse"] = {"correct": c, "total": t, "acc": c / max(t, 1), "degenerate": d}
        print(f"[collapse] ALL-{len(PATIENTS)} {c}/{t} = {c/max(t,1):.3f}  degenerate={d}")
        del cache; torch.cuda.empty_cache()

    json.dump(res, open(OUT_JSON, "w"), indent=2)
    print(f"CAS_COMBINE_EVAL_DONE {OUT_JSON}")


if __name__ == "__main__":
    main()

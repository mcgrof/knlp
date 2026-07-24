#!/usr/bin/env python3
"""Serving A/B microbenchmark: full-context baseline vs Cartridge, same queries.

Fills the serving half of a cartridge-economics benchmark record — the half the
build-cost record cannot supply. For each patient with a trained Cartridge it runs
the SAME LongHealth letter-format questions two ways and times them:

  baseline  : the patient's document is in the prompt; the model prefills
              [document + question] fresh every query (no Cartridge).
  cartridge : the Cartridge KV is loaded as a prefix; the model prefills only
              [question]. The document is never re-processed.

It reports time-to-first-token (TTFT) P50/P95 for both paths, the prefill wall-
time saved per query (baseline TTFT − cartridge TTFT — the calculator's
`prefillMsSaved` input), a decode-throughput estimate, and letter-accuracy for
both paths (qualityBaseline vs qualityCartridge, measured apples-to-apples in one
pass). Output is a JSON block shaped for the cartridge-benchmarks record
(ttftMs / qualityBaseline / qualityCartridge / prefill saving / per-path
completed-requests + wall seconds).

Reuses the combine-eval building blocks verbatim (LETTER prompt +
enable_thinking=True matched to training, sink-vs-distractor cart reconstruction,
flex_generate), so quality here is comparable to the combine-eval numbers.

Env: CART_DIR, PATIENTS, MAX_Q, OUT_JSON, DEVICE, ANSWER_TOKENS, WARMUP.
Run on the same hardware named in the record (H100) so TTFT is representative.
"""
import os, json, time, statistics, torch

os.environ.setdefault("CARTRIDGES_DIR", "/home/mcgrof/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", os.environ.get("OUT_DIR", "/home/mcgrof/cas_out"))
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

from transformers import AutoTokenizer
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
from cartridges.data.longhealth.utils import load_longhealth_dataset
from cartridges.generation import flex_generate

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
CART_DIR = os.environ["CART_DIR"]
PATIENTS = os.environ["PATIENTS"].split()
MAX_Q = int(os.environ.get("MAX_Q", "10"))
OUT_JSON = os.environ.get("OUT_JSON", "/home/mcgrof/cas_out/serve_ab.json")
DEVICE = os.environ.get("DEVICE", "cuda")
ANSWER_TOKENS = int(os.environ.get("ANSWER_TOKENS", "32"))
WARMUP = int(os.environ.get("WARMUP", "2"))
SINK_MAX = int(os.environ.get("SINK_MAX", "4"))

di = int(DEVICE.split(":")[1]) if ":" in DEVICE else 0
torch.cuda.set_device(di)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = FlexQwen3ForCausalLM.from_pretrained(MODEL).to(DEVICE).to(torch.bfloat16)
model.eval()
ac = AttnConfig(n_layers=model.config.num_hidden_layers,
                n_heads=model.config.num_key_value_heads,
                head_dim=model.config.hidden_size // model.config.num_attention_heads)


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

    return (cat(ck.get("frozen_keys"), ck["trainable_keys"]),
            cat(ck.get("frozen_values"), ck["trainable_values"]))


def make_cache(paths):
    per = [load_cart(p) for p in paths]
    nl = ac.n_layers
    ik = [torch.cat([per[c][0][li] for c in range(len(per))], dim=2) for li in range(nl)]
    iv = [torch.cat([per[c][1][li] for c in range(len(per))], dim=2) for li in range(nl)]
    return TrainableCache(config=ac, init_keys=ik, init_values=iv).to(DEVICE)


def letter_prompt(q):
    return (f"Question: {q.question}\nA) {q.answer_a}\nB) {q.answer_b}\nC) {q.answer_c}\n"
            f"D) {q.answer_d}\nE) {q.answer_e}\n\nAnswer with ONLY the letter "
            f"(A, B, C, D, or E). Do not explain.")


def encode(user_content):
    return tok.apply_chat_template(
        [{"role": "user", "content": user_content}], tokenize=True,
        add_generation_prompt=True, return_tensors="pt", enable_thinking=True
    ).to(DEVICE).flatten()


@torch.no_grad()
def gen(ids, cache, max_new_tokens):
    if cache is not None:
        cache.clear()
    return flex_generate(
        model, tok, ids,
        seq_ids=torch.zeros(ids.shape[0], dtype=torch.long, device=DEVICE),
        position_ids=torch.arange(ids.shape[0], device=DEVICE),
        max_new_tokens=max_new_tokens, cache=cache, temperature=0.0)


def score(out, q):
    resp = tok.decode(out.get(0, []), skip_special_tokens=True).strip()
    if "</think>" in resp:
        resp = resp.split("</think>")[-1].strip()
    letter = next((ch.upper() for ch in resp if ch.upper() in "ABCDE"), "")
    amap = {"A": q.answer_a, "B": q.answer_b, "C": q.answer_c, "D": q.answer_d, "E": q.answer_e}
    return int(amap.get(letter, "\0") == q.correct)


def timed(ids, cache, max_new_tokens):
    """Wall-clock ms for a generation of max_new_tokens (cuda-synced)."""
    torch.cuda.synchronize()
    t0 = time.time()
    out = gen(ids, cache, max_new_tokens)
    torch.cuda.synchronize()
    return (time.time() - t0) * 1000.0, out


def pctl(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1))))
    return round(xs[k], 1)


def main():
    os.makedirs(os.path.dirname(OUT_JSON) or ".", exist_ok=True)
    for p in PATIENTS:
        assert os.path.exists(os.path.join(CART_DIR, f"{p}.pt")), f"missing cartridge {p}.pt"

    ttft_base, ttft_cart = [], []
    dec_base, dec_cart = [], []            # decode tok/s per query
    base_prefill_toks, cart_prefill_toks = [], []
    correct_base = correct_cart = total = 0
    warmed = False

    for patient in load_longhealth_dataset(PATIENTS):
        pid = patient.patient_id if hasattr(patient, "patient_id") else None
        doc = "\n\n".join(patient.texts.values())
        cache = make_cache([os.path.join(CART_DIR, f"{patient.patient_id}.pt")])
        for q in patient.questions[:MAX_Q]:
            qp = letter_prompt(q)
            base_ids = encode(f"{doc}\n\n{qp}")
            cart_ids = encode(qp)

            if not warmed:
                for _ in range(WARMUP):
                    gen(base_ids, None, 4)
                    gen(cart_ids, cache, 4)
                warmed = True

            # TTFT = time to first token (max_new_tokens=1); prefill dominates
            tb1, _ = timed(base_ids, None, 1)
            tc1, _ = timed(cart_ids, cache, 1)
            # full answer: timing gives decode throughput, output gives accuracy
            tbN, ob = timed(base_ids, None, ANSWER_TOKENS)
            tcN, oc = timed(cart_ids, cache, ANSWER_TOKENS)

            ttft_base.append(tb1)
            ttft_cart.append(tc1)
            if tbN > tb1:
                dec_base.append((ANSWER_TOKENS - 1) / ((tbN - tb1) / 1000.0))
            if tcN > tc1:
                dec_cart.append((ANSWER_TOKENS - 1) / ((tcN - tc1) / 1000.0))
            base_prefill_toks.append(int(base_ids.shape[0]))
            cart_prefill_toks.append(int(cart_ids.shape[0]))
            correct_base += score(ob, q)
            correct_cart += score(oc, q)
            total += 1
        del cache
        torch.cuda.empty_cache()
        print(f"[serve-ab] {patient.patient_id}: {total} queries so far", flush=True)

    med_base = statistics.median(ttft_base) if ttft_base else None
    med_cart = statistics.median(ttft_cart) if ttft_cart else None
    prefill_ms_saved = round(med_base - med_cart, 1) if (med_base and med_cart) else None
    res = {
        "model": MODEL, "cart_dir": CART_DIR, "patients": PATIENTS,
        "n_queries": total, "answer_tokens": ANSWER_TOKENS,
        "avg_baseline_prefill_tokens": round(statistics.mean(base_prefill_toks), 0) if base_prefill_toks else None,
        "avg_cartridge_prefill_tokens": round(statistics.mean(cart_prefill_toks), 0) if cart_prefill_toks else None,
        "ttftMs": {
            "baselineP50": pctl(ttft_base, 50), "baselineP95": pctl(ttft_base, 95),
            "cartridgeP50": pctl(ttft_cart, 50), "cartridgeP95": pctl(ttft_cart, 95),
        },
        "prefillMsSavedMedian": prefill_ms_saved,
        "decodeTokPerSec": {
            "baseline": round(statistics.mean(dec_base), 1) if dec_base else None,
            "cartridge": round(statistics.mean(dec_cart), 1) if dec_cart else None,
        },
        "qualityMetric": "LongHealth letter accuracy",
        "qualityBaseline": round(correct_base / max(total, 1), 4),
        "qualityCartridge": round(correct_cart / max(total, 1), 4),
    }
    json.dump(res, open(OUT_JSON, "w"), indent=2)
    print(json.dumps(res, indent=2))
    print(f"CAS_SERVE_AB_DONE {OUT_JSON}")


if __name__ == "__main__":
    main()

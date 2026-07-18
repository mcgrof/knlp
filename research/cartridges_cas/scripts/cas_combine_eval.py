#!/usr/bin/env python3
"""CAS smoke -- gap #1: COMBINE-AT-INFERENCE + collapse/oracle eval.

Cartridge trainable keys are position-free free vectors (never RoPE-rotated), and the
query's positions are offset by num_cartridge_tokens() automatically. So combining N
cartridges == concatenating their trainable K/V along the token axis, all with the
CARTRIDGE_SEQ_ID (-1 = globally visible). That co-loads every patient's memory at once,
which is exactly the CAS collapse condition.

Modes:
  oracle   -- each cartridge evaluated ALONE on its own patient's questions (upper bound)
  collapse -- ALL cartridges combined, evaluated on every patient's questions

Env:
  CART_DIR      dir of per-patient cartridges (<patient>.pt)   default /root/cart_out/carts
  PATIENTS      space-sep patient ids to include
  MAX_Q         max questions per patient (default 10)
  MAX_NEW       max new tokens per generation (default 256)
  OUT_JSON      results path (default /root/cas_out/collapse_eval.json)
  MODES         space-sep subset of {oracle,collapse} (default both)
Single-process, no torchrun -> no distributed."""
import os, json, glob, time
os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/root/cas_out")
os.environ["WANDB_DISABLED"] = "true"; os.environ["WANDB_MODE"] = "disabled"

import torch
from transformers import AutoTokenizer
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.models.config import HFModelConfig
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
from cartridges.data.longhealth.evals import LongHealthMultipleChoiceGenerateDataset
from cartridges.generation import flex_generate

MODEL = "Qwen/Qwen3-8B"
CART_DIR = os.environ.get("CART_DIR", "/root/cart_out/carts")
PATIENTS = os.environ.get("PATIENTS", "").split()
MAX_Q = int(os.environ.get("MAX_Q", "10"))
MAX_NEW = int(os.environ.get("MAX_NEW", "256"))
OUT_JSON = os.environ.get("OUT_JSON", "/root/cas_out/collapse_eval.json")
MODES = os.environ.get("MODES", "oracle collapse").split()
DEVICE = "cuda"
assert PATIENTS, "set PATIENTS"


def load_cart_tensors(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # RELOAD FIX: reconstruct the FULL trained cart = [frozen | trainable]. The cart trained
    # with a leading frozen sink token (KVFromText defaults num_frozen_tokens=1); dropping it
    # (loading trainable_keys only) turns the cart into a degenerate control prefix.
    def _t(p):
        return torch.as_tensor(p.data if hasattr(p, "data") else p).to(torch.bfloat16)

    def cat(fro, tra):
        t = [_t(p) for p in tra]
        if fro:
            f = [_t(p) for p in fro]
            return [torch.cat([f[i], t[i]], dim=2) for i in range(len(t))]
        return t

    tk = cat(ckpt.get("frozen_keys"), ckpt["trainable_keys"])
    tv = cat(ckpt.get("frozen_values"), ckpt["trainable_values"])
    return tk, tv


def build_combined_cache(cart_paths, attn_config):
    """Concatenate N cartridges' trainable K/V along the token axis -> one cache,
    all tokens seq_id=-1 (globally visible)."""
    per_cart = [load_cart_tensors(p) for p in cart_paths]
    n_layers = attn_config.n_layers
    init_keys, init_values = [], []
    for li in range(n_layers):
        ks = torch.cat([pc[0][li] for pc in per_cart], dim=2)  # (1,H,sum_T,D)
        vs = torch.cat([pc[1][li] for pc in per_cart], dim=2)
        init_keys.append(ks)
        init_values.append(vs)
    tok_counts = [pc[0][0].shape[2] for pc in per_cart]
    cache = TrainableCache(config=attn_config, init_keys=init_keys, init_values=init_values)
    return cache.to(DEVICE), tok_counts


@torch.no_grad()
def eval_cache(model, tokenizer, cache, patient_ids, max_q):
    ds = LongHealthMultipleChoiceGenerateDataset.Config(
        patient_ids=patient_ids, max_questions=max_q * len(patient_ids), cot=False,
    ).instantiate(tokenizer=tokenizer, seed=42)
    n = len(ds)
    bs = 16
    correct, total = 0, 0
    details = []

    def no_think_ids(prompt):
        # MATCHED BOUNDARY: training built assistant turns as `<|im_start|>assistant\n`
        # + content with NO empty <think></think> block, so eval must use
        # enable_thinking=True (which ends the prompt at `assistant\n`, matching training).
        # enable_thinking=False injects an empty think block the cart never saw -> mismatch.
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True,
                return_tensors="pt", enable_thinking=True)
        except TypeError:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True,
                return_tensors="pt")

    for start in range(0, n, bs):
        elems = [(i, ds[i]) for i in range(start, min(start + bs, n))]
        # re-tokenize each prompt with thinking disabled
        ids_list = [no_think_ids(e.prompt) for _, e in elems]
        cache.clear()  # drop any appended generation KV; keep trainable cartridge KV
        input_ids = torch.cat([ii[0] for ii in ids_list]).to(DEVICE)
        seq_ids = torch.cat([torch.full((ii.shape[1],), idx, dtype=torch.long, device=DEVICE)
                             for (idx, _), ii in zip(elems, ids_list)])
        position_ids = torch.cat([torch.arange(ii.shape[1], device=DEVICE) for ii in ids_list])
        pred_ids = flex_generate(
            model=model, tokenizer=tokenizer, input_ids=input_ids, seq_ids=seq_ids,
            position_ids=position_ids, cache=cache, max_new_tokens=MAX_NEW, temperature=0.0,
        )
        emap = {idx: e for idx, e in elems}
        for seq_id, ids in pred_ids.items():
            e = emap[seq_id]
            pred = tokenizer.decode(ids, skip_special_tokens=True)
            ok, extras = ds.score(pred=pred, answer=e.answer, convo_id=e.convo_id)
            correct += int(bool(ok)); total += 1
            details.append({"convo_id": e.convo_id, "correct": bool(ok),
                            "extracted": extras.get("extracted_pred")})
    return correct, total, details


def main():
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = HFModelConfig(pretrained_model_name_or_path=MODEL, model_cls=FlexQwen3ForCausalLM
                          ).instantiate().to(DEVICE).to(torch.bfloat16)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    attn_config = AttnConfig(
        n_layers=model.config.num_hidden_layers,
        n_heads=model.config.num_key_value_heads,
        head_dim=getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads),
    )
    cart_paths = {p: os.path.join(CART_DIR, f"{p}.pt") for p in PATIENTS}
    for p, cp in cart_paths.items():
        assert os.path.exists(cp), f"missing cartridge {cp}"

    results = {"model": MODEL, "patients": PATIENTS, "max_q": MAX_Q, "cart_dir": CART_DIR, "modes": {}}

    if "oracle" in MODES:
        t0 = time.time(); per_pt = {}; c_sum = t_sum = 0
        for p in PATIENTS:
            cache, toks = build_combined_cache([cart_paths[p]], attn_config)
            c, t, _ = eval_cache(model, tokenizer, cache, [p], MAX_Q)
            per_pt[p] = {"correct": c, "total": t, "acc": (c / t if t else 0.0), "cart_tokens": toks[0]}
            c_sum += c; t_sum += t
            print(f"[oracle] {p}: {c}/{t} = {c/t if t else 0:.3f}")
            del cache; torch.cuda.empty_cache()
        results["modes"]["oracle"] = {"per_patient": per_pt, "correct": c_sum, "total": t_sum,
                                      "acc": (c_sum / t_sum if t_sum else 0.0), "secs": time.time() - t0}
        print(f"[oracle] TOTAL {c_sum}/{t_sum} = {c_sum/t_sum if t_sum else 0:.3f}")

    if "collapse" in MODES:
        t0 = time.time()
        cache, toks = build_combined_cache([cart_paths[p] for p in PATIENTS], attn_config)
        print(f"[collapse] combined cart tokens={sum(toks)} ({toks})")
        c, t, _ = eval_cache(model, tokenizer, cache, PATIENTS, MAX_Q)
        results["modes"]["collapse"] = {"correct": c, "total": t, "acc": (c / t if t else 0.0),
                                        "combined_tokens": sum(toks), "secs": time.time() - t0}
        print(f"[collapse] ALL-{len(PATIENTS)}-LOADED {c}/{t} = {c/t if t else 0:.3f}")
        del cache; torch.cuda.empty_cache()

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"CAS_COMBINE_EVAL_DONE {OUT_JSON}")


if __name__ == "__main__":
    main()

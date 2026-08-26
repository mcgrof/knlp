#!/usr/bin/env python3
"""OP-Cart: on-policy distillation for a trainable KV cartridge.

Implements the first-run contract of op-cart-20260825 in knlp-key-results:
train ONE isolated cartridge from a fixed starting checkpoint under a
declared (prefix_source, divergence) cell, with the base model frozen and
only the cartridge updated.

Arms (env ARM):
  tp_fkl_stored   continued baseline: stored teacher answers + stored
                  top-20 teacher logprobs (the existing recipe's loss)
  tp_fkl          teacher-prefix, live full-vocab forward KL
  tp_rkl          teacher-prefix, live full-vocab reverse KL
  sp_fkl          student-prefix (fresh rollout each step), forward KL
  sp_rkl          student-prefix, reverse KL

For the live arms the frozen teacher scores (record + question + prefix)
and the cartridge student scores (cartridge + question + prefix) on the
IDENTICAL prefix, full vocabulary, at the answer positions.  Rollouts are
sampled from the current cartridge student with no grad and persisted.
K = OPCART_REFRESH optimizer steps reuse one rollout batch (default 1).

Freeze contract and trajectory-source assertions are enforced every run;
the run aborts rather than train under a broken contract.

Env (beyond the arm):
  PATIENT, DATA_PARQUET, RECORDS_DIR, CART_INIT (starting checkpoint .pt),
  STEPS, ACCUM, LR, ROLLOUT_TOKENS, ROLLOUT_TEMP, EVAL_EVERY, MAX_Q,
  OUT_DIR, SEED.
"""

import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridges")
OUT_DIR = os.environ.get("OUT_DIR", "/root/cas_out/opcart")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", OUT_DIR)
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.datasets import TrainDataset, DataSource
from cartridges.data.longhealth.utils import load_longhealth_dataset
from cartridges.generation import flex_generate
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
ARM = os.environ["ARM"]
PATIENT = os.environ["PATIENT"]
DATA_PARQUET = os.environ["DATA_PARQUET"]
RECORDS_DIR = os.environ.get("RECORDS_DIR", "/root/cart_records")
CART_INIT = os.environ["CART_INIT"]
STEPS = int(os.environ.get("STEPS", "100"))
ACCUM = int(os.environ.get("ACCUM", "8"))
LR = float(os.environ.get("LR", "2e-2"))
ROLLOUT_TOKENS = int(os.environ.get("ROLLOUT_TOKENS", "96"))
ROLLOUT_TEMP = float(os.environ.get("ROLLOUT_TEMP", "0.7"))
REFRESH = int(os.environ.get("OPCART_REFRESH", "1"))
EVAL_EVERY = [int(x) for x in os.environ.get("EVAL_AT", "0,10,50,100").split(",")]
MAX_Q = int(os.environ.get("MAX_Q", "20"))
SEED = int(os.environ.get("SEED", "0"))
DEVICE = "cuda"

assert ARM in ("tp_fkl_stored", "tp_fkl", "tp_rkl", "sp_fkl", "sp_rkl"), ARM
assert (
    REFRESH == 1
), "only K=1 (fresh rollouts every step) is implemented; do not mislabel staleness"
PREFIX_SOURCE = "student" if ARM.startswith("sp_") else "teacher"
DIVERGENCE = "rkl" if ARM.endswith("rkl") else "fkl"

torch.manual_seed(SEED)
rng = random.Random(SEED)


# ---------------------------------------------------------------------------
# model, cartridge, data
# ---------------------------------------------------------------------------


def load_cart_as_trainable(path, attn_config):
    """Rebuild the checkpoint as a TrainableCache preserving the
    frozen/trainable split (isolated carts carry a small attention sink)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)

    def t(p):
        return torch.as_tensor(p.data if hasattr(p, "data") else p).to(torch.bfloat16)

    tk = [t(p) for p in ck["trainable_keys"]]
    tv = [t(p) for p in ck["trainable_values"]]
    fk = ck.get("frozen_keys") or []
    fv = ck.get("frozen_values") or []
    if fk:
        ik = [torch.cat([t(fk[i]), tk[i]], dim=2) for i in range(len(tk))]
        iv = [torch.cat([t(fv[i]), tv[i]], dim=2) for i in range(len(tv))]
        nfrozen = t(fk[0]).shape[2]
    else:
        ik, iv, nfrozen = tk, tv, 0
    cache = TrainableCache(
        config=attn_config, init_keys=ik, init_values=iv, num_frozen_tokens=nfrozen
    ).to(DEVICE)
    return cache, nfrozen


def assert_freeze_contract(models, cache):
    for m in models:
        for p in m.parameters():
            assert not p.requires_grad, "frozen base model has requires_grad param"
    assert any(p.requires_grad for p in cache.parameters()), "cartridge not trainable"


def assert_no_model_grads(models):
    for m in models:
        for name, p in m.named_parameters():
            assert p.grad is None, f"frozen model param received grad: {name}"


print(f"[opcart] arm={ARM} prefix_source={PREFIX_SOURCE} divergence={DIVERGENCE}")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = FlexQwen3ForCausalLM.from_pretrained(MODEL).to(DEVICE).to(torch.bfloat16)
model.eval()
model.requires_grad_(False)
attn_config = AttnConfig(
    n_layers=model.config.num_hidden_layers,
    n_heads=model.config.num_key_value_heads,
    head_dim=model.config.head_dim,
)
cache, NUM_FROZEN = load_cart_as_trainable(CART_INIT, attn_config)
# the teacher is a stock transformers Qwen3 with the SAME weights: standard
# forward semantics, no flex-mask plumbing, provably cartridge-free prompt
teacher = (
    AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    .to(DEVICE)
    .eval()
)
teacher.requires_grad_(False)
assert_freeze_contract([model, teacher], cache)

record_text = (Path(RECORDS_DIR) / f"{PATIENT}.txt").read_text()
record_ids = tok(record_text, return_tensors="pt").input_ids.to(DEVICE)

# training elements: question ids, stored teacher answer ids, stored top-20
dataset = TrainDataset.Config(
    data_sources=[DataSource(path=DATA_PARQUET, type="local")],
    top_k_logits=20,
    packed_seq_length=2048,
    packing_mode="truncate",
).instantiate(tokenizer=tok, seed=SEED)
print(f"[opcart] dataset elements: {len(dataset.elements)}")


def split_element(el):
    """An element packs [prompt tokens | answer tokens]; the stored sparse
    targets index answer positions.  The first target index marks the
    answer start (library convention: topk_token_idxs are 1-based next-token
    positions)."""
    idxs = el.topk_token_idxs
    ans_start = int(idxs.min().item()) - 1
    assert ans_start > 0, "cannot locate answer start in element"
    prompt_ids = el.input_ids[:ans_start]
    answer_ids = el.input_ids[ans_start:]
    return prompt_ids, answer_ids, ans_start


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


@torch.no_grad()
def teacher_logits_on(prefix_prompt_ids, answer_ids):
    """Frozen teacher: full record + the element prompt + the answer prefix.
    Returns logits at the answer positions (predicting each answer token)."""
    ids = torch.cat([record_ids.flatten(), prefix_prompt_ids, answer_ids]).unsqueeze(0)
    out = teacher(input_ids=ids)
    n_ans = answer_ids.shape[0]
    start = ids.shape[1] - n_ans - 1
    return out.logits[0, start : start + n_ans].float()


def student_logits_on(prompt_ids, answer_ids, with_grad):
    """Cartridge student: cartridge + element prompt + answer prefix."""
    ids = torch.cat([prompt_ids, answer_ids])
    seq_ids = torch.zeros(ids.shape[0], dtype=torch.long, device=DEVICE)
    ncart = cache.num_cartridge_tokens()
    pos = torch.arange(ids.shape[0], device=DEVICE)
    cache.clear()
    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with ctx, torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(
            input_ids=ids,
            seq_ids=seq_ids,
            position_ids=pos,
            use_cache=True,
            past_key_values=cache,
        )
    n_ans = answer_ids.shape[0]
    start = ids.shape[0] - n_ans - 1
    return out.logits[0, start : start + n_ans].float()


@torch.no_grad()
def rollout(prompt_ids):
    """Sample an answer from the current cartridge student.  No grads; the
    sampled ids are data."""
    cache.clear()
    out = flex_generate(
        model,
        tok,
        prompt_ids,
        seq_ids=torch.zeros(prompt_ids.shape[0], dtype=torch.long, device=DEVICE),
        position_ids=torch.arange(prompt_ids.shape[0], device=DEVICE),
        max_new_tokens=ROLLOUT_TOKENS,
        cache=cache,
        temperature=ROLLOUT_TEMP,
    )
    ids = out.get(0, [])
    ids = torch.as_tensor(ids, dtype=torch.long, device=DEVICE)
    return ids


def divergence_loss(t_logits, s_logits):
    lp_t = t_logits.log_softmax(-1)
    lp_s = s_logits.log_softmax(-1)
    if DIVERGENCE == "fkl":
        return (lp_t.exp() * (lp_t - lp_s)).sum(-1).mean()
    return (lp_s.exp() * (lp_s - lp_t)).sum(-1).mean()


def stored_topk_loss(el):
    """The existing recipe's sparse top-20 distillation CE (baseline arm)."""
    ids = el.input_ids.to(DEVICE)
    idxs = el.topk_token_idxs.to(DEVICE)
    tids = el.topk_token_ids.to(DEVICE)
    tlp = el.topk_logprobs.to(DEVICE)
    seq_ids = torch.zeros(ids.shape[0], dtype=torch.long, device=DEVICE)
    pos = torch.arange(ids.shape[0], device=DEVICE)
    cache.clear()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(
            input_ids=ids,
            seq_ids=seq_ids,
            position_ids=pos,
            use_cache=True,
            past_key_values=cache,
        )
        vocab = out.logits.shape[-1]
        gi = idxs - 1
        valid = (gi >= 0) & (gi < ids.shape[0]) & (tids >= 0) & (tids < vocab)
        if int(valid.sum()) == 0:
            return None
        lp = F.log_softmax(out.logits.float(), dim=-1)[0, gi[valid], tids[valid]]
        return (-tlp[valid].exp() * lp).mean()


# ---------------------------------------------------------------------------
# exact reference test for the divergences (contract section 4)
# ---------------------------------------------------------------------------


def reference_test():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(3, 7, generator=g)
    b = torch.randn(3, 7, generator=g)
    lp_a, lp_b = a.log_softmax(-1), b.log_softmax(-1)
    fkl = (lp_a.exp() * (lp_a - lp_b)).sum(-1)
    rkl = (lp_b.exp() * (lp_b - lp_a)).sum(-1)
    ref_f = torch.stack(
        [F.kl_div(lp_b[i], lp_a[i].exp(), reduction="sum") for i in range(3)]
    )
    ref_r = torch.stack(
        [F.kl_div(lp_a[i], lp_b[i].exp(), reduction="sum") for i in range(3)]
    )
    assert torch.allclose(fkl, ref_f, atol=1e-6), "FKL reference mismatch"
    assert torch.allclose(rkl, ref_r, atol=1e-6), "RKL reference mismatch"
    print("[opcart] FKL/RKL reference test passed")


# ---------------------------------------------------------------------------
# held-out free-generation eval (LongHealth letters) + pathology metrics
# ---------------------------------------------------------------------------


def ngram_repeat_fraction(ids, n=3):
    if len(ids) < n + 1:
        return 0.0
    grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


@torch.no_grad()
def free_gen_eval():
    correct = total = degen = 0
    rep3 = []
    lengths = []
    for patient in load_longhealth_dataset([PATIENT]):
        for q in patient.questions[:MAX_Q]:
            prompt = (
                f"Question: {q.question}\nA) {q.answer_a}\nB) {q.answer_b}\n"
                f"C) {q.answer_c}\nD) {q.answer_d}\nE) {q.answer_e}\n\n"
                "Answer with ONLY the letter (A, B, C, D, or E). Do not explain."
            )
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
            cache.clear()
            out = flex_generate(
                model,
                tok,
                ids,
                seq_ids=torch.zeros(ids.shape[0], dtype=torch.long, device=DEVICE),
                position_ids=torch.arange(ids.shape[0], device=DEVICE),
                max_new_tokens=32,
                cache=cache,
                temperature=0.0,
            )
            out_ids = list(out.get(0, []))
            resp = tok.decode(out_ids, skip_special_tokens=True).strip()
            lengths.append(len(out_ids))
            rep3.append(ngram_repeat_fraction(out_ids))
            if resp.replace("</think>", "").strip() == "":
                degen += 1
            if "</think>" in resp:
                resp = resp.split("</think>")[-1].strip()
            letter = next((c.upper() for c in resp if c.upper() in "ABCDE"), "")
            amap = {
                "A": q.answer_a,
                "B": q.answer_b,
                "C": q.answer_c,
                "D": q.answer_d,
                "E": q.answer_e,
            }
            correct += int(amap.get(letter, "") == q.correct)
            total += 1
    return dict(
        acc=correct / max(total, 1),
        correct=correct,
        total=total,
        degenerate=degen,
        mean_rep3=sum(rep3) / max(len(rep3), 1),
        mean_len=sum(lengths) / max(len(lengths), 1),
    )


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def main():
    reference_test()
    out_dir = Path(OUT_DIR) / ARM
    out_dir.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(
        cache.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.0
    )
    run = dict(
        arm=ARM,
        prefix_source=PREFIX_SOURCE,
        divergence=DIVERGENCE,
        patient=PATIENT,
        cart_init=CART_INIT,
        steps=STEPS,
        accum=ACCUM,
        lr=LR,
        rollout_tokens=ROLLOUT_TOKENS,
        rollout_temp=ROLLOUT_TEMP,
        refresh=REFRESH,
        seed=SEED,
        history=[],
        evals={},
        cost=dict(
            rollout_tokens=0,
            teacher_tokens=0,
            train_tokens=0,
            rollout_s=0.0,
            teacher_s=0.0,
            update_s=0.0,
            eval_s=0.0,
        ),
        rollout_samples=[],
    )

    def maybe_eval(step):
        if step in EVAL_EVERY:
            t0 = time.time()
            run["evals"][str(step)] = free_gen_eval()
            run["cost"]["eval_s"] += time.time() - t0
            e = run["evals"][str(step)]
            print(
                f"[opcart] eval@{step}: acc={e['acc']:.3f} degen={e['degenerate']} "
                f"rep3={e['mean_rep3']:.3f} len={e['mean_len']:.1f}",
                flush=True,
            )

    maybe_eval(0)
    t_start = time.time()
    rollout_bank = []
    for step in range(1, STEPS + 1):
        opt.zero_grad()
        if PREFIX_SOURCE == "student" and (step - 1) % REFRESH == 0:
            rollout_bank = []
        accum_loss = 0.0
        n_used = 0
        for a in range(ACCUM):
            el = dataset.elements[rng.randrange(len(dataset.elements))]
            if ARM == "tp_fkl_stored":
                loss = stored_topk_loss(el)
                if loss is None:
                    continue
                run["cost"]["train_tokens"] += int(el.input_ids.shape[0])
            else:
                prompt_ids, teacher_answer_ids, _ = split_element(el)
                prompt_ids = prompt_ids.to(DEVICE)
                if PREFIX_SOURCE == "teacher":
                    answer_ids = teacher_answer_ids.to(DEVICE)
                else:
                    bank_key = a % max(len(rollout_bank), 1)
                    if len(rollout_bank) <= a:
                        t0 = time.time()
                        y = rollout(prompt_ids)
                        run["cost"]["rollout_s"] += time.time() - t0
                        run["cost"]["rollout_tokens"] += int(y.shape[0])
                        rollout_bank.append((prompt_ids, y))
                        if len(run["rollout_samples"]) < 20:
                            run["rollout_samples"].append(
                                dict(step=step, text=tok.decode(y.tolist()))
                            )
                    prompt_ids, answer_ids = rollout_bank[
                        bank_key if REFRESH > 1 else a
                    ]
                if answer_ids.shape[0] == 0:
                    continue
                # trajectory-source assertion: student prefixes must not be
                # byte-identical to the stored teacher answer (contract 9)
                if (
                    PREFIX_SOURCE == "student"
                    and answer_ids.shape[0] == teacher_answer_ids.shape[0]
                ):
                    assert not torch.equal(
                        answer_ids.cpu(), teacher_answer_ids
                    ), "student arm is replaying stored teacher answers"
                t0 = time.time()
                t_logits = teacher_logits_on(prompt_ids, answer_ids)
                run["cost"]["teacher_s"] += time.time() - t0
                run["cost"]["teacher_tokens"] += int(
                    record_ids.shape[1] + prompt_ids.shape[0] + answer_ids.shape[0]
                )
                s_logits = student_logits_on(prompt_ids, answer_ids, with_grad=True)
                run["cost"]["train_tokens"] += int(
                    prompt_ids.shape[0] + answer_ids.shape[0]
                )
                loss = divergence_loss(t_logits, s_logits)
            (loss / ACCUM).backward()
            accum_loss += float(loss.detach()) / ACCUM
            n_used += 1
        assert_no_model_grads([model, teacher])
        t0 = time.time()
        torch.nn.utils.clip_grad_norm_(cache.parameters(), 1.0)
        opt.step()
        run["cost"]["update_s"] += time.time() - t0
        if step % 5 == 0 or step == 1:
            print(
                f"[opcart] step {step:4d} loss={accum_loss:.4f} used={n_used} "
                f"wall={time.time() - t_start:.0f}s",
                flush=True,
            )
        run["history"].append(dict(step=step, loss=accum_loss, used=n_used))
        maybe_eval(step)

    run["cost"]["total_wall_s"] = time.time() - t_start
    run["cost"]["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    # save the trained cartridge in library format, preserving the frozen
    # sink so the eval loader reconstructs the same geometry as the baseline
    tk = [p.detach() for p in cache.trainable_keys]
    tv = [p.detach() for p in cache.trainable_values]
    fk = fv = None
    if NUM_FROZEN:
        raw_fk = getattr(cache, "frozen_keys", None) or getattr(
            cache, "_frozen_keys", None
        )
        raw_fv = getattr(cache, "frozen_values", None) or getattr(
            cache, "_frozen_values", None
        )
        assert raw_fk is not None, "cannot locate frozen keys on TrainableCache"
        fk = [torch.as_tensor(t).detach().contiguous().cpu() for t in raw_fk]
        fv = [torch.as_tensor(t).detach().contiguous().cpu() for t in raw_fv]
    torch.save(
        {
            "trainable_keys": [t.contiguous().cpu() for t in tk],
            "trainable_values": [t.contiguous().cpu() for t in tv],
            "frozen_keys": fk,
            "frozen_values": fv,
        },
        out_dir / f"{PATIENT}.pt",
    )
    with open(out_dir / "run.json", "w") as f:
        json.dump(run, f, indent=1)
    print(f"OPCART_DONE arm={ARM} out={out_dir}")


if __name__ == "__main__":
    main()

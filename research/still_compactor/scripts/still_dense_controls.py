#!/usr/bin/env python3
"""STILL dense-supervision controls - isolate WHERE arbitrary-fact preservation
fails, before spending on a big run.

The multifact generalization run failed at chance, but it used ONE question per
context (sparse supervision). The decisive tests build the compact cache ONCE
per context and supervise/evaluate against MANY questions of that same context,
and separate the possible causes with control modes:

  identity      : inject the FULL source KV (t=T). Must score ~100% -> validates
                  the cache/position path at the task level (not just cosine).
  oracle        : keep only the ORIGINAL K/V rows of the key+value tokens, up to
                  the budget t. If it approaches the teacher, the slot budget is
                  enough and STILL's synthesis/training is the problem.
  still_overfit : train the STILL compactor on a few FIXED contexts, all
                  questions each, to convergence. If it can't even memorize, the
                  arbitrary-fact thesis is dead for this architecture.

Task: N key->passcode facts; each of the N keys is one question (4-way MCQ over
values from the same context). Cache is query-blind (built before the question).
"""
import os, argparse, json, sys, random, string, statistics
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from still_compactor import (STILLCompactorLayer, apply_rope, cache_to_legacy,
                             legacy_to_cache, rope_theta)  # noqa: E402


def gen_context(rng, n_facts):
    keys, vals, seen = [], [], set()
    while len(keys) < n_facts:
        k = "".join(rng.choice(string.ascii_uppercase) for _ in range(4))
        if k in seen:
            continue
        seen.add(k); keys.append(k)
        vals.append("".join(rng.choice(string.digits) for _ in range(4)))
    order = list(range(n_facts)); rng.shuffle(order)
    ctx = " ".join(f"Record {p + 1}: the passcode for {keys[i]} is {vals[i]}."
                   for p, i in enumerate(order))
    return ctx, keys, vals


def all_questions(rng, tok, keys, vals):
    """One question per key: 4-way MCQ over values from this context."""
    n = len(keys); qs = []
    letters = [tok(l, add_special_tokens=False).input_ids[0] for l in "ABCD"]
    for j in range(n):
        opts = rng.sample([vals[k] for k in range(n) if k != j], 3) + [vals[j]]
        rng.shuffle(opts); correct = opts.index(vals[j])
        q = (f"\nQuestion: What is the passcode for {keys[j]}?\n(A) {opts[0]} "
             f"(B) {opts[1]} (C) {opts[2]} (D) {opts[3]}\nAnswer: (")
        cont = (f"\nThe passcode for {keys[j]} is {vals[j]}, so the answer is "
                f"({'ABCD'[correct]}).")
        qs.append(dict(q=q, cont=cont, correct=correct, letters=letters))
    return qs


@torch.no_grad()
def build_item(model, tok, rng, n_facts, dev, topk=200):
    ctx, keys, vals = gen_context(rng, n_facts)
    ci = tok(ctx, return_tensors="pt").input_ids.to(dev)
    qs = all_questions(rng, tok, keys, vals)
    out = []
    for q in qs:
        qi = tok(q["q"], add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
        coi = tok(q["cont"], add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
        lg = model(torch.cat([ci, qi, coi], 1)).logits[0]
        cl = ci.shape[1] + qi.shape[1]
        pred = lg[cl - 1: cl - 1 + coi.shape[1]].float()
        gold = coi[0]
        idx = torch.cat([pred.topk(topk, -1).indices, gold[:, None]], 1)
        out.append(dict(q_ids=qi.cpu(), cont_ids=coi.cpu(),
                        support=idx.cpu(), tprob=torch.softmax(pred.gather(1, idx), -1).cpu(),
                        correct=q["correct"], letters=q["letters"]))
    return dict(ctx_ids=ci.cpu(), ctxlen=ci.shape[1], qs=out)


def still_cache(comp, model, ctx_ids, theta, dev, grad):
    with torch.no_grad():
        leg = cache_to_legacy(model(ctx_ids.to(dev), use_cache=True).past_key_values)
    sp = torch.arange(ctx_ids.shape[1], device=dev)
    ctxm = torch.enable_grad if grad else torch.no_grad
    out = []
    with ctxm():
        for (k, v) in leg:
            ck, cv, _ = comp(apply_rope(k[0].to(dev), sp, theta, inverse=True),
                             v[0].to(dev), sp)
            out.append((ck.unsqueeze(0), cv.unsqueeze(0)))
    return out


@torch.no_grad()
def full_cache(model, ctx_ids, dev):
    return cache_to_legacy(model(ctx_ids.to(dev), use_cache=True).past_key_values)


def student_letter_logits(model, legacy, ctxlen, q_ids, dev):
    cache = legacy_to_cache(legacy); phys = cache.get_seq_length()
    qn = q_ids.shape[1]
    return model(q_ids.to(dev), past_key_values=cache, use_cache=True,
                 position_ids=torch.arange(ctxlen, ctxlen + qn, device=dev).unsqueeze(0),
                 cache_position=torch.arange(phys, phys + qn, device=dev)).logits[0, -1]


def student_cont_logits(model, legacy, ctxlen, q_ids, cont_ids, dev):
    cache = legacy_to_cache(legacy); phys = cache.get_seq_length()
    seq = torch.cat([q_ids.to(dev), cont_ids.to(dev)], 1); n = seq.shape[1]
    out = model(seq, past_key_values=cache, use_cache=True,
                position_ids=torch.arange(ctxlen, ctxlen + n, device=dev).unsqueeze(0),
                cache_position=torch.arange(phys, phys + n, device=dev))
    s = q_ids.shape[1]
    return out.logits[0, s - 1: s - 1 + cont_ids.shape[1]]


@torch.no_grad()
def eval_dense(model, legacy_builder, items, dev):
    """Accuracy over ALL questions, each against the query-blind cache."""
    correct = tot = 0
    for it in items:
        legacy = legacy_builder(it)
        for q in it["qs"]:
            lg = student_letter_logits(model, legacy, it["ctxlen"], q["q_ids"], dev).float()
            correct += int(int(F.log_softmax(lg[q["letters"]], -1).argmax()) == q["correct"])
            tot += 1
    return correct / tot, correct, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--control", choices=["identity", "still_overfit"], required=True)
    ap.add_argument("--n-facts", type=int, default=64)
    ap.add_argument("--t-compact", type=int, default=256)
    ap.add_argument("--n-ctx", type=int, default=16)     # fixed contexts (overfit)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--q-batch", type=int, default=8)    # questions/context/step
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    di = int(args.device.split(":")[1]); torch.cuda.set_device(di); dev = args.device
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()
    model.requires_grad_(False)
    cfg = model.config
    H, d, theta = cfg.num_key_value_heads, cfg.head_dim, rope_theta(cfg)
    rng = random.Random(args.seed)
    print(f"control={args.control} n_facts={args.n_facts} t={args.t_compact} "
          f"n_ctx={args.n_ctx}")
    items = [build_item(model, tok, rng, args.n_facts, dev) for _ in range(args.n_ctx)]
    print(f"built {len(items)} contexts x {args.n_facts} questions; "
          f"ctxlen~{items[0]['ctxlen']}")

    if args.control == "identity":
        acc, c, t = eval_dense(model, lambda it: full_cache(model, it["ctx_ids"], dev),
                               items, dev)
        print(f"[identity t=T] dense acc = {acc:.3f} ({c}/{t}) "
              f"-> {'PATH OK (approx 1.0 expected)' if acc > 0.9 else 'PATH BROKEN'}")
        return

    # still_overfit: train the compactor on the fixed contexts, all questions
    comp = STILLCompactorLayer(H, d, t=args.t_compact, base_theta=theta).to(dev, torch.bfloat16)
    decay = [p for p in comp.parameters() if p.dim() > 1]
    nod = [p for p in comp.parameters() if p.dim() <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                             {"params": nod, "weight_decay": 0.0}], lr=4e-5, betas=(0.9, 0.95))
    base_lr, warm = 4e-5, 100
    ev = lambda it: still_cache(comp, model, it["ctx_ids"], theta, dev, False)
    a0, c0, t0 = eval_dense(model, ev, items, dev)
    print(f"[still_overfit] untrained dense acc = {a0:.3f} ({c0}/{t0})")
    for step in range(args.steps):
        it = items[step % len(items)]
        cl = still_cache(comp, model, it["ctx_ids"], theta, dev, True)
        qs = random.sample(it["qs"], min(args.q_batch, len(it["qs"])))
        loss = 0.0
        for q in qs:
            slg = student_cont_logits(model, cl, it["ctxlen"], q["q_ids"], q["cont_ids"], dev).float()
            sup = q["support"].to(dev); tp = q["tprob"].to(dev)
            loss = loss + F.kl_div(F.log_softmax(slg.gather(1, sup), -1), tp, reduction="batchmean")
        loss = loss / len(qs)
        lr = base_lr * min(1.0, (step + 1) / warm)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(comp.parameters(), 1.0); opt.step()
        if step % 100 == 0 or step == args.steps - 1:
            a, c, t = eval_dense(model, ev, items, dev)
            print(f"  step {step:>4} lr{lr:.1e} loss={loss.item():.3f} "
                  f"train-dense acc={a:.3f} ({c}/{t})")
    a, c, t = eval_dense(model, ev, items, dev)
    print(f"[still_overfit FINAL] dense acc = {a:.3f} ({c}/{t}) over "
          f"{args.n_ctx}x{args.n_facts} pairs -> "
          f"{'CAN memorize (>=0.95)' if a >= 0.95 else 'CANNOT memorize the dictionary'}")
    print("READ: if STILL cannot even overfit these fixed contexts on dense "
          "supervision, arbitrary-fact preservation is dead for this architecture "
          "at t={}. Next controls (oracle/cartridge) localize further.".format(args.t_compact))


if __name__ == "__main__":
    main()

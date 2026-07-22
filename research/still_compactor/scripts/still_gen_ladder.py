#!/usr/bin/env python3
"""STILL generalization ladder - the cheap decisive test (paper-matched recipe).

Fixes every training confound the review named and asks the binary question:
can the fidelity-corrected compactor learn an AMORTIZED compaction rule (compact
UNSEEN contexts and preserve their answers), or only a lookup table?

Recipe (paper-matched): AdamW lr 4e-5, betas (0.9,0.95), 100-step warmup from
1e-6, weight decay 0.01 on matrices only, grad clip 1.0, effective batch 16.
Loss = forward-KL at EVERY continuation (rationale+answer) token over the
teacher's top-200 vocab with the gold token forced in. Thousands of FRESH
procedural contexts (never a 24-context recycle), balanced A/B/C/D, randomized
animal / needle depth / option order / filler. Track held-out accuracy, gold
margin, and KL - not just discrete accuracy. Base model frozen.
"""
import os
import argparse, json, sys, random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from still_compactor import STILLCompactorLayer, apply_rope  # noqa: E402

ANIMALS = ("owl fox cat dog bat elk ram jay cod hen ape eel yak boa asp gnu koi "
           "pug sow wren crow moth wolf lynx hare seal toad newt crab moose "
           "otter finch heron stoat vole quail robin skunk trout weasel").split()
SENTS = ["The quarterly review covered logistics and staffing at length.",
         "A notice about network maintenance windows circulated widely.",
         "The committee weighed budget allocations for the coming term.",
         "Weather advisories mentioned scattered showers through the week.",
         "An archival note described the cataloguing of older records.",
         "The bulletin listed several updates to the visitor policy.",
         "Attendance figures were tabulated and filed without comment.",
         "A brief memo reminded staff about the revised parking layout."]


def make_item(rng, tok, target_tokens=512):
    opts = rng.sample(ANIMALS, 4)
    correct = rng.randrange(4)
    secret = opts[correct]
    body = []
    n = 0
    needle_at = rng.randint(3, 20)
    i = 0
    while n < target_tokens - 40:
        if i == needle_at:
            s = f"IMPORTANT RECORD: the secret animal is {secret}."
        else:
            s = rng.choice(SENTS)
        body.append(s); n += len(tok(s, add_special_tokens=False).input_ids) + 1
        i += 1
    if needle_at >= i:
        body.insert(len(body) // 2,
                    f"IMPORTANT RECORD: the secret animal is {secret}.")
    ctx = " ".join(body)
    q = (f"\nQuestion: What is the secret animal?\n(A) {opts[0]} (B) {opts[1]} "
         f"(C) {opts[2]} (D) {opts[3]}\nAnswer: (")
    letter = "ABCD"[correct]
    cont = (f"\nBased on the record the secret animal is {secret}, so the "
            f"answer is ({letter}).")
    return dict(ctx=ctx, q=q, cont=cont, correct=correct,
                letters=[tok(l, add_special_tokens=False).input_ids[0] for l in "ABCD"])


@torch.no_grad()
def teacher_targets(model, tok, it, dev, topk=200):
    ci = tok(it["ctx"], return_tensors="pt").input_ids.to(dev)
    qi = tok(it["q"], add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    coi = tok(it["cont"], add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    seq = torch.cat([ci, qi, coi], 1)
    lg = model(seq).logits[0]
    cl = ci.shape[1] + qi.shape[1]                         # continuation start
    pred = lg[cl - 1: cl - 1 + coi.shape[1]].float()       # [contlen, vocab]
    gold = coi[0]
    vals, idx = pred.topk(topk, dim=-1)
    idx = torch.cat([idx, gold[:, None]], 1)               # force gold in support
    tprob = torch.softmax(pred.gather(1, idx), -1)
    return dict(ctx_ids=ci.cpu(), q_ids=qi.cpu(), cont_ids=coi.cpu(),
                ctxlen=ci.shape[1], support=idx.cpu(), tprob=tprob.cpu(),
                correct=it["correct"], letters=it["letters"])


def compact_prefix(comp, model, ctx_ids, theta, dev, grad):
    with torch.no_grad():
        leg = model(ctx_ids.to(dev), use_cache=True).past_key_values.to_legacy_cache()
    sp = torch.arange(ctx_ids.shape[1], device=dev)
    ctxm = torch.enable_grad if grad else torch.no_grad
    out = []
    with ctxm():
        for (k, v) in leg:
            ck, cv, _ = comp(apply_rope(k[0].to(dev), sp, theta, inverse=True),
                             v[0].to(dev), sp)
            out.append((ck.unsqueeze(0), cv.unsqueeze(0)))
    return out


def student_cont_logits(model, cl, ctxlen, q_ids, cont_ids, dev):
    cache = DynamicCache.from_legacy_cache(tuple(cl))
    phys = cache.get_seq_length()
    seq = torch.cat([q_ids.to(dev), cont_ids.to(dev)], 1)
    n = seq.shape[1]
    out = model(seq, past_key_values=cache, use_cache=True,
                position_ids=torch.arange(ctxlen, ctxlen + n, device=dev).unsqueeze(0),
                cache_position=torch.arange(phys, phys + n, device=dev))
    start = q_ids.shape[1]
    return out.logits[0, start - 1: start - 1 + cont_ids.shape[1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-eval", type=int, default=512)
    ap.add_argument("--ctx-tokens", type=int, default=512)
    ap.add_argument("--t-compact", type=int, default=128)     # 512/128 = 4x
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ladder-results")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    di = int(args.device.split(":")[1]); torch.cuda.set_device(di); dev = args.device
    torch.manual_seed(args.seed)
    import os; os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()
    model.requires_grad_(False)
    cfg = model.config
    H, d, theta = cfg.num_key_value_heads, cfg.head_dim, float(cfg.rope_theta)
    comp = STILLCompactorLayer(H, d, t=args.t_compact, base_theta=theta).to(dev, torch.bfloat16)

    rng = random.Random(args.seed)
    print(f"building {args.n_train} train + {args.n_eval} held-out fresh "
          f"contexts (~{args.ctx_tokens} tok, {args.ctx_tokens//args.t_compact}x)...")
    train = [teacher_targets(model, tok, make_item(rng, tok, args.ctx_tokens), dev)
             for _ in range(args.n_train)]
    heldrng = random.Random(args.seed + 99991)
    held = [teacher_targets(model, tok, make_item(heldrng, tok, args.ctx_tokens), dev)
            for _ in range(args.n_eval)]
    balance = [0, 0, 0, 0]
    for h in held:
        balance[h["correct"]] += 1
    print(f"  held-out letter balance A/B/C/D = {balance}")

    def evaluate(tag, get_letters):
        acc, nll, marg = 0, [], []
        for h in held:
            lg = get_letters(h).float()
            lp = F.log_softmax(lg[h["letters"]], -1)
            acc += int(int(lp.argmax()) == h["correct"])
            nll.append(-lp[h["correct"]].item())
            srt = lp.sort(descending=True).values
            marg.append((lp[h["correct"]] - srt[1] if int(lp.argmax()) == h["correct"]
                         else lp[h["correct"]] - srt[0]).item())
        a = acc / len(held)
        print(f"  {tag:24} acc={a:.3f} ({acc}/{len(held)}) "
              f"NLL={sum(nll)/len(nll):.3f} gold_margin={sum(marg)/len(marg):+.3f}")
        return a

    @torch.no_grad()
    def letters_nocontext(h):
        return model(h["q_ids"].to(dev)).logits[0, -1]

    @torch.no_grad()
    def letters_still(h):
        cl = compact_prefix(comp, model, h["ctx_ids"], theta, dev, False)
        cache = DynamicCache.from_legacy_cache(tuple(cl))
        phys = cache.get_seq_length()
        qn = h["q_ids"].shape[1]
        return model(h["q_ids"].to(dev), past_key_values=cache, use_cache=True,
                     position_ids=torch.arange(h["ctxlen"], h["ctxlen"] + qn, device=dev).unsqueeze(0),
                     cache_position=torch.arange(phys, phys + qn, device=dev)).logits[0, -1]

    print("\n[held-out baselines]")
    evaluate("no-context", letters_nocontext)
    evaluate("STILL untrained", letters_still)

    # paper optimizer: AdamW, wd 0.01 on matrices (dim>1) only
    decay = [p for p in comp.parameters() if p.dim() > 1]
    nodecay = [p for p in comp.parameters() if p.dim() <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=4e-5, betas=(0.9, 0.95))
    warmup, base_lr, step = 100, 4e-5, 0
    log = {"config": vars(args), "held_balance": balance, "eval": []}
    for ep in range(args.epochs):
        rng.shuffle(train)
        opt.zero_grad()
        run = 0.0
        for i, t in enumerate(train):
            cl = compact_prefix(comp, model, t["ctx_ids"], theta, dev, True)
            slg = student_cont_logits(model, cl, t["ctxlen"], t["q_ids"],
                                      t["cont_ids"], dev).float()
            sup = t["support"].to(dev); tp = t["tprob"].to(dev)
            slp = F.log_softmax(slg.gather(1, sup), -1)
            loss = F.kl_div(slp, tp, reduction="batchmean") / args.batch
            loss.backward(); run += loss.item()
            if (i + 1) % args.batch == 0:
                step += 1
                lr = base_lr * min(1.0, step / warmup) if step <= warmup else base_lr
                lr = max(lr, 1e-6)
                for g in opt.param_groups:
                    g["lr"] = lr
                torch.nn.utils.clip_grad_norm_(comp.parameters(), 1.0)
                opt.step(); opt.zero_grad()
                if step % 20 == 0:
                    print(f"  ep{ep} step{step} lr{lr:.1e} KL/batch={run:.3f}"); run = 0.0
        acc = evaluate(f"STILL trained (ep{ep+1})", letters_still)
        log["eval"].append({"epoch": ep + 1, "held_acc": acc})
        torch.save(comp.state_dict(), f"{args.out}/compactor_seed{args.seed}.pt")
        with open(f"{args.out}/ladder_seed{args.seed}.json", "w") as f:
            json.dump(log, f, indent=2)

    print("\n[full-context ceiling]")
    @torch.no_grad()
    def letters_full(h):
        seq = torch.cat([h["ctx_ids"].to(dev), h["q_ids"].to(dev)], 1)
        return model(seq).logits[0, -1]
    evaluate("full-context(teacher)", letters_full)
    print("\nVERDICT: held-out STILL-trained ABOVE no-context (~0.25 chance) and "
          "toward full-context, with rising gold_margin and >2 seeds, = STILL "
          "GENERALIZES (amortized compaction). Flat-at-chance across seeds after "
          "this fidelity-corrected, paper-recipe run = a real limit, not funding.")


if __name__ == "__main__":
    main()

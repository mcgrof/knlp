#!/usr/bin/env python3
"""STILL real-document salience validation (repositioned-role check).

The dense controls showed STILL does not generalize arbitrary-binding
preservation, but it does generalize SALIENCE extraction. This validates that
narrower, defensible role on REAL text rather than synthetic marked fields:
compact ~8K-token chunks of public-domain books and test whether salient facts
(proper nouns in context) survive under 32x/50x compression.

Task: cloze MCQ over a salient proper noun in a real sentence, 4 options drawn
from proper nouns of the same chunk (so all are in-context-plausible). Cache is
query-blind. Teacher-filtered (keep questions full-context answers correctly and
no-context misses), so we measure preservation, not base ability.

This is a salience-compression validation on natural text, NOT a faithful
reproduction of the paper's 4-domain corpus and question generator.
"""
import os, argparse, json, sys, random, re, urllib.request
from collections import Counter
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from still_compactor import (STILLCompactorLayer, apply_rope, rope_theta)  # noqa
from still_dense_controls import (still_cache, student_cont_logits,       # noqa
                                  student_letter_logits, eval_dense)

BOOKS = {  # public-domain Project Gutenberg plain-text
    "pride": "https://www.gutenberg.org/files/1342/1342-0.txt",
    "sherlock": "https://www.gutenberg.org/files/1661/1661-0.txt",
    "moby": "https://www.gutenberg.org/files/2701/2701-0.txt",
}
COMMON = set("The This That These Those There Here When Where While What Which "
             "And But For Nor Yet So Chapter Project Gutenberg Mr Mrs Dr His Her "
             "Their They Him She You Your Our We It He I A An Of In On At To".split())


def get_text(key, url, cache_dir):
    p = os.path.join(cache_dir, f"{key}.txt")
    if not os.path.exists(p):
        txt = urllib.request.urlopen(url, timeout=30).read().decode("utf-8", "ignore")
        open(p, "w").write(txt)
    txt = open(p).read()
    # strip Gutenberg header/footer
    s = txt.find("*** START"); e = txt.find("*** END")
    if s != -1:
        txt = txt[txt.find("\n", s) + 1:]
    if e != -1:
        txt = txt[:txt.rfind("*** END")]
    return re.sub(r"\s+", " ", txt).strip()


def chunk_contexts(text, tok, ctx_tokens, n_chunks, rng):
    ids = tok(text, add_special_tokens=False).input_ids
    out, i = [], 0
    starts = list(range(0, len(ids) - ctx_tokens, ctx_tokens))
    rng.shuffle(starts)
    for s in starts[:n_chunks]:
        out.append(tok.decode(ids[s:s + ctx_tokens]))
    return out


def cloze_questions(chunk, tok, rng, max_q):
    sents = re.split(r"(?<=[.!?]) ", chunk)
    # proper-noun vocabulary of the chunk (appear >=2x, mid-sentence, not common)
    nouns = [w for w in re.findall(r"\b[A-Z][a-z]{3,}\b", chunk) if w not in COMMON]
    vocab = [w for w, n in Counter(nouns).items() if n >= 2]
    if len(vocab) < 6:
        return []
    qs = []
    rng.shuffle(sents)
    for sent in sents:
        if len(qs) >= max_q:
            break
        cands = [w for w in re.findall(r"\b[A-Z][a-z]{3,}\b", sent)
                 if w in vocab]
        if not cands or len(sent) < 40 or len(sent) > 300:
            continue
        ans = rng.choice(cands)
        blanked = sent.replace(ans, "_____", 1)
        distract = rng.sample([w for w in vocab if w != ans], 3)
        opts = distract + [ans]; rng.shuffle(opts); correct = opts.index(ans)
        q = (f'\nQuestion: Fill in the blank in this sentence from the text: '
             f'"{blanked}"\n(A) {opts[0]} (B) {opts[1]} (C) {opts[2]} '
             f'(D) {opts[3]}\nAnswer: (')
        cont = (f"\nThe blank is {ans}, so the answer is ({'ABCD'[correct]}).")
        qs.append(dict(q=q, cont=cont, correct=correct,
                       letters=[tok(l, add_special_tokens=False).input_ids[0] for l in "ABCD"]))
    return qs


@torch.no_grad()
def build_item(model, tok, ctx, qs, dev, topk=200, filter_q=True):
    ci = tok(ctx, return_tensors="pt").input_ids.to(dev)
    kept = []
    for q in qs:
        qi = tok(q["q"], add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
        if filter_q:  # keep only teacher-correct, no-context-wrong
            full = model(torch.cat([ci, qi], 1)).logits[0, -1]
            noctx = model(qi).logits[0, -1]
            tc = int(F.log_softmax(full[q["letters"]], -1).argmax()) == q["correct"]
            nw = int(F.log_softmax(noctx[q["letters"]], -1).argmax()) != q["correct"]
            if not (tc and nw):
                continue
        coi = tok(q["cont"], add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
        lg = model(torch.cat([ci, qi, coi], 1)).logits[0]
        cl = ci.shape[1] + qi.shape[1]
        pred = lg[cl - 1: cl - 1 + coi.shape[1]].float()
        idx = torch.cat([pred.topk(topk, -1).indices, coi[0][:, None]], 1)
        kept.append(dict(q_ids=qi.cpu(), cont_ids=coi.cpu(), support=idx.cpu(),
                         tprob=torch.softmax(pred.gather(1, idx), -1).cpu(),
                         correct=q["correct"], letters=q["letters"]))
    return dict(ctx_ids=ci.cpu(), ctxlen=ci.shape[1], qs=kept) if kept else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--ctx-tokens", type=int, default=8192)
    ap.add_argument("--t-compact", type=int, default=256)     # 8192/256 = 32x
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--n-eval", type=int, default=32)
    ap.add_argument("--max-q", type=int, default=10)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--q-batch", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="/home/mcgrof/still_work/books")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    di = int(args.device.split(":")[1]); torch.cuda.set_device(di); dev = args.device
    torch.manual_seed(args.seed); rng = random.Random(args.seed)
    os.makedirs(args.cache_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()
    model.requires_grad_(False)
    cfg = model.config
    H, d, theta = cfg.num_key_value_heads, cfg.head_dim, rope_theta(cfg)
    print(f"real-doc salience: ctx={args.ctx_tokens} t={args.t_compact} "
          f"(~{args.ctx_tokens // args.t_compact}x)")

    texts = {k: get_text(k, u, args.cache_dir) for k, u in BOOKS.items()}
    chunks = []
    for k, t in texts.items():
        chunks += chunk_contexts(t, tok, args.ctx_tokens,
                                 (args.n_train + args.n_eval), rng)
    rng.shuffle(chunks)
    items = []
    for ch in chunks:
        if len(items) >= args.n_train + args.n_eval:
            break
        qs = cloze_questions(ch, tok, rng, args.max_q)
        if len(qs) < 3:
            continue
        it = build_item(model, tok, ch, qs, dev)
        if it and len(it["qs"]) >= 3:
            items.append(it)
    print(f"built {len(items)} usable contexts (teacher-filtered), "
          f"mean {sum(len(x['qs']) for x in items)/max(len(items),1):.1f} Q/ctx")
    train, held = items[:args.n_train], items[args.n_train:]
    if len(held) < 4:
        print("not enough held-out; reduce n-train or max-q"); return

    comp = STILLCompactorLayer(H, d, t=args.t_compact, base_theta=theta).to(dev, torch.bfloat16)
    dec = [p for p in comp.parameters() if p.dim() > 1]
    nod = [p for p in comp.parameters() if p.dim() <= 1]
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": 0.01},
                             {"params": nod, "weight_decay": 0.0}], lr=4e-5, betas=(0.9, 0.95))
    ev = lambda it: still_cache(comp, model, it["ctx_ids"], theta, dev, False)
    a0, c0, t0 = eval_dense(model, ev, held, dev)
    print(f"[real-doc] untrained STILL held-out acc = {a0:.3f} ({c0}/{t0})")
    for step in range(args.steps):
        it = train[step % len(train)]
        cl = still_cache(comp, model, it["ctx_ids"], theta, dev, True)
        qs = random.sample(it["qs"], min(args.q_batch, len(it["qs"])))
        loss = 0.0
        for q in qs:
            slg = student_cont_logits(model, cl, it["ctxlen"], q["q_ids"], q["cont_ids"], dev).float()
            loss = loss + F.kl_div(F.log_softmax(slg.gather(1, q["support"].to(dev)), -1),
                                   q["tprob"].to(dev), reduction="batchmean")
        loss = loss / len(qs)
        lr = 4e-5 * min(1.0, (step + 1) / 100)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(comp.parameters(), 1.0); opt.step()
        if step % 200 == 0 or step == args.steps - 1:
            a, c, t = eval_dense(model, ev, held, dev)
            print(f"  step {step:>4} loss={loss.item():.3f} held-out acc={a:.3f} ({c}/{t})")
    a, c, t = eval_dense(model, ev, held, dev)
    print(f"[real-doc FINAL] held-out salience acc = {a:.3f} ({c}/{t}) at "
          f"{args.ctx_tokens // args.t_compact}x. Full-context teacher ~1.0 by "
          f"construction (teacher-filtered); untrained STILL was {a0:.3f}. "
          f"Above untrained/no-context = STILL preserves salient real-doc facts.")


if __name__ == "__main__":
    main()

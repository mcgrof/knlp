#!/usr/bin/env python3
"""Small-scale cartridge harness: a trainable KV prefix for any Hugging
Face causal language model, trained by context distillation on a
synthetic record task with exact scoring, cheap enough for a CPU.

The purpose is a scaling ladder for cartridge research. A cartridge is
a small trained KV cache that stands in for a document in front of a
frozen reader (Eyuboglu et al., arXiv 2506.06266). Training one for an
8B reader costs hours on a rented GPU, so algorithmic questions (what
to initialize from, what objective to use, whether the reader attends
the slots, how far keys and values move) are answered here first on
models from 14M to 1B parameters, and only the winner goes to the big
reader.

The task is synthetic so that a small reader can answer it from
context and the scoring is exact: a record of entities with attribute
values, questions about those values, and forced-choice scoring over
the other entities' values for the same attribute. The teacher is the
reader with the record in its context; the student is the reader with
the prefix in place of the record; the objective is the top-k context
distillation loss the Cartridges library uses. Informative positions
are those where the teacher's top token differs from the bare reader's
(no record, no prefix), the same split the 8B work uses.

Instruments, all exact: held-out distillation loss (all, informative,
rest), forced-choice accuracy and margin against the full-context
ceiling and no-context floor, attention mass on the prefix slots per
layer at informative and rest positions, key and value drift from the
initialization, and off-context damage: the loss on an unrelated
record's text and the forced-choice accuracy on an unrelated record
given in context while the prefix is loaded (the "off-context
derailment" of Cartridges++, arXiv 2609.35621).

Subcommands:
  run    train one arm and write curves + final metrics as JSON
  bench  time the training step for a fixed workload (host comparison)

Everything runs in float32 on CPU by default; DEVICE=cuda works too.
"""

import argparse
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ----------------------------------------------------------------------
# synthetic record task
# ----------------------------------------------------------------------

FIRST = (
    "Alice Bruno Carla Dmitri Elena Farid Greta Hugo Iris Jonas Keiko "
    "Lars Mira Nadia Omar Priya Quentin Rosa Sven Talia Umar Vera Wen "
    "Xavier Yara Zoe Anton Bea Cyrus Dana Emil Fay Gino Hana Ivo Jade "
    "Kai Lena Milo Nora Otto Pia Remy Sara Tomas Una Vik Willa"
).split()
LAST = (
    "Morgan Okafor Lindqvist Petrov Haddad Nakamura Costa Fischer Duarte "
    "Novak Reyes Ibrahim Kowalski Baptiste Oyelaran Sato Marin Bergman "
    "Castillo Varga Hughes Mensah Rinaldi Sorensen Tanaka Weber Yilmaz "
    "Zhang Almeida Brennan"
).split()
CITIES = (
    "Lisbon Oslo Nairobi Kyoto Lima Porto Zagreb Tallinn Quito Dakar "
    "Perth Malmo Tbilisi Hanoi Bogota Riga Cairo Leeds Turin Jaipur "
    "Bergen Seoul Accra Vilnius Cusco Lyon Ghent Doha Sapporo Valencia "
    "Tunis Odense Dublin Lagos Manila Prague Sofia Basel Graz Hobart"
).split()
COLORS = "red blue green amber violet teal coral ivory olive navy maroon silver".split()
PETS = "cat dog parrot rabbit turtle ferret hamster goldfish canary gecko".split()
JOBS = (
    "baker pilot nurse welder florist tailor chemist surveyor plumber "
    "sculptor archivist beekeeper locksmith cartographer glazier farrier "
    "notary optician brewer cooper"
).split()
BLOOD = "O+ O- A+ A- B+ B- AB+ AB-".split()
MONTHS = (
    "January February March April May June July August September October "
    "November December"
).split()
CARS = (
    "sedan hatchback coupe wagon pickup minivan roadster convertible "
    "crossover limousine microcar camper tractor trike buggy"
).split()


def pick(rng, pool, n):
    """n values from pool, distinct when the pool allows it."""
    pool = list(pool)
    if n <= len(pool):
        return rng.sample(pool, n)
    return [rng.choice(pool) for _ in range(n)]


ATTRS = {
    "age": lambda rng, n: pick(rng, range(18, 91), n),
    "city": lambda rng, n: pick(rng, CITIES, n),
    "favorite color": lambda rng, n: [rng.choice(COLORS) for _ in range(n)],
    "pet": lambda rng, n: [rng.choice(PETS) for _ in range(n)],
    "job": lambda rng, n: pick(rng, JOBS, n),
    "blood type": lambda rng, n: [rng.choice(BLOOD) for _ in range(n)],
    "birth month": lambda rng, n: [rng.choice(MONTHS) for _ in range(n)],
    "car": lambda rng, n: pick(rng, CARS, n),
    "extension": lambda rng, n: pick(rng, range(100, 1000), n),
}

Q_TEMPLATES = [
    "Q: What is the {attr} of {name}?\nA:",
    "Q: {name}'s {attr} is what?\nA:",
    "Q: According to the record, what {attr} does {name} have?\nA:",
    "Q: Tell me the {attr} of {name}.\nA:",
]


ACTIVE = {"attrs": list(ATTRS)}  # set from --attrs: the first N attributes


def attrs():
    return ACTIVE["attrs"]


def make_record(rng, n_entities):
    names = rng.sample([f"{a} {b}" for a in FIRST for b in LAST], n_entities)
    cols = {a: ATTRS[a](rng, n_entities) for a in attrs()}
    ents = [
        {"name": names[i], **{a: cols[a][i] for a in attrs()}}
        for i in range(n_entities)
    ]
    return ents


FMT = {"name": "qa"}  # set from --fmt; "qa" prose record, "kv" one fact per line

KV_TEMPLATES = ["{name} {attr}:", "The {attr} of {name} is"]


def render_record(ents, title="Record"):
    if FMT["name"] == "kv":
        lines = [f"{title}."]
        for e in ents:
            lines += [f"{e['name']} {a}: {e[a]}" for a in attrs()]
        return "\n".join(lines) + "\n"
    lines = [f"{title}."]
    for i, e in enumerate(ents, 1):
        parts = [f"Name: {e['name']}"] + [f"{a}: {e[a]}" for a in attrs()]
        lines.append(f"Entry {i}. " + ". ".join(parts) + ".")
    return "\n".join(lines) + "\n"


def templates():
    return KV_TEMPLATES if FMT["name"] == "kv" else Q_TEMPLATES


def qa_text(e, attr, template):
    return template.format(attr=attr, name=e["name"]), f" {e[attr]}"


def task_split(ents, rng, holdout=0.0):
    """Facts used for training conversations and facts scored at
    evaluation. By default every fact is in both: as in the 8B protocol,
    the held-out set is new conversations about the same document, not
    unseen facts (a cartridge is a compression of the document, and
    self-study covers all of it). holdout > 0 withholds that fraction of
    the facts from training, which tests the initialization alone."""
    pairs = [(i, a) for i in range(len(ents)) for a in attrs()]
    rng.shuffle(pairs)
    if holdout <= 0:
        return pairs, list(pairs)
    n_hold = int(len(pairs) * holdout)
    return pairs[n_hold:], pairs[:n_hold]


def train_templates():
    """Training conversations use every template but the first; the
    first is reserved for evaluation, so a cartridge is scored on a
    phrasing it never saw."""
    t = templates()
    return t[1:] if len(t) > 1 else t


def make_conversation(ents, pairs, rng, n_turns, eval_style=False):
    """A self-study conversation: n_turns question/answer pairs about the
    record, varied templates, answers given in full. eval_style uses the
    reserved evaluation template."""
    chosen = rng.sample(pairs, min(n_turns, len(pairs)))
    text = ""
    for i, a in chosen:
        tpl = templates()[0] if eval_style else rng.choice(train_templates())
        q, ans = qa_text(ents[i], a, tpl)
        text += q + ans + "\n"
    return text


# ----------------------------------------------------------------------
# model and prefix cache
# ----------------------------------------------------------------------


def load_model(name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.float32, attn_implementation="eager"
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def cache_layers(cache):
    """Per-layer (key, value) tensors from a transformers cache object,
    across the API change between the 4.x and 5.x releases."""
    if hasattr(cache, "layers"):
        return [(l.keys, l.values) for l in cache.layers]
    return list(zip(cache.key_cache, cache.value_cache))


def make_cache(keys, values, model):
    from transformers import DynamicCache

    try:
        c = DynamicCache(config=model.config)
    except TypeError:
        c = DynamicCache()
    for i, (k, v) in enumerate(zip(keys, values)):
        c.update(k, v, i)
    return c


@torch.no_grad()
def kv_of_text(model, tok, text, n_tokens, device):
    """Key/value tensors of the first n_tokens tokens of text: the
    first-tokens initialization."""
    ids = tok(text, return_tensors="pt").input_ids[:, :n_tokens].to(device)
    out = model(input_ids=ids, use_cache=True)
    return [(k.clone(), v.clone()) for k, v in cache_layers(out.past_key_values)]


@torch.no_grad()
def kv_of_chunks(model, tok, text, n_tokens, rng, device, chunk=8):
    """Spread initialization: KV of random chunks of the document,
    concatenated (Cartridges' chunk sampling)."""
    ids = tok(text, return_tensors="pt").input_ids[0]
    n_chunks = n_tokens // chunk
    starts = sorted(rng.sample(range(0, max(1, len(ids) - chunk)), n_chunks))
    pieces = [ids[s : s + chunk] for s in starts]
    cat = torch.cat(pieces)[None].to(device)
    out = model(input_ids=cat, use_cache=True)
    return [(k.clone(), v.clone()) for k, v in cache_layers(out.past_key_values)]


class Prefix(torch.nn.Module):
    """Trainable KV prefix. Slot 0 is frozen when freeze_first is set
    (the Cartridges attention sink)."""

    def __init__(self, kv, freeze_first=True):
        super().__init__()
        self.freeze_first = freeze_first
        self.k0 = torch.nn.ParameterList()
        self.v0 = torch.nn.ParameterList()
        self.k = torch.nn.ParameterList()
        self.v = torch.nn.ParameterList()
        for k, v in kv:
            if freeze_first:
                self.k0.append(
                    torch.nn.Parameter(k[:, :, :1].clone(), requires_grad=False)
                )
                self.v0.append(
                    torch.nn.Parameter(v[:, :, :1].clone(), requires_grad=False)
                )
                k, v = k[:, :, 1:], v[:, :, 1:]
            self.k.append(torch.nn.Parameter(k.clone()))
            self.v.append(torch.nn.Parameter(v.clone()))
        self.init_k = [p.detach().clone() for p in self.k]
        self.init_v = [p.detach().clone() for p in self.v]

    @property
    def n_slots(self):
        return self.k[0].shape[2] + (1 if self.freeze_first else 0)

    def layers(self, batch):
        ks, vs = [], []
        for i in range(len(self.k)):
            k, v = self.k[i], self.v[i]
            if self.freeze_first:
                k = torch.cat([self.k0[i], k], dim=2)
                v = torch.cat([self.v0[i], v], dim=2)
            ks.append(k.expand(batch, -1, -1, -1))
            vs.append(v.expand(batch, -1, -1, -1))
        return ks, vs

    def drift(self):
        """Mean per-slot cosine similarity to the initialization and the
        relative norm of the displacement, keys and values."""
        out = {}
        for name, cur, ini in (("k", self.k, self.init_k), ("v", self.v, self.init_v)):
            cos, rel = [], []
            for p, q in zip(cur, ini):
                a = p.detach().flatten(0, 1).transpose(0, 1).flatten(1)  # [slots, H*d]
                b = q.flatten(0, 1).transpose(0, 1).flatten(1)
                cos.append(F.cosine_similarity(a, b, dim=1).mean().item())
                rel.append(((a - b).norm() / b.norm().clamp_min(1e-9)).item())
            out[name] = {"cos": cos, "rel": rel}
        return out

    def state(self):
        return {
            "k": [p.detach().cpu() for p in self.k],
            "v": [p.detach().cpu() for p in self.v],
            "k0": [p.detach().cpu() for p in self.k0],
            "v0": [p.detach().cpu() for p in self.v0],
        }


def forward_with_prefix(model, prefix, ids, attn_mask, output_attentions=False):
    ks, vs = prefix.layers(ids.shape[0])
    cache = make_cache(ks, vs, model)
    full_mask = torch.cat(
        [
            torch.ones(
                ids.shape[0], prefix.n_slots, dtype=attn_mask.dtype, device=ids.device
            ),
            attn_mask,
        ],
        dim=1,
    )
    return model(
        input_ids=ids,
        attention_mask=full_mask,
        past_key_values=cache,
        use_cache=True,
        output_attentions=output_attentions,
    )


# ----------------------------------------------------------------------
# batches, teacher, loss
# ----------------------------------------------------------------------


def encode_batch(tok, texts, device, max_len=None):
    enc = tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=max_len is not None,
        max_length=max_len,
    )
    return enc.input_ids.to(device), enc.attention_mask.to(device)


class DocContext:
    """The document's KV cache computed once: the teacher then reads only
    the last document token plus the conversation, so row j of its logits
    predicts conversation token j without re-encoding the document."""

    def __init__(self, model, doc_ids):
        self.model = model
        self.n = doc_ids.shape[1]
        with torch.no_grad():
            out = model(input_ids=doc_ids[:, :-1], use_cache=True)
        self.kv = [(k.clone(), v.clone()) for k, v in cache_layers(out.past_key_values)]
        self.last = doc_ids[:, -1:]

    def logits(self, ids, mask):
        """Logits over [last doc token] + ids, with the document in the
        cache; returns [B, 1+T, V] so that row j predicts ids[:, j]."""
        B = ids.shape[0]
        ks = [k.expand(B, -1, -1, -1) for k, _ in self.kv]
        vs = [v.expand(B, -1, -1, -1) for _, v in self.kv]
        cache = make_cache(ks, vs, self.model)
        inp = torch.cat([self.last.expand(B, -1), ids], dim=1)
        full = torch.cat(
            [torch.ones(B, self.n - 1 + 1, dtype=mask.dtype, device=ids.device), mask],
            dim=1,
        )
        return self.model(
            input_ids=inp, attention_mask=full, past_key_values=cache, use_cache=True
        ).logits


@torch.no_grad()
def teacher_topk(model, tok, doc_ctx, conv_ids, conv_mask, k):
    """Top-k teacher distribution at every conversation position with
    the document in context. Returns indices [B,T,k], probs [B,T,k]
    (renormalized over the k), and the teacher argmax [B,T]. Row j
    predicts conversation token j; row 0 comes from the last document
    position."""
    logits = doc_ctx.logits(conv_ids, conv_mask)[:, :-1]
    probs = torch.softmax(logits.float(), dim=-1)
    top_p, top_i = probs.topk(k, dim=-1)
    top_p = top_p / top_p.sum(-1, keepdim=True)
    return top_i, top_p, probs.argmax(-1)


@torch.no_grad()
def bare_argmax(model, conv_ids, conv_mask, bos_id):
    """Bare reader's argmax at every conversation position (no record,
    no prefix); a BOS token stands in for the empty context."""
    B = conv_ids.shape[0]
    bos = torch.full((B, 1), bos_id, dtype=conv_ids.dtype, device=conv_ids.device)
    ids = torch.cat([bos, conv_ids], dim=1)
    mask = torch.cat([torch.ones_like(bos), conv_mask], dim=1)
    logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1]
    return logits.argmax(-1)


def student_logits(model, prefix, conv_ids, conv_mask, output_attentions=False):
    """Student logits aligned like the teacher's: row j predicts conv
    token j. The prefix stands in for the document, so the first conv
    token is predicted from the prefix alone; we feed a BOS-free
    sequence and read logits shifted by one, using the last prefix slot
    through a dummy first token is not possible, so position 0 of the
    conversation is scored from the conversation's own first token:
    rows 1..T-1 are comparable, row 0 is dropped from the loss."""
    out = forward_with_prefix(model, prefix, conv_ids, conv_mask, output_attentions)
    return out.logits[:, :-1], out


def distill_loss(s_logits, top_i, top_p, loss_mask):
    """Top-k context distillation: cross-entropy of the student against
    the renormalized top-k teacher distribution, averaged over unmasked
    positions. Inputs aligned so that index j predicts conv token j+1."""
    logp = torch.log_softmax(s_logits.float(), dim=-1)
    g = logp.gather(-1, top_i[:, 1:])
    per_pos = -(top_p[:, 1:] * g).sum(-1)  # [B, T-1]
    m = loss_mask[:, 1:].float()
    return (per_pos * m).sum() / m.sum().clamp_min(1.0), per_pos


# ----------------------------------------------------------------------
# instruments
# ----------------------------------------------------------------------


@torch.no_grad()
def forced_choice(
    model,
    tok,
    prefix,
    ents,
    pairs,
    device,
    context_text=None,
    n_wrong=3,
    rng=None,
    bos_id=None,
    doc_ctx=None,
):
    """Forced-choice over the attribute's values: score each candidate's
    answer log-probability after the question; accuracy and mean margin
    (correct minus best wrong). context_text puts the record text in
    front (ceiling) instead of the prefix; prefix=None and
    context_text=None is the no-context floor."""
    rng = rng or random.Random(0)
    correct = 0
    margins = []
    for i, a in pairs:
        q, ans = qa_text(ents[i], a, templates()[0])
        others = sorted(
            {str(e[a]) for j, e in enumerate(ents) if j != i and e[a] != ents[i][a]}
        )
        wrong = rng.sample(others, min(n_wrong, len(others)))
        cands = [ans] + [f" {w}" for w in wrong]
        if doc_ctx is not None:
            # document in the cache: logits row 0 is the last doc token's
            ids, mask = encode_batch(tok, [q + c for c in cands], device)
            q_len = len(tok(q).input_ids)
            logits = doc_ctx.logits(ids, mask)[:, 1:]
            context_text = None
        else:
            texts = [(context_text or "") + q + c for c in cands]
            ids, mask = encode_batch(tok, texts, device)
            q_len = len(tok((context_text or "") + q).input_ids)
        if doc_ctx is not None:
            pass
        elif prefix is not None:
            logits = forward_with_prefix(model, prefix, ids, mask).logits
        elif context_text is None:
            bos = torch.full((ids.shape[0], 1), bos_id, dtype=ids.dtype, device=device)
            logits = model(
                input_ids=torch.cat([bos, ids], 1),
                attention_mask=torch.cat([torch.ones_like(bos), mask], 1),
            ).logits[:, 1:]
        else:
            logits = model(input_ids=ids, attention_mask=mask).logits
        logp = torch.log_softmax(logits.float(), -1)
        scores = []
        for c in range(ids.shape[0]):
            n = int(mask[c].sum())
            tgt = ids[c, q_len:n]
            lp = logp[c, q_len - 1 : n - 1].gather(-1, tgt[:, None]).sum()
            scores.append(lp.item())
        best_wrong = max(scores[1:]) if len(scores) > 1 else float("-inf")
        margins.append(scores[0] - best_wrong)
        correct += int(scores[0] > best_wrong)
    return {
        "acc": correct / len(pairs),
        "margin": float(np.mean(margins)),
        "n": len(pairs),
    }


@torch.no_grad()
def text_loss(model, prefix, tok, text, device, bos_id):
    """Mean next-token loss on a text, with the prefix loaded (prefix)
    or bare (prefix=None)."""
    ids, mask = encode_batch(tok, [text], device)
    if prefix is not None:
        logits = forward_with_prefix(model, prefix, ids, mask).logits[:, :-1]
        tgt = ids[:, 1:]
    else:
        bos = torch.full((1, 1), bos_id, dtype=ids.dtype, device=device)
        logits = model(input_ids=torch.cat([bos, ids], 1)).logits[:, :-1]
        tgt = ids
    return F.cross_entropy(logits.flatten(0, 1).float(), tgt.flatten()).item()


@torch.no_grad()
def eval_heldout(model, tok, prefix, doc_ctx, convs, device, k, bos_id, info_cache):
    """Held-out distillation loss split by informative / rest, plus the
    attention mass on the prefix slots per layer at those positions."""
    conv_ids, conv_mask = encode_batch(tok, convs, device)
    if "top" not in info_cache:
        top_i, top_p, t_arg = teacher_topk(model, tok, doc_ctx, conv_ids, conv_mask, k)
        b_arg = bare_argmax(model, conv_ids, conv_mask, bos_id)
        info_cache["top"] = (top_i, top_p)
        info_cache["info"] = (t_arg != b_arg) & conv_mask.bool()
    top_i, top_p = info_cache["top"]
    info = info_cache["info"]
    s_logits, out = student_logits(
        model, prefix, conv_ids, conv_mask, output_attentions=True
    )
    loss, per_pos = distill_loss(s_logits, top_i, top_p, conv_mask)
    m = conv_mask[:, 1:].bool()
    inf = info[:, 1:] & m
    rest = (~info[:, 1:]) & m
    res = {
        "loss": loss.item(),
        "loss_info": per_pos[inf].mean().item() if inf.any() else float("nan"),
        "loss_rest": per_pos[rest].mean().item(),
        "n_info": int(inf.sum()),
        "n_pos": int(m.sum()),
    }
    p = prefix.n_slots
    slot_info, slot_rest, sink_info, sink_rest = [], [], [], []
    for att in out.attentions:  # [B,H,T,p+T]
        a = att[:, :, 1:, :]  # align with per_pos rows
        slots = a[..., 1:p].sum(-1).mean(1)  # [B,T-1], trainable slots
        sink = a[..., 0].mean(1)
        slot_info.append(slots[inf].mean().item() if inf.any() else float("nan"))
        slot_rest.append(slots[rest].mean().item())
        sink_info.append(sink[inf].mean().item() if inf.any() else float("nan"))
        sink_rest.append(sink[rest].mean().item())
    res["slot_mass_info"] = slot_info
    res["slot_mass_rest"] = slot_rest
    res["sink_mass_info"] = sink_info
    res["sink_mass_rest"] = sink_rest
    res["slot_mass_info_mean"] = float(np.nanmean(slot_info))
    res["slot_mass_rest_mean"] = float(np.mean(slot_rest))
    return res


# ----------------------------------------------------------------------
# run one arm
# ----------------------------------------------------------------------


def build_task(args, rng):
    ents = make_record(rng, args.entities)
    doc = render_record(ents)
    train_pairs, hold_pairs = task_split(ents, rng, args.holdout)
    other = make_record(random.Random(args.seed + 1000), args.entities)
    other_doc = render_record(other, title="Record")
    o_train, o_hold = task_split(other, random.Random(args.seed + 1000), args.holdout)
    return ents, doc, train_pairs, hold_pairs, other, other_doc, o_hold


def run(args):
    device = args.device
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    tok, model = load_model(args.model, device)
    bos_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    ents, doc, train_pairs, hold_pairs, other, other_doc, o_hold = build_task(args, rng)
    doc_ids = tok(doc, return_tensors="pt").input_ids.to(device)
    n_doc = doc_ids.shape[1]
    doc_ctx = DocContext(model, doc_ids)
    other_ids = tok(other_doc, return_tensors="pt").input_ids.to(device)
    other_ctx = DocContext(model, other_ids)
    print(
        f"[task] entities={args.entities} doc_tokens={n_doc} slots={args.slots} "
        f"train_pairs={len(train_pairs)} heldout_pairs={len(hold_pairs)}",
        flush=True,
    )

    # initialization
    if args.init == "first_tokens":
        kv = kv_of_text(model, tok, doc, args.slots, device)
    elif args.init == "chunks":
        kv = kv_of_chunks(model, tok, doc, args.slots, rng, device)
    elif args.init == "other_doc":
        kv = kv_of_text(model, tok, other_doc, args.slots, device)
    elif args.init == "random":
        kv0 = kv_of_text(model, tok, doc, args.slots, device)
        kv = [
            (torch.randn_like(k) * k.std(), torch.randn_like(v) * v.std())
            for k, v in kv0
        ]
    elif args.init.startswith("file:"):
        st = torch.load(args.init[5:], map_location=device)
        kv = [
            (torch.cat([k0, k], 2), torch.cat([v0, v], 2))
            for k0, k, v0, v in zip(st["k0"], st["k"], st["v0"], st["v"])
        ]
    else:
        raise SystemExit(f"unknown init {args.init}")
    prefix = Prefix(kv, freeze_first=True).to(device)

    # held-out conversations (fixed) and the informative split
    hrng = random.Random(args.seed + 7)
    hold_convs = [
        make_conversation(ents, hold_pairs, hrng, args.turns, eval_style=True)
        for _ in range(args.heldout_convs)
    ]
    info_cache = {}

    # generic mix (Cartridges++): conversations about entities not in the
    # record, teacher = bare reader, so the student learns not to answer
    # from the prefix when the prefix has nothing to say
    generic_ents = make_record(random.Random(args.seed + 2000), args.entities)
    g_pairs = [(i, a) for i in range(len(generic_ents)) for a in attrs()]

    opt = torch.optim.Adam(list(prefix.k) + list(prefix.v), lr=args.lr)
    sched_total = args.sched_steps or args.steps

    def lr_at(step):
        if step < args.warmup:
            return args.warmup_min_lr + (args.lr - args.warmup_min_lr) * step / max(
                1, args.warmup
            )
        frac = (step - args.warmup) / max(1, sched_total - args.warmup)
        return args.lr * max(args.alpha_f, 1.0 - (1.0 - args.alpha_f) * frac)

    curve = []

    def evaluate(step):
        prefix.eval()
        r = eval_heldout(
            model,
            tok,
            prefix,
            doc_ctx,
            hold_convs,
            device,
            args.topk,
            bos_id,
            info_cache,
        )
        r["step"] = step
        r["drift"] = prefix.drift()
        if args.eval_choice_every and (
            step % args.eval_choice_every == 0 or step == args.steps
        ):
            r["choice"] = forced_choice(
                model,
                tok,
                prefix,
                ents,
                hold_pairs,
                device,
                rng=random.Random(1),
                bos_id=bos_id,
            )
            r["offctx_text_loss"] = text_loss(
                model, prefix, tok, other_doc, device, bos_id
            )
            r["offctx_choice"] = forced_choice(
                model,
                tok,
                prefix,
                other,
                o_hold,
                device,
                context_text=other_doc,
                rng=random.Random(2),
                bos_id=bos_id,
            )
        curve.append(r)
        print(
            f"VAL_CSV,{step},{r['loss']:.6f},{r['loss_info']:.6f},{r['loss_rest']:.6f},"
            f"{r['slot_mass_info_mean']:.4f},{r['slot_mass_rest_mean']:.4f}"
            + (
                f",choice_acc={r['choice']['acc']:.3f},margin={r['choice']['margin']:.3f},"
                f"offctx_loss={r['offctx_text_loss']:.4f},offctx_acc={r['offctx_choice']['acc']:.3f}"
                if "choice" in r
                else ""
            ),
            flush=True,
        )
        return r

    # references: floor, ceiling, bare off-context
    refs = {
        "floor": forced_choice(
            model,
            tok,
            None,
            ents,
            hold_pairs,
            device,
            rng=random.Random(1),
            bos_id=bos_id,
        ),
        "ceiling": forced_choice(
            model,
            tok,
            None,
            ents,
            hold_pairs,
            device,
            rng=random.Random(1),
            bos_id=bos_id,
            doc_ctx=doc_ctx,
        ),
        "offctx_text_loss_bare": text_loss(model, None, tok, other_doc, device, bos_id),
        "offctx_choice_bare": forced_choice(
            model,
            tok,
            None,
            other,
            o_hold,
            device,
            rng=random.Random(2),
            bos_id=bos_id,
            doc_ctx=other_ctx,
        ),
    }
    print(
        f"[refs] floor={refs['floor']['acc']:.3f} ceiling={refs['ceiling']['acc']:.3f} "
        f"offctx_loss_bare={refs['offctx_text_loss_bare']:.4f} offctx_acc_bare={refs['offctx_choice_bare']['acc']:.3f}",
        flush=True,
    )

    evaluate(0)
    t0 = time.time()
    step_times = []
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        ts = time.time()
        convs, generic_flags = [], []
        for _ in range(args.batch):
            if args.generic_frac > 0 and rng.random() < args.generic_frac:
                convs.append(make_conversation(generic_ents, g_pairs, rng, args.turns))
                generic_flags.append(True)
            else:
                convs.append(make_conversation(ents, train_pairs, rng, args.turns))
                generic_flags.append(False)
        conv_ids, conv_mask = encode_batch(tok, convs, device)
        top_i, top_p, _ = teacher_topk(
            model, tok, doc_ctx, conv_ids, conv_mask, args.topk
        )
        if any(generic_flags):
            # bare-reader teacher for the generic rows
            B = conv_ids.shape[0]
            bos = torch.full((B, 1), bos_id, dtype=conv_ids.dtype, device=device)
            with torch.no_grad():
                bl = model(
                    input_ids=torch.cat([bos, conv_ids], 1),
                    attention_mask=torch.cat([torch.ones_like(bos), conv_mask], 1),
                ).logits[:, :-1]
                bp = torch.softmax(bl.float(), -1)
                b_p, b_i = bp.topk(args.topk, -1)
                b_p = b_p / b_p.sum(-1, keepdim=True)
            gf = torch.tensor(generic_flags, device=device)
            top_i = torch.where(gf[:, None, None], b_i, top_i)
            top_p = torch.where(gf[:, None, None], b_p, top_p)
        prefix.train()
        s_logits, _ = student_logits(model, prefix, conv_ids, conv_mask)
        loss, _ = distill_loss(s_logits, top_i, top_p, conv_mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(list(prefix.k) + list(prefix.v), args.clip)
        opt.step()
        step_times.append(time.time() - ts)
        if step % args.log_every == 0:
            print(
                f"TRAIN_CSV,{step},{loss.item():.6f},{lr_at(step):.5f},{step_times[-1]:.3f}",
                flush=True,
            )
        if step % args.val_every == 0 or step == args.steps:
            evaluate(step)
        if args.save_every and step % args.save_every == 0:
            torch.save(prefix.state(), Path(args.out) / f"prefix-step{step}.pt")
    total = time.time() - t0

    result = {
        "model": args.model,
        "args": vars(args),
        "doc_tokens": n_doc,
        "slots": prefix.n_slots,
        "refs": refs,
        "curve": curve,
        "final": curve[-1],
        "seconds": total,
        "sec_per_step": float(np.mean(step_times)) if step_times else None,
        "host": platform.node() if args.record_host else None,
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "result.json", "w") as f:
        json.dump(result, f, indent=1)
    torch.save(prefix.state(), Path(args.out) / "prefix-final.pt")
    f = curve[-1]
    sps = result["sec_per_step"] if result["sec_per_step"] is not None else float("nan")
    print(
        f"RUN_DONE loss={f['loss']:.5f} info={f['loss_info']:.5f} choice={f.get('choice', {}).get('acc', float('nan')):.3f} "
        f"floor={refs['floor']['acc']:.3f} ceiling={refs['ceiling']['acc']:.3f} "
        f"slot_mass_info={f['slot_mass_info_mean']:.4f} sec/step={sps:.3f} ({total:.0f}s)",
        flush=True,
    )


# ----------------------------------------------------------------------
# host benchmark
# ----------------------------------------------------------------------


def lscpu_summary():
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    except Exception:
        return {}
    keep = {}
    for line in out.splitlines():
        for key in (
            "Model name",
            "L2 cache",
            "L3 cache",
            "Core(s) per socket",
            "Thread(s) per core",
        ):
            if line.startswith(key):
                keep[key] = line.split(":", 1)[1].strip()
    return keep


def bench(args):
    """Fixed training workload: pythia-class reader, slots, batch, turns,
    a few warmup steps then timed steps at each thread count. Also a
    long-context forward (the attention-over-KV pattern) and a plain
    matmul, to separate memory-bound from compute-bound behaviour."""
    device = args.device
    torch.manual_seed(0)
    rng = random.Random(0)
    tok, model = load_model(args.model, device)
    bos_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    ents, doc, train_pairs, hold_pairs, other, other_doc, o_hold = build_task(args, rng)
    doc_ids = tok(doc, return_tensors="pt").input_ids.to(device)
    doc_ctx = DocContext(model, doc_ids)
    kv = kv_of_text(model, tok, doc, args.slots, device)
    prefix = Prefix(kv).to(device)
    opt = torch.optim.Adam(list(prefix.k) + list(prefix.v), lr=1e-3)
    convs = [
        make_conversation(ents, train_pairs, rng, args.turns) for _ in range(args.batch)
    ]
    conv_ids, conv_mask = encode_batch(tok, convs, device)
    results = {
        "model": args.model,
        "lscpu": lscpu_summary(),
        "torch": torch.__version__,
        "slots": prefix.n_slots,
        "batch": args.batch,
        "conv_tokens": int(conv_ids.shape[1]),
        "doc_tokens": int(doc_ids.shape[1]),
        "threads": {},
    }
    print(
        f"[bench] {results['lscpu'].get('Model name','?')} L3={results['lscpu'].get('L3 cache','?')} "
        f"doc={results['doc_tokens']} conv={results['conv_tokens']}x{args.batch} slots={prefix.n_slots}",
        flush=True,
    )
    for nt in [int(x) for x in args.threads.split(",")]:
        torch.set_num_threads(nt)

        # training step
        def train_step():
            top_i, top_p, _ = teacher_topk(
                model, tok, doc_ctx, conv_ids, conv_mask, args.topk
            )
            s_logits, _ = student_logits(model, prefix, conv_ids, conv_mask)
            loss, _ = distill_loss(s_logits, top_i, top_p, conv_mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        for _ in range(args.warmup_steps):
            train_step()
        ts = []
        for _ in range(args.timed_steps):
            t = time.time()
            train_step()
            ts.append(time.time() - t)
        # long-context forward: the record repeated to ~ctx tokens
        long_ids = doc_ids.repeat(1, max(1, args.ctx // doc_ids.shape[1]))[
            :, : args.ctx
        ]
        with torch.no_grad():
            model(input_ids=long_ids)
            tl = []
            for _ in range(3):
                t = time.time()
                model(input_ids=long_ids)
                tl.append(time.time() - t)
        # matmul
        a = torch.randn(2048, 2048)
        b = torch.randn(2048, 2048)
        a @ b
        tm = []
        for _ in range(5):
            t = time.time()
            a @ b
            tm.append(time.time() - t)
        r = {
            "train_step_s": float(np.median(ts)),
            "train_step_min_s": float(min(ts)),
            "long_ctx_fwd_s": float(np.median(tl)),
            "long_ctx_tokens": int(long_ids.shape[1]),
            "matmul_2048_s": float(np.median(tm)),
            "matmul_gflops": 2 * 2048**3 / float(np.median(tm)) / 1e9,
        }
        results["threads"][str(nt)] = r
        print(
            f"BENCH,{nt},{r['train_step_s']:.4f},{r['train_step_min_s']:.4f},{r['long_ctx_fwd_s']:.4f},{r['matmul_gflops']:.1f}",
            flush=True,
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"BENCH_DONE {args.out}", flush=True)


# ----------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model", default="EleutherAI/pythia-14m")
        p.add_argument("--device", default=os.environ.get("DEVICE", "cpu"))
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--entities", type=int, default=16)
        p.add_argument(
            "--holdout",
            type=float,
            default=0.0,
            help="fraction of facts withheld from training (0 = all facts trained, held-out = new conversations)",
        )
        p.add_argument("--slots", type=int, default=64)
        p.add_argument("--turns", type=int, default=4)
        p.add_argument("--batch", type=int, default=8)
        p.add_argument("--topk", type=int, default=20)
        p.add_argument(
            "--fmt",
            default="qa",
            choices=["qa", "kv"],
            help="record format: prose entries with Q/A, or one fact per line",
        )
        p.add_argument(
            "--attrs",
            type=int,
            default=len(ATTRS),
            help="number of attributes per entity (first N)",
        )

    r = sub.add_parser("run")
    common(r)
    r.add_argument(
        "--init",
        default="first_tokens",
        help="first_tokens | chunks | other_doc | random | file:<prefix.pt>",
    )
    r.add_argument("--steps", type=int, default=200)
    r.add_argument(
        "--sched-steps", type=int, default=0, help="schedule horizon (0 = steps)"
    )
    r.add_argument("--lr", type=float, default=1e-2)
    r.add_argument("--warmup", type=int, default=20)
    r.add_argument("--warmup-min-lr", type=float, default=1e-4)
    r.add_argument("--alpha-f", type=float, default=0.02)
    r.add_argument("--clip", type=float, default=0.0)
    r.add_argument(
        "--generic-frac", type=float, default=0.0, help="Cartridges++ data mix fraction"
    )
    r.add_argument("--heldout-convs", type=int, default=32)
    r.add_argument("--val-every", type=int, default=10)
    r.add_argument("--eval-choice-every", type=int, default=50)
    r.add_argument("--log-every", type=int, default=10)
    r.add_argument("--save-every", type=int, default=0)
    r.add_argument("--record-host", action="store_true")
    r.add_argument("--out", required=True)

    b = sub.add_parser("bench")
    common(b)
    b.add_argument("--threads", default="8,16")
    b.add_argument("--warmup-steps", type=int, default=2)
    b.add_argument("--timed-steps", type=int, default=5)
    b.add_argument("--ctx", type=int, default=2048)
    b.add_argument("--out", required=True)

    args = ap.parse_args()
    FMT["name"] = args.fmt
    ACTIVE["attrs"] = list(ATTRS)[: args.attrs]
    if args.cmd == "run":
        run(args)
    else:
        bench(args)


if __name__ == "__main__":
    main()

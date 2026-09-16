#!/usr/bin/env python3
"""Recurrent-state and KV-cache precision on a real Gated DeltaNet hybrid.

Runs a Hugging Face hybrid (Qwen3.5 class: Gated DeltaNet layers with a
full-attention layer every fourth) through chunked prefill and token-by-
token greedy decode, and between steps rounds the cache in place: each
linear layer's recurrent state to bf16, fp8 e4m3, fp8 e5m2 or int8 (per-
head absmax scale, optionally with error feedback), and each attention
layer's keys and values to fp8. The model computes in bf16 with an fp32
state, as shipped; only the stored cache changes, so the score is what a
serving stack storing the cache in that format would see.

Tasks:
  multikey   RULER-style multi-key retrieval: N "key: value" needles in
             prose, ask for one key's value; exact match of the value.
  needle     single needle, the older knlp templates.
  ppl        perplexity over held-out prose windows.

    python3 scripts/hybrid_state_precision_eval.py --model Qwen/Qwen3.5-4B \\
        --task multikey --lengths 2048,4096,8192 --conditions base,state:bf16,...
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch

FP8_MAX = dict(e4m3=448.0, e5m2=57344.0)
INT_MAX = dict(int8=127.0, int6=31.0, int4=7.0)


# ---------------------------------------------------------------------------
# fake quantizers (round trip through the format; compute stays as shipped)
# ---------------------------------------------------------------------------


def quantize(x, fmt, dims):
    if fmt in ("fp32", "none"):
        return x
    if fmt == "bf16":
        return x.to(torch.bfloat16).to(x.dtype)
    xf = x.float()
    amax = xf.abs().amax(dim=dims, keepdim=True).clamp_min(1e-12)
    if fmt in FP8_MAX:
        scale = amax / FP8_MAX[fmt]
        dt = torch.float8_e4m3fn if fmt == "e4m3" else torch.float8_e5m2
        return ((xf / scale).to(dt).to(torch.float32) * scale).to(x.dtype)
    if fmt in INT_MAX:
        qmax = INT_MAX[fmt]
        scale = amax / qmax
        return (torch.round(xf / scale).clamp_(-qmax, qmax) * scale).to(x.dtype)
    raise ValueError(fmt)


def parse_layers(text):
    """'' -> None (all layers); '3.7' -> (False, {3, 7}); '!11' -> (True, {11})."""
    if not text:
        return None
    invert = text.startswith("!")
    idx = {int(t) for t in text.lstrip("!").split(".") if t}
    return (invert, idx)


def layer_selected(i, sel):
    if sel is None:
        return True
    invert, idx = sel
    return (i not in idx) if invert else (i in idx)


KV_FORMATS = {
    "v8": ("none", "e4m3"),
    "k8v8": ("e4m3", "e4m3"),
    "v6": ("none", "int6"),
    "v4": ("none", "int4"),
    "k8v4": ("e4m3", "int4"),
    "k8v6": ("e4m3", "int6"),
}


class Condition:
    """state:<fmt>[:ef][/every N][@layers]  kv:<fmt>[@layers]  phase:<prefill|decode>,
    joined by '+'.
    State formats: bf16, e4m3, e5m2, int8, int6, int4. KV formats: v8, k8v8,
    v6, v4, k8v6, k8v4 (fp8 keys; values fp8 or integer, per-token scale).
    <layers> is a '.'-separated list of layer indices, or '!' followed by a
    list to mean every layer except those.  Examples: base | state:bf16 |
    state:e4m3/64 | state:e4m3:ef | state:e4m3+kv:k8v8 | state:e4m3@3.7 |
    state:e4m3@!11 | kv:k8v8@31"""

    def __init__(self, spec):
        self.spec = spec
        self.state_fmt, self.state_ef, self.state_every = "none", False, 1
        self.k_fmt, self.v_fmt = "none", "none"
        self.state_layers, self.kv_layers = None, None
        self.phase = "both"
        for part in spec.split("+"):
            if part == "base":
                continue
            kind, _, rest = part.partition(":")
            rest, _, layers = rest.partition("@")
            layers = parse_layers(layers)
            if kind == "state":
                fmt, _, every = rest.partition("/")
                if fmt.endswith(":ef"):
                    fmt, self.state_ef = fmt[:-3], True
                self.state_fmt = fmt
                self.state_every = int(every) if every else 1
                self.state_layers = layers
            elif kind == "kv":
                self.k_fmt, self.v_fmt = KV_FORMATS[rest]
                self.kv_layers = layers
            elif kind == "phase":
                # round only during prefill (chunk boundaries) or only during
                # decode (after each generated token); default is both
                assert rest in ("prefill", "decode"), part
                self.phase = rest
            else:
                raise ValueError(part)
        self.residual = {}
        self.step = 0

    def reset(self):
        self.residual = {}
        self.step = 0

    def apply(self, cache, force=False, phase="prefill"):
        """Round the cache in place after a forward step."""
        self.step += 1
        if self.phase != "both" and phase != self.phase:
            return
        do_state = self.state_fmt != "none" and (
            force or self.step % self.state_every == 0
        )
        for i, layer in enumerate(cache.layers):
            rs = getattr(layer, "recurrent_states", None)
            if (
                rs is not None
                and do_state
                and len(rs)
                and rs[0] is not None
                and layer_selected(i, self.state_layers)
            ):
                s = rs[0]
                x = s.float()
                if self.state_ef:
                    r = self.residual.get(i)
                    if r is not None and r.shape == x.shape:
                        x = x + r
                y = quantize(x, self.state_fmt, (-2, -1))
                if self.state_ef:
                    self.residual[i] = x - y
                s.copy_(y.to(s.dtype))
            keys = getattr(layer, "keys", None)
            if (
                keys is not None
                and keys.numel()
                and (self.k_fmt != "none" or self.v_fmt != "none")
                and layer_selected(i, self.kv_layers)
            ):
                # per-token, per-head absmax scale, as a KV cache quantizer applies
                if self.k_fmt != "none":
                    layer.keys.copy_(quantize(layer.keys, self.k_fmt, (-1,)))
                if self.v_fmt != "none":
                    layer.values.copy_(quantize(layer.values, self.v_fmt, (-1,)))


# ---------------------------------------------------------------------------
# model driving: chunked prefill + greedy decode with cache rounding
# ---------------------------------------------------------------------------


@torch.no_grad()
def prefill(model, ids, cond, chunk):
    from transformers import DynamicCache

    cache = DynamicCache(config=model.config)
    logits = None
    t = ids.shape[1]
    for s in range(0, t, chunk):
        out = model(
            input_ids=ids[:, s : s + chunk], past_key_values=cache, use_cache=True
        )
        cache = out.past_key_values
        logits = out.logits[:, -1]
        cond.apply(cache)
    return cache, logits


@torch.no_grad()
def greedy(model, tok, ids, cond, chunk, max_new, stop_newline=True):
    cond.reset()
    cache, logits = prefill(model, ids, cond, chunk)
    out = []
    for _ in range(max_new):
        nxt = logits.argmax(-1)
        tid = nxt.item()
        if tid == tok.eos_token_id:
            break
        out.append(tid)
        text = tok.decode(out)
        if stop_newline and "\n" in text and text.strip():
            break
        o = model(input_ids=nxt.view(1, 1), past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        logits = o.logits[:, -1]
        cond.apply(cache, phase="decode")
    return tok.decode(out).strip()


@torch.no_grad()
def perplexity(model, ids, cond, chunk):
    cond.reset()
    from transformers import DynamicCache

    cache = DynamicCache(config=model.config)
    nll, count = 0.0, 0
    t = ids.shape[1]
    prev_logits = None
    for s in range(0, t, chunk):
        piece = ids[:, s : s + chunk]
        out = model(input_ids=piece, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        logits = out.logits.float()
        if prev_logits is not None:
            lp = torch.log_softmax(prev_logits, -1)
            nll -= lp.gather(-1, piece[:, :1, None]).sum().item()
            count += 1
        lp = torch.log_softmax(logits[:, :-1], -1)
        nll -= lp.gather(-1, piece[:, 1:, None]).sum().item()
        count += piece.shape[1] - 1
        prev_logits = logits[:, -1:]
        cond.apply(cache)
    return math.exp(nll / max(1, count))


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------


def load_prose(tok, min_tokens):
    """Held-out prose for haystacks and perplexity: wikitext-103 test."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n".join(x["text"] for x in ds if len(x["text"]) > 200)
    ids = tok(text, add_special_tokens=False).input_ids
    assert len(ids) >= min_tokens, f"need {min_tokens} prose tokens, have {len(ids)}"
    return ids


KEYS = [
    "latch",
    "orchid",
    "granite",
    "lantern",
    "meadow",
    "copper",
    "harbor",
    "velvet",
    "falcon",
    "cinder",
    "juniper",
    "saddle",
    "quartz",
    "willow",
    "anchor",
    "beacon",
    "cobalt",
    "dagger",
    "ember",
    "fjord",
    "glacier",
    "hazel",
    "ivory",
    "jasper",
    "kestrel",
    "lagoon",
    "marble",
    "nickel",
    "otter",
    "pepper",
    "quiver",
    "raven",
    "sable",
    "timber",
    "umber",
    "vortex",
    "walnut",
    "yarrow",
    "zephyr",
    "bramble",
]


def build_multikey(tok, prose_ids, length, n_keys, rng, ask="random"):
    """N needles 'The secret number for <key> is <6 digits>.' spread evenly
    through prose of about `length` tokens, then one key is asked."""
    keys = rng.sample(KEYS, n_keys)
    vals = [f"{rng.randint(100000, 999999)}" for _ in keys]
    needles = [f" The secret number for {k} is {v}. " for k, v in zip(keys, vals)]
    ask = {"random": rng.randrange(n_keys), "first": 0, "last": n_keys - 1}[ask]
    q = f"\n\nQuestion: What is the secret number for {keys[ask]}? Answer with the number only.\nAnswer:"
    q_ids = tok(q, add_special_tokens=False).input_ids
    n_ids = [tok(n, add_special_tokens=False).input_ids for n in needles]
    budget = length - len(q_ids) - sum(len(x) for x in n_ids) - 8
    start = rng.randrange(0, max(1, len(prose_ids) - budget - 1))
    hay = prose_ids[start : start + budget]
    seg = budget // (n_keys + 1)
    pieces = []
    for i, n in enumerate(n_ids):
        pieces += hay[i * seg : (i + 1) * seg] + n
    pieces += hay[n_keys * seg :]
    ids = pieces + q_ids
    return ids, vals[ask]


NEEDLE_TEMPLATES = [
    (
        "The secret code for the vault is {v}.",
        "What is the secret code for the vault?",
        ["7392", "4851", "2069", "8476", "3215"],
    ),
    (
        "The capital of the fictional country Zephyria is {v}.",
        "What is the capital of Zephyria?",
        ["Windholm", "Crystalburg", "Starview", "Moondale", "Sunridge"],
    ),
    (
        "Professor Smith discovered that the optimal temperature is {v} degrees.",
        "What temperature did Professor Smith discover was optimal?",
        ["42", "73", "28", "91", "56"],
    ),
]


def build_needle(tok, prose_ids, length, depth, rng):
    tmpl, question, values = rng.choice(NEEDLE_TEMPLATES)
    v = rng.choice(values)
    n_ids = tok(" " + tmpl.format(v=v) + " ", add_special_tokens=False).input_ids
    q = f"\n\nQuestion: {question} Answer briefly.\nAnswer:"
    q_ids = tok(q, add_special_tokens=False).input_ids
    budget = length - len(q_ids) - len(n_ids) - 8
    start = rng.randrange(0, max(1, len(prose_ids) - budget - 1))
    hay = prose_ids[start : start + budget]
    pos = int(depth * budget)
    return hay[:pos] + n_ids + hay[pos:] + q_ids, v


GSM_INSTRUCTION = (
    " Solve the problem with at most four short lines of plain-text arithmetic"
    " (no LaTeX, no headings), then give the final answer on the last line"
    " in the form '#### <number>'."
)


def build_gsm8k(tok, n, rng):
    """First n GSM8K test problems in a seeded order, chat-wrapped."""
    from datasets import load_dataset

    d = load_dataset("openai/gsm8k", "main", split="test")
    idx = list(range(len(d)))
    rng.shuffle(idx)
    out = []
    for i in idx[:n]:
        q = d[i]["question"] + GSM_INSTRUCTION
        ans = d[i]["answer"].split("####")[-1].strip().replace(",", "")
        ids = tok(q, add_special_tokens=False).input_ids
        out.append((chat_wrap(tok, ids), ans))
    return out


def gsm_match(text, ans):
    """The number after the last '####', else the last number in the text."""
    import re

    t = text.replace(",", "")
    m = re.findall(r"####\s*(-?\d+(?:\.\d+)?)", t)
    if not m:
        m = re.findall(r"-?\d+(?:\.\d+)?", t)
    if not m:
        return False
    try:
        return abs(float(m[-1]) - float(ans)) < 1e-6
    except ValueError:
        return False


def chat_wrap(tok, ids):
    """Wrap a raw prompt (token ids) in the model's chat template as a user
    turn with thinking disabled, so an instruct model answers directly."""
    text = tok.decode(ids)
    msgs = [{"role": "user", "content": text}]
    try:
        rendered = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        rendered = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    return tok(rendered, add_special_tokens=False).input_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument(
        "--task", choices=["multikey", "needle", "ppl", "gsm8k"], required=True
    )
    ap.add_argument(
        "--conditions",
        default="base,state:bf16,state:e4m3/64,state:e4m3,state:e4m3:ef,state:int8,kv:v8,kv:k8v8",
    )
    ap.add_argument("--lengths", default="2048,4096,8192")
    ap.add_argument("--prompts", type=int, default=20)
    ap.add_argument("--keys", type=int, default=4)
    ap.add_argument(
        "--ask",
        choices=["random", "first", "last"],
        default="random",
        help="which needle to ask for: first = the oldest write in the state",
    )
    ap.add_argument(
        "--chunk",
        type=int,
        default=64,
        help="prefill chunk (tokens between cache roundings)",
    )
    ap.add_argument("--max-new", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
        .to(args.device)
        .eval()
    )
    lengths = [int(x) for x in args.lengths.split(",")]
    if args.task == "gsm8k":
        lengths, prose = [0], None
    else:
        prose = load_prose(tok, max(lengths) * 2)
    conds = [Condition(c) for c in args.conditions.split(",")]
    rows = []
    manifest = dict(
        model=args.model,
        task=args.task,
        args=vars(args),
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    )
    import transformers

    manifest["transformers"] = transformers.__version__
    for length in lengths:
        rng = random.Random(args.seed + length)
        prompts = []
        if args.task == "gsm8k":
            prompts = build_gsm8k(tok, args.prompts, rng)
        for _ in range(args.prompts if args.task != "gsm8k" else 0):
            if args.task == "multikey":
                ids, ans = build_multikey(tok, prose, length, args.keys, rng, args.ask)
                ids = chat_wrap(tok, ids)
            elif args.task == "needle":
                ids, ans = build_needle(
                    tok, prose, length, rng.choice([0.1, 0.3, 0.5, 0.7, 0.9]), rng
                )
                ids = chat_wrap(tok, ids)
            else:
                start = rng.randrange(0, len(prose) - length - 1)
                ids, ans = prose[start : start + length], None
            prompts.append((ids, ans))
        for cond in conds:
            t0 = time.time()
            hits, ppls, samples = 0, [], []
            for j, (ids, ans) in enumerate(prompts):
                x = torch.tensor([ids], device=args.device)
                if args.task == "ppl":
                    ppls.append(perplexity(model, x, cond, args.chunk))
                else:
                    gsm = args.task == "gsm8k"
                    text = greedy(
                        model,
                        tok,
                        x,
                        cond,
                        args.chunk,
                        args.max_new,
                        stop_newline=not gsm,
                    )
                    ok = gsm_match(text, ans) if gsm else ans in text
                    hits += int(ok)
                    # keep every miss and the first three hits
                    if not ok or len(samples) < 3:
                        got = text[-80:] if gsm else text[:60]
                        samples.append(dict(i=j, expected=ans, got=got, ok=ok))
            row = dict(
                task=args.task,
                length=length,
                condition=cond.spec,
                prompts=len(prompts),
                wall_s=round(time.time() - t0, 1),
            )
            if args.task == "ppl":
                row["ppl"] = sum(ppls) / len(ppls)
                row["ppl_windows"] = [round(v, 4) for v in ppls]
            else:
                row["accuracy"] = hits / len(prompts)
                row["samples"] = samples
            rows.append(row)
            print(
                json.dumps({k: v for k, v in row.items() if k != "samples"}), flush=True
            )
            with open(args.out, "w") as f:
                json.dump(dict(manifest=manifest, rows=rows), f, indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    sys.exit(main())

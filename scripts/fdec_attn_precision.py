#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check whether 16-bit attention scores degrade a model.

The eager attention function of the transformers library computes the
query-key scores in the model dtype and applies the softmax to the rounded
result.  bf16 keeps 8 significant bits, so a score near 700 moves in steps
of 4.  A fused attention kernel accumulates the scores in float32.  A
harness that patches eager attention to emulate a KV-cache change therefore
measures a different model from the one a serving engine runs, unless it
also computes the scores in float32.

For one model this script measures WikiText-2 perplexity, and optionally a
GSM8K sample, with four attention paths:

  sdpa         the default scaled_dot_product_attention path;
  eager        the stock eager function, scores rounded to the model dtype;
  eager-s32    eager with float32 scores; the probabilities are cast to the
               model dtype before the value product;
  eager-all32  eager with float32 scores, probabilities and value sum; only
               the output is cast to the model dtype.

For each layer it also reports the mean absolute difference, in nats,
between the score the stock function computes and the float32 score, with
the mean and the maximum score magnitude.  A model is at risk when that
difference is not small against 1.

Measured in bf16: Qwen2.5-7B has a perplexity of 7.46 with sdpa, 8.84 with
eager and 7.46 with either float32 path (8 x 512 tokens), and solves 24, 7
and 25 of 32 GSM8K problems; its last layer has a mean score magnitude of
713 and a mean score error of 1.4 nat.  Llama-3.2-3B and Qwen3-8B have
scores below 50 and agree within 0.03 in perplexity on all four paths.
"""

import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fdec_compose import load_ids  # noqa: E402

STAT = {}


def make_attention(all32):
    """Eager attention with float32 scores that also records, per layer, how
    far the stock computation is from them."""

    def fwd(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        g = module.num_key_value_groups
        ks = key.repeat_interleave(g, dim=1)
        vs = value.repeat_interleave(g, dim=1)
        aw = torch.matmul(query.float(), ks.float().transpose(2, 3)).mul_(scaling)
        if query.shape[0] == 1 and query.shape[2] > 1:
            # teacher-forced pass: compare with what the stock function
            # computes (product and scale in the model dtype)
            stock = (torch.matmul(query, ks.transpose(2, 3)) * scaling).float()
            li = getattr(module, "layer_idx", -1)
            s = STAT.setdefault(li, dict(n=0, err=0.0, mag=0.0, mx=0.0))
            s["n"] += 1
            s["err"] += float((stock - aw).abs().mean())
            s["mag"] += float(aw.abs().mean())
            s["mx"] = max(s["mx"], float(aw.abs().max()))
            del stock
        if attention_mask is not None:
            aw += attention_mask[:, :, :, : ks.shape[-2]]
        p = F.softmax(aw, dim=-1)
        if all32:
            out = torch.matmul(p, vs.float()).to(query.dtype)
        else:
            out = torch.matmul(p.to(query.dtype), vs)
        return out.transpose(1, 2).contiguous(), p

    return fwd


@torch.no_grad()
def perplexity(model, ids, device):
    tot = n = 0
    for b in range(ids.shape[0]):
        x = ids[b : b + 1].to(device)
        lp = F.log_softmax(model(x, use_cache=False).logits[0, :-1].float(), -1)
        tot += float(-lp.gather(1, x[0, 1:, None]).sum())
        n += x.shape[1] - 1
    return math.exp(tot / n)


@torch.no_grad()
def gsm_solved(model, tok, items, device, batch, max_new):
    from fdec_task_acc import gsm_answer

    ok = 0
    for s in range(0, len(items), batch):
        g = list(range(s, min(s + batch, len(items))))
        enc = tok([items[i]["prompt"] for i in g], return_tensors="pt", padding=True)
        enc = enc.to(device)
        gen = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new,
            stop_strings=["Question:"],
            tokenizer=tok,
            pad_token_id=tok.pad_token_id,
        )
        texts = tok.batch_decode(
            gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
        )
        for i, t in zip(g, texts):
            ok += gsm_answer(t.split("Question:")[0]) == items[i]["gold"]
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument(
        "--ppl",
        default="8x512,4x1024",
        help="perplexity sets as SEQUENCESxLENGTH, comma separated",
    )
    ap.add_argument("--gsm", type=int, default=32, help="GSM8K problems; 0 skips")
    ap.add_argument("--gsm-batch", type=int, default=8)
    ap.add_argument("--gsm-max-new", type=int, default=200)
    ap.add_argument("--variants", default="sdpa,eager,eager-s32,eager-all32")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    items = []
    if args.gsm > 0:
        from fdec_task_acc import gsm_items

        items = gsm_items(8, args.gsm)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation="sdpa"
    )
    model = model.to(args.device).eval()
    attn_mod = sys.modules[type(model.model.layers[0].self_attn).__module__]
    stock_fn = attn_mod.eager_attention_forward
    sets = []
    for spec in args.ppl.split(","):
        n, length = spec.split("x")
        sets.append((int(length), load_ids(tok, "test", int(n), int(length))))
    res = dict(
        model=args.model,
        dtype=args.dtype,
        torch=torch.__version__,
        transformers=transformers.__version__,
        rows={},
    )
    if args.device.startswith("cuda"):
        res["gpu"] = torch.cuda.get_device_name(torch.device(args.device))
    for v in args.variants.split(","):
        STAT.clear()
        model.config._attn_implementation = "sdpa" if v == "sdpa" else "eager"
        attn_mod.eager_attention_forward = (
            stock_fn if v in ("sdpa", "eager") else make_attention(v == "eager-all32")
        )
        row = {f"ppl{n}": perplexity(model, ids, args.device) for n, ids in sets}
        layers = {
            li: dict(err=s["err"] / s["n"], mag=s["mag"] / s["n"], mx=s["mx"])
            for li, s in sorted(STAT.items())
        }
        if items:
            row["gsm_ok"] = gsm_solved(
                model, tok, items, args.device, args.gsm_batch, args.gsm_max_new
            )
            row["gsm_n"] = len(items)
        msg = " ".join(
            f"{k}={x:.3f}" if isinstance(x, float) else f"{k}={x}"
            for k, x in row.items()
        )
        if layers:
            worst = max(layers, key=lambda li: layers[li]["err"])
            w = layers[worst]
            msg += (
                f" | worst layer {worst}: mean|score err|={w['err']:.3f} nat "
                f"mean|score|={w['mag']:.1f} max|score|={w['mx']:.0f}"
            )
            row["layers"] = layers
        print(f"[attn-precision] {args.model} {v} {msg}", flush=True)
        res["rows"][v] = row
    attn_mod.eager_attention_forward = stock_fn
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

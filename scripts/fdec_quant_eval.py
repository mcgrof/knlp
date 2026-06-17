"""Real eval harness for the fdec weight-quant brick (the merged Claude x Codex
plan, step 1+2). Replaces the 6-prompt teacher-forced top-1 proxy with:

  - a held-out, long-context corpus eval (wikitext-2 test, ~4k-token context) so
    quality is measured over thousands of positions and real attention distance,
    NOT 6 short prompts;
  - calib (Fisher/AWQ) strictly disjoint from the eval text;
  - metrics: top-1 + top-5 agreement vs fp16, mean KL(fp16||q), and perplexity
    (absolute + ratio vs fp16) -- a generation-quality proxy, not just argmax;
  - the SAME pipeline across model families (cross-family generality gate).

Quant functions are imported from fdec_matrix so there is no algorithm drift.
"""

import argparse
import sys
import os

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fdec_matrix import fim_fake_quant, gptq_quantize_model  # noqa: E402

# calib for Fisher / AWQ -- diverse single-topic sentences, DISJOINT from wikitext
CALIB = " ".join(
    [
        "The mitochondria generate ATP through oxidative phosphorylation.",
        "Quicksort partitions an array around a pivot and recurses.",
        "Inflation erodes purchasing power as the money supply grows.",
        "Photosynthesis converts carbon dioxide and water into glucose.",
        "Neural networks learn by gradient descent on a loss function.",
        "The French Revolution began in 1789 at the Bastille prison.",
        "Antibiotics kill bacteria but not viruses like the influenza.",
        "Compound interest grows savings as returns accrue on returns.",
        "Tectonic plates drift a few centimeters per year over the mantle.",
        "Caching frequently accessed data reduces database latency.",
    ]
)

# name -> fim_fake_quant kwargs (group-wise everywhere; LM-head excluded by brick)
CONFIGS = {
    "int8": dict(base_bits=8, upgrade_frac=0.0, group=128, awq_alpha=0.0),
    "int4": dict(base_bits=4, upgrade_frac=0.2, group=128, awq_alpha=0.0),
    "int4_awq": dict(base_bits=4, upgrade_frac=0.2, group=128, awq_alpha=0.5),
    "int4_kvpin_awq": dict(
        base_bits=4,
        upgrade_frac=0.2,
        group=128,
        awq_alpha=0.5,
        pin_proj=("k_proj", "v_proj"),
    ),
    # GPTQ path (real Hessian error compensation); no awq_alpha here
    "int8_gptq": dict(base_bits=8, upgrade_frac=0.0, group=128),
    "int4_gptq": dict(base_bits=4, upgrade_frac=0.2, group=128),
    "int4_gptq_kvpin": dict(
        base_bits=4, upgrade_frac=0.2, group=128, pin_proj=("k_proj", "v_proj")
    ),
}


def _wikitext(split):
    from datasets import load_dataset

    for name, cfg in [
        ("Salesforce/wikitext", "wikitext-2-raw-v1"),
        ("wikitext", "wikitext-2-raw-v1"),
    ]:
        try:
            ds = load_dataset(name, cfg, split=split)
            return "\n".join(t for t in ds["text"] if t and not t.isspace())
        except Exception as e:
            print(f"[load {name}/{split}] {e}")
    raise RuntimeError("could not load wikitext")


def load_eval_ids(tok, n_tokens):
    ids = tok(_wikitext("test"), return_tensors="pt").input_ids[0][: n_tokens + 1]
    return ids.unsqueeze(0)


def _c4_text(n_chars):
    # stream a different domain (web text) -- the cross-domain generality check
    from datasets import load_dataset

    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    parts, total = [], 0
    for ex in ds:
        t = ex["text"]
        parts.append(t)
        total += len(t)
        if total >= n_chars:
            break
    return "\n".join(parts)


def load_gptq_calib(tok, n_tokens, source="wikitext"):
    if source == "c4":
        # C4 calib (web text), eval stays wikitext-test -> removes domain alignment
        text = _c4_text(n_tokens * 6)
    else:
        # wikitext TRAIN -- disjoint from the test eval set used for scoring
        text = _wikitext("train")
    ids = tok(text, return_tensors="pt").input_ids[0][:n_tokens]
    return tok.decode(ids)


@torch.no_grad()
def logprobs(model, ids, device):
    out = model(ids.to(device)).logits[0]  # [T, V]
    return out.float()


@torch.no_grad()
def eval_against_ref(lp_q, ref_lp, gold, chunk=512):
    """lp_q, ref_lp: [T,V] log-probs (ref already log_softmax'd, on CPU fp16).
    gold: [T] next-token ids. Returns dict of metrics accumulated over positions."""
    T = lp_q.shape[0]
    lp_q = F.log_softmax(lp_q, dim=-1)
    n = top1 = top5 = 0
    kl_sum = nll_q = nll_ref = 0.0
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        q = lp_q[s:e].to("cpu")
        r = ref_lp[s:e].float()
        g = gold[s:e]
        # agreement vs fp16 argmax
        ref_arg = r.argmax(-1)
        q_top5 = q.topk(5, dim=-1).indices
        top1 += (q.argmax(-1) == ref_arg).sum().item()
        top5 += (q_top5 == ref_arg.unsqueeze(-1)).any(-1).sum().item()
        # KL(fp16 || q)
        p = r.exp()
        kl_sum += (p * (r - q)).sum().item()
        # perplexity on gold
        nll_q += -q.gather(-1, g.unsqueeze(-1)).squeeze(-1).sum().item()
        nll_ref += -r.gather(-1, g.unsqueeze(-1)).squeeze(-1).sum().item()
        n += e - s
    import math

    return dict(
        top1=top1 / n,
        top5=top5 / n,
        meanKL=kl_sum / n,
        ppl_q=math.exp(nll_q / n),
        ppl_ref=math.exp(nll_ref / n),
        ppl_ratio=math.exp((nll_q - nll_ref) / n),
        n=n,
    )


def load_model(model_id, device, dtype):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=getattr(torch, dtype), attn_implementation="eager"
    ).to(device)
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--configs", default="int8,int4,int4_awq")
    ap.add_argument("--eval-tokens", type=int, default=4096)
    ap.add_argument("--gptq-calib-tokens", type=int, default=2048)
    ap.add_argument(
        "--gptq-calib-source", default="wikitext", choices=["wikitext", "c4"]
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = load_eval_ids(tok, args.eval_tokens)
    gold = ids[0, 1:].contiguous()  # next-token targets, [T]
    inp = ids[:, :-1]  # [1, T]
    print(f"[eval] {args.model}  positions={inp.shape[1]}  vocab={len(tok)}")

    cfgs = [c.strip() for c in args.configs.split(",") if c.strip()]
    gptq_calib = (
        load_gptq_calib(tok, args.gptq_calib_tokens, args.gptq_calib_source)
        if any("gptq" in c for c in cfgs)
        else None
    )

    # fp16 reference (the ground truth), store log-probs on CPU fp16
    model = load_model(args.model, device, args.dtype)
    ref_lp = F.log_softmax(logprobs(model, inp, device), dim=-1).to("cpu").half()
    del model
    torch.cuda.empty_cache()

    rows = []
    for cfg in cfgs:
        kw = CONFIGS[cfg]
        model = load_model(args.model, device, args.dtype)
        if "gptq" in cfg:
            _, wf = gptq_quantize_model(model, tok, gptq_calib, device=device, **kw)
        else:
            _, wf = fim_fake_quant(model, tok, CALIB, device=device, **kw)
        lp = logprobs(model, inp, device).to("cpu")
        m = eval_against_ref(lp, ref_lp, gold)
        m["cfg"] = cfg
        m["weight_factor"] = wf
        rows.append(m)
        print(
            f"  {cfg:<16} wfac={wf:.3f}  top1={m['top1']:.3f}  top5={m['top5']:.3f}  "
            f"KL={m['meanKL']:.4f}  pplq={m['ppl_q']:.3f}  pplratio={m['ppl_ratio']:.3f}"
        )
        del model
        torch.cuda.empty_cache()

    print(f"\n[ref] fp16 ppl={rows[0]['ppl_ref']:.3f}")
    import json

    print("JSON " + json.dumps({"model": args.model, "rows": rows}))


if __name__ == "__main__":
    main()

"""Brick C confirmation: end-model perplexity under oracle V-page selection.

The per-head oracle (fdec_pageoracle) showed the worst attention head needs ~all V
pages -- but per-head error can overstate end-token damage (it may wash out through
o_proj + later layers). This measures the END model: patch eager attention so each
query keeps only its top-B V pages (union across heads, ranked by cross-head mass,
renormalized), then report perplexity + top-1 vs full attention. Page = 128 tokens.
"""

import argparse
import sys

import torch
import torch.nn.functional as F

PAGE = 128
_PV = {"B": 0}  # 0 = full attention (no paging)


def pagev_attention(module, q, k, v, attention_mask, scaling, dropout=0.0, **kw):
    g = module.num_key_value_groups
    ks = k.repeat_interleave(g, dim=1)
    vs = v.repeat_interleave(g, dim=1)
    aw = torch.matmul(q, ks.transpose(2, 3)) * scaling
    if attention_mask is not None:
        aw = aw + attention_mask[:, :, :, : ks.shape[-2]]
    aw = F.softmax(aw, dim=-1)  # bf16 softmax (memory; precision fine for this gate)
    B = _PV["B"]
    if B and B > 0:
        Bsz, Hq, T, Tk = aw.shape
        npg = (Tk + PAGE - 1) // PAGE
        if B < npg:
            pad = npg * PAGE - Tk
            pm = F.pad(aw, (0, pad)).view(Bsz, Hq, T, npg, PAGE).sum(-1)  # [B,Hq,T,np]
            tot = pm.sum(1).float()  # [B,T,np] cross-head page mass per query
            sel = tot.topk(B, dim=-1).indices  # [B,T,B]
            psel = torch.zeros(Bsz, T, npg, dtype=torch.bool, device=aw.device)
            psel.scatter_(2, sel, True)
            keymask = psel.repeat_interleave(PAGE, dim=2)[:, :, :Tk]  # [B,T,Tk]
            del pm, tot, sel, psel
            aw.mul_(keymask.unsqueeze(1))  # in-place, no second copy
            aw.div_(aw.sum(-1, keepdim=True).clamp(min=1e-9))
    out = torch.matmul(aw.to(q.dtype), vs).transpose(1, 2).contiguous()
    return out, aw


def install(model):
    model.config._attn_implementation = "eager"
    saved = {}
    for m in model.modules():
        if hasattr(m, "num_key_value_groups") and hasattr(m, "layer_idx"):
            mn = type(m).__module__
            mod = sys.modules.get(mn)
            if mod is not None and hasattr(mod, "eager_attention_forward"):
                saved[mn] = mod.eager_attention_forward
                mod.eager_attention_forward = pagev_attention
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--budgets", default="20,64,128,0")
    ap.add_argument(
        "--needle", action="store_true", help="retrieval test instead of ppl"
    )
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(args.device)
    model.eval()
    install(model)
    if args.needle:
        # bury a unique passcode early, query at the end (retrieval under paging)
        filler = (
            "The weather was mild and the market stayed calm through the afternoon. "
        )
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        body = "\n".join(t for t in ds["text"] if t and not t.isspace())
        pre = tok(body, return_tensors="pt").input_ids[0][: args.ctx - 200]
        passcode = " The secret passcode is bluefin7732."
        pc_ids = tok(passcode, return_tensors="pt").input_ids[0]
        depth = int(0.15 * len(pre))  # needle at 15% depth
        q_ids = tok(" The secret passcode is", return_tensors="pt").input_ids[0]
        seq = torch.cat([pre[:depth], pc_ids, pre[depth:], q_ids])
        ids = seq.unsqueeze(0).to(args.device)
        inp = ids
        # the answer token = the token after "is" in the passcode (first of bluefin7732)
        ans_tok = pc_ids[len(tok(" The secret passcode is").input_ids) :][0].item()
        print(f"[pagev-needle] ctx={inp.shape[1]} answer_tok={ans_tok}")
    else:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n".join(t for t in ds["text"] if t and not t.isspace())
        ids = (
            tok(text, return_tensors="pt").input_ids[:, : args.ctx + 1].to(args.device)
        )
        inp = ids[:, :-1]
        print(f"[pagev-ppl] {args.model} ctx={inp.shape[1]} page={PAGE}")
    gold = ids[0, 1:] if not args.needle else None

    import math

    @torch.no_grad()
    def run(B):
        _PV["B"] = B
        lg = model(inp).logits[0]  # [T, V] bf16
        arg = lg.argmax(-1)
        nll = 0.0
        for s in range(0, lg.shape[0], 2048):  # chunk to avoid fp32 full-logit blowup
            e = min(s + 2048, lg.shape[0])
            lpc = F.log_softmax(lg[s:e].float(), dim=-1)
            nll += -lpc.gather(-1, gold[s:e].unsqueeze(-1)).squeeze(-1).sum().item()
        return math.exp(nll / lg.shape[0]), arg

    npg = (inp.shape[1] + PAGE - 1) // PAGE
    if args.needle:
        # report: does the model still predict the passcode token at the query end?
        print(
            f"\n{'budget B':>9}{'%ctx':>7}{'answer top-1?':>15}{'answer logprob':>16}"
        )
        print("-" * 47)
        for B in [int(x) for x in args.budgets.split(",")]:
            _PV["B"] = B
            with torch.no_grad():
                lg = model(inp).logits[0, -1].float()
            lp = F.log_softmax(lg, dim=-1)
            ok = int(lg.argmax().item() == ans_tok)
            tag = "full" if B == 0 else f"{100*B/npg:.1f}%"
            print(
                f"{B if B else 'full':>9}{tag:>7}{('YES' if ok else 'no'):>15}"
                f"{lp[ans_tok].item():>16.3f}"
            )
        return
    ppl_full, arg_full = run(0)
    print(
        f"\n{'budget B':>9}{'%ctx':>7}{'ppl':>9}{'ppl ratio':>11}{'top1 vs full':>14}"
    )
    print("-" * 50)
    for B in [int(x) for x in args.budgets.split(",")]:
        ppl, arg = run(B)
        top1 = (arg == arg_full).float().mean().item()
        tag = "full" if B == 0 else f"{100*B/npg:.1f}%"
        print(
            f"{B if B else 'full':>9}{tag:>7}{ppl:>9.3f}{ppl/ppl_full:>11.3f}{top1:>14.3f}"
        )


if __name__ == "__main__":
    main()

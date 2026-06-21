"""Phase 9 (autoregressive): does a quantized KV cache make the model GENERATE different text? The
teacher-forced atlas measures one-step logit error; this measures error COMPOUNDING through greedy
decode with an actual incremental cache (measurement_level=hf_dynamic_cache). For each prompt it
greedily generates N tokens natively and with the harness installed, then records the first token
position where they diverge and the cumulative token-agreement fraction. A model can look fine
teacher-forced yet drift badly once errors feed back through the cache.
"""

import os
import sys

import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # tools/kv
import k_bias_common as kbc  # noqa: E402


@torch.no_grad()
def _greedy(model, ids, device, new_tokens, pad_id, harness=None):
    if harness is not None:
        harness.install()
    try:
        inp = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        out = model.generate(
            inp,
            attention_mask=torch.ones_like(
                inp
            ),  # single unpadded seq; silences pad==eos warning
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,  # FIXED window: every prompt gets exactly new_tokens (no EOS)
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=pad_id,
        )
        return out[0, len(ids) :].tolist()
    finally:
        if harness is not None:
            harness.remove()


def _pad_id(model):
    cfg = model.config
    return (
        getattr(cfg, "pad_token_id", None)
        if getattr(cfg, "pad_token_id", None) is not None
        else getattr(cfg, "eos_token_id", None) or 0
    )


@torch.no_grad()
def _margin_at(model, ids, device, prefix_tokens, native_tok, quant_tok):
    """bf16 logit gap the quantized cache OVERTURNED at the divergence step: native picks
    native_tok, quant picks quant_tok; margin = logit[native_tok] - logit[quant_tok] from the NATIVE
    logits at that position. Small margin => a numerical tie-flip (not real drift); large => real.
    """
    seq = (
        torch.tensor(list(ids) + list(prefix_tokens), dtype=torch.long)
        .unsqueeze(0)
        .to(device)
    )
    lg = model(seq, attention_mask=torch.ones_like(seq)).logits[0, -1].float()
    return (lg[native_tok] - lg[quant_tok]).item()


@torch.no_grad()
def ar_divergence(model, infos, ids_list, device, new_tokens, k_spec, v_spec="bf16"):
    """Greedy-generate new_tokens per prompt (FIXED window), native vs (k_spec/v_spec)-quantized
    cache. first_divergence_frac normalized by the realized window; cumulative_agreement is a
    cascade-contaminated lower bound (one early flip forks the rest). margin_at_divergence is the
    bf16 logit gap the quant overturned -- a tiny margin means a numerical tie-flip, not real drift.
    Use k_spec='fp8:per_token' to model a real write-time FP8 cache (per_tensor over the whole cache
    is the pessimistic global-amax bound)."""
    pad_id = _pad_id(model)
    first_div, agree, margins = [], [], []
    for ids in ids_list:
        nat = _greedy(model, ids, device, new_tokens, pad_id, harness=None)
        h = kbc.FlexKVHarness(
            model, infos, kbc.parse_spec(k_spec), kbc.parse_spec(v_spec)
        )
        q = _greedy(model, ids, device, new_tokens, pad_id, harness=h)
        n = min(len(nat), len(q))
        if n == 0:
            continue
        fd = n  # n = never diverged within the window
        for i in range(n):
            if nat[i] != q[i]:
                fd = i
                break
        first_div.append(fd / max(n, 1))
        agree.append(sum(1 for i in range(n) if nat[i] == q[i]) / n)
        if fd < n:  # record the margin the quant overturned at the first divergence
            margins.append(_margin_at(model, ids, device, nat[:fd], nat[fd], q[fd]))
    import statistics as _S

    return dict(
        first_divergence_frac=round(_S.mean(first_div), 4) if first_div else 1.0,
        cumulative_agreement=round(_S.mean(agree), 4) if agree else 1.0,
        margin_at_divergence=round(_S.mean(margins), 4) if margins else None,
        frac_diverged=(
            round(sum(1 for f in first_div if f < 1.0) / len(first_div), 4)
            if first_div
            else 0.0
        ),
        n_prompts=len(first_div),
        new_tokens=new_tokens,
    )

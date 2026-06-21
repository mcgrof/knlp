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
def ar_divergence(model, infos, ids_list, device, new_tokens, k_spec, v_spec="bf16"):
    """Greedy-generate new_tokens per prompt, native vs (k_spec/v_spec)-quantized cache. Returns the
    mean first-divergence position (normalized 0..1, 1.0 = never diverged) and the mean cumulative
    token agreement over the generated continuation."""
    pad_id = _pad_id(model)
    first_div, agree = [], []
    for ids in ids_list:
        nat = _greedy(model, ids, device, new_tokens, pad_id, harness=None)
        h = kbc.FlexKVHarness(
            model, infos, kbc.parse_spec(k_spec), kbc.parse_spec(v_spec)
        )
        q = _greedy(model, ids, device, new_tokens, pad_id, harness=h)
        n = min(len(nat), len(q))
        if n == 0:
            continue
        # first divergence index (n = never diverged within the window)
        fd = n
        for i in range(n):
            if nat[i] != q[i]:
                fd = i
                break
        first_div.append(fd / max(new_tokens, 1))
        agree.append(sum(1 for i in range(n) if nat[i] == q[i]) / n)
    import statistics as _S

    return dict(
        first_divergence_frac=round(_S.mean(first_div), 4) if first_div else 1.0,
        cumulative_agreement=round(_S.mean(agree), 4) if agree else 1.0,
        n_prompts=len(first_div),
        new_tokens=new_tokens,
    )

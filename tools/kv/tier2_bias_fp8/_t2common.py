"""Tier-2 shared helpers: path to tier-1 k_bias_common, model-list loading, logit metrics."""

import os
import sys

# tools/kv is the parent dir -> import the tier-1 common
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import k_bias_common as kbc  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def load_models_spec(models_file=None, models=None, only=None):
    out = []
    if models_file:
        import yaml

        spec = yaml.safe_load(open(models_file))
        for m in spec["models"]:
            out.append(m)
    if models:
        for m in models:
            out.append(dict(short_name=m, model_id=m))
    if only:  # filter by short_name (case-insensitive, dash/underscore-insensitive)
        norm = lambda s: s.lower().replace("-", "").replace("_", "").replace(".", "")
        keep = {norm(x) for x in only}
        out = [
            m
            for m in out
            if norm(m["short_name"]) in keep
            or norm(m["model_id"].split("/")[-1]) in keep
        ]
    return out


def trc_for(model_id):
    return any(x in model_id.lower() for x in ("deepseek", "phi"))


@torch.no_grad()
def logits_list(model, ids_list, device):
    return [
        model(torch.tensor(ids).unsqueeze(0).to(device)).logits[0].float().cpu()
        for ids in ids_list
    ]


def metrics(base, other):
    import statistics as S

    me, mx, t1, t5, nll = [], [], [], [], []
    for b, c in zip(base, other):
        d = (b - c).abs()
        me.append(d.mean().item())
        mx.append(d.max().item())
        t1.append((b.argmax(-1) == c.argmax(-1)).float().mean().item())
        b5 = b.topk(5, -1).indices
        c1 = c.argmax(-1, keepdim=True)
        t5.append((b5 == c1).any(-1).float().mean().item())
        tgt = b.argmax(-1)
        nb = -F.log_softmax(b, -1).gather(-1, tgt.unsqueeze(-1)).mean().item()
        nc = -F.log_softmax(c, -1).gather(-1, tgt.unsqueeze(-1)).mean().item()
        nll.append(nc - nb)
    return dict(
        mean_logit_err=S.mean(me),
        max_logit_err=max(mx),
        top1=S.mean(t1),
        top5=S.mean(t5),
        nll_delta=S.mean(nll),
    )


def recovery(normal_err, prebias_err, bf16_err=0.0):
    denom = normal_err - bf16_err
    if denom <= 1e-9:
        return 0.0
    return (normal_err - prebias_err) / denom

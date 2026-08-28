#!/usr/bin/env python3
"""Does the document-induced likelihood shift carry the information?

Tests the premise behind distilling an information update rather than a
teacher.  The proposal is to run a small model twice, once with the
document and once without, and train a cartridge on the difference:
idiosyncrasies the small model has either way cancel, and what survives
is what the document changed.  Two assumptions have to hold before that
is worth any training compute, and both can be checked with forward
passes on trajectories we already have.

The first is concentration.  If the shift is spread evenly over every
token, or sits on style and formatting, then there is no information
signal to distil and the cancellation argument fails.  Reported as the
share of total positive shift carried by the top few percent of
positions, against the uniform expectation, plus the highest-shift
tokens so the claim can be eyeballed rather than taken on faith.

The second is transfer, and it is the one the economics rest on: a
cheap informed teacher is only useful if its shift resembles the shift
a large model would have produced.  Reported as the correlation of
per-token shift between a small and a large model on identical
trajectories.  If a 0.6B model's shift does not predict an 8B model's,
weak-teacher supervision has no foundation and no choice of divergence
repairs that.

No training, no cartridge, no cache surgery: this is the same model
scored twice per trajectory per size.

Env: SMALL, LARGE (HF ids), DATA_PARQUET, RECORD, N (trajectories),
MAX_ANSWER (cap answer tokens scored), OUT_JSON.
"""

import json
import os
import statistics
from pathlib import Path

os.environ.setdefault("CARTRIDGES_DIR", "/data/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/tmp/info_ratio")
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from cartridges.datasets import TrainDataset, DataSource

SMALL = os.environ.get("SMALL", "Qwen/Qwen3-0.6B")
LARGE = os.environ.get("LARGE", "Qwen/Qwen3-8B")
DATA_PARQUET = os.environ["DATA_PARQUET"]
RECORD = os.environ["RECORD"]
N = int(os.environ.get("N", "60"))
MAX_ANSWER = int(os.environ.get("MAX_ANSWER", "96"))
OUT_JSON = os.environ.get("OUT_JSON", "/tmp/info_ratio/info_ratio.json")
DEVICE = "cuda"


def load(model_id):
    m = (
        AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to(DEVICE)
        .eval()
    )
    m.requires_grad_(False)
    return m


tok = AutoTokenizer.from_pretrained(LARGE, trust_remote_code=True)
record_ids = tok(Path(RECORD).read_text(), return_tensors="pt").input_ids[0]
print(f"[ratio] record is {record_ids.shape[0]} tokens", flush=True)

dataset = TrainDataset.Config(
    data_sources=[DataSource(path=DATA_PARQUET, type="local")],
    top_k_logits=20,
    packed_seq_length=2048,
    packing_mode="truncate",
).instantiate(tokenizer=tok, seed=0)


def split(el):
    """Prompt/answer boundary: the first stored target row predicts the
    first answer token, so everything at or before it is prompt."""
    start = int(el.topk_token_idxs.min().item())
    return el.input_ids[:start], el.input_ids[start:]


trajectories = []
for el in dataset.elements:
    p, a = split(el)
    if a.shape[0] < 8:
        continue
    trajectories.append((p, a[:MAX_ANSWER]))
    if len(trajectories) >= N:
        break
print(f"[ratio] {len(trajectories)} trajectories", flush=True)


@torch.no_grad()
def answer_logprobs(model, prompt_ids, answer_ids, with_record):
    """Log probability of each answer token, with and without the
    document in front of the same prompt and answer."""
    parts = [record_ids.to(DEVICE)] if with_record else []
    parts += [prompt_ids.to(DEVICE), answer_ids.to(DEVICE)]
    ids = torch.cat(parts).unsqueeze(0)
    out = model(input_ids=ids)
    n = answer_ids.shape[0]
    start = ids.shape[1] - n - 1
    lp = F.log_softmax(out.logits[0, start : start + n].float(), dim=-1)
    return lp.gather(1, answer_ids.to(DEVICE).unsqueeze(1)).squeeze(1).cpu()


def shifts_for(model_id):
    model = load(model_id)
    per_traj = []
    for i, (p, a) in enumerate(trajectories):
        with_r = answer_logprobs(model, p, a, True)
        without_r = answer_logprobs(model, p, a, False)
        per_traj.append((a, (with_r - without_r)))
        if (i + 1) % 10 == 0:
            print(f"[ratio] {model_id}: {i + 1}/{len(trajectories)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return per_traj


def concentration(all_shift):
    """Share of total positive shift held by the top slices of
    positions.  Uniform would put 5% of the mass in the top 5%."""
    pos = torch.clamp(all_shift, min=0)
    total = float(pos.sum())
    if total <= 0:
        return None
    srt = torch.sort(pos, descending=True).values
    out = {}
    for frac in (0.01, 0.05, 0.10, 0.25):
        k = max(1, int(len(srt) * frac))
        out[f"top_{int(frac * 100)}pct_share"] = float(srt[:k].sum()) / total
    return out


def main():
    Path(os.path.dirname(OUT_JSON)).mkdir(parents=True, exist_ok=True)
    small = shifts_for(SMALL)
    large = shifts_for(LARGE)

    s_all = torch.cat([s for _, s in small])
    l_all = torch.cat([s for _, s in large])
    toks = torch.cat([a for a, _ in small])
    assert s_all.shape == l_all.shape

    def corr(x, y):
        x = x.double()
        y = y.double()
        xc, yc = x - x.mean(), y - y.mean()
        return float((xc @ yc) / (xc.norm() * yc.norm()).clamp_min(1e-12))

    def spearman(x, y):
        rx = torch.argsort(torch.argsort(x)).double()
        ry = torch.argsort(torch.argsort(y)).double()
        return corr(rx, ry)

    # do the models agree on WHERE the document matters most?
    k = max(1, int(0.10 * len(l_all)))
    top_idx = torch.topk(l_all, k).indices
    top_large = set(top_idx.tolist())
    top_small = set(torch.topk(s_all, k).indices.tolist())
    overlap = len(top_large & top_small) / k

    # A correlation over all positions is dominated by the many
    # low-shift tokens where both models trivially agree.  The
    # decision-relevant question is whether the small model tracks the
    # large one exactly where the document matters most, so restrict to
    # the large model's top decile and report agreement there.
    s_top, l_top = s_all[top_idx], l_all[top_idx]
    sign_agree = float(((s_top > 0) == (l_top > 0)).float().mean())
    small_negative_on_large_top = float((s_top <= 0).float().mean())

    top = torch.topk(l_all, 25)
    top_tokens = [
        dict(
            token=tok.decode([int(toks[i])]),
            shift_large=float(l_all[i]),
            shift_small=float(s_all[i]),
        )
        for i in top.indices.tolist()
    ]

    report = dict(
        small=SMALL,
        large=LARGE,
        trajectories=len(trajectories),
        positions=int(s_all.numel()),
        record_tokens=int(record_ids.shape[0]),
        concentration_large=concentration(l_all),
        concentration_small=concentration(s_all),
        mean_shift_large=float(l_all.mean()),
        mean_shift_small=float(s_all.mean()),
        pearson_small_vs_large=corr(s_all, l_all),
        spearman_small_vs_large=spearman(s_all, l_all),
        top10pct_position_overlap=overlap,
        pearson_on_large_top10pct=corr(s_top, l_top),
        spearman_on_large_top10pct=spearman(s_top, l_top),
        sign_agreement_on_large_top10pct=sign_agree,
        small_wrong_sign_on_large_top10pct=small_negative_on_large_top,
        top_tokens_by_large_shift=top_tokens,
    )
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=1)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "top_tokens_by_large_shift"},
            indent=1,
        )
    )
    print("top tokens by document-induced shift (large model):")
    for t in top_tokens[:15]:
        print(
            f"  {t['token']!r:24} large={t['shift_large']:+.2f} small={t['shift_small']:+.2f}"
        )
    print(f"INFO_RATIO_DONE {OUT_JSON}")


if __name__ == "__main__":
    main()

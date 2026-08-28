#!/usr/bin/env python3
"""Does the current objective already train where the document matters?

Selecting a few percent of positions is only worth building if the loss
does not already concentrate there.  Two quantities per answer
position, on the same stored trajectories:

  relevance  the document-induced log-probability shift, how much
             seeing the record changed the model's belief at that token
  weight     the coefficient the stored objective actually applies
             there, the summed teacher probability of that target row

If the objective's weight already tracks relevance, then selection is a
no-op dressed as an idea and it should die here rather than after a
training campaign.  If the two are close to unrelated, the objective is
spending most of its budget on positions the document did not affect,
which is a concrete mechanism for the measured result that training
improves loss while document specificity falls.

Reports the correlation, and the more decision-relevant number: the
share of total loss weight sitting on the most document-relevant
positions, against the uniform expectation.

Env: MODEL, DATA_PARQUET, RECORD, N, MAX_ANSWER, OUT_JSON.
"""

import json
import os
import statistics
from pathlib import Path

os.environ.setdefault("CARTRIDGES_DIR", "/data/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/tmp/rel_weight")
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from cartridges.datasets import TrainDataset, DataSource

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
DATA_PARQUET = os.environ["DATA_PARQUET"]
RECORD = os.environ["RECORD"]
N = int(os.environ.get("N", "60"))
MAX_ANSWER = int(os.environ.get("MAX_ANSWER", "96"))
OUT_JSON = os.environ.get("OUT_JSON", "/tmp/rel_weight/relevance_vs_weight.json")
DEVICE = "cuda"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
record_ids = tok(Path(RECORD).read_text(), return_tensors="pt").input_ids[0]
model = (
    AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    .to(DEVICE)
    .eval()
)
model.requires_grad_(False)

dataset = TrainDataset.Config(
    data_sources=[DataSource(path=DATA_PARQUET, type="local")],
    top_k_logits=20,
    packed_seq_length=2048,
    packing_mode="truncate",
).instantiate(tokenizer=tok, seed=0)


@torch.no_grad()
def answer_logprobs(prompt_ids, answer_ids, with_record):
    parts = [record_ids.to(DEVICE)] if with_record else []
    parts += [prompt_ids.to(DEVICE), answer_ids.to(DEVICE)]
    ids = torch.cat(parts).unsqueeze(0)
    out = model(input_ids=ids)
    n = answer_ids.shape[0]
    start = ids.shape[1] - n - 1
    lp = F.log_softmax(out.logits[0, start : start + n].float(), dim=-1)
    return lp.gather(1, answer_ids.to(DEVICE).unsqueeze(1)).squeeze(1).cpu()


def main():
    rel, wgt = [], []
    used = 0
    for el in dataset.elements:
        idxs = el.topk_token_idxs
        if idxs.numel() == 0:
            continue
        first = int(idxs.min().item())
        prompt_ids, answer_ids = el.input_ids[:first], el.input_ids[first:]
        if answer_ids.shape[0] < 8:
            continue
        answer_ids = answer_ids[:MAX_ANSWER]

        # weight per answer position: the summed teacher probability the
        # stored objective applies at the row predicting that token
        probs = el.topk_logprobs.exp()
        by_row = {}
        for r, p in zip(idxs.tolist(), probs.tolist()):
            by_row[r] = by_row.get(r, 0.0) + p
        w = [by_row.get(first + j, 0.0) for j in range(answer_ids.shape[0])]

        shift = answer_logprobs(prompt_ids, answer_ids, True) - answer_logprobs(
            prompt_ids, answer_ids, False
        )
        rel.extend(shift.tolist())
        wgt.extend(w)
        used += 1
        if used % 10 == 0:
            print(f"[rel-weight] {used}/{N}", flush=True)
        if used >= N:
            break

    r = torch.tensor(rel, dtype=torch.float64)
    w = torch.tensor(wgt, dtype=torch.float64)

    def corr(x, y):
        xc, yc = x - x.mean(), y - y.mean()
        return float((xc @ yc) / (xc.norm() * yc.norm()).clamp_min(1e-12))

    def spearman(x, y):
        return corr(
            torch.argsort(torch.argsort(x)).double(),
            torch.argsort(torch.argsort(y)).double(),
        )

    total_w = float(w.sum())
    shares = {}
    for frac in (0.05, 0.10, 0.25):
        k = max(1, int(len(r) * frac))
        top = torch.topk(r, k).indices
        shares[f"weight_share_on_top_{int(frac * 100)}pct_relevant"] = (
            float(w[top].sum()) / total_w
        )

    report = dict(
        model=MODEL,
        trajectories=used,
        positions=len(rel),
        pearson=corr(r, w),
        spearman=spearman(r, w),
        mean_relevance=float(r.mean()),
        mean_weight=float(w.mean()),
        **shares,
    )
    Path(os.path.dirname(OUT_JSON)).mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    print(
        "\nuniform expectation for the share columns is the percentage "
        "itself; well above means the objective already trains where the "
        "document matters and selection buys little"
    )
    print(f"REL_WEIGHT_DONE {OUT_JSON}")


if __name__ == "__main__":
    main()

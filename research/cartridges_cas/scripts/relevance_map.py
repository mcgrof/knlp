#!/usr/bin/env python3
"""Score every target row by how much the document changed it.

The objective currently spends its budget almost independently of where
the document matters: measured correlation between a row's loss weight
and its document-induced shift is 0.09, and only 5.5% of the weight
lands on the top 5% most document-relevant positions, against 5% for
uniform.  Meanwhile the document's effect is about ten times more
concentrated than chance.  So most of the gradient is spent on
positions the document did not touch.

This builds the map needed to do something about that: for each target
row the trainer will actually see, the log-probability the model
assigns to that row's chosen token with the record in context minus the
same without it.  High values are rows the document is responsible for.

Written for the schedule the trainer uses, so selection can be applied
to exactly the rows that would otherwise be trained, and saved as a
plain map from element index and row index to a score.

Env: MODEL, DATA_PARQUET, RECORD, SCHEDULE_JSON, OUT_JSON.
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CARTRIDGES_DIR", "/data/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/tmp/relmap")
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from cartridges.datasets import TrainDataset, DataSource

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
DATA_PARQUET = os.environ["DATA_PARQUET"]
RECORD = os.environ["RECORD"]
SCHEDULE_JSON = os.environ["SCHEDULE_JSON"]
OUT_JSON = os.environ.get("OUT_JSON", "/tmp/relmap/relevance_map.json")
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

schedule = json.loads(Path(SCHEDULE_JSON).read_text())
elements = sorted({i for step in schedule["schedule"] for i in step})
print(f"[relmap] {len(elements)} scheduled elements", flush=True)


@torch.no_grad()
def shift_for(el):
    """Per answer-token document-induced shift, aligned to target rows."""
    idxs = el.topk_token_idxs
    if idxs.numel() == 0:
        return {}
    first = int(idxs.min().item())
    prompt_ids, answer_ids = el.input_ids[:first], el.input_ids[first:]
    if answer_ids.shape[0] == 0:
        return {}

    def lp(with_record):
        parts = [record_ids.to(DEVICE)] if with_record else []
        parts += [prompt_ids.to(DEVICE), answer_ids.to(DEVICE)]
        ids = torch.cat(parts).unsqueeze(0)
        out = model(input_ids=ids)
        n = answer_ids.shape[0]
        start = ids.shape[1] - n - 1
        l = F.log_softmax(out.logits[0, start : start + n].float(), dim=-1)
        return l.gather(1, answer_ids.to(DEVICE).unsqueeze(1)).squeeze(1).cpu()

    delta = (lp(True) - lp(False)).tolist()
    # row r predicts element token r+1, and the first answer token is at
    # `first`, so row `first + j` corresponds to answer position j
    return {first + j: float(d) for j, d in enumerate(delta)}


def main():
    out = {}
    for n, i in enumerate(elements):
        out[str(i)] = {str(k): v for k, v in shift_for(dataset.elements[i]).items()}
        if (n + 1) % 20 == 0:
            print(f"[relmap] {n + 1}/{len(elements)}", flush=True)
    Path(os.path.dirname(OUT_JSON)).mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(dict(model=MODEL, parquet=DATA_PARQUET, map=out), f)
    vals = [v for e in out.values() for v in e.values()]
    vals.sort()
    print(
        f"[relmap] {len(vals)} scored rows; "
        f"median {vals[len(vals) // 2]:.3f}, "
        f"p90 {vals[int(0.9 * len(vals))]:.3f}, "
        f"max {vals[-1]:.3f}"
    )
    print(f"RELEVANCE_MAP_DONE {OUT_JSON}")


if __name__ == "__main__":
    main()

"""Multi-dataset pruned-PPL eval (Phase 2.3).

Wraps eval_pruned_ppl over a list of datasets so we can check
whether the single-dataset TinyStories result holds elsewhere.
For each dataset we also bucket loss by sequence position so a
"loss concentrated in dense prefix" failure mode is visible.

Default dataset list: TinyStories (easy local), FineWeb-Edu held-
out (train tail since it has no validation split when streaming),
WikiText-103, optionally PG-19 if accessible. The OpenWebText
dataset (Skylion007/openwebtext) is on the menu but is not in the
default list because the pull is large and slow.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import DataConfig, collate, get_tokenizer, get_train_val_streams  # noqa: E402
from src.eval_pruned_ppl import load_model, retention_to_policy_params, eval_one  # noqa: E402
from src.utils import pick_device, pick_dtype, report_device, set_seed  # noqa: E402


DEFAULT_DATASETS = [
    # (name, config-or-None, text-column, streaming, val_split, train_split)
    ("roneneldan/TinyStories", None, "text", False, "validation", "train"),
    ("Salesforce/wikitext", "wikitext-103-raw-v1", "text", False, "validation", "train"),
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text", True, "train", "train"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=str, required=True)
    p.add_argument("--datasets_yaml", type=str, default=None,
                   help="Optional path to a JSON with [{name,config,text_column,streaming,val_split,train_split},...]; otherwise uses the default list above.")
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--block_size", type=int, default=16)
    p.add_argument("--policies", type=str, default="full,recent,sink_recent,kri")
    p.add_argument("--retention_fracs", type=str, default="1.0,0.5,0.25,0.125,0.0625")
    p.add_argument("--sink_blocks", type=int, default=1)
    p.add_argument("--n_batches", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--precision", type=str, default="auto",
                   choices=["auto", "fp32", "fp16", "bf16"])
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _csv_floats(s: str):
    return tuple(float(x) for x in s.split(","))


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    report_device()
    device = pick_device()
    dtype = pick_dtype(args.precision)

    out_jsonl = Path(args.output)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_csv = out_jsonl.with_suffix(".csv")

    if args.datasets_yaml:
        datasets_list = json.loads(Path(args.datasets_yaml).read_text())
        ds_specs = [
            (d["name"], d.get("config"), d.get("text_column", "text"),
             d.get("streaming", False), d.get("val_split", "validation"),
             d.get("train_split", "train"))
            for d in datasets_list
        ]
    else:
        ds_specs = DEFAULT_DATASETS

    policies = args.policies.split(",")
    retention_fracs = _csv_floats(args.retention_fracs)
    models = args.models.split(",")

    tok = get_tokenizer("openai-community/gpt2")
    rows = []
    fh = out_jsonl.open("w")

    for ds_name, ds_config, text_col, streaming, val_split, train_split in ds_specs:
        ds_label = ds_name + (f"/{ds_config}" if ds_config else "")
        print(f"\n=== dataset: {ds_label} ===")
        data_cfg = DataConfig(
            dataset_name=ds_name, dataset_config=ds_config,
            text_column=text_col, streaming=streaming,
            val_split=val_split, train_split=train_split,
            seq_len=args.seq_len,
        )
        try:
            _, val_ds = get_train_val_streams(data_cfg, tok)
        except Exception as e:
            print(f"  SKIP: dataset load failed: {e}")
            continue

        for m_path in models:
            print(f"  -- model {m_path}")
            model, tag = load_model(m_path, device)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                    collate_fn=collate, num_workers=0)
            cached = []
            for i, b in enumerate(val_loader):
                if i >= args.n_batches:
                    break
                cached.append(b)
            if not cached:
                print(f"     no batches available, skipping")
                del model
                continue

            for policy in policies:
                if policy == "full":
                    params = retention_to_policy_params(1.0, args.seq_len,
                                                        args.block_size,
                                                        args.sink_blocks, "full")
                    res = eval_one(model, cached, "full", params, args.n_batches,
                                   device, dtype, args.block_size)
                    row = {"dataset": ds_label, "model": tag, "policy": "full",
                           "retention_target": 1.0, **params, **res}
                    rows.append(row)
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    print(f"     full: CE={res['cross_entropy']:.4f} PPL={res['perplexity']:.2f}")
                    continue
                for frac in retention_fracs:
                    if frac >= 0.999:
                        continue
                    params = retention_to_policy_params(frac, args.seq_len,
                                                        args.block_size,
                                                        args.sink_blocks, policy)
                    res = eval_one(model, cached, policy, params, args.n_batches,
                                   device, dtype, args.block_size)
                    row = {"dataset": ds_label, "model": tag, "policy": policy,
                           "retention_target": frac, **params, **res}
                    rows.append(row)
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    print(f"     {policy:12s} f={frac:.4f}  "
                          f"CE={res['cross_entropy']:.4f} PPL={res['perplexity']:.2f}")

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    fh.close()
    if rows:
        with out_csv.open("w", newline="") as cf:
            w = csv.DictWriter(cf, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"\nwrote {len(rows)} rows -> {out_jsonl} and {out_csv}")
    return 0


if __name__ == "__main__":
    import os
    rc = main()
    os._exit(rc)

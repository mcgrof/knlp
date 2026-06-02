"""Evaluate validation perplexity for several models under several
attention-pruning policies and retention budgets.

We compare:

  - Models: any list given via `--models`. Each entry is either a HF
    name (`openai-community/gpt2`) or a path to a checkpoint produced
    by `train_kri.py` (`runs/.../checkpoint_final.pt`).
  - Policies: `full`, `recent`, `sink_recent`, `kri`.
  - Retention budgets: a list of target fractions (1.0, 0.5, 0.25, ...)
    of the sequence retained PER QUERY in the decode region.

For each (model, policy, budget) we report:

    cross_entropy, perplexity, retained_fraction, n_eval_tokens

Outputs are emitted as a JSONL file and a CSV file (same data).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import DataConfig, collate, get_tokenizer, get_train_val_streams  # noqa: E402
from src.kri_mask import fixed_policy_mask  # noqa: E402
from src.model_gpt2_kri import GPT2KRI  # noqa: E402
from src.utils import pick_device, pick_dtype, report_device, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=str, required=True,
                   help="Comma-separated list of HF names or checkpoint paths.")
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--block_size", type=int, default=16)
    p.add_argument("--policies", type=str, default="full,recent,sink_recent,kri")
    p.add_argument("--retention_fracs", type=str, default="1.0,0.5,0.25,0.125,0.0625")
    p.add_argument("--sink_blocks", type=int, default=1)
    p.add_argument("--n_batches", type=int, default=32,
                   help="Number of validation batches per (model, policy, budget).")
    p.add_argument("--batch_size", type=int, default=2)

    p.add_argument("--dataset_name", type=str, default="roneneldan/TinyStories")
    p.add_argument("--dataset_config", type=str, default=None)
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--streaming", type=str, default="false")
    p.add_argument("--val_split", type=str, default="validation")
    p.add_argument("--train_split", type=str, default="train")

    p.add_argument("--precision", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"])

    p.add_argument(
        "--tokenizer_name",
        type=str,
        default="openai-community/gpt2",
        help=(
            "Tokenizer to use for the validation dataset. Must match the "
            "model under test or you will get out-of-bounds token ids "
            "(SmolLM2 vocab=49152 vs GPT-2 vocab=50257). For SmolLM2 "
            "checkpoints pass HuggingFaceTB/SmolLM2-360M."
        ),
    )

    p.add_argument("--output", type=str, required=True,
                   help="Path to JSONL output. A sibling .csv is also written.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _truthy(s: str) -> bool:
    return str(s).lower() in ("1", "true", "t", "yes", "y")


def _csv_floats(s: str):
    return tuple(float(x) for x in s.split(","))


_LLAMA_FAMILY_NAME_FRAGMENTS = (
    "smollm",
    "qwen",
    "llama",
    "mistral",
    "gemma",
    "phi-3",
)


def _is_llama_family(name: str) -> bool:
    """Substring test for HF causal LMs that the SmolLM2KRI wrapper
    handles generically (Llama-style decoder blocks with k_proj /
    v_proj / o_proj and model.model.layers[i].self_attn access).
    GPT-2 family does NOT match.
    """
    s = name.lower()
    return any(frag in s for frag in _LLAMA_FAMILY_NAME_FRAGMENTS)


def load_model(path_or_name: str, device: torch.device) -> Tuple[GPT2KRI, str]:
    """Load either an HF name or a local checkpoint and tag the model.

    Auto-detects GPT-2 vs Llama-family checkpoints by looking at the
    checkpoint's state-dict keys: GPT-2 checkpoints have `wte.weight`
    at the top level; Llama-family checkpoints (SmolLM2, Qwen3,
    Llama-3.2, etc.) have `hf.model.*` keys and go through
    SmolLM2KRI's generic from_hf loader. HF names dispatch on
    substring match: anything containing smollm / qwen / llama /
    mistral / gemma / phi-3 loads through SmolLM2KRI (it's
    architecture-agnostic for any Llama-style decoder); otherwise
    through GPT2KRI.
    """
    p = Path(path_or_name)
    if p.exists() and p.is_file():
        ck = torch.load(path_or_name, map_location="cpu", weights_only=False)
        sd = ck["model"]
        is_llama_family = any(k.startswith("hf.") for k in sd.keys())
        if is_llama_family:
            from src.model_smollm2_kri import SmolLM2KRI
            # We need to instantiate from the same base name as training.
            # The base name lives in ck["args"]["init_model"] if saved.
            base_name = (ck.get("args") or {}).get(
                "init_model", "HuggingFaceTB/SmolLM2-360M"
            )
            model = SmolLM2KRI.from_hf(base_name)
            model.load_state_dict(sd, strict=False)
            model = model.to(device).eval()
        else:
            from src.model_gpt2_kri import GPT2Config
            cfg = GPT2Config(**ck["cfg"])
            model = GPT2KRI(cfg)
            model.load_state_dict(sd, strict=False)
            model.lm_head.weight = model.wte.weight
            model = model.to(device).eval()
        # tag from filename stem
        if p.stem.startswith("checkpoint"):
            tag = p.parent.name if p.parent.name else p.stem
        else:
            tag = p.stem
        return model, f"ckpt:{tag}"
    # HF name path: dispatch on the name
    if _is_llama_family(path_or_name):
        from src.model_smollm2_kri import SmolLM2KRI
        model = SmolLM2KRI.from_hf(path_or_name).to(device).eval()
    else:
        model = GPT2KRI.from_hf_gpt2(path_or_name).to(device).eval()
    return model, path_or_name


def retention_to_policy_params(frac: float, seq_len: int, block_size: int,
                              sink_blocks: int, policy: str) -> dict:
    """Translate a target retention fraction into concrete policy params.

    For all policies except "full":
        target_keep_tokens = round(frac * seq_len)

    Budget allocation:
      - `recent`     : local_window = target_keep_tokens
      - `sink_recent`: keep sink_blocks*block_size tokens for the sink,
                       the rest goes to the local window
      - `kri`        : keep sink + 1/3 budget local, 2/3 in KRI top-k
                       blocks
    """
    target = max(1, int(round(frac * seq_len)))
    sink_tokens = sink_blocks * block_size
    if policy == "full":
        return {"local_window_tokens": seq_len, "sink_blocks": 0, "topk_blocks": 0}
    if policy == "recent":
        return {"local_window_tokens": target, "sink_blocks": 0, "topk_blocks": 0}
    if policy == "sink_recent":
        rem = max(block_size, target - sink_tokens)
        return {"local_window_tokens": rem, "sink_blocks": sink_blocks, "topk_blocks": 0}
    if policy == "kri":
        rem = max(block_size, target - sink_tokens)
        local = max(block_size, rem // 3)
        for_kri = max(0, rem - local)
        topk = max(0, for_kri // block_size)
        return {"local_window_tokens": local, "sink_blocks": sink_blocks, "topk_blocks": topk}
    raise ValueError(policy)


@torch.no_grad()
def eval_one(model: GPT2KRI, val_loader, policy: str, params: dict,
             n_batches: int, device, dtype, block_size: int) -> dict:
    """Run one (model, policy, params) sweep and return summary stats."""
    H = model.cfg.n_head
    losses = []
    n_tokens = 0
    retained_acc = 0.0
    retained_count = 0
    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        B, T = ids.shape
        if policy == "full":
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
                _, loss = model(ids, labels=labels)
        else:
            # If we need KRI's K/V scoring, collect from model first.
            need_kv = policy == "kri" and params["topk_blocks"] > 0
            kvs = model.collect_kv(ids) if need_kv else None
            if need_kv:
                k_per = [kv[0] for kv in kvs]
                v_per = [kv[1] for kv in kvs]
                q_per = [kv[0] for kv in kvs]
            else:
                k_per = v_per = q_per = None
            mask, info = fixed_policy_mask(
                policy=policy, seq_len=T, batch_size=B, n_head=H,
                block_size=block_size,
                local_window_tokens=params["local_window_tokens"],
                sink_blocks=params["sink_blocks"],
                topk_blocks=params["topk_blocks"],
                device=device,
                k_per_layer=k_per, v_per_layer=v_per, q_per_layer=q_per,
            )
            retained_acc += info["retained_per_query_avg"]
            retained_count += 1
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
                _, loss = model(ids, labels=labels, attn_mask=mask)
        losses.append(float(loss.item()) * (B * (T - 1)))
        n_tokens += B * (T - 1)
    avg_ce = sum(losses) / max(1, n_tokens)
    ppl = math.exp(avg_ce) if avg_ce < 30 else float("inf")
    # `retained_per_query_avg` is the mean of `mask.sum(-1).mean()`, i.e.
    # the average number of attended key positions per query. Under a
    # purely causal mask of length T, this is ~T/2 (the lower-triangle
    # average). Normalize by that to get a "fraction of dense" number.
    seq_len_eval = model.cfg.n_positions  # we always run at full seq_len here
    dense_avg = seq_len_eval / 2.0
    if policy == "full":
        retained_per_q = dense_avg
    else:
        retained_per_q = retained_acc / max(1, retained_count)
    retained_frac = retained_per_q / dense_avg
    return {
        "cross_entropy": avg_ce,
        "perplexity": ppl,
        "n_eval_tokens": n_tokens,
        "retained_per_query_avg": retained_per_q,
        "retained_fraction_of_dense": retained_frac,
    }


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    report_device()
    device = pick_device()
    dtype = pick_dtype(args.precision)

    out_jsonl = Path(args.output)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_csv = out_jsonl.with_suffix(".csv")

    print(f"output jsonl: {out_jsonl}")
    print(f"output csv  : {out_csv}")

    data_cfg = DataConfig(
        dataset_name=args.dataset_name, dataset_config=args.dataset_config,
        text_column=args.text_column, streaming=_truthy(args.streaming),
        train_split=args.train_split, val_split=args.val_split,
        seq_len=args.seq_len,
    )
    tok = get_tokenizer(args.tokenizer_name)
    _, val_ds = get_train_val_streams(data_cfg, tok)

    policies = args.policies.split(",")
    retention_fracs = _csv_floats(args.retention_fracs)
    models = args.models.split(",")

    rows = []
    for m_path in models:
        print(f"\n=== model: {m_path} ===")
        model, tag = load_model(m_path, device)
        # Fresh loader for each model so we hit deterministic batches.
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate, num_workers=0)
        # Cache the first n_batches of val into memory so we evaluate every
        # (policy, budget) on the SAME tokens.
        cached = []
        for i, b in enumerate(val_loader):
            if i >= args.n_batches:
                break
            cached.append(b)
        print(f"  cached {len(cached)} val batches")

        for policy in policies:
            if policy == "full":
                params = retention_to_policy_params(1.0, args.seq_len, args.block_size, args.sink_blocks, "full")
                res = eval_one(model, cached, "full", params, args.n_batches, device, dtype, args.block_size)
                row = {
                    "model": tag, "policy": "full", "retention_target": 1.0,
                    **params, **res,
                }
                print(f"  full: CE={res['cross_entropy']:.4f} PPL={res['perplexity']:.2f} kept={res['retained_fraction_of_dense']:.3f}")
                rows.append(row)
                continue
            for frac in retention_fracs:
                if frac >= 0.999:
                    continue  # covered by full
                params = retention_to_policy_params(frac, args.seq_len, args.block_size, args.sink_blocks, policy)
                res = eval_one(model, cached, policy, params, args.n_batches, device, dtype, args.block_size)
                row = {
                    "model": tag, "policy": policy, "retention_target": frac,
                    **params, **res,
                }
                print(
                    f"  {policy:12s} frac={frac:.4f} "
                    f"local_W={params['local_window_tokens']:4d} topk={params['topk_blocks']:3d} "
                    f"sinkB={params['sink_blocks']} "
                    f"CE={res['cross_entropy']:.4f} PPL={res['perplexity']:.2f} kept={res['retained_fraction_of_dense']:.3f}"
                )
                rows.append(row)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Write outputs.
    with out_jsonl.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    if rows:
        with out_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"\nwrote {len(rows)} rows -> {out_jsonl} and {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

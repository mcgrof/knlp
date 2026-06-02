"""Synthetic key=value retrieval eval scored by logprob ranking.

We construct a base-LM-friendly example:

    <doc>
    key_001 = blue fox
    key_002 = red moon
    ...
    key_099 = silver river
    </doc>
    lookup key_042 =

The model is then expected to assign higher logprob to the *correct*
value (e.g., "silver river") than to a set of distractor values drawn
from other keys in the same document.

We vary:
  - number of key-value pairs
  - distance between key and query (where in the doc the key was placed)
  - pruning policy
  - retention budget

The score is the rank (1 = correct best) of the gold value among the
distractor values. We report mean rank, top-1 accuracy, and the gold
logprob minus the best distractor logprob.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import get_tokenizer  # noqa: E402
from src.eval_pruned_ppl import load_model, retention_to_policy_params  # noqa: E402
from src.kri_mask import fixed_policy_mask  # noqa: E402
from src.model_gpt2_kri import GPT2KRI  # noqa: E402
from src.utils import pick_device, pick_dtype, report_device, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=str, required=True)
    p.add_argument("--num_examples", type=int, default=200)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--block_size", type=int, default=16)
    p.add_argument("--policies", type=str, default="full,recent,sink_recent,kri")
    p.add_argument("--retention_fracs", type=str, default="1.0,0.5,0.25,0.125")
    p.add_argument("--sink_blocks", type=int, default=1)
    p.add_argument("--n_distractors", type=int, default=8)
    p.add_argument("--num_pairs_choices", type=str, default="32,64,96")
    p.add_argument("--key_position", type=str, default="random",
                   choices=["random", "early", "middle", "late"])
    p.add_argument("--precision", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# A small fixed vocabulary so the tokenization is short and predictable.
# We use simple two-token noun-modifier pairs.
ADJECTIVES = [
    "blue", "red", "silver", "golden", "rusty", "ancient", "tiny", "huge",
    "quiet", "loud", "soft", "bright", "dark", "warm", "cold", "sharp",
    "smooth", "rough", "fresh", "stale", "new", "old", "lone", "happy",
]
NOUNS = [
    "fox", "moon", "river", "stone", "lantern", "feather", "dragon",
    "kettle", "candle", "garden", "harbor", "anchor", "ladder", "compass",
    "shadow", "valley", "mountain", "branch", "island", "marble", "willow",
    "snowflake", "harvest", "ember",
]


def gen_pair_value(rng: random.Random) -> str:
    return f"{rng.choice(ADJECTIVES)} {rng.choice(NOUNS)}"


def make_example(rng: random.Random, num_pairs: int, key_position: str,
                 n_distractors: int) -> dict:
    """Build one synthetic example. Returns:

        prefix : the doc + "lookup key_XYZ = " (string, no answer)
        gold   : the gold answer string (" blue fox")
        distractors: list of distractor answer strings
        key_index: index of the queried key in the doc (0-based)
    """
    pairs = []
    seen_values = set()
    while len(pairs) < num_pairs:
        v = gen_pair_value(rng)
        if v in seen_values:
            continue
        seen_values.add(v)
        pairs.append(v)
    # which key is queried?
    if key_position == "early":
        qi = rng.randint(0, max(0, num_pairs // 4))
    elif key_position == "middle":
        qi = rng.randint(num_pairs // 3, 2 * num_pairs // 3)
    elif key_position == "late":
        qi = rng.randint(3 * num_pairs // 4, num_pairs - 1)
    else:
        qi = rng.randint(0, num_pairs - 1)

    lines = []
    lines.append("<doc>")
    for i, v in enumerate(pairs):
        lines.append(f"key_{i:03d} = {v}")
    lines.append("</doc>")
    prefix = "\n".join(lines) + f"\nlookup key_{qi:03d} ="

    gold = pairs[qi]
    # distractors: pick from other pairs
    others = [v for j, v in enumerate(pairs) if j != qi]
    rng.shuffle(others)
    distractors = others[:n_distractors]
    return {"prefix": prefix, "gold": gold, "distractors": distractors, "key_index": qi}


@torch.no_grad()
def logprob_of_continuation(model: GPT2KRI, tok, prefix: str, continuation: str,
                           device: torch.device, dtype: torch.dtype, seq_len: int,
                           policy: str, params: dict, block_size: int) -> Tuple[float, dict]:
    """Return summed logprob of `continuation` tokens given `prefix`.

    Builds the full input as [prefix + " " + continuation] (continuation
    is preceded by a space so token boundaries are clean), feeds it
    through the model once, then sums log-softmax at the continuation
    positions of the gold tokens.

    The same attention policy is applied to BOTH prefix and continuation;
    that mirrors how a real KV-cache server would prune.
    """
    full_text = f"{prefix} {continuation}".strip()
    enc = tok(full_text, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"][:, :seq_len].to(device)
    # We also need to know how many tokens the prefix consumes.
    enc_prefix = tok(prefix, return_tensors="pt", add_special_tokens=False)
    pre_len = min(enc_prefix["input_ids"].size(1), ids.size(1))
    T = ids.size(1)
    if pre_len >= T:
        # No room for the continuation — return -inf
        return float("-inf"), {"retained_per_query_avg": 0.0, "n_cont_tokens": 0}
    B = 1
    H = model.cfg.n_head

    if policy == "full":
        mask = None
        # `retained_per_query_avg` matches the key emitted by
        # fixed_policy_mask. Under a dense causal mask, the average
        # number of attended keys per query is T/2.
        info = {"retained_per_query_avg": T / 2.0}
    else:
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

    with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
        logits, _ = model(ids, attn_mask=mask)
    log_probs = F.log_softmax(logits.float(), dim=-1)

    # Sum log-prob of continuation tokens.
    # logits[t] predicts token at t+1; for continuation tokens at
    # positions [pre_len, T-1], we sum log_probs[t-1, ids[t]] for t in
    # [pre_len, T-1].
    cont_positions = list(range(pre_len, T))
    if not cont_positions:
        return float("-inf"), info | {"n_cont_tokens": 0}
    summed = 0.0
    for t in cont_positions:
        summed += float(log_probs[0, t - 1, ids[0, t].item()].item())
    info["n_cont_tokens"] = len(cont_positions)
    return summed, info


def run_one_example(model, tok, ex, device, dtype, seq_len, block_size,
                    policy, params) -> dict:
    """Score gold + distractors and return ranking stats."""
    gold_lp, info = logprob_of_continuation(
        model, tok, ex["prefix"], ex["gold"], device, dtype, seq_len, policy, params, block_size,
    )
    distractor_lps = []
    for d in ex["distractors"]:
        lp, _ = logprob_of_continuation(
            model, tok, ex["prefix"], d, device, dtype, seq_len, policy, params, block_size,
        )
        distractor_lps.append(lp)
    # rank: 1 if gold beats all distractors
    all_lps = [gold_lp] + distractor_lps
    rank = 1 + sum(1 for x in distractor_lps if x > gold_lp)
    margin = gold_lp - max(distractor_lps) if distractor_lps else float("inf")
    return {
        "rank": rank,
        "gold_lp": gold_lp,
        "best_distractor_lp": max(distractor_lps) if distractor_lps else float("-inf"),
        "margin": margin,
        "top1": int(rank == 1),
        "retained_per_query_avg": info.get("retained_per_query_avg", 0.0),
        "n_cont_tokens": info.get("n_cont_tokens", 0),
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

    tok = get_tokenizer("openai-community/gpt2")
    rng = random.Random(args.seed)
    pair_choices = [int(x) for x in args.num_pairs_choices.split(",")]
    examples = []
    for i in range(args.num_examples):
        num_pairs = pair_choices[i % len(pair_choices)]
        examples.append(make_example(rng, num_pairs, args.key_position, args.n_distractors))
    print(f"generated {len(examples)} synthetic examples (num_pairs in {pair_choices})")

    policies = args.policies.split(",")
    retention_fracs = [float(x) for x in args.retention_fracs.split(",")]
    models = args.models.split(",")

    rows = []
    for m_path in models:
        print(f"\n=== model: {m_path} ===")
        model, tag = load_model(m_path, device)

        for policy in policies:
            if policy == "full":
                params = retention_to_policy_params(1.0, args.seq_len, args.block_size, args.sink_blocks, "full")
                stats = [run_one_example(model, tok, ex, device, dtype, args.seq_len, args.block_size, "full", params) for ex in examples]
                row = _aggregate(tag, "full", 1.0, params, stats)
                rows.append(row)
                _print(row)
                continue
            for frac in retention_fracs:
                if frac >= 0.999:
                    continue
                params = retention_to_policy_params(frac, args.seq_len, args.block_size, args.sink_blocks, policy)
                stats = [run_one_example(model, tok, ex, device, dtype, args.seq_len, args.block_size, policy, params) for ex in examples]
                row = _aggregate(tag, policy, frac, params, stats)
                rows.append(row)
                _print(row)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

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


def _aggregate(tag: str, policy: str, frac: float, params: dict, stats: List[dict]) -> dict:
    n = len(stats)
    top1 = sum(s["top1"] for s in stats) / n
    mean_rank = sum(s["rank"] for s in stats) / n
    mean_margin = sum(s["margin"] for s in stats) / n
    mean_gold_lp = sum(s["gold_lp"] for s in stats) / n
    retained_per_q = sum(s["retained_per_query_avg"] for s in stats) / n
    # Convert to "fraction of dense" using the eval seq_len.
    # Stats were computed at the configured seq_len; n_cont_tokens > 0.
    # We approximate the dense average as 0.5 * seq_len; ratio is
    # reported instead of the absolute count for easier reading.
    return {
        "model": tag,
        "policy": policy,
        "retention_target": frac,
        **params,
        "n_examples": n,
        "top1_accuracy": top1,
        "mean_rank": mean_rank,
        "mean_margin": mean_margin,
        "mean_gold_logprob": mean_gold_lp,
        "retained_per_query_avg": retained_per_q,
    }


def _print(row: dict) -> None:
    print(
        f"  {row['policy']:12s} frac={row['retention_target']:.4f} "
        f"local_W={row['local_window_tokens']:4d} topk={row['topk_blocks']:3d} "
        f"top1={row['top1_accuracy']:.3f} rank={row['mean_rank']:.2f} margin={row['mean_margin']:.2f}"
    )


if __name__ == "__main__":
    raise SystemExit(main())

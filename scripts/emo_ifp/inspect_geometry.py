#!/usr/bin/env python3
"""Summarize EMO tensor geometry from the BF16 conversion manifest (CPU).

Reports per-layer expert counts, which expert is shared, expert matrix
shapes, bytes per expert, the non-expert core, pool sizes for given routed
budgets, and the file layout of one layer's experts (needed to plan range
reads). No model code is executed.

Usage:
    inspect_geometry.py MODEL_DIR BF16_MANIFEST.json OUT.json [--layer 15]
"""

import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EXPERT_MATS, expert_tensor_name, write_json  # noqa

EXPERT_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(\w+)\.weight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("manifest")
    ap.add_argument("out")
    ap.add_argument("--layer", type=int, default=15)
    args = ap.parse_args()

    with open(os.path.join(args.model_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(args.manifest) as f:
        man = json.load(f)
    tensors = man["tensors"]

    experts = collections.defaultdict(set)
    expert_bytes = collections.Counter()
    shapes = {}
    core_bytes = 0
    for name, t in tensors.items():
        m = EXPERT_RE.fullmatch(name)
        if m:
            L, E, mat = int(m.group(1)), int(m.group(2)), m.group(3)
            experts[L].add(E)
            expert_bytes[(L, E)] += t["nbytes"]
            shapes.setdefault(mat, t["shape"])
        else:
            core_bytes += t["nbytes"]

    n_layers = cfg["num_hidden_layers"]
    n_exp = cfg["num_experts"]
    n_shared = cfg["num_shared_experts"]
    per_expert = sorted(set(expert_bytes.values()))
    assert len(per_expert) == 1, per_expert
    per_expert = per_expert[0]
    for L in range(n_layers):
        assert experts[L] == set(range(n_exp)), L

    def pool_bytes(routed):
        return core_bytes + n_layers * (routed + n_shared) * per_expert

    # One layer's expert layout in the BF16 files.
    L = args.layer
    layout = []
    for E in range(n_exp):
        for mat in EXPERT_MATS:
            t = tensors[expert_tensor_name(L, E, mat)]
            layout.append((t["shard"], t["file_offset"], t["nbytes"], E, mat))
    layout.sort()
    shards = sorted(set(x[0] for x in layout))
    lo = min(x[1] for x in layout)
    hi = max(x[1] + x[2] for x in layout)
    out = {
        "config": {
            k: cfg[k]
            for k in (
                "num_hidden_layers",
                "hidden_size",
                "intermediate_size",
                "num_experts",
                "num_shared_experts",
                "num_experts_per_tok",
                "vocab_size",
                "tie_word_embeddings",
                "norm_topk_prob",
            )
        },
        "routing_rule": (
            "legacy shared path: experts 0..%d routed (softmax over routed, "
            "top-%d); expert %d shared (own softmax, weight 1.0)"
            % (n_exp - n_shared - 1, cfg["num_experts_per_tok"] - n_shared, n_exp - 1)
        ),
        "shared_expert_ids": list(range(n_exp - n_shared, n_exp)),
        "expert_matrix_shapes": shapes,
        "bytes_per_expert": per_expert,
        "core_nonexpert_bytes": core_bytes,
        "total_bytes": core_bytes + n_layers * n_exp * per_expert,
        "pool_bytes": {
            f"{r}_routed_plus_{n_shared}_shared": pool_bytes(r)
            for r in (n_exp - n_shared, 64, 32, 16)
        },
        "layer_layout": {
            "layer": L,
            "shards": shards,
            "first_offset": lo,
            "end_offset": hi,
            "span_bytes": hi - lo,
            "payload_bytes": sum(x[2] for x in layout),
            "contiguous": hi - lo == sum(x[2] for x in layout),
            "order_head": [f"{x[3]}.{x[4]}" for x in layout[:6]],
        },
    }
    write_json(args.out, out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Translate the knlp .config into the JSON the matched-micro driver consumes,
so no experiment policy lives in shell or Python constants. Reads
CONFIG_MATCHED_MICRO_* keys from .config (path via --config, default
../../.config) and writes config.json."""

import argparse
import json
import os
import re

KEYS = {
    "CONFIG_MATCHED_MICRO": ("enabled", bool),
    "CONFIG_MATCHED_MICRO_ARMS": ("arms", str),
    "CONFIG_MATCHED_MICRO_PHASE_BATCH_PROBE": ("phase_batch_probe", bool),
    "CONFIG_MATCHED_MICRO_PROBE_BATCHES": ("probe_batches", str),
    "CONFIG_MATCHED_MICRO_PROBE_STEPS": ("probe_steps", int),
    "CONFIG_MATCHED_MICRO_DATA_TOKENS": ("data_tokens", int),
    "CONFIG_MATCHED_MICRO_PHASE_CAMPAIGN": ("phase_campaign", bool),
    "CONFIG_MATCHED_MICRO_CAMPAIGN_SEEDS": ("campaign_seeds", str),
    "CONFIG_MATCHED_MICRO_CAMPAIGN_BATCH": ("campaign_batch", int),
    "CONFIG_MATCHED_MICRO_CAMPAIGN_TOKEN_BUDGET": ("campaign_token_budget", int),
    "CONFIG_MATCHED_MICRO_CAMPAIGN_EVAL_EVERY": ("campaign_eval_every", int),
    "CONFIG_MATCHED_MICRO_PHASE_RANK_EVAL": ("phase_rank_eval", bool),
    "CONFIG_MATCHED_MICRO_RANK_SEED": ("rank_seed", int),
    "CONFIG_MATCHED_MICRO_RANK_PER_FAMILY": ("rank_per_family", int),
    "CONFIG_MATCHED_MICRO_RANK_CONTROLS": ("rank_controls", str),
    "CONFIG_MATCHED_MICRO_RANK_NORM": ("rank_norm", str),
    "CONFIG_MATCHED_MICRO_RANK_MAX_TOKENS": ("rank_max_tokens", int),
    "CONFIG_MATCHED_MICRO_RANK_SHARDS": ("rank_shards", int),
    "CONFIG_MATCHED_MICRO_RANK_TAG": ("rank_tag", str),
    "CONFIG_MATCHED_MICRO_RANK_NO_MEMORY_UPDATE": ("rank_no_memory_update", bool),
}


def parse(path):
    cfg = {}
    line_re = re.compile(r"^(CONFIG_[A-Z0-9_]+)=(.*)$")
    for line in open(path):
        m = line_re.match(line.strip())
        if not m:
            continue
        k, v = m.group(1), m.group(2)
        if k not in KEYS:
            continue
        name, typ = KEYS[k]
        v = v.strip()
        if typ is str:
            cfg[name] = v.strip('"').replace('\\"', '"')
        elif typ is int:
            cfg[name] = int(v)
        elif typ is bool:
            cfg[name] = v == "y"
    return cfg


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--config", default=os.path.join(here, "..", "..", ".config"))
    ap.add_argument("--out", default=os.path.join(here, "config.json"))
    a = ap.parse_args()
    cfg = parse(a.config)
    # defaults so the driver runs even from a partial .config
    cfg.setdefault("enabled", False)
    cfg.setdefault("arms", "hope")
    cfg.setdefault("phase_batch_probe", False)
    cfg.setdefault("probe_batches", "8 32 64 128")
    cfg.setdefault("probe_steps", 10)
    cfg.setdefault("data_tokens", 40_000_000)
    cfg.setdefault("phase_campaign", False)
    cfg.setdefault("campaign_seeds", "1234 2027 31337")
    cfg.setdefault("campaign_batch", 128)
    cfg.setdefault("campaign_token_budget", 40_000_000)
    cfg.setdefault("campaign_eval_every", 150)
    cfg.setdefault("phase_rank_eval", False)
    cfg.setdefault("rank_seed", 7)
    cfg.setdefault("rank_per_family", 25)
    cfg.setdefault("rank_controls", '{"filler_sentences": 2}')
    cfg.setdefault("rank_norm", "mean")
    cfg.setdefault("rank_max_tokens", 512)
    cfg.setdefault("rank_shards", 1)
    cfg.setdefault("rank_tag", "rank-eval")
    cfg.setdefault("rank_no_memory_update", False)
    json.dump(cfg, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}:")
    print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()

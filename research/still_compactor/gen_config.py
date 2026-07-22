#!/usr/bin/env python3
"""Translate the knlp .config into a JSON the STILL driver consumes, so no
experiment policy lives in shell or Python constants. Reads CONFIG_STILL_* keys
from .config (path via --config, default ../../.config) and writes config.json."""
import argparse, json, os, re

EXP = {
    "CONFIG_STILL_EXP_KERNEL": "kernel",
    "CONFIG_STILL_EXP_LEDGER": "ledger",
    "CONFIG_STILL_EXP_BASELINES": "baselines",
    "CONFIG_STILL_EXP_CHUNKED": "chunked",
    "CONFIG_STILL_EXP_CONCURRENCY": "concurrency",
    "CONFIG_STILL_EXP_IO": "io",
    "CONFIG_STILL_EXP_LADDER": "ladder",
}
KEYS = {
    "CONFIG_STILL_MODEL": ("model", str),
    "CONFIG_STILL_CTX_TOKENS": ("ctx_tokens", int),
    "CONFIG_STILL_CHUNK": ("chunk", int),
    "CONFIG_STILL_T_COMPACT": ("t_compact", int),
    "CONFIG_STILL_N_TRAIN": ("n_train", int),
    "CONFIG_STILL_N_EVAL": ("n_eval", int),
    "CONFIG_STILL_EPOCHS": ("epochs", int),
    "CONFIG_STILL_BATCH": ("batch", int),
    "CONFIG_STILL_SEED": ("seed", int),
}


def parse(path):
    cfg = {}
    line_re = re.compile(r'^(CONFIG_[A-Z0-9_]+)=(.*)$')
    for line in open(path):
        m = line_re.match(line.strip())
        if not m:
            continue
        k, v = m.group(1), m.group(2).strip()
        if k in EXP and v == "y":
            cfg["experiment"] = EXP[k]
        elif k in KEYS:
            name, typ = KEYS[k]
            cfg[name] = v.strip('"') if typ is str else int(v)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--config", default=os.path.join(here, "..", "..", ".config"))
    ap.add_argument("--out", default=os.path.join(here, "config.json"))
    a = ap.parse_args()
    cfg = parse(a.config)
    cfg.setdefault("experiment", "kernel")
    cfg.setdefault("model", "Qwen/Qwen3-4B")
    cfg.setdefault("ctx_tokens", 512)
    cfg.setdefault("chunk", 2048)
    cfg.setdefault("t_compact", 64)
    cfg.setdefault("n_train", 2000)
    cfg.setdefault("n_eval", 512)
    cfg.setdefault("epochs", 4)
    cfg.setdefault("batch", 16)
    cfg.setdefault("seed", 0)
    json.dump(cfg, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}:")
    print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()

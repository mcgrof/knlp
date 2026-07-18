#!/usr/bin/env python3
"""Translate the knlp .config into a JSON the CAS driver consumes, so no experiment
policy lives in shell or Python constants. Reads CONFIG_CARTRIDGES_CAS_* keys from
.config (path via --config, default ../../.config) and writes config.json."""
import argparse, json, os, re

KEYS = {
    "CONFIG_CARTRIDGES_CAS_MODEL": ("model", str),
    "CONFIG_CARTRIDGES_CAS_NUM_PATIENTS": ("num_patients", int),
    "CONFIG_CARTRIDGES_CAS_CONVOS_PER_PATIENT": ("convos_per_patient", int),
    "CONFIG_CARTRIDGES_CAS_MIN_PROB_MASS": ("min_prob_mass", float),
    "CONFIG_CARTRIDGES_CAS_KV_TOKENS": ("kv_tokens", int),
    "CONFIG_CARTRIDGES_CAS_LR": ("lr", str),
    "CONFIG_CARTRIDGES_CAS_GLOBAL_BATCH": ("global_batch", int),
    "CONFIG_CARTRIDGES_CAS_STEPS": ("steps", int),
    "CONFIG_CARTRIDGES_CAS_EPOCHS": ("epochs", int),
    "CONFIG_CARTRIDGES_CAS_COMPILE_FLEX": ("compile_flex", bool),
    "CONFIG_CARTRIDGES_CAS_PHASE_SYNTH": ("phase_synth", bool),
    "CONFIG_CARTRIDGES_CAS_PHASE_TRAIN_ISOLATED": ("phase_train_isolated", bool),
    "CONFIG_CARTRIDGES_CAS_PHASE_COLLAPSE": ("phase_collapse", bool),
    "CONFIG_CARTRIDGES_CAS_PHASE_TRAIN_JOINT": ("phase_train_joint", bool),
    "CONFIG_CARTRIDGES_CAS_PHASE_RESCUE": ("phase_rescue", bool),
}


def parse(path):
    cfg = {}
    line_re = re.compile(r'^(CONFIG_[A-Z0-9_]+)=(.*)$')
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
            cfg[name] = v.strip('"')
        elif typ is float:
            cfg[name] = float(v.strip('"'))
        elif typ is int:
            cfg[name] = int(v)
        elif typ is bool:
            cfg[name] = (v == "y")
    return cfg


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--config", default=os.path.join(here, "..", "..", ".config"))
    ap.add_argument("--out", default=os.path.join(here, "config.json"))
    a = ap.parse_args()
    cfg = parse(a.config)
    # defaults so the driver runs even from a partial .config
    cfg.setdefault("model", "Qwen/Qwen3-8B")
    cfg.setdefault("num_patients", 5)
    cfg.setdefault("convos_per_patient", 8000)
    json.dump(cfg, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}:")
    print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()

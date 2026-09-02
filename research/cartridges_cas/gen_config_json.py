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
    "CONFIG_CARTRIDGES_CAS_PHASE_OPT_ABLATION": ("phase_opt_ablation", bool),
    "CONFIG_CARTRIDGES_CAS_PHASE_CONTROL_SCREEN": ("phase_control_screen", bool),
    "CONFIG_CARTRIDGES_CAS_PHASE_PAPER_REGIME": ("phase_paper_regime", bool),
    "CONFIG_CARTRIDGES_CAS_PAPER_PATIENTS": ("paper_patients", str),
    "CONFIG_CARTRIDGES_CAS_PAPER_KV_DIVISOR": ("paper_kv_divisor", int),
    "CONFIG_CARTRIDGES_CAS_PAPER_EVAL_RUNS": ("paper_eval_runs", int),
    "CONFIG_CARTRIDGES_CAS_CTRL_ARMS": ("ctrl_arms", str),
    "CONFIG_CARTRIDGES_CAS_CTRL_PATIENT": ("ctrl_patient", str),
    "CONFIG_CARTRIDGES_CAS_CTRL_STEPS": ("ctrl_steps", int),
    "CONFIG_CARTRIDGES_CAS_CTRL_ACCUM": ("ctrl_accum", int),
    "CONFIG_CARTRIDGES_CAS_CTRL_LR": ("ctrl_lr", str),
    "CONFIG_CARTRIDGES_CAS_CTRL_SEED": ("ctrl_seed", int),
    "CONFIG_CARTRIDGES_CAS_CTRL_CHECKPOINT_AT": ("ctrl_checkpoint_at", str),
    "CONFIG_CARTRIDGES_CAS_CTRL_MAX_Q": ("ctrl_max_q", int),
    "CONFIG_CARTRIDGES_CAS_CTRL_PROBE_N": ("ctrl_probe_n", int),
    "CONFIG_CARTRIDGES_CAS_OPT_ARMS": ("opt_arms", str),
    "CONFIG_CARTRIDGES_CAS_OPT_PATIENT": ("opt_patient", str),
    "CONFIG_CARTRIDGES_CAS_OPT_STEPS": ("opt_steps", int),
    "CONFIG_CARTRIDGES_CAS_OPT_ACCUM": ("opt_accum", int),
    "CONFIG_CARTRIDGES_CAS_OPT_LR": ("opt_lr", str),
    "CONFIG_CARTRIDGES_CAS_OPT_SEED": ("opt_seed", int),
    "CONFIG_CARTRIDGES_CAS_OPT_CHECKPOINT_AT": ("opt_checkpoint_at", str),
    "CONFIG_CARTRIDGES_CAS_OPT_SOAP_PRECOND_FREQ": ("opt_soap_precond_freq", int),
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
            cfg[name] = v.strip('"')
        elif typ is float:
            cfg[name] = float(v.strip('"'))
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
    cfg.setdefault("model", "Qwen/Qwen3-8B")
    cfg.setdefault("num_patients", 5)
    cfg.setdefault("convos_per_patient", 8000)
    json.dump(cfg, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}:")
    print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()

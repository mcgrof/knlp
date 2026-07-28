#!/usr/bin/env python3
"""Build an UNTRAINED full-document KV "cartridge" per patient — the cache-path
parity control.

The reproduction's full-document baseline (~0.85 on LongHealth) runs through
vLLM, but the trained cartridge runs through the custom FlexQwen3 / TrainableCache
path. So the 0.85 anchor does NOT validate the cartridge execution path. This
builds, for each patient, a cartridge whose KV IS the exact prefill of the full
record (via the library's own KVFromText, which forward-passes the document with
position_ids=arange and captures real K/V) — no compression, no training. Eval it
through the UNCHANGED cartridge evaluator (cas_eval_matched.py):

  * if the full-doc cartridge reaches ~0.85  -> the cartridge cache/geometry path
    is lossless; the 0.55 gap is genuinely upstream (synthesis/init/training).
  * if it lands ~0.55                          -> the cartridge PATH itself is
    lossy (positional/RoPE/serialization); synthesis is not the culprit and
    re-synthesis would be fixing the wrong thing.

Uses the same {trainable_keys,trainable_values,frozen_keys,frozen_values} dict
format as a trained cart, so the evaluator loads it identically. The document is
"\n\n".join(patient.texts) — the exact text the vLLM full-doc baseline used.

Env: PATIENTS, OUT (cart dir), MAXTOK (cap on doc tokens; default 16384 covers
the longest LongHealth record), DEVICE.
"""

import os
import sys
import tempfile

os.environ.setdefault("CARTRIDGES_DIR", "/home/mcgrof/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/home/mcgrof/cas_out")
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
sys.path.insert(0, os.environ["CARTRIDGES_DIR"])

import torch
from transformers import AutoTokenizer

from cartridges.cache import AttnConfig
from cartridges.initialization.text import KVFromText
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
from cartridges.data.longhealth.utils import load_longhealth_dataset

PATIENTS = os.environ.get(
    "PATIENTS", "patient_01 patient_02 patient_03 patient_05 patient_06"
).split()
OUT = os.environ.get("OUT", "/home/mcgrof/cas_out/fulldoc_carts")
MAXTOK = int(os.environ.get("MAXTOK", "16384"))
DEVICE = os.environ.get("DEVICE", "cuda:0")
MODEL = "Qwen/Qwen3-8B"

os.makedirs(OUT, exist_ok=True)
di = int(DEVICE.split(":")[1]) if ":" in DEVICE else 0
torch.cuda.set_device(di)

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = FlexQwen3ForCausalLM.from_pretrained(MODEL).to(DEVICE).to(torch.bfloat16)
model.eval()
ac = AttnConfig(
    n_layers=model.config.num_hidden_layers,
    n_heads=model.config.num_key_value_heads,
    head_dim=getattr(
        model.config,
        "head_dim",
        model.config.hidden_size // model.config.num_attention_heads,
    ),
)

patients = load_longhealth_dataset(PATIENTS)
for p in patients:
    fulltext = "\n\n".join(p.texts.values())
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(fulltext)
        tmp = f.name
    try:
        init = KVFromText.Config(
            max_tokens=MAXTOK, text_source=tmp, num_frozen_tokens=0
        ).instantiate()
        cache = init.initialize_kv_cache(tok, model, ac)
        ntok = cache.trainable_keys[0].shape[2] if len(cache.trainable_keys) else 0
        nfrozen = (
            cache.frozen_keys[0].shape[2]
            if cache.frozen_keys is not None and len(cache.frozen_keys)
            else 0
        )
        torch.save(
            {
                "trainable_keys": [k.detach().cpu() for k in cache.trainable_keys],
                "trainable_values": [v.detach().cpu() for v in cache.trainable_values],
                "frozen_keys": (
                    [k.detach().cpu() for k in cache.frozen_keys]
                    if cache.frozen_keys is not None
                    else []
                ),
                "frozen_values": (
                    [v.detach().cpu() for v in cache.frozen_values]
                    if cache.frozen_values is not None
                    else []
                ),
            },
            os.path.join(OUT, f"{p.patient_id}.pt"),
        )
        print(
            f"built {p.patient_id}: doc-KV trainable_tokens={ntok} frozen={nfrozen}",
            flush=True,
        )
    finally:
        os.unlink(tmp)
    del cache
    torch.cuda.empty_cache()

print("FULLDOC_CARTS_DONE ->", OUT, flush=True)

#!/usr/bin/env python3
"""CAS smoke -- ISOLATED cartridge training (the 'split' step).
Trains ONE trainable KV-cache cartridge for ONE LongHealth patient on that patient's
own self-study corpus (context distillation via captured top-20 teacher logprobs = the
KL(teacher||student) rule). Cart-specific truncation init from the patient's record.
Saves cache-<patient>.pt for later combine-at-inference (gap #1) and joint training (gap #2).

Env:
  PATIENT       patient id (e.g. patient_01)
  DATA_PARQUET  path to that patient's synth dataset.parquet
  KV_TOKENS     cartridge size in tokens (default 1024)
  STEPS         max optimizer steps (default 300)
  LR            learning rate (default 2e-2)
  EPOCHS        epochs (default 4)
  OUT_DIR       where to write the cartridge (default /root/cart_out)
Single-process (NO torchrun) -> is_ddp False -> no gloo teardown hang."""
import os
os.environ.setdefault("CARTRIDGES_DIR", os.environ.get("CARTRIDGES_DIR", "/root/cartridges"))
OUT_DIR = os.environ.get("OUT_DIR", "/root/cart_out")
os.environ["CARTRIDGES_OUTPUT_DIR"] = OUT_DIR
os.environ["WANDB_DISABLED"] = "true"; os.environ["WANDB_MODE"] = "disabled"

from cartridges.initialization import KVFromText
from cartridges.train import GenerationEvalConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import TrainDataset, DataSource
from cartridges.data.longhealth.evals import LongHealthMultipleChoiceGenerateDataset
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
from cartridges.utils.wandb import WandBConfig
from cartridges.train import CosWithWarmup

PATIENT = os.environ["PATIENT"]
DATA_PARQUET = os.environ["DATA_PARQUET"]
KV_TOKENS = int(os.environ.get("KV_TOKENS", "1024"))
STEPS = int(os.environ.get("STEPS", "300"))
LR = float(os.environ.get("LR", "2e-2"))
EPOCHS = int(os.environ.get("EPOCHS", "4"))
GLOBAL_BS = int(os.environ.get("GLOBAL_BS", "16"))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "0"))
WARMUP_MIN_LR = float(os.environ.get("WARMUP_MIN_LR", "1e-4"))
REC = f"{os.environ.get('RECORDS_DIR', '/root/cart_records')}/{PATIENT}.txt"

config = TrainConfig(
    model=HFModelConfig(
        pretrained_model_name_or_path="Qwen/Qwen3-8B",
        model_cls=FlexQwen3ForCausalLM,
    ),
    kv_cache_initializer=KVFromText.Config(max_tokens=KV_TOKENS, text_source=REC),
    lr=LR,
    lr_scheduler=(CosWithWarmup.Config(warmup_steps=WARMUP_STEPS, warmup_min_lr=WARMUP_MIN_LR, max_steps=STEPS)
                  if WARMUP_STEPS > 0 else None),
    epochs=EPOCHS,
    global_batch_size=GLOBAL_BS,
    max_optimizer_steps=STEPS,
    dataset=TrainDataset.Config(
        data_sources=[DataSource(path=DATA_PARQUET, type="local")],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ),
    save_every_n_steps=None,
    save_after_training=True,          # persist the cartridge
    keep_last_n_saved=1,
    save_to_wandb=False,
    generate_before_training=False,    # skip -- oracle eval done separately in combine step
    generate_evals=[],                 # keep training lean; eval in cas_combine_eval.py
    distributed_backend="gloo",
    wandb=WandBConfig(tags=["cas", "isolated", PATIENT]),
    output_dir=OUT_DIR,
    name=f"cas_iso_{PATIENT}",
    seed=42,
)

if __name__ == "__main__":
    # We invoke config.run() directly (not pydrantic.main), so run_dir is unset ->
    # save_cache() would crash. Set a deterministic run_dir before training.
    import os as _os
    config.run_dir = _os.path.join(OUT_DIR, "runs", config.name)
    _os.makedirs(config.run_dir, exist_ok=True)
    config.run()
    # Copy the saved cartridge to a deterministic path for combine/joint steps.
    import glob, shutil
    from pathlib import Path
    carts = Path(OUT_DIR) / "carts"; carts.mkdir(parents=True, exist_ok=True)
    dst = carts / f"{PATIENT}.pt"
    src = None
    rd = getattr(config, "run_dir", None)
    if rd and (Path(rd) / "cache_last.pt").exists():
        src = str(Path(rd) / "cache_last.pt")
    if src is None:
        # fallback: newest cache-step*.pt anywhere under OUT_DIR mentioning this patient
        cands = glob.glob(f"{OUT_DIR}/**/cache*.pt", recursive=True)
        cands = [c for c in cands if PATIENT in c] or cands
        if cands:
            src = max(cands, key=os.path.getmtime)
    assert src is not None, f"no saved cartridge found for {PATIENT} under {OUT_DIR}"
    shutil.copyfile(os.path.realpath(src), dst)
    print(f"CAS_ISO_DONE {PATIENT} cart={dst} from={src}")

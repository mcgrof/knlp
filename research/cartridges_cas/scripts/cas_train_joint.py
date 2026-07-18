#!/usr/bin/env python3
"""CAS smoke -- gap #2: JOINT / mixed-visibility cartridge training (the rescue).
Retrains the target patient's cartridge with the other patients' (isolated-trained)
cartridges present as FROZEN, globally-visible distractors -- the co-loaded condition
that collapses isolated cartridges. Saves the rescued target to <OUT_DIR>/carts_joint/<p>.pt.

Env: PATIENT, DATA_PARQUET, DISTRACTORS (space-sep patient ids), ISO_CART_DIR,
     KV_TOKENS, STEPS, LR, EPOCHS, OUT_DIR."""
import os
os.environ.setdefault("CARTRIDGES_DIR", os.environ.get("CARTRIDGES_DIR", "/root/cartridges"))
OUT_DIR = os.environ.get("OUT_DIR", "/root/cart_out")
os.environ["CARTRIDGES_OUTPUT_DIR"] = OUT_DIR
os.environ["WANDB_DISABLED"] = "true"; os.environ["WANDB_MODE"] = "disabled"

from cartridges.initialization.cas_joint import KVFromCarts
from cartridges.train import TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import TrainDataset, DataSource
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
from cartridges.utils.wandb import WandBConfig

PATIENT = os.environ["PATIENT"]
DATA_PARQUET = os.environ["DATA_PARQUET"]
DISTRACTORS = os.environ.get("DISTRACTORS", "").split()
ISO_CART_DIR = os.environ.get("ISO_CART_DIR", "/root/cart_out/carts")
KV_TOKENS = int(os.environ.get("KV_TOKENS", "1024"))
STEPS = int(os.environ.get("STEPS", "300"))
LR = float(os.environ.get("LR", "2e-2"))
EPOCHS = int(os.environ.get("EPOCHS", "4"))
REC = f"{os.environ.get('RECORDS_DIR', '/root/cart_records')}/{PATIENT}.txt"
dpaths = [os.path.join(ISO_CART_DIR, f"{d}.pt") for d in DISTRACTORS]

config = TrainConfig(
    model=HFModelConfig(pretrained_model_name_or_path="Qwen/Qwen3-8B",
                        model_cls=FlexQwen3ForCausalLM),
    kv_cache_initializer=KVFromCarts.Config(
        target_text_source=REC, target_max_tokens=KV_TOKENS,
        distractor_paths=dpaths, num_frozen_tokens=0,  # frozen count comes from distractors
    ),
    lr=LR, epochs=EPOCHS, global_batch_size=16, max_optimizer_steps=STEPS,
    dataset=TrainDataset.Config(
        data_sources=[DataSource(path=DATA_PARQUET, type="local")],
        top_k_logits=20, packed_seq_length=2048, packing_mode="truncate",
    ),
    save_every_n_steps=None, save_after_training=True, keep_last_n_saved=1,
    save_to_wandb=False, generate_before_training=False, generate_evals=[],
    distributed_backend="gloo",
    wandb=WandBConfig(tags=["cas", "joint", PATIENT]),
    output_dir=OUT_DIR, name=f"cas_joint_{PATIENT}", seed=42,
)

if __name__ == "__main__":
    import os as _os
    config.run_dir = _os.path.join(OUT_DIR, "runs", config.name)
    _os.makedirs(config.run_dir, exist_ok=True)
    config.run()
    import glob, shutil
    from pathlib import Path
    carts = Path(OUT_DIR) / "carts_joint"; carts.mkdir(parents=True, exist_ok=True)
    dst = carts / f"{PATIENT}.pt"
    src = None
    rd = getattr(config, "run_dir", None)
    if rd and (Path(rd) / "cache_last.pt").exists():
        src = str(Path(rd) / "cache_last.pt")
    if src is None:
        cands = [c for c in glob.glob(f"{OUT_DIR}/**/cache*.pt", recursive=True) if "cas_joint" in c and PATIENT in c]
        cands = cands or [c for c in glob.glob(f"{OUT_DIR}/**/cache*.pt", recursive=True) if PATIENT in c]
        if cands:
            src = max(cands, key=os.path.getmtime)
    assert src is not None, f"no saved joint cartridge for {PATIENT}"
    shutil.copyfile(os.path.realpath(src), dst)
    print(f"CAS_JOINT_DONE {PATIENT} cart={dst} from={src}")

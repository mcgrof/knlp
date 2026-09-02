#!/usr/bin/env python3
"""CAS -- ISOLATED cartridge training (the 'split' step).
Trains ONE trainable KV-cache cartridge for ONE LongHealth patient on that patient's
own self-study corpus (context distillation via captured top-20 teacher logprobs = the
KL(teacher||student) rule). Cart-specific truncation init from the patient's record.
Saves cache-<patient>.pt for later combine-at-inference (gap #1) and joint training (gap #2).

Env:
  PATIENT       patient id (e.g. patient_01)
  DATA_PARQUET  path to that patient's synth dataset.parquet
  KV_TOKENS     cartridge size in tokens; integer, or "auto" for the paper's
                per-document size = ceil(doc_tokens / 20) (default 1024)
  KV_DIVISOR    divisor for "auto" sizing (default 20, the paper's ratio)
  KV_MIN/KV_MAX clamp for "auto" sizing (default 256 / 2048)
  STEPS         stop after this many optimizer steps (default 300), or at
                EPOCHS, whichever comes first. The library itself never
                stops at its max_optimizer_steps (that only triggers a
                save), so this script saves cache-step<STEPS>.pt and
                ends the run from its save hook; a non-positive STEPS
                leaves EPOCHS as the only end
  SCHED_STEPS   horizon of the LR decay (default = STEPS). Set it
                separately to stop early on an unchanged schedule
  LR            peak learning rate (default 2e-2)
  SCHEDULE      linear | cosine | none  (default linear = paper-faithful:
                warmup WARMUP_MIN_LR->LR over WARMUP_STEPS, then linear decay
                to LR*ALPHA_F)
  WARMUP_STEPS  warmup steps (default 200, the paper's value)
  WARMUP_MIN_LR warmup start LR (default 2e-3, the paper's value)
  ALPHA_F       final-LR fraction of peak (default 0.02 -> 0.1*0.02=0.002 paper)
  EPOCHS        epochs (default 4)
  GLOBAL_BS     global batch size in conversations (default 16; paper 128)
  SAVE_EVERY    write cache-step<N>.pt every N optimizer steps (default off)
  SAVE_AT       comma list of extra steps to save at (e.g. "1,30,90")
                cache-step<N>.pt is the state after exactly N updates, the
                same state the step-N loss eval sees; the library would
                rewrite it once per micro-batch (the last write being after
                N+1 updates), which this script dedupes to the first write
  KEEP_LAST_N   step checkpoints to keep (default 1; the library only prunes
                cache-epoch*.pt, so step checkpoints persist regardless)
  SEED          training seed (default 42)
  NAME          run name under OUT_DIR/runs (default cas_iso_<PATIENT>)
  CART_INIT     initialize from a cartridge FILE instead of the document's
                first p tokens (a saved checkpoint, or a meta-learned init);
                the file's trainable token count overrides KV_TOKENS
  SAVE_INIT     also write the step-0 cartridge to OUT_DIR/carts/<PATIENT>_init.pt
                (default 1) so the run's starting point is recoverable
  VAL_PARQUET   held-out parquet for a distillation-loss eval (same top-20
                loss as training) every VAL_EVERY optimizer steps (default
                60), including step 0 and the end; VAL_LIMIT caps its rows
                (default 256). Printed as VAL_CSV,<step>,<loss> lines, one
                per eval, where <step> is the number of updates applied.
                The eval at a step precedes that step's save, so a run
                that ends at STEPS gets its final eval only when
                VAL_EVERY divides STEPS.
  OUT_DIR       where to write the cartridge (default /root/cart_out)

The paper-regime baseline (patient_02 = 0.65 accuracy, RUNS=3) was trained
with EPOCHS=80 LR=0.1 GLOBAL_BS=128 KV_TOKENS=auto SCHEDULE=linear
WARMUP_STEPS=200 WARMUP_MIN_LR=2e-3 ALPHA_F=0.02 STEPS=5000. There
STEPS is only the length of the LR decay: EPOCHS ends the run near
step 1020 (about 13 steps per epoch, one packed 2048-token sequence per
micro-batch), so the final LR is ~0.083, not LR*ALPHA_F. Pass STEPS
equal to the real step count for a completed decay; the baseline was
measured with the truncated one.
Single-process (NO torchrun) -> is_ddp False -> no gloo teardown hang."""

import inspect
import logging
import os
import sys

os.environ.setdefault(
    "CARTRIDGES_DIR", os.environ.get("CARTRIDGES_DIR", "/root/cartridges")
)
OUT_DIR = os.environ.get("OUT_DIR", "/root/cart_out")
os.environ["CARTRIDGES_OUTPUT_DIR"] = OUT_DIR
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

from cartridges.initialization import KVFromText
from cartridges.train import LossEvalConfig
from cartridges.train import GenerationEvalConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import TrainDataset, DataSource
from cartridges.data.longhealth.evals import LongHealthMultipleChoiceGenerateDataset
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
from cartridges.utils.wandb import WandBConfig
from cartridges.train import CosWithWarmup, LinearWithWarmup

PATIENT = os.environ["PATIENT"]
DATA_PARQUET = os.environ["DATA_PARQUET"]
STEPS = int(os.environ.get("STEPS", "300"))
SCHED_STEPS = int(os.environ.get("SCHED_STEPS", str(STEPS)))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "0"))
SAVE_AT = sorted({int(x) for x in os.environ.get("SAVE_AT", "").split(",") if x})
LR = float(os.environ.get("LR", "2e-2"))
EPOCHS = int(os.environ.get("EPOCHS", "4"))
GLOBAL_BS = int(os.environ.get("GLOBAL_BS", "16"))
SCHEDULE = os.environ.get("SCHEDULE", "linear").lower()
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "200"))
WARMUP_MIN_LR = float(os.environ.get("WARMUP_MIN_LR", "2e-3"))
ALPHA_F = float(os.environ.get("ALPHA_F", "0.02"))
SEED = int(os.environ.get("SEED", "42"))
NAME = os.environ.get("NAME", f"cas_iso_{PATIENT}")
CART_INIT = os.environ.get("CART_INIT", "")
SAVE_INIT = os.environ.get("SAVE_INIT", "1") not in ("0", "", "n", "no")
VAL_PARQUET = os.environ.get("VAL_PARQUET", "")
VAL_EVERY = int(os.environ.get("VAL_EVERY", "60"))
VAL_LIMIT = int(os.environ.get("VAL_LIMIT", "256"))
REC = f"{os.environ.get('RECORDS_DIR', '/root/cart_records')}/{PATIENT}.txt"
CARTS_DIR = os.path.join(OUT_DIR, "carts")
os.makedirs(CARTS_DIR, exist_ok=True)

# Cartridge size: fixed integer, or "auto" = ceil(doc_tokens / KV_DIVISOR) with
# the record itself as the document (the paper sizes each cart ~doc/20).
_kv_raw = os.environ.get("KV_TOKENS", "1024")
if _kv_raw.lower() == "auto":
    from transformers import AutoTokenizer

    _div = int(os.environ.get("KV_DIVISOR", "20"))
    _lo = int(os.environ.get("KV_MIN", "256"))
    _hi = int(os.environ.get("KV_MAX", "2048"))
    _tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    with open(REC) as _f:
        _doc_toks = len(_tok(_f.read())["input_ids"])
    KV_TOKENS = max(_lo, min(_hi, -(-_doc_toks // _div)))  # ceil div, clamped
    print(
        f"[iso] {PATIENT} doc_tokens={_doc_toks} -> KV_TOKENS={KV_TOKENS} "
        f"(auto, /{_div}, clamp [{_lo},{_hi}])",
        flush=True,
    )
else:
    KV_TOKENS = int(_kv_raw)

if CART_INIT:
    # Start from a cartridge file (saved checkpoint or a constructed init).
    from cas_cart_init import KVFromCartFile, cart_shape

    _nl, _nh, _nt, _hd, _nf = cart_shape(CART_INIT)
    if _nt != KV_TOKENS:
        print(
            f"[iso] CART_INIT has {_nt} trainable tokens; overriding KV_TOKENS={KV_TOKENS}",
            flush=True,
        )
        KV_TOKENS = _nt
    print(
        f"[iso] init from file {CART_INIT}: layers={_nl} heads={_nh} "
        f"tokens={_nt} frozen={_nf}",
        flush=True,
    )
    _initializer = KVFromCartFile.Config(path=CART_INIT, num_frozen_tokens=_nf)
elif SAVE_INIT:
    from cas_cart_init import KVFromTextSaved

    _initializer = KVFromTextSaved.Config(
        max_tokens=KV_TOKENS,
        text_source=REC,
        save_path=os.path.join(CARTS_DIR, f"{PATIENT}_init.pt"),
    )
else:
    _initializer = KVFromText.Config(max_tokens=KV_TOKENS, text_source=REC)


def _make_loss_evals():
    # Held-out distillation loss: the identical top-20 objective on rows
    # the run never trains on. Runs at step 0 too (init-only quality).
    if not VAL_PARQUET:
        return [], None
    ds = TrainDataset.Config(
        data_sources=[DataSource(path=VAL_PARQUET, type="local", limit=VAL_LIMIT)],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    )
    return [LossEvalConfig(dataset=ds, name_for_wandb="val")], VAL_EVERY


_loss_evals, _loss_eval_every = _make_loss_evals()


def _make_scheduler():
    # "none" (or a non-positive warmup, which would divide by zero in the
    # linear-warmup ramp) means a flat LR = no scheduler.
    if SCHEDULE == "none" or WARMUP_STEPS <= 0:
        return None
    common = dict(
        warmup_steps=WARMUP_STEPS,
        warmup_min_lr=WARMUP_MIN_LR,
        max_steps=SCHED_STEPS,
        alpha_f=ALPHA_F,
    )
    if SCHEDULE == "cosine":
        return CosWithWarmup.Config(**common)
    # default: linear warmup + linear decay (paper-faithful)
    return LinearWithWarmup.Config(**common)


# The library calls save_cache at every micro-batch of a save step, so a
# file would be rewritten GLOBAL_BS times and end up holding the state
# after N+1 updates. Write each step once, at the first call, which is
# the state after N updates: the state the step-N VAL_CSV line measured.
import cartridges.train as _ct

_orig_save_cache = _ct.save_cache
_saved_steps = set()
# the end-of-training save is the last save_cache call in train(); its
# line number tells it apart from the in-loop calls at the same step
_train_src, _train_first = inspect.getsourcelines(_ct.train)
_final_save_line = _train_first + max(
    i for i, l in enumerate(_train_src) if "save_cache(" in l
)


def _wanted(step):
    if step in SAVE_AT or step == STEPS:
        return True
    return bool(SAVE_EVERY) and step > 0 and step % SAVE_EVERY == 0


class _StopTraining(Exception):
    """Raised from the save hook once STEPS updates have been applied.

    The library only saves at max_optimizer_steps and then keeps
    training until EPOCHS, so the stop has to come from here. The hook
    runs at the first micro-batch of step STEPS, after that step's
    held-out eval and before any further update, so the saved file is
    the state the VAL_CSV,<STEPS> line measured."""


def _save_cache_once(config, cache, optimizer_step):
    # the end-of-training save arrives after the loop with the final step
    # count; it is never a repeat of an in-loop write, so it always lands
    final = sys._getframe(1).f_lineno == _final_save_line
    if optimizer_step not in _saved_steps and (final or _wanted(optimizer_step)):
        _saved_steps.add(optimizer_step)
        _orig_save_cache(config, cache, optimizer_step)
        print(f"SAVE_CSV,{optimizer_step}", flush=True)
    if not final and STEPS > 0 and optimizer_step >= STEPS:
        raise _StopTraining(optimizer_step)


_ct.save_cache = _save_cache_once

# Print the held-out loss as a parseable line. The library only logs
# "Eval loss - <x>" through its logger, without the step.
_orig_eval_ppl = _ct.evaluate_perplexity
_eval_step = {"step": None}


def _evaluate_perplexity_csv(*args, **kwargs):
    _eval_step["step"] = kwargs.get("optimizer_step")
    return _orig_eval_ppl(*args, **kwargs)


class _EvalLossToCsv(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if msg.startswith("Eval loss - "):
            print(
                f"VAL_CSV,{_eval_step['step']},{float(msg.split('-', 1)[1]):.6f}",
                flush=True,
            )
        return True


_ct.evaluate_perplexity = _evaluate_perplexity_csv
_ct.logger.addFilter(_EvalLossToCsv())


config = TrainConfig(
    model=HFModelConfig(
        pretrained_model_name_or_path="Qwen/Qwen3-8B",
        model_cls=FlexQwen3ForCausalLM,
    ),
    kv_cache_initializer=_initializer,
    lr=LR,
    lr_scheduler=_make_scheduler(),
    epochs=EPOCHS,
    global_batch_size=GLOBAL_BS,
    max_optimizer_steps=STEPS,
    dataset=TrainDataset.Config(
        data_sources=[DataSource(path=DATA_PARQUET, type="local")],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ),
    # every step reaches _save_cache_once below, which applies SAVE_EVERY
    # and SAVE_AT and writes each step at most once
    save_every_n_steps=(1 if (SAVE_EVERY or SAVE_AT) else None),
    save_after_training=True,  # persist the cartridge
    keep_last_n_saved=int(os.environ.get("KEEP_LAST_N", "1")),
    loss_evals=_loss_evals,
    loss_eval_every_n_steps=_loss_eval_every,
    save_to_wandb=False,
    generate_before_training=False,  # skip -- oracle eval done separately in combine step
    generate_evals=[],  # keep training lean; eval in cas_combine_eval.py
    distributed_backend="gloo",
    wandb=WandBConfig(tags=["cas", "isolated", PATIENT]),
    output_dir=OUT_DIR,
    name=NAME,
    seed=SEED,
)

if __name__ == "__main__":
    # We invoke config.run() directly (not pydrantic.main), so run_dir is unset ->
    # save_cache() would crash. Set a deterministic run_dir before training.
    import os as _os

    config.run_dir = _os.path.join(OUT_DIR, "runs", config.name)
    _os.makedirs(config.run_dir, exist_ok=True)
    print(
        f"[iso] {PATIENT} steps={STEPS} sched_steps={SCHED_STEPS} epochs={EPOCHS} "
        f"lr={LR} global_bs={GLOBAL_BS} seed={SEED} save_every={SAVE_EVERY} "
        f"save_at={SAVE_AT} val_every={VAL_EVERY if VAL_PARQUET else None}",
        flush=True,
    )
    try:
        config.run()
    except _StopTraining as stop:
        print(f"[iso] stopped after {stop.args[0]} optimizer steps (STEPS)", flush=True)
    # Copy the saved cartridge to a deterministic path for combine/joint steps.
    import glob, shutil
    from pathlib import Path

    carts = Path(OUT_DIR) / "carts"
    carts.mkdir(parents=True, exist_ok=True)
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

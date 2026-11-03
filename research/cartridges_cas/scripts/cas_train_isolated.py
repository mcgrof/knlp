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
  CLIP_NORM     clip the cartridge gradient to this global L2 norm before
                every Adam step (default 0 = off, the library's behaviour).
                The library has no clipping knob; the clip is applied
                in the optimizer's step() so nothing else in the loop moves
  CLIP_REL      clip relative to recent history instead of a fixed norm:
                once CLIP_WINDOW norms are recorded, the gradient is
                scaled down to CLIP_REL times the median of the last
                CLIP_WINDOW norms whenever it is larger (default 0 =
                off). The history stores each norm after clipping, so
                one spike cannot raise the bar for the next. Adam is
                nearly scale-invariant, so a fixed bar that clips every
                step changes little; a relative bar clips only outliers
  CLIP_WINDOW   steps of history for CLIP_REL (default 50)
  ADAM_BETA2    Adam's second-moment decay (default unset = the library's
                torch default 0.999). A smaller value such as 0.95 lets
                the per-coordinate scale follow a sudden large gradient
                within tens of steps instead of hundreds, so one large
                gradient after a quiet stretch makes a smaller step
  DECAY_LAMBDA  decoupled weight decay on the trainable cartridge tokens,
                applied after every Adam step as
                p -= lr_t * DECAY_LAMBDA * (p - target), where lr_t is
                that step's scheduled rate (default 0 = off, the
                library's behaviour). The frozen tokens never move. Hold
                DECAY_LAMBDA, not the timescale 1 / (LR * DECAY_LAMBDA),
                fixed when LR changes: with an anchor the equilibrium
                drift is set by DECAY_LAMBDA alone
  DECAY_TO      init | zero (default init): the decay target is the
                run's starting cartridge (the document's first tokens or
                CART_INIT), or zero as in ordinary AdamW decay
  LOG_GRADNORM  print GRADNORM_CSV,<step>,<norm>,<clipped> once per
                optimizer step (default on whenever CLIP_NORM is set):
                <norm> is the pre-clip global L2 norm of the cartridge
                gradient in fp32, <step> the number of updates already
                applied, <clipped> 1 when the clip scaled the gradient
  VALUE_CAP     after every update, shrink each trainable slot's value
                vector (per layer and head) to at most VALUE_CAP times
                that layer's median value norm in the starting cartridge
                (default 0, off). A loss spike multiplies the value norms
                of the first layers about eightfold in one step; keys
                stay near the reader's scale (cas_cart_norms.py)
  PACKED_LEN    tokens per packed training sequence (default 2048, the
                upstream LongHealth example; the CAS paper uses 8192)
  PACKING_MODE  truncate (default, the upstream example) or pad. The
                library forms the packs once per run, so in truncate
                mode the conversation that overflows a sequence loses
                its tail in every epoch. In pad mode it starts the next
                sequence instead and the rest is padded; only a single
                conversation longer than PACKED_LEN is still cut.
                Applies to the validation loss data too
  MICRO_SEQS    packed sequences per forward pass (default 1, the
                library's behaviour). Values above 1 concatenate that
                many packed sequences into one pass with separate
                element ids and positions, accumulate GLOBAL_BS /
                MICRO_SEQS passes per optimizer step, and weight each
                target entry so the loss is still the sum of
                per-sequence means over GLOBAL_BS (cas_microbatch.py).
                The groups of sequences are fixed by the packing seed,
                so the epoch shuffle permutes groups. Needs the
                position_weights.py library patch; excludes POS_WEIGHTS
  PIECE_CARTS   comma list of piece cartridges of one split document
                (cas_split.py), in piece order: train all pieces in one
                run (cas_pieces.py). Their combined cartridge replaces
                CART_INIT; needs PIECE_PARQUETS, one Self-Study parquet
                per piece, which replace DATA_PARQUET
  PIECE_MODE    isolated (default): each conversation sees only its own
                piece at its standalone positions; joint: the pieces sit
                at their bundle positions and a conversation sees its own
                piece plus, with probability 1 - P_ISO, others
  P_ISO         joint mode: probability a conversation sees only its own
                piece (default 0, the whole bundle every time)
  NAME          run name under OUT_DIR/runs (default cas_iso_<PATIENT>)
  CART_INIT     initialize from a cartridge FILE instead of the document's
                first p tokens (a saved checkpoint, or a meta-learned init);
                the file's trainable token count overrides KV_TOKENS
  SAVE_INIT     also write the step-0 cartridge to OUT_DIR/carts/<PATIENT>_init.pt
                (default 1) so the run's starting point is recoverable
  VAL_PARQUET   held-out parquet for a distillation-loss eval (same top-20
                loss as training) every VAL_EVERY optimizer steps (default
                60), including step 0 and the end; VAL_LIMIT caps its rows
                (default 256, "all" for none; the library keeps the FIRST
                VAL_LIMIT rows). Printed as VAL_CSV,<step>,<loss> lines, one
                per eval, where <step> is the number of updates applied.
                The eval at a step precedes that step's save, so a run
                that ends at STEPS gets its final eval only when
                VAL_EVERY divides STEPS.
  DATA_SHA256, DATA_ROWS, VAL_SHA256, VAL_ROWS
                when set, the run stops at startup unless DATA_PARQUET and
                VAL_PARQUET have exactly that SHA-256 and row count, and
                VAL_ROWS also requires VAL_LIMIT to cover every row. Hash
                and row count are printed as INPUT_RECEIPT either way.
  LOAD_BF16     1 loads the model weights directly in bf16 instead of the
                library's fp32-then-cast (same weights; fits a 24 GiB GPU)
  POS_WEIGHTS   .npz from cas_position_weights.py built on DATA_PARQUET
                with this SEED: weight the per-entry loss by position
                (POS_MODE informative | rest | uniform, POS_FLOOR for the
                rest weight in informative mode, default 0.1). Needs the
                position_weights.py library patch. The held-out eval
                stays unweighted. Unset = the library's plain mean
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
import types

import torch

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

# The library dispatches its chat-template and thinking-mode handling on the
# tokenizer's reported name, and that table spells the model with a lowercase
# b. Two spellings also mean two Hugging Face cache directories and a second
# sixteen-gigabyte download of the same weights, so the name is a knob with the
# historical default rather than a literal in two places.
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B")
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
CLIP_NORM = float(os.environ.get("CLIP_NORM", "0"))
CLIP_REL = float(os.environ.get("CLIP_REL", "0"))
CLIP_WINDOW = int(os.environ.get("CLIP_WINDOW", "50"))
assert not (CLIP_NORM > 0 and CLIP_REL > 0), "set CLIP_NORM or CLIP_REL, not both"
LOG_GRADNORM = (
    os.environ.get("LOG_GRADNORM", "1" if CLIP_NORM > 0 or CLIP_REL > 0 else "0") == "1"
)
ADAM_BETA2 = os.environ.get("ADAM_BETA2", "")
ADAM_BETA2 = float(ADAM_BETA2) if ADAM_BETA2 else None
assert ADAM_BETA2 is None or 0 < ADAM_BETA2 < 1, "ADAM_BETA2 must be in (0, 1)"
DECAY_LAMBDA = float(os.environ.get("DECAY_LAMBDA", "0"))
DECAY_TO = os.environ.get("DECAY_TO", "init").lower()
MICRO_SEQS = int(os.environ.get("MICRO_SEQS", "1"))
PACKED_LEN = int(os.environ.get("PACKED_LEN", "2048"))
PACKING_MODE = os.environ.get("PACKING_MODE", "truncate")
assert PACKING_MODE in ("truncate", "pad"), "PACKING_MODE: truncate or pad"
VALUE_CAP = float(os.environ.get("VALUE_CAP", "0"))
assert GLOBAL_BS % MICRO_SEQS == 0, "MICRO_SEQS must divide GLOBAL_BS"
assert DECAY_TO in ("init", "zero"), f"DECAY_TO must be init or zero, not {DECAY_TO}"
NAME = os.environ.get("NAME", f"cas_iso_{PATIENT}")
CART_INIT = os.environ.get("CART_INIT", "")
PIECE_CARTS = [x for x in os.environ.get("PIECE_CARTS", "").split(",") if x]
PIECE_PARQUETS = [x for x in os.environ.get("PIECE_PARQUETS", "").split(",") if x]
PIECE_MODE = os.environ.get("PIECE_MODE", "isolated")
P_ISO = float(os.environ.get("P_ISO", "0"))
PIECE_LENGTHS = []
if PIECE_CARTS:
    # all pieces of one split document in one cache (cas_pieces.py)
    assert len(PIECE_PARQUETS) == len(PIECE_CARTS), "one parquet per piece"
    from cas_cart_init import load_cart, save_cart
    from cas_pieces import concat

    _parts = [load_cart(p) for p in PIECE_CARTS]
    PIECE_LENGTHS = [p[0][0].shape[2] for p in _parts]
    os.makedirs(OUT_DIR, exist_ok=True)  # not created yet at this point
    CART_INIT = os.path.join(OUT_DIR, f"pieces_{PIECE_MODE}_init.pt")
    save_cart(*concat(_parts, shift=PIECE_MODE == "joint"), CART_INIT)
    del _parts
SAVE_INIT = os.environ.get("SAVE_INIT", "1") not in ("0", "", "n", "no")
VAL_PARQUET = os.environ.get("VAL_PARQUET", "")
VAL_EVERY = int(os.environ.get("VAL_EVERY", "60"))
VAL_LIMIT = os.environ.get("VAL_LIMIT", "256")
VAL_LIMIT = None if VAL_LIMIT == "all" else int(VAL_LIMIT)


def _input_receipt(path, sha_env, rows_env):
    """Hash and count one input parquet; check it against env if given."""
    import hashlib
    import pyarrow.parquet as pq

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    rec = dict(
        path=os.path.abspath(path),
        sha256=h.hexdigest(),
        rows=pq.ParquetFile(path).metadata.num_rows,
    )
    want_sha, want_rows = os.environ.get(sha_env), os.environ.get(rows_env)
    assert not want_sha or rec["sha256"] == want_sha, f"{sha_env}: {path} differs"
    assert not want_rows or rec["rows"] == int(want_rows), f"{rows_env}: {path}"
    return rec


INPUT_RECEIPT = {}
if not PIECE_PARQUETS:
    INPUT_RECEIPT["data"] = _input_receipt(DATA_PARQUET, "DATA_SHA256", "DATA_ROWS")
if VAL_PARQUET:
    _val = _input_receipt(VAL_PARQUET, "VAL_SHA256", "VAL_ROWS")
    _val["rows_read"] = min(_val["rows"], VAL_LIMIT or _val["rows"])
    if os.environ.get("VAL_ROWS"):
        assert _val["rows_read"] == _val["rows"], "VAL_LIMIT drops held-out rows"
    INPUT_RECEIPT["val"] = _val
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
    _tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
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
        packed_seq_length=PACKED_LEN,
        packing_mode=PACKING_MODE,
    )
    return [LossEvalConfig(dataset=ds, name_for_wandb="val")], VAL_EVERY


_loss_evals, _loss_eval_every = _make_loss_evals()

# The library loads the checkpoint in fp32 on the GPU and casts to bf16
# afterwards, which needs 32 GiB for Qwen3-8B before the cast and fails
# on a 24 GiB card. LOAD_BF16=1 loads the stored bf16 weights directly;
# the weights after the library's cast are identical either way.
_load_kwargs = (
    {"torch_dtype": torch.bfloat16} if os.environ.get("LOAD_BF16") == "1" else {}
)


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

# The library builds ``optim.Adam(cache.parameters(), ...)`` and calls
# ``optimizer.step()`` straight after the last accumulated backward, with
# no clipping in between. Clipping and the gradient-norm log are put in
# the optimizer's step() itself, so the training loop is untouched and
# a run with CLIP_NORM unset is bit-identical to one from before this
# knob existed (the norm is only measured, never applied).
#
# The decay toward the starting cartridge goes in the same place, after
# the Adam update. Its target is a copy of each trainable tensor taken
# just before the first update, so it is the run's own start whatever
# the initializer was. Parameters with requires_grad off (the frozen
# sink) are left out, so a decay toward zero cannot shrink them either.


class _ClippedAdam(torch.optim.Adam):
    _n_steps = 0
    _history = []
    _decay_pairs = None
    _value_refs = None

    def __init__(self, params, **kw):
        # the library passes lr and weight_decay only, so betas are
        # torch's default unless ADAM_BETA2 is set
        if ADAM_BETA2 is not None:
            kw["betas"] = (kw.get("betas", (0.9, 0.999))[0], ADAM_BETA2)
        super().__init__(params, **kw)

    def step(self, closure=None):
        if CLIP_NORM > 0 or CLIP_REL > 0 or LOG_GRADNORM:
            params = [p for g in self.param_groups for p in g["params"]]
            grads = [p.grad for p in params if p.grad is not None]
            norm = torch.norm(torch.stack([g.float().norm() for g in grads]))
            clipped = 0
            bar = CLIP_NORM
            if CLIP_REL > 0 and len(_ClippedAdam._history) >= CLIP_WINDOW:
                recent = sorted(_ClippedAdam._history[-CLIP_WINDOW:])
                bar = CLIP_REL * recent[len(recent) // 2]
            if bar > 0 and float(norm) > bar:
                torch.nn.utils.clip_grad_norm_(params, bar)
                clipped = 1
            _ClippedAdam._history.append(min(float(norm), bar) if bar else float(norm))
            if LOG_GRADNORM:
                print(
                    f"GRADNORM_CSV,{_ClippedAdam._n_steps},{float(norm):.6g},{clipped}",
                    flush=True,
                )
        if DECAY_LAMBDA > 0 and self._decay_pairs is None:
            self._decay_pairs = [
                (p, p.detach().clone() if DECAY_TO == "init" else None)
                for g in self.param_groups
                for p in g["params"]
                if p.requires_grad
            ]
        if VALUE_CAP > 0 and self._value_refs is None:
            # reference: the starting cartridge's median slot value norm,
            # per layer, taken before the first update
            with torch.no_grad():
                self._value_refs = {
                    id(p): float(p.float().norm(dim=-1).median())
                    for g in self.param_groups
                    for p in g["params"]
                    if hasattr(p, "_cr_value_layer")
                }
        _ClippedAdam._n_steps += 1
        out = super().step(closure)
        if VALUE_CAP > 0:
            with torch.no_grad():
                vals = [
                    p
                    for g in self.param_groups
                    for p in g["params"]
                    if hasattr(p, "_cr_value_layer")
                ]
                assert vals, "VALUE_CAP set but no tagged value parameters"
                n_capped = 0
                for p in vals:
                    n = p.float().norm(dim=-1, keepdim=True)
                    bound = VALUE_CAP * self._value_refs[id(p)]
                    scale = (bound / n.clamp_min(1e-12)).clamp(max=1.0)
                    n_capped += int((scale < 1).sum())
                    p.mul_(scale.to(p.dtype))
                if LOG_GRADNORM:
                    print(
                        f"VALUECAP_CSV,{_ClippedAdam._n_steps},{n_capped}", flush=True
                    )
        if DECAY_LAMBDA > 0:
            # the library builds one parameter group, so one rate
            f = self.param_groups[0]["lr"] * DECAY_LAMBDA
            assert 0 < f < 1, f"decay pull {f} per step is outside (0, 1)"
            with torch.no_grad():
                for p, start in self._decay_pairs:
                    if start is None:
                        p.mul_(1.0 - f)
                    else:
                        p.lerp_(start, f)
        return out


if VALUE_CAP > 0:
    from cartridges.cache import TrainableCache as _VTC

    _orig_vtc_init = _VTC.__init__

    def _vtc_init(self, *a, **kw):
        _orig_vtc_init(self, *a, **kw)
        for i, p in enumerate(self.trainable_values or []):
            p._cr_value_layer = i

    _VTC.__init__ = _vtc_init

# train.py uses its ``optim`` import for optim.Adam only
_ct.optim = types.SimpleNamespace(Adam=_ClippedAdam)

# Per-position loss weights. The library patch routes the loop's
# ``ce_by_token.mean()`` through ``_cr_entry_loss``; install the weighted
# version only when asked, so an unweighted run is the unpatched arithmetic.
POS_WEIGHTS = os.environ.get("POS_WEIGHTS", "")
if POS_WEIGHTS:
    assert hasattr(
        _ct, "_cr_entry_loss"
    ), "POS_WEIGHTS needs the position_weights.py patch"
    from cas_position_weights import load_weights

    _pos = load_weights(
        POS_WEIGHTS,
        os.environ.get("POS_MODE", "informative"),
        float(os.environ.get("POS_FLOOR", "0.1")),
    )
    _ct._cr_entry_loss = _pos.entry_loss
    print(
        f"[iso] position weights {POS_WEIGHTS}: mode={_pos.meta['mode']} "
        f"w_info={_pos.w_info:.4f} w_rest={_pos.w_rest:.4f} "
        f"info_entries={_pos.meta['n_info_entries']}/{_pos.meta['n_entries']} "
        f"batches={len(_pos.masks)}",
        flush=True,
    )

if MICRO_SEQS > 1:
    assert not POS_WEIGHTS, "MICRO_SEQS and POS_WEIGHTS both use the loss hook"
    from cas_microbatch import install as _install_micro

    _install_micro(TrainDataset, _ct, MICRO_SEQS)
    print(f"[iso] {MICRO_SEQS} packed sequences per forward pass", flush=True)

if PIECE_CARTS:
    import cartridges.models.qwen.modeling_qwen3 as _mq
    from cartridges.cache import TrainableCache as _TC
    from cas_pieces import install as _install_pieces

    _install_pieces(TrainDataset, _mq, _TC, PIECE_LENGTHS, PIECE_MODE, P_ISO, SEED)

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
        pretrained_model_name_or_path=MODEL_NAME,
        model_cls=FlexQwen3ForCausalLM,
        load_kwargs=_load_kwargs,
    ),
    kv_cache_initializer=_initializer,
    lr=LR,
    lr_scheduler=_make_scheduler(),
    epochs=EPOCHS,
    # the library counts accumulated passes; each pass holds MICRO_SEQS
    global_batch_size=GLOBAL_BS // MICRO_SEQS,
    max_optimizer_steps=STEPS,
    dataset=TrainDataset.Config(
        data_sources=[
            DataSource(path=p, type="local") for p in (PIECE_PARQUETS or [DATA_PARQUET])
        ],
        top_k_logits=20,
        packed_seq_length=PACKED_LEN,
        packing_mode=PACKING_MODE,
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
        f"save_at={SAVE_AT} val_every={VAL_EVERY if VAL_PARQUET else None} "
        f"clip_norm={CLIP_NORM or None} clip_rel={CLIP_REL or None}/"
        f"{CLIP_WINDOW} adam_beta2={ADAM_BETA2} "
        f"micro_seqs={MICRO_SEQS} "
        f"decay={f'{DECAY_LAMBDA} to {DECAY_TO}' if DECAY_LAMBDA > 0 else None}",
        flush=True,
    )
    import json as _json

    print("INPUT_RECEIPT " + _json.dumps(INPUT_RECEIPT), flush=True)
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

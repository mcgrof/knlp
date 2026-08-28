#!/usr/bin/env python3
"""Gate 0 fixed-trajectory objective-decomposition screen trainer.

Trains ONE cartridge for a small number of steps under exactly one
target-construction arm, with everything else frozen and shared across
arms: the starting cartridge, a saved zero-moment AdamW state, a frozen
example schedule, the learning rate, clipping, and the loss denominator
(the count of VALID legacy serialized entries per element — never
renormalized).  The arms decompose the legacy stored objective per the
control-aware plan:

    legacy_raw               exact historical serialized objective
    dedup_only               unique support, original denominator
    dedup_scale_matched      unique support scaled once to legacy
                             coefficient mass
    control_anchor           unique + per-row anchors on first-answer
                             and natural end-of-turn rows
    content_anchor_matched   unique + count/mass-matched anchors on
                             non-control duplicated rows
    parity                   no training: proves legacy_raw equals
                             unique + explicit anchors in loss and
                             cartridge gradients on the real model

Checkpoints are saved at the declared steps for the separate evaluator;
this trainer never evaluates.  Every run writes a manifest with the
schedule hash, transform hashes, calibration report, and artifact
SHA256s so the run is re-runnable from the manifest alone.

Env: MODEL, ARM, PATIENT, DATA_PARQUET, CART_INIT, OPT_INIT,
SCHEDULE_JSON, STEPS, ACCUM, LR, SEED, CHECKPOINT_AT, OUT_DIR.
"""

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridges")
OUT_DIR = os.environ.get("OUT_DIR", "/root/cas_out/control_screen")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", OUT_DIR)
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.datasets import TrainDataset, DataSource
from cartridges.initialization.tokenization_utils import (
    MODEL_TO_SYSTEM_PROMPT_TOKENIZER,
)
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM

for _d in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    sys.path.insert(0, str(_d))
from control_aware.targets import (  # noqa: E402
    TARGET_SCHEMA_VERSION,
    build_target_set,
    calibrate_content_anchors,
    calibrate_scale,
    parse_element,
    transform_hash,
)

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
ARM = os.environ["ARM"]
PATIENT = os.environ["PATIENT"]
DATA_PARQUET = os.environ["DATA_PARQUET"]
CART_INIT = os.environ["CART_INIT"]
OPT_INIT = os.environ.get("OPT_INIT", "")
SCHEDULE_JSON = os.environ["SCHEDULE_JSON"]
STEPS = int(os.environ.get("STEPS", "10"))
ACCUM = int(os.environ.get("ACCUM", "8"))
LR = float(os.environ.get("LR", "2e-2"))
SEED = int(os.environ.get("SEED", "0"))
CHECKPOINT_AT = [
    int(x) for x in os.environ.get("CHECKPOINT_AT", "0,1,2,5,10").split(",")
]
CLIP = float(os.environ.get("CLIP", "1.0"))
# Position selection: keep only the rows the document is most
# responsible for.  The objective otherwise spends its budget almost
# independently of document relevance (measured correlation 0.09), so
# this asks whether training a few percent of positions on purpose
# beats training all of them by default.
RELEVANCE_MAP = os.environ.get("RELEVANCE_MAP", "")
KEEP_FRAC = float(os.environ.get("KEEP_FRAC", "1.0"))
SELECT_INVERT = os.environ.get("SELECT_INVERT", "0") == "1"
RECORDS_DIR = os.environ.get("RECORDS_DIR", "/root/cart_records")
KV_TOKENS = int(os.environ.get("KV_TOKENS", "512"))
DEVICE = "cuda"

ARM_TO_MODE = dict(
    legacy_raw="legacy_raw",
    dedup_only="dedup_legacy_support",
    dedup_scale_matched="dedup_scale_matched",
    control_anchor="control_anchor",
    content_anchor_matched="content_anchor_matched",
)
assert ARM in list(ARM_TO_MODE) + ["parity"], ARM


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def flatten_mode():
    """Which flattener the installed cartridges package carries.

    Recorded for provenance only: flatten() runs at SYNTHESIS time
    (synthesizers/self_study.py) and the parquet stores the already
    flattened rows, so the installed flattener cannot change the
    targets this screen consumes.  It matters for any future synthesis
    — the legacy flattener stops early on duplicate-inflated
    cumulative mass, the fixed one retains all K when the threshold is
    never reached."""
    import inspect

    import cartridges.clients.base as base

    src = inspect.getsource(base)
    return "edge_fixed" if "EDGE FIX" in src else "legacy_early_stop"


@torch.no_grad()
def build_init_cart(path, model, tok, attn_config):
    """Truncation init from the patient record, so a run can start from
    an untrained cartridge instead of only continuing someone else's.

    Answering whether cartridge training overshoots needs the whole
    curve from step zero, and the checkpoints of the original run no
    longer exist.  Forward the first KV_TOKENS system-prompt tokens of
    the record through the frozen model and keep the resulting cache;
    the first token becomes a frozen attention sink, matching the
    geometry every evaluator here expects."""
    record = (Path(RECORDS_DIR) / f"{PATIENT}.txt").read_text()
    tok_fn = MODEL_TO_SYSTEM_PROMPT_TOKENIZER[tok.name_or_path.lower()]
    ids = tok_fn(tokenizer=tok, content=record, max_tokens=KV_TOKENS).squeeze(0)
    assert ids.shape[0] >= KV_TOKENS, (
        f"{PATIENT}: record gives only {ids.shape[0]} tokens "
        f"(< KV_TOKENS={KV_TOKENS}); lower KV_TOKENS"
    )
    ids = ids[:KV_TOKENS].to(DEVICE)
    tmp = TrainableCache(config=attn_config)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        model(
            input_ids=ids,
            seq_ids=torch.zeros_like(ids, dtype=torch.long),
            position_ids=torch.arange(ids.shape[-1], dtype=torch.long, device=DEVICE),
            use_cache=True,
            past_key_values=tmp,
            mode="generate",
        )
    keys = [t.detach().to(torch.bfloat16).cpu() for t in tmp._keys]
    values = [t.detach().to(torch.bfloat16).cpu() for t in tmp._values]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "trainable_keys": [k[:, :, 1:].contiguous() for k in keys],
            "trainable_values": [v[:, :, 1:].contiguous() for v in values],
            "frozen_keys": [k[:, :, :1].contiguous() for k in keys],
            "frozen_values": [v[:, :, :1].contiguous() for v in values],
        },
        path,
    )
    print(f"[ctrl-screen] built truncation-init cartridge: {path}")


def load_cart_as_trainable(path, attn_config):
    """Same loader contract as the legacy continuation trainer:
    preserve the frozen/trainable split of the checkpoint."""
    ck = torch.load(path, map_location="cpu", weights_only=False)

    def t(p):
        return torch.as_tensor(p.data if hasattr(p, "data") else p).to(torch.bfloat16)

    tk = [t(p) for p in ck["trainable_keys"]]
    tv = [t(p) for p in ck["trainable_values"]]
    fk = ck.get("frozen_keys") or []
    fv = ck.get("frozen_values") or []
    if fk:
        ik = [torch.cat([t(fk[i]), tk[i]], dim=2) for i in range(len(tk))]
        iv = [torch.cat([t(fv[i]), tv[i]], dim=2) for i in range(len(tv))]
        nfrozen = t(fk[0]).shape[2]
    else:
        ik, iv, nfrozen = tk, tv, 0
    cache = TrainableCache(
        config=attn_config, init_keys=ik, init_values=iv, num_frozen_tokens=nfrozen
    ).to(DEVICE)
    return cache, nfrozen


print(f"[ctrl-screen] arm={ARM} model={MODEL} patient={PATIENT}")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
EOT = tok.convert_tokens_to_ids("<|im_end|>")
model = FlexQwen3ForCausalLM.from_pretrained(MODEL).to(DEVICE).to(torch.bfloat16)
model.eval()
model.requires_grad_(False)
attn_config = AttnConfig(
    n_layers=model.config.num_hidden_layers,
    n_heads=model.config.num_key_value_heads,
    head_dim=model.config.head_dim,
)
if not Path(CART_INIT).is_file():
    build_init_cart(CART_INIT, model, tok, attn_config)
cache, NUM_FROZEN = load_cart_as_trainable(CART_INIT, attn_config)
for p in model.parameters():
    assert not p.requires_grad
assert any(p.requires_grad for p in cache.parameters())

dataset = TrainDataset.Config(
    data_sources=[DataSource(path=DATA_PARQUET, type="local")],
    top_k_logits=20,
    packed_seq_length=2048,
    packing_mode="truncate",
).instantiate(tokenizer=tok, seed=SEED)
print(f"[ctrl-screen] dataset elements: {len(dataset.elements)}")


# ---------------------------------------------------------------------------
# frozen schedule and calibration
# ---------------------------------------------------------------------------


def load_or_create_schedule():
    if Path(SCHEDULE_JSON).is_file():
        sched = json.loads(Path(SCHEDULE_JSON).read_text())
        assert sched["dataset_elements"] == len(dataset.elements)
        assert sched["cart_sha256"] == sha256_file(
            CART_INIT
        ), "schedule was frozen against a different starting cartridge"
        assert sched["seed"] == SEED and sched["accum"] == ACCUM
        assert sched["steps"] >= STEPS
        assert sched["parquet_sha256"] == sha256_file(
            DATA_PARQUET
        ), "schedule was frozen against a different parquet"
        return sched
    rng = random.Random(SEED)
    steps = [
        [rng.randrange(len(dataset.elements)) for _ in range(ACCUM)]
        for _ in range(STEPS)
    ]
    sched = dict(
        seed=SEED,
        accum=ACCUM,
        steps=STEPS,
        schedule=steps,
        dataset_elements=len(dataset.elements),
        parquet_sha256=sha256_file(DATA_PARQUET),
        cart_sha256=sha256_file(CART_INIT),
        model=MODEL,
        target_schema=TARGET_SCHEMA_VERSION,
        flatten_mode=flatten_mode(),
        relevance_map=RELEVANCE_MAP or None,
        keep_frac=KEEP_FRAC,
        select_invert=SELECT_INVERT,
    )
    Path(SCHEDULE_JSON).parent.mkdir(parents=True, exist_ok=True)
    Path(SCHEDULE_JSON).write_text(json.dumps(sched, indent=1))
    print(f"[ctrl-screen] froze schedule: {SCHEDULE_JSON}")
    return sched


schedule = load_or_create_schedule()
sched_ids = sorted({i for step in schedule["schedule"][:STEPS] for i in step})
parsed = {}
for i in sched_ids:
    el = dataset.elements[i]
    parsed[i] = parse_element(
        el.topk_token_idxs, el.topk_token_ids, el.topk_logprobs, EOT
    )

parsed_list = [parsed[i] for i in sched_ids]

# The objective divides each element by the count of LEGACY entries that
# survive the position/vocabulary validity mask, then averages elements
# equally, so an element's anchor coefficient is weighted by 1/denom.
# Calibration must use the same weights, which means computing the real
# denominators up front — they need only the packed length and the
# vocabulary size, no forward pass.
VOCAB = int(model.config.vocab_size)


def legacy_valid_count(i):
    et = parsed[i]
    seq_len = int(dataset.elements[i].input_ids.shape[0])
    ts = build_target_set(et, "legacy_raw")
    gi, tids, _ = ts.tensors()
    gi = gi - 1
    valid = (gi >= 0) & (gi < seq_len) & (tids >= 0) & (tids < VOCAB)
    return int(valid.sum())


DENOM = {i: legacy_valid_count(i) for i in sched_ids}

# Build the keep-set once: a global threshold over every scheduled row,
# so the budget is spent where the document actually acted rather than
# per element.  SELECT_INVERT keeps the LEAST relevant rows instead, as
# the control that separates "selection helps" from "training less
# helps".
KEEP_ROWS = None
if RELEVANCE_MAP and KEEP_FRAC < 1.0:
    _rm = json.loads(Path(RELEVANCE_MAP).read_text())["map"]
    _scored = [
        (float(v), int(ei), int(ri))
        for ei, rows in _rm.items()
        if int(ei) in DENOM
        for ri, v in rows.items()
    ]
    _scored.sort(reverse=not SELECT_INVERT)
    _k = max(1, int(len(_scored) * KEEP_FRAC))
    KEEP_ROWS = {}
    for _v, _ei, _ri in _scored[:_k]:
        KEEP_ROWS.setdefault(_ei, set()).add(_ri)
    print(
        f"[ctrl-screen] selection: keeping {_k}/{len(_scored)} rows "
        f"({100 * KEEP_FRAC:.1f}%, {'LEAST' if SELECT_INVERT else 'most'} "
        f"document-relevant) across {len(KEEP_ROWS)} elements"
    )
denom_list = [DENOM[i] for i in sched_ids]
A2_SCALE = calibrate_scale(parsed_list, denoms=denom_list)
content_rows_list, CONTENT_SCALE, content_report = calibrate_content_anchors(
    parsed_list, denoms=denom_list
)
content_rows = {i: rows for i, rows in zip(sched_ids, content_rows_list)}
print(
    f"[ctrl-screen] calibration: scale={A2_SCALE:.6f} "
    f"content_scale={CONTENT_SCALE:.6f} "
    f"control_anchors={content_report['control_count']} "
    f"denom_min={min(denom_list)} denom_max={max(denom_list)}"
)


def target_set_for(i, mode):
    et = parsed[i]
    d = DENOM[i]
    if mode == "dedup_scale_matched":
        return build_target_set(et, mode, scale=A2_SCALE, denom=d)
    if mode == "content_anchor_matched":
        return build_target_set(
            et,
            mode,
            content_rows=content_rows[i],
            content_scale=CONTENT_SCALE,
            denom=d,
        )
    return build_target_set(et, mode, denom=d)


# ---------------------------------------------------------------------------
# loss: sum(p * student_nll) over valid entries / valid legacy count
# ---------------------------------------------------------------------------


def element_forward(el):
    ids = el.input_ids.to(DEVICE)
    seq_ids = torch.zeros(ids.shape[0], dtype=torch.long, device=DEVICE)
    pos = torch.arange(ids.shape[0], device=DEVICE)
    cache.clear()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(
            input_ids=ids,
            seq_ids=seq_ids,
            position_ids=pos,
            use_cache=True,
            past_key_values=cache,
        )
    return out.logits, ids.shape[0]


def arm_loss(i, mode):
    el = dataset.elements[i]
    logits, seq_len = element_forward(el)
    vocab = logits.shape[-1]
    assert vocab == VOCAB, f"vocab drift: {vocab} vs {VOCAB}"
    denom = DENOM[i]
    if denom == 0:
        return None
    ts = target_set_for(i, mode)
    if KEEP_ROWS is not None:
        keep = KEEP_ROWS.get(i, set())
        sel = [j for j, r in enumerate(ts.row_idxs) if r in keep]
        if not sel:
            return None
        ts = type(ts)(
            [ts.row_idxs[j] for j in sel],
            [ts.token_ids[j] for j in sel],
            [ts.probs[j] for j in sel],
            ts.denom,
        )
    gi, tids, probs = ts.tensors(device=DEVICE)
    gi = gi - 1
    valid = (gi >= 0) & (gi < seq_len) & (tids >= 0) & (tids < vocab)
    if int(valid.sum()) == 0:
        return None
    lp = F.log_softmax(logits.float(), dim=-1)[0, gi[valid], tids[valid]]
    return -(probs[valid] * lp).sum() / denom


def save_cart(path):
    ck = torch.load(CART_INIT, map_location="cpu", weights_only=False)
    torch.save(
        {
            "trainable_keys": [
                p.detach().contiguous().cpu() for p in cache.trainable_keys
            ],
            "trainable_values": [
                p.detach().contiguous().cpu() for p in cache.trainable_values
            ],
            "frozen_keys": ck.get("frozen_keys"),
            "frozen_values": ck.get("frozen_values"),
        },
        path,
    )


# ---------------------------------------------------------------------------
# parity mode: legacy_raw == unique + explicit anchors, on the real model
# ---------------------------------------------------------------------------


def cart_grads():
    return [
        p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
        for p in cache.parameters()
        if p.requires_grad
    ]


def run_parity():
    report = []
    for i in sched_ids[:2]:
        results = {}
        for mode in ("legacy_raw", "legacy_grouped_replay"):
            for p in cache.parameters():
                p.grad = None
            loss = arm_loss(i, mode)
            assert loss is not None
            loss.backward()
            results[mode] = (float(loss.detach()), cart_grads())
        l_raw, g_raw = results["legacy_raw"]
        l_grp, g_grp = results["legacy_grouped_replay"]
        loss_rel = abs(l_raw - l_grp) / max(abs(l_raw), 1e-12)
        grad_max_rel = 0.0
        for a, b in zip(g_raw, g_grp):
            denom = a.abs().max().clamp_min(1e-12)
            grad_max_rel = max(grad_max_rel, float((a - b).abs().max() / denom))
        report.append(
            dict(
                element=i,
                loss_raw=l_raw,
                loss_grouped=l_grp,
                loss_rel=loss_rel,
                grad_max_rel=grad_max_rel,
            )
        )
        print(
            f"[parity] element {i}: loss {l_raw:.6f} vs {l_grp:.6f} "
            f"(rel {loss_rel:.2e}), grad max rel {grad_max_rel:.2e}"
        )
    ok = all(r["loss_rel"] < 1e-4 and r["grad_max_rel"] < 1e-2 for r in report)
    out = Path(OUT_DIR) / "parity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(ok=ok, results=report), indent=1))
    print(f"PARITY_{'OK' if ok else 'FAIL'} {out}")
    if not ok:
        sys.exit(1)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def main():
    if ARM == "parity":
        run_parity()
        return
    mode = ARM_TO_MODE[ARM]
    out_dir = Path(OUT_DIR) / ARM
    out_dir.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(
        cache.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.0
    )
    if OPT_INIT:
        if Path(OPT_INIT).is_file():
            opt.load_state_dict(torch.load(OPT_INIT, weights_only=False))
            # A saved state_dict carries param_groups, so a stale file
            # would silently reinstate the learning rate it was written
            # with while the manifest still recorded the configured one.
            # The shared state exists to equalize MOMENTS across arms,
            # not hyperparameters: reassert those from config and fail
            # loudly if the file disagrees.
            stale = [
                (g.get("lr"), g.get("betas"), g.get("weight_decay"))
                for g in opt.param_groups
                if g.get("lr") != LR
                or tuple(g.get("betas", ())) != (0.9, 0.95)
                or g.get("weight_decay") != 0.0
            ]
            for g in opt.param_groups:
                g["lr"] = LR
                g["betas"] = (0.9, 0.95)
                g["weight_decay"] = 0.0
            print(
                f"[ctrl-screen] restored optimizer state {OPT_INIT}"
                + (f" (overrode stale hyperparameters {stale})" if stale else "")
            )
            assert not stale, (
                "optimizer-state file was written under different "
                f"hyperparameters {stale}; delete it and refreeze so every "
                "arm starts from one declared state"
            )
        else:
            Path(OPT_INIT).parent.mkdir(parents=True, exist_ok=True)
            torch.save(opt.state_dict(), OPT_INIT)
            print(f"[ctrl-screen] saved zero-moment optimizer state {OPT_INIT}")

    run = dict(
        arm=ARM,
        mode=mode,
        model=MODEL,
        patient=PATIENT,
        parquet=DATA_PARQUET,
        parquet_sha256=schedule["parquet_sha256"],
        cart_init=CART_INIT,
        cart_sha256=schedule["cart_sha256"],
        opt_init=OPT_INIT,
        opt_sha256=(
            sha256_file(OPT_INIT) if OPT_INIT and Path(OPT_INIT).is_file() else None
        ),
        schedule_json=SCHEDULE_JSON,
        schedule_sha256=sha256_file(SCHEDULE_JSON),
        target_schema=TARGET_SCHEMA_VERSION,
        flatten_mode=flatten_mode(),
        relevance_map=RELEVANCE_MAP or None,
        keep_frac=KEEP_FRAC,
        select_invert=SELECT_INVERT,
        transform_hash=transform_hash(
            mode, scale=A2_SCALE, content_scale=CONTENT_SCALE
        ),
        a2_scale=A2_SCALE,
        content_scale=CONTENT_SCALE,
        content_report=content_report,
        steps=STEPS,
        accum=ACCUM,
        lr=LR,
        seed=SEED,
        history=[],
        cost=dict(train_s=0.0, train_tokens=0),
    )

    if 0 in CHECKPOINT_AT:
        save_cart(out_dir / f"{PATIENT}_step0.pt")
    t0 = time.time()
    for step in range(1, STEPS + 1):
        opt.zero_grad()
        accum_loss, n_used = 0.0, 0
        for i in schedule["schedule"][step - 1]:
            loss = arm_loss(i, mode)
            if loss is None:
                continue
            run["cost"]["train_tokens"] += int(dataset.elements[i].input_ids.shape[0])
            (loss / ACCUM).backward()
            accum_loss += float(loss.detach()) / ACCUM
            n_used += 1
        grad_norm = float(torch.nn.utils.clip_grad_norm_(cache.parameters(), CLIP))
        opt.step()
        run["history"].append(
            dict(
                step=step,
                loss=accum_loss,
                used=n_used,
                grad_norm=grad_norm,
                # clipping equalizes gradient magnitude across arms; a
                # step where one arm clips and another does not is the
                # only place a pure loss-scale difference can survive
                clipped=grad_norm > CLIP,
            )
        )
        print(
            f"[ctrl-screen] {ARM} step {step:2d} loss={accum_loss:.4f} "
            f"grad={grad_norm:.3f} wall={time.time() - t0:.0f}s",
            flush=True,
        )
        if step in CHECKPOINT_AT:
            save_cart(out_dir / f"{PATIENT}_step{step}.pt")
    run["cost"]["train_s"] = time.time() - t0
    run["cost"]["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    with open(out_dir / "run.json", "w") as f:
        json.dump(run, f, indent=1)
    print(f"CTRL_SCREEN_DONE arm={ARM} out={out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a Latent Compaction memory for a cartridge document, with
switches that remove one part of the recipe at a time.

Latent Compaction (arXiv 2610.20582; code github.com/thunlp/
latent-compaction) compacts a document's key-value cache in two stages.
It keeps the highest-scoring source entries under KVZip+ reconstruction
scores (preservers, 3/4 of the budget, uneven across heads), seeds the
other quarter from high-attention anchors (first position forced in)
and farthest-key diversifiers, and gives every entry a scalar logit
bias. Stage one fits each layer alone, preservers frozen, so attention
over the memory reproduces the full document's attention outputs and
log-normalizers on teacher queries. Stage two refines everything
jointly through the frozen reader on teacher rollouts, with a top-20
plus tail KL and the teacher's greedy token.

This script runs the authors' functions in their order, with their
defaults (configs/paper.json), on one record wrapped in the system-
prompt layout the native cartridge distills, and saves the memory
after each stage. The switches below change one part each; with all
at their defaults the run is the authors' recipe. Their stage-one
loop is copied here (as ``stage_one``) only so the switches can reach
its loss and its parameter list; with the switches at their defaults
it is the same computation.

Env:
    LC_SRC         checkout of thunlp/latent-compaction
    RECORD         record text; DOC_LAYOUT=system wraps it as
                   <|im_start|>system\\n...<|im_end|>\\n (default), raw
                   leaves it as is
    BUDGET         total entries per head (the native cartridge length)
    SOFT           learned entries per head (default BUDGET // 4)
    STAGE1_STEPS   per-layer updates (default 1000)
    STAGE2_STEPS   joint updates (default 1000)
    QUERIES        generic (the authors' 480 prompts, default) or
                   selfstudy (first user turns of PARQUET, chat template)
    PARQUET        Self-Study parquet for QUERIES=selfstudy
    N_PROMPTS      prompts for QUERIES=selfstudy (default 480)
    LC_BIAS        1 (default) learn the per-entry bias; 0 keeps it zero
    LC_LOGZ        1 (default) fit log-normalizers in stage one; 0 drops it
    LC_FREEZE      1 (default) preservers frozen in stage one; 0 trains them
    SEED           default 401 (configs/paper.json)
    MODEL, DEVICE  reader and device
    OUT_DIR        writes stage1.pt, stage2.pt (authors' format) and
                   meta.json with per-phase seconds and peak memory

Usage:
    python cas_lc_fit.py
"""

import json
import os
import sys
import time

import torch
from torch.nn import functional as F

sys.path.insert(0, os.environ["LC_SRC"])

from latent_compaction.memory import CompactedMemory  # noqa: E402
from latent_compaction.methods import latent as L  # noqa: E402
from latent_compaction.models import (  # noqa: E402
    DocumentCache,
    load_model,
    resolve_model_name,
)
from latent_compaction.prompts import SOURCE_PROMPTS  # noqa: E402
from latent_compaction.selection import (  # noqa: E402
    capture_kvzip_scores,
    gather_preservers,
    initialize_soft_slots,
    select_preservers,
)

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
DEVICE = os.environ.get("DEVICE", "cuda")
BIAS = os.environ.get("LC_BIAS", "1") == "1"
LOGZ = os.environ.get("LC_LOGZ", "1") == "1"
FREEZE = os.environ.get("LC_FREEZE", "1") == "1"


def _stage_two_no_bias(self):
    for name, p in self.named_parameters():
        p.requires_grad_(not name.endswith("_bias"))


if not BIAS:
    CompactedMemory.stage_two = _stage_two_no_bias


def stage_one(memory, queries, full_keys, full_values, *, steps, batch_size, lr, seed):
    """The authors' fit_attention_stage, with the three switches."""
    memory.stage_one()
    params = [memory.soft_keys, memory.soft_values]
    if not FREEZE:
        for p in (memory.preserver_keys, memory.preserver_values):
            p.requires_grad_(True)
        params += [memory.preserver_keys, memory.preserver_values]
    if BIAS:
        params.append(memory.soft_bias)
        if not FREEZE:
            memory.preserver_bias.requires_grad_(True)
            params.append(memory.preserver_bias)
    else:
        memory.soft_bias.requires_grad_(False)
    device = full_keys.device
    queries = queries.to(device)
    targets, target_log_z = L.exact_attention_targets(
        queries, full_keys, full_values, scaling=memory.scaling
    )
    value_mean = targets.mean((-2, -1), keepdim=True)
    value_scale = targets.std((-2, -1), keepdim=True).clamp_min(1e-4)
    log_mean = target_log_z.mean(-1, keepdim=True)
    log_scale = target_log_z.std(-1, keepdim=True).clamp_min(1e-4)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0)
    gen = torch.Generator(device=device).manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(
            queries.shape[-2],
            (min(batch_size, queries.shape[-2]),),
            device=device,
            generator=gen,
        )
        value, log_z = memory(queries.index_select(-2, idx))
        expected = targets.index_select(-2, idx)
        loss = F.mse_loss(
            (value.float() - value_mean) / value_scale,
            (expected - value_mean) / value_scale,
        )
        if LOGZ:
            exp_z = target_log_z.index_select(-1, idx)
            loss = loss + F.mse_loss(
                (log_z - log_mean) / log_scale, (exp_z - log_mean) / log_scale
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    return float(loss)


def document_text():
    text = open(os.environ["RECORD"]).read()
    if os.environ.get("DOC_LAYOUT", "system") == "system":
        text = f"<|im_start|>system\n{text}<|im_end|>\n"
    return text


def prompts(tok):
    kind = os.environ.get("QUERIES", "generic")
    if kind == "generic":
        return list(SOURCE_PROMPTS)
    import pandas as pd

    df = pd.read_parquet(os.environ["PARQUET"])
    firsts = []
    for msgs in df["messages"]:
        u = next(m["content"] for m in msgs if m["role"] == "user")
        if u not in firsts:
            firsts.append(u)
    g = torch.Generator().manual_seed(int(os.environ.get("SEED", "401")))
    n = min(int(os.environ.get("N_PROMPTS", "480")), len(firsts))
    pick = torch.randperm(len(firsts), generator=g)[:n].tolist()
    return [
        tok.apply_chat_template(
            [{"role": "user", "content": firsts[i]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for i in pick
    ]


def main():
    out = os.environ["OUT_DIR"]
    os.makedirs(out, exist_ok=True)
    seed = int(os.environ.get("SEED", "401"))
    budget = int(os.environ["BUDGET"])
    soft = int(os.environ.get("SOFT", budget // 4))
    cfg = L.FitConfig(
        budget=budget,
        soft_slots=soft,
        stage1_steps=int(os.environ.get("STAGE1_STEPS", "1000")),
        stage2_steps=int(os.environ.get("STAGE2_STEPS", "1000")),
        seed=seed,
    )
    secs = {}
    t = time.time()
    model, tok = load_model(MODEL, DEVICE)
    secs["model_load"] = time.time() - t
    torch.cuda.reset_peak_memory_stats()
    doc = document_text()
    ids = tok(doc, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
    D = ids.shape[-1]
    torch.manual_seed(seed)
    full = DocumentCache(model.config)
    t = time.time()
    scores = capture_kvzip_scores(model, ids, full, (budget - soft) / D)
    selected = select_preservers(scores, budget - soft)
    torch.cuda.synchronize()
    secs["prefill_and_kvzip"] = time.time() - t
    t = time.time()
    plist = prompts(tok)
    banks, rollouts = L.collect_teacher_rollouts(
        model, tok, full, prompts=plist, max_tokens=cfg.rollout_tokens
    )
    secs["teacher_rollouts"] = time.time() - t
    t = time.time()
    memories, s1_loss, counts = {}, {}, {}
    for li, layer in enumerate(full.layers):
        q = banks[li]
        if q.shape[-2] > cfg.stage1_queries:
            q = q.index_select(
                -2, torch.linspace(0, q.shape[-2] - 1, cfg.stage1_queries).long()
            )
        q = q.to(DEVICE)
        pk, pv, mask = gather_preservers(layer.keys, layer.values, selected[li, 0])
        attn = model.model.layers[li].self_attn
        sk, sv = initialize_soft_slots(
            layer.keys, layer.values, q, slots=soft, scaling=float(attn.scaling)
        )
        mem = CompactedMemory(
            pk,
            pv,
            mask,
            sk,
            sv,
            query_heads=attn.config.num_attention_heads,
            scaling=float(attn.scaling),
        ).to(DEVICE)
        s1_loss[li] = stage_one(
            mem,
            q,
            layer.keys,
            layer.values,
            steps=cfg.stage1_steps,
            batch_size=cfg.stage1_batch_size,
            lr=cfg.stage1_learning_rate,
            seed=seed + li,
        )
        counts[li] = mask.sum(-1).tolist()
        memories[li] = mem
    torch.cuda.synchronize()
    secs["stage1"] = time.time() - t
    mid = resolve_model_name(MODEL)
    L.save_memory(os.path.join(out, "stage1.pt"), memories, D, model_id=mid)
    del full, banks
    torch.cuda.empty_cache()
    t = time.time()
    L.fit_distillation_stage(
        model,
        memories,
        D,
        rollouts,
        steps=cfg.stage2_steps,
        batch_size=cfg.stage2_batch_size,
        learning_rate=cfg.stage2_learning_rate,
        seed=seed,
    )
    torch.cuda.synchronize()
    secs["stage2"] = time.time() - t
    L.save_memory(os.path.join(out, "stage2.pt"), memories, D, model_id=mid)
    bias_abs = [
        float(
            torch.cat([m.preserver_bias[m.preserver_mask], m.soft_bias.flatten()])
            .abs()
            .mean()
        )
        for m in memories.values()
    ]
    meta = dict(
        model=mid,
        record=os.path.abspath(os.environ["RECORD"]),
        doc_layout=os.environ.get("DOC_LAYOUT", "system"),
        document_tokens=D,
        budget=budget,
        soft=soft,
        preserver_counts_per_layer_head=counts,
        config=vars(cfg),
        queries=os.environ.get("QUERIES", "generic"),
        n_prompts=len(plist),
        switches=dict(bias=BIAS, logz=LOGZ, freeze=FREEZE),
        stage1_last_loss=s1_loss,
        mean_abs_bias_per_layer=bias_abs,
        seconds=secs,
        prep_seconds_total=sum(v for k, v in secs.items() if k != "model_load"),
        peak_gib=torch.cuda.max_memory_allocated() / 2**30,
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
    )
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(
        f"LC_FIT_DONE {out} {json.dumps(secs)} peak={meta['peak_gib']:.1f}GiB",
        flush=True,
    )


if __name__ == "__main__":
    main()

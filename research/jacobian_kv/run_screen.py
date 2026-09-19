# SPDX-License-Identifier: GPL-2.0
"""Stage 0: does a receiver-sensitive second moment rank real cache damage?

The free kill test.  Nothing downstream in the plan happens if this fails, and
failing it cheaply is a success of the plan rather than a disappointment.

What it does, in one sentence: build a bank of controlled KV-cache errors whose
*size* is held fixed within groups, measure what each one actually costs the
frozen receiver by exact teacher-forced KL, then ask which candidate metric
ranks those costs best.

Six metrics are compared, per tensor kind and per model:

  ``mse``              raw cache reconstruction error -- the metric to beat
  ``wo``               the per-head output-projection metric already tested and
                       already beaten by plain SVD in this program
  ``attn_local``       CacheBridge-style attention-local sensitivity
  ``meanj``            the signed mean-Jacobian control, built by averaging the
                       VJP across contexts *before* squaring
  ``jtfj``             the proposed second moment, fitted on calibration
                       contexts and applied to held-out ones -- the deployable
                       arm, and the one the gate judges
  ``jtfj_oracle``      the same second moment computed on the scored context
                       itself.  Not deployable: it needs a backward pass at
                       scoring time.  It is the ceiling, and it is what
                       separates "the receiver signal does not exist" from "it
                       exists but does not survive being averaged into a global
                       metric" -- two very different negative results.

Two views of every correlation are reported.  Pooled across groups, error size
varies and ``mse`` is a strong baseline.  Within a matched-norm group, size is
constant, ``mse`` has exactly zero ranking power, and what remains is the only
question that matters: given the bytes already spent, does the metric know
which direction to spend them in.

Env: HIP_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
Run:
    python -m research.jacobian_kv.run_screen \\
        --model Qwen/Qwen3-0.6B --ctx 512 --seed 0 --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.jacobian_kv import artifacts as art  # noqa: E402
from research.jacobian_kv.evaluate import (  # noqa: E402
    argmax_flip_rate,
    delta_nll,
    evaluate_gate,
    kendall_distance,
    spearman,
    teacher_forced_kl,
)
from research.jacobian_kv.perturb import build_bank, matched_norm_report  # noqa: E402
from research.jacobian_kv.sensitivity import (  # noqa: E402
    fisher_probe,
    freeze_model,
    score_attention_local,
    score_mse,
    vjp_from_probe,
)

PROBE_COUNTS = (1, 2, 4, 8)
MAX_PROBES = max(PROBE_COUNTS)


# ---------------------------------------------------------------------------
# cache helpers
#
# DynamicCache.update rebinds (`self.keys = torch.cat([...])`) rather than
# mutating in place, so one set of base tensors can back thousands of
# perturbation forwards with a single copy for the perturbed layer.  Cloning
# the whole cache per perturbation would move ~120 MiB each time for nothing.
# ---------------------------------------------------------------------------


def make_cache(keys, values):
    from transformers import DynamicCache

    c = DynamicCache()
    for i, (k, v) in enumerate(zip(keys, values)):
        c.update(k, v, i)
    return c


def base_cache_tensors(model, input_ids):
    from transformers import DynamicCache

    c = DynamicCache()
    with torch.no_grad():
        model(input_ids=input_ids, past_key_values=c, use_cache=True)
    keys = [layer.keys.detach() for layer in c.layers]
    values = [layer.values.detach() for layer in c.layers]
    return keys, values


def forward_continuation(model, keys, values, cont_ids, prompt_len, want_attn=False):
    cache = make_cache(keys, values)
    total = prompt_len + cont_ids.shape[1]
    attn = torch.ones(1, total, dtype=torch.long, device=cont_ids.device)
    pos = torch.arange(prompt_len, total, device=cont_ids.device).unsqueeze(0)
    out = model(
        input_ids=cont_ids,
        attention_mask=attn,
        position_ids=pos,
        past_key_values=cache,
        use_cache=True,
        output_attentions=want_attn,
    )
    return out


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def wikitext_chunks(tok, n_chunks, ctx, cont_len, seed):
    """Disjoint fixed-length chunks from wikitext-2 test.  Offsets recorded."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    span = ctx + cont_len
    n_avail = ids.shape[0] // span
    if n_avail < n_chunks:
        raise RuntimeError(f"only {n_avail} chunks of {span} tokens available")
    rng = np.random.default_rng(seed)
    picks = rng.choice(n_avail, size=n_chunks, replace=False)
    return [(int(p), ids[p * span : (p + 1) * span].clone()) for p in sorted(picks)]


NEEDLE_FACTS = [
    ("the harbour master at Vardo", "Ingeborg Rasmussen"),
    ("the reserve fuel pump code", "four seven two nine"),
    ("the winter ferry berth number", "berth nineteen"),
    ("the lighthouse relief keeper", "Tomas Lindqvist"),
    ("the cold-store access phrase", "northern gannet"),
]

FILLER = (
    "The coastal route has been surveyed many times, and the charts are "
    "reissued whenever a channel shifts. Crews rotate at the larger ports. "
    "Cargo manifests are checked twice, once at loading and once on arrival. "
)


def needle_chunks(tok, n_chunks, ctx, cont_len, seed):
    """Long-range retrieval contexts: a fact planted early, asked about late.

    Pretraining-like text under-weights the one thing a cache is for.  These
    contexts make retrieval damage visible in the same KL column.
    """
    rng = np.random.default_rng(seed + 9001)
    out = []
    for i in range(n_chunks):
        subject, answer = NEEDLE_FACTS[int(rng.integers(len(NEEDLE_FACTS)))]
        depth = float(rng.uniform(0.05, 0.4))
        fact = f"Note carefully: {subject} is {answer}. "
        body = FILLER * 200
        ids_fact = tok(fact, add_special_tokens=False).input_ids
        ids_body = tok(body, add_special_tokens=False).input_ids
        need = ctx - len(ids_fact)
        at = max(0, int(depth * need))
        ids = ids_body[:at] + ids_fact + ids_body[at : at + (need - at)]
        ids = ids[:ctx]
        q = f"\nQuestion: what is {subject}?\nAnswer: {answer}"
        ids_q = tok(q, add_special_tokens=False).input_ids[:cont_len]
        while len(ids_q) < cont_len:
            ids_q = ids_q + ids_q
        ids_q = ids_q[:cont_len]
        out.append((10_000 + i, torch.tensor(ids + ids_q, dtype=torch.long)))
    return out


# ---------------------------------------------------------------------------
# metric fitting on calibration contexts
# ---------------------------------------------------------------------------


def fit_global_metrics(model, tok, chunks, ctx, cont_len, layers, device, seed, log):
    """Accumulate the per-(layer, kind, head) second moment and its control.

    ``jtfj`` averages ``u u'`` over contexts, probes and positions.  ``meanj``
    holds the probe directions *fixed across contexts* and averages ``u``
    before squaring, which is what builds ``E[J]' F E[J]`` -- the object that
    loses any direction cancelling across contexts.  The contrast between them
    is the plan's actual question.
    """
    acc2: dict = {}
    accm: dict = {}
    accpos: dict = {}
    counts: dict = {}
    fixed_probes = None
    t0 = time.time()
    tokens_seen = 0

    for ci, (_, ids) in enumerate(chunks):
        ids = ids.to(device)
        prompt, cont = ids[:ctx].unsqueeze(0), ids[ctx : ctx + cont_len].unsqueeze(0)
        keys, values = base_cache_tensors(model, prompt)
        leaves = {}
        kk = [k.clone() for k in keys]
        vv = [v.clone() for v in values]
        for li in layers:
            kk[li] = kk[li].detach().clone().requires_grad_(True)
            vv[li] = vv[li].detach().clone().requires_grad_(True)
            leaves[(li, "k")] = kk[li]
            leaves[(li, "v")] = vv[li]

        with torch.enable_grad():
            out = forward_continuation(model, kk, vv, cont, ctx)
            logits = out.logits
            if fixed_probes is None:
                g0 = torch.Generator(device=device).manual_seed(seed + 555)
                fixed_probes = [
                    fisher_probe(logits, g0).detach() for _ in range(MAX_PROBES)
                ]
            g = torch.Generator(device=device).manual_seed(seed * 1000 + ci)
            for p in range(MAX_PROBES):
                retain = p < MAX_PROBES - 1
                z = fisher_probe(logits, g)
                u = vjp_from_probe(logits, leaves, z, retain=True)
                for key, ug in u.items():
                    x = ug.float()[0]  # [H, T, D]
                    acc2.setdefault((key, p), 0)
                    acc2[(key, p)] = acc2[(key, p)] + torch.einsum("htd,hte->hde", x, x)
                    accpos.setdefault((key, p), 0)
                    accpos[(key, p)] = accpos[(key, p)] + x.pow(2).sum(dim=-1)
                    counts[key] = counts.get(key, 0) + x.shape[1]
                zf = fixed_probes[p].to(logits.dtype)
                um = vjp_from_probe(logits, leaves, zf, retain=retain)
                for key, ug in um.items():
                    xm = ug.float()[0].mean(dim=1)  # [H, D], averaged over positions
                    accm.setdefault((key, p), 0)
                    accm[(key, p)] = accm[(key, p)] + xm
        tokens_seen += ctx + cont_len
        del kk, vv, leaves, out, logits
        if (ci + 1) % 8 == 0:
            log(f"  fitted on {ci + 1}/{len(chunks)} calibration contexts")

    n_ctx = len(chunks)
    # jtfj: cumulative over probe counts so 1/2/4/8 are all available
    G2: dict = {}
    for pc in PROBE_COUNTS:
        for key in {k for (k, _) in acc2}:
            tot = sum(acc2[(key, p)] for p in range(pc))
            G2[(key, pc)] = tot / float(counts[key] / MAX_PROBES * pc)
    Gm: dict = {}
    for key in {k for (k, _) in accm}:
        ubar = torch.stack([accm[(key, p)] / n_ctx for p in range(MAX_PROBES)], 0)
        Gm[key] = torch.einsum("phd,phe->hde", ubar, ubar) / MAX_PROBES
    # Positional energy profile, normalised to mean one per head.  Contexts are
    # a fixed length here, so absolute position transfers between them; this is
    # the cheapest transferable thing the Jacobian knows that a position-uniform
    # metric throws away, and isolating it is what makes a later win
    # attributable to the matrix shape rather than to the position weighting.
    W: dict = {}
    for pc in PROBE_COUNTS:
        for key in {k for (k, _) in accpos}:
            tot = sum(accpos[(key, p)] for p in range(pc)) / float(n_ctx * pc)
            W[(key, pc)] = tot / tot.mean(dim=-1, keepdim=True).clamp(min=1e-30)
    return G2, Gm, W, {"fit_s": time.time() - t0, "fit_tokens": tokens_seen}


def quad_score(error: torch.Tensor, G: torch.Tensor, w=None) -> float:
    """``sum_{h,t} w_ht * e_ht' G_h e_ht`` -- the block-diagonal receiver score.

    ``G`` is the per-head *shape* of the receiver metric; ``w`` is an optional
    per-(head, position) *scale*.  Keeping them apart is deliberate.  The shape
    is what the second moment uniquely claims to contribute; the scale is
    available from cheaper sources -- a fitted positional profile, or the
    receiver's own attention mass -- so separating them makes any win
    attributable to one or the other rather than only to their product.
    """
    e = error.float()[0]  # [H, T, D]
    q = torch.einsum("htd,hde,hte->ht", e, G, e)
    if w is not None:
        q = q * w[..., : q.shape[-1]]
    return float(q.sum())


def energy_score(error: torch.Tensor, w: torch.Tensor) -> float:
    """``sum_{h,t} w_ht ||e_ht||^2`` -- a positional profile with no matrix shape.

    The control that decides whether the second moment's matrix structure earns
    its keep, or whether a scalar per-position weight was the whole story.
    """
    e = error.float()[0]
    return float((e.pow(2).sum(-1) * w[..., : e.shape[1]]).sum())


def attention_position_weight(attn_layer, n_kv_heads, n_positions):
    """Per-(kv head, position) attention mass, normalised to mean one.

    Free at encode time -- it needs no backward pass -- which is what makes the
    ``jtfj_attn`` arm a deployable instantiation rather than an oracle.
    """
    a = attn_layer.float().mean(dim=2)[0]  # [n_q, T_kv]
    if a.shape[0] != n_kv_heads:
        grp = a.shape[0] // n_kv_heads
        a = a.reshape(n_kv_heads, grp, a.shape[-1]).mean(dim=1)
    a = a[:, :n_positions]
    return a / a.mean(dim=-1, keepdim=True).clamp(min=1e-30)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None, help="pinned snapshot hash")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--cont-len", type=int, default=64)
    ap.add_argument("--calib", type=int, default=32)
    ap.add_argument("--eval", type=int, default=12)
    ap.add_argument("--layers", default="", help="comma list; default = 5 spread")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--context-source", default="wikitext", choices=["wikitext", "needle"]
    )
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument(
        "--analyse-only",
        action="store_true",
        help="recompute the gates from an existing perturbations.jsonl; "
        "measured data is never re-measured just to change a statistic",
    )
    args = ap.parse_args()

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    if args.analyse_only:
        return _analyse_only(args, log)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    t_start = time.time()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation="eager", revision=args.revision
    ).to(device)
    freeze_model(model)
    cfg = model.config
    n_layer = int(cfg.num_hidden_layers)
    n_q = int(cfg.num_attention_heads)
    n_kv = int(getattr(cfg, "num_key_value_heads", n_q))
    head_dim = int(getattr(cfg, "head_dim", None) or cfg.hidden_size // n_q)
    group = n_q // n_kv

    layers = (
        [int(x) for x in args.layers.split(",") if x != ""]
        if args.layers
        else sorted({0, n_layer // 4, n_layer // 2, (3 * n_layer) // 4, n_layer - 1})
    )
    log(f"{args.model}: {n_layer}L n_q={n_q} n_kv={n_kv} D={head_dim}; layers {layers}")

    src = wikitext_chunks if args.context_source == "wikitext" else needle_chunks
    all_chunks = src(tok, args.calib + args.eval, args.ctx, args.cont_len, args.seed)
    calib = all_chunks[: args.calib]
    held = all_chunks[args.calib :]
    calib_idx = tuple(int(i) for i, _ in calib)
    eval_idx = tuple(int(i) for i, _ in held)
    assert not (set(calib_idx) & set(eval_idx)), "calibration and evaluation overlap"
    log(f"{len(calib)} calibration / {len(held)} held-out contexts, ctx={args.ctx}")

    run_id = args.run_id or (
        f"{args.model.replace('/', '_')}_ctx{args.ctx}_{args.context_source}_s{args.seed}"
    )
    out_dir = art.ensure_run_dir(args.out_dir, run_id)
    rows_path = os.path.join(out_dir, "perturbations.jsonl")

    # ---- fit the global metrics on calibration contexts ------------------
    log("fitting global receiver metrics on calibration contexts")
    G2, Gm, Wpos, fit_stats = fit_global_metrics(
        model, tok, calib, args.ctx, args.cont_len, layers, device, args.seed, log
    )
    log(f"fit done in {fit_stats['fit_s']:.1f}s")

    wo = {
        li: model.model.layers[li].self_attn.o_proj.weight.detach().float()
        for li in layers
    }

    # ---- score the held-out perturbation bank ----------------------------
    records = []
    writer = art.JsonlWriter(rows_path)
    eval_tokens = 0
    t_eval = time.time()
    try:
        for ci, (chunk_id, ids) in enumerate(held):
            ids = ids.to(device)
            prompt = ids[: args.ctx].unsqueeze(0)
            cont = ids[args.ctx : args.ctx + args.cont_len].unsqueeze(0)
            keys, values = base_cache_tensors(model, prompt)

            with torch.no_grad():
                clean_out = forward_continuation(
                    model, keys, values, cont, args.ctx, want_attn=True
                )
            clean_logits = clean_out.logits.detach().float()
            attns = [a.detach() for a in clean_out.attentions]
            del clean_out
            attn_w = {
                li: attention_position_weight(attns[li], n_kv, args.ctx)
                for li in layers
            }

            # zero-perturbation control: the numerical floor of this run
            with torch.no_grad():
                ctrl = forward_continuation(model, keys, values, cont, args.ctx)
            floor_kl = float(
                teacher_forced_kl(clean_logits, ctrl.logits.float()).mean()
            )
            del ctrl

            # oracle sketch for this context: one backward set, all layers
            kk = [k.clone() for k in keys]
            vv = [v.clone() for v in values]
            leaves = {}
            for li in layers:
                kk[li] = kk[li].detach().clone().requires_grad_(True)
                vv[li] = vv[li].detach().clone().requires_grad_(True)
                leaves[(li, "k")] = kk[li]
                leaves[(li, "v")] = vv[li]
            oracle: dict = {}
            with torch.enable_grad():
                o = forward_continuation(model, kk, vv, cont, args.ctx)
                g = torch.Generator(device=device).manual_seed(args.seed * 7717 + ci)
                for p in range(MAX_PROBES):
                    z = fisher_probe(o.logits, g)
                    u = vjp_from_probe(o.logits, leaves, z, retain=(p < MAX_PROBES - 1))
                    for key, ug in u.items():
                        oracle.setdefault(key, []).append(ug.float().reshape(-1).cpu())
            del kk, vv, leaves, o

            oracle_U = {k: torch.stack(v, 0) for k, v in oracle.items()}

            for li in layers:
                for kind in ("k", "v"):
                    block = (keys if kind == "k" else values)[li]
                    gen = torch.Generator(device=device).manual_seed(
                        args.seed * 104729 + ci * 1013 + li * 17 + (kind == "v")
                    )
                    bank = build_bank(
                        block,
                        kind=kind,
                        layer=li,
                        generator=gen,
                        context_tag=f"c{chunk_id}",
                    )
                    for pert in bank:
                        pk = list(keys)
                        pv = list(values)
                        if kind == "k":
                            pk[li] = block + pert.delta
                        else:
                            pv[li] = block + pert.delta
                        with torch.no_grad():
                            po = forward_continuation(model, pk, pv, cont, args.ctx)
                        pl = po.logits.float()
                        del po

                        kl = float(teacher_forced_kl(clean_logits, pl).mean())
                        dnll = float(delta_nll(clean_logits, pl, cont).mean())
                        flips = argmax_flip_rate(clean_logits, pl)
                        del pl

                        e = pert.delta
                        scores = {
                            "mse": score_mse(e),
                            "wo": _wo_score(e, wo[li], group, head_dim),
                            "attn_local": score_attention_local(e, attns[li], kind),
                            "attn_local_lin": score_attention_local(
                                e, attns[li], kind, power=1.0
                            ),
                            "meanj": quad_score(e, Gm[(li, kind)]),
                        }
                        for pc in PROBE_COUNTS:
                            G = G2[((li, kind), pc)]
                            wfit = Wpos[((li, kind), pc)]
                            scores[f"jtfj_p{pc}"] = quad_score(e, G)
                            scores[f"jtfj_pos_p{pc}"] = quad_score(e, G, wfit)
                            scores[f"jtfj_attn_p{pc}"] = quad_score(e, G, attn_w[li])
                            scores[f"pos_energy_p{pc}"] = energy_score(e, wfit)
                        U = oracle_U[(li, kind)]
                        ev = e.float().reshape(-1).cpu()
                        for pc in PROBE_COUNTS:
                            scores[f"jtfj_oracle_p{pc}"] = float(
                                (U[:pc] @ ev).pow(2).mean()
                            )

                        row = {
                            "context": int(chunk_id),
                            "layer": li,
                            "kind": kind,
                            "family": pert.family,
                            "params": pert.params,
                            "norm_group": pert.norm_group,
                            "frob": pert.frob,
                            "exact_kl": kl,
                            "delta_nll": dnll,
                            "argmax_flip_rate": flips,
                            "floor_kl": floor_kl,
                            "scores": scores,
                        }
                        writer.write(row)
                        records.append(row)
                        eval_tokens += args.ctx + args.cont_len
            del attns, attn_w, clean_logits, oracle, oracle_U
            torch.cuda.empty_cache() if device == "cuda" else None
            log(
                f"  context {ci + 1}/{len(held)} done "
                f"({len(records)} perturbations, floor KL {floor_kl:.2e})"
            )
    finally:
        writer.close()

    eval_s = time.time() - t_eval
    log(f"scored {len(records)} perturbations in {eval_s:.1f}s")

    # ---- gates -----------------------------------------------------------
    summary = analyse(records, args, model_name=args.model)
    summary["fit_stats"] = fit_stats
    summary["matched_norm_check"] = _norm_check(records)

    wall_h = (time.time() - t_start) / 3600.0
    man = art.JKVManifest(
        run_id=run_id,
        stage="stage0",
        target_model_id=args.model,
        target_model_revision=args.revision or _resolve_revision(args.model),
        dtype=args.dtype,
        attn_backend="eager",
        device=device,
        rope_status="post_rope",
        rope_theta_target=float(getattr(cfg, "rope_theta", 0.0) or 0.0),
        rope_scaling_target=getattr(cfg, "rope_scaling", None),
        kv_split="both",
        selected_layers=tuple(layers),
        probe_count=MAX_PROBES,
        measurement_level="fake_quant_teacher_forced",
        calib_dataset=f"{args.context_source}:calib",
        calib_indices=calib_idx,
        eval_dataset=f"{args.context_source}:eval",
        eval_indices=eval_idx,
        seed=args.seed,
        versions=art.versions(),
        gpu_model=(torch.cuda.get_device_name(0) if device == "cuda" else "cpu"),
        gpu_hours=wall_h,
        wall_hours=wall_h,
        evaluated_tokens=eval_tokens,
        observed_tokens_per_s=eval_tokens / max(eval_s, 1e-9),
        hourly_price_usd=0.0,
        billed_cost_usd=0.0,
        peak_mem_bytes=(
            int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
        ),
        cmdline=" ".join(sys.argv),
    )
    man.code_commit, man.code_dirty = art.git_state()
    man.config_hash = art.config_hash(vars(args))
    art.write_json(os.path.join(out_dir, "manifest.json"), man.finalize())
    art.write_json(os.path.join(out_dir, "summary.json"), summary)

    _print_table(summary, log)
    return 0


def _analyse_only(args, log):
    """Redo the statistics on rows already measured, touching no GPU."""
    run_id = args.run_id
    if not run_id:
        raise SystemExit("--analyse-only needs --run-id")
    out_dir = os.path.join(args.out_dir, run_id)
    rows_path = os.path.join(out_dir, "perturbations.jsonl")
    with open(rows_path) as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    log(f"re-analysing {len(records)} rows from {rows_path}")
    summary = analyse(records, args, model_name=args.model)
    summary["matched_norm_check"] = _norm_check(records)
    summary["reanalysed_from"] = rows_path
    art.write_json(os.path.join(out_dir, "summary.json"), summary)
    _print_table(summary, log)
    return 0


def _wo_score(e, w_o, group, head_dim):
    """Expand a kv-head error to its query heads, then push through o_proj."""
    b, h, t, d = e.shape
    eq = e.float().repeat_interleave(group, dim=1)  # [B, n_q, T, D]
    flat = eq.permute(0, 2, 1, 3).reshape(b, t, -1)
    return float((flat @ w_o.transpose(0, 1)).pow(2).mean())


def _resolve_revision(model_id):
    """Best-effort pinned revision from the local snapshot path."""
    try:
        from huggingface_hub import snapshot_download

        p = snapshot_download(model_id, local_files_only=True)
        return os.path.basename(p.rstrip("/"))
    except Exception:  # noqa: BLE001
        return "local-snapshot-unresolved"


def _norm_check(records):
    groups: dict = {}
    for r in records:
        groups.setdefault(r["norm_group"], []).append(r["frob"])
    worst = 0.0
    for norms in groups.values():
        lo, hi = min(norms), max(norms)
        worst = max(worst, (hi - lo) / max(lo, 1e-30))
    return {"n_groups": len(groups), "worst_relative_norm_spread": worst}


BASELINES = ("mse", "wo", "attn_local", "attn_local_lin")


def analyse(records, args, model_name):
    """Pooled and within-matched-norm gates, per tensor kind."""
    out = {
        "model": model_name,
        "ctx": args.ctx,
        "seed": args.seed,
        "context_source": args.context_source,
        "n_records": len(records),
        "cells": {},
    }
    score_names = sorted(records[0]["scores"].keys()) if records else []

    for kind in ("k", "v"):
        rs = [r for r in records if r["kind"] == kind]
        if len(rs) < 8:
            continue
        truth = [r["exact_kl"] for r in rs]
        scores = {n: [r["scores"][n] for r in rs] for n in score_names}

        cell = {"n": len(rs), "pooled": {}, "within_group": {}}
        cell["pooled"]["rho"] = {n: spearman(scores[n], truth) for n in score_names}
        cell["pooled"]["floor_kl_max"] = max(r["floor_kl"] for r in rs)
        cell["pooled"]["exact_kl_median"] = float(np.median(truth))

        for cand in (
            "jtfj_p4",
            "jtfj_p8",
            "jtfj_pos_p4",
            "jtfj_pos_p8",
            "jtfj_attn_p4",
            "jtfj_attn_p8",
            "pos_energy_p8",
            "jtfj_oracle_p4",
            "jtfj_oracle_p8",
            "meanj",
        ):
            if cand not in scores:
                continue
            pv = {}
            for pc in (4, 8):
                name = cand.replace("p8", f"p{pc}").replace("p4", f"p{pc}")
                if name in scores:
                    pv[pc] = scores[name]
            g = evaluate_gate(
                scores,
                truth,
                model=model_name,
                kind=kind,
                candidate=cand,
                baselines=BASELINES,
                probe_variants=(pv if len(pv) == 2 else None),
                n_boot=args.n_boot,
                seed=args.seed,
            )
            cell["pooled"][cand] = g.to_dict()

        # within-group: rank inside each matched-norm group, then pool the
        # per-group correlations.  mse is constant here, so it cannot compete.
        by_group: dict = {}
        for r in rs:
            by_group.setdefault(r["norm_group"], []).append(r)
        per_group = {n: [] for n in score_names}
        for grs in by_group.values():
            if len(grs) < 5:
                continue
            gt = [r["exact_kl"] for r in grs]
            for n in score_names:
                v = spearman([r["scores"][n] for r in grs], gt)
                if not math.isnan(v):
                    per_group[n].append(v)
        cell["within_group"] = {
            "n_groups": sum(1 for g in by_group.values() if len(g) >= 5),
            "mean_rho": {
                n: (float(np.mean(v)) if v else float("nan"))
                for n, v in per_group.items()
            },
            "sem_rho": {
                n: (
                    float(np.std(v, ddof=1) / max(len(v) ** 0.5, 1))
                    if len(v) > 1
                    else float("nan")
                )
                for n, v in per_group.items()
            },
        }
        cell["attribution"] = _attribution(scores, truth, args)
        out["cells"][kind] = cell
    return out


def _attribution(scores, truth, args):
    """Where any ranking power actually comes from.

    Three paired comparisons, each with its own bootstrap interval, so a
    verdict is not just "it did or did not clear 0.10" but an account of which
    ingredient earned the number:

      ``shape_over_position``  the D x D second-moment shape, over a scalar
                               positional profile fitted from the same probes
      ``position_over_uniform`` the positional weighting, over the
                               position-uniform second moment
      ``oracle_over_global``   what is lost by averaging the receiver metric
                               across contexts instead of computing it on the
                               context being scored
    """
    from research.jacobian_kv.evaluate import paired_bootstrap_spearman_delta

    pairs = {
        "shape_over_position": ("jtfj_pos_p8", "pos_energy_p8"),
        "position_over_uniform": ("jtfj_pos_p8", "jtfj_p8"),
        "oracle_over_global": ("jtfj_oracle_p8", "jtfj_pos_p8"),
        "attn_weight_over_fitted": ("jtfj_attn_p8", "jtfj_pos_p8"),
    }
    out = {}
    for name, (a, b) in pairs.items():
        if a not in scores or b not in scores:
            continue
        d, lo, hi = paired_bootstrap_spearman_delta(
            scores[a], scores[b], truth, n_boot=args.n_boot, seed=args.seed
        )
        out[name] = {"a": a, "b": b, "delta": d, "ci_lo": lo, "ci_hi": hi}
    return out


def _print_table(summary, log):
    for kind, cell in summary.get("cells", {}).items():
        log(f"--- {summary['model']} kind={kind} n={cell['n']}")
        rho = cell["pooled"]["rho"]
        wg = cell["within_group"]["mean_rho"]
        log(f"    {'metric':22s} {'pooled rho':>11s} {'within-group rho':>18s}")
        for n in sorted(rho, key=lambda x: -rho[x] if not math.isnan(rho[x]) else 1):
            log(f"    {n:22s} {rho[n]:11.4f} {wg.get(n, float('nan')):18.4f}")
        for name, at in cell.get("attribution", {}).items():
            log(
                f"    attrib {name:24s} {at['delta']:+.4f} "
                f"CI[{at['ci_lo']:+.4f},{at['ci_hi']:+.4f}]  ({at['a']} - {at['b']})"
            )
        for cand in (
            "jtfj_p8",
            "jtfj_pos_p8",
            "jtfj_attn_p8",
            "pos_energy_p8",
            "jtfj_oracle_p8",
        ):
            g = cell["pooled"].get(cand)
            if g:
                verdict = "PASS" if g["passed"] else "fail"
                log(
                    f"    gate {cand:16s} {verdict}  delta={g['delta']:+.4f} "
                    f"CI[{g['ci_lo']:+.4f},{g['ci_hi']:+.4f}] p={g['p_value']:.4f} "
                    f"vs {g['best_baseline']}"
                )
                for r in g["reasons"]:
                    log(f"        - {r}")


if __name__ == "__main__":
    raise SystemExit(main())

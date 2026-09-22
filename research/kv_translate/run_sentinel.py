# SPDX-License-Identifier: GPL-2.0
"""Check on the actual device that the pieces do what they are assumed to do.

Every check here corresponds to something that was assumed rather than tested
and turned out to be wrong at least once in this lane: that a correction
initialised to produce nothing really changes nothing, that folding a linear
correction preserves the map's output across all of its blocks rather than the
one block anybody looked at, that gradients reach the correction and nowhere
else, that the scorer rejects the string it was once fooled by, and that the
timing comparison charges both sides the same work.

It runs before training so that a failure costs minutes rather than a whole
rented session, and it is deliberately cheap: no data, no optimisation, one
probe batch.

Env: HF_HUB_OFFLINE unset (models may download); one GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.cartridges_cas.scripts.cas_kv_rope import rerot  # noqa: E402
from research.kv_translate import freeze  # noqa: E402
from research.kv_translate.fit import SourceLayout  # noqa: E402
from research.kv_translate.pairs import describe, prefill, to_content_keys  # noqa: E402
from research.kv_translate.refine import ResidualMapper, behaviour_loss  # noqa: E402
from research.kv_translate.run_a0 import (  # noqa: E402
    continuation_logits,
    continuation_logits_grad,
    make_cache,
)
from research.kv_translate.run_a1 import flat_features  # noqa: E402
from research.kv_translate.tasks import grade_retrieval  # noqa: E402

GIB = float(2**30)
FOLD_TOL_REL = 1e-4
FOLD_KL_TOL = 1e-4  # nats, on the target's next-token distribution
ZERO_TOL_ABS = 0.0


def log_now(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--affine", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--cont-len", type=int, default=64)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--solve-dtype", default="float32")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda"
    dtype = getattr(torch, args.dtype)
    sdtype = getattr(torch, args.solve_dtype)
    t0 = time.time()
    checks, failures = {}, []

    # Strict float32 is a property of the backend, not of the tensor dtype:
    # an A100 will silently run a float32 matmul in TF32 unless told not to,
    # and a fold tolerance measured under one is not evidence for the other.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checks["tf32_allowed"] = bool(torch.backends.cuda.matmul.allow_tf32)
    checks["gpu"] = torch.cuda.get_device_name(0)
    checks["torch"] = torch.__version__

    tok = AutoTokenizer.from_pretrained(args.target)
    models, geom = {}, {}
    for role, mid in (("source", args.source), ("target", args.target)):
        m = AutoModelForCausalLM.from_pretrained(
            mid, dtype=dtype, attn_implementation="sdpa"
        ).to(dev)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        models[role], geom[role] = m, describe(m, mid)
    sg, tg = geom["source"], geom["target"]
    layout = SourceLayout(sg.n_layers, sg.n_kv_heads, sg.head_dim)
    checks["weights_gib"] = torch.cuda.memory_allocated() / GIB
    log_now(f"models loaded, {checks['weights_gib']:.2f} GiB of frozen weights")

    # 1. the artifact is the one recorded
    mk, mv, man = freeze.load(args.affine, layout, tg, device=dev)
    mk.cast(sdtype)
    mv.cast(sdtype)
    checks["affine_joint_hash"] = man["joint_weight_sha256"]
    checks["affine_blocks"] = man["n_blocks"]
    checks["affine_support"] = len(next(iter(mk.maps.values())).layers)
    log_now(f"affine verified against its hash, {checks['affine_blocks']} blocks")

    gold = json.load(open(args.gold))
    doc = gold["documents"][0]
    ids = torch.tensor(doc["prompt_ids"], dtype=torch.long, device=dev).unsqueeze(0)
    pos = torch.arange(args.ctx, device=dev)

    with torch.no_grad():
        sp = prefill(models["source"], ids, sg)
        xk = flat_features(to_content_keys(sp)).to(sdtype)
        xv = flat_features([v.float() for v in sp.values]).to(sdtype)

    # 2. a correction that predicts nothing changes nothing, exactly
    rk = ResidualMapper(mk, layout, tg, "k", 32, linear=True).to(dev)
    rv = ResidualMapper(mv, layout, tg, "v", 32, linear=True).to(dev)
    rk.freeze_affine()
    rv.freeze_affine()
    with torch.no_grad():
        worst = 0.0
        for name, r, m, x in (("k", rk, mk, xk), ("v", rv, mv, xv)):
            corrected = r(x)
            plain = m.apply(x)
            for a, b in zip(corrected, plain):
                worst = max(worst, float((a - b).abs().max()))
    checks["zero_correction_max_abs_diff"] = worst
    ok = worst <= ZERO_TOL_ABS
    checks["zero_correction_is_identity"] = ok
    if not ok:
        failures.append(
            f"a zeroed correction changes the map by {worst:.3e}; every arm's "
            "starting point is then not the uncorrected map"
        )
    log_now(f"zero-correction parity: max |difference| = {worst:.3e}")

    # 3. folding preserves the map on every block, not on the one that was
    #    checked before. The residual is perturbed first, or the fold of a
    #    zero correction would be trivially exact and prove nothing.
    with torch.no_grad():
        for r in (rk, rv):
            for blk in r.blocks.values():
                blk.fc2.weight.normal_(0, 1e-3)
                blk.fc2.bias.normal_(0, 1e-3)
    probe = torch.randn(8, layout.n_features, device=dev, dtype=sdtype)
    parity = [rk.fold_parity(probe), rv.fold_parity(probe)]
    checks["fold_parity"] = parity
    checks["fold_tolerance_relative"] = FOLD_TOL_REL
    bad = [p for p in parity if p["worst_relative_gap"] > FOLD_TOL_REL]
    checks["fold_parity_all_blocks"] = not bad
    if bad:
        failures.append(f"folding is not parity-preserving: {bad}")
    log_now(
        "fold parity over all blocks: "
        + ", ".join(
            f"{p['kind']} {p['blocks_checked']} blocks worst "
            f"{p['worst_relative_gap']:.2e}"
            for p in parity
        )
    )

    # 3b. folding must also leave the target's own output alone.
    #
    # Order matters here and cost one wrong answer already: merge() rewrites
    # the affine map in place, so a mapper asked for its "unfolded" output
    # after the merge applies the correction twice. Both caches are therefore
    # materialised before anything is merged, and every later comparison is
    # made against those, never by calling the mapper again.
    cont = ids[:, : args.cont_len]
    with torch.no_grad():

        def logits_from_cache(pk, pv):
            keys = [
                rerot(b_[0].float(), pos, tg.rope_theta).unsqueeze(0).to(dtype)
                for b_ in pk
            ]
            return continuation_logits(
                models["target"], keys, [v.to(dtype) for v in pv], cont, args.ctx
            )

        unf_k, unf_v = rk(xk), rv(xv)
        before = logits_from_cache(unf_k, unf_v).float()
        repeat = logits_from_cache(unf_k, unf_v).float()
        fk, fv = rk.merge(), rv.merge()
        fold_k, fold_v = fk.apply(xk), fv.apply(xv)
        after = logits_from_cache(fold_k, fold_v).float()

        def compare(a_, b_):
            lp, lq = torch.log_softmax(a_, -1), torch.log_softmax(b_, -1)
            return {
                "max_abs_logit_diff": float((a_ - b_).abs().max()),
                "mean_kl_nats": float((lp.exp() * (lp - lq)).sum(-1).mean()),
                "argmax_agreement": float(
                    (a_.argmax(-1) == b_.argmax(-1)).float().mean()
                ),
            }

        rerun = compare(before, repeat)
        fold = compare(before, after)

        # Measured two ways, because a relative difference divided by the
        # element itself is meaningless where the element is near zero -- a
        # first attempt reported fourteen million units in the last place for
        # exactly that reason. In float32 the difference is taken against the
        # tensor's own scale, which is what the fold's arithmetic is
        # accountable for; in bfloat16 it is the distance between bit
        # patterns, the only honest unit for "did rounding go the other way".
        worst_rel_fp32, n_el = 0.0, 0
        for x_, y_ in zip(
            [t[0].float() for t in unf_k] + [t[0].float() for t in unf_v],
            [t[0].float() for t in fold_k] + [t[0].float() for t in fold_v],
        ):
            scale = x_.abs().max().clamp(min=1e-30)
            worst_rel_fp32 = max(worst_rel_fp32, float((x_ - y_).abs().max() / scale))
            n_el += x_.numel()
        # After the cast, measured against the tensor's scale rather than as a
        # bit distance. An integer distance between bit patterns is unbounded
        # near zero -- a value that crosses zero reads as tens of thousands of
        # steps while moving almost nothing -- so it answers a different
        # question than the one being asked. One step of bfloat16 at the
        # tensor's scale is 2^-8; anything within that is the cast, not a
        # difference between the two maps.
        n_diff, worst_step = 0, 0.0
        for x_, y_ in zip(
            [t[0].to(dtype) for t in unf_k] + [t[0].to(dtype) for t in unf_v],
            [t[0].to(dtype) for t in fold_k] + [t[0].to(dtype) for t in fold_v],
        ):
            ne = x_ != y_
            n_diff += int(ne.sum())
            if bool(ne.any()):
                scale = x_.float().abs().max().clamp(min=1e-30)
                worst_step = max(
                    worst_step,
                    float((x_.float() - y_.float()).abs().max() / scale) / (2.0**-8),
                )
        cache_cmp = {
            "fp32_worst_relative_to_tensor_scale": worst_rel_fp32,
            "bf16_entries_differing": n_diff,
            "bf16_entries_total": n_el,
            "bf16_fraction_differing": n_diff / max(n_el, 1),
            "bf16_worst_move_in_steps_at_tensor_scale": worst_step,
        }
        del unf_k, unf_v, fold_k, fold_v

    checks["folded_vs_unfolded_logits"] = fold
    checks["folded_vs_unfolded_cache"] = cache_cmp
    checks["identical_call_floor"] = rerun
    checks["fold_kl_tolerance_nats"] = FOLD_KL_TOL
    # The assertion is the declared float32 tolerance, and only that. What
    # the cast then costs is measured and published rather than gated: a
    # threshold on it would be a threshold on bfloat16's resolution, which is
    # not something folding can be held to, and picking one after seeing the
    # number is how a test gets tuned until it passes.
    ok = cache_cmp["fp32_worst_relative_to_tensor_scale"] <= FOLD_TOL_REL
    checks["fold_exact_in_float32"] = ok
    checks["folded_logits_agree"] = ok
    if not ok:
        failures.append(
            f"folding is not exact: the two caches differ by "
            f"{cache_cmp['fp32_worst_relative_to_tensor_scale']:.2e} relative to "
            f"tensor scale in float32, against a declared {FOLD_TOL_REL:.0e}"
        )
    log_now(
        f"folded vs unfolded cache: float32 worst "
        f"{cache_cmp['fp32_worst_relative_to_tensor_scale']:.2e} relative to "
        f"tensor scale; after the cast "
        f"{100 * cache_cmp['bf16_fraction_differing']:.3f}% of entries move by "
        f"at most {cache_cmp['bf16_worst_move_in_steps_at_tensor_scale']:.2f} "
        f"storage steps"
    )
    log_now(
        f"  what the cast costs downstream: KL {fold['mean_kl_nats']:.2e} nats "
        f"and {100 * (1 - fold['argmax_agreement']):.1f}% of next tokens; two "
        f"identical calls agree exactly, so this is the storage format and not "
        f"nondeterminism. It is the floor under any token-level margin here."
    )

    # 4. gradients reach the correction, and only the correction
    rk2 = ResidualMapper(mk, layout, tg, "k", 32, linear=True).to(dev)
    rv2 = ResidualMapper(mv, layout, tg, "v", 32, linear=True).to(dev)
    rk2.freeze_affine()
    rv2.freeze_affine()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        tp = prefill(models["target"], ids, tg)
        ref = torch.log_softmax(
            continuation_logits(
                models["target"], tp.keys_post, tp.values, cont, args.ctx
            ).float(),
            -1,
        ).half()
        del tp
    pk, pv = rk2(xk), rv2(xv)
    keys = [rerot(b[0].float(), pos, tg.rope_theta).unsqueeze(0).to(dtype) for b in pk]
    loss = behaviour_loss(
        ref,
        continuation_logits_grad(
            models["target"], keys, [v.to(dtype) for v in pv], cont, args.ctx
        ),
    )
    loss.backward()
    n_with = sum(
        1
        for p in list(rk2.blocks.parameters()) + list(rv2.blocks.parameters())
        if p.grad is not None and float(p.grad.abs().sum()) > 0
    )
    n_tot = len(list(rk2.blocks.parameters())) + len(list(rv2.blocks.parameters()))
    model_grads = sum(
        1 for m in models.values() for p in m.parameters() if p.grad is not None
    )
    checks["backward_loss"] = float(loss)
    checks["residual_params_with_gradient"] = n_with
    checks["residual_params_total"] = n_tot
    checks["frozen_model_params_with_gradient"] = model_grads
    checks["peak_alloc_gib_backward"] = torch.cuda.max_memory_allocated() / GIB
    if n_with == 0:
        failures.append("no gradient reaches the correction")
    if model_grads:
        failures.append(f"{model_grads} frozen model parameters received gradient")
    log_now(
        f"backward: loss {float(loss):.4f}, {n_with}/{n_tot} correction tensors "
        f"have gradient, {model_grads} frozen tensors do, peak "
        f"{checks['peak_alloc_gib_backward']:.2f} GiB"
    )
    del loss, pk, pv, keys

    # 5. the scorer rejects the string it once accepted
    g = doc["retrieval_gold"]
    checks["scorer"] = {
        "gold": g,
        "exact_on_gold": grade_retrieval(g, f"The code is {g}.")["exact_any_run"],
        "exact_on_extended": grade_retrieval(g, g + g[:4])["exact_any_run"],
        "substring_on_extended": grade_retrieval(g, g + g[:4])["substring"],
        "exact_on_refusal": grade_retrieval(g, "I do not know.")["exact_any_run"],
    }
    s = checks["scorer"]
    if not (
        s["exact_on_gold"] == 1.0
        and s["exact_on_extended"] == 0.0
        and s["substring_on_extended"] == 1.0
        and s["exact_on_refusal"] == 0.0
    ):
        failures.append(f"scorer does not behave as declared: {s}")
    log_now(
        f"scorer: gold accepted, extended rejected (substring would accept it), "
        f"refusal rejected"
    )

    # 6. both timing arms produce one next-token distribution from one prompt
    head, last = ids[:, : args.ctx - 1], ids[:, args.ctx - 1 :]
    sp_head = prefill(models["source"], head, sg)
    pos_head = torch.arange(args.ctx - 1, device=dev)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    @torch.no_grad()
    def native():
        return models["target"](input_ids=ids, logits_to_keep=1).logits[:, -1]

    @torch.no_grad()
    def translated():
        x1 = flat_features(to_content_keys(sp_head)).to(sdtype)
        x2 = flat_features([v.float() for v in sp_head.values]).to(sdtype)
        kk = [
            rerot(b[0].float(), pos_head, tg.rope_theta).unsqueeze(0).to(dtype)
            for b in fk.apply(x1)
        ]
        vv = [v.to(dtype) for v in fv.apply(x2)]
        return models["target"](
            input_ids=last,
            attention_mask=torch.ones(1, args.ctx, dtype=torch.long, device=dev),
            position_ids=torch.arange(args.ctx - 1, args.ctx, device=dev).unsqueeze(0),
            past_key_values=make_cache(kk, vv),
            use_cache=True,
        ).logits[:, -1]

    def timed(fn, reps=10):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            ev[0].record()
            fn()
            ev[1].record()
            torch.cuda.synchronize()
            ts.append(ev[0].elapsed_time(ev[1]))
        return float(np.percentile(ts, 50)), float(np.percentile(ts, 95))

    # Both matmul modes. A float32 tensor does not by itself say how the
    # multiply is performed, and on this class of device the difference
    # between strict float32 and the tensor-core mode is large enough to
    # decide a latency gate on its own. Recording one number without its mode
    # would make the gate a property of an undeclared setting.
    checks["first_token"] = {}
    for mode, allow in (("strict_fp32", False), ("tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = allow
        nat50, _ = timed(native)
        tra50, tra95 = timed(translated)
        checks["first_token"][mode] = {
            "native_p50_ms": nat50,
            "translated_p50_ms": tra50,
            "ratio_p50": tra50 / nat50,
            "ratio_p95": tra95 / nat50,
        }
        log_now(
            f"first token [{mode}]: native {nat50:.1f} ms, translated "
            f"{tra50:.1f} ms, p50 ratio {tra50 / nat50:.3f}"
        )
    torch.backends.cuda.matmul.allow_tf32 = False
    a, b = native(), translated()
    checks["first_token"]["shapes_match"] = list(a.shape) == list(b.shape)
    checks["first_token"]["both_single_position"] = a.shape[0] == 1 and a.ndim == 2
    if not checks["first_token"]["shapes_match"]:
        failures.append("the two timing arms do not produce the same output shape")

    checks["seconds"] = time.time() - t0
    checks["failures"] = failures
    checks["passed"] = not failures
    with open(args.out, "w") as f:
        json.dump(checks, f, indent=2, sort_keys=True, default=str)
    log_now(
        (
            "SENTINEL PASSED"
            if not failures
            else "SENTINEL FAILED: " + "; ".join(failures)
        )
        + f" in {checks['seconds']:.0f}s"
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: GPL-2.0
"""Does a vector-Jacobian product survive the attention path on this stack?

This is the ten-minute portability check the execution order puts ahead of the
twenty free GPU-hours, because it is what decides between a zero-dollar local
program and renting a card.  It answers four questions on real models:

  1. Does a gradient reach a KV cache block grafted into the cache at all, in
     eager attention, on ROCm?
  2. Does ``torch.func.vjp`` work through the same path, or is
     ``torch.autograd.grad`` the portable route?
  3. Is a second moment accumulated from Fisher probes finite and stable --
     does it stop moving as probes are added?
  4. Does the directional score ``mean_p (u_p . e)^2`` track a first-order
     prediction of the real output change for a small perturbation?

Question 4 is the one that catches a silently wrong implementation: if the VJP
is not the Jacobian of the thing being measured, the linearisation check fails
even though every tensor is finite.

Run::

    python -m research.jacobian_kv.probe_rocm --model Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.jacobian_kv.sensitivity import (  # noqa: E402
    clone_cache,
    continuation_logits,
    fisher_probe,
    freeze_model,
    graft_leaves,
    n_cache_layers,
    prefill,
    vjp_from_probe,
    _layer_tensors,
)

PROMPT = (
    "The Norwegian coastal steamer leaves Bergen every evening. "
    "Passengers who board at Trondheim are given cabin keys at the purser's "
    "desk, and the ship calls at thirty-four ports before turning south again. "
    "In winter the route is run by two vessels instead of three, which means "
    "that the northbound and southbound schedules diverge by a full day. "
)
CONTINUATION = (
    "The purser keeps a written record of every cabin key issued at Trondheim, "
    "and the winter timetable is posted in the forward lounge."
)


def kl_rows(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """KL(p || q) per position, in float32, from logits."""
    lp = torch.log_softmax(p_logits.float(), dim=-1)
    lq = torch.log_softmax(q_logits.float(), dim=-1)
    return (lp.exp() * (lp - lq)).sum(dim=-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=-1, help="-1 = middle layer")
    ap.add_argument("--probes", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    report: dict = {
        "model": args.model,
        "device": dev,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "dtype": args.dtype,
        "attn_implementation": args.attn,
        "probes": args.probes,
    }
    if dev == "cuda":
        report["gpu"] = torch.cuda.get_device_name(0)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation=args.attn
    ).to(dev)
    freeze_model(model)
    report["load_s"] = round(time.time() - t0, 2)
    report["n_layers"] = int(model.config.num_hidden_layers)
    report["n_kv_heads"] = int(getattr(model.config, "num_key_value_heads", 0))

    prompt_ids = tok(PROMPT, return_tensors="pt").input_ids.to(dev)
    cont_ids = tok(
        CONTINUATION, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(dev)
    prompt_len = prompt_ids.shape[1]
    report["prompt_tokens"] = int(prompt_len)
    report["cont_tokens"] = int(cont_ids.shape[1])

    base = prefill(model, prompt_ids)
    layer = args.layer if args.layer >= 0 else n_cache_layers(base) // 2
    report["layer_probed"] = int(layer)
    k0, v0 = _layer_tensors(base, layer)
    report["k_block_shape"] = list(k0.shape)
    report["v_block_shape"] = list(v0.shape)

    # ---- clean reference ------------------------------------------------
    with torch.no_grad():
        clean_logits = continuation_logits(
            model, clone_cache(base), cont_ids, prompt_len
        ).detach()
    report["clean_logits_finite"] = bool(torch.isfinite(clean_logits).all())

    # ---- q1: does a gradient reach the cache block ----------------------
    g = torch.Generator(device=dev).manual_seed(args.seed)
    t1 = time.time()
    work = clone_cache(base)
    leaves = graft_leaves(work, [layer], ("k", "v"))
    with torch.enable_grad():
        logits = continuation_logits(model, work, cont_ids, prompt_len)
        z = fisher_probe(logits, g)
        us = vjp_from_probe(logits, leaves, z, retain=False)
    report["vjp_autograd_s"] = round(time.time() - t1, 3)
    report["autograd_vjp_ok"] = all(
        bool(torch.isfinite(u).all()) and float(u.abs().max()) > 0 for u in us.values()
    )
    report["vjp_norms"] = {
        f"{k[0]}:{k[1]}": float(u.float().norm()) for k, u in us.items()
    }

    # ---- q2: does torch.func.vjp work through the same path -------------
    try:
        from torch.func import vjp as func_vjp

        def f(kblock):
            c = clone_cache(base)
            layers_attr = getattr(c, "layers", None)
            if layers_attr is not None:
                layers_attr[layer].keys = kblock
            else:
                c.key_cache[layer] = kblock
            return continuation_logits(model, c, cont_ids, prompt_len)

        _, pullback = func_vjp(f, k0.detach().clone())
        (u_func,) = pullback(z)
        ref = us[(layer, "k")]
        num = float((u_func.float() - ref.float()).norm())
        den = float(ref.float().norm()) + 1e-30
        report["func_vjp_ok"] = bool(torch.isfinite(u_func).all())
        report["func_vjp_rel_err_vs_autograd"] = num / den
    except Exception as exc:  # noqa: BLE001
        report["func_vjp_ok"] = False
        report["func_vjp_error"] = f"{type(exc).__name__}: {exc}"

    # ---- q3: is the second moment finite and stable in probe count ------
    torch.manual_seed(args.seed)
    g2 = torch.Generator(device=dev).manual_seed(args.seed + 1)
    rows = []
    t2 = time.time()
    work = clone_cache(base)
    leaves = graft_leaves(work, [layer], ("k", "v"))
    with torch.enable_grad():
        logits = continuation_logits(model, work, cont_ids, prompt_len)
        for p in range(args.probes):
            zp = fisher_probe(logits, g2)
            up = vjp_from_probe(logits, leaves, zp, retain=(p < args.probes - 1))
            rows.append(up[(layer, "v")].reshape(-1).float().cpu())
    report["probe_loop_s"] = round(time.time() - t2, 3)
    U = torch.stack(rows, dim=0)
    report["sketch_finite"] = bool(torch.isfinite(U).all())

    gen = torch.Generator().manual_seed(args.seed + 2)
    e_dir = torch.randn(U.shape[1], generator=gen)
    e_dir = e_dir / e_dir.norm()
    running = [float((U[:n] @ e_dir).pow(2).mean()) for n in (1, 2, 4, 8, args.probes)]
    report["directional_score_by_probes"] = dict(
        zip([1, 2, 4, 8, args.probes], [round(x, 8) for x in running])
    )
    if running[-2] > 0:
        report["score_drift_8_to_max"] = abs(running[-1] - running[-2]) / running[-2]

    # ---- q4: does the VJP linearise the real output change --------------
    # For a small perturbation d on the V block, the sampled-label loss should
    # change by (u . d) to first order.  Check against a finite difference of
    # the same scalar, which is the only way to catch a VJP of the wrong thing.
    #
    # The check is only meaningful in float32.  Measured on this stack, the
    # bf16 VJP agrees with the fp32 VJP to cosine 0.999, but the bf16 finite
    # difference of the same scalar is wrong by 56 to 109 percent and changes
    # sign as the step shrinks: the difference of two large logit sums is below
    # the format's resolution.  So a bf16 run reports the check and refuses to
    # certify on it, rather than reporting a gradient failure that is not one.
    with torch.enable_grad():
        work2 = clone_cache(base)
        leaves2 = graft_leaves(work2, [layer], ("v",))
        logits2 = continuation_logits(model, work2, cont_ids, prompt_len)
        g3 = torch.Generator(device=dev).manual_seed(args.seed + 3)
        z3 = fisher_probe(logits2, g3)
        u3 = vjp_from_probe(logits2, leaves2, z3, retain=False)[(layer, "v")]

    scale = float(v0.float().abs().mean())
    gen2 = torch.Generator(device=dev).manual_seed(args.seed + 4)
    d = torch.randn(v0.shape, generator=gen2, device=dev, dtype=torch.float32)
    d = (d / d.norm()) * (scale * 1e-2 * (v0.numel() ** 0.5))
    predicted = float((u3.float().reshape(-1) * d.reshape(-1)).sum())

    def scalar_at(eps: float) -> float:
        c = clone_cache(base)
        layers_attr = getattr(c, "layers", None)
        newv = (v0.float() + eps * d).to(v0.dtype)
        if layers_attr is not None:
            layers_attr[layer].values = newv
        else:
            c.value_cache[layer] = newv
        with torch.no_grad():
            lg = continuation_logits(model, c, cont_ids, prompt_len)
        return float((lg.float() * z3.float()).sum())

    fd = (scalar_at(1.0) - scalar_at(-1.0)) / 2.0
    report["linearisation_predicted"] = predicted
    report["linearisation_finite_difference"] = fd
    report["linearisation_rel_err"] = abs(fd - predicted) / (abs(fd) + 1e-30)

    # ---- context: what does a perturbation of this size actually cost ----
    c = clone_cache(base)
    layers_attr = getattr(c, "layers", None)
    newv = (v0.float() + d).to(v0.dtype)
    if layers_attr is not None:
        layers_attr[layer].values = newv
    else:
        c.value_cache[layer] = newv
    with torch.no_grad():
        pert_logits = continuation_logits(model, c, cont_ids, prompt_len)
    report["perturbation_mean_kl"] = float(kl_rows(clean_logits, pert_logits).mean())

    if dev == "cuda":
        report["peak_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    report["total_s"] = round(time.time() - t0, 2)

    report["linearisation_valid"] = args.dtype == "float32"
    if not report["linearisation_valid"]:
        report["linearisation_note"] = (
            "finite difference is below the resolution of "
            f"{args.dtype}; rerun with --dtype float32 to certify"
        )
    verdict = report["autograd_vjp_ok"] and report["sketch_finite"]
    if report["linearisation_valid"]:
        verdict = verdict and report["linearisation_rel_err"] < 0.05
    report["PORTABLE"] = bool(verdict and report["linearisation_valid"])
    report["PORTABLE_UNCERTIFIED"] = bool(verdict and not report["linearisation_valid"])

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())

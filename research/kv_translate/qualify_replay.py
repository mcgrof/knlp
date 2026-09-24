#!/usr/bin/env python3
"""Establish on the real card that the repair computes what it claims to.

The archive holds two qualifications and neither is this one. The mapper was
qualified on the card, which says the fitted operator reproduces its reference
and is deterministic; it says nothing about replay. Replay was qualified on the
processor against a tiny random pair, which says the scatter, the positions and
the masking are right at that scale; it says nothing about the card, the
precision, the kernel or the real models. A review looked for a real-card
full-replay parity receipt and did not find one, which is correct: there was
none.

So this is that receipt. Recomputing every position of an approximate cache
must reproduce what a native prefill of the same tokens produces, on this card,
in this precision, through this kernel. If that fails, every score and every
timing measured through the replay path describes something other than the
operation they are attributed to, and the run stops here rather than spending
its remaining budget.

Four things are checked, in the order that makes a failure interpretable.
Whether the fitted map still matches its frozen hash and its reference. Whether
the selector picks the same positions twice. Whether full replay reproduces
native. And whether the repaired cache carries the same continuation
distribution as the native one it should equal, which is the property the task
scores actually depend on and the one a tensor comparison can miss.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from research.cartridges_cas.scripts.cas_kv_rope import rerot  # noqa: E402
from research.kv_translate import freeze  # noqa: E402
from research.kv_translate.fit import SourceLayout  # noqa: E402
from research.kv_translate.pairs import describe, prefill, to_content_keys  # noqa: E402
from research.kv_translate.recovery import BLOCK, mask_from_blocks  # noqa: E402
from research.kv_translate.recovery_runner import make_cache, select  # noqa: E402
from research.kv_translate.replay import replay_selected  # noqa: E402
from research.kv_translate.run_a1 import flat_features  # noqa: E402

# A repair is not required to be bit-identical to a prefill: it runs a
# different number of query positions through the same kernel, and floating
# point addition is not associative. It is required to be the same computation
# to within that, and these are the limits that distinguishes "the same thing,
# reassociated" from "a different thing".
LIMITS = {
    # Largest elementwise difference, relative to the tensor's own scale.
    "full_replay_matches_native_keys": 2e-2,
    "full_replay_matches_native_values": 2e-2,
    # Cosine is the shape check: a cache that is uniformly scaled or shifted
    # would pass an elementwise tolerance chosen loosely and fail this.
    "full_replay_cosine_keys": 1e-4,  # reported as 1 - cos
    "full_replay_cosine_values": 1e-4,
    # What the task actually reads. A cache can differ in its tensors and
    # still induce the same next-token distribution, and the reverse is the
    # failure that matters.
    "continuation_kl_nats": 1e-3,
    "continuation_argmax_disagreement": 0.0,
    # The selector must choose the same blocks twice or nothing downstream is
    # reproducible.
    "selector_is_deterministic": 0.0,
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def rel_max(a, b):
    """Largest absolute difference, scaled by the reference's magnitude."""
    a32, b32 = a.float(), b.float()
    scale = b32.abs().max().clamp(min=1e-30)
    return float((a32 - b32).abs().max() / scale)


def one_minus_cos(a, b):
    a32, b32 = a.float().reshape(-1), b.float().reshape(-1)
    c = torch.nn.functional.cosine_similarity(a32, b32, dim=0)
    return float(1.0 - c)


@torch.no_grad()
def continuation_stats(model, keys, values, query_ids, prompt_len, device):
    """The next-token distribution a cache induces for a question."""
    n = query_ids.shape[1]
    out = model(
        input_ids=query_ids,
        attention_mask=torch.ones(1, prompt_len + n, dtype=torch.long, device=device),
        position_ids=torch.arange(prompt_len, prompt_len + n, device=device).unsqueeze(
            0
        ),
        past_key_values=make_cache(keys, values),
        use_cache=True,
        logits_to_keep=1,
    )
    return out.logits[0, -1].float()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--source-revision", required=True, help="a commit, not a branch")
    ap.add_argument("--target-revision", required=True)
    ap.add_argument("--arm", required=True, help="name=path.pt")
    ap.add_argument("--expect-joint-sha256", default="")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--docs", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--blocks", type=int, default=64)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--solve-dtype", default="float32")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype, sdtype = getattr(torch, args.dtype), getattr(torch, args.solve_dtype)
    if dev == "cuda":
        # The queue names this numeric mode. TF32 would silently reduce the
        # precision of the very matmuls the map is fitted and applied in.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    checks, failures = {}, []

    def require(name, measured, limit, why):
        ok = bool(measured <= limit)
        checks[name] = {
            "measured": float(measured),
            "limit": float(limit),
            "passed": ok,
            "why": why,
        }
        if not ok:
            failures.append(f"{name}: {measured:.4e} exceeds {limit:.4e}")
        log(f"  {name:38s} {measured:.4e} <= {limit:.1e}  {'pass' if ok else 'FAIL'}")
        return ok

    name, path = args.arm.split("=", 1)
    tok = AutoTokenizer.from_pretrained(args.target, revision=args.target_revision)
    models, geom = {}, {}
    for role, mid, rev in (
        ("source", args.source, args.source_revision),
        ("target", args.target, args.target_revision),
    ):
        m = AutoModelForCausalLM.from_pretrained(
            mid, revision=rev, dtype=dtype, attn_implementation="sdpa"
        ).to(dev)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        models[role], geom[role] = m, describe(m, mid)
    sg, tg = geom["source"], geom["target"]
    layout = SourceLayout(sg.n_layers, sg.n_kv_heads, sg.head_dim)
    mk, mv, man = freeze.load(path, layout, tg, device=dev)
    mk, mv = mk.cast(sdtype), mv.cast(sdtype)
    joint = man["joint_weight_sha256"]
    log(f"arm {name}: joint {joint[:16]}")
    if args.expect_joint_sha256 and joint != args.expect_joint_sha256:
        raise SystemExit(
            f"INVALID: arm hashes to {joint}, expected {args.expect_joint_sha256}"
        )

    fixture = json.load(open(args.manifest))
    docs = [d for d in fixture["documents"] if d["in_screen"]][: args.docs]
    pos = torch.arange(args.ctx, device=dev)
    per_doc = []

    for d in docs:
        ids = torch.tensor(d["prompt_ids"], dtype=torch.long, device=dev).unsqueeze(0)
        qids = tok(d["retrieval_query"], return_tensors="pt").input_ids.to(dev)
        log(f"{d['doc'][:46]}")

        # native, the thing everything is compared against
        tp = prefill(models["target"], ids, tg)
        nk = [k.to(dtype) for k in tp.keys_post]
        nv = [v.to(dtype) for v in tp.values]

        # the translated cache this repair starts from
        sp = prefill(models["source"], ids, sg)
        xk = flat_features(to_content_keys(sp)).to(sdtype)
        xv = flat_features([v.float() for v in sp.values]).to(sdtype)
        ak = [
            rerot(b[0].float(), pos, tg.rope_theta).unsqueeze(0).to(dtype)
            for b in mk.apply(xk)
        ]
        av = [v.to(dtype) for v in mv.apply(xv)]

        # full replay must reproduce native
        full = torch.ones(args.ctx, dtype=torch.bool, device=dev)
        rk, rv, info = replay_selected(models["target"], ids, full, ak, av)
        if info["replayed"] != args.ctx:
            raise SystemExit(f"INVALID: replayed {info['replayed']} of {args.ctx}")
        if len(info["layers_written"]) != tg.n_layers:
            raise SystemExit(
                f"INVALID: wrote {len(info['layers_written'])} of {tg.n_layers} layers"
            )
        km = max(rel_max(a, b) for a, b in zip(rk, nk))
        vm = max(rel_max(a, b) for a, b in zip(rv, nv))
        kc = max(one_minus_cos(a, b) for a, b in zip(rk, nk))
        vc = max(one_minus_cos(a, b) for a, b in zip(rv, nv))

        # and must induce the same continuation
        ln = continuation_stats(models["target"], nk, nv, qids, args.ctx, dev)
        lr = continuation_stats(models["target"], rk, rv, qids, args.ctx, dev)
        kl = float(
            torch.nn.functional.kl_div(
                torch.log_softmax(lr, -1),
                torch.log_softmax(ln, -1),
                log_target=True,
                reduction="sum",
            )
        )
        disagree = float(int(ln.argmax()) != int(lr.argmax()))

        # the selector must be reproducible
        b1, _ = select(
            "mapped_attn",
            args.blocks,
            model=models["target"],
            mapped=(ak, av),
            native=None,
            query_ids=qids,
            n_positions=args.ctx,
        )
        b2, _ = select(
            "mapped_attn",
            args.blocks,
            model=models["target"],
            mapped=(ak, av),
            native=None,
            query_ids=qids,
            n_positions=args.ctx,
        )
        sel_same = 0.0 if list(b1) == list(b2) else 1.0

        # and a partial repair must move exactly the positions it selected
        m = mask_from_blocks(args.ctx, b1).to(dev)
        pk, pv, pinfo = replay_selected(models["target"], ids, m, ak, av)
        moved = torch.zeros(args.ctx, dtype=torch.bool)
        for a, b in zip(ak, pk):
            moved |= (a != b).any(dim=-1).any(dim=1).squeeze(0).cpu()
        expected = set(int(x) for x in torch.nonzero(m).flatten().cpu())
        got = set(int(x) for x in torch.nonzero(moved).flatten())
        per_doc.append(
            {
                "doc": d["doc"],
                "keys_rel_max": km,
                "values_rel_max": vm,
                "keys_one_minus_cos": kc,
                "values_one_minus_cos": vc,
                "continuation_kl_nats": kl,
                "continuation_argmax_disagreement": disagree,
                "selector_repeat_differs": sel_same,
                "selected_blocks": sorted(int(x) for x in b1),
                "partial_moved_exactly_selected": bool(got == expected),
                "partial_replayed": pinfo["replayed"],
            }
        )
        log(
            f"  keys {km:.3e} values {vm:.3e} kl {kl:.3e} argmax_differs {disagree:.0f}"
        )

    agg = lambda k: max(x[k] for x in per_doc)  # noqa: E731
    require(
        "full_replay_matches_native_keys",
        agg("keys_rel_max"),
        LIMITS["full_replay_matches_native_keys"],
        "recomputing every position must reproduce a native prefill",
    )
    require(
        "full_replay_matches_native_values",
        agg("values_rel_max"),
        LIMITS["full_replay_matches_native_values"],
        "the same, for values",
    )
    require(
        "full_replay_cosine_keys",
        agg("keys_one_minus_cos"),
        LIMITS["full_replay_cosine_keys"],
        "a uniformly scaled cache would pass a loose elementwise bound",
    )
    require(
        "full_replay_cosine_values",
        agg("values_one_minus_cos"),
        LIMITS["full_replay_cosine_values"],
        "the same, for values",
    )
    require(
        "continuation_kl_nats",
        agg("continuation_kl_nats"),
        LIMITS["continuation_kl_nats"],
        "what the task reads is the next-token distribution, not the tensors",
    )
    require(
        "continuation_argmax_disagreement",
        agg("continuation_argmax_disagreement"),
        LIMITS["continuation_argmax_disagreement"],
        "a fully repaired cache must predict the same token as native",
    )
    require(
        "selector_is_deterministic",
        agg("selector_repeat_differs"),
        LIMITS["selector_is_deterministic"],
        "nothing downstream reproduces if the selection does not",
    )
    if not all(x["partial_moved_exactly_selected"] for x in per_doc):
        failures.append("a partial repair moved positions it did not select")

    receipt = {
        "contract": "replay_qualification_v1",
        "scope": (
            "full replay against a native prefill, on this card, in this "
            "precision, through this kernel. The mapper qualification is a "
            "different receipt and does not cover replay; the processor "
            "parity test is a different scale and does not cover this card."
        ),
        "arm": {
            "name": name,
            "path": os.path.abspath(path),
            "joint_weight_sha256": joint,
        },
        "models": {
            "source": {"id": args.source, "revision": args.source_revision},
            "target": {"id": args.target, "revision": args.target_revision},
        },
        "numeric_mode": {
            "model_dtype": args.dtype,
            "map_dtype": args.solve_dtype,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        },
        "ctx": args.ctx,
        "documents": [d["doc"] for d in docs],
        "blocks": args.blocks,
        "block_tokens": BLOCK,
        "per_document": per_doc,
        "checks": checks,
        "contract_limits": LIMITS,
        "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
        "torch": torch.__version__,
        "failures": failures,
        "passed": not failures,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    json.dump(receipt, open(args.out, "w"), indent=2, sort_keys=True, default=str)
    log("replay_qualification " + ("QUALIFIED" if not failures else "FAILED"))
    for f in failures:
        log(f"  {f}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

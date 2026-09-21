# SPDX-License-Identifier: GPL-2.0
"""Train the correction on a declared set of documents, and change nothing else.

The previous runner fitted the affine map, chose its support, sampled its
training windows and trained the correction in one pass, so asking it for more
training data also redrew the calibration set and refitted the map. Whatever
the resulting difference measured, it was not the value of more correction
data.

This does one thing. It loads an already-fitted affine map and its support,
casts once to the declared precision, and trains only the residual. The map is
never refitted, the support is never reselected, and a larger arm never starts
from a smaller arm's learned correction -- every arm begins from one
serialised initialisation, loaded and hashed, so the starting point is an
input rather than a side effect of the order the objects were built in.

The unit of data is the document. Each contributes one window, fixed in the
manifest, and a document that appears in training cannot appear in validation
or in any scored set; the manifest builder refuses to emit roles that overlap.

What is held constant across arms, and logged so it can be checked rather than
asserted: the learning rate, the number of updates, the objective, the number
of supervised continuation tokens, the batch size, the sampling policy and its
seed, the frozen source and target weights, and the initialisation hash. What
varies is which documents are in the role.

Env: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.cartridges_cas.scripts.cas_kv_rope import rerot  # noqa: E402
from research.kv_translate import freeze  # noqa: E402
from research.kv_translate.fit import SourceLayout  # noqa: E402
from research.kv_translate.pairs import (  # noqa: E402
    describe,
    prefill,
    to_content_keys,
)
from research.kv_translate.refine import (  # noqa: E402
    ResidualMapper,
    behaviour_loss,
)
from research.kv_translate.run_a0 import (  # noqa: E402
    continuation_logits,
    continuation_logits_grad,
)
from research.kv_translate.run_a1 import Mapper, flat_features  # noqa: E402
from research.kv_translate.tasks import wikitext_documents  # noqa: E402

GIB = float(2**30)


def log_now(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_role_windows(manifests, role, tok, ctx, cont_len, split_cache):
    """The token window each document in a role contributes, one per document."""
    roles = manifests["roles"][role]
    want = {d["doc"]: d for d in roles}
    found, need = {}, ctx + cont_len
    splits = sorted({d["split"] for d in roles})
    for sp in splits:
        if sp not in split_cache:
            split_cache[sp] = {
                t: b for t, b in wikitext_documents(split=sp, max_docs=100000)
            }
        for title, body in split_cache[sp].items():
            if title not in want or title in found:
                continue
            ids = tok(body, return_tensors="pt").input_ids[0]
            if ids.shape[0] < need:
                continue
            found[title] = ids[:need]
    missing = sorted(set(want) - set(found))
    if missing:
        raise SystemExit(
            f"INVALID: {len(missing)} documents of role {role} could not be "
            f"rebuilt, first: {missing[:3]}"
        )
    return [(t, found[t]) for t in (d["doc"] for d in roles)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--manifests", required=True)
    ap.add_argument("--role", required=True, help="train77 | train154 | train308")
    ap.add_argument("--val-role", default="val19")
    ap.add_argument("--affine", required=True, help="the frozen uncorrected map")
    ap.add_argument("--init", required=True, help="shared residual initialisation")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--cont-len", type=int, default=64)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16", help="frozen model weights")
    ap.add_argument("--solve-dtype", default="float32", help="mapper and training")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    sdtype = getattr(torch, args.solve_dtype)
    t_start = time.time()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    manifests = json.load(open(args.manifests))
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
    pos = torch.arange(args.ctx, device=dev)

    mk, mv, man = freeze.load(args.affine, layout, tg, device=dev)
    mk.cast(sdtype)
    mv.cast(sdtype)
    log_now(
        f"affine loaded, joint hash {man['joint_weight_sha256'][:16]}, "
        f"support fixed at {len(next(iter(mk.maps.values())).layers)} source layers"
    )

    rk = ResidualMapper(mk, layout, tg, "k", args.hidden, linear=True).to(dev)
    rv = ResidualMapper(mv, layout, tg, "v", args.hidden, linear=True).to(dev)
    rk.freeze_affine()
    rv.freeze_affine()
    init_k, init_v = args.init + ".k.pt", args.init + ".v.pt"
    if os.path.exists(init_k):
        hk, hv = rk.load_init(init_k), rv.load_init(init_v)
        log_now(f"residual initialisation loaded, hashes {hk[:12]} / {hv[:12]}")
    else:
        hk, hv = rk.save_init(init_k), rv.save_init(init_v)
        log_now(f"residual initialisation created, hashes {hk[:12]} / {hv[:12]}")

    split_cache = {}
    train = load_role_windows(
        manifests, args.role, tok, args.ctx, args.cont_len, split_cache
    )
    val = load_role_windows(
        manifests, args.val_role, tok, args.ctx, args.cont_len, split_cache
    )
    log_now(f"{len(train)} training documents, {len(val)} validation documents")

    def prepare(items, tag):
        """One reference distribution per document, computed once.

        Source features are not cached: recomputing them costs a 1.5B prefill
        per step and saves several gigabytes of host memory, which matters
        more at three hundred documents than the time does.
        """
        out = []
        for i, (title, ids) in enumerate(items):
            ids = ids.to(dev)
            prompt = ids[: args.ctx].unsqueeze(0)
            cont = ids[args.ctx : args.ctx + args.cont_len].unsqueeze(0)
            with torch.no_grad():
                tp = prefill(models["target"], prompt, tg)
                ref = torch.log_softmax(
                    continuation_logits(
                        models["target"], tp.keys_post, tp.values, cont, args.ctx
                    ).float(),
                    -1,
                ).half()
            out.append(
                {
                    "doc": title,
                    "prompt": prompt.cpu(),
                    "cont": cont.cpu(),
                    "ref_lp": ref.cpu(),
                }
            )
            del tp, ref
            if (i + 1) % 32 == 0:
                log_now(f"  {tag}: prepared {i + 1}/{len(items)}")
        return out

    fit_set = prepare(train, args.role)
    val_set = prepare(val, args.val_role)

    def loss_on(item, grad):
        p_ids = item["prompt"].to(dev)
        with torch.no_grad():
            sp = prefill(models["source"], p_ids, sg)
            xk = flat_features(to_content_keys(sp)).to(sdtype)
            xv = flat_features([v.float() for v in sp.values]).to(sdtype)
        pk, pv = rk(xk), rv(xv)
        keys = [
            rerot(b[0].float(), pos, tg.rope_theta).unsqueeze(0).to(dtype) for b in pk
        ]
        vals = [v.to(dtype) for v in pv]
        fn = continuation_logits_grad if grad else continuation_logits
        lg = fn(models["target"], keys, vals, item["cont"].to(dev), args.ctx)
        return behaviour_loss(item["ref_lp"], lg)

    @torch.no_grad()
    def mean_loss(items, tag):
        """The same objective, the same checkpoint, averaged the same way.

        The earlier record compared a single sampled training step against an
        averaged development figure and read the difference as overfitting.
        Both sides are means here, over whole sets, at the same precision.
        """
        vals = [float(loss_on(it, False)) for it in items]
        return {"mean": float(np.mean(vals)), "n": len(vals), "tag": tag}

    params = list(rk.blocks.parameters()) + list(rv.blocks.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    n_res = rk.n_residual_params + rv.n_residual_params

    # A deterministic shuffled pass over the role, repeated: every document is
    # seen the same number of times up to the remainder, which a
    # sample-with-replacement policy does not guarantee and which matters when
    # the point of the experiment is how many documents there were.
    g = np.random.default_rng(args.seed)
    order = []
    while len(order) < args.steps:
        perm = g.permutation(len(fit_set))
        order.extend(int(i) for i in perm)
    order = order[: args.steps]
    order_hash = hashlib.sha256(json.dumps(order).encode()).hexdigest()[:32]
    log_now(
        f"sample order hash {order_hash}, {len(set(order))} distinct documents "
        f"over {args.steps} updates"
    )

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    losses = []
    for step, idx in enumerate(order):
        loss = loss_on(fit_set[idx], True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 100 == 0:
            log_now(f"  step {step + 1}/{args.steps} loss {np.mean(losses[-100:]):.5f}")
        del loss
    train_seconds = time.time() - t0

    final_train = mean_loss(fit_set, args.role)
    final_val = mean_loss(val_set, args.val_role)
    log_now(
        f"trained in {train_seconds:.0f}s; mean train loss {final_train['mean']:.5f}, "
        f"mean validation loss {final_val['mean']:.5f}"
    )

    probe = torch.randn(8, layout.n_features, device=dev, dtype=sdtype)
    parity = [rk.fold_parity(probe), rv.fold_parity(probe)]
    log_now(
        "fold parity: "
        + ", ".join(
            f"{p['kind']} {p['blocks_checked']} blocks, worst relative "
            f"{p['worst_relative_gap']:.2e}"
            for p in parity
        )
    )
    TOL = 1e-4
    bad = [p for p in parity if p["worst_relative_gap"] > TOL]
    if bad:
        raise SystemExit(f"INVALID: folding is not parity-preserving: {bad}")

    folded_k, folded_v = rk.merge(), rv.merge()
    out_arm = os.path.join(args.out_dir, f"{args.role}_lin.pt")
    src_id = freeze.model_identity(args.source, models["source"], tok)
    tgt_id = freeze.model_identity(args.target, models["target"], tok)
    saved = freeze.save(
        out_arm,
        folded_k,
        folded_v,
        source=src_id,
        target=tgt_id,
        fit_context=args.ctx,
        calib_doc_ids=[d["doc"] for d in manifests["roles"][args.role]],
        dev_doc_ids=[d["doc"] for d in manifests["roles"][args.val_role]],
        eval_doc_ids=[],
        config={**vars(args), "arm": f"{args.role}+lin"},
    )

    bytes_k = folded_k.n_bytes + folded_v.n_bytes
    report = {
        "role": args.role,
        "n_train_documents": len(fit_set),
        "n_val_documents": len(val_set),
        "distinct_documents_seen": len(set(order)),
        "steps": args.steps,
        "lr": args.lr,
        "sample_order_sha256": order_hash,
        "residual_params": n_res,
        "init_hashes": {"k": hk, "v": hv},
        "affine_joint_hash": man["joint_weight_sha256"],
        "valid_supervised_tokens": len(order) * args.cont_len,
        "first_100_mean_loss": float(np.mean(losses[:100])),
        "last_100_mean_loss": float(np.mean(losses[-100:])),
        "mean_train_loss": final_train,
        "mean_val_loss": final_val,
        "train_seconds": train_seconds,
        "wall_seconds": time.time() - t_start,
        "peak_alloc_gib": (
            torch.cuda.max_memory_allocated() / GIB if dev == "cuda" else 0.0
        ),
        "fold_parity": parity,
        "fold_tolerance_relative": TOL,
        "folded_arm": out_arm,
        "folded_joint_hash": saved["joint_weight_sha256"],
        "folded_weight_bytes": bytes_k,
        "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
        "tf32_allowed": (
            bool(torch.backends.cuda.matmul.allow_tf32) if dev == "cuda" else None
        ),
        "dtype": args.dtype,
        "solve_dtype": args.solve_dtype,
        "manifest_role_hash": manifests["role_hashes"][args.role],
    }
    with open(os.path.join(args.out_dir, f"{args.role}_train.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    log_now(
        f"folded arm {out_arm}, {bytes_k / 2**20:.1f} MiB of weights, "
        f"joint hash {saved['joint_weight_sha256'][:16]}"
    )
    log_now(f"wrote {args.out_dir}/{args.role}_train.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

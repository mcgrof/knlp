# SPDX-License-Identifier: GPL-2.0
"""The objective probe: teach the correction to answer, and to stop.

This is the rescue runner with one thing changed, which is why it is built
from that file rather than beside it. Half the updates keep the continuation
objective; the other half ask a question about the document the cache holds
and supervise the answer tokens and the end of the answer.

The reason is a measured failure mode. On the development set, sixteen of the
anchor's twenty-five wrong answers are one edit from the gold code and
fourteen of those are a dropped character, while the target itself and a
latency-matched suffix prefill never do this. Continuing prose and emitting an
exact token then stopping are different demands, and only the first has ever
been supervised.

Everything else is held to the rescue's settings: the same frozen map and
support, the same serialised zero-output initialisation, the same rate,
updates, precision and data roles, no warm start from a trained arm. What
varies is what half the examples ask for.

Original docstring follows.

Train the correction on a declared set of documents, and change nothing else.

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
    make_cache,
)
from research.kv_translate.run_a1 import Mapper, flat_features  # noqa: E402
from research.kv_translate.eos_objective import (  # noqa: E402
    answer_logits,
    answer_loss_from_logits,
    build_labels,
    legacy_answer_loss_from_logits,
    masks_for_arm,
)
from research.kv_translate.objective_examples import (  # noqa: E402
    build_objective_examples,
    supervised_targets,
    verify_examples,
)
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
    ap.add_argument(
        "--source-revision",
        default="main",
        help="immutable snapshot id for the source weights. A config hash "
        "identifies a configuration, not the weight files, and 'main' moves, "
        "so both arms and every control pin this explicitly.",
    )
    ap.add_argument("--target-revision", default="main")
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
    ap.add_argument(
        "--objective-mix",
        type=float,
        default=0.0,
        help="fraction of updates spent on answer/termination examples "
        "built from the training documents; 0.0 reproduces the rescue",
    )
    ap.add_argument(
        "--forbid-codes",
        default="",
        help="a gold manifest whose answers may never appear in training",
    )
    ap.add_argument(
        "--require-sentinel",
        default="",
        help="a sentinel.json that must record passed=true before training; "
        "a run whose device did not qualify is not evidence about the method",
    )
    ap.add_argument(
        "--eos-arm",
        default="on",
        choices=("on", "off", "legacy_off"),
        help="which terminal-supervision arm to train. 'on' supervises the "
        "answer and its end. 'off' supervises the answer and masks the end, "
        "keeping the same labels, the same fed sequence and the same "
        "denominator, so the two arms differ by that one term and nothing "
        "else. 'legacy_off' reproduces the earlier arm, which deleted the "
        "terminal label and so also shrank the mean it divided by; it exists "
        "to reproduce that run, not to test termination, and its output is "
        "stamped confounded.",
    )
    ap.add_argument(
        "--require-init",
        action="store_true",
        help="refuse to mint a fresh residual initialisation. Set for any "
        "matched comparison: an arm that starts somewhere else is not a "
        "control, and nothing in the recorded fields would show it.",
    )
    ap.add_argument(
        "--expect-train-documents",
        type=int,
        default=0,
        help="assert the contract's declared training-document count once it "
        "is knowable, rather than trusting that staging delivered it.",
    )
    ap.add_argument("--expect-examples", type=int, default=0)
    ap.add_argument("--expect-example-documents", type=int, default=0)
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

    if args.require_sentinel:
        sent = json.load(open(args.require_sentinel))
        if not sent.get("passed"):
            bad = [
                f"{k} measured {v['measured']:.3e} against {v['limit']:.0e}"
                for k, v in sent.get("required_checks", {}).items()
                if not v.get("passed")
            ]
            raise SystemExit(
                "INVALID: the sentinel for this device did not pass, so any "
                "result trained here is a measurement of an unqualified "
                "configuration. Failing checks: " + "; ".join(bad or ["unknown"])
            )
        log_now(f"sentinel {args.require_sentinel} passed; proceeding")

    manifests = json.load(open(args.manifests))
    tok = AutoTokenizer.from_pretrained(args.target, revision=args.target_revision)
    eos_id = tok.eos_token_id
    for t in ("<|im_end|>",):
        i = tok.convert_tokens_to_ids(t)
        if i is not None and i >= 0:
            eos_id = i

    models, geom = {}, {}
    revs = {"source": args.source_revision, "target": args.target_revision}
    for role, mid in (("source", args.source), ("target", args.target)):
        m = AutoModelForCausalLM.from_pretrained(
            mid, revision=revs[role], dtype=dtype, attn_implementation="sdpa"
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
    if os.path.exists(init_k) and os.path.exists(init_v):
        hk, hv = rk.load_init(init_k), rv.load_init(init_v)
        log_now(f"residual initialisation loaded, hashes {hk[:12]} / {hv[:12]}")
    elif args.require_init:
        # Two arms that start from different weights are not a matched pair,
        # and an absent initialisation is the one way that happens silently.
        # Minting a replacement here would produce a comparison that looks
        # identical in every recorded field and is not one.
        raise SystemExit(
            f"BLOCKED_INPUTS: the frozen initialisation {init_k} / {init_v} is "
            "absent. It is not created here, because an arm started from a "
            "fresh initialisation is not matched to one that was not."
        )
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
    if args.expect_train_documents and len(train) != args.expect_train_documents:
        raise SystemExit(
            f"BLOCKED_INPUTS: {len(train)} training documents, the contract "
            f"declares {args.expect_train_documents}"
        )

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

    obj_set = []
    if args.objective_mix > 0:
        forbid = set()
        if args.forbid_codes:
            forbid = {
                d["retrieval_gold"]
                for d in json.load(open(args.forbid_codes))["documents"]
                if "retrieval_gold" in d
            }
        train_titles = {t for t, _ in train}
        bodies = {}
        for sp_name in sorted({d["split"] for d in manifests["roles"][args.role]}):
            if sp_name not in split_cache:
                split_cache[sp_name] = {
                    t: b for t, b in wikitext_documents(split=sp_name, max_docs=100000)
                }
            for t, b in split_cache[sp_name].items():
                if t in train_titles:
                    bodies[t] = b
        ordered = [(t, bodies[t]) for t, _ in train if t in bodies]
        ex = build_objective_examples(
            ordered, tok, args.ctx, seed=args.seed, forbid_codes=forbid
        )
        problems = verify_examples(
            tok,
            ex,
            eos_id,
            gold_answers=forbid,
            supervise_eos=args.eos_arm != "legacy_off",
        )
        if problems:
            raise SystemExit(
                f"INVALID: {len(problems)} objective examples are unusable, "
                f"first: {problems[:3]}"
            )
        for e in ex:
            if args.eos_arm == "legacy_off":
                q, lab = supervised_targets(tok, e, eos_id, supervise_eos=False)
                terminal = [0] * len(lab)
            else:
                q, lab, terminal = build_labels(tok, e, eos_id)
            valid, contribute = masks_for_arm(terminal, args.eos_arm)
            obj_set.append(
                {
                    "doc": e.doc_id,
                    "kind": e.kind,
                    "prompt": torch.tensor(e.prompt_ids).unsqueeze(0),
                    "query_ids": q,
                    "labels": torch.tensor(lab).unsqueeze(0),
                    "n_labels": len(lab),
                    "valid": valid.unsqueeze(0),
                    "contribute": contribute.unsqueeze(0),
                    # The denominator is the same in both arms by construction;
                    # recorded per example so the receipt can show it.
                    "denominator": float(valid.sum()),
                    "supervised_terms": float(contribute.sum()),
                }
            )
        if args.expect_examples and len(obj_set) != args.expect_examples:
            raise SystemExit(
                f"BLOCKED_INPUTS: built {len(obj_set)} objective examples, the "
                f"contract declares {args.expect_examples}. The arms are only "
                "matched if they see the same examples."
            )
        n_docs_with = len({o["doc"] for o in obj_set})
        if (
            args.expect_example_documents
            and n_docs_with != args.expect_example_documents
        ):
            raise SystemExit(
                f"BLOCKED_INPUTS: examples span {n_docs_with} documents, the "
                f"contract declares {args.expect_example_documents}"
            )
        kinds = {}
        for o in obj_set:
            kinds[o["kind"]] = kinds.get(o["kind"], 0) + 1
        log_now(
            f"{len(obj_set)} objective examples over "
            f"{len({o['doc'] for o in obj_set})} training documents: {kinds}; "
            f"all answers verified present in their prompt, "
            + (
                "all label sequences end at EOS"
                if args.eos_arm != "legacy_off"
                else "no label sequence carries EOS (the earlier arm, which also "
                "shortened its denominator)"
            )
            + f", no training code collides with a "
            f"held-out answer ({len(forbid)} checked)"
        )

    def answer_loss(item, grad):
        """Cross-entropy on the answer tokens, and on its end in the on arm.

        The first answer token is predicted from the query's last position,
        so the logits used are those at positions [len(query)-1 .. ]. Getting
        that offset wrong by one trains the model to continue an answer it
        never learns to start, which is exactly the failure being corrected.
        """
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
        q = item["query_ids"].to(dev)
        lab = item["labels"].to(dev)
        feed = torch.cat([q, lab[:, :-1]], dim=1)
        n_prompt = p_ids.shape[1]
        total = n_prompt + feed.shape[1]
        out = models["target"](
            input_ids=feed,
            attention_mask=torch.ones(1, total, dtype=torch.long, device=dev),
            position_ids=torch.arange(n_prompt, total, device=dev).unsqueeze(0),
            past_key_values=make_cache(keys, vals),
            use_cache=True,
        ).logits
        # positions q_len-1 .. end predict label[0] .. label[-1]
        pred = answer_logits(out, q.shape[1])
        if args.eos_arm == "legacy_off":
            return legacy_answer_loss_from_logits(pred, lab)
        return answer_loss_from_logits(
            pred, lab, item["valid"].to(dev), item["contribute"].to(dev)
        )

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

    def cycle(n, k):
        out = []
        while len(out) < k:
            out.extend(int(i) for i in g.permutation(n))
        return out[:k]

    if obj_set:
        # The schedule is fixed before the run and hashed. Interleaving is
        # deterministic rather than sampled, so the two objectives are mixed
        # in a known order and a rerun sees the same one.
        n_obj = int(round(args.steps * args.objective_mix))
        n_cont = args.steps - n_obj
        co, oo = cycle(len(fit_set), n_cont), cycle(len(obj_set), n_obj)
        order = []
        ci = oi = 0
        for step in range(args.steps):
            take_obj = oi < n_obj and (
                ci >= n_cont or (step * n_obj) // args.steps >= oi
            )
            if take_obj:
                order.append(("obj", oo[oi]))
                oi += 1
            else:
                order.append(("cont", co[ci]))
                ci += 1
    else:
        order = [("cont", i) for i in cycle(len(fit_set), args.steps)]
    order_hash = hashlib.sha256(json.dumps(order).encode()).hexdigest()[:32]
    n_obj_steps = sum(1 for k, _ in order if k == "obj")
    log_now(
        f"sample order hash {order_hash}: {args.steps} updates, "
        f"{n_obj_steps} on answer/termination examples and "
        f"{args.steps - n_obj_steps} on continuation"
    )

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    losses = []
    valid_tokens = 0
    for step, (which, idx) in enumerate(order):
        if which == "obj":
            item = obj_set[idx]
            loss = answer_loss(item, True)
            valid_tokens += item["n_labels"]
        else:
            loss = loss_on(fit_set[idx], True)
            valid_tokens += args.cont_len
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

    # The residual before folding, kept so fold equivalence can be checked on
    # this artifact rather than recorded unavailable as it was for the probe.
    premerge = os.path.join(args.out_dir, f"{args.role}_premerge.pt")
    torch.save(
        {
            "contract": "premerge_residual_v1",
            "eos_arm": args.eos_arm,
            "source_revision": args.source_revision,
            "target_revision": args.target_revision,
            "k": {k: v.detach().cpu() for k, v in rk.blocks.state_dict().items()},
            "v": {k: v.detach().cpu() for k, v in rv.blocks.state_dict().items()},
        },
        premerge,
    )
    log_now(f"saved premerge residual {premerge}")

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
        "distinct_documents_seen": len({i for k, i in order if k == "cont"}),
        "steps": args.steps,
        "lr": args.lr,
        "sample_order_sha256": order_hash,
        "residual_params": n_res,
        "init_hashes": {"k": hk, "v": hv},
        "affine_joint_hash": man["joint_weight_sha256"],
        "valid_supervised_tokens": valid_tokens,
        "eos_arm": args.eos_arm,
        "source_revision": args.source_revision,
        "target_revision": args.target_revision,
        "eos_arm_contract": (
            "labels, fed sequence and denominator are identical across the on "
            "and off arms; the arm masks the terminal term in the numerator "
            "only. legacy_off reproduces the earlier arm, which deleted the "
            "terminal label and shrank the denominator with it, and is "
            "confounded by construction."
        ),
        "eos_arm_confounded": args.eos_arm == "legacy_off",
        "denominator_total": float(sum(o["denominator"] for o in obj_set)),
        "supervised_terms_total": float(sum(o["supervised_terms"] for o in obj_set)),
        "objective_mix": args.objective_mix,
        "objective_updates": n_obj_steps,
        "objective_examples": len(obj_set),
        "continuation_updates": args.steps - n_obj_steps,
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
        # Retained rather than summarised: a per-update trace and the exact
        # order are what let two arms be compared update by update, and a mean
        # over the last hundred cannot be recovered from afterwards.
        "update_order": order,
        "loss_trace": losses,
        "per_example_denominators": [
            {
                "doc": o["doc"],
                "kind": o["kind"],
                "n_labels": o["n_labels"],
                "denominator": o.get("denominator"),
                "supervised_terms": o.get("supervised_terms"),
            }
            for o in obj_set
        ],
        "premerge_residual": premerge,
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

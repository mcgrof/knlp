#!/usr/bin/env python3
"""Fit a cheaper map, by narrowing what each target block reads.

Applying the fitted map is the largest single term in the path a translated
cache is served on -- more than running the source model over the whole
document -- so making it cheaper is the one change that moves the economics.
The map is a per-block affine, and its cost is set by how many source features
each block reads. The fitted map in hand reads sixteen source layers; these
read four.

Two ways to narrow, fitted in order. Keeping every source head over four layers
quarters the input. Keeping only the source head a target head corresponds to
halves it again, which is possible here only because the pair's head counts
divide: the source carries two key/value heads and the receiver four, so each
receiver head is assigned one source head and that assignment is declared
rather than inferred.

Which four layers is the part worth care. Each block already carries a
sixteen-layer support chosen when the original map was fitted, and those are
stored **in depth order, not in importance order**, so taking the first four
would be taking the shallowest four for no reason. They are re-ranked here by
the residual each leaves on its own, and the ranking is computed from
calibration documents only -- never from the documents the screen scores, whose
questions and answers the fit must not see.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from research.kv_translate import freeze  # noqa: E402
from research.kv_translate.fit import (  # noqa: E402
    Accumulator,
    SourceLayout,
    fit_block,
    solve_ridge,
)
from research.kv_translate.pairs import describe, prefill, to_content_keys  # noqa: E402
from research.kv_translate.run_a1 import (
    Mapper,
    flat_features,
    target_blocks,
)  # noqa: E402

# Which source head each receiver head reads, for the head-local variant. The
# receiver carries four key/value heads and the source two; the pairing is
# declared here rather than derived, because a derivation would silently do
# something else for a pair whose counts did not divide.
HEAD_ASSIGNMENT = (0, 0, 1, 1)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def rank_within_support(acc, target, head, support, ridge, head_local, k):
    """The ``k`` layers of this block's own support that leave least behind.

    Restricted to the support the original fit chose for this block, which is
    what the plan asks for: the question is which of those sixteen carry the
    block, not which of all twenty-eight. Each candidate costs one small solve
    on a slice of statistics already accumulated, so this is cheap.

    Ties break by depth, so the selection is reproducible rather than dependent
    on the order a sort happened to produce.
    """
    scored = []
    for li in support:
        cols = acc.layout.columns_for([li], head, head_local)
        G, C, _, _ = acc.centred(cols, target)
        M = solve_ridge(G, C, ridge)
        residual = -2.0 * float((M * C).sum()) + float(
            (M.transpose(0, 1) @ G @ M).diagonal().sum()
        )
        scored.append((residual, li))
    scored.sort(key=lambda x: (x[0], x[1]))
    return tuple(sorted(li for _, li in scored[:k]))


def calibration_documents(manifests_path, role, n_docs, seed_tag="e2-compact-v1"):
    """Documents to fit on, in an order that does not depend on anything measured.

    Drawn from the training role the original map used, ranked by a hash of the
    document identity so the same set falls out anywhere. The development and
    evaluation roles are not touched: a map fitted on the documents a screen
    scores would be measuring its own training set.
    """
    man = json.load(open(manifests_path))
    if role not in man["roles"]:
        raise SystemExit(f"INVALID: role {role!r} not in the manifest")
    entries = man["roles"][role]
    ranked = sorted(
        entries,
        key=lambda e: hashlib.sha256(f"{seed_tag}|{e['doc']}".encode()).hexdigest(),
    )
    return [e["doc"] for e in ranked[:n_docs]], entries[0]["split"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--source-revision", required=True)
    ap.add_argument("--target-revision", required=True)
    ap.add_argument(
        "--reference-arm",
        required=True,
        help="the fitted map whose per-block supports the compact one narrows",
    )
    ap.add_argument("--manifests", required=True)
    ap.add_argument(
        "--documents-json",
        default="",
        help="calibration text extracted beforehand, as {title: text}. The "
        "corpus loader pulls a multi-gigabyte dataset over the network, which "
        "is a dependency a rented machine does not need and a way for a run "
        "to fail after the weights are already paid for.",
    )
    ap.add_argument("--role", default="train154")
    ap.add_argument("--docs", type=int, default=64)
    ap.add_argument("--tokens-each", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--support-k", type=int, default=4)
    ap.add_argument(
        "--variants",
        required=True,
        help="comma-separated: full_head, head_local. Both read slices of the "
        "same accumulated statistics, so fitting them together costs one pass "
        "over the calibration documents rather than two.",
    )
    ap.add_argument("--ridge", type=float, default=0.0, help="0 reuses the reference's")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument(
        "--out-prefix",
        required=True,
        help="each variant is written to <prefix>_<variant>.pt",
    )
    args = ap.parse_args()
    variants = [v for v in args.variants.split(",") if v]
    for v in variants:
        if v not in ("full_head", "head_local"):
            raise SystemExit(f"INVALID: unknown variant {v!r}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not args.documents_json:
        from research.kv_translate.tasks import wikitext_documents

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    tok_for_identity = AutoTokenizer.from_pretrained(
        args.target, revision=args.target_revision
    )
    models, geom, identity = {}, {}, {}
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
        # The full identity the freeze format records -- geometry, tokenizer
        # and template hashes -- rather than a name and a revision string. A
        # reader can check the former.
        ident = freeze.model_identity(mid, m, tok_for_identity)
        ident["revision_requested"] = rev
        identity[role] = ident
    sg, tg = geom["source"], geom["target"]
    layout = SourceLayout(sg.n_layers, sg.n_kv_heads, sg.head_dim)
    if "head_local" in variants and tg.n_kv_heads != len(HEAD_ASSIGNMENT):
        raise SystemExit(
            f"INVALID: the head assignment names {len(HEAD_ASSIGNMENT)} receiver "
            f"heads and the receiver has {tg.n_kv_heads}"
        )
    if "head_local" in variants and max(HEAD_ASSIGNMENT) >= sg.n_kv_heads:
        raise SystemExit(
            f"INVALID: the head assignment reads source head "
            f"{max(HEAD_ASSIGNMENT)} and the source has {sg.n_kv_heads}"
        )
    log(
        f"source {sg.n_layers}x{sg.n_kv_heads}x{sg.head_dim} -> "
        f"target {tg.n_layers}x{tg.n_kv_heads}x{tg.head_dim}, variants {variants}"
    )

    # ---- the supports to narrow, and the ridge to reuse ------------------
    blob = torch.load(args.reference_arm, map_location="cpu", weights_only=False)
    ref_blocks, ref_man = blob["blocks"], blob["manifest"]
    supports = {
        (k.split(".")[0], int(k.split(".")[1]), int(k.split(".")[2])): tuple(
            rec["layers"]
        )
        for k, rec in ref_blocks.items()
    }
    ridge = args.ridge or float(next(iter(ref_blocks.values()))["ridge"])
    ref_support_size = len(next(iter(supports.values())))
    log(
        f"reference {ref_man['joint_weight_sha256'][:16]}: {len(supports)} blocks, "
        f"support {ref_support_size}, ridge {ridge:.4g}"
    )
    if args.support_k >= ref_support_size:
        raise SystemExit(
            f"INVALID: asked for {args.support_k} of {ref_support_size} layers, "
            "which narrows nothing"
        )

    # ---- calibration, from the training role only -------------------------
    docs, split = calibration_documents(args.manifests, args.role, args.docs)
    tok = AutoTokenizer.from_pretrained(args.target, revision=args.target_revision)
    if args.documents_json:
        by_title = json.load(open(args.documents_json))
        log(f"calibration text read from {args.documents_json}")
    else:
        by_title = dict(wikitext_documents(split=split, max_docs=100000))
    missing = [d for d in docs if d not in by_title]
    if missing:
        raise SystemExit(f"INVALID: {len(missing)} calibration documents not found")
    log(f"{len(docs)} calibration documents from role {args.role}, split {split}")

    n_targets = tg.n_layers * tg.n_kv_heads
    accs = {
        kind: Accumulator(layout, n_targets, tg.head_dim, device=dev)
        for kind in ("k", "v")
    }
    t0 = time.time()
    used = 0
    for i, title in enumerate(docs, 1):
        ids = (
            tok(by_title[title], return_tensors="pt")
            .input_ids[:, : args.tokens_each]
            .to(dev)
        )
        if ids.shape[1] < 64:
            continue
        with torch.no_grad():
            sp = prefill(models["source"], ids, sg)
            tp = prefill(models["target"], ids, tg)
            xk = flat_features(to_content_keys(sp))
            xv = flat_features([v.float() for v in sp.values])
            yk = target_blocks(to_content_keys(tp))
            yv = target_blocks([v.float() for v in tp.values])
            # Every stride-th token. The statistics a ridge solve needs are
            # sums over tokens, and neighbouring positions in one document are
            # highly correlated, so a stride buys independence per unit of
            # arithmetic rather than costing information.
            sl = slice(None, None, args.stride)
            accs["k"].add(xk[sl], yk[:, sl])
            accs["v"].add(xv[sl], yv[:, sl])
        used += 1
        if i % 16 == 0:
            log(
                f"  {i}/{len(docs)} documents, {accs['k'].n} tokens, "
                f"{time.time() - t0:.0f}s"
            )
    log(
        f"accumulated {accs['k'].n} tokens from {used} documents in "
        f"{time.time() - t0:.0f}s"
    )

    # ---- narrow each block's support, then fit it -------------------------
    results = []
    for variant in variants:
        head_local = variant == "head_local"
        t1 = time.time()
        maps = {"k": {}, "v": {}}
        chosen, same_as_shallowest = {}, 0
        for kind in ("k", "v"):
            t = 0
            for li in range(tg.n_layers):
                for h in range(tg.n_kv_heads):
                    src_head = HEAD_ASSIGNMENT[h] if head_local else 0
                    support = supports[(kind, li, h)]
                    sel = rank_within_support(
                        accs[kind],
                        t,
                        src_head,
                        support,
                        ridge,
                        head_local,
                        args.support_k,
                    )
                    chosen[f"{kind}.{li}.{h}"] = list(sel)
                    if sel == tuple(sorted(support)[: args.support_k]):
                        same_as_shallowest += 1
                    maps[kind][(li, h)] = fit_block(
                        accs[kind],
                        target=t,
                        target_layer=li,
                        head=src_head,
                        kind=kind,
                        layers=sel,
                        ridge=ridge,
                        head_local=head_local,
                    )
                    t += 1
            log(
                f"  {variant}/{kind}: {len(maps[kind])} blocks, {time.time() - t1:.0f}s"
            )

        out_path = f"{args.out_prefix}_{variant}.pt"
        mk = Mapper(maps["k"], layout, tg, "k")
        mv = Mapper(maps["v"], layout, tg, "v")
        man = freeze.save(
            out_path,
            mk,
            mv,
            source=identity["source"],
            target=identity["target"],
            fit_context=args.tokens_each,
            calib_doc_ids=docs,
            dev_doc_ids=[],
            eval_doc_ids=[],
            config={
                "variant": variant,
                "support_k": args.support_k,
                "head_assignment": list(HEAD_ASSIGNMENT) if head_local else None,
                "reference_arm": os.path.abspath(args.reference_arm),
                "reference_joint_sha256": ref_man["joint_weight_sha256"],
                "reference_support_size": ref_support_size,
                "ridge": ridge,
                "ridge_source": "reference" if not args.ridge else "explicit",
                "calibration_role": args.role,
                "calibration_documents": used,
                "calibration_tokens": int(accs["k"].n),
                "stride": args.stride,
                "support_rule": (
                    "single-layer residual ranking restricted to each block's "
                    "own reference support, ties broken by depth; calibration "
                    "documents only, never the documents a screen scores"
                ),
            },
        )
        json.dump(
            {
                "contract": "compact_support_v1",
                "variant": variant,
                "support_k": args.support_k,
                "head_assignment": list(HEAD_ASSIGNMENT) if head_local else None,
                "reference_joint_sha256": ref_man["joint_weight_sha256"],
                "blocks_whose_choice_equals_the_shallowest": same_as_shallowest,
                "blocks": len(chosen),
                "note": (
                    "the reference supports are stored in depth order, so a "
                    "block whose re-ranked choice equals the shallowest four "
                    "is worth counting rather than being the default"
                ),
                "chosen": chosen,
            },
            open(out_path + ".supports.json", "w"),
            indent=1,
        )
        d_full = ref_support_size * sg.n_kv_heads * sg.head_dim
        d_sel = args.support_k * (1 if head_local else sg.n_kv_heads) * sg.head_dim
        results.append(
            {
                "variant": variant,
                "path": os.path.abspath(out_path),
                "joint_weight_sha256": man["joint_weight_sha256"],
                "features_per_block": d_sel,
                "reference_features_per_block": d_full,
                "narrowing": d_full / d_sel,
                "serialized_bytes": man["serialized_bytes"],
                "blocks_choosing_the_shallowest": same_as_shallowest,
                "blocks": len(chosen),
            }
        )
        log(
            f"  {variant}: {d_full} -> {d_sel} features ({d_full / d_sel:.1f}x), "
            f"{man['serialized_bytes'] / 1e6:.0f} MB, joint "
            f"{man['joint_weight_sha256'][:16]}"
        )
        log(
            f"  {variant}: {same_as_shallowest}/{len(chosen)} blocks chose the "
            f"shallowest four"
        )

    json.dump(
        {
            "contract": "compact_fit_v1",
            "calibration": {
                "role": args.role,
                "documents": used,
                "tokens": int(accs["k"].n),
                "tokens_each": args.tokens_each,
                "stride": args.stride,
                "document_ids": docs,
            },
            "reference_joint_sha256": ref_man["joint_weight_sha256"],
            "ridge": ridge,
            "support_k": args.support_k,
            "variants": results,
            "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
            "models": {
                "source": {"id": args.source, "revision": args.source_revision},
                "target": {"id": args.target, "revision": args.target_revision},
            },
        },
        open(f"{args.out_prefix}_FIT.json", "w"),
        indent=2,
    )
    log(f"wrote {args.out_prefix}_FIT.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

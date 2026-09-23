# SPDX-License-Identifier: GPL-2.0
"""Build a tiny two-model pair the real runner can be driven against.

The point is not to predict the rented run's numbers. It is to exercise the
runner's own control flow — condition parsing, translation, selection, repair,
scoring, resume — on a processor, before a card is paying for the first
discovery that a flag was never wired up. A four-second argparse failure has
already cost one allocation on this lane.

Two random Qwen2 models with the receiver's shape relationship (more layers,
more key-value heads, same head width) and a fitted map between them, plus a
manifest in the fixture's own schema whose answers really are present in the
contexts. Everything is seeded and written once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def tiny_config(layers, kv_heads, heads, head_dim, vocab):
    from transformers import Qwen2Config

    hidden = heads * head_dim
    return Qwen2Config(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        max_position_embeddings=2048,
        tie_word_embeddings=False,
    )


def build_models(out_dir, tok):
    """Two random models with the receiver's shape relationship to the source.

    The tokenizer is written beside each one. A model directory without it
    loads as a model and fails as a pair, which is how the first attempt at
    this fixture got a stack trace out of the runner rather than a result.
    """
    from transformers import Qwen2ForCausalLM

    torch.manual_seed(11)
    pair = {}
    for role, (L, KV, H, D) in {
        "source": (2, 1, 4, 32),  # fewer layers, one key-value head
        "target": (3, 2, 4, 32),  # more layers, two, same head width
    }.items():
        m = Qwen2ForCausalLM(tiny_config(L, KV, H, D, len(tok)))
        m = m.eval().to(torch.float32)
        path = os.path.join(out_dir, role)
        m.save_pretrained(path)
        tok.save_pretrained(path)
        pair[role] = path
    return pair


def build_manifest(path, tok, ctx, n_docs, seed=7):
    """A manifest in the v2 schema, with answers that really are present."""
    rng = np.random.default_rng(seed)
    docs = []
    for i in range(n_docs):
        code = "".join(rng.choice(list(CODE_ALPHABET), size=8))
        filler = rng.integers(20, 400, size=ctx).tolist()
        depth = 0.15 + 0.7 * (i % 4) / 3.0
        at = int(depth * ctx)
        code_ids = tok(code, add_special_tokens=False).input_ids
        ids = filler[:at] + code_ids + filler[at:]
        ids = ids[:ctx]
        # The cloze answer is a run of real tokens from the document body,
        # taken from a region the planted code does not overlap.
        cs = (at + len(code_ids) + 8) % max(ctx - 16, 1)
        cloze_ids = ids[cs : cs + 4]
        docs.append(
            {
                "doc": f"tiny_{i:02d}",
                "dev_index": i,
                "quartile": (i // 4) % 4,
                "in_screen": i % 2 == 0,
                "prompt_ids": ids,
                "prompt_len": len(ids),
                "prompt_ids_sha256_le_i64": hashlib.sha256(
                    np.asarray(ids, dtype="<i8").tobytes()
                ).hexdigest(),
                "retrieval_query": f"\nWhat is the code in document {i}? Answer: ",
                "retrieval_gold": code,
                "cloze_query": "\nContinue the passage: ",
                "cloze_gold": tok.decode(cloze_ids),
                "free_query": "\nWrite a short paragraph. ",
                "plant_depth_frac": depth,
                "plant_depth_tokens": at,
                "wrong_source_doc": f"tiny_{(i + 1) % n_docs:02d}",
            }
        )
    man = {
        "contract": "kv_translate_recovery_fixture",
        "version": "cpu-tiny-v1",
        "ctx": ctx,
        "documents": docs,
        "task_joint_sha256": hashlib.sha256(
            json.dumps([d["prompt_ids_sha256_le_i64"] for d in docs]).encode()
        ).hexdigest(),
    }
    json.dump(man, open(path, "w"))
    return man


def fit_tiny_mapper(paths, man, out_path, ctx, n_calib=6, ridge=1e-2):
    """Fit a real map between the two tiny models and freeze it.

    A random map would exercise the same plumbing, but a frozen artifact whose
    weights were never solved cannot distinguish a translation bug from a bad
    fit, and the repair conditions are read against the unrepaired mapping.
    """
    from transformers import AutoModelForCausalLM

    from research.kv_translate import freeze
    from research.kv_translate.fit import Accumulator, SourceLayout, fit_block
    from research.kv_translate.pairs import describe, prefill, to_content_keys
    from research.kv_translate.run_a1 import Mapper, flat_features, target_blocks

    models, geom = {}, {}
    for role in ("source", "target"):
        m = AutoModelForCausalLM.from_pretrained(paths[role], dtype=torch.float32)
        m.eval()
        models[role], geom[role] = m, describe(m, paths[role])
    sg, tg = geom["source"], geom["target"]
    layout = SourceLayout(sg.n_layers, sg.n_kv_heads, sg.head_dim)
    n_targets = tg.n_layers * tg.n_kv_heads

    rng = np.random.default_rng(99)
    accs = {k: Accumulator(layout, n_targets, tg.head_dim) for k in ("k", "v")}
    with torch.no_grad():
        for _ in range(n_calib):
            ids = torch.tensor(
                rng.integers(20, 400, size=(1, ctx)).tolist(), dtype=torch.long
            )
            sp = prefill(models["source"], ids, sg)
            tp = prefill(models["target"], ids, tg)
            accs["k"].add(
                flat_features(to_content_keys(sp)),
                target_blocks(to_content_keys(tp)),
            )
            accs["v"].add(
                flat_features([v.float() for v in sp.values]),
                target_blocks([v.float() for v in tp.values]),
            )

    maps = {"k": {}, "v": {}}
    layers = tuple(range(sg.n_layers))
    for kind in ("k", "v"):
        t = 0
        for li in range(tg.n_layers):
            for h in range(tg.n_kv_heads):
                maps[kind][(li, h)] = fit_block(
                    accs[kind],
                    target=t,
                    target_layer=li,
                    head=h,
                    kind=kind,
                    layers=layers,
                    ridge=ridge,
                    head_local=False,  # the pair does not share head counts
                )
                t += 1
    mk = Mapper(maps["k"], layout, tg, "k")
    mv = Mapper(maps["v"], layout, tg, "v")
    return freeze.save(
        out_path,
        mk,
        mv,
        source={"id": paths["source"], "revision": "local-random"},
        target={"id": paths["target"], "revision": "local-random"},
        fit_context=ctx,
        calib_doc_ids=[f"synthetic_{i}" for i in range(n_calib)],
        dev_doc_ids=[d["doc"] for d in man["documents"]],
        eval_doc_ids=[],
        config={"ridge": ridge, "support": "full fan-in, all source layers"},
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--docs", type=int, default=4)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    paths = build_models(args.out_dir, tok)
    man_path = os.path.join(args.out_dir, "manifest.json")
    man = build_manifest(man_path, tok, args.ctx, args.docs)
    arm = os.path.join(args.out_dir, "tiny_arm.pt")
    amn = fit_tiny_mapper(paths, man, arm, args.ctx)
    json.dump(
        {
            "source": paths["source"],
            "target": paths["target"],
            "manifest": man_path,
            "arm": arm,
            "joint_weight_sha256": amn["joint_weight_sha256"],
        },
        open(os.path.join(args.out_dir, "fixture.json"), "w"),
        indent=1,
    )
    print(f"fixture in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

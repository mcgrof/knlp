# SPDX-License-Identifier: GPL-2.0
"""Walk the whole CPU side of training before any of it costs GPU time.

Two attempts have now died seconds into training on things a check could have
caught: arguments the parser did not declare, and a corpus module that was
never installed. Both were invisible to the local tests, which exercise the
loss rather than the run, and to the earlier preflight, which imported the
top-level modules but never reached an import made lazily inside a function.

So this does what the trainer does, minus the model: resolve the weights,
verify the frozen inputs, read the corpus, cut the role windows, build the
objective examples, check them, and build both arms' masks. If it passes,
what remains untested is the part that genuinely needs the card.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--contract", required=True)
    ap.add_argument("--source", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--source-revision", required=True)
    ap.add_argument("--target-revision", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    checks, problems = {}, []

    def ok(name, cond, detail=""):
        checks[name] = {"passed": bool(cond), "detail": detail}
        if not cond:
            problems.append(f"{name}: {detail}")
        print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail}")
        return bool(cond)

    c = json.load(open(args.contract))
    sched = c["schedule"]
    exp = c["expected"]

    # 1. Every import the run reaches, including the ones made inside
    #    functions, which the previous preflight never touched.
    try:
        import datasets  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401

        from research.kv_translate.eos_objective import build_labels, masks_for_arm
        from research.kv_translate.objective_examples import (
            build_objective_examples,
            verify_examples,
        )
        from research.kv_translate.run_probe import load_role_windows
        from research.kv_translate.tasks import wikitext_documents

        ok("imports", True, f"datasets {datasets.__version__}")
    except Exception as e:
        ok("imports", False, f"{type(e).__name__}: {e}")
        json.dump(
            {"passed": False, "checks": checks, "problems": problems},
            open(args.out, "w"),
            indent=2,
        )
        return 1

    # 2. The weights resolve offline at the pinned revisions.
    from transformers import AutoConfig, AutoTokenizer

    for role, mid, rev in (
        ("source", args.source, args.source_revision),
        ("target", args.target, args.target_revision),
    ):
        try:
            cfg = AutoConfig.from_pretrained(mid, revision=rev)
            ok(f"resolve_{role}", True, f"{rev[:12]} {cfg.num_hidden_layers}L")
        except Exception as e:
            ok(f"resolve_{role}", False, f"{type(e).__name__}: {e}")
    tok = AutoTokenizer.from_pretrained(args.target, revision=args.target_revision)

    # 3. The frozen inputs are present and are the contracted bytes.
    for name, rec in c["inputs"].items():
        local = os.path.join(args.artifacts, os.path.basename(rec["path"]))
        if not os.path.exists(local):
            ok(f"input_{name}", False, "absent")
            continue
        h = hashlib.sha256()
        with open(local, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        ok(f"input_{name}", h.hexdigest() == rec["sha256"], "")

    if problems:
        json.dump(
            {"passed": False, "checks": checks, "problems": problems},
            open(args.out, "w"),
            indent=2,
        )
        print(f"BLOCKED_INPUTS: {problems[:3]}", file=sys.stderr)
        return 1

    # 4. The corpus reads, and the role windows cut to the declared counts.
    manifests = json.load(open(os.path.join(args.artifacts, "manifests.json")))
    try:
        split_cache = {}
        train = load_role_windows(
            manifests, sched["role"], tok, sched["ctx"], sched["cont_len"], split_cache
        )
        val = load_role_windows(
            manifests,
            sched["val_role"],
            tok,
            sched["ctx"],
            sched["cont_len"],
            split_cache,
        )
        ok(
            "train_documents",
            len(train) == exp["train_documents"],
            f"{len(train)} vs {exp['train_documents']}",
        )
        ok("val_documents", len(val) > 0, f"{len(val)}")
    except Exception as e:
        ok("role_windows", False, f"{type(e).__name__}: {e}")
        json.dump(
            {"passed": False, "checks": checks, "problems": problems},
            open(args.out, "w"),
            indent=2,
        )
        return 1

    # 5. The examples build, count as declared, and pass their own checks.
    gold = json.load(open(os.path.join(args.artifacts, "dev64_gold.json")))
    forbid = {d.get("gold") for d in gold.get("documents", []) if d.get("gold")}
    eos_id = tok.eos_token_id
    i = tok.convert_tokens_to_ids("<|im_end|>")
    if i is not None and i >= 0:
        eos_id = i
    # load_role_windows returns (title, token window). The example builder
    # wants the document text, which lives in the split cache the loader
    # filled, exactly as the trainer reassembles it.
    train_titles = {t for t, _ in train}
    bodies = {}
    for sp in split_cache:
        for t, b in split_cache[sp].items():
            if t in train_titles:
                bodies[t] = b
    ordered = [(t, bodies[t]) for t, _ in train if t in bodies]
    ex = build_objective_examples(
        ordered, tok, sched["ctx"], seed=sched["seed"], forbid_codes=forbid
    )
    ok(
        "objective_examples",
        len(ex) == exp["objective_examples"],
        f"{len(ex)} vs {exp['objective_examples']}",
    )
    ok(
        "example_documents",
        len({e.doc_id for e in ex}) == exp["documents_with_examples"],
        f"{len({e.doc_id for e in ex})} vs {exp['documents_with_examples']}",
    )
    bad = verify_examples(tok, ex, eos_id, gold_answers=forbid, supervise_eos=True)
    ok("examples_valid", not bad, f"{len(bad)} problems" if bad else "")

    # 6. Both arms build, and differ only in what contributes.
    try:
        _, labels, terminal = build_labels(tok, ex[0], eos_id)
        von, con = masks_for_arm(terminal, "on")
        voff, coff = masks_for_arm(terminal, "off")
        ok(
            "arms_share_denominator",
            float(von.sum()) == float(voff.sum()),
            f"{float(von.sum())}",
        )
        ok(
            "arms_differ_in_numerator",
            float(con.sum()) - float(coff.sum()) == 1.0,
            f"{float(con.sum())} vs {float(coff.sum())}",
        )
    except Exception as e:
        ok("arm_masks", False, f"{type(e).__name__}: {e}")

    out = {
        "contract": "eos_preflight_v1",
        "passed": not problems,
        "checks": checks,
        "problems": problems,
    }
    json.dump(out, open(args.out, "w"), indent=2, sort_keys=True, default=str)
    if problems:
        print(f"BLOCKED: {problems[:3]}", file=sys.stderr)
        return 1
    print("preflight passed: the CPU side of training is exercised end to end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

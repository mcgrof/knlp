#!/usr/bin/env python3
"""Document advantage of a reader on the synthetic record task: forced-
choice accuracy with the record in context (ceiling) and without it
(floor), averaged over several records, for a grid of record sizes.
Picks the smallest reader and the record size for a ladder rung.

usage: ceiling_grid.py --models m1,m2 --entities 4,8 --attrs 3,9 --docs 10
"""

import argparse, json, random, sys, os, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smallcart as sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--entities", default="4,8")
    ap.add_argument("--attrs", default="3,9")
    ap.add_argument("--docs", type=int, default=10)
    ap.add_argument("--fmt", default="kv")
    ap.add_argument("--holdout", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sc.FMT["name"] = args.fmt
    rows = []
    for m in args.models.split(","):
        tok, model = sc.load_model(m, "cpu")
        bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        for e in [int(x) for x in args.entities.split(",")]:
            for na in [int(x) for x in args.attrs.split(",")]:
                sc.ACTIVE["attrs"] = list(sc.ATTRS)[:na]
                t0 = time.time()
                fl, ce, n, ntok = 0, 0, 0, 0
                for d in range(args.docs):
                    rng = random.Random(100 + d)
                    ents = sc.make_record(rng, e)
                    doc = sc.render_record(ents)
                    _, hold = sc.task_split(ents, rng, args.holdout)
                    f = sc.forced_choice(
                        model,
                        tok,
                        None,
                        ents,
                        hold,
                        "cpu",
                        rng=random.Random(1),
                        bos_id=bos,
                    )
                    c = sc.forced_choice(
                        model,
                        tok,
                        None,
                        ents,
                        hold,
                        "cpu",
                        context_text=doc,
                        rng=random.Random(1),
                        bos_id=bos,
                    )
                    fl += f["acc"] * f["n"]
                    ce += c["acc"] * c["n"]
                    n += f["n"]
                    ntok += len(tok(doc).input_ids)
                r = {
                    "model": m,
                    "entities": e,
                    "attrs": na,
                    "n": n,
                    "floor": fl / n,
                    "ceiling": ce / n,
                    "gap": (ce - fl) / n,
                    "doc_tokens": ntok / args.docs,
                    "seconds": time.time() - t0,
                }
                rows.append(r)
                print(
                    f"GRID,{m},{e},{na},{n},{r['floor']:.3f},{r['ceiling']:.3f},{r['gap']:.3f},{r['doc_tokens']:.0f},{r['seconds']:.0f}s",
                    flush=True,
                )
        del model
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)
    print("GRID_DONE", args.out)


if __name__ == "__main__":
    main()

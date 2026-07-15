"""Summarize the bidirectional per-layer K-format intervention matrix.

Reads intervention_results.json + layer_features.json (from per_layer_intervention.py) and answers:
  T0  does useful layer selectivity exist? (is the demote-damage concentrated on a few layers?)
  T1/T2  does the causal sensitivity ranking match the STATIC K-bias ranking, or depth?
  bytes  if only the sensitive layers need K16/INT8, what is r = 0.5 + 0.25*f vs uniform asym 0.75?

Spearman rank-agreement is computed without scipy (rank + Pearson-on-ranks)."""

import argparse
import json
import os


def spearman(a, b):
    n = len(a)
    if n < 3:
        return float("nan")

    def rank(xs):
        order = sorted(range(n), key=lambda i: xs[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = sum((ra[i] - ma) ** 2 for i in range(n)) ** 0.5
    vb = sum((rb[i] - mb) ** 2 for i in range(n)) ** 0.5
    return cov / (va * vb) if va > 0 and vb > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    d = json.load(open(os.path.join(a.dir, "intervention_results.json")))
    feats = {
        x["layer"]: x
        for x in json.load(open(os.path.join(a.dir, "layer_features.json")))
    }
    m = d["meta"]
    nL = m["nL"]
    cfgs = d["configs"]
    anch = {x["name"]: x for x in cfgs if x["direction"] == "anchor"}

    def by(name):
        return sorted([x for x in cfgs if x["name"] == name], key=lambda x: x["layer"])

    print(
        f"model={m['model']}  nL={nL}  ctx={m['seq_len']}  prompts={m['num_prompts']}"
    )
    print(f"ref_ppl={m['ref_ppl']:.3f}")
    for k in ("asym_k16v8", "fail_fp8v8", "int8_all_v8"):
        x = anch[k]
        print(
            f"  ANCHOR {k:14s} dNLL={x['d_nll']:+.4f} ppl={x['ppl']:.2f} dis={x['disagree']:.4f}"
        )
    fail = anch["fail_fp8v8"]["d_nll"]

    demote = by("demote_fp8")
    dvals = [x["d_nll"] for x in demote]
    tot = sum(dvals) or 1e-9
    top = sorted(demote, key=lambda x: -x["d_nll"])
    print(
        "\n-- T0: demote-damage concentration (base=all-K16/V8, drop 1 layer K->FP8) --"
    )
    print(
        f"  top layer L{top[0]['layer']} dNLL={top[0]['d_nll']:+.4f} = {top[0]['d_nll']/tot:.1%} of total"
    )
    print(
        f"  top-2 share={sum(x['d_nll'] for x in top[:2])/tot:.1%}  top-3={sum(x['d_nll'] for x in top[:3])/tot:.1%}"
    )
    nsig = sum(1 for x in dvals if x > 0.05)
    print(f"  layers with dNLL>0.05: {nsig}/{nL}  -> selective set candidate")

    # correlation of causal sensitivity vs static features
    layers = [x["layer"] for x in demote]
    bias_max = [feats[l]["max_abs"] for l in layers]
    depth = [feats[l]["depth_frac"] for l in layers]
    print("\n-- T1/T2: does static geometry predict causal damage? --")
    print(f"  Spearman(demote dNLL, K-bias max_abs) = {spearman(dvals, bias_max):+.3f}")
    print(f"  Spearman(demote dNLL, depth_frac)     = {spearman(dvals, depth):+.3f}")
    fb = sorted(layers, key=lambda l: -feats[l]["max_abs"])[0]
    print(
        f"  highest-K-bias layer = L{fb} (max_abs={feats[fb]['max_abs']:.1f}); "
        f"its demote dNLL={next(x['d_nll'] for x in demote if x['layer']==fb):+.4f}"
    )
    print(
        f"  => naive 'protect highest-bias layer' {'MISSES' if fb != top[0]['layer'] else 'hits'} the culprit L{top[0]['layer']}"
    )

    # repair direction
    rb = by("repair_bf16")
    ri = by("repair_int8")
    if rb:
        topr = sorted(
            rb, key=lambda x: x["d_nll"]
        )  # smallest residual dNLL = best rescue
        print(
            "\n-- repair (base=all-FP8/V8, restore 1 layer): best single-layer rescue --"
        )
        for x in topr[:3]:
            resc = fail - x["d_nll"]
            print(
                f"  restore L{x['layer']:2d} -> BF16: dNLL={x['d_nll']:+.4f} (rescues {resc/fail:.1%} of failure)"
            )
        if ri:
            x0 = min(ri, key=lambda x: x["d_nll"])
            print(
                f"  best INT8 restore: L{x0['layer']} dNLL={x0['d_nll']:+.4f} (rescues {(fail-x0['d_nll'])/fail:.1%})"
            )

    # byte story
    print("\n-- bytes: selective protection vs uniform asym --")
    prot = [x["layer"] for x in demote if x["d_nll"] > 0.05]
    f = len(prot) / nL
    print(
        f"  protect {len(prot)} of {nL} layers at K16 (f={f:.3f}): r = 0.5+0.25f = {0.5+0.25*f:.4f}x"
    )
    print(f"  vs uniform asym K16/V8 = 0.75x ; vs symmetric FP8 = 0.5x")
    print(f"  protected set = {prot}")
    intr = anch["int8_all_v8"]["d_nll"]
    print(
        f"  (INT8-K-everywhere already dNLL={intr:+.4f} at 0.5x -- if admissible, that dominates selective)"
    )


if __name__ == "__main__":
    main()

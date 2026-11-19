#!/usr/bin/env python3
"""Serve a mixed-domain request stream with prompt-selected pools (GPU).

A server that keeps one expert pool in GPU memory must pick each
request's pool from its prompt and switch pools when the pick changes.
This script runs that whole path on a stream of multiple-choice
requests from several domains and measures what it costs against a
reference with every pool resident.

Requests. Each request is one test item: its prompt (demonstrations,
question and "Answer:") is prefilled and --decode tokens are generated
greedily, batch one. The selector picks the request's pool from the
question text alone: the cosine-nearest class centroid of the mean input
embedding, with centroids from the first --n-train training questions
of each pooled domain in the features file (selector_features.py). The
live embedding is computed as in selector_features.py, and every live
pick must equal the pick from the features file.

Streams. For each mean run length R in --run-lengths, a seeded stream
orders all test items. Requests come in runs of one domain; a run's
length is geometric with mean R and the next run's domain is drawn,
weighted by the items left, from the other domains. R = 1 instead draws
every request's domain independently, weighted by the items left. Items
of a domain keep their test order.

Passes, for each stream:

    resident    all pools stay in GPU memory; a request runs on its
                pool with no I/O (the reference)
    switch-QJ   one pool is resident. Before each request the scheduler
                looks at that request and the next J; if no load is in
                flight and one of them needs another pool, it starts
                loading the first such pool from its packed per-layer
                files through the fastsafetensors io_uring DMA-BUF
                copier, from a background thread, two files in flight.
                A request whose pool is still loading waits, layer by
                layer, for each file. When it finishes, its pool becomes
                the resident one and the previous pool is released.

The stream is saturated: requests run back to back, so a request J
positions ahead is visible J requests early.

Checks, per request of every switch pass: the generated tokens and the
first-token logits equal the resident pass bitwise. Per load: every
layer's loaded bytes equal the resident copy and no layer fell back to
another reader. Device bytes read are recorded per pass.

Measures: per request, the time to first token (TTFT) and total time,
paired with the resident pass; per pass, the wall time of the stream
(excluding the byte checks), the number of pool switches and, for
switched requests, the lead time and the TTFT excess. Served quality is
read from the pool-by-domain matrix (pool_matrix.py) for the pool each
request ran on; it is the same in every pass because the outputs are.

Policies (--policies) set which pool each request runs on, and every
policy gets its own resident and switch passes:

    selector    the selector's pick
    debounce    the resident pool, unless this request and the next one
                both pick the same other pool; then that pool. This
                needs one more request of visibility than the lookahead.

The oracle sequence (each request's own pool; requests from a domain
without a pool go to --unpooled-pool) is reported offline only: its
switch count and served quality.

Usage:
    mixed_stream.py MODEL_DIR FEATURES.pt POOL_MATRIX.json OUT_DIR \\
        --configs TUNED.json \\
        --pool arc POOL.json PACKED_DIR --pool health ... \\
        --domain arc TEST.json --domain health TEST.json ... \\
        [--run-lengths 1 4 16] [--lookahead 0 2] [--decode 16]
"""

import argparse
import gc
import json
import os
import random
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    apply_pool,
    load_model,
    mem_snapshot,
    runtime_identity,
    stacked_name,
    write_json,
)
from domain_selector import question  # noqa: E402
from layer_offload import block_device_of, evict, sectors_read  # noqa: E402
from pool_switch import (  # noqa: E402
    Pool,
    PoolLoader,
    check_loaded,
    run_request,
    set_layer,
    set_pool,
    wait_hooks,
)

KEY = "correct_per_char"


def make_stream(domains, n_items, mean_run, seed):
    """Order (domain, index) pairs into domain runs of mean length mean_run."""
    rng = random.Random(seed)
    left = {d: list(range(n_items[d])) for d in domains}
    out, prev = [], None
    while any(left.values()):
        cand = [d for d in domains if left[d] and (mean_run == 1 or d != prev)]
        if not cand:  # only the previous domain has items left
            cand = [prev]
        d = rng.choices(cand, weights=[len(left[c]) for c in cand])[0]
        run = 1
        if mean_run > 1:
            while rng.random() > 1.0 / mean_run:
                run += 1
        for _ in range(min(run, len(left[d]))):
            out.append((d, left[d].pop(0)))
        prev = d
    return out


def debounce(picks, start):
    """Keep the resident pool unless two consecutive requests pick another."""
    cur, out = start, []
    for i, p in enumerate(picks):
        if p != cur and i + 1 < len(picks) and picks[i + 1] == p:
            cur = p
        out.append(cur)
    return out


def switches(seq):
    return sum(1 for a, b in zip(seq, seq[1:]) if a != b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("features")
    ap.add_argument("matrix")
    ap.add_argument("out_dir")
    ap.add_argument("--configs", required=True)
    ap.add_argument("--pool", nargs=3, action="append", required=True)
    ap.add_argument("--domain", nargs=2, action="append", required=True)
    ap.add_argument("--n-train", type=int, default=128)
    ap.add_argument("--run-lengths", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--lookahead", type=int, nargs="+", default=[0, 2])
    ap.add_argument("--decode", type=int, default=16)
    ap.add_argument("--inflight", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--unpooled-pool", default="history")
    ap.add_argument("--policies", nargs="+", default=["selector", "debounce"])
    ap.add_argument("--limit", type=int, help="first N requests per stream (smoke)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if os.environ.get("FASTSAFETENSORS_DMABUF_STRICT") != "1":
        sys.exit("set FASTSAFETENSORS_DMABUF_STRICT=1")

    from safetensors.torch import load_file

    from fused_moe import ConfigTable, install_fused

    out_path = os.path.join(args.out_dir, "mixed-stream.json")
    pools = {name: (pj, pdir) for name, pj, pdir in args.pool}
    classes = list(pools)
    items = {}
    for name, path in args.domain:
        with open(path) as f:
            items[name] = json.load(f)["items"]
    domains = list(items)
    F = torch.load(args.features, weights_only=False)
    with open(args.matrix) as f:
        M = json.load(f)["models"]
    ok = {
        m: {d: {r["id"]: r[KEY] for r in M[m][d]["rows"]} for d in domains} for m in M
    }

    # Selector: cosine-nearest centroid of the mean input embedding.
    C = []
    for c in classes:
        X = F["domains"][c]["train"]["emb"][: args.n_train].double()
        assert len(X) == args.n_train, (c, len(X))
        X = X / X.norm(dim=1, keepdim=True)
        m = X.mean(0)
        C.append(m / m.norm())
    C = torch.stack(C)

    def pick_of(emb):
        e = emb.double()
        return classes[int((C @ (e / e.norm())).argmax())]

    offline = {}
    for d in domains:
        fd = F["domains"][d]["test"]
        assert fd["ids"] == [it["id"] for it in items[d]], d
        offline[d] = [pick_of(e) for e in fd["emb"]]

    res = {
        "runtime": runtime_identity(),
        "args": vars(args),
        "classes": classes,
        "env": {
            k: v
            for k, v in os.environ.items()
            if k.startswith("FASTSAFETENSORS_") or k.startswith("TORCH_ROCM")
        },
    }

    model, tok = load_model(args.model_dir)
    nl = len(model.model.layers)
    full_gate = [x.mlp.gate.weight.data.clone() for x in model.model.layers]
    emb_w = model.model.embed_tokens.weight
    with open(pools[classes[0]][0]) as f:
        apply_pool(model, json.load(f))
    res["fused_configs"] = install_fused(model, ConfigTable(args.configs)).source
    gc.collect()
    torch.cuda.empty_cache()
    names = [(stacked_name(li, "w13"), stacked_name(li, "w2")) for li in range(nl)]
    files, refs = {}, {}
    for c in classes:
        with open(pools[c][0]) as f:
            pj = json.load(f)
        keep = [
            pj["layers"][str(li)]["routed"] + pj["layers"][str(li)]["shared"]
            for li in range(nl)
        ]
        files[c] = [
            os.path.join(pools[c][1], f"layer{li}.safetensors") for li in range(nl)
        ]
        w13, w2 = [], []
        for li, p in enumerate(files[c]):
            t = load_file(p, device="cuda")
            w13.append(t[names[li][0]])
            w2.append(t[names[li][1]])
        gates = [full_gate[li][keep[li], :].contiguous() for li in range(nl)]
        refs[c] = Pool(gates, w13, w2)
    # The applied first pool must equal its packed copy.
    for li, layer in enumerate(model.model.layers):
        r = refs[classes[0]]
        assert torch.equal(layer.mlp.w13.data, r.w13[li]), li
        assert torch.equal(layer.mlp.w2.data, r.w2[li]), li
        assert torch.equal(layer.mlp.gate.weight.data, r.gate[li]), li
    set_pool(model, refs[classes[0]])
    del full_gate
    gc.collect()
    torch.cuda.empty_cache()
    res["memory_after_setup"] = mem_snapshot()
    dev = block_device_of(files[classes[0]][0])
    eos = tok.eos_token_id

    # Requests and live picks.
    reqs = {}
    pick_mismatch = []
    for d in domains:
        reqs[d] = []
        for k, it in enumerate(items[d]):
            ids = tok(it["context"], add_special_tokens=False)["input_ids"]
            q = tok(question(it["context"]), add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                e = emb_w[torch.tensor(q, device="cuda")].float().mean(0).cpu()
            p = pick_of(e)
            if p != offline[d][k]:
                pick_mismatch.append((d, it["id"]))
            reqs[d].append({"id": it["id"], "ids": ids, "pick": p})
    res["live_pick_mismatches"] = pick_mismatch
    if pick_mismatch:
        write_json(out_path, res)
        sys.exit(f"live picks differ from the features file: {pick_mismatch[:5]}")

    def quality(stream, assign):
        """Served correct answers per domain for one pool assignment."""
        out = {}
        for d in domains:
            idx = [i for i, (dd, _) in enumerate(stream) if dd == d]
            iid = [reqs[d][stream[i][1]]["id"] for i in idx]
            got = [ok[assign[i]][d][x] for i, x in zip(idx, iid)]
            full = [ok["full"][d][x] for x in iid]

            def net(base):
                lost = sum(b and not g for b, g in zip(base, got))
                return lost - sum(g and not b for b, g in zip(base, got))

            o = {"correct": sum(got), "full_correct": sum(full), "n": len(idx)}
            o["net_losses_vs_full"] = net(full)
            if d in classes:
                o["net_losses_vs_oracle"] = net([ok[d][d][x] for x in iid])
            out[d] = o
        return out

    def run_pass(stream, assign, look, ref):
        """look None = resident pass. ref: resident records, for checks."""
        recs, loads = [], []
        evicted_before = sectors_read(dev)[0]
        cur_name = assign[0]
        cur_loader = None
        if look is not None:
            for p in files[cur_name]:
                evict(p)
            cur_loader = PoolLoader(files[cur_name], args.inflight)
            cur_loader.start()
            cur_loader.join()
            for li in range(nl):
                ts = cur_loader.tensors[li]
                set_layer(
                    model,
                    li,
                    refs[cur_name].gate[li],
                    ts[names[li][0]],
                    ts[names[li][1]],
                )
        else:
            set_pool(model, refs[cur_name])
        torch.cuda.synchronize()
        s0 = sectors_read(dev)[0]
        inflight = None  # (name, loader)
        check_s = 0.0
        t_pass0 = time.perf_counter()
        for i, (d, k) in enumerate(stream):
            rq = reqs[d][k]
            want = assign[i]
            rec = {"i": i, "domain": d, "id": rq["id"], "pool": want}
            rec["switched"] = i > 0 and want != assign[i - 1]
            if look is None:
                set_pool(model, refs[want])
                r, lg = run_request(model, rq["ids"], args.decode, eos)
            else:
                if inflight is None:
                    for j in range(i, min(i + look + 1, len(stream))):
                        if assign[j] != cur_name:
                            for p in files[assign[j]]:
                                evict(p)
                            ld = PoolLoader(files[assign[j]], args.inflight)
                            ld.start()
                            inflight = (assign[j], ld)
                            break
                if want != cur_name:
                    nm, ld = inflight
                    assert nm == want, (nm, want, i)
                    r, lg = run_request(
                        model,
                        rq["ids"],
                        args.decode,
                        eos,
                        wait_hooks(model, ld, refs[want].gate, names),
                    )
                    ld.join()
                    rec["lead_s"] = r["t_start"] - ld.t0
                    rec["load_s"] = ld.t_ready[-1] - ld.t0
                    t0 = time.perf_counter()
                    bad, fb = check_loaded(ld, refs[want], names)
                    torch.cuda.synchronize()
                    check_s += time.perf_counter() - t0
                    loads.append(
                        {"i": i, "pool": want, "bad_layers": bad, "fallback": fb}
                    )
                    if cur_loader is not None:
                        cur_loader.close()
                    cur_loader, cur_name, inflight = ld, want, None
                    gc.collect()
                else:
                    r, lg = run_request(model, rq["ids"], args.decode, eos)
            rec.update(
                {
                    "ttft_s": r["ttft_s"],
                    "total_s": r["total_s"],
                    "tokens": r["tokens"],
                }
            )
            if ref is None:
                rec["logits"] = lg.cpu()
            else:
                rec["tokens_match"] = r["tokens"] == ref[i]["tokens"]
                rec["logits_bitwise"] = bool(torch.equal(lg.cpu(), ref[i]["logits"]))
            recs.append(rec)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t_pass0 - check_s
        s1 = sectors_read(dev)[0]
        if inflight is not None:  # a load started for a request past the end
            inflight[1].join()
            inflight[1].close()
        if cur_loader is not None:
            set_pool(model, refs[cur_name])
            cur_loader.close()
        gc.collect()
        torch.cuda.empty_cache()
        return recs, {
            "wall_s": wall,
            "check_s": check_s,
            "device_bytes_read": (s1 - s0) * 512,
            "initial_load_bytes": (s0 - evicted_before) * 512,
            "loads": loads,
        }

    # Warm-up: untimed, a short stream through every pass type.
    n_items = {d: len(reqs[d]) for d in domains}
    warm = make_stream(domains, n_items, 1, args.seed - 1)[:12]
    wa = [reqs[d][k]["pick"] for d, k in warm]
    wref, _ = run_pass(warm, wa, None, None)
    run_pass(warm, wa, max(args.lookahead), wref)

    res["streams"] = {}
    for R in args.run_lengths:
        stream = make_stream(domains, n_items, R, args.seed + R)
        if args.limit:
            stream = stream[: args.limit]
        picks = [reqs[d][k]["pick"] for d, k in stream]
        oracle = [d if d in classes else args.unpooled_pool for d, _ in stream]
        deb = debounce(picks, picks[0])
        st = {
            "n": len(stream),
            "domain_switches": switches([d for d, _ in stream]),
            "order": [[d, reqs[d][k]["id"]] for d, k in stream],
            "offline": {
                name: {"pool_switches": switches(seq), "quality": quality(stream, seq)}
                for name, seq in (
                    ("selector", picks),
                    ("oracle", oracle),
                    ("debounce", deb),
                )
            },
            "passes": {},
        }
        res["streams"][str(R)] = st
        assigns = {"selector": picks, "debounce": deb}
        med = statistics.median
        for pol in args.policies:
            seq = assigns[pol]
            key = lambda q: f"{pol}/switch-Q{q}"  # noqa: E731
            ref, info = run_pass(stream, seq, None, None)
            st["passes"][f"{pol}/resident"] = {"info": info, "rows": ref}
            order = list(args.lookahead)
            if args.run_lengths.index(R) % 2:
                order = order[::-1]
            for q in order:
                rows, info = run_pass(stream, seq, q, ref)
                st["passes"][key(q)] = {"info": info, "rows": rows}
                print(
                    f"R={R} {pol} Q={q}: wall {info['wall_s']:.1f} s vs resident "
                    f"{st['passes'][pol + '/resident']['info']['wall_s']:.1f} s, "
                    f"{len(info['loads'])} loads",
                    flush=True,
                )
            summ = {
                "pool_switches": switches(seq),
                "domain_switches": st["domain_switches"],
                "resident_wall_s": st["passes"][f"{pol}/resident"]["info"]["wall_s"],
                "resident_ttft_ms_median": med(r["ttft_s"] for r in ref) * 1e3,
                "quality": st["offline"][pol]["quality"],
            }
            for q in args.lookahead:
                p = st["passes"][key(q)]
                rows = p["rows"]
                ex = [(a["ttft_s"] - b["ttft_s"]) * 1e3 for a, b in zip(rows, ref)]
                sw = [e for e, r in zip(ex, rows) if r["switched"]]
                limit = max(10.0, 0.05 * summ["resident_ttft_ms_median"])
                leads = [r["lead_s"] * 1e3 for r in rows if "lead_s" in r]
                loads = [r["load_s"] * 1e3 for r in rows if "load_s" in r]
                s_ = {
                    "wall_s": p["info"]["wall_s"],
                    "throughput_loss": p["info"]["wall_s"] / summ["resident_wall_s"]
                    - 1,
                    "ttft_excess_ms_median": med(ex),
                    "ttft_excess_ms_p90": sorted(ex)[int(0.9 * len(ex))],
                    "switched_n": len(sw),
                    "switched_excess_ms_median": med(sw) if sw else None,
                    "switched_excess_ms_p90": (
                        sorted(sw)[int(0.9 * len(sw))] if sw else None
                    ),
                    "switched_hidden_fraction": (
                        sum(e <= limit for e in sw) / len(sw) if sw else None
                    ),
                    "lead_ms_median": med(leads) if leads else None,
                    "load_ms_median": med(loads) if loads else None,
                    "device_bytes_read": p["info"]["device_bytes_read"],
                    "correct": all(
                        r["tokens_match"] and r["logits_bitwise"] for r in rows
                    )
                    and all(
                        not ld["bad_layers"] and not ld["fallback"]
                        for ld in p["info"]["loads"]
                    ),
                }
                s_["keeps_up"] = (
                    s_["correct"]
                    and s_["throughput_loss"] <= 0.05
                    and (s_["switched_excess_ms_median"] or 0.0) <= limit
                )
                summ[f"switch-Q{q}"] = s_
            st.setdefault("summary", {})[pol] = summ
            print(json.dumps({R: {pol: summ}}, indent=1), flush=True)
            for r in ref:
                r.pop("logits", None)
        write_json(out_path, res)
    res["memory_final"] = mem_snapshot()
    write_json(out_path, res)


if __name__ == "__main__":
    main()

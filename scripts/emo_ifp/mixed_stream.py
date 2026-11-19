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
    MODE-QJ     one pool is resident. Before each request the scheduler
                looks at that request and the next J; if no load is in
                flight and one of them needs another pool, it starts
                loading the first such pool through the fastsafetensors
                io_uring DMA-BUF copier, from a background thread, two
                files in flight. A request whose pool is still loading
                waits, layer by layer, for each file. When it finishes,
                its pool becomes the resident one and the previous pool
                is released.

Load modes (--modes):

    full        read the new pool's packed per-layer files
    delta       read only the routed experts of the new pool that the
                resident pool lacks, from per-pair delta files
                (--delta-root/FROM-to-TO/layerL.safetensors, packed from
                delta_pool.py); as each layer arrives the loader thread
                builds the new pool's stacked weights on the GPU from
                the resident pool's copies and the loaded experts. The
                delta buffers are released once the switch is done.

The first pool of every switch pass is read in full before timing.

The stream is saturated: requests run back to back, so a request J
positions ahead is visible J requests early.

Checks, per request of every switch pass: the generated tokens and the
first-token logits equal the resident pass bitwise. Per load: every
layer's loaded bytes equal the resident copy and no layer fell back to
another reader. Device bytes read are recorded per pass.

Measures: per request, the time to first token (TTFT) and total time,
paired with the resident pass; per pass, the wall time of the stream,
the number of pool switches and, for switched requests, the lead time
and the TTFT excess. Three steps exist only for the test and are timed
but left out of the wall time: the byte check of each load, the page
cache eviction of the files about to be read (it guarantees a cold
read) and a Python garbage-collection pass after each switch. With
--gc none a switch pass makes no collection call, so the released
pool must be freed when its last reference drops; Python's automatic
collections still run, inside the wall time, and are timed. Every
switch pass reads the GPU memory allocated right after each switch
(left out of the wall time), which shows whether the released pool
was freed. The
reference pools wait in pinned host memory during switch passes, so
those passes run with the GPU memory of a one-pool server; the byte
check copies each reference layer to the GPU to compare. The
other steps between requests are timed one by one: starting a load,
joining the loader thread and closing the released pool's loads. Served quality is
read from the pool-by-domain matrix (pool_matrix.py) for the pool each
request ran on; it is the same in every pass because the outputs are.

Policies (--policies) set which pool each request runs on, and every
policy gets its own resident and switch passes:

    selector    the selector's pick
    debounce    the resident pool, unless this request and the next one
                both pick the same other pool; then that pool. This
                needs one more request of visibility than the lookahead.
    oracle      each request's own pool; requests from a domain without
                a pool go to --unpooled-pool. A bound, not a policy a
                server could run.

Full-model fallback. A pool named "full" (full_pool.py, every expert)
is the full model as one more target. With --abstain-domain D the
selector gets one more centroid, from D's training questions, and a
request nearest to it is sent to "full" (abstain_eval.py's known-other
rule). A delta switch into "full" releases each layer's delta buffers
as soon as the layer is built, so the resident pool, the full model
being built and two layers in flight fit in GPU memory. A switch out of
"full" reads nothing: every pool is a subset of the full model, so a
background thread gathers the new pool from the resident weights. The
resident pass cannot hold the four pools and the full model at once;
it runs in two parts, the pools' requests with the pools resident and
the full model's requests with it resident, and its wall time is the
sum.

The offline record of every stream holds all three sequences' switch
counts and served quality.

Usage:
    mixed_stream.py MODEL_DIR FEATURES.pt POOL_MATRIX.json OUT_DIR \\
        --configs TUNED.json \\
        --pool arc POOL.json PACKED_DIR --pool health ... \\
        --domain arc TEST.json --domain health TEST.json ... \\
        [--run-lengths 1 4 16] [--lookahead 0 2] [--decode 16] \\
        [--modes full delta --delta-root DIR] \\
        [--policies selector debounce oracle] \\
        [--pool full FULL.json DIR --abstain-domain politics \\
         --unpooled-pool full]
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
    GatherLoader,
    Pool,
    PoolLoader,
    run_request,
    set_layer,
    set_pool,
    wait_hooks,
)

KEY = "correct_per_char"
FULL = "full"
# PyTorch allocator counters read after each request.
ALLOC_KEYS = ("num_device_alloc", "num_device_free", "num_alloc_retries")


def allocator_counters():
    s = torch.cuda.memory_stats()
    return {k: s.get(k, 0) for k in ALLOC_KEYS}


# Steps that exist only for the test; left out of the wall time.
HARNESS = ("check", "evict", "gc", "mem")


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
    ap.add_argument(
        "--abstain-domain",
        help="unpooled domain whose centroid sends requests to the 'full' pool",
    )
    ap.add_argument(
        "--policies",
        nargs="+",
        default=["selector", "debounce"],
        choices=["selector", "debounce", "oracle"],
    )
    ap.add_argument("--modes", nargs="+", default=["full"], choices=["full", "delta"])
    ap.add_argument("--delta-root")
    ap.add_argument(
        "--gc",
        nargs="+",
        default=["collect"],
        choices=["collect", "none"],
        help="per-switch garbage collection: a full pass, or none",
    )
    ap.add_argument("--limit", type=int, help="first N requests per stream (smoke)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if os.environ.get("FASTSAFETENSORS_DMABUF_STRICT") != "1":
        sys.exit("set FASTSAFETENSORS_DMABUF_STRICT=1")

    from safetensors.torch import load_file

    from fused_moe import ConfigTable, install_fused

    if "delta" in args.modes and not args.delta_root:
        sys.exit("--modes delta needs --delta-root")
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

    # Selector: cosine-nearest centroid of the mean input embedding. The
    # full model, if it is a target, is picked only through the centroid of
    # --abstain-domain.
    if FULL in classes:
        assert args.abstain_domain, "pool 'full' needs --abstain-domain"
    labels = [c for c in classes if c != FULL]
    cdoms = list(labels)
    if args.abstain_domain:
        assert FULL in classes, "--abstain-domain needs a pool named 'full'"
        labels.append(FULL)
        cdoms.append(args.abstain_domain)
    C = []
    for c in cdoms:
        X = F["domains"][c]["train"]["emb"][: args.n_train].double()
        assert len(X) == args.n_train, (c, len(X))
        X = X / X.norm(dim=1, keepdim=True)
        m = X.mean(0)
        C.append(m / m.norm())
    C = torch.stack(C)

    def pick_of(emb):
        e = emb.double()
        return labels[int((C @ (e / e.norm())).argmax())]

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
    with open(pools[classes[0]][0]) as f:
        apply_pool(model, json.load(f))
    res["fused_configs"] = install_fused(model, ConfigTable(args.configs)).source
    gc.collect()
    torch.cuda.empty_cache()
    # The pruned model's weights are scattered over the allocator segments
    # of the full model's load, which then stay reserved (about 27 GB).
    # A round trip through host memory packs them into new segments.
    res["reserved_before_compact"] = torch.cuda.memory_reserved()
    model.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    model.to("cuda")
    torch.cuda.synchronize()
    res["reserved_after_compact"] = torch.cuda.memory_reserved()
    emb_w = model.model.embed_tokens.weight
    names = [(stacked_name(li, "w13"), stacked_name(li, "w2")) for li in range(nl)]
    files, refs, pjs, host = {}, {}, {}, {}
    for c in classes:
        with open(pools[c][0]) as f:
            pj = json.load(f)
        pjs[c] = pj
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
        if c == classes[0]:
            # The applied first pool must equal its packed copy.
            for li, layer in enumerate(model.model.layers):
                assert torch.equal(layer.mlp.w13.data, w13[li]), li
                assert torch.equal(layer.mlp.w2.data, w2[li]), li
                assert torch.equal(layer.mlp.gate.weight.data, gates[li]), li
        # The reference pools wait in host memory (pinned, except the full
        # model's, which is only read by the excluded byte check). They move
        # to the GPU only for the resident passes, so switch passes run with
        # the memory a one-pool server would have. Router rows are small and
        # stay on the GPU.
        pin = (lambda x: x.pin_memory()) if c != FULL else (lambda x: x)
        host[c] = Pool(gates, [pin(x.cpu()) for x in w13], [pin(x.cpu()) for x in w2])
        del w13, w2, t, gates
        torch.cuda.empty_cache()
    del full_gate
    refs.update(host)

    def refs_to_gpu(group=None):
        for c in group or classes:
            h = host[c]
            refs[c] = Pool(
                h.gate,
                [t.to("cuda") for t in h.w13],
                [t.to("cuda") for t in h.w2],
            )
        torch.cuda.synchronize()

    def refs_to_host():
        # The model points at host tensors until a pass sets its pool; any
        # stray use fails on the device check instead of reading freed memory.
        for c in classes:
            refs[c] = host[c]
        set_pool(model, host[classes[0]])
        gc.collect()
        torch.cuda.empty_cache()

    refs_to_host()
    res["memory_after_setup"] = mem_snapshot()
    res["free_after_setup"] = torch.cuda.mem_get_info()[0]
    dev = block_device_of(files[classes[0]][0])
    eos = tok.eos_token_id

    # Delta files and index maps for every ordered pair of pools.
    dfiles, didx = {}, {}
    if "delta" in args.modes:
        for x in classes:
            for y in classes:
                if x == y:
                    continue
                if x != FULL:  # out of the full model nothing is read
                    dfiles[x, y] = [
                        os.path.join(
                            args.delta_root, f"{x}-to-{y}", f"layer{li}.safetensors"
                        )
                        for li in range(nl)
                    ]
                maps = []
                for li in range(nl):
                    rx = pjs[x]["layers"][str(li)]["routed"]
                    ry = pjs[y]["layers"][str(li)]["routed"]
                    pos_x = {e: k for k, e in enumerate(rx)}
                    ov = [k for k, e in enumerate(ry) if e in pos_x]
                    dl = [k for k, e in enumerate(ry) if e not in pos_x]
                    dev_i = lambda v: torch.tensor(v, device="cuda")  # noqa: E731
                    maps.append(
                        (
                            dev_i(ov),
                            dev_i([pos_x[ry[k]] for k in ov]),
                            dev_i(dl),
                            len(dl),
                        )
                    )
                didx[x, y] = maps
        res["delta_file_bytes"] = {
            f"{x}-to-{y}": sum(os.path.getsize(p) for p in v)
            for (x, y), v in dfiles.items()
        }

    def assembler(x, y, src):
        """Loader post-step: pool y's layer from pool x's tensors and a delta."""

        def post(li, ts):
            ov_y, ov_x, dl, n = didx[x, y][li]
            n13, n2 = names[li]
            hy = host[y]
            w13 = torch.empty(hy.w13[li].shape, dtype=hy.w13[li].dtype, device="cuda")
            w2 = torch.empty(hy.w2[li].shape, dtype=hy.w2[li].dtype, device="cuda")
            w13[ov_y] = src[li][n13][ov_x]
            w2[ov_y] = src[li][n2][ov_x]
            if ts is None:  # a gather: the new pool is a subset of the old
                assert n == 0, (x, y, li)
            else:
                d13, d2 = ts[n13], ts[n2]
                assert d13.shape[0] == n and d2.shape[0] == n, (x, y, li)
                w13[dl] = d13
                w2[dl] = d2
            return {n13: w13, n2: w2}

        return post

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

    def run_pass(stream, assign, look, ref, mode="full", gc_mode="collect", only=None):
        """look None = resident pass. ref: resident records, for checks.

        only: for a resident pass, run only the requests assigned to these
        pools (the others are left out of the records).
        """
        recs, loads = [], []
        evicted_before = sectors_read(dev)[0]
        cur_name = assign[0]
        if only is not None:
            cur_name = next(a for a in assign if a in only)
        cur_loader = None
        if look is not None:
            mem_point(f"switch pass start, first target {cur_name}")
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
        steps = dict.fromkeys(
            ("check", "evict", "gc", "mem", "start", "join", "close", "record"),
            0.0,
        )
        clock = time.perf_counter
        free_min = [torch.cuda.mem_get_info()[0]]
        alloc_after, other_after = [], []
        dev_ctr = [allocator_counters()]

        def count_device_calls():
            # Device allocations, frees and allocator retries during the
            # request just served; only nonzero changes are recorded.
            now = allocator_counters()
            for k, v in now.items():
                if v != dev_ctr[0][k]:
                    rec.setdefault("allocator", {})[k] = v - dev_ctr[0][k]
            dev_ctr[0] = now

        # Python's automatic collections, timed per generation.
        auto = {"n": [0, 0, 0], "s": [0.0, 0.0, 0.0], "t": None, "on": False}

        def gc_cb(phase, info):
            if not auto["on"]:
                return
            if phase == "start":
                auto["t"] = clock()
            elif auto["t"] is not None:
                g = info["generation"]
                auto["n"][g] += 1
                auto["s"][g] += clock() - auto["t"]
                auto["t"] = None

        def timed(key, fn, *a):
            # Automatic collections inside a test step are not counted:
            # that step is left out of the wall time already.
            on = auto["on"]
            if key in HARNESS:
                auto["on"] = False
            t = clock()
            try:
                return fn(*a)
            finally:
                dt = clock() - t
                auto["on"] = on
                steps[key] += dt
                rec.setdefault("steps", {})[key] = (
                    rec.get("steps", {}).get(key, 0.0) + dt
                )

        gc.callbacks.append(gc_cb)
        auto["on"] = True
        t_pass0 = clock()
        for i, (d, k) in enumerate(stream):
            if only is not None and assign[i] not in only:
                continue
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
                        nxt = assign[j]
                        if nxt != cur_name:
                            gather = mode == "delta" and cur_name == FULL
                            if gather:
                                fl = []
                                post = assembler(cur_name, nxt, cur_loader.tensors)
                            elif mode == "delta":
                                fl = dfiles[cur_name, nxt]
                                post = assembler(cur_name, nxt, cur_loader.tensors)
                            else:
                                fl, post = files[nxt], None

                            def evict_all(fl=fl):
                                for p in fl:
                                    evict(p)

                            def start(fl=fl, post=post, gather=gather, nxt=nxt):
                                if gather:
                                    ld = GatherLoader(nl, post)
                                else:
                                    # Into the full model, free each delta
                                    # layer once built (GPU memory).
                                    ld = PoolLoader(
                                        fl,
                                        args.inflight,
                                        post,
                                        release_after_post=nxt == FULL,
                                    )
                                ld.start()
                                return ld

                            timed("evict", evict_all)
                            ld = timed("start", start)
                            # Only the loader may hold the assembler (and
                            # through it the resident pool's tensors).
                            post = start = None
                            inflight = (nxt, ld)
                            rec["load_started_for"] = j
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
                    timed("join", ld.join)
                    rec["lead_s"] = r["t_start"] - ld.t0
                    rec["load_s"] = ld.t_ready[-1] - ld.t0

                    def check(ld=ld):
                        # Layer by layer against the host copy of the pool.
                        h, bad = host[want], []
                        for li, (n13, n2) in enumerate(names):
                            ts = ld.tensors[li]
                            if not (
                                torch.equal(ts[n13], h.w13[li].to("cuda"))
                                and torch.equal(ts[n2], h.w2[li].to("cuda"))
                            ):
                                bad.append(li)
                        fb = [
                            li
                            for li, x in enumerate(ld.load_info)
                            if x["fallback"]
                            or not x["copier_class"]
                            .lower()
                            .startswith(("dmab", "gath"))
                        ]
                        torch.cuda.synchronize()
                        return bad, fb

                    bad, fb = timed("check", check)
                    loads.append(
                        {
                            "i": i,
                            "pool": want,
                            "source": cur_name,
                            "mode": mode,
                            "bad_layers": bad,
                            "fallback": fb,
                        }
                    )

                    def close(old=cur_loader, new=ld):
                        if old is not None:
                            old.close()
                        if mode == "delta":
                            # The delta buffers are no longer needed.
                            for x in new.loads:
                                if x is not None:
                                    x.close()
                            new.loads = [None] * len(new.loads)
                        # The assembler holds the old pool's tensors.
                        new.post = None

                    timed("close", close)
                    cur_loader, cur_name, inflight = ld, want, None

                    if gc_mode == "collect":
                        timed("gc", gc.collect)

                    def mem():
                        # Live PyTorch tensors, and device memory held
                        # outside PyTorch's allocator (the copier's buffers
                        # for a whole-pool load).
                        a = torch.cuda.memory_allocated()
                        free, total = torch.cuda.mem_get_info()
                        o = total - free - torch.cuda.memory_reserved()
                        alloc_after.append(a)
                        other_after.append(o)
                        rec["alloc_after_switch"] = a
                        rec["other_after_switch"] = o
                        free_min[0] = min(free_min[0], free)

                    timed("mem", mem)
                else:
                    r, lg = run_request(model, rq["ids"], args.decode, eos)
            timed("mem", count_device_calls)
            rec.update(
                {
                    "ttft_s": r["ttft_s"],
                    "total_s": r["total_s"],
                    "tokens": r["tokens"],
                }
            )
            t = clock()
            if ref is None:
                rec["logits"] = lg.cpu()
            else:
                rec["tokens_match"] = r["tokens"] == ref[i]["tokens"]
                rec["logits_bitwise"] = bool(torch.equal(lg.cpu(), ref[i]["logits"]))
            steps["record"] += clock() - t
            recs.append(rec)
        torch.cuda.synchronize()
        elapsed = clock() - t_pass0
        auto["on"] = False
        gc.callbacks.remove(gc_cb)
        harness = sum(steps[x] for x in HARNESS)
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
            "wall_s": elapsed - harness,
            "wall_with_harness_s": elapsed,
            "check_s": steps["check"],
            "steps_s": steps,
            "between_s": elapsed - sum(r["total_s"] for r in recs),
            "device_bytes_read": (s1 - s0) * 512,
            "initial_load_bytes": (s0 - evicted_before) * 512,
            "loads": loads,
            "mode": mode,
            "lookahead": look,
            "min_free_after_switch": free_min[0],
            "gc_mode": gc_mode,
            "alloc_after_switch": (
                {
                    "min": min(alloc_after),
                    "median": statistics.median(alloc_after),
                    "max": max(alloc_after),
                    "first": alloc_after[0],
                    "last": alloc_after[-1],
                }
                if alloc_after
                else None
            ),
            "other_after_switch": (
                {
                    "min": min(other_after),
                    "median": statistics.median(other_after),
                    "max": max(other_after),
                }
                if other_after
                else None
            ),
            "auto_gc": {"count": auto["n"], "seconds": auto["s"]},
            "memory": mem_snapshot(),
        }

    def mem_point(tag):
        """Record and print GPU memory at a stage (a diagnostic)."""
        free, total = torch.cuda.mem_get_info()
        m = mem_snapshot()
        m["free"], m["outside"] = free, total - free - m["reserved"]
        res.setdefault("memory_points", []).append([tag, m])
        print(
            f"mem {tag}: allocated {m['allocated'] / 1e9:.2f} reserved "
            f"{m['reserved'] / 1e9:.2f} outside {m['outside'] / 1e9:.2f} "
            f"free {free / 1e9:.2f} GB",
            flush=True,
        )

    def run_resident(stream, assign):
        """The resident pass. The four pools and the full model do not fit
        in GPU memory together, so with the full model as a target the pass
        runs in two parts, each with its targets resident, and its records
        are merged in request order; the wall time is the sum."""
        groups = [[c for c in classes if c != FULL]]
        if FULL in set(assign):
            groups.append([FULL])
        recs, infos = [], []
        for g in groups:
            if not any(a in g for a in assign):
                continue
            refs_to_gpu(g)
            mem_point(f"resident {'+'.join(g)} loaded")
            r, info = run_pass(stream, assign, None, None, only=set(g))
            refs_to_host()
            mem_point(f"resident {'+'.join(g)} released")
            recs += r
            infos.append(info)
        recs.sort(key=lambda r: r["i"])
        assert [r["i"] for r in recs] == list(range(len(stream)))
        info = dict(infos[0])
        for k in ("wall_s", "wall_with_harness_s", "check_s", "between_s"):
            info[k] = sum(x[k] for x in infos)
        info["parts"] = infos
        return recs, info

    # Warm-up: untimed, a short stream through every pass type.
    n_items = {d: len(reqs[d]) for d in domains}
    warm = make_stream(domains, n_items, 1, args.seed - 1)[:12]
    wa = [reqs[d][k]["pick"] for d, k in warm]
    mem_point("setup done")
    wref, _ = run_resident(warm, wa)
    for mode in args.modes:
        for g in args.gc:
            run_pass(warm, wa, max(args.lookahead), wref, mode, g)

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
        assigns = {"selector": picks, "debounce": deb, "oracle": oracle}
        med = statistics.median
        for pol in args.policies:
            seq = assigns[pol]
            key = lambda m, q, g: (  # noqa: E731
                f"{pol}/{m}-Q{q}" + ("-nogc" if g == "none" else "")
            )
            ref, info = run_resident(stream, seq)
            st["passes"][f"{pol}/resident"] = {"info": info, "rows": ref}
            combos = [
                (m, q, g) for m in args.modes for q in args.lookahead for g in args.gc
            ]
            # Rotate the pass order across streams and policies.
            rot = (args.run_lengths.index(R) + args.policies.index(pol)) % len(combos)
            for m, q, g in combos[rot:] + combos[:rot]:
                rows, info = run_pass(stream, seq, q, ref, m, g)
                st["passes"][key(m, q, g)] = {"info": info, "rows": rows}
                aa = info["alloc_after_switch"] or {}
                print(
                    f"R={R} {pol} {m} Q={q} gc={g}: wall {info['wall_s']:.1f} s "
                    f"(with test steps {info['wall_with_harness_s']:.1f} s) vs "
                    f"resident {st['passes'][pol + '/resident']['info']['wall_s']:.1f}"
                    f" s, {len(info['loads'])} loads, min free "
                    f"{info['min_free_after_switch'] / 1e9:.1f} GB, allocated "
                    f"after switch {aa.get('min', 0) / 1e9:.2f}-"
                    f"{aa.get('max', 0) / 1e9:.2f} GB, auto gc "
                    f"{sum(info['auto_gc']['seconds']):.2f} s "
                    f"{info['auto_gc']['count']}, steps "
                    + " ".join(f"{a}={b:.1f}" for a, b in info["steps_s"].items()),
                    flush=True,
                )
            summ = {
                "pool_switches": switches(seq),
                "domain_switches": st["domain_switches"],
                "resident_wall_s": st["passes"][f"{pol}/resident"]["info"]["wall_s"],
                "resident_ttft_ms_median": med(r["ttft_s"] for r in ref) * 1e3,
                "quality": st["offline"][pol]["quality"],
            }
            for m, q, g in combos:
                p = st["passes"][key(m, q, g)]
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
                    "wall_with_harness_s": p["info"]["wall_with_harness_s"],
                    "steps_s": p["info"]["steps_s"],
                    "between_s": p["info"]["between_s"],
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
                s_["alloc_after_switch"] = p["info"]["alloc_after_switch"]
                s_["auto_gc"] = p["info"]["auto_gc"]
                summ[key(m, q, g).split("/", 1)[1]] = s_
            st.setdefault("summary", {})[pol] = summ
            print(json.dumps({R: {pol: summ}}, indent=1), flush=True)
            for r in ref:
                r.pop("logits", None)
        write_json(out_path, res)
    res["memory_final"] = mem_snapshot()
    write_json(out_path, res)


if __name__ == "__main__":
    main()

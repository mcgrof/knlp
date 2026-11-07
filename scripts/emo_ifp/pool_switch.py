#!/usr/bin/env python3
"""Switch the restricted EMO model from one expert pool to another (GPU).

A fixed expert pool keeps 32 routed experts per layer, about 6 GiB of
routed weights for all 16 layers; loading it takes most of a second
from one drive, longer than a request's prefill. A serving process
therefore keeps a pool across requests and must switch when the domain
changes. This script measures what that switch costs.

The model runs with pool A (calibrate_pool.py) and the fused MoE kernel
(fused_moe.py). All non-expert weights, the shared experts and the router
rows of both pools stay in GPU memory. Pool B's routed weights live in one
stacked file per layer (pack_pool_layer.py --stacked). A switch loads
those files in order through the fastsafetensors io_uring DMA-BUF copier
from one background thread, two files in flight, and points each layer
at its new weights. The arms, per prompt length and trial:

    control-A   the A requests of the hint arms, with no load running
    resident-B  the B request with pool B already in GPU memory
    sync        the B request loads all layers, then starts
    on-demand   the load starts with the B request; each layer waits
                for its own file
    hint-J      the load starts, J A requests run on pool A, then the B
                request; layers still missing wait as in on-demand
    delta-...   with --delta-dir, the on-demand and hint arms again, but
                the load reads only the routed experts of B that A lacks
                (pack_pool_layer.py on a pool file listing, per layer, B's
                experts absent from A, in B's order); as each layer's file
                arrives, the loader thread builds B's stacked weights in a
                new buffer from A's resident copies of the shared experts
                and the loaded ones, on the GPU

Requests are batch one: a prompt of N tokens (layer_offload.build_requests)
and greedy decode. For each switched B request the script records the
time to first token (TTFT) from the B request's start (for sync and
on-demand, from the load start), the lead time between load start and B
request start, and each layer's ready time. It checks the tokens and the
first-token logits against resident-B, the A requests against control-A,
and every loaded layer against the resident copy of pool B.

Usage:
    pool_switch.py MODEL_DIR POOL_A.json POOL_B.json PACKED_B_DIR \\
        REQUESTS_A.json REQUESTS_B.json OUT_DIR --configs TUNED.json \\
        [--trials 10] [--lengths 128 1024] [--decode 32] [--hints 1 2 4] \\
        [--delta-dir DELTA_DIR]

PACKED_B_DIR and DELTA_DIR hold layerL.safetensors for L = 0..15.
"""

import argparse
import collections
import gc
import json
import os
import statistics
import sys
import threading
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
from layer_offload import Load, block_device_of, build_requests  # noqa: E402
from layer_offload import evict, sectors_read  # noqa: E402

LOAD_ARMS = ("sync", "on-demand", "delta-on-demand")


class Pool:
    """Router rows and stacked routed weights for every layer."""

    def __init__(self, gate, w13, w2):
        self.gate, self.w13, self.w2 = gate, w13, w2


def set_layer(model, li, gate, w13, w2):
    m = model.model.layers[li].mlp
    m.gate.weight.data = gate
    m.w13.data = w13
    m.w2.data = w2


def set_pool(model, pool):
    for li in range(len(model.model.layers)):
        set_layer(model, li, pool.gate[li], pool.w13[li], pool.w2[li])


class PoolLoader:
    """Read one file per layer, in order, from a background thread."""

    def __init__(self, files, inflight=2, post=None):
        self.files = files
        self.inflight = inflight
        # post(i, tensors) -> tensors runs in the loader thread per layer
        self.post = post
        n = len(files)
        self.events = [threading.Event() for _ in range(n)]
        self.tensors = [None] * n
        self.loads = [None] * n
        self.t_ready = [None] * n
        self.err = None

    def start(self):
        self.t0 = time.perf_counter()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            pending = collections.deque()
            nxt = 0

            def submit():
                nonlocal nxt
                ld = Load(self.files[nxt], "dmabuf")
                ld.submit()
                pending.append((nxt, ld))
                nxt += 1

            while nxt < len(self.files) and len(pending) < self.inflight:
                submit()
            while pending:
                i, ld = pending.popleft()
                ts = ld.wait()
                self.tensors[i] = self.post(i, ts) if self.post else ts
                self.loads[i] = ld
                self.t_ready[i] = time.perf_counter()
                self.events[i].set()
                if nxt < len(self.files):
                    submit()
        except Exception as exc:  # surfaced to the waiting request
            self.err = exc
            for e in self.events:
                e.set()

    def wait_layer(self, i):
        self.events[i].wait()
        if self.err is not None:
            raise RuntimeError(f"pool load failed: {self.err!r}")

    def join(self):
        self.thread.join()
        if self.err is not None:
            raise RuntimeError(f"pool load failed: {self.err!r}")

    def close(self):
        for ld in self.loads:
            if ld is not None:
                ld.close()
        self.tensors = None
        self.loads = None


@torch.no_grad()
def run_request(model, ids, decode, eos, prefill_hooks=None):
    """One batch-one request; prefill_hooks() installs hooks, returns them."""
    x = torch.tensor([ids], device="cuda")
    hooks = prefill_hooks() if prefill_hooks else []
    t0 = time.perf_counter()
    try:
        out = model(input_ids=x, use_cache=True)
        logits = out.logits[0, -1].float()
        nxt = int(logits.argmax())
        t_first = time.perf_counter()
    finally:
        for h in hooks:
            h.remove()
    toks = [nxt]
    past = out.past_key_values
    for _ in range(decode - 1):
        if nxt == eos:
            break
        o = model(
            input_ids=torch.tensor([[nxt]], device="cuda"),
            past_key_values=past,
            use_cache=True,
        )
        past = o.past_key_values
        nxt = int(o.logits[0, -1].argmax())
        toks.append(nxt)
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    return {
        "t_start": t0,
        "ttft_s": t_first - t0,
        "total_s": t_end - t0,
        "decode_s_per_token": (t_end - t_first) / max(1, len(toks) - 1),
        "tokens": toks,
    }, logits


def wait_hooks(model, loader, gates, li_names):
    """Pre-hooks that make each MoE layer wait for its own loaded file."""

    def install():
        hooks = []
        for li, layer in enumerate(model.model.layers):

            def pre(mod, inp, li=li):
                loader.wait_layer(li)
                ts = loader.tensors[li]
                n13, n2 = li_names[li]
                set_layer(model, li, gates[li], ts[n13], ts[n2])

            hooks.append(layer.mlp.register_forward_pre_hook(pre))
        return hooks

    return install


def check_loaded(loader, poolB, li_names):
    bad = []
    for li, (n13, n2) in enumerate(li_names):
        ts = loader.tensors[li]
        if not (
            torch.equal(ts[n13], poolB.w13[li]) and torch.equal(ts[n2], poolB.w2[li])
        ):
            bad.append(li)
    fallback = [
        li
        for li, ld in enumerate(loader.loads)
        if ld.fallback or not ld.copier_class.lower().startswith("dmab")
    ]
    return bad, fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("pool_a")
    ap.add_argument("pool_b")
    ap.add_argument("packed_b_dir")
    ap.add_argument("requests_a")
    ap.add_argument("requests_b")
    ap.add_argument("out_dir")
    ap.add_argument("--configs", required=True)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--lengths", type=int, nargs="+", default=[128, 1024])
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--hints", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--inflight", type=int, default=2)
    ap.add_argument("--delta-dir")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if os.environ.get("FASTSAFETENSORS_DMABUF_STRICT") != "1":
        sys.exit("set FASTSAFETENSORS_DMABUF_STRICT=1")

    from safetensors.torch import load_file

    from fused_moe import VLLM_SOURCE, ConfigTable, install_fused

    with open(args.pool_a) as f:
        pool_a = json.load(f)
    with open(args.pool_b) as f:
        pool_b = json.load(f)
    with open(args.requests_a) as f:
        req_a = json.load(f)
    with open(args.requests_b) as f:
        req_b = json.load(f)
    res = {
        "runtime": runtime_identity(),
        "args": vars(args),
        "kernel_source": VLLM_SOURCE,
        "env": {
            k: v
            for k, v in os.environ.items()
            if k.startswith("FASTSAFETENSORS_") or k.startswith("TORCH_ROCM")
        },
    }

    model, tok = load_model(args.model_dir)
    nl = len(model.model.layers)
    full_gate = [layer.mlp.gate.weight.data.clone() for layer in model.model.layers]
    apply_pool(model, pool_a)
    res["fused_configs"] = install_fused(model, ConfigTable(args.configs)).source
    gc.collect()
    torch.cuda.empty_cache()
    layers = model.model.layers
    A = Pool(
        [l.mlp.gate.weight.data for l in layers],
        [l.mlp.w13.data for l in layers],
        [l.mlp.w2.data for l in layers],
    )
    keep_b = [
        pool_b["layers"][str(li)]["routed"] + pool_b["layers"][str(li)]["shared"]
        for li in range(nl)
    ]
    gates_b = [full_gate[li][keep_b[li], :].contiguous() for li in range(nl)]
    del full_gate
    files = [
        os.path.join(args.packed_b_dir, f"layer{li}.safetensors") for li in range(nl)
    ]
    names = [(stacked_name(li, "w13"), stacked_name(li, "w2")) for li in range(nl)]
    w13b, w2b = [], []
    for li, p in enumerate(files):
        t = load_file(p, device="cuda")
        w13b.append(t[names[li][0]])
        w2b.append(t[names[li][1]])
    B = Pool(gates_b, w13b, w2b)
    overlap = []
    for li in range(nl):
        a = set(pool_a["layers"][str(li)]["routed"])
        b = set(pool_b["layers"][str(li)]["routed"])
        overlap.append(len(a & b))
    res["routed_overlap_per_layer"] = overlap
    delta_files, delta_idx = None, None
    if args.delta_dir:
        delta_files = [
            os.path.join(args.delta_dir, f"layer{li}.safetensors") for li in range(nl)
        ]
        delta_idx = []
        for li in range(nl):
            ra = pool_a["layers"][str(li)]["routed"]
            rb = pool_b["layers"][str(li)]["routed"]
            pos_a = {e: k for k, e in enumerate(ra)}
            ov = [k for k, e in enumerate(rb) if e in pos_a]
            dl = [k for k, e in enumerate(rb) if e not in pos_a]
            dev_i = lambda xs: torch.tensor(xs, device="cuda")  # noqa
            delta_idx.append(
                (dev_i(ov), dev_i([pos_a[rb[k]] for k in ov]), dev_i(dl), len(dl))
            )
        res["delta_file_bytes"] = [os.path.getsize(p) for p in delta_files]

    def assemble(li, ts):
        """B's stacked weights for layer li from A's copies and the delta."""
        ov_b, ov_a, dl, n = delta_idx[li]
        d13, d2 = ts[names[li][0]], ts[names[li][1]]
        assert d13.shape[0] == n and d2.shape[0] == n
        w13 = torch.empty_like(B.w13[li])
        w2 = torch.empty_like(B.w2[li])
        w13[ov_b] = A.w13[li][ov_a]
        w2[ov_b] = A.w2[li][ov_a]
        w13[dl] = d13
        w2[dl] = d2
        return {names[li][0]: w13, names[li][1]: w2}

    res["file_bytes"] = [os.path.getsize(p) for p in files]
    res["memory_after_setup"] = mem_snapshot()
    eos = tok.eos_token_id
    dev = block_device_of(files[0])
    print("overlap", overlap, flush=True)

    def do_load_arm(arm, b_ids, a_list):
        """Run one switched arm; return its record and the B logits."""
        delta = arm.startswith("delta-")
        arm_files = delta_files if delta else files
        for p in arm_files:
            evict(p)
        s0, _ = sectors_read(dev)
        set_pool(model, A)
        torch.cuda.synchronize()
        ld = PoolLoader(arm_files, args.inflight, assemble if delta else None)
        rec = {"arm": arm}
        a_recs = []
        ld.start()
        if arm == "sync":
            ld.join()
            for li in range(nl):
                ts = ld.tensors[li]
                set_layer(model, li, B.gate[li], ts[names[li][0]], ts[names[li][1]])
            r, lg = run_request(model, b_ids, args.decode, eos)
        else:
            a_logits = []
            for ids in a_list:
                ra, la = run_request(model, ids, args.decode, eos)
                a_recs.append(ra)
                a_logits.append(la)
            r, lg = run_request(
                model, b_ids, args.decode, eos, wait_hooks(model, ld, B.gate, names)
            )
            rec["a_logits"] = a_logits
        ld.join()
        s1, _ = sectors_read(dev)
        rec["lead_s"] = r["t_start"] - ld.t0
        rec["b_ttft_s"] = r["ttft_s"] + (rec["lead_s"] if arm in LOAD_ARMS else 0.0)
        rec["b"] = r
        rec["a"] = a_recs
        rec["load_s"] = ld.t_ready[-1] - ld.t0
        rec["layer_ready_s"] = [t - ld.t0 for t in ld.t_ready]
        rec["device_bytes_read"] = (s1 - s0) * 512
        bad, fb = check_loaded(ld, B, names)
        rec["byte_mismatch_layers"] = bad
        rec["fallback_layers"] = fb
        set_pool(model, A)
        torch.cuda.synchronize()
        ld.close()
        del ld
        gc.collect()
        torch.cuda.empty_cache()
        return rec, lg

    def run_trial(n, i, a_ids, b_ids, arms):
        out = {}
        for arm in arms:
            if arm == "control-A":
                set_pool(model, A)
                recs, lgs = [], []
                for ids in a_ids:
                    r, lg = run_request(model, ids, args.decode, eos)
                    recs.append(r)
                    lgs.append(lg)
                out[arm] = ({"arm": arm, "a": recs}, lgs)
            elif arm == "resident-B":
                set_pool(model, B)
                r, lg = run_request(model, b_ids, args.decode, eos)
                set_pool(model, A)
                out[arm] = ({"arm": arm, "b": r, "b_ttft_s": r["ttft_s"]}, lg)
            elif arm in LOAD_ARMS:
                out[arm] = do_load_arm(arm, b_ids, [])
            else:
                j = int(arm.rsplit("-", 1)[1])
                out[arm] = do_load_arm(arm, b_ids, a_ids[:j])
        ref_b = out["resident-B"]
        ctrl = out["control-A"]
        rows = []
        for arm, (rec, lg) in out.items():
            rec.update({"length": n, "trial": i, "order": arms.index(arm)})
            if "b" in rec:
                rec["b_tokens_match"] = rec["b"]["tokens"] == ref_b[0]["b"]["tokens"]
                rec["b_logits_bitwise"] = bool(torch.equal(lg, ref_b[1]))
            if "hint-" in arm:
                j = len(rec["a"])
                rec["a_tokens_match"] = all(
                    x["tokens"] == y["tokens"]
                    for x, y in zip(rec["a"], ctrl[0]["a"][:j])
                )
                rec["a_logits_bitwise"] = all(
                    bool(torch.equal(x, y))
                    for x, y in zip(rec.pop("a_logits"), ctrl[1][:j])
                )
            rec.pop("a_logits", None)
            rows.append(rec)
        return rows

    hint_arms = [f"hint-{j}" for j in args.hints]
    if args.delta_dir:
        hint_arms += ["delta-on-demand"] + [f"delta-hint-{j}" for j in args.hints]
    arms_all = ["control-A", "resident-B", "sync", "on-demand"] + hint_arms
    jmax = max(args.hints)

    # Warm-up: one untimed trial at the shortest length.
    n0 = min(args.lengths)
    wa = build_requests(tok, req_a, n0, jmax)
    wb = build_requests(tok, req_b, n0, 1)[0]
    run_trial(n0, -1, wa, wb, arms_all)

    trials = []
    for n in args.lengths:
        a_all = build_requests(tok, req_a, n, jmax * args.trials)
        b_all = build_requests(tok, req_b, n, args.trials)
        for i in range(args.trials):
            k = i % len(arms_all)
            arms = arms_all[k:] + arms_all[:k]
            rows = run_trial(n, i, a_all[i * jmax : (i + 1) * jmax], b_all[i], arms)
            trials += rows
            show = {
                r["arm"]: round(r.get("b_ttft_s", 0) * 1e3, 1)
                for r in rows
                if "b_ttft_s" in r
            }
            print(n, i, show, flush=True)
            write_json(
                os.path.join(args.out_dir, "pool-switch.json"),
                {**res, "trials": trials},
            )
    res["trials"] = trials

    # Summaries and the registered decision rule.
    med = statistics.median
    summ = {}
    for n in args.lengths:
        rows = {
            a: [t for t in trials if t["length"] == n and t["arm"] == a]
            for a in arms_all
        }
        for a in rows:
            rows[a].sort(key=lambda t: t["trial"])
        res_b = [t["b_ttft_s"] for t in rows["resident-B"]]
        s = {"resident_b_ttft_ms": med(res_b) * 1e3}
        for a in ["sync", "on-demand"] + hint_arms:
            ex = [(t["b_ttft_s"] - r) * 1e3 for t, r in zip(rows[a], res_b)]
            d = {
                "b_excess_ms": med(ex),
                "b_excess_min_ms": min(ex),
                "b_excess_max_ms": max(ex),
                "lead_ms": med(t["lead_s"] for t in rows[a]) * 1e3,
                "load_ms": med(t["load_s"] for t in rows[a]) * 1e3,
                "correct": all(
                    t["b_tokens_match"]
                    and t["b_logits_bitwise"]
                    and not t["byte_mismatch_layers"]
                    and not t["fallback_layers"]
                    and t.get("a_tokens_match", True)
                    and t.get("a_logits_bitwise", True)
                    for t in rows[a]
                ),
            }
            if "hint-" in a:
                j = int(a.rsplit("-", 1)[1])
                infl = []
                for t, c in zip(rows[a], rows["control-A"]):
                    got = sum(x["total_s"] for x in t["a"])
                    ref = sum(x["total_s"] for x in c["a"][:j])
                    infl.append(got / ref - 1)
                d["a_inflation"] = med(infl)
                d["a_ttft_inflation"] = med(
                    sum(x["ttft_s"] for x in t["a"])
                    / sum(x["ttft_s"] for x in c["a"][:j])
                    - 1
                    for t, c in zip(rows[a], rows["control-A"])
                )
                d["a_decode_inflation"] = med(
                    med(x["decode_s_per_token"] for x in t["a"])
                    / med(x["decode_s_per_token"] for x in c["a"][:j])
                    - 1
                    for t, c in zip(rows[a], rows["control-A"])
                )
                limit = max(10.0, 0.05 * s["resident_b_ttft_ms"])
                d["hidden"] = d["b_excess_ms"] <= limit and d["a_inflation"] <= 0.05
            s[a] = d
        s["on_demand_minus_sync_ms"] = (
            s["on-demand"]["b_excess_ms"] - s["sync"]["b_excess_ms"]
        )
        for prefix in ("hint-", "delta-hint-"):
            hidden = [a for a in hint_arms if a.startswith(prefix) and s[a]["hidden"]]
            key = "smallest_hidden_" + prefix.rstrip("-").replace("-", "_")
            s[key] = hidden[0] if hidden else None
        summ[str(n)] = s
    res["summary"] = summ
    res["memory_final"] = mem_snapshot()
    write_json(os.path.join(args.out_dir, "pool-switch.json"), res)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()

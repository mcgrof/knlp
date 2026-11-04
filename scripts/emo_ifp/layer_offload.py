#!/usr/bin/env python3
"""Load one MoE layer's selected experts from NVMe during a request (GPU).

The restricted EMO model (calibrate_pool.py) stays resident except for the
routed experts of one late layer, which start absent from GPU memory and
from the page cache. A request then loads them from a packed file
(pack_pool_layer.py) under one of these policies:

    resident     weights present before the request; lower bound
    sync-nogds   read through host bounce buffers when the layer needs them
    sync-dmabuf  read with io_uring DMA-BUF straight into GPU memory, same point
    early-dmabuf submit the DMA-BUF read when the request starts; wait only
                 when the layer first needs the weights
    early-nogds  the same early submission through the bounce-buffer reader

Loaded tensors are used in place (views into the reader's allocation) and
kept for the rest of the request. Before any timing, a correctness gate
checks loaded bytes against the manifest and the layer's output on a
captured real input against the resident weights. Each timed trial records
the wall-clock time to the first generated token (TTFT), submission, wait
and readiness times, NVMe sectors read, host-mapped GPU memory (GTT) use,
and checks the generated tokens and first-token logits against the
resident run of the same request.

Usage:
    layer_offload.py MODEL_DIR POOL.json PACKED.safetensors BF16_MANIFEST.json \
        SCREEN.json OUT_DIR [--trials 10] [--lengths 128 1024]
"""

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import threading
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa
    EXPERT_MATS,
    apply_pool,
    expert_tensor_name,
    load_model,
    mem_snapshot,
    runtime_identity,
    tensor_sha256,
    write_json,
)

POLICIES = ["resident", "sync-nogds", "sync-dmabuf", "early-dmabuf", "early-nogds"]


# -- host-side probes -------------------------------------------------------


def block_device_of(path):
    """Return the /proc/diskstats name of the partition holding ``path``."""
    st = os.stat(path)
    maj, mn = os.major(st.st_dev), os.minor(st.st_dev)
    with open("/proc/diskstats") as f:
        for line in f:
            p = line.split()
            if int(p[0]) == maj and int(p[1]) == mn:
                return p[2]
    raise RuntimeError(f"no diskstats entry for {path}")


def sectors_read(dev):
    with open("/proc/diskstats") as f:
        for line in f:
            p = line.split()
            if p[2] == dev:
                return int(p[5]), int(p[3])  # sectors read, reads completed
    raise RuntimeError(dev)


def gpu_sysfs():
    import glob

    for d in sorted(glob.glob("/sys/class/drm/card[0-9]")):
        p = os.path.join(d, "device", "mem_info_vram_total")
        if os.path.exists(p) and int(open(p).read()) > (40 << 30):
            return os.path.join(d, "device")
    return None


class GttSampler:
    """Sample host-mapped GPU memory (GTT) use every 2 ms."""

    def __init__(self, dev):
        self.path = os.path.join(dev, "mem_info_gtt_used") if dev else None
        self.max = 0
        self.base = self.read()
        self.stop = threading.Event()
        self.t = None

    def read(self):
        if not self.path:
            return 0
        with open(self.path) as f:
            return int(f.read())

    def __enter__(self):
        self.base = self.read()
        self.max = self.base
        self.stop.clear()

        def run():
            while not self.stop.is_set():
                v = self.read()
                if v > self.max:
                    self.max = v
                time.sleep(0.002)

        self.t = threading.Thread(target=run, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.t.join()

    @property
    def rise(self):
        return self.max - self.base


def evict(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def cached_pages(path):
    try:
        out = subprocess.check_output(
            ["fincore", "-b", "-n", "-o", "RES", path], text=True
        )
        return int(out.split()[0])
    except Exception:
        return None


# -- loading ----------------------------------------------------------------


class Load:
    """One read of the packed file through one reader."""

    def __init__(self, path, reader, names=None):
        from fastsafetensors import SafeTensorsFileLoader, SingleGroup

        self.t_construct0 = time.perf_counter()
        self.loader = SafeTensorsFileLoader(
            SingleGroup(),
            "cuda:0",
            nogds=(reader == "nogds"),
            use_dmabuf=(reader == "dmabuf"),
        )
        self.loader.add_filenames({0: [path]})
        if names is not None:
            self.loader.set_tensor_filter(lambda n: n in names)
        meta, _ = next(iter(self.loader.meta.values()))
        self.meta = meta
        self.copier = self.loader.copier_constructor(
            meta, self.loader.device, self.loader.framework
        )
        if names is not None:
            self.copier.set_byte_ranges(meta.select_byte_ranges(lambda n: n in names))
        self.t_construct1 = time.perf_counter()

    def submit(self):
        self.t_submit0 = time.perf_counter()
        self.gbuf = self.copier.submit_io(True, 16 << 30)
        self.t_submit1 = time.perf_counter()
        self.copier_class = type(self.copier).__name__
        self.fallback = getattr(self.copier, "_fallback", None) is not None
        pieces = getattr(self.copier, "pieces", None)
        self.alloc_span = sum(p.span for p in pieces) if pieces else None
        self.n_pieces = len(pieces) if pieces else None

    def wait(self):
        from fastsafetensors.allocation import SharedDeviceAllocation

        self.t_wait0 = time.perf_counter()
        owner = SharedDeviceAllocation(
            self.gbuf, self.loader.framework, self.loader.device
        )
        try:
            ts = self.copier.wait_io(self.gbuf, owner=owner)
        finally:
            owner.release()
        self.tensors = {k: v.get_raw() for k, v in ts.items()}
        self.t_wait1 = time.perf_counter()
        return self.tensors

    def close(self):
        self.tensors = None
        try:
            self.loader.close()
        except Exception:
            pass


class LayerSlot:
    """The routed experts of one restricted layer, swappable at run time."""

    def __init__(self, model, layer, routed_ids):
        self.moe = model.model.layers[layer].mlp
        self.layer = layer
        self.routed_ids = routed_ids
        self.names = [
            [expert_tensor_name(layer, e, m) for m in EXPERT_MATS] for e in routed_ids
        ]
        # Reference copy of the resident weights (kept for the resident arm
        # and correctness checks; counted in peak memory, see RESULT).
        self.ref = {}
        for i, e in enumerate(routed_ids):
            for m, n in zip(EXPERT_MATS, self.names[i]):
                self.ref[n] = getattr(self.moe.experts[i], m).weight.data
        self.empty = torch.empty(0, device="cuda", dtype=torch.bfloat16)

    def assign(self, tensors):
        for i in range(len(self.routed_ids)):
            for m, n in zip(EXPERT_MATS, self.names[i]):
                getattr(self.moe.experts[i], m).weight.data = tensors[n]

    def clear(self):
        for i in range(len(self.routed_ids)):
            for m in EXPERT_MATS:
                getattr(self.moe.experts[i], m).weight.data = self.empty

    def all_names(self):
        return [n for ns in self.names for n in ns]


# -- requests ---------------------------------------------------------------


def build_requests(tok, screen, n_tokens, count):
    items = screen["items"]
    reqs = []
    for r in range(count):
        ids = []
        j = r
        while len(ids) < n_tokens:
            ids += tok(
                items[j % len(items)]["context"] + "\n\n", add_special_tokens=False
            )["input_ids"]
            j += 7
        reqs.append(ids[-n_tokens:])
    return reqs


@torch.no_grad()
def run_request(model, slot, ids, policy, packed, decode, eos):
    """Run one request; return timings, tokens and first-token logits."""
    rec = {"policy": policy}
    load = None
    reader = policy.split("-")[1] if "-" in policy else None
    early = policy.startswith("early")
    hook_state = {}

    def pre_hook(mod, inp):
        nonlocal load
        if "done" in hook_state:
            return
        hook_state["t_hook"] = time.perf_counter()
        if reader is not None:
            if not early:
                load = Load(packed, reader)
                load.submit()
            tensors = load.wait()
            slot.assign(tensors)
        hook_state["t_ready"] = time.perf_counter()
        hook_state["done"] = True

    if policy == "resident":
        slot.assign(slot.ref)
    else:
        slot.clear()
    torch.cuda.synchronize()
    h = slot.moe.register_forward_pre_hook(pre_hook)
    x = torch.tensor([ids], device="cuda")
    try:
        t0 = time.perf_counter()
        if early:
            load = Load(packed, reader)
            load.submit()
        out = model(input_ids=x, use_cache=True)
        logits = out.logits[0, -1].float()
        nxt = int(logits.argmax())
        torch.cuda.synchronize()
        t_first = time.perf_counter()
    finally:
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
    rec["ttft_s"] = t_first - t0
    rec["t_hook_s"] = hook_state.get("t_hook", t0) - t0
    rec["t_ready_s"] = hook_state.get("t_ready", t0) - t0
    rec["exposed_wait_s"] = hook_state.get("t_ready", 0) - hook_state.get("t_hook", 0)
    if load is not None:
        rec["construct_s"] = load.t_construct1 - load.t_construct0
        rec["submit_call_s"] = load.t_submit1 - load.t_submit0
        rec["submit_end_s"] = load.t_submit1 - t0
        rec["wait_call_s"] = load.t_wait1 - load.t_wait0
        rec["copier"] = load.copier_class
        rec["fallback"] = load.fallback
        rec["alloc_span"] = load.alloc_span
        rec["pieces"] = load.n_pieces
    rec["tokens"] = toks
    return rec, logits, load


def verify_loaded(slot, load, man_hashes=None):
    bad = []
    for n in slot.all_names():
        t = load.tensors[n]
        if not torch.equal(t, slot.ref[n]):
            bad.append(n)
        elif man_hashes is not None and tensor_sha256(t) != man_hashes[n]:
            bad.append(n)
    return bad


@torch.no_grad()
def correctness_gate(model, slot, tok, screen, packed, man_hashes, out):
    """Bytes and FFN output on a real captured input, for each reader."""
    cap = {}

    def grab(mod, inp):
        cap["x"] = inp[0].detach().clone()

    slot.assign(slot.ref)
    h = slot.moe.register_forward_pre_hook(grab)
    ids = tok(screen["items"][0]["context"], add_special_tokens=False)["input_ids"]
    model(input_ids=torch.tensor([ids], device="cuda"))
    h.remove()
    x = cap["x"]
    ref_out, _ = slot.moe(x)
    gate = {"input_tokens": len(ids)}
    for reader in ("nogds", "dmabuf"):
        evict(packed)
        load = Load(packed, reader)
        load.submit()
        load.wait()
        bad = verify_loaded(slot, load, man_hashes)
        slot.assign(load.tensors)
        got, _ = slot.moe(x)
        gate[reader] = {
            "copier": load.copier_class,
            "fallback": load.fallback,
            "byte_mismatches": bad,
            "ffn_output_bitwise_equal": bool(torch.equal(got, ref_out)),
            "ffn_max_abs_diff": float((got.float() - ref_out.float()).abs().max()),
        }
        slot.clear()
        load.close()
        del load
        gc.collect()
    # Short-read check: a truncated copy must fail, never return tensors.
    trunc = os.path.join(out, "truncated.safetensors")
    with open(packed, "rb") as src, open(trunc, "wb") as dst:
        size = os.path.getsize(packed)
        dst.write(src.read(size - (8 << 20)))
    for reader in ("nogds", "dmabuf"):
        try:
            load = Load(trunc, reader)
            load.submit()
            load.wait()
            gate[f"truncated_{reader}"] = "NO ERROR (failure)"
            load.close()
        except Exception as exc:  # expected
            gate[f"truncated_{reader}"] = (
                f"raised {type(exc).__name__}: {str(exc)[:200]}"
            )
    os.unlink(trunc)
    gc.collect()
    torch.cuda.empty_cache()
    gate["pass"] = all(
        gate[r]["copier"].lower().startswith(r[:4])
        and not gate[r]["fallback"]
        and not gate[r]["byte_mismatches"]
        and gate[r]["ffn_output_bitwise_equal"]
        for r in ("nogds", "dmabuf")
    ) and all(gate[f"truncated_{r}"].startswith("raised") for r in ("nogds", "dmabuf"))
    return gate


@torch.no_grad()
def original_shard_probe(slot, model_dir, man, out):
    """Read the same experts from the original shard: bytes and span."""
    shard = man[slot.all_names()[0]]["shard"]
    path = os.path.join(model_dir, shard)
    dev = block_device_of(path)
    evict(path)
    s0, _ = sectors_read(dev)
    load = Load(path, "dmabuf", names=set(slot.all_names()))
    load.submit()
    load.wait()
    s1, _ = sectors_read(dev)
    bad = verify_loaded(slot, load)
    res = {
        "file": shard,
        "copier": load.copier_class,
        "fallback": load.fallback,
        "pieces": load.n_pieces,
        "alloc_span_bytes": load.alloc_span,
        "payload_bytes": sum(man[n]["nbytes"] for n in slot.all_names()),
        "device_bytes_read": (s1 - s0) * 512,
        "byte_mismatches": bad,
    }
    load.close()
    del load
    gc.collect()
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("pool")
    ap.add_argument("packed")
    ap.add_argument("manifest")
    ap.add_argument("screen")
    ap.add_argument("out_dir")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--lengths", type=int, nargs="+", default=[128, 1024])
    ap.add_argument("--decode", type=int, default=8)
    ap.add_argument("--policies", nargs="+", default=POLICIES)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if os.environ.get("FASTSAFETENSORS_DMABUF_STRICT") != "1":
        sys.exit("set FASTSAFETENSORS_DMABUF_STRICT=1")

    import fastsafetensors

    res = {
        "runtime": runtime_identity(),
        "fastsafetensors_path": os.path.dirname(fastsafetensors.__file__),
        "env": {
            k: v for k, v in os.environ.items() if k.startswith("FASTSAFETENSORS_")
        },
        "args": vars(args),
    }
    with open(args.pool) as f:
        pool = json.load(f)
    with open(args.manifest) as f:
        man = json.load(f)["tensors"]
    with open(args.screen) as f:
        screen = json.load(f)
    man_hashes = {n: man[n]["bf16_sha256"] for n in man}

    model, tok = load_model(args.model_dir)
    apply_pool(model, pool)
    gc.collect()
    torch.cuda.empty_cache()
    slot = LayerSlot(model, args.layer, pool["layers"][str(args.layer)]["routed"])
    res["memory_after_pool"] = mem_snapshot()
    eos = tok.eos_token_id
    dev = block_device_of(args.packed)
    res["packed_device"] = dev
    gtt_dev = gpu_sysfs()

    res["correctness_gate"] = correctness_gate(
        model, slot, tok, screen, args.packed, man_hashes, args.out_dir
    )
    print("gate", json.dumps(res["correctness_gate"])[:600], flush=True)
    res["original_shard_probe"] = original_shard_probe(
        slot, args.model_dir, man, args.out_dir
    )
    print("orig", res["original_shard_probe"], flush=True)
    if not res["correctness_gate"]["pass"]:
        write_json(os.path.join(args.out_dir, "offload-summary.json"), res)
        sys.exit("correctness gate failed; no timing")

    # Warm-up: one untimed request per policy (library, kernel and ring init).
    warm = build_requests(tok, screen, 128, 1)[0]
    for p in args.policies:
        evict(args.packed)
        r, _, load = run_request(model, slot, warm, p, args.packed, 2, eos)
        slot.clear()
        if load is not None:
            load.close()
        gc.collect()
        torch.cuda.empty_cache()

    trials = []
    for n in args.lengths:
        reqs = build_requests(tok, screen, n, args.trials)
        for i, ids in enumerate(reqs):
            k = i % len(args.policies)
            order = args.policies[k:] + args.policies[:k]
            ref_logits = None
            ref_tokens = None
            per = {}
            for p in order:
                evict(args.packed)
                pages = cached_pages(args.packed)
                s0, r0 = sectors_read(dev)
                with GttSampler(gtt_dev) as g:
                    rec, logits, load = run_request(
                        model, slot, ids, p, args.packed, args.decode, eos
                    )
                s1, r1 = sectors_read(dev)
                rec.update(
                    {
                        "length": n,
                        "request": i,
                        "order": order.index(p),
                        "cached_pages_before": pages,
                        "device_bytes_read": (s1 - s0) * 512,
                        "device_reads": r1 - r0,
                        "gtt_rise_bytes": g.rise,
                    }
                )
                if load is not None:
                    rec["byte_mismatches"] = len(verify_loaded(slot, load))
                per[p] = (rec, logits)
                slot.clear()
                if load is not None:
                    load.close()
                del load
                gc.collect()
                torch.cuda.empty_cache()
            ref_rec, ref_logits = per["resident"]
            for p, (rec, logits) in per.items():
                rec["tokens_match_resident"] = rec["tokens"] == ref_rec["tokens"]
                rec["logits_bitwise_resident"] = bool(torch.equal(logits, ref_logits))
                trials.append(rec)
            print(
                n,
                i,
                {p: round(per[p][0]["ttft_s"] * 1e3, 1) for p in args.policies},
                flush=True,
            )
    res["trials"] = trials

    # Paired summaries.
    summ = {}
    for n in args.lengths:
        rows = {
            p: [t for t in trials if t["length"] == n and t["policy"] == p]
            for p in args.policies
        }
        for p in args.policies:
            rows[p].sort(key=lambda t: t["request"])
        s = {}
        for p in args.policies:
            tt = [t["ttft_s"] for t in rows[p]]
            s[p] = {
                "ttft_median_ms": statistics.median(tt) * 1e3,
                "ttft_min_ms": min(tt) * 1e3,
                "ttft_max_ms": max(tt) * 1e3,
                "exposed_wait_median_ms": statistics.median(
                    t["exposed_wait_s"] for t in rows[p]
                )
                * 1e3,
                "correct_all": all(
                    t["tokens_match_resident"]
                    and t["logits_bitwise_resident"]
                    and t.get("byte_mismatches", 0) == 0
                    and not t.get("fallback", False)
                    for t in rows[p]
                ),
            }

        def pair(a, b):
            d = [(x["ttft_s"] - y["ttft_s"]) * 1e3 for x, y in zip(rows[a], rows[b])]
            return {
                "median_saving_ms": statistics.median(d),
                "positive_pairs": sum(1 for v in d if v > 0),
                "pairs": len(d),
            }

        cmp = {}
        for a, b in (
            ("sync-nogds", "sync-dmabuf"),
            ("sync-dmabuf", "early-dmabuf"),
            ("sync-nogds", "early-nogds"),
            ("early-nogds", "early-dmabuf"),
            ("early-dmabuf", "resident"),
            ("sync-dmabuf", "resident"),
        ):
            if a in rows and b in rows:
                cmp[f"{a} minus {b}"] = pair(a, b)
        s["paired"] = cmp
        summ[str(n)] = s
    res["summary"] = summ
    res["memory_final"] = mem_snapshot()
    write_json(os.path.join(args.out_dir, "offload-summary.json"), res)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()

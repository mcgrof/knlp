#!/usr/bin/env python3
"""Run the frozen workload through the three request paths, a shard at a time.

For each shard the source model builds and stores its caches, then is
unloaded. The receiver is loaded once with the mappers resident, and for
each context it stores its own cache, answers every fixed question by every
route, and is timed to its first token. When a context's outputs and byte
receipts are safely on disk its cache files are deleted, so the disk never
holds more than one shard.

Quality and timing are separate requests. A quality request generates the
whole answer once. A timing request stops at the first token and is repeated,
with the order of the routes drawn from a fixed seed.

Everything a later scorer needs is kept raw: the generated tokens, the text,
whether the model stopped by itself, and every timing sample.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import resource
import subprocess
import sys
import time

import numpy as np

from . import cache_store, contexts
from .cache_store import storage_facts

STOP = ("<|im_end|>", "<|endoftext|>")
NEW_TOKENS = {"document": 48, "code": 768}
ROUTES = ("recompute", "restore", "translate")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)


def package_digest():
    """One hash over the source that is about to run."""
    h = hashlib.sha256()
    here = os.path.dirname(os.path.abspath(__file__))
    for root in (here, os.path.join(os.path.dirname(here), "published")):
        for fn in sorted(os.listdir(root)):
            if fn.endswith((".py", ".sh")):
                h.update(fn.encode())
                with open(os.path.join(root, fn), "rb") as f:
                    h.update(f.read())
    return h.hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def identity(cfg, records_sha, mapper_shas):
    blob = json.dumps(
        {
            "code": package_digest(),
            "records": records_sha,
            "mappers": mapper_shas,
            "models": [
                cfg["source"],
                cfg["source_revision"],
                cfg["target"],
                cfg["target_revision"],
            ],
            "new_tokens": NEW_TOKENS,
            "rounds": cfg["rounds"],
            "seed": cfg["seed"],
            "read_state": cfg["read_state"],
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def append(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_model(name, revision):
    import torch
    from transformers import AutoModelForCausalLM

    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, torch_dtype=torch.bfloat16, device_map={"": "cuda"}
    ).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, round(time.time() - t0, 2)


def unload(model):
    import torch

    del model
    gc.collect()
    torch.cuda.empty_cache()


def tokens_of(work, rel):
    return np.load(os.path.join(work, rel)).astype(np.int64).tolist()


def cache_path(cfg, role, cid):
    return os.path.join(cfg["cache_dir"], f"{cid}.{role}.kv")


def expect(name, revision, record):
    return {
        "model": name,
        "revision": revision,
        "prefix_tokens_sha256": record["prefix_tokens_sha256"],
        "tokens": record["actual_prefix_tokens"],
    }


def build_source(cfg, records, ledger):
    """Every context of the shard through the source, stored and receipted."""
    import torch

    from ..published.capture import cache_layers

    todo = [
        r
        for r in records
        if not os.path.exists(cache_path(cfg, "source", r["context_id"]))
        and not os.path.exists(
            os.path.join(cfg["out"], "contexts", r["context_id"] + ".json")
        )
    ]
    if not todo:
        return {"loaded": False}
    model, load_s = load_model(cfg["source"], cfg["source_revision"])
    log(f"source resident in {load_s}s")
    for r in todo:
        ids = torch.as_tensor(
            tokens_of(cfg["workload"], r["prefix_token_file"]), device="cuda"
        )[None]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.model(input_ids=ids, use_cache=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        pairs = [(k, v) for k, v in cache_layers(out.past_key_values)]
        rec = cache_store.write(
            cache_path(cfg, "source", r["context_id"]),
            pairs,
            {
                "model": cfg["source"],
                "revision": cfg["source_revision"],
                "prefix_tokens_sha256": r["prefix_tokens_sha256"],
                "role": "source",
            },
        )
        rec.update(
            context_id=r["context_id"],
            seconds_prefill=round(t1 - t0, 4),
            seconds_write=round(time.perf_counter() - t1, 4),
        )
        append(ledger, rec)
        del out, pairs
        log(f"{r['context_id']}: source cache {rec['logical_bytes']} bytes")
    unload(model)
    return {"loaded": True, "seconds_loading": load_s, "contexts": len(todo)}


def decode(tok, ids):
    return tok.decode(ids, skip_special_tokens=False)


def one_request(rx, route, cfg, record, query, prefix, suffix, *, direct, mapper=None):
    if route == "recompute":
        return rx.recompute(prefix, suffix)
    if route == "restore":
        return rx.restore(
            cache_path(cfg, "target", record["context_id"]),
            suffix,
            direct=direct,
            expect=expect(cfg["target"], cfg["target_revision"], record),
        )
    if route == "translate":
        rx.mapper = rx.mappers[mapper]
        return rx.translate(
            cache_path(cfg, "source", record["context_id"]),
            suffix,
            direct=direct,
            expect=expect(cfg["source"], cfg["source_revision"], record),
        )
    raise ValueError(route)


def strip(result):
    return {k: v for k, v in result.items() if k not in ("cache", "logits")}


def serve(cfg, records, ledger, out_dir, ident):
    import torch
    from transformers import AutoTokenizer

    from ..published import fit_pair
    from ..published.capture import geometry
    from .request_path import Receiver

    gen_path = os.path.join(out_dir, "generations.jsonl")
    time_path = os.path.join(out_dir, "timings.jsonl")
    done_dir = os.path.join(out_dir, "contexts")
    os.makedirs(done_dir, exist_ok=True)
    todo = [
        r
        for r in records
        if not os.path.exists(os.path.join(done_dir, r["context_id"] + ".json"))
    ]
    if not todo:
        return {"loaded": False}
    tok = AutoTokenizer.from_pretrained(cfg["target"], revision=cfg["target_revision"])
    stop_ids = {tok.convert_tokens_to_ids(s) for s in STOP}
    model, load_s = load_model(cfg["target"], cfg["target_revision"])
    log(f"receiver resident in {load_s}s")
    geom = geometry(model)
    largest = cache_store.payload_bytes(
        geom["layers"],
        geom["kv_heads"],
        max(r["actual_prefix_tokens"] for r in records),
        geom["head_dim"],
    )
    # the staging buffer holds whichever cache is read, and the source's is
    # the larger one when the source is the larger model
    for r in todo:
        head = cache_store.header(cache_path(cfg, "source", r["context_id"]))
        largest = max(largest, head["payload_bytes"])
    t0 = time.time()
    rx = Receiver(model, largest_payload=largest, source_theta=cfg["source_rope_theta"])
    rx.mappers = {}
    for name, path in cfg["mappers"].items():
        m = fit_pair.load_mapper(path)
        rx.mappers[name] = {**m, "W": m["W"].to(rx.device), "b": m["b"].to(rx.device)}
    warm_s = round(time.time() - t0, 2)
    log(f"mappers and staging buffer resident in {warm_s}s: {sorted(rx.mappers)}")
    heads = {
        "document": tok.encode(contexts.DOCUMENT_HEAD, add_special_tokens=False),
        "code": tok.encode(contexts.CODE_HEAD, add_special_tokens=False),
    }
    rng = random.Random(f"{cfg['seed']}|order")
    primary_direct = cfg["read_state"] == "direct"

    for r in todo:
        cid = r["context_id"]
        torch.cuda.reset_peak_memory_stats()
        prefix = tokens_of(cfg["workload"], r["prefix_token_file"])
        tpath = cache_path(cfg, "target", cid)
        if not os.path.exists(tpath):
            torch.cuda.synchronize()
            a = time.perf_counter()
            pairs = rx.prefix_cache(prefix)
            torch.cuda.synchronize()
            b = time.perf_counter()
            rec = cache_store.write(
                tpath,
                pairs,
                {
                    "model": cfg["target"],
                    "revision": cfg["target_revision"],
                    "prefix_tokens_sha256": r["prefix_tokens_sha256"],
                    "role": "target",
                },
            )
            rec.update(
                context_id=cid,
                seconds_prefill=round(b - a, 4),
                seconds_write=round(time.perf_counter() - b, 4),
            )
            append(ledger, rec)
            del pairs
        # what was written is what is read back, for both files
        checks = {}
        for role, name, rev in (
            ("target", cfg["target"], cfg["target_revision"]),
            ("source", cfg["source"], cfg["source_revision"]),
        ):
            head, io = rx.reader.read(
                cache_path(cfg, role, cid), direct=False, expect=expect(name, rev, r)
            )
            checks[role] = rx.reader.verify(head)
            if not checks[role]:
                raise OSError(
                    f"{cid}: the {role} cache read back differs from what was written"
                )

        arms = [("restore", None), ("recompute", None), ("translate", cfg["lead"])]
        if cid in cfg["diagnostic_contexts"]:
            arms += [("translate", m) for m in rx.mappers if m != cfg["lead"]]
        for q in r["fixed_queries"]:
            suffix = tokens_of(cfg["workload"], q["suffix_token_file"])
            cap = NEW_TOKENS[r["domain"]]
            for route, mapper in arms + [("closed_book", None)]:
                if route == "closed_book":
                    res = rx.closed_book(heads[r["domain"]] + suffix)
                else:
                    res = one_request(
                        rx,
                        route,
                        cfg,
                        r,
                        q,
                        prefix,
                        suffix,
                        direct=primary_direct,
                        mapper=mapper,
                    )
                gen = rx.generate(
                    res["token"], res["cache"], max_new_tokens=cap, stop_ids=stop_ids
                )
                append(
                    gen_path,
                    {
                        "identity": ident,
                        "context_id": cid,
                        "source_group_id": r["source_group_id"],
                        "domain": r["domain"],
                        "length_bucket": r["length_bucket"],
                        "query_id": q["query_id"],
                        "route": route,
                        "mapper": mapper,
                        "first_token": res["token"],
                        "tokens": gen["tokens"],
                        "text": decode(tok, gen["tokens"]),
                        "stopped": gen["stopped"],
                        "new_tokens": gen["new_tokens"],
                        "max_new_tokens": cap,
                        "seconds_generating": gen["seconds_generating"],
                        "request": strip(res),
                    },
                )
                del res, gen
            # timing: first token only, routes in a drawn order, repeated
            for state, rounds in (
                ("primary", cfg["rounds"]),
                ("page_cache", cfg["warm_rounds"]),
            ):
                direct = primary_direct if state == "primary" else False
                for rnd in range(rounds):
                    order = list(ROUTES)
                    rng.shuffle(order)
                    for route in order:
                        if state == "page_cache" and route == "recompute":
                            continue
                        res = one_request(
                            rx,
                            route,
                            cfg,
                            r,
                            q,
                            prefix,
                            suffix,
                            direct=direct,
                            mapper=cfg["lead"],
                        )
                        append(
                            time_path,
                            {
                                "identity": ident,
                                "context_id": cid,
                                "source_group_id": r["source_group_id"],
                                "domain": r["domain"],
                                "length_bucket": r["length_bucket"],
                                "query_id": q["query_id"],
                                "route": route,
                                "mapper": cfg["lead"] if route == "translate" else None,
                                "read_state": (
                                    "none"
                                    if route == "recompute"
                                    else ("direct" if direct else "page_cache")
                                ),
                                "round": rnd,
                                "position_in_round": order.index(route),
                                "suffix_tokens": len(suffix),
                                "first_token": res["token"],
                                **strip(res),
                            },
                        )
                        del res
        receipt = {
            "context_id": cid,
            "identity": ident,
            "round_trip_verified": checks,
            "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "peak_host_rss_gib": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2
            ),
            "cache_dir_free_bytes": storage_facts(cfg["cache_dir"])["free_bytes"],
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        tmp = os.path.join(done_dir, cid + ".json.tmp")
        with open(tmp, "w") as f:
            json.dump(receipt, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, os.path.join(done_dir, cid + ".json"))
        if not cfg["keep_caches"]:
            for role in ("source", "target"):
                os.remove(cache_path(cfg, role, cid))
        log(f"{cid}: done, peak card memory {receipt['peak_gpu_gib']} GiB")
    unload(model)
    return {
        "loaded": True,
        "seconds_loading": load_s,
        "seconds_warming": warm_s,
        "contexts": len(todo),
    }


def shards(records, size):
    for i in range(0, len(records), size):
        yield records[i : i + size]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    import torch

    from ..published.run_pair import usable_processors

    threads = min(usable_processors(), 16)
    torch.set_num_threads(threads)
    out_dir = cfg["out"]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cfg["cache_dir"], exist_ok=True)
    with open(os.path.join(cfg["workload"], "records.jsonl")) as f:
        records = [json.loads(x) for x in f if x.strip()]
    records_sha = file_sha(os.path.join(cfg["workload"], "records.jsonl"))
    if cfg.get("contexts"):
        keep = set(cfg["contexts"])
        records = [r for r in records if r["context_id"] in keep]
        if len(records) != len(keep):
            raise SystemExit("a requested context is not in the workload")
    mapper_shas = {k: file_sha(v) for k, v in cfg["mappers"].items()}
    for name, want in cfg.get("mapper_sha256", {}).items():
        if mapper_shas[name] != want:
            raise SystemExit(
                f"the {name} mapper on disk is not the one that was fitted"
            )
    ident = identity(cfg, records_sha, mapper_shas)
    facts = {
        "identity": ident,
        "code_digest": package_digest(),
        "records_sha256": records_sha,
        "mapper_sha256": mapper_shas,
        "threads": threads,
        "processors_visible": os.cpu_count(),
        "torch": torch.__version__,
        "card": torch.cuda.get_device_name(0),
        "card_memory_gib": round(
            torch.cuda.get_device_properties(0).total_memory / 2**30, 1
        ),
        "storage": storage_facts(cfg["cache_dir"]),
        "config": cfg,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    import transformers

    facts["transformers"] = transformers.__version__
    with open(os.path.join(out_dir, "RUN.json"), "w") as f:
        json.dump(facts, f, indent=2, sort_keys=True)
    ledger = os.path.join(out_dir, "CACHE_BYTES.jsonl")
    loads = []
    for shard in shards(records, cfg["shard_contexts"]):
        t0 = time.time()
        a = build_source(cfg, shard, ledger)
        t1 = time.time()
        b = serve(cfg, shard, ledger, out_dir, ident)
        loads.append(
            {
                "contexts": [r["context_id"] for r in shard],
                "source": a,
                "receiver": b,
                "seconds_preparing_source": round(t1 - t0, 1),
                "seconds_serving": round(time.time() - t1, 1),
            }
        )
        with open(os.path.join(out_dir, "SHARDS.json"), "w") as f:
            json.dump(loads, f, indent=2, sort_keys=True)
    log("SHARING_RUN_COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

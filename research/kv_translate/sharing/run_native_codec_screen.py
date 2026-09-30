#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""Compare native recompute, BF16 restore, and K16/V8 restore."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import resource
import sys
import time

import torch

from . import asym_cache_store, cache_store
from .request_path import Clock, Receiver
from .run_repair_screen import (
    NEW_TOKENS,
    STOP,
    append_jsonl,
    choose_contexts,
    combine_context_outputs,
    decode,
    expected,
    file_sha,
    load_jsonl,
    load_model,
    log,
    strip_result,
    tokens_of,
)

ARMS = ("native_recompute", "native_bf16_restore", "native_k16_v8_restore")


def choose_qualification_contexts(records, lengths):
    """Choose the first frozen document and code group at each length."""
    chosen = []
    for length in lengths:
        chosen.extend(choose_contexts(records, int(length), 1))
    return chosen


def source_digest():
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    paths = [
        os.path.abspath(__file__),
        os.path.join(here, "asym_cache_store.py"),
        os.path.join(here, "cache_store.py"),
        os.path.join(here, "request_path.py"),
        os.path.join(here, "run_repair_screen.py"),
        os.path.join(parent, "published", "capture.py"),
        os.path.join(parent, "published", "torch_ops.py"),
    ]
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(os.path.relpath(path, os.path.dirname(parent)).encode())
        with open(path, "rb") as source:
            digest.update(source.read())
    return digest.hexdigest()


def cache_path(cfg, kind, context_id):
    return os.path.join(cfg["cache_dir"], f"{context_id}.{kind}.kv")


def prepare_caches(rx, cfg, record, prefix, ledger):
    bf16 = cache_path(cfg, "bf16", record["context_id"])
    asym = cache_path(cfg, "k16v8", record["context_id"])
    if os.path.exists(bf16) and os.path.exists(asym):
        return cache_store.header(bf16), asym_cache_store.header(asym)
    torch.cuda.synchronize()
    start = time.perf_counter()
    pairs = rx.prefix_cache(prefix)
    torch.cuda.synchronize()
    prefetched = time.perf_counter()
    meta = {
        "model": cfg["model"],
        "revision": cfg["model_revision"],
        "prefix_tokens_sha256": record["prefix_tokens_sha256"],
        "role": "native",
        "attention_backend": "sdpa",
    }
    bf16_receipt = cache_store.write(bf16, pairs, meta)
    bf16_receipt.update(
        context_id=record["context_id"],
        kind="bf16",
        seconds_prefill=round(prefetched - start, 6),
        seconds_write=round(time.perf_counter() - prefetched, 6),
    )
    append_jsonl(ledger, bf16_receipt)
    asym_receipt = asym_cache_store.write(
        asym,
        pairs,
        meta,
        codec_checkout=cfg["codec_checkout"],
        codec_revision=cfg["codec_revision"],
    )
    asym_receipt.update(context_id=record["context_id"], kind="k16v8")
    append_jsonl(ledger, asym_receipt)
    return cache_store.header(bf16), asym_cache_store.header(asym)


def request(rx, asym_reader, cfg, record, prefix, suffix, arm, *, direct):
    if arm == "native_recompute":
        return rx.recompute(prefix, suffix)
    expect = expected(cfg["model"], cfg["model_revision"], record)
    if arm == "native_bf16_restore":
        return rx.restore(
            cache_path(cfg, "bf16", record["context_id"]),
            suffix,
            direct=direct,
            expect=expect,
        )
    if arm != "native_k16_v8_restore":
        raise ValueError(f"unknown native codec arm {arm!r}")
    clock = Clock(rx.device)
    _, pairs, io = asym_reader.read(
        cache_path(cfg, "k16v8", record["context_id"]),
        direct=direct,
        expect=expect,
    )
    clock.mark("read_unpack_to_card_dequant")
    from ..published.capture import make_cache

    cache = make_cache(pairs)
    clock.mark("install")
    token, cache, logits = rx._finish(clock, suffix, cache)
    return {
        "token": token,
        "cache": cache,
        "logits": logits,
        "io": io,
        **clock.report(),
    }


def run_context(rx, asym_reader, tokenizer, stop_ids, cfg, record, identity, ledger, rng):
    cid = record["context_id"]
    out_dir = os.path.join(cfg["out"], "contexts", cid)
    done = os.path.join(out_dir, "DONE.json")
    if os.path.exists(done):
        return False
    os.makedirs(out_dir, exist_ok=True)
    gen_tmp = os.path.join(out_dir, "generations.jsonl.tmp")
    timing_tmp = os.path.join(out_dir, "timings.jsonl.tmp")
    for path in (gen_tmp, timing_tmp):
        with open(path, "w"):
            pass
    prefix = tokens_of(cfg["workload"], record["prefix_token_file"])
    if len(prefix) != int(record["length_bucket"]):
        raise ValueError(f"{cid}: prefix has {len(prefix)} tokens")
    _, asym_head = prepare_caches(rx, cfg, record, prefix, ledger)
    reduction = 1 - os.stat(cache_path(cfg, "k16v8", cid)).st_size / os.stat(
        cache_path(cfg, "bf16", cid)
    ).st_size
    if reduction < cfg["serialized_reduction_min"]:
        raise RuntimeError(f"serialized byte reduction is only {reduction:.6f}")
    torch.cuda.reset_peak_memory_stats()
    direct = cfg["read_state"] == "direct"
    for query in record["fixed_queries"]:
        suffix = tokens_of(cfg["workload"], query["suffix_token_file"])
        cap = NEW_TOKENS[record["domain"]]
        for arm in ARMS:
            result = request(
                rx, asym_reader, cfg, record, prefix, suffix, arm, direct=direct
            )
            generated = rx.generate(
                result["token"],
                result["cache"],
                max_new_tokens=cap,
                stop_ids=stop_ids,
            )
            append_jsonl(
                gen_tmp,
                {
                    "identity": identity,
                    "context_id": cid,
                    "source_group_id": record["source_group_id"],
                    "domain": record["domain"],
                    "length_bucket": record["length_bucket"],
                    "query_id": query["query_id"],
                    "route": arm,
                    "mapper": None,
                    "first_token": result["token"],
                    "tokens": generated["tokens"],
                    "text": decode(tokenizer, generated["tokens"]),
                    "stopped": generated["stopped"],
                    "new_tokens": generated["new_tokens"],
                    "max_new_tokens": cap,
                    "seconds_generating": generated["seconds_generating"],
                    "request": strip_result(result),
                },
            )
            del result, generated
        for round_id in range(cfg["rounds"]):
            order = list(ARMS)
            rng.shuffle(order)
            for position, arm in enumerate(order):
                result = request(
                    rx, asym_reader, cfg, record, prefix, suffix, arm, direct=direct
                )
                append_jsonl(
                    timing_tmp,
                    {
                        "identity": identity,
                        "context_id": cid,
                        "source_group_id": record["source_group_id"],
                        "domain": record["domain"],
                        "length_bucket": record["length_bucket"],
                        "query_id": query["query_id"],
                        "route": arm,
                        "mapper": None,
                        "read_state": "none" if arm == "native_recompute" else cfg["read_state"],
                        "round": round_id,
                        "position_in_round": position,
                        "suffix_tokens": len(suffix),
                        "first_token": result["token"],
                        **strip_result(result),
                    },
                )
                del result
    os.replace(gen_tmp, os.path.join(out_dir, "generations.jsonl"))
    os.replace(timing_tmp, os.path.join(out_dir, "timings.jsonl"))
    receipt = {
        "context_id": cid,
        "identity": identity,
        "serialized_reduction": reduction,
        "bf16_serialized_bytes": os.stat(cache_path(cfg, "bf16", cid)).st_size,
        "k16v8_serialized_bytes": os.stat(cache_path(cfg, "k16v8", cid)).st_size,
        "codec_payload_bytes": asym_head["asym_payload_bytes"],
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_host_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = done + ".tmp"
    with open(tmp, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, done)
    if not cfg["keep_caches"]:
        for kind in ("bf16", "k16v8"):
            os.remove(cache_path(cfg, kind, cid))
    log(f"{cid}: native codec comparison complete")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as source:
        cfg = json.load(source)
    import transformers
    from transformers import AutoTokenizer
    from ..published.capture import geometry
    from ..published.run_pair import usable_processors

    cfg = {
        "qualification_lengths": [4096, 16384, 32768],
        "rounds": 3,
        "seed": 0,
        "read_state": "direct",
        "keep_caches": False,
        "serialized_reduction_min": 0.24,
        **cfg,
    }
    if cfg["read_state"] not in ("direct", "page_cache"):
        raise ValueError("read_state must be direct or page_cache")
    threads = min(usable_processors(), 16)
    torch.set_num_threads(threads)
    os.makedirs(cfg["out"], exist_ok=True)
    os.makedirs(cfg["cache_dir"], exist_ok=True)
    os.makedirs(os.path.join(cfg["out"], "contexts"), exist_ok=True)
    records_path = os.path.join(cfg["workload"], "records.jsonl")
    records = choose_qualification_contexts(
        load_jsonl(records_path), cfg["qualification_lengths"]
    )
    identity_blob = {
        "source_digest": source_digest(),
        "records_sha256": file_sha(records_path),
        "model": [cfg["model"], cfg["model_revision"]],
        "contexts": [record["context_id"] for record in records],
        "qualification_lengths": cfg["qualification_lengths"],
        "arms": ARMS,
        "rounds": cfg["rounds"],
        "seed": cfg["seed"],
        "read_state": cfg["read_state"],
        "codec_revision": cfg["codec_revision"],
        "codec_source_sha256": cfg["codec_source_sha256"],
        "codec_contract": {
            "k_dtype": "bfloat16",
            "v_storage_dtype": asym_cache_store.FP8_FORMAT,
            "v_runtime_dtype": "bfloat16",
            "scale_dtype": asym_cache_store.SCALE_DTYPE,
            "scale_scope": asym_cache_store.SCALE_SCOPE,
            "chunk_tokens": "full prefix length for each frozen context",
            "head_axis": None,
            "page_axis": None,
            "outer_header_bytes": asym_cache_store.OUTER_HEADER,
            "padding_bytes": cache_store.BLOCK,
            "expansion": "target device before ordinary attention",
        },
    }
    identity = hashlib.sha256(
        json.dumps(identity_blob, sort_keys=True).encode()
    ).hexdigest()[:16]
    run_path = os.path.join(cfg["out"], "RUN.json")
    if os.path.exists(run_path):
        with open(run_path) as source:
            previous = json.load(source)
        if previous["identity"] != identity:
            raise SystemExit("output directory belongs to a different run identity")
    else:
        with open(run_path, "w") as output:
            json.dump(
                {
                    "identity": identity,
                    **identity_blob,
                    "threads": threads,
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "config": cfg,
                    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                output,
                indent=2,
                sort_keys=True,
            )
    model, load_seconds = load_model(cfg["model"], cfg["model_revision"])
    geom = geometry(model)
    largest = cache_store.payload_bytes(
        geom["layers"],
        geom["kv_heads"],
        max(int(record["length_bucket"]) for record in records),
        geom["head_dim"],
    )
    rx = Receiver(model, largest_payload=largest)
    largest_segment = _largest_codec_segment(
        max(int(record["length_bucket"]) for record in records), geom
    )
    asym_reader = asym_cache_store.Reader(
        largest_segment,
        device=rx.device,
        codec_checkout=cfg["codec_checkout"],
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["model_revision"])
    stop_ids = {tokenizer.convert_tokens_to_ids(token) for token in STOP}
    ledger = os.path.join(cfg["out"], "CACHE_BYTES.jsonl")
    rng = random.Random(f"{cfg['seed']}|native-codec-order")
    completed = 0
    for record in records:
        completed += bool(
            run_context(
                rx,
                asym_reader,
                tokenizer,
                stop_ids,
                cfg,
                record,
                identity,
                ledger,
                rng,
            )
        )
    combine_context_outputs(cfg, records)
    with open(os.path.join(cfg["out"], "COMPLETE.json"), "w") as output:
        json.dump(
            {
                "identity": identity,
                "receiver_seconds_loading": load_seconds,
                "contexts_completed_this_invocation": completed,
                "contexts_total": len(records),
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            output,
            indent=2,
            sort_keys=True,
        )
    log("NATIVE_CODEC_SCREEN_COMPLETE")
    return 0


def _largest_codec_segment(tokens, geom):
    payload = geom["kv_heads"] * tokens * geom["head_dim"]
    return cache_store.BLOCK + payload * 3 + 4096


if __name__ == "__main__":
    sys.exit(main())

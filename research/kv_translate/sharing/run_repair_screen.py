#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""Run a bounded sparse-replay comparison on a frozen sharing workload.

The source model is loaded first and its cache files are written, then it is
unloaded.  One warm receiver compares native recomputation, native cache
restore, unrepaired translation, token-level replay, and (when admitted by a
separate cost check) block-level replay.  Every mapped arm keeps the full
translated prefix; replay replaces selected positions and never prunes or
appends them.

Outputs are one atomic directory per context so an interrupted run resumes
without duplicating generations.  The raw generations stay unscored here and
can be passed to ``score_outputs`` after the accelerator is released.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import resource
import sys
import time

import numpy as np

from . import cache_store
from .cache_store import storage_facts
from .request_path import Clock, Receiver

STOP = ("<|im_end|>", "<|endoftext|>")
NEW_TOKENS = {"document": 48, "code": 768}
BASE_ARMS = (
    "native_recompute",
    "native_restore",
    "mapped_unrepaired",
    "mapped_uniform",
    "mapped_key_deviation",
)
KRI_ARMS = ("mapped_uniform_blocks", "mapped_kri_blocks")


def arms_for(cfg):
    if cfg.get("native_donor_arm"):
        return (cfg["native_donor_arm"],)
    return BASE_ARMS + (KRI_ARMS if cfg.get("kri_enabled", False) else ())


def log(msg):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def source_digest():
    """Hash the executable path, including replay code outside this package."""
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    paths = [
        os.path.abspath(__file__),
        os.path.join(here, "cache_store.py"),
        os.path.join(here, "request_path.py"),
        os.path.join(parent, "blend_select.py"),
        os.path.join(parent, "kri_cost.py"),
        os.path.join(parent, "replay.py"),
        os.path.join(parent, "published", "capture.py"),
        os.path.join(parent, "published", "fit_pair.py"),
        os.path.join(parent, "published", "torch_ops.py"),
    ]
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update(os.path.relpath(path, os.path.dirname(parent)).encode())
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def choose_contexts(records, length_bucket, per_domain):
    """First groups in each domain, ordered independently of file order."""
    chosen = []
    for domain in ("document", "code"):
        eligible = [
            r
            for r in records
            if r["domain"] == domain and int(r["length_bucket"]) == length_bucket
        ]
        eligible.sort(key=lambda r: (r["source_group_id"], r["context_id"]))
        seen = set()
        for row in eligible:
            if row["source_group_id"] in seen:
                continue
            seen.add(row["source_group_id"])
            chosen.append(row)
            if len(seen) == per_domain:
                break
        if len(seen) != per_domain:
            raise ValueError(
                f"wanted {per_domain} {domain} groups at {length_bucket}, "
                f"found {len(seen)}"
            )
    return chosen


def uniform_token_indices(tokens, count, device=None):
    """Deterministic position coverage at an exact token budget."""
    import torch

    tokens, count = int(tokens), int(count)
    if not 0 <= count <= tokens:
        raise ValueError(f"selected token count {count} is outside 0..{tokens}")
    if count == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.div(
        torch.arange(count, dtype=torch.long, device=device) * tokens,
        count,
        rounding_mode="floor",
    )


def mask_from_indices(tokens, selected, device):
    import torch

    mask = torch.zeros(tokens, dtype=torch.bool, device=device)
    mask[selected.to(device)] = True
    if int(mask.sum()) != int(selected.numel()):
        raise ValueError("selected token indices repeat or fall outside the prefix")
    return mask


def expand_block_indices(blocks, block_size, *, total_tokens, device=None):
    """Expand distinct block numbers into sorted contiguous token positions."""
    import torch

    blocks = torch.as_tensor(blocks, dtype=torch.long, device=device).flatten()
    block_size, total_tokens = int(block_size), int(total_tokens)
    if block_size <= 0 or total_tokens % block_size:
        raise ValueError("block size must divide the prefix length")
    total_blocks = total_tokens // block_size
    if blocks.numel() and (
        int(blocks.min()) < 0 or int(blocks.max()) >= total_blocks
    ):
        raise ValueError(f"selected block is outside 0..{total_blocks - 1}")
    if blocks.unique().numel() != blocks.numel():
        raise ValueError("selected block indices repeat")
    offsets = torch.arange(block_size, dtype=torch.long, device=blocks.device)
    return (blocks.sort().values[:, None] * block_size + offsets).flatten()


def cache_path(cfg, role, context_id):
    return os.path.join(cfg["cache_dir"], f"{context_id}.{role}.kv")


def expected(name, revision, record):
    return {
        "model": name,
        "revision": revision,
        "prefix_tokens_sha256": record["prefix_tokens_sha256"],
        "tokens": record["actual_prefix_tokens"],
    }


def tokens_of(workload, rel):
    return np.load(os.path.join(workload, rel)).astype(np.int64).tolist()


def load_model(name, revision):
    import torch
    from transformers import AutoModelForCausalLM

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        name,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda"},
        attn_implementation="sdpa",
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, round(time.time() - t0, 2)


def write_source_caches(cfg, records, ledger):
    import torch

    from ..published.capture import cache_layers

    todo = [
        r
        for r in records
        if not os.path.exists(cache_path(cfg, "source", r["context_id"]))
        and not os.path.exists(
            os.path.join(cfg["out"], "contexts", r["context_id"], "DONE.json")
        )
    ]
    if not todo:
        return {"loaded": False, "contexts": 0}
    model, load_s = load_model(cfg["source"], cfg["source_revision"])
    log(f"source resident in {load_s}s")
    for record in todo:
        cid = record["context_id"]
        ids = torch.as_tensor(
            tokens_of(cfg["workload"], record["prefix_token_file"]), device="cuda"
        )[None]
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            out = model.model(input_ids=ids, use_cache=True)
        torch.cuda.synchronize()
        prefetched = time.perf_counter()
        pairs = [(k, v) for k, v in cache_layers(out.past_key_values)]
        receipt = cache_store.write(
            cache_path(cfg, "source", cid),
            pairs,
            {
                "model": cfg["source"],
                "revision": cfg["source_revision"],
                "prefix_tokens_sha256": record["prefix_tokens_sha256"],
                "role": "source",
            },
        )
        receipt.update(
            context_id=cid,
            seconds_prefill=round(prefetched - start, 4),
            seconds_write=round(time.perf_counter() - prefetched, 4),
        )
        append_jsonl(ledger, receipt)
        del out, pairs, ids
        log(f"{cid}: source cache {receipt['logical_bytes']} bytes")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"loaded": True, "seconds_loading": load_s, "contexts": len(todo)}


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def strip_result(result):
    return {k: v for k, v in result.items() if k not in ("cache", "logits")}


def selection_sha(selected):
    return hashlib.sha256(
        selected.detach().cpu().numpy().astype("<i8").tobytes()
    ).hexdigest()


def read_and_map(rx, cfg, record, *, direct):
    """Read and translate one source cache, retaining the request clock."""
    from ..published import fit_pair

    clock = Clock(rx.device)
    head, io = rx.reader.read(
        cache_path(cfg, "source", record["context_id"]),
        direct=direct,
        expect=expected(cfg["source"], cfg["source_revision"], record),
    )
    clock.mark("read")
    pairs = cache_store.as_pairs(rx.reader.tensor(head))
    clock.mark("to_card")
    moved = fit_pair.transfer(
        rx.mapper,
        pairs,
        rx.source_theta,
        rx.geom["rope_theta"],
        dtype=rx.dtype,
    )
    clock.mark("map")
    return moved, io, clock


def mapped_selection(rx, cfg, record, prefix_ids, keys, arm):
    """Select the exact receiver positions named by one mapped arm."""
    import torch

    from ..blend_select import select_qwen3_layer0_key_deviation

    if arm == "mapped_uniform":
        return uniform_token_indices(
            len(prefix_ids), cfg["selected_tokens"], device=rx.device
        )
    if arm == "mapped_key_deviation":
        ids = torch.as_tensor(prefix_ids, dtype=torch.long, device=rx.device)[None]
        selected, _, _ = select_qwen3_layer0_key_deviation(
            rx.model, ids, keys[0], cfg["selected_tokens"]
        )
        return selected
    if arm == "mapped_uniform_blocks":
        total_blocks = len(prefix_ids) // cfg["kri_block_size"]
        blocks = uniform_token_indices(
            total_blocks, cfg["kri_selected_blocks"], device=rx.device
        )
        return expand_block_indices(
            blocks,
            cfg["kri_block_size"],
            total_tokens=len(prefix_ids),
            device=rx.device,
        )
    if arm == "mapped_kri_blocks":
        with open(kri_prior_path(cfg, record["context_id"])) as f:
            prior = json.load(f)
        validate_kri_prior(prior, cfg, record)
        return expand_block_indices(
            prior["selected_blocks"],
            cfg["kri_block_size"],
            total_tokens=len(prefix_ids),
            device=rx.device,
        )
    if arm == "mapped_unrepaired":
        return None
    raise ValueError(f"unknown mapped arm {arm!r}")


def mapped_request(rx, cfg, record, prefix_ids, suffix, arm, *, direct):
    """One complete mapped request, including selection and replay costs."""
    import torch

    from ..published.capture import make_cache
    from ..replay import replay_selected

    pairs, io, clock = read_and_map(rx, cfg, record, direct=direct)
    keys, values = [p[0] for p in pairs], [p[1] for p in pairs]
    selected = mapped_selection(rx, cfg, record, prefix_ids, keys, arm)
    if selected is not None:
        clock.mark("select")

    info = {"replayed": 0, "layers_written": []}
    if selected is not None:
        mask = mask_from_indices(len(prefix_ids), selected, rx.device)
        keys, values, info = replay_selected(
            rx.model,
            torch.as_tensor(prefix_ids, dtype=torch.long, device=rx.device)[None],
            mask,
            keys,
            values,
            in_place=True,
        )
        clock.mark("repair")
        if info["replayed"] != cfg["selected_tokens"]:
            raise RuntimeError(
                f"replayed {info['replayed']} tokens, expected {cfg['selected_tokens']}"
            )
        if len(info["layers_written"]) != rx.geom["layers"]:
            raise RuntimeError("replay did not write every receiver layer")

    cache = make_cache(zip(keys, values))
    clock.mark("install")
    token, cache, logits = rx._finish(clock, suffix, cache)
    return {
        "token": token,
        "cache": cache,
        "logits": logits,
        "io": io,
        "selected_tokens": 0 if selected is None else int(selected.numel()),
        "selection_sha256": None if selected is None else selection_sha(selected),
        "replay": info,
        **clock.report(),
    }


def native_donor_request(rx, cfg, record, prefix_ids, suffix, arm, *, direct):
    """Oracle diagnostic: scatter native K/V at one practical arm's mask."""
    from ..published.capture import make_cache

    base_arm = arm.removesuffix("_native_donor")
    pairs, source_io, clock = read_and_map(rx, cfg, record, direct=direct)
    keys, values = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
    selected = mapped_selection(rx, cfg, record, prefix_ids, keys, base_arm)
    if selected is None:
        raise ValueError("native-donor diagnostic requires a repair mask")
    if selected.numel() != cfg["selected_tokens"]:
        raise RuntimeError(
            f"donor mask selected {selected.numel()} tokens, "
            f"expected {cfg['selected_tokens']}"
        )
    clock.mark("select")
    head, native_io = rx.reader.read(
        cache_path(cfg, "target", record["context_id"]),
        direct=False,
        expect=expected(cfg["target"], cfg["target_revision"], record),
    )
    clock.mark("native_read")
    native = cache_store.as_pairs(rx.reader.tensor(head))
    clock.mark("native_to_card")
    if len(native) != len(pairs):
        raise RuntimeError("native donor and mapped cache differ in layer count")
    for (key, value), (native_key, native_value) in zip(pairs, native):
        if key.shape != native_key.shape or value.shape != native_value.shape:
            raise RuntimeError("native donor and mapped cache shapes differ")
        key[:, :, selected] = native_key[:, :, selected]
        value[:, :, selected] = native_value[:, :, selected]
    clock.mark("donor_scatter")
    cache = make_cache(zip(keys, values))
    clock.mark("install")
    token, cache, logits = rx._finish(clock, suffix, cache)
    return {
        "token": token,
        "cache": cache,
        "logits": logits,
        "io": {"source": source_io, "native_donor": native_io},
        "selected_tokens": int(selected.numel()),
        "selection_sha256": selection_sha(selected),
        "oracle_assisted": True,
        **clock.report(),
    }


def request(rx, cfg, record, prefix, suffix, arm, *, direct):
    if arm == "native_recompute":
        return rx.recompute(prefix, suffix)
    if arm == "native_restore":
        return rx.restore(
            cache_path(cfg, "target", record["context_id"]),
            suffix,
            direct=direct,
            expect=expected(cfg["target"], cfg["target_revision"], record),
        )
    if arm.endswith("_native_donor"):
        return native_donor_request(
            rx, cfg, record, prefix, suffix, arm, direct=direct
        )
    return mapped_request(rx, cfg, record, prefix, suffix, arm, direct=direct)


def decode(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=False)


def prepare_target_cache(rx, cfg, record, prefix, ledger):
    path = cache_path(cfg, "target", record["context_id"])
    if os.path.exists(path):
        return cache_store.header(path)
    import torch

    torch.cuda.synchronize()
    start = time.perf_counter()
    pairs = rx.prefix_cache(prefix)
    torch.cuda.synchronize()
    prefetched = time.perf_counter()
    receipt = cache_store.write(
        path,
        pairs,
        {
            "model": cfg["target"],
            "revision": cfg["target_revision"],
            "prefix_tokens_sha256": record["prefix_tokens_sha256"],
            "role": "target",
        },
    )
    receipt.update(
        context_id=record["context_id"],
        seconds_prefill=round(prefetched - start, 4),
        seconds_write=round(time.perf_counter() - prefetched, 4),
    )
    append_jsonl(ledger, receipt)
    del pairs
    return cache_store.header(path)


def run_context(rx, tokenizer, stop_ids, cfg, record, identity, ledger, rng):
    import torch

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
    if len(prefix) != cfg["length_bucket"]:
        raise ValueError(f"{cid}: prefix has {len(prefix)} tokens")
    prepare_target_cache(rx, cfg, record, prefix, ledger)
    torch.cuda.reset_peak_memory_stats()
    primary_direct = cfg["read_state"] == "direct"

    for query in record["fixed_queries"]:
        suffix = tokens_of(cfg["workload"], query["suffix_token_file"])
        cap = NEW_TOKENS[record["domain"]]
        for arm in arms_for(cfg):
            result = request(
                rx, cfg, record, prefix, suffix, arm, direct=primary_direct
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
                    "mapper": cfg["mapper_name"] if arm.startswith("mapped_") else None,
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
            order = list(arms_for(cfg))
            rng.shuffle(order)
            for position, arm in enumerate(order):
                result = request(
                    rx, cfg, record, prefix, suffix, arm, direct=primary_direct
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
                        "mapper": (
                            cfg["mapper_name"] if arm.startswith("mapped_") else None
                        ),
                        "read_state": (
                            "none" if arm == "native_recompute" else cfg["read_state"]
                        ),
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
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_host_rss_gib": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2
        ),
        "cache_dir_free_bytes": storage_facts(cfg["cache_dir"])["free_bytes"],
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = done + ".tmp"
    with open(tmp, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, done)
    for role in ("source", "target"):
        path = cache_path(cfg, role, cid)
        if os.path.exists(path) and not cfg["keep_caches"]:
            os.remove(path)
    log(f"{cid}: done, peak card memory {receipt['peak_gpu_gib']} GiB")
    return True


def combine_context_outputs(cfg, records):
    for name in ("generations.jsonl", "timings.jsonl"):
        tmp = os.path.join(cfg["out"], name + ".tmp")
        with open(tmp, "wb") as dst:
            for record in records:
                path = os.path.join(cfg["out"], "contexts", record["context_id"], name)
                if not os.path.exists(path):
                    raise RuntimeError(f"missing completed output {path}")
                with open(path, "rb") as src:
                    for block in iter(lambda: src.read(1 << 20), b""):
                        dst.write(block)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(tmp, os.path.join(cfg["out"], name))


def profile_kri_cost(rx, tokenizer, cfg, record):
    """Time a fixed block sample and forecast a reusable 16-prefix bake."""
    import statistics

    from ..kri_cost import measure_kri_d_kv_blocks

    path = os.path.join(cfg["out"], "KRI_COST.json")
    if os.path.exists(path) or not cfg.get("kri_cost_sample_blocks"):
        return
    pairs, _, clock = read_and_map(rx, cfg, record, direct=False)
    clock.mark("mapped_for_kri")
    neutral_probe = "\n\nGive a brief high-level description of the preceding text."
    probe_ids = tokenizer(
        neutral_probe, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(rx.device)
    total_blocks = cfg["length_bucket"] // cfg["kri_block_size"]
    sample_count = min(int(cfg["kri_cost_sample_blocks"]), total_blocks)
    blocks = uniform_token_indices(total_blocks, sample_count, rx.device).tolist()
    measured = measure_kri_d_kv_blocks(
        rx.model,
        [pair[0] for pair in pairs],
        [pair[1] for pair in pairs],
        probe_ids,
        prefix_tokens=cfg["length_bucket"],
        block_size=cfg["kri_block_size"],
        blocks=blocks,
    )
    seconds = [sample["seconds"] for sample in measured["samples"]]
    per_ablation = statistics.median(seconds)
    measured.update(
        {
            "context_id": record["context_id"],
            "probe": neutral_probe,
            "probe_token_ids_sha256": hashlib.sha256(
                probe_ids.detach().cpu().numpy().astype("<i8").tobytes()
            ).hexdigest(),
            "map_path_seconds": clock.report(),
            "estimate_basis": "sample median times 1024 blocks; no overlap credited",
            "estimated_seconds_per_prefix": per_ablation * total_blocks,
            "estimated_seconds_16_prefixes": per_ablation * total_blocks * 16,
            "estimated_hours_16_prefixes": per_ablation * total_blocks * 16 / 3600,
        }
    )
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(measured, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def kri_prior_path(cfg, context_id):
    root = cfg.get("kri_prior_dir", os.path.join(cfg["out"], "kri"))
    return os.path.join(root, f"{context_id}.json")


def validate_kri_prior(prior, cfg, record, *, identity=None):
    """Refuse a baked selector whose identity, geometry, or budget changed."""
    if identity is not None and prior.get("identity") != identity:
        raise ValueError("KRI prior belongs to a different run identity")
    if prior.get("prefix_tokens_sha256") != record["prefix_tokens_sha256"]:
        raise ValueError("KRI prior belongs to a different prefix")
    if prior.get("block_size") != cfg["kri_block_size"]:
        raise ValueError("KRI prior block size differs from the configured selector")
    blocks = prior.get("selected_blocks", [])
    if len(blocks) != cfg["kri_selected_blocks"]:
        raise ValueError("KRI prior block count differs from the configured budget")
    if prior.get("selected_tokens") != cfg["selected_tokens"]:
        raise ValueError("KRI prior token count differs from the configured budget")
    return prior


def bake_kri_prior(rx, tokenizer, cfg, record, identity):
    """Bake one query-independent KRI-D-kv block prior for a prefix."""
    if not cfg.get("kri_enabled", False):
        return None
    path = kri_prior_path(cfg, record["context_id"])
    if os.path.exists(path):
        with open(path) as f:
            prior = json.load(f)
        return validate_kri_prior(prior, cfg, record, identity=identity)

    from ..kri_cost import measure_kri_d_kv_blocks

    pairs, io, clock = read_and_map(rx, cfg, record, direct=False)
    clock.mark("mapped_for_kri")
    neutral_probe = "\n\nGive a brief high-level description of the preceding text."
    probe_ids = tokenizer(
        neutral_probe, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(rx.device)
    total_blocks = cfg["length_bucket"] // cfg["kri_block_size"]
    measured = measure_kri_d_kv_blocks(
        rx.model,
        [pair[0] for pair in pairs],
        [pair[1] for pair in pairs],
        probe_ids,
        prefix_tokens=cfg["length_bucket"],
        block_size=cfg["kri_block_size"],
        blocks=range(total_blocks),
    )
    ranked = sorted(
        measured["samples"],
        key=lambda sample: (-sample["hidden_l2_sum"], sample["block"]),
    )
    selected_blocks = sorted(
        sample["block"] for sample in ranked[: cfg["kri_selected_blocks"]]
    )
    selected = expand_block_indices(
        selected_blocks,
        cfg["kri_block_size"],
        total_tokens=cfg["length_bucket"],
    )
    if selected.numel() != cfg["selected_tokens"]:
        raise RuntimeError(
            f"KRI selected {selected.numel()} tokens, expected {cfg['selected_tokens']}"
        )
    measured.update(
        {
            "identity": identity,
            "context_id": record["context_id"],
            "prefix_tokens_sha256": record["prefix_tokens_sha256"],
            "probe": neutral_probe,
            "probe_token_ids_sha256": hashlib.sha256(
                probe_ids.detach().cpu().numpy().astype("<i8").tobytes()
            ).hexdigest(),
            "selection_rule": "largest hidden_l2_sum, stable tie break by block",
            "block_size": cfg["kri_block_size"],
            "selected_blocks": selected_blocks,
            "selected_tokens": int(selected.numel()),
            "selection_sha256": selection_sha(selected),
            "source_cache_io": io,
            "map_path_seconds": clock.report(),
        }
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(measured, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    del pairs
    log(f"{record['context_id']}: baked KRI prior over {total_blocks} blocks")
    return measured


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)

    import torch
    import transformers
    from transformers import AutoTokenizer

    from ..published import fit_pair
    from ..published.capture import geometry
    from ..published.run_pair import usable_processors

    threads = min(usable_processors(), 16)
    torch.set_num_threads(threads)
    cfg = {
        "length_bucket": 16384,
        "contexts_per_domain": 8,
        "repair_fraction": 0.25,
        "rounds": 3,
        "seed": 0,
        "read_state": "direct",
        "keep_caches": False,
        "kri_cost_sample_blocks": 0,
        "kri_enabled": False,
        "kri_block_size": 16,
        "kri_selected_blocks": 256,
        "native_donor_arm": "",
        **cfg,
    }
    cfg["selected_tokens"] = int(cfg["length_bucket"] * cfg["repair_fraction"])
    if cfg["selected_tokens"] != 4096 or cfg["length_bucket"] != 16384:
        raise ValueError("this bounded comparison is fixed at 4096 of 16384 tokens")
    if (
        cfg["kri_enabled"]
        and cfg["kri_block_size"] * cfg["kri_selected_blocks"]
        != cfg["selected_tokens"]
    ):
        raise ValueError("the KRI and token replay budgets must match")
    if cfg["read_state"] not in ("direct", "page_cache"):
        raise ValueError("read_state must be direct or page_cache")
    os.makedirs(cfg["out"], exist_ok=True)
    os.makedirs(cfg["cache_dir"], exist_ok=True)
    os.makedirs(os.path.join(cfg["out"], "contexts"), exist_ok=True)

    records_path = os.path.join(cfg["workload"], "records.jsonl")
    all_records = load_jsonl(records_path)
    records = choose_contexts(
        all_records, cfg["length_bucket"], cfg["contexts_per_domain"]
    )
    records_sha = file_sha(records_path)
    mapper_sha = file_sha(cfg["mapper"])
    identity_blob = {
        "source_digest": source_digest(),
        "records_sha256": records_sha,
        "mapper_sha256": mapper_sha,
        "models": [
            cfg["source"],
            cfg["source_revision"],
            cfg["target"],
            cfg["target_revision"],
        ],
        "contexts": [r["context_id"] for r in records],
        "arms": arms_for(cfg),
        "selected_tokens": cfg["selected_tokens"],
        "rounds": cfg["rounds"],
        "seed": cfg["seed"],
        "read_state": cfg["read_state"],
        "kri": {
            "enabled": bool(cfg["kri_enabled"]),
            "block_size": cfg["kri_block_size"],
            "selected_blocks": cfg["kri_selected_blocks"],
            "cost_sample_blocks": cfg["kri_cost_sample_blocks"],
        },
        "native_donor_arm": cfg["native_donor_arm"],
    }
    identity = hashlib.sha256(
        json.dumps(identity_blob, sort_keys=True).encode()
    ).hexdigest()[:16]
    run_path = os.path.join(cfg["out"], "RUN.json")
    if os.path.exists(run_path):
        with open(run_path) as f:
            old = json.load(f)
        if old["identity"] != identity:
            raise SystemExit("output directory belongs to a different run identity")
    else:
        with open(run_path, "w") as f:
            json.dump(
                {
                    "identity": identity,
                    **identity_blob,
                    "threads": threads,
                    "processors_visible": os.cpu_count(),
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "storage": storage_facts(cfg["cache_dir"]),
                    "config": cfg,
                    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                f,
                indent=2,
                sort_keys=True,
            )

    ledger = os.path.join(cfg["out"], "CACHE_BYTES.jsonl")
    source_info = write_source_caches(cfg, records, ledger)
    model, load_s = load_model(cfg["target"], cfg["target_revision"])
    geom = geometry(model)
    largest = cache_store.payload_bytes(
        geom["layers"], geom["kv_heads"], cfg["length_bucket"], geom["head_dim"]
    )
    for record in records:
        path = cache_path(cfg, "source", record["context_id"])
        if os.path.exists(path):
            largest = max(largest, cache_store.header(path)["payload_bytes"])
    mapper = fit_pair.load_mapper(cfg["mapper"])
    if mapper["method"] != cfg["mapper_name"]:
        raise ValueError(
            f"mapper method {mapper['method']!r} != configured {cfg['mapper_name']!r}"
        )
    mapper = {**mapper, "W": mapper["W"].to("cuda"), "b": mapper["b"].to("cuda")}
    rx = Receiver(
        model,
        largest_payload=largest,
        mapper=mapper,
        source_theta=cfg["source_rope_theta"],
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["target"], revision=cfg["target_revision"]
    )
    stop_ids = {tokenizer.convert_tokens_to_ids(token) for token in STOP}
    profile_kri_cost(rx, tokenizer, cfg, records[0])
    rng = random.Random(f"{cfg['seed']}|repair-order")
    completed = 0
    for record in records:
        done = os.path.join(
            cfg["out"], "contexts", record["context_id"], "DONE.json"
        )
        if not os.path.exists(done):
            bake_kri_prior(rx, tokenizer, cfg, record, identity)
        completed += bool(
            run_context(rx, tokenizer, stop_ids, cfg, record, identity, ledger, rng)
        )
    combine_context_outputs(cfg, records)
    with open(os.path.join(cfg["out"], "COMPLETE.json"), "w") as f:
        json.dump(
            {
                "identity": identity,
                "source": source_info,
                "receiver_seconds_loading": load_s,
                "contexts_completed_this_invocation": completed,
                "contexts_total": len(records),
                "card": torch.cuda.get_device_name(0),
                "card_memory_gib": round(
                    torch.cuda.get_device_properties(0).total_memory / 2**30, 1
                ),
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            f,
            indent=2,
            sort_keys=True,
        )
    log("REPAIR_SCREEN_COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

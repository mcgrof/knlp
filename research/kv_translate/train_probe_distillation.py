#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""Qualify cache translation trained through document-independent probes.

This runner is deliberately a qualification, not a model-selection campaign.
It first proves that the frozen receiver can answer held-out planted-fact tasks
from its native cache.  It then fits one affine translator and starts three
matched residual arms from exactly the same serialized initialization:
receiver-cache reconstruction, output KL on relevant continuations, and output
KL on document-independent continuations.  Each arm receives the same measured
wall-time budget.

The primary readout is greedy free generation on unseen documents.  KL is kept
as a diagnostic, and a successful run admits a larger fresh-seed comparison;
it does not by itself establish a subliminal-learning result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.jacobian_kv.evaluate import teacher_forced_kl  # noqa: E402
from research.kv_translate.fit import (  # noqa: E402
    Accumulator,
    SourceLayout,
    fit_block,
    select_layers,
)
from research.kv_translate.pairs import (  # noqa: E402
    check_pair,
    describe,
    prefill,
    to_content_keys,
    to_stored_keys,
)
from research.kv_translate.refine import (  # noqa: E402
    ResidualMapper,
    behaviour_loss,
    kv_error_loss,
)
from research.kv_translate.run_a0 import (  # noqa: E402
    continuation_logits,
    continuation_logits_grad,
    wikitext_chunks,
)
from research.kv_translate.run_a1 import (  # noqa: E402
    Mapper,
    flat_features,
    target_blocks,
)
from research.kv_translate.run_h1_smoke import greedy  # noqa: E402
from research.kv_translate.tasks import (  # noqa: E402
    build_items,
    grade_cloze,
    grade_free,
    grade_retrieval,
    wikitext_documents,
)

DEFAULT_SOURCE = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_SOURCE_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DEFAULT_TARGET_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(state: dict[str, torch.Tensor]) -> str:
    """Stable digest of named tensors, independent of serialization details."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def module_state(rk: ResidualMapper, rv: ResidualMapper) -> dict[str, torch.Tensor]:
    state = {}
    for prefix, mapper in (("k", rk), ("v", rv)):
        for name, value in mapper.blocks.state_dict().items():
            state[f"{prefix}.{name}"] = value.detach().cpu().clone()
    return state


def cyclic_schedule(n_items: int, n_steps: int, seed: int) -> list[int]:
    """Deterministic shuffled passes, with no sampling-with-replacement gaps."""
    if n_items <= 0:
        raise ValueError("a schedule needs at least one item")
    rng = np.random.default_rng(seed)
    order = []
    while len(order) < n_steps:
        order.extend(int(index) for index in rng.permutation(n_items))
    return order[:n_steps]


def summarize_task_rows(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["kind"], row["condition"])].append(float(row["primary"]))
    out = {}
    for (kind, condition), values in sorted(grouped.items()):
        out.setdefault(kind, {})[condition] = {
            "mean": float(np.mean(values)),
            "n": len(values),
        }
    return out


def mechanics_decision(training: dict, diagnostics: dict, identity_failures: list):
    """Apply the narrow qualification gate without treating task KL as quality."""
    before = diagnostics["unrelated"]["affine"]["mean_kl"]
    after = diagnostics["unrelated"]["unrelated"]["mean_kl"]
    changed = bool(training["unrelated"]["parameters_changed"])
    passed = changed and after < before and not identity_failures
    return {
        "unrelated_parameters_changed": changed,
        "unrelated_heldout_kl_before": before,
        "unrelated_heldout_kl_after": after,
        "unrelated_heldout_kl_improved": after < before,
        "identity_preserved": not identity_failures,
        "passes": passed,
    }


def choose_roles(tok, *, ctx: int, probe_len: int, counts: dict, seed: int):
    """Split complete documents before any fitting, training, or task building."""
    documents = wikitext_documents(max_docs=1024, split="train[:3%]")
    rng = np.random.default_rng(seed)
    rng.shuffle(documents)
    need = ctx + probe_len
    windows = []
    unused = []
    for title, body in documents:
        ids = tok(
            body,
            return_tensors="pt",
            truncation=True,
            max_length=need,
        ).input_ids[0]
        if len(ids) == need:
            windows.append((title, body, ids.clone()))
        else:
            unused.append((title, body))
    required = sum(counts.values())
    if len(windows) < required:
        raise RuntimeError(f"only {len(windows)} documents satisfy the {need}-token floor")
    roles, offset = {}, 0
    for role in ("calibration", "training", "validation"):
        count = counts[role]
        roles[role] = windows[offset : offset + count]
        offset += count
    # Task prompts are rebuilt with planted facts and therefore draw from
    # wholly different documents, not from unused tails of training windows.
    candidates = [(title, body) for title, body, _ in windows[offset:]] + unused
    task_items = build_items(
        candidates[:256],
        tok,
        ctx,
        seed=seed + 17,
        kinds=("retrieval", "cloze", "free"),
    )
    by_doc = defaultdict(list)
    for item in task_items:
        by_doc[item.doc_id].append(item)
    complete = [
        title
        for title, items in by_doc.items()
        if {item.kind for item in items} == {"retrieval", "cloze", "free"}
    ][: counts["task"]]
    if len(complete) != counts["task"]:
        raise RuntimeError(
            f"only {len(complete)} documents produced all three held-out tasks"
        )
    roles["task"] = [item for title in complete for item in by_doc[title]]
    role_names = {
        role: [entry[0] for entry in entries]
        for role, entries in roles.items()
        if role != "task"
    }
    role_names["task"] = complete
    flat = [name for names in role_names.values() for name in names]
    if len(flat) != len(set(flat)):
        raise RuntimeError("document roles overlap")
    return roles, role_names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--source-revision", default=DEFAULT_SOURCE_REVISION)
    parser.add_argument("--target-revision", default=DEFAULT_TARGET_REVISION)
    parser.add_argument("--signal-result", required=True)
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--probe-len", type=int, default=32)
    parser.add_argument("--calibration-documents", type=int, default=24)
    parser.add_argument("--training-documents", type=int, default=16)
    parser.add_argument("--validation-documents", type=int, default=4)
    parser.add_argument("--task-documents", type=int, default=8)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-5)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--arm-seconds", type=float, default=60.0)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--task-max-new", type=int, default=16)
    parser.add_argument("--free-max-new", type=int, default=32)
    parser.add_argument("--noise-scale", type=float, default=0.05)
    parser.add_argument("--headroom-min", type=float, default=0.125)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    def log(message):
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.out_dir, exist_ok=True)
    started = time.time()
    source_path = os.path.abspath(__file__)
    signal = json.load(open(args.signal_result))
    if not signal.get("decision", {}).get("passes"):
        raise SystemExit("the frozen unrelated-probe signal gate did not pass")
    if signal.get("target") != args.target or signal.get("target_revision") != args.target_revision:
        raise SystemExit("the signal receipt does not identify this receiver revision")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    tok = AutoTokenizer.from_pretrained(args.target, revision=args.target_revision)
    source_tok = AutoTokenizer.from_pretrained(
        args.source, revision=args.source_revision
    )
    tokenizers_match = tok.get_vocab() == source_tok.get_vocab()
    models, geometry = {}, {}
    for role, model_id, revision in (
        ("source", args.source, args.source_revision),
        ("target", args.target, args.target_revision),
    ):
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=dtype,
            attn_implementation="eager",
        ).to(device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models[role] = model
        geometry[role] = describe(model, model_id)
    sg, tg = geometry["source"], geometry["target"]
    compatibility = check_pair(sg, tg, tokenizers_match=tokenizers_match)
    if compatibility["blockers"]:
        raise SystemExit(f"incompatible model pair: {compatibility['blockers']}")
    log(
        f"loaded frozen pair: {sg.n_layers}L/{sg.n_kv_heads}H -> "
        f"{tg.n_layers}L/{tg.n_kv_heads}H"
    )

    counts = {
        "calibration": args.calibration_documents,
        "training": args.training_documents,
        "validation": args.validation_documents,
        "task": args.task_documents,
    }
    roles, role_names = choose_roles(
        tok, ctx=args.ctx, probe_len=args.probe_len, counts=counts, seed=args.seed
    )
    log(
        "frozen document roles: "
        + ", ".join(f"{name}={len(values)}" for name, values in role_names.items())
    )

    unrelated_all = [
        ids[: args.probe_len].unsqueeze(0).to(device)
        for _, ids in wikitext_chunks(tok, 16, 0, args.probe_len, args.seed + 91)
    ]
    unrelated_train, unrelated_held = unrelated_all[:12], unrelated_all[12:]
    unrelated_hashes = [
        hashlib.sha256(ids.cpu().numpy().astype("<i8").tobytes()).hexdigest()
        for ids in unrelated_all
    ]

    eos_ids = {tok.eos_token_id} | {
        tok.convert_tokens_to_ids(token)
        for token in ("<|im_end|>", "<|endoftext|>")
        if tok.convert_tokens_to_ids(token) is not None
    }
    eos_ids = {token for token in eos_ids if token is not None and token >= 0}

    neutral_ids = tok(
        "\n", return_tensors="pt", add_special_tokens=False
    ).input_ids[0]
    neutral_ids = neutral_ids.repeat((args.ctx + len(neutral_ids) - 1) // len(neutral_ids))[
        : args.ctx
    ].unsqueeze(0).to(device)
    neutral = prefill(models["target"], neutral_ids, tg)

    task_items = roles["task"]
    task_docs = role_names["task"]
    task_prompts = {}
    for doc in task_docs:
        prompt_ids = next(
            item.meta["prompt_ids"] for item in task_items if item.doc_id == doc
        )
        task_prompts[doc] = torch.tensor(
            prompt_ids, dtype=torch.long, device=device
        ).unsqueeze(0)
    native_task = {
        doc: prefill(models["target"], ids, tg) for doc, ids in task_prompts.items()
    }
    shift = {
        doc: task_docs[(index + 1) % len(task_docs)]
        for index, doc in enumerate(task_docs)
    }

    def noisy_cache(cache, index):
        generator = torch.Generator(device=device).manual_seed(args.seed + 1000 + index)
        keys, values = [], []
        for tensor in cache.keys_post:
            noise = torch.randn(
                tensor.shape, generator=generator, device=device, dtype=tensor.dtype
            )
            keys.append(tensor + noise * tensor.float().std().to(tensor.dtype) * args.noise_scale)
        for tensor in cache.values:
            noise = torch.randn(
                tensor.shape, generator=generator, device=device, dtype=tensor.dtype
            )
            values.append(
                tensor + noise * tensor.float().std().to(tensor.dtype) * args.noise_scale
            )
        return keys, values

    # Native task headroom is an admission test, run before fitting or training.
    preflight_rows = []
    identity_exact = True
    retrieval_items = [item for item in task_items if item.kind == "retrieval"]
    for index, item in enumerate(retrieval_items):
        query = tok(
            item.query, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        native = native_task[item.doc_id]
        wrong = native_task[shift[item.doc_id]]
        noisy_keys, noisy_values = noisy_cache(native, index)
        conditions = {
            "target_native": (native.keys_post, native.values),
            "native_copy_identity": (
                [tensor.clone() for tensor in native.keys_post],
                [tensor.clone() for tensor in native.values],
            ),
            "neutral_cache": (neutral.keys_post, neutral.values),
            "wrong_native": (wrong.keys_post, wrong.values),
            "noisy_native": (noisy_keys, noisy_values),
        }
        generated = {}
        for condition, (keys, values) in conditions.items():
            tokens = greedy(
                models["target"], keys, values, query, args.ctx, args.task_max_new, eos_ids
            )
            generated[condition] = tokens
            score = grade_retrieval(
                item.answer, tok.decode(tokens, skip_special_tokens=True)
            )
            preflight_rows.append(
                {
                    "doc": item.doc_id,
                    "condition": condition,
                    "tokens": tokens,
                    "produced": tok.decode(tokens, skip_special_tokens=True),
                    **score,
                }
            )
        identity_exact = identity_exact and (
            generated["target_native"] == generated["native_copy_identity"]
        )
    preflight_accuracy = {
        condition: float(
            np.mean(
                [
                    row["correct"]
                    for row in preflight_rows
                    if row["condition"] == condition
                ]
            )
        )
        for condition in sorted({row["condition"] for row in preflight_rows})
    }
    task_headroom = preflight_accuracy["target_native"] - max(
        preflight_accuracy["neutral_cache"], preflight_accuracy["wrong_native"]
    )
    task_gate = identity_exact and task_headroom >= args.headroom_min
    log(
        f"native retrieval gate: {preflight_accuracy}, headroom={task_headroom:.3f}, "
        f"identity_exact={identity_exact} => {'PASS' if task_gate else 'FAIL'}"
    )
    with open(os.path.join(args.out_dir, "preflight_rows.jsonl"), "w") as output:
        for row in preflight_rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    if not task_gate:
        stopped = {
            "schema": "unrelated_probe_training_qualification_v1",
            "status": "STOPPED_NATIVE_TASK_GATE",
            "preflight_accuracy": preflight_accuracy,
            "task_headroom": task_headroom,
            "headroom_min": args.headroom_min,
            "identity_exact": identity_exact,
            "source_digest": sha256_file(source_path),
            "config": vars(args),
            "wall_seconds": time.time() - started,
        }
        with open(os.path.join(args.out_dir, "RESULT.json"), "w") as output:
            json.dump(stopped, output, indent=2, sort_keys=True)
        return 0

    layout = SourceLayout(sg.n_layers, sg.n_kv_heads, sg.head_dim)
    n_targets = tg.n_layers * tg.n_kv_heads
    accumulators = {
        kind: Accumulator(layout, n_targets, tg.head_dim, device=device)
        for kind in ("k", "v")
    }
    fit_started = time.time()
    for index, (_, _, ids) in enumerate(roles["calibration"]):
        prompt = ids[: args.ctx].unsqueeze(0).to(device)
        sp = prefill(models["source"], prompt, sg)
        tp = prefill(models["target"], prompt, tg)
        accumulators["k"].add(
            flat_features(to_content_keys(sp)), target_blocks(to_content_keys(tp))
        )
        accumulators["v"].add(
            flat_features([value.float() for value in sp.values]),
            target_blocks([value.float() for value in tp.values]),
        )
        if (index + 1) % 8 == 0:
            log(f"  calibrated {index + 1}/{len(roles['calibration'])} documents")
        del sp, tp
    scales = {
        kind: float(accumulators[kind].xtx.diagonal().mean()) for kind in ("k", "v")
    }
    selected = {
        (kind, layer, head): select_layers(
            accumulators[kind],
            layer * tg.n_kv_heads + head,
            head,
            args.ridge * scales[kind],
            args.k,
            False,
        )
        for kind in ("k", "v")
        for layer in range(tg.n_layers)
        for head in range(tg.n_kv_heads)
    }
    affine = {
        kind: Mapper(
            {
                (layer, head): fit_block(
                    accumulators[kind],
                    target=layer * tg.n_kv_heads + head,
                    target_layer=layer,
                    head=head,
                    kind=kind,
                    layers=selected[(kind, layer, head)],
                    ridge=args.ridge * scales[kind],
                    head_local=False,
                )
                for layer in range(tg.n_layers)
                for head in range(tg.n_kv_heads)
            },
            layout,
            tg,
            kind,
        ).cast(torch.float32)
        for kind in ("k", "v")
    }
    calibration_seconds = time.time() - fit_started
    log(f"affine map fitted in {calibration_seconds:.1f}s")
    del accumulators

    proto_k = ResidualMapper(affine["k"], layout, tg, "k", args.hidden).to(device)
    proto_v = ResidualMapper(affine["v"], layout, tg, "v", args.hidden).to(device)
    proto_k.freeze_affine()
    proto_v.freeze_affine()
    init_state = module_state(proto_k, proto_v)
    init_digest = tensor_digest(init_state)
    init_path = os.path.join(args.out_dir, "residual_initialization.pt")
    torch.save(init_state, init_path)
    log(f"serialized common residual initialization {init_digest[:16]}")
    del proto_k, proto_v

    def make_residual_pair():
        rk = ResidualMapper(affine["k"], layout, tg, "k", args.hidden).to(device)
        rv = ResidualMapper(affine["v"], layout, tg, "v", args.hidden).to(device)
        rk.freeze_affine()
        rv.freeze_affine()
        rk.blocks.load_state_dict(
            {name[2:]: value for name, value in init_state.items() if name.startswith("k.")}
        )
        rv.blocks.load_state_dict(
            {name[2:]: value for name, value in init_state.items() if name.startswith("v.")}
        )
        if tensor_digest(module_state(rk, rv)) != init_digest:
            raise RuntimeError("residual initialization changed while loading")
        return rk, rv

    def prepare_window(entry):
        title, _, ids = entry
        ids = ids.to(device)
        prompt = ids[: args.ctx].unsqueeze(0)
        relevant = ids[args.ctx : args.ctx + args.probe_len].unsqueeze(0)
        sp = prefill(models["source"], prompt, sg)
        tp = prefill(models["target"], prompt, tg)
        return {
            "doc": title,
            "Xk": flat_features(to_content_keys(sp)).float(),
            "Xv": flat_features([value.float() for value in sp.values]).float(),
            "Yk": [value.float() for value in to_content_keys(tp)],
            "Yv": [value.float() for value in tp.values],
            "native_keys": [value.detach() for value in tp.keys_post],
            "native_values": [value.detach() for value in tp.values],
            "relevant": relevant,
        }

    train_set = [prepare_window(entry) for entry in roles["training"]]
    validation_set = [prepare_window(entry) for entry in roles["validation"]]
    log(f"prepared {len(train_set)} training and {len(validation_set)} validation caches")

    schedule = cyclic_schedule(len(train_set), args.max_steps, args.seed + 301)
    probe_schedule = cyclic_schedule(
        len(unrelated_train), args.max_steps, args.seed + 302
    )

    def translated(rk, rv, item):
        content = rk(item["Xk"])
        keys = [value.to(dtype) for value in to_stored_keys(content, tg, args.ctx)]
        values = [value.to(dtype) for value in rv(item["Xv"])]
        return content, keys, values

    def train_arm(name):
        rk, rv = make_residual_pair()
        optimizer = torch.optim.Adam(
            list(rk.blocks.parameters()) + list(rv.blocks.parameters()), lr=args.lr
        )
        losses = []
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        began = time.perf_counter()
        for step, item_index in enumerate(schedule):
            item = train_set[item_index]
            content, keys, values = translated(rk, rv, item)
            if name == "tensor":
                loss = kv_error_loss(content, item["Yk"]) + kv_error_loss(
                    values, item["Yv"]
                )
            else:
                probe = (
                    item["relevant"]
                    if name == "relevant"
                    else unrelated_train[probe_schedule[step]]
                )
                with torch.no_grad():
                    reference = continuation_logits(
                        models["target"],
                        item["native_keys"],
                        item["native_values"],
                        probe,
                        args.ctx,
                    ).float()
                    reference_lp = torch.log_softmax(reference, -1)
                logits = continuation_logits_grad(
                    models["target"], keys, values, probe, args.ctx
                )
                loss = behaviour_loss(reference_lp, logits)
                del reference, reference_lp, logits
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            del content, keys, values, loss
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - began
            if elapsed >= args.arm_seconds or step + 1 >= args.max_steps:
                break
        final_state = module_state(rk, rv)
        report = {
            "arm": name,
            "updates": len(losses),
            "budget_seconds": args.arm_seconds,
            "train_seconds": time.perf_counter() - began,
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "first_10_mean": float(np.mean(losses[:10])),
            "last_10_mean": float(np.mean(losses[-10:])),
            "initial_digest": init_digest,
            "final_digest": tensor_digest(final_state),
            "parameters_changed": tensor_digest(final_state) != init_digest,
            "peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
            ),
            "loss_trace": losses,
        }
        log(
            f"{name}: {report['updates']} updates in {report['train_seconds']:.1f}s, "
            f"loss {report['first_loss']:.5f}->{report['last_loss']:.5f}"
        )
        return (rk.eval(), rv.eval()), report

    trained, training = {}, {}
    for arm_name in ("tensor", "relevant", "unrelated"):
        trained[arm_name], training[arm_name] = train_arm(arm_name)

    def apply_arm(name, item):
        if name == "affine":
            content = affine["k"].apply(item["Xk"])
            values = affine["v"].apply(item["Xv"])
        else:
            rk, rv = trained[name]
            with torch.no_grad():
                content, values = rk(item["Xk"]), rv(item["Xv"])
        keys = [value.to(dtype) for value in to_stored_keys(content, tg, args.ctx)]
        return keys, [value.to(dtype) for value in values]

    diagnostic_rows = []
    for item in validation_set:
        probes = [("relevant", "document", item["relevant"])] + [
            ("unrelated", f"held_{index}", probe)
            for index, probe in enumerate(unrelated_held)
        ]
        for probe_type, probe_id, probe in probes:
            reference = continuation_logits(
                models["target"],
                item["native_keys"],
                item["native_values"],
                probe,
                args.ctx,
            ).float()
            for arm_name in ("affine", "tensor", "relevant", "unrelated"):
                keys, values = apply_arm(arm_name, item)
                candidate = continuation_logits(
                    models["target"], keys, values, probe, args.ctx
                ).float()
                diagnostic_rows.append(
                    {
                        "doc": item["doc"],
                        "probe_type": probe_type,
                        "probe_id": probe_id,
                        "arm": arm_name,
                        "kl": float(teacher_forced_kl(reference, candidate).mean()),
                    }
                )
                del keys, values, candidate
            del reference
    diagnostic_summary = {}
    for probe_type in ("relevant", "unrelated"):
        diagnostic_summary[probe_type] = {}
        for arm_name in ("affine", "tensor", "relevant", "unrelated"):
            values = [
                row["kl"]
                for row in diagnostic_rows
                if row["probe_type"] == probe_type and row["arm"] == arm_name
            ]
            diagnostic_summary[probe_type][arm_name] = {
                "mean_kl": float(np.mean(values)),
                "median_kl": float(np.median(values)),
                "n": len(values),
            }
    log(f"held-out KL: {json.dumps(diagnostic_summary, sort_keys=True)}")

    # Rebuild source-side features for the planted, wholly unseen task prompts.
    task_source = {}
    for doc, prompt in task_prompts.items():
        sp = prefill(models["source"], prompt, sg)
        task_source[doc] = {
            "Xk": flat_features(to_content_keys(sp)).float(),
            "Xv": flat_features([value.float() for value in sp.values]).float(),
        }
    translated_task = {
        arm_name: {
            doc: apply_arm(arm_name, task_source[doc]) for doc in task_docs
        }
        for arm_name in ("affine", "tensor", "relevant", "unrelated")
    }
    noisy_task = {
        doc: noisy_cache(native_task[doc], index)
        for index, doc in enumerate(task_docs)
    }

    task_rows = []
    identity_failures = []
    translator_arms = ("affine", "tensor", "relevant", "unrelated")
    for item_index, item in enumerate(task_items):
        query = tok(
            item.query, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        native = native_task[item.doc_id]
        conditions = {
            "target_native": (native.keys_post, native.values),
            "native_copy_identity": (
                [tensor.clone() for tensor in native.keys_post],
                [tensor.clone() for tensor in native.values],
            ),
            "no_cache": (None, None),
            "neutral_cache": (neutral.keys_post, neutral.values),
            "noisy_native": noisy_task[item.doc_id],
        }
        for arm_name in translator_arms:
            conditions[arm_name] = translated_task[arm_name][item.doc_id]
            if item.kind != "free":
                conditions[f"{arm_name}|wrong"] = translated_task[arm_name][
                    shift[item.doc_id]
                ]
        native_tokens = None
        for condition, (keys, values) in conditions.items():
            cap = args.free_max_new if item.kind == "free" else args.task_max_new
            prompt_len = 0 if keys is None else args.ctx
            tokens = greedy(
                models["target"], keys, values, query, prompt_len, cap, eos_ids
            )
            if condition == "target_native":
                native_tokens = tokens
            elif condition == "native_copy_identity" and tokens != native_tokens:
                identity_failures.append(
                    {"doc": item.doc_id, "kind": item.kind, "item": item_index}
                )
            text = tok.decode(tokens, skip_special_tokens=True)
            row = {
                "doc": item.doc_id,
                "kind": item.kind,
                "condition": condition,
                "produced": text,
                "tokens": tokens,
            }
            if item.kind == "retrieval":
                row.update(grade_retrieval(item.answer, text))
                row["primary"] = row["correct"]
            elif item.kind == "cloze":
                row.update(grade_cloze(item.answer, text))
                row["primary"] = row["overlap"]
            else:
                row.update(grade_free(text, tokens, eos_ids, cap))
                row["primary"] = row["healthy"]
            task_rows.append(row)
        if (item_index + 1) % 8 == 0:
            log(f"  generated {item_index + 1}/{len(task_items)} task items")

    task_summary = summarize_task_rows(task_rows)
    with open(os.path.join(args.out_dir, "diagnostic_rows.jsonl"), "w") as output:
        for row in diagnostic_rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    with open(os.path.join(args.out_dir, "task_rows.jsonl"), "w") as output:
        for row in task_rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")

    mechanics = mechanics_decision(training, diagnostic_summary, identity_failures)
    mechanics_pass = mechanics["passes"]
    result = {
        "schema": "unrelated_probe_training_qualification_v1",
        "status": "QUALIFIED_MECHANICS" if mechanics_pass else "MECHANICS_FAIL",
        "scope": (
            "One-seed qualification only. No arm is promoted, and this is not "
            "a restricted-channel or subliminal-learning result."
        ),
        "source_digest": sha256_file(source_path),
        "signal_result_sha256": sha256_file(args.signal_result),
        "source": {"model": args.source, "revision": args.source_revision},
        "target": {"model": args.target, "revision": args.target_revision},
        "compatibility": compatibility,
        "config": vars(args),
        "document_roles": role_names,
        "unrelated_probe_sha256": {
            "training": unrelated_hashes[:12],
            "held_out": unrelated_hashes[12:],
        },
        "preflight": {
            "accuracy": preflight_accuracy,
            "task_headroom": task_headroom,
            "headroom_min": args.headroom_min,
            "identity_exact": identity_exact,
        },
        "affine": {
            "calibration_seconds": calibration_seconds,
            "layer_budget": args.k,
            "ridge": args.ridge,
            "bytes": affine["k"].n_bytes + affine["v"].n_bytes,
        },
        "initialization": {
            "sha256": init_digest,
            "path": os.path.basename(init_path),
        },
        "training": training,
        "held_out_diagnostics": diagnostic_summary,
        "tasks": task_summary,
        "identity_failures": identity_failures,
        "mechanics": mechanics,
        "device_type": torch.device(device).type,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch": torch.__version__,
        "wall_seconds": time.time() - started,
    }
    with open(os.path.join(args.out_dir, "RESULT.json"), "w") as output:
        json.dump(result, output, indent=2, sort_keys=True)
    log(
        f"qualification {'PASS' if mechanics_pass else 'FAIL'}; task summary "
        f"{json.dumps(task_summary, sort_keys=True)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

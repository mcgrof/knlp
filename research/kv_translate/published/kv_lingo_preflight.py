#!/usr/bin/env python3
"""Measure the true frozen-receiver backward for KV-Lingo's 4B/8B pair."""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import resource
import time

import torch

from .capture import cache_layers, make_cache
from .kv_lingo import (
    LinearTranslator,
    capture_pre_norm_span,
    forward_kl,
    freeze_model,
    geometry,
    span_bytes,
)
from .coqa_runner import write_json

GIB = float(2**30)
PINS = {
    "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
    "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
}


def log(message):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {message}", flush=True)


def deterministic_ids(tokenizer, tokens: int, device) -> torch.Tensor:
    phrase = (
        "A cache handoff must preserve the exact token ledger while the frozen "
        "receiver predicts a continuation from translated history. "
    )
    base = tokenizer(phrase, add_special_tokens=False, return_tensors="pt").input_ids[0]
    repeats = (tokens + len(base) - 1) // len(base)
    return base.repeat(repeats)[:tokens].unsqueeze(0).to(device)


@torch.no_grad()
def native_teacher(target, prefix_without_last, scored_tokens):
    # The base model produces the native cache without allocating prefix
    # logits.  Only the short continuation needs the language-model head.
    output = target.model(input_ids=prefix_without_last, use_cache=True)
    pairs = cache_layers(output.past_key_values)
    keys = [pair[0] for pair in pairs]
    values = [pair[1] for pair in pairs]
    logits = continuation_logits(
        target, keys, values, scored_tokens, int(prefix_without_last.shape[1])
    )
    return torch.log_softmax(logits.float(), dim=-1).to(torch.float16)


def continuation_logits(target, keys, values, scored_tokens, prompt_tokens):
    """Differentiable teacher-forced logits on a supplied cache."""

    total = prompt_tokens + scored_tokens.shape[1]
    attention_mask = torch.ones(1, total, dtype=torch.long, device=scored_tokens.device)
    position_ids = torch.arange(
        prompt_tokens, total, dtype=torch.long, device=scored_tokens.device
    ).unsqueeze(0)
    return target(
        input_ids=scored_tokens,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=make_cache(list(zip(keys, values, strict=True))),
        use_cache=True,
    ).logits


def parse_cases(value: str):
    cases = []
    for cell in value.split(","):
        context, continuation = cell.split("x", 1)
        cases.append((int(context), int(continuation)))
    return cases


def run_case(source, target, tokenizer, context, continuation, seed):
    device = next(target.parameters()).device
    torch.manual_seed(seed)
    ids = deterministic_ids(tokenizer, context + continuation, device)[0]
    prefix = ids[:context].unsqueeze(0)
    continuation_ids = ids[context:].unsqueeze(0)
    prefix_without_last = prefix[:, :-1]
    scored_tokens = torch.cat((prefix[:, -1:], continuation_ids[:, :-1]), dim=1)

    translator = LinearTranslator(geometry(source), geometry(target)).to(device)
    optimizer = torch.optim.AdamW(translator.parameters(), lr=3e-5, weight_decay=0.0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    allocated_start = torch.cuda.memory_allocated()
    timings = {}
    started = time.perf_counter()

    before = torch.stack(
        [
            parameter.detach().float().square().sum()
            for parameter in translator.parameters()
        ]
    )
    t0 = time.perf_counter()
    source_span = capture_pre_norm_span(source, prefix_without_last)
    torch.cuda.synchronize()
    timings["source_capture"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    with torch.no_grad():
        teacher = native_teacher(target, prefix_without_last, scored_tokens)
    torch.cuda.synchronize()
    timings["native_teacher"] = time.perf_counter() - t0

    optimizer.zero_grad(set_to_none=True)
    positions = torch.arange(context - 1, device=device, dtype=torch.long).unsqueeze(0)
    t0 = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        keys, values = translator(source_span, target, positions)
        student = continuation_logits(target, keys, values, scored_tokens, context - 1)
        loss = forward_kl(teacher, student)
    torch.cuda.synchronize()
    timings["translated_forward"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(translator.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    timings["backward_and_step"] = time.perf_counter() - t0
    timings["total"] = time.perf_counter() - started

    after = torch.stack(
        [
            parameter.detach().float().square().sum()
            for parameter in translator.parameters()
        ]
    )
    changed = not torch.equal(before, after)
    receipt = {
        "context_tokens": context,
        "continuation_tokens": continuation,
        "source_span_tokens": source_span.tokens,
        "source_span_gib": span_bytes(source_span) / GIB,
        "trainable_parameters": translator.trainable_parameters,
        "parameter_gib_fp32": translator.parameter_bytes / GIB,
        "loss": float(loss.detach()),
        "loss_finite": bool(torch.isfinite(loss.detach())),
        "gradient_norm": float(gradient_norm.detach()),
        "gradient_finite": bool(torch.isfinite(gradient_norm.detach())),
        "parameters_changed": changed,
        "timings_seconds": timings,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
        "incremental_peak_over_loaded_models_gib": (
            torch.cuda.max_memory_allocated() - allocated_start
        )
        / GIB,
    }
    receipt["passed"] = all(
        (
            receipt["loss_finite"],
            receipt["gradient_finite"],
            receipt["parameters_changed"],
        )
    )
    del (
        translator,
        optimizer,
        before,
        after,
        source_span,
        teacher,
        keys,
        values,
        student,
        loss,
    )
    gc.collect()
    torch.cuda.empty_cache()
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--direction", choices=("4b-to-8b", "8b-to-4b"), default="4b-to-8b"
    )
    parser.add_argument("--cases", default="256x64,2048x256,16384x2048")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    source_name, target_name = (
        ("Qwen/Qwen3-4B", "Qwen/Qwen3-8B")
        if args.direction == "4b-to-8b"
        else ("Qwen/Qwen3-8B", "Qwen/Qwen3-4B")
    )
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(
        target_name, revision=PINS[target_name], local_files_only=True
    )
    models = {}
    for role, name in (("source", source_name), ("target", target_name)):
        log(f"loading {role} {name}@{PINS[name]}")
        model = AutoModelForCausalLM.from_pretrained(
            name,
            revision=PINS[name],
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": device},
        )
        freeze_model(model)
        models[role] = model
    source, target = models["source"], models["target"]
    loaded_models_gib = torch.cuda.memory_allocated() / GIB
    rows = []
    for context, continuation in parse_cases(args.cases):
        log(f"preflight context={context} continuation={continuation}")
        try:
            row = run_case(source, target, tokenizer, context, continuation, args.seed)
        except torch.cuda.OutOfMemoryError as error:
            row = {
                "context_tokens": context,
                "continuation_tokens": continuation,
                "passed": False,
                "oom": True,
                "error": str(error)[:400],
            }
            rows.append(row)
            log(f"OOM at context={context} continuation={continuation}")
            break
        rows.append(row)
        log(
            f"case {'PASS' if row['passed'] else 'FAIL'}: "
            f"{row['peak_allocated_gib']:.2f} GiB peak, "
            f"{row['timings_seconds']['total']:.2f}s"
        )

    output = {
        "schema": "kv_lingo_true_backward_preflight_v1",
        "method_identity": "independent paper-faithful mechanics; no author code/checkpoint available",
        "direction": args.direction,
        "source": {"model": source_name, "revision": PINS[source_name]},
        "target": {"model": target_name, "revision": PINS[target_name]},
        "capture_point": "pre-k-norm keys; v_proj values",
        "translator": "head-mixing, split K/V, one bias-free linear map per target layer",
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 3e-5,
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
        },
        "loaded_models_gib": loaded_models_gib,
        "gpu": torch.cuda.get_device_name(),
        "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory / GIB,
        "torch": torch.__version__,
        "python": os.sys.version,
        "rows": rows,
        "peak_host_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "passed": len(rows) == len(parse_cases(args.cases))
        and all(row["passed"] for row in rows),
    }
    write_json(args.out, output)
    log(f"wrote {args.out}; terminal={'PASS' if output['passed'] else 'FAIL'}")
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

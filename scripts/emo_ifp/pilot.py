#!/usr/bin/env python3
"""Phase 0 GPU pilot for the BF16 EMO reference.

Loads the verified BF16 copy on one GPU and records: load time and memory,
a spot check of resident weights against the conversion manifest, the
tokenizer's special-token behavior, the routing rule actually executed
(seven routed plus one shared expert per token), greedy outputs for eight
short prompts with a bitwise replay, prefill and decode rates after a
warm-up, and the cost of the reference implementation's per-expert loop.

Usage:
    pilot.py MODEL_DIR BF16_MANIFEST.json OUT_DIR
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa
    expert_tensor_name,
    load_model,
    mem_snapshot,
    runtime_identity,
    tensor_sha256,
    write_json,
)

# Eight short pilot prompts. The model is a base model, so prompts are plain
# text with no chat template.
PILOT_PROMPTS = [
    "The capital of France is",
    "Photosynthesis is the process by which plants",
    "def fibonacci(n):\n    ",
    "Question: Which gas do plants absorb from the air for photosynthesis?\nAnswer:",
    "Question: What is 12 times 7?\nAnswer:",
    "The three primary colors of light are",
    "In 1969, the Apollo 11 mission",
    "A mixture of experts model routes each token to",
]

# Natural filler text for timing prompts of exact token lengths. Disclosed
# as a repeated-text timing fixture; it is not a quality input.
FILLER = (
    "Mixture of experts language models route each token through a small "
    "subset of feed-forward experts selected by a learned router. Because "
    "only a few experts run per token, the active compute is much smaller "
    "than the total parameter count, but every expert must still be stored "
    "somewhere the accelerator can reach when the router selects it. "
)


def sync():
    torch.cuda.synchronize()


def timing_ids(tok, n):
    ids = tok(FILLER * (n // 40 + 2), add_special_tokens=False)["input_ids"]
    assert len(ids) >= n
    return torch.tensor([ids[:n]], device="cuda")


class RoutingAudit:
    """Count the tokens each expert actually processes in every layer.

    With seven routed experts plus one shared expert per token, the experts
    of a layer together process eight rows per token, and the shared
    expert (the last index) processes every token.
    """

    def __init__(self, model):
        self.rows = []
        self.handles = []
        for li, layer in enumerate(model.model.layers):
            self.rows.append([0] * len(layer.mlp.experts))
            for ei, exp in enumerate(layer.mlp.experts):
                self.handles.append(exp.register_forward_hook(self._mk(li, ei)))

    def _mk(self, li, ei):
        def hook(mod, inp, out):
            self.rows[li][ei] += int(inp[0].shape[0])

        return hook

    def remove(self):
        for h in self.handles:
            h.remove()

    def summary(self, tokens):
        per = []
        for li, r in enumerate(self.rows):
            per.append(
                {
                    "layer": li,
                    "rows_total": sum(r),
                    "rows_per_token": sum(r) / tokens,
                    "shared_rows": r[-1],
                    "experts_used": sum(1 for x in r if x),
                }
            )
        return per


@torch.no_grad()
def greedy(model, tok, prompt, new_tokens):
    enc = tok(prompt, return_tensors="pt", return_token_type_ids=False).to("cuda")
    out = model.generate(
        **enc,
        max_new_tokens=new_tokens,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tok.pad_token_id,
    )
    seq = out.sequences[0, enc["input_ids"].shape[1] :].tolist()
    scores = torch.stack([s[0].float() for s in out.scores])  # (new, vocab)
    return enc["input_ids"][0].tolist(), seq, scores


@torch.no_grad()
def time_prefill(model, ids, reps):
    ts = []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        model(input_ids=ids, use_cache=True)
        sync()
        ts.append(time.perf_counter() - t0)
    return ts


@torch.no_grad()
def time_decode(model, ids, steps):
    """Prefill once, then time ``steps`` single-token greedy decode steps."""
    out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[:, -1:].argmax(-1)
    sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        out = model(input_ids=nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1:].argmax(-1)
    sync()
    return (time.perf_counter() - t0) / steps


@torch.no_grad()
def expert_loop_overhead(model, reps=20):
    """Compare one MoE block on a single token with its eight active MLPs."""
    moe = model.model.layers[8].mlp
    x = torch.randn(1, 1, model.config.hidden_size, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        moe(x)
    sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        moe(x)
    sync()
    block = (time.perf_counter() - t0) / reps
    flat = x.view(1, -1)
    logits = moe.gate(flat).float()
    n, ns = moe.num_experts, moe.num_shared_experts
    _, sel = torch.topk(torch.softmax(logits[:, : n - ns], 1), moe.top_k - ns)
    active = sel[0].tolist() + list(range(n - ns, n))
    sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        for e in active:
            moe.experts[e](flat)
    sync()
    direct = (time.perf_counter() - t0) / reps
    return {
        "moe_block_seconds_1tok": block,
        "active_mlps_only_seconds_1tok": direct,
        "loop_overhead_fraction": (block - direct) / block,
        "layer": 8,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("manifest")
    ap.add_argument("out_dir")
    ap.add_argument("--new-tokens", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    res = {"runtime": runtime_identity(), "model_dir": os.path.abspath(args.model_dir)}

    torch.cuda.reset_peak_memory_stats()
    m0 = mem_snapshot()
    t0 = time.perf_counter()
    model, tok = load_model(args.model_dir)
    sync()
    res["load_seconds"] = time.perf_counter() - t0
    m1 = mem_snapshot()
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buf_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    res["memory"] = {
        "before_load": m0,
        "after_load": m1,
        "param_bytes": param_bytes,
        "buffer_bytes": buf_bytes,
        "load_transient_bytes": m1["max_allocated"] - m1["allocated"],
    }
    dtypes = sorted(set(str(p.dtype) for p in model.parameters()))
    res["param_dtypes"] = dtypes

    # Spot check resident weights against the conversion manifest.
    with open(args.manifest) as f:
        man = json.load(f)["tensors"]
    sd = model.state_dict()
    names = [
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.layers.0.mlp.gate.weight",
        expert_tensor_name(0, 0, "gate_proj"),
        expert_tensor_name(15, 127, "down_proj"),
        expert_tensor_name(15, 63, "up_proj"),
        "model.layers.15.self_attn.q_proj.weight",
    ]
    res["weight_spot_check"] = {
        n: tensor_sha256(sd[n]) == man[n]["bf16_sha256"] for n in names
    }

    # Tokenizer and special tokens.
    enc = tok("Hello", add_special_tokens=True)["input_ids"]
    res["tokenizer"] = {
        "class": type(tok).__name__,
        "bos_token": tok.bos_token,
        "eos_token": tok.eos_token,
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.pad_token_id,
        "config_eos_token_id": model.config.eos_token_id,
        "generation_eos_token_id": model.generation_config.eos_token_id,
        "hello_ids_with_special": enc,
        "hello_ids_plain": tok("Hello", add_special_tokens=False)["input_ids"],
        "chat_template_used": False,
    }

    # Routing rule actually executed.
    audit = RoutingAudit(model)
    rin = tok(
        "The router picks experts for every token in this sentence.",
        return_tensors="pt",
        return_token_type_ids=False,
    ).to("cuda")
    with torch.no_grad():
        model(**rin)
    audit.remove()
    ntok = int(rin["input_ids"].shape[1])
    per = audit.summary(ntok)
    res["routing_audit"] = {"tokens": ntok, "layers": per}
    res["routing_seven_plus_shared"] = all(
        x["rows_per_token"] == 8 and x["shared_rows"] == ntok for x in per
    )

    # Greedy outputs with bitwise replay.
    gens = []
    replay_ok = True
    for p in PILOT_PROMPTS:
        pid, s1, sc1 = greedy(model, tok, p, args.new_tokens)
        _, s2, sc2 = greedy(model, tok, p, args.new_tokens)
        same_tok = s1 == s2
        same_logits = torch.equal(sc1, sc2)
        replay_ok &= same_tok and same_logits
        eos = tok.eos_token_id
        gens.append(
            {
                "prompt": p,
                "prompt_tokens": len(pid),
                "output_ids": s1,
                "output_text": tok.decode(s1),
                "stopped_at_eos": eos in s1,
                "replay_same_tokens": same_tok,
                "replay_same_scores": same_logits,
                "first_step_top5": torch.topk(sc1[0], 5).indices.tolist(),
                "first_step_max_logit": float(sc1[0].max()),
            }
        )
    res["generations"] = gens
    res["replay_bitwise_identical"] = replay_ok

    # Prefill and decode timing after warm-up.
    timing = {}
    for n in (128, 512, 1024):
        ids = timing_ids(tok, n)
        time_prefill(model, ids, 2)  # warm-up
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        ts = time_prefill(model, ids, 5)
        peak = torch.cuda.max_memory_allocated() - base
        dec = time_decode(model, ids, 32)
        timing[str(n)] = {
            "prefill_seconds": ts,
            "prefill_median_s": statistics.median(ts),
            "prefill_tok_per_s": n / statistics.median(ts),
            "prefill_peak_extra_bytes": peak,
            "decode_seconds_per_token": dec,
            "decode_tok_per_s": 1.0 / dec,
        }
        print(n, timing[str(n)]["prefill_tok_per_s"], 1.0 / dec, flush=True)
    res["timing"] = timing
    res["expert_loop"] = expert_loop_overhead(model)
    res["memory"]["final"] = mem_snapshot()
    write_json(os.path.join(args.out_dir, "pilot.json"), res)
    print(
        json.dumps(
            {
                k: res[k]
                for k in (
                    "load_seconds",
                    "param_dtypes",
                    "replay_bitwise_identical",
                    "routing_seven_plus_shared",
                    "weight_spot_check",
                    "expert_loop",
                )
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()

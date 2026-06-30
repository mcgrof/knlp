#!/usr/bin/env python3
"""Train a cartridge (KV cache state) from skill text content.

Usage:
    python train_cartridge.py \
        --model /workspace/models/Qwen2.5-7B-Instruct \
        --skill-text /path/to/SKILL.md \
        --output-dir /workspace/cartridges/citation-check/default-50pct \
        --budget-pct 50

Produces:
    cartridge.pt          - KV cache state
    prefix_token_ids.json - token IDs used to initialize the saved KV prefix
    meta.json             - training metadata

init_method options:
    default, first_k
        First-k initialization: run a normal forward pass on the first p
        tokens of the skill text and save the resulting K/V prefix.

    sci, sci_chunk
        Paper-style Sampled Chunk Initialization: sample contiguous chunks
        from the tokenized skill text, concatenate those chunks into a
        synthetic initializer sequence, and run a normal forward pass on
        that sequence. This does not extract scattered K/V positions and
        does not need RoPE rephasing.

    strided_rope
        Legacy KNLP initializer formerly called "sci": run the full skill
        text, pick uniformly spaced K/V positions, then rephase K vectors
        from their original RoPE positions to consecutive prefix positions.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def token_ids_sha256(token_ids: list[int]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def to_legacy_cache(past_key_values):
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    return past_key_values


def normalize_init_method(init_method: str) -> str:
    aliases = {
        "default": "first_k",
        "first-k": "first_k",
        "first_k": "first_k",
        "sci": "sci_chunk",
        "sci_chunk": "sci_chunk",
        "sci_chunk_64": "sci_chunk",
        "strided": "strided_rope",
        "strided_kv": "strided_rope",
        "legacy_sci": "strided_rope",
        "strided_rope": "strided_rope",
    }
    try:
        return aliases[init_method]
    except KeyError as exc:
        raise ValueError(f"Unknown init method: {init_method}") from exc


def build_sampled_chunk_init_tokens(
    corpus_token_ids: torch.Tensor,
    budget_tokens: int,
    chunk_size: int = 64,
    seed: int = 0,
) -> tuple[torch.Tensor, list[int], int]:
    """Build paper-style SCI initializer tokens.

    This follows Sampled Chunk Initialization: sample contiguous chunks from
    the tokenized corpus, concatenate them, and forward-pass that synthetic
    token sequence as positions 0..p-1. The implementation intentionally does
    not sample individual K/V positions from a full-document cache.
    """
    x = corpus_token_ids.flatten().cpu()
    n_total = x.numel()
    if budget_tokens <= 0:
        raise ValueError("budget_tokens must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if budget_tokens % chunk_size != 0:
        raise ValueError(
            f"paper-style SCI needs budget_tokens divisible by chunk_size: "
            f"budget_tokens={budget_tokens}, chunk_size={chunk_size}. Use "
            "--budget-tokens with a multiple of the chunk size for exact SCI."
        )
    if n_total < chunk_size:
        reps = (budget_tokens + n_total - 1) // n_total
        x_init = x.repeat(reps)[:budget_tokens].contiguous()
        return x_init, [0] * (budget_tokens // chunk_size), n_total

    n_chunks = budget_tokens // chunk_size
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    starts = torch.randint(
        low=0,
        high=n_total - chunk_size + 1,
        size=(n_chunks,),
        generator=gen,
        device="cpu",
    ).tolist()
    chunks = [x[start : start + chunk_size] for start in starts]
    x_init = torch.cat(chunks, dim=0)[:budget_tokens].contiguous()
    return x_init, starts, n_total


def rotate_half(x):
    """GPT-NeoX/Qwen-style RoPE pair rotation."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def get_cos_sin_table(model, max_pos: int):
    inv_freq = model.model.rotary_emb.inv_freq.detach().float().cpu()
    positions = torch.arange(max_pos, dtype=torch.float32)
    freqs = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def reroute_rope_phase(k, orig_positions, new_positions, cos_table, sin_table):
    """Rephase K vectors from original RoPE positions to new prefix positions."""
    cos_orig = (
        cos_table[orig_positions].unsqueeze(0).unsqueeze(0).to(k.device).to(k.dtype)
    )
    sin_orig = (
        sin_table[orig_positions].unsqueeze(0).unsqueeze(0).to(k.device).to(k.dtype)
    )
    cos_new = (
        cos_table[new_positions].unsqueeze(0).unsqueeze(0).to(k.device).to(k.dtype)
    )
    sin_new = (
        sin_table[new_positions].unsqueeze(0).unsqueeze(0).to(k.device).to(k.dtype)
    )

    k_unrot = k * cos_orig - rotate_half(k) * sin_orig
    return k_unrot * cos_new + rotate_half(k_unrot) * sin_new


def materialize_forward_cache(model, input_ids: torch.Tensor):
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    return to_legacy_cache(outputs.past_key_values)


def train_cartridge(
    model_path: str,
    skill_text_path: str,
    output_dir: str,
    budget_pct: int = 50,
    budget_tokens: int | None = None,
    init_method: str = "default",
    sci_chunk_size: int = 64,
    sci_seed: int = 0,
    dtype: str = "float16",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_init_method = normalize_init_method(init_method)

    # Load tokenizer and model
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=DTYPES[dtype],
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Tokenize skill text
    skill_text = Path(skill_text_path).read_text()
    inputs = tokenizer(skill_text, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"].to(model_device(model))
    n_tokens = input_ids.shape[1]
    print(f"Skill text: {len(skill_text)} chars, {n_tokens} tokens")

    # Compute budget
    requested_budget_tokens = budget_tokens
    if requested_budget_tokens is None:
        requested_budget_tokens = int(n_tokens * budget_pct / 100)
    requested_budget_tokens = min(requested_budget_tokens, n_tokens)
    if requested_budget_tokens <= 0:
        raise ValueError("budget resolves to zero tokens")
    print(
        f"Budget: {budget_pct}% request = {requested_budget_tokens} tokens "
        f"(of {n_tokens})"
    )

    t0 = time.time()
    init_metadata = {
        "init_method_requested": init_method,
        "init_method_canonical": canonical_init_method,
        "requires_force_inject": canonical_init_method != "first_k",
    }

    if canonical_init_method == "first_k":
        init_ids = input_ids[:, :requested_budget_tokens]
        kv_cache = materialize_forward_cache(model, init_ids)
        cartridge_kv = []
        for layer_kv in kv_cache:
            k, v = layer_kv[0], layer_kv[1]
            cartridge_kv.append((k.cpu(), v.cpu()))
        prefix_ids = init_ids[0].cpu().tolist()

    elif canonical_init_method == "sci_chunk":
        x_init, starts, n_total = build_sampled_chunk_init_tokens(
            input_ids[0],
            requested_budget_tokens,
            chunk_size=sci_chunk_size,
            seed=sci_seed,
        )
        init_ids = x_init.unsqueeze(0).to(model_device(model))
        kv_cache = materialize_forward_cache(model, init_ids)
        cartridge_kv = []
        for layer_kv in kv_cache:
            k, v = layer_kv[0], layer_kv[1]
            cartridge_kv.append((k.cpu(), v.cpu()))
        prefix_ids = x_init.tolist()
        init_metadata.update(
            {
                "sci_chunk_size": sci_chunk_size,
                "sci_seed": sci_seed,
                "sci_n_total_tokens": n_total,
                "sci_n_chunks": len(starts),
                "sci_sample_starts": starts,
                "sci_sampling_replacement": True,
                "sci_sort_starts": False,
                "sci_inserted_separators": False,
                "sci_exact_budget_multiple": True,
            }
        )

    elif canonical_init_method == "strided_rope":
        kv_cache = materialize_forward_cache(model, input_ids)
        orig_positions = torch.linspace(0, n_tokens - 1, requested_budget_tokens).long()
        new_positions = torch.arange(requested_budget_tokens, dtype=torch.long)
        max_pos = int(orig_positions.max().item()) + 1
        cos_table, sin_table = get_cos_sin_table(model, max_pos)

        cartridge_kv = []
        for layer_kv in kv_cache:
            k, v = layer_kv[0], layer_kv[1]
            k_sel = k[:, :, orig_positions, :]
            v_sel = v[:, :, orig_positions, :]
            k_rephased = reroute_rope_phase(
                k_sel,
                orig_positions,
                new_positions,
                cos_table,
                sin_table,
            )
            cartridge_kv.append((k_rephased.cpu(), v_sel.cpu()))
        prefix_ids = input_ids[0, orig_positions].cpu().tolist()
        init_metadata.update(
            {
                "legacy_name": "sci",
                "strided_orig_positions_first10": orig_positions[:10].tolist(),
                "strided_orig_positions_last10": orig_positions[-10:].tolist(),
                "strided_stride_approx": (n_tokens - 1)
                / max(requested_budget_tokens - 1, 1),
            }
        )

    else:
        raise ValueError(f"Unknown canonical init method: {canonical_init_method}")

    fwd_time = time.time() - t0
    actual_budget_tokens = cartridge_kv[0][0].shape[2]
    print(f"Forward/init pass: {fwd_time:.2f}s")
    print(f"Actual cartridge length: {actual_budget_tokens} tokens")

    # Save cartridge
    cartridge_path = output_dir / "cartridge.pt"
    torch.save(cartridge_kv, cartridge_path)
    cart_size = cartridge_path.stat().st_size
    print(f"Cartridge saved: {cartridge_path} ({cart_size / 1024 / 1024:.1f} MB)")

    # Save prefix token IDs
    prefix_path = output_dir / "prefix_token_ids.json"
    with open(prefix_path, "w") as f:
        json.dump(prefix_ids, f)
    print(f"Prefix token IDs saved: {len(prefix_ids)} tokens")

    # Save metadata
    meta = {
        "model": model_path,
        "dtype": dtype,
        "skill_text": skill_text_path,
        "init_method": init_method,
        "init_method_canonical": canonical_init_method,
        "budget_pct": budget_pct,
        "budget_tokens_requested": requested_budget_tokens,
        "original_tokens": n_tokens,
        "budget_tokens": actual_budget_tokens,
        "cartridge_size_bytes": cart_size,
        "n_layers": len(cartridge_kv),
        "prefix_token_ids_sha256": token_ids_sha256(prefix_ids),
        "fwd_time_sec": round(fwd_time, 3),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **init_metadata,
    }
    meta_path = output_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata: {json.dumps(meta, indent=2)}")

    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--skill-text", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budget-pct", type=int, default=50)
    parser.add_argument("--budget-tokens", type=int, default=None)
    parser.add_argument(
        "--init-method",
        choices=[
            "default",
            "first-k",
            "first_k",
            "sci",
            "sci_chunk",
            "sci_chunk_64",
            "strided",
            "strided_kv",
            "legacy_sci",
            "strided_rope",
        ],
        default="default",
    )
    parser.add_argument("--sci-chunk-size", type=int, default=64)
    parser.add_argument("--sci-seed", type=int, default=0)
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPES),
        default="float16",
        help="Model/cache dtype used for the initializer forward pass.",
    )
    args = parser.parse_args()

    train_cartridge(
        model_path=args.model,
        skill_text_path=args.skill_text,
        output_dir=args.output_dir,
        budget_pct=args.budget_pct,
        budget_tokens=args.budget_tokens,
        init_method=args.init_method,
        sci_chunk_size=args.sci_chunk_size,
        sci_seed=args.sci_seed,
        dtype=args.dtype,
    )

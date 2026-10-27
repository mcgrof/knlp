# SPDX-License-Identifier: MIT
"""Parity harness between the native cartridge reader and this reader.

The native path is the HazyResearch ``cartridges`` library: its own Qwen3
implementation with FlexAttention and a ``TrainableCache`` that holds the
cartridge rows in front of the request. This module runs matched prompts
through that path and through the strict artifact path of this package and
stores last-position logits, greedy continuations and teacher-forced
per-step logits so a separate comparison can report the numerical envelope
between the two stacks and between two kernels of the same stack.

The library is imported lazily; the comparison needs only the saved files.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import artifact as A
from . import cas_bridge as B
from . import longhealth as LH
from . import model as M
from . import runs as R


def _require_library() -> None:
    try:
        import cartridges  # noqa: F401
    except ImportError as error:  # pragma: no cover - needs the library
        raise RuntimeError(
            "the native cartridges library is not importable; add its "
            "checkout to PYTHONPATH and set CARTRIDGES_DIR"
        ) from error


def load_native_model(model_name: str, device: str) -> torch.nn.Module:
    _require_library()
    os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", str(Path.cwd() / "cas_out"))
    from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM

    # the native evaluator loads FP32 weights and casts on the device; on a
    # 24 GiB card the 8B model must arrive in BF16 (the same stored dtype)
    model = FlexQwen3ForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def native_cache(model: torch.nn.Module, cart: B.NativeCartridge, device: str) -> Any:
    """The native cache exactly as the native evaluator builds it: sink and
    trainable rows concatenated with no frozen split."""
    from cartridges.cache import AttnConfig, TrainableCache

    config = AttnConfig(
        n_layers=model.config.num_hidden_layers,
        n_heads=model.config.num_key_value_heads,
        head_dim=getattr(
            model.config,
            "head_dim",
            model.config.hidden_size // model.config.num_attention_heads,
        ),
    )
    if cart.layers != config.n_layers:
        raise ValueError("cartridge layer count does not match the model")
    return TrainableCache(
        config=config,
        init_keys=[k.to(device) for k in cart.keys],
        init_values=[v.to(device) for v in cart.values],
    ).to(device)


@torch.no_grad()
def native_step(
    model: torch.nn.Module, cache: Any, ids: list[int], start: int
) -> torch.Tensor:
    """Forward ``ids`` at request positions ``start ..`` through the native
    model in generation mode (the model adds the cartridge length itself)
    and return FP32 logits for every input position."""
    device = next(model.parameters()).device
    cur = torch.tensor(ids, dtype=torch.long, device=device)
    pos = torch.arange(start, start + len(ids), device=device)
    sids = torch.zeros(len(ids), dtype=torch.long, device=device)
    out = model(
        input_ids=cur,
        seq_ids=sids,
        position_ids=pos,
        past_key_values=cache,
        use_cache=True,
        mode="generate",
    )
    return out.logits[0].float()


@torch.no_grad()
def native_greedy(
    model: torch.nn.Module,
    cache: Any,
    prompt: list[int],
    steps: int,
    eos_ids: tuple[int, ...],
) -> dict[str, Any]:
    cache.clear()
    logits = native_step(model, cache, prompt, 0)[-1]
    step_logits = [logits.cpu().numpy()]
    generated: list[int] = []
    position = len(prompt)
    for _ in range(steps):
        token = int(logits.argmax(-1))
        generated.append(token)
        if token in eos_ids or len(generated) == steps:
            break
        logits = native_step(model, cache, [token], position)[-1]
        position += 1
        step_logits.append(logits.cpu().numpy())
    cache.clear()
    return {"ids": generated, "step_logits": np.stack(step_logits)}


@torch.no_grad()
def native_teacher_forced(
    model: torch.nn.Module, cache: Any, prompt: list[int], continuation: list[int]
) -> np.ndarray:
    """Logits that predict each continuation token, from one forward over
    prompt plus continuation (the last continuation token is not scored)."""
    cache.clear()
    logits = native_step(model, cache, prompt + continuation, 0)
    cache.clear()
    start = len(prompt) - 1
    return logits[start : start + len(continuation)].cpu().numpy()


def run_native_stage(
    model_name: str,
    cartridge: Path,
    questions: list[LH.Question],
    out_dir: Path,
    device: str,
    steps: int,
    eos_ids: tuple[int, ...],
    tokenizer: Any,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cart = B.load_native_cartridge(cartridge)
    t0 = time.perf_counter()
    model = load_native_model(model_name, device)
    load_seconds = time.perf_counter() - t0
    cache = native_cache(model, cart, device)
    rows = []
    last = []
    for q in questions:
        prompt = LH.prompt_ids(tokenizer, q)
        greedy = native_greedy(model, cache, prompt, steps, eos_ids)
        forced = native_teacher_forced(model, cache, prompt, greedy["ids"])
        key = q.question_id
        np.save(out_dir / f"{key}.greedy_logits.npy", greedy["step_logits"])
        np.save(out_dir / f"{key}.forced_logits.npy", forced)
        last.append(greedy["step_logits"][0])
        rows.append(
            {
                "question_id": key,
                "prompt_ids": prompt,
                "greedy_ids": greedy["ids"],
                "greedy_steps": int(greedy["step_logits"].shape[0]),
            }
        )
    np.save(out_dir / "last_logits.npy", np.stack(last))
    record = {
        "stage": "native",
        "model": model_name,
        "cartridge": str(cartridge),
        "cartridge_digest": cart.file_digest,
        "native_tensor_digest": B.digest_native(cart),
        "rows": cart.rows,
        "frozen_rows": cart.frozen_rows,
        "cartridge_positions": [0, cart.rows - 1],
        "reader_start": cart.reader_start,
        "load_seconds": load_seconds,
        "questions": rows,
        "environment": R.environment(),
        "attention": "flex",
    }
    R.dump_json(out_dir / "stage.json", record)
    return record


@torch.no_grad()
def run_reader_stage(
    spec: M.ModelSpec,
    cartridge: Path,
    questions: list[LH.Question],
    out_dir: Path,
    device: str,
    steps: int,
    source_revision: str,
    native_dir: Path | None,
) -> dict[str, Any]:
    """This package's reader on the imported cartridge through the strict
    artifact path. With ``native_dir`` the native greedy continuations are
    teacher-forced so per-step logits can be compared."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = M.load_tokenizer(spec)
    t0 = time.perf_counter()
    base = M.load_base(spec, device)
    load_seconds = time.perf_counter() - t0
    eos = tuple(
        int(i)
        for i in (
            [tokenizer.eos_token_id]
            if isinstance(tokenizer.eos_token_id, int)
            else tokenizer.eos_token_id
        )
    )
    reader = M.Reader(base, eos_ids=eos, pad_id=int(tokenizer.pad_token_id))
    identity = M.reader_identity(base, tokenizer, spec)
    cart = B.load_native_cartridge(cartridge, dtype=spec.torch_dtype())
    art_dir = out_dir / "artifact"
    result = A.export_prefix(
        art_dir,
        cart.prefix("cpu"),
        identity,
        cart.positions,
        cart.reader_start,
        source_revision,
        cart.file_digest,
        "cas-native-import",
        {
            "native_path": str(cartridge),
            "native_digest": cart.file_digest,
            "native_tensor_digest": B.digest_native(cart),
            "frozen_rows": cart.frozen_rows,
        },
    )
    loaded = A.load_prefix(art_dir, identity, device)
    if loaded.tensor_digest != M.prefix_digest(cart.prefix("cpu")):
        raise RuntimeError("imported rows differ from the native file")
    native = (
        {
            q["question_id"]: q
            for q in R.load_json(native_dir / "stage.json")["questions"]
        }
        if native_dir
        else {}
    )
    rows = []
    last = []
    for q in questions:
        prompt = LH.prompt_ids(tokenizer, q)
        ids = torch.tensor([prompt], dtype=torch.long, device=device)
        mask = torch.ones_like(ids)
        out = reader.greedy(
            loaded.prefix, ids, mask, loaded.first_position, steps, keep_logits=True
        )
        gen = M.strip_generation(out["ids"][0].tolist(), reader.eos_ids, reader.pad_id)
        raw = out["ids"][0].tolist()
        step_logits = out["step_logits"][0].cpu().numpy()
        key = q.question_id
        np.save(out_dir / f"{key}.greedy_logits.npy", step_logits)
        last.append(step_logits[0])
        row = {
            "question_id": key,
            "prompt_ids": prompt,
            "greedy_ids": raw[: len(gen) + 1] if len(gen) < len(raw) else raw,
            "greedy_steps": int(step_logits.shape[0]),
        }
        if key in native:
            if native[key]["prompt_ids"] != prompt:
                raise RuntimeError("prompt token ids differ from the native stage")
            cont = native[key]["greedy_ids"]
            full = torch.tensor([prompt + cont], dtype=torch.long, device=device)
            hidden, _ = reader.forward(
                loaded.prefix, full, torch.ones_like(full), loaded.first_position
            )
            start = len(prompt) - 1
            forced = reader.model.lm_head(hidden[0, start : start + len(cont)]).float()
            np.save(out_dir / f"{key}.forced_logits.npy", forced.cpu().numpy())
            row["forced_continuation"] = cont
        rows.append(row)
    np.save(out_dir / "last_logits.npy", np.stack(last))
    record = {
        "stage": f"reader-{spec.attn_implementation}",
        "attention": spec.attn_implementation,
        "model": spec.name,
        "revision": spec.revision,
        "identity": identity,
        "artifact_id": result.artifact_id,
        "payload_digest": result.payload_digest,
        "cartridge": str(cartridge),
        "cartridge_digest": cart.file_digest,
        "rows": loaded.rows,
        "cartridge_positions": [loaded.positions[0], loaded.positions[-1]],
        "reader_start": loaded.first_position,
        "load_seconds": load_seconds,
        "questions": rows,
        "environment": R.environment(),
    }
    R.dump_json(out_dir / "stage.json", record)
    return record


def _kl(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    p = torch.log_softmax(torch.from_numpy(p_logits).double(), -1)
    q = torch.log_softmax(torch.from_numpy(q_logits).double(), -1)
    return float((p.exp() * (p - q)).sum())


def compare_stages(
    stage_dirs: dict[str, Path], reference: str, out_path: Path
) -> dict[str, Any]:
    """Per-stage agreement with ``reference`` (the dense-attention reader by
    design) on last-position logits, first-token argmax, greedy tokens and
    teacher-forced per-step logits along the native continuation."""
    records = {name: R.load_json(d / "stage.json") for name, d in stage_dirs.items()}
    ref = records[reference]
    ref_last = np.load(stage_dirs[reference] / "last_logits.npy")
    ref_q = {q["question_id"]: q for q in ref["questions"]}
    report: dict[str, Any] = {"reference": reference, "stages": {}}
    for name, record in records.items():
        if name == reference:
            continue
        last = np.load(stage_dirs[name] / "last_logits.npy")
        diff = np.abs(last - ref_last)
        argmax_equal = int((last.argmax(-1) == ref_last.argmax(-1)).sum())
        kls = [_kl(ref_last[i], last[i]) for i in range(last.shape[0])]
        greedy_equal = 0
        first_divergence = []
        forced_max = 0.0
        forced_argmax_total = 0
        forced_argmax_equal = 0
        for q in record["questions"]:
            key = q["question_id"]
            r = ref_q[key]
            if q["prompt_ids"] != r["prompt_ids"]:
                raise RuntimeError(f"{name}: prompt ids differ for {key}")
            a, b = q["greedy_ids"], r["greedy_ids"]
            n = min(len(a), len(b))
            div = next((i for i in range(n) if a[i] != b[i]), None)
            if div is None and len(a) == len(b):
                greedy_equal += 1
            first_divergence.append(
                div if div is not None else (None if len(a) == len(b) else n)
            )
            fa = stage_dirs[name] / f"{key}.forced_logits.npy"
            fb = stage_dirs[reference] / f"{key}.forced_logits.npy"
            if fa.exists() and fb.exists():
                la, lb = np.load(fa), np.load(fb)
                steps = min(la.shape[0], lb.shape[0])
                forced_max = max(
                    forced_max, float(np.abs(la[:steps] - lb[:steps]).max())
                )
                forced_argmax_total += steps
                forced_argmax_equal += int(
                    (la[:steps].argmax(-1) == lb[:steps].argmax(-1)).sum()
                )
        report["stages"][name] = {
            "attention": record.get("attention"),
            "questions": int(last.shape[0]),
            "last_logits_max_abs": float(diff.max()),
            "last_logits_mean_abs": float(diff.mean()),
            "last_argmax_equal": argmax_equal,
            "last_kl_max": max(kls),
            "last_kl_mean": float(np.mean(kls)),
            "greedy_sequences_equal": greedy_equal,
            "greedy_first_divergence": first_divergence,
            "forced_steps_compared": forced_argmax_total,
            "forced_argmax_equal": forced_argmax_equal,
            "forced_logits_max_abs": forced_max,
        }
    R.dump_json(out_path, report)
    return report

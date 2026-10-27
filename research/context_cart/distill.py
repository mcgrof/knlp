# SPDX-License-Identifier: MIT
"""Source-conditioned writer trained on Self-Study teacher distributions.

The writer reads a complete patient record and emits a cartridge in the
native frame (shared sink at position 0, generated rows behind it). The
frozen reader scores Self-Study conversations from that cartridge with
the native objective: for every assistant token, the cross-entropy of
the reader's next-token distribution against the captured top-k teacher
probabilities. Only the writer's adapters and memory bank train.

One optimizer update encodes one source and shares its differentiable
cartridge across that source's conversation examples: the reader scores
each example against detached copies of the rows, their gradients
accumulate, and one backward pass carries the sum into the writer. The
held-out loss panel of the selection patient, a fixed foreign cartridge
and the native cartridge are scored with the same loss.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import cas_bridge as B
from . import model as M
from . import runs as R
from . import selfstudy as S

Prefix = list[tuple[torch.Tensor, torch.Tensor]]


@dataclasses.dataclass
class DistillConfig:
    run_dir: str
    prepared_dir: str  # output of ``prepare``: splits, sources, sink
    mode: str = "pilot"  # pilot | tiny | profile
    seed: int = 61031
    updates: int = 850
    examples_per_update: int = 4
    lr: float = 1e-4
    lr_floor: float = 1e-5
    warmup_fraction: float = 0.05
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip: float = 1.0
    validate_every: int = 100
    validate_at_start: bool = True
    profile_warmup: int = 10
    profile_timed: int = 40
    tiny_patient: str = "patient_01"
    tiny_groups: int = 16
    log_every: int = 1
    device: str = "cuda"
    checkpoint_writer: bool = True
    # directory holding the native cartridge files by basename when the
    # prepared manifest's absolute paths belong to another host
    cart_dir: str | None = None
    model: dict[str, Any] = dataclasses.field(
        default_factory=lambda: dataclasses.asdict(M.ModelSpec())
    )

    def spec(self) -> M.ModelSpec:
        values = dict(self.model)
        values["lora_targets"] = tuple(
            values.get("lora_targets", M.ModelSpec().lora_targets)
        )
        return M.ModelSpec(**values)


@dataclasses.dataclass
class Source:
    patient_id: str
    record_path: str
    record_digest: str
    source_ids: list[int]
    rows: int  # native cartridge size P for this record
    native_cartridge: str | None = None
    native_digest: str | None = None

    @property
    def content_rows(self) -> int:
        return self.rows - 1


def record_source(
    patient_id: str, record_path: Path, tokenizer: Any, divisor: int = 20
) -> Source:
    text = Path(record_path).read_text(encoding="utf-8")
    ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
    return Source(
        patient_id=patient_id,
        record_path=str(record_path),
        record_digest=M.sha256_file(Path(record_path)),
        source_ids=[int(t) for t in ids],
        rows=S.cartridge_rows(len(ids), divisor),
    )


def topk_loss(
    reader: M.Reader,
    prefix: Prefix,
    element: S.Element,
    first_position: int,
) -> dict[str, Any]:
    """Native objective on one conversation: for each scored token t with
    teacher rows (ids_k, p_k), ``-sum_k p_k log q_t(ids_k)``. Returns the
    sum over tokens, the token count, the retained teacher mass and the
    count of positions where the reader's argmax is the teacher's top-1.
    ``loss_sum / (tokens * K)`` is the native code's mean over entries."""
    device = reader.device
    ids = torch.tensor([element.input_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    hidden, _ = reader.forward(prefix, ids, mask, first_position)
    positions = torch.tensor(element.target_positions, device=device) - 1
    logits = reader.model.lm_head(hidden[0, positions]).float()
    logq = torch.log_softmax(logits, dim=-1)
    target_ids = torch.from_numpy(element.target_ids).to(device)
    weights = torch.from_numpy(element.target_logprobs).to(device).exp()
    valid = target_ids >= 0
    weights = torch.where(valid, weights, torch.zeros_like(weights))
    gathered = logq.gather(1, target_ids.clamp(min=0))
    per_token = -(weights * gathered).sum(1)
    with torch.no_grad():
        teacher_top = element.target_logprobs.argmax(1)
        teacher_top_ids = torch.from_numpy(
            element.target_ids[np.arange(len(teacher_top)), teacher_top]
        ).to(device)
        agree = int((logq.argmax(1) == teacher_top_ids).sum())
        mass = float(weights.sum(1).mean())
    return {
        "loss_sum": per_token.sum(),
        "tokens": int(per_token.numel()),
        "mass": mass,
        "agree": agree,
        "k": int(target_ids.shape[1]),
    }


def framed_prefix(
    writer: M.Writer,
    reader: M.Reader,
    source: Source,
    sink: Prefix,
    dtype: torch.dtype,
    checkpoint: bool,
) -> tuple[Prefix, int]:
    """Writer rows for ``source`` moved into the native frame; the result
    carries the writer's graph. Returns the prefix and the reader start."""
    ids = torch.tensor([source.source_ids], dtype=torch.long, device=reader.device)
    generated, length = writer.encode(
        ids, slots=source.content_rows, checkpoint=checkpoint
    )
    framed = B.generated_to_native_frame(reader.model, generated, length, sink, dtype)
    return framed, source.rows


def shared_prefix_step(
    writer: M.Writer,
    reader: M.Reader,
    source: Source,
    elements: list[S.Element],
    sink: Prefix,
    dtype: torch.dtype,
    checkpoint: bool,
) -> dict[str, Any]:
    """Encode once, score every example against detached copies of the
    rows, accumulate their gradients, and push the sum through the writer.
    The loss is normalised by the examples' total scored tokens."""
    framed, start = framed_prefix(writer, reader, source, sink, dtype, checkpoint)
    leaves = [
        (k.detach().requires_grad_(True), v.detach().requires_grad_(True))
        for k, v in framed
    ]
    total_tokens = sum(e.scored_tokens for e in elements)
    if total_tokens == 0:
        raise ValueError("no scored tokens in this update")
    loss_total, agree, mass = 0.0, 0, 0.0
    for element in elements:
        out = topk_loss(reader, leaves, element, start)
        loss = out["loss_sum"] / total_tokens
        loss.backward()
        loss_total += float(out["loss_sum"].item())
        agree += out["agree"]
        mass += out["mass"] * out["tokens"]
    tensors, grads = [], []
    for (k, v), (lk, lv) in zip(framed, leaves):
        if lk.grad is None or lv.grad is None:
            raise RuntimeError("an example left prefix rows without gradient")
        tensors += [k, v]
        grads += [lk.grad, lv.grad]
    torch.autograd.backward(tensors, grads)
    return {
        "loss_per_token": loss_total / total_tokens,
        "tokens": total_tokens,
        "teacher_agreement": agree / total_tokens,
        "teacher_mass": mass / total_tokens,
        "examples": len(elements),
    }


@torch.no_grad()
def panel_loss(
    reader: M.Reader, prefix: Prefix, elements: list[S.Element], start: int
) -> dict[str, Any]:
    total, tokens, agree, mass = 0.0, 0, 0, 0.0
    for element in elements:
        out = topk_loss(reader, prefix, element, start)
        total += float(out["loss_sum"].item())
        tokens += out["tokens"]
        agree += out["agree"]
        mass += out["mass"] * out["tokens"]
    return {
        "loss_per_token": total / max(tokens, 1),
        "loss_native_mean": total / max(tokens, 1) / S.TOP_K,
        "tokens": tokens,
        "rows": len(elements),
        "teacher_agreement": agree / max(tokens, 1),
        "teacher_mass": mass / max(tokens, 1),
    }


def load_prepared(
    prepared_dir: Path, cart_dir: str | Path | None = None
) -> dict[str, Any]:
    """The prepared manifest; with ``cart_dir`` every native cartridge path
    is re-rooted there by basename (the files travel with the bundle)."""
    prepared = R.load_json(prepared_dir / "prepared.json")
    if cart_dir:
        root = Path(cart_dir)

        def move(path: str) -> str:
            return str(root / Path(path).name)

        prepared["sink"]["path"] = move(prepared["sink"]["path"])
        prepared["controls"] = {k: move(v) for k, v in prepared["controls"].items()}
        for source in prepared["sources"].values():
            if source.get("native_cartridge"):
                source["native_cartridge"] = move(source["native_cartridge"])
    return prepared


def load_elements(path: Path) -> list[S.Element]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return [S.Element(**item) for item in payload]


def save_elements(path: Path, elements: list[S.Element]) -> None:
    torch.save([dataclasses.asdict(e) for e in elements], path)


def cosine_with_warmup(
    step: int, total: int, warmup: int, peak: float, floor: float
) -> float:
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * min(progress, 1.0)))


def reader_state_digest(reader: M.Reader) -> str:
    return M.weights_digest(reader.model)


def train(cfg: DistillConfig) -> dict[str, Any]:
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    R.dump_json(run_dir / "distill_config.json", dataclasses.asdict(cfg))
    threads = R.configure_threads()
    spec = cfg.spec()
    device = cfg.device
    dtype = spec.torch_dtype()
    prepared = load_prepared(Path(cfg.prepared_dir), cfg.cart_dir)
    sources = {
        p: Source(**{k: v for k, v in s.items()})
        for p, s in prepared["sources"].items()
    }
    bank = max(s.content_rows for s in sources.values())
    if spec.memory_slots != bank:
        raise ValueError(
            f"model.memory_slots must equal the largest content row count {bank}"
        )
    tokenizer = M.load_tokenizer(spec)
    torch.manual_seed(cfg.seed)
    reader_base = M.load_base(spec, device)
    writer_base = M.load_base(spec, device)
    rows = M.nonspecial_vocab_rows(
        tokenizer, bank, cfg.seed, reader_base.config.vocab_size
    )
    writer = M.Writer(writer_base, spec, rows)
    eos = tuple(
        int(i)
        for i in (
            [tokenizer.eos_token_id]
            if isinstance(tokenizer.eos_token_id, int)
            else tokenizer.eos_token_id
        )
    )
    reader = M.Reader(reader_base, eos_ids=eos, pad_id=int(tokenizer.pad_token_id))
    identity = M.reader_identity(reader_base, tokenizer, spec)
    reader_digest_before = identity["weights_digest"]
    sink_cart = B.load_native_cartridge(Path(prepared["sink"]["path"]), dtype=dtype)
    sink = [
        (k[:, :, :1].to(device), v[:, :, :1].to(device))
        for k, v in sink_cart.prefix("cpu")
    ]
    if B.digest_native(sink_cart) != prepared["sink"]["native_tensor_digest"]:
        raise RuntimeError("sink cartridge changed since preparation")

    # data
    prepared_dir = Path(cfg.prepared_dir)
    train_examples: list[tuple[str, int]] = []  # (patient, element index)
    elements: dict[str, list[S.Element]] = {}
    for patient in prepared["train_patients"]:
        elements[patient] = load_elements(prepared_dir / f"{patient}_train.pt")
        train_examples += [(patient, i) for i in range(len(elements[patient]))]
    panels: dict[str, list[S.Element]] = {}
    for name, path in prepared["panels"].items():
        panels[name] = load_elements(prepared_dir / path)
    if cfg.mode == "tiny":
        patient = cfg.tiny_patient
        groups: dict[str, list[int]] = {}
        for i, e in enumerate(elements[patient]):
            groups.setdefault(e.prompt, []).append(i)
        chosen = sorted(groups)[: cfg.tiny_groups]
        tiny_idx = [groups[g][0] for g in chosen]
        train_examples = [(patient, i) for i in tiny_idx]
        panels = {"tiny_fit": [elements[patient][i] for i in tiny_idx]}
    order = list(range(len(train_examples)))
    rng = random.Random(cfg.seed)
    rng.shuffle(order)
    # one patient per update: the examples of an update share one source
    by_patient: dict[str, list[int]] = {}
    for index in order:
        by_patient.setdefault(train_examples[index][0], []).append(index)
    patients_cycle = sorted(by_patient)
    cursors = {p: 0 for p in patients_cycle}
    exposure: dict[str, int] = {}

    def next_batch(update: int) -> tuple[str, list[S.Element]]:
        patient = patients_cycle[update % len(patients_cycle)]
        pool = by_patient[patient]
        picked = []
        for _ in range(cfg.examples_per_update):
            if cursors[patient] >= len(pool):
                cursors[patient] = 0
                rng.shuffle(pool)
            index = pool[cursors[patient]]
            cursors[patient] += 1
            picked.append(index)
            key = f"{patient}:{train_examples[index][1]}"
            exposure[key] = exposure.get(key, 0) + 1
        return patient, [elements[patient][train_examples[i][1]] for i in picked]

    params = writer.trainable_parameters()
    optimizer = torch.optim.AdamW(
        params,
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
        eps=1e-8,
    )
    total_updates = (
        cfg.updates if cfg.mode != "profile" else cfg.profile_warmup + cfg.profile_timed
    )
    warmup = max(1, int(round(cfg.warmup_fraction * cfg.updates)))
    log = (run_dir / "train_log.jsonl").open("a", encoding="utf-8")
    summary: dict[str, Any] = {
        "identity": identity,
        "init_rows": rows,
        "memory_bank_rows": bank,
        "sources": {
            p: {k: v for k, v in dataclasses.asdict(s).items() if k != "source_ids"}
            | {"source_tokens": len(s.source_ids)}
            for p, s in sources.items()
        },
        "trainable_parameters": int(sum(p.numel() for p in params)),
        "train_examples": len(train_examples),
        "train_examples_by_patient": {p: len(v) for p, v in by_patient.items()},
        "panels": {k: len(v) for k, v in panels.items()},
        "warmup_updates": warmup,
        "threads": threads,
        "environment": R.environment(),
        "validations": {},
        "checkpoints": {},
        "mode": cfg.mode,
    }
    sync = (
        torch.cuda.synchronize
        if torch.device(device).type == "cuda"
        else (lambda: None)
    )

    def save(tag: str, update: int) -> str:
        path = run_dir / f"ckpt_{tag}.pt"
        payload = {
            "trainable": writer.trainable_state(),
            "init_rows": writer.init_rows,
            "update": update,
            "config": dataclasses.asdict(cfg),
            "torch_rng": torch.get_rng_state(),
        }
        torch.save(payload, path)
        digest = R.file_digest(path)
        summary["checkpoints"][tag] = {"path": str(path), "digest": digest}
        return digest

    @torch.no_grad()
    def validate(tag: str, update: int) -> dict[str, Any]:
        writer.eval()
        result: dict[str, Any] = {"update": update}
        for name, panel in panels.items():
            patient = (
                prepared["panel_patients"][name]
                if name in prepared.get("panel_patients", {})
                else cfg.tiny_patient
            )
            source = sources[patient]
            t0 = time.perf_counter()
            framed, start = framed_prefix(writer, reader, source, sink, dtype, False)
            framed = [(k.detach(), v.detach()) for k, v in framed]
            sync()
            build = time.perf_counter() - t0
            scores = {"generated": panel_loss(reader, framed, panel, start)}
            scores["generated"]["build_seconds"] = build
            if name == prepared.get("selection_panel"):
                for control, cart_path in prepared["controls"].items():
                    cart = B.load_native_cartridge(Path(cart_path), dtype=dtype)
                    scores[control] = panel_loss(
                        reader, cart.prefix(device), panel, cart.reader_start
                    )
                    scores[control]["rows"] = cart.rows
                wrong = sources[prepared["wrong_source_patient"]]
                wrong_source = dataclasses.replace(wrong, rows=source.rows)
                framed_w, start_w = framed_prefix(
                    writer, reader, wrong_source, sink, dtype, False
                )
                framed_w = [(k.detach(), v.detach()) for k, v in framed_w]
                scores["wrong_source_generated"] = panel_loss(
                    reader, framed_w, panel, start_w
                )
                del framed_w
            result[name] = scores
            del framed
        summary["validations"][tag] = result
        R.dump_json(run_dir / f"val_{tag}.json", result)
        writer.train()
        return result

    if cfg.validate_at_start and cfg.mode != "profile":
        validate("0", 0)
        save("0", 0)
    writer.train()
    timings: list[float] = []
    R.reset_peak_memory()
    t_train = time.perf_counter()
    for update in range(total_updates):
        lr = cosine_with_warmup(update, cfg.updates, warmup, cfg.lr, cfg.lr_floor)
        for group in optimizer.param_groups:
            group["lr"] = lr
        patient, batch = next_batch(update)
        sync()
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        stats = shared_prefix_step(
            writer, reader, sources[patient], batch, sink, dtype, cfg.checkpoint_writer
        )
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip))
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient norm at update {update + 1}")
        optimizer.step()
        sync()
        seconds = time.perf_counter() - t0
        timings.append(seconds)
        row = {
            "update": update + 1,
            "patient": patient,
            "lr": lr,
            "grad_norm": grad_norm,
            "seconds": seconds,
            **stats,
        }
        log.write(json.dumps(row) + "\n")
        log.flush()
        if cfg.log_every and (update + 1) % cfg.log_every == 0:
            print(
                f"update {update + 1}/{total_updates} {patient} loss/token {stats['loss_per_token']:.4f} "
                f"agree {stats['teacher_agreement']:.3f} tokens {stats['tokens']} gn {grad_norm:.3f} {seconds:.2f}s",
                flush=True,
            )
        if (
            cfg.mode != "profile"
            and cfg.validate_every
            and (update + 1) % cfg.validate_every == 0
        ):
            validate(str(update + 1), update + 1)
            save(str(update + 1), update + 1)
    sync()
    summary["train_seconds"] = time.perf_counter() - t_train
    summary["updates_run"] = total_updates
    summary["peak_memory"] = R.peak_memory()
    summary["exposure"] = {
        "unique_examples_seen": len(exposure),
        "max_exposure": max(exposure.values()) if exposure else 0,
        "mean_exposure": (sum(exposure.values()) / len(exposure)) if exposure else 0,
        "passes_over_train_set": (sum(exposure.values()) / max(len(train_examples), 1)),
    }
    if cfg.mode == "profile":
        timed = timings[cfg.profile_warmup :]
        summary["profile"] = {
            "warmup_updates": cfg.profile_warmup,
            "timed_updates": len(timed),
            "seconds_per_update_mean": float(np.mean(timed)),
            "seconds_per_update_p50": float(np.median(timed)),
            "seconds_per_update_p90": float(np.percentile(timed, 90)),
            "projected_seconds_1000_updates": 1000 * float(np.median(timed)),
        }
    elif cfg.mode != "profile" and (str(total_updates) not in summary["validations"]):
        validate(str(total_updates), total_updates)
        save(str(total_updates), total_updates)
    if cfg.mode != "profile":
        panel = (
            "tiny_fit"
            if cfg.mode == "tiny"
            else prepared.get("selection_panel", next(iter(panels)))
        )
        best = None
        for tag, val in summary["validations"].items():
            score = val[panel]["generated"]["loss_per_token"]
            if best is None or score < best[1]:
                best = (tag, score)
        summary["selected_checkpoint"] = best[0] if best else None
        summary["selection_rule"] = (
            f"lowest {panel} generated loss per token, earliest on ties"
        )
    reader_digest_after = reader_state_digest(reader)
    summary["reader_unchanged"] = reader_digest_after == reader_digest_before
    if not summary["reader_unchanged"]:
        raise RuntimeError("reader weights changed during training")
    log.close()
    R.dump_json(run_dir / "train_summary.json", summary)
    return summary


def export_generated(
    writer: M.Writer,
    reader: M.Reader,
    source: Source,
    sink: Prefix,
    dtype: torch.dtype,
    rows_override: int | None = None,
) -> tuple[Prefix, int]:
    """One cartridge in the native frame from a writer state, detached. With
    ``rows_override`` the source is written into another patient's size (the
    wrong-source control keeps the target's position contract)."""
    src = (
        source
        if rows_override is None
        else dataclasses.replace(source, rows=rows_override)
    )
    with torch.no_grad():
        framed, start = framed_prefix(writer, reader, src, sink, dtype, False)
    return [(k.detach().clone(), v.detach().clone()) for k, v in framed], start


def prepare(
    out_dir: Path,
    tokenizer: Any,
    records_dir: Path,
    parquet_dir: Path,
    native_cartridges: dict[str, Path],
    benchmark: Path,
    train_patients: tuple[str, ...],
    selection_patient: str,
    test_patient: str,
    wrong_source_patient: str,
    fixed_cartridge_patient: str,
    train_groups: int,
    question_groups: int,
    holdout_groups: int,
    split_seed: int,
    max_reader_tokens: int,
    divisor: int = 20,
) -> dict[str, Any]:
    """Freeze every input of a writer screen before any training: sources
    with their native cartridge sizes, the shared sink verified identical
    across the retained cartridges, per-patient prompt-group splits, the
    reader-ready elements, the control cartridges and an overlap audit
    against the evaluation questions."""
    from . import longhealth as LH

    out_dir.mkdir(parents=True, exist_ok=True)
    data, benchmark_digest = LH.load_benchmark(benchmark)
    patients = tuple(
        dict.fromkeys(
            train_patients
            + (
                selection_patient,
                test_patient,
                wrong_source_patient,
                fixed_cartridge_patient,
            )
        )
    )
    sources: dict[str, Source] = {}
    carts: dict[str, B.NativeCartridge] = {}
    for patient in patients:
        source = record_source(
            patient, records_dir / f"{patient}.txt", tokenizer, divisor
        )
        if patient in native_cartridges:
            cart = B.load_native_cartridge(native_cartridges[patient])
            carts[patient] = cart
            if cart.rows != source.rows:
                raise ValueError(
                    f"{patient}: size rule gives {source.rows} rows, native cartridge has {cart.rows}"
                )
            source.native_cartridge = str(native_cartridges[patient])
            source.native_digest = cart.file_digest
        sources[patient] = source
    # the shared sink: the frozen row of every retained cartridge, bit-equal
    sink_patient = fixed_cartridge_patient
    sink_cart = carts[sink_patient]
    sink_rows = [
        (k[:, :, :1], v[:, :, :1]) for k, v in zip(sink_cart.keys, sink_cart.values)
    ]
    sink_checks = {}
    for patient, cart in carts.items():
        rows = [(k[:, :, :1], v[:, :, :1]) for k, v in zip(cart.keys, cart.values)]
        sink_checks[patient] = {
            "frozen_rows": cart.frozen_rows,
            "equal_to_sink": B.prefix_equal(rows, sink_rows),
        }
        if cart.frozen_rows != 1 or not sink_checks[patient]["equal_to_sink"]:
            raise ValueError(f"{patient}: frozen sink differs from {sink_patient}")
    splits: dict[str, Any] = {}
    panels: dict[str, str] = {}
    panel_patients: dict[str, str] = {}
    audit_prompts: list[str] = []
    for patient in train_patients:
        rows = S.read_rows(parquet_dir / f"{patient}.parquet")
        split = S.split_patient(
            patient,
            parquet_dir / f"{patient}.parquet",
            rows,
            split_seed,
            train_groups,
            question_groups,
            0,
        )
        train_elements = [
            S.element_from_row(rows[i], i, max_reader_tokens, tokenizer)
            for i in split.train_rows
        ]
        question_elements = [
            S.element_from_row(rows[i], i, max_reader_tokens, tokenizer)
            for i in split.question_rows
        ]
        save_elements(out_dir / f"{patient}_train.pt", train_elements)
        save_elements(out_dir / f"{patient}_question.pt", question_elements)
        panels[f"{patient}_question"] = f"{patient}_question.pt"
        panel_patients[f"{patient}_question"] = patient
        splits[patient] = split.to_dict() | {
            "train_elements": len(train_elements),
            "train_scored_tokens": sum(e.scored_tokens for e in train_elements),
            "train_truncated": sum(e.truncated for e in train_elements),
            "teacher_mass_mean": float(
                np.mean([e.teacher_mass() for e in train_elements])
            ),
            "kinds": {
                k: sum(e.kind == k for e in train_elements)
                for k in sorted({e.kind for e in train_elements})
            },
        }
        audit_prompts += split.train_groups
    rows = S.read_rows(parquet_dir / f"{selection_patient}.parquet")
    split = S.split_patient(
        selection_patient,
        parquet_dir / f"{selection_patient}.parquet",
        rows,
        split_seed,
        0,
        0,
        holdout_groups,
    )
    holdout = [
        S.element_from_row(rows[i], i, max_reader_tokens, tokenizer)
        for i in split.holdout_rows
    ]
    save_elements(out_dir / f"{selection_patient}_holdout.pt", holdout)
    panels[f"{selection_patient}_holdout"] = f"{selection_patient}_holdout.pt"
    panel_patients[f"{selection_patient}_holdout"] = selection_patient
    splits[selection_patient] = split.to_dict() | {
        "holdout_elements": len(holdout),
        "holdout_scored_tokens": sum(e.scored_tokens for e in holdout),
    }
    questions = []
    for patient in patients:
        questions += [q.question for q in LH.patient_questions(data, patient)]
    audit = S.overlap_audit(audit_prompts, questions)
    prepared = {
        "benchmark": str(benchmark),
        "benchmark_digest": benchmark_digest,
        "records_dir": str(records_dir),
        "parquet_dir": str(parquet_dir),
        "train_patients": list(train_patients),
        "selection_patient": selection_patient,
        "test_patient": test_patient,
        "wrong_source_patient": wrong_source_patient,
        "selection_panel": f"{selection_patient}_holdout",
        "panels": panels,
        "panel_patients": panel_patients,
        "controls": {
            f"native_{selection_patient}": str(native_cartridges[selection_patient]),
            f"fixed_{fixed_cartridge_patient}": str(
                native_cartridges[fixed_cartridge_patient]
            ),
        },
        "sink": {
            "patient": sink_patient,
            "path": str(native_cartridges[sink_patient]),
            "native_tensor_digest": B.digest_native(sink_cart),
            "checks": sink_checks,
        },
        "sources": {p: dataclasses.asdict(s) for p, s in sources.items()},
        "memory_bank_rows": max(s.content_rows for s in sources.values()),
        "splits": splits,
        "max_reader_tokens": max_reader_tokens,
        "divisor": divisor,
        "overlap_audit": {
            "max_jaccard": max((a["jaccard"] for a in audit), default=0.0),
            "questions": len(audit),
            "worst": sorted(audit, key=lambda a: -a["jaccard"])[:5],
        },
    }
    R.dump_json(out_dir / "prepared.json", prepared)
    return prepared

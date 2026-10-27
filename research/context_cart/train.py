# SPDX-License-Identifier: MIT
"""Writer training with answer-only cross-entropy through the frozen reader.

Each optimizer update sees ``effective_batch`` source-question examples. The
writer encodes each source alone; the reader scores the teacher-forced answer
from the exported rows plus instruction and question. Only the adapters and
memory embeddings train, in FP32 master precision. Validation runs greedy
generation on a fixed panel at the configured updates and saves a checkpoint
each time. Checkpoint selection uses validation macro exact match only.
"""

from __future__ import annotations

import dataclasses
import random
import time
from pathlib import Path
from typing import Any

import torch

from . import dataset as D
from . import evaluate as E
from . import model as M
from . import prompts as P
from . import runs as R


@dataclasses.dataclass
class TrainConfig:
    run_dir: str
    dataset_dir: str
    mode: str = "pilot"  # pilot | tiny | profile
    seed: int = 61031
    pool: str = "train"
    updates: int = 1000
    effective_batch: int = 8
    microbatch: int = 8
    lr: float = 1e-4
    lr_floor: float = 1e-5
    warmup: int = 50
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip: float = 1.0
    validate_at: tuple[int, ...] = (250, 500, 1000)
    validate_at_start: bool = False
    validation_pool: str = "validation"
    validation_worlds: int = 64
    eval_batch: int = 16
    max_new_tokens: int = 16
    order_seed: int = D.DATA_SEED
    log_every: int = 10
    profile_warmup: int = 10
    profile_timed: int = 50
    device: str = "cuda"
    model: dict[str, Any] = dataclasses.field(
        default_factory=lambda: dataclasses.asdict(M.ModelSpec())
    )
    init_state: str | None = None

    def spec(self) -> M.ModelSpec:
        values = dict(self.model)
        values["lora_targets"] = tuple(
            values.get("lora_targets", M.ModelSpec().lora_targets)
        )
        return M.ModelSpec(**values)


def load_worlds(dataset_dir: Path, pool: str) -> list[dict[str, Any]]:
    return D.read_jsonl(dataset_dir / f"worlds_{pool}.jsonl")


def validation_items(
    cfg: TrainConfig, worlds: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[E.Item]]:
    if (
        cfg.validation_worlds <= 0
        or cfg.validation_worlds >= len(worlds)
        and cfg.pool == cfg.validation_pool
    ):
        return worlds, E.items_from_worlds(worlds)
    panel = D.validation_panel(worlds, cfg.validation_worlds)
    wanted = {(w, q) for w, q in panel}
    by_id = {w["world_id"]: w for w in worlds}
    chosen_worlds = [by_id[w] for w in dict.fromkeys(w for w, _ in panel)]
    items = [
        i for i in E.items_from_worlds(chosen_worlds) if (i.world_id, i.qid) in wanted
    ]
    return chosen_worlds, items


def build_models(
    spec: M.ModelSpec, device: str, seed: int, tokenizer: Any
) -> tuple[M.Writer, M.Reader, dict[str, Any], list[int]]:
    torch.manual_seed(seed)
    reader_base = M.load_base(spec, device)
    writer_base = M.load_base(spec, device)
    rows = M.nonspecial_vocab_rows(
        tokenizer, spec.memory_slots, seed, reader_base.config.vocab_size
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
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id not in eos:
        eos = eos + (int(tokenizer.pad_token_id),)
    reader = M.Reader(reader_base, eos_ids=eos, pad_id=int(tokenizer.pad_token_id))
    identity = M.reader_identity(reader_base, tokenizer, spec)
    return writer, reader, identity, rows


def save_checkpoint(
    path: Path,
    writer: M.Writer,
    optimizer: torch.optim.Optimizer | None,
    update: int,
    cfg: TrainConfig,
) -> str:
    payload = {
        "trainable": writer.trainable_state(),
        "init_rows": writer.init_rows,
        "update": update,
        "config": dataclasses.asdict(cfg),
        "torch_rng": torch.get_rng_state(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if torch.cuda.is_available():
        payload["cuda_rng"] = torch.cuda.get_rng_state()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return R.file_digest(path)


def load_trainable(path: Path, writer: M.Writer) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    writer.load_trainable_state(payload["trainable"])
    return payload


@torch.no_grad()
def validate(
    writer: M.Writer,
    reader: M.Reader,
    tokenizer: Any,
    worlds: list[dict[str, Any]],
    items: list[E.Item],
    cfg: TrainConfig,
    tag: str,
    run_dir: Path,
) -> dict[str, Any]:
    writer.eval()
    prefixes, length, build_seconds = E.encode_worlds(
        writer, worlds, cfg.eval_batch, reader.device
    )
    rows, timing, _ = E.answer_memory_arm(
        reader,
        tokenizer,
        "learned",
        prefixes,
        items,
        length + writer.memory.shape[0],
        cfg.eval_batch,
        cfg.max_new_tokens,
    )
    result = E.metrics(rows)
    result["timing"] = dataclasses.asdict(timing)
    result["timing"]["build_seconds"] = build_seconds
    result["source_tokens"] = length
    result["memory_rows"] = int(writer.memory.shape[0])
    E.write_rows(run_dir / f"val_{tag}_predictions.jsonl", rows)
    R.dump_json(run_dir / f"val_{tag}.json", result)
    del prefixes
    return result


def train(cfg: TrainConfig) -> dict[str, Any]:
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    R.dump_json(run_dir / "train_config.json", dataclasses.asdict(cfg))
    threads = R.configure_threads()
    spec = cfg.spec()
    device = cfg.device
    tokenizer = M.load_tokenizer(spec)
    writer, reader, identity, init_rows = build_models(
        spec, device, cfg.seed, tokenizer
    )
    if cfg.init_state:
        load_trainable(Path(cfg.init_state), writer)
    init_digest = save_checkpoint(run_dir / "ckpt_init.pt", writer, None, 0, cfg)
    dataset_dir = Path(cfg.dataset_dir)
    worlds = load_worlds(dataset_dir, cfg.pool)
    val_worlds_all = (
        worlds
        if cfg.validation_pool == cfg.pool
        else load_worlds(dataset_dir, cfg.validation_pool)
    )
    val_worlds, val_items = validation_items(cfg, val_worlds_all)
    slots = spec.memory_slots
    source_tokens = len(worlds[0]["source_ids"])
    first_position = source_tokens + slots

    # pre-tokenise every training example once; order is fixed by order_seed
    examples = []
    for w_index, world in enumerate(worlds):
        for q in world["questions"]:
            prompt = P.prompt_ids(tokenizer, P.user_message("learned", q["text"]))
            answer = P.answer_ids(tokenizer, q["answer"])
            if len(prompt) + len(answer) > 128:
                raise ValueError("reader sequence exceeds 128 positions")
            examples.append((w_index, prompt, answer))
    order = list(range(len(examples)))
    random.Random(cfg.order_seed).shuffle(order)

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
    schedule_updates = cfg.updates
    log_path = run_dir / "train_log.jsonl"
    log = log_path.open("a", encoding="utf-8")
    summary: dict[str, Any] = {
        "identity": identity,
        "init_rows": init_rows,
        "init_checkpoint_digest": init_digest,
        "trainable_parameters": int(sum(p.numel() for p in params)),
        "examples": len(examples),
        "source_tokens": source_tokens,
        "memory_rows": slots,
        "first_reader_position": first_position,
        "threads": threads,
        "environment": R.environment(),
        "validations": {},
        "checkpoints": {},
    }
    if cfg.validate_at_start and cfg.mode != "profile":
        summary["validations"]["0"] = validate(
            writer, reader, tokenizer, val_worlds, val_items, cfg, "0", run_dir
        )
        summary["checkpoints"]["0"] = {
            "path": str(run_dir / "ckpt_init.pt"),
            "digest": init_digest,
        }

    sync = (
        torch.cuda.synchronize
        if torch.device(device).type == "cuda"
        else (lambda: None)
    )
    R.reset_peak_memory()
    pointer = 0
    timed_seconds: list[float] = []
    train_start = time.perf_counter()
    for update in range(1, total_updates + 1):
        lr = M.cosine_lr(update - 1, schedule_updates, cfg.warmup, cfg.lr, cfg.lr_floor)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch = [
            examples[order[(pointer + i) % len(order)]]
            for i in range(cfg.effective_batch)
        ]
        pointer += cfg.effective_batch
        total_tokens = sum(len(a) for _, _, a in batch)
        sync()
        t0 = time.perf_counter()
        loss_total = 0.0
        for micro in E.chunks(batch, cfg.microbatch):
            source = torch.tensor(
                [worlds[w]["source_ids"] for w, _, _ in micro],
                dtype=torch.long,
                device=device,
            )
            prefix, length = writer.encode(source)
            assert length == source_tokens
            ids, mask, labels = M.right_pad_with_labels(
                [p for _, p, _ in micro],
                [a for _, _, a in micro],
                reader.pad_id,
                device,
            )
            loss_sum, count, _ = reader.answer_loss(
                prefix, ids, mask, labels, first_position
            )
            (loss_sum / total_tokens).backward()
            loss_total += float(loss_sum.item())
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip).item())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        sync()
        seconds = time.perf_counter() - t0
        if cfg.mode == "profile" and update > cfg.profile_warmup:
            timed_seconds.append(seconds)
        record = {
            "update": update,
            "loss": loss_total / total_tokens,
            "answer_tokens": total_tokens,
            "lr": lr,
            "grad_norm": grad_norm,
            "seconds": seconds,
        }
        if update % cfg.log_every == 0 or update == 1 or update == total_updates:
            record.update(R.peak_memory())
            print(
                f"update {update} loss {record['loss']:.4f} lr {lr:.2e} gn {grad_norm:.3f} {seconds:.2f}s",
                flush=True,
            )
        log.write(D.canonical_json(record) + "\n")
        log.flush()
        if cfg.mode != "profile" and update in set(cfg.validate_at):
            tag = str(update)
            summary["validations"][tag] = validate(
                writer, reader, tokenizer, val_worlds, val_items, cfg, tag, run_dir
            )
            path = run_dir / f"ckpt_{tag}.pt"
            summary["checkpoints"][tag] = {
                "path": str(path),
                "digest": save_checkpoint(path, writer, optimizer, update, cfg),
            }
            print(
                f"validation at {tag}: macro EM {summary['validations'][tag]['macro_em']:.3f}",
                flush=True,
            )
    log.close()
    summary["train_seconds"] = time.perf_counter() - train_start
    summary["updates_run"] = total_updates
    summary["peak_memory"] = R.peak_memory()
    if cfg.mode == "profile":
        timed_seconds.sort()
        n = len(timed_seconds)
        summary["profile"] = {
            "warmup_updates": cfg.profile_warmup,
            "timed_updates": n,
            "seconds_per_update_mean": sum(timed_seconds) / max(1, n),
            "seconds_per_update_p50": timed_seconds[n // 2] if n else None,
            "seconds_per_update_max": timed_seconds[-1] if n else None,
            "effective_batch": cfg.effective_batch,
            "microbatch": cfg.microbatch,
            "writer_positions": source_tokens + slots,
            "projected_seconds_1000_updates": 1000 * sum(timed_seconds) / max(1, n),
        }
        t_eval = time.perf_counter()
        summary["profile"]["validation_sample"] = validate(
            writer, reader, tokenizer, val_worlds, val_items, cfg, "profile", run_dir
        )
        summary["profile"]["validation_seconds"] = time.perf_counter() - t_eval
    else:
        best_tag, best_score = None, -1.0
        for tag in sorted(summary["validations"], key=int):
            if tag == "0":
                continue
            score = summary["validations"][tag]["macro_em"]
            if score > best_score:
                best_tag, best_score = tag, score
        summary["selected_checkpoint"] = best_tag
        summary["selected_macro_em"] = best_score
    R.dump_json(run_dir / "train_summary.json", summary)
    return summary

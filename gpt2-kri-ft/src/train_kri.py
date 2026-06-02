"""KRI-aware fine-tuning of GPT-2 small.

Modes:
  - dense : no KRI mask (control fine-tune)
  - kri   : KRI mask on every batch
  - mixed : random per-batch choice between dense and KRI (default)

Optional teacher-KL distillation against frozen HF GPT-2 on the dense
logits, weighted by `--teacher_kl_alpha`.

The CLI mirrors the example in the project README. Defaults match the
"smoke" config; pass `--max_steps 10000 --seq_len 1024 ...` for a
serious run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import DataConfig, collate, get_tokenizer, get_train_val_streams  # noqa: E402
from src.kri_mask import KRIConfig, build_kri_mask  # noqa: E402
from src.model_gpt2_kri import GPT2KRI  # noqa: E402
from src.model_smollm2_kri import SmolLM2KRI  # noqa: E402


def _load_base_model(base_model: str, init_name: str, device):
    """Return (model, teacher_factory). teacher_factory()
    instantiates a frozen teacher with the same weights."""
    if base_model == "gpt2":
        return (
            GPT2KRI.from_hf_gpt2(init_name).to(device),
            lambda: GPT2KRI.from_hf_gpt2(init_name).to(device).eval(),
        )
    if base_model == "smollm2":
        return (
            SmolLM2KRI.from_hf_smollm2(init_name).to(device),
            lambda: SmolLM2KRI.from_hf_smollm2(init_name).to(device).eval(),
        )
    raise ValueError(f"unknown base_model: {base_model}")
from src.utils import (  # noqa: E402
    StepLog,
    Timer,
    banner,
    log_step,
    pick_device,
    pick_dtype,
    report_device,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--init_model", type=str, default="openai-community/gpt2")
    p.add_argument("--base_model", type=str, default="gpt2",
                   choices=["gpt2", "smollm2"],
                   help="Which model wrapper to use. 'gpt2' for GPT-2 "
                        "small via GPT2KRI; 'smollm2' for SmolLM2 via "
                        "SmolLM2KRI (RoPE+GQA, longer context).")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--mode", type=str, default="mixed", choices=["dense", "kri", "mixed"])

    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--block_size", type=int, default=16)
    p.add_argument("--local_windows", type=str, default="64,128,256")
    p.add_argument("--topk_blocks", type=str, default="2,4,8")
    p.add_argument("--prefill_splits", type=str, default="128,192,256,384")
    p.add_argument("--sparse_prob", type=float, default=0.7)
    p.add_argument("--per_head_mask", action="store_true",
                   help="Use strict per-head KRI selection (more memory).")
    p.add_argument("--use_novelty", action="store_true",
                   help="Enable the novelty term in the KRI score.")
    p.add_argument("--score_layer_index", type=int, default=6)

    # Dataset
    p.add_argument("--dataset_name", type=str, default="roneneldan/TinyStories")
    p.add_argument("--dataset_config", type=str, default=None)
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--streaming", type=str, default="false")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--val_split", type=str, default="validation")

    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--precision", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    p.add_argument("--grad_checkpointing", type=str, default="true")

    p.add_argument("--teacher_kl_alpha", type=float, default=0.0)

    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def _truthy(s: str) -> bool:
    return str(s).lower() in ("1", "true", "t", "yes", "y")


def _csv_ints(s: str):
    return tuple(int(x) for x in s.split(","))


def cosine_lr(step: int, base_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if total <= warmup:
        return base_lr
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_kri_for_batch(model: GPT2KRI, ids: torch.Tensor, cfg_template: KRIConfig,
                       seed: int) -> torch.Tensor:
    """Collect K/V from the student (frozen forward, no grad) and build
    the KRI mask. We use the student's own K/V so the routing scores
    are computed in the same space the model is being trained on.
    """
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    cfg = cfg_template.sample(rng)
    cfg.prefill_split = min(cfg.prefill_split or ids.shape[1] // 2, ids.shape[1] - 1)

    B, T = ids.shape
    H = model.cfg.n_head
    with torch.no_grad():
        kvs = model.collect_kv(ids)
    k_per_layer = [kv[0] for kv in kvs]
    v_per_layer = [kv[1] for kv in kvs]
    # Use the K projection as a stand-in for q for scoring; same space.
    q_per_layer = [kv[0] for kv in kvs]

    mask = build_kri_mask(
        cfg, T, B, H,
        k_per_layer=k_per_layer, v_per_layer=v_per_layer, q_per_layer=q_per_layer,
        device=ids.device,
    )
    return mask, cfg


def evaluate(model: GPT2KRI, val_iter, device, kri_cfg_template: KRIConfig,
            n_batches: int, dtype: torch.dtype):
    model.eval()
    losses_dense = []
    losses_sparse = []
    seen = 0
    with torch.no_grad():
        for batch in val_iter:
            if seen >= n_batches:
                break
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
                logits, loss_dense = model(ids, labels=labels)
            losses_dense.append(float(loss_dense.item()))

            mask, _ = build_kri_for_batch(model, ids, kri_cfg_template, seed=seen + 100000)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
                _, loss_sparse = model(ids, labels=labels, attn_mask=mask)
            losses_sparse.append(float(loss_sparse.item()))
            seen += 1
    model.train()
    mean_dense = sum(losses_dense) / max(1, len(losses_dense))
    mean_sparse = sum(losses_sparse) / max(1, len(losses_sparse))
    return mean_dense, mean_sparse


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    report_device()
    device = pick_device()
    dtype = pick_dtype(args.precision)
    print(f"precision dtype = {dtype}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "train_config.json"
    cfg_path.write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    banner(f"loading init model: {args.init_model}")
    model, teacher_factory = _load_base_model(args.base_model, args.init_model, device)
    if _truthy(args.grad_checkpointing):
        model.gradient_checkpointing_enable()
        print("gradient checkpointing: ON")
    model.train()

    teacher = None
    if args.teacher_kl_alpha > 0.0:
        banner("loading frozen teacher")
        teacher = teacher_factory()
        for p in teacher.parameters():
            p.requires_grad = False

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95),
        weight_decay=args.weight_decay, eps=1e-8,
    )

    banner("loading data")
    data_cfg = DataConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        text_column=args.text_column,
        streaming=_truthy(args.streaming),
        train_split=args.train_split,
        val_split=args.val_split,
        seq_len=args.seq_len,
    )
    tok = get_tokenizer(args.init_model)
    train_ds, val_ds = get_train_val_streams(data_cfg, tok)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate, num_workers=0)

    kri_template = KRIConfig(
        block_size=args.block_size,
        local_window_choices=_csv_ints(args.local_windows),
        topk_block_choices=_csv_ints(args.topk_blocks),
        prefill_split_choices=_csv_ints(args.prefill_splits),
        per_head=args.per_head_mask,
        use_novelty=args.use_novelty,
        score_layer_index=args.score_layer_index,
    )

    banner(f"training: mode={args.mode}  max_steps={args.max_steps}")
    timer = Timer()
    step = 0
    micro = 0
    optim.zero_grad(set_to_none=True)
    tokens_in_acc = 0
    loss_running = 0.0
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    metrics_log = (out_dir / "metrics.jsonl").open("w")

    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        B, T = ids.shape
        tokens_in_acc += B * T

        # Decide dense vs KRI per micro-batch.
        if args.mode == "dense":
            use_sparse = False
        elif args.mode == "kri":
            use_sparse = True
        else:
            use_sparse = (torch.rand(()).item() < args.sparse_prob)

        attn_mask = None
        if use_sparse:
            attn_mask, _ = build_kri_for_batch(model, ids, kri_template, seed=step * 997 + micro)

        # Teacher pass first (frozen, no grad) — uses dense attention.
        teacher_log_probs = None
        if teacher is not None and args.teacher_kl_alpha > 0.0:
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
                    t_logits, _ = teacher(ids)
            teacher_log_probs = F.log_softmax(t_logits.float(), dim=-1)

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
            logits, ce_loss = model(ids, labels=labels, attn_mask=attn_mask)

        loss = ce_loss
        if teacher_log_probs is not None:
            student_log_probs = F.log_softmax(logits.float(), dim=-1)
            # KL(student || teacher) = sum_x exp(student) * (student - teacher)
            # Use the symmetric forward-KL form used for distillation:
            # KL(teacher || student) on shifted positions, common for LM
            # distillation. We average over all (B, T-1, V).
            kl = F.kl_div(student_log_probs[:, :-1, :], teacher_log_probs[:, :-1, :].exp(),
                          reduction="batchmean", log_target=False)
            loss = ce_loss + args.teacher_kl_alpha * kl

        loss = loss / args.grad_accum
        loss.backward()
        loss_running += float(ce_loss.detach().item()) / args.grad_accum
        micro += 1

        if micro % args.grad_accum == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            lr_now = cosine_lr(step, args.lr, args.warmup_steps, args.max_steps)
            for g in optim.param_groups:
                g["lr"] = lr_now
            optim.step()
            optim.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                tps = tokens_in_acc / max(1e-6, timer.lap())
                # `loss_running` accumulated ce_loss/grad_accum over
                # `grad_accum * log_every` micro-batches, which sums to
                # `log_every` step-means. Divide to get per-step mean.
                avg_loss = loss_running / max(1, args.log_every)
                log = StepLog(
                    step=step, loss=avg_loss, lr=lr_now,
                    tokens_per_sec=tps, seconds=timer.total(),
                    extra={"mode": args.mode, "sparse_used": int(use_sparse)},
                )
                log_step(log)
                metrics_log.write(json.dumps({
                    "step": step, "loss": avg_loss, "lr": lr_now, "tokens_per_sec": tps,
                    "seconds_total": timer.total(), "sparse_used": int(use_sparse),
                }) + "\n")
                metrics_log.flush()
                loss_running = 0.0
                tokens_in_acc = 0
            if step > 0 and step % args.eval_every == 0:
                # Build a fresh val iter each time so we always start
                # from the same point — cheap deterministic-ish signal.
                try:
                    next(val_iter)
                except StopIteration:
                    pass
                val_iter = iter(val_loader)
                vd, vs = evaluate(model, val_iter, device, kri_template, args.eval_batches, dtype)
                print(f"[eval @ step {step}] val_loss_dense={vd:.4f} val_loss_sparse={vs:.4f}")
                metrics_log.write(json.dumps({
                    "step": step, "eval_loss_dense": vd, "eval_loss_sparse": vs,
                }) + "\n")
                metrics_log.flush()
                val_iter = iter(val_loader)
            if step > 0 and step % args.save_every == 0:
                ckpt = out_dir / f"checkpoint_step{step}.pt"
                torch.save({
                    "model": model.state_dict(),
                    "cfg": (asdict(model.cfg) if hasattr(model.cfg, '__dataclass_fields__')
                else {k: getattr(model.cfg, k) for k in
                      ('n_head','n_embd','n_layer','n_positions','vocab_size')
                      if hasattr(model.cfg, k)}),
                    "step": step,
                    "args": vars(args),
                }, ckpt)
                print(f"saved {ckpt}")

    # Final save.
    final = out_dir / "checkpoint_final.pt"
    torch.save({
        "model": model.state_dict(),
        "cfg": (asdict(model.cfg) if hasattr(model.cfg, '__dataclass_fields__')
                else {k: getattr(model.cfg, k) for k in
                      ('n_head','n_embd','n_layer','n_positions','vocab_size')
                      if hasattr(model.cfg, k)}),
        "step": step,
        "args": vars(args),
    }, final)
    metrics_log.close()
    print(f"DONE: total time {timer.total():.1f}s, steps {step}", flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    # PyTorch + HF datasets + ROCm have a known GIL-release race during
    # Python interpreter shutdown that fires an Abort *after* training
    # completes and the checkpoint is on disk. Bypass interpreter
    # finalizers and exit cleanly with the right status code so any
    # wrapper script can continue.
    os._exit(rc)

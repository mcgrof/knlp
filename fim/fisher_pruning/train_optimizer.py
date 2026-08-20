"""Phase B trainer: GPT-2 baseline with a pluggable optimizer.

Mirrors the spectral harness train-baseline loop (same data stream,
LR schedule shape, bf16 autocast, fork-ready checkpoints with full
optimizer state) but builds the optimizer by name:

  adamw  torch.optim.AdamW via the harness make_optimizer
  soap   vendored SOAP (fim/fisher_pruning/soap.py, MIT, from
         github.com/nikhilvyas/SOAP) — Adam in Shampoo's eigenbasis

Checkpoints keep the optimizer state so Shampoo-family factors
(GG/Q per parameter) can be harvested later as pruning signals.

  python3 fim/fisher_pruning/train_optimizer.py --config CFG
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fim.fisher_pruning.soap import SOAP  # noqa: E402
from fim.reciprocal_attention.gpt2_spectral_ra import (  # noqa: E402
    BinData,
    _amp_ctx,
    _jsonl,
    build_model,
    cosine_lr,
    evaluate,
    make_optimizer,
    save_checkpoint,
)
from fim.fisher_pruning.phase_a import _manifest  # noqa: E402


def build_optimizer(model: nn.Module, cfg: dict):
    name = cfg.get("optimizer", "adamw")
    if name == "adamw":
        return make_optimizer(model, cfg)
    if name == "soap":
        decay, no_decay = [], []
        for _, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        return SOAP(
            [
                {
                    "params": decay,
                    "weight_decay": cfg.get("weight_decay", 0.1),
                },
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=cfg["lr"],
            betas=tuple(cfg.get("betas", (0.95, 0.95))),
            precondition_frequency=cfg.get("precondition_frequency", 10),
        )
    raise ValueError(f"unknown optimizer {name!r}")


def cmd_train(cfg: dict, device: str) -> None:
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "train.jsonl"
    torch.manual_seed(cfg["seed"])
    model = build_model(cfg["model"], selection={})
    model.to(device)
    train_data = BinData(cfg["data"]["train_bin"], cfg["model"]["block_size"])
    val_data = BinData(cfg["data"]["val_bin"], cfg["model"]["block_size"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    val_gen_seed = cfg["seed"] + 1000
    tcfg = cfg["train"]
    optimizer = build_optimizer(model, tcfg)
    batch_counter = 0
    t0 = time.time()
    tokens_per_step = tcfg["batch_size"] * cfg["model"]["block_size"]
    _jsonl(
        log,
        _manifest(
            cfg,
            {"event": "start", "device": device, "optimizer": type(optimizer).__name__},
        ),
    )
    for step in range(tcfg["max_steps"]):
        lr = cosine_lr(step, tcfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = train_data.batch(tcfg["batch_size"], gen, device)
        batch_counter += 1
        with _amp_ctx(device):
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.get("clip", 1.0))
        optimizer.step()
        if step % tcfg.get("log_interval", 50) == 0:
            elapsed = time.time() - t0
            _jsonl(
                log,
                {
                    "event": "train",
                    "step": step,
                    "loss": float(loss.item()),
                    "lr": lr,
                    "elapsed_s": elapsed,
                    "tokens_per_s": tokens_per_step * (step + 1) / elapsed,
                },
            )
        if step % tcfg.get("eval_interval", 500) == 0 or step == tcfg["max_steps"] - 1:
            val_loss = evaluate(
                model,
                val_data,
                tcfg.get("eval_batches", 20),
                tcfg["batch_size"],
                device,
                seed=val_gen_seed,
            )
            _jsonl(
                log,
                {
                    "event": "eval",
                    "step": step,
                    "val_loss": val_loss,
                    "val_ppl": math.exp(val_loss),
                },
            )
        if (step > 0 and step % tcfg.get("ckpt_interval", 5000) == 0) or step == tcfg[
            "max_steps"
        ] - 1:
            save_checkpoint(
                out_dir / f"ckpt_step{step}.pt",
                model,
                optimizer,
                step,
                batch_counter,
                cfg,
            )
            save_checkpoint(
                out_dir / "ckpt_latest.pt", model, optimizer, step, batch_counter, cfg
            )
    _jsonl(log, {"event": "done", "elapsed_s": time.time() - t0})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    cmd_train(cfg, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())

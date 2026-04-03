#!/usr/bin/env python3
"""Matched GPT-2 baseline / surgical-RA harness for paper ablation.

Adapted from llama150m_matched.py for GPT-2 (124M).  Provides the same
three-arm comparison lane (baseline / FIM-trace / attention-derived) with
identical RA mixing, training loop, and evaluation infrastructure.

Key differences from the LLaMA harness:
- Uses HuggingFace GPT2LMHeadModel (learned positional embeddings, no RoPE)
- No GQA — GPT-2 has 12 query heads and 12 KV heads (MHA)
- Patches GPT2Attention.forward instead of LlamaAttention.forward
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2Config, GPT2LMHeadModel

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except Exception:
    SDPBackend = None
    sdpa_kernel = None


DEFAULT_CONFIG: Dict[str, Any] = {
    "run_name": "gpt2-baseline-smoke",
    "seed": 1337,
    "model": {
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "n_positions": 1024,
        "vocab_size": 50304,
        "resid_pdrop": 0.1,
        "embd_pdrop": 0.1,
        "attn_pdrop": 0.0,
        "activation_function": "gelu_new",
    },
    "training": {
        "train_steps": 8,
        "max_time": None,
        "eval_interval": 4,
        "eval_batches": 2,
        "batch_size": 8,
        "gradient_accumulation_steps": 1,
        "learning_rate": 6e-4,
        "min_lr": 6e-5,
        "warmup_steps": 0,
        "weight_decay": 0.1,
        "max_grad_norm": 1.0,
        "seq_len": 1024,
        "bf16": True,
    },
    "data": {
        "train_bin": "gpt2/data/finewebedu/train.bin",
        "val_bin": "gpt2/data/finewebedu/val.bin",
        "dtype": "uint16",
    },
    "attention": {
        "impl": "sdpa",
        "backend": "auto",
        "record_backend": True,
        "capture_attention_stats": False,
    },
    "optimizer": {
        "type": "adamw",
    },
    "ra": {
        "enabled": False,
        "alpha_std": 0.9375,
        "alpha_rec": 0.0625,
        "selection_file": "",
        "layers": {},
    },
    "fim": {
        "enabled": False,
        "batches": 4,
        "candidate_low_threshold": 0.15,
        "candidate_keep_top_k_layers": 3,
        "select_top_heads": 8,
        "head_score_metric": "exact_eigmax",
        "output_file": "configs/ra_surgical_gpt2.json",
    },
    "tracking": {
        "wandb": False,
        "wandb_project": "gpt2-paper-ablation",
        "wandb_mode": "offline",
        "log_jsonl": True,
        "out_dir": "out/gpt2-matched",
    },
}


# ── helpers ──────────────────────────────────────────────────────────────────


def deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    result = json.loads(json.dumps(base))
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


@dataclass
class AttentionStatsCollector:
    num_layers: int
    num_heads: int
    enabled: bool = False
    score_metric: str = "exact_eigmax"
    synced: bool = False
    layer_sum: torch.Tensor = field(init=False)
    layer_count: torch.Tensor = field(init=False)
    head_sum: torch.Tensor = field(init=False)
    head_count: torch.Tensor = field(init=False)
    head_score: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.layer_sum = torch.zeros(self.num_layers, dtype=torch.float64)
        self.layer_count = torch.zeros(self.num_layers, dtype=torch.float64)
        self.head_sum = torch.zeros(
            self.num_layers, self.num_heads, dtype=torch.float64
        )
        self.head_count = torch.zeros(
            self.num_layers, self.num_heads, dtype=torch.float64
        )
        self.head_score = torch.zeros(
            self.num_layers, self.num_heads, dtype=torch.float64
        )

    def _score_head(self, mean_attn: torch.Tensor) -> float:
        if self.score_metric == "inbound_mass_var":
            inbound = mean_attn.sum(dim=0)
            return float(inbound.var(unbiased=False).item())
        try:
            eigvals = torch.linalg.eigvals(mean_attn).real
            return float(eigvals.abs().max().item())
        except Exception:
            return float("nan")

    def update(self, layer_idx: int, attn_probs: torch.Tensor) -> None:
        if not self.enabled:
            return
        with torch.no_grad():
            probs = attn_probs.detach().float().cpu()
            head_trace = probs.square().mean(dim=(0, 2, 3))
            self.head_sum[layer_idx] += head_trace.double()
            self.head_count[layer_idx] += 1
            self.layer_sum[layer_idx] += head_trace.mean().item()
            self.layer_count[layer_idx] += 1
            mean_mats = probs.mean(dim=0)
            for h in range(mean_mats.shape[0]):
                score = self._score_head(mean_mats[h])
                cur = self.head_score[layer_idx, h].item()
                if math.isnan(cur) or score > cur:
                    self.head_score[layer_idx, h] = score

    def summary(self) -> Dict[str, Any]:
        per_layer_traces = []
        for i in range(self.num_layers):
            cnt = self.layer_count[i].item()
            per_layer_traces.append(float(self.layer_sum[i].item() / max(cnt, 1)))
        per_head_scores = self.head_score.tolist()
        return {
            "per_layer_traces": per_layer_traces,
            "per_head_max_eigenvalue": per_head_scores,
            "head_score_metric": self.score_metric,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
        }


@dataclass
class BackendRecorder:
    impl: str = "sdpa"
    backend: str = "auto"
    counts: Dict[str, int] = field(default_factory=dict)
    requested_backend_names: List[str] = field(default_factory=list)

    def note(self, tag: str, q, k, v, mask, dp, causal):
        key = f"{tag}:{self.impl}:{self.backend}"
        self.counts[key] = self.counts.get(key, 0) + 1

    def summary(self) -> Dict[str, Any]:
        return {
            "impl": self.impl,
            "backend": self.backend,
            "requested_backends": self.requested_backend_names,
            "call_counts": dict(self.counts),
        }


class JSONLLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def log(self, payload: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(payload, sort_keys=True) + "\n")
        self._fh.flush()


class MemmapTokenDataset:
    def __init__(
        self, path: Path, seq_len: int, vocab_size: int, dtype: str = "uint16"
    ):
        dt = {"uint16": np.uint16, "uint32": np.uint32, "int32": np.int32}[dtype]
        self.data = np.memmap(str(path), dtype=dt, mode="r")
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.n_tokens = len(self.data)

    def sample_batch(
        self,
        batch_size: int,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ):
        max_start = self.n_tokens - self.seq_len - 1
        starts = torch.randint(0, max_start, (batch_size,), generator=generator)
        x = torch.stack(
            [
                torch.from_numpy(self.data[s : s + self.seq_len].astype(np.int64))
                for s in starts
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    self.data[s + 1 : s + 1 + self.seq_len].astype(np.int64)
                )
                for s in starts
            ]
        )
        return x.to(device), y.to(device)


def resolve_sdpa_backend(name: str):
    if SDPBackend is None or name == "auto":
        return [], ["auto"]
    mapping = {
        "math": [SDPBackend.MATH],
        "flash": [SDPBackend.FLASH_ATTENTION],
        "efficient": [SDPBackend.EFFICIENT_ATTENTION],
        "cudnn": (
            [SDPBackend.CUDNN_ATTENTION]
            if hasattr(SDPBackend, "CUDNN_ATTENTION")
            else []
        ),
    }
    backends = mapping.get(name, [])
    return backends, [name]


def maybe_sdpa_backend(name: str):
    backends, _ = resolve_sdpa_backend(name)
    if backends and sdpa_kernel is not None:
        return sdpa_kernel(backends)
    return contextlib.nullcontext()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return world_size, rank, local_rank, device, True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 1, 0, 0, device, False


def barrier():
    if dist.is_initialized():
        dist.barrier()


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def rank0(rank: int) -> bool:
    return rank == 0


# ── model ────────────────────────────────────────────────────────────────────


def build_model(cfg: Dict[str, Any]) -> GPT2LMHeadModel:
    mcfg = cfg["model"]
    gpt2_cfg = GPT2Config(
        n_embd=mcfg["n_embd"],
        n_layer=mcfg["n_layer"],
        n_head=mcfg["n_head"],
        n_positions=mcfg.get("n_positions", 1024),
        vocab_size=mcfg.get("vocab_size", 50304),
        resid_pdrop=mcfg.get("resid_pdrop", 0.1),
        embd_pdrop=mcfg.get("embd_pdrop", 0.1),
        attn_pdrop=mcfg.get("attn_pdrop", 0.0),
        activation_function=mcfg.get("activation_function", "gelu_new"),
        scale_attn_weights=True,
        tie_word_embeddings=True,
        # Force SDPA attention for consistent RA patching
        _attn_implementation="sdpa",
    )
    model = GPT2LMHeadModel(gpt2_cfg)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"GPT2LMHeadModel: {n_params:.1f}M parameters", flush=True)
    return model


# ── RA patch ─────────────────────────────────────────────────────────────────


def parse_selection_file(path: Path) -> Dict[int, List[int]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    layers = raw.get("layers", raw)
    parsed: Dict[int, List[int]] = {}
    for k, v in layers.items():
        parsed[int(k)] = [int(x) for x in v]
    return parsed


@dataclass
class RAPatchContext:
    backend_recorder: BackendRecorder
    stats_collector: AttentionStatsCollector
    selected_heads: Dict[int, List[int]]
    ra_enabled: bool
    alpha_std: float
    alpha_rec: float
    impl: str
    backend_name: str


def resolve_repo_path(repo_root: Path, path_value: str) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    return repo_root / p


def apply_gpt2_sdpa_patch(
    model: GPT2LMHeadModel, cfg: Dict[str, Any], repo_root: Path
) -> RAPatchContext:
    num_layers = model.config.n_layer
    num_heads = model.config.n_head
    attn_cfg = cfg["attention"]
    ra_cfg = cfg["ra"]
    fim_cfg = cfg["fim"]

    ra_enabled = bool(ra_cfg.get("enabled", False))
    selected_heads: Dict[int, List[int]] = {}
    if ra_enabled:
        selected_heads = {
            int(k): list(map(int, v)) for k, v in ra_cfg.get("layers", {}).items()
        }
        if not selected_heads and ra_cfg.get("selection_file"):
            sf = ra_cfg["selection_file"]
            if sf:
                selected_heads = parse_selection_file(resolve_repo_path(repo_root, sf))

    backend_recorder = BackendRecorder(
        impl=attn_cfg.get("impl", "sdpa"), backend=attn_cfg.get("backend", "auto")
    )
    _, backend_names = resolve_sdpa_backend(attn_cfg.get("backend", "auto"))
    backend_recorder.requested_backend_names = backend_names
    stats_collector = AttentionStatsCollector(
        num_layers=num_layers,
        num_heads=num_heads,
        enabled=bool(attn_cfg.get("capture_attention_stats") or fim_cfg.get("enabled")),
        score_metric=str(fim_cfg.get("head_score_metric", "exact_eigmax")),
    )
    ctx = RAPatchContext(
        backend_recorder=backend_recorder,
        stats_collector=stats_collector,
        selected_heads=selected_heads,
        ra_enabled=ra_enabled,
        alpha_std=float(ra_cfg.get("alpha_std", 0.9375)),
        alpha_rec=float(ra_cfg.get("alpha_rec", 0.0625)),
        impl=attn_cfg.get("impl", "sdpa"),
        backend_name=attn_cfg.get("backend", "auto"),
    )

    for layer_idx, block in enumerate(model.transformer.h):
        block.attn.forward = MethodType(
            make_gpt2_patched_forward(ctx, layer_idx), block.attn
        )

    if ra_enabled:
        total_ra = sum(len(v) for v in selected_heads.items())
        layer_str = ", ".join(
            f"L{k}({len(v)}h)" for k, v in sorted(selected_heads.items())
        )
        print(
            f"RA enabled: {total_ra} heads across {len(selected_heads)} layers [{layer_str}]",
            flush=True,
        )
        print(f"  alpha_std={ctx.alpha_std}, alpha_rec={ctx.alpha_rec}", flush=True)

    return ctx


def make_gpt2_patched_forward(ctx: RAPatchContext, layer_idx: int):
    """Create a patched forward for GPT2Attention that adds RA overlay."""

    def forward(
        self,
        hidden_states,
        past_key_values=None,
        cache_position=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        **kwargs,
    ):
        # Standard QKV computation
        query_states, key_states, value_states = self.c_attn(hidden_states).split(
            self.split_size, dim=2
        )
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)
        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)

        # Cache update (for generation)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                {"cache_position": cache_position},
            )

        q = query_states.contiguous()
        k = key_states.contiguous()
        v = value_states.contiguous()

        scale = 1.0 / math.sqrt(self.head_dim)
        dropout_p = self.attn_dropout.p if self.training else 0.0
        is_causal = attention_mask is None and q.shape[2] > 1

        attn_mask = attention_mask
        if attn_mask is not None and attn_mask.ndim == 4:
            attn_mask = attn_mask[:, :, :, : k.shape[-2]]

        ctx.backend_recorder.note("standard", q, k, v, attn_mask, dropout_p, is_causal)

        # Attention stats collection (for FIM)
        if ctx.stats_collector.enabled:
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
            if attn_mask is not None:
                scores = scores + attn_mask.float()
            probs = scores.softmax(dim=-1)
            ctx.stats_collector.update(layer_idx, probs)

        # Standard SDPA
        with maybe_sdpa_backend(ctx.backend_name):
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                scale=scale,
                is_causal=is_causal,
            )

            # RA overlay on selected heads
            selected = ctx.selected_heads.get(layer_idx, []) if ctx.ra_enabled else []
            if selected:
                sel = torch.tensor(selected, device=q.device, dtype=torch.long)
                q_sel = q.index_select(1, sel)
                k_sel = k.index_select(1, sel)
                v_sel = v.index_select(1, sel)
                ctx.backend_recorder.note(
                    "reciprocal", k_sel, q_sel, v_sel, attn_mask, dropout_p, is_causal
                )
                rec_out = F.scaled_dot_product_attention(
                    k_sel,
                    q_sel,
                    v_sel,  # swapped Q and K
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    scale=scale,
                    is_causal=is_causal,
                )
                mixed = (
                    attn_output.index_select(1, sel) * ctx.alpha_std
                    + rec_out * ctx.alpha_rec
                )
                attn_output = attn_output.clone()
                attn_output.index_copy_(1, sel, mixed)

        # Reshape and project
        attn_output = (
            attn_output.transpose(1, 2)
            .reshape(*hidden_states.shape[:-1], -1)
            .contiguous()
        )
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, None

    return forward


# ── wandb ────────────────────────────────────────────────────────────────────


class MaybeWandb:
    def __init__(self, enabled, project, mode, run_name, config, rank):
        env_mode = os.environ.get("WANDB_MODE", mode).strip() or mode
        env_disabled = os.environ.get("WANDB_DISABLED", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.enabled = (
            enabled and rank == 0 and not env_disabled and env_mode != "disabled"
        )
        self.run = None
        if self.enabled:
            import wandb

            self.run = wandb.init(
                project=project, mode=env_mode, name=run_name, config=config
            )

    def log(self, payload, step=None):
        if self.run is not None:
            self.run.log(payload, step=step)

    def finish(self):
        if self.run is not None:
            self.run.finish()


# ── eval ─────────────────────────────────────────────────────────────────────


def eval_model(model, dataset, cfg, device, generator=None):
    model.eval()
    losses = []
    eval_batches = int(cfg["training"].get("eval_batches", 2))
    with torch.no_grad():
        for _ in range(eval_batches):
            x, y = dataset.sample_batch(
                cfg["training"]["batch_size"], device, generator
            )
            out = model(input_ids=x, labels=y)
            losses.append(float(out.loss.detach().cpu().item()))
    total_loss = float(sum(losses))
    total_batches = len(losses)
    if dist.is_initialized():
        reduce_device = (
            torch.device(f"cuda:{torch.cuda.current_device()}")
            if dist.get_backend() == "nccl"
            else torch.device("cpu")
        )
        stats = torch.tensor(
            [total_loss, float(total_batches)],
            dtype=torch.float64,
            device=reduce_device,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss = float(stats[0].item())
        total_batches = int(stats[1].item())
    model.train()
    return float(total_loss / max(1, total_batches))


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def generate_surgical_config(fim_summary, cfg):
    per_layer = fim_summary["per_layer_traces"]
    per_head = (
        fim_summary.get("per_head_scores") or fim_summary["per_head_max_eigenvalue"]
    )
    score_metric = fim_summary.get(
        "head_score_metric", cfg["fim"].get("head_score_metric", "exact_eigmax")
    )
    low_threshold = float(cfg["fim"].get("candidate_low_threshold", 0.15))
    keep_top_layers = int(cfg["fim"].get("candidate_keep_top_k_layers", 3))
    select_top_heads = int(cfg["fim"].get("select_top_heads", 8))

    ranked_layers = sorted(
        range(len(per_layer)), key=lambda i: per_layer[i], reverse=True
    )
    highest_layer = ranked_layers[0] if ranked_layers else None
    candidates = [
        i for i in ranked_layers if i != highest_layer and per_layer[i] >= low_threshold
    ][:keep_top_layers]
    if not candidates:
        candidates = [i for i in ranked_layers[1 : 1 + keep_top_layers]]

    scored_heads = []
    for layer_idx in candidates:
        for head_idx, score in enumerate(per_head[layer_idx]):
            scored_heads.append((float(score), layer_idx, head_idx))
    scored_heads.sort(reverse=True)
    chosen = scored_heads[:select_top_heads]

    layers = {}
    scores = {}
    for score, layer_idx, head_idx in chosen:
        layers.setdefault(str(layer_idx), []).append(head_idx)
        scores.setdefault(str(layer_idx), {})[str(head_idx)] = score
    for v in layers.values():
        v.sort()

    return {
        "model": "gpt2",
        "selection_method": f"attention-proxy-{score_metric}",
        "head_score_metric": score_metric,
        "candidate_layers": candidates,
        "candidate_layer_traces": {str(i): per_layer[i] for i in candidates},
        "selected_head_count": len(chosen),
        "layers": layers,
        "scores": scores,
    }


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="GPT-2 matched RA ablation harness")
    parser.add_argument("--config", default=None, help="JSON config path")
    parser.add_argument(
        "--override", action="append", default=[], help="key=value overrides"
    )
    args = parser.parse_args()

    if not args.config:
        parser.error("--config is required")

    cfg_path = Path(args.config)
    cfg = deep_update(DEFAULT_CONFIG, json.loads(cfg_path.read_text()))
    for item in args.override:
        key, value = item.split("=", 1)
        cur = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = json.loads(value)

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / cfg["tracking"].get("out_dir", "out/gpt2-matched")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg["tracking"]["out_dir"] = str(out_dir)

    world_size, rank, local_rank, device, distributed = init_distributed()
    set_seed(int(cfg.get("seed", 1337)) + rank)

    run_name = cfg.get("run_name", cfg_path.stem)
    jsonl = (
        JSONLLogger(out_dir / f"{run_name}.jsonl")
        if rank0(rank) and cfg["tracking"].get("log_jsonl", True)
        else None
    )
    wandb_logger = MaybeWandb(
        enabled=bool(cfg["tracking"].get("wandb", False)),
        project=cfg["tracking"].get("wandb_project", "gpt2-paper-ablation"),
        mode=cfg["tracking"].get("wandb_mode", "offline"),
        run_name=run_name,
        config=cfg,
        rank=rank,
    )

    if rank0(rank):
        save_json(out_dir / f"{run_name}.resolved.json", cfg)

    model = build_model(cfg).to(device)
    patch_ctx = apply_gpt2_sdpa_patch(model, cfg, repo_root)

    train_ds = MemmapTokenDataset(
        repo_root / cfg["data"]["train_bin"],
        cfg["training"]["seq_len"],
        cfg["model"]["vocab_size"],
        cfg["data"].get("dtype", "uint16"),
    )
    val_ds = MemmapTokenDataset(
        repo_root / cfg["data"]["val_bin"],
        cfg["training"]["seq_len"],
        cfg["model"]["vocab_size"],
        cfg["data"].get("dtype", "uint16"),
    )

    use_bf16 = (
        bool(cfg["training"].get("bf16", False))
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    autocast = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if use_bf16
        else contextlib.nullcontext()
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"].get("weight_decay", 0.1),
    )

    if distributed:
        ddp_kwargs = (
            {"device_ids": [local_rank], "output_device": local_rank}
            if device.type == "cuda"
            else {}
        )
        model = DDP(model, **ddp_kwargs)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(cfg.get("seed", 1337)) + rank)

    startup = {
        "event": "startup",
        "run_name": run_name,
        "rank": rank,
        "world_size": world_size,
        "device": str(device),
        "distributed": distributed,
        "ra_enabled": patch_ctx.ra_enabled,
        "ra_heads": {str(k): v for k, v in patch_ctx.selected_heads.items()},
    }
    print(json.dumps(startup, sort_keys=True), flush=True)
    if jsonl:
        jsonl.log(startup)

    training_cfg = cfg["training"]
    train_steps = int(training_cfg["train_steps"])
    eval_interval = int(training_cfg.get("eval_interval", max(1, train_steps)))
    grad_accum = int(training_cfg.get("gradient_accumulation_steps", 1))
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
    max_time = training_cfg.get("max_time")
    max_time = None if max_time in (None, "", 0) else float(max_time)

    model.train()
    started = time.time()
    completed_steps = 0
    exit_reason = "max_steps"

    for step in range(1, train_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for micro in range(grad_accum):
            x, y = train_ds.sample_batch(training_cfg["batch_size"], device, generator)
            with autocast:
                out = model(input_ids=x, labels=y)
                loss = out.loss / grad_accum
            loss.backward()
            step_loss += float(loss.detach().cpu().item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        completed_steps = step

        elapsed_s = time.time() - started
        time_limit_reached = max_time is not None and elapsed_s >= max_time

        if dist.is_initialized():
            stop_tensor = torch.tensor(
                [int(time_limit_reached)], dtype=torch.int32, device=device
            )
            dist.broadcast(stop_tensor, src=0)
            time_limit_reached = bool(stop_tensor.item())

        should_eval = (
            step == 1
            or step % eval_interval == 0
            or step == train_steps
            or time_limit_reached
        )
        stop_elapsed = round(elapsed_s, 3)

        if should_eval:
            eval_loss = eval_model(model, val_ds, cfg, device, generator)
            payload = {
                "event": "eval",
                "step": step,
                "train_loss": step_loss,
                "eval_loss": eval_loss,
                "perplexity": float(math.exp(min(eval_loss, 20.0))),
                "elapsed_s": stop_elapsed,
            }
            if rank0(rank):
                print(json.dumps(payload, sort_keys=True), flush=True)
                if jsonl:
                    jsonl.log(payload)
                wandb_logger.log(
                    {k: v for k, v in payload.items() if k not in {"event"}},
                    step=step,
                )

        barrier()
        if time_limit_reached:
            exit_reason = "max_time"
            break

    if exit_reason != "max_time":
        stop_elapsed = round(time.time() - started, 3)

    # Save checkpoint
    if rank0(rank):
        ckpt_path = out_dir / f"{run_name}.checkpoint.pt"
        raw = model.module if distributed else model
        try:
            torch.save(
                {
                    "model_state_dict": raw.state_dict(),
                    "config": cfg,
                    "completed_steps": completed_steps,
                    "exit_reason": exit_reason,
                },
                ckpt_path,
            )
            print(
                json.dumps(
                    {"event": "checkpoint_saved", "path": str(ckpt_path)},
                    sort_keys=True,
                ),
                flush=True,
            )
        except Exception as e:
            print(
                json.dumps(
                    {"event": "checkpoint_save_failed", "error": repr(e)},
                    sort_keys=True,
                ),
                flush=True,
            )

    # FIM summary
    fim_summary = (
        patch_ctx.stats_collector.summary()
        if patch_ctx.stats_collector.enabled
        else None
    )
    if rank0(rank):
        backend_summary = patch_ctx.backend_recorder.summary()
        save_json(out_dir / f"{run_name}.backend.json", backend_summary)

        if fim_summary is not None:
            save_json(out_dir / f"{run_name}.fim_summary.json", fim_summary)
            generated = generate_surgical_config(fim_summary, cfg)
            save_json(out_dir / f"{run_name}.generated_selection.json", generated)
            print(
                json.dumps({"event": "fim_summary", **fim_summary}, sort_keys=True),
                flush=True,
            )

        elapsed_total = round(time.time() - started, 3)
        done = {
            "event": "complete",
            "run_name": run_name,
            "stop_elapsed_s": stop_elapsed,
            "total_elapsed_s": elapsed_total,
            "completed_steps": completed_steps,
            "exit_reason": exit_reason,
        }
        print(json.dumps(done, sort_keys=True), flush=True)
        if jsonl:
            jsonl.log(done)

    barrier()
    wandb_logger.finish()
    barrier()
    cleanup_distributed()
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Matched LLaMA-150M baseline / surgical-RA harness.

This script exists because knlp's existing training path is GPT-2-centric and
there is not currently a first-class LLaMA-150M reciprocal-attention runner in
this repo. The goal here is to provide the minimum clean comparison lane
requested by FIM-RA-LLAMA-150:

- baseline and RA both use the same SDPA-family attention core
- no torch.compile
- cheap smoke runs before cloud allocation
- optional DDP smoke on CPU (gloo) or GPU (nccl)
- optional attention-stat collection to generate a surgical-head JSON

The RA path is intentionally conservative: it reuses the same SDPA call shape as
baseline and only mixes an additional reciprocal SDPA output on selected heads.
That preserves backend-family parity while giving us a real distributed smoke
path in knlp.
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
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

import functools

try:
    from torch.distributed.fsdp import (
        FullStateDictConfig,
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
        StateDictType,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except Exception:  # pragma: no cover - older torch fallback
    SDPBackend = None
    sdpa_kernel = None


DEFAULT_CONFIG: Dict[str, Any] = {
    "run_name": "llama150m-baseline-smoke",
    "seed": 1337,
    "model": {
        "hidden_size": 512,
        "intermediate_size": 1792,
        "num_hidden_layers": 6,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "max_position_embeddings": 1024,
        "vocab_size": 50304,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
    },
    "training": {
        "train_steps": 8,
        "max_time": None,
        "eval_interval": 4,
        "eval_batches": 2,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "learning_rate": 3e-4,
        "min_lr": 3e-5,
        "warmup_steps": 0,
        "weight_decay": 0.1,
        "max_grad_norm": 1.0,
        "seq_len": 128,
        "bf16": False,
    },
    "data": {
        "train_bin": "gpt2/data/finewebedu/train.bin",
        "val_bin": "gpt2/data/finewebedu/val.bin",
        "dtype": "uint16",
    },
    "attention": {
        "impl": "sdpa",
        "backend": "auto",  # auto|math|flash|efficient|cudnn
        "record_backend": True,
        "capture_attention_stats": False,
    },
    "optimizer": {
        "type": "adamw",
        "spam_theta": 50.0,
        "spam_enable_clip": True,
        "spam_interval": 1000,
        "spam_warmup_steps": 1000,
    },
    "ra": {
        "enabled": False,
        "alpha_std": 0.9375,
        "alpha_rec": 0.0625,
        "selection_file": "configs/ra_surgical_llama150m.json",
        "layers": {},
    },
    "fim": {
        "enabled": False,
        "batches": 4,
        "candidate_low_threshold": 0.15,
        "candidate_keep_top_k_layers": 3,
        "select_top_heads": 8,
        "head_score_metric": "exact_eigmax",
        "output_file": "configs/ra_surgical_llama150m.json",
    },
    "tracking": {
        "wandb": False,
        "wandb_project": "llama150m-ra-surgical-b200x4",
        "wandb_mode": "offline",
        "log_jsonl": True,
        "out_dir": "out/llama150m-matched",
    },
}


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
        # attn_probs: [B, H, T, T]
        with torch.no_grad():
            probs = attn_probs.detach().float().cpu()
            head_trace = probs.square().mean(dim=(0, 2, 3))
            self.head_sum[layer_idx] += head_trace.double()
            self.head_count[layer_idx] += 1
            self.layer_sum[layer_idx] += head_trace.mean().item()
            self.layer_count[layer_idx] += 1
            mean_mats = probs.mean(dim=0)  # [H, T, T]
            for h in range(mean_mats.shape[0]):
                score = self._score_head(mean_mats[h])
                cur = self.head_score[layer_idx, h].item()
                if math.isnan(cur) or score > cur:
                    self.head_score[layer_idx, h] = score

    def sync_distributed(self) -> None:
        if self.synced or not dist.is_initialized():
            return
        backend = dist.get_backend()
        sync_device = (
            torch.device(f"cuda:{torch.cuda.current_device()}")
            if backend == "nccl"
            else torch.device("cpu")
        )

        def reduce_sum(name: str) -> None:
            value = getattr(self, name).to(sync_device)
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            setattr(self, name, value.cpu())

        def reduce_max(name: str) -> None:
            value = getattr(self, name).to(sync_device)
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
            setattr(self, name, value.cpu())

        reduce_sum("layer_sum")
        reduce_sum("layer_count")
        reduce_sum("head_sum")
        reduce_sum("head_count")
        reduce_max("head_score")
        self.synced = True

    def summary(self) -> Dict[str, Any]:
        self.sync_distributed()
        layer_trace = torch.where(
            self.layer_count > 0,
            self.layer_sum / torch.clamp(self.layer_count, min=1),
            torch.zeros_like(self.layer_sum),
        )
        head_trace = torch.where(
            self.head_count > 0,
            self.head_sum / torch.clamp(self.head_count, min=1),
            torch.zeros_like(self.head_sum),
        )
        return {
            "per_layer_traces": [float(x) for x in layer_trace.tolist()],
            "per_head_traces": [[float(y) for y in row] for row in head_trace.tolist()],
            "head_score_metric": self.score_metric,
            "per_head_scores": [[float(y) for y in row] for row in self.head_score.tolist()],
            "per_head_max_eigenvalue": [
                [float(y) for y in row] for row in self.head_score.tolist()
            ],
        }


@dataclass
class BackendRecorder:
    impl: str
    backend: str
    requested_backend_names: List[str] = field(default_factory=list)
    calls: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def note(
        self,
        label: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        dropout_p: float,
        is_causal: bool,
    ) -> None:
        if label in self.calls:
            return
        with torch.no_grad():
            probe = infer_runtime_backend(
                q=q,
                k=k,
                v=v,
                mask=mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                requested_backend_names=self.requested_backend_names,
            )
            self.calls[label] = {
                "actual_backend": probe.get("actual_backend", "unknown"),
                "probe": probe,
                "sample": {
                    "device": str(q.device),
                    "dtype": str(q.dtype),
                    "shape": list(q.shape),
                    "mask": None if mask is None else list(mask.shape),
                    "dropout_p": float(dropout_p),
                    "is_causal": bool(is_causal),
                },
            }

    def summary(self) -> Dict[str, Any]:
        standard = self.calls.get("standard", {})
        reciprocal = self.calls.get("reciprocal", {})
        parity_ok = True
        if standard and reciprocal:
            parity_ok = standard.get("actual_backend") == reciprocal.get(
                "actual_backend"
            )
        return {
            "impl": self.impl,
            "requested_backend": self.backend,
            "requested_backend_names": list(self.requested_backend_names),
            "actual_backend": standard.get("actual_backend", "unknown"),
            "calls": self.calls,
            "parity_ok": parity_ok,
        }


class JSONLLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            backup = self.path.with_suffix(
                self.path.suffix + f".prev-{int(time.time())}"
            )
            self.path.replace(backup)

    def log(self, payload: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True) + "\n")


class MemmapTokenDataset:
    def __init__(
        self, path: Path, seq_len: int, vocab_size: int, dtype: str = "uint16"
    ) -> None:
        self.path = path
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        np_dtype = np.uint16 if dtype == "uint16" else np.int32
        self.data = np.memmap(path, dtype=np_dtype, mode="r")
        self.length = int(max(0, len(self.data) - seq_len - 1))

    def sample_batch(
        self,
        batch_size: int,
        device: torch.device,
        generator: Optional[torch.Generator],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.length <= 0:
            raise RuntimeError(
                f"dataset too short for seq_len={self.seq_len}: {self.path}"
            )
        starts = torch.randint(0, self.length, (batch_size,), generator=generator)
        xs = []
        ys = []
        for start in starts.tolist():
            chunk = np.array(
                self.data[start : start + self.seq_len + 1], dtype=np.int64
            )
            chunk %= self.vocab_size
            xs.append(torch.from_numpy(chunk[:-1]))
            ys.append(torch.from_numpy(chunk[1:]))
        x = torch.stack(xs).to(device=device, dtype=torch.long)
        y = torch.stack(ys).to(device=device, dtype=torch.long)
        return x, y


def backend_name(value: Any) -> str:
    try:
        return str(value).split(".")[-1]
    except Exception:
        return str(value)


def make_sdpa_params(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor],
    dropout_p: float,
    is_causal: bool,
) -> Any:
    ctor = torch.backends.cuda.SDPAParams
    last_exc: Optional[TypeError] = None
    for args in (
        (q, k, v, mask, float(dropout_p), bool(is_causal)),
        (q, k, v, mask, float(dropout_p), bool(is_causal), False),
    ):
        try:
            return ctor(*args)
        except TypeError as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("unable to construct SDPAParams")


def infer_runtime_backend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor],
    dropout_p: float,
    is_causal: bool,
    requested_backend_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "device": str(q.device),
        "dtype": str(q.dtype),
        "requested_backend_names": list(requested_backend_names or []),
        "eligible_backends": [],
        "priority_order": [],
        "global_backend_state": {},
        "actual_backend": "unknown",
    }
    if q.device.type != "cuda":
        summary["eligible_backends"] = ["MATH"]
        summary["actual_backend"] = "MATH"
        return summary

    try:
        summary["device_name"] = torch.cuda.get_device_name(q.device)
    except Exception:
        summary["device_name"] = "cuda"

    try:
        priority = [backend_name(x) for x in torch._C._get_sdp_priority_order()]
    except Exception:
        priority = []
    summary["priority_order"] = priority

    global_state = {
        "flash_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "efficient_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "cudnn_enabled": bool(torch.backends.cuda.cudnn_sdp_enabled()),
        "math_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
    }
    summary["global_backend_state"] = global_state

    availability: Dict[str, Dict[str, Any]] = {}
    try:
        params = make_sdpa_params(q, k, v, mask, float(dropout_p), bool(is_causal))
    except Exception as exc:
        summary["actual_backend"] = f"probe_error:{type(exc).__name__}"
        summary["probe_error"] = f"{type(exc).__name__}: {exc}"
        summary["probe_context"] = {
            "q_shape": list(q.shape),
            "k_shape": list(k.shape),
            "v_shape": list(v.shape),
            "mask_shape": None if mask is None else list(mask.shape),
            "dropout_p": float(dropout_p),
            "is_causal": bool(is_causal),
        }
        return summary

    checks = [
        (
            "FLASH_ATTENTION",
            global_state["flash_enabled"],
            getattr(torch.backends.cuda, "can_use_flash_attention", None),
        ),
        (
            "EFFICIENT_ATTENTION",
            global_state["efficient_enabled"],
            getattr(torch.backends.cuda, "can_use_efficient_attention", None),
        ),
        (
            "CUDNN_ATTENTION",
            global_state["cudnn_enabled"],
            getattr(torch.backends.cuda, "can_use_cudnn_attention", None),
        ),
    ]
    eligible: List[str] = []
    for name, enabled, fn in checks:
        entry: Dict[str, Any] = {"enabled": bool(enabled), "eligible": False}
        if enabled and fn is not None:
            try:
                entry["eligible"] = bool(fn(params, debug=False))
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
        elif enabled and fn is None:
            entry["error"] = "backend probe helper unavailable in this torch build"
        if entry["eligible"]:
            eligible.append(name)
        availability[name] = entry

    availability["MATH"] = {
        "enabled": global_state["math_enabled"],
        "eligible": global_state["math_enabled"],
    }
    if global_state["math_enabled"]:
        eligible.append("MATH")

    summary["availability"] = availability
    summary["eligible_backends"] = eligible

    runnable_backends: List[str] = []
    probe_errors: Dict[str, str] = {}
    q_probe = q[:1, :, : min(8, q.shape[2]), :].contiguous()
    k_probe = k[:1, :, : min(8, k.shape[2]), :].contiguous()
    v_probe = v[:1, :, : min(8, v.shape[2]), :].contiguous()
    mask_probe = None
    if mask is not None:
        mask_probe = mask[:1, :, : q_probe.shape[2], : k_probe.shape[2]].contiguous()
    force_map = {
        "FLASH_ATTENTION": dict(
            enable_flash=True,
            enable_mem_efficient=False,
            enable_math=False,
            enable_cudnn=False,
        ),
        "EFFICIENT_ATTENTION": dict(
            enable_flash=False,
            enable_mem_efficient=True,
            enable_math=False,
            enable_cudnn=False,
        ),
        "CUDNN_ATTENTION": dict(
            enable_flash=False,
            enable_mem_efficient=False,
            enable_math=False,
            enable_cudnn=True,
        ),
        "MATH": dict(
            enable_flash=False,
            enable_mem_efficient=False,
            enable_math=True,
            enable_cudnn=False,
        ),
    }
    for name, kwargs in force_map.items():
        try:
            with torch.backends.cuda.sdp_kernel(**kwargs):
                F.scaled_dot_product_attention(
                    q_probe,
                    k_probe,
                    v_probe,
                    attn_mask=mask_probe,
                    dropout_p=float(dropout_p),
                    is_causal=bool(is_causal and mask_probe is None),
                )
            torch.cuda.synchronize(q.device)
            runnable_backends.append(name)
        except Exception as exc:
            probe_errors[name] = f"{type(exc).__name__}: {exc}"
    summary["runnable_backends"] = runnable_backends
    if probe_errors:
        summary["probe_errors"] = probe_errors

    requested = list(requested_backend_names or [])
    if not requested:
        requested = priority or [
            "FLASH_ATTENTION",
            "EFFICIENT_ATTENTION",
            "CUDNN_ATTENTION",
            "MATH",
        ]

    actual_candidates = runnable_backends or eligible
    for name in requested:
        if name in actual_candidates:
            summary["actual_backend"] = name
            break
    else:
        summary["actual_backend"] = (
            actual_candidates[0] if actual_candidates else "none-eligible"
        )
    return summary


def resolve_sdpa_backend(name: str) -> Tuple[List[Any], List[str]]:
    if sdpa_kernel is None or SDPBackend is None:
        return [], []
    mapping = {
        "math": [SDPBackend.MATH],
        "flash": [SDPBackend.FLASH_ATTENTION],
        "efficient": [SDPBackend.EFFICIENT_ATTENTION],
        "cudnn": [SDPBackend.CUDNN_ATTENTION],
        "auto": [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.MATH,
        ],
    }
    backends = mapping.get(name, mapping["auto"])
    names = [str(x).split(".")[-1] for x in backends]
    return backends, names


@contextlib.contextmanager
def maybe_sdpa_backend(name: str):
    backends, _ = resolve_sdpa_backend(name)
    if sdpa_kernel is None or not backends:
        yield
    else:
        with sdpa_kernel(backends=backends):
            yield


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_distributed() -> Tuple[int, int, int, torch.device, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return world_size, rank, local_rank, device, distributed


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def distributed_max_float(value: float, device: torch.device) -> float:
    if not dist.is_initialized():
        return float(value)
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def rank0(rank: int) -> bool:
    return rank == 0


def build_model(cfg: Dict[str, Any]) -> LlamaForCausalLM:
    m = cfg["model"]
    attn_cfg = cfg["attention"]
    llama_cfg = LlamaConfig(
        hidden_size=m["hidden_size"],
        intermediate_size=m["intermediate_size"],
        num_hidden_layers=m["num_hidden_layers"],
        num_attention_heads=m["num_attention_heads"],
        num_key_value_heads=m["num_key_value_heads"],
        max_position_embeddings=m["max_position_embeddings"],
        vocab_size=m["vocab_size"],
        rms_norm_eps=m.get("rms_norm_eps", 1e-6),
        rope_theta=m.get("rope_theta", 10000.0),
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    llama_cfg._attn_implementation = attn_cfg.get("impl", "sdpa")
    # For FSDP with large models, build in bf16 to halve CPU memory
    init_dtype = None
    if cfg.get("distributed_strategy") == "fsdp" and cfg["training"].get("bf16"):
        init_dtype = torch.bfloat16
    model = LlamaForCausalLM(llama_cfg)
    if init_dtype is not None:
        model = model.to(dtype=init_dtype)
    model.config.use_cache = False
    return model


def resolve_repo_path(repo_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


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


def apply_llama_sdpa_patch(
    model: LlamaForCausalLM, cfg: Dict[str, Any], repo_root: Path
) -> RAPatchContext:
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
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
            selected_heads = parse_selection_file(
                resolve_repo_path(repo_root, ra_cfg["selection_file"])
            )

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

    for layer_idx, layer in enumerate(model.model.layers):
        layer.self_attn.forward = MethodType(
            make_patched_forward(ctx, layer_idx), layer.self_attn
        )

    return ctx


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match query heads for GQA models."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def make_patched_forward(ctx: RAPatchContext, layer_idx: int):
    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # GQA: expand KV heads to match query heads
        n_kv_groups = getattr(self, "num_key_value_groups", 1)
        key_states = _repeat_kv(key_states, n_kv_groups)
        value_states = _repeat_kv(value_states, n_kv_groups)

        scale = self.scaling
        is_causal = attention_mask is None and query_states.shape[2] > 1
        attn_mask = attention_mask
        if attn_mask is not None and attn_mask.ndim == 4:
            attn_mask = attn_mask[:, :, :, : key_states.shape[-2]]

        q = query_states.contiguous()
        k = key_states.contiguous()
        v = value_states.contiguous()
        dropout_p = 0.0 if not self.training else self.attention_dropout
        ctx.backend_recorder.note("standard", q, k, v, attn_mask, dropout_p, is_causal)

        if ctx.stats_collector.enabled:
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
            if attn_mask is not None:
                scores = scores + attn_mask.float()
            probs = scores.softmax(dim=-1)
            ctx.stats_collector.update(layer_idx, probs)

        if ctx.impl != "sdpa":
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
            if attn_mask is not None:
                scores = scores + attn_mask.float()
            probs = scores.softmax(dim=-1).to(q.dtype)
            attn_output = torch.matmul(probs, v)
        else:
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

                selected = (
                    ctx.selected_heads.get(layer_idx, []) if ctx.ra_enabled else []
                )
                if selected:
                    sel = torch.tensor(selected, device=q.device, dtype=torch.long)
                    q_sel = q.index_select(1, sel)
                    k_sel = k.index_select(1, sel)
                    v_sel = v.index_select(1, sel)
                    ctx.backend_recorder.note(
                        "reciprocal",
                        k_sel,
                        q_sel,
                        v_sel,
                        attn_mask,
                        dropout_p,
                        is_causal,
                    )
                    rec_out = F.scaled_dot_product_attention(
                        k_sel,
                        q_sel,
                        v_sel,
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

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    return forward


class MaybeWandb:
    def __init__(
        self,
        enabled: bool,
        project: str,
        mode: str,
        run_name: str,
        config: Dict[str, Any],
        rank: int,
    ):
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
            import wandb  # noqa: F401

            self.run = wandb.init(
                project=project, mode=env_mode, name=run_name, config=config
            )

    def log(self, payload: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.run is not None:
            self.run.log(payload, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def eval_model(
    model: torch.nn.Module,
    dataset: MemmapTokenDataset,
    cfg: Dict[str, Any],
    device: torch.device,
    generator: Optional[torch.Generator],
) -> float:
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


def generate_surgical_config(
    fim_summary: Dict[str, Any], cfg: Dict[str, Any]
) -> Dict[str, Any]:
    per_layer = fim_summary["per_layer_traces"]
    per_head = fim_summary.get("per_head_scores") or fim_summary["per_head_max_eigenvalue"]
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

    scored_heads: List[Tuple[float, int, int]] = []
    for layer_idx in candidates:
        for head_idx, score in enumerate(per_head[layer_idx]):
            scored_heads.append((float(score), layer_idx, head_idx))
    scored_heads.sort(reverse=True)
    chosen = scored_heads[:select_top_heads]

    layers: Dict[str, List[int]] = {}
    scores: Dict[str, Dict[str, float]] = {}
    for score, layer_idx, head_idx in chosen:
        layers.setdefault(str(layer_idx), []).append(head_idx)
        scores.setdefault(str(layer_idx), {})[str(head_idx)] = score
    for v in layers.values():
        v.sort()

    model_name = cfg.get("run_name", "unknown")
    if "llama1b" in model_name or "llama-1b" in model_name:
        model_label = "llama-1b"
    elif "llama150m" in model_name or "llama-150m" in model_name:
        model_label = "llama-150m"
    else:
        model_label = model_name.split("-")[0] if model_name else "unknown"

    return {
        "model": model_label,
        "selection_method": f"attention-proxy-{score_metric}",
        "head_score_metric": score_metric,
        "candidate_layers": candidates,
        "candidate_layer_traces": {str(i): per_layer[i] for i in candidates},
        "selected_head_count": len(chosen),
        "layers": layers,
        "scores": scores,
    }


def derive_topk(input_path: str, topk: int, output_path: str) -> int:
    """Trim a surgical selection JSON to the top-k highest-scoring heads."""
    src = json.loads(Path(input_path).read_text())
    scores = src.get("scores", {})
    scored_heads = []
    for layer, heads in scores.items():
        for head, score in heads.items():
            scored_heads.append((float(score), layer, head))
    scored_heads.sort(reverse=True)
    chosen = scored_heads[:topk]

    layers: Dict[str, List[int]] = {}
    new_scores: Dict[str, Dict[str, float]] = {}
    for score, layer, head in chosen:
        layers.setdefault(layer, []).append(int(head))
        new_scores.setdefault(layer, {})[head] = score
    for v in layers.values():
        v.sort()

    result = {
        "candidate_layer_traces": src.get("candidate_layer_traces", {}),
        "candidate_layers": src.get("candidate_layers", []),
        "layers": layers,
        "model": src.get("model", "llama-150m"),
        "scores": new_scores,
        "selected_head_count": len(chosen),
        "selection_method": src.get("selection_method", "unknown") + f"+top{topk}-trim",
        "source_selection_file": input_path,
    }
    save_json(Path(output_path), result)
    print(
        json.dumps(
            {
                "event": "derive_topk",
                "input": input_path,
                "topk": topk,
                "output": output_path,
                "heads": len(chosen),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def eval_hellaswag(
    model: torch.nn.Module,
    cfg: Dict[str, Any],
    device: torch.device,
    max_examples: int = 1000,
) -> Dict[str, Any]:
    """Evaluate on HellaSwag (completion selection accuracy)."""
    try:
        from datasets import load_dataset
    except ImportError:
        return {"error": "datasets library not installed", "accuracy": -1.0}

    ds = load_dataset("Rowan/hellaswag", split="validation")

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as exc:
        return {"error": f"tokenizer load failed: {exc}", "accuracy": -1.0}

    model.eval()
    correct = 0
    total = 0
    n = min(max_examples, len(ds))

    with torch.no_grad():
        for i in range(n):
            ex = ds[i]
            ctx_text = ex["ctx"]
            endings = ex["endings"]
            label = int(ex["label"])

            scores = []
            for ending in endings:
                text = ctx_text + " " + ending
                tokens = tokenizer.encode(text, return_tensors="pt").to(device)
                if tokens.shape[1] > cfg["model"].get("max_position_embeddings", 2048):
                    tokens = tokens[:, : cfg["model"]["max_position_embeddings"]]
                out = model(input_ids=tokens)
                logits = out.logits if hasattr(out, "logits") else out[0]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = tokens[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="mean",
                )
                scores.append(-float(loss.item()))

            pred = max(range(len(scores)), key=lambda j: scores[j])
            if pred == label:
                correct += 1
            total += 1

            if total % 200 == 0:
                print(
                    json.dumps(
                        {
                            "event": "hellaswag_progress",
                            "done": total,
                            "of": n,
                            "acc_so_far": round(correct / total, 4),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    accuracy = correct / max(total, 1)
    return {
        "task": "hellaswag",
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
    }


def eval_lambada(
    model: torch.nn.Module,
    cfg: Dict[str, Any],
    device: torch.device,
    max_examples: int = 1000,
) -> Dict[str, Any]:
    """Evaluate on LAMBADA (last-word prediction accuracy)."""
    try:
        from datasets import load_dataset
    except ImportError:
        return {"error": "datasets library not installed", "accuracy": -1.0}

    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as exc:
        return {"error": f"tokenizer load failed: {exc}", "accuracy": -1.0}

    model.eval()
    correct = 0
    total = 0
    n = min(max_examples, len(ds))

    with torch.no_grad():
        for i in range(n):
            text = ds[i]["text"]
            tokens = tokenizer.encode(text, return_tensors="pt").to(device)
            if tokens.shape[1] < 2:
                continue
            if tokens.shape[1] > cfg["model"].get("max_position_embeddings", 2048):
                tokens = tokens[:, : cfg["model"]["max_position_embeddings"]]

            last_word = text.split()[-1]
            last_word_tokens = tokenizer.encode(" " + last_word)
            n_last = len(last_word_tokens)

            input_ids = tokens[:, :-1]
            out = model(input_ids=input_ids)
            logits = out.logits if hasattr(out, "logits") else out[0]

            predicted_ids = logits[:, -n_last:, :].argmax(dim=-1).squeeze(0)
            target_ids = tokens[:, -n_last:].squeeze(0)

            if torch.equal(predicted_ids, target_ids):
                correct += 1
            total += 1

    accuracy = correct / max(total, 1)
    return {
        "task": "lambada",
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
    }


def eval_winogrande(
    model: torch.nn.Module,
    cfg: Dict[str, Any],
    device: torch.device,
    max_examples: int = 1000,
) -> Dict[str, Any]:
    """Evaluate on Winogrande (coreference-style sentence completion)."""
    try:
        from datasets import load_dataset
    except ImportError:
        return {"error": "datasets library not installed", "accuracy": -1.0}

    try:
        ds = load_dataset("allenai/winogrande", "winogrande_debiased", split="validation")
    except Exception as exc:
        return {"error": f"dataset load failed: {exc}", "accuracy": -1.0}

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as exc:
        return {"error": f"tokenizer load failed: {exc}", "accuracy": -1.0}

    model.eval()
    correct = 0
    total = 0
    n = min(max_examples, len(ds))

    with torch.no_grad():
        for i in range(n):
            ex = ds[i]
            sentence = ex["sentence"]
            option1 = ex["option1"]
            option2 = ex["option2"]
            label = int(ex["answer"]) - 1  # 1-indexed -> 0-indexed

            candidates = [
                sentence.replace("_", option1),
                sentence.replace("_", option2),
            ]

            scores = []
            for text in candidates:
                tokens = tokenizer.encode(text, return_tensors="pt").to(device)
                if tokens.shape[1] > cfg["model"].get(
                    "max_position_embeddings", 2048
                ):
                    tokens = tokens[:, : cfg["model"]["max_position_embeddings"]]
                if tokens.shape[1] < 2:
                    scores.append(float("-inf"))
                    continue
                out = model(input_ids=tokens)
                logits = out.logits if hasattr(out, "logits") else out[0]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = tokens[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="mean",
                )
                scores.append(-float(loss.item()))

            pred = max(range(len(scores)), key=lambda j: scores[j])
            if pred == label:
                correct += 1
            total += 1

            if total % 200 == 0:
                print(
                    json.dumps(
                        {
                            "event": "winogrande_progress",
                            "done": total,
                            "of": n,
                            "acc_so_far": round(correct / total, 4),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    accuracy = correct / max(total, 1)
    return {
        "task": "winogrande",
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
    }


def run_eval_checkpoint(
    checkpoint_path: str,
    tasks: str = "hellaswag,lambada,winogrande",
    max_examples: int = 1000,
    output_dir: Optional[str] = None,
) -> int:
    """Load a checkpoint and run downstream evaluations."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    use_bf16 = (
        bool(cfg["training"].get("bf16", False))
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    if use_bf16:
        model = model.to(dtype=torch.bfloat16)

    task_list = [t.strip() for t in tasks.split(",")]
    results: Dict[str, Any] = {
        "checkpoint": checkpoint_path,
        "completed_steps": ckpt.get("completed_steps"),
        "exit_reason": ckpt.get("exit_reason"),
        "device": str(device),
        "bf16": use_bf16,
        "tasks": {},
    }

    for task in task_list:
        print(
            json.dumps({"event": "eval_start", "task": task}, sort_keys=True),
            flush=True,
        )
        if task == "hellaswag":
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                r = eval_hellaswag(model, cfg, device, max_examples=max_examples)
        elif task == "lambada":
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                r = eval_lambada(model, cfg, device, max_examples=max_examples)
        elif task == "winogrande":
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                r = eval_winogrande(model, cfg, device, max_examples=max_examples)
        else:
            r = {"error": f"unknown task: {task}"}
        results["tasks"][task] = r
        print(
            json.dumps({"event": "eval_result", "task": task, **r}, sort_keys=True),
            flush=True,
        )

    out_dir = Path(output_dir) if output_dir else Path(checkpoint_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = Path(checkpoint_path).stem.replace(".checkpoint", "")
    save_json(out_dir / f"{run_name}.eval_results.json", results)
    print(
        json.dumps({"event": "eval_complete", "results": results}, sort_keys=True),
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="JSON config path")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="key=value overrides (e.g. training.train_steps=2)",
    )
    parser.add_argument(
        "--derive-topk",
        type=int,
        default=None,
        metavar="K",
        help="Trim a selection JSON to top-K heads (requires --input-selection and --output)",
    )
    parser.add_argument(
        "--input-selection", default=None, help="Input selection JSON for --derive-topk"
    )
    parser.add_argument("--output", default=None, help="Output path for --derive-topk")
    parser.add_argument(
        "--eval-checkpoint",
        default=None,
        help="Path to checkpoint for downstream eval (skip training)",
    )
    parser.add_argument(
        "--eval-tasks",
        default="hellaswag,lambada,winogrande",
        help="Comma-separated eval tasks (default: hellaswag,lambada,winogrande)",
    )
    parser.add_argument(
        "--eval-max-examples",
        type=int,
        default=1000,
        help="Max examples per eval task (default: 1000)",
    )
    args = parser.parse_args()

    if args.derive_topk is not None:
        if not args.input_selection or not args.output:
            parser.error("--derive-topk requires --input-selection and --output")
        return derive_topk(args.input_selection, args.derive_topk, args.output)

    if args.eval_checkpoint is not None:
        return run_eval_checkpoint(
            args.eval_checkpoint,
            tasks=args.eval_tasks,
            max_examples=args.eval_max_examples,
            output_dir=args.output,
        )

    if not args.config:
        parser.error("--config is required for training runs")

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
    out_dir = repo_root / cfg["tracking"].get("out_dir", "out/llama150m-matched")
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
        project=cfg["tracking"].get("wandb_project", "llama150m-ra-surgical-b200x4"),
        mode=cfg["tracking"].get("wandb_mode", "offline"),
        run_name=run_name,
        config=cfg,
        rank=rank,
    )

    if rank0(rank):
        save_json(out_dir / f"{run_name}.resolved.json", cfg)

    use_fsdp = cfg.get("distributed_strategy", "ddp") == "fsdp"
    if use_fsdp:
        # FSDP: build on CPU, apply patch, then let FSDP shard to GPUs
        model = build_model(cfg)
    else:
        model = build_model(cfg).to(device)
    patch_ctx = apply_llama_sdpa_patch(model, cfg, repo_root)
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

    if cfg["training"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    _bf16_ok = (
        bool(cfg["training"].get("bf16", False))
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    if distributed and use_fsdp:
        if not FSDP_AVAILABLE:
            raise RuntimeError("FSDP requested but not available in this torch build")
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        fsdp_mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ) if _bf16_ok else None
        auto_wrap = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={LlamaDecoderLayer},
        )
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=fsdp_mp,
            device_id=local_rank,
            use_orig_params=True,
        )
    elif distributed:
        ddp_kwargs = (
            {"device_ids": [local_rank], "output_device": local_rank}
            if device.type == "cuda"
            else {}
        )
        model = DDP(model, **ddp_kwargs)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(cfg.get("seed", 1337)) + rank)

    training_cfg = cfg["training"]
    use_bf16 = (
        bool(training_cfg.get("bf16", False))
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
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg.get("weight_decay", 0.1),
    )

    startup = {
        "event": "startup",
        "run_name": run_name,
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device": str(device),
        "distributed": distributed,
    }
    print(json.dumps(startup, sort_keys=True), flush=True)
    if jsonl:
        jsonl.log(startup)

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
        # In DDP every rank must agree on the stop decision, otherwise
        # fast ranks break out while slow ranks enter the next DDP step,
        # causing an NCCL timeout.  Broadcast rank-0's verdict.
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
        stop_elapsed_candidate = round(elapsed_s, 3)
        if should_eval:
            eval_loss = eval_model(model, val_ds, cfg, device, generator)
            payload = {
                "event": "eval",
                "step": step,
                "train_loss": step_loss,
                "eval_loss": eval_loss,
                "perplexity": float(math.exp(min(eval_loss, 20.0))),
                "elapsed_s": stop_elapsed_candidate,
            }
            if rank0(rank):
                print(json.dumps(payload, sort_keys=True), flush=True)
                if jsonl:
                    jsonl.log(payload)
                wandb_logger.log(
                    {k: v for k, v in payload.items() if k not in {"event"}}, step=step
                )
        barrier()
        if time_limit_reached:
            exit_reason = "max_time"
            stop_elapsed_s = stop_elapsed_candidate
            break

    if exit_reason != "max_time":
        stop_elapsed_s = round(time.time() - started, 3)

    # Save checkpoint for downstream evaluation
    if use_fsdp and distributed:
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            fsdp_state = model.state_dict()
        if rank0(rank):
            ckpt_path = out_dir / f"{run_name}.checkpoint.pt"
            try:
                torch.save(
                    {
                        "model_state_dict": fsdp_state,
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
                        {
                            "event": "checkpoint_save_failed",
                            "path": str(ckpt_path),
                            "error": repr(e),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    elif rank0(rank):
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
                    {"event": "checkpoint_saved", "path": str(ckpt_path)}, sort_keys=True
                ),
                flush=True,
            )
        except Exception as e:
            print(
                json.dumps(
                    {
                        "event": "checkpoint_save_failed",
                        "path": str(ckpt_path),
                        "error": repr(e),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    fim_summary = (
        patch_ctx.stats_collector.summary()
        if patch_ctx.stats_collector.enabled
        else None
    )
    if rank0(rank):
        backend_summary = patch_ctx.backend_recorder.summary()
        save_json(out_dir / f"{run_name}.backend.json", backend_summary)
        print(
            json.dumps({"event": "backend_summary", **backend_summary}, sort_keys=True),
            flush=True,
        )
        if jsonl:
            jsonl.log({"event": "backend_summary", **backend_summary})

        if fim_summary is not None:
            save_json(out_dir / f"{run_name}.fim_summary.json", fim_summary)
            generated = generate_surgical_config(fim_summary, cfg)
            selection_path = repo_root / cfg["fim"].get(
                "output_file", "configs/ra_surgical_llama150m.json"
            )
            save_json(selection_path, generated)
            save_json(out_dir / f"{run_name}.generated_selection.json", generated)
            print(
                json.dumps({"event": "fim_summary", **fim_summary}, sort_keys=True),
                flush=True,
            )
            print(
                json.dumps(
                    {"event": "generated_selection", **generated}, sort_keys=True
                ),
                flush=True,
            )
            if jsonl:
                jsonl.log({"event": "fim_summary", **fim_summary})
                jsonl.log({"event": "generated_selection", **generated})

        elapsed_total = round(time.time() - started, 3)
        done = {
            "event": "complete",
            "run_name": run_name,
            "stop_elapsed_s": stop_elapsed_s,
            "total_elapsed_s": elapsed_total,
            "elapsed_s": stop_elapsed_s,
            "world_size": world_size,
            "completed_steps": completed_steps,
            "exit_reason": exit_reason,
        }
        teardown_delta = elapsed_total - stop_elapsed_s
        if teardown_delta > 5.0:
            done["teardown_overhead_s"] = round(teardown_delta, 3)
            done["teardown_warning"] = (
                f"total wall-clock exceeds training stop by "
                f"{teardown_delta:.1f}s; use stop_elapsed_s for "
                f"matched comparisons"
            )
        print(json.dumps(done, sort_keys=True), flush=True)
        if jsonl:
            jsonl.log(done)

        max_time_shortfall = max_time is not None and elapsed_total + 1e-6 < max_time
        if exit_reason == "max_steps" and max_time_shortfall:
            warning = {
                "event": "wallclock_mismatch_warning",
                "run_name": run_name,
                "elapsed_s": elapsed_total,
                "configured_max_time": max_time,
                "completed_steps": completed_steps,
                "message": "run hit training.train_steps before training.max_time; wall-clock matching is invalid unless train_steps is increased",
            }
            print(json.dumps(warning, sort_keys=True), flush=True)
            if jsonl:
                jsonl.log(warning)

    barrier()
    wandb_logger.finish()
    barrier()
    cleanup_distributed()
    if (
        exit_reason == "max_steps"
        and max_time is not None
        and (time.time() - started) + 1e-6 < max_time
    ):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

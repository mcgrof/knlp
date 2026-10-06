#!/usr/bin/env python3
"""Resumable Stage-1 and Stage-2 training for the KV-Lingo 4B/8B pair."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import platform
import resource
import time

import torch
import torch.distributed as dist

from .coqa_runner import write_json
from .kv_lingo import (
    CacheGeometry,
    LinearTranslator,
    PreNormSpan,
    capture_pre_norm_span,
    forward_kl,
    freeze_model,
    geometry,
)
from .kv_lingo_data import TokenStore, file_sha256, iter_jsonl, token_sha256
from .kv_lingo_preflight import continuation_logits, native_teacher

PINS = {
    "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
    "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
}
DIRECTIONS = {
    "4b-to-8b": ("Qwen/Qwen3-4B", "Qwen/Qwen3-8B"),
    "8b-to-4b": ("Qwen/Qwen3-8B", "Qwen/Qwen3-4B"),
}
STAGE2_STEPS = 5_000
WARMUP_STEPS = 250
EFFECTIVE_BATCH = 8
LEARNING_RATE = 3e-5
CAPTURE_CUT = "pre-k_norm keys and v_proj values; target k_norm then RoPE"
LOSS_REDUCTION = "mean_token_forward_kl_then_mean_of_8_samples"
MAP_ARCHITECTURE = "per_layer_dense_key_and_value_maps_fp32"
DISTRIBUTED_CONTRACT_SCHEMA = "kv_lingo_resume_contract_v2"
DISTRIBUTED_EXECUTION_SCHEMA = "kv_lingo_distributed_execution_v1"
DISTRIBUTED_ASSIGNMENT = "length_balanced_within_fixed_global_batch_v1"


def log(message: str) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {message}", flush=True)


def distributed_environment() -> dict:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size not in (1, 4, 8):
        raise ValueError("KV-Lingo distributed training requires world size 1, 4, or 8")
    if not 0 <= rank < world_size:
        raise ValueError("distributed rank is outside the requested world size")
    if not 0 <= local_rank < world_size:
        raise ValueError("local rank is outside the requested world size")
    return {
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "enabled": world_size > 1,
    }


def initialize_distributed(context: dict) -> None:
    if not context["enabled"]:
        return
    torch.cuda.set_device(context["local_rank"])
    dist.init_process_group(backend="nccl")


def execution_topology(args, *, migration_from_world_size: int | None) -> dict:
    context = distributed_environment()
    if not context["enabled"]:
        raise ValueError("distributed execution topology requested for a serial run")
    source_commit = getattr(args, "source_commit", None)
    if not source_commit:
        raise ValueError("distributed training requires --source-commit")
    return {
        "schema": DISTRIBUTED_EXECUTION_SCHEMA,
        "world_size": context["world_size"],
        "backend": "nccl",
        "assignment": DISTRIBUTED_ASSIGNMENT,
        "global_batch": EFFECTIVE_BATCH,
        "source_commit": source_commit,
        "implementation_sha256": file_sha256(Path(__file__)),
        "migration_from_world_size": migration_from_world_size,
        "stochastic_mapping": (
            "models are eval/frozen with no dropout; per-rank RNG state is recorded"
        ),
    }


def distributed_batch_assignments(batch_rows: list[dict], world_size: int):
    """Assign one fixed eight-row global batch without changing membership."""

    if len(batch_rows) != EFFECTIVE_BATCH:
        raise ValueError("distributed assignment requires exactly eight rows")
    if world_size not in (1, 4, 8):
        raise ValueError("distributed assignment requires world size 1, 4, or 8")
    indexed = list(enumerate(batch_rows))
    if world_size == 1:
        return [indexed]
    ordered = sorted(
        indexed,
        key=lambda item: (
            -(item[1]["prefix_tokens"] + item[1]["continuation_tokens"]),
            item[1]["index"],
        ),
    )
    if world_size == 8:
        return [[item] for item in ordered]
    assignments = [[] for _ in range(world_size)]
    for rank in range(world_size):
        assignments[rank] = sorted(
            (ordered[rank], ordered[-(rank + 1)]), key=lambda item: item[0]
        )
    return assignments


def atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def geometry_dict(value: CacheGeometry) -> dict:
    return {
        "layers": value.layers,
        "kv_heads": value.kv_heads,
        "head_dim": value.head_dim,
        "width": value.width,
    }


def read_rows(path: Path) -> list[dict]:
    return list(iter_jsonl(path))


def _require_equal(label: str, actual, expected) -> None:
    if actual != expected:
        raise ValueError(
            f"resume contract {label} mismatch: {actual!r} != {expected!r}"
        )


def _require_hash(label: str, path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"resume contract {label} is absent: {path}")
    _require_equal(f"{label} sha256", file_sha256(path), expected)


def validate_resume_contract(args) -> dict | None:
    """Reject an unsafe Stage-2 continuation before loading either model.

    Legacy checkpoints do not contain the complete experiment identity.  The
    immutable JSON sidecar therefore binds every omitted field to the exact
    files and constants used by the continuation.  It is deliberately checked
    before CUDA is initialized or model weights are allocated.
    """

    if args.action != "train" or args.resume is None:
        return None
    if not args.resume.is_file():
        raise FileNotFoundError(f"explicit resume checkpoint is absent: {args.resume}")
    if args.resume_contract is None or not args.resume_contract.is_file():
        raise FileNotFoundError(
            f"explicit resume requires a readable --resume-contract: "
            f"{args.resume_contract}"
        )
    contract = json.loads(args.resume_contract.read_text(encoding="utf-8"))
    if contract.get("schema") not in (
        "kv_lingo_resume_contract_v1",
        DISTRIBUTED_CONTRACT_SCHEMA,
    ):
        raise ValueError(f"resume contract schema mismatch: {contract.get('schema')!r}")
    _require_equal("direction", args.direction, contract.get("direction"))
    _require_equal(
        "checkpoint schema",
        contract.get("checkpoint_schema"),
        "kv_lingo_stage2_checkpoint_v1",
    )
    _require_hash("checkpoint", args.resume, contract["checkpoint_sha256"])
    if args.stage1 is None:
        raise ValueError("resume requires the exact --stage1 artifact")
    _require_hash("Stage-1 artifact", args.stage1, contract["stage1_sha256"])
    _require_hash("training rows", args.rows, contract["train_rows_sha256"])
    _require_hash("training tokens", args.tokens, contract["train_tokens_sha256"])
    if args.validation_rows is None or args.validation_tokens is None:
        raise ValueError("resume requires --validation-rows and --validation-tokens")
    _require_hash(
        "validation rows", args.validation_rows, contract["validation_rows_sha256"]
    )
    _require_hash(
        "validation tokens",
        args.validation_tokens,
        contract["validation_tokens_sha256"],
    )
    _require_equal("model revisions", contract.get("model_revisions"), PINS)
    _require_equal(
        "tokenizer revision",
        contract.get("tokenizer_revision"),
        PINS["Qwen/Qwen3-4B"],
    )
    _require_equal("capture cut", contract.get("capture_cut"), CAPTURE_CUT)
    _require_equal(
        "map architecture", contract.get("map_architecture"), MAP_ARCHITECTURE
    )
    _require_equal("loss reduction", contract.get("loss_reduction"), LOSS_REDUCTION)
    _require_equal("effective batch", contract.get("effective_batch"), EFFECTIVE_BATCH)
    _require_equal("warmup steps", contract.get("warmup_steps"), WARMUP_STEPS)
    _require_equal("total schedule", contract.get("total_schedule_steps"), STAGE2_STEPS)
    _require_equal(
        "peak learning rate", contract.get("peak_learning_rate"), LEARNING_RATE
    )

    stage1 = torch.load(args.stage1, map_location="cpu", weights_only=False)
    source_name, target_name = DIRECTIONS[args.direction]
    _require_equal("Stage-1 schema", stage1.get("schema"), "kv_lingo_translator_v1")
    _require_equal("Stage-1 direction", stage1.get("direction"), args.direction)
    _require_equal(
        "Stage-1 source",
        stage1.get("source"),
        {"name": source_name, "revision": PINS[source_name]},
    )
    _require_equal(
        "Stage-1 target",
        stage1.get("target"),
        {"name": target_name, "revision": PINS[target_name]},
    )
    _require_equal("Stage-1 capture cut", stage1.get("cut"), CAPTURE_CUT)
    _require_equal(
        "Stage-1 train rows",
        stage1.get("provenance", {}).get("train_rows_sha256"),
        contract["train_rows_sha256"],
    )
    _require_equal(
        "Stage-1 token store",
        stage1.get("provenance", {}).get("token_store_sha256"),
        contract["train_tokens_sha256"],
    )

    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    _require_equal(
        "embedded checkpoint schema",
        checkpoint.get("schema"),
        contract["checkpoint_schema"],
    )
    _require_equal("embedded direction", checkpoint.get("direction"), args.direction)
    step = int(checkpoint.get("global_step", -1))
    cursor = int(checkpoint.get("sample_cursor", -1))
    _require_equal("global step", step, int(contract["global_step"]))
    _require_equal("sample cursor", cursor, int(contract["sample_cursor"]))
    _require_equal("cursor semantics", cursor, step * EFFECTIVE_BATCH)
    _require_equal(
        "embedded total schedule", checkpoint.get("total_schedule_steps"), STAGE2_STEPS
    )
    _require_equal("embedded warmup", checkpoint.get("warmup_steps"), WARMUP_STEPS)
    _require_equal(
        "embedded effective batch", checkpoint.get("effective_batch"), EFFECTIVE_BATCH
    )
    _require_equal(
        "embedded learning rate", checkpoint.get("learning_rate"), LEARNING_RATE
    )
    _require_equal(
        "embedded Stage-1 hash",
        checkpoint.get("stage1_artifact", {}).get("sha256"),
        contract["stage1_sha256"],
    )
    if not isinstance(checkpoint.get("optimizer"), dict) or "rng" not in checkpoint:
        raise ValueError("resume checkpoint lacks optimizer or RNG state")
    context = distributed_environment()
    checkpoint_topology = checkpoint.get("distributed")
    contract_topology = contract.get("execution_topology")
    if contract.get("schema") == DISTRIBUTED_CONTRACT_SCHEMA:
        if not isinstance(contract_topology, dict):
            raise ValueError("distributed resume contract lacks execution topology")
        _require_equal(
            "execution schema",
            contract_topology.get("schema"),
            DISTRIBUTED_EXECUTION_SCHEMA,
        )
        _require_equal(
            "execution world size",
            contract_topology.get("world_size"),
            context["world_size"],
        )
        _require_equal(
            "execution assignment",
            contract_topology.get("assignment"),
            DISTRIBUTED_ASSIGNMENT,
        )
        _require_equal(
            "execution source commit",
            contract_topology.get("source_commit"),
            getattr(args, "source_commit", None),
        )
        _require_equal(
            "execution implementation",
            contract_topology.get("implementation_sha256"),
            file_sha256(Path(__file__)),
        )
        if not isinstance(checkpoint_topology, dict):
            raise ValueError("distributed checkpoint lacks rank execution state")
        _require_equal(
            "checkpoint world size",
            checkpoint_topology.get("world_size"),
            context["world_size"],
        )
        _require_equal(
            "checkpoint assignment",
            checkpoint_topology.get("assignment"),
            DISTRIBUTED_ASSIGNMENT,
        )
        rank_rng = checkpoint_topology.get("rank_rng")
        if not isinstance(rank_rng, list) or len(rank_rng) != context["world_size"]:
            raise ValueError("distributed checkpoint has incomplete per-rank RNG state")
    elif checkpoint_topology is not None:
        raise ValueError("legacy resume contract cannot authorize distributed state")
    elif context["enabled"] and not getattr(args, "source_commit", None):
        raise ValueError("serial-to-distributed migration requires --source-commit")
    metrics = checkpoint.get("metrics")
    if not isinstance(metrics, list) or len(metrics) != step:
        raise ValueError("resume checkpoint metrics do not cover the recorded cursor")
    if step and (
        int(metrics[-1].get("step", -1)) != step
        or int(metrics[-1].get("sample_cursor", -1)) != cursor
    ):
        raise ValueError("resume checkpoint final metric disagrees with its cursor")
    if not step < args.max_steps <= STAGE2_STEPS:
        raise ValueError(
            "resume target must advance within the original 5000-step schedule"
        )
    return contract


def validate_new_run_contract(args) -> dict | None:
    """Bind a fresh Stage-2 run to its Stage-1 and frozen v2 data identity."""

    if args.action != "train" or args.resume is not None:
        return None
    if args.stage1 is None or not args.stage1.is_file():
        raise FileNotFoundError("fresh Stage-2 training requires a readable --stage1")
    if args.validation_rows is None or args.validation_tokens is None:
        raise ValueError("fresh Stage-2 training requires validation rows and tokens")
    for label, path in (
        ("training rows", args.rows),
        ("training tokens", args.tokens),
        ("validation rows", args.validation_rows),
        ("validation tokens", args.validation_tokens),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"fresh Stage-2 {label} is absent: {path}")
    stage1 = torch.load(args.stage1, map_location="cpu", weights_only=False)
    source_name, target_name = DIRECTIONS[args.direction]
    train_rows_hash = file_sha256(args.rows)
    train_tokens_hash = file_sha256(args.tokens)
    _require_equal("Stage-1 schema", stage1.get("schema"), "kv_lingo_translator_v1")
    _require_equal("Stage-1 direction", stage1.get("direction"), args.direction)
    _require_equal(
        "Stage-1 source",
        stage1.get("source"),
        {"name": source_name, "revision": PINS[source_name]},
    )
    _require_equal(
        "Stage-1 target",
        stage1.get("target"),
        {"name": target_name, "revision": PINS[target_name]},
    )
    _require_equal("Stage-1 capture cut", stage1.get("cut"), CAPTURE_CUT)
    _require_equal(
        "Stage-1 train rows",
        stage1.get("provenance", {}).get("train_rows_sha256"),
        train_rows_hash,
    )
    _require_equal(
        "Stage-1 token store",
        stage1.get("provenance", {}).get("token_store_sha256"),
        train_tokens_hash,
    )
    context = distributed_environment()
    contract = {
        "schema": (
            DISTRIBUTED_CONTRACT_SCHEMA
            if context["enabled"]
            else "kv_lingo_resume_contract_v1"
        ),
        "direction": args.direction,
        "checkpoint_schema": "kv_lingo_stage2_checkpoint_v1",
        "checkpoint_sha256": None,
        "global_step": 0,
        "sample_cursor": 0,
        "stage1_sha256": file_sha256(args.stage1),
        "train_rows_sha256": train_rows_hash,
        "train_tokens_sha256": train_tokens_hash,
        "validation_rows_sha256": file_sha256(args.validation_rows),
        "validation_tokens_sha256": file_sha256(args.validation_tokens),
        "model_revisions": PINS,
        "tokenizer_revision": PINS["Qwen/Qwen3-4B"],
        "capture_cut": CAPTURE_CUT,
        "map_architecture": MAP_ARCHITECTURE,
        "loss_reduction": LOSS_REDUCTION,
        "effective_batch": EFFECTIVE_BATCH,
        "warmup_steps": WARMUP_STEPS,
        "total_schedule_steps": STAGE2_STEPS,
        "peak_learning_rate": LEARNING_RATE,
        "Generated-by": "OpenAI Codex",
    }
    if context["enabled"]:
        contract["execution_topology"] = execution_topology(
            args, migration_from_world_size=None
        )
    return contract


def sample_ids(
    row: dict, store: TokenStore, device
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = store.read(row["prefix_offset_uint32"], row["prefix_tokens"])
    continuation = store.read(
        row["continuation_offset_uint32"], row["continuation_tokens"]
    )
    if token_sha256(prefix) != row["prefix_sha256"]:
        raise RuntimeError(f"prefix hash mismatch at sample {row['index']}")
    if token_sha256(continuation) != row["continuation_sha256"]:
        raise RuntimeError(f"continuation hash mismatch at sample {row['index']}")
    return (
        torch.tensor([prefix], dtype=torch.long, device=device),
        torch.tensor([continuation], dtype=torch.long, device=device),
    )


def _rows(span: PreNormSpan, layer: int, component: str) -> torch.Tensor:
    tensor = span.keys[layer] if component == "k" else span.values[layer]
    return tensor.transpose(1, 2).reshape(-1, tensor.shape[1] * tensor.shape[3])


class BidirectionalMoments:
    """FP64 sufficient statistics shared by both shape-preserving directions."""

    def __init__(self, geom: CacheGeometry, device):
        if geom.width <= 0:
            raise ValueError("invalid geometry")
        shape = (geom.layers, 2, geom.width, geom.width)
        self.geom = geom
        self.gram4 = torch.zeros(shape, dtype=torch.float64, device=device)
        self.gram8 = torch.zeros(shape, dtype=torch.float64, device=device)
        self.cross8_4 = torch.zeros(shape, dtype=torch.float64, device=device)
        self.samples = 0
        self.positions = 0

    @torch.no_grad()
    def update(self, span4: PreNormSpan, span8: PreNormSpan) -> None:
        if span4.tokens != span8.tokens:
            raise ValueError("stage-1 spans cover different token counts")
        for layer in range(self.geom.layers):
            for side, component in enumerate(("k", "v")):
                rows4 = _rows(span4, layer, component).to(torch.float64)
                rows8 = _rows(span8, layer, component).to(torch.float64)
                self.gram4[layer, side].addmm_(rows4.T, rows4)
                self.gram8[layer, side].addmm_(rows8.T, rows8)
                self.cross8_4[layer, side].addmm_(rows8.T, rows4)
                del rows4, rows8
        self.samples += 1
        self.positions += span4.tokens

    def state_dict(self) -> dict:
        return {
            "schema": "kv_lingo_stage1_moments_v1",
            "geometry": geometry_dict(self.geom),
            "gram4": self.gram4,
            "gram8": self.gram8,
            "cross8_4": self.cross8_4,
            "samples": self.samples,
            "positions": self.positions,
        }

    @classmethod
    def from_state_dict(cls, value: dict, device):
        gd = value["geometry"]
        result = cls(
            CacheGeometry(gd["layers"], gd["kv_heads"], gd["head_dim"]), device
        )
        result.gram4.copy_(value["gram4"].to(device))
        result.gram8.copy_(value["gram8"].to(device))
        result.cross8_4.copy_(value["cross8_4"].to(device))
        result.samples = int(value["samples"])
        result.positions = int(value["positions"])
        return result

    @staticmethod
    @torch.no_grad()
    def _solve(gram: torch.Tensor, cross: torch.Tensor, tolerance: float):
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        largest = eigenvalues[-1].clamp_min(0)
        keep = eigenvalues > largest * tolerance
        inverse = torch.where(keep, eigenvalues.reciprocal(), 0)
        # cross @ Q @ diag(inv) @ Q.T, without materialising a pseudoinverse.
        weight = ((cross @ eigenvectors) * inverse.unsqueeze(0)) @ eigenvectors.T
        receipt = {
            "rank": int(keep.sum().item()),
            "width": int(gram.shape[0]),
            "largest_eigenvalue": float(largest),
            "smallest_retained_eigenvalue": (
                float(eigenvalues[keep][0]) if bool(keep.any()) else None
            ),
            "relative_tolerance": tolerance,
            "finite": bool(torch.isfinite(weight).all()),
        }
        if not receipt["finite"]:
            raise FloatingPointError("stage-1 solve produced nonfinite coefficients")
        return weight.to(torch.float32), receipt

    @torch.no_grad()
    def solve(self, tolerance: float = 1e-8):
        weights = {
            "4b-to-8b": {
                "key": torch.empty_like(self.cross8_4[:, 0], dtype=torch.float32),
                "value": torch.empty_like(self.cross8_4[:, 1], dtype=torch.float32),
            },
            "8b-to-4b": {
                "key": torch.empty_like(self.cross8_4[:, 0], dtype=torch.float32),
                "value": torch.empty_like(self.cross8_4[:, 1], dtype=torch.float32),
            },
        }
        receipts = {direction: [] for direction in weights}
        for layer in range(self.geom.layers):
            for side, component in enumerate(("key", "value")):
                forward, f_receipt = self._solve(
                    self.gram4[layer, side],
                    self.cross8_4[layer, side],
                    tolerance,
                )
                reverse, r_receipt = self._solve(
                    self.gram8[layer, side],
                    self.cross8_4[layer, side].T,
                    tolerance,
                )
                weights["4b-to-8b"][component][layer].copy_(forward)
                weights["8b-to-4b"][component][layer].copy_(reverse)
                receipts["4b-to-8b"].append(
                    {"layer": layer, "component": component, **f_receipt}
                )
                receipts["8b-to-4b"].append(
                    {"layer": layer, "component": component, **r_receipt}
                )
        return weights, receipts


def load_models(device="cuda") -> dict[str, torch.nn.Module]:
    from transformers import AutoModelForCausalLM

    models = {}
    for size, name in (("4b", "Qwen/Qwen3-4B"), ("8b", "Qwen/Qwen3-8B")):
        log(f"loading {name}@{PINS[name]}")
        model = AutoModelForCausalLM.from_pretrained(
            name,
            revision=PINS[name],
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device)
        freeze_model(model)
        models[size] = model
    if geometry(models["4b"]) != geometry(models["8b"]):
        raise RuntimeError("the pinned pair no longer has shape-preserving KV geometry")
    return models


def translator_artifact(
    direction: str,
    translator: LinearTranslator,
    *,
    stage: str,
    provenance: dict,
) -> dict:
    source_name, target_name = DIRECTIONS[direction]
    return {
        "schema": "kv_lingo_translator_v1",
        "method_identity": "independent KV-Lingo reproduction",
        "direction": direction,
        "source": {"name": source_name, "revision": PINS[source_name]},
        "target": {"name": target_name, "revision": PINS[target_name]},
        "source_geometry": geometry_dict(translator.source),
        "target_geometry": geometry_dict(translator.target),
        "cut": CAPTURE_CUT,
        "parameter_dtype": "float32",
        "stage": stage,
        "state_dict": {
            key: value.detach().cpu() for key, value in translator.state_dict().items()
        },
        "provenance": provenance,
    }


def translator_from_artifact(value: dict, device) -> LinearTranslator:
    sg, tg = value["source_geometry"], value["target_geometry"]
    translator = LinearTranslator(
        CacheGeometry(sg["layers"], sg["kv_heads"], sg["head_dim"]),
        CacheGeometry(tg["layers"], tg["kv_heads"], tg["head_dim"]),
    ).to(device)
    translator.load_state_dict(value["state_dict"], strict=True)
    return translator


def run_stage1(args, models, rows, store):
    device = next(models["4b"].parameters()).device
    checkpoint_path = args.out / "stage1-moments.pt"
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        moments = BidirectionalMoments.from_state_dict(state, device)
        log(f"resumed Stage 1 at sample {moments.samples}")
    else:
        moments = BidirectionalMoments(geometry(models["4b"]), device)

    started = time.time()
    sample_seconds = []
    for row in rows[moments.samples : args.stage1_samples]:
        prefix, _continuation = sample_ids(row, store, device)
        translated_prefix = prefix[:, :-1]
        before = time.perf_counter()
        span4 = capture_pre_norm_span(models["4b"], translated_prefix)
        span8 = capture_pre_norm_span(models["8b"], translated_prefix)
        moments.update(span4, span8)
        torch.cuda.synchronize()
        sample_seconds.append(time.perf_counter() - before)
        del prefix, translated_prefix, span4, span8
        gc.collect()
        torch.cuda.empty_cache()
        if moments.samples % args.stage1_checkpoint_every == 0:
            atomic_torch_save(moments.state_dict(), checkpoint_path)
        log(
            f"Stage1 {moments.samples}/{args.stage1_samples}: "
            f"positions={moments.positions:,}, last={sample_seconds[-1]:.2f}s"
        )

    atomic_torch_save(moments.state_dict(), checkpoint_path)
    weights, solve_receipts = moments.solve(args.stage1_tolerance)
    artifacts = {}
    for direction, pair_weights in weights.items():
        source_size, target_size = (
            ("4b", "8b") if direction == "4b-to-8b" else ("8b", "4b")
        )
        translator = LinearTranslator(
            geometry(models[source_size]), geometry(models[target_size])
        ).to(device)
        with torch.no_grad():
            translator.key_weight.copy_(pair_weights["key"])
            translator.value_weight.copy_(pair_weights["value"])
        path = args.out / f"stage1-{direction}.pt"
        provenance = {
            "samples": moments.samples,
            "translated_prefix_positions": moments.positions,
            "relative_eigen_tolerance": args.stage1_tolerance,
            "solve": solve_receipts[direction],
            "train_rows_sha256": file_sha256(args.rows),
            "token_store_sha256": file_sha256(args.tokens),
        }
        atomic_torch_save(
            translator_artifact(
                direction, translator, stage="stage1_closed_form", provenance=provenance
            ),
            path,
        )
        artifacts[direction] = {"path": path.name, "sha256": file_sha256(path)}
        del translator
    receipt = {
        "schema": "kv_lingo_stage1_receipt_v1",
        "samples": moments.samples,
        "translated_prefix_positions": moments.positions,
        "sample_seconds": sample_seconds,
        "wall_seconds_this_invocation": time.time() - started,
        "artifacts": artifacts,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
    write_json(args.out / "STAGE1_RECEIPT.json", receipt)
    return receipt


def learning_rate(step: int) -> float:
    if step <= 0:
        return 0.0
    if step <= WARMUP_STEPS:
        return LEARNING_RATE * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (STAGE2_STEPS - WARMUP_STEPS)
    return LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * progress))


def rng_state() -> dict:
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(value: dict) -> None:
    torch.set_rng_state(value["cpu"])
    torch.cuda.set_rng_state_all(value["cuda"])


def local_rng_state() -> dict:
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
    }


def gather_rank_rng(context: dict) -> list[dict] | None:
    if not context["enabled"]:
        return None
    gathered = [None] * context["world_size"]
    dist.all_gather_object(
        gathered,
        {"rank": context["rank"], "rng": local_rng_state()},
    )
    ordered = sorted(gathered, key=lambda value: value["rank"])
    return [value["rng"] for value in ordered]


def checkpoint_state(
    direction: str,
    translator: LinearTranslator,
    optimizer,
    *,
    global_step: int,
    sample_cursor: int,
    source_artifact: Path,
    metrics: list[dict],
    execution: dict | None = None,
    rank_rng: list[dict] | None = None,
) -> dict:
    state = {
        "schema": "kv_lingo_stage2_checkpoint_v1",
        "direction": direction,
        "global_step": global_step,
        "sample_cursor": sample_cursor,
        "total_schedule_steps": STAGE2_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "effective_batch": EFFECTIVE_BATCH,
        "learning_rate": LEARNING_RATE,
        "source_geometry": geometry_dict(translator.source),
        "target_geometry": geometry_dict(translator.target),
        "translator": {
            key: value.detach().cpu() for key, value in translator.state_dict().items()
        },
        "optimizer": optimizer.state_dict(),
        "rng": rank_rng[0] if rank_rng is not None else rng_state(),
        "stage1_artifact": {
            "path": str(source_artifact),
            "sha256": file_sha256(source_artifact),
        },
        "metrics": metrics,
    }
    if execution is not None:
        state["distributed"] = {
            "schema": DISTRIBUTED_EXECUTION_SCHEMA,
            "world_size": execution["world_size"],
            "backend": execution["backend"],
            "assignment": execution["assignment"],
            "rank_rng": rank_rng,
        }
    return state


def load_stage2(
    direction: str,
    stage1_path: Path,
    resume: Path | None,
    device,
    context: dict,
):
    artifact = torch.load(stage1_path, map_location="cpu", weights_only=False)
    translator = translator_from_artifact(artifact, device)
    optimizer = torch.optim.AdamW(
        translator.parameters(), lr=LEARNING_RATE, weight_decay=0.0
    )
    step = cursor = 0
    metrics = []
    if resume is not None:
        if not resume.is_file():
            raise FileNotFoundError(f"explicit resume checkpoint is absent: {resume}")
        value = torch.load(resume, map_location="cpu", weights_only=False)
        if value["direction"] != direction:
            raise ValueError("resume direction differs from requested direction")
        if value["total_schedule_steps"] != STAGE2_STEPS:
            raise ValueError("resume checkpoint would restart a different schedule")
        translator.load_state_dict(value["translator"], strict=True)
        optimizer.load_state_dict(value["optimizer"])
        for state in optimizer.state.values():
            for key, tensor in state.items():
                if torch.is_tensor(tensor):
                    state[key] = tensor.to(device)
        step = int(value["global_step"])
        cursor = int(value["sample_cursor"])
        metrics = list(value.get("metrics", []))
        distributed_state = value.get("distributed")
        if distributed_state is not None:
            rank_rng = distributed_state["rank_rng"][context["rank"]]
            torch.set_rng_state(rank_rng["cpu"])
            torch.cuda.set_rng_state(rank_rng["cuda"], device=device)
        elif context["enabled"]:
            torch.set_rng_state(value["rng"]["cpu"])
            torch.cuda.set_rng_state(value["rng"]["cuda"][0], device=device)
        else:
            restore_rng_state(value["rng"])
        log(f"resumed {direction} at step={step}, sample_cursor={cursor}")
    return translator, optimizer, step, cursor, metrics


def one_stage2_sample(source, target, translator, prefix, continuation):
    context = prefix.shape[1]
    prefix_without_last = prefix[:, :-1]
    scored_tokens = torch.cat((prefix[:, -1:], continuation[:, :-1]), dim=1)
    source_span = capture_pre_norm_span(source, prefix_without_last)
    with torch.no_grad():
        teacher = native_teacher(target, prefix_without_last, scored_tokens)
    positions = torch.arange(
        context - 1, device=prefix.device, dtype=torch.long
    ).unsqueeze(0)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        keys, values = translator(source_span, target, positions)
        student = continuation_logits(target, keys, values, scored_tokens, context - 1)
        loss = forward_kl(teacher, student)
    return loss, (source_span, teacher, keys, values, student)


def synchronize_mapper_gradients(translator, context: dict) -> float:
    if not context["enabled"]:
        return 0.0
    parameters = list(translator.parameters())
    local_valid = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().item())
        for parameter in parameters
    )
    valid = torch.tensor(
        int(local_valid), device=parameters[0].device, dtype=torch.int32
    )
    dist.all_reduce(valid, op=dist.ReduceOp.MIN)
    if not bool(valid.item()):
        raise FloatingPointError("missing or nonfinite mapper gradient on a rank")
    started = time.perf_counter()
    for parameter in parameters:
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    return time.perf_counter() - started


def gather_rank_records(local: dict, context: dict) -> list[dict]:
    if not context["enabled"]:
        return [local]
    gathered = [None] * context["world_size"]
    dist.all_gather_object(gathered, local)
    return sorted(gathered, key=lambda value: value["rank"])


def current_execution_topology(args, context: dict) -> dict | None:
    if not context["enabled"]:
        return None
    contract = args.verified_resume_contract
    if contract.get("schema") == DISTRIBUTED_CONTRACT_SCHEMA:
        return contract["execution_topology"]
    return execution_topology(args, migration_from_world_size=1)


def next_resume_contract(args, checkpoint_hash, step, cursor, execution):
    contract = {
        **args.verified_resume_contract,
        "checkpoint_sha256": checkpoint_hash,
        "global_step": step,
        "sample_cursor": cursor,
        "parent_checkpoint_sha256": args.verified_resume_contract.get(
            "checkpoint_sha256"
        ),
    }
    if args.resume_contract is not None:
        contract["parent_contract_sha256"] = file_sha256(args.resume_contract)
    if execution is not None:
        contract["schema"] = DISTRIBUTED_CONTRACT_SCHEMA
        contract["execution_topology"] = execution
    return contract


def run_stage2(args, models, rows, store):
    context = args.distributed_context
    source_size, target_size = (
        ("4b", "8b") if args.direction == "4b-to-8b" else ("8b", "4b")
    )
    source, target = models[source_size], models[target_size]
    device = next(target.parameters()).device
    stage1_path = args.stage1 or args.out / f"stage1-{args.direction}.pt"
    resume = args.resume
    translator, optimizer, step, cursor, metrics = load_stage2(
        args.direction, stage1_path, resume, device, context
    )
    if cursor != step * EFFECTIVE_BATCH:
        raise RuntimeError("sample cursor does not match effective-batch semantics")
    if args.max_steps > STAGE2_STEPS:
        raise ValueError("requested step exceeds the frozen 5000-step schedule")
    if len(rows) < args.max_steps * EFFECTIVE_BATCH:
        raise ValueError("frozen stream is shorter than the requested run")

    execution = current_execution_topology(args, context)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    while step < args.max_steps:
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        batch_rows = rows[cursor : cursor + EFFECTIVE_BATCH]
        assignments = distributed_batch_assignments(batch_rows, context["world_size"])
        local_samples = []
        local_compute_started = time.perf_counter()
        for offset, row in assignments[context["rank"]]:
            prefix, continuation = sample_ids(row, store, device)
            sample_started = time.perf_counter()
            loss, temporaries = one_stage2_sample(
                source, target, translator, prefix, continuation
            )
            if not bool(torch.isfinite(loss.detach())):
                raise FloatingPointError(
                    f"nonfinite Stage-2 loss at step={step + 1}, sample={row['index']}"
                )
            (loss / EFFECTIVE_BATCH).backward()
            torch.cuda.synchronize()
            local_samples.append(
                {
                    "offset": offset,
                    "index": row["index"],
                    "kl": float(loss.detach()),
                    "seconds": time.perf_counter() - sample_started,
                }
            )
            del prefix, continuation, loss, temporaries
            gc.collect()
            torch.cuda.empty_cache()
        local_compute_seconds = time.perf_counter() - local_compute_started
        reduction_seconds = synchronize_mapper_gradients(translator, context)
        next_step = step + 1
        lr = learning_rate(next_step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        gradient_norm = torch.nn.utils.clip_grad_norm_(translator.parameters(), 1.0)
        if not bool(torch.isfinite(gradient_norm.detach())):
            raise FloatingPointError(f"nonfinite gradient norm at step={next_step}")
        optimizer.step()
        if any(
            parameter.grad is not None
            for model in models.values()
            for parameter in model.parameters()
        ):
            raise RuntimeError("a frozen language-model parameter received a gradient")
        rank_records = gather_rank_records(
            {
                "rank": context["rank"],
                "samples": local_samples,
                "compute_seconds": local_compute_seconds,
                "reduction_seconds": reduction_seconds,
            },
            context,
        )
        sample_records = sorted(
            (
                sample
                for rank_record in rank_records
                for sample in rank_record["samples"]
            ),
            key=lambda value: value["offset"],
        )
        if [value["offset"] for value in sample_records] != list(
            range(EFFECTIVE_BATCH)
        ):
            raise RuntimeError("distributed batch did not cover every row exactly once")
        step = next_step
        cursor += EFFECTIVE_BATCH
        metric = {
            "step": step,
            "sample_cursor": cursor,
            "mean_kl": sum(value["kl"] for value in sample_records) / EFFECTIVE_BATCH,
            "sample_kls": [value["kl"] for value in sample_records],
            "sample_seconds": [value["seconds"] for value in sample_records],
            "learning_rate": lr,
            "gradient_norm_before_clip": float(gradient_norm),
            "prefix_tokens": [row["prefix_tokens"] for row in batch_rows],
            "continuation_tokens": [row["continuation_tokens"] for row in batch_rows],
            "world_size": context["world_size"],
            "rank_assignment": [
                [row["index"] for _offset, row in assignment]
                for assignment in assignments
            ],
            "rank_compute_seconds": [
                value["compute_seconds"] for value in rank_records
            ],
            "rank_reduction_seconds": [
                value["reduction_seconds"] for value in rank_records
            ],
            "global_step_seconds": time.perf_counter() - step_started,
        }
        metrics.append(metric)
        log(
            f"{args.direction} step {step}/{args.max_steps}: "
            f"KL={metric['mean_kl']:.6f}, lr={lr:.3e}, "
            f"elapsed={metric['global_step_seconds']:.2f}s"
        )
        if step in set(args.checkpoint_steps) or step == args.max_steps:
            rank_rng = gather_rank_rng(context)
            rank_resources = gather_rank_records(
                {
                    "rank": context["rank"],
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                    "wall_seconds_this_invocation": time.time() - started,
                },
                context,
            )
            path = args.out / f"stage2-{args.direction}-step{step}.pt"
            if context["rank"] == 0:
                atomic_torch_save(
                    checkpoint_state(
                        args.direction,
                        translator,
                        optimizer,
                        global_step=step,
                        sample_cursor=cursor,
                        source_artifact=stage1_path,
                        metrics=metrics,
                        execution=execution,
                        rank_rng=rank_rng,
                    ),
                    path,
                )
                checkpoint_hash = file_sha256(path)
                resume_contract_receipt = None
                if args.verified_resume_contract is not None:
                    resume_contract = next_resume_contract(
                        args, checkpoint_hash, step, cursor, execution
                    )
                    contract_path = (
                        args.out
                        / f"RESUME_CONTRACT_{args.direction.upper()}_STEP{step}.json"
                    )
                    write_json(contract_path, resume_contract)
                    resume_contract_receipt = {
                        "path": contract_path.name,
                        "sha256": file_sha256(contract_path),
                    }
                write_json(
                    args.out / f"STAGE2_{args.direction.upper()}_STEP{step}.json",
                    {
                        "schema": (
                            "kv_lingo_stage2_distributed_receipt_v1"
                            if context["enabled"]
                            else "kv_lingo_stage2_receipt_v1"
                        ),
                        "direction": args.direction,
                        "global_step": step,
                        "sample_cursor": cursor,
                        "checkpoint": {
                            "path": path.name,
                            "sha256": checkpoint_hash,
                        },
                        "resume_contract": resume_contract_receipt,
                        "execution_topology": execution,
                        "wall_seconds_this_invocation": max(
                            value["wall_seconds_this_invocation"]
                            for value in rank_resources
                        ),
                        "aggregate_gpu_hours_this_invocation": sum(
                            value["wall_seconds_this_invocation"]
                            for value in rank_resources
                        )
                        / 3600,
                        "peak_allocated_gib": max(
                            value["peak_allocated_gib"] for value in rank_resources
                        ),
                        "peak_reserved_gib": max(
                            value["peak_reserved_gib"] for value in rank_resources
                        ),
                        "rank_resources": rank_resources,
                        "metrics": metrics,
                    },
                )
            if context["enabled"]:
                dist.barrier()
    return step


@torch.no_grad()
def validate(args, models, rows, store):
    source_size, target_size = (
        ("4b", "8b") if args.direction == "4b-to-8b" else ("8b", "4b")
    )
    source, target = models[source_size], models[target_size]
    device = next(target.parameters()).device
    value = torch.load(args.translator, map_location="cpu", weights_only=False)
    if value.get("schema") == "kv_lingo_stage2_checkpoint_v1":
        sg, tg = value["source_geometry"], value["target_geometry"]
        translator = LinearTranslator(
            CacheGeometry(sg["layers"], sg["kv_heads"], sg["head_dim"]),
            CacheGeometry(tg["layers"], tg["kv_heads"], tg["head_dim"]),
        )
        translator = translator.to(device)
        translator.load_state_dict(value["translator"])
        stage = f"stage2_step{value['global_step']}"
    else:
        translator = translator_from_artifact(value, device)
        stage = value["stage"]
    losses = []
    samples = []
    started = time.time()
    for row in rows[: args.validation_samples]:
        prefix, continuation = sample_ids(row, store, device)
        loss, temporaries = one_stage2_sample(
            source, target, translator, prefix, continuation
        )
        losses.append(float(loss))
        samples.append(
            {
                "index": row["index"],
                "kind": row["kind"],
                "reasoning": row["reasoning"],
                "prefix_tokens": row["prefix_tokens"],
                "continuation_tokens": row["continuation_tokens"],
                "kl": float(loss),
            }
        )
        del prefix, continuation, loss, temporaries
        gc.collect()
        torch.cuda.empty_cache()
    receipt = {
        "schema": "kv_lingo_validation_v1",
        "direction": args.direction,
        "stage": stage,
        "translator": {
            "path": str(args.translator),
            "sha256": file_sha256(args.translator),
        },
        "samples": len(losses),
        "mean_kl": sum(losses) / len(losses),
        "sample_rows": samples,
        "wall_seconds": time.time() - started,
    }
    suffix = args.translator.stem.upper().replace("-", "_")
    write_json(args.out / f"VALIDATION_{suffix}.json", receipt)
    return receipt


def resource_receipt(models) -> dict:
    return {
        "gpu": torch.cuda.get_device_name(),
        "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "loaded_models_gib": torch.cuda.memory_allocated() / 2**30,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "peak_host_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "models_frozen": all(
            not parameter.requires_grad
            for model in models.values()
            for parameter in model.parameters()
        ),
    }


def collect_resource_receipt(models, context: dict) -> dict:
    local = {
        "rank": context["rank"],
        "local_rank": context["local_rank"],
        **resource_receipt(models),
    }
    records = gather_rank_records(local, context)
    if not context["enabled"]:
        return records[0]
    return {
        "schema": "kv_lingo_distributed_resource_v1",
        "world_size": context["world_size"],
        "backend": "nccl",
        "ranks": records,
        "aggregate_allocated_gpus": context["world_size"],
        "all_models_frozen": all(value["models_frozen"] for value in records),
    }


def parse_steps(value: str) -> list[int]:
    return sorted({int(cell) for cell in value.split(",") if cell})


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("stage1", "train", "validate"))
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--direction", choices=tuple(DIRECTIONS))
    parser.add_argument("--stage1", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-contract", type=Path)
    parser.add_argument("--validation-rows", type=Path)
    parser.add_argument("--validation-tokens", type=Path)
    parser.add_argument("--translator", type=Path)
    parser.add_argument("--stage1-samples", type=int, default=400)
    parser.add_argument("--stage1-checkpoint-every", type=int, default=100)
    parser.add_argument("--stage1-tolerance", type=float, default=1e-8)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--checkpoint-steps", type=parse_steps, default=[4, 250, 1000])
    parser.add_argument("--validation-samples", type=int, default=64)
    parser.add_argument("--source-commit")
    args = parser.parse_args(argv)
    if args.action in ("train", "validate") and not args.direction:
        parser.error(f"{args.action} requires --direction")
    if args.action == "validate" and not args.translator:
        parser.error("validate requires --translator")

    context = distributed_environment()
    if context["enabled"] and args.action != "train":
        parser.error("distributed execution currently supports Stage-2 training only")
    resume_contract = validate_resume_contract(args)
    if resume_contract is None:
        resume_contract = validate_new_run_contract(args)
    args.verified_resume_contract = resume_contract
    args.distributed_context = context
    args.out.mkdir(parents=True, exist_ok=True)
    if resume_contract is not None and context["rank"] == 0:
        write_json(args.out / "RESUME_PREFLIGHT.json", resume_contract)
    rows = read_rows(args.rows)
    store = TokenStore(args.tokens)
    initialize_distributed(context)
    try:
        torch.manual_seed(42)
        if context["enabled"]:
            torch.cuda.manual_seed(42)
            device = f"cuda:{context['local_rank']}"
        else:
            torch.cuda.manual_seed_all(42)
            device = "cuda"
        models = load_models(device)
        start_receipt = collect_resource_receipt(models, context)
        if context["rank"] == 0:
            write_json(args.out / "RESOURCE_START.json", start_receipt)
        if args.action == "stage1":
            run_stage1(args, models, rows, store)
        elif args.action == "train":
            run_stage2(args, models, rows, store)
        else:
            validate(args, models, rows, store)
        receipt = collect_resource_receipt(models, context)
        if context["enabled"]:
            for rank_receipt in receipt["ranks"]:
                rank_receipt["peak_allocated_gib"] = rank_receipt.get(
                    "peak_allocated_gib", 0
                )
                rank_receipt["peak_reserved_gib"] = rank_receipt.get(
                    "peak_reserved_gib", 0
                )
        else:
            receipt["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            receipt["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        if context["rank"] == 0:
            write_json(args.out / "RESOURCE_END.json", receipt)
        return 0
    finally:
        if context["enabled"] and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())

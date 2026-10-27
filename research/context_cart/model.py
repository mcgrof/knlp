# SPDX-License-Identifier: MIT
"""Writer and reader for generated prefix key/value memory.

The writer is a frozen base model with rank-limited adapters on the attention
projections plus ``memory_slots`` trainable input embeddings. It reads the
source tokens followed by the memory embeddings and returns the per-layer
key/value rows at the memory positions. The reader is a second, untouched
instance of the same checkpoint. It receives only those rows as its initial
attention cache, then the instruction and question tokens.

Positions follow one contract. Let ``L`` be the number of source positions
the writer processed. Memory rows carry logical positions ``L .. L+M-1`` and
the first reader token has logical position ``L+M`` while its physical cache
index is ``M``. Keys are stored after rotary embedding, exactly as the writer
produced them. The reader never allocates blank rows for the source.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


@dataclasses.dataclass
class ModelSpec:
    name: str = DEFAULT_MODEL
    revision: str = DEFAULT_REVISION
    dtype: str = "bfloat16"
    memory_slots: int = 64
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    attn_implementation: str = "sdpa"
    local_files_only: bool = True
    tokenizer_name: str | None = None

    @property
    def is_tiny(self) -> bool:
        """``tiny`` selects a random small Qwen3 for CPU pipeline smoke tests;
        it shares the real tokenizer so prompts and datasets are unchanged."""
        return self.name.startswith("tiny")

    def torch_dtype(self) -> torch.dtype:
        return {
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "float16": torch.float16,
        }[self.dtype]


@dataclasses.dataclass(frozen=True)
class Geometry:
    layers: int
    kv_heads: int
    head_dim: int
    hidden: int
    heads: int
    vocab: int

    @classmethod
    def from_config(cls, config: Any) -> "Geometry":
        return cls(
            layers=int(config.num_hidden_layers),
            kv_heads=int(config.num_key_value_heads),
            head_dim=int(
                getattr(
                    config, "head_dim", config.hidden_size // config.num_attention_heads
                )
            ),
            hidden=int(config.hidden_size),
            heads=int(config.num_attention_heads),
            vocab=int(config.vocab_size),
        )

    def bytes_per_position(self, dtype_bytes: int = 2) -> int:
        return 2 * self.layers * self.kv_heads * self.head_dim * dtype_bytes


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def load_tokenizer(spec: ModelSpec) -> Any:
    from transformers import AutoTokenizer

    name = spec.tokenizer_name or (DEFAULT_MODEL if spec.is_tiny else spec.name)
    revision = (
        DEFAULT_REVISION if spec.is_tiny and not spec.tokenizer_name else spec.revision
    )
    return AutoTokenizer.from_pretrained(
        name, revision=revision, local_files_only=spec.local_files_only
    )


def load_base(spec: ModelSpec, device: torch.device | str) -> nn.Module:
    """A frozen causal LM instance in evaluation mode."""
    from transformers import AutoModelForCausalLM

    if spec.is_tiny:
        vocab = int(spec.name.split(":")[1]) if ":" in spec.name else 151936
        model = tiny_base(
            seed=0,
            dtype=spec.torch_dtype(),
            vocab_size=vocab,
            hidden_size=64,
            num_hidden_layers=3,
        )
        model.to(device)
        return model
    model = AutoModelForCausalLM.from_pretrained(
        spec.name,
        revision=spec.revision,
        dtype=spec.torch_dtype(),
        attn_implementation=spec.attn_implementation,
        local_files_only=spec.local_files_only,
    )
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def tiny_config(**overrides: Any) -> Any:
    """A deterministic small Qwen3 configuration for CPU tests."""
    from transformers import Qwen3Config

    values = dict(
        vocab_size=512,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=8192,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        rms_norm_eps=1e-6,
    )
    values.update(overrides)
    return Qwen3Config(**values)


def tiny_base(
    seed: int = 0, dtype: torch.dtype = torch.float32, **overrides: Any
) -> nn.Module:
    from transformers import Qwen3ForCausalLM

    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(tiny_config(**overrides)).to(dtype)
    model.config._attn_implementation = "sdpa"
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def rope_digest(config: Any, head_dim: int) -> str:
    """Identity of the rotary convention the stored keys were produced under."""
    scaling = getattr(config, "rope_scaling", None)
    if scaling is not None and not isinstance(scaling, dict):
        scaling = dict(scaling)
    return sha256_bytes(
        canonical(
            {
                "rope_theta": float(config.rope_theta),
                "rope_scaling": scaling,
                "head_dim": head_dim,
                "max_position_embeddings": int(config.max_position_embeddings),
                "convention": "keys stored after rotary embedding at logical positions",
            }
        ).encode()
    )


def config_digest(config: Any) -> str:
    data = config.to_dict()
    for key in (
        "transformers_version",
        "_name_or_path",
        "_attn_implementation_autoset",
    ):
        data.pop(key, None)
    data.pop("_attn_implementation_internal", None)
    return sha256_bytes(canonical(data).encode())


def weights_digest(model: nn.Module, sample: int = 8) -> str:
    """Digest of a fixed sample of parameter tensors; cheap identity check
    that two processes loaded the same weights."""
    names = sorted(name for name, _ in model.named_parameters())
    step = max(1, len(names) // sample)
    chosen = names[::step][:sample]
    params = dict(model.named_parameters())
    digest = hashlib.sha256()
    for name in chosen:
        tensor = params[name].detach().to(torch.float32).cpu().contiguous()
        digest.update(name.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def tokenizer_digest(tokenizer: Any) -> str:
    """Digest over the tokenizer vocabulary and chat template."""
    vocab = tokenizer.get_vocab()
    items = sorted(vocab.items(), key=lambda kv: kv[1])
    digest = hashlib.sha256()
    for token, index in items:
        digest.update(f"{index}\t{token}\n".encode())
    digest.update((tokenizer.chat_template or "").encode())
    return digest.hexdigest()


def reader_identity(
    model: nn.Module, tokenizer: Any, spec: ModelSpec
) -> dict[str, Any]:
    geometry = Geometry.from_config(model.config)
    return {
        "model": spec.name,
        "revision": spec.revision,
        "dtype": spec.dtype,
        "config_digest": config_digest(model.config),
        "weights_digest": weights_digest(model),
        "tokenizer_digest": tokenizer_digest(tokenizer),
        "rope_digest": rope_digest(model.config, geometry.head_dim),
        "geometry": dataclasses.asdict(geometry),
    }


def nonspecial_vocab_rows(
    tokenizer: Any, count: int, seed: int, vocab: int
) -> list[int]:
    """``count`` distinct ordinary vocabulary ids chosen with ``seed``."""
    import random

    special = set(tokenizer.all_special_ids)
    added = set(getattr(tokenizer, "added_tokens_decoder", {}).keys())
    rng = random.Random(seed)
    chosen: list[int] = []
    seen: set[int] = set()
    limit = min(vocab, len(tokenizer))
    while len(chosen) < count:
        index = rng.randrange(0, limit)
        if index in special or index in added or index in seen:
            continue
        seen.add(index)
        chosen.append(index)
    return chosen


class Writer(nn.Module):
    """Frozen base + adapters + trainable memory embeddings."""

    def __init__(self, base: nn.Module, spec: ModelSpec, init_rows: list[int]):
        super().__init__()
        from peft import LoraConfig, get_peft_model

        self.spec = spec
        self.geometry = Geometry.from_config(base.config)
        config = LoraConfig(
            r=spec.lora_rank,
            lora_alpha=spec.lora_alpha,
            lora_dropout=spec.lora_dropout,
            target_modules=list(spec.lora_targets),
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.peft = get_peft_model(base, config)
        embed = self.inner.embed_tokens.weight.detach()
        rows = (
            embed[torch.tensor(init_rows, device=embed.device)]
            .to(torch.float32)
            .clone()
        )
        self.memory = nn.Parameter(rows)
        self.init_rows = list(init_rows)
        for name, parameter in self.peft.named_parameters():
            if "lora_" in name:
                parameter.data = parameter.data.to(torch.float32)
                parameter.requires_grad_(True)
            else:
                parameter.requires_grad_(False)

    @property
    def inner(self) -> nn.Module:
        """The decoder stack (``Qwen3Model``) with adapters injected."""
        return self.peft.base_model.model.model

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_state(self) -> dict[str, torch.Tensor]:
        from peft import get_peft_model_state_dict

        state = {
            f"lora.{k}": v.detach().cpu()
            for k, v in get_peft_model_state_dict(self.peft).items()
        }
        state["memory"] = self.memory.detach().cpu()
        return state

    def load_trainable_state(self, state: dict[str, torch.Tensor]) -> None:
        from peft import set_peft_model_state_dict

        lora = {k[len("lora.") :]: v for k, v in state.items() if k.startswith("lora.")}
        set_peft_model_state_dict(self.peft, lora)
        with torch.no_grad():
            self.memory.copy_(state["memory"].to(self.memory.device, self.memory.dtype))

    def encode(
        self, source_ids: torch.Tensor
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], int]:
        """Return per-layer (key, value) rows at the memory positions, each
        shaped ``[B, kv_heads, M, head_dim]`` in the base dtype, and ``L``.
        Source rows are discarded; nothing else leaves the writer."""
        if source_ids.dim() != 2:
            raise ValueError("source_ids must be [B, L]")
        batch, length = source_ids.shape
        slots = self.memory.shape[0]
        embed = self.inner.embed_tokens(source_ids)
        memory = self.memory.to(embed.dtype).unsqueeze(0).expand(batch, -1, -1)
        inputs = torch.cat([embed, memory], dim=1)
        positions = (
            torch.arange(length + slots, device=inputs.device)
            .unsqueeze(0)
            .expand(batch, -1)
        )
        out = self.inner(
            inputs_embeds=inputs,
            position_ids=positions,
            attention_mask=torch.ones(
                batch, length + slots, dtype=torch.long, device=inputs.device
            ),
            use_cache=True,
        )
        cache = out.past_key_values
        prefix = []
        for layer in range(self.geometry.layers):
            keys, values = cache_layer(cache, layer)
            if keys.shape[2] != length + slots:
                raise RuntimeError("writer cache length does not match the input")
            prefix.append((keys[:, :, length:, :], values[:, :, length:, :]))
        return prefix, length


def cache_layer(cache: Any, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(cache, "layers"):
        return cache.layers[layer].keys, cache.layers[layer].values
    return cache.key_cache[layer], cache.value_cache[layer]


def build_cache(prefix: list[tuple[torch.Tensor, torch.Tensor]] | None) -> Any:
    """A fresh cache holding exactly the prefix rows. Each reader request
    receives its own cache object, so extension never touches the prefix."""
    from transformers import DynamicCache

    cache = DynamicCache()
    if prefix is None:
        return cache
    for layer, (keys, values) in enumerate(prefix):
        cache.update(keys, values, layer)
    return cache


def prefix_rows(prefix: list[tuple[torch.Tensor, torch.Tensor]] | None) -> int:
    return 0 if prefix is None else int(prefix[0][0].shape[2])


def to_thd(
    prefix: list[tuple[torch.Tensor, torch.Tensor]], index: int = 0
) -> dict[str, torch.Tensor]:
    """One batch element as ``key.N``/``value.N`` tensors in THD layout:
    token, kv head, head dimension."""
    tensors = {}
    for layer, (keys, values) in enumerate(prefix):
        tensors[f"key.{layer}"] = (
            keys[index].detach().permute(1, 0, 2).contiguous().cpu()
        )
        tensors[f"value.{layer}"] = (
            values[index].detach().permute(1, 0, 2).contiguous().cpu()
        )
    return tensors


def from_thd(
    tensors: dict[str, torch.Tensor], layers: int, device: torch.device | str
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    prefix = []
    for layer in range(layers):
        keys = (
            tensors[f"key.{layer}"]
            .to(device)
            .permute(1, 0, 2)
            .unsqueeze(0)
            .contiguous()
        )
        values = (
            tensors[f"value.{layer}"]
            .to(device)
            .permute(1, 0, 2)
            .unsqueeze(0)
            .contiguous()
        )
        prefix.append((keys, values))
    return prefix


def stack_prefixes(
    items: list[list[tuple[torch.Tensor, torch.Tensor]]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Concatenate single-element prefixes along the batch dimension."""
    layers = len(items[0])
    return [
        (
            torch.cat([p[layer][0] for p in items], dim=0),
            torch.cat([p[layer][1] for p in items], dim=0),
        )
        for layer in range(layers)
    ]


def prefix_digest(
    prefix: list[tuple[torch.Tensor, torch.Tensor]], index: int = 0
) -> str:
    digest = hashlib.sha256()
    for layer, (keys, values) in enumerate(prefix):
        for tensor in (keys[index], values[index]):
            digest.update(
                tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
            )
    return digest.hexdigest()


@dataclasses.dataclass
class Trace:
    """What the reader actually processed in one forward call."""

    input_tokens: int
    cache_rows_before: int
    cache_rows_after: int
    first_position: int
    source_tokens_processed: int = 0


class Reader:
    """Frozen reader with a differentiable prefix path and a greedy decoder."""

    def __init__(self, model: nn.Module, eos_ids: tuple[int, ...], pad_id: int):
        self.model = model
        self.inner = model.model
        self.geometry = Geometry.from_config(model.config)
        self.eos_ids = tuple(int(i) for i in eos_ids)
        self.pad_id = int(pad_id)
        self.traces: list[Trace] = []
        self.record_traces = False

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def forward(
        self,
        prefix: list[tuple[torch.Tensor, torch.Tensor]] | None,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        first_position: int | torch.Tensor,
        cache: Any = None,
        position_ids: torch.Tensor | None = None,
        prefix_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Any]:
        """Hidden states for ``input_ids`` given the prefix rows.

        ``attention_mask`` covers the reader tokens only. Prefix rows are
        visible unless ``prefix_mask`` marks them as padding (text arms that
        prefill contexts of different lengths in one batch). ``first_position``
        is the logical position of the first non-padded reader token, one
        integer or one per batch row.
        """
        batch, length = input_ids.shape
        rows = prefix_rows(prefix) if cache is None else cache.get_seq_length()
        if cache is None:
            cache = build_cache(prefix)
        if position_ids is None:
            start = first_position
            if isinstance(start, torch.Tensor):
                start = start.to(attention_mask.device).view(-1, 1)
            position_ids = start + (attention_mask.cumsum(-1) - 1).clamp(min=0)
        if prefix_mask is None:
            prefix_mask = torch.ones(
                batch, rows, dtype=attention_mask.dtype, device=attention_mask.device
            )
        full_mask = torch.cat(
            [prefix_mask.to(attention_mask.dtype), attention_mask], dim=1
        )
        out = self.inner(
            input_ids=input_ids,
            attention_mask=full_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
        if self.record_traces:
            self.traces.append(
                Trace(
                    input_tokens=(
                        int(attention_mask.sum().item()) if batch == 1 else int(length)
                    ),
                    cache_rows_before=int(rows),
                    cache_rows_after=int(cache.get_seq_length()),
                    first_position=int(first_position),
                )
            )
        return out.last_hidden_state, cache

    def answer_loss(
        self,
        prefix: list[tuple[torch.Tensor, torch.Tensor]] | None,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        first_position: int,
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        """Summed answer-token cross-entropy in FP32, the token count, and
        per-example summed losses. Vocabulary logits are materialised only at
        scored positions."""
        hidden, _ = self.forward(prefix, input_ids, attention_mask, first_position)
        shift_labels = labels[:, 1:]
        scored = shift_labels != -100
        selected = hidden[:, :-1][scored]
        logits = self.model.lm_head(selected).float()
        targets = shift_labels[scored]
        losses = nn.functional.cross_entropy(logits, targets, reduction="none")
        per_example = torch.zeros(
            input_ids.shape[0], device=losses.device, dtype=losses.dtype
        )
        example_index = (
            torch.arange(input_ids.shape[0], device=losses.device)
            .unsqueeze(1)
            .expand_as(shift_labels)[scored]
        )
        per_example.index_add_(0, example_index, losses)
        return losses.sum(), int(targets.numel()), per_example

    @torch.no_grad()
    def greedy(
        self,
        prefix: list[tuple[torch.Tensor, torch.Tensor]] | None,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        first_position: int | torch.Tensor,
        max_new_tokens: int,
        keep_logits: bool = False,
        prefix_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Batched greedy decoding with left-padded prompts.

        Returns generated ids (padded with ``pad_id`` after the end token),
        FP32 first-token logits, optional FP32 per-step logits, and timings.
        """
        device = prompt_ids.device
        batch = prompt_ids.shape[0]
        sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
        sync()
        t0 = time.perf_counter()
        hidden, cache = self.forward(
            prefix, prompt_ids, prompt_mask, first_position, prefix_mask=prefix_mask
        )
        logits = self.model.lm_head(hidden[:, -1]).float()
        sync()
        prefill_seconds = time.perf_counter() - t0
        first_logits = logits.clone()
        step_logits = [logits.clone()] if keep_logits else []
        if prefix_mask is None:
            prefix_mask = torch.ones(
                batch, prefix_rows(prefix), dtype=prompt_mask.dtype, device=device
            )
        mask = torch.cat([prefix_mask.to(prompt_mask.dtype), prompt_mask], dim=1)
        start = first_position
        if isinstance(start, torch.Tensor):
            start = start.to(device).view(-1)
        last_position = start + (prompt_mask.sum(-1) - 1).clamp(min=0)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        generated = []
        eos = torch.tensor(self.eos_ids, device=device)
        t1 = time.perf_counter()
        for step in range(max_new_tokens):
            next_token = logits.argmax(-1)
            next_token = torch.where(
                finished, torch.full_like(next_token, self.pad_id), next_token
            )
            generated.append(next_token)
            finished = finished | torch.isin(next_token, eos)
            if bool(finished.all()) or step == max_new_tokens - 1:
                break
            mask = torch.cat(
                [mask, torch.ones(batch, 1, dtype=mask.dtype, device=device)], dim=1
            )
            last_position = last_position + 1
            position_ids = last_position.unsqueeze(1)
            out = self.inner(
                input_ids=next_token.unsqueeze(1),
                attention_mask=mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
            )
            logits = self.model.lm_head(out.last_hidden_state[:, -1]).float()
            if keep_logits:
                step_logits.append(logits.clone())
        sync()
        decode_seconds = time.perf_counter() - t1
        ids = torch.stack(generated, dim=1)
        return {
            "ids": ids,
            "first_logits": first_logits,
            "step_logits": torch.stack(step_logits, dim=1) if keep_logits else None,
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "prompt_tokens": int(prompt_mask.sum().item()),
            "cache_rows": int(cache.get_seq_length()),
        }


def strip_generation(
    ids: list[int], eos_ids: tuple[int, ...], pad_id: int
) -> list[int]:
    out = []
    for token in ids:
        if token in eos_ids or token == pad_id:
            break
        out.append(int(token))
    return out


def left_pad(
    sequences: list[list[int]], pad_id: int, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, seq in enumerate(sequences):
        ids[row, width - len(seq) :] = torch.tensor(seq, dtype=torch.long)
        mask[row, width - len(seq) :] = 1
    return ids.to(device), mask.to(device)


def right_pad_with_labels(
    prompts: list[list[int]],
    answers: list[list[int]],
    pad_id: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(p) + len(a) for p, a in zip(prompts, answers))
    ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(prompts), width), dtype=torch.long)
    labels = torch.full((len(prompts), width), -100, dtype=torch.long)
    for row, (p, a) in enumerate(zip(prompts, answers)):
        seq = p + a
        ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[row, : len(seq)] = 1
        labels[row, len(p) : len(seq)] = torch.tensor(a, dtype=torch.long)
    return ids.to(device), mask.to(device), labels.to(device)


def reference_hidden(
    model: nn.Module,
    prefix: list[tuple[torch.Tensor, torch.Tensor]],
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Dense masked reference for the prefix path, batch one, no padding.

    Recomputes the decoder with explicit attention over ``[prefix rows,
    reader tokens]`` using the model's own submodules. Prefix rows are
    visible to every reader token; reader tokens are causal among themselves.
    """
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

    inner = model.model
    hidden = inner.embed_tokens(input_ids)
    cos, sin = inner.rotary_emb(hidden, position_ids)
    batch, length, _ = hidden.shape
    rows = prefix[0][0].shape[2]
    allowed = torch.ones(length, rows + length, dtype=torch.bool, device=hidden.device)
    allowed[:, rows:] = torch.tril(
        torch.ones(length, length, dtype=torch.bool, device=hidden.device)
    )
    for layer, (pk, pv) in zip(inner.layers, prefix):
        attn = layer.self_attn
        x = layer.input_layernorm(hidden)
        heads, kv_heads, dim = (
            attn.config.num_attention_heads,
            attn.config.num_key_value_heads,
            attn.head_dim,
        )
        q = attn.q_norm(attn.q_proj(x).view(batch, length, heads, dim)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(x).view(batch, length, kv_heads, dim)).transpose(
            1, 2
        )
        v = attn.v_proj(x).view(batch, length, kv_heads, dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        keys = repeat_kv(torch.cat([pk, k], dim=2), heads // kv_heads)
        values = repeat_kv(torch.cat([pv, v], dim=2), heads // kv_heads)
        scores = torch.matmul(q, keys.transpose(-1, -2)) * attn.scaling
        scores = scores.masked_fill(~allowed, float("-inf"))
        probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        context = (
            torch.matmul(probs, values)
            .transpose(1, 2)
            .reshape(batch, length, heads * dim)
        )
        hidden = hidden + attn.o_proj(context)
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    return inner.norm(hidden)


def cosine_lr(step: int, total: int, warmup: int, peak: float, floor: float) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))

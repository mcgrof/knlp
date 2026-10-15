"""Transformers decoder backend for the DroidSpeak recomputation mechanism.

This backend measures an in-process algorithm baseline. It does not model
LMCache or network transfer time.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Any, Iterator

from .profile import RecomputeInterval

ARCHITECTURE_FIELDS = (
    "model_type",
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "hidden_act",
    "max_position_embeddings",
    "rope_theta",
    "sliding_window",
)


HOTPOT_PREFIX = (
    "Answer the question based on the given passages. Only give me the answer "
    "and do not output any other words.\n\n"
    "The following are given passages.\n"
)
HOTPOT_SUFFIX = (
    "\n\nAnswer the question based on the given passages. Only give me the "
    "answer and do not output any other words.\n\nQuestion: {question}\nAnswer:"
)


@dataclass
class TokenizedPrompt:
    context_ids: Any
    query_ids: Any
    context_tokens_original: int
    context_tokens_used: int


@dataclass
class SenderState:
    kv_cache: Any
    transition_states: dict[int, Any]


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int]
    ttft_s: float
    prefill_s: float
    decode_s: float
    context_tokens: int
    query_tokens: int


def architecture_signature(config: Any) -> dict[str, Any]:
    return {field: getattr(config, field, None) for field in ARCHITECTURE_FIELDS}


def validate_pair(
    sender: Any, receiver: Any, sender_tokenizer: Any, receiver_tokenizer: Any
) -> None:
    """Reject model pairs whose caches cannot be combined layer by layer."""

    sender_signature = architecture_signature(sender.config)
    receiver_signature = architecture_signature(receiver.config)
    differences = {
        field: (sender_signature[field], receiver_signature[field])
        for field in ARCHITECTURE_FIELDS
        if sender_signature[field] != receiver_signature[field]
    }
    if differences:
        raise ValueError(f"sender and receiver architecture mismatch: {differences}")
    if sender_tokenizer.get_vocab() != receiver_tokenizer.get_vocab():
        raise ValueError("sender and receiver token vocabularies differ")


def decoder_layers(model: Any) -> Any:
    """Return decoder layers from a plain or PEFT-wrapped causal LM."""

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if not hasattr(base, "model") or not hasattr(base.model, "layers"):
        raise ValueError("model does not expose a supported decoder layer stack")
    return base.model.layers


def _middle_truncate(token_ids: list[int], limit: int) -> list[int]:
    if len(token_ids) <= limit:
        return token_ids
    left = limit // 2
    return token_ids[:left] + token_ids[-(limit - left) :]


def tokenize_hotpot_prompt(
    tokenizer: Any,
    context: str,
    question: str,
    max_context_tokens: int,
) -> tuple[list[int], list[int], int]:
    """Tokenize the whole input prompt and reserve its final input token.

    DroidSpeak reuses the sender KV for the whole input prompt. As with a
    normal full-cache hit, the receiver processes the final input token to
    produce logits for the first output token.
    """

    if max_context_tokens <= 0:
        raise ValueError("max_context_tokens must be positive")
    passage_ids = tokenizer.encode(
        HOTPOT_PREFIX + context,
        add_special_tokens=True,
    )
    suffix_ids = tokenizer.encode(
        HOTPOT_SUFFIX.format(question=question),
        add_special_tokens=False,
    )
    original_prompt_tokens = len(passage_ids) + len(suffix_ids)
    passage_ids = _middle_truncate(passage_ids, max_context_tokens)
    prompt_ids = passage_ids + suffix_ids
    if len(prompt_ids) < 2:
        raise ValueError("HotpotQA prompt must contain at least two tokens")
    return prompt_ids[:-1], prompt_ids[-1:], original_prompt_tokens - 1


class Qwen2DroidSpeak:
    """Run sender capture and receiver selective recomputation."""

    def __init__(
        self,
        sender: Any,
        receiver: Any,
        tokenizer: Any,
        device: str = "cuda",
    ) -> None:
        import torch

        self.torch = torch
        self.sender = sender.eval()
        self.receiver = receiver.eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.sender_layers = decoder_layers(self.sender)
        self.receiver_layers = decoder_layers(self.receiver)
        self.num_layers = len(self.receiver_layers)
        if len(self.sender_layers) != self.num_layers:
            raise ValueError("sender and receiver layer counts differ")
        eos_ids = receiver.generation_config.eos_token_id
        if eos_ids is None:
            eos_ids = tokenizer.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        self.eos_ids = set(eos_ids or [])

    @classmethod
    def load(
        cls,
        sender_model: str,
        receiver_model: str,
        sender_revision: str | None = None,
        receiver_revision: str | None = None,
        sender_adapter: str | None = None,
        receiver_adapter: str | None = None,
        sender_adapter_revision: str | None = None,
        receiver_adapter_revision: str | None = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
    ) -> "Qwen2DroidSpeak":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_value = getattr(torch, dtype)
        common = {
            "torch_dtype": dtype_value,
            "attn_implementation": attn_implementation,
            "trust_remote_code": False,
        }
        sender_tokenizer = AutoTokenizer.from_pretrained(
            sender_model,
            revision=sender_revision,
            trust_remote_code=False,
        )
        receiver_tokenizer = AutoTokenizer.from_pretrained(
            receiver_model,
            revision=receiver_revision,
            trust_remote_code=False,
        )
        sender = AutoModelForCausalLM.from_pretrained(
            sender_model,
            revision=sender_revision,
            **common,
        )
        receiver = AutoModelForCausalLM.from_pretrained(
            receiver_model,
            revision=receiver_revision,
            **common,
        )
        if sender_adapter or receiver_adapter:
            from peft import PeftModel

            if sender_adapter:
                sender = PeftModel.from_pretrained(
                    sender,
                    sender_adapter,
                    revision=sender_adapter_revision,
                )
            if receiver_adapter:
                receiver = PeftModel.from_pretrained(
                    receiver,
                    receiver_adapter,
                    revision=receiver_adapter_revision,
                )
        sender = sender.to(device)
        receiver = receiver.to(device)
        validate_pair(sender, receiver, sender_tokenizer, receiver_tokenizer)
        return cls(sender, receiver, receiver_tokenizer, device=device)

    def tokenize_hotpot(
        self,
        context: str,
        question: str,
        max_context_tokens: int,
    ) -> TokenizedPrompt:
        """Split the full input prompt at its final input token."""

        context_ids, query_ids, original_count = tokenize_hotpot_prompt(
            self.tokenizer,
            context,
            question,
            max_context_tokens,
        )
        return TokenizedPrompt(
            context_ids=self.torch.tensor(
                [context_ids], dtype=self.torch.long, device=self.device
            ),
            query_ids=self.torch.tensor(
                [query_ids], dtype=self.torch.long, device=self.device
            ),
            context_tokens_original=original_count,
            context_tokens_used=len(context_ids),
        )

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    @contextlib.contextmanager
    def _receiver_interval(
        self,
        interval: RecomputeInterval,
        transition_state: Any | None,
    ) -> Iterator[None]:
        layers = self.receiver_layers
        original_forwards: dict[int, Any] = {}
        hook = None
        for layer_idx, layer in enumerate(layers):
            if interval.contains(layer_idx):
                continue
            original_forwards[layer_idx] = layer.forward
            layer.forward = lambda hidden_states, *args, **kwargs: hidden_states

        if interval.start > 0 and interval.layer_count:
            if transition_state is None:
                raise ValueError("transition state is required for a nonzero start")

            def inject_state(
                module: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
            ) -> tuple[tuple[Any, ...], dict[str, Any]]:
                del module
                return (transition_state, *args[1:]), kwargs

            hook = layers[interval.start].register_forward_pre_hook(
                inject_state,
                with_kwargs=True,
            )
        try:
            yield
        finally:
            if hook is not None:
                hook.remove()
            for layer_idx, forward in original_forwards.items():
                layers[layer_idx].forward = forward

    def capture_sender(self, context_ids: Any) -> SenderState:
        """Build sender KV and all profiler transition E states."""

        transition_states: dict[int, Any] = {}
        handles = []

        def capture(layer_idx: int):
            def hook(module: Any, args: tuple[Any, ...]) -> None:
                del module
                transition_states[layer_idx] = args[0].detach()

            return hook

        for layer_idx, layer in enumerate(self.sender_layers):
            handles.append(layer.register_forward_pre_hook(capture(layer_idx)))
        try:
            with self.torch.inference_mode():
                output = self.sender(
                    input_ids=context_ids,
                    use_cache=True,
                    logits_to_keep=1,
                )
        finally:
            for handle in handles:
                handle.remove()
        if len(transition_states) != self.num_layers:
            raise RuntimeError("did not capture every sender transition state")
        return SenderState(output.past_key_values, transition_states)

    def _recompute_context(
        self,
        context_ids: Any,
        sender_state: SenderState,
        interval: RecomputeInterval,
    ) -> Any:
        from transformers import DynamicCache

        if interval.end > self.num_layers:
            raise ValueError("recompute interval exceeds the model")
        if interval.layer_count == 0:
            return self._hybrid_cache(sender_state.kv_cache, None, interval)

        transition_state = (
            sender_state.transition_states[interval.start]
            if interval.start > 0
            else None
        )
        receiver_cache = DynamicCache(config=self.receiver.config)
        with self._receiver_interval(interval, transition_state):
            with self.torch.inference_mode():
                self.receiver(
                    input_ids=context_ids,
                    past_key_values=receiver_cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
        return self._hybrid_cache(sender_state.kv_cache, receiver_cache, interval)

    def _hybrid_cache(
        self,
        sender_cache: Any,
        receiver_cache: Any | None,
        interval: RecomputeInterval,
    ) -> Any:
        from transformers import DynamicCache

        layers = []
        for layer_idx in range(self.num_layers):
            source = (
                receiver_cache
                if interval.contains(layer_idx) and receiver_cache is not None
                else sender_cache
            )
            key, value = source[layer_idx]
            layers.append((key.clone(), value.clone()))
        return DynamicCache(layers, config=self.receiver.config)

    def _generate_from_cache(
        self,
        model: Any,
        query_ids: Any,
        cache: Any,
        max_new_tokens: int,
        start_time: float,
        prefill_s: float,
        context_tokens: int,
    ) -> GenerationResult:
        if query_ids.shape[1] == 0:
            raise ValueError("query suffix must contain at least one token")
        token_ids: list[int] = []
        self._synchronize()
        with self.torch.inference_mode():
            output = model(
                input_ids=query_ids,
                past_key_values=cache,
                use_cache=True,
            )
            next_token = int(output.logits[0, -1].argmax().item())
            token_ids.append(next_token)
            self._synchronize()
            ttft_s = time.perf_counter() - start_time
            decode_start = time.perf_counter()
            for _ in range(max_new_tokens - 1):
                if token_ids[-1] in self.eos_ids:
                    break
                next_ids = self.torch.tensor(
                    [[token_ids[-1]]], dtype=self.torch.long, device=self.device
                )
                output = model(
                    input_ids=next_ids,
                    past_key_values=cache,
                    use_cache=True,
                )
                token_ids.append(int(output.logits[0, -1].argmax().item()))
        self._synchronize()
        decode_s = time.perf_counter() - decode_start
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        return GenerationResult(
            text=text,
            token_ids=token_ids,
            ttft_s=ttft_s,
            prefill_s=prefill_s,
            decode_s=decode_s,
            context_tokens=context_tokens,
            query_tokens=int(query_ids.shape[1]),
        )

    def run_interval(
        self,
        prompt: TokenizedPrompt,
        sender_state: SenderState,
        interval: RecomputeInterval,
        max_new_tokens: int,
    ) -> GenerationResult:
        """Generate with receiver KVs inside one interval and sender KVs outside."""

        self._synchronize()
        start = time.perf_counter()
        cache = self._recompute_context(prompt.context_ids, sender_state, interval)
        self._synchronize()
        prefill_s = time.perf_counter() - start
        return self._generate_from_cache(
            self.receiver,
            prompt.query_ids,
            cache,
            max_new_tokens,
            start,
            prefill_s,
            prompt.context_tokens_used,
        )

    def run_native(
        self,
        model: Any,
        prompt: TokenizedPrompt,
        max_new_tokens: int,
    ) -> GenerationResult:
        """Run an ordinary full context prefill for a control model."""

        self._synchronize()
        start = time.perf_counter()
        with self.torch.inference_mode():
            output = model(
                input_ids=prompt.context_ids,
                use_cache=True,
                logits_to_keep=1,
            )
        self._synchronize()
        prefill_s = time.perf_counter() - start
        return self._generate_from_cache(
            model,
            prompt.query_ids,
            output.past_key_values,
            max_new_tokens,
            start,
            prefill_s,
            prompt.context_tokens_used,
        )

    def audit_full_recompute(
        self,
        prompt: TokenizedPrompt,
        sender_state: SenderState,
    ) -> dict[str, float]:
        """Check that the full interval reconstructs receiver-native context KV."""

        full = RecomputeInterval(0, self.num_layers)
        hybrid = self._recompute_context(prompt.context_ids, sender_state, full)
        with self.torch.inference_mode():
            native = self.receiver(
                input_ids=prompt.context_ids,
                use_cache=True,
                logits_to_keep=1,
            )
        max_key_error = 0.0
        max_value_error = 0.0
        for layer_idx in range(self.num_layers):
            hybrid_key, hybrid_value = hybrid[layer_idx]
            native_key, native_value = native.past_key_values[layer_idx]
            max_key_error = max(
                max_key_error,
                float((hybrid_key - native_key).abs().max().item()),
            )
            max_value_error = max(
                max_value_error,
                float((hybrid_value - native_value).abs().max().item()),
            )
        return {
            "max_key_abs_error": max_key_error,
            "max_value_abs_error": max_value_error,
        }

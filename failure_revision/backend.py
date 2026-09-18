"""Generation backends with per-call deadlines.

``VLLMBackend`` wraps vLLM's asynchronous engine.  Every call is one
request; a caller-supplied deadline (``time.monotonic()`` seconds)
aborts the request mid-decode, and the partial token count is still
reported so it can be charged.  The same code serves batch-one timed
runs (the caller awaits one call at a time) and batched runs (the
caller awaits many calls concurrently and vLLM batches them).

``FakeBackend`` is a CPU stand-in with the same interface for tests.
"""

from __future__ import annotations

import asyncio
import time
import uuid


class VLLMBackend:
    def __init__(
        self,
        model: str,
        revision: str | None = None,
        max_model_len: int = 12800,
        gpu_memory_utilization: float = 0.85,
        enable_prefix_caching: bool = True,
        seed: int = 0,
    ):
        from transformers import AutoTokenizer
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        self.tok = AutoTokenizer.from_pretrained(model, revision=revision)
        args = AsyncEngineArgs(
            model=model,
            revision=revision,
            tokenizer_revision=revision,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
            dtype="bfloat16",
            seed=seed,
        )
        self.engine = AsyncLLM.from_engine_args(args)
        self.info = {
            "model": model,
            "revision": revision,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enable_prefix_caching": enable_prefix_caching,
        }

    def n_tokens(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))

    async def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        deadline: float | None = None,
    ) -> dict:
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        ids = self.tok.encode(prompt, add_special_tokens=False)
        sp = SamplingParams(
            n=1,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            output_kind=RequestOutputKind.DELTA,
        )
        rid = uuid.uuid4().hex
        st = {"text": [], "gen": 0, "cached": 0, "finish": None}

        async def consume():
            async for out in self.engine.generate(
                {"prompt_token_ids": ids}, sp, request_id=rid
            ):
                o = out.outputs[0]
                st["text"].append(o.text)
                st["gen"] += len(o.token_ids)
                if out.num_cached_tokens:
                    st["cached"] = out.num_cached_tokens
                if out.finished:
                    st["finish"] = o.finish_reason

        t0 = time.monotonic()
        if deadline is not None and deadline - t0 <= 0:
            return _result("", 0, len(ids), 0, "deadline", t0, t0)
        try:
            if deadline is None:
                await consume()
            else:
                await asyncio.wait_for(consume(), timeout=deadline - t0)
        except asyncio.TimeoutError:
            st["finish"] = "deadline"
            try:
                await self.engine.abort(rid)
            except Exception:
                pass
        t1 = time.monotonic()
        return _result(
            "".join(st["text"]), st["gen"], len(ids), st["cached"], st["finish"], t0, t1
        )

    def shutdown(self):
        try:
            self.engine.shutdown()
        except Exception:
            pass


def _result(text, gen, prompt_tokens, cached, finish, t0, t1) -> dict:
    return {
        "text": text,
        "gen_tokens": gen,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached,
        "finish": finish,
        "t_start": t0,
        "t_end": t1,
    }


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class FakeBackend:
    """Deterministic CPU backend.  ``responder(prompt, seed) -> text``."""

    def __init__(self, responder, sec_per_token: float = 0.0):
        self.tok = CharTokenizer()
        self.responder = responder
        self.sec_per_token = sec_per_token
        self.info = {"model": "fake"}
        self.calls: list[dict] = []

    def n_tokens(self, text: str) -> int:
        return len(text)

    async def generate(
        self, prompt, max_tokens, temperature, top_p, seed, deadline=None
    ):
        t0 = time.monotonic()
        self.calls.append({"prompt": prompt, "seed": seed, "temperature": temperature})
        text = self.responder(prompt, seed)
        finish = "stop"
        if len(text) > max_tokens:
            text, finish = text[:max_tokens], "length"
        need = len(text) * self.sec_per_token
        if deadline is not None and t0 + need > deadline:
            avail = max(0.0, deadline - t0)
            await asyncio.sleep(avail)
            n = int(avail / self.sec_per_token) if self.sec_per_token else 0
            return _result(
                text[:n], n, len(prompt), 0, "deadline", t0, time.monotonic()
            )
        if need:
            await asyncio.sleep(need)
        return _result(text, len(text), len(prompt), 0, finish, t0, time.monotonic())

    def shutdown(self):
        pass

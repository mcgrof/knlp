# SPDX-License-Identifier: MIT
"""Control offline request admission without changing model inputs or sampling."""

import os
import weakref


def configure_scheduling(mode):
    """Select the process model before importing vLLM."""
    if mode == "inprocess":
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    elif mode == "batch":
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
    elif mode != "runtime":
        raise ValueError(f"unknown offline scheduling mode: {mode}")


class BatchAdmission:
    """Queue a complete batch before the owned offline engine can execute it.

    This needs vLLM's level-zero scheduling pause, enqueue and completion
    APIs. It preserves multiprocessing, cache state, weights and sampling.
    It does not make different numerical batch shapes equivalent. Any API
    failure poisons this wrapper: discard the engine rather than resuming
    an incompletely submitted batch.
    """

    def __init__(self, engine):
        required = ("sleep", "wake_up", "enqueue", "wait_for_completion")
        if not all(callable(getattr(engine, name, None)) for name in required):
            raise ValueError(
                "batch scheduling requires vLLM level-0 pause/enqueue APIs; "
                "use inprocess scheduling for an older runtime"
            )
        # The engine owns this bound generate method. Do not keep it alive
        # through a reference cycle when the decoder is freed for scoring.
        self.engine = weakref.proxy(engine)
        self.failed = False

    def generate(
        self,
        prompts,
        sampling_params=None,
        *,
        use_tqdm=True,
        lora_request=None,
        priority=None,
        tokenization_kwargs=None,
    ):
        if self.failed:
            raise RuntimeError("failed batch admission: discard this offline engine")
        if self.engine.llm_engine.has_unfinished_requests():
            raise RuntimeError(
                "batch admission requires an idle, exclusive offline engine"
            )
        if not prompts:
            return []
        try:
            self.engine.sleep(level=0, mode="keep")
            self.engine.enqueue(
                prompts,
                sampling_params=sampling_params,
                lora_request=lora_request,
                priority=priority,
                use_tqdm=False,
                tokenization_kwargs=tokenization_kwargs,
            )
            self.engine.wake_up(tags=["scheduling"])
            return self.engine.wait_for_completion(use_tqdm=use_tqdm)
        except BaseException:
            self.failed = True
            raise


def runtime_settings(engine):
    """Record effective settings rather than guessing from environment defaults."""
    llm = engine.llm_engine
    config = getattr(llm, "vllm_config", llm)
    cache = getattr(config, "cache_config", None)
    scheduler = getattr(config, "scheduler_config", None)
    core = getattr(llm, "engine_core", None)
    return {
        "multiprocessing_env": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"),
        "engine_class": f"{type(llm).__module__}.{type(llm).__name__}",
        "core_client_class": (
            f"{type(core).__module__}.{type(core).__name__}"
            if core is not None
            else None
        ),
        "enable_prefix_caching": getattr(cache, "enable_prefix_caching", None),
        "async_scheduling": getattr(scheduler, "async_scheduling", None),
        "enable_chunked_prefill": getattr(scheduler, "enable_chunked_prefill", None),
        "max_num_seqs": getattr(scheduler, "max_num_seqs", None),
        "max_num_batched_tokens": getattr(scheduler, "max_num_batched_tokens", None),
    }

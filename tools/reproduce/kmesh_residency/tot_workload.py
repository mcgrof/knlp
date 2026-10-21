# SPDX-License-Identifier: MIT
"""Capture the official Tree of Thoughts search using a bounded local provider."""

import argparse
import ast
from collections import Counter
from contextlib import redirect_stdout
from fractions import Fraction
from functools import wraps
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

from .cpu import authenticate, VLLM_COMMIT
from .workload_capture import InvocationCapture

TOT_COMMIT = "8050e67d0e3a0fddc424d7fa5801538722a4c4cc"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
TASK_INDICES = (0, 1, 2)
MAX_OUTPUT = 192
WORKLOAD_SECONDS = 75 * 60


def grade(numbers, output):
    """Check arithmetic through bounded syntax and exact rational operations."""
    expression = (
        output.strip()
        .split("\n")[-1]
        .lower()
        .removeprefix("answer: ")
        .split("=", 1)[0]
        .strip()
    )
    if not expression or len(expression) > 256:
        return False
    try:
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 64:
            return False
        used = []

        def evaluate(node):
            if (
                isinstance(node, ast.Constant)
                and type(node.value) is int
                and 0 <= node.value <= 100
            ):
                used.append(node.value)
                return Fraction(node.value)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                return -evaluate(node.operand)
            if isinstance(node, ast.BinOp) and isinstance(
                node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
            ):
                left, right = evaluate(node.left), evaluate(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                return left / right
            raise ValueError("unsupported arithmetic syntax")

        answer = evaluate(tree.body)
        return answer == 24 and Counter(used) == Counter(numbers)
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError):
        return False


def source_identity(source):
    source = Path(source).resolve()
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != TOT_COMMIT or subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"]
    ):
        raise ValueError("Tree of Thoughts source must be clean at the pinned revision")
    return source


def prompt_tokens(tokenizer, prompt):
    """Render one chat prompt as a flat sequence of integer token IDs."""
    tokens = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(type(token) is not int or token < 0 for token in tokens)
    ):
        raise ValueError("chat template must return a flat list of integer token IDs")
    if len(tokens) + MAX_OUTPUT > 2048:
        raise ValueError("upstream prompt exceeds the bounded model context")
    return tokens


class LocalProvider:
    """Replace the upstream remote model provider and retain exact engine IDs."""

    def __init__(self, llm, capture):
        self.tokenizer = llm.get_tokenizer()
        self.capture = capture
        self.family = None
        self.parent = None
        self.purpose = None
        self.calls = 0
        self.last_request = None
        self.started_ns = time.perf_counter_ns()

    def __call__(
        self, prompt, model=None, temperature=0.7, max_tokens=1000, n=1, stop=None
    ):
        if n != 1 or stop is not None or self.family is None or self.purpose is None:
            raise ValueError("unexpected upstream invocation for the frozen profile")
        if self.calls >= 384:
            raise RuntimeError("predeclared model-call limit of 384 exhausted")
        if time.perf_counter_ns() - self.started_ns > WORKLOAD_SECONDS * 1_000_000_000:
            raise TimeoutError(
                f"workload exceeded its {WORKLOAD_SECONDS}-second wall-clock limit"
            )
        from vllm import SamplingParams

        tokens = prompt_tokens(self.tokenizer, prompt)
        name = f"{self.family}-call-{self.calls:04}"
        params = SamplingParams(
            temperature=temperature,
            max_tokens=min(max_tokens, MAX_OUTPUT),
            seed=123 + self.calls,
        )
        self.calls += 1
        text, _ = self.capture.invoke(
            name, self.family, self.parent, tokens, params, purpose=self.purpose
        )
        self.last_request = name
        return [text]


def attach_search(source, provider):
    """Load the original search while substituting only its model-call boundary."""
    sys.path.insert(0, str(source / "src"))
    boundary = types.ModuleType("tot.models")
    boundary.gpt = provider
    if "tot.models" in sys.modules:
        raise RuntimeError("remote model provider was imported before capture")
    sys.modules["tot.models"] = boundary
    bfs = importlib.import_module("tot.methods.bfs")
    task_type = importlib.import_module("tot.tasks.game24").Game24Task
    if Path(bfs.__file__).resolve() != source / "src/tot/methods/bfs.py":
        raise ValueError("search did not load from authenticated source")
    original_proposals, original_value = bfs.get_proposals, bfs.get_value
    producers, application = {}, []

    @wraps(original_proposals)
    def proposals(task, x, y):
        provider.purpose, provider.parent = "propose", producers.get(y)
        result = original_proposals(task, x, y)
        for candidate in result:
            producers[candidate] = provider.last_request
        return result

    @wraps(original_value)
    def value(task, x, y, n_evaluate_sample, cache_value=True):
        provider.purpose, provider.parent = "value", producers.get(y)
        before = provider.calls
        result = original_value(task, x, y, n_evaluate_sample, cache_value)
        if before == provider.calls:
            application.append(
                {
                    "event": "application-value-cache-hit",
                    "family_id": provider.family,
                    "state_sha256": hashlib.sha256(y.encode()).hexdigest(),
                }
            )
        return result

    bfs.get_proposals, bfs.get_value = proposals, value
    return bfs, task_type, producers, application


def model_identity(model):
    expected = json.loads(Path(__file__).with_name("tot-model-files.json").read_text())
    for name, digest in expected.items():
        with (model / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError("model snapshot differs: " + name)
    return expected


def run(args):
    source = source_identity(args.workload_source)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    authenticate(args.source)
    from vllm import LLM
    import torch
    import vllm
    import numpy
    import pandas
    import sympy
    import transformers

    model_files = model_identity(args.model)
    args.output.mkdir(parents=True)
    identity = {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "model_config_sha256": hashlib.sha256(
            (args.model / "config.json").read_bytes()
        ).hexdigest(),
        "vllm_commit": VLLM_COMMIT,
        "dtype": "bfloat16",
        "block_tokens": 16,
        "adapter": None,
    }
    manifest = {
        "schema": "knlp.kmesh.workload-capture.v1",
        "workload_source_commit": TOT_COMMIT,
        "model_files_sha256": model_files,
        "workload_files_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in (
                "src/tot/methods/bfs.py",
                "src/tot/tasks/game24.py",
                "src/tot/tasks/base.py",
                "src/tot/prompts/game24.py",
                "src/tot/data/24/24.csv",
            )
        },
        "identity": identity,
        "task_indices": list(TASK_INDICES),
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "vllm": vllm.__version__,
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "sympy": sympy.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "cpu_threads": torch.get_num_threads(),
        },
        "configuration": {
            "cache_bytes": 512 << 20,
            "max_model_len": 2048,
            "active_requests": 1,
            "generate": "propose",
            "evaluate": "value",
            "select": "greedy",
            "selected_states": 2,
            "evaluation_samples": 1,
            "max_output_tokens": MAX_OUTPUT,
            "temperature": 0.7,
            "seed_base": 123,
            "request_limit": 384,
            "workload_deadline_seconds": WORKLOAD_SECONDS,
        },
        "limits": [
            "Official search and prompts, adapted to a local Qwen model with smaller search/output budgets; not a reproduction of paper accuracy.",
            "This is observed benchmark demand in sequential closed-loop serving, not production traffic or an assumed arrival-rate distribution.",
            "Logical parentage does not imply token-prefix equality; rejected states are not cancelled engine requests.",
            "Instrumentation is active and capture writing can change observed gaps; no latency benchmark claim.",
        ],
    }
    (args.output / "workload.json").write_text(json.dumps(manifest, indent=2) + "\n")
    llm = LLM(
        model=str(args.model),
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=True,
        block_size=16,
        kv_cache_memory_bytes=512 << 20,
        max_model_len=2048,
        max_num_batched_tokens=2048,
        max_num_seqs=1,
        enable_chunked_prefill=False,
        async_scheduling=False,
        gpu_memory_utilization=0.9,
        seed=123,
    )
    capture = InvocationCapture(llm, args.output)
    results = []
    started = time.perf_counter_ns()
    try:
        if not llm.reset_prefix_cache():
            raise RuntimeError("initial empty-cache reset refused")
        with capture:
            provider = LocalProvider(llm, capture)
            bfs, task_type, producers, application = attach_search(source, provider)
            task = task_type()
            search_args = argparse.Namespace(
                backend="local-vllm",
                temperature=0.7,
                method_generate="propose",
                method_evaluate="value",
                method_select="greedy",
                n_generate_sample=1,
                n_evaluate_sample=1,
                n_select_sample=2,
                prompt_sample=None,
            )
            for index in TASK_INDICES:
                provider.family = f"game24-{index:04}"
                producers.clear()
                bfs.gpt = provider
                with redirect_stdout(io.StringIO()):
                    answers, info = bfs.solve(search_args, task, index, to_print=False)
                numbers = [int(n) for n in task.get_input(index).split()]
                grades = [grade(numbers, answer) for answer in answers]
                results.append(
                    {
                        "task_index": index,
                        "puzzle": numbers,
                        "answers": answers,
                        "correct": grades,
                        "search": info,
                    }
                )
                (args.output / "search.json").write_text(
                    json.dumps(results, indent=2, allow_nan=False) + "\n"
                )
                if time.perf_counter_ns() - started > WORKLOAD_SECONDS * 1_000_000_000:
                    raise TimeoutError(
                        f"workload exceeded its {WORKLOAD_SECONDS}-second wall-clock limit"
                    )
            manifest["application_events"] = application
        manifest["engine_cache"] = capture.manifest()
        if manifest["engine_cache"]["final_nonnull_references"] != 0:
            raise RuntimeError("request references did not drain")
        manifest["task_results"] = [
            {
                "task_index": r["task_index"],
                "solved": any(r["correct"]),
                "selected_answers": len(r["answers"]),
                "proposed_states": sum(
                    len(step["new_ys"]) for step in r["search"]["steps"]
                ),
                "selected_states": sum(
                    len(step["select_new_ys"]) for step in r["search"]["steps"]
                ),
            }
            for r in results
        ]
        manifest["successful_capture"] = True
    finally:
        manifest["engine_cache"] = capture.manifest()
        (args.output / "workload.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        llm.llm_engine.engine_core.shutdown(timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--workload-source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a fresh directory")
    run(args)


if __name__ == "__main__":
    main()

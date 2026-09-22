#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run baseline and SnapKV through LMCache's real vLLM decode path.

The SnapKV scoring and key rerotation follow LMCache's public
``examples/token_dropping/snapkv_token_dropping.ipynb`` example.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from datasets import load_dataset
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

import lmcache.sdk as lmc_sdk


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def launch(argv: list[str], log_path: Path, env: dict[str, str]) -> subprocess.Popen:
    log = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        argv,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )


def wait_ready(
    url: str, name: str, process: subprocess.Popen, log_path: Path, timeout: int
) -> None:
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if process.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"{name} exited with {process.returncode}:\n{tail}")
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    print(f"{name} ready in {time.monotonic() - started:.1f}s")
                    return
        except Exception:
            pass
        time.sleep(2)
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    raise TimeoutError(f"{name} was not ready after {timeout}s:\n{tail}")


def stop(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def normalize_q(q_tensor: torch.Tensor) -> torch.Tensor:
    if q_tensor.ndim == 4 and q_tensor.shape[0] == 1:
        return q_tensor[0]
    if q_tensor.ndim == 3:
        return q_tensor
    raise ValueError(f"expected a 3D Q tensor or leading singleton: {q_tensor.shape}")


def layer_scores(
    k_flat: torch.Tensor,
    q_flat: torch.Tensor,
    *,
    seq_len: int,
    window_size: int,
    kernel_size: int,
    pooling: str,
    num_kv_heads: int,
    num_q_heads: int,
    head_size: int,
    device: torch.device,
) -> torch.Tensor:
    groups = num_q_heads // num_kv_heads
    past_len = seq_len - window_size
    key = k_flat[:seq_len].to(device=device, dtype=torch.float32)
    query = q_flat[:seq_len].to(device=device, dtype=torch.float32)
    key = key.reshape(seq_len, num_kv_heads, head_size).permute(1, 0, 2)
    query = query.reshape(seq_len, num_q_heads, head_size)
    query = query.reshape(seq_len, num_kv_heads, groups, head_size)
    query = query[-window_size:].permute(1, 2, 0, 3)
    attention = torch.einsum("kgwd,ktd->kgwt", query, key) / math.sqrt(head_size)
    causal = torch.triu(
        torch.full(
            (window_size, window_size),
            torch.finfo(attention.dtype).min,
            dtype=attention.dtype,
            device=device,
        ),
        diagonal=1,
    )
    attention[..., -window_size:] += causal
    attention = F.softmax(attention, dim=-1, dtype=torch.float32)
    scores = attention[..., :past_len].sum(dim=-2).reshape(1, -1, past_len)
    pool = F.avg_pool1d if pooling == "avgpool" else F.max_pool1d
    scores = pool(scores, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
    return scores[..., :past_len].reshape(-1, past_len).mean(dim=0)


def make_dropper(config, rerotate_k_cache, window_size: int, drop_ratio: float):
    work_device = torch.device("cpu")

    def drop_tokens_fn(tensors, token_source):
        kind = lmc_sdk.LMCacheSDKCacheKind
        query = normalize_q(tensors[kind.QUERY])
        kv = tensors[kind.KV]
        seq_len = min(kv.shape[2], query.shape[1], len(token_source))
        num_q_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_q_heads)
        head_size = getattr(config, "head_dim", config.hidden_size // num_q_heads)
        if num_q_heads % num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if seq_len <= window_size:
            raise ValueError(f"sequence {seq_len} does not exceed window {window_size}")
        past_len = seq_len - window_size
        total_keep = int(round(seq_len * (1 - drop_ratio)))
        keep_past = max(0, min(total_keep - window_size, past_len))
        scores = torch.zeros(past_len, dtype=torch.float32, device=work_device)
        for layer in range(kv.shape[1]):
            scores += layer_scores(
                kv[0, layer],
                query[layer],
                seq_len=seq_len,
                window_size=window_size,
                kernel_size=5,
                pooling="avgpool",
                num_kv_heads=num_kv_heads,
                num_q_heads=num_q_heads,
                head_size=head_size,
                device=work_device,
            )
        scores /= kv.shape[1]
        past = torch.topk(scores, k=keep_past, largest=True).indices.sort().values
        recent = torch.arange(past_len, seq_len, device=work_device)
        keep = torch.cat([past, recent]).to(dtype=torch.long)
        kept_ids = [token_source[index] for index in keep.tolist()]
        compacted = rerotate_k_cache(
            kv[:, :, keep.cpu(), :].clone().to(work_device),
            old_positions=keep,
            new_positions=torch.arange(keep.numel(), device=work_device),
            model_config=config,
        )
        print(f"compacted {compacted.shape[2]}/{kv.shape[2]} tokens")
        return compacted.cpu(), kept_ids

    return drop_tokens_fn


def output_tput(result) -> float:
    return float(result.to_dict()["metrics"]["results"]["output_tput"])


def create_batch(ctx, qctx, post_completion, prompts):
    batch = lmc_sdk.batch.LMCacheBatchedStream()
    for prompt in prompts:
        request = lmc_sdk.request.create_request(
            contexts=[ctx, qctx],
            post_completion=post_completion,
            prompt_token_ids=prompt,
        )
        batch.add(request)
    return batch


def decode(batch, max_tokens: int):
    prefill = batch.prefill(
        sampling_params={"max_tokens": 1, "temperature": 1.0, "ignore_eos": True}
    )
    prefill.emit()
    result = batch.decode(
        sampling_params={
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
        }
    )
    result.emit()
    texts = [request.output_text for request in batch.request_streams.values()]
    return result, texts


def git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmcache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--drop-ratio", type=float, default=0.5)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--l1-size-gb", type=float, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    example_dir = args.lmcache_dir / "examples/token_dropping"
    sys.path.insert(0, str(example_dir))
    from utils import make_post_completion, rerotate_k_cache, score_answers

    ports = {"mq": free_port(), "lmcache": free_port(), "vllm": free_port()}
    log_dir = args.output.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    executable_dir = Path(sys.executable).parent
    shm_name = f"knlp_snapkv_{os.getpid()}"
    lmc = None
    vllm = None
    ctx = None
    qctx = None
    try:
        lmc = launch(
            [
                str(executable_dir / "lmcache"),
                "server",
                "--l1-size-gb",
                str(args.l1_size_gb),
                "--eviction-policy",
                "LRU",
                "--chunk-size",
                "256",
                "--port",
                str(ports["mq"]),
                "--http-port",
                str(ports["lmcache"]),
                "--shm-name",
                shm_name,
                "--no-l1-use-lazy",
                "--enable",
                "transfer_query",
                "--max-workers",
                "4",
                "--eviction-trigger-watermark",
                "0.98",
            ],
            log_dir / "lmcache.log",
            env,
        )
        lmcache_url = f"http://127.0.0.1:{ports['lmcache']}"
        wait_ready(
            f"{lmcache_url}/healthcheck", "LMCache", lmc, log_dir / "lmcache.log", 180
        )
        connector = json.dumps(
            {
                "kv_connector": "LMCacheMPConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {
                    "lmcache.mp.port": ports["mq"],
                    "lmcache.mp.transfer_intermediate_tensors": True,
                    "lmcache.mp.q.ring_depth": 2,
                    "lmcache.mp.heartbeat_interval": 30,
                },
            }
        )
        vllm = launch(
            [
                str(executable_dir / "vllm"),
                "serve",
                args.model,
                "--port",
                str(ports["vllm"]),
                "--served-model-name",
                args.model,
                "--no-enable-prefix-caching",
                "--enforce-eager",
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
                "--kv-transfer-config",
                connector,
                "--trust-remote-code",
                "--return-tokens-as-token-ids",
                "--max-model-len",
                "2048",
                "--max-num-batched-tokens",
                "2048",
            ],
            log_dir / "vllm.log",
            env,
        )
        vllm_url = f"http://127.0.0.1:{ports['vllm']}"
        wait_ready(f"{vllm_url}/v1/models", "vLLM", vllm, log_dir / "vllm.log", 1800)

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        dataset = load_dataset(args.dataset, split="train")
        rows = list(dataset)[: args.num_prompts]
        prompts = [tokenizer.encode(row["prompt"]) for row in rows]
        answers = [row["answer"] for row in rows]
        post_completion = make_post_completion(vllm_url, args.model, 60)
        mq_url = f"tcp://127.0.0.1:{ports['mq']}"
        ctx = lmc_sdk.kvcache.connect(mq_url, lmcache_url, args.model, timeout=60)
        qctx = lmc_sdk.qcache.connect(mq_url, lmcache_url, args.model, timeout=60)

        baseline_batch = create_batch(ctx, qctx, post_completion, prompts)
        baseline_result, baseline_texts = decode(baseline_batch, args.max_tokens)
        baseline_correct, total, _ = score_answers(answers, baseline_texts)

        request = urllib.request.Request(f"{lmcache_url}/cache/clear", method="POST")
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()

        snap_batch = create_batch(ctx, qctx, post_completion, prompts)
        prefill = snap_batch.prefill(
            sampling_params={"max_tokens": 1, "temperature": 1.0, "ignore_eos": True}
        )
        prefill.emit()
        modified = snap_batch.modify(
            make_dropper(
                model_config, rerotate_k_cache, args.window_size, args.drop_ratio
            )
        )
        modified.emit()
        snap_result = snap_batch.decode(
            sampling_params={
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
                "ignore_eos": True,
            }
        )
        snap_result.emit()
        snap_texts = [
            request.output_text for request in snap_batch.request_streams.values()
        ]
        snap_correct, _, _ = score_answers(answers, snap_texts)

        report = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "dataset": args.dataset,
            "num_prompts": len(prompts),
            "prompt_tokens": [len(prompt) for prompt in prompts],
            "drop_ratio": args.drop_ratio,
            "window_size": args.window_size,
            "lmcache_revision": git_revision(args.lmcache_dir),
            "vllm_version": version("vllm"),
            "lmcache_version": version("lmcache"),
            "torch_version": torch.__version__,
            "model_revision": getattr(model_config, "_commit_hash", None),
            "baseline": {
                "correct": baseline_correct,
                "total": total,
                "output_tput": output_tput(baseline_result),
            },
            "snapkv": {
                "correct": snap_correct,
                "total": total,
                "output_tput": output_tput(snap_result),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    finally:
        for context in (qctx, ctx):
            if context is None:
                continue
            try:
                context.close()
            except Exception as error:
                print(f"warning: failed to close SDK context: {error}", file=sys.stderr)
        stop(vllm)
        stop(lmc)


if __name__ == "__main__":
    main()

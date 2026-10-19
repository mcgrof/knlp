# SPDX-License-Identifier: MIT
"""Run the established in-process entry point for HTTP parity/quality checks."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .current import save
from .serving_server import prepare


def reference(inputs, output, variant, cache_bytes, length, sequential=False):
    from vllm import SamplingParams
    from .october import build_engine

    output.mkdir()
    arm = "k16_v8" if variant == "mixed_prefill" else "k16_v16"
    engine, manifest, _ = build_engine(
        inputs, output, variant, arm, cache_bytes, "profile"
    )
    samples = [
        s
        for s in manifest["samples"]
        if s["kind"] == "nll" and s["length"] == length and not s.get("reserve")
    ]
    if len(samples) != 32:
        raise ValueError("reference requires the frozen panel at the selected length")
    prompts = [{"prompt_token_ids": s["token_ids"][:length]} for s in samples]
    save(output / "provenance.json", engine._study_provenance)
    for order, selected in enumerate((prompts, prompts[1:] + prompts[:1])):
        params = SamplingParams(
            temperature=0, seed=20261019, max_tokens=128, ignore_eos=True
        )
        values = []
        for group in ([p] for p in selected) if sequential else [selected]:
            values.extend(engine.generate(group, params, use_tqdm=False))
        save(
            output / f"generation-{order}.json",
            [
                {
                    "index": index,
                    "prompt_token_ids": item.prompt_token_ids,
                    "output_token_ids": list(item.outputs[0].token_ids),
                    "finish_reason": item.outputs[0].finish_reason,
                }
                for index, item in enumerate(values)
            ],
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("reference", "nll", "retrieval", "gsm8k", "child"))
    p.add_argument("--install", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--format", choices=("awq", "fp16"), required=True)
    p.add_argument(
        "--arm", choices=("candidate", "native_fi", "native_fa3"), required=True
    )
    p.add_argument("--length", type=int, choices=(1024, 8192), default=8192)
    p.add_argument("--cache-bytes", type=int)
    p.add_argument("--sequential-reference", action="store_true")
    args = p.parse_args()
    variant = "mixed_prefill" if args.arm == "candidate" else "native"
    inputs = args.output.resolve() / "inputs"
    if args.stage == "child":
        reference(
            inputs,
            args.output / "generation",
            variant,
            args.cache_bytes,
            args.length,
            args.sequential_reference,
        )
        return
    _, env = prepare(
        args.install.resolve(),
        args.output.resolve(),
        args.format,
        args.arm,
        args.cache_bytes,
    )
    protocol = json.loads((inputs / "protocol.json").read_text())
    protocol["engine"]["worker_cls"] = "tools.kv.native.memory_worker.ObservedWorker"
    save(inputs / "protocol.json", protocol)
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    budget = str(protocol["engine"]["kv_cache_memory_bytes"])
    geometry = args.output / "geometry"
    common = ["--inputs", str(inputs), "--variant", variant, "--cache-bytes", budget]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.kv.native.october",
            "geometry",
            *common,
            "--output",
            str(geometry),
        ],
        env=env,
        check=True,
    )
    shutil.copy2(geometry / "geometry.json", inputs / "geometry.json")
    if args.stage == "reference":
        argv = [
            sys.executable,
            "-m",
            "tools.kv.native.serving_reference",
            "child",
            "--install",
            str(args.install),
            "--output",
            str(args.output),
            "--format",
            args.format,
            "--arm",
            args.arm,
            "--length",
            str(args.length),
            "--cache-bytes",
            budget,
        ]
        if args.sequential_reference:
            argv.append("--sequential-reference")
    else:
        argv = [
            sys.executable,
            "-m",
            "tools.kv.native.october",
            args.stage,
            *common,
            "--output",
            str(args.output / args.stage),
            "--length",
            str(args.length),
            "--batch",
            "32",
        ]
    save(
        args.output / "command.json",
        {"argv": argv, "purpose": "untimed parity/quality"},
    )
    subprocess.run(argv, env=env, check=True)


if __name__ == "__main__":
    main()

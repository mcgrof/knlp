#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Replay the frozen serial model workload using registered native span inputs."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT.parent / "tokendance"
LMCACHE_CONFIG = BASELINE / "lmcache-current.yaml"
sys.path[:0] = [str(ROOT), str(BASELINE), str(BASELINE / "harness")]


def service(channel, vllm_source: Path, wheel: Path, lmcache_source: Path):
    """Own the importing process; its actual exit is the final release proof."""
    import collective_gate_common as common

    sys.path.insert(0, str(lmcache_source))
    common.serial._load_vllm_overlay(vllm_source, wheel)
    from lmcache.v1.multiprocess.kmesh_raw_server import create_managed_raw_server
    from lmcache.v1.multiprocess.transport.grpc_impl.server import (
        GrpcMultiprocessServer,
    )
    import zmq

    server = engine = None
    try:
        server, engine = create_managed_raw_server(
            port=0, chunk_tokens=256, dram_bytes=64 << 20, enable_span_execution=True
        )
        server.start()
        address = (
            f"grpc://127.0.0.1:{server.bound_port}"
            if isinstance(server, GrpcMultiprocessServer)
            else server.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        )
        channel.send(("ready", address))
        if channel.recv() != "stop":
            raise RuntimeError("unexpected service control message")
        server.close()
        server = None
        engine.close()
        engine = None
        channel.send(("closed", None))
    except BaseException:
        channel.send(("error", traceback.format_exc()))
        raise
    finally:
        if server is not None:
            server.close()
        if engine is not None:
            engine.close()
        channel.close()


def receive(channel, timeout=90):
    if not channel.poll(timeout):
        raise RuntimeError("owned cache process did not reply")
    kind, value = channel.recv()
    if kind == "error":
        raise RuntimeError(value)
    return kind, value


def stop_importer(process):
    """Settle actual owned-process exit before engine allocations can be freed."""
    if process.is_alive():
        process.terminate()
    process.join(30)
    if process.is_alive():
        process.kill()
        process.join(30)
    if process.is_alive():
        # Do not let Python finalizers release exporter allocations while the
        # child still imports them. The enclosing experiment supervises both.
        os._exit(70)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "vllm-source",
        "lmcache-source",
        "wheel-vllm-package",
        "model-dir",
        "trace",
        "manifest",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=ROOT / "profile.json")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument(
        "--route", choices=("reference", "registered"), default="registered"
    )
    return parser.parse_args()


def main():
    from bridge import RegisteredReplay, identity
    import selection

    args = parse_args()
    pins = json.loads(args.profile.read_text())
    if args.plan:
        print(json.dumps(pins, indent=2))
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("output directory must be empty")
    for source_path in (args.lmcache_source, args.vllm_source, ROOT.parents[2]):
        if (
            args.output_dir.resolve() == source_path.resolve()
            or source_path.resolve() in args.output_dir.resolve().parents
        ):
            raise ValueError("capture output must be outside source checkouts")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TOKENDANCE_PROFILE"] = "rtx4090-cuda"
    os.environ["TOKENDANCE_MODEL_DIR"] = str(args.model_dir)
    unexpected = [
        name
        for name in os.environ
        if name.startswith("LMCACHE_") and name != "LMCACHE_CONFIG_FILE"
    ]
    if unexpected:
        raise ValueError(
            "unreviewed LMCache environment overrides: " + ", ".join(sorted(unexpected))
        )
    os.environ["LMCACHE_CONFIG_FILE"] = str(LMCACHE_CONFIG)
    for name, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "PYTHONHASHSEED": "0",
    }.items():
        if os.environ.get(name, value) != value:
            raise ValueError(f"unsupported execution setting: {name}")
        os.environ[name] = value
    import collective_gate_common as common
    import contract
    import numerical_instrumentation as numerical
    import run_online_collective_lockstep_gate as baseline

    base = common.serial
    contract.validate_assets()
    contract.validate_model(args.model_dir)
    source = {
        name: base._source_identity(path, pins[name]["commit"], pins[name]["tree"])
        for name, path in (("lmcache", args.lmcache_source), ("vllm", args.vllm_source))
    }
    _, records = base._load_trace(args.trace, args.manifest)
    sys.path.insert(0, str(args.lmcache_source))
    vllm = base._load_vllm_overlay(args.vllm_source, args.wheel_vllm_package)
    import lmcache
    import torch
    import transformers
    from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
    from vllm import LLM

    extensions = {
        name: Path(importlib.import_module(f"vllm.{name}").__file__)
        for name in base.EXPECTED_EXTENSION_HASHES
    }
    hashes = {name: base._sha256(path) for name, path in extensions.items()}
    if hashes != base.EXPECTED_EXTENSION_HASHES:
        raise ValueError("compiled vLLM extensions differ")
    runtime = base._runtime_identity(
        vllm, lmcache, torch, transformers, extensions, hashes
    )
    for name, digest in pins["lmcache_native_sha256"].items():
        if (
            base._sha256(Path(importlib.import_module("lmcache." + name).__file__))
            != digest
        ):
            raise ValueError(f"compiled LMCache extension differs: {name}")

    # Extend only source admission for this separately pinned descendant. Keep
    # the frozen capture geometry, raw tensor policy and warmup checks intact.
    class QualifiedCapture(numerical.NumericalInstrumentation):
        @staticmethod
        def _validate_source(value):
            if value != source:
                raise ValueError("capture source differs from qualified descendant")
            return copy.deepcopy(value)

    ledger = base.EventLedger(args.output_dir / "events.jsonl")
    base._atomic_json(args.output_dir / "status.json", {"status": "running"})
    llm = bridge = client = process = None
    parent = None
    try:
        baseline._install_instrumentation(ledger, inject_sibling_failure=False)
        capture = QualifiedCapture(
            ledger=ledger,
            arm="serial_oracle",
            output_dir=args.output_dir,
            records=tuple(copy.deepcopy(records)),
            source=source,
        )
        selection.install(ledger)
        if args.route == "registered":
            parent, child = mp.get_context("spawn").Pipe()
            process = mp.get_context("spawn").Process(
                target=service,
                args=(
                    child,
                    args.vllm_source,
                    args.wheel_vllm_package,
                    args.lmcache_source,
                ),
            )
            process.start()
            child.close()
            kind, address = receive(parent)
            if kind != "ready":
                raise RuntimeError("cache process did not initialize")
            client = RequestClientFactory.create(address)
            model_files = contract.profile()["model_files"]
            bridge = RegisteredReplay(
                client,
                {
                    "weights": identity(
                        {
                            k: v
                            for k, v in model_files.items()
                            if k.endswith(".safetensors")
                        }
                    ),
                    "tokenizer": identity(
                        {
                            k: v
                            for k, v in model_files.items()
                            if k not in {"config.json", "generation_config.json"}
                            and not k.endswith(".safetensors")
                        }
                    ),
                    "execution": identity(pins),
                },
                ledger,
                lambda: stop_importer(process),
            )
            bridge.install()
        ledger.set_phase("construction")
        kwargs = common.engine_kwargs(connectors=True, profile=common.SERIAL_PROFILE)
        llm = LLM(**kwargs)
        from vllm.v1.kmesh.config import validate_kmesh_target_profile

        if validate_kmesh_target_profile(llm.llm_engine.vllm_config) != "serial_pic_v1":
            raise ValueError("runtime is not the bounded serial profile")
        if llm.llm_engine.vllm_config.attention_config.backend.name != "TRITON_ATTN":
            raise ValueError("runtime attention backend changed")
        observations = baseline._execute_arm("serial_oracle", llm, records, ledger)
        events = ledger.snapshot()
        recovery = baseline._recovery_evidence(observations, events)
        numeric_result = capture.finish(
            arm="serial_oracle",
            observations=observations,
            recovery_evidence=recovery,
            source=source,
            runtime=runtime,
        )
        if args.route == "registered":
            stored = [e for e in events if e["action"] == "registered.span.stored"]
            restored = [e for e in events if e["action"] == "registered.span.restored"]
            consumed = [e for e in events if e["action"] == "registered.model.consumed"]
            if len(stored) != 4 or len(restored) != 16 or len(consumed) != 4:
                raise ValueError("registered operation/model coverage is incomplete")
            if any(
                e["span_count"] != 4
                or e["restored_tokens"] != 416
                or not e["source_closed"]
                or e["pending"]
                for e in consumed
            ):
                raise ValueError(
                    "model did not consume and settle all registered inputs"
                )
            if any(e["source_start"] == e["destination_start"] for e in restored):
                raise ValueError("model replay did not exercise shifted positions")
            bridge.close()
            parent.send("stop")
            if receive(parent)[0] != "closed":
                raise RuntimeError("cache process did not close")
            process.join(30)
            if process.exitcode != 0:
                raise RuntimeError("cache process did not exit successfully")
            client.close()
            client = None
        base._cleanup_llm(llm, ledger)
        llm = None
        for name, path in (
            ("lmcache", args.lmcache_source),
            ("vllm", args.vllm_source),
        ):
            if (
                base._source_identity(path, pins[name]["commit"], pins[name]["tree"])
                != source[name]
            ):
                raise ValueError("source identity changed during model execution")
        contract.validate_assets()
        base._atomic_json(
            args.output_dir / "result.json",
            {
                "status": "complete",
                "kind": (
                    "registered_native_serial_model"
                    if args.route == "registered"
                    else "serial_model_reference"
                ),
                "route": args.route,
                "source": source,
                "runtime": runtime,
                "profile": pins,
                "lmcache_config_sha256": base._sha256(LMCACHE_CONFIG),
                "profile_sha256": base._sha256(args.profile),
                "runner_sha256": base._sha256(Path(__file__)),
                "bridge_sha256": base._sha256(ROOT / "bridge.py"),
                "observations": observations,
                "recovery_evidence": recovery,
                "numerical_capture": numeric_result,
                "selection_evidence": [
                    e for e in events if e["action"] == "registered.model.selection"
                ],
                "event_ledger_sha256": base._sha256(ledger.path),
                "helper_sha256": {p.name: base._sha256(p) for p in ROOT.glob("*.py")},
                "claim_boundary": "CPU-L1 serial model correctness; duplicated reference storage; no serving latency or DMA-BUF claim.",
            },
        )
        base._atomic_json(args.output_dir / "status.json", {"status": "complete"})
    except BaseException:
        base._atomic_json(
            args.output_dir / "failure.json", {"traceback": traceback.format_exc()}
        )
        base._atomic_json(args.output_dir / "status.json", {"status": "failed"})
        raise
    finally:
        # Resolve importer lifetime before any worker engine/exporter teardown,
        # including bootstrap and transport uncertainty. The owned process is
        # supervised inside this experiment; no unrelated service is touched.
        if process is not None:
            stop_importer(process)
        if client is not None:
            client.close()
        if llm is not None:
            base._cleanup_llm(llm, ledger)
        if parent is not None:
            parent.close()
        ledger.close()


if __name__ == "__main__":
    main()

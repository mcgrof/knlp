#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Replay the frozen serial model workload using registered native span inputs."""

from __future__ import annotations

import argparse
import copy
import hashlib
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


def service(
    channel,
    vllm_source: Path,
    wheel: Path,
    lmcache_source: Path,
    control="none",
    storage_route="cpu-l1",
    storage_file=None,
    storage_settings=None,
    poison_arena=False,
    checksum_library=None,
):
    """Own the importing process; its actual exit is the final release proof."""
    import collective_gate_common as common

    sys.path.insert(0, str(lmcache_source))
    common.serial._load_vllm_overlay(vllm_source, wheel)
    from lmcache.v1.multiprocess.kmesh_raw_server import create_managed_raw_server
    from lmcache.v1.multiprocess.transport.grpc_impl.server import (
        GrpcMultiprocessServer,
    )
    import zmq

    import faults
    import storage

    fault = faults.install_service(control)
    server = engine = observer = None
    try:
        storage_config = (
            None
            if storage_route == "cpu-l1"
            else storage.create_config(storage_route, storage_file, storage_settings)
        )
        service_args = dict(
            port=0, chunk_tokens=256, dram_bytes=64 << 20, enable_span_execution=True
        )
        if storage_config is not None:
            service_args["storage_config"] = storage_config
        server, engine = create_managed_raw_server(**service_args)
        if storage_config is not None:
            observer = storage.StorageObserver(
                engine.storage_manager,
                storage_settings,
                poison_arena=poison_arena,
                checksum_library=checksum_library,
            )
            observer.install()
        server.start()
        address = (
            f"grpc://127.0.0.1:{server.bound_port}"
            if isinstance(server, GrpcMultiprocessServer)
            else server.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        )
        channel.send(("ready", address))
        while True:
            command = channel.recv()
            if command == "stop":
                break
            if command == "arm":
                fault.armed = True
                channel.send(("armed", dict(fault.observed)))
            elif command == "clean":
                fault.armed = False
                channel.send(("cleaned", dict(fault.observed)))
            elif (
                observer is not None
                and isinstance(command, tuple)
                and len(command) == 2
            ):
                kind, request_id = command
                if type(request_id) is not str or not request_id:
                    raise ValueError(
                        "storage observation requires the actual request ID"
                    )
                if kind == "cold":
                    channel.send(("cold", observer.cold(request_id)))
                elif kind == "measure":
                    channel.send(("measured", observer.measure(request_id)))
                else:
                    raise RuntimeError("unexpected storage observation")
            else:
                raise RuntimeError("unexpected service control message")
        server.close()
        server = None
        storage_evidence = None if observer is None else observer.snapshot()
        engine.close()
        engine = None
        if observer is not None:
            outcome = observer.adapter.close_with_outcome()
            storage_evidence["close_buffers_reusable"] = outcome.buffers_reusable
            if not outcome.buffers_reusable:
                raise RuntimeError(
                    "raw-block arena shutdown remains physically unresolved"
                )
        channel.send(("closed", storage_evidence))
    except BaseException:
        channel.send(("error", traceback.format_exc()))
        raise
    finally:
        if server is not None:
            server.close()
        if engine is not None:
            engine.close()
        if observer is not None:
            observer.close()
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
    parser.add_argument(
        "--control",
        choices=(
            "none",
            "partial_miss",
            "corrupt",
            "evict",
            "post_enqueue",
            "transient_drain",
            "persistent_drain",
            "worker_drain",
        ),
        default="none",
    )
    parser.add_argument("--ordinary-result", type=Path)
    parser.add_argument("--ordinary-verification", type=Path)
    parser.add_argument("--serial-result", type=Path)
    parser.add_argument("--serial-comparison", type=Path)
    parser.add_argument("--group-failure", action="store_true")
    parser.add_argument(
        "--storage-route", choices=("cpu-l1", "staged", "dma-buf"), default="cpu-l1"
    )
    parser.add_argument("--storage-file", type=Path)
    parser.add_argument("--poison-arena", action="store_true")
    parser.add_argument("--checksum-library", type=Path)
    parser.add_argument(
        "--model-execution", choices=("serial", "collective-lockstep"), default="serial"
    )
    return parser.parse_args()


def engine_options(common, execution):
    """Change only the admitted scheduler mode for layer-major model repair."""
    options = common.engine_kwargs(connectors=True, profile=common.SERIAL_PROFILE)
    profile = common.SERIAL_PROFILE
    if execution == "collective-lockstep":
        profile = "collective_lockstep_v1"
        extra = options["kv_transfer_config"]["kv_connector_extra_config"]
        extra["kmesh_target_profile"] = profile
        extra["connectors"][0]["kv_connector_extra_config"]["kmesh"][
            "mode"
        ] = "collective_lockstep"
    elif execution != "serial":
        raise ValueError("unsupported registered model execution")
    return profile, options


def main():
    from bridge import RegisteredReplay, identity
    import selection
    import faults
    import controls

    args = parse_args()
    is_control = args.control != "none" or args.group_failure
    if is_control and (
        args.route != "registered"
        or args.ordinary_result is None
        or args.ordinary_verification is None
    ):
        raise ValueError(
            "model controls require registered route and a verified ordinary result"
        )
    if args.group_failure and (
        args.model_execution != "collective-lockstep"
        or args.control != "none"
        or args.serial_result is None
        or args.serial_comparison is None
    ):
        raise ValueError("group member control requires its verified serial reference")
    pins = json.loads(args.profile.read_text())
    if args.model_execution not in pins.get("model_executions", ["serial"]):
        raise ValueError("model execution differs from its pinned input profile")
    if args.model_execution != "serial" and (
        args.route != "registered"
        or args.storage_route != "cpu-l1"
        or args.control != "none"
    ):
        raise ValueError(
            "the first registered model group requires CPU-L1 and no source fault"
        )
    profile_hash = hashlib.sha256(args.profile.read_bytes()).hexdigest()
    helper_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.py")
    }
    if args.plan:
        print(json.dumps(pins, indent=2))
        return
    if args.storage_route != "cpu-l1" and (
        args.route != "registered"
        or args.control != "none"
        or pins["storage"]["kind"] != "raw_block_comparison"
    ):
        raise ValueError(
            "storage comparison requires its pinned registered model profile"
        )
    if args.storage_route == "cpu-l1" and args.storage_file is not None:
        raise ValueError("a backing file requires an explicit storage route")
    if args.storage_route == "cpu-l1" and args.poison_arena:
        raise ValueError("arena conditioning requires an explicit storage route")
    if args.checksum_library is not None and (
        args.route != "registered" or args.storage_route != "dma-buf"
    ):
        raise ValueError("GPU checksums require registered DMA-BUF restoration")
    gpu_checksum = None
    if args.checksum_library is not None:
        args.checksum_library = args.checksum_library.resolve(strict=True)
        gpu_checksum = {
            "library_sha256": hashlib.sha256(
                args.checksum_library.read_bytes()
            ).hexdigest(),
            "kernel_source_sha256": hashlib.sha256(
                (ROOT / "gpu_sha256.cu").read_bytes()
            ).hexdigest(),
        }
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("output directory must be empty")
    for source_path in (args.lmcache_source, args.vllm_source, ROOT.parents[2]):
        if (
            args.output_dir.resolve() == source_path.resolve()
            or source_path.resolve() in args.output_dir.resolve().parents
        ):
            raise ValueError("capture output must be outside source checkouts")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.storage_route != "cpu-l1":
        selected = (
            args.output_dir / "canonical-kv.raw"
            if args.storage_file is None
            else args.storage_file
        )
        if selected.exists() or selected.is_symlink():
            raise ValueError("storage comparison requires a fresh caller-selected file")
        args.storage_file = selected.resolve()
        for source_path in (args.lmcache_source, args.vllm_source, ROOT.parents[2]):
            if args.storage_file.is_relative_to(source_path.resolve()):
                raise ValueError("storage files must remain outside source checkouts")
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
    ordinary_outputs = {}
    if is_control:
        verification = json.loads(args.ordinary_verification.read_text())
        if verification.get("claims", {}).get(
            "full_collective_lockstep_gate_pass"
        ) is not True or verification["arm_results"]["ordinary"][
            "sha256"
        ] != base._sha256(
            args.ordinary_result
        ):
            raise ValueError("ordinary result differs from independent verification")
        ordinary_outputs = base._baseline_outputs(args.ordinary_result)
    if args.group_failure:
        checked = json.loads(args.serial_comparison.read_text())
        serial_reference = json.loads(args.serial_result.read_text())
        if (
            checked.get("passed") is not True
            or checked.get("candidate_result_sha256")
            != base._sha256(args.serial_result)
            or serial_reference["source"] != source
            or serial_reference["kind"] != "serial_model_reference"
            or serial_reference.get("model_execution", "serial") != "serial"
        ):
            raise ValueError(
                "group control serial reference is not independently bound"
            )
        from group_controls import serial_reference_outputs

        if set(serial_reference_outputs(serial_reference)) != set(ordinary_outputs):
            raise ValueError("serial and ordinary references cover different requests")
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
    native_files = {"vllm." + name: path for name, path in extensions.items()}
    native_hashes = {"vllm." + name: digest for name, digest in hashes.items()}
    if hashes != base.EXPECTED_EXTENSION_HASHES:
        raise ValueError("compiled vLLM extensions differ")
    runtime = base._runtime_identity(
        vllm, lmcache, torch, transformers, extensions, hashes
    )
    for name, digest in pins["lmcache_native_sha256"].items():
        native_path = Path(importlib.import_module("lmcache." + name).__file__)
        native_files["lmcache." + name] = native_path
        native_hashes["lmcache." + name] = digest
        if base._sha256(native_path) != digest:
            raise ValueError(f"compiled LMCache extension differs: {name}")
    for name, expected in pins.get("auxiliary_native_sha256", {}).items():
        native_path = Path(importlib.import_module(name).__file__)
        native_files[name], native_hashes[name] = native_path, expected
        if base._sha256(native_path) != expected:
            raise ValueError(f"compiled storage extension differs: {name}")

    # Extend only source admission for this separately pinned descendant. Keep
    # the frozen capture geometry, raw tensor policy and warmup checks intact.
    class QualifiedCapture(numerical.NumericalInstrumentation):
        @staticmethod
        def _validate_source(value):
            if value != source:
                raise ValueError("capture source differs from qualified descendant")
            return copy.deepcopy(value)

        def _recovery_contract(self, recovery):
            if args.model_execution == "collective-lockstep":
                from group_capture import recovery_contract

                return recovery_contract(recovery, self._captured)
            return super()._recovery_contract(recovery)

    ledger = base.EventLedger(args.output_dir / "events.jsonl")
    base._atomic_json(args.output_dir / "status.json", {"status": "running"})
    llm = bridge = client = process = None
    parent = None
    storage_evidence = None
    storage_verification = None
    capture_arm = (
        "serial_oracle" if args.model_execution == "serial" else "collective_forward"
    )
    try:
        injection = baseline._install_instrumentation(
            ledger, inject_sibling_failure=False
        )
        if args.group_failure:
            from group_controls import install_member_failure
            from lmcache.v1.compute.blend.blender import LMCBlender, LMCBlendSession

            injection = install_member_failure(ledger, LMCBlendSession, LMCBlender)
        capture = (
            None
            if is_control
            else QualifiedCapture(
                ledger=ledger,
                arm=capture_arm,
                output_dir=args.output_dir,
                records=tuple(copy.deepcopy(records)),
                source=source,
            )
        )
        selection.install(ledger)
        worker_fault = faults.install_worker(args.control)
        if args.route == "registered":
            parent, child = mp.get_context("spawn").Pipe()
            process = mp.get_context("spawn").Process(
                target=service,
                args=(
                    child,
                    args.vllm_source,
                    args.wheel_vllm_package,
                    args.lmcache_source,
                    args.control,
                    args.storage_route,
                    args.storage_file,
                    pins["storage"],
                    args.poison_arena,
                    args.checksum_library,
                ),
            )
            process.start()
            child.close()
            kind, address = receive(parent)
            if kind != "ready":
                raise RuntimeError("cache process did not initialize")
            client = RequestClientFactory.create(address)
            model_files = contract.profile()["model_files"]

            def before_consume(request_id):
                parent.send(("cold", request_id))
                kind, value = receive(parent)
                if kind != "cold":
                    raise RuntimeError("owned service did not confirm cold sources")
                ledger.emit("registered.storage.before", **value)

            def after_consume(request_id):
                parent.send(("measure", request_id))
                kind, value = receive(parent)
                if kind != "measured":
                    raise RuntimeError(
                        "owned service did not report actual storage work"
                    )
                ledger.emit(
                    "registered.storage.after", request_id=request_id, evidence=value
                )

            replay_type = RegisteredReplay
            if args.model_execution == "collective-lockstep":
                from group_bridge import RegisteredGroupReplay

                replay_type = RegisteredGroupReplay
            bridge = replay_type(
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
                    "execution": identity(
                        {"profile": pins, "model_execution": args.model_execution}
                    ),
                },
                ledger,
                lambda: stop_importer(process),
                before_consume=(
                    None if args.storage_route == "cpu-l1" else before_consume
                ),
                after_consume=None if args.storage_route == "cpu-l1" else after_consume,
            )
            bridge.install()
        ledger.set_phase("construction")
        target_profile, kwargs = engine_options(common, args.model_execution)
        llm = LLM(**kwargs)
        from vllm.v1.kmesh.config import validate_kmesh_target_profile

        if validate_kmesh_target_profile(llm.llm_engine.vllm_config) != target_profile:
            raise ValueError("runtime is not the bounded selected model profile")
        if llm.llm_engine.vllm_config.attention_config.backend.name != "TRITON_ATTN":
            raise ValueError("runtime attention backend changed")
        control_evidence = None
        if args.group_failure:
            from group_controls import verify_member_failure

            observations = baseline._execute_arm(
                "sibling_failure", llm, records, ledger
            )
            events = ledger.snapshot()
            # The original evidence builder also enforces the full failed-member
            # invalidation -> empty retry plan -> completed request causal chain.
            recovery = baseline._recovery_evidence(observations, events)
            control_evidence = verify_member_failure(
                events,
                recovery,
                observations,
                ordinary_outputs,
                serial_reference,
                injection,
            )
            numeric_result = {
                "enabled": False,
                "status": "disabled",
                "reason": "mixed group repair and member recomputation control",
            }
        elif args.control == "none":
            observations = baseline._execute_arm(capture_arm, llm, records, ledger)
            events = ledger.snapshot()
            recovery = baseline._recovery_evidence(observations, events)
            numeric_result = capture.finish(
                arm=capture_arm,
                observations=observations,
                recovery_evidence=recovery,
                source=source,
                runtime=runtime,
            )
        else:
            observations, control_evidence = controls.execute(
                args.control, llm, records, ledger, bridge, parent, worker_fault
            )
            if any(
                o["output_token_ids"] != ordinary_outputs[o["trace_request_id"]]
                for o in observations
            ):
                raise ValueError(
                    "control model output differs from the ordinary baseline"
                )
            events = ledger.snapshot()
            recovery = {}
            numeric_result = {
                "enabled": False,
                "status": "disabled",
                "reason": "model failure control",
            }
        if args.route == "registered" and not is_control:
            stored = [e for e in events if e["action"] == "registered.span.stored"]
            restored = [e for e in events if e["action"] == "registered.span.restored"]
            consumed = [e for e in events if e["action"] == "registered.model.consumed"]
            if len(stored) != 4 or len(restored) != 16:
                raise ValueError("registered operation/model coverage is incomplete")
            if args.model_execution == "serial" and (
                len(consumed) != 4
                or any(
                    e["span_count"] != 4
                    or e["restored_tokens"] != 416
                    or not e["source_closed"]
                    or e["pending"]
                    for e in consumed
                )
            ):
                raise ValueError(
                    "model did not consume and settle all registered inputs"
                )
            if args.model_execution == "collective-lockstep":
                groups = [
                    e
                    for e in events
                    if e["action"] == "registered.model.group.consumed"
                ]
                if (
                    consumed
                    or len(groups) != 1
                    or groups[0]["span_counts"] != [4] * 4
                    or groups[0]["restored_tokens"] != [416] * 4
                    or groups[0]["outcomes"] != ["recovered"] * 4
                    or not groups[0]["source_closed"]
                    or groups[0]["pending"]
                ):
                    raise ValueError(
                        "model group did not consume and settle every registered input"
                    )
            if any(e["source_start"] == e["destination_start"] for e in restored):
                raise ValueError("model replay did not exercise shifted positions")
        if args.route == "registered":
            bridge.close()
            parent.send("stop")
            kind, storage_evidence = receive(parent)
            if kind != "closed":
                raise RuntimeError("cache process did not close")
            process.join(30)
            if process.exitcode != 0:
                raise RuntimeError("cache process did not exit successfully")
            if args.storage_route != "cpu-l1":
                import storage_verify

                storage_verification = storage_verify.verify(
                    args.storage_route,
                    storage_evidence,
                    pins["storage"],
                    require_poison=args.poison_arena,
                    checksum_backend="cuda-sha256" if gpu_checksum else "cpu",
                )
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
        if (
            profile_hash != base._sha256(args.profile)
            or helper_hashes != {p.name: base._sha256(p) for p in ROOT.glob("*.py")}
            or native_hashes
            != {name: base._sha256(p) for name, p in native_files.items()}
        ):
            raise ValueError(
                "profile, helper or native inputs changed during model execution"
            )
        base._atomic_json(
            args.output_dir / "result.json",
            {
                "status": "complete",
                "kind": (
                    (
                        "registered_group_member_control"
                        if args.group_failure
                        else "registered_model_control"
                    )
                    if is_control
                    else (
                        (
                            "registered_native_serial_model"
                            if args.model_execution == "serial"
                            else "registered_native_group_model"
                        )
                        if args.route == "registered"
                        else "serial_model_reference"
                    )
                ),
                "route": args.route,
                "model_execution": args.model_execution,
                "group_failure": args.group_failure,
                "storage_route": args.storage_route,
                "gpu_checksum": gpu_checksum,
                "poison_arena": args.poison_arena,
                "storage_evidence": storage_evidence,
                "storage_verification": storage_verification,
                "control_evidence": control_evidence,
                "ordinary_result_sha256": (
                    None
                    if args.ordinary_result is None
                    else base._sha256(args.ordinary_result)
                ),
                "serial_result_sha256": (
                    None
                    if args.serial_result is None
                    else base._sha256(args.serial_result)
                ),
                "source": source,
                "runtime": runtime,
                "profile": pins,
                "lmcache_config_sha256": base._sha256(LMCACHE_CONFIG),
                "profile_sha256": profile_hash,
                "native_sha256": native_hashes,
                "runner_sha256": base._sha256(Path(__file__)),
                "bridge_sha256": base._sha256(ROOT / "bridge.py"),
                "observations": observations,
                "recovery_evidence": recovery,
                "numerical_capture": numeric_result,
                "selection_evidence": [
                    e for e in events if e["action"] == "registered.model.selection"
                ],
                "event_ledger_sha256": base._sha256(ledger.path),
                "helper_sha256": helper_hashes,
                "claim_boundary": (
                    (
                        "CPU-L1 model repair with request-local operations in layer-major order; duplicated reference storage; no batched tensor work, serving speed or DMA-BUF claim."
                        if args.model_execution == "collective-lockstep"
                        else "CPU-L1 serial model correctness; duplicated reference storage; no serving latency or DMA-BUF claim."
                    )
                    if args.storage_route == "cpu-l1"
                    else "Controlled cold registered restore and serial repair; duplicated reference storage and capture overhead; no natural eviction-policy, total-memory or production serving-speed claim."
                ),
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

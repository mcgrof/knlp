# SPDX-License-Identifier: MIT
"""Worker-local calibration and explicit, untimed qualification RPCs.

Import this class through vLLM's worker_cls. Parent-process monkey patches do
not establish calibration in a spawned API worker. The development RPC endpoint
must be bound to loopback on a dedicated experiment host.
"""

import collections
import gc
import gzip
import hashlib
import json
import os
import shutil
import time
from contextlib import ExitStack
from pathlib import Path

import torch

from . import instrument
from .memory_worker import ObservedWorker
from .october import check_protocol, check_scales


class ServingWorker(ObservedWorker):
    def compile_or_warm_up_model(self):
        import flashinfer
        import vllm

        self.study_inputs = Path(os.environ["KNLP_SERVING_INPUTS"])
        self.study_output = Path(os.environ["KNLP_SERVING_OUTPUT"])
        self.study_arm = os.environ["KNLP_SERVING_ARM"]
        protocol = json.loads((self.study_inputs / "protocol.json").read_text())
        self.study_torch_cap = protocol["torch_allocator_reserved_cap_bytes"]
        definition = check_protocol(
            protocol, os.environ["KNLP_SERVING_VARIANT"], os.environ
        )
        definition.verify_sources(
            {name: self.study_inputs / name for name in definition.sources},
            {"vllm": vllm.__file__, "flashinfer": flashinfer.__file__},
        )
        if torch.__version__ != protocol["torch"]:
            raise RuntimeError("worker Torch differs from the frozen protocol")
        for name, expected in protocol["inputs_sha256"].items():
            if (
                hashlib.sha256((self.study_inputs / name).read_bytes()).hexdigest()
                != expected
            ):
                raise RuntimeError("worker input hash differs: " + name)
        actual = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        }
        if actual != protocol["runner_files_sha256"]:
            raise RuntimeError("worker harness changed after preparation")
        scales = check_scales(
            json.loads((self.study_inputs / "scales.json").read_text()),
            protocol["geometry"],
        )
        capture = self.study_output / "before-capture"
        capture.mkdir()
        instrument.configure(
            self.model_runner.get_model(),
            scales,
            self.study_arm,
            str(capture),
            protocol["packed_execution"],
        )
        original = self.model_runner.capture_model
        checked = []

        def capture_checked(*args, **kwargs):
            records = []
            for name, module in self.model_runner.get_model().named_modules():
                if hasattr(module, "_v_scale") and hasattr(module, "kv_cache"):
                    index = int(name.split("layers.")[1].split(".")[0])
                    expected = (
                        scales[f"{index}.v"] if self.study_arm == "k16_v8" else 1.0
                    )
                    if module._v_scale_float != expected or float(
                        module._v_scale
                    ) != float(torch.tensor(expected, dtype=torch.float32)):
                        raise RuntimeError("worker value scale differs before capture")
                    if module.calculate_kv_scales:
                        raise RuntimeError("dynamic scale calculation is enabled")
                    records.append(
                        {
                            "layer": name,
                            "host": module._v_scale_float,
                            "gpu": float(module._v_scale),
                        }
                    )
            if len(records) != protocol["geometry"]["layers"]:
                raise RuntimeError("worker attention inventory differs")
            instrument.save(self.study_output / "capture-time-scales.json", records)
            checked.append(True)
            return original(*args, **kwargs)

        self.model_runner.capture_model = capture_checked
        try:
            result = super().compile_or_warm_up_model()
        finally:
            self.model_runner.capture_model = original
        if checked != [True]:
            raise RuntimeError("worker did not execute exactly one validated capture")
        self.study_sequence = 0
        self.study_rows = []
        self.study_context = None
        self.study_graphs = []
        self.study_label = None
        self.study_record_steps = True
        instrument.save(
            self.study_output / "worker-identity.json",
            {
                "pid": os.getpid(),
                "sources": dict(definition.sources),
                "loaded": {"vllm": vllm.__file__, "flashinfer": flashinfer.__file__},
                "arm": self.study_arm,
                "torch_threads": torch.get_num_threads(),
                "configuration": repr(self.vllm_config),
                "calibration_before_worker_warmup_and_capture": True,
            },
        )
        return result

    def execute_model(self, scheduler_output):
        # CPU metadata only, buffered. No synchronization or CUDA event in timing.
        if hasattr(self, "study_sequence"):
            scheduler_output._knlp_batch_id = self.study_sequence
            self.study_sequence += 1
            if self.study_record_steps:
                self.study_rows.append(
                    {
                        "batch_id": scheduler_output._knlp_batch_id,
                        "worker_monotonic_ns": time.monotonic_ns(),
                        "label": self.study_label,
                        "scheduled_tokens": dict(scheduler_output.num_scheduled_tokens),
                        "new_request_ids": [
                            r.req_id for r in scheduler_output.scheduled_new_reqs
                        ],
                        "resumed_request_ids": list(
                            scheduler_output.scheduled_cached_reqs.resumed_req_ids
                        ),
                        "finished_request_ids": list(scheduler_output.finished_req_ids),
                    }
                )
        return super().execute_model(scheduler_output)

    def study_begin(self, label, mode="timing"):
        from . import schedule

        if self.study_context is not None or mode not in (
            "timing",
            "qualification",
            "profile",
            "overhead",
        ):
            raise ValueError("invalid or overlapping worker interval")
        if not label or Path(label).name != label:
            raise ValueError("label must be one path component")
        destination = self.study_output / label
        destination.mkdir()
        self.study_label, self.study_mode = label, mode
        self.study_record_steps = mode != "overhead"
        self.study_rows, self.study_graphs = [], []
        schedule._label = label
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        self.study_context = ExitStack()
        self.study_checks = []
        if mode == "qualification":
            from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

            # Numerical oracle buffers are separately accounted instrumentation.
            torch.cuda.set_per_process_memory_fraction(1.0)

            old = CudaGraphManager.run_fullgraph

            def replay(manager, desc):
                n = desc.num_reqs
                values = {
                    name: getattr(self.model_runner.input_buffers, name)[:n]
                    .cpu()
                    .tolist()
                    for name in ("input_ids", "positions", "seq_lens")
                }
                self.study_graphs.append(
                    {
                        "requests": n,
                        "tokens": desc.num_tokens,
                        "values": values,
                        "sha256": hashlib.sha256(
                            json.dumps(values, sort_keys=True).encode()
                        ).hexdigest(),
                    }
                )
                return old(manager, desc)

            CudaGraphManager.run_fullgraph = replay
            self.study_context.callback(setattr, CudaGraphManager, "run_fullgraph", old)
            if self.study_arm == "k16_v8":
                from .compact import observe_routes
                from .routing_oracle import validate_routing_layers

                self.study_context.enter_context(
                    observe_routes(destination / "route.jsonl")
                )
                self.study_checks = self.study_context.enter_context(
                    validate_routing_layers(destination / "layers")
                )
        if mode == "profile":
            from .profile_ranges import trace_attention_routes

            self.study_context.enter_context(
                trace_attention_routes(destination / "operations.json")
            )
            self.study_profile = self.study_context.enter_context(
                torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=False,
                )
            )
        return {"label": label, "mode": mode, "pid": os.getpid()}

    def study_end(self):
        from . import schedule
        from .storage_receipts import inventory

        if self.study_context is None:
            raise ValueError("no worker interval")
        torch.cuda.synchronize()
        context, self.study_context = self.study_context, None
        context.close()
        destination = self.study_output / self.study_label
        result = {
            "label": self.study_label,
            "mode": self.study_mode,
            "steps": len(self.study_rows),
            "oracle_checks": len(self.study_checks),
            "memory": instrument.native_memory_state(self),
        }
        instrument.save(destination / "steps.json", self.study_rows)
        instrument.save(destination / "graphs.json", self.study_graphs)
        instrument.save(destination / "receipt.json", result)
        if self.study_mode in ("qualification", "profile"):
            instrument.save(
                destination / "storage.json", inventory(self, self.study_arm)
            )
        if self.study_mode == "profile":
            path = destination / "trace.json"
            self.study_profile.export_chrome_trace(str(path))
            with path.open("rb") as src, gzip.open(str(path) + ".gz", "wb") as dst:
                shutil.copyfileobj(src, dst)
            path.unlink()
            del self.study_profile
        self.study_rows, self.study_graphs, self.study_checks = [], [], []
        self.study_label = schedule._label = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(
            self.study_torch_cap / torch.cuda.get_device_properties(0).total_memory
        )
        return result

    def study_snapshot(self):
        from .storage_receipts import inventory

        return inventory(self, self.study_arm)

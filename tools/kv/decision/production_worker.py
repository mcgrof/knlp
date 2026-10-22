# SPDX-License-Identifier: MIT
"""Worker setup for the separately qualified, native-query paged comparison.

This class does not reuse historical worker certificates. Numerical checks
and a complete quality panel must be retained for this exact execution.
"""

import gc
import gzip
import os
import resource
from pathlib import Path
import shutil

import torch

from tools.kv.native import instrument
from tools.kv.native.memory_worker import ObservedWorker
from tools.kv.native.serving_worker import ServingWorker
from .paged_memory import inventory
from .inline_scales import verify_inline_scale_aliases
from .production import environment, read_inputs, sha, variant, verify_runner


class ProductionWorker(ServingWorker):
    def check_scales(self):
        records = []
        for name, module in self.model_runner.get_model().named_modules():
            if not hasattr(module, "_k_scale") or not hasattr(module, "kv_cache"):
                continue
            index = int(name.split("layers.")[1].split(".")[0])
            expected = {
                "q": 1.0,
                "k": (
                    self.decision_protocol["scales"][f"{index}.k"]
                    if self.decision_arm == "SF"
                    else 1.0
                ),
                "v": (
                    self.decision_protocol["scales"][f"{index}.v"]
                    if self.decision_arm in ("AF", "SF")
                    else 1.0
                ),
            }
            for component, value in expected.items():
                if (
                    getattr(module, f"_{component}_scale_float") != value
                    or float(getattr(module, f"_{component}_scale")) != value
                ):
                    raise RuntimeError("Frozen worker scale changed: " + name)
            if module.calculate_kv_scales:
                raise RuntimeError("Runtime scale calibration is enabled")
            if self.decision_arm == "SD":
                if (
                    type(module.impl).__name__ != "TritonAttentionImpl"
                    or module.kv_cache_dtype != "fp8_per_token_head"
                    or not module.impl.reference_fp8_per_token_head
                    or not module.impl._is_per_token_head_quant
                    or not hasattr(module, "query_quant")
                    or module.query_quant is not None
                ):
                    raise RuntimeError("Dynamic writer or native query policy differs")
                verify_inline_scale_aliases(
                    module.kv_cache,
                    self.decision_protocol["geometry"],
                    (module.impl._k_scale_cache, module.impl._v_scale_cache),
                )
            records.append({"layer": name, **expected})
        if len(records) != self.decision_protocol["geometry"]["layers"]:
            raise RuntimeError("Incomplete worker scale inventory")
        return records

    def compile_or_warm_up_model(self):
        import flashinfer
        import transformers
        import vllm
        from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

        self.study_inputs = Path(os.environ["KNLP_DECISION_INPUTS"])
        self.study_output = Path(os.environ["KNLP_DECISION_OUTPUT"])
        self.decision_arm = os.environ["KNLP_DECISION_ARM"]
        self.decision_allocation = os.environ["KNLP_DECISION_ALLOCATION"]
        self.study_arm = {"AF": "k16_v8", "SD": "k8_v8"}.get(
            self.decision_arm, "k16_v16"
        )
        protocol, _ = read_inputs(
            self.study_inputs, os.environ["KNLP_DECISION_PROTOCOL_SHA256"]
        )
        self.decision_protocol = protocol
        self.study_torch_cap = protocol["torch_allocator_reserved_cap_bytes"]
        verify_runner(protocol)
        definition = variant(self.decision_arm)
        definition.verify_environment()
        for name, value in environment(self.decision_arm).items():
            if os.environ.get(name) != value:
                raise RuntimeError("Frozen worker environment differs: " + name)
        install = Path(os.environ["KNLP_DECISION_INSTALL"])
        definition.verify_sources(
            {name: install / name for name in definition.sources},
            {"vllm": vllm.__file__, "flashinfer": flashinfer.__file__},
        )
        if (
            torch.__version__ != protocol["torch"]
            or transformers.__version__ != protocol["transformers"]
        ):
            raise RuntimeError("Worker package identity differs")
        config = self.vllm_config
        if config.model_config.dtype != torch.float16:
            raise RuntimeError("Worker model arithmetic must remain FP16")
        if (
            self.decision_arm in ("native_fi", "AF")
            and not config.attention_config.disable_flashinfer_q_quantization
        ):
            raise RuntimeError("Native query dtype was not requested")
        capture = self.study_output / "before-capture"
        capture.mkdir()
        scales = protocol["scales"]
        if self.decision_arm == "SD":
            scales = {key: 1.0 for key in scales}
            for _, module in self.model_runner.get_model().named_modules():
                if hasattr(module, "_k_scale") and hasattr(module, "kv_cache"):
                    if type(module.impl).__name__ != "TritonAttentionImpl":
                        raise RuntimeError("Dynamic candidate did not select Triton")
                    # Construct and initialize scale aliases before any graph.
                    module.impl._ensure_scale_caches(module.kv_cache)
        instrument.configure(
            self.model_runner.get_model(),
            scales,
            self.study_arm,
            str(capture),
            {
                "format": "fp16",
                "attention_layers": 36,
                "linear_modules": 144,
                "embedding_modules": 2,
            },
        )
        original_capture = self.model_runner.capture_model
        original_manager_capture = CudaGraphManager.capture
        checked = []
        self.decision_graph_forwards = {}

        def retain_forwards(manager, factory, *args, **kwargs):
            def retained(desc, warmup):
                forward, state = factory(desc, warmup)
                if not warmup:
                    key = (id(manager), desc)
                    if key in self.decision_graph_forwards:
                        raise RuntimeError("Repeated graph descriptor capture")
                    # Retain the unchanged closure and its original metadata for
                    # excluded graph-versus-eager numerical checks. No operation
                    # is added to the captured graph itself.
                    self.decision_graph_forwards[key] = (forward, state)
                return forward, state

            return original_manager_capture(manager, retained, *args, **kwargs)

        def capture_checked(*args, **kwargs):
            instrument.save(
                self.study_output / "capture-time-scales.json", self.check_scales()
            )
            checked.append(True)
            return original_capture(*args, **kwargs)

        self.model_runner.capture_model = capture_checked
        CudaGraphManager.capture = retain_forwards
        try:
            # Bypass the historical ServingWorker's asymmetric-only scale gate.
            result = ObservedWorker.compile_or_warm_up_model(self)
        finally:
            self.model_runner.capture_model = original_capture
            CudaGraphManager.capture = original_manager_capture
        if checked != [True] or not self.decision_graph_forwards:
            raise RuntimeError("Expected exactly one checked graph-capture phase")
        self.check_scales()
        self.study_sequence = 0
        self.study_rows, self.study_graphs = [], []
        self.study_context = self.study_label = None
        self.study_record_steps = True
        from tools.kv.native import schedule

        # The uni executor shares the engine-core process with its scheduler.
        # Buffer CPU scheduling metadata; leave CUDA event instrumentation off.
        schedule._mode = "buffered"
        schedule.install(self.study_output / "scheduler.jsonl")
        from vllm.v1.core.sched.scheduler import Scheduler

        self.decision_scheduler = None
        observed_schedule = Scheduler.schedule

        def retain_scheduler(scheduler, *args, **kwargs):
            self.decision_scheduler = scheduler
            return observed_schedule(scheduler, *args, **kwargs)

        Scheduler.schedule = retain_scheduler
        instrument.save(
            self.study_output / "worker-identity.json",
            {
                "pid": os.getpid(),
                "sources": dict(definition.sources),
                "loaded": {"vllm": vllm.__file__, "flashinfer": flashinfer.__file__},
                "arm": self.decision_arm,
                "allocation": self.decision_allocation,
                "protocol_sha256": os.environ["KNLP_DECISION_PROTOCOL_SHA256"],
                "source_sha": protocol["source_sha"],
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "torch_threads": torch.get_num_threads(),
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "fd_limit": list(resource.getrlimit(resource.RLIMIT_NOFILE)),
                "configuration": repr(config),
                "calibration_before_worker_warmup_and_capture": True,
                "retained_graph_descriptors": [
                    str(d) for _, d in self.decision_graph_forwards
                ],
            },
        )
        instrument.save(
            self.study_output / "initial-storage.json", self.study_snapshot()
        )
        return result

    def study_pending(self):
        """Observe scheduler state between cells without admitting more work."""
        import time

        scheduler = self.decision_scheduler
        return {
            "at_ns": time.monotonic_ns(),
            "scheduler_seen": scheduler is not None,
            "running": len(scheduler.running) if scheduler is not None else 0,
            "waiting": len(scheduler.waiting) if scheduler is not None else 0,
            "requests": len(scheduler.requests) if scheduler is not None else 0,
        }

    def execute_model(self, scheduler_output):
        result = super().execute_model(scheduler_output)
        if getattr(self, "study_rows", None):
            observed = getattr(scheduler_output, "_knlp_observation", None)
            self.study_rows[-1]["scheduler_batch_id"] = (
                None if observed is None else observed["batch_id"]
            )
        return result

    def study_snapshot(self):
        self.check_scales()
        result = inventory(
            self,
            self.decision_arm,
            geometry=self.decision_protocol["geometry"],
            persistent_budget_bytes=self.decision_protocol["persistent_budget_bytes"],
        )
        if (
            self.decision_allocation == "equal_slots"
            and result["physical_blocks_per_layer"]
            != self.decision_protocol["equal_slot_blocks"]
        ):
            raise RuntimeError("Actual equal-slot allocation differs from the protocol")
        return result

    def study_build_identity(self, label):
        import flashinfer.jit.env as jit_environment
        import importlib.metadata

        if label not in ("numerical", "quality", "timing"):
            raise ValueError("Unknown build-inventory boundary")
        destination = self.study_output / ("build-identity-" + label + ".json")
        if destination.exists():
            raise ValueError("Build inventory boundary was already recorded")

        loaded = set()
        for line in Path("/proc/self/maps").read_text().splitlines():
            parts = line.split(maxsplit=5)
            if len(parts) != 6:
                continue
            path = Path(parts[5])
            if (
                ".so" in path.name
                and any(
                    x in str(path).lower() for x in ("flashinfer", "vllm", "triton")
                )
                and path.is_file()
            ):
                loaded.add(path)
        roots = {
            Path(getattr(jit_environment, name))
            for name in (
                "FLASHINFER_JIT_DIR",
                "FLASHINFER_CUBIN_DIR",
                "FLASHINFER_AOT_DIR",
                "FLASHINFER_GEN_SRC_DIR",
            )
            if hasattr(jit_environment, name)
        }
        roots.add(
            Path(os.environ.get("TRITON_CACHE_DIR", str(Path.home() / ".triton/cache")))
        )
        generated = set()
        for root in roots:
            for extension in (
                "*.so",
                "*.cubin",
                "*.ptx",
                "*.json",
                "*.cu",
                "*.cuh",
                "*.h",
            ):
                generated.update(p for p in root.rglob(extension) if p.is_file())
        result = {
            "loaded_shared_objects_sha256": {str(p): sha(p) for p in sorted(loaded)},
            "generated_cache_files_sha256": {str(p): sha(p) for p in sorted(generated)},
            "generated_cache_scope": "Compiled cache inventory can include unused specializations; actual dispatch is observed separately during numerical qualification.",
            "torch_cuda_runtime": torch.version.cuda,
            "package_versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "vllm", "triton")
            },
        }
        instrument.save(destination, result)
        return {
            "file": destination.name,
            "sha256": sha(destination),
            "loaded_shared_objects": len(loaded),
            "generated_files": len(generated),
        }

    def study_begin(self, label, mode="timing"):
        from tools.kv.native import schedule

        self.check_scales()
        schedule._completed.clear()
        self.decision_checks = []
        # Use this comparison's all-arm raw-page oracle. The older AF-only
        # oracle retains full compact-prefill witnesses and has a different
        # recipe/coverage contract; it is not a substitute for this gate.
        result = super().study_begin(
            label, "timing" if mode == "qualification" else mode
        )
        self.study_mode = mode
        result["mode"] = mode
        if mode == "qualification":
            from .production_oracle import check_worker

            torch.cuda.set_per_process_memory_fraction(1.0)
            self.decision_checks = self.study_context.enter_context(
                check_worker(self, self.study_output / label / "numerics")
            )
        return result

    def study_end(self):
        from tools.kv.native import schedule

        if self.study_context is None:
            raise ValueError("No active worker interval")
        torch.cuda.synchronize()
        context, self.study_context = self.study_context, None
        context.close()
        destination = self.study_output / self.study_label
        result = {
            "label": self.study_label,
            "mode": self.study_mode,
            "steps": len(self.study_rows),
            "preemption_events": sum(
                len(r["preempted_request_ids"]) for r in self.study_rows
            ),
            "oracle_checks": len(self.study_checks),
            "decision_numerical_checks": len(self.decision_checks),
            "scheduler_events": len(schedule._completed),
            "memory": instrument.native_memory_state(self),
        }
        instrument.save(destination / "steps.json", self.study_rows)
        instrument.save(destination / "graphs.json", self.study_graphs)
        instrument.save(destination / "receipt.json", result)
        schedule.flush()
        instrument.save(destination / "scheduler-completions.json", schedule._completed)
        schedule._completed.clear()
        instrument.save(destination / "storage.json", self.study_snapshot())
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

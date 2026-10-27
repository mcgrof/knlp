# SPDX-License-Identifier: MIT
"""Checkpoint evaluator and deterministic replay using unchanged author modules."""

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .replay import BatchAdmission, configure_scheduling, runtime_settings
from .run import (
    BENCHMARKS,
    CHECKPOINT_RELEASE_LABELS,
    DATA,
    ENCODER,
    LMCACHE,
    READER,
    VLLM,
    digest,
    write,
)
from .workloads import fingerprint, percentile, request_record, schedule


def tensor_digest(tensor):
    """Hash tensor values with dtype and shape, including bfloat16."""
    import torch

    owned = tensor.detach().cpu().contiguous()
    return fingerprint(
        {
            "dtype": str(owned.dtype),
            "shape": list(owned.shape),
            "sha256": hashlib.sha256(
                owned.reshape(-1).view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        }
    )


def observe_tensors(model):
    """Observe reader output and exact prompt embeddings without changing them."""
    observed = {}

    def reader_hook(module, args, output):
        observed["reader_sha256"] = [tensor_digest(row) for row in output]

    hook = model.memory.register_forward_hook(reader_hook)
    original = model.engine.generate

    def generate(requests, *args, **kwargs):
        observed["prompt_sha256"] = [
            tensor_digest(request["prompt_embeds"]) for request in requests
        ]
        return original(requests, *args, **kwargs)

    model.engine.generate = generate
    return observed, hook


def catalog_class(root):
    """Load the fork's CPU-only catalog without replacing a built engine."""
    source = Path(root) / "source/vllm"
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != VLLM:
        raise ValueError("catalog source revision mismatch")
    subprocess.run(
        ["git", "-C", str(source), "diff", "--exit-code", "HEAD", "--"], check=True
    )
    # Load exactly three independent modules. Do not change the engine's
    # package search path or resolve its implementation from another tree.
    for name in ["artifact_registry", "cartridge_manifest", "cartridge_registry"]:
        filename = source / "vllm" / f"{name}.py"
        key = f"vllm.{name}"
        module = sys.modules.get(key)
        if module is not None:
            if Path(module.__file__).resolve() != filename.resolve():
                raise ValueError(
                    "an existing catalog shadows the pinned implementation"
                )
        else:
            spec = importlib.util.spec_from_file_location(key, filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules[key] = module
            spec.loader.exec_module(module)
    return module.CartridgeRegistry


class ArtifactEmbeddingStore:
    """Author collator API backed by leased LMCache collections.

    The collator expects owned tensors after documents() returns. Copy while
    leased, then let its unchanged padding code consume that owned tensor.
    Document conversion/publication precedes measured generation.
    """

    def __init__(self, native, cfg, checkpoint, output):
        from lmcache.v1.memory_allocators.host_memory_allocator import (
            HostMemoryAllocator,
        )
        from safetensors.torch import save_file

        from tools.reproduce.context_artifacts.artifacts import (
            ConsumerIdentity,
            ContextArtifactManifest,
            DocumentMetadata,
            inspect_payload,
        )
        from tools.reproduce.context_artifacts.lmcache_adapter import (
            LMCacheArtifactRegistry,
        )

        self.native = native
        self.consumer = ConsumerIdentity("memorilla/Memorilla-Qwen3-8B", READER)
        directory = Path(cfg["root"]) / "artifacts" / native.config
        directory.mkdir(parents=True, exist_ok=True)
        self.catalog = catalog_class(cfg["root"])(directory / "cartridges.sqlite3")
        self.registry = LMCacheArtifactRegistry(
            self.catalog, HostMemoryAllocator(cfg["pool_bytes"])
        )
        self.ids = {}
        self._shapes = {}
        checkpoint_hash = digest(checkpoint / "memory.pt")
        for collection in native.collection_ids.tolist():
            source = directory / f"{DATA}-{int(collection)}.safetensors"
            vectors = native.documents(collection)
            if not source.exists():
                save_file({"embeddings": vectors.contiguous()}, str(source))
            # A retained bank must still match its pinned native source.
            import torch
            from safetensors.torch import load_file

            if not torch.equal(load_file(str(source))["embeddings"], vectors):
                raise ValueError(
                    "retained collection differs from pinned author source"
                )
            manifest = ContextArtifactManifest(
                1,
                "memorilla-evaluation",
                "document_embeddings",
                "Memorilla",
                cfg["source_revision"],
                checkpoint_hash,
                (self.consumer,),
                digest(source),
                inspect_payload(source),
                DocumentMetadata(
                    ConsumerIdentity("Qwen/Qwen3-Embedding-4B", ENCODER),
                    fingerprint(
                        {
                            "data": DATA,
                            "benchmark": native.config,
                            "split": native.split,
                        }
                    ),
                    fingerprint(
                        {
                            "data": DATA,
                            "benchmark": native.config,
                            "collection": collection,
                        }
                    ),
                    vectors.shape[0],
                    vectors.shape[1],
                ),
            )
            self.ids[int(collection)] = self.registry.register(manifest, source)
            self._shapes[manifest.artifact_id] = manifest.tensors[0].shape
        write(output / "artifacts.json", self.ids)

    def documents(self, collection):
        with self.registry.lease(
            self.ids[int(collection)], self.consumer, timeout=120.0
        ) as tensors:
            return tensors["embeddings"].clone()

    def collection_shape(self, collection):
        """Resolve admitted geometry without reserving payload memory."""
        return self._shapes[self.ids[int(collection)]]

    def copy_documents(self, collection, destination):
        """Finish a synchronous CPU copy before releasing the bank lease."""
        import torch

        from .tensor_copy import copy_fp16_cpu

        if destination.device.type != "cpu" or destination.dtype not in (
            torch.float16,
            torch.bfloat16,
        ):
            raise ValueError("document destination must be CPU float16 or bfloat16")
        with self.registry.lease(
            self.ids[int(collection)], self.consumer, timeout=120.0
        ) as tensors:
            source = tensors["embeddings"]
            if source.dtype != torch.float16 or source.shape != destination.shape:
                raise ValueError("document destination differs from bank geometry")
            copy_fp16_cpu(source, destination)

    def question(self, row):
        return self.native.question(row)

    def evict(self):
        for artifact in self.ids.values():
            self.registry.evict(artifact)

    def stats(self):
        store = self.registry.storage
        return {
            "range_loads": store.read_count,
            "logical_bytes": store.read_bytes,
            "resident_hits": store.cache_hits,
            "resident_objects": len(store.resident),
            "capacity_waits": store.capacity_waits,
            "capacity_wait_seconds": store.capacity_wait_seconds,
        }

    def close(self):
        self.registry.close()
        self.catalog.close()


def usage():
    observed = resource.getrusage(resource.RUSAGE_SELF)
    result = {
        "max_rss_kib": observed.ru_maxrss,
        "major_faults": observed.ru_majflt,
        "minor_faults": observed.ru_minflt,
    }
    try:
        result["proc_io"] = dict(
            line.split(": ", 1)
            for line in Path("/proc/self/io").read_text().splitlines()
        )
    except OSError:
        pass
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["benchmark", "stress"], required=True)
    opts = p.parse_args()
    cfg, inputs = json.loads(opts.config.read_text()), json.loads(
        opts.inputs.read_text()
    )
    # Historical resolved captures did not have this key. Preserve their
    # runtime mode; new configurations always record an explicit selection.
    scheduling = cfg.get("scheduling", "runtime")
    configure_scheduling(scheduling)
    cfg["source_revision"] = inputs["source_revision"]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(Path(cfg["root"]) / "hf")
    source = Path(inputs["source"])
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location(
        "memorilla_author_evaluate", source / "evaluate.py"
    )
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)
    import memorilla
    import torch
    import vllm
    from memorilla.benchmarks import BENCHMARKS as protocols
    from memorilla.data import MemoryCollator, MemoryDataset
    from memorilla.metrics import clean_generation
    from memorilla.utils import release_gpu_memory, seed_everything
    from memorilla.vllm import MemoryVLLM

    input_pipeline = cfg.get("input_pipeline", "author")
    if input_pipeline not in {"author", "leased-fp16", "leased-bf16"}:
        raise ValueError("unknown document input pipeline")
    if cfg["profile"] == "author" and input_pipeline != "author":
        raise ValueError("original author mode rejects collator overrides")
    collators = []
    bank_clients = None
    if input_pipeline != "author":
        from .input_batch import LeasedMemoryCollator

        class ConfiguredMemoryCollator(LeasedMemoryCollator):
            def __init__(self, *args, **kwargs):
                super().__init__(
                    *args,
                    clients=cfg.get("clients", 1),
                    final_dtype=(
                        torch.bfloat16
                        if input_pipeline == "leased-bf16"
                        else torch.float16
                    ),
                    **kwargs,
                )
                collators.append(self)

        MemoryCollator = ConfiguredMemoryCollator
        author.MemoryCollator = ConfiguredMemoryCollator

    if cfg.get("engine_options"):
        from functools import partial

        import memorilla.vllm as serving

        serving.LLM = partial(serving.LLM, **cfg["engine_options"])

    lmcache_installation = None
    if cfg["profile"] == "lmcache":
        import lmcache

        from .environment import verify_package

        lmcache_installation = verify_package(
            lmcache, Path(cfg["root"]) / "source/lmcache", LMCACHE
        )

    if cfg["profile"] == "author" and (
        torch.version.hip is not None or not torch.cuda.is_available()
    ):
        raise ValueError("the original author profile requires the pinned CUDA runtime")
    if Path(memorilla.__file__).resolve().parent != (source / "memorilla").resolve():
        raise ValueError("imported author package differs from pinned source")
    name = opts.benchmark
    folder = name if name in {"pmv2", "pv4"} else "stage3"
    checkpoint_release = {
        "published_label": CHECKPOINT_RELEASE_LABELS[folder],
        "model_card": (
            "https://huggingface.co/memorilla/Memorilla-Qwen3-8B/"
            f"blob/{READER}/README.md"
        ),
        "paper_lineage_verified": False,
    }
    checkpoint = Path(inputs["assets"]["reader"]["path"]) / folder
    decoder = inputs["assets"]["decoder"]["path"]
    engine_root = Path(vllm.__file__).resolve().parent
    engine_files = {
        str(path.relative_to(engine_root)): digest(path)
        for path in sorted(engine_root.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".so"}
    }
    write(opts.output / "engine-files.json", engine_files)
    args = argparse.Namespace(
        benchmark=name,
        config=None,
        split="test",
        scoring="generation",
        checkpoint=str(checkpoint),
        decoder=decoder,
        encoder="Qwen/Qwen3-Embedding-4B",
        system_prompt=None,
        system_prompt_from=None,
        max_length=None,
        max_new_tokens=None,
        batch_size=cfg["batch"],
        gpu_memory_utilization=cfg["gpu_fraction"],
        scorer_gpu_memory_utilization=cfg["scorer_fraction"],
        data_repo=inputs["assets"]["data"]["path"],
        data_dir=inputs["assets"]["data"]["path"],
        cache_dir=str(Path(cfg["root"]) / "hf/hub"),
        output_dir=str(opts.output / "author-results"),
        seed=cfg["seed"],
    )
    provenance = {
        "author_source": inputs["source_revision"],
        "runner_sha256": digest(Path(__file__)),
        "workloads_sha256": digest(Path(__file__).with_name("workloads.py")),
        "bank_prefetch_sha256": digest(Path(__file__).with_name("bank_prefetch.py")),
        "checkpoint_folder": folder,
        "checkpoint_release": checkpoint_release,
        "checkpoint_sha256": digest(checkpoint / "memory.pt"),
        "data_revision": DATA,
        "decoder_revision": inputs["assets"]["decoder"]["revision"],
        "encoder_revision": ENCODER,
        "runtime_profile": cfg["profile"],
        "offline_scheduling": scheduling,
        "bank_fetch_clients": cfg.get("clients", 1),
        "input_pipeline": input_pipeline,
        "singleton_reference": cfg.get("singleton_reference") == "y",
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "author_module": memorilla.__file__,
        "vllm_engine_module": vllm.__file__,
        "vllm_engine_version": vllm.__version__,
        "vllm_engine_tree_sha256": fingerprint(engine_files),
        "catalog_source_revision": VLLM if cfg["profile"] == "lmcache" else None,
        "lmcache_installation": lmcache_installation,
        "engine_options": cfg.get("engine_options", {}),
        "diagnostics": cfg.get("diagnostics") == "y",
        "torch_precision": {
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul": torch.get_float32_matmul_precision(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "protocol": asdict(protocols[name]),
        "page_cache_state": "unknown",
        "training": False,
    }
    write(opts.output / "provenance.json", provenance)
    original_model = MemoryVLLM

    class ReplayMemoryVLLM(original_model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            effective = runtime_settings(self.engine)
            provenance["effective_runtime"] = effective
            write(opts.output / "provenance.json", provenance)
            if cfg.get("singleton_reference") == "y" and (
                effective["enable_prefix_caching"] is not False
                or effective["max_num_seqs"] != 1
            ):
                raise ValueError("runtime did not apply singleton reference options")
            if scheduling == "batch":
                self.engine.generate = BatchAdmission(self.engine).generate

    MemoryVLLM = ReplayMemoryVLLM
    author.MemoryVLLM = ReplayMemoryVLLM
    original_load = author.load_target
    stores = []

    def load_target(benchmark, settings):
        benchmark, dataset, native = original_load(benchmark, settings)
        expected = BENCHMARKS[name][0]
        if len(dataset) != expected:
            raise ValueError(f"expected {expected} rows, got {len(dataset)}")
        if cfg["profile"] == "lmcache":
            native = ArtifactEmbeddingStore(native, cfg, checkpoint, opts.output)
            stores.append(native)
        return benchmark, dataset, native

    author.load_target = load_target
    try:
        before = usage()
        start = time.monotonic()
        if opts.mode == "benchmark":
            # The original driver constructs its collator/reader/engine, generates
            # the complete split and calls its original scorers.
            author.main(args)
            metric_path = opts.output / "author-results" / name / "metrics.json"
            metrics = json.loads(metric_path.read_text())
            key, target = BENCHMARKS[name][1:]
            write(
                opts.output / "paper-comparison.json",
                {
                    "paper_table": 11,
                    "metric": key,
                    "paper_percent": target,
                    "observed_percent": metrics[key],
                    "delta_percentage_points": metrics[key] - target,
                    "checkpoint_lineage_verified_against_paper": False,
                    "checkpoint_release": checkpoint_release,
                    "claim": "released-checkpoint evaluation; scores are observations, not an exact-reproduction gate",
                },
            )
        else:
            seed_everything(cfg["seed"])
            benchmark, dataset, store = load_target(protocols[name], args)
            items = MemoryDataset([(benchmark.config, benchmark.split, dataset)])
            rows = min(len(items), cfg["rows"] or len(items))
            order = schedule(rows, cfg["requests"], cfg["seed"], cfg["pattern"])

            def request(row, request_id, bank_row):
                if type(bank_row) is not int or not 0 <= bank_row < len(items):
                    raise ValueError("trace bank row is outside the selected split")
                return request_record(
                    dataset[row],
                    dataset[bank_row],
                    row,
                    bank_row,
                    request_id,
                    {
                        "data": DATA,
                        "checkpoint": provenance["checkpoint_sha256"],
                        "decoder_revision": provenance["decoder_revision"],
                        "encoder_revision": ENCODER,
                        "protocol": provenance["protocol"],
                    },
                )

            trace = [
                request(0 if cfg.get("bank_swap") == "y" else row, f"{name}:{i}", row)
                for i, row in enumerate(order)
            ]
            if cfg.get("trace"):
                from .workloads import validate_trace

                supplied = json.loads(Path(cfg["trace"]).read_text())
                validate_trace(supplied, len(items))
                for entry in supplied:
                    if entry != request(
                        entry["row"], entry["request_id"], entry.get("bank_row")
                    ):
                        raise ValueError(
                            "trace input identity differs from the selected checkpoint/data/protocol"
                        )
                trace = supplied
            write(opts.output / "trace.json", trace)
            memory = author.MemoryModule.from_pretrained(checkpoint)
            model = MemoryVLLM(
                decoder,
                memory,
                max_model_len=benchmark.max_length,
                gpu_memory_utilization=cfg["gpu_fraction"],
                cache_dir=args.cache_dir,
            )
            bank_clients = None
            if cfg.get("clients", 1) > 1 and input_pipeline == "author":
                from memorilla.embeddings import EmbeddingStore

                from .bank_prefetch import ConcurrentBanks

                factory = (
                    None
                    if cfg["profile"] == "lmcache"
                    else lambda: EmbeddingStore(
                        benchmark.config,
                        benchmark.split,
                        root=inputs["assets"]["data"]["path"],
                    )
                )
                bank_clients = ConcurrentBanks(store, cfg["clients"], factory)
            collator = MemoryCollator(
                model.tokenizer,
                {(benchmark.config, benchmark.split): bank_clients or store},
                num_memories=model.memory.num_memories,
                max_length=benchmark.max_length,
                system_prompt=benchmark.system_prompt,
                suffix=benchmark.suffix,
                train=False,
            )
            observed, hook = (
                observe_tensors(model) if cfg.get("diagnostics") == "y" else ({}, None)
            )
            durations, inconsistent = [], []
            seen = {}
            total_generation = 0
            with (opts.output / "requests.jsonl").open("w") as receipt:
                for index, offset in enumerate(range(0, len(trace), cfg["batch"])):
                    current = trace[offset : offset + cfg["batch"]]
                    if (
                        cfg["evict_every"]
                        and index % cfg["evict_every"] == 0
                        and stores
                    ):
                        store.evict()
                    tick = time.monotonic()
                    if bank_clients:
                        bank_clients.prepare(
                            [entry["collection_id"] for entry in current]
                        )
                    banks_ready = time.monotonic()
                    batch = collator(
                        [
                            items[entry["row"]]
                            | {"collection_id": entry["collection_id"]}
                            for entry in current
                        ]
                    )
                    if bank_clients:
                        bank_clients.finish()
                    collated = time.monotonic()
                    diagnostics = {}
                    if observed or hook is not None:
                        diagnostics["batch_input_sha256"] = fingerprint(
                            {key: tensor_digest(value) for key, value in batch.items()}
                        )
                    texts = model.generate(
                        **batch, max_new_tokens=benchmark.max_new_tokens
                    )
                    torch.cuda.synchronize()
                    elapsed = time.monotonic() - tick
                    durations.append(elapsed)
                    total_generation += elapsed
                    for position, (request, text) in enumerate(
                        zip(current, texts, strict=True)
                    ):
                        prediction = clean_generation(text)
                        semantic_key = request["input_digest"]
                        if semantic_key in seen and seen[semantic_key] != prediction:
                            inconsistent.append(request["request_id"])
                        seen[semantic_key] = prediction
                        if hook is not None:
                            diagnostics["reader_sha256"] = observed["reader_sha256"][
                                position
                            ]
                            diagnostics["prompt_sha256"] = observed["prompt_sha256"][
                                position
                            ]
                        receipt.write(
                            json.dumps(
                                request
                                | {
                                    "prediction": prediction,
                                    "batch_seconds": elapsed,
                                    "batch_size": len(current),
                                    "bank_prepare_seconds": banks_ready - tick,
                                    "collate_seconds": collated - banks_ready,
                                    "model_seconds": elapsed - (collated - tick),
                                    **diagnostics,
                                }
                            )
                            + "\n"
                        )
                    receipt.flush()
            write(
                opts.output / "stress-summary.json",
                {
                    "requests": len(trace),
                    "unique_inputs": len(seen),
                    "unique_questions": len({entry["row"] for entry in trace}),
                    "unique_collections": len(
                        {entry["collection_id"] for entry in trace}
                    ),
                    "synthetic_bank_pairs": any(
                        entry["row"] != entry["bank_row"] for entry in trace
                    ),
                    "batch_seconds_p50": percentile(durations, 0.5),
                    "batch_seconds_p95": percentile(durations, 0.95),
                    "batch_seconds_p99": percentile(durations, 0.99),
                    "generation_seconds": total_generation,
                    "bank_fetch_clients": cfg.get("clients", 1),
                    "bank_capacity_retries": (
                        bank_clients.retries if bank_clients else 0
                    ),
                    "requests_per_second": len(trace) / total_generation,
                    "repeated_question_prediction_changes": inconsistent,
                    "note": "Synchronous batched replay; batch latency is not individual request latency. Numeric output changes are reported, not silently tolerated.",
                },
            )
            del model, memory
            if hook is not None:
                hook.remove()
            release_gpu_memory()
        write(
            opts.output / "resources.json",
            {
                "before": before,
                "after": usage(),
                "elapsed_seconds": time.monotonic() - start,
                "note": "Driver-process usage only; engine children and device I/O are not fully attributed.",
            },
        )
    finally:
        for collator in collators:
            collator.close()
        if bank_clients:
            bank_clients.close(discard=sys.exc_info()[0] is not None)
        for store in stores:
            write(opts.output / "storage.json", store.stats())
            store.close()


if __name__ == "__main__":
    main()

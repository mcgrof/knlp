# SPDX-License-Identifier: MIT
"""Excluded actual-worker page, write, attention and CUDA-graph checks.

These checks synchronize and allocate diagnostic buffers. They must never be
installed during quality confirmation or serving timing. Bounds come from
the frozen worker protocol; this module does not fit tolerances to outputs.
"""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .oracle import oracle
from .production_numerics import attention, comparison, decoded, planes, slots, within


def byte_hash(tensor):
    return hashlib.sha256(
        tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


@contextmanager
def check_worker(worker, destination):
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    from vllm.v1.attention.backends.flashinfer import FlashInferImpl
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

    destination = Path(destination)
    destination.mkdir()
    runner = worker.model_runner
    bounds = worker.decision_protocol["numerical_bounds"]
    state, rows, originals = {}, [], []
    layers = [
        m
        for _, m in runner.get_model().named_modules()
        if hasattr(m, "_k_scale") and hasattr(m, "kv_cache")
    ]

    def emit(value):
        value = {"batch_id": worker.study_sequence - 1, **value}
        with (destination / "checks.jsonl").open("a") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")
        rows.append(value)

    original_prepare = runner.prepare_attn

    def prepare(batch):
        tables, mappings = original_prepare(batch)
        if len(tables) != 1 or batch.num_draft_tokens:
            raise RuntimeError("Oracle requires one cache group and ordinary decode")
        n = batch.num_reqs
        # Validate the gather against original persistent request-indexed tables.
        raw = runner.block_tables.block_tables[0].gpu
        indices = torch.tensor(
            batch.idx_mapping_np, device=raw.device, dtype=torch.long
        )
        original = raw.index_select(0, indices).cpu().tolist()
        gathered = tables[0][:n].cpu().tolist()
        lengths = batch.seq_lens[:n].cpu().tolist()
        for i, length in enumerate(lengths):
            count = (length + 15) // 16
            if gathered[i][:count] != original[i][:count]:
                raise RuntimeError("Request-indexed page gather changed raw page IDs")
        offsets = batch.query_start_loc_np[: n + 1].tolist()
        positions = batch.positions[: batch.num_tokens].cpu().tolist()
        expected = slots(offsets, lengths, original, positions)
        actual = mappings[0].cpu().tolist()
        if actual[: len(expected)] != expected or any(
            x != -1 for x in actual[len(expected) :]
        ):
            raise RuntimeError(
                "Actual writer slots differ from raw page/position oracle"
            )
        state.clear()
        state.update(
            offsets=offsets,
            lengths=lengths,
            pages=original,
            positions=positions,
            expected_slots=expected,
            request_ids=list(batch.req_ids),
            padded_tokens=batch.num_tokens_after_padding,
            phase=(
                "decode"
                if all(x == 1 for x in batch.num_scheduled_tokens)
                else (
                    "mixed"
                    if any(x == 1 for x in batch.num_scheduled_tokens)
                    else "prefill"
                )
            ),
        )
        emit(
            {
                "kind": "routing",
                "phase": state["phase"],
                "requests": n,
                "positions": positions,
                "slots": expected,
                "request_ids": state["request_ids"],
                "page_ids": [
                    r[: (length + 15) // 16] for r, length in zip(original, lengths)
                ],
            }
        )
        return tables, mappings

    runner.prepare_attn = prepare

    def install(cls):
        original_write, original_forward = cls.do_kv_cache_update, cls.forward
        originals.extend(
            [
                (cls, "do_kv_cache_update", original_write),
                (cls, "forward", original_forward),
            ]
        )

        def write(impl, layer, key, value, cache, mapping):
            if not state:
                raise RuntimeError(
                    "Cache write lacks independently observed raw requests"
                )
            expected = state["expected_slots"]
            actual = mapping.cpu().tolist()
            if actual[: len(expected)] != expected or any(
                x != -1 for x in actual[len(expected) :]
            ):
                raise RuntimeError("Layer writer received a different slot mapping")
            selected = np.unique(
                np.linspace(0, len(expected) - 1, min(16, len(expected)), dtype=int)
            ).tolist()
            ids = torch.tensor(selected, device=key.device, dtype=torch.long)
            assigned = torch.tensor(
                [expected[i] for i in selected], device=key.device, dtype=torch.long
            )
            operands = [
                x.reshape(-1, 8, 128).index_select(0, ids).detach().cpu()
                for x in (key, value)
            ]
            result = original_write(impl, layer, key, value, cache, mapping)
            records = []
            for component, plane, operand in zip(
                ("k", "v"), planes(cache, type(impl).__name__), operands
            ):
                scale = getattr(layer, "_" + component + "_scale_float")
                if not bool(torch.isfinite(operand).all()):
                    raise RuntimeError("Nonfinite native cache-write operand")
                actual_bytes = plane.view(torch.uint8)[
                    assigned // 16, assigned % 16
                ].cpu()
                quantized = plane.dtype in (torch.uint8, torch.float8_e4m3fn)
                expected_bytes = (
                    torch.from_numpy(oracle(operand, scale))
                    if quantized
                    else operand.contiguous().view(torch.uint8)
                )
                if not torch.equal(actual_bytes, expected_bytes):
                    np.savez_compressed(
                        destination / (layer.layer_name + "-writer-failure.npz"),
                        operand=operand.float().numpy(),
                        scale=scale,
                        actual=actual_bytes.numpy(),
                        expected=expected_bytes.numpy(),
                    )
                    raise RuntimeError(
                        "Stored cache bytes differ from independent writer oracle"
                    )
                records.append(
                    {
                        "component": component,
                        "scale": scale,
                        "dtype": str(plane.dtype),
                        "bytes_checked": actual_bytes.numel(),
                        "sha256": byte_hash(actual_bytes),
                    }
                )
            emit(
                {
                    "kind": "writer",
                    "phase": state["phase"],
                    "layer": layer.layer_name,
                    "sampled_input_rows": selected,
                    "records": records,
                    "exact": True,
                }
            )
            return result

        def forward(
            impl, layer, query, key, value, cache, metadata, output, *args, **kwargs
        ):
            if query.dtype != torch.float16 or layer._q_scale_float != 1.0:
                raise RuntimeError("Actual query arithmetic changed from native FP16")
            if type(impl).__name__ == "FlashInferImpl" and any(
                getattr(metadata, "q_data_type_" + phase) != torch.float16
                for phase in ("prefill", "decode")
            ):
                raise RuntimeError("Backend metadata would convert native queries")
            count = len(state["expected_slots"])
            # Two causal rows cover both ends of mixed batches. Retain all heads.
            selected = sorted(set([0, count - 1]))
            cached = planes(cache, type(impl).__name__)
            snapshots = []
            for index in selected:
                request = next(
                    i for i, end in enumerate(state["offsets"][1:]) if index < end
                )
                length = state["positions"][index] + 1
                ids = torch.tensor(
                    state["pages"][request][: (length + 15) // 16],
                    device=query.device,
                    dtype=torch.long,
                )
                raw = [p.view(torch.uint8).index_select(0, ids).clone() for p in cached]
                snapshots.append((index, length, ids, raw))
            result = original_forward(
                impl, layer, query, key, value, cache, metadata, output, *args, **kwargs
            )
            for index, length, ids, raw in snapshots:
                for plane, before in zip(cached, raw):
                    if not torch.equal(
                        plane.view(torch.uint8).index_select(0, ids), before
                    ):
                        raise RuntimeError(
                            "Attention read changed persistent cache bytes"
                        )
                values = [
                    decoded(
                        x.view(plane.dtype), getattr(layer, "_" + side + "_scale_float")
                    ).reshape(-1, 8, 128)[:length]
                    for x, plane, side in zip(raw, cached, ("k", "v"))
                ]
                q = query.reshape(-1, 32, 128)[index]
                observed = output.reshape(-1, 32, 128)[index]
                ideal = attention(q, *values, softmax_scale=impl.scale)
                reconstructed = attention(
                    q, *(x.half().float() for x in values), softmax_scale=impl.scale
                )
                numerical = comparison(observed, ideal)
                diagnostic = comparison(observed, reconstructed)
                emit(
                    {
                        "kind": "attention",
                        "phase": state["phase"],
                        "layer": layer.layer_name,
                        "query_row": index,
                        "causal_length": length,
                        "query_dtype": str(query.dtype),
                        "implementation": type(impl).__name__,
                        "prefill_dispatch": type(
                            getattr(metadata, "prefill", None)
                        ).__name__,
                        "decode_dispatch": str(
                            getattr(getattr(metadata, "decode", None), "kernel", None)
                        ),
                        "scalar_scale_fp32": numerical,
                        "native_reconstruction_diagnostic": diagnostic,
                        "bounds": bounds["attention"],
                        "passed": within(numerical, bounds["attention"]),
                    }
                )
                if not within(numerical, bounds["attention"]):
                    raise RuntimeError(
                        "Actual attention exceeds frozen numerical bounds"
                    )
            return result

        cls.do_kv_cache_update = write
        cls.forward = forward

    install(FlashInferImpl)
    install(FlashAttentionImpl)
    original_graph = CudaGraphManager.run_fullgraph

    def graph(manager, desc):
        result = original_graph(manager, desc)
        actual = manager.hidden_states[: desc.num_tokens].clone()
        assigned = torch.tensor(
            state["expected_slots"], device=actual.device, dtype=torch.long
        )
        snapshots = []
        for layer in layers:
            for plane in planes(layer.kv_cache, type(layer.impl).__name__):
                view = plane.view(torch.uint8)
                snapshots.append((view, view[assigned // 16, assigned % 16].clone()))
        try:
            original_graph(manager, desc)
            repeat = manager.hidden_states[: desc.num_tokens]
            if not torch.equal(actual[: len(assigned)], repeat[: len(assigned)]):
                raise RuntimeError("Graph replay is not repeatable on identical inputs")
            if any(
                not torch.equal(view[assigned // 16, assigned % 16], before)
                for view, before in snapshots
            ):
                raise RuntimeError("Repeated graph write changed current cache bytes")
            fn, _ = worker.decision_graph_forwards[(id(manager), desc)]
            fn(CUDAGraphMode.NONE)
            eager = manager.hidden_states[: desc.num_tokens]
            metrics = comparison(actual[: len(assigned)], eager[: len(assigned)])
            cache_exact = all(
                torch.equal(view[assigned // 16, assigned % 16], before)
                for view, before in snapshots
            )
            emit(
                {
                    "kind": "graph",
                    "descriptor": str(desc),
                    "requests": len(assigned),
                    "padded_tokens": desc.num_tokens,
                    "comparison": metrics,
                    "cache_codes_exact": cache_exact,
                    "bounds": bounds["graph"],
                    "passed": cache_exact and within(metrics, bounds["graph"]),
                }
            )
            if not cache_exact or not within(metrics, bounds["graph"]):
                raise RuntimeError(
                    "Graph/eager output or cache-write qualification failed"
                )
        finally:
            for view, before in snapshots:
                view[assigned // 16, assigned % 16] = before
            manager.hidden_states[: desc.num_tokens].copy_(actual)
        return result

    CudaGraphManager.run_fullgraph = graph
    try:
        yield rows
    finally:
        CudaGraphManager.run_fullgraph = original_graph
        runner.prepare_attn = original_prepare
        for cls, name, original in reversed(originals):
            setattr(cls, name, original)

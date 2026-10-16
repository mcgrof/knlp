# SPDX-License-Identifier: MIT
"""Check a bounded serving gather against stored cache bytes and mixed attention."""

import hashlib
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from .prefill import compare_close, save, select_pages, tensor_info


def tensor_hash(tensor):
    data = tensor.detach().contiguous().cpu().view(torch.uint8).numpy()
    return hashlib.sha256(data.tobytes()).hexdigest()


def full_comparison(actual, reference):
    """Check every element while bounding temporary float comparison storage."""
    if actual.shape != reference.shape:
        raise ValueError("full comparison shapes differ")
    parts = [
        compare_close(actual[i : i + 4096], reference[i : i + 4096])
        for i in range(0, len(actual), 4096)
    ]
    finite = all(x["finite"] for x in parts)
    return dict(
        passed=all(x["passed"] for x in parts),
        finite=finite,
        rtol=1e-3,
        atol=1e-3,
        elements=actual.numel(),
        violating_elements=sum(x["violating_elements"] for x in parts),
        max_absolute_error=(
            max(x["max_absolute_error"] for x in parts) if finite else None
        ),
    )


def workspace_receipt(workspace):
    row = workspace.receipt()
    row["allocated"] = torch.cuda.memory_allocated()
    row["reserved"] = torch.cuda.memory_reserved()
    row["peak_reserved"] = torch.cuda.max_memory_reserved()
    row["wrapper_tensors"] = {
        name: tensor_info(value)
        for name, value in vars(workspace.wrapper).items()
        if isinstance(value, torch.Tensor)
    }
    row["wrapper_storage"] = {
        name: dict(
            pointer=value.untyped_storage().data_ptr(),
            storage_bytes=value.untyped_storage().nbytes(),
        )
        for name, value in vars(workspace.wrapper).items()
        if isinstance(value, torch.Tensor)
    }
    return row


def worker_workspace(worker):
    workspace = getattr(worker.vllm_config, "_compact_prefill_workspace", None)
    return None if workspace is None else workspace_receipt(workspace)


@contextmanager
def _observe_route_calls(destination, runner_type, workspace_type, label):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    original_execute = runner_type.execute_model
    original_prepare = runner_type.prepare_inputs
    original_run = workspace_type.run
    contexts = []

    def execute(runner, scheduler_output, *args, **kwargs):
        phase = label()
        batch_id = getattr(scheduler_output, "_knlp_batch_id", None)
        if phase is not None and batch_id is None:
            raise RuntimeError(
                "labeled worker batch missing scheduler identity"
            )
        contexts.append(dict(label=phase, batch_id=batch_id, request_ids=None))
        try:
            return original_execute(runner, scheduler_output, *args, **kwargs)
        finally:
            contexts.pop()

    def prepare(runner, scheduler_output, *args, **kwargs):
        batch = original_prepare(runner, scheduler_output, *args, **kwargs)
        if contexts:
            contexts[-1]["request_ids"] = list(batch.req_ids)
        return batch

    def run(workspace, query, cache, source_indices, page_plan, **kwargs):
        if not contexts or contexts[-1]["request_ids"] is None:
            raise RuntimeError(
                "actual worker batch missing from compact coverage"
            )
        row = dict(
            **contexts[-1],
            lengths=list(page_plan.query_lengths),
            pages=page_plan.pages,
            workspace_ptr=workspace.backing.data_ptr(),
            source_cache_pointer=cache[0].untyped_storage().data_ptr(),
        )
        with destination.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        return original_run(
            workspace, query, cache, source_indices, page_plan, **kwargs
        )

    runner_type.execute_model = execute
    runner_type.prepare_inputs = prepare
    workspace_type.run = run
    try:
        yield
    finally:
        runner_type.execute_model = original_execute
        runner_type.prepare_inputs = original_prepare
        workspace_type.run = original_run


@contextmanager
def observe_routes(destination):
    """Copy CPU batch identity at worker entry, without retaining model frames."""
    from vllm.v1.attention.backends.compact_prefill import (
        CompactPrefillWorkspace,
    )
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from . import schedule

    with _observe_route_calls(
        destination,
        GPUModelRunner,
        CompactPrefillWorkspace,
        lambda: schedule._label,
    ):
        yield


def validate_gather_fixture(fixture_path, destination):
    """Use stored quantized values in sparse, repeated, and page-zero layouts."""
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper
    from vllm.v1.attention.backends.compact_prefill import (
        CompactPrefillWorkspace,
    )
    from vllm.v1.attention.backends.compact_prefill_metadata import (
        plan_whole_prompt,
    )

    fixture_path, destination = Path(fixture_path), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    data = torch.load(fixture_path, map_location="cpu", weights_only=True)
    meta = json.loads(fixture_path.with_suffix(".json").read_text())
    assert (
        hashlib.sha256(fixture_path.read_bytes()).hexdigest() == meta["sha256"]
    )
    workspace = CompactPrefillWorkspace("cuda")
    mixed = BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(32 * 2**20, dtype=torch.uint8, device="cuda"),
        "NHD",
        backend="fa3",
    )
    # Both planes share one storage, with guards around regions and matching
    # padded page strides. The page map includes zero and high nonzero IDs.
    page_count, padded_tokens = 2053, 18
    plane_elements = page_count * padded_tokens * 8 * 128
    guard = 4096
    raw = torch.full(
        (plane_elements * 3 + guard * 3,),
        0x25,
        dtype=torch.uint8,
        device="cuda",
    )
    k_bytes = raw[guard : guard + 2 * plane_elements]
    v_start = 2 * guard + 2 * plane_elements
    v_bytes = raw[v_start : v_start + plane_elements]
    k = k_bytes.view(torch.float16).view(page_count, padded_tokens, 8, 128)[
        :, :16
    ]
    v = v_bytes.view(torch.float8_e4m3fn).view(
        page_count, padded_tokens, 8, 128
    )[:, :16]
    ids = list(range(1, page_count))
    random.Random(20261015).shuffle(ids)
    page_map = torch.tensor([0] + ids[:511], device="cuda", dtype=torch.long)
    k[page_map] = data["k"].to("cuda")
    v.view(torch.uint8)[page_map] = data["v"].view(torch.uint8).to("cuda")
    source_hash = tensor_hash(raw)
    rows = []
    cases = [
        [8191],
        [8191, 8187],
        [8192] * 8,
        [31 + i * 7 for i in range(32)],
        [33, 17],
    ]
    for case, lengths in enumerate(cases):
        qo = [0]
        for length in lengths:
            qo.append(qo[-1] + length)
        plan, reason = plan_whole_prompt(qo, lengths, max_sequences=96)
        assert reason == "candidate"
        source = torch.cat([page_map[: (n + 15) // 16] for n in lengths]).to(
            torch.int32
        )
        q = torch.cat([data["q"][:n] for n in lengths]).to("cuda")
        qptr = torch.tensor(qo, dtype=torch.int32)
        ptr = torch.tensor(plan.page_indptr, dtype=torch.int32)
        tails = torch.tensor(plan.last_page_lengths, dtype=torch.int32)
        workspace.k.fill_(-37)
        workspace.v.fill_(37)
        kwargs = dict(
            sm_scale=meta["sm_scale"], window_left=-1, logits_soft_cap=0.0
        )
        started = time.perf_counter()
        workspace.plan(plan, qptr, ptr, tails, **kwargs)
        mixed.plan(
            qo_indptr=qptr,
            paged_kv_indptr=ptr,
            paged_kv_indices=source,
            paged_kv_last_page_len=tails,
            num_qo_heads=32,
            num_kv_heads=8,
            head_dim_qk=128,
            page_size=16,
            causal=True,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
            k_data_type=torch.float16,
            v_data_type=torch.float8_e4m3fn,
            o_data_type=torch.float16,
            **kwargs,
        )
        actual, lse = workspace.run(
            q, (k, v), source, plan, v_scale=meta["scale"], return_lse=True
        )
        reference, reference_lse = mixed.run(
            q, (k, v), v_scale=meta["scale"], return_lse=True
        )
        index = source.long()
        expected_k = select_pages(k, index)
        expected_v = select_pages(v, index).to(torch.float16)
        k_exact = torch.equal(
            workspace.k[: plan.pages].view(torch.uint8),
            expected_k.view(torch.uint8),
        )
        v_exact = torch.equal(
            workspace.v[: plan.pages].view(torch.uint8),
            expected_v.view(torch.uint8),
        )
        untouched = bool(
            (workspace.k[plan.pages :] == -37).all()
            and (workspace.v[plan.pages :] == 37).all()
        )
        unchanged = tensor_hash(raw) == source_hash
        output_check, lse_check = full_comparison(
            actual, reference
        ), full_comparison(lse, reference_lse)
        row = dict(
            case=case,
            lengths=lengths,
            pages=plan.pages,
            includes_page_zero=bool((source == 0).any()),
            repeated_pages=source.unique().numel() < source.numel(),
            source_ids=source.cpu().tolist(),
            page_indptr=list(plan.page_indptr),
            tails=list(plan.last_page_lengths),
            k_exact=k_exact,
            v_exact=v_exact,
            unused_destination_unchanged=untouched,
            source_and_guards_unchanged=unchanged,
            output=output_check,
            lse=lse_check,
            workspace=workspace_receipt(workspace),
            setup_and_check_seconds=time.perf_counter() - started,
            source_cache=[tensor_info(k), tensor_info(v)],
        )
        rows.append(row)
        save(destination / f"case-{case}.json", row)
        if not all(
            [
                k_exact,
                v_exact,
                untouched,
                unchanged,
                output_check["passed"],
                lse_check["passed"],
            ]
        ):
            torch.save(
                dict(
                    q=q.cpu(),
                    k=expected_k.cpu(),
                    v=select_pages(v, index).cpu(),
                    actual=actual.cpu(),
                    reference=reference.cpu(),
                    source_ids=source.cpu(),
                ),
                destination / "failure.pt",
            )
            raise RuntimeError("real-layout compact gather check failed")
        del actual, reference, lse, reference_lse, expected_k, expected_v, q
    # Overlapping streams and host submissions are explicitly rejected.
    with torch.cuda.stream(torch.cuda.Stream()):
        try:
            with workspace.serial():
                pass
        except RuntimeError:
            stream_rejected = True
        else:
            raise RuntimeError("compact workspace accepted another stream")
    with workspace.serial():
        try:
            with workspace.serial():
                pass
        except RuntimeError:
            overlap_rejected = True
        else:
            raise RuntimeError(
                "compact workspace accepted overlapping submission"
            )
    save(
        destination / "summary.json",
        dict(
            passed=True,
            cases=len(rows),
            stream_rejected=stream_rejected,
            overlap_rejected=overlap_rejected,
            fixture_sha256=meta["sha256"],
            source_sha256=source_hash,
        ),
    )
    return rows


@contextmanager
def validate_model_layers(destination):
    """Check every routed layer in actual model forwards, outside timing."""
    from vllm.v1.attention.backends.compact_prefill import (
        CompactPrefillWorkspace,
    )
    from vllm.v1.attention.backends.flashinfer import FlashInferImpl

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    original_forward = FlashInferImpl.forward
    original_run = CompactPrefillWorkspace.run
    current = []
    rows = []

    def forward(
        impl,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        current.append((layer.layer_name, attn_metadata))
        try:
            return original_forward(
                impl,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )
        finally:
            current.pop()

    def run(workspace, query, cache, source_indices, page_plan, **kwargs):
        name, metadata = current[-1]
        before = workspace_receipt(workspace)
        result = original_run(
            workspace, query, cache, source_indices, page_plan, **kwargs
        )
        reference_kwargs = {k: v for k, v in kwargs.items() if k != "out"}
        reference, reference_lse = metadata.prefill.wrapper.run(
            query, cache, return_lse=True, **reference_kwargs
        )
        _, actual_lse = workspace.wrapper.run(
            query,
            (workspace.k[: page_plan.pages], workspace.v[: page_plan.pages]),
            return_lse=True,
            **reference_kwargs,
        )
        output_check = full_comparison(result, reference)
        lse_check = full_comparison(actual_lse, reference_lse)
        row = dict(
            layer=name,
            query_tokens=query.shape[0],
            lengths=list(page_plan.query_lengths),
            pages=page_plan.pages,
            tails=list(page_plan.last_page_lengths),
            output=output_check,
            lse=lse_check,
            output_sha256=tensor_hash(result),
            reference_sha256=tensor_hash(reference),
            before=before,
            after=workspace_receipt(workspace),
            source_cache=[tensor_info(t) for t in cache],
            source_cache_ptrs=[t.untyped_storage().data_ptr() for t in cache],
            v_scale=kwargs.get("v_scale"),
        )
        rows.append(row)
        with (destination / "layers.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if not output_check["passed"] or not lse_check["passed"]:
            indices = source_indices.long()
            torch.save(
                dict(
                    q=query.cpu(),
                    k=select_pages(cache[0], indices).cpu(),
                    v=select_pages(cache[1], indices).cpu(),
                    output=result.cpu(),
                    reference=reference.cpu(),
                    actual_lse=actual_lse.cpu(),
                    reference_lse=reference_lse.cpu(),
                    source_ids=indices.cpu(),
                ),
                destination / "failure.pt",
            )
            raise RuntimeError(
                "compact full-tensor layer comparison failed: " + name
            )
        return result

    FlashInferImpl.forward = forward
    CompactPrefillWorkspace.run = run
    try:
        yield rows
    finally:
        FlashInferImpl.forward = original_forward
        CompactPrefillWorkspace.run = original_run
        save(
            destination / "summary.json",
            dict(
                checked_calls=len(rows),
                layers=sorted({x["layer"] for x in rows}),
                passed=bool(rows)
                and all(
                    x["output"]["passed"] and x["lse"]["passed"] for x in rows
                ),
            ),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate_gather_fixture(args.fixture, args.output)

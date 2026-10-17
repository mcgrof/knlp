# SPDX-License-Identifier: MIT
"""Validate metadata routing from raw request block tables, outside timing.

This explicit mode checks equivalence to the accepted compact dispatch. Ideal
FP32 and mixed-kernel comparisons keep their original bounds as diagnostics.
"""

import inspect
from contextlib import contextmanager
from itertools import accumulate
from pathlib import Path

import torch

from .compact import full_comparison, tensor_hash, workspace_receipt
from .instrument import save


def raw_suffix(
    query_offsets, kv_lengths, block_table, *, decode_requests, page_size=16
):
    """Use original per-request metadata; accept no candidate slices or plans."""
    q, kv = list(map(int, query_offsets)), list(map(int, kv_lengths))
    assert len(q) == len(kv) + 1 and q[0] == 0
    assert 0 <= decode_requests < len(kv)
    lengths = [q[i + 1] - q[i] for i in range(decode_requests, len(kv))]
    assert lengths == kv[decode_requests:] and all(n > 0 for n in lengths)
    ids, counts = [], []
    for i in range(decode_requests, len(kv)):
        count = (kv[i] + page_size - 1) // page_size
        row = list(map(int, block_table[i][:count]))
        assert len(row) == count and min(row) >= 0
        ids.extend(row)
        counts.append(count)
    return {
        "lengths": lengths,
        "source_ids": ids,
        "query_offset": q[decode_requests],
        "query_indptr": [0, *accumulate(lengths)],
        "page_indptr": [0, *accumulate(counts)],
        "tails": [(n - 1) % page_size + 1 for n in lengths],
    }


def require_equal(actual, expected, what):
    if actual.shape != expected.shape or not torch.equal(actual, expected):
        raise RuntimeError("routing equivalence failed: " + what)
    if actual.is_floating_point() and not bool(torch.isfinite(actual.float()).all()):
        raise RuntimeError("nonfinite routing operand: " + what)


@contextmanager
def validate_routing_layers(destination):
    """Use an independent workspace and preserve the live plan identity."""
    from vllm.v1.attention.backends.compact_prefill import CompactPrefillWorkspace
    from vllm.v1.attention.backends.compact_prefill_metadata import CompactPagePlan
    from vllm.v1.attention.backends.flashinfer import (
        FlashInferImpl,
        FlashInferMetadataBuilder,
    )

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    old_build, old_forward, old_run = (
        FlashInferMetadataBuilder.build,
        FlashInferImpl.forward,
        CompactPrefillWorkspace.run,
    )
    accepted_run = inspect.unwrap(old_run)
    oracle = CompactPrefillWorkspace("cuda")
    current, rows = [], []

    def build(builder, common_prefix_len, common_attn_metadata, fast_build=False):
        raw = common_attn_metadata
        # Snapshot before the candidate performs any pointer/page slicing.
        n = raw.num_reqs
        snapshot = {
            "query_offsets": raw.query_start_loc_cpu[: n + 1].tolist(),
            "kv_lengths": raw.seq_lens[:n].cpu().tolist(),
            "block_table": raw.block_table_tensor[:n].cpu().tolist(),
        }
        result = old_build(builder, common_prefix_len, common_attn_metadata, fast_build)
        if result.num_prefill_tokens and result.prefill.compact is not None:
            fresh_start = next(
                i
                for i, n in enumerate(snapshot["kv_lengths"])
                if snapshot["query_offsets"][i + 1] - snapshot["query_offsets"][i] == n
                and n > 1
            )
            assert fresh_start == result.num_decodes
            expected = raw_suffix(**snapshot, decode_requests=fresh_start)
            assert expected["query_offset"] == result.num_decode_tokens
            plan = CompactPagePlan(
                tuple(expected["lengths"]),
                tuple(expected["page_indptr"]),
                tuple(expected["tails"]),
            )
            oracle.plan(
                plan,
                torch.tensor(expected["query_indptr"], dtype=torch.int32),
                torch.tensor(expected["page_indptr"], dtype=torch.int32),
                torch.tensor(expected["tails"], dtype=torch.int32),
                sm_scale=builder.sm_scale,
                window_left=builder.window_left,
                logits_soft_cap=builder.logits_soft_cap,
                fixed_split_size=builder.prefill_fixed_split_size,
                disable_split_kv=builder.disable_split_kv,
            )
            result._routing_oracle = (snapshot, expected, plan)
        return result

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
        before_rows = len(rows)
        active = attn_metadata is not None and hasattr(attn_metadata, "_routing_oracle")
        if active:
            n = attn_metadata.num_actual_tokens
            current.append((layer.layer_name, attn_metadata, query[:n], output[:n]))
        try:
            result = old_forward(
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
            if active:
                nd = attn_metadata.num_decode_tokens
                if nd:
                    # Writer and decode operation are unchanged. Repeat identical
                    # dispatch on actual inputs; compare the complete output prefix.
                    scale = {
                        "q_scale": layer._q_scale_float,
                        "k_scale": layer._k_scale_float,
                        "v_scale": layer._v_scale_float,
                        "kv_cache_sf": None,
                    }
                    dq = impl.maybe_quant_query(
                        query[:nd], attn_metadata.q_data_type_decode, layer._q_scale
                    )
                    from vllm.v1.attention.backends.flashinfer import (
                        _prepare_flashinfer_paged_kv_cache,
                    )

                    cache = _prepare_flashinfer_paged_kv_cache(kv_cache)
                    expected = attn_metadata.decode.wrapper.run(dq, cache, **scale)
                    repeat = attn_metadata.decode.wrapper.run(dq, cache, **scale)
                    require_equal(expected, repeat, "decode repeatability")
                    require_equal(output[:nd], expected, "decode output placement")
                rows[-1]["decode_tokens_checked"] = nd
                rows[-1]["forward_placement_verified"] = True
            return result
        except BaseException:
            if active:
                torch.save(
                    {"query": query.cpu(), "output": output.cpu()},
                    destination / "forward-failure.pt",
                )
            raise
        finally:
            if active:
                current.pop()
                _save_forward_receipt(destination, rows, before_rows)

    def run(workspace, query, cache, source_indices, page_plan, **kwargs):
        name, metadata, full_query, full_output = current[-1]
        snapshot, expected, trusted_plan = metadata._routing_oracle
        ids = torch.tensor(
            expected["source_ids"], dtype=torch.int32, device=query.device
        )
        original_identity = workspace.last_plan
        row = {
            "layer": name,
            "raw": snapshot,
            "expected": expected,
            "before": workspace_receipt(workspace),
            "routing_equivalence_passed": False,
            "forward_placement_verified": False,
        }
        try:
            require_equal(source_indices, ids, "raw block-table page mapping")
            assert list(page_plan.query_lengths) == expected["lengths"]
            assert list(page_plan.page_indptr) == expected["page_indptr"]
            assert list(page_plan.last_page_lengths) == expected["tails"]
            offset = expected["query_offset"]
            require_placement(query, kwargs["out"], full_query, full_output, offset)
            args = {k: v for k, v in kwargs.items() if k != "out"}
            source_before = [
                x.view(torch.uint8).index_select(0, ids.long()) for x in cache
            ]
            result = old_run(
                workspace, query, cache, source_indices, page_plan, **kwargs
            )
            # Independently select original bytes with torch, then check the
            # production gather's K bytes and FP8->FP16 cast before scaling.
            for side in (0, 1):
                require_equal(
                    cache[side].view(torch.uint8).index_select(0, ids.long()),
                    source_before[side],
                    "source preservation",
                )
                expected_plane = source_before[side].view(cache[side].dtype)
                gathered = (workspace.k, workspace.v)[side][: len(ids)]
                expected_plane = expected_plane if side == 0 else expected_plane.half()
                require_equal(
                    gathered.view(torch.uint8),
                    expected_plane.contiguous().view(torch.uint8),
                    "K bytes" if side == 0 else "V cast",
                )
                del expected_plane
            del source_before
            trusted = accepted_run(oracle, query, cache, ids, trusted_plan, **args)
            repeat = accepted_run(oracle, query, cache, ids, trusted_plan, **args)
            require_equal(trusted, repeat, "prefill repeatability")
            _, trusted_lse = oracle.wrapper.run(
                query,
                (oracle.k[: len(ids)], oracle.v[: len(ids)]),
                return_lse=True,
                **args,
            )
            _, repeat_lse = oracle.wrapper.run(
                query,
                (oracle.k[: len(ids)], oracle.v[: len(ids)]),
                return_lse=True,
                **args,
            )
            _, actual_lse = workspace.wrapper.run(
                query,
                (workspace.k[: len(ids)], workspace.v[: len(ids)]),
                return_lse=True,
                **args,
            )
            require_equal(trusted_lse, repeat_lse, "LSE repeatability")
            require_equal(result, trusted, "prefill output")
            require_equal(actual_lse, trusted_lse, "prefill LSE")
            require_equal(full_output[offset:], result, "prefill output suffix")
            assert workspace.last_plan is original_identity is page_plan
            fallback, fallback_lse = metadata.prefill.wrapper.run(
                query, cache, return_lse=True, **args
            )
            from .stored_prefill import check

            # The candidate plan is intentionally replaced by the independently
            # reconstructed trusted plan/IDs in this diagnostic input selection.
            diagnostic = check(
                oracle,
                query,
                cache,
                ids,
                trusted_plan,
                result,
                actual_lse,
                kwargs,
                destination / f"stored-{len(rows):04d}.pt",
            )
            row.update(
                routing_equivalence_passed=True,
                finite=True,
                output_sha256=tensor_hash(result),
                trusted_sha256=tensor_hash(trusted),
                lse_sha256=tensor_hash(actual_lse),
                trusted_lse_sha256=tensor_hash(trusted_lse),
                ideal_fp32_diagnostic=diagnostic,
                trusted_ideal_fp32_diagnostic=diagnostic,
                fallback_output_diagnostic=full_comparison(result, fallback),
                fallback_lse_diagnostic=full_comparison(actual_lse, fallback_lse),
                after=workspace_receipt(workspace),
                query_tokens=len(query),
                pages=len(ids),
                lengths=expected["lengths"],
                source_cache_ptrs=[x.untyped_storage().data_ptr() for x in cache],
            )
        except BaseException as exc:
            row["error"] = repr(exc)
            torch.save(
                {
                    "query": query.cpu(),
                    "k": cache[0].index_select(0, ids.long()).cpu(),
                    "v": cache[1]
                    .view(torch.uint8)
                    .index_select(0, ids.long())
                    .view(cache[1].dtype)
                    .cpu(),
                    "source_indices": source_indices.cpu(),
                    "raw": snapshot,
                    "output": kwargs["out"].cpu(),
                },
                destination / "failure.pt",
            )
            raise
        finally:
            rows.append(row)
        return result

    (
        FlashInferMetadataBuilder.build,
        FlashInferImpl.forward,
        CompactPrefillWorkspace.run,
    ) = (build, forward, run)
    try:
        yield rows
    finally:
        (
            FlashInferMetadataBuilder.build,
            FlashInferImpl.forward,
            CompactPrefillWorkspace.run,
        ) = (old_build, old_forward, old_run)
        save(
            destination / "summary.json",
            {
                "checked_calls": len(rows),
                "layers": sorted({x["layer"] for x in rows}),
                "passed": bool(rows)
                and all(
                    x["routing_equivalence_passed"] and x["forward_placement_verified"]
                    for x in rows
                ),
                "contract": "raw-block-table and identical accepted dispatch; ideal FP32/fallback diagnostics retain original bounds",
            },
        )
        oracle = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def require_placement(query, output, full_query, full_output, offset):
    if (
        query.data_ptr() != full_query[offset:].data_ptr()
        or query.shape != full_query[offset:].shape
        or output.data_ptr() != full_output[offset:].data_ptr()
        or output.shape != full_output[offset:].shape
    ):
        raise RuntimeError("routing equivalence failed: output suffix placement")


def validate_routing_fixture(destination):
    """Exercise independent maps, production gather, guards and four faults."""
    from vllm.v1.attention.backends.compact_prefill import CompactPrefillWorkspace
    from vllm.v1.attention.backends.compact_prefill_metadata import (
        CompactPagePlan,
        plan_prefill_subset,
    )

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    workspace, oracle = CompactPrefillWorkspace("cuda"), CompactPrefillWorkspace("cuda")
    pos = torch.arange(8 * 16 * 8 * 128, device="cuda", dtype=torch.float32).reshape(
        8, 16, 8, 128
    )
    cache = (
        torch.sin(pos / 83).half(),
        (16 * torch.cos(pos / 31)).to(torch.float8_e4m3fn),
    )
    q = (
        torch.sin(torch.arange(52 * 32 * 128, device="cuda", dtype=torch.float32) / 71)
        .reshape(52, 32, 128)
        .half()
    )
    checks, faults = [], []
    previous_ids = None
    for replay, table in enumerate(
        ([[7, 6, 0], [3, 0, 2], [5, 3, 0]], [[6, 7, 0], [2, 3, 0], [0, 5, 0]])
    ):
        raw = {
            "query_offsets": [0, 2, 35, 52],
            "kv_lengths": [17, 33, 17],
            "block_table": table,
        }
        expected = raw_suffix(**raw, decode_requests=1)
        ids = torch.tensor(expected["source_ids"], dtype=torch.int32, device="cuda")
        source_ids = torch.tensor(
            [x for row, n in zip(table, [2, 3, 2]) for x in row[:n]],
            dtype=torch.int32,
            device="cuda",
        )
        plan, reason = plan_prefill_subset(
            raw["query_offsets"],
            raw["kv_lengths"],
            [0, 2, 5, 7],
            max_sequences=96,
            num_decodes=1,
        )
        assert reason == "candidate"
        trusted = CompactPagePlan(
            tuple(expected["lengths"]),
            tuple(expected["page_indptr"]),
            tuple(expected["tails"]),
        )
        for work, pp in [(workspace, plan), (oracle, trusted)]:
            work.plan(
                pp,
                torch.tensor(expected["query_indptr"], dtype=torch.int32),
                torch.tensor(expected["page_indptr"], dtype=torch.int32),
                torch.tensor(expected["tails"], dtype=torch.int32),
                sm_scale=128**-0.5,
                window_left=-1,
                logits_soft_cap=0.0,
                fixed_split_size=None,
                disable_split_kv=False,
            )
        candidate_ids = source_ids[plan.source_page_start : plan.source_page_end]
        require_equal(candidate_ids, ids, "fixture independent mapping")
        out = torch.full_like(q, -91)
        guard = out[:2].clone()
        before = [x.view(torch.uint8).clone() for x in cache]
        result = workspace.run(
            q[2:], cache, candidate_ids, plan, out=out[2:], k_scale=1.0, v_scale=1 / 64
        )
        ref = oracle.run(q[2:], cache, ids, trusted, k_scale=1.0, v_scale=1 / 64)
        repeat = oracle.run(q[2:], cache, ids, trusted, k_scale=1.0, v_scale=1 / 64)
        require_equal(ref, repeat, "fixture repeatability")
        require_equal(result, ref, "fixture output")
        require_equal(out[:2], guard, "decode guard")
        require_placement(q[2:], out[2:], q, out, expected["query_offset"])
        for side in (0, 1):
            require_equal(cache[side].view(torch.uint8), before[side], "source guard")
            gathered = (workspace.k, workspace.v)[side][: len(ids)]
            independent = (
                cache[side]
                .view(torch.uint8)
                .index_select(0, ids.long())
                .view(cache[side].dtype)
                .half()
            )
            require_equal(
                gathered.view(torch.uint8),
                independent.view(torch.uint8),
                "fixture K/V gather",
            )
        wrongs = {
            "omit_source_offset": source_ids[: len(ids)],
            "decode_page_substitution": torch.cat([source_ids[:1], ids[1:]]),
        }
        if replay:
            assert previous_ids is not None
            wrongs["stale_mapping"] = previous_ids
        for name, wrong in wrongs.items():
            try:
                require_equal(wrong, ids, name)
            except RuntimeError:
                faults.append({"replay": replay, "fault": name, "detected": True})
            else:
                raise RuntimeError("injected fault not detected: " + name)
        try:
            require_placement(q[2:], out[1:-1], q, out, 2)
        except RuntimeError:
            faults.append(
                {"replay": replay, "fault": "shift_output_suffix", "detected": True}
            )
        else:
            raise RuntimeError("shifted suffix not detected")
        _, lse = workspace.wrapper.run(
            q[2:],
            (workspace.k[: len(ids)], workspace.v[: len(ids)]),
            return_lse=True,
            k_scale=1.0,
            v_scale=1 / 64,
        )
        _, rlse = oracle.wrapper.run(
            q[2:],
            (oracle.k[: len(ids)], oracle.v[: len(ids)]),
            return_lse=True,
            k_scale=1.0,
            v_scale=1 / 64,
        )
        require_equal(lse, rlse, "fixture LSE")
        torch.save(
            {
                "raw": raw,
                "source_ids": candidate_ids.cpu(),
                "query": q.cpu(),
                "key": cache[0].cpu(),
                "value": cache[1].cpu(),
                "output": out.cpu(),
                "reference": ref.cpu(),
                "lse": lse.cpu(),
                "reference_lse": rlse.cpu(),
            },
            destination / f"replay-{replay}.pt",
        )
        checks.append(
            {
                "replay": replay,
                "expected": expected,
                "output_sha256": tensor_hash(result),
                "reference_sha256": tensor_hash(ref),
                "passed": True,
            }
        )
        previous_ids = ids.clone()
    assert {x["fault"] for x in faults} == {
        "omit_source_offset",
        "decode_page_substitution",
        "shift_output_suffix",
        "stale_mapping",
    }
    result = {"passed": True, "checks": checks, "faults": faults}
    save(destination / "RESULT.json", result)
    return result


def _save_forward_receipt(destination, rows, before):
    """Create a call receipt once, after attention and output placement finish."""
    if len(rows) > before:
        if len(rows) != before + 1:
            raise RuntimeError("one attention call per layer forward is required")
        save(Path(destination) / f"call-{before:04d}.json", rows[-1])

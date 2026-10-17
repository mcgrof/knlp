# SPDX-License-Identifier: MIT
"""Check a mixed D64 loader separately from model quality and serving speed."""

import argparse
import json
from pathlib import Path

import torch

from .geometry import qualify
from .instrument import save
from .qualification import record_attention_check


def reference(query, key, value, *, causal):
    """Compute attention over the actual stored values in FP32."""
    groups = query.shape[1] // key.shape[1]
    keys = key.float().repeat_interleave(groups, dim=1)
    values = value.float().repeat_interleave(groups, dim=1)
    logits = torch.einsum("qhd,thd->hqt", query.float(), keys) / query.shape[-1] ** 0.5
    if causal:
        allowed = torch.arange(key.shape[0], device=key.device)[None, :] <= (
            key.shape[0]
            - query.shape[0]
            + torch.arange(query.shape[0], device=key.device)[:, None]
        )
        logits.masked_fill_(~allowed[None, :, :], -torch.inf)
    return torch.einsum("hqt,thd->qhd", logits.softmax(-1), values)


def populate_cache(k, v, slots, positions, vscale, replay):
    key = torch.sin((positions + 7 * replay) / 91).half()
    value = (torch.cos((positions + 11 * replay) / 79) / 4).half()
    encoded = (value.float() / vscale).to(v.dtype)
    k.flatten(0, 1)[slots] = key
    # Advanced indexing assignment for float8 is not supported by every torch build.
    v.view(torch.uint8).flatten(0, 1)[slots] = encoded.view(torch.uint8)
    return key, encoded.float() * vscale


def grid(output, *, phase):
    import flashinfer

    cases = [(64, False, 14, 2), (128, True, 32, 8)]
    if phase == "candidate":
        cases.insert(0, (64, True, 14, 2))
    receipts = []
    for dim, mixed, queries, heads in cases:
        lengths = [1, 16, 17, 32, 33, 47, 65] if mixed and dim == 64 else [17, 33, 65]
        for length in lengths:
            pages = (length + 15) // 16
            mapping = list(range(pages + 1))
            mapping = mapping[1::2] + mapping[::2]
            mapping = mapping[:pages]
            if length == 33:
                mapping = [3, 0, 2]
            count = max(mapping) + 1
            shape = (count, 16, heads, dim)
            kbytes = count * 16 * heads * dim * 2
            vbytes = count * 16 * heads * dim * (1 if mixed else 2)
            storage = torch.zeros(kbytes + vbytes, dtype=torch.uint8, device="cuda")
            k = storage[:kbytes].view(torch.float16).view(shape)
            v = (
                storage[kbytes:]
                .view(torch.float8_e4m3fn if mixed else torch.float16)
                .view(shape)
            )
            slots = torch.tensor(
                [mapping[i // 16] * 16 + i % 16 for i in range(length)], device="cuda"
            )
            indptr = torch.tensor([0, pages], dtype=torch.int32, device="cuda")
            indices = torch.tensor(mapping, dtype=torch.int32, device="cuda")
            tails = torch.tensor(
                [(length - 1) % 16 + 1], dtype=torch.int32, device="cuda"
            )
            vscale = 1 / 64 if mixed else 1.0
            positions = torch.arange(
                length * heads * dim, dtype=torch.float32, device="cuda"
            ).reshape(length, heads, dim)

            key, value = populate_cache(k, v, slots, positions, vscale, 0)
            q = (
                torch.sin(
                    torch.arange(queries * dim, dtype=torch.float32, device="cuda") / 23
                )
                .reshape(1, queries, dim)
                .half()
            )
            work = torch.empty(128 * 2**20, dtype=torch.uint8, device="cuda")
            graph_case = length == 33
            wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                work,
                "NHD",
                use_tensor_cores=True,
                backend="fa2",
                use_cuda_graph=graph_case,
                paged_kv_indptr_buffer=indptr if graph_case else None,
                paged_kv_indices_buffer=indices if graph_case else None,
                paged_kv_last_page_len_buffer=tails if graph_case else None,
            )
            wrapper.plan(
                indptr,
                indices,
                tails,
                queries,
                heads,
                dim,
                16,
                q_data_type=torch.float16,
                k_data_type=torch.float16,
                v_data_type=v.dtype,
            )
            result = wrapper.run(q, (k, v), k_scale=1.0, v_scale=vscale)
            torch.cuda.synchronize()
            config = {
                "phase": phase,
                "head_dim": dim,
                "query_heads": queries,
                "kv_heads": heads,
                "stored_tokens": length,
                "k_scale": 1.0,
                "v_scale": vscale,
                "mixed": mixed,
                "requested_backend": "fa2",
                "page_size": 16,
                "graph": False,
            }
            name = f"D{dim}-{'mixed' if mixed else 'symmetric'}-L{length}"
            tensors = {
                "query": q,
                "page_indptr": indptr,
                "page_indices": indices,
                "last_page_len": tails,
            }
            receipts.append(
                record_attention_check(
                    output,
                    name + "-decode",
                    wrapper=wrapper,
                    result=result,
                    reference=reference(q, key, value, causal=False),
                    tensors=tensors,
                    cache=(k, v),
                    configuration=config,
                )
            )
            prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                work, "NHD", backend="fa2"
            )
            nq = min(3, length)
            qo = torch.tensor([0, nq], dtype=torch.int32, device="cuda")
            prefill.plan(
                qo,
                indptr,
                indices,
                tails,
                queries,
                heads,
                dim,
                16,
                causal=True,
                q_data_type=torch.float16,
                k_data_type=torch.float16,
                v_data_type=v.dtype,
            )
            pq = (
                torch.sin(
                    torch.arange(nq * queries * dim, dtype=torch.float32, device="cuda")
                    / 23
                )
                .reshape(nq, queries, dim)
                .half()
            )
            result = prefill.run(pq, (k, v), k_scale=1.0, v_scale=vscale)
            receipts.append(
                record_attention_check(
                    output,
                    name + "-prefill",
                    wrapper=prefill,
                    result=result,
                    reference=reference(pq, key, value, causal=True),
                    tensors={**tensors, "query": pq, "query_indptr": qo},
                    cache=(k, v),
                    configuration={**config, "causal": True},
                )
            )
            if graph_case:
                # Keep all buffers alive and change both the query and stored payload between replays.
                wrapper.run(q, (k, v), k_scale=1.0, v_scale=vscale)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = wrapper.run(q, (k, v), k_scale=1.0, v_scale=vscale)
                for replay in range(1, 4):
                    key, value = populate_cache(k, v, slots, positions, vscale, replay)
                    q.copy_(
                        torch.sin(
                            torch.arange(
                                queries * dim, dtype=torch.float32, device="cuda"
                            )
                            / 23
                            + replay / 5
                        )
                        .reshape_as(q)
                        .half()
                    )
                    graph.replay()
                    torch.cuda.synchronize()
                    receipts.append(
                        record_attention_check(
                            output,
                            name + f"-graph-{replay}",
                            wrapper=wrapper,
                            result=captured,
                            reference=reference(q, key, value, causal=False),
                            tensors={**tensors, "query": q},
                            cache=(k, v),
                            configuration={**config, "graph": True, "replay": replay},
                        )
                    )
                del graph, captured
            del wrapper, prefill, work, storage, k, v
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["original", "candidate"], required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(22)
    if args.phase == "original":
        try:
            qualify(
                args.stack,
                args.output / "original-fixture",
                geometry={
                    "layers": 24,
                    "query_heads": 14,
                    "kv_heads": 2,
                    "head_dim": 64,
                },
            )
        except RuntimeError:
            path = args.output / "original-fixture/geometry-decode/check.json"
            if not path.exists():
                raise
            failure = json.loads(path.read_text())
            if failure["passed"] or failure["backend"] != "fa2":
                raise
            save(args.output / "expected-original-failure.json", failure)
        else:
            raise RuntimeError(
                "original D64 failure did not reproduce; preserve evidence and stop"
            )
    else:
        qualify(
            args.stack,
            args.output / "repaired-fixture",
            geometry={"layers": 24, "query_heads": 14, "kv_heads": 2, "head_dim": 64},
        )
    qualify(
        args.stack,
        args.output / "D128-control",
        geometry={"layers": 36, "query_heads": 32, "kv_heads": 8, "head_dim": 128},
    )
    receipts = grid(args.output / "grid", phase=args.phase)
    save(
        args.output / "RESULT.json",
        {
            "phase": args.phase,
            "passed": True,
            "grid_checks": len(receipts),
            "original_failure_expected": args.phase == "original",
            "all_grid_checks_passed": all(r["passed"] for r in receipts),
        },
    )


if __name__ == "__main__":
    main()

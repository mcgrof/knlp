# SPDX-License-Identifier: Apache-2.0
"""No-GPU regression for the frozen MRv2 kernel-warmup admission."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch

VLLM_SOURCE = Path(os.environ.get("TOKENDANCE_VLLM_SOURCE", "/sources/vllm"))
INPUT_SOURCE = Path(__file__).parent
for source in (INPUT_SOURCE, VLLM_SOURCE):
    sys.path.insert(0, str(source))

from numerical_instrumentation import (  # noqa: E402
    NumericalCaptureError,
    NumericalInstrumentation,
    REQUEST_ID_CONTRACT as NUMERICAL_REQUEST_ID_CONTRACT,
    _atomic_safetensors,
    _internal_request_id_matches,
    _prompt_digest,
)
from run_online_collective_lockstep_gate import (  # noqa: E402
    GateError,
    REQUEST_ID_CONTRACT as RUNNER_REQUEST_ID_CONTRACT,
    _bind_internal_request_id,
    _canonical_projection_events as _runner_canonical_projection_events,
    _domain_aware_layer_call_details,
    _evaluate as _runner_evaluate,
    _expected_projection_phases as _runner_expected_projection_phases,
    _external_trace_bindings,
    _recovery_evidence,
)
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.v1.core.sched.output import (  # noqa: E402
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.worker.gpu.kv_connector import (  # noqa: E402
    NO_OP_KV_CONNECTOR,
    ActiveKVConnector,
)
from verify_online_collective_lockstep_gate import (  # noqa: E402
    REQUEST_ID_CONTRACT as VERIFIER_REQUEST_ID_CONTRACT,
)
from verify_online_collective_lockstep_gate import (  # noqa: E402
    _load as _independent_load,
)
from verify_online_collective_lockstep_gate import (  # noqa: E402
    _canonical_projection_events as _verifier_canonical_projection_events,
)
from verify_online_collective_lockstep_gate import (  # noqa: E402
    _expected_projection_phases as _verifier_expected_projection_phases,
)
from verify_online_collective_lockstep_gate import (  # noqa: E402
    ARMS,
    _build_claims,
    _compare_numerical_families,
    _semantic_group,
    _verify_arm_claims,
    _verify_cross_arm_output_parity,
    _verify_event_id_domains,
    _verify_layer_operation_claims,
    _verify_lookup_claims,
    _verify_numerical_event_bindings,
    _verify_recovery_ledger,
    _verify_sibling_failure_lookup_claim,
)


class _Ledger:
    phase = "construction"

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def emit(self, event: str, **payload: Any) -> None:
        self.events.append((event, payload))


def _instrumentation() -> NumericalInstrumentation:
    value = NumericalInstrumentation.__new__(NumericalInstrumentation)
    value.ledger = _Ledger()
    value._lock = threading.RLock()
    value._kernel_warmup_seen = False
    value._engine_to_sequence = {}
    value._sequence_to_engine = {}
    value._captured = {}
    value._records_by_digest = {}
    return value


def _runner(connector: object = NO_OP_KV_CONNECTOR) -> SimpleNamespace:
    return SimpleNamespace(
        decode_query_len=1,
        scheduler_config=SimpleNamespace(
            max_num_seqs=4,
            max_num_batched_tokens=2048,
        ),
        kv_connector=connector,
    )


def _warmup_output() -> SchedulerOutput:
    request_ids = tuple(f"_warmup_{index}_" for index in range(4))
    output = SchedulerOutput.make_empty()
    output.scheduled_new_reqs = [
        NewRequestData(
            req_id=request_id,
            prompt_token_ids=[0, 1],
            prefill_token_ids=[0, 1],
            mm_features=[],
            sampling_params=SamplingParams.for_sampler_warmup(),
            pooling_params=None,
            block_ids=([index],),
            num_computed_tokens=0,
            lora_request=None,
            prompt_embeds=None,
            prompt_is_token_ids=None,
        )
        for index, request_id in enumerate(request_ids, start=1)
    ]
    output.num_scheduled_tokens = {request_id: 2 for request_id in request_ids}
    output.total_num_scheduled_tokens = 8
    output.num_common_prefix_blocks = [0]
    return output


def _is_admitted(
    output: SchedulerOutput,
    *,
    instrumentation: NumericalInstrumentation | None = None,
    runner: SimpleNamespace | None = None,
) -> bool:
    value = instrumentation or _instrumentation()
    active_runner = runner or _runner()
    return value._is_exact_kernel_warmup_batch(
        active_runner,
        output,
        output.scheduled_new_reqs,
    )


def _assert_rejected(output: SchedulerOutput, label: str) -> None:
    if _is_admitted(output):
        raise AssertionError(f"warmup mutation was admitted: {label}")


def _mutated(
    mutate: Callable[[SchedulerOutput], None],
) -> SchedulerOutput:
    output = _warmup_output()
    mutate(output)
    return output


def _expect_gate_error(call: Callable[[], object], label: str) -> None:
    try:
        call()
    except GateError:
        return
    raise AssertionError(f"request-ID drift was admitted: {label}")


def _verify_request_id_contract() -> dict[str, int]:
    if not (
        RUNNER_REQUEST_ID_CONTRACT
        == NUMERICAL_REQUEST_ID_CONTRACT
        == VERIFIER_REQUEST_ID_CONTRACT
    ):
        raise AssertionError("runner/capture/verifier request-ID contracts differ")
    observations = [
        {
            "external_request_id": str(index),
            "trace_request_id": f"trace-{index}",
            "phase": "prime" if index == 0 else "collective_forward",
        }
        for index in range(5)
    ]
    external_to_trace = _external_trace_bindings(observations)
    internal_ids = {
        str(index): f"{index}-{index:08x}" for index in range(len(observations))
    }
    canonical_events = [
        {
            "action": "LMCacheConnectorV1Impl.prepare_request_lookup.begin",
            "phase": observation["phase"],
            "request_id": internal_ids[observation["external_request_id"]],
            "owner_type": "LMCacheConnectorV1Impl",
            "role": "SCHEDULER",
        }
        for observation in observations
    ]
    for external, internal in internal_ids.items():
        if _bind_internal_request_id(internal, external_to_trace) != (
            external,
            f"trace-{external}",
        ):
            raise AssertionError("runner request-ID binding changed")
        if not _internal_request_id_matches(internal, external):
            raise AssertionError("numerical request-ID binding changed")

    invalid_pairs: tuple[tuple[object, object], ...] = (
        ("1-0000000", "1"),
        ("1-000000000", "1"),
        ("1-0000000A", "1"),
        ("1-0000000g", "1"),
        ("1_00000000", "1"),
        ("2-00000000", "1"),
        (True, "1"),
        ("1-00000000", True),
    )
    for internal, external in invalid_pairs:
        if _internal_request_id_matches(internal, external):
            raise AssertionError("numerical request-ID lookalike was admitted")
    invalid_internals = [invalid_pairs[index][0] for index in (0, 1, 2, 3, 4, 6)]
    for internal in invalid_internals:
        _expect_gate_error(
            lambda internal=internal: _bind_internal_request_id(
                internal,
                external_to_trace,
            ),
            repr((internal, external)),
        )

    observation_mutations: list[Callable[[list[dict[str, str]]], None]] = [
        lambda values: values[0].update(external_request_id="1"),
        lambda values: values[0].update(external_request_id=True),
        lambda values: values[0].update(trace_request_id="trace-1"),
        lambda values: values[0].pop("external_request_id"),
        lambda values: values.reverse(),
    ]
    for index, mutate in enumerate(observation_mutations):
        changed = copy.deepcopy(observations)
        mutate(changed)
        _expect_gate_error(
            lambda changed=changed: _external_trace_bindings(changed),
            f"observation mutation {index}",
        )

    probe_ids = [internal_ids[str(index)] for index in range(1, 5)]
    plan_members = [
        {
            "request_id": request_id,
            "generation": 1,
            "explicit_spans": {"span": [index]},
        }
        for index, request_id in enumerate(probe_ids, start=1)
    ]
    projection = {
        "action": "gate.collective.plan_projection",
        "phase": "collective_forward",
        "sequence": 200,
        "plans": [
            {
                "plan_id": "plan",
                "reuse_group_id": "group",
                "members": copy.deepcopy(plan_members),
            }
        ],
    }
    prime_projection = {
        "action": "gate.collective.plan_projection",
        "phase": "prime",
        "plans": [],
        "sequence": 100,
    }
    prime_bind_projection = copy.deepcopy(prime_projection)
    prime_bind_projection["sequence"] = 101
    bind_projection = copy.deepcopy(projection)
    bind_projection["sequence"] = 201
    projection_receipts = [
        prime_projection,
        prime_bind_projection,
        projection,
        bind_projection,
    ]
    expected_projection_phases = ("prime", "collective_forward")
    if (
        _runner_expected_projection_phases("collective_forward")
        != expected_projection_phases
        or _verifier_expected_projection_phases("collective_forward")
        != expected_projection_phases
        or [
            event["sequence"]
            for event in _runner_canonical_projection_events(
                projection_receipts,
                expected_projection_phases,
            )
        ]
        != [101, 201]
        or [
            event["sequence"]
            for event in _verifier_canonical_projection_events(
                Path("projection-bind-selection"),
                projection_receipts,
                expected_projection_phases,
            )
        ]
        != [101, 201]
    ):
        raise AssertionError("projection receipt canonicalization did not select bind")
    blend = {
        "action": "gate.collective.blend_group.end",
        "phase": "collective_forward",
        "num_layers": 24,
        "input_members": copy.deepcopy(plan_members),
        "result_members": [
            {
                "request_id": member["request_id"],
                "generation": 1,
                "outcome": "recovered",
                "retrieval_mask": {"true_count": 1},
                "operation_counts": {},
                "error": None,
            }
            for member in plan_members
        ],
    }
    recovery = _recovery_evidence(
        observations,
        [*canonical_events, *projection_receipts, blend],
    )
    recovered_members = recovery["groups"][0]["members"]
    if [member["external_request_id"] for member in recovered_members] != [
        "1",
        "2",
        "3",
        "4",
    ] or [member["trace_request_id"] for member in recovered_members] != [
        "trace-1",
        "trace-2",
        "trace-3",
        "trace-4",
    ]:
        raise AssertionError("collective recovery lost an ID-domain binding")

    missing_projection_receipt = copy.deepcopy(projection_receipts)
    missing_projection_receipt.pop()
    _expect_gate_error(
        lambda: _recovery_evidence(
            observations,
            [*canonical_events, *missing_projection_receipt, blend],
        ),
        "missing projected-plan bind receipt",
    )

    divergent_projection_receipts = copy.deepcopy(projection_receipts)
    divergent_projection_receipts[-1]["plans"][0]["plan_id"] = "divergent-plan"
    _expect_gate_error(
        lambda: _recovery_evidence(
            observations,
            [
                *canonical_events,
                *divergent_projection_receipts,
                blend,
            ],
        ),
        "divergent projected-plan replay",
    )
    _expect_gate_error(
        lambda: _recovery_evidence(
            observations,
            [
                *canonical_events,
                *projection_receipts,
                {**copy.deepcopy(projection), "sequence": 202},
                blend,
            ],
        ),
        "third projected-plan event",
    )

    serial_events = [
        {
            "action": "gate.serial_blend.evidence",
            "phase": "collective_forward",
            "request_id": request_id,
            "explicit_spans": {"span": [index]},
            "retrieval_mask": {"true_count": 1},
        }
        for index, request_id in enumerate(probe_ids, start=1)
    ]
    serial = _recovery_evidence(
        observations,
        [*canonical_events, *projection_receipts, *serial_events],
    )["serial_members"]
    if [member["external_request_id"] for member in serial] != ["1", "2", "3", "4"]:
        raise AssertionError("serial recovery lost an ID-domain binding")
    serial_observations = copy.deepcopy(observations)
    serial_events_without_recovery = copy.deepcopy(canonical_events)
    for observation, event in zip(
        serial_observations,
        serial_events_without_recovery,
        strict=True,
    ):
        if observation["phase"] != "prime":
            observation["phase"] = "serial_oracle"
            event["phase"] = "serial_oracle"
    empty = _recovery_evidence(
        serial_observations,
        serial_events_without_recovery,
    )
    if empty["groups"] or empty["serial_members"]:
        raise AssertionError("empty recovery fabricated request bindings")

    invalid_projection = copy.deepcopy(projection)
    invalid_projection["plans"][0]["members"][0]["request_id"] = "1-0000000A"
    invalid_bind_projection = copy.deepcopy(invalid_projection)
    invalid_bind_projection["sequence"] = 201
    _expect_gate_error(
        lambda: _recovery_evidence(
            observations,
            [
                *canonical_events,
                prime_projection,
                prime_bind_projection,
                invalid_projection,
                invalid_bind_projection,
            ],
        ),
        "uppercase projected nonce",
    )
    return {
        "request_id_contract_copies_matched": 3,
        "exact_request_id_pairs_admitted": len(internal_ids),
        "request_id_lookalikes_rejected": len(invalid_pairs),
        "observation_mutations_rejected": len(observation_mutations),
        "collective_members_bound": len(recovered_members),
        "serial_members_bound": len(serial),
        "runner_projection_receipt_pairs_admitted": 2,
        "runner_projection_receipt_mutations_rejected": 3,
    }


def _verify_finish_id_domains() -> dict[str, int]:
    source = {"synthetic": "request-id-contract"}
    instrumentation = NumericalInstrumentation.__new__(NumericalInstrumentation)
    instrumentation.arm = "ordinary"
    instrumentation.phase = "ordinary_batch_forward"
    instrumentation.source = copy.deepcopy(source)
    instrumentation._validate_source = lambda value: copy.deepcopy(value)
    instrumentation._captured = {
        sequence: SimpleNamespace(
            binding=SimpleNamespace(trace_request_id=f"trace-{sequence}"),
            engine_request_id=f"{sequence}-{sequence:08x}",
        )
        for sequence in range(1, 5)
    }
    instrumentation._records_by_sequence = {
        sequence: SimpleNamespace(trace_request_id=f"trace-{sequence}")
        for sequence in range(5)
    }
    instrumentation._sequence_to_engine = {
        sequence: f"{sequence}-{sequence:08x}" for sequence in range(5)
    }
    observations = [
        {
            "sequence_index": sequence,
            "phase": ("ordinary_prime" if sequence == 0 else "ordinary_batch_forward"),
            "trace_request_id": f"trace-{sequence}",
            "external_request_id": str(sequence),
            "prompt_tokens": 512,
            "output_token_ids": [sequence],
            "num_cached_tokens": 0,
        }
        for sequence in range(5)
    ]
    recovery_evidence = {
        "request_id_bindings": [
            {
                "trace_request_id": observation["trace_request_id"],
                "external_request_id": observation["external_request_id"],
                "engine_request_id": instrumentation._sequence_to_engine[
                    observation["sequence_index"]
                ],
                "phase": observation["phase"],
                "binding_event_action": "gate.numerical.request_bound",
            }
            for observation in observations
        ]
    }
    import collective_gate_common as common

    runtime = {
        "image": common.EXPECTED_IMAGE,
        "target_volume": common.EXPECTED_TARGET_VOLUME,
        "accelerator_count": 1,
        "device_arch": common.EXPECTED_RUNTIME["device_arch"],
    }
    selected, recovery, copied_runtime = instrumentation._validate_finish_evidence(
        arm="ordinary",
        observations=observations,
        recovery_evidence=recovery_evidence,
        source=source,
        runtime=runtime,
    )
    if (
        selected != observations[1:]
        or recovery != recovery_evidence
        or copied_runtime != runtime
    ):
        raise AssertionError("finish ID-domain validation altered exact evidence")

    instrumentation._captured[1].engine_request_id = "1-0000000A"
    try:
        instrumentation._validate_finish_evidence(
            arm="ordinary",
            observations=observations,
            recovery_evidence=recovery_evidence,
            source=source,
            runtime=runtime,
        )
    except NumericalCaptureError:
        pass
    else:
        raise AssertionError("finish admitted an uppercase internal nonce")
    instrumentation._captured[1].engine_request_id = "1-00000001"

    changed = copy.deepcopy(observations)
    changed[0]["external_request_id"] = "2"
    try:
        instrumentation._validate_finish_evidence(
            arm="ordinary",
            observations=changed,
            recovery_evidence=recovery_evidence,
            source=source,
            runtime=runtime,
        )
    except NumericalCaptureError:
        pass
    else:
        raise AssertionError("finish admitted a duplicate external request ID")
    return {
        "finish_domain_pairs_bound": len(observations),
        "finish_domain_mutations_rejected": 2,
    }


def _expect_runtime_error(call: Callable[[], object], label: str) -> None:
    try:
        call()
    except RuntimeError:
        return
    raise AssertionError(f"independent verifier admitted drift: {label}")


def _verify_layer_api_id_domains() -> dict[str, int]:
    request_id = "1-11111111"
    lookup_id = "11111111-2222-4333-8444-555555555555"

    def inherited(
        name: str,
        _owner: object,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> dict[str, object]:
        if name == "prepare_request_lookup":
            peer_sidecars = (
                args[1] if len(args) > 1 else kwargs.get("peer_sidecars", ())
            )
            return {
                "request_id": str(args[0].request_id),
                "sidecar_count": len(peer_sidecars),
            }
        if name not in {"store_layer", "retrieve_layer"}:
            return {"unrelated": True}
        return {"request_id": kwargs.get("request_id") or kwargs.get("req_id")}

    retrieve_lookup_only = _domain_aware_layer_call_details(
        inherited,
        "retrieve_layer",
        object(),
        (),
        {"req_id": lookup_id},
    )
    if retrieve_lookup_only != {"lookup_id": lookup_id}:
        raise AssertionError("retrieve_layer did not publish lookup identity only")
    store_request_only = _domain_aware_layer_call_details(
        inherited,
        "store_layer",
        object(),
        (),
        {"req_id": request_id},
    )
    if store_request_only != {"request_id": request_id}:
        raise AssertionError("store_layer did not publish request identity only")

    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorRequestSidecar,
    )

    sidecar_payload = (
        "kmesh-request-lookup-v1",
        "serial_pic",
        request_id,
        1,
        512,
        "synthetic-cache-domain:a",
        "no_shared_segments",
        (),
    )
    sidecar = KVConnectorRequestSidecar(
        provider_id="vllm.kmesh.request_lookup.v1",
        payload=sidecar_payload,
    )
    request = SimpleNamespace(request_id=request_id)
    captured = _domain_aware_layer_call_details(
        inherited,
        "prepare_request_lookup",
        object(),
        (request, (sidecar,)),
        {},
    )
    if captured != {
        "request_id": request_id,
        "sidecar_count": 1,
        "sidecars": [_zero_sidecar_evidence(request_id, 1)],
    }:
        raise AssertionError("prepare_request_lookup sidecar capture drifted")

    class SidecarSubclass(KVConnectorRequestSidecar):
        pass

    sidecar_rejection_cases: list[tuple[str, Callable[[], object]]] = [
        (
            "prepare sidecars passed by keyword",
            lambda: _domain_aware_layer_call_details(
                inherited,
                "prepare_request_lookup",
                object(),
                (request,),
                {"peer_sidecars": (sidecar,)},
            ),
        ),
        (
            "prepare sidecars mutable outer list",
            lambda: _domain_aware_layer_call_details(
                inherited,
                "prepare_request_lookup",
                object(),
                (request, [sidecar]),
                {},
            ),
        ),
        (
            "prepare sidecar subclass",
            lambda: _domain_aware_layer_call_details(
                inherited,
                "prepare_request_lookup",
                object(),
                (request, (SidecarSubclass(sidecar.provider_id, sidecar.payload),)),
                {},
            ),
        ),
        (
            "prepare sidecar wrong provider",
            lambda: _domain_aware_layer_call_details(
                inherited,
                "prepare_request_lookup",
                object(),
                (
                    request,
                    (
                        KVConnectorRequestSidecar(
                            provider_id="wrong.provider",
                            payload=sidecar_payload,
                        ),
                    ),
                ),
                {},
            ),
        ),
        (
            "prepare sidecar wrong wire tag",
            lambda: _domain_aware_layer_call_details(
                inherited,
                "prepare_request_lookup",
                object(),
                (
                    request,
                    (
                        KVConnectorRequestSidecar(
                            provider_id=sidecar.provider_id,
                            payload=("wrong-wire", *sidecar_payload[1:]),
                        ),
                    ),
                ),
                {},
            ),
        ),
        (
            "prepare sidecar wrong wire arity",
            lambda: _domain_aware_layer_call_details(
                inherited,
                "prepare_request_lookup",
                object(),
                (
                    request,
                    (
                        KVConnectorRequestSidecar(
                            provider_id=sidecar.provider_id,
                            payload=sidecar_payload[:-1],
                        ),
                    ),
                ),
                {},
            ),
        ),
        (
            "prepare inherited boolean sidecar count",
            lambda: _domain_aware_layer_call_details(
                lambda *_args, **_kwargs: {"sidecar_count": True},
                "prepare_request_lookup",
                object(),
                (request, (sidecar,)),
                {},
            ),
        ),
    ]
    for label, call in sidecar_rejection_cases:
        _expect_gate_error(call, label)

    rejection_cases: list[tuple[str, Callable[[], object]]] = [
        (
            "retrieve missing req_id",
            lambda: _domain_aware_layer_call_details(
                inherited,
                "retrieve_layer",
                object(),
                (),
                {},
            ),
        ),
        (
            "retrieve explicit request with lookup",
            lambda: _domain_aware_layer_call_details(
                inherited,
                "retrieve_layer",
                object(),
                (),
                {"request_id": request_id, "req_id": lookup_id},
            ),
        ),
        (
            "retrieve explicit request without lookup",
            lambda: _domain_aware_layer_call_details(
                inherited,
                "retrieve_layer",
                object(),
                (),
                {"request_id": request_id},
            ),
        ),
        (
            "retrieve collapsed aliases",
            lambda: _domain_aware_layer_call_details(
                inherited,
                "retrieve_layer",
                object(),
                (),
                {"request_id": request_id, "req_id": request_id},
            ),
        ),
        (
            "retrieve non-UUID lookup",
            lambda: _domain_aware_layer_call_details(
                inherited,
                "retrieve_layer",
                object(),
                (),
                {"req_id": "lookup-not-a-uuid"},
            ),
        ),
        (
            "store conflicting aliases",
            lambda: _domain_aware_layer_call_details(
                inherited,
                "store_layer",
                object(),
                (),
                {"request_id": request_id, "req_id": "2-22222222"},
            ),
        ),
        (
            "inherited behavior drift",
            lambda: _domain_aware_layer_call_details(
                lambda *_args, **_kwargs: {"request_id": "wrong"},
                "retrieve_layer",
                object(),
                (),
                {"req_id": lookup_id},
            ),
        ),
    ]
    for label, call in rejection_cases:
        _expect_gate_error(call, label)

    bindings = [
        {
            "trace_request_id": "trace-1",
            "external_request_id": "1",
            "engine_request_id": request_id,
            "phase": "probe",
        },
        {
            "trace_request_id": "trace-2",
            "external_request_id": "2",
            "engine_request_id": "2-22222222",
            "phase": "other-probe",
        },
    ]
    valid_layer_events = [
        {
            "sequence": 0,
            "phase": "probe",
            "action": "LMCacheLookupClient.lookup.begin",
            "lookup_id": lookup_id,
        },
        {
            "sequence": 1,
            "phase": "probe",
            "action": "LMCacheLookupClient.lookup.end",
            "lookup_id": lookup_id,
        },
        {
            "sequence": 2,
            "phase": "probe",
            "action": "LMCacheEngine.retrieve_layer.begin",
            "lookup_id": lookup_id,
        },
        {
            "sequence": 3,
            "phase": "probe",
            "action": "LMCacheEngine.retrieve_layer.closed",
            "lookup_id": lookup_id,
        },
        {
            "sequence": 4,
            "phase": "probe",
            "action": "LMCacheEngine.store_layer.begin",
            "request_id": request_id,
        },
        {
            "sequence": 5,
            "phase": "probe",
            "action": "LMCacheEngine.store_layer.closed",
            "request_id": request_id,
        },
    ]
    _verify_event_id_domains(Path("layer-api-exact"), valid_layer_events, bindings)
    verifier_mutations: list[tuple[str, Callable[[list[dict[str, object]]], None]]] = [
        (
            "retrieve carries request_id",
            lambda events: events[2].update(request_id=request_id),
        ),
        (
            "retrieve omits lookup_id",
            lambda events: events[2].pop("lookup_id"),
        ),
        (
            "retrieve lookup lacks canonical owner",
            lambda events: events[2].update(
                lookup_id="aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
            ),
        ),
        (
            "store carries lookup_id",
            lambda events: events[4].update(lookup_id=lookup_id),
        ),
        (
            "store carries lookup as request",
            lambda events: events[4].update(request_id=lookup_id),
        ),
        (
            "orphan retrieve begin",
            lambda events: events.pop(3),
        ),
        (
            "duplicate retrieve terminal",
            lambda events: events.append(
                {
                    **copy.deepcopy(events[3]),
                    "sequence": 6,
                    "action": "LMCacheEngine.retrieve_layer.end",
                }
            ),
        ),
        (
            "lookup UUID reused across phases",
            lambda events: events.extend(
                [
                    {
                        "sequence": 6,
                        "phase": "other-probe",
                        "action": "LMCacheLookupClient.lookup.begin",
                        "lookup_id": lookup_id,
                    },
                    {
                        "sequence": 7,
                        "phase": "other-probe",
                        "action": "LMCacheLookupClient.lookup.end",
                        "lookup_id": lookup_id,
                    },
                ]
            ),
        ),
        (
            "store lifecycle moved to another request phase",
            lambda events: (
                events[4].update(phase="other-probe"),
                events[5].update(phase="other-probe"),
            ),
        ),
    ]
    for label, mutate in verifier_mutations:
        changed = copy.deepcopy(valid_layer_events)
        mutate(changed)
        _expect_runtime_error(
            lambda changed=changed: _verify_event_id_domains(
                Path(label), changed, bindings
            ),
            label,
        )
    return {
        "layer_api_domain_cases_admitted": 2,
        "layer_api_domain_mutations_rejected": len(rejection_cases),
        "request_sidecar_capture_cases_admitted": 1,
        "request_sidecar_capture_mutations_rejected": len(sidecar_rejection_cases),
        "verifier_layer_events_admitted": len(valid_layer_events),
        "verifier_layer_mutations_rejected": len(verifier_mutations),
    }


def _layout_observations(arm: str) -> list[dict[str, object]]:
    sequences = {
        "collective_reverse": [0, 4, 3, 2, 1],
        "zero_sharing": [1, 2, 3, 4],
        "incompatible_partition": [0, 1, 2, 3, 4],
        "sibling_failure": [0, 1, 2, 3, 4],
    }[arm]
    phases = [arm] * 4 if arm == "zero_sharing" else ["prime"] + [arm] * 4
    return [
        {
            "external_request_id": str(external),
            "trace_request_id": f"trace-{sequence}",
            "sequence_index": sequence,
            "phase": phase,
            "prompt_tokens": 512,
        }
        for external, (sequence, phase) in enumerate(
            zip(sequences, phases, strict=True)
        )
    ]


def _layout_spans() -> dict[str, object]:
    return {
        "token_count": 512,
        "cached_span": [0, 512],
        "spans": [[24, 440]],
    }


def _layout_mask() -> dict[str, object]:
    true_indices = list(range(24, 440))
    encoded = bytes(int(index in set(true_indices)) for index in range(512))
    return {
        "token_count": 512,
        "true_count": len(true_indices),
        "true_indices": true_indices,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _zero_sidecar_evidence(request_id: str, generation: int) -> dict[str, object]:
    payload: list[object] = [
        "kmesh-request-lookup-v1",
        "serial_pic",
        request_id,
        generation,
        512,
        "synthetic-cache-domain:a",
        "no_shared_segments",
        [],
    ]
    payload_sha256 = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "provider_id": "vllm.kmesh.request_lookup.v1",
        "payload": payload,
        "payload_sha256": payload_sha256,
        "wire_tag": "kmesh-request-lookup-v1",
        "intent": "serial_pic",
        "descriptor_request_id": request_id,
        "admission_generation": generation,
        "prompt_token_count": 512,
        "cache_domain": "synthetic-cache-domain:a",
        "fallback": "no_shared_segments",
        "span_count": 0,
    }


def _layout_case(
    arm: str,
) -> tuple[
    list[dict[str, object]],
    dict[str, object],
    list[dict[str, object]],
]:
    observations = _layout_observations(arm)
    bindings = [
        {
            "trace_request_id": observation["trace_request_id"],
            "external_request_id": observation["external_request_id"],
            "engine_request_id": (
                f"{observation['external_request_id']}-{external:08x}"
            ),
            "phase": observation["phase"],
            "binding_event_action": (
                "LMCacheConnectorV1Impl.prepare_request_lookup.begin"
            ),
        }
        for external, observation in enumerate(observations)
    ]
    binding_by_sequence = {
        observation["sequence_index"]: binding
        for observation, binding in zip(observations, bindings, strict=True)
    }
    events: list[dict[str, object]] = []
    for index, binding in enumerate(bindings):
        prepare = {
            "phase": binding["phase"],
            "request_id": binding["engine_request_id"],
            "owner_type": "LMCacheConnectorV1Impl",
            "role": "SCHEDULER",
        }
        begin = {
            "action": "LMCacheConnectorV1Impl.prepare_request_lookup.begin",
            **prepare,
        }
        events.append(begin)
        if arm == "zero_sharing":
            begin["sidecar_count"] = 1
            begin["sidecars"] = [
                _zero_sidecar_evidence(binding["engine_request_id"], index + 1)
            ]
            events.append(
                {
                    "action": "LMCacheConnectorV1Impl.prepare_request_lookup.end",
                    **prepare,
                    "sidecar_count": 1,
                    "sidecars": copy.deepcopy(begin["sidecars"]),
                    "result": None,
                }
            )
            matched = {
                "phase": arm,
                "request_id": binding["engine_request_id"],
                "owner_type": "LMCacheConnectorV1Impl",
                "role": "SCHEDULER",
                "num_computed_tokens": 0,
            }
            events.extend(
                [
                    {
                        "action": (
                            "LMCacheConnectorV1Impl.get_num_new_matched_tokens.begin"
                        ),
                        **matched,
                    },
                    {
                        "action": (
                            "LMCacheConnectorV1Impl.get_num_new_matched_tokens.end"
                        ),
                        **matched,
                        "result": 0,
                    },
                ]
            )
            allocated = {
                "phase": arm,
                "request_id": binding["engine_request_id"],
                "owner_type": "LMCacheConnectorV1Impl",
                "role": "SCHEDULER",
                "num_external_tokens": 0,
            }
            events.extend(
                [
                    {
                        "action": (
                            "LMCacheConnectorV1Impl.update_state_after_alloc.begin"
                        ),
                        **allocated,
                    },
                    {
                        "action": (
                            "LMCacheConnectorV1Impl.update_state_after_alloc.end"
                        ),
                        **allocated,
                        "result": None,
                    },
                ]
            )
    sibling_lookup_id = "11111111-2222-4333-8444-555555555555"
    if arm == "sibling_failure":
        events.extend(
            [
                {
                    "action": "LMCacheLookupClient.lookup.begin",
                    "phase": arm,
                    "lookup_id": sibling_lookup_id,
                },
                {
                    "action": "LMCacheLookupClient.lookup.end",
                    "phase": arm,
                    "lookup_id": sibling_lookup_id,
                },
            ]
        )
    groups: list[dict[str, object]] = []

    if arm == "zero_sharing":
        projection = {
            "action": "gate.collective.plan_projection",
            "phase": arm,
            "plans": [],
        }
        events.extend([projection, copy.deepcopy(projection)])
        meta = {
            "phase": arm,
            "owner_type": "LMCacheConnectorV1Impl",
            "role": "SCHEDULER",
        }
        events.extend(
            [
                {"action": "gate.lmcache_connector_meta.begin", **meta},
                {
                    "action": "gate.lmcache_connector_meta.end",
                    **meta,
                    "request_count": 0,
                },
            ]
        )
        for observation, binding in zip(observations, bindings, strict=True):
            finished = {
                "phase": arm,
                "request_id": binding["engine_request_id"],
                "owner_type": "LMCacheConnectorV1Impl",
                "role": "SCHEDULER",
            }
            events.extend(
                [
                    {
                        "action": "LMCacheConnectorV1Impl.request_finished.begin",
                        **finished,
                    },
                    {
                        "action": "LMCacheConnectorV1Impl.request_finished.end",
                        **finished,
                        "result": {"type": "tuple"},
                    },
                ]
            )
        events.extend(
            {"action": "gate.request.complete", **copy.deepcopy(observation)}
            for observation in observations
        )
    else:
        prime_projection = {
            "action": "gate.collective.plan_projection",
            "phase": "prime",
            "plans": [],
        }
        events.extend([prime_projection, copy.deepcopy(prime_projection)])
        if arm == "incompatible_partition":
            plan_sequences = ((1, 2), (3, 4))
        elif arm == "collective_reverse":
            plan_sequences = (
                tuple(
                    observation["sequence_index"] for observation in observations[1:]
                ),
            )
        else:
            plan_sequences = ((1, 2, 3, 4),)
        plans: list[dict[str, object]] = []
        for plan_index, sequences in enumerate(plan_sequences):
            members = [
                {
                    "request_id": binding_by_sequence[sequence]["engine_request_id"],
                    "explicit_spans": _layout_spans(),
                }
                for sequence in sequences
            ]
            plans.append(
                {
                    "plan_id": f"plan-{arm}-{plan_index}",
                    "reuse_group_id": f"reuse-{arm}-{plan_index}",
                    "members": members,
                }
            )
        projection = {
            "action": "gate.collective.plan_projection",
            "phase": arm,
            "plans": copy.deepcopy(plans),
        }
        events.extend([projection, copy.deepcopy(projection)])

        group_sequences = (
            (1, 2)
            if arm == "incompatible_partition"
            else tuple(
                observation["sequence_index"] for observation in observations[1:]
            )
        )
        plan = plans[0]
        input_members = [
            {
                "request_id": binding_by_sequence[sequence]["engine_request_id"],
                "generation": 1,
                "token_count": 512,
                "explicit_spans": _layout_spans(),
            }
            for sequence in group_sequences
        ]
        events.append(
            {
                "action": "gate.collective.blend_group.begin",
                "phase": arm,
                "num_layers": 24,
                "members": copy.deepcopy(input_members),
            }
        )
        failed_id = input_members[1]["request_id"] if arm == "sibling_failure" else None
        if failed_id is not None:
            events.append(
                {
                    "action": "gate.sibling_failure.injected",
                    "phase": arm,
                    "request_id": failed_id,
                    "completed_serial_operations": {
                        "serial_rope_operations": 0,
                        "serial_diff_operations": 0,
                        "collective_rope_operations": 0,
                        "collective_diff_operations": 0,
                        "position_recovery_serial_operations": 0,
                        "position_recovery_collective_operations": 0,
                    },
                }
            )
        result_members: list[dict[str, object]] = []
        normalized_members: list[dict[str, object]] = []
        for input_member, sequence in zip(input_members, group_sequences, strict=True):
            failed = input_member["request_id"] == failed_id
            operation_counts = {
                "serial_rope_operations": 0,
                "serial_diff_operations": 0,
                "collective_rope_operations": 0 if failed else 24,
                "collective_diff_operations": 0 if failed else 1,
                "position_recovery_serial_operations": 0,
                "position_recovery_collective_operations": 0 if failed else 24,
            }
            retrieval_mask = None if failed else _layout_mask()
            error = (
                {"type": "RuntimeError", "message": "synthetic sibling failure"}
                if failed
                else None
            )
            outcome = "failed" if failed else "recovered"
            result_members.append(
                {
                    "request_id": input_member["request_id"],
                    "generation": 1,
                    "outcome": outcome,
                    "retrieval_mask": retrieval_mask,
                    "operation_counts": operation_counts,
                    "error": error,
                }
            )
            binding = binding_by_sequence[sequence]
            normalized_members.append(
                {
                    "trace_request_id": binding["trace_request_id"],
                    "external_request_id": binding["external_request_id"],
                    "engine_request_id": binding["engine_request_id"],
                    "generation": 1,
                    "outcome": outcome,
                    "selected_spans": input_member["explicit_spans"],
                    "retrieval_mask": retrieval_mask,
                    "operation_counts": operation_counts,
                    "error": error,
                }
            )
        events.append(
            {
                "action": "gate.collective.blend_group.end",
                "phase": arm,
                "num_layers": 24,
                "input_members": copy.deepcopy(input_members),
                "result_members": result_members,
            }
        )
        groups.append(
            {
                "call_index": 0,
                "plan_id": plan["plan_id"],
                "reuse_group_id": plan["reuse_group_id"],
                "num_layers": 24,
                "engine_request_order": [
                    member["request_id"] for member in input_members
                ],
                "members": normalized_members,
            }
        )
        if failed_id is not None:
            failure_details = {
                "phase": arm,
                "owner_type": "LMCacheConnectorV1Impl",
                "role": "WORKER",
                "request_id": failed_id,
                "reason": "request-local collective CacheBlend failure: synthetic",
                "invalidate_full_prompt": True,
                "contract_slot_count": 512,
                "advertised_tokens": 440,
                "lookup_id": sibling_lookup_id,
            }
            events.extend(
                [
                    {
                        "action": (
                            "LMCacheConnectorV1Impl._record_kmesh_load_failure.begin"
                        ),
                        **failure_details,
                    },
                    {
                        "action": (
                            "LMCacheConnectorV1Impl._record_kmesh_load_failure.end"
                        ),
                        **failure_details,
                    },
                ]
            )
            finish_common = {
                "phase": arm,
                "owner_type": "LMCacheConnectorV1Impl",
                "role": "SCHEDULER",
            }
            for member in input_members:
                request_id = member["request_id"]
                if request_id == failed_id:
                    continue
                events.extend(
                    [
                        {
                            "action": "LMCacheConnectorV1Impl.request_finished.begin",
                            "request_id": request_id,
                            **finish_common,
                        },
                        {
                            "action": "LMCacheConnectorV1Impl.request_finished.end",
                            "request_id": request_id,
                            "result": {"type": "tuple"},
                            **finish_common,
                        },
                    ]
                )
            events.extend(
                [
                    {
                        "action": "gate.lmcache_connector_meta.begin",
                        **finish_common,
                    },
                    {
                        "action": "gate.lmcache_connector_meta.end",
                        "request_count": 0,
                        **finish_common,
                    },
                ]
            )
            recovery_projection = {
                "action": "gate.collective.plan_projection",
                "phase": arm,
                "plans": [],
            }
            events.extend([recovery_projection, copy.deepcopy(recovery_projection)])
            events.extend(
                [
                    {
                        "action": "LMCacheConnectorV1Impl.request_finished.begin",
                        "request_id": failed_id,
                        **finish_common,
                    },
                    {
                        "action": "LMCacheConnectorV1Impl.request_finished.end",
                        "request_id": failed_id,
                        "result": {"type": "tuple"},
                        **finish_common,
                    },
                ]
            )

    for sequence, event in enumerate(events):
        event["schema_version"] = 1
        event["sequence"] = sequence
    recovery: dict[str, object] = {
        "groups": groups,
        "serial_members": [],
        "request_id_bindings": bindings,
        "trace_submission_order": [
            observation["trace_request_id"]
            for observation in observations
            if observation["phase"] != "prime"
        ],
    }
    return observations, recovery, events


def _complete_zero_sharing_case() -> tuple[
    list[dict[str, object]],
    dict[str, object],
    list[dict[str, object]],
]:
    observations, recovery, layout_events = _layout_case("zero_sharing")
    for ordinal, observation in enumerate(observations):
        observation.update(
            batch_ordinal=ordinal,
            num_cached_tokens=0,
            output_token_ids=[100 + observation["sequence_index"]],
            finish_reason="length",
        )
    trace_ids = [observation["trace_request_id"] for observation in observations]
    events: list[dict[str, object]] = [
        {
            "phase": "construction",
            "action": "gate.construction.begin",
            "arm": "zero_sharing",
        },
        {
            "phase": "construction",
            "action": "gate.construction.end",
            "arm": "zero_sharing",
            "resolved_target_profile": "collective_batched_qkv_v1",
            "resolved_attention_backend": "TRITON_ATTN",
            "kv_transfer_config_present": True,
        },
        {
            "phase": "zero_sharing",
            "action": "gate.batch.begin",
            "batch_size": 4,
            "submitted_trace_request_ids": trace_ids,
            "prompt_tokens": 2048,
        },
    ]
    completion_index = 0
    for event in copy.deepcopy(layout_events):
        event.pop("sequence", None)
        event.pop("schema_version", None)
        if event["action"] == "gate.request.complete":
            event = {
                "action": "gate.request.complete",
                **copy.deepcopy(observations[completion_index]),
            }
            completion_index += 1
        events.append(event)
    if completion_index != 4:
        raise AssertionError("zero-sharing fixture lost completion events")
    events.append(
        {
            "phase": "zero_sharing",
            "action": "gate.batch.end",
            "batch_size": 4,
            "submitted_trace_request_ids": trace_ids,
            "returned_external_request_ids": [
                observation["external_request_id"] for observation in observations
            ],
        }
    )
    worker_state = {
        field: 0
        for field in (
            "_kmesh_store_sessions_count",
            "_kmesh_outstanding_lookup_ids_count",
            "_kmesh_store_cleanup_errors_count",
            "_kmesh_failed_worker_requests_count",
        )
    }
    scheduler_state = {
        field: 0
        for field in (
            "_prealloc_lookup_ids_count",
            "_allocated_lookup_ids_count",
            "_worker_owned_lookup_requests_count",
            "_kmesh_prepared_lookups_count",
            "_kmesh_prepared_request_objects_count",
            "_kmesh_allocated_requests_count",
            "_unfinished_requests_count",
            "_request_trackers_count",
            "load_specs_count",
        )
    }
    for role, state in (
        ("WORKER", worker_state),
        ("SCHEDULER", scheduler_state),
    ):
        events.extend(
            [
                {
                    "phase": "shutdown",
                    "action": "LMCacheConnectorV1Impl.shutdown.begin",
                    "owner_type": "LMCacheConnectorV1Impl",
                    "role": role,
                },
                {
                    "phase": "shutdown",
                    "action": "LMCacheConnectorV1Impl.shutdown.end",
                    "owner_type": "LMCacheConnectorV1Impl",
                    "role": role,
                    "manager_services_stopped": True,
                    "result": None,
                    **state,
                },
            ]
        )
    events.append({"phase": "shutdown", "action": "gate.shutdown.complete"})
    for sequence, event in enumerate(events):
        event["schema_version"] = 1
        event["sequence"] = sequence
    return observations, recovery, events


def _verify_non_numerical_id_layouts() -> dict[str, object]:
    arms = (
        "collective_reverse",
        "zero_sharing",
        "incompatible_partition",
        "sibling_failure",
    )
    cases = {arm: _layout_case(arm) for arm in arms}
    for arm, (observations, recovery, events) in cases.items():
        _verify_recovery_ledger(
            Path(f"{arm}-exact"),
            arm,
            observations,
            recovery,
            recovery["request_id_bindings"],
            events,
        )

    sibling_observations, sibling_recovery, sibling_events = copy.deepcopy(
        cases["sibling_failure"]
    )
    if _recovery_evidence(sibling_observations, sibling_events) != sibling_recovery:
        raise AssertionError("runner changed the sibling two-round fixture")

    sibling_round_phases = ("prime", "sibling_failure", "sibling_failure")
    runner_sibling_rounds = _runner_canonical_projection_events(
        sibling_events,
        sibling_round_phases,
    )
    verifier_sibling_rounds = _verifier_canonical_projection_events(
        Path("sibling-projection-rounds-exact"),
        sibling_events,
        sibling_round_phases,
    )
    for label, canonical in (
        ("runner", runner_sibling_rounds),
        ("verifier", verifier_sibling_rounds),
    ):
        if (
            [event["phase"] for event in canonical]
            != ["prime", "sibling_failure", "sibling_failure"]
            or canonical[0]["plans"] != []
            or not canonical[1]["plans"]
            or canonical[2]["plans"] != []
        ):
            raise AssertionError(f"{label} changed the sibling projection rounds")

    def projection_receipts(
        events: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        return [
            event
            for event in events
            if event["action"] == "gate.collective.plan_projection"
        ]

    def resequence_events(events: list[dict[str, object]]) -> None:
        for sequence, event in enumerate(events):
            event["sequence"] = sequence

    def add_extra_recovery_pair(events: list[dict[str, object]]) -> None:
        receipts = projection_receipts(events)
        extra = copy.deepcopy(receipts[-2:])
        insertion = events.index(receipts[-1]) + 1
        events[insertion:insertion] = extra
        resequence_events(events)

    def drop_recovery_pair(events: list[dict[str, object]]) -> None:
        for receipt in projection_receipts(events)[-2:]:
            events.remove(receipt)

    def drop_recovery_bind(events: list[dict[str, object]]) -> None:
        events.remove(projection_receipts(events)[-1])

    def diverge_recovery_bind(events: list[dict[str, object]]) -> None:
        receipts = projection_receipts(events)
        receipts[-1]["plans"] = [{"forged": True}]

    def inject_unexpected_recovery_phase(
        events: list[dict[str, object]],
    ) -> None:
        receipts = projection_receipts(events)
        receipts[-2]["phase"] = "cold_miss"
        receipts[-1]["phase"] = "cold_miss"

    def make_recovery_bind_malformed(events: list[dict[str, object]]) -> None:
        receipts = projection_receipts(events)
        receipts[-1]["plans"] = {}

    def reverse_sibling_round_payloads(events: list[dict[str, object]]) -> None:
        receipts = projection_receipts(events)
        nonempty = copy.deepcopy(receipts[2]["plans"])
        receipts[2]["plans"] = []
        receipts[3]["plans"] = []
        receipts[4]["plans"] = copy.deepcopy(nonempty)
        receipts[5]["plans"] = copy.deepcopy(nonempty)

    def interleave_projection_pairs(events: list[dict[str, object]]) -> None:
        receipts = projection_receipts(events)
        left = events.index(receipts[1])
        right = events.index(receipts[2])
        events[left], events[right] = events[right], events[left]
        resequence_events(events)

    def tie_recovery_sequence(events: list[dict[str, object]]) -> None:
        receipts = projection_receipts(events)
        receipts[-1]["sequence"] = receipts[-2]["sequence"]

    def reverse_recovery_sequence(events: list[dict[str, object]]) -> None:
        receipts = projection_receipts(events)
        receipts[-1]["sequence"] = receipts[-2]["sequence"] - 1

    def alias_recovery_sequence_type(events: list[dict[str, object]]) -> None:
        receipts = projection_receipts(events)
        receipts[-1]["sequence"] = True

    def move_recovery_pair_before_failure(events: list[dict[str, object]]) -> None:
        receipts = projection_receipts(events)
        retry_pair = receipts[-2:]
        for receipt in retry_pair:
            events.remove(receipt)
        injection_index = next(
            index
            for index, event in enumerate(events)
            if event["action"] == "gate.sibling_failure.injected"
        )
        events[injection_index:injection_index] = retry_pair
        resequence_events(events)

    def make_retry_metadata_nonzero(events: list[dict[str, object]]) -> None:
        retry_validate_sequence = projection_receipts(events)[-2]["sequence"]
        retry_meta = [
            event
            for event in events
            if event["action"] == "gate.lmcache_connector_meta.end"
            and event["phase"] == "sibling_failure"
            and event["sequence"] < retry_validate_sequence
        ][-1]
        retry_meta["request_count"] = 1

    def select_sibling_events(
        events: list[dict[str, object]],
        action: str,
    ) -> list[dict[str, object]]:
        return [
            event
            for event in events
            if event["action"] == action and event["phase"] == "sibling_failure"
        ]

    def drop_recovered_finish_begin(events: list[dict[str, object]]) -> None:
        events.remove(
            select_sibling_events(
                events,
                "LMCacheConnectorV1Impl.request_finished.begin",
            )[0]
        )

    def add_extra_finish_pair(events: list[dict[str, object]]) -> None:
        begin = select_sibling_events(
            events,
            "LMCacheConnectorV1Impl.request_finished.begin",
        )[0]
        end = select_sibling_events(
            events,
            "LMCacheConnectorV1Impl.request_finished.end",
        )[0]
        insertion = events.index(end) + 1
        events[insertion:insertion] = [copy.deepcopy(begin), copy.deepcopy(end)]
        resequence_events(events)

    def drop_recovered_finish_end(events: list[dict[str, object]]) -> None:
        events.remove(
            select_sibling_events(
                events,
                "LMCacheConnectorV1Impl.request_finished.end",
            )[0]
        )

    def reorder_recovered_finish_pair(events: list[dict[str, object]]) -> None:
        begin = select_sibling_events(
            events,
            "LMCacheConnectorV1Impl.request_finished.begin",
        )[0]
        end = select_sibling_events(
            events,
            "LMCacheConnectorV1Impl.request_finished.end",
        )[0]
        left = events.index(begin)
        right = events.index(end)
        events[left], events[right] = events[right], events[left]
        resequence_events(events)

    def drift_finish_owner(events: list[dict[str, object]]) -> None:
        select_sibling_events(
            events,
            "LMCacheConnectorV1Impl.request_finished.begin",
        )[
            0
        ]["owner_type"] = "WrongOwner"

    def drift_finish_role(events: list[dict[str, object]]) -> None:
        select_sibling_events(
            events,
            "LMCacheConnectorV1Impl.request_finished.end",
        )[
            0
        ]["role"] = "WORKER"

    def alias_finish_sequence_type(events: list[dict[str, object]]) -> None:
        select_sibling_events(
            events,
            "LMCacheConnectorV1Impl.request_finished.begin",
        )[
            0
        ]["sequence"] = True

    def alias_finish_result_type(events: list[dict[str, object]]) -> None:
        select_sibling_events(
            events,
            "LMCacheConnectorV1Impl.request_finished.end",
        )[
            0
        ]["result"] = {"type": True}

    def move_meta_begin_before_recovered_cleanup(
        events: list[dict[str, object]],
    ) -> None:
        meta_begin = select_sibling_events(events, "gate.lmcache_connector_meta.begin")[
            0
        ]
        events.remove(meta_begin)
        first_finish = select_sibling_events(
            events,
            "LMCacheConnectorV1Impl.request_finished.begin",
        )[0]
        events.insert(events.index(first_finish), meta_begin)
        resequence_events(events)

    def drop_retry_meta_begin(events: list[dict[str, object]]) -> None:
        events.remove(
            select_sibling_events(events, "gate.lmcache_connector_meta.begin")[0]
        )

    def add_extra_retry_meta_pair(events: list[dict[str, object]]) -> None:
        meta_begin = select_sibling_events(events, "gate.lmcache_connector_meta.begin")[
            0
        ]
        meta_end = select_sibling_events(events, "gate.lmcache_connector_meta.end")[0]
        insertion = events.index(meta_end) + 1
        events[insertion:insertion] = [
            copy.deepcopy(meta_begin),
            copy.deepcopy(meta_end),
        ]
        resequence_events(events)

    def add_post_retry_meta_event(
        events: list[dict[str, object]],
        action: str,
    ) -> None:
        extra = copy.deepcopy(select_sibling_events(events, action)[0])
        retry_bind = projection_receipts(events)[-1]
        events.insert(events.index(retry_bind) + 1, extra)
        resequence_events(events)

    def drift_meta_owner(events: list[dict[str, object]]) -> None:
        select_sibling_events(events, "gate.lmcache_connector_meta.begin")[0][
            "owner_type"
        ] = "WrongOwner"

    def drift_meta_role(events: list[dict[str, object]]) -> None:
        select_sibling_events(events, "gate.lmcache_connector_meta.end")[0][
            "role"
        ] = "WORKER"

    def alias_meta_request_count_type(events: list[dict[str, object]]) -> None:
        select_sibling_events(events, "gate.lmcache_connector_meta.end")[0][
            "request_count"
        ] = False

    sibling_receipt_adversaries: list[
        tuple[str, Callable[[list[dict[str, object]]], None]]
    ] = [
        ("sibling extra recovery pair", add_extra_recovery_pair),
        ("sibling missing recovery pair", drop_recovery_pair),
        ("sibling half recovery pair", drop_recovery_bind),
        ("sibling divergent recovery bind", diverge_recovery_bind),
        ("sibling unexpected recovery phase", inject_unexpected_recovery_phase),
        ("sibling malformed recovery bind", make_recovery_bind_malformed),
        ("sibling reversed round payloads", reverse_sibling_round_payloads),
        ("sibling interleaved projection pairs", interleave_projection_pairs),
        ("sibling tied recovery sequence", tie_recovery_sequence),
        ("sibling reversed recovery sequence", reverse_recovery_sequence),
        ("sibling boolean recovery sequence", alias_recovery_sequence_type),
        ("sibling retry pair before failure", move_recovery_pair_before_failure),
        ("sibling nonzero retry metadata", make_retry_metadata_nonzero),
        ("sibling orphan finish end", drop_recovered_finish_begin),
        ("sibling extra finish pair", add_extra_finish_pair),
        ("sibling missing finish end", drop_recovered_finish_end),
        ("sibling reordered finish pair", reorder_recovered_finish_pair),
        ("sibling finish owner drift", drift_finish_owner),
        ("sibling finish role drift", drift_finish_role),
        ("sibling boolean finish sequence", alias_finish_sequence_type),
        ("sibling finish result type alias", alias_finish_result_type),
        (
            "sibling metadata begin before recovered cleanup",
            move_meta_begin_before_recovered_cleanup,
        ),
        ("sibling missing metadata begin", drop_retry_meta_begin),
        ("sibling extra metadata pair", add_extra_retry_meta_pair),
        (
            "sibling post-retry metadata begin",
            lambda events: add_post_retry_meta_event(
                events,
                "gate.lmcache_connector_meta.begin",
            ),
        ),
        (
            "sibling post-retry metadata end",
            lambda events: add_post_retry_meta_event(
                events,
                "gate.lmcache_connector_meta.end",
            ),
        ),
        ("sibling metadata owner drift", drift_meta_owner),
        ("sibling metadata role drift", drift_meta_role),
        ("sibling boolean metadata count", alias_meta_request_count_type),
    ]
    for label, mutate in sibling_receipt_adversaries:
        changed = copy.deepcopy(sibling_events)
        mutate(changed)
        _expect_gate_error(
            lambda changed=changed: _runner_canonical_projection_events(
                changed,
                sibling_round_phases,
            ),
            f"runner {label}",
        )
        _expect_runtime_error(
            lambda changed=changed: _verifier_canonical_projection_events(
                Path(label),
                changed,
                sibling_round_phases,
            ),
            f"verifier {label}",
        )

    zero_observations, zero_recovery, zero_events = copy.deepcopy(cases["zero_sharing"])
    if _recovery_evidence(zero_observations, zero_events) != zero_recovery:
        raise AssertionError("runner changed the exact zero-sharing sidecar fixture")

    complete_observations, complete_recovery, complete_events = (
        _complete_zero_sharing_case()
    )
    complete_runner_recovery = _recovery_evidence(
        complete_observations,
        complete_events,
    )
    if complete_runner_recovery != complete_recovery:
        raise AssertionError("runner changed the complete zero-sharing fixture")
    baseline = {
        observation["trace_request_id"]: observation["output_token_ids"]
        for observation in complete_observations
    }
    runner_checks = _runner_evaluate(
        "zero_sharing",
        complete_observations,
        complete_events,
        baseline,
        None,
        complete_runner_recovery,
    )
    if runner_checks.get("all_passed") is not True:
        raise AssertionError("runner rejected the complete zero-sharing control")
    _verify_recovery_ledger(
        Path("zero-sharing-complete"),
        "zero_sharing",
        complete_observations,
        complete_recovery,
        complete_recovery["request_id_bindings"],
        complete_events,
    )
    verifier_checks = _verify_arm_claims(
        Path("zero-sharing-complete"),
        "zero_sharing",
        complete_observations,
        complete_recovery,
        complete_events,
    )
    if verifier_checks.get("all_passed") is not True:
        raise AssertionError("verifier rejected the complete zero-sharing control")

    def prepare_events(
        events: list[dict[str, object]],
        terminal: str,
    ) -> list[dict[str, object]]:
        action = f"LMCacheConnectorV1Impl.prepare_request_lookup.{terminal}"
        return [event for event in events if event["action"] == action]

    def resequence(events: list[dict[str, object]]) -> None:
        for sequence, event in enumerate(events):
            event["sequence"] = sequence

    def set_sidecar_count(
        events: list[dict[str, object]],
        value: object,
    ) -> None:
        prepare_events(events, "begin")[0]["sidecar_count"] = value
        prepare_events(events, "end")[0]["sidecar_count"] = value

    def omit_sidecar_count(events: list[dict[str, object]]) -> None:
        prepare_events(events, "begin")[0].pop("sidecar_count")

    def mismatch_sidecar_count(events: list[dict[str, object]]) -> None:
        prepare_events(events, "end")[0]["sidecar_count"] = 0

    def duplicate_prepare(events: list[dict[str, object]]) -> None:
        events.extend(
            [
                copy.deepcopy(prepare_events(events, "begin")[0]),
                copy.deepcopy(prepare_events(events, "end")[0]),
            ]
        )

    def drop_prepare_end(events: list[dict[str, object]]) -> None:
        events.remove(prepare_events(events, "end")[0])

    def drift_prepare_field(
        events: list[dict[str, object]],
        field: str,
        value: object,
    ) -> None:
        prepare_events(events, "begin")[0][field] = value
        prepare_events(events, "end")[0][field] = value

    def mismatch_prepare_request(events: list[dict[str, object]]) -> None:
        prepare_events(events, "end")[0]["request_id"] = "9-deadbeef"

    def swap_prepare_request_order(events: list[dict[str, object]]) -> None:
        begins = prepare_events(events, "begin")
        ends = prepare_events(events, "end")
        for receipts in (begins, ends):
            receipts[0]["request_id"], receipts[1]["request_id"] = (
                receipts[1]["request_id"],
                receipts[0]["request_id"],
            )

    def inject_lookup_rpc(events: list[dict[str, object]]) -> None:
        lookup_id = "00000000-0000-4000-8000-000000000099"
        events.extend(
            [
                {
                    "action": "LMCacheLookupClient.lookup.begin",
                    "phase": "zero_sharing",
                    "owner_type": "LMCacheLookupClient",
                    "lookup_id": lookup_id,
                    "token_count": 512,
                },
                {
                    "action": "LMCacheLookupClient.lookup.end",
                    "phase": "zero_sharing",
                    "owner_type": "LMCacheLookupClient",
                    "lookup_id": lookup_id,
                    "token_count": 512,
                    "result": 0,
                },
            ]
        )

    def mutate_sidecar_summary(
        events: list[dict[str, object]],
        field: str,
        value: object,
    ) -> None:
        for terminal in ("begin", "end"):
            prepare_events(events, terminal)[0]["sidecars"][0][field] = value

    def mutate_sidecar_payload(
        events: list[dict[str, object]],
        index: int,
        summary_field: str,
        value: object,
    ) -> None:
        for terminal in ("begin", "end"):
            sidecar = prepare_events(events, terminal)[0]["sidecars"][0]
            sidecar["payload"][index] = copy.deepcopy(value)
            sidecar[summary_field] = copy.deepcopy(value)
            sidecar["payload_sha256"] = hashlib.sha256(
                json.dumps(
                    sidecar["payload"],
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()

    def mutate_sidecar_spans(events: list[dict[str, object]]) -> None:
        for terminal in ("begin", "end"):
            sidecar = prepare_events(events, terminal)[0]["sidecars"][0]
            sidecar["payload"][7] = [["forged-span"]]
            sidecar["span_count"] = 1
            sidecar["payload_sha256"] = hashlib.sha256(
                json.dumps(
                    sidecar["payload"],
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()

    def mismatch_sidecar_payload(events: list[dict[str, object]]) -> None:
        sidecar = prepare_events(events, "end")[0]["sidecars"][0]
        sidecar["payload"][3] = 2
        sidecar["admission_generation"] = 2
        sidecar["payload_sha256"] = hashlib.sha256(
            json.dumps(
                sidecar["payload"],
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    def selected(
        events: list[dict[str, object]],
        action: str,
    ) -> list[dict[str, object]]:
        return [event for event in events if event["action"] == action]

    def drop_first(events: list[dict[str, object]], action: str) -> None:
        events.remove(selected(events, action)[0])

    def duplicate_pair(events: list[dict[str, object]], stem: str) -> None:
        events.extend(
            [
                copy.deepcopy(selected(events, f"{stem}.begin")[0]),
                copy.deepcopy(selected(events, f"{stem}.end")[0]),
            ]
        )

    def move_pair_before_prepare(
        events: list[dict[str, object]],
        stem: str,
    ) -> None:
        pair = [
            selected(events, f"{stem}.begin")[0],
            selected(events, f"{stem}.end")[0],
        ]
        for event in pair:
            events.remove(event)
        insert_at = events.index(prepare_events(events, "begin")[0])
        events[insert_at:insert_at] = pair

    def drift_first_pair(
        events: list[dict[str, object]],
        stem: str,
        field: str,
        value: object,
    ) -> None:
        selected(events, f"{stem}.begin")[0][field] = value
        selected(events, f"{stem}.end")[0][field] = value

    def mismatch_first_pair(
        events: list[dict[str, object]],
        stem: str,
        field: str,
        value: object,
    ) -> None:
        selected(events, f"{stem}.end")[0][field] = value

    def omit_first_end_field(
        events: list[dict[str, object]],
        stem: str,
        field: str,
    ) -> None:
        selected(events, f"{stem}.end")[0].pop(field)

    def move_pair_before_action(
        events: list[dict[str, object]],
        stem: str,
        action: str,
    ) -> None:
        pair = [
            selected(events, f"{stem}.begin")[0],
            selected(events, f"{stem}.end")[0],
        ]
        for event in pair:
            events.remove(event)
        insert_at = events.index(selected(events, action)[0])
        events[insert_at:insert_at] = pair

    def inject_forbidden_event(
        events: list[dict[str, object]],
        action: str,
    ) -> None:
        events.append(
            {
                "action": action,
                "phase": "zero_sharing",
                "owner_type": action.split(".", 1)[0],
            }
        )

    def duplicate_meta(events: list[dict[str, object]]) -> None:
        events.extend(
            [
                copy.deepcopy(selected(events, "gate.lmcache_connector_meta.begin")[0]),
                copy.deepcopy(selected(events, "gate.lmcache_connector_meta.end")[0]),
            ]
        )

    def drift_meta_field(
        events: list[dict[str, object]],
        field: str,
        value: object,
    ) -> None:
        selected(events, "gate.lmcache_connector_meta.begin")[0][field] = value
        selected(events, "gate.lmcache_connector_meta.end")[0][field] = value

    def move_meta_before_updates(events: list[dict[str, object]]) -> None:
        begin = selected(events, "gate.lmcache_connector_meta.begin")[0]
        end = selected(events, "gate.lmcache_connector_meta.end")[0]
        events.remove(begin)
        events.remove(end)
        insert_at = events.index(
            selected(
                events,
                "LMCacheConnectorV1Impl.update_state_after_alloc.begin",
            )[0]
        )
        events[insert_at:insert_at] = [begin, end]

    def move_projections_before_updates(events: list[dict[str, object]]) -> None:
        projections = selected(events, "gate.collective.plan_projection")
        for event in projections:
            events.remove(event)
        insert_at = events.index(
            selected(
                events,
                "LMCacheConnectorV1Impl.update_state_after_alloc.begin",
            )[0]
        )
        events[insert_at:insert_at] = projections

    def inject_store(events: list[dict[str, object]]) -> None:
        events.extend(
            [
                {
                    "action": "LMCacheEngine.store_layer.begin",
                    "phase": "zero_sharing",
                    "request_id": "0-00000000",
                    "owner_type": "LMCacheEngine",
                    "role": "WORKER",
                },
                {
                    "action": "LMCacheEngine.store_layer.end",
                    "phase": "zero_sharing",
                    "request_id": "0-00000000",
                    "owner_type": "LMCacheEngine",
                    "role": "WORKER",
                    "result": None,
                },
            ]
        )

    def inject_transfer(events: list[dict[str, object]]) -> None:
        events.extend(
            [
                {
                    "action": "LMCacheLookupClient.transfer_lookups.begin",
                    "phase": "zero_sharing",
                    "owner_type": "LMCacheLookupClient",
                    "lookup_ids": [],
                },
                {
                    "action": "LMCacheLookupClient.transfer_lookups.end",
                    "phase": "zero_sharing",
                    "owner_type": "LMCacheLookupClient",
                    "lookup_ids": [],
                    "result": None,
                },
            ]
        )

    zero_sidecar_adversaries: list[
        tuple[str, Callable[[list[dict[str, object]]], None]]
    ] = [
        ("zero-sharing missing sidecar_count", omit_sidecar_count),
        (
            "zero-sharing sidecar_count zero",
            lambda events: set_sidecar_count(events, 0),
        ),
        (
            "zero-sharing boolean sidecar_count",
            lambda events: set_sidecar_count(events, True),
        ),
        (
            "zero-sharing sidecar_count two",
            lambda events: set_sidecar_count(events, 2),
        ),
        ("zero-sharing sidecar_count begin/end mismatch", mismatch_sidecar_count),
        ("zero-sharing duplicate prepare", duplicate_prepare),
        ("zero-sharing missing prepare", drop_prepare_end),
        (
            "zero-sharing prepare owner drift",
            lambda events: drift_prepare_field(events, "owner_type", "WrongOwner"),
        ),
        (
            "zero-sharing prepare role drift",
            lambda events: drift_prepare_field(events, "role", "WORKER"),
        ),
        (
            "zero-sharing prepare phase drift",
            lambda events: drift_prepare_field(events, "phase", "prime"),
        ),
        ("zero-sharing prepare request mismatch", mismatch_prepare_request),
        ("zero-sharing prepare request order drift", swap_prepare_request_order),
        ("zero-sharing injected lookup RPC", inject_lookup_rpc),
        (
            "zero-sharing wrong sidecar provider",
            lambda events: mutate_sidecar_summary(
                events, "provider_id", "wrong.provider"
            ),
        ),
        (
            "zero-sharing wrong payload digest",
            lambda events: mutate_sidecar_summary(events, "payload_sha256", "0" * 64),
        ),
        ("zero-sharing begin/end payload mismatch", mismatch_sidecar_payload),
        (
            "zero-sharing type-sensitive generation summary",
            lambda events: mutate_sidecar_summary(events, "admission_generation", True),
        ),
        (
            "zero-sharing wrong wire tag",
            lambda events: mutate_sidecar_payload(events, 0, "wire_tag", "wrong-wire"),
        ),
        (
            "zero-sharing actionable intent",
            lambda events: mutate_sidecar_payload(events, 1, "intent", "observe_only"),
        ),
        (
            "zero-sharing descriptor request mismatch",
            lambda events: mutate_sidecar_payload(
                events, 2, "descriptor_request_id", "9-deadbeef"
            ),
        ),
        (
            "zero-sharing boolean generation",
            lambda events: mutate_sidecar_payload(
                events, 3, "admission_generation", True
            ),
        ),
        (
            "zero-sharing zero generation",
            lambda events: mutate_sidecar_payload(events, 3, "admission_generation", 0),
        ),
        (
            "zero-sharing ordinal generation drift",
            lambda events: mutate_sidecar_payload(events, 3, "admission_generation", 5),
        ),
        (
            "zero-sharing prompt count drift",
            lambda events: mutate_sidecar_payload(events, 4, "prompt_token_count", 511),
        ),
        (
            "zero-sharing cache domain drift",
            lambda events: mutate_sidecar_payload(
                events, 5, "cache_domain", "synthetic-cache-domain:b"
            ),
        ),
        (
            "zero-sharing fallback drift",
            lambda events: mutate_sidecar_payload(
                events, 6, "fallback", "internal_error"
            ),
        ),
        ("zero-sharing nonempty spans", mutate_sidecar_spans),
        (
            "zero-sharing prepare end omits result",
            lambda events: omit_first_end_field(
                events, "LMCacheConnectorV1Impl.prepare_request_lookup", "result"
            ),
        ),
        (
            "zero-sharing missing matched-token result",
            lambda events: drop_first(
                events,
                "LMCacheConnectorV1Impl.get_num_new_matched_tokens.end",
            ),
        ),
        (
            "zero-sharing duplicate matched-token pair",
            lambda events: duplicate_pair(
                events, "LMCacheConnectorV1Impl.get_num_new_matched_tokens"
            ),
        ),
        (
            "zero-sharing matched owner drift",
            lambda events: drift_first_pair(
                events,
                "LMCacheConnectorV1Impl.get_num_new_matched_tokens",
                "owner_type",
                "WrongOwner",
            ),
        ),
        (
            "zero-sharing matched phase drift",
            lambda events: drift_first_pair(
                events,
                "LMCacheConnectorV1Impl.get_num_new_matched_tokens",
                "phase",
                "prime",
            ),
        ),
        (
            "zero-sharing matched computed-token drift",
            lambda events: drift_first_pair(
                events,
                "LMCacheConnectorV1Impl.get_num_new_matched_tokens",
                "num_computed_tokens",
                1,
            ),
        ),
        (
            "zero-sharing matched end computed-token boolean",
            lambda events: mismatch_first_pair(
                events,
                "LMCacheConnectorV1Impl.get_num_new_matched_tokens",
                "num_computed_tokens",
                False,
            ),
        ),
        (
            "zero-sharing matched result boolean",
            lambda events: mismatch_first_pair(
                events,
                "LMCacheConnectorV1Impl.get_num_new_matched_tokens",
                "result",
                False,
            ),
        ),
        (
            "zero-sharing missing allocation update",
            lambda events: drop_first(
                events, "LMCacheConnectorV1Impl.update_state_after_alloc.end"
            ),
        ),
        (
            "zero-sharing duplicate allocation update",
            lambda events: duplicate_pair(
                events, "LMCacheConnectorV1Impl.update_state_after_alloc"
            ),
        ),
        (
            "zero-sharing allocation token drift",
            lambda events: drift_first_pair(
                events,
                "LMCacheConnectorV1Impl.update_state_after_alloc",
                "num_external_tokens",
                1,
            ),
        ),
        (
            "zero-sharing allocation end token boolean",
            lambda events: mismatch_first_pair(
                events,
                "LMCacheConnectorV1Impl.update_state_after_alloc",
                "num_external_tokens",
                False,
            ),
        ),
        (
            "zero-sharing allocation role drift",
            lambda events: drift_first_pair(
                events,
                "LMCacheConnectorV1Impl.update_state_after_alloc",
                "role",
                "WORKER",
            ),
        ),
        (
            "zero-sharing allocation request drift",
            lambda events: drift_first_pair(
                events,
                "LMCacheConnectorV1Impl.update_state_after_alloc",
                "request_id",
                "9-deadbeef",
            ),
        ),
        (
            "zero-sharing allocation end omits result",
            lambda events: omit_first_end_field(
                events, "LMCacheConnectorV1Impl.update_state_after_alloc", "result"
            ),
        ),
        (
            "zero-sharing matched lookup before prepare",
            lambda events: move_pair_before_prepare(
                events, "LMCacheConnectorV1Impl.get_num_new_matched_tokens"
            ),
        ),
        (
            "zero-sharing allocation before prepare",
            lambda events: move_pair_before_prepare(
                events, "LMCacheConnectorV1Impl.update_state_after_alloc"
            ),
        ),
        (
            "zero-sharing allocation before matched-token result",
            lambda events: move_pair_before_action(
                events,
                "LMCacheConnectorV1Impl.update_state_after_alloc",
                "LMCacheConnectorV1Impl.get_num_new_matched_tokens.begin",
            ),
        ),
        ("zero-sharing projections before allocation", move_projections_before_updates),
        ("zero-sharing injected engine store", inject_store),
        ("zero-sharing injected lookup transfer", inject_transfer),
        *[
            (
                f"zero-sharing forbidden event {action}",
                lambda events, action=action: inject_forbidden_event(events, action),
            )
            for action in (
                "LMCacheEngine.retrieve_layer.begin",
                "LMCacheLookupClient.cancel_lookup.begin",
                "LMCacheLookupClient.finish_lookup.begin",
                "LMCacheEngine.register_lookup_cleanup.begin",
                "LMCacheEngine.schedule_lookup_unpin.begin",
                "LMCacheEngine.lookup_unpin.begin",
                "LMCacheConnectorV1Impl._record_kmesh_load_failure.begin",
                "_KMeshLayerwiseStoreSession.finish.begin",
                "LMCBlender.blend.begin",
                "gate.collective.blend_group.begin",
                "gate.serial_blend.evidence",
            )
        ],
        (
            "zero-sharing nonempty connector metadata",
            lambda events: selected(events, "gate.lmcache_connector_meta.end")[
                0
            ].update(request_count=1),
        ),
        (
            "zero-sharing boolean connector metadata count",
            lambda events: selected(events, "gate.lmcache_connector_meta.end")[
                0
            ].update(request_count=False),
        ),
        (
            "zero-sharing missing connector metadata end",
            lambda events: drop_first(events, "gate.lmcache_connector_meta.end"),
        ),
        ("zero-sharing duplicate connector metadata", duplicate_meta),
        (
            "zero-sharing connector metadata owner drift",
            lambda events: drift_meta_field(events, "owner_type", "WrongOwner"),
        ),
        (
            "zero-sharing connector metadata role drift",
            lambda events: drift_meta_field(events, "role", "WORKER"),
        ),
        (
            "zero-sharing connector metadata phase drift",
            lambda events: drift_meta_field(events, "phase", "prime"),
        ),
        (
            "zero-sharing connector metadata error",
            lambda events: inject_forbidden_event(
                events, "gate.lmcache_connector_meta.error"
            ),
        ),
        ("zero-sharing connector metadata before updates", move_meta_before_updates),
        (
            "zero-sharing missing request finish",
            lambda events: drop_first(
                events, "LMCacheConnectorV1Impl.request_finished.end"
            ),
        ),
        (
            "zero-sharing duplicate request finish",
            lambda events: duplicate_pair(
                events, "LMCacheConnectorV1Impl.request_finished"
            ),
        ),
        (
            "zero-sharing request finish owner drift",
            lambda events: drift_first_pair(
                events,
                "LMCacheConnectorV1Impl.request_finished",
                "owner_type",
                "WrongOwner",
            ),
        ),
        (
            "zero-sharing request finish role drift",
            lambda events: drift_first_pair(
                events,
                "LMCacheConnectorV1Impl.request_finished",
                "role",
                "WORKER",
            ),
        ),
        (
            "zero-sharing request finish phase drift",
            lambda events: drift_first_pair(
                events,
                "LMCacheConnectorV1Impl.request_finished",
                "phase",
                "prime",
            ),
        ),
        (
            "zero-sharing request finish owner request drift",
            lambda events: drift_first_pair(
                events,
                "LMCacheConnectorV1Impl.request_finished",
                "request_id",
                "9-deadbeef",
            ),
        ),
        (
            "zero-sharing request finish result drift",
            lambda events: mismatch_first_pair(
                events,
                "LMCacheConnectorV1Impl.request_finished",
                "result",
                [False, None],
            ),
        ),
        (
            "zero-sharing request finish omits result",
            lambda events: omit_first_end_field(
                events, "LMCacheConnectorV1Impl.request_finished", "result"
            ),
        ),
        (
            "zero-sharing request finish before metadata",
            lambda events: move_pair_before_action(
                events,
                "LMCacheConnectorV1Impl.request_finished",
                "gate.lmcache_connector_meta.begin",
            ),
        ),
    ]
    runner_zero_sidecar_rejections = 0
    verifier_zero_sidecar_rejections = 0
    for label, mutate in zero_sidecar_adversaries:
        observations, recovery, events = copy.deepcopy(cases["zero_sharing"])
        mutate(events)
        resequence(events)
        _expect_gate_error(
            lambda observations=observations, events=events: _recovery_evidence(
                observations,
                events,
            ),
            label,
        )
        runner_zero_sidecar_rejections += 1
        _expect_runtime_error(
            lambda observations=observations, recovery=recovery, events=events: (
                _verify_recovery_ledger(
                    Path(label),
                    "zero_sharing",
                    observations,
                    recovery,
                    recovery["request_id_bindings"],
                    events,
                )
            ),
            label,
        )
        verifier_zero_sidecar_rejections += 1

    receipt_pair_rejections = 0
    receipt_diagnoses: list[str] = []
    phase_cases = (
        ("zero_sharing", "zero_sharing"),
        ("incompatible_partition", "prime"),
        ("incompatible_partition", "incompatible_partition"),
    )
    for arm, phase in phase_cases:
        for mutation in ("missing", "divergent", "third"):
            observations, recovery, events = copy.deepcopy(cases[arm])
            projection_indices = [
                index
                for index, event in enumerate(events)
                if event["action"] == "gate.collective.plan_projection"
                and event["phase"] == phase
            ]
            if len(projection_indices) != 2:
                raise AssertionError("synthetic projection receipt pair drifted")
            if mutation == "missing":
                events.pop(projection_indices[1])
            elif mutation == "divergent":
                second = events[projection_indices[1]]
                if second["plans"]:
                    second["plans"][0]["plan_id"] = "divergent-plan"
                else:
                    second["plans"] = [{"forged": True}]
            else:
                events.insert(
                    projection_indices[1] + 1,
                    copy.deepcopy(events[projection_indices[1]]),
                )
            _expect_runtime_error(
                lambda arm=arm, observations=observations, recovery=recovery, events=events: _verify_recovery_ledger(
                    Path(f"{arm}-{phase}-{mutation}-projection-receipt"),
                    arm,
                    observations,
                    recovery,
                    recovery["request_id_bindings"],
                    events,
                ),
                f"{arm} {phase} {mutation} projection receipt",
            )
            receipt_pair_rejections += 1
            receipt_diagnoses.append(f"{arm}:{phase}:{mutation}")

    def receipt_pair(
        events: list[dict[str, object]],
        phase: str,
    ) -> list[dict[str, object]]:
        receipts = [
            event
            for event in events
            if event["action"] == "gate.collective.plan_projection"
            and event["phase"] == phase
        ]
        if len(receipts) != 2:
            raise AssertionError("synthetic projection receipt pair drifted")
        return receipts

    def reject_receipt_adversary(
        label: str,
        arm: str,
        mutate: Callable[[dict[str, object], list[dict[str, object]]], None],
    ) -> None:
        nonlocal receipt_pair_rejections
        observations, recovery, events = copy.deepcopy(cases[arm])
        mutate(recovery, events)
        _expect_runtime_error(
            lambda: _verify_recovery_ledger(
                Path(label),
                arm,
                observations,
                recovery,
                recovery["request_id_bindings"],
                events,
            ),
            label,
        )
        receipt_pair_rejections += 1
        receipt_diagnoses.append(label)

    def diverge_reuse_group(
        _recovery: dict[str, object],
        events: list[dict[str, object]],
    ) -> None:
        receipt_pair(events, "incompatible_partition")[1]["plans"][0][
            "reuse_group_id"
        ] = "divergent-reuse-group"

    def diverge_member_set(
        _recovery: dict[str, object],
        events: list[dict[str, object]],
    ) -> None:
        receipt_pair(events, "collective_reverse")[1]["plans"][0]["members"].pop()

    def diverge_member_order(
        _recovery: dict[str, object],
        events: list[dict[str, object]],
    ) -> None:
        receipt_pair(events, "collective_reverse")[1]["plans"][0]["members"].reverse()

    def diverge_span_geometry(
        _recovery: dict[str, object],
        events: list[dict[str, object]],
    ) -> None:
        receipt_pair(events, "collective_reverse")[1]["plans"][0]["members"][0][
            "explicit_spans"
        ]["token_count"] = 511

    def diverge_exact_type_alias(
        _recovery: dict[str, object],
        events: list[dict[str, object]],
    ) -> None:
        receipt_pair(events, "collective_reverse")[1]["plans"][0]["members"][0][
            "explicit_spans"
        ]["token_count"] = True

    def one_sided_coherent_plan_id(
        recovery: dict[str, object],
        events: list[dict[str, object]],
    ) -> None:
        receipt_pair(events, "collective_reverse")[1]["plans"][0][
            "plan_id"
        ] = "coherent-bind-plan"
        recovery["groups"][0]["plan_id"] = "coherent-bind-plan"

    def drift_bind_phase(
        _recovery: dict[str, object],
        events: list[dict[str, object]],
    ) -> None:
        receipt_pair(events, "collective_reverse")[1]["phase"] = "cold_miss"

    def tie_receipt_sequence(
        _recovery: dict[str, object],
        events: list[dict[str, object]],
    ) -> None:
        receipts = receipt_pair(events, "collective_reverse")
        receipts[1]["sequence"] = receipts[0]["sequence"]

    for label, arm, mutate in (
        (
            "projection receipt reuse-group divergence",
            "incompatible_partition",
            diverge_reuse_group,
        ),
        (
            "projection receipt member-set divergence",
            "collective_reverse",
            diverge_member_set,
        ),
        (
            "projection receipt member-order divergence",
            "collective_reverse",
            diverge_member_order,
        ),
        (
            "projection receipt span-geometry divergence",
            "collective_reverse",
            diverge_span_geometry,
        ),
        (
            "projection receipt exact-type alias",
            "collective_reverse",
            diverge_exact_type_alias,
        ),
        (
            "one-sided coherent bind/downstream plan ID",
            "collective_reverse",
            one_sided_coherent_plan_id,
        ),
        (
            "projection bind phase drift",
            "collective_reverse",
            drift_bind_phase,
        ),
        (
            "projection receipt non-increasing sequence",
            "collective_reverse",
            tie_receipt_sequence,
        ),
    ):
        reject_receipt_adversary(label, arm, mutate)

    def target_projections(
        events: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        return [
            event
            for event in events
            if event["action"] == "gate.collective.plan_projection"
            and event["phase"] != "prime"
        ]

    def projection(events: list[dict[str, object]]) -> dict[str, object]:
        return next(event for event in target_projections(events) if event["plans"])

    def reverse_projected_members(
        _observations: list[dict[str, object]],
        _recovery: dict[str, object],
        events: list[dict[str, object]],
    ) -> None:
        for receipt in target_projections(events):
            receipt["plans"][0]["members"].reverse()

    def duplicate_partition_owner(
        _observations: list[dict[str, object]],
        _recovery: dict[str, object],
        events: list[dict[str, object]],
    ) -> None:
        for receipt in target_projections(events):
            duplicate = copy.deepcopy(receipt["plans"][0])
            duplicate["plan_id"] = "plan-incompatible_partition-2"
            duplicate["reuse_group_id"] = "reuse-incompatible_partition-2"
            receipt["plans"].append(duplicate)

    def duplicate_plan_id(
        _observations: list[dict[str, object]],
        _recovery: dict[str, object],
        events: list[dict[str, object]],
    ) -> None:
        for receipt in target_projections(events):
            plans = receipt["plans"]
            plans[1]["plan_id"] = plans[0]["plan_id"]

    def unsort_plan_ids(
        _observations: list[dict[str, object]],
        _recovery: dict[str, object],
        events: list[dict[str, object]],
    ) -> None:
        for receipt in target_projections(events):
            plans = receipt["plans"]
            plans[0]["plan_id"] = "zz-plan"
            plans[1]["plan_id"] = "aa-plan"

    def duplicate_reuse_group_id(
        _observations: list[dict[str, object]],
        _recovery: dict[str, object],
        events: list[dict[str, object]],
    ) -> None:
        for receipt in target_projections(events):
            plans = receipt["plans"]
            plans[1]["reuse_group_id"] = plans[0]["reuse_group_id"]

    def duplicate_projected_member_id(
        _observations: list[dict[str, object]],
        _recovery: dict[str, object],
        events: list[dict[str, object]],
    ) -> None:
        for receipt in target_projections(events):
            members = receipt["plans"][0]["members"]
            members[1]["request_id"] = members[0]["request_id"]

    def drop_projected_member(
        _observations: list[dict[str, object]],
        _recovery: dict[str, object],
        events: list[dict[str, object]],
    ) -> None:
        for receipt in target_projections(events):
            receipt["plans"][0]["members"].pop()

    def mutate_projected_token_count(
        _observations: list[dict[str, object]],
        _recovery: dict[str, object],
        events: list[dict[str, object]],
    ) -> None:
        for receipt in target_projections(events):
            receipt["plans"][0]["members"][0]["explicit_spans"]["token_count"] = 511

    def boolean_projected_token_count(
        _observations: list[dict[str, object]],
        _recovery: dict[str, object],
        events: list[dict[str, object]],
    ) -> None:
        for receipt in target_projections(events):
            receipt["plans"][0]["members"][0]["explicit_spans"]["token_count"] = True

    def move_sibling_failure_to_index_zero(
        _observations: list[dict[str, object]],
        recovery: dict[str, object],
        events: list[dict[str, object]],
    ) -> None:
        group = recovery["groups"][0]
        result_event = next(
            event
            for event in events
            if event["action"] == "gate.collective.blend_group.end"
        )
        identity_fields = (
            "outcome",
            "retrieval_mask",
            "operation_counts",
            "error",
        )
        for members in (group["members"], result_event["result_members"]):
            first = {
                field: copy.deepcopy(members[0][field]) for field in identity_fields
            }
            second = {
                field: copy.deepcopy(members[1][field]) for field in identity_fields
            }
            members[0].update(second)
            members[1].update(first)
        failed_id = group["engine_request_order"][0]
        for event in events:
            if event["action"] in {
                "gate.sibling_failure.injected",
                "LMCacheConnectorV1Impl._record_kmesh_load_failure.begin",
                "LMCacheConnectorV1Impl._record_kmesh_load_failure.end",
            }:
                event["request_id"] = failed_id

    def mutate_recovered_operation_count(
        _observations: list[dict[str, object]],
        recovery: dict[str, object],
        events: list[dict[str, object]],
    ) -> None:
        recovery["groups"][0]["members"][0]["operation_counts"][
            "serial_rope_operations"
        ] = 23
        next(
            event
            for event in events
            if event["action"] == "gate.collective.blend_group.end"
        )["result_members"][0]["operation_counts"]["serial_rope_operations"] = 23

    def mutate_failed_operation_count(
        _observations: list[dict[str, object]],
        recovery: dict[str, object],
        events: list[dict[str, object]],
    ) -> None:
        recovery["groups"][0]["members"][1]["operation_counts"][
            "serial_rope_operations"
        ] = 1
        next(
            event
            for event in events
            if event["action"] == "gate.collective.blend_group.end"
        )["result_members"][1]["operation_counts"]["serial_rope_operations"] = 1
        next(
            event
            for event in events
            if event["action"] == "gate.sibling_failure.injected"
        )["completed_serial_operations"]["serial_rope_operations"] = 1

    def mutate_boolean_operation_count(
        _observations: list[dict[str, object]],
        recovery: dict[str, object],
        events: list[dict[str, object]],
    ) -> None:
        recovery["groups"][0]["members"][0]["operation_counts"][
            "serial_diff_operations"
        ] = True
        next(
            event
            for event in events
            if event["action"] == "gate.collective.blend_group.end"
        )["result_members"][0]["operation_counts"]["serial_diff_operations"] = True

    def mutate_num_layers_coherently(
        _observations: list[dict[str, object]],
        recovery: dict[str, object],
        events: list[dict[str, object]],
    ) -> None:
        recovery["groups"][0]["num_layers"] = 23
        for member in recovery["groups"][0]["members"]:
            if member["outcome"] == "recovered":
                member["operation_counts"]["serial_rope_operations"] = 23
        for event in events:
            if event["action"] in {
                "gate.collective.blend_group.begin",
                "gate.collective.blend_group.end",
            }:
                event["num_layers"] = 23
            if event["action"] == "gate.collective.blend_group.end":
                for member in event["result_members"]:
                    if member["outcome"] == "recovered":
                        member["operation_counts"]["serial_rope_operations"] = 23

    def mutate_group_generation_and_tokens(
        _observations: list[dict[str, object]],
        recovery: dict[str, object],
        events: list[dict[str, object]],
    ) -> None:
        for member in recovery["groups"][0]["members"]:
            member["generation"] = "bogus"
        for event in events:
            if event["action"] == "gate.collective.blend_group.begin":
                for member in event["members"]:
                    member["generation"] = "bogus"
                    member["token_count"] = 0
            elif event["action"] == "gate.collective.blend_group.end":
                for member in event["input_members"]:
                    member["generation"] = "bogus"
                    member["token_count"] = 0
                for member in event["result_members"]:
                    member["generation"] = "bogus"

    def move_failure_chain_phase(
        _observations: list[dict[str, object]],
        _recovery: dict[str, object],
        events: list[dict[str, object]],
    ) -> None:
        for event in events:
            if event["action"] in {
                "gate.sibling_failure.injected",
                "LMCacheConnectorV1Impl._record_kmesh_load_failure.begin",
                "LMCacheConnectorV1Impl._record_kmesh_load_failure.end",
            }:
                event["phase"] = "prime"

    def forge_selected_spans(
        _observations: list[dict[str, object]],
        recovery: dict[str, object],
        events: list[dict[str, object]],
    ) -> None:
        for member in recovery["groups"][0]["members"]:
            member["selected_spans"] = {}
        for event in target_projections(events):
            for member in event["plans"][0]["members"]:
                member["explicit_spans"] = {}
        for event in events:
            if event["action"] == "gate.collective.blend_group.begin":
                for member in event["members"]:
                    member["explicit_spans"] = {}
            elif event["action"] == "gate.collective.blend_group.end":
                for member in event["input_members"]:
                    member["explicit_spans"] = {}

    mutations: list[
        tuple[
            str,
            str,
            Callable[
                [list[dict[str, object]], dict[str, object], list[dict[str, object]]],
                None,
            ],
        ]
    ] = [
        (
            "reverse engine order",
            "collective_reverse",
            lambda _observations, recovery, _events: recovery["groups"][0][
                "engine_request_order"
            ].reverse(),
        ),
        (
            "zero-sharing missing empty projection",
            "zero_sharing",
            lambda _observations, _recovery, events: events.pop(),
        ),
        (
            "incompatible duplicate partition owner",
            "incompatible_partition",
            duplicate_partition_owner,
        ),
        (
            "projected member order",
            "collective_reverse",
            reverse_projected_members,
        ),
        (
            "duplicate plan ID",
            "incompatible_partition",
            duplicate_plan_id,
        ),
        (
            "unsorted plan IDs",
            "incompatible_partition",
            unsort_plan_ids,
        ),
        (
            "duplicate reuse-group ID",
            "incompatible_partition",
            duplicate_reuse_group_id,
        ),
        (
            "duplicate projected member ID",
            "collective_reverse",
            duplicate_projected_member_id,
        ),
        (
            "projected member set",
            "collective_reverse",
            drop_projected_member,
        ),
        (
            "projected token geometry",
            "collective_reverse",
            mutate_projected_token_count,
        ),
        (
            "boolean projected token count",
            "collective_reverse",
            boolean_projected_token_count,
        ),
        (
            "sibling injection owner",
            "sibling_failure",
            lambda _observations, recovery, events: next(
                event
                for event in events
                if event["action"] == "gate.sibling_failure.injected"
            ).update(request_id=recovery["groups"][0]["engine_request_order"][0]),
        ),
        (
            "self-consistent sibling owner index",
            "sibling_failure",
            move_sibling_failure_to_index_zero,
        ),
        (
            "coherent recovered operation count",
            "collective_reverse",
            mutate_recovered_operation_count,
        ),
        (
            "coherent failed operation count",
            "sibling_failure",
            mutate_failed_operation_count,
        ),
        (
            "boolean operation count",
            "collective_reverse",
            mutate_boolean_operation_count,
        ),
        (
            "coherent group layer count",
            "collective_reverse",
            mutate_num_layers_coherently,
        ),
        (
            "coherent group generation and token count",
            "collective_reverse",
            mutate_group_generation_and_tokens,
        ),
        (
            "failure-chain phase drift",
            "sibling_failure",
            move_failure_chain_phase,
        ),
        (
            "coherent selected-span forgery",
            "collective_reverse",
            forge_selected_spans,
        ),
    ]
    for label, arm, mutate in mutations:
        observations, recovery, events = copy.deepcopy(cases[arm])
        mutate(observations, recovery, events)
        _expect_runtime_error(
            lambda observations=observations, recovery=recovery, events=events, arm=arm: (
                _verify_recovery_ledger(
                    Path(label),
                    arm,
                    observations,
                    recovery,
                    recovery["request_id_bindings"],
                    events,
                )
            ),
            label,
        )
    return {
        "non_numerical_layouts_admitted": len(cases),
        "non_numerical_layout_mutations_rejected": len(mutations),
        "non_numerical_layout_diagnoses": [label for label, _arm, _ in mutations],
        "projection_receipt_pair_layouts_admitted": len(cases),
        "projection_receipt_pair_mutations_rejected": receipt_pair_rejections,
        "projection_receipt_diagnoses": receipt_diagnoses,
        "runner_sibling_projection_round_layouts_admitted": 1,
        "verifier_sibling_projection_round_layouts_admitted": 1,
        "runner_sibling_projection_round_mutations_rejected": len(
            sibling_receipt_adversaries
        ),
        "verifier_sibling_projection_round_mutations_rejected": len(
            sibling_receipt_adversaries
        ),
        "sibling_projection_round_diagnoses": [
            label for label, _mutate in sibling_receipt_adversaries
        ],
        "zero_sharing_sidecar_layouts_admitted": 4,
        "zero_sharing_complete_controls_admitted": 2,
        "runner_zero_sharing_sidecar_mutations_rejected": (
            runner_zero_sidecar_rejections
        ),
        "verifier_zero_sharing_sidecar_mutations_rejected": (
            verifier_zero_sidecar_rejections
        ),
        "zero_sharing_sidecar_diagnoses": [
            label for label, _mutate in zero_sidecar_adversaries
        ],
    }


def _verify_cross_arm_semantic_normalization() -> dict[str, int]:
    operation_counts = {
        "serial_rope_operations": 0,
        "serial_diff_operations": 0,
        "collective_rope_operations": 24,
        "collective_diff_operations": 1,
        "position_recovery_serial_operations": 0,
        "position_recovery_collective_operations": 24,
    }
    forward_group = {
        "call_index": 0,
        "plan_id": "forward-plan",
        "reuse_group_id": "forward-reuse",
        "num_layers": 24,
        "engine_request_order": [f"{index}-{index:08x}" for index in range(1, 5)],
        "members": [
            {
                "trace_request_id": f"trace-{index}",
                "external_request_id": str(index),
                "engine_request_id": f"{index}-{index:08x}",
                "generation": 1,
                "outcome": "recovered",
                "selected_spans": {"spans": [[0, 440]]},
                "retrieval_mask": {"true_count": 440},
                "operation_counts": copy.deepcopy(operation_counts),
                "error": None,
            }
            for index in range(1, 5)
        ],
    }
    forward = {
        "arm": "collective_forward",
        "recovery_evidence": {"groups": [forward_group]},
    }
    reverse = copy.deepcopy(forward)
    reverse["arm"] = "collective_reverse"
    reverse_group = reverse["recovery_evidence"]["groups"][0]
    reverse_group["plan_id"] = "reverse-plan"
    reverse_group["reuse_group_id"] = "reverse-reuse"
    reverse_group["engine_request_order"] = [
        f"{index}-deadbeef" for index in range(1, 5)
    ]
    for index, member in enumerate(reverse_group["members"], start=1):
        member["external_request_id"] = str(5 - index)
        member["engine_request_id"] = f"{index}-deadbeef"
    if _semantic_group(forward) != _semantic_group(reverse):
        raise AssertionError("arm-local IDs leaked into semantic group comparison")

    mutations: list[tuple[str, Callable[[dict[str, object]], None]]] = [
        (
            "mask",
            lambda value: value["recovery_evidence"]["groups"][0]["members"][0][
                "retrieval_mask"
            ].update(true_count=439),
        ),
        (
            "span",
            lambda value: value["recovery_evidence"]["groups"][0]["members"][0][
                "selected_spans"
            ].update(spans=[[1, 440]]),
        ),
        (
            "outcome",
            lambda value: value["recovery_evidence"]["groups"][0]["members"][0].update(
                outcome="failed"
            ),
        ),
    ]
    expected = _semantic_group(forward)
    for label, mutate in mutations:
        changed = copy.deepcopy(reverse)
        mutate(changed)
        if _semantic_group(changed) == expected:
            raise AssertionError(f"semantic group ignored a {label} mutation")
    return {
        "cross_arm_local_id_normalizations_admitted": 1,
        "cross_arm_semantic_mutations_rejected": len(mutations),
    }


def _verify_numerical_acceptance_policy() -> dict[str, int]:
    def family(
        *,
        kv_value: float = 0.0,
        logits: tuple[float, ...] = (0.0, 2.0, 1.0),
        output_token_id: int = 1,
    ) -> dict[str, dict[str, object]]:
        key = torch.tensor([kv_value, kv_value + 0.25], dtype=torch.bfloat16)
        value = torch.tensor([kv_value + 0.5, kv_value + 0.75], dtype=torch.bfloat16)
        return {
            "trace-1": {
                "key": key,
                "value": value,
                "raw_logits": torch.tensor(logits, dtype=torch.float32),
                "output_token_id": output_token_id,
            }
        }

    exact = _compare_numerical_families(family(), family(), family())
    if (
        exact["lockstep_primary_pass"] is not True
        or exact["ordinary_pic_diagnostic_pass"] is not True
        or exact["ordinary_pic_claim_accepted"] is not False
    ):
        raise AssertionError("all-pass numerical policy was rejected")

    ordinary_drift = _compare_numerical_families(
        family(kv_value=2.0),
        family(),
        family(),
    )
    if (
        ordinary_drift["lockstep_primary_pass"] is not True
        or ordinary_drift["ordinary_pic_diagnostic_pass"] is not False
    ):
        raise AssertionError("ordinary-only numerical drift became terminal")
    ordinary_key_metrics = ordinary_drift["members"]["trace-1"]["key"]
    if not {
        "ordinary_serial_within_tolerance",
        "ordinary_serial_delta",
        "ordinary_collective_within_tolerance",
        "ordinary_collective_delta",
    }.issubset(ordinary_key_metrics):
        raise AssertionError("ordinary/PIC K/V diagnostics were not preserved")

    kv_failure = _compare_numerical_families(
        family(),
        family(),
        family(kv_value=1.0),
    )
    if kv_failure["lockstep_primary_pass"] is not False:
        raise AssertionError("serial/collective K/V drift was admitted")

    signed_zero_collective = family()
    signed_zero_collective["trace-1"]["key"] = torch.tensor(
        [-0.0, 0.25],
        dtype=torch.bfloat16,
    )
    signed_zero_failure = _compare_numerical_families(
        family(),
        family(),
        signed_zero_collective,
    )
    signed_zero_metrics = signed_zero_failure["members"]["trace-1"]["key"]
    raw_bits = signed_zero_metrics["serial_collective_raw_bits"]
    numerical_delta = signed_zero_metrics["serial_collective_delta"]
    if (
        signed_zero_failure["lockstep_primary_pass"] is not False
        or signed_zero_metrics["serial_collective_exact"] is not False
        or raw_bits["bitwise_equal"] is not False
        or raw_bits["raw_bit_mismatch_count"] != 1
        or numerical_delta["mismatch_count"] != 0
        or numerical_delta["max_abs"] != 0.0
    ):
        raise AssertionError("signed-zero BF16 storage drift was admitted")

    logits_failure = _compare_numerical_families(
        family(),
        family(),
        family(logits=(0.0, 2.1, 1.0)),
    )
    if logits_failure["lockstep_primary_pass"] is not False:
        raise AssertionError("serial/collective raw-logit drift was admitted")

    serial_near_tie = family(
        logits=(1.0, 1.00005),
        output_token_id=1,
    )
    collective_near_tie = family(
        logits=(1.00006, 1.00005),
        output_token_id=0,
    )
    output_failure = _compare_numerical_families(
        serial_near_tie,
        serial_near_tie,
        collective_near_tie,
    )
    output_metrics = output_failure["members"]["trace-1"]["raw_logits"]
    if (
        output_metrics["serial_collective_within_tolerance"] is not True
        or output_metrics["serial_collective_top1_and_generated_tokens_are_exact"]
        is not False
        or output_failure["lockstep_primary_pass"] is not False
    ):
        raise AssertionError("serial/collective output drift was admitted")

    ordinary_output_diagnostic = _compare_numerical_families(
        collective_near_tie,
        serial_near_tie,
        serial_near_tie,
    )
    diagnostic_metrics = ordinary_output_diagnostic["members"]["trace-1"]["raw_logits"]
    if (
        ordinary_output_diagnostic["lockstep_primary_pass"] is not True
        or ordinary_output_diagnostic["ordinary_pic_diagnostic_pass"] is not False
        or diagnostic_metrics["top1_and_generated_tokens_are_exact"] is not False
    ):
        raise AssertionError("ordinary/PIC output drift became terminal")

    output_policy = {
        "lockstep_terminal_by_arm_against_serial_oracle": {
            arm: True
            for arm in (
                "serial_oracle",
                "collective_forward",
                "collective_reverse",
            )
        },
        "lockstep_terminal_pass": True,
        "ordinary_pic_diagnostic_by_arm_against_ordinary": {
            arm: False
            for arm in (
                "serial_oracle",
                "collective_forward",
                "collective_reverse",
            )
        },
        "ordinary_pic_output_diagnostic_pass": False,
        "control_terminal_by_arm_against_ordinary": {
            arm: True
            for arm in (
                "cold_miss",
                "zero_sharing",
                "incompatible_partition",
                "sibling_failure",
            )
        },
        "control_terminal_pass": True,
    }
    claims = _build_claims(
        {arm: True for arm in ARMS},
        output_policy,
        order_stable=True,
        serial_collective_mask_parity=True,
        terminal_cleanup_pass=True,
        numerical_comparison=ordinary_drift,
    )
    if (
        claims["full_collective_lockstep_gate_pass"] is not True
        or claims["lockstep_numerical_pass"] is not True
        or claims["ordinary_pic_diagnostic_pass"] is not False
        or claims["ordinary_pic_claim_accepted"] is not False
        or "numerical_parity_pass" in claims
    ):
        raise AssertionError("ordinary/PIC diagnostics leaked into full acceptance")
    return {
        "numerical_acceptance_contracts_admitted": 2,
        "ordinary_only_numerical_drifts_nonterminal": 1,
        "terminal_numerical_failures_rejected": 4,
        "signed_zero_raw_bit_adversaries_rejected": 1,
        "ordinary_pic_output_diagnostics_nonterminal": 1,
        "full_gate_diagnostic_exclusions_verified": 1,
    }


def _verify_cross_arm_output_contract() -> dict[str, int]:
    def observations(
        token: int,
        sequences: tuple[int, ...] = (0, 1, 2, 3, 4),
    ) -> list[dict[str, object]]:
        return [
            {
                "sequence_index": sequence,
                "trace_request_id": f"trace-{sequence}",
                "output_token_ids": [token],
            }
            for sequence in sequences
        ]

    def results(*, ordinary_token: int = 7, pic_token: int = 7) -> dict[str, object]:
        return {
            "ordinary": {"observations": observations(ordinary_token)},
            "serial_oracle": {"observations": observations(pic_token)},
            "collective_forward": {"observations": observations(pic_token)},
            "collective_reverse": {
                "observations": observations(pic_token, (0, 4, 3, 2, 1))
            },
            "cold_miss": {"observations": observations(ordinary_token, (4,))},
            "zero_sharing": {
                "observations": observations(ordinary_token, (1, 2, 3, 4))
            },
            "incompatible_partition": {"observations": observations(ordinary_token)},
            "sibling_failure": {"observations": observations(ordinary_token)},
        }

    exact = _verify_cross_arm_output_parity(results())
    if (
        exact["lockstep_terminal_pass"] is not True
        or exact["ordinary_pic_output_diagnostic_pass"] is not True
        or exact["control_terminal_pass"] is not True
    ):
        raise AssertionError("exact cross-arm output contract was rejected")

    diagnostic = _verify_cross_arm_output_parity(results(ordinary_token=7, pic_token=8))
    if (
        diagnostic["lockstep_terminal_pass"] is not True
        or diagnostic["ordinary_pic_output_diagnostic_pass"] is not False
        or diagnostic["control_terminal_pass"] is not True
    ):
        raise AssertionError("ordinary/PIC output drift became terminal")

    lockstep_drift = results()
    lockstep_drift["collective_forward"]["observations"][0]["output_token_ids"] = [8]
    _expect_runtime_error(
        lambda: _verify_cross_arm_output_parity(lockstep_drift),
        "serial/collective output drift",
    )

    trace_permutation = results()
    reverse_observations = trace_permutation["collective_reverse"]["observations"]
    reverse_observations[2]["trace_request_id"] = "trace-2"
    reverse_observations[3]["trace_request_id"] = "trace-3"
    _expect_runtime_error(
        lambda: _verify_cross_arm_output_parity(trace_permutation),
        "sequence-to-trace permutation",
    )

    control_arms = (
        "cold_miss",
        "zero_sharing",
        "incompatible_partition",
        "sibling_failure",
    )
    for arm in control_arms:
        changed = results()
        changed[arm]["observations"][0]["output_token_ids"] = [8]
        _expect_runtime_error(
            lambda changed=changed: _verify_cross_arm_output_parity(changed),
            f"{arm} output drift",
        )
    return {
        "cross_arm_output_contracts_admitted": 2,
        "ordinary_pic_output_diagnostics_nonterminal": 1,
        "lockstep_output_mutations_rejected": 2,
        "control_output_mutations_rejected": len(control_arms),
    }


def _verify_lookup_claim_matrix() -> dict[str, int]:
    def build(
        arm: str,
    ) -> tuple[
        list[dict[str, object]],
        list[dict[str, object]],
        list[dict[str, object]],
    ]:
        if arm == "zero_sharing":
            sequences = [1, 2, 3, 4]
            phases = [arm] * 4
        else:
            sequences = [0, 1, 2, 3, 4]
            phases = ["prime", *(arm for _ in range(4))]
        observations = []
        bindings = []
        events = []
        for index, (sequence, phase) in enumerate(zip(sequences, phases, strict=True)):
            positive = (arm == "serial_oracle" and sequence in {1, 2, 3, 4}) or (
                arm == "incompatible_partition" and sequence in {1, 2}
            )
            cached = 440 if positive else 0
            trace_id = f"trace-{sequence}"
            engine_id = f"{index}-{index:08x}"
            observations.append(
                {
                    "phase": phase,
                    "sequence_index": sequence,
                    "trace_request_id": trace_id,
                    "num_cached_tokens": cached,
                }
            )
            bindings.append(
                {
                    "trace_request_id": trace_id,
                    "engine_request_id": engine_id,
                }
            )
            events.append(
                {
                    "phase": phase,
                    "action": (
                        "LMCacheConnectorV1Impl.get_num_new_matched_tokens.begin"
                    ),
                    "request_id": engine_id,
                    "owner_type": "LMCacheConnectorV1Impl",
                    "role": "SCHEDULER",
                    "num_computed_tokens": 0,
                }
            )
            if arm != "zero_sharing":
                lookup_id = f"00000000-0000-4000-8000-{index + 1:012x}"
                events.extend(
                    [
                        {
                            "phase": phase,
                            "action": "LMCacheLookupClient.lookup.begin",
                            "owner_type": "LMCacheLookupClient",
                            "lookup_id": lookup_id,
                            "token_count": 512,
                        },
                        {
                            "phase": phase,
                            "action": "LMCacheLookupClient.lookup.end",
                            "owner_type": "LMCacheLookupClient",
                            "lookup_id": lookup_id,
                            "token_count": 512,
                            "result": cached,
                        },
                    ]
                )
            events.append(
                {
                    "phase": phase,
                    "action": ("LMCacheConnectorV1Impl.get_num_new_matched_tokens.end"),
                    "request_id": engine_id,
                    "owner_type": "LMCacheConnectorV1Impl",
                    "role": "SCHEDULER",
                    "num_computed_tokens": 0,
                    "result": cached,
                }
            )
        for sequence, event in enumerate(events):
            event["sequence"] = sequence
        return observations, bindings, events

    exact_cases = {
        arm: build(arm)
        for arm in ("serial_oracle", "zero_sharing", "incompatible_partition")
    }
    for arm, (observations, bindings, events) in exact_cases.items():
        _verify_lookup_claims(
            Path(f"{arm}-lookup-exact"), arm, observations, bindings, events
        )

    positive_to_zero = copy.deepcopy(exact_cases["serial_oracle"])
    positive_to_zero[0][1]["num_cached_tokens"] = 0
    [
        event
        for event in positive_to_zero[2]
        if event["action"] == "LMCacheLookupClient.lookup.end"
    ][1]["result"] = 0
    [
        event
        for event in positive_to_zero[2]
        if event["action"] == "LMCacheConnectorV1Impl.get_num_new_matched_tokens.end"
    ][1]["result"] = 0

    zero_extra_lookup = copy.deepcopy(exact_cases["zero_sharing"])
    zero_extra_lookup[2][1:1] = [
        {
            "phase": "zero_sharing",
            "action": "LMCacheLookupClient.lookup.begin",
            "owner_type": "LMCacheLookupClient",
            "lookup_id": "00000000-0000-4000-8000-000000000099",
            "token_count": 512,
        },
        {
            "phase": "zero_sharing",
            "action": "LMCacheLookupClient.lookup.end",
            "owner_type": "LMCacheLookupClient",
            "lookup_id": "00000000-0000-4000-8000-000000000099",
            "token_count": 512,
            "result": 0,
        },
    ]
    for sequence, event in enumerate(zero_extra_lookup[2]):
        event["sequence"] = sequence

    incompatible_b_positive = copy.deepcopy(exact_cases["incompatible_partition"])
    incompatible_b_positive[0][3]["num_cached_tokens"] = 440
    [
        event
        for event in incompatible_b_positive[2]
        if event["action"] == "LMCacheLookupClient.lookup.end"
    ][3]["result"] = 440
    [
        event
        for event in incompatible_b_positive[2]
        if event["action"] == "LMCacheConnectorV1Impl.get_num_new_matched_tokens.end"
    ][3]["result"] = 440

    for label, arm, changed in (
        ("positive lookup changed to zero", "serial_oracle", positive_to_zero),
        ("zero-sharing extra lookup", "zero_sharing", zero_extra_lookup),
        (
            "incompatible partition B changed positive",
            "incompatible_partition",
            incompatible_b_positive,
        ),
    ):
        observations, bindings, events = changed
        _expect_runtime_error(
            lambda arm=arm, observations=observations, bindings=bindings, events=events: _verify_lookup_claims(
                Path(label), arm, observations, bindings, events
            ),
            label,
        )
    return {
        "lookup_claim_matrices_admitted": len(exact_cases),
        "lookup_claim_mutations_rejected": 3,
    }


def _verify_capture_batch_row_contract() -> dict[str, int]:
    sequence_by_engine = {
        f"{sequence}-{sequence:08x}": sequence for sequence in range(1, 5)
    }
    sequence_to_row = {1: 2, 2: 0, 3: 3, 4: 1}
    batch_engine_request_ids = tuple(
        next(
            engine_request_id
            for engine_request_id, sequence in sequence_by_engine.items()
            if sequence_to_row[sequence] == row
        )
        for row in range(4)
    )

    instrumentation = _instrumentation()
    instrumentation._engine_to_sequence = copy.deepcopy(sequence_by_engine)
    input_batch = SimpleNamespace(
        req_ids=list(batch_engine_request_ids),
        num_reqs=4,
        num_draft_tokens=0,
        cu_num_logits_np=SimpleNamespace(tolist=lambda: [0, 1, 2, 3, 4]),
        has_structured_output_reqs=False,
        prefill_len_np=[512, 512, 512, 512],
        is_prefilling_np=[True, True, True, True],
    )
    expected_rows = [
        (row, engine_request_id, sequence_by_engine[engine_request_id])
        for row, engine_request_id in enumerate(batch_engine_request_ids)
    ]
    if instrumentation._target_rows(input_batch) != expected_rows:
        raise AssertionError(
            "non-identity InputBatch row permutation was not preserved"
        )

    captured = {
        sequence: SimpleNamespace(
            binding=SimpleNamespace(sequence_index=sequence),
            capture_batch_row=sequence_to_row[sequence],
            engine_request_id=f"{sequence}-{sequence:08x}",
            batch_engine_request_ids=batch_engine_request_ids,
        )
        for sequence in range(1, 5)
    }
    if (
        NumericalInstrumentation._validate_capture_batch_ownership(captured)
        != batch_engine_request_ids
    ):
        raise AssertionError("capture ownership changed the physical batch row order")

    def expect_capture_error(call: Callable[[], object], label: str) -> None:
        try:
            call()
        except NumericalCaptureError:
            return
        raise AssertionError(f"numerical capture admitted row drift: {label}")

    duplicate_input_row = copy.deepcopy(input_batch)
    duplicate_input_row.req_ids[3] = duplicate_input_row.req_ids[2]
    nonbijective_mapping = _instrumentation()
    nonbijective_mapping._engine_to_sequence = copy.deepcopy(sequence_by_engine)
    nonbijective_mapping._engine_to_sequence[batch_engine_request_ids[0]] = 1
    target_row_mutations = (
        (
            "duplicate InputBatch engine owner",
            lambda: instrumentation._target_rows(duplicate_input_row),
        ),
        (
            "non-bijective engine-to-sequence map",
            lambda: nonbijective_mapping._target_rows(input_batch),
        ),
    )
    for label, call in target_row_mutations:
        expect_capture_error(call, label)

    duplicate_capture_row = copy.deepcopy(captured)
    duplicate_capture_row[1].capture_batch_row = duplicate_capture_row[
        2
    ].capture_batch_row
    swapped_capture_rows = copy.deepcopy(captured)
    (
        swapped_capture_rows[1].capture_batch_row,
        swapped_capture_rows[2].capture_batch_row,
    ) = (
        swapped_capture_rows[2].capture_batch_row,
        swapped_capture_rows[1].capture_batch_row,
    )
    differing_batch_identity = copy.deepcopy(captured)
    differing_batch_identity[1].batch_engine_request_ids = tuple(
        reversed(batch_engine_request_ids)
    )
    duplicated_batch_engine = copy.deepcopy(captured)
    duplicated_identity = list(batch_engine_request_ids)
    duplicated_identity[0] = duplicated_identity[1]
    for member in duplicated_batch_engine.values():
        member.batch_engine_request_ids = tuple(duplicated_identity)
    changed_trace_owner = copy.deepcopy(captured)
    changed_trace_owner[1].binding.sequence_index = 2
    finish_mutations = (
        ("duplicate capture row", duplicate_capture_row),
        ("row/engine ownership swap", swapped_capture_rows),
        ("per-member batch identity", differing_batch_identity),
        ("duplicate batch engine owner", duplicated_batch_engine),
        ("capture trace-sequence owner", changed_trace_owner),
    )
    for label, changed in finish_mutations:
        expect_capture_error(
            lambda changed=changed: NumericalInstrumentation._validate_capture_batch_ownership(
                changed
            ),
            label,
        )
    return {
        "capture_row_permutations_admitted": 2,
        "capture_row_mutations_rejected": len(target_row_mutations)
        + len(finish_mutations),
    }


def _verify_numerical_event_contract() -> dict[str, int]:
    arm = "ordinary"
    phase = "ordinary_batch_forward"
    manifest_path = "post-pic-capture-v1/family-manifest.json"
    manifest_sha = "f" * 64
    numerical = {
        "enabled": True,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha,
    }
    capture_row_by_sequence = {1: 2, 2: 0, 3: 3, 4: 1}
    engine_id_by_sequence = {
        sequence: f"{sequence}-{sequence:08x}" for sequence in range(1, 5)
    }
    batch_engine_request_ids = [
        engine_id_by_sequence[
            next(
                sequence
                for sequence, capture_row in capture_row_by_sequence.items()
                if capture_row == row
            )
        ]
        for row in range(4)
    ]
    members = []
    events = []
    for sequence in range(1, 5):
        trace_id = f"trace-{sequence}"
        engine_id = engine_id_by_sequence[sequence]
        tensor_sha = f"{sequence:064x}"
        logits_sha = f"{sequence + 8:064x}"
        members.append(
            {
                "sequence_index": sequence,
                "trace_request_id": trace_id,
                "engine_request_id": engine_id,
                "capture_batch_row": capture_row_by_sequence[sequence],
                "batch_engine_request_ids": batch_engine_request_ids,
                "physical_block_ids": [sequence],
                "tensor_file": {
                    "path": f"members/{sequence:03d}/kv.safetensors",
                    "sha256": tensor_sha,
                    "nbytes": 100 + sequence,
                },
                "logits_file": {
                    "path": f"members/{sequence:03d}/logits.safetensors",
                    "sha256": logits_sha,
                    "nbytes": 200 + sequence,
                },
            }
        )
    sequence_by_engine_id = {
        engine_request_id: sequence
        for sequence, engine_request_id in engine_id_by_sequence.items()
    }
    for capture_batch_row, engine_id in enumerate(batch_engine_request_ids):
        sequence = sequence_by_engine_id[engine_id]
        member = members[sequence - 1]
        events.append(
            {
                "phase": phase,
                "action": "gate.numerical.kv_captured",
                "arm": arm,
                "sequence_index": sequence,
                "trace_request_id": member["trace_request_id"],
                "engine_request_id": engine_id,
                "capture_batch_row": capture_batch_row,
                "physical_block_ids": [sequence],
                "tensor_path": (
                    f"post-pic-capture-v1/members/{sequence:03d}/kv.safetensors"
                ),
                "tensor_sha256": member["tensor_file"]["sha256"],
            }
        )
    for capture_batch_row, engine_id in enumerate(batch_engine_request_ids):
        sequence = sequence_by_engine_id[engine_id]
        member = members[sequence - 1]
        events.append(
            {
                "phase": phase,
                "action": "gate.numerical.logits_captured",
                "arm": arm,
                "sequence_index": sequence,
                "trace_request_id": member["trace_request_id"],
                "engine_request_id": engine_id,
                "capture_batch_row": capture_batch_row,
                "vocab_size": 151936,
                "logits_path": (
                    f"post-pic-capture-v1/members/{sequence:03d}/logits.safetensors"
                ),
                "logits_sha256": member["logits_file"]["sha256"],
            }
        )
    events.append(
        {
            "phase": phase,
            "action": "gate.numerical.finish",
            "arm": arm,
            "status": "complete",
            "manifest_path": manifest_path,
            "manifest_sha256": manifest_sha,
            "member_count": 4,
        }
    )
    for sequence, event in enumerate(events):
        event["sequence"] = sequence
    manifest = {"capture_phase": phase, "members": members}
    _verify_numerical_event_bindings(
        Path("numerical-events-exact"), arm, numerical, manifest, events
    )

    swapped = copy.deepcopy((numerical, manifest, events))
    swapped[1]["members"][0], swapped[1]["members"][1] = (
        swapped[1]["members"][1],
        swapped[1]["members"][0],
    )
    _expect_runtime_error(
        lambda: _verify_numerical_event_bindings(
            Path("numerical-member-swap"), arm, *swapped
        ),
        "numerical event/member swap",
    )

    duplicate_row = copy.deepcopy((numerical, manifest, events))
    duplicate_row[1]["members"][0]["capture_batch_row"] = duplicate_row[1]["members"][
        1
    ]["capture_batch_row"]

    coherent_row_swap = copy.deepcopy((numerical, manifest, events))
    first_row = coherent_row_swap[1]["members"][0]["capture_batch_row"]
    second_row = coherent_row_swap[1]["members"][1]["capture_batch_row"]
    coherent_row_swap[1]["members"][0]["capture_batch_row"] = second_row
    coherent_row_swap[1]["members"][1]["capture_batch_row"] = first_row
    for event in coherent_row_swap[2]:
        if event.get("action") in {
            "gate.numerical.kv_captured",
            "gate.numerical.logits_captured",
        }:
            if event.get("sequence_index") == 1:
                event["capture_batch_row"] = second_row
            elif event.get("sequence_index") == 2:
                event["capture_batch_row"] = first_row

    event_row_drift = copy.deepcopy((numerical, manifest, events))
    next(
        event
        for event in event_row_drift[2]
        if event.get("action") == "gate.numerical.logits_captured"
        and event.get("sequence_index") == 1
    )["capture_batch_row"] = 0

    member_batch_drift = copy.deepcopy((numerical, manifest, events))
    member_batch_drift[1]["members"][0]["batch_engine_request_ids"] = list(
        reversed(batch_engine_request_ids)
    )

    row_engine_drift = copy.deepcopy((numerical, manifest, events))
    row_engine_drift[1]["members"][0]["engine_request_id"] = row_engine_drift[1][
        "members"
    ][1]["engine_request_id"]
    boolean_member_sequence = copy.deepcopy((numerical, manifest, events))
    boolean_member_sequence[1]["members"][0]["sequence_index"] = True
    for label, changed in (
        ("duplicate numerical capture row", duplicate_row),
        ("coherent row swap with stale batch tuple", coherent_row_swap),
        ("event/member capture-row mismatch", event_row_drift),
        ("per-member batch engine tuple", member_batch_drift),
        ("row/engine identity mismatch", row_engine_drift),
        ("boolean member sequence", boolean_member_sequence),
    ):
        _expect_runtime_error(
            lambda changed=changed, label=label: _verify_numerical_event_bindings(
                Path(label), arm, *changed
            ),
            label,
        )

    disabled = {"enabled": False}
    _verify_numerical_event_bindings(
        Path("numerical-disabled-exact"), "cold_miss", disabled, None, []
    )
    _expect_runtime_error(
        lambda: _verify_numerical_event_bindings(
            Path("numerical-disabled-event"),
            "cold_miss",
            disabled,
            None,
            [events[0]],
        ),
        "disabled numerical capture event",
    )
    return {
        "numerical_event_contracts_admitted": 2,
        "numerical_event_mutations_rejected": 8,
    }


def _verify_layer_operation_contract() -> dict[str, int]:
    observations = [
        {
            "phase": "prime" if sequence == 0 else "serial_oracle",
            "trace_request_id": f"trace-{sequence}",
            "num_cached_tokens": 0 if sequence == 0 else 440,
        }
        for sequence in range(5)
    ]
    bindings = [
        {
            "trace_request_id": f"trace-{sequence}",
            "engine_request_id": f"{sequence}-{sequence:08x}",
        }
        for sequence in range(5)
    ]
    lookup_by_engine = {
        binding["engine_request_id"]: (f"00000000-0000-4000-8000-{sequence + 1:012x}")
        for sequence, binding in enumerate(bindings)
    }
    events: list[dict[str, object]] = [
        {
            "phase": "prime",
            "action": "LMCacheEngine.store_layer.begin",
            "request_id": bindings[0]["engine_request_id"],
            "explicit_spans": _layout_spans(),
        },
        {
            "phase": "prime",
            "action": "LMCacheEngine.store_layer.closed",
            "request_id": bindings[0]["engine_request_id"],
            "explicit_spans": _layout_spans(),
        },
        {
            "phase": "prime",
            "action": "_KMeshLayerwiseStoreSession.finish.end",
        },
    ]
    for sequence in range(1, 5):
        lookup_id = lookup_by_engine[bindings[sequence]["engine_request_id"]]
        events.extend(
            [
                {
                    "phase": "serial_oracle",
                    "action": "LMCacheEngine.retrieve_layer.begin",
                    "lookup_id": lookup_id,
                },
                {
                    "phase": "serial_oracle",
                    "action": "LMCacheEngine.retrieve_layer.closed",
                    "lookup_id": lookup_id,
                },
            ]
        )
    for sequence, event in enumerate(events):
        event["sequence"] = sequence
    exact = (observations, bindings, lookup_by_engine, events)
    _verify_layer_operation_claims(
        Path("layer-operations-exact"), "serial_oracle", *exact
    )

    no_retrieves = copy.deepcopy(exact)
    no_retrieves[3][:] = [
        event
        for event in no_retrieves[3]
        if not str(event["action"]).startswith("LMCacheEngine.retrieve_layer.")
    ]
    no_store = copy.deepcopy(exact)
    no_store[3][:] = [
        event
        for event in no_store[3]
        if not str(event["action"]).startswith("LMCacheEngine.store_layer.")
    ]
    for label, changed in (
        ("missing retrieve layer calls", no_retrieves),
        ("missing prime store layer call", no_store),
    ):
        _expect_runtime_error(
            lambda changed=changed: _verify_layer_operation_claims(
                Path(label), "serial_oracle", *changed
            ),
            label,
        )
    return {
        "layer_operation_contracts_admitted": 1,
        "layer_operation_mutations_rejected": 2,
    }


def _verify_sibling_lookup_contract() -> dict[str, int]:
    observations = [
        {
            "trace_request_id": f"trace-{sequence}",
            "num_cached_tokens": 440,
        }
        for sequence in range(1, 5)
    ]
    members = [
        {
            "trace_request_id": f"trace-{sequence}",
            "engine_request_id": f"{sequence}-{sequence:08x}",
            "outcome": "failed" if sequence == 2 else "recovered",
        }
        for sequence in range(1, 5)
    ]
    lookup_by_engine = {
        member["engine_request_id"]: (f"00000000-0000-4000-8000-{sequence:012x}")
        for sequence, member in enumerate(members, start=1)
    }
    failed = members[1]
    failure_details = {
        "phase": "sibling_failure",
        "request_id": failed["engine_request_id"],
        "lookup_id": lookup_by_engine[failed["engine_request_id"]],
        "contract_slot_count": 512,
        "advertised_tokens": 440,
    }
    events = [
        {
            "action": "LMCacheConnectorV1Impl._record_kmesh_load_failure.begin",
            **failure_details,
        },
        {
            "action": "LMCacheConnectorV1Impl._record_kmesh_load_failure.end",
            **failure_details,
        },
    ]
    _verify_sibling_failure_lookup_claim(
        Path("sibling-lookup-exact"),
        observations,
        members,
        lookup_by_engine,
        events,
    )
    swapped = copy.deepcopy(events)
    wrong_lookup_id = lookup_by_engine[members[0]["engine_request_id"]]
    for event in swapped:
        event["lookup_id"] = wrong_lookup_id
    _expect_runtime_error(
        lambda: _verify_sibling_failure_lookup_claim(
            Path("sibling-lookup-swap"),
            observations,
            members,
            lookup_by_engine,
            swapped,
        ),
        "sibling failure lookup swap",
    )
    return {
        "sibling_lookup_contracts_admitted": 1,
        "sibling_lookup_mutations_rejected": 1,
    }


def _verify_independent_ledger_bindings() -> dict[str, int]:
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def publish_case(
        root: Path,
        *,
        arm: str,
        event: dict[str, object],
        binding_internal: str,
        mutation: (
            Callable[[list[dict[str, object]], list[dict[str, object]]], None] | None
        ) = None,
    ) -> Path:
        root.mkdir()
        sequences = [0, 1, 2, 3, 4] if arm == "ordinary" else [4]
        phases = (
            ["ordinary_prime", *(["ordinary_batch_forward"] * 4)]
            if arm == "ordinary"
            else ["cold_miss"]
        )
        observations = [
            {
                "phase": phase,
                "batch_ordinal": (
                    0 if external == 0 or arm != "ordinary" else external - 1
                ),
                "external_request_id": str(external),
                "trace_request_id": f"trace-{sequence}",
                "sequence_index": sequence,
                "prompt_tokens": 512,
                "num_cached_tokens": 0,
                "output_token_ids": [100 + sequence],
                "finish_reason": "length",
            }
            for external, (sequence, phase) in enumerate(
                zip(sequences, phases, strict=True)
            )
        ]
        action = (
            "gate.numerical.request_bound"
            if arm == "ordinary"
            else "LMCacheConnectorV1Impl.prepare_request_lookup.begin"
        )
        bindings = [
            {
                "trace_request_id": observation["trace_request_id"],
                "external_request_id": observation["external_request_id"],
                "engine_request_id": (
                    binding_internal if external == 0 else f"{external}-{external:08x}"
                ),
                "phase": observation["phase"],
                "binding_event_action": action,
            }
            for external, observation in enumerate(observations)
        ]
        canonical_events = []
        for binding in bindings:
            canonical = {
                "action": action,
                "phase": binding["phase"],
                ("engine_request_id" if arm == "ordinary" else "request_id"): binding[
                    "engine_request_id"
                ],
            }
            if arm == "ordinary":
                canonical["trace_request_id"] = binding["trace_request_id"]
            else:
                canonical.update(
                    owner_type="LMCacheConnectorV1Impl",
                    role="SCHEDULER",
                )
            canonical_events.append(canonical)
        canonical_events[0] = copy.deepcopy(event)

        events: list[dict[str, object]] = [
            {
                "phase": "construction",
                "action": "gate.construction.begin",
                "arm": arm,
            },
            {
                "phase": "construction",
                "action": "gate.construction.end",
                "arm": arm,
                "resolved_target_profile": (
                    None if arm == "ordinary" else "collective_batched_qkv_v1"
                ),
                "resolved_attention_backend": "TRITON_ATTN",
                "kv_transfer_config_present": arm != "ordinary",
            },
        ]
        batches = (
            [
                ("ordinary_prime", observations[:1], canonical_events[:1]),
                (
                    "ordinary_batch_forward",
                    observations[1:],
                    canonical_events[1:],
                ),
            ]
            if arm == "ordinary"
            else [("cold_miss", observations, canonical_events)]
        )
        for phase, batch, batch_canonical_events in batches:
            trace_ids = [observation["trace_request_id"] for observation in batch]
            events.append(
                {
                    "phase": phase,
                    "action": "gate.batch.begin",
                    "batch_size": len(batch),
                    "submitted_trace_request_ids": trace_ids,
                    "prompt_tokens": sum(
                        observation["prompt_tokens"] for observation in batch
                    ),
                }
            )
            events.extend(copy.deepcopy(batch_canonical_events))
            if arm == "cold_miss":
                lookup_id = "11111111-2222-4333-8444-555555555555"
                projection_receipt = {
                    "phase": phase,
                    "action": "gate.collective.plan_projection",
                    "plans": [],
                }
                events.extend(
                    [
                        projection_receipt,
                        copy.deepcopy(projection_receipt),
                        {
                            "phase": phase,
                            "action": (
                                "LMCacheConnectorV1Impl."
                                "get_num_new_matched_tokens.begin"
                            ),
                            "owner_type": "LMCacheConnectorV1Impl",
                            "role": "SCHEDULER",
                            "request_id": bindings[0]["engine_request_id"],
                            "num_computed_tokens": 0,
                        },
                        {
                            "phase": phase,
                            "action": "LMCacheLookupClient.lookup.begin",
                            "owner_type": "LMCacheLookupClient",
                            "lookup_id": lookup_id,
                            "token_count": 512,
                        },
                        {
                            "phase": phase,
                            "action": "LMCacheLookupClient.lookup.end",
                            "owner_type": "LMCacheLookupClient",
                            "lookup_id": lookup_id,
                            "token_count": 512,
                            "result": 0,
                        },
                        {
                            "phase": phase,
                            "action": (
                                "LMCacheConnectorV1Impl.get_num_new_matched_tokens.end"
                            ),
                            "owner_type": "LMCacheConnectorV1Impl",
                            "role": "SCHEDULER",
                            "request_id": bindings[0]["engine_request_id"],
                            "num_computed_tokens": 0,
                            "result": 0,
                        },
                    ]
                )
            events.extend(
                {
                    "action": "gate.request.complete",
                    **copy.deepcopy(observation),
                }
                for observation in batch
            )
            events.append(
                {
                    "phase": phase,
                    "action": "gate.batch.end",
                    "batch_size": len(batch),
                    "submitted_trace_request_ids": trace_ids,
                    "returned_external_request_ids": [
                        observation["external_request_id"] for observation in batch
                    ],
                }
            )
        if arm != "ordinary":
            worker_state = {
                field: 0
                for field in (
                    "_kmesh_store_sessions_count",
                    "_kmesh_outstanding_lookup_ids_count",
                    "_kmesh_store_cleanup_errors_count",
                    "_kmesh_failed_worker_requests_count",
                )
            }
            scheduler_state = {
                field: 0
                for field in (
                    "_prealloc_lookup_ids_count",
                    "_allocated_lookup_ids_count",
                    "_worker_owned_lookup_requests_count",
                )
            }
            for role in ("WORKER", "SCHEDULER"):
                events.append(
                    {
                        "phase": "shutdown",
                        "action": "LMCacheConnectorV1Impl.shutdown.begin",
                        "owner_type": "LMCacheConnectorV1Impl",
                        "role": role,
                    }
                )
                events.append(
                    {
                        "phase": "shutdown",
                        "action": "LMCacheConnectorV1Impl.shutdown.end",
                        "owner_type": "LMCacheConnectorV1Impl",
                        "role": role,
                        "manager_services_stopped": True,
                        **(worker_state if role == "WORKER" else scheduler_state),
                    }
                )
        events.append(
            {
                "phase": "shutdown",
                "action": "gate.shutdown.complete",
            }
        )
        if mutation is not None:
            mutation(observations, events)
        events = [
            {"schema_version": 1, "sequence": sequence, **item}
            for sequence, item in enumerate(events)
        ]
        ledger = root / "events.jsonl"
        ledger.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in events),
            encoding="utf-8",
        )
        result = {
            "schema_version": 1,
            "status": "complete",
            "arm": arm,
            "request_id_contract": VERIFIER_REQUEST_ID_CONTRACT,
            "checks": {"all_passed": True},
            "observations": observations,
            "recovery_evidence": {
                "groups": [],
                "serial_members": [],
                "request_id_bindings": bindings,
                "trace_submission_order": [
                    observation["trace_request_id"]
                    for observation in observations
                    if observation["phase"] not in {"prime", "ordinary_prime"}
                ],
            },
            "numerical_evidence": {"enabled": False, "status": "disabled"},
            "event_ledger": {"sha256": sha256(ledger), "event_count": len(events)},
        }
        result_path = root / "result.json"
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (root / "status.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "arm": arm,
                    "result_sha256": sha256(result_path),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return result_path

    with tempfile.TemporaryDirectory(prefix="kmesh-id-verifier-") as temporary:
        root = Path(temporary)
        cold_event = {
            "action": "LMCacheConnectorV1Impl.prepare_request_lookup.begin",
            "phase": "cold_miss",
            "request_id": "0-00000000",
            "owner_type": "LMCacheConnectorV1Impl",
            "role": "SCHEDULER",
        }
        ordinary_event = {
            "action": "gate.numerical.request_bound",
            "phase": "ordinary_prime",
            "engine_request_id": "0-00000000",
            "trace_request_id": "trace-0",
        }
        exact_paths = [
            publish_case(
                root / "cold-exact",
                arm="cold_miss",
                event=cold_event,
                binding_internal="0-00000000",
            ),
            publish_case(
                root / "ordinary-exact",
                arm="ordinary",
                event=ordinary_event,
                binding_internal="0-00000000",
            ),
        ]
        for path, arm in zip(exact_paths, ("cold_miss", "ordinary"), strict=True):
            _independent_load(path, arm)

        binding_mutations = [
            (
                "self-consistent-result-drift",
                "cold_miss",
                cold_event,
                "0-deadbeef",
            ),
            (
                "connector-role-drift",
                "cold_miss",
                {**cold_event, "role": "WORKER"},
                "0-00000000",
            ),
            (
                "ordinary-trace-drift",
                "ordinary",
                {**ordinary_event, "trace_request_id": "trace-wrong"},
                "0-00000000",
            ),
        ]
        rejected = 0
        for label, arm, event, binding_internal in binding_mutations:
            path = publish_case(
                root / label,
                arm=arm,
                event=event,
                binding_internal=binding_internal,
            )
            try:
                _independent_load(path, arm)
            except RuntimeError:
                rejected += 1
            else:
                raise AssertionError(
                    f"independent verifier admitted ledger drift: {label}"
                )

        evidence_mutations: list[
            tuple[
                str,
                str,
                dict[str, object],
                Callable[[list[dict[str, object]], list[dict[str, object]]], None],
            ]
        ] = [
            (
                "observation-result-differs-from-completion",
                "cold_miss",
                cold_event,
                lambda observations, _events: observations[0].update(
                    output_token_ids=[999]
                ),
            ),
            (
                "coherent-cleanup-residue",
                "cold_miss",
                cold_event,
                lambda _observations, events: next(
                    event
                    for event in events
                    if event["action"] == "LMCacheConnectorV1Impl.shutdown.end"
                    and event["role"] == "WORKER"
                ).update(_kmesh_store_sessions_count=1),
            ),
            (
                "boolean-cleanup-residue",
                "cold_miss",
                cold_event,
                lambda _observations, events: next(
                    event
                    for event in events
                    if event["action"] == "LMCacheConnectorV1Impl.shutdown.end"
                    and event["role"] == "WORKER"
                ).update(_kmesh_store_sessions_count=False),
            ),
            (
                "boolean-lookup-result",
                "cold_miss",
                cold_event,
                lambda _observations, events: (
                    next(
                        event
                        for event in events
                        if event["action"] == "LMCacheLookupClient.lookup.end"
                    ).update(result=False),
                    next(
                        event
                        for event in events
                        if event["action"]
                        == ("LMCacheConnectorV1Impl.get_num_new_matched_tokens.end")
                    ).update(result=False),
                ),
            ),
            (
                "boolean-singleton-batch-size",
                "ordinary",
                ordinary_event,
                lambda _observations, events: [
                    event.update(batch_size=True)
                    for event in events
                    if event["phase"] == "ordinary_prime"
                    and event["action"] in {"gate.batch.begin", "gate.batch.end"}
                ],
            ),
            (
                "missing-shutdown-begins",
                "cold_miss",
                cold_event,
                lambda _observations, events: events.__setitem__(
                    slice(None),
                    [
                        event
                        for event in events
                        if event["action"] != "LMCacheConnectorV1Impl.shutdown.begin"
                    ],
                ),
            ),
            (
                "coherent-cold-positive-control",
                "cold_miss",
                cold_event,
                lambda observations, events: (
                    observations[0].update(num_cached_tokens=1),
                    next(
                        event
                        for event in events
                        if event["action"] == "gate.request.complete"
                    ).update(num_cached_tokens=1),
                    next(
                        event
                        for event in events
                        if event["action"] == "LMCacheLookupClient.lookup.end"
                    ).update(result=1),
                    next(
                        event
                        for event in events
                        if event["action"]
                        == ("LMCacheConnectorV1Impl.get_num_new_matched_tokens.end")
                    ).update(result=1),
                ),
            ),
            (
                "complete-ledger-gate-failure",
                "cold_miss",
                cold_event,
                lambda _observations, events: events.insert(
                    -1,
                    {
                        "phase": "shutdown",
                        "action": "gate.failure",
                        "exception_type": "RuntimeError",
                        "message": "synthetic prior failure",
                    },
                ),
            ),
            (
                "duplicate-terminal-shutdown",
                "cold_miss",
                cold_event,
                lambda _observations, events: events.append(
                    {"phase": "shutdown", "action": "gate.shutdown.complete"}
                ),
            ),
        ]
        for label, arm, event, mutation in evidence_mutations:
            path = publish_case(
                root / label,
                arm=arm,
                event=event,
                binding_internal="0-00000000",
                mutation=mutation,
            )
            try:
                _independent_load(path, arm)
            except RuntimeError:
                rejected += 1
            else:
                raise AssertionError(
                    f"independent verifier admitted claim drift: {label}"
                )
    return {
        "independent_ledger_bindings_admitted": len(exact_paths),
        "independent_ledger_mutations_rejected": rejected,
    }


def _verify_safetensor_publication() -> dict[str, object]:
    import torch
    from safetensors.torch import load_file

    with tempfile.TemporaryDirectory(prefix="kmesh-safetensor-mode-") as temporary:
        path = Path(temporary) / "probe.safetensors"
        expected = torch.arange(4, dtype=torch.float32)
        _atomic_safetensors(path, {"probe": expected})
        mode = path.stat().st_mode & 0o777
        loaded = load_file(str(path), device="cpu")
        if mode != 0o644 or not torch.equal(loaded["probe"], expected):
            raise AssertionError("atomic safetensor mode or content drifted")
    return {
        "safetensor_publication_mode": "0644",
        "safetensor_round_trips": 1,
    }


def main() -> dict[str, object]:
    exact = _warmup_output()
    instrumentation = _instrumentation()
    instrumentation._observe_new_requests(_runner(), exact)
    if instrumentation._engine_to_sequence or instrumentation._sequence_to_engine:
        raise AssertionError("kernel warmup created a frozen-trace binding")
    if instrumentation.ledger.events != [
        (
            "gate.numerical.kernel_warmup_ignored",
            {
                "request_ids": [f"_warmup_{index}_" for index in range(4)],
                "prompt_token_ids": [0, 1],
                "total_num_scheduled_tokens": 8,
            },
        )
    ]:
        raise AssertionError("kernel warmup did not emit the exact audit event")
    try:
        instrumentation._observe_new_requests(_runner(), exact)
    except NumericalCaptureError:
        pass
    else:
        raise AssertionError("a second exact kernel warmup was admitted")

    active_connector = ActiveKVConnector.__new__(ActiveKVConnector)
    active_connector._disabled = True
    if not _is_admitted(_warmup_output(), runner=_runner(active_connector)):
        raise AssertionError("the exact disabled active connector was rejected")
    active_connector._disabled = False
    if _is_admitted(_warmup_output(), runner=_runner(active_connector)):
        raise AssertionError("an enabled active connector was admitted")

    scheduler_mutations: dict[str, Callable[[SchedulerOutput], None]] = {
        "scheduled_new_reqs": lambda value: setattr(
            value, "scheduled_new_reqs", list(reversed(value.scheduled_new_reqs))
        ),
        "scheduled_cached_reqs": lambda value: setattr(
            value, "scheduled_cached_reqs", None
        ),
        "num_scheduled_tokens": lambda value: setattr(
            value,
            "num_scheduled_tokens",
            {"_warmup_0_": True, **dict(list(value.num_scheduled_tokens.items())[1:])},
        ),
        "total_num_scheduled_tokens": lambda value: setattr(
            value, "total_num_scheduled_tokens", True
        ),
        "scheduled_spec_decode_tokens": lambda value: setattr(
            value, "scheduled_spec_decode_tokens", {"x": []}
        ),
        "scheduled_encoder_inputs": lambda value: setattr(
            value, "scheduled_encoder_inputs", {"x": []}
        ),
        "num_common_prefix_blocks": lambda value: setattr(
            value, "num_common_prefix_blocks", [False]
        ),
        "finished_req_ids": lambda value: setattr(value, "finished_req_ids", {"x"}),
        "free_encoder_mm_hashes": lambda value: setattr(
            value, "free_encoder_mm_hashes", ["x"]
        ),
        "scheduled_encoder_input_stats": lambda value: setattr(
            value, "scheduled_encoder_input_stats", object()
        ),
        "preempted_req_ids": lambda value: setattr(value, "preempted_req_ids", set()),
        "has_structured_output_requests": lambda value: setattr(
            value, "has_structured_output_requests", True
        ),
        "pending_structured_output_tokens": lambda value: setattr(
            value, "pending_structured_output_tokens", True
        ),
        "num_invalid_spec_tokens": lambda value: setattr(
            value, "num_invalid_spec_tokens", {}
        ),
        "kv_connector_metadata": lambda value: setattr(
            value, "kv_connector_metadata", object()
        ),
        "ec_connector_metadata": lambda value: setattr(
            value, "ec_connector_metadata", object()
        ),
        "ec_manager_metadata": lambda value: setattr(
            value, "ec_manager_metadata", object()
        ),
        "new_block_ids_to_zero": lambda value: setattr(
            value, "new_block_ids_to_zero", []
        ),
        "kv_cache_block_copies": lambda value: setattr(
            value, "kv_cache_block_copies", []
        ),
        "partial_tail_offloads": lambda value: setattr(
            value, "partial_tail_offloads", {}
        ),
        "num_spec_tokens_to_schedule": lambda value: setattr(
            value, "num_spec_tokens_to_schedule", False
        ),
    }
    for label, mutate in scheduler_mutations.items():
        _assert_rejected(_mutated(mutate), f"SchedulerOutput.{label}")

    cached_mutations: dict[str, object] = {
        "req_ids": ["x"],
        "resumed_req_ids": {"x"},
        "new_token_ids": [[]],
        "all_token_ids": {"x": []},
        "new_block_ids": [None],
        "num_computed_tokens": [False],
        "num_output_tokens": [False],
    }
    for field_name, replacement in cached_mutations.items():
        output = _warmup_output()
        setattr(output.scheduled_cached_reqs, field_name, replacement)
        _assert_rejected(output, f"CachedRequestData.{field_name}")

    request_mutations: dict[str, object] = {
        "req_id": "_warmup_spoof_",
        "prompt_token_ids": [False, 1],
        "prefill_token_ids": [False, 1],
        "mm_features": [object()],
        "sampling_params": None,
        "pooling_params": object(),
        "block_ids": ([True],),
        "num_computed_tokens": False,
        "lora_request": object(),
        "prompt_embeds": object(),
        "prompt_is_token_ids": [],
    }
    for field_name, replacement in request_mutations.items():
        output = _warmup_output()
        setattr(output.scheduled_new_reqs[0], field_name, replacement)
        _assert_rejected(output, f"NewRequestData.{field_name}")

    output = _warmup_output()
    altered_sampling = copy.deepcopy(output.scheduled_new_reqs[0].sampling_params)
    altered_sampling.max_tokens = 999
    output.scheduled_new_reqs[0].sampling_params = altered_sampling
    _assert_rejected(output, "SamplingParams.max_tokens")

    output = _warmup_output()
    output.scheduled_new_reqs[0].req_id = "_warmup_4_"
    output.num_scheduled_tokens = {
        request.req_id: 2 for request in output.scheduled_new_reqs
    }
    _assert_rejected(output, "warmup prefix spoof")

    output = _warmup_output()
    output.scheduled_new_reqs[0].prompt_token_ids = [9]
    _assert_rejected(output, "unknown prompt")

    output = _warmup_output()
    output.scheduled_new_reqs[0].prompt_token_ids = [9]
    mixed = _instrumentation()
    mixed._records_by_digest[_prompt_digest([9])] = object()
    try:
        mixed._observe_new_requests(_runner(), output)
    except NumericalCaptureError:
        pass
    else:
        raise AssertionError("a mixed warmup/frozen-trace batch was admitted")

    return {
        "status": "pass",
        "scheduler_fields_rejected": len(scheduler_mutations),
        "cached_request_fields_rejected": len(cached_mutations),
        "new_request_fields_rejected": len(request_mutations),
        "exact_noop_admissions": 1,
        "exact_active_connector_admissions": 1,
        "second_exact_admissions": 0,
        **_verify_request_id_contract(),
        **_verify_finish_id_domains(),
        **_verify_layer_api_id_domains(),
        **_verify_non_numerical_id_layouts(),
        **_verify_cross_arm_semantic_normalization(),
        **_verify_numerical_acceptance_policy(),
        **_verify_cross_arm_output_contract(),
        **_verify_lookup_claim_matrix(),
        **_verify_capture_batch_row_contract(),
        **_verify_numerical_event_contract(),
        **_verify_layer_operation_contract(),
        **_verify_sibling_lookup_contract(),
        **_verify_independent_ledger_bindings(),
        **_verify_safetensor_publication(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = main()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(payload, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        with arguments.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(arguments.output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

# SPDX-License-Identifier: MIT
"""Check one actual group member failure after completed native restoration."""

from dataclasses import asdict
import functools

if __package__:
    from .group_capture import MODEL_COUNTS
else:
    from group_capture import MODEL_COUNTS


def serial_reference_outputs(result):
    """Validate the completed serial reference for the group failure control."""
    if (
        type(result) is not dict
        or result.get("status") != "complete"
        or result.get("kind") != "serial_model_reference"
        or result.get("model_execution") != "serial"
        or type(result.get("observations")) is not list
        or len(result["observations"]) != 5
    ):
        raise ValueError(
            "group control requires a completed five-request serial reference"
        )
    outputs = {}
    for row in result["observations"]:
        if type(row) is not dict:
            raise ValueError("serial reference observation is malformed")
        trace, tokens = row.get("trace_request_id"), row.get("output_token_ids")
        if (
            type(trace) is not str
            or not trace
            or trace in outputs
            or type(tokens) is not list
            or not tokens
            or any(type(t) is not int or not 0 <= t < 2**63 for t in tokens)
        ):
            raise ValueError("serial reference identity or generated tokens changed")
        outputs[trace] = tokens
    return outputs


def install_member_failure(ledger, session_type, blender_type):
    """Fail the second member after its first actual model layer completes."""
    state = {"target_request_id": None, "count": 0}
    original_group, original_model = blender_type.blend_group, session_type.step_model

    @functools.wraps(original_group)
    def group(owner, requests, **kwargs):
        requests = tuple(requests)
        if state["target_request_id"] is None:
            if len(requests) != 4:
                raise ValueError(
                    "member failure requires the complete four-child group"
                )
            state["target_request_id"] = requests[1].request_id
        return original_group(owner, requests, **kwargs)

    @functools.wraps(original_model)
    def model(owner, *args, **kwargs):
        result = original_model(owner, *args, **kwargs)
        if owner.request_id == state["target_request_id"] and state["count"] == 0:
            if owner.completed_layers != 1:
                raise ValueError(
                    "member failure missed its first completed model layer"
                )
            state["count"] += 1
            ledger.emit(
                "gate.sibling_failure.injected",
                request_id=owner.request_id,
                completed_layers=owner.completed_layers,
                completed_serial_operations=asdict(owner.operation_counts),
            )
            raise RuntimeError("injected member failure after the first repaired layer")
        return result

    blender_type.blend_group, session_type.step_model = group, model
    return state


def verify_member_failure(
    events, recovery, observations, ordinary, serial_reference, injection
):
    """Check member outcomes after the inherited retry-causality gate passes."""
    if len(recovery["groups"]) != 1 or recovery["serial_members"]:
        raise ValueError("member failure did not use one actual model group")
    group = recovery["groups"][0]
    members = group["members"]
    failed = [m for m in members if m["outcome"] == "failed"]
    recovered = [m for m in members if m["outcome"] == "recovered"]
    injections = [e for e in events if e["action"] == "gate.sibling_failure.injected"]
    failed_counts = MODEL_COUNTS | {
        "serial_rope_operations": 1,
        "serial_diff_operations": 0,
    }
    if (
        len(members) != 4
        or len(failed) != 1
        or len(recovered) != 3
        or injection["count"] != 1
        or injection["target_request_id"] != failed[0]["engine_request_id"]
        or any(
            m["operation_counts"] != MODEL_COUNTS or m["error"] is not None
            for m in recovered
        )
        or failed[0]["operation_counts"] != failed_counts
        or any(
            type(v) is not int for m in members for v in m["operation_counts"].values()
        )
        or len(injections) != 1
        or injections[0].get("request_id") != failed[0]["engine_request_id"]
        or type(injections[0].get("completed_layers")) is not int
        or injections[0].get("completed_layers") != 1
        or injections[0].get("completed_serial_operations") != failed_counts
        or any(
            type(v) is not int
            for v in injections[0]["completed_serial_operations"].values()
        )
        or failed[0]["retrieval_mask"] is not None
        or failed[0]["error"] is None
    ):
        raise ValueError("member failure or request-local model work is incomplete")
    restored = [e for e in events if e["action"] == "registered.span.restored"]
    consumed = [e for e in events if e["action"] == "registered.model.group.consumed"]
    if (
        len([e for e in events if e["action"] == "registered.span.stored"]) != 4
        or len(restored) != 16
        or len(consumed) != 1
        or consumed[0]["span_counts"] != [4] * 4
        or consumed[0]["restored_tokens"] != [416] * 4
        or consumed[0]["request_order"] != group["engine_request_order"]
        or not consumed[0]["source_closed"]
        or consumed[0]["pending"]
        or any(e["source_start"] == e["destination_start"] for e in restored)
    ):
        raise ValueError("member failure lacks actual settled shifted group inputs")
    failures = [
        e
        for e in events
        if e["action"] == "LMCacheConnectorV1Impl._record_kmesh_load_failure.begin"
    ]
    if (
        len(failures) != 1
        or failures[0]["request_id"] != failed[0]["engine_request_id"]
        or failures[0].get("invalidate_full_prompt") is not True
    ):
        raise ValueError("failed member did not invalidate its full prompt")
    if (
        len(observations) != 5
        or len({o["trace_request_id"] for o in observations}) != 5
    ):
        raise ValueError("member failure did not complete the actual five requests")
    failed_trace = failed[0]["trace_request_id"]
    serial = serial_reference_outputs(serial_reference)
    reference_members = {
        m["trace_request_id"]: m
        for m in serial_reference["recovery_evidence"]["serial_members"]
    }
    for member in recovered:
        expected = reference_members[member["trace_request_id"]]
        if any(member[k] != expected[k] for k in ("selected_spans", "retrieval_mask")):
            raise ValueError("surviving group masks differ from the serial reference")
    bindings = {
        b["engine_request_id"]: b["trace_request_id"]
        for b in recovery["request_id_bindings"]
    }
    serial_bindings = {
        b["engine_request_id"]: b["trace_request_id"]
        for b in serial_reference["recovery_evidence"]["request_id_bindings"]
    }

    def selections(rows, owners):
        result = {}
        for row in rows:
            key = owners[row["request_id"]], row["layer_id"]
            if key in result:
                raise ValueError("duplicate group model selection")
            result[key] = row["selected_indices"]
        return result

    selected = selections(
        [e for e in events if e["action"] == "registered.model.selection"], bindings
    )
    expected_selection = selections(
        serial_reference["selection_evidence"], serial_bindings
    )
    if len(expected_selection) != 96 or len(selected) != 73:
        raise ValueError("group failure actual model-layer coverage is incomplete")
    for member in members:
        trace = member["trace_request_id"]
        layers = set(range(1 if trace == failed_trace else 24))
        if {layer for owner, layer in selected if owner == trace} != layers or any(
            selected[(trace, layer)] != expected_selection[(trace, layer)]
            for layer in layers
        ):
            raise ValueError("group failure changed actual model repair selections")
    for observation in observations:
        trace = observation["trace_request_id"]
        expected = ordinary[trace] if trace == failed_trace else serial[trace]
        if observation["output_token_ids"] != expected:
            raise ValueError("member fallback or surviving model output changed")
    return {
        "passed": True,
        "control": "one model member fails after its first repaired layer",
        "failed_trace_request_id": failed_trace,
        "recovered_trace_request_ids": [m["trace_request_id"] for m in recovered],
        "fallback_matches_ordinary_tokens": True,
        "survivors_match_serial_tokens": True,
        "survivor_masks_and_selections_match_serial": True,
        "all_registered_sources_closed": True,
        "limit": "Generated tokens and cleanup control; mixed repaired/recomputed K/V are not a complete raw-tensor numerical comparison.",
    }

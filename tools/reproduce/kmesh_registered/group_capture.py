# SPDX-License-Identifier: MIT
"""Validate actual layer-major group recovery without inventing batched work."""

MODEL_COUNTS = {
    "serial_rope_operations": 24,
    "serial_diff_operations": 1,
    "collective_rope_operations": 0,
    "collective_diff_operations": 0,
    "position_recovery_serial_operations": 0,
    "position_recovery_collective_operations": 0,
}


def recovery_contract(recovery, captured):
    """Bind one successful four-member group to actual numerical capture owners."""
    groups = recovery.get("groups")
    if (
        type(groups) is not list
        or len(groups) != 1
        or recovery.get("serial_members") != []
    ):
        raise ValueError("registered group recovery membership changed")
    group = groups[0]
    if (
        type(group.get("call_index")) is not int
        or group["call_index"] != 0
        or type(group.get("num_layers")) is not int
        or group["num_layers"] != 24
        or type(group.get("members")) is not list
        or len(group["members"]) != 4
        or set(captured) != {1, 2, 3, 4}
        or any(
            type(group.get(k)) is not str or not group[k]
            for k in ("plan_id", "reuse_group_id")
        )
    ):
        raise ValueError("registered group geometry or plan identity changed")
    engine_order = [captured[i].engine_request_id for i in range(1, 5)]
    if (
        len(set(engine_order)) != 4
        or group.get("engine_request_order") != engine_order
        or [m.get("engine_request_id") for m in group["members"]] != engine_order
    ):
        raise ValueError("registered group capture order changed")
    by_trace = {m.get("trace_request_id"): m for m in group["members"]}
    if len(by_trace) != 4:
        raise ValueError("registered group trace membership is ambiguous")
    evidence, generations = {}, set()
    for sequence in range(1, 5):
        actual = captured[sequence]
        member = by_trace.get(actual.binding.trace_request_id)
        if (
            member is None
            or member.get("engine_request_id") != actual.engine_request_id
            or member.get("outcome") != "recovered"
            or member.get("error") is not None
            or member.get("operation_counts") != MODEL_COUNTS
            or any(type(v) is not int for v in member["operation_counts"].values())
            or type(member.get("generation")) is not int
            or member["generation"] < 0
        ):
            raise ValueError(
                "registered group member or actual operation counts changed"
            )
        generations.add(member["generation"])
        evidence[sequence] = {
            "plan_member_index": sequence - 1,
            "outcome": "recovered",
            "fallback_reason": None,
            "pic_executed": True,
        }
    if len(generations) != 1:
        raise ValueError("registered group scheduling generation changed")
    return {
        "plan_id": group["plan_id"],
        "reuse_group_id": group["reuse_group_id"],
        "profile": "collective_lockstep_v1",
        "member_count": 4,
        "all_members_successful": True,
    }, evidence

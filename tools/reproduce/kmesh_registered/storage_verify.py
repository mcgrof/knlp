# SPDX-License-Identifier: MIT
"""Check exact cold-read coverage without treating a configured route as proof."""


def verify(
    route: str, evidence: dict, settings: dict, *, require_poison: bool = False
) -> dict:
    if route not in ("staged", "dma-buf"):
        raise ValueError("unknown storage route")
    if evidence.get("close_buffers_reusable") is not True:
        raise ValueError("storage arena lacks checked physical shutdown")
    if evidence["storage_manager"].get("is_healthy") is not True:
        raise ValueError("storage manager or controller is unhealthy")
    if evidence.get("poisoned_l1_arena") is not require_poison:
        raise ValueError("arena conditioning differs from the requested experiment")
    rows = evidence["rows"]
    cold = [row for row in rows if row["kind"] == "cold"]
    after = [row for row in rows if row["kind"] == "after"]
    expected_children = settings["children"]
    objects, payload = settings["source_objects"], settings["chunk_payload_bytes"]
    if len(cold) != expected_children or len(after) != expected_children:
        raise ValueError("cold model request coverage is incomplete")
    ids = [row["request_id"] for row in cold]
    if len(set(ids)) != len(ids) or ids != [row["request_id"] for row in after]:
        raise ValueError("cold and completed model requests differ")
    path = "host_fixed" if route == "staged" else "dmabuf_fixed"
    registration = "classic" if route == "staged" else "dmabuf"
    source_keys = cold[0]["source_key_hashes"]
    if len(source_keys) != objects or len(set(source_keys)) != objects:
        raise ValueError("canonical source key coverage is incomplete")
    results = []
    for before, completed in zip(cold, after, strict=True):
        if (
            before["confirmed_l2_keys"] != objects
            or before["deleted_l1_keys"] != objects
            or before["source_key_hashes"] != source_keys
        ):
            raise ValueError("L1 eviction lacks exact immutable L2 admission")
        poison = before.get("poison")
        if require_poison:
            if (
                not poison
                or poison["bytes"] != settings["arena_bytes"]
                or poison["drained"] is not True
                or poison["device"] != ("cpu" if route == "staged" else "cuda:0")
            ):
                raise ValueError(
                    "restored bytes were not qualified against a poisoned arena"
                )
        elif poison is not None:
            raise ValueError("timed restore includes unexpected arena conditioning")
        states = [before["before"], completed["adapter"]]
        if any(state["is_healthy"] is not True for state in states):
            raise ValueError("storage adapter is unhealthy")
        cores = [state["core"] for state in states]
        if any(
            core["buffer_registration_mode"] != registration
            or core["io_engine"] != "io_uring"
            or core["use_uring_cmd"]
            or core["poisoned"]
            or core["inflight_io_count"]
            or core["quarantined_slot_count"]
            for core in cores
        ):
            raise ValueError("actual storage route or physical completion differs")
        tallies = [core["payload_reads"] for core in cores]
        delta = {name: tallies[1][name] - tallies[0][name] for name in tallies[0]}
        if (
            delta["logical_requests"] != objects
            or delta["logical_bytes"] != objects * payload
            or delta["padded_bytes"] != objects * payload
            or delta["submitted_operations"] <= 0
            or delta["submitted_operations"] != delta["completed_operations"]
            or delta["submitted_padded_bytes"] != objects * payload
            or delta["completed_padded_bytes"] != objects * payload
        ):
            raise ValueError("child did not complete every exact storage payload read")
        journals = [core["io_by_request"] for core in cores]
        if any(
            journal["unanswered"] or journal["evidence_failures"]
            for journal in journals
        ):
            raise ValueError("native submission/completion evidence is incomplete")

        def identity(operation):
            return operation["device_instance_id"], operation["operation_id"]

        old = {identity(op) for op in journals[0]["operations"]}
        operations = [
            op
            for op in journals[1]["operations"]
            if identity(op) not in old
            and op["direction"] == "read"
            and op["path"] == path
        ]
        if (
            not operations
            or any(op["complete"] is not True for op in operations)
            or sum(op["requested_bytes"] for op in operations) != objects * payload
            or sum(op["completed_bytes"] for op in operations) != objects * payload
        ):
            raise ValueError("native registered-read route lacks exact byte coverage")
        restores = [
            row
            for row in rows
            if row["kind"] == "restore" and row["phase"] == before["request_id"]
        ]
        hashes = [
            row
            for row in rows
            if row["kind"] == "checksum" and row["phase"] == before["request_id"]
        ]
        reads = [
            row
            for row in rows
            if row["kind"] == "storage_read" and row["phase"] == before["request_id"]
        ]
        if (
            not reads
            or any(row["success"] is not True for row in reads)
            or sum(row["payload_bytes"] for row in reads) != objects * payload
            or sorted(key for row in reads for key in row["source_key_hashes"])
            != sorted(source_keys)
        ):
            raise ValueError(
                "actual raw reads do not name the exact canonical source set"
            )
        contexts = {row["io_request_id"] for row in reads}
        if any(
            not op["context"] or op["context"]["request_id"] not in contexts
            for op in operations
        ):
            raise ValueError("native read completions belong to another storage task")
        if (
            len(restores) != objects
            or len(hashes) != objects
            or any(row["state"] != "loaded-needs-repair" for row in restores)
            or any(row["payload_bytes"] != payload for row in hashes)
            or sum(row["host_payload_copy_bytes"] for row in hashes)
            != (objects * payload if route == "dma-buf" else 0)
        ):
            raise ValueError("restored source or full integrity-check coverage differs")
        results.append(
            {
                "request_id": before["request_id"],
                "cold_prepare_ns": before["elapsed_ns"],
                "restore_ns": sum(row["elapsed_ns"] for row in restores),
                "storage_read_ns": sum(row["elapsed_ns"] for row in reads),
                "checksum_ns": sum(row["elapsed_ns"] for row in hashes),
                "integrity_host_copy_bytes": sum(
                    row["host_payload_copy_bytes"] for row in hashes
                ),
                "payload_read_bytes": objects * payload,
                "native_payload_read_operations": len(operations),
                "native_route": path,
            }
        )
    return {
        "passed": True,
        "route": route,
        "children": results,
        "canonical_payload_bytes": objects * payload,
        "configured_arena_bytes": settings["arena_bytes"],
        "memory_accounting": evidence["storage_manager"],
        "location": evidence["location"],
        "poisoned_l1_arena": require_poison,
    }

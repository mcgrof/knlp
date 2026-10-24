# SPDX-License-Identifier: MIT
"""Exercise real model fallback and blocked fallback under native failures."""


def execute(mode, llm, records, ledger, bridge, channel, worker_fault):
    import run_online_collective_lockstep_gate as baseline
    from run import receive

    observations = baseline._run_batch(
        llm, [records[0]], "prime", ledger, connectors=True
    )
    if mode == "partial_miss":
        bridge.omitted_sources.add(next(iter(bridge.handles)))
    channel.send("arm")
    if receive(channel)[0] != "armed":
        raise RuntimeError("owned service did not arm its control")
    worker_fault.armed = True
    raised = None
    try:
        observations += baseline._run_batch(
            llm, [records[1]], "fault_probe", ledger, connectors=True
        )
    except Exception as error:
        raised = {"type": type(error).__name__, "message": str(error)}
        ledger.emit("registered.control.probe_error", **raised)
    events = ledger.snapshot()
    failures = [
        e
        for e in events
        if e["phase"] == "fault_probe"
        and e["action"] == "LMCacheConnectorV1Impl._record_kmesh_load_failure.begin"
    ]
    completed = [
        e
        for e in events
        if e["phase"] == "fault_probe" and e["action"] == "gate.request.complete"
    ]
    pending_before_cleanup = bridge.worker.pending
    blocked = mode in ("persistent_drain", "worker_drain")
    if blocked:
        if raised is None or failures or completed or not pending_before_cleanup:
            raise ValueError("unsettled physical work did not block model fallback")
    elif raised is not None or len(completed) != 1:
        raise ValueError("drained failure did not finish the actual model request")
    elif mode == "partial_miss":
        consumed = [e for e in events if e["action"] == "registered.model.consumed"]
        if (
            len(consumed) != 1
            or consumed[0]["span_count"] != 3
            or consumed[0]["restored_tokens"] != 293
        ):
            raise ValueError("declared miss was not excluded from actual model input")
        mandatory = {i for i, hit in enumerate(consumed[0]["returned_mask"]) if not hit}
        selections = [
            e
            for e in events
            if e["phase"] == "fault_probe"
            and e["action"] == "registered.model.selection"
            and e["layer_id"] >= 1
        ]
        if len(selections) != 23 or any(
            not mandatory.issubset(e["selected_indices"]) for e in selections
        ):
            raise ValueError(
                "actual model repair omitted mandatory missing/private positions"
            )
    elif not failures or not any(
        e.get("invalidate_full_prompt") is True for e in failures
    ):
        raise ValueError("actual failed restore did not invalidate its full prompt")

    channel.send("clean")
    kind, service_observed = receive(channel)
    if kind != "cleaned":
        raise RuntimeError("owned service did not disarm its control")
    worker_fault.armed = False
    bridge.worker.drain_pending()
    if bridge.worker.pending:
        raise ValueError("actual cleanup did not settle physical work")
    ledger.emit(
        "registered.control.actual_cleanup",
        pending_before=pending_before_cleanup,
        pending_after=bridge.worker.pending,
    )
    if mode == "corrupt" and (
        not service_observed["corruptions"] or service_observed["enqueues"]
    ):
        raise ValueError("corrupt source was not refused before native scatter")
    if mode == "evict" and (
        not service_observed["evictions"] or service_observed["enqueues"]
    ):
        raise ValueError("evicted source was not refused before native scatter")
    if (
        mode in ("post_enqueue", "transient_drain", "persistent_drain")
        and not service_observed["enqueues"]
    ):
        raise ValueError("control did not issue an actual native restore")
    if (
        mode in ("transient_drain", "persistent_drain")
        and not service_observed["refused_drains"]
    ):
        raise ValueError("control did not refuse actual service cleanup")
    if mode == "worker_drain" and (
        not worker_fault.observed["worker_gathers"]
        or not worker_fault.observed["refused_drains"]
    ):
        raise ValueError("worker control did not retain a real native layer gather")

    if not blocked:
        bridge.omitted_sources.clear()
        observations += baseline._run_batch(
            llm, [records[1]], "safe_retry", ledger, connectors=True
        )
    return observations, {
        "control": mode,
        "passed": True,
        "fallback_blocked": blocked,
        "probe_exception": raised,
        "service_observed": service_observed,
        "worker_observed": worker_fault.observed,
        "pending_before_cleanup": pending_before_cleanup,
        "pending_after_cleanup": bridge.worker.pending,
        "availability": "a failed registered lease disables the replay backend; drained requests recompute",
    }

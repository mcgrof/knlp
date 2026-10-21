# SPDX-License-Identifier: MIT
"""Preserve scheduled arrivals while one engine prepares shared context."""

from __future__ import annotations

import math
import sys
import time

from worker import result_rows

ARMS = {"demand", "early", "declaration", "external_resident", "engine_resident"}


def replay_fixed(
    engine,
    ancestor,
    children,
    params,
    warm_params,
    arm,
    lead_ms,
    label,
    *,
    start_prefetch=None,
    declare=None,
    release_declaration=None,
    on_partial=None,
    arrival_offset_ms=250,
    clock=time,
):
    """Charge submission delays to a deadline frozen before any preparation.

    Keep engine submission and stepping on this caller's thread. External
    preparation returns a future and progresses separately. A blocking engine
    step may still delay admission; recorded deadline-based waits include that
    delay. Child completion excludes later preparation or ownership drainage.
    The caller must shut down the engine and join its preparation executor
    after any exception, including a bounded wait that cannot prove drainage.
    """
    if arm not in ARMS:
        raise ValueError("unknown loading control")
    for value in (lead_ms, arrival_offset_ms):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid arrival interval")
    if lead_ms > arrival_offset_ms or (arm != "early" and lead_ms != 0):
        raise ValueError("lead must fit the trace and belong to early loading")
    if arm == "early" and start_prefetch is None:
        raise ValueError("early loading requires an external preparation owner")
    if (declare is None) != (release_declaration is None):
        raise ValueError("declaration and release callbacks must be paired")
    origin = clock.perf_counter_ns()
    arrival = origin + round(arrival_offset_ms * 1e6)
    preparation_due = arrival - round(lead_ms * 1e6)
    names = [f"{label}-child-{index}" for index in range(len(children))]
    warm_name = label + "-warm"
    submitted, first, eighth, finished, outputs = {}, {}, {}, {}, {}
    first_counts = {}
    future = preparation = identity = None
    prepare_start = external_ready = warm_submit = warm_done = None
    warm_output = None
    group_submit = None
    warm_skipped = False
    events = []
    declaration_ids = set()
    receipt = None

    def declaration(warm_id):
        nonlocal identity
        if declare is None:
            return
        returned = declare(warm_id)
        if isinstance(returned, str) and returned:
            declaration_ids.add(returned)
        if returned is None or (identity is not None and returned != identity):
            raise RuntimeError("declaration was refused or changed identity")
        identity = returned
        events.append(
            {
                "event": "declaration",
                "time_ns": clock.perf_counter_ns(),
                "warm": warm_id,
            }
        )

    def submit_children():
        nonlocal group_submit
        group_submit = clock.perf_counter_ns()
        if arm == "declaration":
            declaration(None)
        for name, tokens in zip(names, children, strict=True):
            start = clock.perf_counter_ns()
            internal = engine.add_request(name, {"prompt_token_ids": tokens}, params)
            submitted[name] = {
                "submit_ns": start,
                "submit_return_ns": clock.perf_counter_ns(),
                "engine_request_id": internal,
            }

    try:
        while True:
            now = clock.perf_counter_ns()
            if now - origin > 120_000_000_000:
                raise TimeoutError("fixed-arrival replay exceeded its bound")
            if group_submit is None and now >= arrival:
                submit_children()
            if arm == "early" and future is None and now >= preparation_due:
                prepare_start = clock.perf_counter_ns()
                declaration(None)
                future = start_prefetch()
                if future is None:
                    raise RuntimeError("external preparation returned no owner")
            if group_submit is None and clock.perf_counter_ns() >= arrival:
                submit_children()
            if future is not None and external_ready is None and future.done():
                preparation = future.result()
                external_ready = clock.perf_counter_ns()
                if group_submit is None and external_ready >= arrival:
                    submit_children()
                if names and group_submit is not None:
                    # A child may already own singleflight. Installing a warm
                    # dependency here could make that owner wait on its follower.
                    warm_skipped = True
                    events.append(
                        {"event": "late-warm-skipped", "time_ns": external_ready}
                    )
                else:
                    warm_submit = clock.perf_counter_ns()
                    internal = engine.add_request(
                        warm_name, {"prompt_token_ids": ancestor + [7]}, warm_params
                    )
                    declaration(internal)
            if group_submit is None and clock.perf_counter_ns() >= arrival:
                submit_children()
            if engine.has_unfinished_requests():
                batch = engine.step()
                observed = clock.perf_counter_ns()
                for output in batch:
                    name = output.request_id
                    if name != warm_name and name not in submitted:
                        raise RuntimeError("engine returned an unknown request")
                    if name in finished or (
                        name == warm_name and warm_done is not None
                    ):
                        raise RuntimeError("engine returned a duplicate completion")
                    if (
                        name != warm_name
                        and output.outputs
                        and output.outputs[0].token_ids
                    ):
                        count = len(output.outputs[0].token_ids)
                        if name not in first:
                            if count != 1:
                                raise RuntimeError(
                                    "first output was not one streamed token"
                                )
                            first[name] = observed
                            first_counts[name] = count
                        if len(output.outputs[0].token_ids) >= 8:
                            eighth.setdefault(name, observed)
                    if not output.finished:
                        continue
                    if name == warm_name:
                        warm_done, warm_output = observed, output
                    else:
                        finished[name], outputs[name] = observed, output
            complete = group_submit is not None and set(finished) == set(names)
            prepared = arm != "early" or warm_done is not None or warm_skipped
            if complete and prepared and not engine.has_unfinished_requests():
                break
            if not engine.has_unfinished_requests():
                if group_submit is None:
                    remaining = (arrival - clock.perf_counter_ns()) / 1e9
                    if remaining <= 0:
                        continue
                    clock.sleep(min(0.0005, remaining))
                else:
                    clock.sleep(0.0005)
        drain_end = clock.perf_counter_ns()
        if any(name not in first or name not in eighth for name in names):
            raise RuntimeError("missing observed first output token")
        rows = result_rows([outputs[name] for name in names], 8)
        for row, name in zip(rows, names, strict=True):
            row.update(
                submitted[name]
                | {
                    "first_token_observed_ns": first[name],
                    "first_observed_token_count": first_counts[name],
                    "eighth_token_observed_ns": eighth[name],
                    "terminal_observed_ns": finished[name],
                    "submission_delay_ms": (submitted[name]["submit_ns"] - arrival)
                    / 1e6,
                    "deadline_first_token_ms": (first[name] - arrival) / 1e6,
                    "deadline_completion_ms": (eighth[name] - arrival) / 1e6,
                }
            )
        group_end = max(eighth.values()) if eighth else None
        receipt = {
            "arm": arm,
            "lead_ms": lead_ms,
            "trace_origin_ns": origin,
            "scheduled_arrival_ns": arrival,
            "group_submit_ns": group_submit,
            "group_end_ns": group_end,
            "drain_end_ns": drain_end,
            "prepare_start_ns": prepare_start,
            "external_ready_ns": external_ready,
            "warm_submit_ns": warm_submit,
            "warm_done_ns": warm_done,
            "warm_skipped_due_to_arrival": warm_skipped,
            "preparation": preparation,
            "declaration_identity": identity,
            "events": events,
            "rows": rows,
            "warm_rows": [] if warm_output is None else result_rows([warm_output], 1),
            "children_wall_ms": (
                None if group_end is None else (group_end - group_submit) / 1e6
            ),
            "deadline_group_completion_ms": (
                None if group_end is None else (group_end - arrival) / 1e6
            ),
            "workflow_completion_ms": (
                None if group_end is None else (group_end - origin) / 1e6
            ),
            "ready_by_scheduled_arrival": warm_done is not None
            and warm_done <= arrival,
        }
        return receipt
    except BaseException as error:
        if on_partial is not None:
            try:
                on_partial(
                    {
                        "arm": arm,
                        "lead_ms": lead_ms,
                        "trace_origin_ns": origin,
                        "scheduled_arrival_ns": arrival,
                        "group_submit_ns": group_submit,
                        "failed_ns": clock.perf_counter_ns(),
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "submissions": submitted,
                        "first_tokens": first,
                        "eighth_tokens": eighth,
                        "terminals": finished,
                        "events": events,
                    }
                )
            except BaseException as ledger_error:
                error.add_note(f"partial ledger write failed: {ledger_error}")
        pending = [name for name in submitted if name not in finished]
        if warm_submit is not None and warm_done is None:
            pending.append(warm_name)
        if pending:
            try:
                engine.abort_request(pending)
            except BaseException as cleanup_error:
                error.add_note(f"request abort failed: {cleanup_error}")
        raise
    finally:
        # Logical abandonment does not cancel accepted backend work.
        active_error = sys.exception()
        cleanup_errors = []
        try:
            if future is not None:
                future.result(timeout=60)
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        for declared in declaration_ids:
            try:
                release_declaration(declared)
            except BaseException as release_error:
                cleanup_errors.append(release_error)
        if cleanup_errors:
            failure = (
                receipt or {"arm": arm, "lead_ms": lead_ms, "events": events}
            ) | {
                "failed_ns": clock.perf_counter_ns(),
                "cleanup_errors": [str(error) for error in cleanup_errors],
            }
            if on_partial is not None:
                try:
                    on_partial(failure)
                except BaseException as ledger_error:
                    cleanup_errors.append(ledger_error)
            if active_error is not None:
                active_error.add_note(f"cleanup failures: {cleanup_errors}")
            else:
                raise RuntimeError(
                    "preparation drain or declaration release failed"
                ) from cleanup_errors[0]
        if receipt is not None:
            receipt["cleanup_completed_ns"] = clock.perf_counter_ns()

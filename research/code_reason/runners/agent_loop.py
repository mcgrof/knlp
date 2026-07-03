#!/usr/bin/env python3
"""The execution-free agent loop.

Drive one model through a task: offer it the read-only tools, run each tool
call through the router, and stop when it submits a final answer (or a
certificate in semi-formal mode) or hits the step cap. Every step is recorded
as an `agent_step` row through the ArtifactWriter when one is provided, so a
run is fully auditable from disk. The loop maintains provider-native
assistant/tool_result messages so the real multi-turn tool protocol works;
the mock client ignores them, which keeps offline runs deterministic.
"""

from __future__ import annotations

import json

from model_client import SUBMIT_TOOLS


def _assistant_message(turn):
    """Provider-native assistant content to append for the next turn."""
    if turn.raw is not None:
        return {"role": "assistant", "content": turn.raw}
    blocks = []
    if turn.text:
        blocks.append({"type": "text", "text": turn.text})
    for tc in turn.tool_calls:
        blocks.append(
            {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.args}
        )
    return {"role": "assistant", "content": blocks or turn.text or "(no output)"}


def run_agent(
    task_id,
    model,
    mode,
    prompt_text,
    client,
    router,
    tools,
    writer=None,
    system="",
    max_steps=8,
):
    """Run one (task, model, mode). Returns answer/certificate/stop reason."""
    messages = [{"role": "user", "content": prompt_text}]
    step = 0

    def record(role, **kw):
        nonlocal step
        if writer is not None:
            writer.append_transcript(
                task_id, model, mode, {"step": step, "role": role, **kw}
            )
        step += 1

    record("user", content=prompt_text[:2000])
    answer, certificate, stop = None, None, "max_steps"
    totals = {"input": 0, "output": 0}

    for _ in range(max_steps):
        turn = client.run_turn(system, messages, tools)
        totals["input"] += turn.tokens.get("input", 0)
        totals["output"] += turn.tokens.get("output", 0)
        record("assistant", content=turn.text or "", tokens=turn.tokens)

        if turn.final is not None:
            answer = turn.final.get("answer", {})
            certificate = turn.final.get("certificate")
            stop = "final"
            break

        if not turn.tool_calls:
            stop = "no_action"
            break

        messages.append(_assistant_message(turn))
        tool_results = []
        for tc in turn.tool_calls:
            if tc.name in SUBMIT_TOOLS:
                answer = tc.args.get("answer", {})
                certificate = tc.args.get("certificate")
                stop = "final"
                break
            result = router.dispatch(tc.name, tc.args)
            record(
                "tool", tool_call={"name": tc.name, "args": tc.args}, tool_result=result
            )
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": json.dumps(result)[:8000],
                }
            )
        if stop == "final":
            break
        messages.append({"role": "user", "content": tool_results})

    return {
        "answer": answer if answer is not None else {},
        "certificate": certificate,
        "stop": stop,
        "steps": step,
        "tokens": totals,
    }

# SPDX-License-Identifier: MIT
"""Exercise real HTTP failure boundaries and all-offered goodput accounting."""

import asyncio
import copy
import json
import math
import socket

import pytest

from tools.kv.decision.goodput import compare_blocks, evaluate, median_lower
from tools.kv.decision.serving import AcceptedRequests, replay


def plan():
    return {
        "mode": "open",
        "model": "study",
        "seed": 20261020,
        "arrival_seconds": 0.5,
        "drain_seconds": 0.2,
        "request_timeout_seconds": 2,
        "requests": [
            {
                "id": name,
                "offset_ns": i * 100_000_000,
                "prompt_token_ids": [1, 2],
                "max_tokens": 2,
            }
            for i, name in enumerate(["good-a", "good-b", "http-error", "hang"])
        ],
    }


def event(at, tokens=None, finish=None):
    return {
        "at_ns": at,
        "data": json.dumps(
            {
                "choices": [
                    {"index": 0, "token_ids": tokens or [], "finish_reason": finish}
                ]
            }
        ),
    }


def capture():
    p = plan()
    start = 1_000_000_000
    rows = []
    for i, frozen in enumerate(p["requests"]):
        arrival = start + frozen["offset_ns"]
        rows.append(
            {
                **frozen,
                "intended_ns": arrival,
                "dispatch_ns": arrival,
                "admitted_ns": arrival,
                "send_ns": arrival,
                "end_ns": arrival + 50_000_000,
                "status": "completed",
                "events": [
                    event(arrival + 10_000_000, [10]),
                    event(arrival + 40_000_000, [11], "length"),
                    {"at_ns": arrival + 50_000_000, "data": "[DONE]"},
                ],
            }
        )
    return {
        "plan": p,
        "start_ns": start,
        "arrival_end_ns": start + 500_000_000,
        "drain_deadline_ns": start + 700_000_000,
        "end_ns": start + 700_000_000,
        "requests": rows,
    }


def test_all_offered_and_common_window():
    x = capture()
    x["requests"][1].update(status="error", events=[])
    x["requests"][2]["events"][0]["at_ns"] += 20_000_000  # TTFT fails.
    x["requests"][3]["events"][-1]["at_ns"] = x["drain_deadline_ns"] + 1
    x["requests"][3]["end_ns"] = x["drain_deadline_ns"] + 2
    result = evaluate(x, {"ttft_ms": 20, "tpot_ms": 40, "eos_ms": 1000})
    assert result["offered"] == 4 and result["slo_completed"] == 1
    assert result["goodput_requests_per_second"] == 2
    assert result["slo_fraction_of_all_offered"] == 0.25
    assert result["valid_successful"] == 3
    assert result["valid_within_drain"] == 2
    assert result["latencies_conditional_on_valid_success"]["ttft_arrival_ms"]["n"] == 3
    assert result["completed_requests_per_second_including_drain"] == 2 / 0.7


def test_missing_or_changed_request_is_not_a_smaller_denominator():
    original = capture()
    for change in ("missing", "tokens", "arrival", "duplicate"):
        x = copy.deepcopy(original)
        if change == "missing":
            x["requests"].pop()
        elif change == "tokens":
            x["requests"][0]["max_tokens"] += 1
        elif change == "arrival":
            x["requests"][0]["intended_ns"] += 1
        else:
            x["requests"][1]["id"] = x["requests"][0]["id"]
        with pytest.raises(ValueError):
            evaluate(x, {"ttft_ms": 20, "tpot_ms": 40, "eos_ms": 1000})


def test_wrong_length_and_bad_framing_score_zero():
    x = capture()
    x["requests"][0]["events"][1]["data"] = json.dumps(
        {"choices": [{"token_ids": [], "finish_reason": "length"}]}
    )
    x["requests"][1]["events"].pop()
    x["requests"][2]["events"][0]["data"] = "not json"
    result = evaluate(x, {"ttft_ms": 20, "tpot_ms": 40, "eos_ms": 1000})
    assert result["slo_completed"] == result["valid_successful"] == 1


def test_independent_block_bounds_and_ties():
    bound = median_lower(list(range(10)), 0.05 / 3)
    assert bound["rank"] == 2 and bound["lower"] == 1
    assert bound["coverage_at_least"] == 1 - 11 / 1024
    controls = {name: [1] * 10 for name in ("native_fi", "native_fa3", "symmetric")}
    result = compare_blocks([1.06] * 9 + [0.9], controls)
    assert result["improvement_vs_all_controls"]
    assert all(row["holm_adjusted_p"] == 33 / 1024 for row in result["comparisons"])
    assert not compare_blocks([1.03] * 10, controls)["improvement_vs_all_controls"]
    assert not compare_blocks([1] * 10, controls)["improvement_vs_all_controls"]
    assert not compare_blocks([1.06] * 8 + [0.9, 0.9], controls)[
        "improvement_vs_all_controls"
    ]
    with pytest.raises(ValueError):
        compare_blocks([1.1] * 9, controls)
    with pytest.raises(ValueError):
        compare_blocks([math.nan] * 10, controls)


def test_zero_control_has_no_invented_relative_gain():
    row = compare_blocks([1] * 10, {"native": [0] * 10})["comparisons"][0]
    assert row["median_absolute_delta"] == 1
    assert row["median_relative_percent"] is None
    assert not row["practical_improvement_established"]


def test_real_http_preserves_errors_timeout_reuse_and_partial_frames(tmp_path):
    async def exercise():
        from aiohttp import web

        seen = []

        async def handler(request):
            body = await request.json()
            rid = body["request_id"]
            assert request.headers["X-KNLP-Request-ID"] == rid
            seen.append(rid)
            if rid == "http-error":
                return web.Response(status=503, text="intentional test failure")
            if rid == "hang":
                await asyncio.sleep(1)
                return web.Response()
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            payload = (
                "data: "
                + event(0, [10])["data"]
                + "\n\ndata: "
                + event(0, [11], "length")["data"]
                + "\n\ndata: [DONE]\n\n"
            ).encode()
            for fragment in [payload[:3], payload[3:11], payload[11:]]:
                await response.write(fragment)
            return response

        app = web.Application()
        app.router.add_post("/v1/completions", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            result = await replay(
                plan(), f"http://127.0.0.1:{port}", tmp_path / "ledger.jsonl"
            )
        finally:
            await runner.cleanup()
        assert seen == [r["id"] for r in plan()["requests"]]
        rows = {row["id"]: row for row in result["requests"]}
        assert rows["good-a"]["status"] == rows["good-b"]["status"] == "completed"
        assert rows["http-error"]["status"] == "error"
        assert rows["http-error"]["http_status"] == 503
        assert rows["hang"]["status"] in ("timeout", "cancelled")
        assert any(
            e["event"] == "connection_reuseconn"
            for e in rows["good-b"]["connection_events"]
        )
        assert len((tmp_path / "ledger.jsonl").read_text().splitlines()) == 4
        summary = evaluate(result, {"ttft_ms": 1000, "tpot_ms": 1000, "eos_ms": 1000})
        assert summary["offered"] == 4 and summary["slo_completed"] == 2
        assert summary["goodput_requests_per_second"] == 4

    asyncio.run(exercise())


def test_asgi_observer_passes_messages_without_rebuffering(tmp_path, monkeypatch):
    path = tmp_path / "accepted.jsonl"
    monkeypatch.setenv("KNLP_DECISION_HTTP_EVENTS", str(path))
    sent, received = [], []
    messages = [
        {"type": "http.request", "body": b'{"prompt":'},
        {"type": "http.request", "body": b"[1]}", "more_body": False},
    ]

    async def receive():
        return messages[len(received)]

    async def send(message):
        sent.append(message)

    async def app(scope, receive, send):
        for _ in messages:
            received.append(await receive())
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": b"a", "more_body": True})
        await send({"type": "http.response.body", "body": b"b"})

    asyncio.run(
        AcceptedRequests(app)(
            {
                "type": "http",
                "path": "/v1/completions",
                "headers": [(b"x-knlp-request-id", b"a")],
            },
            receive,
            send,
        )
    )
    assert received == messages
    assert [m.get("body") for m in sent] == [None, b"a", b"b"]
    assert [r["event"] for r in map(json.loads, path.read_text().splitlines())] == [
        "asgi_accepted",
        "asgi_response_start",
        "asgi_response_end",
    ]


def test_failures_before_server_acceptance_remain_offered(tmp_path):
    p = plan()
    p["arrival_seconds"] = 0.1
    p["requests"] = p["requests"][:2]
    p["requests"][1]["offset_ns"] = 10_000_000
    # Reserve an unused port without listening: TCP refuses both connections.
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
        result = asyncio.run(replay(p, f"http://127.0.0.1:{port}"))
    assert all(row["status"] == "error" for row in result["requests"])
    assert all(
        any(e["event"] == "request_exception" for e in row["connection_events"])
        for row in result["requests"]
    )
    summary = evaluate(result, {"ttft_ms": 20, "tpot_ms": 20, "eos_ms": 100})
    assert summary["offered"] == 2 and summary["goodput_requests_per_second"] == 0

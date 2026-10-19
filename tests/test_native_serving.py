# SPDX-License-Identifier: MIT
"""Known client-clock endpoints and a real local streaming transport."""

import asyncio
import copy
import json

import pytest

from tools.kv.native.serving_client import replay
from tools.kv.native.serving_events import SSEDecoder, request_metrics, summarize
from tools.kv.native.serving_verify import verify


def event(ms, tokens=None, text="", finish=None):
    return {
        "at_ns": int(ms * 1e6),
        "data": json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "text": text,
                        "token_ids": tokens or [],
                        "finish_reason": finish,
                    }
                ]
            }
        ),
    }


def known_record():
    planned = {"id": "queued", "offset_ns": 0, "prompt_token_ids": [9], "max_tokens": 4}
    row = {
        **planned,
        "intended_ns": 0,
        "dispatch_ns": 10_000_000,
        "admitted_ns": 110_000_000,
        "send_ns": 120_000_000,
        "headers_ns": 125_000_000,
        "end_ns": 300_000_000,
        "status": "completed",
        "events": [
            event(130),
            event(150, [1], ""),
            event(200, [2, 3], "ab"),
            event(270, [4], "c", "length"),
            {"at_ns": 280_000_000, "data": "[DONE]"},
        ],
    }
    timeout_plan = {**planned, "id": "timeout"}
    timed_out = {
        **timeout_plan,
        "intended_ns": 0,
        "dispatch_ns": 10_000_000,
        "end_ns": 400_000_000,
        "status": "timeout",
        "events": [],
    }
    rows = [row, timed_out]
    return {
        "start_ns": 0,
        "end_ns": 500_000_000,
        "arrival_end_ns": 0,
        "plan": {"requests": [planned, timeout_plan], "arrival_duration_ns": 0},
        "requests": rows,
        "summary": summarize(rows, 0, 500_000_000, 0),
    }


def test_known_queue_chunk_timeout_and_independent_reader():
    record = known_record()
    metrics = request_metrics(record["requests"][0])
    assert metrics["ttft_send_ms"] == 30
    assert metrics["ttft_arrival_ms"] == 150
    assert metrics["first_visible_send_ms"] == 80
    assert metrics["tpot_ms"] == 40
    assert metrics["latency_send_ms"] == 160
    assert metrics["latency_arrival_ms"] == 280
    assert metrics["admission_wait_ms"] == 100
    assert metrics["tokens_per_event"] == [1, 2, 1]
    assert metrics["inter_token_event_gaps_ms"] == [50, 70]
    assert verify(record) == {
        "passed": True,
        "offered": 2,
        "completed": 1,
        "output_tokens": 4,
    }
    tampered = copy.deepcopy(record)
    tampered["summary"]["metrics_conditional_on_valid_completion"]["ttft_arrival_ms"][
        "p50"
    ] = 30
    with pytest.raises(AssertionError):
        verify(tampered)


def test_incremental_sse_and_coalesced_events():
    decoder = SSEDecoder()
    assert decoder.feed(b': comment\r\ndata: {"utf8":"\xc3', 1) == []
    assert decoder.feed(b'\xa9"}\r\n\r', 2) == []
    events = decoder.feed(b"\ndata: [DONE]\n\n", 3)
    assert events == [
        {"at_ns": 3, "data": '{"utf8":"é"}'},
        {"at_ns": 3, "data": "[DONE]"},
    ]
    decoder.finish()
    decoder.feed(b"data: unterminated", 4)
    with pytest.raises(ValueError, match="truncated"):
        decoder.finish()


def test_missing_token_ids_cannot_be_scored():
    row = known_record()["requests"][0]
    row["events"][2] = {
        "at_ns": 200_000_000,
        "data": json.dumps({"choices": [{"text": "missing ids"}]}),
    }
    with pytest.raises(ValueError, match="omitted"):
        request_metrics(row)
    assert summarize([row], 0, 500_000_000, 0)["counts"]["invalid"] == 1


def test_local_http_queue_errors_and_timeout():
    from aiohttp import web

    async def exercise():
        async def handle(request):
            body = await request.json()
            key = body["request_id"]
            if key == "error":
                return web.Response(status=503, text="injected failure")
            if key == "timeout":
                await asyncio.sleep(0.4)
                return web.Response(status=500)
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(("data: " + event(0)["data"] + "\n\n").encode())
            await asyncio.sleep(0.03)
            # An invisible token must start TTFT; two tokens share one event.
            await response.write(("data: " + event(0, [1])["data"] + "\n\n").encode())
            await asyncio.sleep(0.03)
            await response.write(
                (
                    "data: "
                    + event(0, [2, 3], "ab", "length")["data"]
                    + "\n\ndata: [DONE]\n\n"
                ).encode()
            )
            await response.write_eof()
            return response

        app = web.Application()
        app.router.add_post("/v1/completions", handle)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        base = {
            "model": "test",
            "seed": 1,
            "mode": "closed",
            "concurrency": 1,
            "request_timeout_s": 0.3,
            "drain_timeout_s": 1,
            "arrival_duration_ns": 0,
        }
        base["requests"] = [
            {"id": key, "offset_ns": 0, "prompt_token_ids": [1], "max_tokens": 3}
            for key in ["first", "second", "error", "timeout"]
        ]
        try:
            result = await replay(base, f"http://127.0.0.1:{port}")
            result["plan"] = base
            assert verify(result)["completed"] == 2
            assert result["summary"]["counts"] == {
                "completed": 2,
                "error": 1,
                "timeout": 1,
                "cancelled": 0,
                "unfinished": 0,
                "invalid": 0,
            }
            second = request_metrics(result["requests"][1])
            assert second["admission_wait_ms"] >= 40
            assert second["ttft_arrival_ms"] > second["ttft_send_ms"] + 40
            base["mode"] = "open"
            with pytest.raises(ValueError, match="cap"):
                await replay(base, f"http://127.0.0.1:{port}")
            base.pop("concurrency")
            base["requests"] = base["requests"][:2]
            result = await replay(base, f"http://127.0.0.1:{port}")
            result["plan"] = base
            assert verify(result)["completed"] == 2
            assert all(
                request_metrics(row)["admission_wait_ms"] < 10
                for row in result["requests"]
            )
        finally:
            await runner.cleanup()

    asyncio.run(exercise())

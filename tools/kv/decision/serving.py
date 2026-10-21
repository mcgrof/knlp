# SPDX-License-Identifier: MIT
"""Open-loop HTTP evidence with explicit connection and ASGI boundaries.

All offered requests survive in the result, including failures before server
acceptance. A frozen manifest supplies arrival times, timeouts and SLOs. This
module does not choose those values after observing a candidate.
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import time

from tools.kv.native.serving_events import SSEDecoder


def emit(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def validate(plan):
    if plan["mode"] != "open" or plan.get("concurrency") is not None:
        raise ValueError("This experiment requires uncapped open-loop arrivals")
    for name in ("arrival_seconds", "drain_seconds", "request_timeout_seconds"):
        if not math.isfinite(plan[name]) or plan[name] <= 0:
            raise ValueError("Invalid time limit: " + name)
    if not plan["requests"]:
        raise ValueError("Empty arrival trace")
    ids = set()
    for row in plan["requests"]:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", row["id"]):
            raise ValueError("Invalid request ID")
        if row["id"] in ids:
            raise ValueError("Duplicate request ID")
        ids.add(row["id"])
        if (
            type(row["offset_ns"]) is not int
            or not 0 <= row["offset_ns"] < plan["arrival_seconds"] * 1e9
        ):
            raise ValueError("Arrival outside the measured window")
        if not row["prompt_token_ids"] or any(
            type(x) is not int or x < 0 for x in row["prompt_token_ids"]
        ):
            raise ValueError("Invalid prompt tokens")
        if type(row["max_tokens"]) is not int or row["max_tokens"] < 2:
            raise ValueError("At least two forced output tokens are required")


def connection_trace():
    import aiohttp

    trace = aiohttp.TraceConfig()
    for name in (
        "request_start",
        "request_headers_sent",
        "request_chunk_sent",
        "connection_queued_start",
        "connection_queued_end",
        "connection_create_start",
        "connection_create_end",
        "connection_reuseconn",
        "dns_resolvehost_start",
        "dns_resolvehost_end",
        "request_end",
        "request_exception",
    ):

        async def record(session, context, params, _name=name):
            event = {"event": _name, "at_ns": time.monotonic_ns()}
            if _name == "request_exception":
                event["exception"] = repr(params.exception)
            context.trace_request_ctx["connection_events"].append(event)

        getattr(trace, "on_" + name).append(record)
    return trace


async def request(session, url, plan, row, drain_ns, sink):
    try:
        await asyncio.sleep(max(0, (row["intended_ns"] - time.monotonic_ns()) / 1e9))
        row["dispatch_ns"] = row["admitted_ns"] = time.monotonic_ns()
        deadline = min(
            drain_ns, row["intended_ns"] + int(plan["request_timeout_seconds"] * 1e9)
        )
        remaining = (deadline - time.monotonic_ns()) / 1e9
        if remaining <= 0:
            raise TimeoutError("Request deadline elapsed before dispatch")
        async with asyncio.timeout(remaining):
            body = {
                "model": plan["model"],
                "prompt": row["prompt_token_ids"],
                "max_tokens": row["max_tokens"],
                "temperature": 0,
                "seed": plan["seed"],
                "ignore_eos": True,
                "stream": True,
                "return_token_ids": True,
                "request_id": row["id"],
            }
            payload = json.dumps(body).encode()
            row["send_ns"] = time.monotonic_ns()
            async with session.post(
                url + "/v1/completions",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-KNLP-Request-ID": row["id"],
                },
                trace_request_ctx=row,
                allow_redirects=False,
            ) as response:
                row["headers_ns"] = time.monotonic_ns()
                row["http_status"] = response.status
                if response.status != 200:
                    row["error_body"] = await response.text()
                    raise RuntimeError(f"HTTP {response.status}")
                decoder = SSEDecoder()
                async for chunk in response.content.iter_any():
                    at = time.monotonic_ns()
                    row["reads"].append({"at_ns": at, "bytes": len(chunk)})
                    row["events"].extend(decoder.feed(chunk, at))
                decoder.finish()
                if not any(e["data"] == "[DONE]" for e in row["events"]):
                    raise ValueError("Stream ended without DONE")
                row["status"] = "completed"
    except TimeoutError as error:
        row.update(status="timeout", error=repr(error))
    except asyncio.CancelledError:
        row.update(status="cancelled", error="Common drain deadline reached")
    except Exception as error:
        row.update(status="error", error=repr(error))
    finally:
        row["end_ns"] = time.monotonic_ns()
        if sink is not None:
            emit(sink, row)


async def replay(plan, url, sink=None):
    import aiohttp
    import psutil

    validate(plan)
    start = time.monotonic_ns() + 250_000_000
    arrival_end = start + int(plan["arrival_seconds"] * 1e9)
    drain = arrival_end + int(plan["drain_seconds"] * 1e9)
    rows = [
        {
            **r,
            "intended_ns": start + r["offset_ns"],
            "status": "unfinished",
            "events": [],
            "reads": [],
            "connection_events": [],
        }
        for r in plan["requests"]
    ]
    telemetry = []
    process = psutil.Process()
    stop = asyncio.Event()

    async def sample():
        while not stop.is_set():
            cpu = process.cpu_times()
            telemetry.append(
                {
                    "at_ns": time.monotonic_ns(),
                    "cpu_user_seconds": cpu.user,
                    "cpu_system_seconds": cpu.system,
                    "rss_bytes": process.memory_info().rss,
                    "open_fds": process.num_fds(),
                    "network": psutil.net_io_counters()._asdict(),
                }
            )
            try:
                await asyncio.wait_for(stop.wait(), timeout=0.5)
            except TimeoutError:
                pass

    monitor = asyncio.create_task(sample())
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0, keepalive_timeout=300)
    try:
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=None),
            trace_configs=[connection_trace()],
        ) as session:
            tasks = [
                asyncio.create_task(request(session, url, plan, row, drain, sink))
                for row in rows
            ]
            _, pending = await asyncio.wait(
                tasks, timeout=max(0, (drain - time.monotonic_ns()) / 1e9)
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*tasks)
            await asyncio.sleep(max(0, (arrival_end - time.monotonic_ns()) / 1e9))
    finally:
        stop.set()
        await monitor
    return {
        "schema": 1,
        "clock": "time.monotonic_ns; client and server on the same host",
        "start_ns": start,
        "arrival_end_ns": arrival_end,
        "drain_deadline_ns": drain,
        "end_ns": time.monotonic_ns(),
        "requests": rows,
        "client_resources": telemetry,
        "network_counter_scope": "Whole-host network counters; not attributed to the client process",
        "client_cpu_affinity": process.cpu_affinity(),
        "client_fd_limit": list(resource.getrlimit(resource.RLIMIT_NOFILE)),
        "connector_limit": 0,
        "keepalive_timeout_seconds": 300,
        "application_retries": 0,
        "plan": plan,
    }


class AcceptedRequests:
    """Transparent ASGI observer; accepted means the ASGI app was entered."""

    def __init__(self, app):
        self.app = app
        self.path = Path(os.environ["KNLP_DECISION_HTTP_EVENTS"])

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/v1/completions":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        rid = headers.get(b"x-knlp-request-id", b"").decode("ascii", errors="replace")

        def log(event, **extra):
            emit(
                self.path,
                {"id": rid, "event": event, "at_ns": time.monotonic_ns(), **extra},
            )

        log("asgi_accepted", client=scope.get("client"), pid=os.getpid())

        async def observed_receive():
            message = await receive()
            if message["type"] == "http.disconnect":
                log("asgi_disconnect")
            return message

        async def observed_send(message):
            if message["type"] == "http.response.start":
                log("asgi_response_start", status=message["status"])
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body"):
                log("asgi_response_end")

        try:
            return await self.app(scope, observed_receive, observed_send)
        except BaseException as error:
            log("asgi_exception", exception=repr(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    raw = args.trace.read_bytes()
    result = asyncio.run(
        replay(json.loads(raw), args.url, args.output / "completed-requests.jsonl")
    )
    result["trace_sha256"] = hashlib.sha256(raw).hexdigest()
    (args.output / "client.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

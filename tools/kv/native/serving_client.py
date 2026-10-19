# SPDX-License-Identifier: MIT
"""Replay an immutable arrival trace against the local streaming HTTP server.

Open-loop traces have no concurrency semaphore or connector limit. Closed-loop
traces explicitly retain admission wait. All raw events and failed requests
survive. This process and the server must run on the same dedicated test host.
"""

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

from .serving_events import SSEDecoder, summarize


async def request(session, url, plan, row, semaphore, request_timeout):
    row["dispatch_ns"] = time.monotonic_ns()
    acquired = False
    try:
        # Timeout includes client admission, connection, server queue and drain.
        async with asyncio.timeout(request_timeout):
            if semaphore is not None:
                await semaphore.acquire()
                acquired = True
            row["admitted_ns"] = time.monotonic_ns()
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
            # Serialize before the HTTP interval and keep this boundary explicit.
            payload = json.dumps(body).encode()
            row["send_ns"] = time.monotonic_ns()
            async with session.post(
                url + "/v1/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
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
                    raise ValueError("stream ended without DONE")
                row["status"] = "completed"
    except TimeoutError:
        row["status"] = "timeout"
    except asyncio.CancelledError:
        row["status"] = "cancelled"
    except Exception as exc:
        row["status"] = "error"
        row["error"] = repr(exc)
    finally:
        row["end_ns"] = time.monotonic_ns()
        if acquired:
            semaphore.release()


async def replay(plan, url):
    import aiohttp

    if plan["mode"] not in ("open", "closed"):
        raise ValueError("mode must be open or closed")
    if plan["mode"] == "open" and plan.get("concurrency") is not None:
        raise ValueError("open-loop traces cannot hide an admission cap")
    arrivals = [r["offset_ns"] for r in plan["requests"]]
    if not arrivals or arrivals != sorted(arrivals) or min(arrivals) < 0:
        raise ValueError("arrival trace must be nonempty, nonnegative and ordered")
    if len({r["id"] for r in plan["requests"]}) != len(arrivals):
        raise ValueError("request IDs must be unique")
    semaphore = (
        asyncio.Semaphore(plan["concurrency"]) if plan["mode"] == "closed" else None
    )
    if semaphore is not None and plan["concurrency"] < 1:
        raise ValueError("closed-loop concurrency must be positive")
    start = time.monotonic_ns() + 250_000_000
    arrival_end = start + plan["arrival_duration_ns"]
    if arrivals[-1] > plan["arrival_duration_ns"]:
        raise ValueError("arrival lies outside declared interval")
    rows = [
        {
            **r,
            "intended_ns": start + r["offset_ns"],
            "status": "unfinished",
            "events": [],
            "reads": [],
        }
        for r in plan["requests"]
    ]
    tasks = []
    connector = aiohttp.TCPConnector(limit=0)
    # Our timeout includes admission. Disable a second library-wide timeout.
    async with aiohttp.ClientSession(
        connector=connector, timeout=aiohttp.ClientTimeout(total=None)
    ) as session:
        for row in rows:
            await asyncio.sleep(
                max(0, (row["intended_ns"] - time.monotonic_ns()) / 1e9)
            )
            tasks.append(
                asyncio.create_task(
                    request(
                        session, url, plan, row, semaphore, plan["request_timeout_s"]
                    )
                )
            )
        remaining = (
            max(0, (arrival_end - time.monotonic_ns()) / 1e9) + plan["drain_timeout_s"]
        )
        _, pending = await asyncio.wait(tasks, timeout=remaining)
        for task in pending:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=False)
        # An underloaded run still owns the complete declared arrival interval.
        await asyncio.sleep(max(0, (arrival_end - time.monotonic_ns()) / 1e9))
    end = time.monotonic_ns()
    return {
        "schema": 1,
        "clock": "time.monotonic_ns/client",
        "start_ns": start,
        "end_ns": end,
        "arrival_end_ns": arrival_end,
        "requests": rows,
        "summary": summarize(rows, start, end, arrival_end),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("use a fresh output file")
    data = args.trace.read_bytes()
    plan = json.loads(data)
    result = asyncio.run(replay(plan, args.url))
    result.update(trace_sha256=hashlib.sha256(data).hexdigest(), plan=plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()

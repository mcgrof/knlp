#!/usr/bin/env python3
"""Minimal Anthropic Messages client for the code-reason harness (no SDK).

Mirror of openai_client so the fault-loc runner can drive Claude models
(the paper's Opus-4.5 headline model, and the Sonnet-4.5 that regressed in
the paper) with the same interface: complete(prompt) -> text + usage + cost.
Stdlib-only (urllib). Reads ANTHROPIC_API_KEY from the environment.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

_URL = "https://api.anthropic.com/v1/messages"

# usd per 1M tokens -- ESTIMATE, verify against current pricing.
PRICE = {
    "claude-opus-4-5-20251101": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},
}
_DEFAULT_PRICE = {"input": 15.0, "output": 75.0}


class AnthropicClient:
    def __init__(
        self,
        model="claude-opus-4-5-20251101",
        max_tokens=4096,
        api_key=None,
        retries=4,
        reasoning_effort="n/a",
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort  # for report parity
        self.retries = retries
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

    def cost(self, usage):
        p = PRICE.get(self.model, _DEFAULT_PRICE)
        return (
            usage.get("input_tokens", 0) / 1e6 * p["input"]
            + usage.get("output_tokens", 0) / 1e6 * p["output"]
        )

    def complete(self, prompt, system=None):
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        data = json.dumps(payload).encode()
        last = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(
                    _URL,
                    data=data,
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                )
                with urllib.request.urlopen(req, timeout=180) as resp:
                    body = json.load(resp)
                text = "".join(
                    b.get("text", "")
                    for b in body.get("content", [])
                    if b.get("type") == "text"
                )
                usage = body.get("usage", {})
                return {
                    "text": text,
                    "finish_reason": body.get("stop_reason"),
                    "usage": usage,
                    "cost": self.cost(usage),
                    "model": body.get("model"),
                }
            except urllib.error.HTTPError as exc:
                last = f"HTTP {exc.code}: {exc.read()[:200]}"
                if exc.code in (429, 500, 502, 503, 529):
                    time.sleep(2**attempt)
                    continue
                break
            except Exception as exc:  # pragma: no cover - network
                last = str(exc)
                time.sleep(2**attempt)
        raise RuntimeError(f"Anthropic call failed after {self.retries}: {last}")


if __name__ == "__main__":
    import sys

    if "--smoke" in sys.argv:
        c = AnthropicClient(max_tokens=100)
        r = c.complete('Reply with ONLY this JSON: {"ok": true}')
        print(
            "text:",
            r["text"],
            "| stop:",
            r["finish_reason"],
            "| cost $%.5f" % r["cost"],
        )

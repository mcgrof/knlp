#!/usr/bin/env python3
"""Minimal OpenAI chat client for the code-reason harness (no SDK dependency).

A scriptable model backend -- unlike the MCP/agent channel, a plain process
can call this, so a run needs no human in the loop. Uses only the standard
library (urllib) so the harness stays dependency-light. Reads OPENAI_API_KEY
from the environment; returns the text plus the token usage so cost is
recorded per call. Reasoning models (gpt-5.x) spend reasoning tokens before
content, so max_completion_tokens is set high enough to leave room for the
answer and finish_reason is surfaced to catch truncation.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

_URL = "https://api.openai.com/v1/chat/completions"


class OpenAIClient:
    def __init__(
        self,
        model="gpt-5.2-2025-12-11",
        reasoning_effort="medium",
        max_completion_tokens=8000,
        api_key=None,
        retries=4,
    ):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = max_completion_tokens
        self.retries = retries
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

    def complete(self, prompt, system=None):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": self.max_completion_tokens,
            "reasoning_effort": self.reasoning_effort,
        }
        data = json.dumps(payload).encode()
        last = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(
                    _URL,
                    data=data,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )
                with urllib.request.urlopen(req, timeout=180) as resp:
                    body = json.load(resp)
                ch = body["choices"][0]
                usage = body.get("usage", {})
                return {
                    "text": ch["message"].get("content") or "",
                    "finish_reason": ch.get("finish_reason"),
                    "usage": usage,
                    "cost": call_cost(usage),
                    "model": body.get("model"),
                }
            except urllib.error.HTTPError as exc:
                last = f"HTTP {exc.code}: {exc.read()[:200]}"
                if exc.code in (429, 500, 502, 503):
                    time.sleep(2**attempt)
                    continue
                break
            except Exception as exc:  # pragma: no cover - network
                last = str(exc)
                time.sleep(2**attempt)
        raise RuntimeError(f"OpenAI call failed after {self.retries}: {last}")


# usd per 1M tokens -- ESTIMATE, verify against current pricing
PRICE = {"input": 2.0, "output": 10.0}


def call_cost(usage):
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    return pt / 1e6 * PRICE["input"] + ct / 1e6 * PRICE["output"]


if __name__ == "__main__":
    import sys

    if "--smoke" in sys.argv:
        c = OpenAIClient(max_completion_tokens=2000, reasoning_effort="low")
        r = c.complete('Reply with ONLY this JSON: {"ok": true}')
        print(
            "text:",
            r["text"],
            "| finish:",
            r["finish_reason"],
            "| cost $%.5f" % call_cost(r["usage"]),
        )

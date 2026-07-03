#!/usr/bin/env python3
"""Model client abstraction for the code-reason agent loop.

The agent loop talks to a model through one small interface, `run_turn`,
which returns a `Turn`: some assistant text, zero or more tool calls, and an
optional final answer (with an optional certificate). Two clients implement
it. `AnthropicModelClient` calls the real API and is imported lazily so the
harness never hard-depends on the SDK or a key. `MockModelClient` replays a
scripted sequence of turns with no network, which is what the self-tests and
`make check`-style dry runs use: the whole runner is exercisable offline.

Keeping the provider-native assistant content on each `Turn` (`raw`) lets the
loop append faithful assistant/tool_result messages for the next turn, so the
real multi-turn tool protocol works without leaking provider details into the
loop.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

SUBMIT_ANSWER = "submit_answer"
SUBMIT_CERTIFICATE = "submit_certificate"
SUBMIT_TOOLS = {SUBMIT_ANSWER, SUBMIT_CERTIFICATE}


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Turn:
    text: str = ""
    tool_calls: list = field(default_factory=list)  # list[ToolCall]
    final: dict | None = None  # {"answer": {...}, "certificate": {...}|None}
    tokens: dict = field(default_factory=lambda: {"input": 0, "output": 0})
    raw: object = None  # provider-native assistant content, for message replay


class ModelClient:
    model_id = "base"

    def run_turn(self, system, messages, tools):
        raise NotImplementedError


# --------------------------------------------------------------------------
# Offline deterministic client
# --------------------------------------------------------------------------
class MockModelClient(ModelClient):
    """Replay a fixed list of Turns. Ignores messages (deterministic)."""

    def __init__(self, script, model_id="mock"):
        self.model_id = model_id
        self._script = list(script)
        self._i = 0

    def run_turn(self, system, messages, tools):
        if self._i < len(self._script):
            turn = self._script[self._i]
            self._i += 1
            return turn
        # Exhausted script: submit an empty answer so the loop always ends.
        return Turn(text="(mock exhausted)", final={"answer": {}, "certificate": None})


def mock_default_script(task_type, mode, sample_glob="**/*.py"):
    """A generic 3-step plan: grep, read, submit. For offline self-test."""
    grep = Turn(
        text="Looking for definitions.",
        tool_calls=[ToolCall(id="t1", name="grep", args={"pattern": "def "})],
    )
    read = Turn(
        text="Reading the primary file.",
        tool_calls=[ToolCall(id="t2", name="read_file", args={"path": "calc.py"})],
    )
    answer = {"value": "mock", "note": f"{task_type}/{mode}"}
    if mode == "semiformal":
        cert = {
            "task_type": task_type,
            "answer": answer,
            "premises": [
                {
                    "id": "P1",
                    "claim": "mock premise",
                    "evidence": [{"file": "calc.py", "source": "repo_read"}],
                }
            ],
            "formal_conclusion": "mock conclusion",
            "confidence": "low",
        }
        submit = Turn(
            text="Submitting certificate.",
            final={"answer": answer, "certificate": cert},
        )
    else:
        submit = Turn(text="Submitting answer.", final={"answer": answer})
    return [grep, read, submit]


# --------------------------------------------------------------------------
# Real Anthropic client (lazy; never required for import)
# --------------------------------------------------------------------------
class AnthropicModelClient(ModelClient):
    def __init__(self, model_id, api_key=None, max_tokens=2048):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    def _ensure(self):
        if self._client is not None:
            return
        try:
            import anthropic
        except Exception as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "anthropic SDK not installed; use MockModelClient offline"
            ) from exc
        if not self._api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic(api_key=self._api_key)

    def run_turn(self, system, messages, tools):
        self._ensure()
        resp = self._client.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            system=system or "",
            messages=messages,
            tools=tools,
        )
        text_parts, calls, final = [], [], None
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                args = block.input if isinstance(block.input, dict) else {}
                if block.name in SUBMIT_TOOLS:
                    if block.name == SUBMIT_CERTIFICATE:
                        final = {
                            "answer": args.get("answer", {}),
                            "certificate": args.get("certificate", args),
                        }
                    else:
                        final = {"answer": args.get("answer", args)}
                else:
                    calls.append(ToolCall(id=block.id, name=block.name, args=args))
        usage = getattr(resp, "usage", None)
        tokens = {
            "input": getattr(usage, "input_tokens", 0) if usage else 0,
            "output": getattr(usage, "output_tokens", 0) if usage else 0,
        }
        return Turn(
            text="".join(text_parts),
            tool_calls=calls,
            final=final,
            tokens=tokens,
            raw=resp.content,
        )


def _self_test():
    script = mock_default_script("code_qa", "semiformal")
    c = MockModelClient(script, model_id="mock")
    t1 = c.run_turn("", [], [])
    assert t1.tool_calls and t1.tool_calls[0].name == "grep"
    t2 = c.run_turn("", [], [])
    assert t2.tool_calls[0].name == "read_file"
    t3 = c.run_turn("", [], [])
    assert t3.final and t3.final["certificate"]["confidence"] == "low"
    # exhaustion is safe
    t4 = c.run_turn("", [], [])
    assert t4.final == {"answer": {}, "certificate": None}
    print("[model_client] self-test PASS (mock 3-step + exhaustion)")


if __name__ == "__main__":
    import sys

    if "--self-test" in sys.argv:
        _self_test()
    else:
        print(
            json.dumps(
                [t.__dict__ for t in mock_default_script("code_qa", "standard")],
                default=str,
                indent=2,
            )
        )

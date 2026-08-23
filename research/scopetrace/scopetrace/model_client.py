"""Model boundary: an OpenAI-compatible client and an offline replay backend.

The harness talks to models through one narrow interface so that the inference
backend and the model can change without changing challenge semantics. Two
implementations ship here. The replay client serves a scripted list of
responses and is what every test uses: it is fully offline and deterministic.
The OpenAI-compatible client posts to a ``/v1/chat/completions`` endpoint using
``urllib.request`` from the standard library.

Importing and constructing either client touches nothing outside the process.
Only :meth:`OpenAICompatibleClient.complete` opens a socket, so a test suite
can import this module, build a client, and inspect the request it would send
without any network at all.

Parsing is strict. A response that does not carry a usable message, or a tool
call whose arguments are not valid JSON, raises :class:`ParseFailure` rather
than being coerced into something plausible. Backend failures are not
behavioural results, and silently repaired tool arguments would make them look
like one.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

from .ids import FinishReason
from .manifest import SamplingConfig, digest_text

DEFAULT_CHAT_PATH: Final[str] = "/v1/chat/completions"
DEFAULT_TIMEOUT_S: Final[float] = 120.0
DEFAULT_USER_AGENT: Final[str] = "scopetrace/0.1"

ROLE_SYSTEM: Final[str] = "system"
ROLE_USER: Final[str] = "user"
ROLE_ASSISTANT: Final[str] = "assistant"
ROLE_TOOL: Final[str] = "tool"

PARSE_NO_CHOICES: Final[str] = "no_choices"
PARSE_NO_MESSAGE: Final[str] = "no_message"
PARSE_BAD_TOOL_ARGUMENTS: Final[str] = "bad_tool_arguments"
PARSE_EMPTY_RESPONSE: Final[str] = "empty_response"
PARSE_UNKNOWN_FINISH_REASON: Final[str] = "unknown_finish_reason"
PARSE_TRANSPORT_ERROR: Final[str] = "transport_error"

FINISH_REASONS: Final[Mapping[str, FinishReason]] = {
    "tool_call": FinishReason.TOOL_CALL,
    "tool_calls": FinishReason.TOOL_CALL,
    "function_call": FinishReason.TOOL_CALL,
    "stop": FinishReason.STOP,
    "end_turn": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "max_tokens": FinishReason.LENGTH,
    "error": FinishReason.ERROR,
}
"""Backend finish-reason spellings this version accepts. The table is short on
purpose: an unlisted value is reported rather than assumed to mean a clean
stop, because a truncated trajectory that looks complete is a silent error."""


class ParseFailure(Exception):
    """A model response could not be turned into a usable :class:`ModelResponse`.

    Carries a stable ``reason_code`` for aggregation and the digest of the raw
    payload so the offending text can be found in the run's text store without
    embedding it in an event payload.
    """

    def __init__(
        self,
        message: str,
        *,
        reason_code: str,
        raw_text: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.raw_text = raw_text
        self.raw_digest = digest_text(raw_text) if raw_text is not None else None


@dataclass(frozen=True, slots=True)
class ModelToolCall:
    """A tool call as it arrives from the model, arguments still unparsed.

    Keeping the raw JSON string is deliberate: the digest recorded in the event
    stream is of what the model actually emitted, not of a re-serialized
    version of it.
    """

    call_id: str
    tool_name: str
    arguments_json: str

    def parsed_arguments(self) -> dict[str, Any]:
        """Parse the argument JSON, raising :class:`ParseFailure` if malformed."""
        raw = self.arguments_json if isinstance(self.arguments_json, str) else None
        try:
            parsed = json.loads(self.arguments_json, parse_constant=reject_constant)
        except (TypeError, ValueError) as error:
            raise ParseFailure(
                f"tool call {self.call_id} for {self.tool_name} carries "
                "arguments that are not JSON",
                reason_code=PARSE_BAD_TOOL_ARGUMENTS,
                raw_text=raw,
            ) from error
        if not isinstance(parsed, dict):
            raise ParseFailure(
                f"tool call {self.call_id} for {self.tool_name} carries "
                "arguments that are not a JSON object",
                reason_code=PARSE_BAD_TOOL_ARGUMENTS,
                raw_text=raw,
            )
        return parsed

    def to_wire(self) -> dict[str, Any]:
        """Return the OpenAI-compatible representation of this tool call."""
        return {
            "id": self.call_id,
            "type": "function",
            "function": {"name": self.tool_name, "arguments": self.arguments_json},
        }


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One message in the conversation sent to the model."""

    role: str
    content: str | None = None
    tool_calls: tuple[ModelToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    def to_wire(self) -> dict[str, Any]:
        """Return the OpenAI-compatible representation of this message."""
        wire: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            wire["tool_calls"] = [call.to_wire() for call in self.tool_calls]
        if self.tool_call_id is not None:
            wire["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            wire["name"] = self.name
        return wire

    @classmethod
    def system(cls, content: str) -> "ChatMessage":
        """Build a system message."""
        return cls(role=ROLE_SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "ChatMessage":
        """Build a user message."""
        return cls(role=ROLE_USER, content=content)

    @classmethod
    def assistant(
        cls, content: str | None = None, tool_calls: Sequence[ModelToolCall] = ()
    ) -> "ChatMessage":
        """Build an assistant message, optionally carrying tool calls."""
        return cls(role=ROLE_ASSISTANT, content=content, tool_calls=tuple(tool_calls))

    @classmethod
    def tool(cls, content: str, *, tool_call_id: str, name: str) -> "ChatMessage":
        """Build a tool result message answering one call."""
        return cls(
            role=ROLE_TOOL, content=content, tool_call_id=tool_call_id, name=name
        )


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token counts for one inference call.

    ``reasoning_tokens`` is zero for models that expose no reasoning channel
    and is reported separately from output tokens where a backend distinguishes
    them, so a reasoning-budget comparison does not silently change the meaning
    of the output count.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Return the element-wise sum, for per-run totals."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Return the usage block as an event payload fragment."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """One parsed model response.

    ``raw`` keeps the decoded backend payload for diagnosis; it is never
    written into an event. ``reasoning_content`` holds a private reasoning
    channel if the backend returned one, and is treated as sensitive auxiliary
    data: the harness must work identically when it is absent.
    """

    content: str | None = None
    tool_calls: tuple[ModelToolCall, ...] = ()
    finish_reason: FinishReason = FinishReason.STOP
    usage: TokenUsage = field(default_factory=TokenUsage)
    reasoning_content: str | None = None
    latency_ns: int = 0
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        """Report whether the model asked for at least one tool call."""
        return bool(self.tool_calls)

    def text_digest(self) -> str:
        """Return the ``sha256:<hex>`` digest of the assistant text, or of ''."""
        return digest_text(self.content or "")

    def to_assistant_message(self) -> ChatMessage:
        """Return the assistant message to append to the conversation."""
        return ChatMessage.assistant(self.content, self.tool_calls)


class ModelClient(Protocol):
    """The inference boundary the agent depends on.

    Implementations are synchronous and stateless with respect to the
    conversation: the agent owns the message history and passes the whole list
    on every call.
    """

    model_id: str

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> ModelResponse:
        """Return one completion, raising :class:`ParseFailure` on bad output."""


class ReplayModelClient:
    """Serves a scripted list of responses. Offline and deterministic.

    Each :meth:`complete` call pops the next entry. An entry that is a
    :class:`ParseFailure` is raised rather than returned, which is how a test
    exercises the retry path. Running past the end of the script raises
    :class:`ReplayExhausted`, so a loop that takes more turns than the script
    expected fails loudly instead of hanging.
    """

    def __init__(
        self,
        responses: Sequence[ModelResponse | ParseFailure],
        *,
        model_id: str = "replay",
        model_revision: str = "scripted",
    ) -> None:
        self.model_id = model_id
        self.model_revision = model_revision
        self.responses: tuple[ModelResponse | ParseFailure, ...] = tuple(responses)
        self._index = 0
        self._requests: list[tuple[tuple[ChatMessage, ...], tuple[str, ...]]] = []

    @property
    def calls(self) -> int:
        """Return how many completions have been served."""
        return self._index

    @property
    def requests(self) -> tuple[tuple[tuple[ChatMessage, ...], tuple[str, ...]], ...]:
        """Return the recorded message histories and exposed tool names."""
        return tuple(self._requests)

    @property
    def exhausted(self) -> bool:
        """Report whether the script has been fully consumed."""
        return self._index >= len(self.responses)

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> ModelResponse:
        """Return the next scripted response, or raise the next scripted failure."""
        if self.exhausted:
            raise ReplayExhausted(
                f"the script for model {self.model_id!r} holds "
                f"{len(self.responses)} responses and the loop asked for "
                f"number {self._index + 1}; extend the script or lower the "
                "turn budget"
            )
        self._requests.append(
            (tuple(messages), tuple(wire_tool_name(t) for t in tools))
        )
        entry = self.responses[self._index]
        self._index += 1
        if isinstance(entry, ParseFailure):
            raise entry
        return entry

    def reset(self) -> None:
        """Rewind the script so the same client can drive a second run."""
        self._index = 0
        self._requests.clear()


class ReplayExhausted(RuntimeError):
    """The agent asked for more completions than the script provides."""


class OpenAICompatibleClient:
    """Client for an OpenAI-compatible ``/v1/chat/completions`` endpoint.

    Construction performs no I/O: the base URL, model id, sampling parameters,
    and optional extra body fields are recorded and nothing is contacted until
    :meth:`complete` is called. Only the standard library is used, so the
    package pulls in no HTTP dependency for a feature the offline tests never
    exercise.
    """

    def __init__(
        self,
        base_url: str,
        model_id: str,
        *,
        api_key: str | None = None,
        sampling: SamplingConfig | None = None,
        chat_path: str = DEFAULT_CHAT_PATH,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        extra_body: Mapping[str, Any] | None = None,
        model_revision: str = "unknown",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.api_key = api_key
        self.sampling = sampling if sampling is not None else SamplingConfig()
        self.chat_path = chat_path
        self.timeout_s = timeout_s
        self.extra_body: Mapping[str, Any] = dict(extra_body or {})
        self.model_revision = model_revision

    @property
    def endpoint(self) -> str:
        """Return the full URL completions are posted to."""
        return f"{self.base_url}{self.chat_path}"

    def build_payload(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        """Build the request body. Pure, so a test can assert on it offline."""
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages_to_wire(messages),
            "temperature": self.sampling.temperature,
            "top_p": self.sampling.top_p,
            "max_tokens": self.sampling.max_output_tokens,
            "seed": self.sampling.seed,
            "stream": False,
        }
        if tools:
            payload["tools"] = [dict(tool) for tool in tools]
            payload["tool_choice"] = "auto"
        payload.update(self.extra_body)
        return payload

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> ModelResponse:
        """Post one completion request and parse the response.

        This is the only method that touches the network. Transport errors are
        surfaced as :class:`ParseFailure` with the transport reason code, so a
        caller has one failure type to handle and one place to decide whether a
        cell is invalid.
        """
        payload = self.build_payload(messages, tools)
        started_ns = time.monotonic_ns()
        body = self._post(payload)
        latency_ns = time.monotonic_ns() - started_ns
        return self.parse_response(body, latency_ns=latency_ns)

    def _post(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Post a JSON body and return the decoded response."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": DEFAULT_USER_AGENT,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            raise ParseFailure(
                f"request to {self.endpoint} failed: {error}",
                reason_code=PARSE_TRANSPORT_ERROR,
            ) from error
        except ValueError as error:
            raise ParseFailure(
                f"response from {self.endpoint} is not JSON",
                reason_code=PARSE_TRANSPORT_ERROR,
            ) from error
        if not isinstance(decoded, Mapping):
            raise ParseFailure(
                f"response from {self.endpoint} is not a JSON object",
                reason_code=PARSE_TRANSPORT_ERROR,
                raw_text=raw_payload_text(decoded),
            )
        return decoded

    @staticmethod
    def parse_response(
        payload: Mapping[str, Any], *, latency_ns: int = 0
    ) -> ModelResponse:
        """Parse a backend response body into a :class:`ModelResponse`.

        Static so that a recorded payload can be parsed in a test without a
        client instance. Raises :class:`ParseFailure` for a missing choice, a
        missing message, or an unusable tool call.
        """
        raw_text = raw_payload_text(payload)
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ParseFailure(
                "response carries no choices",
                reason_code=PARSE_NO_CHOICES,
                raw_text=raw_text,
            )
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, Mapping) else None
        if not isinstance(message, Mapping):
            raise ParseFailure(
                "first choice carries no message",
                reason_code=PARSE_NO_MESSAGE,
                raw_text=raw_text,
            )
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise ParseFailure(
                "message content is neither text nor absent",
                reason_code=PARSE_NO_MESSAGE,
                raw_text=raw_text,
            )
        tool_calls = parse_tool_calls(message.get("tool_calls"), raw_text=raw_text)
        finish_reason = parse_finish_reason(
            choice.get("finish_reason") if isinstance(choice, Mapping) else None
        )
        if not tool_calls and not (content or "").strip():
            raise ParseFailure(
                "message carries neither text nor a tool call",
                reason_code=PARSE_EMPTY_RESPONSE,
                raw_text=raw_text,
            )
        reasoning = message.get("reasoning_content")
        if not isinstance(reasoning, str):
            reasoning = message.get("reasoning")
        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=parse_usage(payload.get("usage")),
            reasoning_content=reasoning if isinstance(reasoning, str) else None,
            latency_ns=latency_ns,
            raw=payload,
        )


def parse_finish_reason(value: str | None) -> FinishReason:
    """Map a backend finish reason onto the enum.

    ``tool_calls`` and ``function_call`` both map to ``tool_call``. An
    unrecognized value raises :class:`ParseFailure`, because silently treating
    it as a normal stop would hide a truncated trajectory.
    """
    if value is None:
        return FinishReason.STOP
    mapped = FINISH_REASONS.get(value) if isinstance(value, str) else None
    if mapped is None:
        raise ParseFailure(
            f"unrecognized finish reason: {value!r}",
            reason_code=PARSE_UNKNOWN_FINISH_REASON,
        )
    return mapped


def messages_to_wire(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    """Convert a message history to the wire form."""
    return [message.to_wire() for message in messages]


def conversation_digest(messages: Sequence[ChatMessage]) -> str:
    """Return a ``sha256:<hex>`` digest of the conversation sent to the model."""
    return digest_text(
        json.dumps(messages_to_wire(messages), sort_keys=True, separators=(",", ":"))
    )


def wire_tool_name(tool: Mapping[str, Any]) -> str:
    """Return the name of a tool from its OpenAI-compatible declaration."""
    function = tool.get("function")
    if isinstance(function, Mapping):
        name = function.get("name")
        if isinstance(name, str):
            return name
    name = tool.get("name")
    return name if isinstance(name, str) else ""


def parse_tool_calls(
    raw_calls: Any, *, raw_text: str | None = None
) -> tuple[ModelToolCall, ...]:
    """Parse the tool-call block of a backend message.

    Argument strings are checked here rather than at first use, so an unusable
    call is a parse failure of the response that carried it and not a mystery
    several layers later. A backend that returns arguments as an object instead
    of a string is accepted and canonicalized, which is a shape difference
    rather than a repair of malformed output.
    """
    if raw_calls in (None, []):
        return ()
    if not isinstance(raw_calls, list):
        raise ParseFailure(
            "tool_calls is not a list",
            reason_code=PARSE_BAD_TOOL_ARGUMENTS,
            raw_text=raw_text,
        )
    calls: list[ModelToolCall] = []
    for index, entry in enumerate(raw_calls):
        function = entry.get("function") if isinstance(entry, Mapping) else None
        name = function.get("name") if isinstance(function, Mapping) else None
        if not isinstance(name, str) or not name:
            raise ParseFailure(
                f"tool call {index} names no function",
                reason_code=PARSE_BAD_TOOL_ARGUMENTS,
                raw_text=raw_text,
            )
        arguments = function.get("arguments", "{}")
        if isinstance(arguments, Mapping):
            arguments = json.dumps(
                dict(arguments), sort_keys=True, separators=(",", ":")
            )
        if not isinstance(arguments, str):
            raise ParseFailure(
                f"tool call {index} for {name} carries arguments that are "
                "neither text nor an object",
                reason_code=PARSE_BAD_TOOL_ARGUMENTS,
                raw_text=raw_text,
            )
        call_id = entry.get("id") if isinstance(entry, Mapping) else None
        call = ModelToolCall(
            call_id=(
                call_id if isinstance(call_id, str) and call_id else f"call-{index}"
            ),
            tool_name=name,
            arguments_json=arguments,
        )
        call.parsed_arguments()
        calls.append(call)
    return tuple(calls)


def parse_usage(block: Any) -> TokenUsage:
    """Read the token counts from a backend usage block, defaulting to zero."""
    if not isinstance(block, Mapping):
        return TokenUsage()
    details = block.get("completion_tokens_details")
    reasoning = (
        details.get("reasoning_tokens") if isinstance(details, Mapping) else None
    )
    return TokenUsage(
        input_tokens=count_or_zero(block.get("prompt_tokens")),
        output_tokens=count_or_zero(block.get("completion_tokens")),
        reasoning_tokens=count_or_zero(reasoning),
    )


def count_or_zero(value: Any) -> int:
    """Return a non-negative integer count, or zero for anything unusable."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


def reject_constant(literal: str) -> Any:
    """Refuse the JSON constants canonical JSON cannot carry.

    ``NaN`` and the infinities parse by default and then fail to serialize,
    which would turn a malformed tool call into an exception several layers
    away from the boundary that accepted it.
    """
    raise ValueError(f"{literal} is not a value canonical JSON can carry")


def raw_payload_text(payload: Any) -> str:
    """Render a backend payload as text for the run's content-addressed store."""
    return json.dumps(payload, sort_keys=True, default=str)

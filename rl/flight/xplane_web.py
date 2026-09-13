"""X-Plane 12 local Web API client for live flight controllers."""

from __future__ import annotations

import json
import math
import re
import socket
import time
from collections.abc import Callable, Mapping, Sequence
from urllib.parse import urlencode
from urllib.request import urlopen

INDEXED_DATAREF = re.compile(r"^(?P<name>.+)\[(?P<index>[0-9]+)\]$")


def dataref_query(names: Sequence[str]) -> str:
    """Build an exact-name dataref query for X-Plane's REST API."""

    if not names or any(not name for name in names):
        raise ValueError("dataref names must be nonempty")
    return "/datarefs?" + urlencode(
        [("filter[name]", name) for name in names]
    )


class XPlaneWeb:
    """Read and write scalar datarefs through REST and WebSocket APIs."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8086,
        *,
        api_version: str = "v3",
        opener: Callable = urlopen,
        socket_factory: Callable | None = None,
    ):
        if port < 1 or port > 65535:
            raise ValueError("X-Plane Web API port must be valid")
        self.http_root = f"http://{host}:{port}/api/{api_version}"
        self.websocket_url = f"ws://{host}:{port}/api/{api_version}"
        self.opener = opener
        self.socket_factory = socket_factory
        self.socket = None
        self.subscriptions: dict[str, tuple[int, str]] = {}
        self.ids: dict[str, int] = {}
        self.indexes: dict[str, int | None] = {}
        self.values: dict[str, float] = {}
        self.last_update_ns = 0
        self.request_id = 0

    def _connect(self) -> None:
        if self.socket is not None:
            return
        if self.socket_factory is None:
            try:
                import websocket
            except ImportError as error:
                raise RuntimeError(
                    "the websocket-client package is required"
                ) from error
            self.socket = websocket.create_connection(
                self.websocket_url,
                timeout=2.0,
                http_proxy_host=None,
            )
        else:
            self.socket = self.socket_factory(self.websocket_url, timeout=2.0)

    def _request_json(self, path: str) -> object:
        with self.opener(self.http_root + path, timeout=2.0) as response:
            return json.loads(response.read())

    def _resolve(self, names: Sequence[str]) -> None:
        missing = [name for name in names if name not in self.ids]
        if not missing:
            return
        requested = {}
        for name in missing:
            match = INDEXED_DATAREF.fullmatch(name)
            base = match.group("name") if match else name
            index = int(match.group("index")) if match else None
            requested[name] = (base, index)
        payload = self._request_json(
            dataref_query(sorted({base for base, _index in requested.values()}))
        )
        if not isinstance(payload, dict) or not isinstance(
            payload.get("data"), list
        ):
            raise RuntimeError("X-Plane returned an invalid dataref list")
        identifiers = {}
        for item in payload["data"]:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            identifier = item.get("id")
            if isinstance(name, str) and isinstance(identifier, int):
                identifiers[name] = identifier
        for original, (base, index) in requested.items():
            if base in identifiers:
                self.ids[original] = identifiers[base]
                self.indexes[original] = index
        unresolved = sorted(set(missing) - set(self.ids))
        if unresolved:
            raise RuntimeError(f"X-Plane datarefs not found: {unresolved}")

    def _send(self, message_type: str, params: dict) -> None:
        if self.socket is None:
            raise RuntimeError("X-Plane WebSocket is not connected")
        self.request_id += 1
        self.socket.send(
            json.dumps(
                {
                    "req_id": self.request_id,
                    "type": message_type,
                    "params": params,
                },
                separators=(",", ":"),
            )
        )

    def subscribe(self, datarefs: Mapping[str, str], frequency_hz: int) -> None:
        if self.subscriptions:
            raise RuntimeError("X-Plane datarefs are already subscribed")
        if frequency_hz < 1:
            raise ValueError("subscription frequency must be positive")
        self._resolve(list(datarefs.values()))
        self._connect()
        self.subscriptions = {
            key: (self.ids[name], name) for key, name in datarefs.items()
        }
        self._send(
            "dataref_subscribe_values",
            {
                "datarefs": [
                    {"id": identifier}
                    for identifier, _name in self.subscriptions.values()
                ]
            },
        )

    def receive(self, timeout_s: float) -> dict[str, float]:
        if self.socket is None:
            raise RuntimeError("X-Plane WebSocket is not connected")
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("receive timeout must be positive")
        self.socket.settimeout(timeout_s)
        while True:
            try:
                payload = self.socket.recv()
            except TimeoutError as error:
                raise socket.timeout from error
            except Exception as error:
                if error.__class__.__name__ == "WebSocketTimeoutException":
                    raise socket.timeout from error
                raise
            if payload in ("", b""):
                raise ConnectionError(
                    "X-Plane closed the WebSocket connection"
                )
            try:
                message = json.loads(payload)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise RuntimeError(
                    "X-Plane returned an invalid WebSocket message"
                ) from error
            if not isinstance(message, dict):
                raise RuntimeError(
                    "X-Plane returned a non-object WebSocket message"
                )
            if message.get("type") == "result":
                if not message.get("success"):
                    detail = message.get("error_message", "unknown error")
                    raise RuntimeError(f"X-Plane Web API request failed: {detail}")
                continue
            if message.get("type") != "dataref_update_values":
                continue
            updates = message.get("data")
            if not isinstance(updates, dict):
                raise RuntimeError("X-Plane returned an invalid dataref update")
            reverse = {
                str(identifier): key
                for key, (identifier, _name) in self.subscriptions.items()
            }
            for identifier, value in updates.items():
                key = reverse.get(identifier)
                if key is not None and isinstance(value, (int, float)):
                    if not math.isfinite(value):
                        raise RuntimeError("X-Plane returned a non-finite value")
                    self.values[key] = float(value)
            self.last_update_ns = time.monotonic_ns()
            return dict(self.values)

    def fresh(self, maximum_age_s: float) -> bool:
        if not math.isfinite(maximum_age_s) or maximum_age_s <= 0.0:
            raise ValueError("maximum sample age must be positive")
        if set(self.values) != set(self.subscriptions):
            return False
        threshold_ns = time.monotonic_ns() - round(maximum_age_s * 1e9)
        return self.last_update_ns >= threshold_ns

    def write_many(self, values: Sequence[tuple[str, float]]) -> None:
        if not values:
            return
        if any(not math.isfinite(value) for _name, value in values):
            raise ValueError("DREF value must be finite")
        self._resolve([name for name, _value in values])
        self._connect()
        self._send(
            "dataref_set_values",
            {
                "datarefs": [
                    {
                        "id": self.ids[name],
                        "value": float(value),
                        **(
                            {"index": self.indexes[name]}
                            if self.indexes[name] is not None
                            else {}
                        ),
                    }
                    for name, value in values
                ]
            },
        )

    def write(self, dataref: str, value: float) -> None:
        self.write_many(((dataref, value),))

    def close(self) -> None:
        if self.socket is None:
            return
        try:
            self._send("dataref_unsubscribe_values", {"datarefs": "all"})
        except Exception:
            pass
        try:
            self.socket.close()
        finally:
            self.socket = None
            self.subscriptions.clear()

    def __enter__(self) -> XPlaneWeb:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()

"""Small X-Plane RREF/DREF client for local flight-control adapters."""

from __future__ import annotations

import math
import socket
import struct
import time
from collections.abc import Mapping

RREF_NAME_BYTES = 400
DREF_NAME_BYTES = 500
RREF_HEADER = b"RREF\x00"
DREF_HEADER = b"DREF\x00"


def _encoded_name(name: str, width: int) -> bytes:
    encoded = name.encode("ascii")
    if not encoded or b"\x00" in encoded or len(encoded) >= width:
        raise ValueError("dataref name must be nonempty ASCII shorter than field")
    return encoded + bytes(width - len(encoded))


def pack_rref_request(dataref: str, index: int, frequency_hz: int) -> bytes:
    """Encode one X-Plane dataref subscription request."""

    if index < 0 or index > 2**31 - 1:
        raise ValueError("RREF index must fit a signed 32-bit integer")
    if frequency_hz < 0 or frequency_hz > 1000:
        raise ValueError("RREF frequency must be between 0 and 1000 Hz")
    return (
        RREF_HEADER
        + struct.pack("<ii", frequency_hz, index)
        + _encoded_name(dataref, RREF_NAME_BYTES)
    )


def unpack_rref_response(packet: bytes) -> dict[int, float]:
    """Decode one X-Plane RREF response packet."""

    if not packet.startswith(RREF_HEADER):
        raise ValueError("packet is not an RREF response")
    payload = packet[len(RREF_HEADER) :]
    if not payload or len(payload) % 8:
        raise ValueError("RREF response payload is malformed")
    values = {}
    for offset in range(0, len(payload), 8):
        index, value = struct.unpack_from("<if", payload, offset)
        if index < 0 or not math.isfinite(value):
            raise ValueError("RREF response contains an invalid sample")
        values[index] = float(value)
    return values


def pack_dref_write(dataref: str, value: float) -> bytes:
    """Encode one X-Plane scalar dataref write."""

    if not math.isfinite(value):
        raise ValueError("DREF value must be finite")
    return (
        DREF_HEADER
        + struct.pack("<f", value)
        + _encoded_name(dataref, DREF_NAME_BYTES)
    )


class XPlaneUdp:
    """Subscribe to float datarefs and write scalar controls over UDP."""

    def __init__(self, host: str = "127.0.0.1", port: int = 49000):
        if port < 1 or port > 65535:
            raise ValueError("X-Plane UDP port must be between 1 and 65535")
        self.address = (host, port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.connect(self.address)
        self.subscriptions: dict[str, tuple[int, str]] = {}
        self.values: dict[str, float] = {}
        self.updated_ns: dict[str, int] = {}

    def subscribe(self, datarefs: Mapping[str, str], frequency_hz: int) -> None:
        if self.subscriptions:
            raise RuntimeError("X-Plane datarefs are already subscribed")
        for index, (key, dataref) in enumerate(datarefs.items()):
            self.socket.send(pack_rref_request(dataref, index, frequency_hz))
            self.subscriptions[key] = (index, dataref)

    def receive(self, timeout_s: float) -> dict[str, float]:
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("receive timeout must be positive")
        self.socket.settimeout(timeout_s)
        packet = self.socket.recv(64 * 1024)
        received_ns = time.monotonic_ns()
        reverse = {
            index: key for key, (index, _dataref) in self.subscriptions.items()
        }
        for index, value in unpack_rref_response(packet).items():
            key = reverse.get(index)
            if key is not None:
                self.values[key] = value
                self.updated_ns[key] = received_ns
        return dict(self.values)

    def fresh(self, maximum_age_s: float) -> bool:
        if not math.isfinite(maximum_age_s) or maximum_age_s <= 0.0:
            raise ValueError("maximum sample age must be positive")
        if set(self.updated_ns) != set(self.subscriptions):
            return False
        threshold_ns = time.monotonic_ns() - round(maximum_age_s * 1e9)
        return all(updated >= threshold_ns for updated in self.updated_ns.values())

    def write(self, dataref: str, value: float) -> None:
        self.socket.send(pack_dref_write(dataref, value))

    def close(self) -> None:
        for index, dataref in self.subscriptions.values():
            try:
                self.socket.send(pack_rref_request(dataref, index, 0))
            except OSError:
                pass
        self.subscriptions.clear()
        self.socket.close()

    def __enter__(self) -> XPlaneUdp:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()

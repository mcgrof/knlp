"""Tests for the X-Plane local Web API transport."""

import json

import pytest

from rl.flight.xplane_web import XPlaneRest, XPlaneWeb, dataref_query


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback

    def read(self):
        return json.dumps(self.payload).encode()


class FakeSocket:
    def __init__(self):
        self.sent = []
        self.messages = []
        self.timeout = None
        self.closed = False

    def send(self, message):
        self.sent.append(json.loads(message))

    def recv(self):
        return json.dumps(self.messages.pop(0))

    def settimeout(self, timeout):
        self.timeout = timeout

    def close(self):
        self.closed = True


class RawFakeSocket(FakeSocket):
    def recv(self):
        return self.messages.pop(0)


def test_dataref_query_uses_repeated_exact_name_filters():
    query = dataref_query(("sim/time/paused", "sim/flightmodel/position/phi"))
    assert query.startswith("/datarefs?")
    assert query.count("filter%5Bname%5D=") == 2


def test_web_transport_subscribes_reads_and_batches_indexed_writes():
    identifiers = {
        "sim/time/paused": 10,
        "sim/flightmodel/engine/ENGN_thro_use": 11,
    }

    def opener(url, timeout):
        assert timeout == 2.0
        items = [
            {"id": identifier, "name": name, "value_type": "float"}
            for name, identifier in identifiers.items()
            if name.replace("/", "%2F") in url or name in url
        ]
        return FakeResponse({"data": items})

    wire = FakeSocket()
    client = XPlaneWeb(
        opener=opener,
        socket_factory=lambda url, timeout: wire,
    )
    client.subscribe({"paused": "sim/time/paused"}, 50)
    subscription = wire.sent[0]
    assert subscription["type"] == "dataref_subscribe_values"
    assert subscription["params"]["datarefs"] == [{"id": 10}]

    wire.messages = [
        {"req_id": 1, "type": "result", "success": True},
        {"type": "dataref_update_values", "data": {"10": 0}},
    ]
    assert client.receive(0.2) == {"paused": 0.0}
    assert client.fresh(1.0)

    client.write_many(
        (("sim/flightmodel/engine/ENGN_thro_use[0]", 0.75),)
    )
    write = wire.sent[-1]
    assert write["type"] == "dataref_set_values"
    assert write["params"]["datarefs"] == [
        {"id": 11, "value": 0.75, "index": 0}
    ]
    client.close()
    assert wire.closed


def test_web_transport_subscribes_to_multiple_array_indices():
    def opener(url, timeout):
        assert timeout == 2.0
        assert "sim%2Fmultiplayer%2Fcombat%2Fteam_status" in url
        return FakeResponse(
            {
                "data": [
                    {
                        "id": 20,
                        "name": "sim/multiplayer/combat/team_status",
                        "value_type": "int_array",
                    }
                ]
            }
        )

    wire = FakeSocket()
    client = XPlaneWeb(
        opener=opener,
        socket_factory=lambda url, timeout: wire,
    )
    client.subscribe(
        {
            "team_status_1": "sim/multiplayer/combat/team_status[1]",
            "team_status_2": "sim/multiplayer/combat/team_status[2]",
        },
        10,
    )
    assert wire.sent[0]["params"]["datarefs"] == [
        {"id": 20, "index": [1, 2]}
    ]
    wire.messages = [
        {"type": "dataref_update_values", "data": {"20": [1, 2]}}
    ]
    assert client.receive(0.2) == {
        "team_status_1": 1.0,
        "team_status_2": 2.0,
    }


@pytest.mark.parametrize("payload", ("", b""))
def test_web_transport_reports_a_closed_connection(payload):
    wire = RawFakeSocket()
    wire.messages = [payload]
    client = XPlaneWeb(socket_factory=lambda url, timeout: wire)
    client.socket = wire
    with pytest.raises(ConnectionError, match="closed"):
        client.receive(0.2)


@pytest.mark.parametrize("payload", ("not-json", "[]"))
def test_web_transport_rejects_invalid_messages(payload):
    wire = RawFakeSocket()
    wire.messages = [payload]
    client = XPlaneWeb(socket_factory=lambda url, timeout: wire)
    client.socket = wire
    with pytest.raises(RuntimeError, match="WebSocket message"):
        client.receive(0.2)


def test_rest_transport_polls_and_writes_indexed_datarefs():
    identifiers = {
        "sim/time/paused": 10,
        "sim/flightmodel/engine/ENGN_thro_use": 11,
    }
    calls = []

    def opener(request, timeout):
        calls.append((request, timeout))
        url = request.full_url
        if "/datarefs?" in url:
            return FakeResponse(
                {
                    "data": [
                        {"id": identifier, "name": name}
                        for name, identifier in identifiers.items()
                        if name.replace("/", "%2F") in url
                    ]
                }
            )
        if request.get_method() == "GET":
            return FakeResponse({"data": 0})
        assert request.get_method() == "PATCH"
        assert json.loads(request.data) == {"data": 0.75}
        return FakeResponse(None)

    client = XPlaneRest(opener=opener)
    client.subscribe({"paused": "sim/time/paused"}, 50)
    assert client.receive(0.2) == {"paused": 0.0}
    assert client.fresh(1.0)
    client.write_many(
        (("sim/flightmodel/engine/ENGN_thro_use[0]", 0.75),)
    )
    assert calls[-1][0].full_url.endswith("/datarefs/11/value?index=0")

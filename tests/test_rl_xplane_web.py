"""Tests for the X-Plane local Web API transport."""

import json

from rl.flight.xplane_web import XPlaneWeb, dataref_query


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

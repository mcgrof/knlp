"""Run a frozen UFO actor against X-Plane telemetry without sending controls."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import signal
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, TextIO

import numpy as np

from rl.flight.contracts import FlightContract, TelemetryFrame

MAXIMUM_WIRE_BYTES = 64 * 1024


@dataclass
class StreamStats:
    frames: int = 0
    rejected: int = 0
    first_sequence: int | None = None
    last_sequence: int | None = None
    maximum_inference_ns: int = 0
    interrupted: bool = False


class ShadowPolicy:
    """A read-only actor that produces trace records, never control messages."""

    def __init__(self, contract: FlightContract, model: Path):
        self.contract = contract
        self.model = model
        self.model_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
        with np.load(model, allow_pickle=False) as saved:
            if int(saved["format_version"]) != 1:
                raise ValueError("unsupported shadow actor format")
            if str(saved["contract_hash"]) != contract.digest:
                raise ValueError("shadow actor contract hash does not match")
            self.checkpoint_sha256 = str(saved["checkpoint_sha256"])
            self.observation_mean = saved["observation_mean"].astype(np.float32)
            self.observation_scale = saved["observation_scale"].astype(np.float32)
            self.action_mid = saved["action_mid"].astype(np.float32)
            self.action_scale = saved["action_scale"].astype(np.float32)
            self.weights = tuple(
                saved[f"layer_{index}_weight"].astype(np.float32)
                for index in range(3)
            )
            self.biases = tuple(
                saved[f"layer_{index}_bias"].astype(np.float32)
                for index in range(3)
            )
        self._validate_model()

    def _validate_model(self) -> None:
        input_width = self.contract.observation.width + self.contract.goal.width
        action_width = self.contract.action.width
        if self.observation_mean.shape != (input_width,):
            raise ValueError("shadow actor observation mean has the wrong width")
        if self.observation_scale.shape != (input_width,):
            raise ValueError("shadow actor observation scale has the wrong width")
        if self.action_mid.shape != (action_width,) or self.action_scale.shape != (
            action_width,
        ):
            raise ValueError("shadow actor action transform has the wrong width")
        expected_inputs = (input_width, self.weights[0].shape[0], self.weights[1].shape[0])
        for index, (weight, bias, expected_input) in enumerate(
            zip(self.weights, self.biases, expected_inputs)
        ):
            expected_output = action_width if index == 2 else weight.shape[0]
            if weight.ndim != 2 or weight.shape != (expected_output, expected_input):
                raise ValueError(f"shadow actor layer {index} has an invalid weight shape")
            if bias.shape != (expected_output,):
                raise ValueError(f"shadow actor layer {index} has an invalid bias shape")
        arrays = (
            self.observation_mean,
            self.observation_scale,
            self.action_mid,
            self.action_scale,
            *self.weights,
            *self.biases,
        )
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("shadow actor contains non-finite parameters")
        if np.any(self.observation_scale <= 0.0) or np.any(self.action_scale <= 0.0):
            raise ValueError("shadow actor contains an invalid scale")

    def infer(self, frame: TelemetryFrame) -> tuple[list[float], int]:
        values = np.asarray((*frame.observation, *frame.goal), dtype=np.float32)
        started = time.monotonic_ns()
        hidden = (values - self.observation_mean) / self.observation_scale
        hidden = np.tanh(self.weights[0] @ hidden + self.biases[0])
        hidden = np.tanh(self.weights[1] @ hidden + self.biases[1])
        unit_action = np.tanh(self.weights[2] @ hidden + self.biases[2])
        action = self.action_mid + self.action_scale * unit_action
        inference_ns = time.monotonic_ns() - started
        result = [float(value) for value in action]
        self.contract.action.validate(result, "shadow action")
        return result, inference_ns

    def record(self, frame: TelemetryFrame) -> dict:
        action, inference_ns = self.infer(frame)
        return {
            "schema_version": 1,
            "kind": "shadow_action",
            "checkpoint_sha256": self.checkpoint_sha256,
            "model_sha256": self.model_sha256,
            "received_monotonic_ns": time.monotonic_ns(),
            "inference_ns": inference_ns,
            "telemetry": json.loads(frame.to_wire()),
            "proposed_action": action,
        }


def process_stream(
    source: BinaryIO,
    output: TextIO,
    policy: ShadowPolicy,
    *,
    max_frames: int | None = None,
) -> StreamStats:
    stats = StreamStats()
    episode_id = None
    last_sequence = None
    last_monotonic_ns = None
    try:
        while max_frames is None or stats.frames < max_frames:
            line = source.readline(MAXIMUM_WIRE_BYTES + 1)
            if not line:
                break
            if len(line) > MAXIMUM_WIRE_BYTES or not line.endswith(b"\n"):
                stats.rejected += 1
                continue
            try:
                frame = TelemetryFrame.from_wire(line, policy.contract)
                if frame.episode_id == episode_id:
                    if frame.sequence <= last_sequence:
                        raise ValueError("telemetry sequence did not increase")
                    if frame.monotonic_ns < last_monotonic_ns:
                        raise ValueError("telemetry monotonic time moved backwards")
                else:
                    episode_id = frame.episode_id
                record = policy.record(frame)
            except ValueError:
                stats.rejected += 1
                continue
            output.write(
                json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n"
            )
            output.flush()
            last_sequence = frame.sequence
            last_monotonic_ns = frame.monotonic_ns
            if stats.first_sequence is None:
                stats.first_sequence = frame.sequence
            stats.last_sequence = frame.sequence
            stats.maximum_inference_ns = max(
                stats.maximum_inference_ns, record["inference_ns"]
            )
            stats.frames += 1
    except KeyboardInterrupt:
        stats.interrupted = True
    return stats


def default_socket_path() -> Path:
    configured = os.environ.get("XPLANE_UFO_TELEMETRY_SOCKET")
    if configured:
        return Path(configured).expanduser()
    return Path(f"/tmp/xplane-ufo-telemetry-{os.getuid()}.sock")


def connect_telemetry(path: Path, timeout: float) -> socket.socket:
    """Wait for a live listener, tolerating absent or stale socket paths."""

    deadline = time.monotonic() + timeout
    last_error = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError(
                f"timed out waiting for X-Plane telemetry at {path}"
            ) from last_error
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(min(1.0, remaining))
        try:
            client.connect(str(path))
        except OSError as error:
            client.close()
            if error.errno not in {None, errno.ENOENT, errno.ECONNREFUSED}:
                raise
            last_error = error
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
            continue
        client.settimeout(None)
        return client


def request_stop(signum, frame) -> None:
    del signum, frame
    raise KeyboardInterrupt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, default=default_socket_path())
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--connect-timeout", type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if args.connect_timeout <= 0:
        parser.error("--connect-timeout must be positive")
    contract = FlightContract.from_json(args.contract)
    policy = ShadowPolicy(contract, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary or args.output.with_suffix(args.output.suffix + ".summary.json")
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    with connect_telemetry(args.socket, args.connect_timeout) as client:
        with client.makefile("rb") as source, args.output.open("x") as output:
            stats = process_stream(source, output, policy, max_frames=args.max_frames)
    summary = {
        "schema_version": 1,
        "mode": "shadow_read_only",
        "socket": str(args.socket),
        "contract_hash": contract.digest,
        "checkpoint_sha256": policy.checkpoint_sha256,
        "model_sha256": policy.model_sha256,
        **asdict(stats),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if stats.frames else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Client for the Extreme Tux Racer headless bridge (``etr --rl``).

One bridge is one game process driven over stdin/stdout with the line
protocol documented in rl/etr/protocol.md. Requests are single lines;
every reply is one JSON object. The process never opens a window, so
any number of bridges can run on a CPU-only host or next to other GPU
work.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any, Optional

DEFAULT_DT = 1.0 / 60.0

# Raw observation keys every backend (game bridge or simulator) returns.
RAW_KEYS = (
    "ok",
    "tick",
    "time",
    "pos",
    "vel",
    "speed",
    "way",
    "progress",
    "herring",
    "items_left",
    "collisions",
    "airborne",
    "braking",
    "paddling",
    "jumping",
    "turn",
    "friction",
    "nml",
    "finish",
    "done",
    "size",
    "play",
    "probe_h",
    "probe_f",
    "trees",
    "items",
)

# Forward probe fan the bridge samples: 4 distances x 5 lateral offsets.
PROBE_AHEAD = (3.0, 6.0, 12.0, 24.0)
PROBE_SIDE = (-6.0, -3.0, 0.0, 3.0, 6.0)
NEAREST = 4


class BridgeError(RuntimeError):
    pass


def find_binary(explicit: Optional[str] = None) -> str:
    """Locate the patched game binary.

    Order: explicit argument, ``$ETR_RL_BIN``, ``~/devel/extremetuxracer-0.8.4/src/etr``,
    then ``etr`` on PATH (only useful once the patch is installed).
    """
    candidates = [
        explicit,
        os.environ.get("ETR_RL_BIN"),
        os.path.expanduser("~/devel/extremetuxracer-0.8.4/src/etr"),
        shutil.which("etr"),
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    raise BridgeError(
        "no ETR binary with the --rl bridge found; build it with "
        "rl/etr/build_etr.sh or set ETR_RL_BIN"
    )


class EtrBridge:
    """A live ``etr --rl`` process."""

    def __init__(
        self,
        binary: Optional[str] = None,
        stderr_path: Optional[str] = None,
        env: Optional[dict] = None,
    ):
        self.binary = find_binary(binary)
        run_env = dict(os.environ)
        # The bridge must never see a display; SFML would try to use it.
        run_env.pop("DISPLAY", None)
        run_env.pop("WAYLAND_DISPLAY", None)
        if env:
            run_env.update(env)
        self._stderr = open(stderr_path, "ab") if stderr_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(
            [self.binary, "--rl"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            env=run_env,
            bufsize=1,
            text=True,
        )
        self.course_info: Optional[dict] = None

    # ---- protocol ---------------------------------------------------

    def _request(self, line: str) -> dict:
        if self.proc.poll() is not None:
            raise BridgeError(f"etr exited with status {self.proc.returncode}")
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        reply = self.proc.stdout.readline()
        if not reply:
            raise BridgeError("etr closed its stdout")
        try:
            data = json.loads(reply)
        except ValueError as e:
            raise BridgeError(f"bad reply from etr: {reply!r}") from e
        if not data.get("ok", False):
            raise BridgeError(data.get("error", "unknown error"))
        return data

    def info(self) -> dict:
        return self._request("info")

    def reset(
        self,
        seed: int = 0,
        course: str = "bunny_hill",
        group: str = "default",
        wind: int = 0,
        light: int = 0,
        mirror: bool = False,
        dt: float = DEFAULT_DT,
    ) -> dict:
        return self._request(
            f"reset seed={int(seed)} course={course} group={group} wind={int(wind)} "
            f"light={int(light)} mirror={int(mirror)} dt={dt:.9g}"
        )

    def step(
        self,
        turn: float = 0.0,
        brake: bool = False,
        paddle: bool = False,
        jump: bool = False,
        ticks: int = 1,
        dt: Optional[float] = None,
    ) -> dict:
        line = (
            f"step turn={float(turn):.6g} brake={int(bool(brake))} "
            f"paddle={int(bool(paddle))} jump={int(bool(jump))} ticks={int(ticks)}"
        )
        if dt is not None:
            line += f" dt={dt:.9g}"
        return self._request(line)

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                assert self.proc.stdin is not None
                self.proc.stdin.write("quit\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                self.proc.kill()
                self.proc.wait()
        if self._stderr is not subprocess.DEVNULL:
            self._stderr.close()

    def __enter__(self) -> "EtrBridge":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

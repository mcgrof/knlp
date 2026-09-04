"""A small pure-Python stand-in for the game bridge.

It speaks the same raw-observation dictionary as ``etr --rl`` so the
Gymnasium contract, the controls and the trainer can be exercised on
any machine (and in CI) without the game. The dynamics are a
deliberately crude point-mass on a slope: gravity along the course,
friction that depends on a terrain band, steering that rotates the
velocity, paddling and braking as bounded accelerations, round trees
that bounce the racer, and round items that count as herrings. It is
not a model of the game; it is a contract test double that a policy
can learn on in a minute of CPU time.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from rl.envs.etr_bridge import DEFAULT_DT, NEAREST, PROBE_AHEAD, PROBE_SIDE

COURSES = {
    # dir: (width, length, play_width, play_length, start_x, n_trees, n_items)
    "sim_flat": (60.0, 300.0, 60.0, 300.0, 30.0, 0, 20),
    "sim_trees": (60.0, 400.0, 60.0, 400.0, 30.0, 40, 30),
    "sim_ice": (60.0, 400.0, 60.0, 400.0, 30.0, 25, 30),
}


class EtrSim:
    """Same reset/step/info surface as ``EtrBridge``."""

    GRAVITY = 9.81
    SLOPE = math.radians(18.0)
    MAX_SPEED = 30.0
    PADDLE_ACCEL = 6.0
    BRAKE_ACCEL = 8.0
    TURN_RATE = math.radians(70.0)
    MIN_SPEED = 1.4

    def __init__(self, binary: Optional[str] = None, **_: object):
        self.rng = np.random.default_rng(0)
        self.course = "sim_trees"
        self.reset_done = False

    def info(self) -> dict:
        courses = []
        for d, (w, l, pw, pl, sx, _, _) in COURSES.items():
            courses.append(
                {
                    "group": "sim",
                    "dir": d,
                    "name": d,
                    "width": w,
                    "length": l,
                    "play_width": pw,
                    "play_length": pl,
                    "start_x": sx,
                    "start_z": -2.0,
                    "use_keyframe": True,
                }
            )
        return {
            "ok": True,
            "version": "sim",
            "character": "point-mass",
            "data_dir": "",
            "default_dt": DEFAULT_DT,
            "courses": courses,
        }

    # ---- helpers ----------------------------------------------------

    def _height(self, x: float, z: float) -> float:
        # Gentle rolling terrain on top of the slope, so probes carry signal.
        base = z * math.tan(self.SLOPE)  # z <= 0, so heights go down the course
        return base + 0.6 * math.sin(0.05 * z) * math.cos(0.08 * x)

    def _friction(self, x: float, z: float) -> float:
        if self.course == "sim_ice":
            band = math.sin(0.03 * z + 0.1 * x)
            return 0.08 if band > 0.3 else 0.35
        return 0.35

    def reset(
        self,
        seed: int = 0,
        course: str = "sim_trees",
        group: str = "sim",
        wind: int = 0,
        light: int = 0,
        mirror: bool = False,
        dt: float = DEFAULT_DT,
    ) -> dict:
        if course not in COURSES:
            raise ValueError(f"unknown sim course {course}")
        self.course = course
        w, l, pw, pl, sx, n_trees, n_items = COURSES[course]
        self.size = (w, l)
        self.play = (pw, pl)
        self.rng = np.random.default_rng(seed)
        self.dt = dt
        self.tick = 0
        self.time = 0.0
        self.x = sx
        self.z = -2.0
        self.speed = 3.0
        self.heading = 0.0  # 0 = straight down the course (-z)
        self.way = 0.0
        self.herring = 0
        self.collisions = 0
        self.colliding = False
        self.braking = False
        self.paddling = False
        self.turn = 0.0
        margin = 4.0
        self.trees = np.column_stack(
            [
                self.rng.uniform(margin, w - margin, n_trees),
                -self.rng.uniform(20.0, pl - 10.0, n_trees),
                self.rng.uniform(1.0, 2.5, n_trees),
            ]
        ).reshape(-1, 3)
        self.items = np.column_stack(
            [
                self.rng.uniform(margin, w - margin, n_items),
                -self.rng.uniform(10.0, pl - 5.0, n_items),
            ]
        ).reshape(-1, 2)
        self.item_alive = np.ones(len(self.items), dtype=bool)
        self.reset_done = True
        return self._obs(False)

    def _tick(self, turn: float, brake: bool, paddle: bool) -> None:
        dt = self.dt
        fr = self._friction(self.x, self.z)
        # steering authority grows with speed and grip
        self.heading += (
            turn * self.TURN_RATE * dt * min(1.0, self.speed / 8.0) * (fr / 0.35)
        )
        self.heading = max(-1.2, min(1.2, self.heading))
        accel = self.GRAVITY * math.sin(self.SLOPE) * math.cos(self.heading)
        accel -= fr * self.GRAVITY * math.cos(self.SLOPE)
        if paddle:
            accel += self.PADDLE_ACCEL * max(0.0, 1.0 - self.speed / self.MAX_SPEED)
        if brake:
            accel -= self.BRAKE_ACCEL * (fr / 0.35)
        self.speed = max(self.MIN_SPEED, min(self.MAX_SPEED, self.speed + accel * dt))
        dx = self.speed * math.sin(self.heading) * dt
        dz = -self.speed * math.cos(self.heading) * dt
        self.x += dx
        self.z += dz
        self.way += math.hypot(dx, dz)
        # walls
        if self.x < 0.5:
            self.x = 0.5
            self.heading = abs(self.heading) * 0.5
        if self.x > self.size[0] - 0.5:
            self.x = self.size[0] - 0.5
            self.heading = -abs(self.heading) * 0.5
        # trees
        hit = False
        if len(self.trees):
            d = np.hypot(self.trees[:, 0] - self.x, self.trees[:, 1] - self.z)
            idx = int(np.argmin(d))
            if d[idx] < self.trees[idx, 2] / 2.0 + 0.6:
                hit = True
                nx, nz = self.x - self.trees[idx, 0], self.z - self.trees[idx, 1]
                n = math.hypot(nx, nz) or 1.0
                self.x = self.trees[idx, 0] + nx / n * (self.trees[idx, 2] / 2.0 + 0.7)
                self.z = self.trees[idx, 1] + nz / n * (self.trees[idx, 2] / 2.0 + 0.7)
                self.speed = max(self.MIN_SPEED, self.speed * 0.5)
                self.heading = math.atan2(nx, -nz) * 0.5
        if hit and not self.colliding:
            self.collisions += 1
        self.colliding = hit
        # items
        if len(self.items):
            d = np.hypot(self.items[:, 0] - self.x, self.items[:, 1] - self.z)
            got = (d < 1.2) & self.item_alive
            if got.any():
                self.herring += int(got.sum())
                self.item_alive &= ~got
        self.braking = brake
        self.paddling = paddle
        self.turn = turn
        self.time += dt
        self.tick += 1

    def _finished(self) -> bool:
        return -self.z >= self.play[1]

    def step(
        self,
        turn: float = 0.0,
        brake: bool = False,
        paddle: bool = False,
        jump: bool = False,
        ticks: int = 1,
        dt: Optional[float] = None,
    ) -> dict:
        if not self.reset_done:
            raise RuntimeError("reset first")
        if dt is not None:
            self.dt = dt
        turn = max(-1.0, min(1.0, float(turn)))
        done = self._finished()
        for _ in range(int(ticks)):
            if done:
                break
            self._tick(turn, bool(brake), bool(paddle))
            done = self._finished()
        return self._obs(done)

    def _obs(self, done: bool) -> dict:
        y = self._height(self.x, self.z)
        vx = self.speed * math.sin(self.heading)
        vz = -self.speed * math.cos(self.heading)
        heights, fricts = [], []
        for a in PROBE_AHEAD:
            for s in PROBE_SIDE:
                px = min(max(self.x + s, 0.0), self.size[0])
                pz = min(max(self.z - a, -self.size[1]), 0.0)
                heights.append(self._height(px, pz) - y)
                fricts.append(self._friction(px, pz))
        trees = []
        if len(self.trees):
            rel = self.trees[:, :2] - np.array([self.x, self.z])
            keep = rel[:, 1] <= 4.0
            rel, diam = rel[keep], self.trees[keep, 2]
            order = np.argsort((rel**2).sum(1))[:NEAREST]
            trees = [
                [float(rel[i, 0]), float(rel[i, 1]), float(diam[i])] for i in order
            ]
        items = []
        if len(self.items):
            rel = self.items - np.array([self.x, self.z])
            keep = self.item_alive & (rel[:, 1] <= 4.0)
            rel = rel[keep]
            order = np.argsort((rel**2).sum(1))[:NEAREST]
            items = [[float(rel[i, 0]), float(rel[i, 1])] for i in order]
        return {
            "ok": True,
            "tick": self.tick,
            "time": self.time,
            "pos": [self.x, y, self.z],
            "vel": [vx, 0.0, vz],
            "speed": self.speed,
            "way": self.way,
            "progress": -self.z,
            "herring": self.herring,
            "items_left": int(self.item_alive.sum()),
            "collisions": self.collisions,
            "airborne": False,
            "braking": self.braking,
            "paddling": self.paddling,
            "jumping": False,
            "turn": self.turn,
            "friction": self._friction(self.x, self.z),
            "nml": [0.0, math.cos(self.SLOPE), math.sin(self.SLOPE)],
            "finish": done,
            "done": done,
            "size": list(self.size),
            "play": list(self.play),
            "probe_h": heights,
            "probe_f": fricts,
            "trees": trees,
            "items": items,
        }

    RECOVER_TIME = 1.0
    RECOVER_BACK = 10.0

    def recover(self) -> dict:
        """Mirror of the game's reset: 10 m back up the course, 1 s of clock."""
        if not self.reset_done:
            raise RuntimeError("reset first")
        self.z = min(self.z + self.RECOVER_BACK, -1.0)
        self.x = self.size[0] / 2.0
        self.speed = 3.0
        self.heading = 0.0
        self.colliding = False
        self.time += self.RECOVER_TIME
        return self._obs(self._finished())

    def close(self) -> None:
        pass

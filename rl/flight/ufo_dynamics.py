"""Dependency-light binding to xplane-ufo's C flight dynamics."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Sequence

import numpy as np

DYNAMICS_ABI_VERSION = 1


class UfoParameters(ctypes.Structure):
    _fields_ = [
        ("mass_kg", ctypes.c_double),
        ("inertia_kg_m2", ctypes.c_double * 3),
        ("linear_drag_n_per_mps", ctypes.c_double * 3),
        ("angular_drag_nm_per_radps", ctypes.c_double * 3),
        ("gravity_mps2", ctypes.c_double),
    ]


class UfoState(ctypes.Structure):
    _fields_ = [
        ("position_ned_m", ctypes.c_double * 3),
        ("velocity_ned_mps", ctypes.c_double * 3),
        ("quaternion_body_to_ned", ctypes.c_double * 4),
        ("angular_velocity_body_radps", ctypes.c_double * 3),
    ]


class UfoWrench(ctypes.Structure):
    _fields_ = [
        ("force_body_n", ctypes.c_double * 3),
        ("moment_body_nm", ctypes.c_double * 3),
    ]


class UfoDynamics:
    """Typed access to the common C integrator."""

    def __init__(self, library_path: str | Path):
        path = Path(library_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"xplane-ufo dynamics library does not exist: {path}")
        self.library_path = path
        self._library = ctypes.CDLL(str(path))
        self._library.xplane_ufo_dynamics_abi_version.argtypes = []
        self._library.xplane_ufo_dynamics_abi_version.restype = ctypes.c_uint
        actual_abi = self._library.xplane_ufo_dynamics_abi_version()
        if actual_abi != DYNAMICS_ABI_VERSION:
            raise ValueError(
                f"xplane-ufo dynamics ABI {actual_abi}, "
                f"expected {DYNAMICS_ABI_VERSION}"
            )
        self._library.xplane_ufo_default_params.argtypes = [
            ctypes.POINTER(UfoParameters)
        ]
        self._library.xplane_ufo_default_params.restype = None
        self._library.xplane_ufo_reset.argtypes = [ctypes.POINTER(UfoState)]
        self._library.xplane_ufo_reset.restype = None
        self._library.xplane_ufo_step.argtypes = [
            ctypes.POINTER(UfoParameters),
            ctypes.POINTER(UfoWrench),
            ctypes.c_double,
            ctypes.POINTER(UfoState),
        ]
        self._library.xplane_ufo_step.restype = ctypes.c_int
        self.params = UfoParameters()
        self.state = UfoState()
        self._library.xplane_ufo_default_params(ctypes.byref(self.params))
        self.reset()

    def reset(self) -> None:
        self._library.xplane_ufo_reset(ctypes.byref(self.state))

    def state_vector(self) -> np.ndarray:
        return np.asarray(
            [
                *self.state.position_ned_m,
                *self.state.velocity_ned_mps,
                *self.state.quaternion_body_to_ned,
                *self.state.angular_velocity_body_radps,
            ],
            dtype=np.float64,
        )

    def step(self, action: Sequence[float], dt_s: float) -> np.ndarray:
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError("wrench must contain six finite values")
        wrench = UfoWrench()
        for axis in range(3):
            wrench.force_body_n[axis] = values[axis]
            wrench.moment_body_nm[axis] = values[axis + 3]
        result = self._library.xplane_ufo_step(
            ctypes.byref(self.params),
            ctypes.byref(wrench),
            float(dt_s),
            ctypes.byref(self.state),
        )
        if result != 0:
            raise RuntimeError("xplane-ufo dynamics rejected the step")
        return self.state_vector()

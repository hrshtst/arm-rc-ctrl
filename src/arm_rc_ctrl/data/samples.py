# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Canonical processed dataset arrays (``docs/PLAN.md`` section 7.3).

A :class:`SampleSet` holds the arrays of one processed demonstration with a
common leading sample dimension ``N``:

==============  ==================  =======  ==========================================
Array           Shape               dtype    Meaning
==============  ==================  =======  ==========================================
``t``           ``(N,)``            float64  Time in seconds, starting at zero
``q``           ``(N, dof)``        float64  Joint position (rad)
``dq``          ``(N, dof)``        float64  Joint velocity (rad/s)
``ddq``         ``(N, dof)``        float64  Joint acceleration (rad/s^2)
``tip``         ``(N, task_dim)``   float64  Endpoint position (m)
``dtip``        ``(N, task_dim)``   float64  Endpoint velocity (m/s)
``ddtip``       ``(N, task_dim)``   float64  Endpoint acceleration (m/s^2)
``task_code``   ``(N, code_dim)``   float64  Task code; ``code_dim == 0`` for task 1-a
``phase``       ``(N,)``            int64    ``0`` prime, ``1`` move, ``2`` dwell
==============  ==================  =======  ==========================================

This module fixes names, shapes, dtypes, and the phase encoding. Semantic
validation (finiteness, monotonic time, phase completeness, joint limits,
task-code validity) is the dataset validator's job (M1-004).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from arm_rc_ctrl.data.arrays import array_digest, load_npz, save_npz

__all__ = [
    "ARRAY_NAMES",
    "PHASE_CODES",
    "PHASE_DWELL",
    "PHASE_MOVE",
    "PHASE_PRIME",
    "SAMPLES_FILE",
    "SAMPLES_SCHEMA_VERSION",
    "ArrayShape",
    "SampleSet",
    "load_samples",
    "save_samples",
]

SAMPLES_SCHEMA_VERSION: Final = 1
SAMPLES_FILE: Final = "samples.npz"

PHASE_PRIME: Final = 0
PHASE_MOVE: Final = 1
PHASE_DWELL: Final = 2
PHASE_CODES: Final[dict[str, int]] = {"prime": PHASE_PRIME, "move": PHASE_MOVE, "dwell": PHASE_DWELL}
"""Documented integer encoding of the ``phase`` array."""

ARRAY_NAMES: Final[tuple[str, ...]] = ("t", "q", "dq", "ddq", "tip", "dtip", "ddtip", "task_code", "phase")
_JOINT_ARRAYS: Final = ("q", "dq", "ddq")
_TASK_ARRAYS: Final = ("tip", "dtip", "ddtip")
_MIN_SAMPLES: Final = 2

type ArrayShape = tuple[int, ...]


def _frozen(array: NDArray[Any], dtype: type[np.generic]) -> NDArray[Any]:
    """Return a C-contiguous, read-only copy with exactly ``dtype`` (no value conversion)."""
    if array.dtype != np.dtype(dtype):
        msg = f"expected dtype {np.dtype(dtype)}, got {array.dtype}"
        raise TypeError(msg)
    copy = np.ascontiguousarray(array).copy()
    copy.setflags(write=False)
    return copy


@dataclass(frozen=True)
class SampleSet:
    """Immutable, shape-checked processed dataset arrays."""

    t: NDArray[np.float64]
    q: NDArray[np.float64]
    dq: NDArray[np.float64]
    ddq: NDArray[np.float64]
    tip: NDArray[np.float64]
    dtip: NDArray[np.float64]
    ddtip: NDArray[np.float64]
    task_code: NDArray[np.float64]
    phase: NDArray[np.int64]

    def __post_init__(self) -> None:
        """Enforce dtypes, dimensionality, a common sample count, and valid phase codes."""
        for name in ARRAY_NAMES[:-1]:
            object.__setattr__(self, name, _frozen(getattr(self, name), np.float64))
        object.__setattr__(self, "phase", _frozen(self.phase, np.int64))
        if self.t.ndim != 1 or self.phase.ndim != 1:
            msg = f"t and phase must be 1-D, got shapes {self.t.shape} and {self.phase.shape}"
            raise ValueError(msg)
        n = self.t.shape[0]
        if n < _MIN_SAMPLES:
            msg = f"a dataset needs at least {_MIN_SAMPLES} samples, got {n}"
            raise ValueError(msg)
        for name in (*_JOINT_ARRAYS, *_TASK_ARRAYS, "task_code"):
            array: NDArray[Any] = getattr(self, name)
            if array.ndim != 2 or array.shape[0] != n:  # noqa: PLR2004
                msg = f"{name} must have shape ({n}, k), got {array.shape}"
                raise ValueError(msg)
        if self.phase.shape[0] != n:
            msg = f"phase must have shape ({n},), got {self.phase.shape}"
            raise ValueError(msg)
        dof = self.q.shape[1]
        if dof < 1 or any(getattr(self, name).shape[1] != dof for name in _JOINT_ARRAYS):
            msg = f"q, dq, ddq must share a joint dimension >= 1, got {[getattr(self, k).shape for k in _JOINT_ARRAYS]}"
            raise ValueError(msg)
        task_dim = self.tip.shape[1]
        if task_dim < 1 or any(getattr(self, name).shape[1] != task_dim for name in _TASK_ARRAYS):
            shapes = [getattr(self, k).shape for k in _TASK_ARRAYS]
            msg = f"tip, dtip, ddtip must share a task dimension >= 1, got {shapes}"
            raise ValueError(msg)
        allowed = np.array(sorted(PHASE_CODES.values()), dtype=np.int64)
        if not np.isin(self.phase, allowed).all():
            bad = sorted(set(np.unique(self.phase).tolist()) - set(allowed.tolist()))
            msg = f"phase contains undocumented codes {bad}; allowed {PHASE_CODES}"
            raise ValueError(msg)

    @property
    def n_samples(self) -> int:
        """Number of samples ``N``."""
        return int(self.t.shape[0])

    @property
    def dof(self) -> int:
        """Number of joints."""
        return int(self.q.shape[1])

    @property
    def task_dim(self) -> int:
        """Endpoint dimension."""
        return int(self.tip.shape[1])

    @property
    def task_code_dim(self) -> int:
        """Task-code dimension (``0`` when there is no task conditioning)."""
        return int(self.task_code.shape[1])

    def arrays(self) -> dict[str, NDArray[Any]]:
        """Arrays keyed by canonical name, in canonical order."""
        return {name: getattr(self, name) for name in ARRAY_NAMES}

    def shapes(self) -> dict[str, ArrayShape]:
        """Shape of every array."""
        return {name: tuple(int(d) for d in array.shape) for name, array in self.arrays().items()}

    def dtypes(self) -> dict[str, str]:
        """Dtype name of every array."""
        return {name: str(array.dtype) for name, array in self.arrays().items()}

    def digests(self) -> dict[str, str]:
        """SHA-256 of every array (dtype, shape, and bytes)."""
        return {name: array_digest(array) for name, array in self.arrays().items()}

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, NDArray[Any]]) -> SampleSet:
        """Build from a mapping that must contain exactly the canonical arrays."""
        if set(arrays) != set(ARRAY_NAMES):
            msg = f"expected arrays {sorted(ARRAY_NAMES)}, got {sorted(arrays)}"
            raise ValueError(msg)
        return cls(**{name: arrays[name] for name in ARRAY_NAMES})


def save_samples(path: Path, samples: SampleSet) -> None:
    """Write ``samples.npz``."""
    save_npz(path, samples.arrays())


def load_samples(path: Path) -> SampleSet:
    """Read ``samples.npz`` and enforce the schema."""
    return SampleSet.from_arrays(load_npz(path, ARRAY_NAMES))

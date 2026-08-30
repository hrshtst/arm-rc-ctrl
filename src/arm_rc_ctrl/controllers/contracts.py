# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Behavior contracts between the robot, target generators, and low-level trackers (docs/PLAN.md section 8).

Units are SI throughout: time in seconds, joint positions in radians, joint
velocities in rad/s, joint accelerations in rad/s². Every array is a
one-dimensional, finite, read-only ``float64`` vector with one entry per joint.
A target generator must be reset with the robot's initial state before it
steps, may only step forward in time, and may never emit a non-finite target.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "DesiredJointState",
    "GeneratorError",
    "RobotState",
    "TargetGenerator",
    "TargetGeneratorBase",
    "as_joint_vector",
]


class GeneratorError(RuntimeError):
    """A target generator was used outside its contract or produced an invalid target."""


def as_joint_vector(values: ArrayLike, name: str, *, dof: int | None = None) -> NDArray[np.float64]:
    """Return ``values`` as a read-only, finite, one-dimensional ``float64`` vector.

    Parameters
    ----------
    values : ArrayLike
        The joint vector (any real dtype).
    name : str
        Field name used in error messages.
    dof : int | None, optional
        Required length; ``None`` accepts any length of at least one.
    """
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != 1 or array.shape[0] < 1:
        msg = f"{name} must be a one-dimensional vector with at least one joint, got shape {array.shape}"
        raise ValueError(msg)
    if dof is not None and array.shape[0] != dof:
        msg = f"{name} must have {dof} entries, got {array.shape[0]}"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must be finite, got {array.tolist()}"
        raise ValueError(msg)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class RobotState:
    """Measured joint state at one control sample."""

    t: float
    """Time of the measurement (s); finite and non-negative."""
    q: NDArray[np.float64]
    """Joint positions (rad)."""
    dq: NDArray[np.float64]
    """Joint velocities (rad/s)."""

    def __post_init__(self) -> None:
        """Validate time, shapes, dtypes, and finiteness; store read-only copies."""
        t = float(self.t)
        if not math.isfinite(t) or t < 0:
            msg = f"t must be finite and non-negative, got {self.t!r}"
            raise ValueError(msg)
        object.__setattr__(self, "t", t)
        q = as_joint_vector(self.q, "q")
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "dq", as_joint_vector(self.dq, "dq", dof=q.shape[0]))

    @property
    def dof(self) -> int:
        """Number of joints."""
        return int(self.q.shape[0])


@dataclass(frozen=True)
class DesiredJointState:
    """Desired joint position, velocity, and acceleration for the low-level tracker."""

    q: NDArray[np.float64]
    """Desired joint positions (rad)."""
    dq: NDArray[np.float64]
    """Desired joint velocities (rad/s)."""
    ddq: NDArray[np.float64]
    """Desired joint accelerations (rad/s²)."""

    def __post_init__(self) -> None:
        """Validate shapes, dtypes, and finiteness; store read-only copies."""
        q = as_joint_vector(self.q, "q")
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "dq", as_joint_vector(self.dq, "dq", dof=q.shape[0]))
        object.__setattr__(self, "ddq", as_joint_vector(self.ddq, "ddq", dof=q.shape[0]))

    @property
    def dof(self) -> int:
        """Number of joints."""
        return int(self.q.shape[0])

    @classmethod
    def hold(cls, q: ArrayLike) -> DesiredJointState:
        """A stationary target at posture ``q``."""
        posture = as_joint_vector(q, "q")
        zeros = np.zeros_like(posture)
        return cls(posture, zeros, zeros)


@runtime_checkable
class TargetGenerator(Protocol):
    """Produces the desired joint trajectory online from measured state."""

    def reset(self, initial_state: RobotState) -> None:
        """Start a new episode from the robot's initial state."""
        ...

    def step(self, state: RobotState, task_code: NDArray[np.float64] | None = None) -> DesiredJointState:
        """Return the desired joint state for the next control sample given the measured state."""
        ...


class TargetGeneratorBase(ABC):
    """Enforces the generator contract around a concrete implementation.

    Subclasses implement :meth:`_reset` and :meth:`_step`; the base class
    rejects stepping before a reset, a state whose joint count differs from the
    reset state, time that does not advance, a task code that is not a finite
    ``float64`` vector, and any non-finite or mismatched target.
    """

    def __init__(self) -> None:
        self._dof: int | None = None
        self._last_t: float | None = None
        self._steps = 0

    @property
    def is_reset(self) -> bool:
        """Whether an episode has been started."""
        return self._dof is not None

    @property
    def dof(self) -> int:
        """Number of joints of the current episode."""
        if self._dof is None:
            msg = "the generator has not been reset"
            raise GeneratorError(msg)
        return self._dof

    @property
    def steps(self) -> int:
        """Number of prime and step calls since the last reset."""
        return self._steps

    def reset(self, initial_state: RobotState) -> None:
        """Start a new episode from ``initial_state``."""
        self._dof = initial_state.dof
        self._last_t = initial_state.t
        self._steps = 0
        self._reset(initial_state)

    def _validate(self, method: str, state: RobotState, task_code: NDArray[np.float64] | None) -> NDArray[np.float64]:
        if self._dof is None or self._last_t is None:
            msg = f"{method}() called before reset()"
            raise GeneratorError(msg)
        if state.dof != self._dof:
            msg = f"state has {state.dof} joints but the episode was reset with {self._dof}"
            raise GeneratorError(msg)
        if state.t <= self._last_t and self._steps > 0:
            msg = f"time must advance: {state.t} s after {self._last_t} s"
            raise GeneratorError(msg)
        if state.t < self._last_t:
            msg = f"time must not go backwards: {state.t} s before the reset time {self._last_t} s"
            raise GeneratorError(msg)
        return self._task_code(task_code)

    def prime(self, state: RobotState, task_code: NDArray[np.float64] | None = None) -> None:
        """Feed measured state during the priming interval without producing a target.

        Validated exactly like :meth:`step` (an episode must have been reset and
        time must advance); the default implementation ignores the sample.
        """
        code = self._validate("prime", state, task_code)
        self._prime(state, code)
        self._last_t = state.t
        self._steps += 1

    def step(self, state: RobotState, task_code: NDArray[np.float64] | None = None) -> DesiredJointState:
        """Validate ``state`` and ``task_code``, delegate to the implementation, and validate its target."""
        code = self._validate("step", state, task_code)
        target = self._step(state, code)
        if not isinstance(target, DesiredJointState):  # pyright: ignore[reportUnnecessaryIsInstance]
            msg = f"the generator returned {type(target).__name__}, expected DesiredJointState"
            raise GeneratorError(msg)
        if target.dof != self._dof:
            msg = f"the generator returned {target.dof} joints, expected {self._dof}"
            raise GeneratorError(msg)
        self._last_t = state.t
        self._steps += 1
        return target

    @staticmethod
    def _task_code(task_code: NDArray[np.float64] | None) -> NDArray[np.float64]:
        if task_code is None:
            return np.zeros(0, dtype=np.float64)
        code = np.asarray(task_code)
        if code.ndim != 1 or code.dtype != np.float64 or not np.all(np.isfinite(code)):
            msg = (
                f"task_code must be a finite one-dimensional float64 vector, got shape {code.shape} dtype {code.dtype}"
            )
            raise GeneratorError(msg)
        return code

    @abstractmethod
    def _reset(self, initial_state: RobotState) -> None:
        """Implementation hook for :meth:`reset`."""

    def _prime(self, state: RobotState, task_code: NDArray[np.float64]) -> None:  # noqa: B027
        """Implementation hook for :meth:`prime` (default: the sample is ignored)."""

    @abstractmethod
    def _step(self, state: RobotState, task_code: NDArray[np.float64]) -> DesiredJointState:
        """Implementation hook for :meth:`step`; ``task_code`` is a (possibly empty) finite vector."""

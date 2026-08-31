# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Typed termination reasons and success/failure outcomes of simulation or hardware runs.

Every run ends with exactly one :class:`Termination`. ``completed`` means the
configured duration elapsed without a safety stop; everything else is a
structured failure that reports when it happened and why. An :class:`Outcome`
combines the termination with named success criteria so a report can show
*which* criterion failed instead of a bare boolean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final, Literal

__all__ = [
    "FAILURE_KINDS",
    "LIMIT_NAMES",
    "FailureKind",
    "LimitName",
    "Outcome",
    "Termination",
    "TerminationKind",
    "backend_failure",
    "completed",
    "divergence",
    "invalid_output",
    "invalid_state",
    "limit_violation",
    "timeout",
]

type TerminationKind = Literal[
    "completed",
    "invalid_state",
    "invalid_output",
    "limit_violation",
    "divergence",
    "timeout",
    "backend_failure",
]
type LimitName = Literal["joint_position", "joint_velocity", "torque", "endpoint"]
type FailureKind = Literal["non_finite", "shape", "bounds", "stale_time", "model_exception"]

LIMIT_NAMES: Final[tuple[str, ...]] = ("joint_position", "joint_velocity", "torque", "endpoint")
FAILURE_KINDS: Final[tuple[str, ...]] = ("non_finite", "shape", "bounds", "stale_time", "model_exception")
"""Why a command was invalid: the categories of docs/TASKS.md M2-013, each ending the run safely."""


def _finite(value: float | None, name: str) -> None:
    if value is not None and not math.isfinite(value):
        msg = f"{name} must be finite, got {value!r}"
        raise ValueError(msg)


@dataclass(frozen=True)
class Termination:
    """Why and when a run stopped."""

    kind: TerminationKind
    time_s: float
    """Simulation/wall time at which the run stopped."""
    step: int
    """Index of the last executed control step."""
    detail: str = ""
    limit: LimitName | None = None
    joint: int | None = None
    value: float | None = None
    bound: float | None = None
    failure: FailureKind | None = None
    """Category of an invalid command (``invalid_output`` only)."""

    def __post_init__(self) -> None:
        """Validate timing and the fields each kind requires."""
        if not math.isfinite(self.time_s) or self.time_s < 0:
            msg = f"time_s must be finite and non-negative, got {self.time_s!r}"
            raise ValueError(msg)
        if self.step < 0:
            msg = f"step must be non-negative, got {self.step}"
            raise ValueError(msg)
        if self.kind != "completed" and not self.detail.strip():
            msg = f"termination {self.kind!r} requires a non-empty detail"
            raise ValueError(msg)
        if self.kind == "limit_violation":
            if self.limit is None or self.value is None or self.bound is None:
                msg = "limit_violation requires limit, value, and bound"
                raise ValueError(msg)
        elif self.limit is not None or self.joint is not None or self.value is not None or self.bound is not None:
            msg = f"limit, joint, value, and bound are only valid for limit_violation, not {self.kind!r}"
            raise ValueError(msg)
        if self.joint is not None and self.joint < 0:
            msg = f"joint must be non-negative, got {self.joint}"
            raise ValueError(msg)
        _finite(self.value, "value")
        _finite(self.bound, "bound")
        if self.failure is not None and self.kind != "invalid_output":
            msg = f"failure is only valid for invalid_output, not {self.kind!r}"
            raise ValueError(msg)
        if self.failure is not None and self.failure not in FAILURE_KINDS:
            msg = f"failure must be one of {list(FAILURE_KINDS)}, got {self.failure!r}"
            raise ValueError(msg)

    @property
    def is_completed(self) -> bool:
        """Whether the run ran to its configured end."""
        return self.kind == "completed"


@dataclass(frozen=True)
class Outcome:
    """Success/failure of a run with the named criteria that decided it."""

    termination: Termination
    criteria: dict[str, bool] = field(default_factory=dict)
    """Named success criteria (e.g. ``"completed"``, ``"final_dwell_in_tolerance"``)."""

    def __post_init__(self) -> None:
        """A run that did not complete can never satisfy the ``completed`` criterion."""
        if "completed" not in self.criteria:
            msg = "criteria must include 'completed'"
            raise ValueError(msg)
        if self.criteria["completed"] != self.termination.is_completed:
            msg = "criteria['completed'] must equal termination.is_completed"
            raise ValueError(msg)

    @property
    def success(self) -> bool:
        """All criteria satisfied."""
        return all(self.criteria.values())

    @property
    def failed_criteria(self) -> tuple[str, ...]:
        """Names of the criteria that were not met, in order."""
        return tuple(name for name, ok in self.criteria.items() if not ok)


def completed(time_s: float, step: int) -> Termination:
    """The configured duration elapsed."""
    return Termination("completed", time_s, step)


def invalid_state(time_s: float, step: int, detail: str) -> Termination:
    """Measured state was not finite, had the wrong shape, or was stale."""
    return Termination("invalid_state", time_s, step, detail)


def invalid_output(time_s: float, step: int, detail: str, failure: FailureKind | None = None) -> Termination:
    """The target generator or controller produced an invalid command (``failure`` says why)."""
    return Termination("invalid_output", time_s, step, detail, failure=failure)


def limit_violation(
    time_s: float, step: int, limit: LimitName, value: float, bound: float, joint: int | None = None
) -> Termination:
    """A configured joint/velocity/torque/endpoint limit was exceeded."""
    where = f" on joint {joint}" if joint is not None else ""
    detail = f"{limit}{where}: {value!r} exceeds bound {bound!r}"
    return Termination("limit_violation", time_s, step, detail, limit=limit, joint=joint, value=value, bound=bound)


def divergence(time_s: float, step: int, detail: str) -> Termination:
    """The state or output grew without bound."""
    return Termination("divergence", time_s, step, detail)


def timeout(time_s: float, step: int, deadline_s: float) -> Termination:
    """A control deadline or wall-clock budget was missed."""
    return Termination("timeout", time_s, step, f"deadline {deadline_s!r} s missed at t={time_s!r} s")


def backend_failure(time_s: float, step: int, detail: str) -> Termination:
    """The simulator or hardware bridge reported an error."""
    return Termination("backend_failure", time_s, step, detail)

# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Semantic validation of canonical datasets (``docs/PLAN.md`` section 7.3).

:class:`~arm_rc_ctrl.data.samples.SampleSet` already fixes names, shapes,
dtypes, and phase codes. This module checks what the arrays *mean* against a
:class:`ValidationSpec` derived from the scenario: finite values, a uniform
time grid starting at zero, the expected dimensions, complete and ordered
phase intervals, valid one-hot task codes, and joint limits. Every problem is
collected and reported together; nothing is repaired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast

import numpy as np
from numpy.typing import NDArray

from arm_rc_ctrl.data.records import ProcessedDatasetRecord
from arm_rc_ctrl.data.samples import ARRAY_NAMES, PHASE_CODES, SampleSet
from arm_rc_ctrl.validation import require_finite

__all__ = ["DatasetValidationError", "JointLimits", "ValidationSpec", "dataset_problems", "validate_dataset"]

_FLOAT_ARRAYS: Final = ARRAY_NAMES[:-1]
_ONE_HOT_TOLERANCE: Final = 0.0


class DatasetValidationError(ValueError):
    """The dataset violates its specification; ``problems`` lists every finding."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = tuple(problems)
        super().__init__("dataset validation failed:\n" + "\n".join(problems))


@dataclass(frozen=True)
class JointLimits:
    """Per-joint position bounds (rad) and optional speed bounds (rad/s)."""

    lower: tuple[float, ...]
    upper: tuple[float, ...]
    speed: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        """Require matching lengths, finite values, lower < upper, and positive speeds."""
        if not self.lower or len(self.lower) != len(self.upper):
            msg = (
                f"limits.lower and limits.upper need the same non-zero length, got {len(self.lower)}/{len(self.upper)}"
            )
            raise ValueError(msg)
        require_finite(self.lower, "limits.lower")
        require_finite(self.upper, "limits.upper")
        if any(lo >= hi for lo, hi in zip(self.lower, self.upper, strict=True)):
            msg = "limits.lower must be below limits.upper for every joint"
            raise ValueError(msg)
        if self.speed is not None:
            if len(self.speed) != len(self.lower):
                msg = f"limits.speed must have {len(self.lower)} entries, got {len(self.speed)}"
                raise ValueError(msg)
            require_finite(self.speed, "limits.speed")
            if any(s <= 0 for s in self.speed):
                msg = "limits.speed entries must be positive"
                raise ValueError(msg)

    @property
    def dof(self) -> int:
        """Number of joints covered."""
        return len(self.lower)


@dataclass(frozen=True)
class ValidationSpec:
    """What a dataset must satisfy for one scenario."""

    dof: int
    task_dim: int
    task_code_dim: int
    period_s: float
    limits: JointLimits
    require_all_phases: bool = True
    period_tolerance_s: float = 1e-9
    time_start_tolerance_s: float = 1e-12

    def __post_init__(self) -> None:
        """Validate dimensions, period, tolerances, and limit width."""
        if self.dof < 1 or self.task_dim < 1 or self.task_code_dim < 0:
            dims = f"{self.dof}, {self.task_dim}, {self.task_code_dim}"
            msg = f"dof >= 1, task_dim >= 1, task_code_dim >= 0 required, got {dims}"
            raise ValueError(msg)
        if not (self.period_s > 0 and self.period_s < float("inf")):
            msg = f"period_s must be positive and finite, got {self.period_s!r}"
            raise ValueError(msg)
        if self.period_tolerance_s < 0 or self.time_start_tolerance_s < 0:
            msg = "tolerances must be non-negative"
            raise ValueError(msg)
        if self.limits.dof != self.dof:
            msg = f"limits cover {self.limits.dof} joints but dof is {self.dof}"
            raise ValueError(msg)

    @classmethod
    def from_record(
        cls, record: ProcessedDatasetRecord, limits: JointLimits, **overrides: float | bool
    ) -> ValidationSpec:
        """Derive the specification from a processed record and the scenario's joint limits."""
        return cls(
            dof=record.dof,
            task_dim=record.task_dim,
            task_code_dim=record.task_code_dim,
            period_s=record.preprocessing.resample_period_s,
            limits=limits,
            **overrides,  # type: ignore[arg-type]
        )


def dataset_problems(samples: SampleSet, spec: ValidationSpec) -> list[str]:
    """Return every violation of ``spec`` found in ``samples`` (empty when valid)."""
    problems: list[str] = []
    problems.extend(_finite_problems(samples))
    problems.extend(_shape_problems(samples, spec))
    problems.extend(_time_problems(samples, spec))
    problems.extend(_phase_problems(samples, spec))
    if samples.task_code_dim == spec.task_code_dim:
        problems.extend(_task_code_problems(samples))
    if samples.dof == spec.dof:
        problems.extend(_limit_problems(samples, spec.limits))
    return problems


def validate_dataset(samples: SampleSet, spec: ValidationSpec) -> None:
    """Raise :class:`DatasetValidationError` listing every problem; return silently when valid."""
    problems = dataset_problems(samples, spec)
    if problems:
        raise DatasetValidationError(problems)


def _finite_problems(samples: SampleSet) -> list[str]:
    problems: list[str] = []
    for name in _FLOAT_ARRAYS:
        array = samples.arrays()[name]
        bad = np.argwhere(~np.isfinite(array))
        if bad.size:
            first = tuple(int(i) for i in bad[0])
            problems.append(f"{name} contains {bad.shape[0]} non-finite value(s), first at index {first}")
    return problems


def _shape_problems(samples: SampleSet, spec: ValidationSpec) -> list[str]:
    problems: list[str] = []
    for label, actual, expected in (
        ("dof", samples.dof, spec.dof),
        ("task_dim", samples.task_dim, spec.task_dim),
        ("task_code_dim", samples.task_code_dim, spec.task_code_dim),
    ):
        if actual != expected:
            problems.append(f"{label} is {actual}, expected {expected}")
    return problems


def _time_problems(samples: SampleSet, spec: ValidationSpec) -> list[str]:
    problems: list[str] = []
    t = samples.t
    if not np.all(np.isfinite(t)):
        return problems  # already reported as non-finite
    if abs(float(t[0])) > spec.time_start_tolerance_s:
        problems.append(f"t must start at 0, got {float(t[0])!r}")
    steps = np.diff(t)
    if not bool(np.all(steps > 0)):
        first = int(np.argmax(steps <= 0))
        problems.append(f"t is not strictly increasing (first violation between samples {first} and {first + 1})")
        return problems
    deviation = np.abs(steps - spec.period_s)
    worst = int(np.argmax(deviation))
    if float(deviation[worst]) > spec.period_tolerance_s:
        problems.append(
            f"t is not uniformly sampled at {spec.period_s} s: interval {worst} is {float(steps[worst])!r} s"
        )
    return problems


def _phase_problems(samples: SampleSet, spec: ValidationSpec) -> list[str]:
    problems: list[str] = []
    phase = samples.phase
    present = set(np.unique(phase).tolist())
    if spec.require_all_phases:
        missing = [name for name, code in PHASE_CODES.items() if code not in present]
        if missing:
            problems.append(f"missing phase interval(s): {missing}")
    if not bool(np.all(np.diff(phase) >= 0)):
        first = int(np.argmax(np.diff(phase) < 0))
        problems.append(
            f"phases must run prime -> move -> dwell without returning "
            f"(violation between samples {first} and {first + 1})"
        )
    return problems


def _task_code_problems(samples: SampleSet) -> list[str]:
    if samples.task_code_dim == 0:
        return []
    codes = samples.task_code
    if not np.all(np.isfinite(codes)):
        return []  # already reported as non-finite
    binary: NDArray[np.bool_] = np.all((codes == 0.0) | (codes == 1.0), axis=1)
    ones = cast("NDArray[np.intp]", np.count_nonzero(codes == 1.0, axis=1))
    one_hot: NDArray[np.bool_] = ones == 1
    bad = np.argwhere(~(binary & one_hot)).ravel()
    if bad.size:
        return [f"task_code has {bad.size} row(s) that are not one-hot, first at sample {int(bad[0])}"]
    return []


def _limit_problems(samples: SampleSet, limits: JointLimits) -> list[str]:
    problems: list[str] = []
    q = samples.q
    if not np.all(np.isfinite(q)):
        return problems
    lower = np.asarray(limits.lower)
    upper = np.asarray(limits.upper)
    outside = np.argwhere((q < lower) | (q > upper))
    if outside.size:
        sample, joint = (int(i) for i in outside[0])
        problems.append(
            f"q violates joint limits at {outside.shape[0]} sample/joint pair(s), first at sample {sample} "
            f"joint {joint}: {float(q[sample, joint])!r} not in [{limits.lower[joint]}, {limits.upper[joint]}]"
        )
    if limits.speed is not None and np.all(np.isfinite(samples.dq)):
        too_fast = np.argwhere(np.abs(samples.dq) > np.asarray(limits.speed))
        if too_fast.size:
            sample, joint = (int(i) for i in too_fast[0])
            problems.append(
                f"dq exceeds the speed limit at {too_fast.shape[0]} sample/joint pair(s), first at sample {sample} "
                f"joint {joint}: |{float(samples.dq[sample, joint])!r}| > {limits.speed[joint]}"
            )
    return problems

# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Joint-speed diagnostics of one simulated run (M3REP-004; repetition plan section 7.1, D5).

The pilot aborts a run at the evaluation-side speed bound (12 rad/s per
joint) but keeps the historical 6 rad/s limit as a non-terminating
diagnostic: for every joint, the peak absolute measured speed, the first
crossing of each threshold on the run and task clocks, and the time above the
historical limit split into warm-up, movement, and dwell. The figures are
taken from the simulator's checked states at the control cadence, including
the terminal offending state of an abort, which is retained here separately
because no controller telemetry exists for it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arm_rc_ctrl.experiments.simulation import CheckedState
    from arm_rc_ctrl.experiments.termination import Termination

__all__ = [
    "PHASES",
    "AbortDetail",
    "JointVelocityDiagnostic",
    "TerminalCheckedState",
    "ThresholdCrossing",
    "VelocityDiagnostics",
    "phase_of",
    "velocity_diagnostics",
]

PHASES: tuple[str, ...] = ("warmup", "movement", "dwell")
"""Run phases by run time: before activation, from activation to the dwell start, and the dwell."""


def phase_of(t: float, *, activation_s: float, dwell_start_s: float) -> str:
    """The phase of run time ``t`` (``dwell_start_s`` is on the task clock)."""
    if t < activation_s:
        return "warmup"
    if t - activation_s < dwell_start_s:
        return "movement"
    return "dwell"


@dataclass(frozen=True)
class ThresholdCrossing:
    """The first checked state at which a joint's speed exceeded a threshold."""

    step: int
    run_time_s: float
    task_time_s: float
    value: float
    """The measured speed at the crossing (rad/s, signed)."""
    phase: str


@dataclass(frozen=True)
class JointVelocityDiagnostic:
    """One joint's speed record over the run."""

    joint: int
    peak_abs_speed: float
    peak_step: int
    peak_run_time_s: float
    peak_task_time_s: float
    first_historical_crossing: ThresholdCrossing | None
    first_abort_crossing: ThresholdCrossing | None
    time_above_historical_s: dict[str, float]
    """Seconds (checked samples times the control period) with speed above the historical limit, per phase."""

    def __post_init__(self) -> None:
        """Every phase is reported and the figures are finite."""
        if set(self.time_above_historical_s) != set(PHASES):
            msg = f"time_above_historical_s must report exactly the phases {PHASES}"
            raise ValueError(msg)
        if not math.isfinite(self.peak_abs_speed) or self.peak_abs_speed < 0:
            msg = f"peak_abs_speed must be finite and non-negative, got {self.peak_abs_speed!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class TerminalCheckedState:
    """The measured state at which the simulator stopped a run (retained apart from the telemetry)."""

    step: int
    run_time_s: float
    task_time_s: float
    q: tuple[float, ...]
    dq: tuple[float, ...]


@dataclass(frozen=True)
class AbortDetail:
    """A velocity abort's limit, joint, measured value, bound, and both timestamps."""

    limit: str
    joint: int | None
    value: float
    bound: float
    step: int
    run_time_s: float
    task_time_s: float


@dataclass(frozen=True)
class VelocityDiagnostics:
    """The speed diagnostics of one run against the historical and the abort limits."""

    historical_limit: tuple[float, ...]
    abort_limit: tuple[float, ...]
    activation_s: float
    dwell_start_s: float
    control_period_s: float
    samples: int
    """Checked states the figures were taken from (the terminal offending state included)."""
    joints: tuple[JointVelocityDiagnostic, ...]
    crossed_historical: bool
    """Whether any joint exceeded the historical limit at any checked state."""
    abort: AbortDetail | None
    terminal_state: TerminalCheckedState | None

    def __post_init__(self) -> None:
        """The summary flag re-derives from the joints and an abort carries its terminal state."""
        if self.crossed_historical != any(j.first_historical_crossing is not None for j in self.joints):
            msg = "crossed_historical contradicts the joint records"
            raise ValueError(msg)
        if (self.abort is None) != (self.terminal_state is None):
            msg = "an abort and its terminal checked state are recorded together"
            raise ValueError(msg)
        if len(self.historical_limit) != len(self.joints) or len(self.abort_limit) != len(self.joints):
            msg = "one limit per joint is required"
            raise ValueError(msg)


def _crossing(
    states: Sequence[CheckedState], joint: int, threshold: float, *, activation_s: float, dwell_start_s: float
) -> ThresholdCrossing | None:
    for state in states:
        value = float(state.dq[joint])
        if abs(value) > threshold:
            return ThresholdCrossing(
                step=state.step,
                run_time_s=state.t,
                task_time_s=state.t - activation_s,
                value=value,
                phase=phase_of(state.t, activation_s=activation_s, dwell_start_s=dwell_start_s),
            )
    return None


def velocity_diagnostics(
    states: Sequence[CheckedState],
    termination: Termination,
    *,
    historical: Sequence[float],
    abort_limit: Sequence[float],
    activation_s: float,
    dwell_start_s: float,
    dt: float,
) -> VelocityDiagnostics:
    """Evaluate the checked states of one run against both limits.

    ``historical`` is the scenario's own per-joint limit (6 rad/s for task 1-a),
    ``abort_limit`` the bound the simulator applied. A ``limit_violation`` of
    the joint speed is recorded with its detail and the last checked state is
    retained as the terminal state; any other non-completed termination keeps
    its terminal state too (the abort detail is then ``None``).
    """
    if not states:
        msg = "velocity diagnostics need at least one checked state"
        raise ValueError(msg)
    dof = int(states[0].dq.shape[0])
    if len(historical) != dof or len(abort_limit) != dof:
        msg = f"historical and abort limits must give {dof} bounds, got {historical!r} and {abort_limit!r}"
        raise ValueError(msg)
    if not (dt > 0 and math.isfinite(dt)):
        msg = f"dt must be positive and finite, got {dt!r}"
        raise ValueError(msg)
    speeds = np.vstack([np.abs(np.asarray(s.dq, dtype=np.float64)) for s in states])
    times = np.array([s.t for s in states], dtype=np.float64)
    phases: list[str] = [phase_of(float(t), activation_s=activation_s, dwell_start_s=dwell_start_s) for t in times]
    joints: list[JointVelocityDiagnostic] = []
    for j in range(dof):
        peak = int(np.argmax(speeds[:, j]))
        above = speeds[:, j] > float(historical[j])
        time_above: dict[str, float] = dict.fromkeys(PHASES, 0.0)
        for flag, phase in zip(above, phases, strict=True):
            if bool(flag):
                time_above[phase] += dt
        joints.append(
            JointVelocityDiagnostic(
                joint=j,
                peak_abs_speed=float(speeds[peak, j]),
                peak_step=states[peak].step,
                peak_run_time_s=float(times[peak]),
                peak_task_time_s=float(times[peak]) - activation_s,
                first_historical_crossing=_crossing(
                    states, j, float(historical[j]), activation_s=activation_s, dwell_start_s=dwell_start_s
                ),
                first_abort_crossing=_crossing(
                    states, j, float(abort_limit[j]), activation_s=activation_s, dwell_start_s=dwell_start_s
                ),
                time_above_historical_s=time_above,
            )
        )
    abort: AbortDetail | None = None
    terminal: TerminalCheckedState | None = None
    if not termination.is_completed:
        last = states[-1]
        terminal = TerminalCheckedState(
            step=last.step,
            run_time_s=last.t,
            task_time_s=last.t - activation_s,
            q=tuple(float(v) for v in last.q),
            dq=tuple(float(v) for v in last.dq),
        )
        if termination.kind == "limit_violation" and termination.limit is not None:
            abort = AbortDetail(
                limit=termination.limit,
                joint=termination.joint,
                value=float(termination.value if termination.value is not None else math.nan),
                bound=float(termination.bound if termination.bound is not None else math.nan),
                step=termination.step,
                run_time_s=termination.time_s,
                task_time_s=termination.time_s - activation_s,
            )
    return VelocityDiagnostics(
        historical_limit=tuple(float(v) for v in historical),
        abort_limit=tuple(float(v) for v in abort_limit),
        activation_s=activation_s,
        dwell_start_s=dwell_start_s,
        control_period_s=dt,
        samples=len(states),
        joints=tuple(joints),
        crossed_historical=any(j.first_historical_crossing is not None for j in joints),
        abort=abort,
        terminal_state=terminal,
    )

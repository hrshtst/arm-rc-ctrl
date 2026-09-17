# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-008: the manual protocol's run verdicts (plan section 6, clarification I5).

Three rules separate this protocol from the inherited recovery one. Completion
is judged against the **configured evaluation horizon** rather than the
demonstration's length, so reaching the target and then drifting away for the
rest of the horizon cannot count as success, and an aborted run still keeps
its partial metrics. The dwell is the acquisition rule -- one uninterrupted
second inside the target radius with every joint slow, any excursion
restarting the timer -- rather than the old occupancy fraction over a
phase-derived window. And the force pulse is triggered by the measured motion,
so its verdict is about whether it fired at all and early enough to leave room
for the final dwell it is meant to disturb.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from arm_rc_ctrl.data.manual import continuous_dwell, dwell_runs

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from arm_rc_ctrl.data.manual import DwellPredicate
    from arm_rc_ctrl.experiments.disturbances import ForcePulse
    from arm_rc_ctrl.experiments.termination import Termination

__all__ = [
    "ManualDwellReport",
    "TriggerOutcome",
    "horizon_completed",
    "manual_dwell_report",
    "trigger_outcome",
]

_GRID_TOLERANCE_S: Final = 1e-9
"""Slack for comparing times that are exact multiples of the control period."""


def horizon_completed(
    t: NDArray[np.float64], termination: Termination, *, activation_s: float, horizon_s: float
) -> bool:
    """Whether the run terminated normally **and** ran the whole configured horizon from activation (I5).

    The inherited check compares the active sample count with the reference
    length, which this protocol cannot use: the horizon is a finite timeout for
    non-convergence, deliberately far longer than any demonstration.
    """
    times = np.asarray(t, dtype=np.float64)
    if times.ndim != 1 or times.shape[0] == 0:
        msg = f"t must be a non-empty 1-D array, got shape {times.shape}"
        raise ValueError(msg)
    if not termination.is_completed:
        return False
    return float(times[-1]) >= activation_s + horizon_s - _GRID_TOLERANCE_S


@dataclass(frozen=True)
class ManualDwellReport:
    """What the continuous dwell rule measured on one run (plan section 6's four reported quantities)."""

    ok: bool
    """The run ending at the last sample satisfies the rule for its full required duration."""
    final_samples: int
    final_duration_s: float
    longest_samples: int
    longest_duration_s: float
    earliest_start_s: float | None
    """Start of the first run that reached the required length, if any."""
    departures_after_hold: int
    """Qualifying holds that ended before the last sample: the arm reached the target and then left it."""

    def __post_init__(self) -> None:
        """Counts are non-negative and a successful report ends in a qualifying run."""
        if self.final_samples < 0 or self.longest_samples < self.final_samples or self.departures_after_hold < 0:
            msg = f"inconsistent dwell counts: {self}"
            raise ValueError(msg)
        if self.ok and self.final_samples == 0:
            msg = "a successful dwell report must end in a qualifying run"
            raise ValueError(msg)


def manual_dwell_report(
    t: NDArray[np.float64],
    tip: NDArray[np.float64],
    dq: NDArray[np.float64],
    *,
    target: NDArray[np.float64],
    predicate: DwellPredicate,
    since_s: float | None = None,
) -> ManualDwellReport:
    """Measure the dwell of a run under the acquisition predicate.

    ``since_s`` restarts the measurement at that time, which is how a force
    case is judged: the pulse resets the success timer, so only the dwell after
    the pulse ends can satisfy the rule.
    """
    times = np.asarray(t, dtype=np.float64)
    tips = np.asarray(tip, dtype=np.float64)
    speeds = np.asarray(dq, dtype=np.float64)
    if since_s is not None:
        first = int(np.searchsorted(times, since_s - _GRID_TOLERANCE_S, side="left"))
        times, tips, speeds = times[first:], tips[first:], speeds[first:]
    measurement = continuous_dwell(times, tips, speeds, target=target, predicate=predicate)
    runs = dwell_runs(times, tips, speeds, target=target, predicate=predicate)
    n = times.shape[0]
    departures = sum(1 for start, end in runs if end < n and end - start >= predicate.min_samples)
    longest = max((float(times[end - 1] - times[start]) for start, end in runs), default=0.0)
    return ManualDwellReport(
        ok=measurement.ok,
        final_samples=measurement.final_samples,
        final_duration_s=measurement.final_duration_s,
        longest_samples=measurement.longest_samples,
        longest_duration_s=longest,
        earliest_start_s=measurement.earliest_start_s,
        departures_after_hold=departures,
    )


@dataclass(frozen=True)
class TriggerOutcome:
    """Whether the dwell-triggered pulse fired, and early enough to leave room for the dwell it disturbs."""

    ok: bool
    triggered: bool
    pulse_start_s: float | None
    pulse_end_s: float | None
    reason: str | None
    """Why the case cannot count as a disturbance-recovery test, or ``None`` when the timing rule was met."""

    def __post_init__(self) -> None:
        """A met rule carries its timing and no reason; a failed one always names itself."""
        if self.ok and (self.reason is not None or not self.triggered):
            msg = f"a satisfied trigger rule carries no reason and did fire: {self}"
            raise ValueError(msg)
        if not self.ok and not self.reason:
            msg = "a failed trigger rule must give its reason"
            raise ValueError(msg)


def trigger_outcome(
    pulse: ForcePulse | None, *, activation_s: float, horizon_s: float, dwell_min_duration_s: float
) -> TriggerOutcome:
    """The verdict on a state-triggered pulse: it must have fired, and ended a full dwell before the horizon.

    A missing or too-late trigger is reported explicitly and never counts as a
    successful disturbance-recovery test, because the run never carried the
    disturbance the case exists to apply.
    """
    horizon_end = activation_s + horizon_s
    if pulse is None:
        return TriggerOutcome(
            ok=False,
            triggered=False,
            pulse_start_s=None,
            pulse_end_s=None,
            reason="the target dwell never qualified for the trigger hold, so the pulse never fired",
        )
    deadline = horizon_end - dwell_min_duration_s
    if pulse.end_s > deadline + _GRID_TOLERANCE_S:
        return TriggerOutcome(
            ok=False,
            triggered=True,
            pulse_start_s=pulse.start_s,
            pulse_end_s=pulse.end_s,
            reason=(
                f"the pulse ended at {pulse.end_s:.3f} s, too late for the {dwell_min_duration_s:.3f} s final "
                f"dwell to complete before the horizon at {horizon_end:.3f} s"
            ),
        )
    return TriggerOutcome(ok=True, triggered=True, pulse_start_s=pulse.start_s, pulse_end_s=pulse.end_s, reason=None)

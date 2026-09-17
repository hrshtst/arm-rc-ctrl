# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-008: horizon completion, the continuous dwell measured on a run, and the pulse-trigger verdicts.

The manual protocol judges completion against the configured evaluation
horizon rather than the demonstration's length (clarification I5), reuses the
acquisition dwell predicate instead of the inherited occupancy rule, and
triggers its force pulse on the measured motion rather than at a fixed time
(plan section 6).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from arm_rc_ctrl.data.manual import DwellPredicate
from arm_rc_ctrl.experiments.disturbances import ForcePulse
from arm_rc_ctrl.experiments.manual_evaluation import horizon_completed, manual_dwell_report, trigger_outcome
from arm_rc_ctrl.experiments.termination import completed, limit_violation

PREDICATE = DwellPredicate(tolerance_m=0.01, max_velocity_rad_s=0.05, min_duration_s=1.0, min_samples=101)
"""The frozen acquisition rule: one second inside 1 cm at joint speeds of at most 0.05 rad/s."""

TARGET: NDArray[np.float64] = np.array([0.10, 0.45])
DT = 0.01
ACTIVATION_S = 0.25
HORIZON_S = 30.0
DWELL_MIN_S = 1.0

Arrays = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]


def _arrays(n: int, qualifying: Sequence[tuple[int, int]]) -> Arrays:
    """A run of ``n`` samples that satisfies the dwell predicate exactly on the given half-open ranges."""
    t: NDArray[np.float64] = np.arange(n, dtype=np.float64) * DT
    tip: NDArray[np.float64] = np.tile(TARGET + np.array([0.05, 0.0]), (n, 1))
    dq: NDArray[np.float64] = np.full((n, 2), 0.5)
    for start, end in qualifying:
        tip[start:end] = TARGET
        dq[start:end] = 0.0
    return t, tip, dq


def _horizon_grid() -> NDArray[np.float64]:
    return np.arange(round((ACTIVATION_S + HORIZON_S) / DT) + 1, dtype=np.float64) * DT


# --- completion against the configured horizon (I5) ------------------------------------------


def test_a_run_reaching_the_configured_horizon_is_complete() -> None:
    """The whole horizon ran and the simulator terminated normally."""
    t = _horizon_grid()
    assert horizon_completed(t, completed(float(t[-1]), t.shape[0] - 1), activation_s=ACTIVATION_S, horizon_s=HORIZON_S)


def test_a_run_ending_at_the_demonstration_length_is_not_complete() -> None:
    """I5: a run as long as the demonstration is far short of the horizon, however normally it ended.

    The inherited check compares the active sample count with the reference
    length, so it would call exactly this run complete; reaching the target and
    then drifting away for the rest of the horizon must not count as success.
    """
    t: NDArray[np.float64] = np.arange(round((ACTIVATION_S + 12.0) / DT) + 1, dtype=np.float64) * DT
    assert not horizon_completed(
        t, completed(float(t[-1]), t.shape[0] - 1), activation_s=ACTIVATION_S, horizon_s=HORIZON_S
    )


def test_an_aborted_run_is_incomplete_yet_keeps_its_partial_dwell_metrics() -> None:
    """I5: aborted runs retain their partial metrics and terminal evidence."""
    t, tip, dq = _arrays(500, [(200, 500)])
    termination = limit_violation(float(t[-1]), 499, "joint_velocity", 7.0, 6.0, joint=1)
    assert not horizon_completed(t, termination, activation_s=ACTIVATION_S, horizon_s=HORIZON_S)
    report = manual_dwell_report(t, tip, dq, target=TARGET, predicate=PREDICATE)
    assert report.final_samples == 300
    assert report.ok


# --- the continuous dwell measured on a run --------------------------------------------------


def test_exactly_one_second_of_final_dwell_qualifies() -> None:
    """One second at 100 Hz is 101 consecutive samples ending at the last sample."""
    n = 400
    t, tip, dq = _arrays(n, [(n - 101, n)])
    assert manual_dwell_report(t, tip, dq, target=TARGET, predicate=PREDICATE).ok


def test_one_sample_short_of_a_second_does_not_qualify() -> None:
    """The boundary is closed on the required duration, so 100 samples is a failure, not a rounding question."""
    n = 400
    t, tip, dq = _arrays(n, [(n - 100, n)])
    assert not manual_dwell_report(t, tip, dq, target=TARGET, predicate=PREDICATE).ok


def test_a_hold_that_ends_before_the_last_sample_is_a_departure_not_a_success() -> None:
    """Success requires the dwell to extend to the end of the run; an earlier hold is reported as a departure."""
    t, tip, dq = _arrays(600, [(100, 400)])
    report = manual_dwell_report(t, tip, dq, target=TARGET, predicate=PREDICATE)
    assert not report.ok
    assert report.final_samples == 0
    assert report.longest_samples == 300
    assert report.departures_after_hold == 1
    assert report.earliest_start_s == pytest.approx(1.0)


def test_a_touch_too_brief_to_qualify_is_not_counted_as_a_departure() -> None:
    """Only a run that actually reached the required length counts as a hold that was then departed from."""
    t, tip, dq = _arrays(600, [(100, 150), (300, 600)])
    report = manual_dwell_report(t, tip, dq, target=TARGET, predicate=PREDICATE)
    assert report.ok
    assert report.departures_after_hold == 0


def test_the_pulse_resets_the_success_dwell_timer() -> None:
    """Force cases need a complete final dwell measured from the pulse end, not from an earlier arrival."""
    t, tip, dq = _arrays(600, [(100, 600)])
    whole = manual_dwell_report(t, tip, dq, target=TARGET, predicate=PREDICATE)
    after = manual_dwell_report(t, tip, dq, target=TARGET, predicate=PREDICATE, since_s=5.50)
    assert whole.ok
    assert whole.final_samples == 500
    assert not after.ok
    assert after.final_samples == 50


# --- the dwell-triggered pulse ---------------------------------------------------------------


def test_a_pulse_that_never_triggered_is_an_explicit_failure() -> None:
    """A missing trigger cannot count as a successful disturbance-recovery test; it is reported by reason."""
    outcome = trigger_outcome(None, activation_s=ACTIVATION_S, horizon_s=HORIZON_S, dwell_min_duration_s=DWELL_MIN_S)
    assert not outcome.ok
    assert outcome.reason is not None
    assert "never" in outcome.reason


def test_a_pulse_too_late_for_a_complete_final_dwell_is_an_explicit_failure() -> None:
    """A pulse ending 0.3 s before the horizon leaves no room for the required second of dwell."""
    pulse = ForcePulse(start_s=ACTIVATION_S + 29.5, duration_s=0.2, force=(12.0, 0.0))
    outcome = trigger_outcome(pulse, activation_s=ACTIVATION_S, horizon_s=HORIZON_S, dwell_min_duration_s=DWELL_MIN_S)
    assert not outcome.ok
    assert outcome.reason is not None
    assert "late" in outcome.reason


def test_a_pulse_leaving_room_for_the_final_dwell_passes() -> None:
    """The verdict is about the timing rule only; whether the dwell was achieved is the dwell report's business."""
    pulse = ForcePulse(start_s=ACTIVATION_S + 5.0, duration_s=0.2, force=(12.0, 0.0))
    outcome = trigger_outcome(pulse, activation_s=ACTIVATION_S, horizon_s=HORIZON_S, dwell_min_duration_s=DWELL_MIN_S)
    assert outcome.ok
    assert outcome.reason is None
    assert outcome.pulse_end_s == pytest.approx(ACTIVATION_S + 5.2)

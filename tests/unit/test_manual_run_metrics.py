# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-010: the diagnostic metrics of one run, read from its stored trajectories (plan section 6).

Every metric is computed over the active segment, from activation to the last
sample, so the negative-time warm-up never contributes a peak or an error. A run
that aborted before activating has no active segment, and its metrics are
absent rather than zero.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from arm_rc_ctrl.experiments.manual_results import run_metrics

DT, ACTIVATION, END = 0.01, 1.0, 3.0
TARGET = (0.1, 0.003)


def _run() -> dict[str, np.ndarray]:
    """A two-joint run: still until 1.503 s, a straight reach to the target by 2.003 s, then held."""
    t = np.round(np.arange(0.0, END + DT / 2, DT), 10)
    n = t.shape[0]
    progress = np.clip((t - 1.503) / 0.5, 0.0, 1.0)
    tip = np.stack([0.1 * progress, np.zeros(n)], axis=1)
    active = t >= ACTIVATION
    dq = np.zeros((n, 2))
    dq[~active] = 5.0  # a warm-up peak that must not count
    dq[t >= 2.0, 0] = 2.0  # the active peak, reached by a single step of 2 rad/s in one sample
    q = np.zeros((n, 2))
    q_desired = np.zeros((n, 2))
    first_active = int(np.argmax(active))
    q_desired[first_active] = (0.03, 0.04)  # the command jumps 0.05 rad at activation
    dq_desired = np.zeros((n, 2))
    dq_desired[active, 1] = -1.5
    dq_desired[~active, 1] = 9.0
    ddq_desired = np.zeros((n, 2))
    ddq_desired[active, 0] = 12.0
    tracking_error = np.zeros((n, 2))
    tracking_error[active] = (0.01, -0.02)
    tracking_error[~active] = (1.0, 1.0)
    tau = np.zeros((n, 2))
    tau[active, 1] = -3.0
    tau[~active, 1] = 9.0
    return {
        "t": t,
        "tip": tip,
        "q": q,
        "dq": dq,
        "q_desired": q_desired,
        "dq_desired": dq_desired,
        "ddq_desired": ddq_desired,
        "tracking_error": tracking_error,
        "tau_applied": tau,
        "tau_requested": tau,
    }


def test_the_metrics_read_the_active_segment_only() -> None:
    """Peaks, errors and effort come from activation onwards; the warm-up's larger values never count."""
    metrics = run_metrics(_run(), activation_s=ACTIVATION, target=TARGET, final_dwell_samples=None)
    assert metrics.n_active_samples == 201
    assert metrics.peak_speed_rad_s == pytest.approx(2.0)
    assert metrics.peak_reference_speed_rad_s == pytest.approx(1.5)
    assert metrics.peak_reference_acceleration_rad_s2 == pytest.approx(12.0)
    assert metrics.torque_peak_nm == pytest.approx(3.0)
    assert metrics.tracking_error_max_rad == pytest.approx(0.02)
    assert metrics.tracking_error_rms_rad == pytest.approx(math.sqrt((0.01**2 + 0.02**2) / 2))


def test_measured_acceleration_is_the_finite_difference_of_measured_velocity_within_the_active_segment() -> None:
    """The step from the warm-up's 5 rad/s into the active segment is not an active acceleration."""
    metrics = run_metrics(_run(), activation_s=ACTIVATION, target=TARGET, final_dwell_samples=None)
    assert metrics.peak_acceleration_rad_s2 == pytest.approx(2.0 / DT)


def test_the_endpoint_error_is_measured_at_the_last_sample() -> None:
    """Final endpoint error: the distance from the target at the end of the run."""
    metrics = run_metrics(_run(), activation_s=ACTIVATION, target=TARGET, final_dwell_samples=None)
    assert metrics.final_endpoint_error_m == pytest.approx(0.003)


def test_departure_latency_is_when_the_endpoint_first_leaves_the_dwell_radius_of_its_activation_position() -> None:
    """The endpoint passes 1 cm from where it was at activation between 1.55 s and 1.56 s."""
    metrics = run_metrics(_run(), activation_s=ACTIVATION, target=TARGET, final_dwell_samples=None)
    assert metrics.departure_latency_s == pytest.approx(0.56)


def test_the_activation_jump_is_the_command_step_at_activation() -> None:
    """The distance between the first active command and the measured posture it starts from."""
    metrics = run_metrics(_run(), activation_s=ACTIVATION, target=TARGET, final_dwell_samples=None)
    assert metrics.activation_jump_rad == pytest.approx(0.05)


def test_time_to_the_final_dwell_counts_from_activation_to_the_start_of_the_dwell_that_ends_the_run() -> None:
    """A final dwell of 101 samples in a 301-sample run starts at 2.0 s, one second after activation."""
    metrics = run_metrics(_run(), activation_s=ACTIVATION, target=TARGET, final_dwell_samples=101)
    assert metrics.time_to_final_dwell_s == pytest.approx(1.0)


def test_a_run_that_never_leaves_has_no_departure_latency() -> None:
    """Never departing is recorded as absent, not as a latency of zero or of the horizon."""
    run = _run()
    run["tip"] = np.zeros_like(run["tip"])
    metrics = run_metrics(run, activation_s=ACTIVATION, target=TARGET, final_dwell_samples=None)
    assert metrics.departure_latency_s is None


def test_a_run_that_aborted_before_activation_has_no_active_metrics() -> None:
    """With no active segment there is nothing to measure, and nothing is reported as zero."""
    run = {name: values[:50] for name, values in _run().items()}
    metrics = run_metrics(run, activation_s=ACTIVATION, target=TARGET, final_dwell_samples=None)
    assert metrics.n_active_samples == 0
    assert metrics.peak_speed_rad_s is None
    assert metrics.final_endpoint_error_m is None
    assert metrics.activation_jump_rad is None


def test_a_final_dwell_longer_than_the_active_segment_is_refused() -> None:
    """The dwell is measured over the active segment, so a longer one cannot belong to this run."""
    with pytest.raises(ValueError, match="final dwell"):
        run_metrics(_run(), activation_s=ACTIVATION, target=TARGET, final_dwell_samples=202)

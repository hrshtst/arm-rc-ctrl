# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3REP-004: the simulation-only velocity abort and the non-terminating historical-limit diagnostics (D5, C2)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from arm_rc_ctrl.controllers.tracking import LimitedTracker, TrackerConfig
from arm_rc_ctrl.experiments.simulation import CheckedState, simulate
from arm_rc_ctrl.experiments.termination import completed, limit_violation
from arm_rc_ctrl.experiments.velocity_diagnostics import (
    PHASES,
    VelocityDiagnostics,
    velocity_diagnostics,
)
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import load_scenario

SCENARIO_FILE = repository_root() / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"


class _Step:
    """A joint reference that jumps far from the start, driving a stiff tracker over the speed limit."""

    def __init__(self, target: np.ndarray) -> None:
        self._target = target

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del t
        return self._target, np.zeros_like(self._target), np.zeros_like(self._target)


def _stiff_tracker() -> LimitedTracker:
    """A PD step response whose peak speed (about 24 rad/s) exceeds the fixture's 20 rad/s bound but not twice it."""
    config = TrackerConfig(type="pd", kp=(15.0, 15.0), kd=(1.5, 1.5))
    target = np.asarray(load_scenario(SCENARIO_FILE).task.initial_q) + np.array([0.6, -0.5])
    return LimitedTracker(cast("Any", _Step(target)), config, (10.0, 10.0))


def test_velocity_abort_replaces_the_scenario_bound_and_records_every_checked_state() -> None:
    """The legacy path aborts at the scenario limit; the override applies its own bound; the sink is complete."""
    scenario = load_scenario(SCENARIO_FILE)
    legacy: list[CheckedState] = []
    _arrays, termination = simulate(scenario, _stiff_tracker(), duration_s=1.0, checked_states=legacy)
    assert termination.kind == "limit_violation"
    assert termination.limit == "joint_velocity"
    assert termination.bound == scenario.limits.velocity[termination.joint or 0]
    # The sink holds every checked state including the terminal offending one, at the control cadence.
    assert len(legacy) == termination.step + 1
    assert legacy[-1].step == termination.step
    assert legacy[-1].t == termination.time_s
    assert abs(legacy[-1].dq[termination.joint or 0]) == abs(termination.value or 0.0)
    assert all(np.isclose(s.t, s.step * scenario.timing.dt) for s in legacy)
    relaxed: list[CheckedState] = []
    doubled = tuple(2.0 * v for v in scenario.limits.velocity)
    _arrays2, termination2 = simulate(
        scenario, _stiff_tracker(), duration_s=1.0, velocity_abort=doubled, checked_states=relaxed
    )
    assert termination2.kind == "completed"  # the peak stays below twice the bound
    assert len(relaxed) == termination2.step + 1 > len(legacy)
    assert max(float(np.max(np.abs(s.dq))) for s in relaxed) > scenario.limits.velocity[0]
    assert max(float(np.max(np.abs(s.dq))) for s in relaxed) < doubled[0]
    with pytest.raises(ValueError, match="velocity_abort"):
        simulate(scenario, _stiff_tracker(), duration_s=0.1, velocity_abort=(1.0,))
    with pytest.raises(ValueError, match="velocity_abort"):
        simulate(scenario, _stiff_tracker(), duration_s=0.1, velocity_abort=(0.0, 1.0))


def test_velocity_diagnostics_report_crossings_peaks_and_time_above_by_phase() -> None:
    """Peaks, first crossings on both clocks, per-phase time above the historical limit, and abort details."""
    dt = 0.01
    activation = 0.25
    dwell_start = 0.8
    states = [
        CheckedState(t=k * dt, step=k, q=np.zeros(2), dq=np.array([speed, 0.0]))
        for k, speed in enumerate([0.0] * 20 + [7.0] * 10 + [2.0] * 60 + [-6.5] * 20 + [1.0] * 10)
    ]
    abort = limit_violation(states[-1].t + dt, len(states), "joint_velocity", 13.0, 12.0, joint=0)
    terminal = CheckedState(t=states[-1].t + dt, step=len(states), q=np.zeros(2), dq=np.array([13.0, 0.0]))
    diagnostics = velocity_diagnostics(
        [*states, terminal],
        abort,
        historical=(6.0, 6.0),
        abort_limit=(12.0, 12.0),
        activation_s=activation,
        dwell_start_s=dwell_start,
        dt=dt,
    )
    assert isinstance(diagnostics, VelocityDiagnostics)
    assert diagnostics.crossed_historical
    joint0, joint1 = diagnostics.joints
    assert joint0.peak_abs_speed == 13.0
    assert joint0.first_historical_crossing is not None
    assert joint0.first_historical_crossing.step == 20
    assert joint0.first_historical_crossing.run_time_s == pytest.approx(0.20)
    assert joint0.first_historical_crossing.task_time_s == pytest.approx(0.20 - activation)  # in the warm-up
    assert joint0.first_abort_crossing is not None
    assert joint0.first_abort_crossing.step == len(states)
    assert joint1.first_historical_crossing is None
    assert joint1.peak_abs_speed == 0.0
    # Ten samples at 7 rad/s: five in the warm-up (0.20-0.24 s) and five in the movement (0.25-0.29 s);
    # twenty at -6.5 rad/s from 0.90 s on: the run clock 0.90-1.09 s is task time 0.65-0.84 s, so fifteen
    # in the movement and five in the dwell (task time >= 0.8 s); the terminal sample counts in the dwell.
    assert joint0.time_above_historical_s == {
        "warmup": pytest.approx(5 * dt),
        "movement": pytest.approx(20 * dt),
        "dwell": pytest.approx(6 * dt),
    }
    assert set(joint0.time_above_historical_s) == set(PHASES)
    assert diagnostics.abort is not None
    assert diagnostics.abort.joint == 0
    assert diagnostics.abort.value == 13.0
    assert diagnostics.abort.bound == 12.0
    assert diagnostics.abort.task_time_s == pytest.approx(abort.time_s - activation)
    assert diagnostics.terminal_state is not None
    assert diagnostics.terminal_state.step == len(states)
    assert tuple(diagnostics.terminal_state.dq) == (13.0, 0.0)
    assert diagnostics.samples == len(states) + 1
    # A completed run has neither abort nor terminal state and no crossings when it stayed slow.
    calm = velocity_diagnostics(
        states[:20],
        completed(states[19].t, 19),
        historical=(6.0, 6.0),
        abort_limit=(12.0, 12.0),
        activation_s=activation,
        dwell_start_s=dwell_start,
        dt=dt,
    )
    assert not calm.crossed_historical
    assert calm.abort is None
    assert calm.terminal_state is None
    assert all(v == 0.0 for v in calm.joints[0].time_above_historical_s.values())
    with pytest.raises(ValueError, match="historical"):
        velocity_diagnostics(
            states,
            completed(0.0, 0),
            historical=(6.0,),
            abort_limit=(12.0, 12.0),
            activation_s=activation,
            dwell_start_s=dwell_start,
            dt=dt,
        )
    with pytest.raises(ValueError, match="at least one"):
        velocity_diagnostics(
            [],
            completed(0.0, 0),
            historical=(6.0, 6.0),
            abort_limit=(12.0, 12.0),
            activation_s=activation,
            dwell_start_s=dwell_start,
            dt=dt,
        )
    with pytest.raises(ValueError, match="contradicts"):
        replace(diagnostics, crossed_historical=False)
    with pytest.raises(ValueError, match="recorded together"):
        replace(diagnostics, terminal_state=None)
    with pytest.raises(ValueError, match="phases"):
        replace(joint0, time_above_historical_s={"warmup": 0.0})

# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-008: replay commands its reference through the same causal derivative policy as RC.

The paired comparison is only about the generator if both arms are driven the
same way. A reference that hands the tracker the recording's offline
derivatives gives replay information RC never has -- derivatives computed from
the whole trajectory, including its future -- and then switches them to zero
the instant the log ends. Replay therefore derives its velocity and
acceleration causally from the positions it commands, before activation, during
the recording, and through the continuation after it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from arm_rc_ctrl.controllers.estimator import CausalDerivativeEstimator, EstimatorConfig
from arm_rc_ctrl.data.manual_scenario import load_manual_scenario
from arm_rc_ctrl.experiments.manual_evaluation import CausalReplayReference
from arm_rc_ctrl.experiments.manual_fixture import fixture_samples
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from arm_rc_ctrl.data.samples import SampleSet

SCENARIO_FILE = repository_root() / "tests" / "fixtures" / "configs" / "planar_2dof_manual_fixture.toml"
SCENARIO = load_manual_scenario(SCENARIO_FILE)
DT = SCENARIO.timing.dt
ACTIVATION_S = 0.1
CUTOFFS = (20.0, 20.0)


def _samples() -> SampleSet:
    """A synthetic take on the task grid, the same shape the study's demonstrations have."""
    return fixture_samples(SCENARIO, 60)


def _reference() -> CausalReplayReference:
    return CausalReplayReference.from_samples(
        _samples(),
        activation_s=ACTIVATION_S,
        hold=np.asarray(SCENARIO.task.initial_q, dtype=np.float64),
        estimator=CausalDerivativeEstimator(
            EstimatorConfig(nominal_dt_s=DT, velocity_cutoff_hz=CUTOFFS[0], acceleration_cutoff_hz=CUTOFFS[1]),
            SCENARIO.robot.dof,
        ),
    )


def _independent(positions: list[NDArray[np.float64]], times: list[float]) -> tuple[NDArray[np.float64], ...]:
    """The causal estimate of the same commanded positions, computed outside the reference."""
    estimator = CausalDerivativeEstimator(
        EstimatorConfig(nominal_dt_s=DT, velocity_cutoff_hz=CUTOFFS[0], acceleration_cutoff_hz=CUTOFFS[1]),
        SCENARIO.robot.dof,
    )
    last = None
    for t, q in zip(times, positions, strict=True):
        last = estimator.update(t, q)
    assert last is not None
    return last.dq, last.ddq


# --- the derivatives are causal ------------------------------------------------------------------


def test_the_first_sample_has_no_derivative_to_estimate_yet() -> None:
    """A causal estimator has seen one sample, so it reports zero rather than a recorded velocity."""
    reference = _reference()
    _q, dq, ddq = reference.sample(0.0)
    assert np.array_equal(dq, np.zeros(SCENARIO.robot.dof))
    assert np.array_equal(ddq, np.zeros(SCENARIO.robot.dof))


def test_the_derivatives_are_the_causal_estimate_of_the_commanded_positions() -> None:
    """What the tracker receives is derived from what was commanded, not read from the recording."""
    reference = _reference()
    times = [round(k * DT, 12) for k in range(30)]
    positions: list[NDArray[np.float64]] = []
    dq = ddq = np.zeros(SCENARIO.robot.dof)
    for t in times:
        q, dq, ddq = reference.sample(t)
        positions.append(np.asarray(q, dtype=np.float64))
    expected_dq, expected_ddq = _independent(positions, times)
    assert np.allclose(dq, expected_dq, rtol=0, atol=1e-12)
    assert np.allclose(ddq, expected_ddq, rtol=0, atol=1e-12)


def test_the_recording_supplies_positions_not_derivatives() -> None:
    """The offline derivatives of the take are never handed to the tracker.

    They are computed from the whole trajectory, so using them would give
    replay information the generator cannot have.
    """
    samples = _samples()
    reference = _reference()
    recorded_dq = np.asarray(samples.dq, dtype=np.float64)
    seen: list[NDArray[np.float64]] = []
    for k in range(40):
        _q, dq, _ddq = reference.sample(round(k * DT, 12))
        seen.append(np.asarray(dq, dtype=np.float64))
    moving = int(np.argmax(np.max(np.abs(recorded_dq), axis=1)))
    assert np.max(np.abs(recorded_dq[moving])) > 0.0
    # At the sample where the recording moves fastest, the causal estimate is not the recorded value.
    active = np.asarray(seen[min(len(seen) - 1, moving + round(ACTIVATION_S / DT))])
    assert not np.allclose(active, recorded_dq[moving], rtol=0, atol=1e-9)


def test_the_continuation_holds_the_final_posture_with_causal_derivatives() -> None:
    """After the log ends the command stays at the final recorded posture, and its derivatives follow from it."""
    samples = _samples()
    reference = _reference()
    duration = float(np.asarray(samples.t, dtype=np.float64)[-1])
    final = np.asarray(samples.q, dtype=np.float64)[-1]
    steps = round((ACTIVATION_S + duration + 0.2) / DT)
    q = dq = np.zeros(SCENARIO.robot.dof)
    for k in range(steps + 1):
        q, dq, _ddq = reference.sample(round(k * DT, 12))
    assert np.allclose(q, final, rtol=0, atol=1e-12)
    # Held position means the causal velocity decays toward zero rather than jumping to it.
    assert float(np.max(np.abs(dq))) == pytest.approx(0.0, abs=1e-6)


def test_an_estimator_of_the_wrong_width_is_refused() -> None:
    """The reference and its estimator must describe the same arm."""
    with pytest.raises(ValueError, match="dof"):
        CausalReplayReference.from_samples(
            _samples(),
            activation_s=ACTIVATION_S,
            hold=np.asarray(SCENARIO.task.initial_q, dtype=np.float64),
            estimator=CausalDerivativeEstimator(EstimatorConfig(nominal_dt_s=DT), SCENARIO.robot.dof + 1),
        )

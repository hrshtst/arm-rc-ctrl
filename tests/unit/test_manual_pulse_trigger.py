# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-008: the dwell-triggered force pulse and the manual protocol driving the simulator (plan section 6).

With natural movement durations a pulse at a fixed task time could land in the
pre-roll, the movement, or the dwell, so the manual protocol triggers it on the
measured motion instead: once the arm satisfies the target predicate
continuously for the trigger hold, the pulse fires once and its actual
timestamp is recorded. The manual task schema is a different configuration
class from the historical one, so the simulator takes both structurally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from arm_rc_ctrl.controllers.tracking import LimitedTracker, TrackerConfig
from arm_rc_ctrl.data.manual_scenario import load_manual_scenario, manual_endpoint_positions
from arm_rc_ctrl.experiments.disturbances import ForcePulse
from arm_rc_ctrl.experiments.simulation import DwellCounter, DwellTrigger, simulate
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import joint_target

if TYPE_CHECKING:
    from numpy.typing import NDArray

SCENARIO_FILE = repository_root() / "tests" / "fixtures" / "configs" / "planar_2dof_manual_fixture.toml"
HOLD_S = 0.05
PULSE_S = 0.2


class _Hold:
    """A joint reference that commands one fixed posture with zero velocity and acceleration."""

    def __init__(self, posture: NDArray[np.float64]) -> None:
        self._posture = posture

    def sample(self, t: float) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        del t
        return self._posture, np.zeros_like(self._posture), np.zeros_like(self._posture)


def _tracker(posture: NDArray[np.float64]) -> LimitedTracker:
    config = TrackerConfig(type="pd", kp=(80.0, 80.0), kd=(12.0, 12.0))
    return LimitedTracker(cast("Any", _Hold(posture)), config, (10.0, 5.0))


def _trigger(scenario: Any) -> DwellTrigger:  # noqa: ANN401 - the widened scenario protocol
    return DwellTrigger.from_polar(
        target=scenario.task.target,
        tolerance_m=scenario.task.tolerance,
        max_velocity_rad_s=scenario.task.dwell_max_velocity,
        hold_s=HOLD_S,
        duration_s=PULSE_S,
        magnitude_n=12.0,
        direction_deg=0.0,
    )


# --- the manual task schema drives the simulator ---------------------------------------------


def test_a_manual_task_configuration_drives_the_simulator() -> None:
    """The manual schema carries the same robot, limits, posture, and period, so the loop takes it structurally."""
    scenario = load_manual_scenario(SCENARIO_FILE)
    start = np.asarray(scenario.task.initial_q, dtype=np.float64)
    arrays, termination = simulate(scenario, _tracker(start), duration_s=0.2)
    assert termination.kind == "completed"
    assert arrays.arrays["t"].shape[0] == round(0.2 / scenario.timing.dt) + 1
    assert np.allclose(arrays.arrays["q"][0], start)


# --- the trigger rule ------------------------------------------------------------------------


def test_the_counter_fires_once_after_a_continuous_hold() -> None:
    """The rule is a continuous run: the count reaches the hold, fires exactly once, and never fires again."""
    counter = DwellCounter(hold_samples=3)
    assert [counter.update(qualifies=True) for _ in range(5)] == [False, False, True, False, False]


def test_an_excursion_restarts_the_counter() -> None:
    """Any sample that fails the predicate restarts the timer, so an interrupted hold never fires."""
    counter = DwellCounter(hold_samples=3)
    assert not any(counter.update(qualifies=q) for q in (True, True, False, True, True))
    assert counter.update(qualifies=True)


def test_the_pulse_fires_once_the_target_predicate_has_held(tmp_path: object) -> None:
    """Starting at the target posture, the arm qualifies immediately and the pulse fires after the hold."""
    del tmp_path
    scenario = load_manual_scenario(SCENARIO_FILE)
    posture = np.asarray(joint_target(scenario), dtype=np.float64)
    assert np.hypot(*(manual_endpoint_positions(scenario, posture[None, :])[0] - scenario.task.target)) < 1e-9
    fired: list[ForcePulse] = []
    arrays, termination = simulate(
        scenario,
        _tracker(posture),
        duration_s=0.4,
        initial_q=tuple(float(v) for v in posture),
        force_trigger=_trigger(scenario),
        triggered=fired,
    )
    assert termination.kind == "completed"
    assert len(fired) == 1
    pulse = fired[0]
    # Six consecutive qualifying samples are 0.05 s on the 0.01 s grid, counted from the first sample.
    assert pulse.start_s == pytest.approx(HOLD_S)
    assert pulse.duration_s == PULSE_S
    run_t = arrays.arrays["t"]
    applied = np.linalg.norm(arrays.arrays["ext_force"], axis=1) > 0
    assert np.array_equal(applied, (run_t >= pulse.start_s) & (run_t < pulse.end_s))


def test_no_pulse_fires_when_the_target_is_never_satisfied() -> None:
    """A run that never reaches the target never triggers, and the caller can tell that from an empty record."""
    scenario = load_manual_scenario(SCENARIO_FILE)
    start = np.asarray(scenario.task.initial_q, dtype=np.float64)
    fired: list[ForcePulse] = []
    arrays, termination = simulate(
        scenario, _tracker(start), duration_s=0.3, force_trigger=_trigger(scenario), triggered=fired
    )
    assert termination.kind == "completed"
    assert fired == []
    assert not np.any(arrays.arrays["ext_force"])


def test_a_scheduled_pulse_and_a_trigger_cannot_both_be_given() -> None:
    """One disturbance rule per run: a fixed schedule and a state trigger would race for the same channel."""
    scenario = load_manual_scenario(SCENARIO_FILE)
    start = np.asarray(scenario.task.initial_q, dtype=np.float64)
    with pytest.raises(ValueError, match="force_trigger"):
        simulate(
            scenario,
            _tracker(start),
            duration_s=0.1,
            force=ForcePulse(start_s=0.01, duration_s=0.02, force=(1.0, 0.0)),
            force_trigger=_trigger(scenario),
        )

# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-008: the verdict of one manual evaluation run (plan section 6).

Success is completing the configured horizon inside every limit, at no more
than the approved torque saturation, with an uninterrupted final dwell that
reaches the end of the run. A force case additionally needs its pulse to have
fired in time and a full dwell after it. Actual motion and the generated
reference are judged by the same rule and reported separately, and the joint
feasibility verdict is given only when both pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from arm_rc_ctrl.data.manual_scenario import load_manual_scenario, manual_endpoint_positions
from arm_rc_ctrl.experiments.disturbances import ForcePulse
from arm_rc_ctrl.experiments.manual_evaluation import ManualRunOutcome, manual_run_outcome
from arm_rc_ctrl.experiments.run_record import RunArrays
from arm_rc_ctrl.experiments.termination import completed, limit_violation
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import joint_target

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from arm_rc_ctrl.experiments.termination import Termination

SCENARIO_FILE = repository_root() / "tests" / "fixtures" / "configs" / "planar_2dof_manual_fixture.toml"
SCENARIO = load_manual_scenario(SCENARIO_FILE)
DT = SCENARIO.timing.dt
ACTIVATION_S = 0.1
HORIZON_S = 0.5
ON_TARGET = np.asarray(joint_target(SCENARIO), dtype=np.float64)
"""A posture whose endpoint is exactly the target, so a hold there satisfies the dwell rule."""

AWAY = np.asarray(SCENARIO.task.initial_q, dtype=np.float64)


def _run(
    *,
    rows: int | None = None,
    hold_from: int = 0,
    generated: NDArray[np.float64] | None = None,
    saturation: float = 0.0,
) -> RunArrays:
    """A crafted run on the evaluation schedule: away from the target, then holding it from ``hold_from``."""
    n = rows if rows is not None else round((ACTIVATION_S + HORIZON_S) / DT) + 1
    t = np.arange(n, dtype=np.float64) * DT
    q = np.tile(AWAY, (n, 1))
    q[hold_from:] = ON_TARGET
    dq = np.zeros((n, 2), dtype=np.float64)
    dq[:hold_from] = 1.0
    tip = manual_endpoint_positions(SCENARIO, q)
    zeros = np.zeros((n, 2), dtype=np.float64)
    saturated = np.zeros(n, dtype=np.int64)
    saturated[: round(saturation * n)] = 1
    data: dict[str, NDArray[np.float64] | NDArray[np.int64]] = {
        "t": t,
        "q": q,
        "dq": dq,
        "tip": tip,
        "q_desired": q.copy(),
        "dq_desired": dq.copy(),
        "dq_desired_raw": dq.copy(),
        "ddq_desired": zeros.copy(),
        "ddq_desired_raw": zeros.copy(),
        "tracking_error": zeros.copy(),
        "tau_requested": zeros.copy(),
        "task_code": np.zeros((n, 0), dtype=np.float64),
        "saturation": saturated,
    }
    if generated is not None:
        active = t >= ACTIVATION_S - 1e-9
        readout = np.full((n, 2), np.nan, dtype=np.float64)
        readout[active] = generated[active]
        data["generator_output_q"] = readout
        # The readout is masked before activation, and phase is what defines that boundary.
        data["phase"] = active.astype(np.int64)
    return RunArrays(dict(data))


def _outcome(
    arrays: RunArrays,
    termination: Termination | None = None,
    *,
    pulse: ForcePulse | None = None,
    force_case: bool = False,
) -> ManualRunOutcome:
    """Judge a crafted run, completing normally at its last sample unless a termination is given."""
    run_t = arrays.arrays["t"]
    term = completed(float(run_t[-1]), int(run_t.shape[0]) - 1) if termination is None else termination
    return manual_run_outcome(
        arrays,
        term,
        scenario=SCENARIO,
        activation_s=ACTIVATION_S,
        horizon_s=HORIZON_S,
        pulse=pulse,
        force_case=force_case,
    )


# --- the nominal verdict ---------------------------------------------------------------------


def test_a_run_that_holds_the_target_to_the_end_succeeds() -> None:
    """Completed the horizon, inside every limit, with the final dwell reaching the last sample."""
    outcome = _outcome(_run(hold_from=20))
    assert outcome.completed
    assert outcome.dwell.ok
    assert outcome.success
    assert outcome.reason is None


def test_a_run_that_reaches_the_target_and_leaves_it_fails_on_the_dwell() -> None:
    """Reaching briefly and drifting away cannot count as stable success."""
    n = round((ACTIVATION_S + HORIZON_S) / DT) + 1
    arrays = _run(hold_from=20)
    q = np.array(arrays.arrays["q"])
    q[n - 5 :] = AWAY
    data = dict(arrays.arrays)
    data["q"] = q
    data["tip"] = manual_endpoint_positions(SCENARIO, q)
    outcome = _outcome(RunArrays(data))
    assert outcome.completed
    assert not outcome.dwell.ok
    assert not outcome.success
    assert outcome.reason is not None
    assert "dwell" in outcome.reason


def test_a_run_short_of_the_horizon_is_incomplete_but_keeps_its_metrics() -> None:
    """An aborted run retains its partial dwell measurement and its terminal evidence (I5)."""
    arrays = _run(rows=30, hold_from=10)
    termination = limit_violation(float(arrays.arrays["t"][-1]), 29, "joint_velocity", 25.0, 20.0, joint=0)
    outcome = _outcome(arrays, termination)
    assert not outcome.completed
    assert not outcome.success
    assert outcome.dwell.final_samples > 0
    assert outcome.reason is not None
    assert "limit_violation" in outcome.reason


def test_saturation_above_the_bound_fails() -> None:
    """Success allows at most the approved fraction of saturated samples."""
    outcome = _outcome(_run(hold_from=20, saturation=0.5))
    assert outcome.saturation_fraction > 0.005
    assert not outcome.success
    assert outcome.reason is not None
    assert "saturation" in outcome.reason


# --- force cases -----------------------------------------------------------------------------


def test_a_force_case_whose_pulse_never_fired_is_reported_explicitly() -> None:
    """A missing trigger cannot count as a successful disturbance-recovery test."""
    outcome = _outcome(_run(hold_from=20), pulse=None, force_case=True)
    assert outcome.trigger is not None
    assert not outcome.trigger.ok
    assert not outcome.success
    assert outcome.reason is not None
    assert "trigger" in outcome.reason


def test_a_force_case_needs_its_final_dwell_after_the_pulse() -> None:
    """The pulse resets the success timer, so only the dwell after it counts."""
    late = ForcePulse(start_s=ACTIVATION_S + HORIZON_S - 0.05, duration_s=0.02, force=(12.0, 0.0))
    outcome = _outcome(_run(hold_from=20), pulse=late, force_case=True)
    assert outcome.trigger is not None
    assert not outcome.trigger.ok
    assert not outcome.success
    assert outcome.reason is not None
    assert "trigger" in outcome.reason


def test_a_force_case_with_room_after_the_pulse_succeeds() -> None:
    """Fired early enough, with the arm holding the target through to the end."""
    early = ForcePulse(start_s=ACTIVATION_S + 0.05, duration_s=0.02, force=(12.0, 0.0))
    outcome = _outcome(_run(hold_from=20), pulse=early, force_case=True)
    assert outcome.trigger is not None
    assert outcome.trigger.ok
    assert outcome.success


# --- the generated reference -------------------------------------------------------------------


def test_a_replay_run_has_no_generated_reference() -> None:
    """Only an RC run carries a readout, so replay reports none rather than an empty one."""
    assert _outcome(_run(hold_from=20)).generated is None


def test_the_generated_reference_is_judged_by_the_same_dwell_rule() -> None:
    """Its endpoint comes from the generated joint positions, checked over the active samples only."""
    n = round((ACTIVATION_S + HORIZON_S) / DT) + 1
    generated = np.tile(ON_TARGET, (n, 1))
    outcome = _outcome(_run(hold_from=20, generated=generated))
    assert outcome.generated is not None
    assert outcome.generated.dwell is not None
    assert outcome.generated.dwell.ok
    assert outcome.generated.within_workspace is True
    assert outcome.generated.ok
    assert outcome.success


def test_a_generated_reference_outside_the_joint_limits_is_infeasible() -> None:
    """Actual motion can look fine while the generated command leaves the arm's bounds; both must pass."""
    n = round((ACTIVATION_S + HORIZON_S) / DT) + 1
    generated = np.tile(ON_TARGET, (n, 1))
    generated[n - 3 :] = np.array([5.0, 0.0])
    outcome = _outcome(_run(hold_from=20, generated=generated))
    assert outcome.generated is not None
    assert not outcome.generated.within_position_limits
    assert not outcome.generated.ok
    assert not outcome.success
    assert outcome.reason is not None
    assert "position_limits" in outcome.reason
    # The endpoint checks are left unevaluated: computing them would clamp the posture and answer
    # for a trajectory the generator never commanded (skelarm warns, and the answer would be wrong).
    assert outcome.generated.within_workspace is None
    assert outcome.generated.dwell is None

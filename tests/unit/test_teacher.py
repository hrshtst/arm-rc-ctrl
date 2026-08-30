# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-013: the scripted teacher plans and records a deterministic minimum-jerk reach."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from skelarm import StateLog

from arm_rc_ctrl.data.teacher import main, minimum_jerk, plan_reach, record_demonstration
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import endpoint_positions, joint_target, load_scenario

REPO_ROOT = repository_root()
TASK_1A = REPO_ROOT / "configs" / "tasks" / "task_1a.toml"


def test_minimum_jerk_profile_boundary_conditions() -> None:
    """s(0)=0, s(1)=1 with zero velocity and acceleration at both ends; derivatives are consistent."""
    tau = np.linspace(0.0, 1.0, 1001)
    s, ds, dds = minimum_jerk(tau)
    assert (s[0], s[-1]) == (0.0, 1.0)
    assert ds[0] == ds[-1] == 0.0
    assert dds[0] == dds[-1] == 0.0
    assert np.allclose(np.gradient(s, tau)[10:-10], ds[10:-10], atol=1e-3)
    assert np.allclose(np.gradient(ds, tau)[10:-10], dds[10:-10], atol=1e-2)
    assert np.all(np.diff(s) >= 0)


def test_plan_holds_prime_moves_and_dwells_on_the_control_grid() -> None:
    """The plan sits on the scenario grid, holds the initial posture, reaches the IK solution, and holds it."""
    scenario = load_scenario(TASK_1A)
    plan = plan_reach(scenario)
    intervals = scenario.timing.intervals
    assert plan.t.shape[0] == round(intervals.duration_s / scenario.timing.dt) + 1
    assert np.allclose(np.diff(plan.t), scenario.timing.dt)
    prime = plan.t <= intervals.prime[1]
    dwell = plan.t >= intervals.dwell[0]
    assert np.allclose(plan.q[prime], scenario.task.initial_q)
    assert np.allclose(plan.q[dwell], joint_target(scenario))
    assert np.all(plan.dq[prime] == 0.0)
    assert np.all(plan.dq[dwell] == 0.0)
    assert np.all(plan.ddq[dwell] == 0.0)
    moving = (plan.t > intervals.move[0]) & (plan.t < intervals.move[1])
    assert np.any(np.abs(plan.dq[moving]) > 0)
    tip = endpoint_positions(scenario, plan.q[-1:])[0]
    assert np.linalg.norm(tip - np.asarray(scenario.task.target)) < 1e-9


def test_recorded_demonstration_tracks_the_plan_and_is_deterministic() -> None:
    """The skelarm recording follows the plan closely, ends on target, and repeats bit for bit."""
    scenario = load_scenario(TASK_1A)
    log, plan = record_demonstration(scenario)
    again, _ = record_demonstration(scenario)
    assert log.channel_names == ["q", "dq", "tau"]
    assert len(log) == plan.t.shape[0]
    assert np.array_equal(log.channel("q"), again.channel("q"))
    assert np.array_equal(log.channel("tau"), again.channel("tau"))
    assert np.max(np.abs(log.channel("q") - plan.q)) < 1e-3
    tip = endpoint_positions(scenario, log.channel("q")[-1:])[0]
    assert np.linalg.norm(tip - np.asarray(scenario.task.target)) < 1e-4
    assert np.all(np.abs(log.channel("dq")) <= np.asarray(scenario.limits.velocity))
    assert np.all(np.abs(log.channel("tau")) <= np.asarray(scenario.limits.torque))
    assert log.channel_meta["q"]["unit"] == "rad"
    assert log.extra["teacher"] == "minimum-jerk"
    assert log.build_skeleton().num_joints == 2


def test_joint_target_rejects_unreachable_and_non_planar_cases() -> None:
    """The closed-form IK guards its assumptions."""
    scenario = load_scenario(TASK_1A)
    import dataclasses

    far = dataclasses.replace(scenario.task, target=(0.35, 0.44))  # 0.562 m > 0.55 m reach
    with pytest.raises(ValueError, match="exceeds the arm's reach"):
        dataclasses.replace(scenario, task=far)
    q = joint_target(scenario)
    assert 0.8 < q[0] < 0.9
    assert 1.1 < q[1] < 1.2
    elbow_down = joint_target(scenario, elbow_up=False)
    assert elbow_down[1] < 0


def test_command_line_writes_a_log_and_never_overwrites(tmp_path: Path) -> None:
    """The CLI writes the log once and refuses an existing output path."""
    out = tmp_path / "demo.sklog.npz"
    assert main(["--scenario", str(TASK_1A), "--out", str(out)]) == 0
    assert len(StateLog.load(out)) == 501
    with pytest.raises(FileExistsError):
        main(["--scenario", str(TASK_1A), "--out", str(out)])

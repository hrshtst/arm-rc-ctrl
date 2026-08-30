# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Scripted teacher for task 1-a: a minimum-jerk reach recorded as a ``skelarm`` log.

The teacher plans a joint-space minimum-jerk motion from the scenario's initial
posture to the closed-form joint solution of its endpoint target over the
scenario's ``move`` interval, holds the initial posture during ``prime`` and the
final posture during ``dwell``, and tracks the plan with a high-gain
computed-torque controller in ``skelarm`` so the recording is a physically
consistent trajectory with torques. The result is deterministic.

Command line::

    python -m arm_rc_ctrl.data.teacher --scenario configs/tasks/task_1a.toml --out demo.sklog.npz
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np
from numpy.typing import NDArray
from skelarm import StateLog

from arm_rc_ctrl.controllers.reference import DemonstrationReference
from arm_rc_ctrl.controllers.tracking import TrackerConfig
from arm_rc_ctrl.experiments.replay import simulate_tracking
from arm_rc_ctrl.scenario import ScenarioConfig, build_skeleton, joint_target, load_scenario

__all__ = ["TEACHER_GAINS", "TeacherPlan", "main", "minimum_jerk", "plan_reach", "record_demonstration"]

TEACHER_GAINS: Final = TrackerConfig("computed_torque", (400.0, 400.0), (40.0, 40.0))
"""High-gain computed torque so the recorded motion follows the plan closely (not a baseline setting)."""

PRODUCER: Final = "arm_rc_ctrl.data.teacher"


@dataclass(frozen=True)
class TeacherPlan:
    """Planned joint trajectory on the control grid."""

    t: NDArray[np.float64]
    q: NDArray[np.float64]
    dq: NDArray[np.float64]
    ddq: NDArray[np.float64]
    q_start: tuple[float, ...]
    q_goal: tuple[float, ...]


def minimum_jerk(tau: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Normalized minimum-jerk profile ``s(tau)`` and its first two derivatives for ``tau`` in ``[0, 1]``."""
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    dds = 60 * tau - 180 * tau**2 + 120 * tau**3
    return s, ds, dds


def plan_reach(scenario: ScenarioConfig) -> TeacherPlan:
    """Plan prime → minimum-jerk move → dwell on the scenario's control grid."""
    intervals = scenario.timing.intervals
    dt = scenario.timing.dt
    n = round(intervals.duration_s / dt) + 1
    t = np.arange(n, dtype=np.float64) * dt
    q_start = np.asarray(scenario.task.initial_q, dtype=np.float64)
    q_goal = np.asarray(joint_target(scenario), dtype=np.float64)
    move_start, move_end = intervals.move
    span = move_end - move_start
    tau = np.clip((t - move_start) / span, 0.0, 1.0)
    s, ds, dds = minimum_jerk(tau)
    delta = q_goal - q_start
    q = q_start[None, :] + s[:, None] * delta[None, :]
    dq = (ds / span)[:, None] * delta[None, :]
    ddq = (dds / span**2)[:, None] * delta[None, :]
    outside = (t < move_start) | (t > move_end)
    dq[outside] = 0.0
    ddq[outside] = 0.0
    return TeacherPlan(t, q, dq, ddq, tuple(float(v) for v in q_start), tuple(float(v) for v in q_goal))


def record_demonstration(
    scenario: ScenarioConfig, *, gains: TrackerConfig = TEACHER_GAINS
) -> tuple[StateLog, TeacherPlan]:
    """Track the plan in ``skelarm`` and return the native log plus the plan."""
    plan = plan_reach(scenario)
    reference = DemonstrationReference(plan.t, plan.q, plan.dq, plan.ddq)
    arrays, termination = simulate_tracking(scenario, reference, gains, duration_s=float(plan.t[-1]))
    if not termination.is_completed:
        msg = f"the teacher tracking run did not complete: {termination.kind}: {termination.detail}"
        raise RuntimeError(msg)
    joints = [f"j{i + 1}" for i in range(scenario.dof)]
    channel_meta = {
        "q": {"unit": "rad", "label": "joint angle", "columns": joints},
        "dq": {"unit": "rad/s", "label": "joint velocity", "columns": joints},
        "tau": {"unit": "N*m", "label": "applied joint torque", "columns": joints},
    }
    extra = {
        "teacher": "minimum-jerk",
        "scenario": scenario.name,
        "q_start": list(plan.q_start),
        "q_goal": list(plan.q_goal),
        "gains": {"type": gains.type, "kp": list(gains.kp), "kd": list(gains.kd)},
    }
    log = StateLog(build_skeleton(scenario), producer=PRODUCER, channel_meta=channel_meta, extra=extra)
    t = cast("NDArray[np.float64]", arrays.arrays["t"])
    q = cast("NDArray[np.float64]", arrays.arrays["q"])
    dq = cast("NDArray[np.float64]", arrays.arrays["dq"])
    tau = cast("NDArray[np.float64]", arrays.arrays["tau_applied"])
    for k in range(t.shape[0]):
        log.record(float(t[k]), q=q[k], dq=dq[k], tau=tau[k])
    return log, plan


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: write the scripted demonstration to a file."""
    parser = argparse.ArgumentParser(description="Record the scripted minimum-jerk task 1-a demonstration.")
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="output .sklog.npz (must not exist)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        msg = f"{out} already exists"
        raise FileExistsError(msg)
    log, plan = record_demonstration(load_scenario(Path(args.scenario)))
    log.save(out)
    print(f"wrote {out}: {len(log)} frames, q_start={plan.q_start}, q_goal={tuple(round(v, 4) for v in plan.q_goal)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

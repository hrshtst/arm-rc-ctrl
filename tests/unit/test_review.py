# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-013: interval proposals follow the speed profile and the review plot is written."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from arm_rc_ctrl.data.review import plot_intervals, propose_intervals
from arm_rc_ctrl.data.teacher import plan_reach
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import load_scenario

TASK_1A = repository_root() / "configs" / "tasks" / "task_1a.toml"


def test_proposal_recovers_the_scripted_intervals_from_speed() -> None:
    """On the minimum-jerk plan the proposal brackets the move interval to within a few samples."""
    scenario = load_scenario(TASK_1A)
    plan = plan_reach(scenario)
    proposal = propose_intervals(plan.t, plan.q, dq=plan.dq, speed_threshold=0.01)
    intervals = proposal.intervals
    assert intervals.prime[0] == 0.0
    assert abs(intervals.move[0] - 1.0) < 0.15  # min-jerk starts slowly, so the threshold crossing lags a little
    assert abs(intervals.move[1] - 4.0) < 0.15
    assert intervals.dwell[1] == pytest.approx(5.0)
    assert proposal.speed.shape == plan.t.shape
    without_dq = propose_intervals(plan.t, plan.q, speed_threshold=0.01)
    assert abs(without_dq.intervals.move[0] - intervals.move[0]) < 0.05


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"speed_threshold": 100.0}, "nothing moves"),
        ({"min_hold_s": 2.0}, "shorter than the minimum"),
    ],
)
def test_proposal_failures_are_explicit(kwargs: dict[str, float], message: str) -> None:
    """No movement or holds that are too short are reported, never silently accepted."""
    plan = plan_reach(load_scenario(TASK_1A))
    with pytest.raises(ValueError, match=message):
        propose_intervals(plan.t, plan.q, dq=plan.dq, **kwargs)


def test_movement_touching_the_ends_has_no_holds() -> None:
    """A recording that starts or ends while moving cannot yield prime/dwell holds."""
    t: NDArray[np.float64] = np.arange(50, dtype=np.float64) * 0.01
    q: NDArray[np.float64] = np.column_stack([t, t])  # constant speed from the first sample
    with pytest.raises(ValueError, match="touches the start or the end"):
        propose_intervals(t, q, speed_threshold=0.5)


def test_review_plot_is_written(tmp_path: Path) -> None:
    """The PNG is created with the phases shaded."""
    plan = plan_reach(load_scenario(TASK_1A))
    proposal = propose_intervals(plan.t, plan.q, dq=plan.dq, speed_threshold=0.01)
    out = tmp_path / "review.png"
    plot_intervals(plan.t, plan.q, proposal.speed, proposal.intervals, out, speed_threshold=0.01, title="test")
    assert out.stat().st_size > 1000
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

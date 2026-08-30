# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-015: termination reasons and outcomes are typed, validated, and serializable."""

from __future__ import annotations

import pytest

from arm_rc_ctrl.config import ConfigError, from_mapping, to_mapping
from arm_rc_ctrl.experiments.termination import (
    Outcome,
    Termination,
    backend_failure,
    completed,
    divergence,
    invalid_output,
    invalid_state,
    limit_violation,
    timeout,
)


def test_factories_cover_every_kind() -> None:
    """Normal end, invalid state/output, limits, divergence, timeout, and backend failure."""
    cases = [
        (completed(5.0, 500), "completed", True),
        (invalid_state(1.23, 123, "q contains NaN"), "invalid_state", False),
        (invalid_output(0.5, 50, "target generator returned shape (3,)"), "invalid_output", False),
        (limit_violation(2.0, 200, "joint_velocity", 7.2, 6.0, joint=1), "limit_violation", False),
        (divergence(3.0, 300, "|q| exceeded 1e3"), "divergence", False),
        (timeout(4.0, 400, 0.01), "timeout", False),
        (backend_failure(0.0, 0, "SimArm reported an error"), "backend_failure", False),
    ]
    for termination, kind, is_completed in cases:
        assert termination.kind == kind
        assert termination.is_completed is is_completed
    limit = limit_violation(2.0, 200, "joint_velocity", 7.2, 6.0, joint=1)
    assert limit.detail == "joint_velocity on joint 1: 7.2 exceeds bound 6.0"
    assert (limit.limit, limit.joint, limit.value, limit.bound) == ("joint_velocity", 1, 7.2, 6.0)
    assert timeout(4.0, 400, 0.01).detail == "deadline 0.01 s missed at t=4.0 s"


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: Termination("completed", -1.0, 0), "time_s must be finite and non-negative"),
        (lambda: Termination("completed", float("nan"), 0), "time_s must be finite"),
        (lambda: Termination("completed", 1.0, -1), "step must be non-negative"),
        (lambda: Termination("divergence", 1.0, 1, ""), "requires a non-empty detail"),
        (lambda: Termination("limit_violation", 1.0, 1, "x"), "requires limit, value, and bound"),
        (lambda: Termination("timeout", 1.0, 1, "x", limit="torque"), "only valid for limit_violation"),
        (
            lambda: Termination("limit_violation", 1.0, 1, "x", limit="torque", value=float("inf"), bound=1.0),
            "value must be finite",
        ),
        (
            lambda: Termination("limit_violation", 1.0, 1, "x", limit="torque", value=2.0, bound=1.0, joint=-1),
            "joint must be non-negative",
        ),
    ],
)
def test_invalid_terminations_are_rejected(build: object, message: str) -> None:
    """Timing, detail, and limit fields are validated per kind."""
    with pytest.raises(ValueError, match=message):
        build()  # type: ignore[operator]


def test_outcome_combines_completion_with_named_criteria() -> None:
    """Success requires every criterion; failed criteria are listed by name."""
    done = completed(5.0, 500)
    assert Outcome(done, {"completed": True, "final_dwell_in_tolerance": True}).success
    partial = Outcome(done, {"completed": True, "final_dwell_in_tolerance": False})
    assert not partial.success
    assert partial.failed_criteria == ("final_dwell_in_tolerance",)
    stopped = Outcome(divergence(1.0, 100, "blew up"), {"completed": False, "final_dwell_in_tolerance": False})
    assert not stopped.success
    assert stopped.failed_criteria == ("completed", "final_dwell_in_tolerance")


def test_outcome_consistency_is_enforced() -> None:
    """The completed criterion must exist and agree with the termination."""
    with pytest.raises(ValueError, match="must include 'completed'"):
        Outcome(completed(1.0, 1), {"dwell": True})
    with pytest.raises(ValueError, match=r"must equal termination\.is_completed"):
        Outcome(divergence(1.0, 1, "x"), {"completed": True})


def test_round_trip_through_plain_mappings() -> None:
    """Terminations and outcomes survive to_mapping/from_mapping with strict kinds."""
    outcome = Outcome(limit_violation(2.0, 200, "torque", 12.0, 10.0, joint=0), {"completed": False})
    mapping = to_mapping(outcome)
    assert from_mapping(mapping, Outcome) == outcome
    bad = dict(mapping)
    bad["termination"] = {**mapping["termination"], "kind": "exploded"}  # type: ignore[dict-item]
    with pytest.raises(ConfigError, match=r"termination\.kind: expected one of"):
        from_mapping(bad, Outcome)

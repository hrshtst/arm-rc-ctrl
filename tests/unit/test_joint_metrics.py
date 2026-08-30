# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-016: joint RMSE with per-joint wrapping matches hand calculations and rejects bad shapes."""

from __future__ import annotations

import math

import numpy as np
import pytest

from arm_rc_ctrl.metrics.joint import JointAnglePolicy, joint_error, joint_rmse, wrap_angle

LIMITED = JointAnglePolicy.limited(2)
MIXED = JointAnglePolicy((True, False))


def test_hand_calculated_aggregate_and_per_joint_rmse() -> None:
    """Errors [[0.1, -0.2], [0.3, 0.0]] give per-joint sqrt(0.05) and 0.1*sqrt(2), aggregate sqrt(0.035)."""
    q_ref = np.zeros((2, 2))
    q = np.array([[0.1, -0.2], [0.3, 0.0]])
    result = joint_rmse(q, q_ref, LIMITED)
    assert result.samples == 2
    assert result.per_joint[0] == pytest.approx(math.sqrt((0.01 + 0.09) / 2))
    assert result.per_joint[1] == pytest.approx(math.sqrt((0.04 + 0.0) / 2))
    assert result.aggregate == pytest.approx(math.sqrt((0.01 + 0.04 + 0.09 + 0.0) / 4))
    assert result.aggregate == pytest.approx(math.sqrt(np.mean(np.square(result.per_joint))))


def test_identical_trajectories_have_zero_error_and_metrics_are_nonnegative() -> None:
    """RMSE is zero for identical signals and non-negative otherwise."""
    q = np.random.default_rng(1).standard_normal((50, 2))
    zero = joint_rmse(q, q, LIMITED)
    assert zero.aggregate == 0.0
    assert zero.per_joint == (0.0, 0.0)
    nonzero = joint_rmse(q, q + 0.1, LIMITED)
    assert nonzero.aggregate == pytest.approx(0.1)
    assert all(v >= 0 for v in nonzero.per_joint)


def test_wrapping_applies_only_to_continuous_joints() -> None:
    """A 2*pi - 0.1 difference wraps to -0.1 on a continuous joint and stays raw on a limited joint."""
    delta = 2 * np.pi - 0.1
    q_ref = np.zeros((1, 2))
    q = np.array([[delta, delta]])
    error = joint_error(q, q_ref, MIXED)
    assert error[0, 0] == pytest.approx(-0.1)
    assert error[0, 1] == pytest.approx(delta)
    assert joint_rmse(q, q_ref, MIXED).per_joint[0] == pytest.approx(0.1)
    assert joint_rmse(q, q_ref, LIMITED).per_joint[0] == pytest.approx(delta)


def test_wrap_angle_maps_into_half_open_interval() -> None:
    """wrap_angle returns values in (-pi, pi], with pi mapped to pi and -pi to pi."""
    values = np.array([0.0, np.pi, -np.pi, 3 * np.pi, -3 * np.pi + 0.5, 2 * np.pi, 0.25])
    wrapped = wrap_angle(values)
    assert np.all(wrapped > -np.pi)
    assert np.all(wrapped <= np.pi)
    assert wrapped[1] == pytest.approx(np.pi)
    assert wrapped[2] == pytest.approx(np.pi)
    assert wrapped[3] == pytest.approx(np.pi)
    assert wrapped[4] == pytest.approx(-np.pi + 0.5)
    assert wrapped[5] == pytest.approx(0.0)
    assert wrapped[6] == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("q", "q_ref", "message"),
    [
        (np.zeros((3, 3)), np.zeros((3, 3)), r"q must have shape \(N, 2\)"),
        (np.zeros(3), np.zeros(3), r"q must have shape \(N, 2\)"),
        (np.zeros((3, 2)), np.zeros((4, 2)), "same shape"),
        (np.zeros((0, 2)), np.zeros((0, 2)), "at least one sample"),
        (np.array([[np.nan, 0.0]]), np.zeros((1, 2)), "must be finite"),
    ],
)
def test_shape_and_finiteness_are_rejected(q: np.ndarray, q_ref: np.ndarray, message: str) -> None:
    """Mismatched, 1-D, empty, or non-finite inputs fail."""
    with pytest.raises(ValueError, match=message):
        joint_rmse(q, q_ref, LIMITED)


def test_policy_requires_joints() -> None:
    """An empty policy is invalid."""
    with pytest.raises(ValueError, match="at least one joint"):
        JointAnglePolicy(())

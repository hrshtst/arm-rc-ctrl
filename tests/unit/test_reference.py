# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-021: the demonstration reference matches the dataset on the grid, between points, and at the boundaries."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.interpolate import make_interp_spline

from arm_rc_ctrl.controllers.reference import DemonstrationReference
from arm_rc_ctrl.data.synthetic import synthetic_samples


def _reference(interpolation: str = "linear") -> DemonstrationReference:
    samples = synthetic_samples(n=20, dof=2, task_dim=2, code_dim=0)
    return DemonstrationReference.from_samples(samples, interpolation)  # type: ignore[arg-type]


def test_grid_points_are_exact() -> None:
    """At every dataset time the reference returns the dataset values."""
    ref = _reference()
    q, dq, ddq = ref.sample_many(ref.t)
    assert np.array_equal(q, ref.q)
    assert np.array_equal(dq, ref.dq)
    assert np.array_equal(ddq, ref.ddq)
    for i in (0, 7, 19):
        qi, dqi, ddqi = ref.sample(float(ref.t[i]))
        assert np.array_equal(qi, ref.q[i])
        assert np.array_equal(dqi, ref.dq[i])
        assert np.array_equal(ddqi, ref.ddq[i])


def test_between_grid_points_follows_the_linear_policy() -> None:
    """Midway between samples the linear reference is the average of the neighbours."""
    ref = _reference("linear")
    mid = 0.5 * (ref.t[4] + ref.t[5])
    q, dq, ddq = ref.sample(float(mid))
    assert np.allclose(q, 0.5 * (ref.q[4] + ref.q[5]))
    assert np.allclose(dq, 0.5 * (ref.dq[4] + ref.dq[5]))
    assert np.allclose(ddq, 0.5 * (ref.ddq[4] + ref.ddq[5]))


def test_between_grid_points_follows_the_cubic_policy() -> None:
    """The cubic reference equals the not-a-knot spline through the samples."""
    ref = _reference("cubic")
    times: NDArray[np.float64] = ref.t[:-1] + 0.3 * np.diff(ref.t)
    q, _, _ = ref.sample_many(times)
    expected = make_interp_spline(ref.t, ref.q, k=3, axis=0)(times)
    assert np.allclose(q, expected, atol=1e-12)
    linear = _reference("linear").sample_many(times)[0]
    assert np.max(np.abs(q - linear)) > 1e-9  # the policies differ between grid points


def test_boundaries_hold_the_posture_with_zero_motion() -> None:
    """Before the start and after the end, q holds the boundary sample and dq/ddq are zero."""
    ref = _reference()
    for t, index in ((-1.0, 0), (float(ref.t[0]) - 1e-6, 0), (ref.duration + 1e-6, -1), (ref.duration + 5.0, -1)):
        q, dq, ddq = ref.sample(t)
        assert np.array_equal(q, ref.q[index])
        assert np.array_equal(dq, np.zeros(2))
        assert np.array_equal(ddq, np.zeros(2))
    q_end, dq_end, _ = ref.sample(ref.duration)
    assert np.array_equal(q_end, ref.q[-1])
    assert np.array_equal(dq_end, ref.dq[-1])  # exactly at the end the recorded derivative applies


def test_vectorized_and_scalar_sampling_agree() -> None:
    """sample_many at several times equals sample at each of them."""
    ref = _reference()
    times = np.array([-0.5, 0.0, 0.037, 0.1, 0.19, 0.4])
    q, dq, ddq = ref.sample_many(times)
    for i, t in enumerate(times):
        qi, dqi, ddqi = ref.sample(float(t))
        assert np.array_equal(q[i], qi)
        assert np.array_equal(dq[i], dqi)
        assert np.array_equal(ddq[i], ddqi)
    assert (ref.dof, ref.duration) == (2, pytest.approx(0.19))


def test_arrays_are_read_only_and_validated() -> None:
    """Stored arrays cannot be mutated; bad grids and shapes are rejected."""
    ref = _reference()
    with pytest.raises(ValueError, match="read-only"):
        ref.q[0, 0] = 1.0
    t = np.arange(5, dtype=np.float64) * 0.01
    ok = np.zeros((5, 2))
    with pytest.raises(ValueError, match="strictly increasing"):
        DemonstrationReference(np.array([0.0, 0.01, 0.01, 0.03, 0.04]), ok, ok, ok)
    with pytest.raises(ValueError, match=r"dq must have shape \(5, dof\)"):
        DemonstrationReference(t, ok, np.zeros((4, 2)), ok)
    with pytest.raises(ValueError, match="share one shape"):
        DemonstrationReference(t, ok, ok, np.zeros((5, 3)))
    with pytest.raises(ValueError, match="q must be finite"):
        DemonstrationReference(t, np.full((5, 2), np.nan), ok, ok)
    with pytest.raises(ValueError, match="cubic interpolation needs at least 4"):
        DemonstrationReference(t[:3], ok[:3], ok[:3], ok[:3], "cubic")
    with pytest.raises(ValueError, match="times must be finite"):
        ref.sample(float("nan"))

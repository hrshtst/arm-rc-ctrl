# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Deterministic synthetic sample sets for tests and fixtures (no randomness)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from skelarm import StateLog

from arm_rc_ctrl.data.samples import PHASE_DWELL, PHASE_MOVE, PHASE_PRIME, SampleSet

__all__ = ["synthetic_arrays", "synthetic_demonstration_log", "synthetic_samples"]


def synthetic_arrays(
    n: int = 6, dof: int = 2, task_dim: int = 2, code_dim: int = 0, period_s: float = 0.01
) -> dict[str, NDArray[Any]]:
    """Schema-conforming arrays from analytic signals: first sample prime, last sample dwell, rest move."""
    t: NDArray[Any] = np.arange(n, dtype=np.float64) * period_s
    q: NDArray[Any] = np.stack([np.sin(t + j) for j in range(dof)], axis=1) if dof else np.zeros((n, 0))
    tip: NDArray[Any] = np.stack([np.cos(t + k) for k in range(task_dim)], axis=1) if task_dim else np.zeros((n, 0))
    phase: NDArray[Any] = np.full(n, PHASE_MOVE, dtype=np.int64)
    if n:
        phase[0] = PHASE_PRIME
        phase[-1] = PHASE_DWELL
    arrays: dict[str, NDArray[Any]] = {
        "t": t,
        "q": q,
        "dq": np.gradient(q, axis=0) if n > 1 else np.zeros_like(q),
        "ddq": np.zeros_like(q),
        "tip": tip,
        "dtip": np.gradient(tip, axis=0) if n > 1 else np.zeros_like(tip),
        "ddtip": np.zeros_like(tip),
        "task_code": np.zeros((n, code_dim), dtype=np.float64),
        "phase": phase,
    }
    return arrays


def synthetic_samples(n: int = 6, dof: int = 2, task_dim: int = 2, code_dim: int = 0) -> SampleSet:
    """A valid :class:`SampleSet` built from :func:`synthetic_arrays`."""
    return SampleSet.from_arrays(synthetic_arrays(n, dof, task_dim, code_dim))


def synthetic_demonstration_log(*, dt: float = 0.01, duration: float = 0.3) -> StateLog:
    """A short, deterministic planar 2-DOF PD reach recorded as a ``skelarm`` log.

    Used to build the committed raw fixture; the log embeds the skeleton and
    the ``q``, ``dq``, ``tau``, ``q_ref``, and ``error`` channels.
    """
    from skelarm import JointPD, LinkProp, SampledJointReference, Skeleton, simulate_controlled

    props = [
        LinkProp(length=0.30, m=1.0, i=0.0075, rgx=0.15, rgy=0.0, qmin=-3.0, qmax=3.0),
        LinkProp(length=0.25, m=0.6, i=0.0031, rgx=0.125, rgy=0.0, qmin=-3.0, qmax=3.0),
    ]
    skeleton = Skeleton(props)
    skeleton.q = np.array([0.3, 0.6])
    skeleton.dq = np.zeros(2)
    target = np.array([0.8, 0.4])
    zeros = np.zeros(2)
    reference = SampledJointReference([0.0, duration], [target, target], [zeros, zeros], [zeros, zeros])
    controller = JointPD(reference, kp=np.array([20.0, 10.0]), kd=np.array([2.0, 1.0]))
    return simulate_controlled(skeleton, controller, duration=duration, dt=dt)

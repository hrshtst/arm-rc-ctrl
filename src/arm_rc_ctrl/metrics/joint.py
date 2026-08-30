# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

r"""Joint-space trajectory error (``docs/PLAN.md`` section 9.1).

.. math::

    \\mathrm{RMSE}_q = \\sqrt{\\frac{1}{N d} \\sum_{k=1}^{N} \\| \\operatorname{wrap}(q_k - q_k^{demo}) \\|_2^2}

Wrapping to ``(-pi, pi]`` is applied only to joints declared *continuous* by the
:class:`JointAnglePolicy`; limited revolute joints keep their raw difference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["JointAnglePolicy", "JointRmse", "joint_error", "joint_rmse", "wrap_angle"]


def wrap_angle(delta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Wrap angular differences into ``(-pi, pi]``."""
    wrapped = np.mod(-delta + np.pi, 2.0 * np.pi)
    return np.ascontiguousarray(np.pi - wrapped, dtype=np.float64)


@dataclass(frozen=True)
class JointAnglePolicy:
    """Which joints are continuous (wrapped) versus limited (raw difference)."""

    continuous: tuple[bool, ...]

    def __post_init__(self) -> None:
        """Require at least one joint."""
        if not self.continuous:
            msg = "policy must cover at least one joint"
            raise ValueError(msg)

    @property
    def dof(self) -> int:
        """Number of joints covered."""
        return len(self.continuous)

    @classmethod
    def limited(cls, dof: int) -> JointAnglePolicy:
        """All joints limited (no wrapping) — the planar arm's convention."""
        return cls((False,) * dof)


@dataclass(frozen=True)
class JointRmse:
    """Aggregate and per-joint RMSE in radians."""

    aggregate: float
    per_joint: tuple[float, ...]
    samples: int


def _check(
    q: NDArray[np.float64], q_ref: NDArray[np.float64], policy: JointAnglePolicy
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    a = np.asarray(q, dtype=np.float64)
    b = np.asarray(q_ref, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != policy.dof:  # noqa: PLR2004
        msg = f"q must have shape (N, {policy.dof}), got {a.shape}"
        raise ValueError(msg)
    if b.shape != a.shape:
        msg = f"q_ref must have the same shape as q {a.shape}, got {b.shape}"
        raise ValueError(msg)
    if a.shape[0] == 0:
        msg = "at least one sample is required"
        raise ValueError(msg)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        msg = "q and q_ref must be finite"
        raise ValueError(msg)
    return a, b


def joint_error(q: NDArray[np.float64], q_ref: NDArray[np.float64], policy: JointAnglePolicy) -> NDArray[np.float64]:
    """Per-sample, per-joint error ``q - q_ref`` with wrapping on continuous joints only."""
    a, b = _check(q, q_ref, policy)
    error = a - b
    mask = np.asarray(policy.continuous, dtype=bool)
    if mask.any():
        error[:, mask] = wrap_angle(error[:, mask])
    return np.ascontiguousarray(error, dtype=np.float64)


def joint_rmse(q: NDArray[np.float64], q_ref: NDArray[np.float64], policy: JointAnglePolicy) -> JointRmse:
    """Aggregate and per-joint RMSE of the (wrapped) joint error."""
    error = joint_error(q, q_ref, policy)
    squared = error * error
    per_joint = np.sqrt(np.mean(squared, axis=0))
    aggregate = float(np.sqrt(np.mean(squared)))
    return JointRmse(aggregate=aggregate, per_joint=tuple(float(v) for v in per_joint), samples=int(error.shape[0]))

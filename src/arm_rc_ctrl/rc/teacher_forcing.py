# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Teacher-forcing input/target pairs for the ESN target generator (docs/PLAN.md sections 5.1 and 5.3).

The input at sample ``k`` is ``u_k = [q̄_k, dq̄_k, c_k]``: the measured joint
state normalized with the dataset's training-only statistics, followed by the
task code. The target is ``y_k = q_(k+1)``, the absolute next demonstrated
joint position in radians. Row ``k`` of an episode pairs sample ``k`` with
sample ``k + 1``, so an episode of ``N`` samples yields ``N - 1`` rows. Rows
whose input sample lies in the prime (initial-hold) interval are *washout*:
they drive the reservoir but are excluded from the ridge loss, exactly as the
runtime priming interval drives the reservoir before generation starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from arm_rc_ctrl.data.normalization import Normalizer
from arm_rc_ctrl.data.records import Normalization
from arm_rc_ctrl.data.samples import PHASE_CODES, SampleSet

__all__ = ["INPUT_CHANNELS", "Episode", "InputEncoder", "build_episode"]

INPUT_CHANNELS: Final[tuple[str, ...]] = ("q", "dq")
"""Joint-state channels that form the normalized part of the ESN input, in order."""


def _finite_2d(array: NDArray[np.float64], name: str, rows: int | None = None) -> NDArray[np.float64]:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim != 2 or (rows is not None and values.shape[0] != rows):  # noqa: PLR2004
        expected = f"({rows}, k)" if rows is not None else "(N, k)"
        msg = f"{name} must have shape {expected}, got {values.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(values)):
        msg = f"{name} must be finite"
        raise ValueError(msg)
    return values


@dataclass(frozen=True)
class InputEncoder:
    """Maps measured joint state (and task code) to the ESN input; shared by training and runtime."""

    normalizer: Normalizer
    dof: int
    task_code_dim: int

    def __post_init__(self) -> None:
        """The statistics must cover exactly the input channels at the joint count."""
        if self.dof < 1 or self.task_code_dim < 0:
            msg = f"dof must be >= 1 and task_code_dim >= 0, got {self.dof} and {self.task_code_dim}"
            raise ValueError(msg)
        channels = self.normalizer.normalization.channels
        for name in INPUT_CHANNELS:
            stats = channels.get(name)
            if stats is None:
                msg = f"normalization lacks statistics for input channel {name!r}"
                raise ValueError(msg)
            if len(stats.mean) != self.dof:
                msg = f"normalization of {name!r} covers {len(stats.mean)} joints, expected {self.dof}"
                raise ValueError(msg)

    @classmethod
    def from_normalization(cls, normalization: Normalization, *, dof: int, task_code_dim: int) -> InputEncoder:
        """Build the encoder from recorded statistics."""
        return cls(Normalizer(normalization), dof, task_code_dim)

    @property
    def input_dim(self) -> int:
        """Width of the ESN input vector."""
        return len(INPUT_CHANNELS) * self.dof + self.task_code_dim

    def encode_many(
        self, q: NDArray[np.float64], dq: NDArray[np.float64], task_code: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        """Encode ``N`` samples into an ``(N, input_dim)`` array."""
        q = _finite_2d(q, "q")
        n = q.shape[0]
        dq = _finite_2d(dq, "dq", n)
        if q.shape[1] != self.dof or dq.shape[1] != self.dof:
            msg = f"q and dq must have {self.dof} joints, got {q.shape[1]} and {dq.shape[1]}"
            raise ValueError(msg)
        code = (
            np.zeros((n, self.task_code_dim), dtype=np.float64)
            if task_code is None
            else _finite_2d(task_code, "task_code", n)
        )
        if code.shape[1] != self.task_code_dim:
            msg = f"task_code must have {self.task_code_dim} columns, got {code.shape[1]}"
            raise ValueError(msg)
        columns = [self.normalizer.transform("q", q), self.normalizer.transform("dq", dq), code]
        return np.ascontiguousarray(np.hstack(columns), dtype=np.float64)

    def encode(
        self, q: NDArray[np.float64], dq: NDArray[np.float64], task_code: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        """Encode one sample into a one-dimensional ``input_dim`` vector."""
        code = None if task_code is None else np.asarray(task_code, dtype=np.float64)[None, :]
        return self.encode_many(
            np.asarray(q, dtype=np.float64)[None, :], np.asarray(dq, dtype=np.float64)[None, :], code
        )[0]


@dataclass(frozen=True)
class Episode:
    """One teacher-forced episode: aligned input/target rows and the washout prefix."""

    source: str
    """Artifact ID of the dataset the episode was built from."""
    t: NDArray[np.float64]
    """Time of each input row (s)."""
    inputs: NDArray[np.float64]
    """``(M, input_dim)`` normalized inputs ``u_k``."""
    targets: NDArray[np.float64]
    """``(M, dof)`` next demonstrated joint positions ``q_(k+1)`` (rad)."""
    loss_rows: NDArray[np.bool_]
    """``(M,)`` mask of rows that contribute to the ridge loss; the leading ``False`` block is the washout."""

    def __post_init__(self) -> None:
        """Validate alignment, finiteness, and that washout rows form a leading block; freeze the arrays."""
        if not self.source.strip():
            msg = "source must name the dataset artifact"
            raise ValueError(msg)
        inputs = _finite_2d(self.inputs, "inputs")
        m = inputs.shape[0]
        targets = _finite_2d(self.targets, "targets", m)
        t = np.asarray(self.t, dtype=np.float64)
        loss = np.asarray(self.loss_rows)
        if t.shape != (m,) or loss.shape != (m,) or loss.dtype != np.bool_:
            msg = f"t and loss_rows must be ({m},) with loss_rows boolean, got {t.shape} and {loss.shape} {loss.dtype}"
            raise ValueError(msg)
        if m < 1 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
            msg = "t must be finite and strictly increasing over at least one row"
            raise ValueError(msg)
        washout = int(np.argmax(loss)) if loss.any() else m
        if not loss.any() or not loss[washout:].all():
            msg = "loss_rows must be a leading washout block of False followed only by True rows (at least one)"
            raise ValueError(msg)
        for name, array in (("t", t), ("inputs", inputs), ("targets", targets), ("loss_rows", loss)):
            frozen = np.ascontiguousarray(array).copy()
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)

    @property
    def n_rows(self) -> int:
        """Number of input/target rows."""
        return int(self.inputs.shape[0])

    @property
    def input_dim(self) -> int:
        """Width of the input rows."""
        return int(self.inputs.shape[1])

    @property
    def dof(self) -> int:
        """Number of joints in the targets."""
        return int(self.targets.shape[1])

    @property
    def washout_len(self) -> int:
        """Number of leading rows excluded from the loss (``rclib``'s ``washout_len``)."""
        return int(np.argmax(self.loss_rows))


def build_episode(samples: SampleSet, encoder: InputEncoder, *, source: str) -> Episode:
    """Pair every sample with its successor: inputs from sample ``k``, target ``q`` of sample ``k + 1``.

    Washout rows are those whose input sample carries the prime phase code; the
    phase annotation guarantees the prime interval leads the episode, and an
    episode must retain at least one loss row after it.
    """
    if samples.dof != encoder.dof or samples.task_code_dim != encoder.task_code_dim:
        msg = (
            f"dataset has dof {samples.dof} and task_code_dim {samples.task_code_dim}; "
            f"the encoder expects {encoder.dof} and {encoder.task_code_dim}"
        )
        raise ValueError(msg)
    inputs = encoder.encode_many(samples.q[:-1], samples.dq[:-1], samples.task_code[:-1])
    targets = samples.q[1:]
    loss_rows = samples.phase[:-1] != PHASE_CODES["prime"]
    return Episode(source=source, t=samples.t[:-1], inputs=inputs, targets=targets, loss_rows=loss_rows)

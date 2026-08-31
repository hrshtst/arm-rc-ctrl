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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal, cast

import numpy as np
from numpy.typing import NDArray

from arm_rc_ctrl.data.records import Normalization, is_artifact_id
from arm_rc_ctrl.data.samples import PHASE_CODES, SampleSet

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "INPUT_CHANNELS",
    "TRANSFORM_POLICIES",
    "ChannelTransform",
    "Episode",
    "InputEncoder",
    "InputTransform",
    "TransformPolicy",
    "build_episode",
]

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


type TransformPolicy = Literal["training_std", "fixed_scale"]
TRANSFORM_POLICIES: Final[tuple[str, ...]] = ("training_std", "fixed_scale")


@dataclass(frozen=True)
class ChannelTransform:
    """Per-column affine map ``(value - center) / scale`` of one input channel."""

    center: tuple[float, ...]
    scale: tuple[float, ...]

    def __post_init__(self) -> None:
        """Centers and scales are finite, equally long, and scales are positive."""
        if not self.center or len(self.center) != len(self.scale):
            msg = f"center and scale must have the same non-zero length, got {len(self.center)} and {len(self.scale)}"
            raise ValueError(msg)
        values = (*self.center, *self.scale)
        if any(not np.isfinite(v) for v in values) or any(s <= 0 for s in self.scale):
            msg = "center must be finite and scale positive and finite"
            raise ValueError(msg)


@dataclass(frozen=True)
class InputTransform:
    """The recipe-level input transform ``ū = (u - center) / scale`` per channel (docs/PLAN.md section 5.1).

    ``training_std`` reproduces the dataset's training-only statistics (centers
    and scales from ``Normalization``); ``fixed_scale`` keeps the training means
    as centers but replaces every scale of a channel by one shared physical
    value, which keeps a barely moving joint from amplifying tracking jitter.
    ``derived_from`` names the dataset whose statistics supplied the centers.
    """

    policy: TransformPolicy
    derived_from: tuple[str, ...]
    channels: dict[str, ChannelTransform]
    fixed_scales: dict[str, float] = field(default_factory=dict)
    """Shared physical scale per channel under ``fixed_scale``; empty under ``training_std``."""

    def __post_init__(self) -> None:
        """Exactly the input channels are transformed; fixed scales match the policy."""
        if self.policy not in TRANSFORM_POLICIES:
            msg = f"policy must be one of {list(TRANSFORM_POLICIES)}, got {self.policy!r}"
            raise ValueError(msg)
        if tuple(self.channels) != INPUT_CHANNELS:
            msg = f"channels must be exactly {INPUT_CHANNELS} in order, got {tuple(self.channels)}"
            raise ValueError(msg)
        if not self.derived_from or any(not is_artifact_id(a) for a in self.derived_from):
            msg = f"derived_from must list the artifact IDs the centers came from, got {self.derived_from}"
            raise ValueError(msg)
        if self.policy == "fixed_scale":
            if tuple(self.fixed_scales) != INPUT_CHANNELS:
                msg = f"fixed_scale needs fixed_scales for exactly {INPUT_CHANNELS}"
                raise ValueError(msg)
            for name, value in self.fixed_scales.items():
                if not (np.isfinite(value) and value > 0):
                    msg = f"fixed_scales[{name!r}] must be positive and finite, got {value!r}"
                    raise ValueError(msg)
                if any(scale != value for scale in self.channels[name].scale):
                    msg = f"channel {name!r} scales must all equal fixed_scales[{name!r}] = {value}"
                    raise ValueError(msg)
        elif self.fixed_scales:
            msg = "fixed_scales is only meaningful for the fixed_scale policy"
            raise ValueError(msg)

    @property
    def dof(self) -> int:
        """Number of joints covered by every channel."""
        return len(self.channels[INPUT_CHANNELS[0]].center)

    @classmethod
    def derive(
        cls, policy: TransformPolicy, normalization: Normalization, *, fixed_scales: Mapping[str, float] | None = None
    ) -> InputTransform:
        """Derive the transform for ``policy`` from a dataset's recorded normalization statistics."""
        channels: dict[str, ChannelTransform] = {}
        for name in INPUT_CHANNELS:
            stats = normalization.channels.get(name)
            if stats is None:
                msg = f"normalization lacks statistics for input channel {name!r}"
                raise ValueError(msg)
            if policy == "fixed_scale":
                if not fixed_scales or name not in fixed_scales:
                    msg = f"fixed_scale needs a scale for channel {name!r}"
                    raise ValueError(msg)
                value = float(fixed_scales[name])
                if not (np.isfinite(value) and value > 0):
                    msg = f"fixed_scales[{name!r}] must be positive and finite, got {value!r}"
                    raise ValueError(msg)
                scale = tuple(value for _ in stats.mean)
            else:
                scale = tuple(stats.scale)
            channels[name] = ChannelTransform(tuple(stats.mean), scale)
        fixed: dict[str, float] = {}
        if policy == "fixed_scale":
            fixed = {n: float(cast("Mapping[str, float]", fixed_scales)[n]) for n in INPUT_CHANNELS}
        return cls(policy, normalization.fitted_on, channels, fixed)

    def _params(self, name: str, values: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        channel = self.channels[name]
        if values.shape[-1] != len(channel.center):
            msg = f"channel {name!r} expects {len(channel.center)} columns, got shape {values.shape}"
            raise ValueError(msg)
        return np.asarray(channel.center, dtype=np.float64), np.asarray(channel.scale, dtype=np.float64)

    def transform(self, name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """``(values - center) / scale`` for channel ``name``."""
        center, scale = self._params(name, values)
        return np.ascontiguousarray((values - center) / scale, dtype=np.float64)

    def inverse(self, name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """``values * scale + center`` for channel ``name``."""
        center, scale = self._params(name, values)
        return np.ascontiguousarray(values * scale + center, dtype=np.float64)


@dataclass(frozen=True)
class InputEncoder:
    """Maps measured joint state (and task code) to the ESN input; shared by training and runtime."""

    transform: InputTransform
    dof: int
    task_code_dim: int

    def __post_init__(self) -> None:
        """The transform must cover the input channels at the joint count."""
        if self.dof < 1 or self.task_code_dim < 0:
            msg = f"dof must be >= 1 and task_code_dim >= 0, got {self.dof} and {self.task_code_dim}"
            raise ValueError(msg)
        for name in INPUT_CHANNELS:
            width = len(self.transform.channels[name].center)
            if width != self.dof:
                msg = f"transform of {name!r} covers {width} joints, expected {self.dof}"
                raise ValueError(msg)

    @classmethod
    def from_normalization(cls, normalization: Normalization, *, dof: int, task_code_dim: int) -> InputEncoder:
        """The ``training_std`` encoder of a dataset's recorded statistics."""
        return cls(InputTransform.derive("training_std", normalization), dof, task_code_dim)

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
        columns = [self.transform.transform("q", q), self.transform.transform("dq", dq), code]
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

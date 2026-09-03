# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Per-episode reservoir reset and the configurable common warm-up (M3R-006; recovery plan section 4.2, D2).

Every training episode and evaluation run independently resets the reservoir
to its deterministic all-zero state. A configured warm-up ``T_w`` from the
approved D2 set (0, 0.25, 0.5, 1.0, or 2.0 s; anchor 1.0 s) then precedes task
time zero:

- **Training** (:func:`build_task_episode`): the warm-up repeats the episode's
  encoded initial state ``[q_0, 0]`` for ``T_w / dt`` rows before the
  teacher-forcing rows; the rows live on ``[-T_w, 0)`` and never enter the
  ridge loss. ``T_w = 0`` is the explicit no-warm-up case: it adds no rows and
  drops none.
- **Evaluation**: the closed-loop adapter holds the initial posture and primes
  the generator with the *measured* ``[q, dq]`` for the same duration
  (``hold_until_s = T_w``) while the readout stays inactive; with ``T_w = 0``
  generation starts at the very first control sample. The shared activation
  contract is that an ideal hold (measured state exactly ``[q_0, 0]``) leaves
  the reservoir bitwise in the training warm-up state
  (:func:`warmup_state`; see ``tests/unit/test_warmup.py``).

No reservoir state ever crosses an episode boundary: harvesting and priming
always begin from :meth:`~arm_rc_ctrl.rc.esn.EsnModel.reset`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from arm_rc_ctrl.data.samples import PHASE_PRIME
from arm_rc_ctrl.rc.teacher_forcing import Episode
from arm_rc_ctrl.rc.training import prime

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from arm_rc_ctrl.data.samples import SampleSet
    from arm_rc_ctrl.rc.esn import EsnModel
    from arm_rc_ctrl.rc.teacher_forcing import InputEncoder

__all__ = [
    "APPROVED_WARMUPS_S",
    "WarmupConfig",
    "build_task_episode",
    "build_task_episode_arrays",
    "warmup_inputs",
    "warmup_state",
]

APPROVED_WARMUPS_S: Final[frozenset[float]] = frozenset({0.0, 0.25, 0.5, 1.0, 2.0})
"""Approved warm-up durations in seconds (D2); anchor 1.0, with 0 as the named no-warm-up case."""

_GRID_TOLERANCE_S: Final = 1e-9


@dataclass(frozen=True)
class WarmupConfig:
    """One warm-up duration from the approved D2 set."""

    duration_s: float

    def __post_init__(self) -> None:
        """Reject durations outside the approved protocol set."""
        if self.duration_s not in APPROVED_WARMUPS_S:
            msg = f"duration_s must be one of the approved values {sorted(APPROVED_WARMUPS_S)}, got {self.duration_s!r}"
            raise ValueError(msg)

    def n_rows(self, period_s: float) -> int:
        """Warm-up samples consumed at the control period (zero for the no-warm-up case).

        Raises
        ------
        ValueError
            If the period is invalid or the duration does not land on the control grid.
        """
        if not (period_s > 0 and period_s < float("inf")):
            msg = f"period_s must be positive and finite, got {period_s!r}"
            raise ValueError(msg)
        rows = round(self.duration_s / period_s)
        if abs(rows * period_s - self.duration_s) > _GRID_TOLERANCE_S:
            msg = f"warm-up duration {self.duration_s!r} s does not lie on the control grid (period {period_s!r} s)"
            raise ValueError(msg)
        return rows

    def times(self, period_s: float) -> NDArray[np.float64]:
        """The warm-up timestamps ``[-T_w, ..., -period]`` (empty for the no-warm-up case)."""
        rows = self.n_rows(period_s)
        return np.arange(-rows, 0, dtype=np.float64) * period_s


def warmup_inputs(encoder: InputEncoder, q0: NDArray[np.float64], n_rows: int) -> NDArray[np.float64]:
    """The encoded warm-up input ``[q_0, 0]`` repeated ``n_rows`` times (``(n_rows, input_dim)``)."""
    if n_rows < 0:
        msg = f"n_rows must be non-negative, got {n_rows}"
        raise ValueError(msg)
    hold = np.tile(np.asarray(q0, dtype=np.float64), (n_rows, 1))
    zeros = np.zeros_like(hold)
    return encoder.encode_many(hold, zeros)


def build_task_episode_arrays(
    t: NDArray[np.float64],
    q: NDArray[np.float64],
    dq: NDArray[np.float64],
    task_code: NDArray[np.float64],
    encoder: InputEncoder,
    *,
    source: str,
    warmup: WarmupConfig,
    period_s: float,
) -> Episode:
    """The warm-up-prefixed episode of raw task-clock arrays (datasets and synthetic episodes alike).

    The warm-up rows repeat the encoded ``[q_0, 0]`` on ``[-T_w, 0)`` and are
    excluded from the loss; every task row enters the loss. With ``T_w = 0``
    the episode is the pure task pairing.
    """
    times = np.asarray(t, dtype=np.float64)
    positions = np.asarray(q, dtype=np.float64)
    velocities = np.asarray(dq, dtype=np.float64)
    codes = np.asarray(task_code, dtype=np.float64)
    if float(times[0]) != 0.0:
        msg = f"a task episode must start at 0.0 s on the task clock, got {float(times[0])!r}"
        raise ValueError(msg)
    n_warm = warmup.n_rows(period_s)
    q0 = np.asarray(positions[0], dtype=np.float64)
    warm_in = warmup_inputs(encoder, q0, n_warm)
    warm_targets = np.tile(q0, (n_warm, 1))
    task_in = encoder.encode_many(positions[:-1], velocities[:-1], codes[:-1])
    task_targets = positions[1:]
    inputs = np.vstack([warm_in, task_in])
    targets = np.vstack([warm_targets, task_targets])
    grid = np.concatenate([warmup.times(period_s), times[:-1]])
    loss_rows = np.concatenate([np.zeros(n_warm, dtype=np.bool_), np.ones(task_in.shape[0], dtype=np.bool_)])
    return Episode(source=source, t=grid, inputs=inputs, targets=targets, loss_rows=loss_rows)


def build_task_episode(
    samples: SampleSet, encoder: InputEncoder, *, source: str, warmup: WarmupConfig, period_s: float
) -> Episode:
    """Pair a move/dwell-only task episode with its configured warm-up prefix.

    Raises
    ------
    ValueError
        If the samples contain prime phases, do not start at task time zero,
        or do not match the encoder.
    """
    if bool(np.any(samples.phase == PHASE_PRIME)):
        msg = "a task episode is move/dwell only; prime samples are not allowed (crop the pre-roll first)"
        raise ValueError(msg)
    if samples.dof != encoder.dof or samples.task_code_dim != encoder.task_code_dim:
        msg = (
            f"dataset has dof {samples.dof} and task_code_dim {samples.task_code_dim}; "
            f"the encoder expects {encoder.dof} and {encoder.task_code_dim}"
        )
        raise ValueError(msg)
    return build_task_episode_arrays(
        samples.t, samples.q, samples.dq, samples.task_code, encoder, source=source, warmup=warmup, period_s=period_s
    )


def warmup_state(
    model: EsnModel, encoder: InputEncoder, q0: NDArray[np.float64], warmup: WarmupConfig, period_s: float
) -> NDArray[np.float64]:
    """The reservoir state at task activation after a reset and the configured warm-up.

    ``T_w = 0`` returns the all-zero reset state. This is the state an ideal
    evaluation hold (measured state exactly ``[q_0, 0]``) reproduces bitwise.
    """
    rows = warmup.n_rows(period_s)
    model.reset()
    if rows == 0:
        return model.state()
    return prime(model, warmup_inputs(encoder, q0, rows))

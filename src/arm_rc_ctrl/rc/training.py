# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Per-episode reservoir driving, priming, and offline readout training (docs/PLAN.md section 5.3).

Every episode starts from a reset reservoir. Its washout rows (the initial
hold) drive the reservoir but are excluded from the ridge loss; at runtime the
same inputs are replayed by :func:`prime` before generation starts, so the
reservoir enters the movement in exactly the state it saw during training.
Episodes are never concatenated without a reset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arm_rc_ctrl.rc.esn import EsnModel
    from arm_rc_ctrl.rc.teacher_forcing import Episode

__all__ = [
    "EpisodeStates",
    "FitReport",
    "TrainingBatch",
    "harvest_episode",
    "harvest_states",
    "one_step_rmse",
    "predict_episode",
    "prime",
    "train_readout",
    "training_rows",
]


def harvest_states(model: EsnModel, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Reset the reservoir and drive it with ``inputs`` row by row; row ``k`` is the state after input ``k``."""
    rows = np.asarray(inputs, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] < 1:  # noqa: PLR2004
        msg = f"inputs must have shape (M >= 1, input_dim), got {rows.shape}"
        raise ValueError(msg)
    model.reset()
    return np.vstack([model.advance(row) for row in rows])


def prime(model: EsnModel, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Reset and replay the priming inputs; return the reservoir state generation starts from.

    Replaying an episode's washout rows leaves the reservoir in the state it
    had after those rows during training (see ``tests/unit/test_training.py``).
    """
    return harvest_states(model, inputs)[-1]


@dataclass(frozen=True)
class EpisodeStates:
    """Reservoir states of one episode with the rows that enter the loss."""

    source: str
    states: NDArray[np.float64]
    """``(M, n_neurons)`` state after each input row."""
    targets: NDArray[np.float64]
    """``(M, dof)`` targets aligned with the states."""
    loss_rows: NDArray[np.bool_]

    @property
    def training_states(self) -> NDArray[np.float64]:
        """States of the loss rows only."""
        return self.states[self.loss_rows]

    @property
    def training_targets(self) -> NDArray[np.float64]:
        """Targets of the loss rows only."""
        return self.targets[self.loss_rows]


def harvest_episode(model: EsnModel, episode: Episode) -> EpisodeStates:
    """Drive the reservoir through one episode from a reset state."""
    if episode.input_dim != model.input_dim or episode.dof != model.output_dim:
        msg = (
            f"episode {episode.source} has input_dim {episode.input_dim} and dof {episode.dof}; "
            f"the model expects {model.input_dim} and {model.output_dim}"
        )
        raise ValueError(msg)
    states = harvest_states(model, episode.inputs)
    return EpisodeStates(episode.source, states, np.asarray(episode.targets), np.asarray(episode.loss_rows))


@dataclass(frozen=True)
class TrainingBatch:
    """The stacked loss rows of every episode, with the rows each episode contributed."""

    states: NDArray[np.float64]
    """``(sum L_i, n_neurons)`` loss-row states, episode blocks in training order."""
    targets: NDArray[np.float64]
    """``(sum L_i, dof)`` targets aligned with the states."""
    loss_rows: tuple[int, ...]
    """``L_i`` per episode, in training order; the stacked blocks follow this order."""

    def row_weights(self, reference_rows: int) -> tuple[NDArray[np.float64], tuple[float, ...]]:
        """Equal-episode weights: ``w_i = reference_rows / L_i`` per episode and the per-row vector they expand to.

        Every episode then carries the same total loss weight regardless of
        its length (manual plan section 4): a take twice as long contributes
        rows of half the weight, not twice the influence.
        """
        if reference_rows < 1:
            msg = f"weight_reference_rows must be >= 1, got {reference_rows}"
            raise ValueError(msg)
        per_episode = tuple(reference_rows / rows for rows in self.loss_rows)
        counts = np.asarray(self.loss_rows, dtype=np.int64)
        rows = np.repeat(np.asarray(per_episode, dtype=np.float64), counts)
        return rows, per_episode


def training_rows(model: EsnModel, episodes: Sequence[Episode]) -> TrainingBatch:
    """Harvest every episode (each from a reset) and stack the loss rows into one ridge problem."""
    if not episodes:
        msg = "at least one episode is required"
        raise ValueError(msg)
    harvested = [harvest_episode(model, episode) for episode in episodes]
    states = np.vstack([h.training_states for h in harvested])
    targets = np.vstack([h.training_targets for h in harvested])
    return TrainingBatch(states, targets, tuple(int(h.training_states.shape[0]) for h in harvested))


def predict_episode(model: EsnModel, episode: Episode) -> NDArray[np.float64]:
    """Teacher-forced one-step prediction of every row of ``episode`` from a reset reservoir."""
    if episode.input_dim != model.input_dim or episode.dof != model.output_dim:
        msg = (
            f"episode {episode.source} has input_dim {episode.input_dim} and dof {episode.dof}; "
            f"the model expects {model.input_dim} and {model.output_dim}"
        )
        raise ValueError(msg)
    model.reset()
    return np.vstack([model.step(row) for row in episode.inputs])


def one_step_rmse(
    prediction: NDArray[np.float64], targets: NDArray[np.float64], rows: NDArray[np.bool_]
) -> tuple[tuple[float, ...], float]:
    """Per-joint and aggregate RMSE of ``prediction`` against ``targets`` over the selected ``rows``."""
    if prediction.shape != targets.shape or rows.shape != (targets.shape[0],) or not rows.any():
        msg = f"prediction {prediction.shape}, targets {targets.shape}, rows {rows.shape} (need at least one selected)"
        raise ValueError(msg)
    error = prediction[rows] - targets[rows]
    per_joint = np.sqrt(np.mean(error**2, axis=0))
    return tuple(float(v) for v in per_joint), float(np.sqrt(np.mean(error**2)))


@dataclass(frozen=True)
class FitReport:
    """Training outcome: what was fitted and how well the readout reproduces the teacher-forced targets."""

    episodes: tuple[str, ...]
    loss_rows: int
    washout_rows: int
    rmse_per_joint: tuple[float, ...]
    """Teacher-forced one-step RMSE (rad) per joint over the loss rows of all episodes."""
    rmse: float
    """Aggregate teacher-forced one-step RMSE (rad) over the loss rows."""
    constant_rmse: float
    """RMSE of predicting the mean loss-row target everywhere; the readout must beat it to be useful."""
    max_abs_error: float
    episode_loss_rows: tuple[int, ...] | None = None
    """Loss rows of each episode (weighted fits only; ``None`` for the historical unweighted stack)."""
    episode_weights: tuple[float, ...] | None = None
    """The loss weight each episode was fitted with (weighted fits only)."""

    def __post_init__(self) -> None:
        """Counts and errors are finite and consistent, and recorded weighting covers every episode."""
        values = (*self.rmse_per_joint, self.rmse, self.constant_rmse, self.max_abs_error)
        if not self.episodes or self.loss_rows < 1 or self.washout_rows < 0:
            msg = "a fit report needs at least one episode and one loss row"
            raise ValueError(msg)
        if any(not math.isfinite(v) or v < 0 for v in values):
            msg = "fit errors must be finite and non-negative"
            raise ValueError(msg)
        self._check_weighting()

    def _check_weighting(self) -> None:
        counts, weights = self.episode_loss_rows, self.episode_weights
        if (counts is None) != (weights is None):
            msg = "episode_loss_rows and episode_weights are recorded together"
            raise ValueError(msg)
        if counts is None or weights is None:
            return
        if len(counts) != len(self.episodes) or len(weights) != len(self.episodes):
            msg = "a weighted fit report needs one loss-row count and one weight per episode"
            raise ValueError(msg)
        if any(rows < 1 for rows in counts):
            msg = f"episode loss rows must be positive, got {counts}"
            raise ValueError(msg)
        if any(not math.isfinite(w) or w <= 0 for w in weights):
            msg = f"episode weights must be finite and positive, got {weights}"
            raise ValueError(msg)


def train_readout(
    model: EsnModel, episodes: Sequence[Episode], *, weight_reference_rows: int | None = None
) -> FitReport:
    """Fit the readout on the loss rows of every episode and evaluate the teacher-forced prediction.

    Deterministic: the same model configuration and episodes give bitwise identical
    readouts and reports (see ``tests/unit/test_training.py``).

    Without ``weight_reference_rows`` every loss row carries weight one, which
    is the historical stacked fit. With it, each episode is weighted
    ``weight_reference_rows / L_i`` so episodes of unequal length carry equal
    total loss weight (manual plan section 4); the weights need the readout's
    explicit-bias layout and are recorded in the report.
    """
    batch = training_rows(model, episodes)
    per_episode: tuple[float, ...] | None = None
    if weight_reference_rows is None:
        model.fit_readout(batch.states, batch.targets)
    else:
        row_weights, per_episode = batch.row_weights(weight_reference_rows)
        model.fit_readout(batch.states, batch.targets, weights=row_weights)
    predictions = [predict_episode(model, episode) for episode in episodes]
    all_prediction = np.vstack(predictions)
    all_targets = np.vstack([episode.targets for episode in episodes])
    all_rows = np.concatenate([episode.loss_rows for episode in episodes])
    per_joint, aggregate = one_step_rmse(all_prediction, all_targets, all_rows)
    constant = np.mean(all_targets[all_rows], axis=0, dtype=np.float64)
    tiled = np.ascontiguousarray(np.tile(constant, (all_targets.shape[0], 1)), dtype=np.float64)
    _, constant_rmse = one_step_rmse(tiled, all_targets, all_rows)
    return FitReport(
        episodes=tuple(episode.source for episode in episodes),
        loss_rows=int(all_rows.sum()),
        washout_rows=int((~all_rows).sum()),
        rmse_per_joint=per_joint,
        rmse=aggregate,
        constant_rmse=constant_rmse,
        max_abs_error=float(np.abs(all_prediction[all_rows] - all_targets[all_rows]).max()),
        episode_loss_rows=None if per_episode is None else batch.loss_rows,
        episode_weights=per_episode,
    )

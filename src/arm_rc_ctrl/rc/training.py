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

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arm_rc_ctrl.rc.esn import EsnModel
    from arm_rc_ctrl.rc.teacher_forcing import Episode

__all__ = ["EpisodeStates", "harvest_episode", "harvest_states", "prime", "training_rows"]


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


def training_rows(model: EsnModel, episodes: Sequence[Episode]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Harvest every episode (each from a reset) and stack the loss rows into one ridge problem."""
    if not episodes:
        msg = "at least one episode is required"
        raise ValueError(msg)
    harvested = [harvest_episode(model, episode) for episode in episodes]
    states = np.vstack([h.training_states for h in harvested])
    targets = np.vstack([h.training_targets for h in harvested])
    return states, targets

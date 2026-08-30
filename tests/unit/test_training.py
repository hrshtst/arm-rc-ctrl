# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M2-003: per-episode reservoir reset and priming."""

from __future__ import annotations

import numpy as np
import pytest

from arm_rc_ctrl.rc.esn import EsnConfig, EsnModel, ReadoutConfig, ReservoirConfig
from arm_rc_ctrl.rc.teacher_forcing import Episode
from arm_rc_ctrl.rc.training import harvest_episode, harvest_states, prime, training_rows

CONFIG = EsnConfig(
    reservoir=ReservoirConfig(
        n_neurons=30, spectral_radius=0.9, sparsity=0.8, leak_rate=0.6, input_scaling=0.5, seed=3
    ),
    readout=ReadoutConfig(alpha=1e-3),
)
RNG = np.random.default_rng(7)


def _episode(source: str, rows: int, washout: int) -> Episode:
    inputs = RNG.standard_normal((rows, 3))
    targets = RNG.standard_normal((rows, 2))
    loss = np.arange(rows) >= washout
    return Episode(source, np.arange(rows, dtype=np.float64) * 0.01, inputs, targets, loss)


def _model() -> EsnModel:
    return EsnModel(CONFIG, input_dim=3, output_dim=2)


def test_each_episode_starts_from_a_reset_reservoir() -> None:
    """Harvesting B after A equals harvesting B alone; the state at the boundary is zero, not A's final state."""
    model = _model()
    a, b = _episode("a", 40, 5), _episode("b", 30, 5)
    harvest_episode(model, a)
    after_a = model.state()
    assert after_a.any()
    b_after_a = harvest_episode(model, b).states
    b_alone = harvest_episode(_model(), b).states
    assert np.array_equal(b_after_a, b_alone)
    # concatenating without a reset (a leak) would drive B from A's final state and differ
    model.reset()
    for row in a.inputs:
        model.advance(row)
    leaked = np.vstack([model.advance(row) for row in b.inputs])
    assert not np.array_equal(leaked, b_alone)


def test_training_rows_stack_only_loss_rows_across_episodes() -> None:
    """Washout rows are dropped per episode; the remaining rows keep their episode order and targets."""
    model = _model()
    a, b = _episode("a", 12, 4), _episode("b", 9, 2)
    states, targets = training_rows(model, [a, b])
    assert states.shape == (8 + 7, 30)
    assert np.array_equal(targets, np.vstack([a.targets[4:], b.targets[2:]]))
    assert np.array_equal(states[:8], harvest_states(_model(), a.inputs)[4:])
    assert np.array_equal(states[8:], harvest_states(_model(), b.inputs)[2:])
    with pytest.raises(ValueError, match="at least one episode"):
        training_rows(model, [])


def test_runtime_priming_reproduces_the_training_washout_state() -> None:
    """Replaying an episode's washout rows leaves the reservoir in the state training saw after them."""
    model = _model()
    episode = _episode("a", 50, 10)
    harvested = harvest_episode(model, episode)
    primed = prime(_model(), episode.inputs[: episode.washout_len])
    assert np.array_equal(primed, harvested.states[episode.washout_len - 1])
    assert np.array_equal(model.state(), harvested.states[-1])  # harvesting leaves the final state in place
    # continuing from the primed state reproduces the training states of the loss rows
    model.reset()
    prime(model, episode.inputs[: episode.washout_len])
    continued = np.vstack([model.advance(row) for row in episode.inputs[episode.washout_len :]])
    assert np.array_equal(continued, harvested.training_states)


def test_harvested_episode_keeps_alignment_and_rejects_mismatched_models() -> None:
    """States align with targets and the loss mask; a model of another width cannot harvest the episode."""
    model = _model()
    episode = _episode("a", 20, 3)
    harvested = harvest_episode(model, episode)
    assert harvested.source == "a"
    assert harvested.states.shape == (20, 30)
    assert np.array_equal(harvested.targets, episode.targets)
    assert np.array_equal(harvested.training_targets, episode.targets[3:])
    assert harvested.training_states.shape == (17, 30)
    wrong = EsnModel(CONFIG, input_dim=4, output_dim=2)
    with pytest.raises(ValueError, match="input_dim 3 and dof 2; the model expects 4 and 2"):
        harvest_episode(wrong, episode)
    with pytest.raises(ValueError, match=r"inputs must have shape \(M >= 1, input_dim\)"):
        harvest_states(model, np.zeros((0, 3)))

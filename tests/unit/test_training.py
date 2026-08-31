# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M2-003/M2-005: per-episode reservoir reset and priming; offline ridge training and validation prediction."""

from __future__ import annotations

import numpy as np
import pytest

from arm_rc_ctrl.rc.esn import EsnConfig, EsnModel, ReadoutConfig, ReservoirConfig
from arm_rc_ctrl.rc.teacher_forcing import Episode
from arm_rc_ctrl.rc.training import (
    FitReport,
    harvest_episode,
    harvest_states,
    one_step_rmse,
    predict_episode,
    prime,
    train_readout,
    training_rows,
)

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


def _sinusoid_episode(source: str, phase: float, rows: int = 300, washout: int = 30) -> Episode:
    """A learnable one-step problem: q_(k+1) is a linear function of [q_k, dq_k] for a sinusoidal joint motion."""
    t = np.arange(rows + 1, dtype=np.float64) * 0.01
    omega = np.array([2.0, 3.0])
    q = np.sin(omega[None, :] * t[:, None] + phase) * np.array([0.5, 0.3])
    dq = omega[None, :] * np.cos(omega[None, :] * t[:, None] + phase) * np.array([0.5, 0.3])
    inputs = np.hstack([q[:-1], dq[:-1]])
    return Episode(source, t[:-1], inputs, q[1:], np.arange(rows) >= washout)


def _sin_model() -> EsnModel:
    config = EsnConfig(
        reservoir=ReservoirConfig(
            n_neurons=100, spectral_radius=0.8, sparsity=0.9, leak_rate=0.3, input_scaling=0.3, seed=5
        ),
        readout=ReadoutConfig(alpha=1e-6),
    )
    return EsnModel(config, input_dim=4, output_dim=2)


def test_training_beats_a_constant_predictor_on_a_learnable_sequence() -> None:
    """The fitted readout predicts the next position far better than the mean target."""
    episode = _sinusoid_episode("sin", 0.0)
    report = train_readout(_sin_model(), [episode])
    assert report.episodes == ("sin",)
    assert (report.loss_rows, report.washout_rows) == (270, 30)
    assert len(report.rmse_per_joint) == 2
    assert report.rmse < 0.05 * report.constant_rmse
    assert report.max_abs_error < 0.05
    assert report.rmse == pytest.approx(np.sqrt(np.mean(np.square(report.rmse_per_joint))))


def test_training_is_deterministic() -> None:
    """Two trainings from the same configuration and episodes give identical reports and predictions."""
    episodes = [_sinusoid_episode("a", 0.0), _sinusoid_episode("b", 1.0)]
    first_model, second_model = _sin_model(), _sin_model()
    first = train_readout(first_model, episodes)
    second = train_readout(second_model, episodes)
    assert first == second
    assert first.episodes == ("a", "b")
    assert first.loss_rows == 540
    assert np.array_equal(predict_episode(first_model, episodes[1]), predict_episode(second_model, episodes[1]))


def test_validation_prediction_starts_from_a_reset_and_is_teacher_forced() -> None:
    """predict_episode resets first (repeatable) and feeds demonstrated inputs, never its own output."""
    episode = _sinusoid_episode("sin", 0.5)
    model = _sin_model()
    train_readout(model, [episode])
    once = predict_episode(model, episode)
    twice = predict_episode(model, episode)
    assert np.array_equal(once, twice)
    assert once.shape == (300, 2)
    # a partial replay gives the same leading rows: outputs depend only on the demonstrated inputs so far
    model.reset()
    partial = np.vstack([model.step(row) for row in episode.inputs[:50]])
    assert np.array_equal(partial, once[:50])
    with pytest.raises(ValueError, match="input_dim 4 and dof 2; the model expects 3 and 2"):
        predict_episode(EsnModel(CONFIG, input_dim=3, output_dim=2), episode)


def test_one_step_rmse_and_report_validation() -> None:
    """RMSE is computed over selected rows only; reports reject inconsistent values."""
    prediction = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    targets = np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 6.0]])
    per_joint, aggregate = one_step_rmse(prediction, targets, np.array([False, True, True]))
    assert per_joint == pytest.approx((np.sqrt(2.0), np.sqrt(0.5)))
    assert aggregate == pytest.approx(np.sqrt((0 + 1 + 4 + 0) / 4))
    with pytest.raises(ValueError, match="need at least one selected"):
        one_step_rmse(prediction, targets, np.zeros(3, dtype=np.bool_))
    with pytest.raises(ValueError, match="a fit report needs at least one episode"):
        FitReport((), 1, 0, (0.1,), 0.1, 0.2, 0.3)
    with pytest.raises(ValueError, match="finite and non-negative"):
        FitReport(("a",), 1, 0, (0.1,), float("nan"), 0.2, 0.3)

# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-005: equal-episode weighted ridge training on the explicit-bias readout (manual plan section 4, I3).

The declared objective of the manual-demonstration arms is

``(1 / K) * sum_i (||X_i W - Y_i||^2 / L_i) + (alpha_0 / 400) * ||W||^2``,

implemented as the equivalent weighted summed-loss fit with per-row weights
``400 / L_i`` and ``solver_alpha = K * alpha_0``. These tests check the fit
against an independently assembled weighted normal system, against the
historical implicit-bias fit at unit weights, and check that the unweighted
path is bitwise what it was.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from arm_rc_ctrl.rc.esn import EsnConfig, EsnModel, ReadoutConfig, ReservoirConfig
from arm_rc_ctrl.rc.teacher_forcing import Episode
from arm_rc_ctrl.rc.training import FitReport, harvest_episode, train_readout, training_rows

ROWS_REFERENCE = 400
"""The historical row-count reference of the weighting rule (400 loss rows per scripted episode)."""
BASE_ALPHA = 0.01
N_NEURONS = 30
DOF = 2
INPUT_DIM = 3
EPS = float(np.finfo(np.float64).eps)
RESERVOIR = ReservoirConfig(
    n_neurons=N_NEURONS, spectral_radius=0.9, sparsity=0.8, leak_rate=0.6, input_scaling=0.5, seed=3
)


def _model(alpha: float, *, explicit_bias: bool) -> EsnModel:
    readout = ReadoutConfig(alpha=alpha, include_bias=not explicit_bias, explicit_bias=explicit_bias or None)
    return EsnModel(EsnConfig(reservoir=RESERVOIR, readout=readout), input_dim=INPUT_DIM, output_dim=DOF)


def _episode(source: str, rows: int, washout: int, seed: int) -> Episode:
    rng = np.random.default_rng(seed)
    inputs = rng.standard_normal((rows, INPUT_DIM))
    targets = rng.standard_normal((rows, DOF))
    return Episode(source, np.arange(rows, dtype=np.float64) * 0.01, inputs, targets, np.arange(rows) >= washout)


def _loss_blocks(alpha: float, episodes: list[Episode]) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Harvested loss-row states and targets per episode, from a fresh model of the same configuration."""
    harvested = [harvest_episode(_model(alpha, explicit_bias=True), episode) for episode in episodes]
    return [h.training_states for h in harvested], [h.training_targets for h in harvested]


def _weighted_normal_system(
    states: list[NDArray[np.float64]],
    targets: list[NDArray[np.float64]],
    weights: list[float],
    alpha: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(sum_i w_i A_i^T A_i + alpha I, sum_i w_i A_i^T Y_i)`` of the declared objective, assembled per episode.

    ``A_i`` is the episode's loss-row state block with the explicit ones column
    appended. The assembly never scales a row, so it is independent of the
    square-root row weighting the implementation uses.
    """
    dim = states[0].shape[1] + 1
    normal = alpha * np.eye(dim)
    rhs = np.zeros((dim, targets[0].shape[1]))
    for block, target, weight in zip(states, targets, weights, strict=True):
        features = np.hstack([block, np.ones((block.shape[0], 1))])
        normal = normal + weight * (features.T @ features)
        rhs = rhs + weight * (features.T @ target)
    return normal, rhs


def _solve_bound(normal: NDArray[np.float64], solution: NDArray[np.float64]) -> float:
    """Backward-stability bound for two different factorizations of one normal system, with a factor-64 margin.

    ``rclib`` solves the system with Eigen's LDLT and the reference with
    LAPACK's LU; both are backward stable, so their solutions may differ by
    about ``cond(N) * eps * ||W||`` in the worst case.
    """
    return 64.0 * EPS * float(np.linalg.cond(normal)) * float(np.max(np.abs(solution)))


def _normalized_residual(normal: NDArray[np.float64], weights: NDArray[np.float64], rhs: NDArray[np.float64]) -> float:
    """``||N W - B||_F / (||N||_F ||W||_F + ||B||_F)``: the normal-equation residual of M3REP-003."""
    scale = float(np.linalg.norm(normal) * np.linalg.norm(weights) + np.linalg.norm(rhs))
    return float(np.linalg.norm(normal @ weights - rhs) / scale)


def test_weighted_fit_solves_the_declared_equal_episode_objective() -> None:
    """Two episodes of unequal length are fitted at ``400 / L_i`` and ``K alpha_0``, matching an independent solve."""
    episodes = [_episode("long", 165, 25, seed=11), _episode("short", 85, 25, seed=12)]
    count = len(episodes)
    alpha = count * BASE_ALPHA
    model = _model(alpha, explicit_bias=True)
    report = train_readout(model, episodes, weight_reference_rows=ROWS_REFERENCE)
    assert report.episode_loss_rows == (140, 60)
    assert report.episode_weights == (ROWS_REFERENCE / 140, ROWS_REFERENCE / 60)

    states, targets = _loss_blocks(alpha, episodes)
    weights = [ROWS_REFERENCE / block.shape[0] for block in states]
    normal, rhs = _weighted_normal_system(states, targets, weights, alpha)
    expected = np.linalg.solve(normal, rhs)
    fitted = model.readout_weights()
    assert fitted.shape == (N_NEURONS + 1, DOF)
    # The fit is a stationary point of the declared objective (gradient residual of the assembled system).
    assert _normalized_residual(normal, fitted, rhs) <= 1e-10
    bound = _solve_bound(normal, expected)
    assert bound < 1e-6  # the problem is well conditioned; the tolerance stays far below any physical scale
    assert float(np.max(np.abs(fitted - expected))) <= bound


def test_equal_length_episodes_receive_equal_weights() -> None:
    """Episodes of one length get one weight; the weights are exactly ``400 / L`` per episode."""
    episodes = [_episode("a", 125, 25, seed=13), _episode("b", 125, 25, seed=14)]
    alpha = len(episodes) * BASE_ALPHA
    report = train_readout(_model(alpha, explicit_bias=True), episodes, weight_reference_rows=ROWS_REFERENCE)
    weights = report.episode_weights
    assert report.episode_loss_rows == (100, 100)
    assert weights is not None
    assert weights == (4.0, 4.0)
    assert len(set(weights)) == 1


def test_unit_weights_reproduce_the_historical_implicit_bias_fit() -> None:
    """At ``L_i = 400`` every weight is one, and the explicit ones column reproduces ``rclib``'s implicit bias."""
    episodes = [_episode("a", 425, 25, seed=21), _episode("b", 425, 25, seed=22)]
    alpha = len(episodes) * BASE_ALPHA
    explicit = _model(alpha, explicit_bias=True)
    weighted = train_readout(explicit, episodes, weight_reference_rows=ROWS_REFERENCE)
    assert weighted.episode_weights == (1.0, 1.0)
    implicit = _model(alpha, explicit_bias=False)
    plain = train_readout(implicit, episodes)
    assert plain.episode_weights is None
    assert plain.episode_loss_rows is None

    fitted, historical = explicit.readout_weights(), implicit.readout_weights()
    assert fitted.shape == historical.shape == (N_NEURONS + 1, DOF)
    states, targets = _loss_blocks(alpha, episodes)
    normal, _rhs = _weighted_normal_system(states, targets, [1.0, 1.0], alpha)
    bound = _solve_bound(normal, fitted)
    assert bound < 1e-6
    assert float(np.max(np.abs(fitted - historical))) <= bound
    assert weighted.rmse == pytest.approx(plain.rmse, abs=1e-12)


def test_unweighted_training_is_unchanged_bitwise() -> None:
    """Without a weight reference the fit is exactly the historical stacked one, and records no weights."""
    episodes = [_episode("a", 60, 10, seed=31), _episode("b", 40, 10, seed=32)]
    model = _model(BASE_ALPHA, explicit_bias=False)
    report = train_readout(model, episodes)
    reference = _model(BASE_ALPHA, explicit_bias=False)
    batch = training_rows(reference, episodes)
    reference.fit_readout(batch.states, batch.targets)
    assert np.array_equal(model.readout_weights(), reference.readout_weights())
    assert (report.episode_loss_rows, report.episode_weights) == (None, None)
    assert batch.loss_rows == (50, 30)
    assert batch.states.shape == (80, N_NEURONS)
    assert np.array_equal(batch.targets, np.vstack([episodes[0].targets[10:], episodes[1].targets[10:]]))


def test_weighted_training_and_weights_are_validated() -> None:
    """A non-positive reference, an implicit-bias model, and malformed weight vectors are refused."""
    episodes = [_episode("a", 60, 10, seed=41)]
    with pytest.raises(ValueError, match="weight_reference_rows must be >= 1"):
        train_readout(_model(BASE_ALPHA, explicit_bias=True), episodes, weight_reference_rows=0)
    with pytest.raises(ValueError, match="explicit_bias"):
        train_readout(_model(BASE_ALPHA, explicit_bias=False), episodes, weight_reference_rows=ROWS_REFERENCE)
    model = _model(BASE_ALPHA, explicit_bias=True)
    states, targets = np.zeros((5, N_NEURONS)), np.zeros((5, DOF))
    with pytest.raises(ValueError, match=r"weights must have shape \(5,\)"):
        model.fit_readout(states, targets, weights=np.ones(4))
    with pytest.raises(ValueError, match="weights must be finite and positive"):
        model.fit_readout(states, targets, weights=np.zeros(5))
    with pytest.raises(ValueError, match="weights must be finite and positive"):
        model.fit_readout(states, targets, weights=np.full(5, np.nan))


def test_fit_report_records_consistent_per_episode_weighting() -> None:
    """Recorded per-episode rows and weights cover every episode and stay positive and finite."""
    report = FitReport(
        ("a", "b"), 10, 2, (0.1, 0.1), 0.1, 0.5, 0.2, episode_loss_rows=(6, 4), episode_weights=(2.0, 3.0)
    )
    assert report.episode_loss_rows == (6, 4)
    with pytest.raises(ValueError, match="one loss-row count and one weight per episode"):
        FitReport(("a", "b"), 10, 2, (0.1,), 0.1, 0.5, 0.2, episode_loss_rows=(10,), episode_weights=(1.0, 1.0))
    with pytest.raises(ValueError, match="recorded together"):
        FitReport(("a",), 10, 2, (0.1,), 0.1, 0.5, 0.2, episode_loss_rows=(10,))
    with pytest.raises(ValueError, match="episode weights must be finite and positive"):
        FitReport(("a",), 10, 2, (0.1,), 0.1, 0.5, 0.2, episode_loss_rows=(10,), episode_weights=(0.0,))
    with pytest.raises(ValueError, match="episode loss rows must be positive"):
        FitReport(("a",), 10, 2, (0.1,), 0.1, 0.5, 0.2, episode_loss_rows=(0,), episode_weights=(1.0,))

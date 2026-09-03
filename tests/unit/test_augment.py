# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-004: deterministic paired AR(1) augmentation with the locked terminal taper."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from numpy.typing import NDArray

from arm_rc_ctrl.data.derivatives import DerivativeConfig, differentiate
from arm_rc_ctrl.data.recovery import TaskIntervals
from arm_rc_ctrl.rc.augment import (
    APPROVED_GAMMA,
    APPROVED_N_SYNTHETIC,
    APPROVED_PHI,
    APPROVED_SIGMA_RAD,
    SEED_NAMESPACE,
    TAPER_DURATION_S,
    TAPER_ZERO_MARGIN_S,
    AugmentationConfig,
    AugmentationError,
    contraction_envelope,
    generate_augmentation,
    terminal_taper,
)
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import endpoint_positions, load_scenario

SCENARIO = load_scenario(repository_root() / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml")
DERIVATIVES = DerivativeConfig(method="central")
N = 101
DT = 0.01
TASK = TaskIntervals(move=(0.0, 0.8), dwell=(0.8, 1.0))
T: NDArray[np.float64] = np.arange(N, dtype=np.float64) * DT


def _reference() -> NDArray[np.float64]:
    """A smooth reach from (0.3, 0.6) to (0.8, 0.4) rad that holds during the dwell."""
    start = np.array([0.3, 0.6])
    goal = np.array([0.8, 0.4])
    s = np.clip(T / TASK.move[1], 0.0, 1.0)
    blend = s * s * (3.0 - 2.0 * s)
    return start[None, :] + blend[:, None] * (goal - start)[None, :]


def _config(**changes: object) -> AugmentationConfig:
    base = AugmentationConfig(n_synthetic=16, sigma_rad=0.05, phi=0.99, gamma=1.0, seed_bank=1, attempt_budget=64)
    return dataclasses.replace(base, **changes)


def _generate(config: AugmentationConfig):  # noqa: ANN202
    return generate_augmentation(T, _reference(), TASK, SCENARIO, config, derivatives=DERIVATIVES)


@pytest.mark.parametrize("n_synthetic", sorted(APPROVED_N_SYNTHETIC))
@pytest.mark.parametrize("sigma_rad", sorted(APPROVED_SIGMA_RAD))
def test_approved_budgets_and_scales_are_accepted(n_synthetic: int, sigma_rad: float) -> None:
    """Every approved (N_aug, sigma) combination constructs."""
    _config(n_synthetic=n_synthetic, sigma_rad=sigma_rad, attempt_budget=4 * n_synthetic)


@pytest.mark.parametrize("phi", sorted(APPROVED_PHI))
@pytest.mark.parametrize("gamma", sorted(APPROVED_GAMMA))
def test_approved_correlations_and_exponents_are_accepted(phi: float, gamma: float) -> None:
    """Every approved (phi, gamma) combination constructs."""
    _config(phi=phi, gamma=gamma)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"n_synthetic": 8}, "approved"),
        ({"sigma_rad": 0.2}, "approved"),
        ({"phi": 0.9}, "approved"),
        ({"gamma": 3.0}, "approved"),
        ({"attempt_budget": 8}, "attempt_budget"),
        ({"seed_bank": -1}, "seed_bank"),
        ({"max_abs_perturbation_rad": 0.0}, "max_abs_perturbation_rad"),
    ],
)
def test_off_protocol_configurations_are_rejected(changes: dict[str, object], message: str) -> None:
    """Values outside the approved D1 grids and inconsistent budgets fail at construction."""
    with pytest.raises(ValueError, match=message):
        _config(**changes)


def test_terminal_taper_is_smooth_fixed_and_reaches_zero_before_dwell() -> None:
    """The locked taper is 1 through most of the move, decays smoothly, and is exactly 0 before dwell onset."""
    taper = terminal_taper(T, TASK.dwell[0])
    zero_from = TASK.dwell[0] - TAPER_ZERO_MARGIN_S
    window_start = zero_from - TAPER_DURATION_S
    assert np.all(taper[window_start >= T] == 1.0)
    assert np.all(taper[zero_from <= T] == 0.0)
    inside = (window_start < T) & (zero_from > T)
    assert np.all((taper[inside] > 0.0) & (taper[inside] < 1.0))
    assert np.all(np.diff(taper) <= 0.0)
    assert float(np.max(np.abs(np.diff(taper)))) < 2.0 * DT / TAPER_DURATION_S
    midpoint = float(np.interp(zero_from - TAPER_DURATION_S / 2.0, T, taper))
    assert midpoint == pytest.approx(0.5, abs=0.05)


def test_terminal_taper_requires_room_before_the_dwell() -> None:
    """An episode whose movement is shorter than the taper window cannot be augmented."""
    with pytest.raises(AugmentationError, match="taper"):
        terminal_taper(np.arange(21, dtype=np.float64) * DT, 0.1)


def test_contraction_envelope_follows_endpoint_distance() -> None:
    """The envelope equals clip(d_tip/d_tip0, 0, 1)^gamma and decreases toward the target."""
    q_ref = _reference()
    gamma = 2.0
    envelope = contraction_envelope(SCENARIO, q_ref, gamma)
    tip = endpoint_positions(SCENARIO, q_ref)
    target = np.asarray(SCENARIO.task.target)
    diff = tip - target[None, :]
    distance = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
    expected = np.clip(distance / distance[0], 0.0, 1.0) ** gamma
    assert np.array_equal(envelope, expected)
    assert envelope[0] == 1.0
    assert envelope[-1] < 0.1


def test_generation_is_deterministic_and_seed_separated() -> None:
    """The same configuration reproduces byte-identical arrays; another seed bank differs."""
    first = _generate(_config())
    second = _generate(_config())
    assert first.digests() == second.digests()
    other = _generate(_config(seed_bank=2))
    assert other.digests() != first.digests()
    assert SEED_NAMESPACE == 20260903


def test_original_episode_is_the_unmodified_demonstration() -> None:
    """Episode 0 carries the reference positions, a zero perturbation, and recomputed velocity."""
    result = _generate(_config())
    q_ref = _reference()
    assert np.array_equal(result.original.q, q_ref)
    assert np.all(result.original.delta == 0.0)
    dq_expected, _ = differentiate(q_ref, DT, DERIVATIVES)
    assert np.array_equal(result.original.dq, dq_expected)
    assert result.original.delta_rms_rad == 0.0
    assert result.original.delta_peak_rad == 0.0


def test_matched_arms_share_the_latent_process() -> None:
    """Before the taper window the non-decaying delta is the latent z and contractive = envelope * z."""
    config = _config()
    result = _generate(config)
    envelope = contraction_envelope(SCENARIO, _reference(), config.gamma)
    taper = terminal_taper(T, TASK.dwell[0])
    untapered = taper == 1.0
    for episode in result.episodes:
        z = episode.non_decaying.delta[untapered]
        assert np.allclose(episode.contractive.delta[untapered], envelope[untapered, None] * z, atol=1e-15)
        assert np.any(z != 0.0)


def test_perturbations_collapse_exactly_before_and_throughout_the_dwell() -> None:
    """Both families are exactly zero from the locked margin before dwell onset through the episode end."""
    result = _generate(_config())
    zero_from = TASK.dwell[0] - TAPER_ZERO_MARGIN_S
    tail = zero_from <= T
    dwell = TASK.dwell[0] <= T
    assert np.any(tail & ~dwell)  # exact zero is reached strictly before dwell onset
    for episode in result.episodes:
        for arrays in (episode.non_decaying, episode.contractive):
            assert np.all(arrays.delta[tail] == 0.0)
            assert np.array_equal(arrays.q[tail], _reference()[tail])


def test_perturbations_are_temporally_smooth() -> None:
    """No sample-to-sample jump approaches the marginal scale; the taper never steps to zero."""
    config = _config()
    result = _generate(config)
    for episode in result.episodes:
        for arrays in (episode.non_decaying, episode.contractive):
            assert float(np.max(np.abs(np.diff(arrays.delta, axis=0)))) < config.sigma_rad


def test_stationary_initialization_varies_across_episodes() -> None:
    """z_0 ~ N(0, sigma^2 I): the first untapered perturbation differs per episode with scale near sigma."""
    config = _config(n_synthetic=64, attempt_budget=256)
    result = _generate(config)
    starts = np.array([episode.non_decaying.delta[0] for episode in result.episodes])
    assert starts.shape[0] == 64
    assert np.unique(starts, axis=0).shape[0] == 64
    spread = float(np.std(starts))
    assert 0.5 * config.sigma_rad < spread < 1.5 * config.sigma_rad


def test_velocity_is_recomputed_from_augmented_positions() -> None:
    """Each episode's velocity equals the versioned derivative of its own augmented positions."""
    result = _generate(_config())
    for episode in result.episodes:
        for arrays in (episode.non_decaying, episode.contractive):
            dq_expected, _ = differentiate(arrays.q, DT, DERIVATIVES)
            assert np.array_equal(arrays.dq, dq_expected)
            assert np.array_equal(arrays.q, _reference() + arrays.delta)


def test_realized_statistics_and_digests_are_recorded() -> None:
    """Per-episode RMS/peak statistics and per-array digests cover every family."""
    config = _config()
    result = _generate(config)
    assert len(result.episodes) == config.n_synthetic
    assert result.attempts_used >= config.n_synthetic
    for episode in result.episodes:
        for arrays in (episode.non_decaying, episode.contractive):
            assert arrays.delta_peak_rad >= arrays.delta_rms_rad > 0.0
            assert arrays.delta_peak_rad == float(np.max(np.abs(arrays.delta)))
    digests = result.digests()
    assert "original/q" in digests
    assert f"episode-{config.n_synthetic:03d}/contractive/dq" in digests
    families = {key.split("/")[1] for key in digests if key.startswith("episode-")}
    assert families == {"non_decaying", "contractive"}


def test_rejection_accounting_and_finite_budget() -> None:
    """Attempts violating the configured perturbation bound are recorded; an exhausted budget fails loudly."""
    probe = _generate(_config())
    peaks = sorted(
        max(episode.non_decaying.delta_peak_rad, episode.contractive.delta_peak_rad) for episode in probe.episodes
    )
    threshold = peaks[len(peaks) // 2]
    config = _config(max_abs_perturbation_rad=threshold, attempt_budget=256)
    result = _generate(config)
    assert len(result.episodes) == config.n_synthetic
    assert result.rejections
    assert all(r.reason and r.attempt >= 1 for r in result.rejections)
    with pytest.raises(AugmentationError, match="attempt budget"):
        _generate(_config(max_abs_perturbation_rad=1e-06, attempt_budget=32))


def test_paired_arms_reject_together() -> None:
    """A configured bound violated by either family rejects the whole attempt, keeping the arms matched."""
    probe = _generate(_config())
    accepted_attempts = {episode.attempt for episode in probe.episodes}
    peaks = sorted(
        max(episode.non_decaying.delta_peak_rad, episode.contractive.delta_peak_rad) for episode in probe.episodes
    )
    threshold = peaks[len(peaks) // 2]
    result = _generate(_config(max_abs_perturbation_rad=threshold, attempt_budget=256))
    rejected = {r.attempt for r in result.rejections}
    assert rejected & accepted_attempts  # some previously accepted attempts now fail the bound
    for episode in result.episodes:
        assert episode.attempt not in rejected

# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-006: the manual contractive arm preserves the recorded boundaries and seeds one stream per parent.

The manual protocol is a separate, versioned arm of the augmentation module:
its envelope starts at exactly zero on the recorded first sample, ramps in over
the frozen 0.5 s, and still collapses through the inherited terminal taper, and
its streams carry a parent term (clarification I8) so two demonstrations never
draw the same latent process. The inherited recovery generator keeps every
number it had.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import pytest

from arm_rc_ctrl.data.derivatives import DerivativeConfig, differentiate
from arm_rc_ctrl.data.manual_scenario import ManualScenarioConfig, load_manual_scenario
from arm_rc_ctrl.rc.augment import (
    APPROVED_GAMMA,
    APPROVED_N_SYNTHETIC,
    APPROVED_PHI,
    APPROVED_SIGMA_RAD,
    MANUAL_ATTEMPT_BUDGET,
    MANUAL_ENVELOPE_VERSION,
    MANUAL_GAMMA,
    MANUAL_N_SYNTHETIC,
    MANUAL_PHI,
    MANUAL_RAMP_DURATION_S,
    MANUAL_SEED_NAMESPACE,
    MANUAL_SEED_WORD,
    MANUAL_SIGMA_RAD,
    SEED_NAMESPACE,
    TAPER_DURATION_S,
    TAPER_ZERO_MARGIN_S,
    AugmentationError,
    ManualAugmentationConfig,
    ManualAugmentationResult,
    ManualBudgetExhaustedError,
    contraction_envelope,
    entropy_word,
    generate_manual_augmentation,
    manual_boundary_envelope,
    terminal_taper,
)
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import load_scenario

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from arm_rc_ctrl.rc.augment import TaskGeometry

REPO_ROOT = repository_root()
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "configs"
SCENARIO = load_manual_scenario(FIXTURES / "planar_2dof_manual_fixture.toml")
RECOVERY_SCENARIO = load_scenario(FIXTURES / "planar_2dof_fixture.toml")
DERIVATIVES = DerivativeConfig(method="central")
DT = 0.01
N = 201
T: NDArray[np.float64] = np.arange(N, dtype=np.float64) * DT
DWELL_START_S = 1.5
ZERO_FROM_S = DWELL_START_S - TAPER_ZERO_MARGIN_S
WINDOW_START_S = ZERO_FROM_S - TAPER_DURATION_S
PARENT = "processed-20260916-a8c94bb35358"
OTHER_PARENT = "processed-20260916-c33a21797eb3"
GOAL_Q = (0.8, 0.4)
HOLD_S = 0.2


def _reference() -> NDArray[np.float64]:
    """A complete recording: a 0.2 s pre-roll hold, a smooth reach, and a final dwell at the goal."""
    start = np.asarray(SCENARIO.task.initial_q, dtype=np.float64)
    goal = np.asarray(GOAL_Q, dtype=np.float64)
    s = np.clip((T - HOLD_S) / (DWELL_START_S - HOLD_S), 0.0, 1.0)
    blend = s * s * (3.0 - 2.0 * s)
    return start[None, :] + blend[:, None] * (goal - start)[None, :]


def _config(**changes: object) -> ManualAugmentationConfig:
    base = ManualAugmentationConfig(parent=PARENT, seed_bank=1)
    return dataclasses.replace(base, **changes)


def _generate(
    config: ManualAugmentationConfig | None = None,
    *,
    scenario: TaskGeometry = SCENARIO,
    q_ref: NDArray[np.float64] | None = None,
) -> ManualAugmentationResult:
    return generate_manual_augmentation(
        T,
        _reference() if q_ref is None else q_ref,
        scenario,
        _config() if config is None else config,
        dwell_start_s=DWELL_START_S,
        period_s=DT,
        derivatives=DERIVATIVES,
    )


def _slow_scenario(velocity: float) -> ManualScenarioConfig:
    """The manual fixture with both joints' speed limit (and the acquisition bound with it) lowered."""
    return dataclasses.replace(
        SCENARIO,
        limits=dataclasses.replace(SCENARIO.limits, velocity=(velocity, velocity)),
        acquisition=dataclasses.replace(SCENARIO.acquisition, velocity_bound_rad_s=velocity),
    )


# --- the versioned boundary envelope -------------------------------------------------------


def test_manual_envelope_is_zero_at_the_first_sample_and_ramps_in_smoothly() -> None:
    """Exactly zero on the first recorded sample, strictly rising through the ramp, one after it."""
    envelope = manual_boundary_envelope(T, DWELL_START_S)
    assert envelope[0] == 0.0
    ramp = T <= MANUAL_RAMP_DURATION_S
    assert np.all(np.diff(envelope[ramp]) > 0.0)
    plateau = (T >= MANUAL_RAMP_DURATION_S) & (T <= WINDOW_START_S)
    assert np.any(plateau)
    assert np.all(envelope[plateau] == 1.0)
    assert np.all(envelope[T >= ZERO_FROM_S] == 0.0)
    assert np.any((T >= ZERO_FROM_S) & (T < DWELL_START_S))  # zero is reached strictly before the dwell
    assert float(np.max(np.abs(np.diff(envelope[T <= MANUAL_RAMP_DURATION_S])))) < 2.0 * DT / MANUAL_RAMP_DURATION_S
    assert float(np.max(np.abs(np.diff(envelope)))) < 2.0 * DT / TAPER_DURATION_S
    midpoint = float(np.interp(MANUAL_RAMP_DURATION_S / 2.0, T, envelope))
    assert midpoint == pytest.approx(0.5, abs=0.05)


def test_manual_envelope_is_the_ramp_times_the_inherited_terminal_taper() -> None:
    """The manual arm reuses the locked taper unchanged and only adds the ramp-in factor."""
    envelope = manual_boundary_envelope(T, DWELL_START_S)
    x = np.clip((T - T[0]) / MANUAL_RAMP_DURATION_S, 0.0, 1.0)
    expected = (x * x * (3.0 - 2.0 * x)) * terminal_taper(T, DWELL_START_S)
    assert np.array_equal(envelope, expected)


def test_manual_envelope_refuses_a_recording_without_room_for_the_ramp_and_taper() -> None:
    """Insufficient transition support is reported, never worked around by altering the timing."""
    short = np.arange(61, dtype=np.float64) * DT  # 0.60 s recording
    with pytest.raises(AugmentationError, match="ramp"):
        manual_boundary_envelope(short, 0.6)
    with pytest.raises(AugmentationError, match="taper"):
        manual_boundary_envelope(np.arange(21, dtype=np.float64) * DT, 0.1)
    with pytest.raises(AugmentationError, match="dwell"):
        manual_boundary_envelope(T, 10.0)  # the dwell lies past the end of the recording


def test_the_inherited_recovery_envelopes_are_unchanged() -> None:
    """The recovery families still perturb their first sample; the manual arm is a separate function."""
    assert terminal_taper(T, DWELL_START_S)[0] == 1.0
    assert contraction_envelope(RECOVERY_SCENARIO, _reference(), 1.0)[0] == 1.0
    assert SEED_NAMESPACE == 415926535
    assert MANUAL_SEED_WORD != SEED_NAMESPACE


# --- the frozen manual configuration -------------------------------------------------------


def test_manual_configuration_is_the_frozen_protocol() -> None:
    """The manual values are frozen constants, not a grid, and the approved D1 grids stay untouched."""
    config = _config()
    assert (config.n_synthetic, config.sigma_rad, config.phi, config.gamma) == (9, 0.05, 0.99, 1.0)
    assert (config.attempt_budget, config.ramp_duration_s) == (36, 0.5)
    assert (MANUAL_N_SYNTHETIC, MANUAL_SIGMA_RAD, MANUAL_PHI, MANUAL_GAMMA) == (9, 0.05, 0.99, 1.0)
    assert (MANUAL_ATTEMPT_BUDGET, MANUAL_RAMP_DURATION_S) == (36, 0.5)
    assert MANUAL_SEED_NAMESPACE == "task_1a_manual_v1/contractive/v1"
    assert MANUAL_ENVELOPE_VERSION == "boundary_ramp_v1"
    assert entropy_word(MANUAL_SEED_NAMESPACE) == MANUAL_SEED_WORD
    # The manual protocol never widens the approved recovery grids.
    assert MANUAL_N_SYNTHETIC not in APPROVED_N_SYNTHETIC
    assert sorted(APPROVED_N_SYNTHETIC) == [16, 32, 64]
    assert sorted(APPROVED_SIGMA_RAD) == [0.01, 0.025, 0.05, 0.10]
    assert sorted(APPROVED_PHI) == [0.98, 0.99, 0.995]
    assert sorted(APPROVED_GAMMA) == [0.5, 1.0, 2.0]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"n_synthetic": 16}, "n_synthetic"),
        ({"sigma_rad": 0.01}, "sigma_rad"),
        ({"phi": 0.98}, "phi"),
        ({"gamma": 2.0}, "gamma"),
        ({"attempt_budget": 64}, "attempt_budget"),
        ({"ramp_duration_s": 0.25}, "ramp_duration_s"),
        ({"seed_bank": -1}, "seed_bank"),
        ({"parent": "  "}, "parent"),
    ],
)
def test_off_protocol_manual_configurations_are_rejected(changes: dict[str, object], message: str) -> None:
    """The frozen values are the protocol: any deviation fails at construction."""
    with pytest.raises(ValueError, match=message):
        _config(**changes)


# --- generation ----------------------------------------------------------------------------


def test_synthetic_episodes_keep_the_recorded_start_and_dwell_bitwise() -> None:
    """Every episode starts at the recorded first sample and holds the recorded dwell, bit for bit."""
    result = _generate()
    q_ref = _reference()
    tail = T >= ZERO_FROM_S
    assert len(result.episodes) == MANUAL_N_SYNTHETIC
    for episode in result.episodes:
        arrays = episode.arrays
        # The envelope is exactly zero there, so the recorded first sample survives bit for bit
        # (the sign of that zero perturbation is free: q + (-0.0) is q).
        assert np.all(arrays.delta[0] == 0.0)
        assert arrays.q[0].tobytes() == q_ref[0].tobytes()
        assert arrays.q[-1].tobytes() == q_ref[-1].tobytes()
        assert np.array_equal(arrays.q[tail], q_ref[tail])
        assert np.all(arrays.delta[tail] == 0.0)
        assert np.any(arrays.delta != 0.0)


def test_contractive_delta_is_the_boundary_envelope_times_the_contraction() -> None:
    """The contractive form is the versioned boundary envelope multiplied by the inherited contraction."""
    config = _config()
    result = _generate(config)
    q_ref = _reference()
    envelope = contraction_envelope(SCENARIO, q_ref, config.gamma) * manual_boundary_envelope(T, DWELL_START_S)
    for episode in result.episodes:
        delta = episode.arrays.delta
        latent = np.zeros_like(delta)
        interior = envelope > 0.0
        latent[interior] = delta[interior] / envelope[interior, None]
        assert np.allclose(delta, envelope[:, None] * latent, atol=1e-15)
        assert float(np.max(np.abs(latent))) < 6.0 * config.sigma_rad


def test_velocity_is_recomputed_from_the_augmented_positions() -> None:
    """Velocity comes from the parent's derivative policy applied to the augmented positions."""
    result = _generate()
    q_ref = _reference()
    for episode in result.episodes:
        arrays = episode.arrays
        expected, _ = differentiate(arrays.q, DT, DERIVATIVES)
        assert np.array_equal(arrays.dq, expected)
        assert np.array_equal(arrays.q, q_ref + arrays.delta)
    assert result.derivative_method == DERIVATIVES.label
    assert result.period_s == DT


def test_parents_sharing_a_seed_bank_draw_different_latents() -> None:
    """Clarification I8: the stream carries a parent term, so two demonstrations never share draws."""
    first = _generate(_config(parent=PARENT))
    second = _generate(_config(parent=OTHER_PARENT))
    assert first.digests() != second.digests()
    for a, b in zip(first.episodes, second.episodes, strict=True):
        assert not np.array_equal(a.arrays.delta, b.arrays.delta)
    assert _generate(_config(parent=PARENT)).digests() == first.digests()
    assert _generate(_config(seed_bank=2)).digests() != first.digests()


def test_generation_is_deterministic_and_independent_of_order() -> None:
    """Regeneration reproduces identical arrays whatever order the parents were generated in."""
    forward = [_generate(_config(parent=PARENT)), _generate(_config(parent=OTHER_PARENT))]
    backward = [_generate(_config(parent=OTHER_PARENT)), _generate(_config(parent=PARENT))]
    assert forward[0].digests() == backward[1].digests()
    assert forward[1].digests() == backward[0].digests()
    assert [e.attempt for e in forward[0].episodes] == sorted(e.attempt for e in forward[0].episodes)


def test_limit_violations_reject_attempts_with_their_reason() -> None:
    """An episode leaving the task limits is rejected with its reason and never clipped or retimed."""
    probe = _generate()
    speeds = sorted(float(np.max(np.abs(episode.arrays.dq))) for episode in probe.episodes)
    limit = speeds[-3]  # the recording itself stays well inside this limit; its fastest episodes do not
    assert float(np.max(np.abs(probe.original.dq))) < limit
    result = _generate(scenario=_slow_scenario(limit))
    assert len(result.episodes) == MANUAL_N_SYNTHETIC
    assert result.rejections
    assert {r.reason for r in result.rejections} == {"velocity limit violated"}
    assert all(r.family == "contractive" for r in result.rejections)
    rejected = {r.attempt for r in result.rejections}
    accepted = {episode.attempt for episode in result.episodes}
    assert not (rejected & accepted)
    assert result.attempts_used == max(accepted)
    q_ref = _reference()
    for episode in result.episodes:
        # Rejected, never clipped or retimed: what survives is a whole episode that meets the limit.
        assert float(np.max(np.abs(episode.arrays.dq))) <= limit
        assert np.array_equal(episode.arrays.q, q_ref + episode.arrays.delta)
        assert np.array_equal(episode.arrays.q[0], q_ref[0])


def test_budget_exhaustion_reports_every_rejection() -> None:
    """A parent that cannot fill its bank fails loudly after exactly the budgeted attempts."""
    probe = _generate()
    limit = 1.5 * float(np.max(np.abs(probe.original.dq)))  # the recording fits comfortably; no episode does
    with pytest.raises(ManualBudgetExhaustedError) as excinfo:
        _generate(scenario=_slow_scenario(limit))
    failure = excinfo.value
    assert isinstance(failure, AugmentationError)
    assert failure.attempts_used == MANUAL_ATTEMPT_BUDGET
    assert len(failure.rejections) == MANUAL_ATTEMPT_BUDGET
    assert {r.attempt for r in failure.rejections} == set(range(1, MANUAL_ATTEMPT_BUDGET + 1))
    assert {r.reason for r in failure.rejections} == {"velocity limit violated"}
    assert failure.accepted == 0
    assert "attempt budget" in str(failure)


# --- the generator refuses malformed inputs instead of guessing ----------------------------


@pytest.mark.parametrize(
    ("times", "dwell_start_s", "ramp_duration_s", "message"),
    [
        (np.zeros((4, 2)), 1.5, MANUAL_RAMP_DURATION_S, "1-D"),
        (np.empty(0), 1.5, MANUAL_RAMP_DURATION_S, "1-D"),
        (np.array([0.0, 0.01, np.nan]), 1.5, MANUAL_RAMP_DURATION_S, "1-D"),
        (T, 1.5, 0.0, "ramp_duration_s"),
        (T, 1.5, float("inf"), "ramp_duration_s"),
        (T, 0.0, MANUAL_RAMP_DURATION_S, "dwell onset"),
    ],
)
def test_malformed_envelope_inputs_are_refused(
    times: NDArray[np.float64], dwell_start_s: float, ramp_duration_s: float, message: str
) -> None:
    """The envelope reports what is wrong with the recording; it never guesses a clock."""
    with pytest.raises(AugmentationError, match=message):
        manual_boundary_envelope(times, dwell_start_s, ramp_duration_s=ramp_duration_s)


def test_mismatched_arrays_and_labels_are_refused() -> None:
    """The time base must match the positions, and a stream needs a non-empty label."""
    with pytest.raises(AugmentationError, match="shapes"):
        generate_manual_augmentation(
            T,
            _reference()[:-1],
            SCENARIO,
            _config(),
            dwell_start_s=DWELL_START_S,
            period_s=DT,
            derivatives=DERIVATIVES,
        )
    with pytest.raises(ValueError, match="label"):
        entropy_word("   ")


def test_results_refuse_inconsistent_episode_numbering() -> None:
    """The result is self-checking: episodes are numbered 1..n in increasing attempt order."""
    result = _generate()
    with pytest.raises(ValueError, match="increasing attempt order"):
        dataclasses.replace(result, episodes=tuple(reversed(result.episodes)))

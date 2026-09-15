# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-013 (I11): the first-sample point reflection keeps the recorded start without a stationary hold."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from arm_rc_ctrl.config import ConfigError, from_mapping, to_mapping
from arm_rc_ctrl.data.derivatives import DerivativeConfig, differentiate
from arm_rc_ctrl.data.manual import (
    HoldSettings,
    ManualDeriveConfig,
    ReflectionSettings,
    load_manual_derive_config,
    smooth_hold_anchored,
    smooth_start_reflected,
)
from arm_rc_ctrl.data.smoothing import SmoothingConfig, smooth
from arm_rc_ctrl.repo import repository_root

RATE_HZ = 100.0
PERIOD_S = 0.01
CONFIG = SmoothingConfig(method="butterworth", cutoff_hz=5.0, order=4)
CENTRAL = DerivativeConfig(method="central")
RESET: NDArray[np.float64] = np.array([0.2, 1.2])
AMPLITUDE: NDArray[np.float64] = np.array([0.6, -0.9])
TAU_S = 0.3
EXTENSION = 500  # 5 s on the 0.01 s grid, as declared by configs/preprocessing/manual_v2.toml
BOUND = 1e-12  # max_start_shift_rad of configs/preprocessing/manual_v2.toml
PREPROCESSING = repository_root() / "configs" / "preprocessing"


def _immediate_reach(duration_s: float = 4.0, reset: NDArray[np.float64] = RESET) -> NDArray[np.float64]:
    """A reach that moves from the first sample on: ``reset + A (1 - exp(-t / tau))``.

    Its exact joint speed ``A / tau`` and acceleration ``A / tau**2`` are largest at the first sample.
    """
    t = np.arange(round(duration_s / PERIOD_S) + 1, dtype=np.float64) * PERIOD_S
    return reset + (1.0 - np.exp(-t / TAU_S))[:, None] * AMPLITUDE


def test_reflected_filter_keeps_the_first_sample_of_a_take_that_moves_at_once() -> None:
    """The first sample stays at the recorded reset posture up to roundoff while the rest is smoothed."""
    q = _immediate_reach()
    result = smooth_start_reflected(q, RATE_HZ, CONFIG, extension_samples=EXTENSION)
    assert result.onset == 1
    assert result.start_shift_rad == float(np.max(np.abs(result.values[0] - q[0])))
    assert result.start_shift_rad <= BOUND
    from_origin = smooth_start_reflected(
        _immediate_reach(reset=np.zeros(2)), RATE_HZ, CONFIG, extension_samples=EXTENSION
    )
    assert from_origin.start_shift_rad < 1e-15  # the filtered deviation itself vanishes at the first sample
    assert not np.array_equal(result.values[1:], q[1:])
    inherited = smooth(q, RATE_HZ, CONFIG)
    assert np.allclose(result.values[200:-200], inherited[200:-200], rtol=0.0, atol=1e-12)  # same filter inside


def test_no_derivative_spike_at_the_first_sample_where_the_inherited_and_hold_anchored_filters_fail() -> None:
    """Smoothed speed and acceleration at the first sample stay within the raw motion's own derivatives.

    The hold-anchored filter extends a take that moves at once with zeros, so its
    smoothed acceleration at the first sample exceeds anything the recording holds and
    the start moves by centiradians; the inherited filter's short odd padding moves the
    start by more than 1e-4 rad.
    """
    q = _immediate_reach()
    peak_speed = np.abs(AMPLITUDE) / TAU_S
    peak_acceleration = np.abs(AMPLITUDE) / TAU_S**2
    reflected = smooth_start_reflected(q, RATE_HZ, CONFIG, extension_samples=EXTENSION)
    dq, ddq = differentiate(reflected.values, PERIOD_S, CENTRAL)
    assert np.all(np.abs(dq[0]) <= peak_speed)
    assert np.all(np.abs(ddq[0]) <= peak_acceleration)
    assert np.all(np.abs(dq[0]) <= 1.1 * np.max(np.abs(dq[1:6]), axis=0))
    assert np.all(np.abs(ddq[0]) <= np.max(np.abs(ddq[1:6]), axis=0))

    anchored = smooth_hold_anchored(q, RATE_HZ, CONFIG, reset=RESET, margin_samples=100)
    assert anchored.start_shift_rad > 1e-2
    _, ddq_anchored = differentiate(anchored.values, PERIOD_S, CENTRAL)
    assert np.all(np.abs(ddq_anchored[0]) > 2.0 * peak_acceleration)
    inherited = smooth(q, RATE_HZ, CONFIG)
    assert float(np.max(np.abs(inherited[0] - RESET))) > 1e-4


def test_pre_roll_fluctuations_are_kept_rather_than_replaced_by_a_constant() -> None:
    """A 5 mrad, 2 Hz wobble from the first sample on survives the 5 Hz filter before a later reach."""
    t = np.arange(301, dtype=np.float64) * PERIOD_S
    wobble = 0.005 * np.sin(2.0 * np.pi * 2.0 * t)
    s = np.clip((t - 0.5) / 1.0, 0.0, 1.0)
    reach = s**3 * (10.0 - 15.0 * s + 6.0 * s**2)
    q = RESET + wobble[:, None] * np.array([1.0, -0.5]) + reach[:, None] * AMPLITUDE
    result = smooth_start_reflected(q, RATE_HZ, CONFIG, extension_samples=EXTENSION)
    assert result.start_shift_rad <= BOUND
    pre_roll = slice(0, 50)
    assert float(np.max(np.abs(result.values[pre_roll] - q[pre_roll]))) < 5e-4
    raw_span = np.ptp(q[pre_roll], axis=0)
    assert np.all(np.ptp(result.values[pre_roll], axis=0) >= 0.9 * raw_span)


def test_the_extension_must_cover_the_filter_transient_and_may_exceed_a_short_take() -> None:
    """A 1 s extension leaves a measurable start shift (reported, not snapped); a 1.2 s take is extended as needed."""
    short_extension = smooth_start_reflected(_immediate_reach(), RATE_HZ, CONFIG, extension_samples=100)
    assert short_extension.start_shift_rad > BOUND
    brief = _immediate_reach(duration_s=1.2)
    covered = smooth_start_reflected(brief, RATE_HZ, CONFIG, extension_samples=EXTENSION)
    assert covered.values.shape == brief.shape
    assert covered.start_shift_rad <= BOUND


def test_stationary_unfiltered_and_invalid_inputs() -> None:
    """A take that never departs or is not filtered passes through; malformed inputs are refused."""
    still = np.tile(RESET, (50, 1))
    stationary = smooth_start_reflected(still, RATE_HZ, CONFIG, extension_samples=EXTENSION)
    assert np.array_equal(stationary.values, still)
    assert stationary.start_shift_rad == 0.0
    assert stationary.onset is None
    q = _immediate_reach(duration_s=1.0)
    unfiltered = smooth_start_reflected(q, RATE_HZ, SmoothingConfig(method="none"), extension_samples=EXTENSION)
    assert np.array_equal(unfiltered.values, q)
    assert unfiltered.start_shift_rad == 0.0
    with pytest.raises(ValueError, match="extension_samples"):
        smooth_start_reflected(q, RATE_HZ, CONFIG, extension_samples=0)
    with pytest.raises(ValueError, match="shape"):
        smooth_start_reflected(q[:, 0], RATE_HZ, CONFIG, extension_samples=EXTENSION)
    corrupt = q.copy()
    corrupt[3, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        smooth_start_reflected(corrupt, RATE_HZ, CONFIG, extension_samples=EXTENSION)
    with pytest.raises(ValueError, match="Nyquist"):
        smooth_start_reflected(
            q, RATE_HZ, SmoothingConfig(method="butterworth", cutoff_hz=60.0), extension_samples=EXTENSION
        )


def test_derivation_configurations_select_exactly_one_boundary_method() -> None:
    """v1 keeps the hold-anchored filter, v2 selects the reflection; the filter, grid, and derivatives are shared."""
    v1 = load_manual_derive_config(PREPROCESSING / "manual_v1.toml")
    v2 = load_manual_derive_config(PREPROCESSING / "manual_v2.toml")
    assert isinstance(v1.boundary, HoldSettings)
    assert v1.reflection is None
    assert isinstance(v2.boundary, ReflectionSettings)
    assert v2.hold is None
    assert v2.reflection == ReflectionSettings(extension_s=5.0, max_start_shift_rad=1e-12)
    assert (v2.smoothing, v2.resampling, v2.derivatives) == (v1.smoothing, v1.resampling, v1.derivatives)
    both = to_mapping(v2)
    both["hold"] = {"margin_s": 1.0, "max_start_shift_rad": 1e-9}
    with pytest.raises(ConfigError, match="exactly one"):
        from_mapping(both, ManualDeriveConfig)
    neither = to_mapping(v2)
    neither["reflection"] = None
    with pytest.raises(ConfigError, match="exactly one"):
        from_mapping(neither, ManualDeriveConfig)
    degenerate = to_mapping(v2)
    degenerate["reflection"] = {"extension_s": 0.0, "max_start_shift_rad": 1e-12}
    with pytest.raises(ConfigError, match="extension_s"):
        from_mapping(degenerate, ManualDeriveConfig)

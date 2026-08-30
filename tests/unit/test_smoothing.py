# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-005: zero-phase smoothing attenuates noise without measurable phase shift."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from arm_rc_ctrl.data.smoothing import SmoothingConfig, smooth

FS = 200.0
CONFIG = SmoothingConfig(method="butterworth", cutoff_hz=5.0, order=4)


def _time(n: int = 4000) -> NDArray[np.float64]:
    return np.arange(n, dtype=np.float64) / FS


def _fit_phase(t: NDArray[np.float64], x: NDArray[np.float64], hz: float) -> tuple[float, float]:
    """Least-squares amplitude and phase of a sinusoid at ``hz`` in ``x``."""
    basis = np.column_stack([np.sin(2 * np.pi * hz * t), np.cos(2 * np.pi * hz * t)])
    a, b = np.linalg.lstsq(basis, x, rcond=None)[0]
    return float(np.hypot(a, b)), float(np.arctan2(b, a))


def _interior(x: NDArray[np.float64], margin: int = 400) -> NDArray[np.float64]:
    return x[margin:-margin]


def test_passband_sinusoid_keeps_amplitude_and_phase() -> None:
    """A 0.5 Hz tone (well below the 5 Hz cutoff) passes with < 0.1 % amplitude loss and no phase shift."""
    t = _time()
    x = np.sin(2 * np.pi * 0.5 * t)
    y = smooth(x, FS, CONFIG)
    amp, phase = _fit_phase(_interior(t), _interior(y), 0.5)
    assert abs(amp - 1.0) < 1e-3
    assert abs(phase) < 1e-4
    peak_in = int(np.argmax(_interior(x)))
    peak_out = int(np.argmax(_interior(y)))
    assert peak_in == peak_out


def test_stopband_noise_is_attenuated_by_the_squared_response() -> None:
    """A 40 Hz tone is attenuated below 1e-3 (the forward-backward pass squares |H|)."""
    t = _time()
    x = np.sin(2 * np.pi * 0.5 * t) + 0.5 * np.sin(2 * np.pi * 40.0 * t)
    y = smooth(x, FS, CONFIG)
    amp_noise, _ = _fit_phase(_interior(t), _interior(y), 40.0)
    amp_signal, phase_signal = _fit_phase(_interior(t), _interior(y), 0.5)
    assert amp_noise < 1e-3 * 0.5
    assert abs(amp_signal - 1.0) < 1e-3
    assert abs(phase_signal) < 1e-4


def test_zero_phase_symmetry_on_an_impulse() -> None:
    """The impulse response is symmetric about the impulse, the signature of zero phase."""
    x = np.zeros(2001)
    x[1000] = 1.0
    y = smooth(x, FS, CONFIG)
    assert np.allclose(y[1000 - 300 : 1000], y[1000 + 1 : 1000 + 301][::-1], atol=1e-12)
    assert int(np.argmax(y)) == 1000


def test_constant_and_linear_signals_are_preserved_in_the_interior() -> None:
    """DC and ramps pass unchanged away from the edges."""
    t = _time(2000)
    for x in (np.full_like(t, 3.0), 2.0 * t - 1.0):
        y = smooth(x, FS, CONFIG)
        assert np.allclose(_interior(y, 200), _interior(x, 200), atol=1e-9)


def test_columns_are_filtered_independently_and_identically() -> None:
    """A 2-D signal is treated column by column."""
    t = _time(1000)
    a = np.sin(2 * np.pi * 1.0 * t)
    b = np.cos(2 * np.pi * 2.0 * t)
    stacked = smooth(np.column_stack([a, b]), FS, CONFIG)
    assert np.array_equal(stacked[:, 0], smooth(a, FS, CONFIG))
    assert np.array_equal(stacked[:, 1], smooth(b, FS, CONFIG))
    assert stacked.dtype == np.float64
    assert stacked.flags["C_CONTIGUOUS"]


def test_none_method_returns_an_equal_copy() -> None:
    """Disabling smoothing yields a copy, never the same object."""
    x = np.arange(50, dtype=np.float64)
    y = smooth(x, FS, SmoothingConfig(method="none"))
    assert np.array_equal(x, y)
    assert y is not x
    assert SmoothingConfig(method="none").label == "none"
    assert SmoothingConfig(method="none").parameters() == {}
    assert CONFIG.label == "butterworth-zero-phase"
    assert CONFIG.parameters() == {"order": 4.0, "cutoff_hz": 5.0}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cutoff_hz": 0.0}, "cutoff_hz must be positive"),
        ({"cutoff_hz": float("inf")}, "cutoff_hz must be positive and finite"),
        ({"order": 0}, r"order must be in \[1, 16\]"),
        ({"order": 17}, r"order must be in \[1, 16\]"),
    ],
)
def test_configuration_is_validated(kwargs: dict[str, float], message: str) -> None:
    """Cutoff and order are range-checked."""
    with pytest.raises(ValueError, match=message):
        SmoothingConfig(**kwargs)  # type: ignore[arg-type]


def test_invalid_inputs_are_rejected_not_repaired() -> None:
    """Non-finite data, cutoff at or above Nyquist, bad rates, and short signals fail."""
    t = _time(500)
    x = np.sin(t)
    bad = x.copy()
    bad[10] = np.nan
    with pytest.raises(ValueError, match="never repairs"):
        smooth(bad, FS, CONFIG)
    with pytest.raises(ValueError, match=r"below the Nyquist frequency 100\.0 Hz"):
        smooth(x, FS, SmoothingConfig(cutoff_hz=100.0))
    with pytest.raises(ValueError, match="sample_rate_hz must be positive"):
        smooth(x, 0.0, CONFIG)
    with pytest.raises(ValueError, match="needs more than"):
        smooth(x[:10], FS, CONFIG)
    with pytest.raises(ValueError, match=r"shape \(N,\) or \(N, k\)"):
        smooth(np.zeros((2, 2, 2)), FS, CONFIG)
    with pytest.raises(ValueError, match="N > 0"):
        smooth(np.zeros(0), FS, CONFIG)

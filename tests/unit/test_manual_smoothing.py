# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-002 (I9): the inherited zero-phase filter shifts a held start; the hold-anchored filter keeps it exact."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from arm_rc_ctrl.data.manual import smooth_hold_anchored
from arm_rc_ctrl.data.smoothing import SmoothingConfig, smooth

RATE_HZ = 100.0
CONFIG = SmoothingConfig(method="butterworth", cutoff_hz=5.0, order=4)
RESET: NDArray[np.float64] = np.array([0.2, 1.2])


def _hold_then_reach(hold_s: float, move_s: float = 1.0, dwell_s: float = 1.5) -> NDArray[np.float64]:
    """A stationary hold at the reset posture, a minimum-jerk reach of ~1 rad, then a stationary dwell."""
    n_hold, n_move, n_dwell = (round(x * RATE_HZ) for x in (hold_s, move_s, dwell_s))
    tau = np.linspace(0.0, 1.0, n_move + 1)[1:]
    profile = tau**3 * (10.0 - 15.0 * tau + 6.0 * tau**2)
    amplitude: NDArray[np.float64] = np.array([0.6, -0.9])
    q: NDArray[np.float64] = np.concatenate(
        [np.tile(RESET, (n_hold, 1)), RESET + profile[:, None] * amplitude, np.tile(RESET + amplitude, (n_dwell, 1))]
    )
    return q


@pytest.mark.parametrize(("hold_s", "low", "high"), [(0.1, 1e-5, 1e-3), (1.0, 1e-11, 1e-8)])
def test_inherited_filter_shifts_the_recorded_start(hold_s: float, low: float, high: float) -> None:
    """Reproduces the 2026-09-15 finding: the odd-extension filter moves the first sample of a held start."""
    q = _hold_then_reach(hold_s)
    shift = float(np.max(np.abs(smooth(q, RATE_HZ, CONFIG)[0] - RESET)))
    assert low < shift < high


def test_hold_anchored_filter_keeps_the_hold_bitwise_when_it_is_longer_than_the_margin() -> None:
    """With a 2 s hold and a 1 s margin the first second is untouched and the start shift is exactly zero."""
    q = _hold_then_reach(2.0)
    result = smooth_hold_anchored(q, RATE_HZ, CONFIG, reset=RESET, margin_samples=100)
    assert result.start_shift_rad == 0.0
    assert result.anchored_from == 100  # onset at sample 200 minus the margin
    assert np.array_equal(result.values[:100], q[:100])
    inherited = smooth(q, RATE_HZ, CONFIG)
    assert np.allclose(result.values[150:-150], inherited[150:-150], atol=1e-9)  # same smoothing away from the edges
    assert np.max(np.abs(result.values[-1] - q[-1])) < 1e-9  # constant-extended stationary tail


@pytest.mark.parametrize("hold_s", [0.2, 0.5])
def test_hold_anchored_filter_measures_the_onset_leakage_of_a_short_hold(hold_s: float) -> None:
    """A hold shorter than the margin cannot be preserved: the measured shift is the onset leakage itself.

    Beyond the filter's 15-sample pad the inherited odd extension of a constant hold is that
    constant, so both methods leak identically; the shift is a property of zero-phase
    smoothing near an onset, not of the padding.
    """
    q = _hold_then_reach(hold_s)
    result = smooth_hold_anchored(q, RATE_HZ, CONFIG, reset=RESET, margin_samples=100)
    assert result.anchored_from == 0
    assert result.start_shift_rad > 1e-7
    inherited_shift = float(np.max(np.abs(smooth(q, RATE_HZ, CONFIG)[0] - RESET)))
    assert result.start_shift_rad == pytest.approx(inherited_shift, rel=1e-9)


def test_zero_extension_removes_the_padding_artifact_of_a_very_short_hold() -> None:
    """Within the pad length the odd extension mirrors motion into the pad; the anchored filter does not."""
    q = _hold_then_reach(0.1)
    result = smooth_hold_anchored(q, RATE_HZ, CONFIG, reset=RESET, margin_samples=100)
    inherited_shift = float(np.max(np.abs(smooth(q, RATE_HZ, CONFIG)[0] - RESET)))
    assert 1e-5 < result.start_shift_rad < inherited_shift


def test_constant_and_invalid_inputs() -> None:
    """A stationary take passes through bitwise; a reset of the wrong width or a mismatch is refused."""
    q = np.tile(RESET, (250, 1))
    result = smooth_hold_anchored(q, RATE_HZ, CONFIG, reset=RESET, margin_samples=100)
    assert np.array_equal(result.values, q)
    assert result.anchored_from == 250  # never departs: nothing to filter
    with pytest.raises(ValueError, match="reset"):
        smooth_hold_anchored(q, RATE_HZ, CONFIG, reset=np.array([0.2]), margin_samples=100)
    with pytest.raises(ValueError, match="first sample"):
        smooth_hold_anchored(q + 1e-3, RATE_HZ, CONFIG, reset=RESET, margin_samples=100)

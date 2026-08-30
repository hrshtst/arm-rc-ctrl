# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-009: normalization statistics come from training rows only; near-zero scales become 1.0."""

from __future__ import annotations

import numpy as np
import pytest

from arm_rc_ctrl.data.normalization import Normalizer, fit_normalization
from arm_rc_ctrl.data.records import ChannelStats, Normalization
from arm_rc_ctrl.data.synthetic import synthetic_arrays

SOURCE = ("processed-20260830-555555555555",)


def _arrays() -> dict[str, np.ndarray]:
    arrays = synthetic_arrays(n=20, dof=2, task_dim=2, code_dim=0)
    arrays["q"] = np.column_stack([np.linspace(-1.0, 1.0, 20), np.full(20, 0.25)])  # second column constant
    return arrays


def test_statistics_match_numpy_on_training_rows_only() -> None:
    """Mean and population standard deviation are computed over the masked rows."""
    arrays = _arrays()
    mask = np.zeros(20, dtype=bool)
    mask[:10] = True
    norm = fit_normalization(arrays, ("q", "tip"), fitted_on=SOURCE, training_rows=mask)
    q_train = arrays["q"][:10]
    assert norm.fitted_on == SOURCE
    assert np.allclose(norm.channels["q"].mean, q_train.mean(axis=0))
    assert norm.channels["q"].scale[0] == pytest.approx(float(q_train[:, 0].std()))
    assert norm.channels["tip"].replaced_near_zero == ()
    assert set(norm.channels) == {"q", "tip"}


def test_evaluation_rows_cannot_leak_into_the_statistics() -> None:
    """Changing evaluation rows arbitrarily leaves the fitted statistics identical."""
    arrays = _arrays()
    mask = np.zeros(20, dtype=bool)
    mask[:12] = True
    reference = fit_normalization(arrays, ("q", "tip", "dq"), fitted_on=SOURCE, training_rows=mask)
    leaked = {k: v.copy() for k, v in arrays.items()}
    leaked["q"][12:] = 1e6
    leaked["tip"][12:] = -1e6
    leaked["dq"][12:] = np.nan  # even NaN outside the training rows is irrelevant
    assert fit_normalization(leaked, ("q", "tip", "dq"), fitted_on=SOURCE, training_rows=mask) == reference
    everything = np.ones(20, dtype=bool)
    assert fit_normalization(arrays, ("q",), fitted_on=SOURCE, training_rows=everything) != reference


def test_near_zero_scale_is_replaced_and_reported() -> None:
    """A constant column gets scale 1.0 and its index is recorded; the threshold is configurable."""
    arrays = _arrays()
    mask = np.ones(20, dtype=bool)
    norm = fit_normalization(arrays, ("q",), fitted_on=SOURCE, training_rows=mask)
    assert norm.channels["q"].scale[1] == 1.0
    assert norm.channels["q"].replaced_near_zero == (1,)
    assert norm.channels["q"].scale[0] > 0.5

    tiny = {**arrays, "q": np.column_stack([arrays["q"][:, 0] * 1e-6, arrays["q"][:, 1]])}
    strict = fit_normalization(tiny, ("q",), fitted_on=SOURCE, training_rows=mask, near_zero=0.0)
    assert strict.channels["q"].replaced_near_zero == (1,)
    loose = fit_normalization(tiny, ("q",), fitted_on=SOURCE, training_rows=mask, near_zero=1e-3)
    assert loose.channels["q"].replaced_near_zero == (0, 1)
    assert loose.channels["q"].scale == (1.0, 1.0)


def test_transform_and_inverse_round_trip() -> None:
    """Normalized training data has zero mean/unit scale (except replaced columns); inverse restores values."""
    arrays = _arrays()
    mask = np.ones(20, dtype=bool)
    normalizer = Normalizer(fit_normalization(arrays, ("q", "tip"), fitted_on=SOURCE, training_rows=mask))
    z = normalizer.transform("q", arrays["q"])
    assert np.allclose(z.mean(axis=0), 0.0)
    assert z[:, 0].std() == pytest.approx(1.0)
    assert np.allclose(z[:, 1], 0.0)  # constant column: (x - mean) / 1.0 == 0
    assert np.allclose(normalizer.inverse("q", z), arrays["q"])
    single = normalizer.transform("tip", arrays["tip"][3])
    assert single.shape == (2,)
    assert np.allclose(normalizer.inverse("tip", single), arrays["tip"][3])


def test_zero_width_channels_are_skipped() -> None:
    """A task_code with dimension 0 yields no statistics."""
    arrays = _arrays()
    norm = fit_normalization(arrays, ("task_code", "q"), fitted_on=SOURCE, training_rows=np.ones(20, dtype=bool))
    assert set(norm.channels) == {"q"}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"training_rows": np.zeros(20, dtype=bool)}, "selects no samples"),
        ({"training_rows": np.ones(20)}, "1-D boolean mask"),
        ({"training_rows": np.ones(19, dtype=bool)}, r"channel 'q' must have shape \(19, k\)"),
        ({"training_rows": np.ones(20, dtype=bool), "channels": ("missing",)}, "not in the dataset"),
        ({"training_rows": np.ones(20, dtype=bool), "near_zero": -1.0}, "near_zero must be non-negative"),
    ],
)
def test_fit_rejects_bad_inputs(kwargs: dict[str, object], message: str) -> None:
    """Masks, channels, and thresholds are validated."""
    arrays = _arrays()
    channels = kwargs.pop("channels", ("q",))
    with pytest.raises(ValueError, match=message):
        fit_normalization(arrays, channels, fitted_on=SOURCE, **kwargs)  # type: ignore[arg-type]


def test_non_finite_training_values_are_rejected() -> None:
    """NaN inside the training rows is an error, never silently dropped."""
    arrays = _arrays()
    arrays["q"][3, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite training values"):
        fit_normalization(arrays, ("q",), fitted_on=SOURCE, training_rows=np.ones(20, dtype=bool))


def test_normalizer_validates_channel_and_width() -> None:
    """Unknown channels and wrong widths are errors."""
    normalizer = Normalizer(Normalization(SOURCE, {"q": ChannelStats((0.0, 0.0), (1.0, 2.0))}))
    with pytest.raises(KeyError, match="no normalization statistics for channel 'tip'"):
        normalizer.transform("tip", np.zeros((3, 2)))
    with pytest.raises(ValueError, match="expects 2 columns"):
        normalizer.transform("q", np.zeros((3, 3)))

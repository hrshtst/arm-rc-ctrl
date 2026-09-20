# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-028: finite-duration endpoint force pulses."""

from __future__ import annotations

import math

import numpy as np
import pytest

from arm_rc_ctrl.experiments.disturbances import FORCE_PULSE_KIND, ForcePulse
from arm_rc_ctrl.experiments.run_record import Disturbance


def test_pulse_acts_only_inside_its_half_open_window() -> None:
    """The force is constant on [start, end) and zero elsewhere."""
    pulse = ForcePulse(start_s=2.0, duration_s=0.2, force=(3.0, -4.0))
    assert pulse.end_s == pytest.approx(2.2)
    assert pulse.magnitude_n == pytest.approx(5.0)
    assert not pulse.active(1.99)
    assert pulse.active(2.0)
    assert pulse.active(2.19)
    assert not pulse.active(2.2)
    assert np.array_equal(pulse.at(2.1), np.array([3.0, -4.0]))
    assert np.array_equal(pulse.at(0.0), np.zeros(2))
    assert pulse.at(2.1).dtype == np.float64


@pytest.mark.parametrize(
    ("direction_deg", "expected"),
    [(0.0, (2.0, 0.0)), (90.0, (0.0, 2.0)), (180.0, (-2.0, 0.0)), (270.0, (0.0, -2.0)), (45.0, (math.sqrt(2),) * 2)],
)
def test_polar_construction(direction_deg: float, expected: tuple[float, float]) -> None:
    """Magnitude and direction from the base x axis give the Cartesian force."""
    pulse = ForcePulse.from_polar(1.0, 0.5, 2.0, direction_deg)
    assert pulse.force == pytest.approx(expected, abs=1e-12)
    assert pulse.magnitude_n == pytest.approx(2.0)


def test_disturbance_record() -> None:
    """The run-record description names the kind, window, components, and magnitude."""
    record = ForcePulse.from_polar(2.0, 0.2, 8.0, 270.0).to_disturbance()
    assert record.kind == FORCE_PULSE_KIND
    assert (record.start_s, record.end_s) == (2.0, pytest.approx(2.2))
    assert record.parameters == pytest.approx({"fx": 0.0, "fy": -8.0, "magnitude_n": 8.0}, abs=1e-12)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"start_s": -0.1, "duration_s": 0.2, "force": (1.0, 0.0)}, "start_s must be >= 0"),
        ({"start_s": 0.0, "duration_s": 0.0, "force": (1.0, 0.0)}, "duration_s must be > 0"),
        ({"start_s": 0.0, "duration_s": 0.2, "force": (1.0, 0.0, 0.0)}, "force must have 2 components"),
        ({"start_s": math.nan, "duration_s": 0.2, "force": (1.0, 0.0)}, "force pulse"),
        ({"start_s": 0.0, "duration_s": 0.2, "force": (math.inf, 0.0)}, "force pulse"),
    ],
)
def test_invalid_pulses_are_rejected(kwargs: dict[str, object], message: str) -> None:
    """Timing and force are validated on construction."""
    with pytest.raises(ValueError, match=message):
        ForcePulse(**kwargs)  # type: ignore[arg-type]


def test_negative_magnitude_is_rejected() -> None:
    """A negative magnitude would silently flip the direction."""
    with pytest.raises(ValueError, match="magnitude_n must be >= 0"):
        ForcePulse.from_polar(0.0, 0.1, -1.0, 0.0)


def test_a_pulse_reads_back_from_a_run_record_with_its_window_intact() -> None:
    """A recorded run says which pulse actually fired; judging that run again needs the same window.

    The window is what the verdict depends on, and ``duration_s`` is stored as
    the difference of two recorded instants, so the reader is held to the
    instants rather than to a duration it would have to round back.
    """
    pulse = ForcePulse(start_s=1.25, duration_s=0.2, force=(3.0, -4.0))
    recovered = ForcePulse.from_disturbance(pulse.to_disturbance())
    assert (recovered.start_s, recovered.end_s, recovered.force) == (pulse.start_s, pulse.end_s, pulse.force)


def test_a_description_of_something_else_is_refused() -> None:
    """Reading a pulse out of a record must not invent one from an unrelated disturbance."""
    with pytest.raises(ValueError, match="endpoint_force_pulse"):
        ForcePulse.from_disturbance(Disturbance("joint_offset", 1.0, 1.2, {"fx": 1.0, "fy": 0.0}))
    with pytest.raises(ValueError, match="fx"):
        ForcePulse.from_disturbance(Disturbance(FORCE_PULSE_KIND, 1.0, 1.2, {"magnitude_n": 2.0}))


def test_a_description_whose_magnitude_disagrees_with_its_components_is_refused() -> None:
    """The recorded magnitude is checked against the components, not trusted beside them."""
    described = ForcePulse(start_s=1.0, duration_s=0.2, force=(3.0, 4.0)).to_disturbance()
    described.parameters["magnitude_n"] = 99.0
    with pytest.raises(ValueError, match="magnitude"):
        ForcePulse.from_disturbance(described)

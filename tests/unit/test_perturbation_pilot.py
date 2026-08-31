# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-028: pilot protocol validation, level classification, and selection rules."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from arm_rc_ctrl.config import ConfigError, from_mapping, to_mapping
from arm_rc_ctrl.experiments.perturbation_pilot import (
    ForceSweep,
    PilotCase,
    PilotProtocol,
    PostureSweep,
    Selection,
    SelectionRules,
    load_protocol,
    select_levels,
    summarize_levels,
)
from arm_rc_ctrl.experiments.termination import Termination, completed, limit_violation
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from arm_rc_ctrl.experiments.perturbation_pilot import PerturbationKind

SCENARIO = repository_root() / "configs" / "tasks" / "task_1a.toml"
RULES = SelectionRules(
    posture_recovery_min_s=0.1,
    posture_recovery_max_s=1.0,
    force_recovery_max_s=1.0,
    force_deviation_min_m=0.01,
    force_max_saturation_fraction=0.0,
)


def _protocol(**overrides: object) -> PilotProtocol:
    base = PilotProtocol(
        name="unit",
        scenario=SCENARIO,
        baselines=("pd", "computed_torque"),
        posture=PostureSweep((0.05, 0.1, 0.2), ((1.0, 0.0), (0.0, -1.0))),
        force=ForceSweep((1.0, 4.0, 16.0), (0.0, 180.0), 2.0, 0.2),
        selection=RULES,
    )
    return dataclasses.replace(base, **overrides)


def _case(
    baseline: str,
    kind: PerturbationKind,
    magnitude: float,
    *,
    termination: Termination | None = None,
    success: bool = True,
    recovery: float | None = 0.05,
    deviation: float = 0.005,
    saturation: float = 0.0,
) -> PilotCase:
    termination = completed(5.0, 500) if termination is None else termination
    return PilotCase(
        baseline=baseline,
        kind=kind,
        magnitude=magnitude,
        direction=(1.0, 0.0) if kind == "posture" else (0.0,),
        initial_q=(0.2, 1.2),
        termination=termination,
        criteria={"completed": termination.is_completed, "dwell_in_tolerance": success, "dwell_stationary": success},
        success=success and termination.is_completed,
        move_joint_rmse=1e-4 if termination.is_completed else None,
        peak_deviation_m=deviation,
        recovery_time_s=recovery,
        peak_torque_fraction=0.5,
        saturation_fraction=saturation,
        peak_velocity=1.0,
    )


def test_committed_pilot_protocol_loads() -> None:
    """The tracked protocol is valid and refers to the task 1-a scenario with both frozen baselines."""
    protocol = load_protocol(repository_root() / "configs" / "studies" / "perturbation_pilot_1a.toml")
    assert protocol.scenario == SCENARIO
    assert protocol.baselines == ("pd", "computed_torque")
    assert protocol.force.start_s == 2.0  # inside the movement interval [1, 4] s
    assert protocol.force.start_s + protocol.force.duration_s < 4.0
    assert protocol.selection.force_max_saturation_fraction == 0.0


def test_posture_directions_are_normalized() -> None:
    """Unit vectors keep the magnitude meaningful in every direction."""
    sweep = PostureSweep((0.1,), ((3.0, 4.0), (0.0, -2.0)))
    assert np.allclose(sweep.unit(0), [0.6, 0.8])
    assert np.allclose(sweep.unit(1), [0.0, -1.0])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"magnitudes": (0.1, 0.1), "directions": ((1.0, 0.0),)}, "strictly increasing"),
        ({"magnitudes": (0.0, 0.1), "directions": ((1.0, 0.0),)}, "positive and strictly increasing"),
        ({"magnitudes": (), "directions": ((1.0, 0.0),)}, "positive and strictly increasing"),
        ({"magnitudes": (0.1,), "directions": ()}, "must not be empty"),
        ({"magnitudes": (0.1,), "directions": ((0.0, 0.0),)}, "non-zero vectors"),
        ({"magnitudes": (0.1,), "directions": ((1.0, 0.0), (1.0,))}, "one dimension"),
        ({"magnitudes": (math.nan,), "directions": ((1.0, 0.0),)}, "posture.magnitudes"),
    ],
)
def test_invalid_posture_sweeps(kwargs: dict[str, object], message: str) -> None:
    """Grids are validated on construction."""
    with pytest.raises(ValueError, match=message):
        PostureSweep(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"magnitudes": (4.0, 2.0), "directions_deg": (0.0,), "start_s": 2.0, "duration_s": 0.2}, "increasing"),
        ({"magnitudes": (2.0,), "directions_deg": (), "start_s": 2.0, "duration_s": 0.2}, "must not be empty"),
        ({"magnitudes": (2.0,), "directions_deg": (0.0,), "start_s": -1.0, "duration_s": 0.2}, "start_s must be >= 0"),
        ({"magnitudes": (2.0,), "directions_deg": (0.0,), "start_s": 2.0, "duration_s": 0.0}, "duration_s must be > 0"),
    ],
)
def test_invalid_force_sweeps(kwargs: dict[str, object], message: str) -> None:
    """Force grids and pulse windows are validated on construction."""
    with pytest.raises(ValueError, match=message):
        ForceSweep(**kwargs)  # type: ignore[arg-type]


def test_force_sweep_builds_pulses() -> None:
    """Each case's pulse has the sweep's window and the requested polar force."""
    pulse = ForceSweep((2.0,), (90.0,), 2.0, 0.2).pulse(2.0, 90.0)
    assert (pulse.start_s, pulse.duration_s) == (2.0, 0.2)
    assert pulse.force == pytest.approx((0.0, 2.0), abs=1e-12)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"posture_recovery_min_s": 0.0}, "must be positive"),
        ({"posture_recovery_min_s": 2.0}, "must not exceed"),
        ({"force_deviation_min_m": -0.01}, "must be positive"),
        ({"force_max_saturation_fraction": 1.5}, r"must lie in \[0, 1\]"),
        ({"force_recovery_max_s": math.inf}, "selection"),
    ],
)
def test_invalid_rules(overrides: dict[str, float], message: str) -> None:
    """Rule bounds are positive, finite, and ordered."""
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(RULES, **overrides)


def test_invalid_protocols(tmp_path: Path) -> None:
    """Empty names, duplicate baselines, and unknown keys are rejected."""
    with pytest.raises(ValueError, match="name must not be empty"):
        _protocol(name=" ")
    with pytest.raises(ValueError, match="distinct methods"):
        _protocol(baselines=("pd", "pd"))
    bad = tmp_path / "pilot.toml"
    bad.write_text('name = "x"\nscenario = "s.toml"\nbaselines = ["pd"]\nunknown = 1\n')
    with pytest.raises(ConfigError):
        load_protocol(bad)


def _grid_cases() -> list[PilotCase]:
    """Cases for ``_protocol()``: posture 0.05 trivial / 0.1 nontrivial / 0.2 unsafe; force 1 / 4 / 16 (saturating)."""
    cases: list[PilotCase] = []
    for baseline in ("pd", "computed_torque"):
        cases += [
            _case(baseline, "posture", 0.05, recovery=0.02),
            _case(baseline, "posture", 0.1, recovery=0.3 if baseline == "pd" else 0.05),
            _case(baseline, "posture", 0.2, recovery=1.5 if baseline == "pd" else 0.4),
            _case(baseline, "force", 1.0, deviation=0.004),
            _case(baseline, "force", 4.0, deviation=0.02 if baseline == "computed_torque" else 0.006, recovery=0.3),
            _case(baseline, "force", 16.0, deviation=0.05, saturation=0.01 if baseline == "pd" else 0.0),
        ]
    return cases


def test_levels_are_classified_by_the_rules() -> None:
    """Safe needs every baseline within the rules; nontrivial needs one baseline past the threshold."""
    protocol = _protocol()
    levels = summarize_levels(protocol, _grid_cases())
    by_key = {(lv.kind, lv.magnitude): lv for lv in levels}
    assert [(lv.kind, lv.magnitude) for lv in levels] == [
        ("posture", 0.05),
        ("posture", 0.1),
        ("posture", 0.2),
        ("force", 1.0),
        ("force", 4.0),
        ("force", 16.0),
    ]
    assert (by_key["posture", 0.05].safe, by_key["posture", 0.05].nontrivial) == (True, False)
    assert (by_key["posture", 0.1].safe, by_key["posture", 0.1].nontrivial) == (True, True)
    assert (by_key["posture", 0.2].safe, by_key["posture", 0.2].nontrivial) == (False, True)  # PD recovers too late
    assert (by_key["force", 1.0].safe, by_key["force", 1.0].nontrivial) == (True, False)
    assert (by_key["force", 4.0].safe, by_key["force", 4.0].nontrivial) == (True, True)
    assert (by_key["force", 16.0].safe, by_key["force", 16.0].nontrivial) == (False, True)  # PD saturates
    pd = by_key["posture", 0.2].baselines["pd"]
    assert pd.max_recovery_time_s == 1.5
    assert pd.terminations == ("completed",)
    assert select_levels(protocol, levels) == Selection(0.1, 0.1, 4.0, 2.0, 0.2, (0.0, 180.0))


def test_large_posture_level_is_the_largest_safe_one() -> None:
    """With a wider safe band the large level exceeds the small one; a relaxed saturation bound admits 16 N."""
    protocol = _protocol(
        selection=dataclasses.replace(RULES, posture_recovery_max_s=2.0, force_max_saturation_fraction=0.05)
    )
    levels = summarize_levels(protocol, _grid_cases())
    assert select_levels(protocol, levels) == Selection(0.1, 0.2, 16.0, 2.0, 0.2, (0.0, 180.0))


def test_unrecovered_or_failed_cases_make_a_level_unsafe() -> None:
    """A run that never returns within tolerance, or fails a criterion, cannot be safe even when it completes."""
    protocol = _protocol(posture=PostureSweep((0.1,), ((1.0, 0.0),)), force=ForceSweep((4.0,), (0.0,), 2.0, 0.2))
    cases = [
        _case("pd", "posture", 0.1, recovery=None),
        _case("computed_torque", "posture", 0.1),
        _case("pd", "force", 4.0, success=False, deviation=0.02),
        _case("computed_torque", "force", 4.0, deviation=0.02),
    ]
    levels = summarize_levels(protocol, cases)
    assert [lv.safe for lv in levels] == [False, False]
    assert levels[0].baselines["pd"].recovered_all is False
    assert levels[0].baselines["pd"].max_recovery_time_s is None
    assert levels[0].nontrivial is True  # an unrecovered baseline is a visible effect
    with pytest.raises(ValueError, match="no safe and nontrivial posture level"):
        select_levels(protocol, levels)


def test_missing_cases_are_an_error() -> None:
    """Every level needs cases for every baseline."""
    protocol = _protocol(posture=PostureSweep((0.1,), ((1.0, 0.0),)), force=ForceSweep((4.0,), (0.0,), 2.0, 0.2))
    with pytest.raises(ValueError, match="no cases for force level 4"):
        summarize_levels(protocol, [_case("pd", "posture", 0.1), _case("computed_torque", "posture", 0.1)])


def test_selection_orders_posture_levels() -> None:
    """The small level never exceeds the large one."""
    with pytest.raises(ValueError, match="posture_small_rad must not exceed"):
        Selection(0.2, 0.1, 4.0, 2.0, 0.2, (0.0,))


VIOLATION = limit_violation(0.01, 1, "joint_velocity", 6.045592, 6.0, joint=1)


def test_levels_keep_the_earliest_failure_with_its_diagnostics() -> None:
    """A level records the earliest non-completed termination in full, not just its kind."""
    protocol = _protocol(
        posture=PostureSweep((0.07,), ((1.0, 0.0), (0.0, 1.0))), force=ForceSweep((4.0,), (0.0,), 2.0, 0.2)
    )
    later = limit_violation(0.03, 3, "joint_velocity", 6.2, 6.0, joint=0)
    cases = [
        _case("pd", "posture", 0.07, termination=later, success=False, recovery=None),
        _case("pd", "posture", 0.07, termination=VIOLATION, success=False, recovery=None),
        _case("computed_torque", "posture", 0.07, recovery=0.23),
        _case("pd", "force", 4.0, deviation=0.02),
        _case("computed_torque", "force", 4.0, deviation=0.02),
    ]
    (posture, force) = summarize_levels(protocol, cases)
    pd = posture.baselines["pd"]
    assert pd.terminations == ("limit_violation",)
    assert pd.first_failure == VIOLATION
    assert pd.first_failure is not None
    assert (pd.first_failure.limit, pd.first_failure.joint) == ("joint_velocity", 1)
    assert (pd.first_failure.value, pd.first_failure.bound, pd.first_failure.time_s) == (6.045592, 6.0, 0.01)
    assert posture.baselines["computed_torque"].first_failure is None
    assert posture.safe is False
    assert force.baselines["pd"].first_failure is None


def test_cases_round_trip_with_their_complete_termination() -> None:
    """The report mapping (JSON) preserves every termination field of a case."""
    case = _case("pd", "posture", 0.07, termination=VIOLATION, success=False, recovery=None)
    mapping = to_mapping(case)
    assert mapping["termination"] == {
        "kind": "limit_violation",
        "time_s": 0.01,
        "step": 1,
        "detail": "joint_velocity on joint 1: 6.045592 exceeds bound 6.0",
        "limit": "joint_velocity",
        "joint": 1,
        "value": 6.045592,
        "bound": 6.0,
        "failure": None,
    }
    assert from_mapping(mapping, PilotCase) == case

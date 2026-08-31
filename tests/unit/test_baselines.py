# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-027: frozen baseline registry and replay snapshot comparison."""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.experiments import baselines
from arm_rc_ctrl.experiments.baselines import (
    BaselineExpectations,
    ReplaySnapshot,
    Tolerances,
    baseline_method,
    compare_snapshots,
    frozen_baseline_digest,
    frozen_baseline_file,
    load_expectations,
    load_frozen_baseline,
    write_expectations,
)
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from pathlib import Path

CRITERIA = {"completed": True, "dwell_in_tolerance": True, "dwell_stationary": True}
DWELL = {
    "in_tolerance_fraction": 1.0,
    "longest_in_tolerance_s": 1.0,
    "endpoint_rms": 2e-3,
    "endpoint_max": 3e-3,
    "velocity_max": 0.01,
}
EFFORT = {"torque_rms": 0.5, "torque_peak": 2.0, "saturation_fraction": 0.0, "effort": 1.5}


def _snapshot(**overrides: object) -> ReplaySnapshot:
    base = ReplaySnapshot(
        method="replay+pd",
        termination="completed",
        success=True,
        criteria=dict(CRITERIA),
        n_samples=501,
        joint_rmse=1e-3,
        joint_rmse_per_joint=(1.2e-3, 0.7e-3),
        dwell=dict(DWELL),
        effort=dict(EFFORT),
        final_q=(0.83, 1.16),
        final_dq=(0.0, 0.0),
        final_tip=(0.10, 0.45),
    )
    return dataclasses.replace(base, **overrides)


@pytest.mark.parametrize("method", sorted(baselines.FROZEN_BASELINES))
def test_frozen_baselines_load_with_their_declared_method(method: str) -> None:
    """Each registered method resolves to a tracked file declaring that tracker type with positive gains."""
    file = frozen_baseline_file(method)
    assert file.is_relative_to(repository_root())
    config = load_frozen_baseline(method)
    assert config.type == baseline_method(method)
    assert config.dof == 2
    assert all(value > 0 for value in (*config.kp, *config.kd))
    assert len(frozen_baseline_digest(method)) == 64


def test_unknown_method_is_rejected() -> None:
    """Asking for gains that were never frozen names the known methods."""
    with pytest.raises(KeyError, match="no frozen baseline for method 'lqr'; known methods"):
        frozen_baseline_file("lqr")


def test_type_mismatch_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A frozen file declaring another tracker type than its registry key cannot be loaded."""
    wrong = tmp_path / "wrong.toml"
    wrong.write_text('type = "computed_torque"\nkp = [1.0, 1.0]\nkd = [1.0, 1.0]\n')
    monkeypatch.setitem(baselines.FROZEN_BASELINES, "pd", str(wrong))
    with pytest.raises(ValueError, match="declares tracker type 'computed_torque', expected 'pd'"):
        load_frozen_baseline("pd")


def test_identical_snapshots_have_no_mismatches() -> None:
    """A reproduced run compares clean."""
    assert compare_snapshots(_snapshot(), _snapshot(), Tolerances()) == []


def test_deviations_within_declared_tolerances_are_accepted() -> None:
    """Metric and state deviations inside the tolerances are not mismatches; just outside, they are."""
    tolerances = Tolerances(metric_rel=1e-6, state_abs=1e-6)
    inside = _snapshot(joint_rmse=1e-3 * (1 + 5e-7), final_q=(0.83 + 5e-7, 1.16))
    assert compare_snapshots(inside, _snapshot(), tolerances) == []
    outside = _snapshot(joint_rmse=1e-3 * (1 + 2e-6), final_q=(0.83 + 2e-6, 1.16))
    mismatches = compare_snapshots(outside, _snapshot(), tolerances)
    assert [m.split(":")[0] for m in mismatches] == ["joint_rmse[0]", "final_q[0]"]


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"termination": "limit_violation", "success": False}, "termination: 'limit_violation'"),
        ({"criteria": {**CRITERIA, "dwell_stationary": False}}, "criteria: "),
        ({"n_samples": 500}, "n_samples: 500 != expected 501"),
        ({"joint_rmse": 1.1e-3}, "joint_rmse[0]: 0.0011 differs"),
        ({"joint_rmse_per_joint": (1.2e-3,)}, "joint_rmse_per_joint: length 1 != 2"),
        ({"dwell": {**DWELL, "velocity_max": 0.02}}, "dwell.velocity_max: 0.02 differs"),
        ({"effort": {"torque_rms": 0.5}}, "effort: keys ['torque_rms'] != "),
        ({"final_q": (0.83 + 1e-6, 1.16)}, "final_q[0]: "),
        ({"final_tip": (0.1,)}, "final_tip: length 1 != 2"),
    ],
)
def test_every_field_is_compared(overrides: dict[str, object], fragment: str) -> None:
    """Each snapshot field contributes a located mismatch."""
    mismatches = compare_snapshots(_snapshot(**overrides), _snapshot(), Tolerances())
    assert any(fragment in message for message in mismatches), mismatches


@pytest.mark.parametrize("value", [0.0, -1e-9, math.inf, math.nan])
def test_tolerances_must_be_positive_and_finite(value: float) -> None:
    """Zero, negative, infinite, or NaN tolerances are configuration errors."""
    with pytest.raises(ValueError, match="must be a positive finite number"):
        Tolerances(metric_rel=value)
    with pytest.raises(ValueError, match="must be a positive finite number"):
        Tolerances(state_abs=value)


def test_expectations_round_trip_through_toml(tmp_path: Path) -> None:
    """Written expectations load back equal and carry the regeneration note."""
    expectations = BaselineExpectations(
        scenario="task-1a-reach",
        dataset="processed-20260830-feaf73e6663c",
        gains={"pd": "ab" * 32},
        tolerances=Tolerances(),
        runs={"pd": _snapshot()},
    )
    path = tmp_path / "baselines.toml"
    write_expectations(path, expectations)
    text = path.read_text()
    assert text.startswith("# Replay snapshots of the frozen task 1-a baselines")
    assert "--update-baselines" in text
    assert load_expectations(path) == expectations


def test_expectations_validate_their_structure() -> None:
    """Gains and runs must cover the same methods, each run must be of its method, schema is checked."""
    with pytest.raises(ValueError, match="must cover the same methods"):
        BaselineExpectations("s", "d", {"pd": "x", "computed_torque": "y"}, Tolerances(), {"pd": _snapshot()})
    with pytest.raises(ValueError, match="holds a snapshot of method 'replay\\+pd'"):
        BaselineExpectations("s", "d", {"computed_torque": "y"}, Tolerances(), {"computed_torque": _snapshot()})
    with pytest.raises(ValueError, match="unsupported expectations schema version 2"):
        BaselineExpectations("s", "d", {"pd": "x"}, Tolerances(), {"pd": _snapshot()}, schema_version=2)


def test_versioned_names_map_to_their_tracker_method() -> None:
    """A study-version suffix does not change the tracker method."""
    assert baseline_method("pd") == "pd"
    assert baseline_method("pd_v2") == "pd"
    assert baseline_method("computed_torque") == "computed_torque"
    assert load_frozen_baseline("pd_v2").type == "pd"
    assert load_frozen_baseline("pd_v2") != load_frozen_baseline("pd")

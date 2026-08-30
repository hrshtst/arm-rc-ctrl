# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-004: dataset validation rejects non-finite values, time errors, shape errors, phase gaps, bad codes, limits."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from arm_rc_ctrl.data.records import (
    CANONICAL_UNITS,
    ArtifactRecord,
    Origin,
    Payload,
    Preprocessing,
    ProcessedDatasetRecord,
    Scenario,
    array_specs,
    make_artifact_id,
)
from arm_rc_ctrl.data.samples import PHASE_CODES, PHASE_DWELL, PHASE_MOVE, PHASE_PRIME, SampleSet
from arm_rc_ctrl.data.synthetic import synthetic_arrays, synthetic_samples
from arm_rc_ctrl.data.validate import (
    DatasetValidationError,
    JointLimits,
    ValidationSpec,
    dataset_problems,
    validate_dataset,
)

LIMITS = JointLimits(lower=(-3.0, -3.0), upper=(3.0, 3.0))
SPEC = ValidationSpec(dof=2, task_dim=2, task_code_dim=0, period_s=0.01, limits=LIMITS)


def _arrays(**changes: NDArray[Any]) -> dict[str, NDArray[Any]]:
    arrays = synthetic_arrays(n=8, dof=2, task_dim=2, code_dim=0)
    arrays.update(changes)
    return arrays


def _samples(**changes: NDArray[Any]) -> SampleSet:
    return SampleSet.from_arrays(_arrays(**changes))


def _one_hot_samples() -> SampleSet:
    arrays = synthetic_arrays(n=8, dof=2, task_dim=2, code_dim=3)
    codes = np.zeros((8, 3))
    codes[:4, 0] = 1.0
    codes[4:, 2] = 1.0
    arrays["task_code"] = codes
    return SampleSet.from_arrays(arrays)


def _processed_record(samples: SampleSet) -> ProcessedDatasetRecord:
    created = "2026-08-30T06:00:00+00:00"
    sha = "6" * 64
    artifact_id = make_artifact_id("processed", created, sha)
    artifact = ArtifactRecord(
        artifact_id=artifact_id,
        kind="processed",
        created_at=created,
        license="LicenseRef-Private",
        access="private",
        payload=Payload(f"armrc://processed/{artifact_id}/samples.npz", sha, 1, "samples.npz", 1),
        origin=Origin("cmd", "2" * 64, "a" * 40, False, {}, sources=("raw-20260830-2a97516c354b",)),  # noqa: FBT003
    )
    return ProcessedDatasetRecord(
        artifact=artifact,
        scenario=Scenario(
            "configs/tasks/task_1a.toml", "2" * 64, "r", "t", samples.dof, (0.0,) * samples.dof, (0.1, 0.4)
        ),
        n_samples=samples.n_samples,
        dof=samples.dof,
        task_dim=samples.task_dim,
        task_code_dim=samples.task_code_dim,
        units=dict(CANONICAL_UNITS),
        phases=dict(PHASE_CODES),
        preprocessing=Preprocessing(0.01, "none", {}, "central-difference"),
        arrays=array_specs(samples),
    )


# --- valid ------------------------------------------------------------------------------


def test_valid_dataset_has_no_problems() -> None:
    """The synthetic dataset satisfies the specification and is left untouched."""
    samples = _samples()
    digests = samples.digests()
    assert dataset_problems(samples, SPEC) == []
    validate_dataset(samples, SPEC)
    assert samples.digests() == digests


def test_valid_one_hot_task_codes_pass() -> None:
    """One-hot rows are accepted, including a switch between targets."""
    spec = ValidationSpec(dof=2, task_dim=2, task_code_dim=3, period_s=0.01, limits=LIMITS)
    assert dataset_problems(_one_hot_samples(), spec) == []


def test_spec_from_record_and_limits() -> None:
    """The specification derives dimensions and period from a processed record."""
    samples = synthetic_samples(n=8, dof=2, task_dim=2, code_dim=0)
    spec = ValidationSpec.from_record(_processed_record(samples), LIMITS, require_all_phases=False)
    assert spec == ValidationSpec(2, 2, 0, 0.01, LIMITS, require_all_phases=False)
    validate_dataset(samples, spec)


# --- non-finite ------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["q", "dq", "ddq", "tip", "dtip", "ddtip", "t"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_values_are_rejected(name: str, value: float) -> None:
    """NaN and ±Inf in any float array are reported with the array name and first index."""
    arrays = _arrays()
    bad = arrays[name].copy()
    bad[(2,) if bad.ndim == 1 else (2, 1)] = value
    problems = dataset_problems(SampleSet.from_arrays({**arrays, name: bad}), SPEC)
    index = "(2,)" if bad.ndim == 1 else "(2, 1)"
    assert any(p.startswith(f"{name} contains 1 non-finite value(s), first at index {index}") for p in problems), (
        problems
    )


# --- time --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("t", "message"),
    [
        (np.arange(8) * 0.01 + 0.5, "t must start at 0, got 0.5"),
        (np.array([0.0, 0.01, 0.02, 0.02, 0.04, 0.05, 0.06, 0.07]), "not strictly increasing .*samples 2 and 3"),
        (np.array([0.0, 0.01, 0.02, 0.03, 0.045, 0.055, 0.065, 0.075]), "not uniformly sampled at 0.01 s: interval 3"),
        (np.arange(8) * 0.02, "not uniformly sampled at 0.01 s"),
    ],
)
def test_time_errors_are_rejected(t: NDArray[np.float64], message: str) -> None:
    """Non-zero start, non-monotonic, and non-uniform time grids are all rejected."""
    problems = dataset_problems(_samples(t=t), SPEC)
    assert any(__import__("re").search(message, p) for p in problems), problems


# --- shapes -----------------------------------------------------------------------------


def test_dimension_mismatches_against_the_spec_are_rejected() -> None:
    """A dataset of the wrong dof, task_dim, or task_code_dim does not fit the scenario."""
    spec = ValidationSpec(
        dof=3, task_dim=1, task_code_dim=2, period_s=0.01, limits=JointLimits((-1.0,) * 3, (1.0,) * 3)
    )
    problems = dataset_problems(_samples(), spec)
    assert "dof is 2, expected 3" in problems
    assert "task_dim is 2, expected 1" in problems
    assert "task_code_dim is 0, expected 2" in problems


# --- phases -----------------------------------------------------------------------------


def test_missing_phase_intervals_are_rejected() -> None:
    """Every phase must be present unless the specification relaxes it."""
    phase = np.full(8, PHASE_MOVE, dtype=np.int64)
    phase[-1] = PHASE_DWELL
    problems = dataset_problems(_samples(phase=phase), SPEC)
    assert "missing phase interval(s): ['prime']" in problems
    relaxed = ValidationSpec(2, 2, 0, 0.01, LIMITS, require_all_phases=False)
    assert dataset_problems(_samples(phase=phase), relaxed) == []


def test_phase_order_violations_are_rejected() -> None:
    """Phases may not return to an earlier phase."""
    phase = np.array(
        [PHASE_PRIME, PHASE_MOVE, PHASE_MOVE, PHASE_PRIME, PHASE_MOVE, PHASE_DWELL, PHASE_DWELL, PHASE_DWELL]
    )
    problems = dataset_problems(_samples(phase=phase), SPEC)
    assert any("without returning (violation between samples 2 and 3)" in p for p in problems), problems


# --- task codes ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("row", "label"),
    [
        (np.array([0.5, 0.5, 0.0]), "fractional"),
        (np.array([1.0, 1.0, 0.0]), "two ones"),
        (np.array([0.0, 0.0, 0.0]), "all zeros"),
        (np.array([2.0, 0.0, 0.0]), "value 2"),
    ],
)
def test_invalid_task_codes_are_rejected(row: NDArray[np.float64], label: str) -> None:
    """Rows that are not exactly one-hot are reported with the first offending sample."""
    samples = _one_hot_samples()
    codes = samples.task_code.copy()
    codes[5] = row
    arrays = {**samples.arrays(), "task_code": codes}
    spec = ValidationSpec(dof=2, task_dim=2, task_code_dim=3, period_s=0.01, limits=LIMITS)
    problems = dataset_problems(SampleSet.from_arrays(arrays), spec)
    assert problems == ["task_code has 1 row(s) that are not one-hot, first at sample 5"], label


# --- limits -----------------------------------------------------------------------------


def test_joint_limit_violations_are_rejected() -> None:
    """Positions outside [lower, upper] and speeds above the limit are reported."""
    q = _arrays()["q"].copy()
    q[3, 1] = 3.5
    problems = dataset_problems(_samples(q=q), SPEC)
    assert problems == [
        "q violates joint limits at 1 sample/joint pair(s), first at sample 3 joint 1: 3.5 not in [-3.0, 3.0]"
    ]

    fast = JointLimits(lower=(-3.0, -3.0), upper=(3.0, 3.0), speed=(0.5, 0.5))
    dq = _arrays()["dq"].copy()
    dq[6, 0] = -0.75
    spec = ValidationSpec(2, 2, 0, 0.01, fast)
    problems = dataset_problems(_samples(dq=dq), spec)
    assert problems == [
        "dq exceeds the speed limit at 1 sample/joint pair(s), first at sample 6 joint 0: |-0.75| > 0.5"
    ]


def test_limit_checks_wait_for_finite_positions() -> None:
    """A non-finite q is reported once as non-finite; limit comparisons are not attempted on it."""
    q = _arrays()["q"].copy()
    q[0, 0] = np.nan
    q[1, 1] = 9.0
    problems = dataset_problems(_samples(q=q), SPEC)
    assert len(problems) == 1
    assert problems[0].startswith("q contains 1 non-finite value(s)")


# --- aggregation and specification invariants ------------------------------------------------


def test_all_problems_are_reported_together() -> None:
    """validate_dataset raises once with every finding, so users never fix problems one at a time."""
    arrays = _arrays()
    ddq = arrays["ddq"].copy()
    ddq[0, 0] = np.nan
    q = arrays["q"].copy()
    q[1, 1] = 9.0
    phase = arrays["phase"].copy()
    phase[:] = PHASE_MOVE
    t = arrays["t"].copy()
    t[-1] += 0.5
    samples = SampleSet.from_arrays({**arrays, "ddq": ddq, "q": q, "phase": phase, "t": t})
    with pytest.raises(DatasetValidationError) as info:
        validate_dataset(samples, SPEC)
    problems = info.value.problems
    assert len(problems) == 4
    assert problems[0].startswith("ddq contains 1 non-finite value(s)")
    assert any("not uniformly sampled" in p for p in problems)
    assert "missing phase interval(s): ['prime', 'dwell']" in problems
    assert any("violates joint limits" in p for p in problems)
    assert "dataset validation failed:\n" in str(info.value)


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: JointLimits((-1.0,), (1.0, 2.0)), "same non-zero length"),
        (lambda: JointLimits((1.0,), (1.0,)), "lower must be below"),
        (lambda: JointLimits((-1.0,), (float("inf"),)), r"limits.upper\[0\] must be finite"),
        (lambda: JointLimits((-1.0,), (1.0,), speed=(0.0,)), "speed entries must be positive"),
        (lambda: JointLimits((-1.0,), (1.0,), speed=(1.0, 1.0)), "limits.speed must have 1 entries"),
        (lambda: ValidationSpec(0, 2, 0, 0.01, LIMITS), "dof >= 1"),
        (lambda: ValidationSpec(2, 2, 0, 0.0, LIMITS), "period_s must be positive"),
        (lambda: ValidationSpec(2, 2, 0, 0.01, LIMITS, period_tolerance_s=-1.0), "tolerances must be non-negative"),
        (lambda: ValidationSpec(3, 2, 0, 0.01, LIMITS), "limits cover 2 joints but dof is 3"),
    ],
)
def test_specification_invariants(build: object, message: str) -> None:
    """Limits and specifications validate themselves."""
    with pytest.raises(ValueError, match=message):
        build()  # type: ignore[operator]

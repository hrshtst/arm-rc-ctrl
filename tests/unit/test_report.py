# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-020: reports equal the pure metric functions and serialize without hidden recomputation."""

from __future__ import annotations

import csv
import io
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from arm_rc_ctrl.data.phases import intervals_from_phases
from arm_rc_ctrl.data.samples import PHASE_DWELL, PHASE_MOVE, PHASE_PRIME, SampleSet
from arm_rc_ctrl.data.synthetic import synthetic_arrays
from arm_rc_ctrl.experiments.run_record import LoadedRun, RunArrays, load_run, write_run
from arm_rc_ctrl.experiments.termination import Outcome, completed, divergence
from arm_rc_ctrl.metrics.dwell import dwell_metrics
from arm_rc_ctrl.metrics.effort import effort_metrics
from arm_rc_ctrl.metrics.joint import JointAnglePolicy, joint_rmse
from arm_rc_ctrl.metrics.report import RunReport, build_report, report_from_json, report_to_csv, report_to_json
from arm_rc_ctrl.provenance import collect_provenance
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageRoot

REPO_ROOT = repository_root()
FIXED_TIME = datetime(2026, 8, 30, 9, 0, 0, tzinfo=UTC)
N = 40
TARGET = (0.10, 0.45)
LIMITS = (10.0, 5.0)
POLICY = JointAnglePolicy.limited(2)


def _reference() -> SampleSet:
    arrays = synthetic_arrays(n=N, dof=2, task_dim=2, code_dim=0)
    phase = np.full(N, PHASE_MOVE, dtype=np.int64)
    phase[:10] = PHASE_PRIME
    phase[30:] = PHASE_DWELL
    arrays["phase"] = phase
    return SampleSet.from_arrays(arrays)


def _run_arrays(reference: SampleSet, n: int, *, applied: bool = False) -> RunArrays:
    rng = np.random.default_rng(3)
    q = reference.q[:n] + 0.01 * rng.standard_normal((n, 2))
    zeros = np.zeros((n, 2))
    requested = np.column_stack([np.sin(reference.t[:n]), 6.0 * np.cos(reference.t[:n])])
    arrays: dict[str, NDArray[Any]] = {
        "t": reference.t[:n],
        "q": q,
        "dq": reference.dq[:n],
        "tip": reference.tip[:n],
        "q_desired": reference.q[:n],
        "dq_desired_raw": zeros,
        "dq_desired": zeros,
        "ddq_desired_raw": zeros,
        "ddq_desired": zeros,
        "tracking_error": reference.q[:n] - q,
        "tau_requested": requested,
        "task_code": np.zeros((n, 0)),
        "saturation": np.zeros(n, dtype=np.int64),
    }
    if applied:
        arrays["tau_applied"] = np.clip(requested, -np.asarray(LIMITS), np.asarray(LIMITS))
    return RunArrays(arrays)


def _stored_run(store: StorageRoot, reference: SampleSet, n: int, *, done: bool, applied: bool = False) -> LoadedRun:
    provenance = collect_provenance({"kp": 20.0}, seeds={}, now=FIXED_TIME, env={}, exploratory=True)
    t_end = float(reference.t[n - 1])
    termination = completed(t_end, n - 1) if done else divergence(t_end, n - 1, "stopped early")
    pointer, _, _ = write_run(
        store,
        _run_arrays(reference, n, applied=applied),
        kind="simulation",
        method=("rc+pd" if done else "rc+pd-early") + ("+applied" if applied else ""),
        scenario="task-1a-reach",
        control_period_s=0.01,
        duration_s=float(reference.t[-1]),
        target=TARGET,
        task_code=(),
        disturbances=(),
        termination=termination,
        outcome=Outcome(termination, {"completed": done}),
        provenance=provenance,
        license_label="LicenseRef-Private",
        access="private",
        command="test",
    )
    return load_run(store, pointer)


@pytest.fixture
def store(tmp_path: Path) -> StorageRoot:
    """Empty storage root."""
    root = tmp_path / "store"
    root.mkdir()
    return StorageRoot(root, repositories=(REPO_ROOT,))


def test_report_fields_equal_the_pure_metric_functions(store: StorageRoot) -> None:
    """Joint RMSE, dwell, and effort are exactly the pure functions' outputs on the same windows."""
    reference = _reference()
    run = _stored_run(store, reference, N, done=True)
    report = build_report(
        run, reference, "processed-20260830-555555555555", tolerance=0.02, torque_limits=LIMITS, policy=POLICY
    )

    intervals = intervals_from_phases(reference.t, reference.phase)
    move = (reference.t >= intervals.move[0]) & (reference.t < intervals.move[1])
    assert report.windows.move == intervals.move
    assert report.windows.dwell == intervals.dwell
    assert report.move_coverage == 1.0
    assert report.dwell_coverage == 1.0
    assert report.joint_rmse == joint_rmse(run.arrays.arrays["q"][move], reference.q[move], POLICY)
    assert report.dwell == dwell_metrics(
        run.arrays.arrays["t"],
        run.arrays.arrays["tip"],
        run.arrays.arrays["dq"],
        np.asarray(TARGET),
        0.02,
        window=(intervals.dwell[0], intervals.dwell[1]),
    )
    assert report.effort_source == "tau_requested"  # no applied torque in this synthetic run
    assert report.effort == effort_metrics(run.arrays.arrays["t"], run.arrays.arrays["tau_requested"], LIMITS)
    assert report.demand == report.effort
    assert report.effort is not None
    assert report.effort.saturation_fraction > 0  # 6 cos t reaches the 5 N*m bound
    assert report.termination_kind == "completed"
    assert report.success is True
    assert report.failed_criteria == ()
    assert report.run_id == run.pointer.artifact.artifact_id


def test_effort_uses_applied_torque_when_available(store: StorageRoot) -> None:
    """With tau_applied recorded, effort is physical (clamped) while demand keeps the requested torque."""
    reference = _reference()
    run = _stored_run(store, reference, N, done=True, applied=True)
    report = build_report(
        run, reference, "processed-20260830-555555555555", tolerance=0.02, torque_limits=LIMITS, policy=POLICY
    )
    assert report.effort_source == "tau_applied"
    assert report.effort == effort_metrics(run.arrays.arrays["t"], run.arrays.arrays["tau_applied"], LIMITS)
    assert report.demand == effort_metrics(run.arrays.arrays["t"], run.arrays.arrays["tau_requested"], LIMITS)
    assert report.effort is not None
    assert report.demand is not None
    assert report.effort.effort < report.demand.effort  # saturation clips the physical effort
    assert report.effort.per_joint_peak[1] <= LIMITS[1]
    assert report.demand.per_joint_peak[1] > LIMITS[1]  # 6 cos t exceeds the 5 N*m bound


def test_early_terminated_run_stays_reportable(store: StorageRoot) -> None:
    """A run that stops in the movement window reports partial coverage and no dwell metrics."""
    reference = _reference()
    run = _stored_run(store, reference, 20, done=False)
    report = build_report(
        run, reference, "processed-20260830-555555555555", tolerance=0.02, torque_limits=LIMITS, policy=POLICY
    )
    assert report.termination_kind == "divergence"
    assert report.success is False
    assert report.failed_criteria == ("completed",)
    assert report.move_coverage == pytest.approx(10 / 20)
    assert report.dwell_coverage == 0.0
    assert report.dwell is None
    assert report.joint_rmse is not None
    assert report.joint_rmse.samples == 10
    assert report.effort is not None
    assert report.effort.samples == 20


def test_json_and_csv_carry_the_same_values(store: StorageRoot) -> None:
    """JSON round-trips to an equal report; the CSV row holds the identical numbers as strings."""
    reference = _reference()
    done = _stored_run(store, reference, N, done=True)
    early = _stored_run(store, reference, 20, done=False)
    reports = [
        build_report(
            run, reference, "processed-20260830-555555555555", tolerance=0.02, torque_limits=LIMITS, policy=POLICY
        )
        for run in (done, early)
    ]
    for report in reports:
        assert report_from_json(report_to_json(report)) == report

    rows = list(csv.DictReader(io.StringIO(report_to_csv(reports))))
    assert len(rows) == 2
    assert rows[0]["run_id"] == reports[0].run_id
    assert float(rows[0]["joint_rmse.aggregate"]) == reports[0].joint_rmse.aggregate  # type: ignore[union-attr]
    assert reports[0].effort is not None
    assert float(rows[0]["effort.effort"]) == reports[0].effort.effort
    assert float(rows[0]["dwell.endpoint.p95"]) == reports[0].dwell.endpoint.p95  # type: ignore[union-attr]
    assert rows[0]["windows.move"] == ";".join(str(v) for v in reports[0].windows.move)
    assert rows[1]["dwell.endpoint.p95"] == ""
    assert rows[1]["failed_criteria"] == "completed"
    assert rows[1]["success"] == "False"


def test_grid_mismatch_is_rejected(store: StorageRoot) -> None:
    """A run sampled at a different period cannot be compared sample by sample."""
    reference = _reference()
    run = _stored_run(store, reference, N, done=True)
    coarse = SampleSet.from_arrays({**reference.arrays(), "t": reference.t * 2})
    with pytest.raises(ValueError, match="does not match the reference period"):
        build_report(
            run, coarse, "processed-20260830-555555555555", tolerance=0.02, torque_limits=LIMITS, policy=POLICY
        )


def test_report_schema_is_strict() -> None:
    """Unknown keys in a stored report are rejected."""
    with pytest.raises(Exception, match=r"unknown key\(s\) 'extra'"):
        report_from_json('{"extra": 1}')
    assert RunReport.__dataclass_fields__["schema_version"].default == 1

# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Metric reports from run records (``docs/PLAN.md`` section 9.1).

A report evaluates one run against the canonical reference dataset it was
meant to reproduce: joint RMSE over the reference's movement window, dwell
metrics over its dwell window, and effort over the whole run, plus the
termination and outcome. Values are exactly what the pure metric functions
return — the report never recomputes or rounds them — and early-terminated
runs stay reportable with coverage fractions and ``None`` where a window has
no samples.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from typing import Final, cast

import numpy as np
from numpy.typing import NDArray

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.data.phases import intervals_from_phases
from arm_rc_ctrl.data.samples import SampleSet
from arm_rc_ctrl.experiments.run_record import LoadedRun
from arm_rc_ctrl.metrics.dwell import DwellMetrics, dwell_metrics
from arm_rc_ctrl.metrics.effort import EffortMetrics, effort_metrics
from arm_rc_ctrl.metrics.joint import JointAnglePolicy, JointRmse, joint_rmse
from arm_rc_ctrl.provenance import canonical_json

__all__ = [
    "REPORT_SCHEMA_VERSION",
    "ReportWindows",
    "RunReport",
    "build_report",
    "report_from_json",
    "report_to_csv",
    "report_to_json",
]

REPORT_SCHEMA_VERSION: Final = 1
_GRID_TOLERANCE: Final = 1e-9


@dataclass(frozen=True)
class ReportWindows:
    """Time windows (s) taken from the reference dataset's phases."""

    move: tuple[float, ...]
    dwell: tuple[float, ...]


@dataclass(frozen=True)
class RunReport:
    """Metrics of one run against its reference."""

    run_id: str
    method: str
    scenario: str
    reference_artifact: str
    termination_kind: str
    success: bool
    failed_criteria: tuple[str, ...]
    windows: ReportWindows
    move_coverage: float
    """Fraction of the reference movement window covered by run samples."""
    dwell_coverage: float
    joint_rmse: JointRmse | None
    dwell: DwellMetrics | None
    effort: EffortMetrics
    schema_version: int = REPORT_SCHEMA_VERSION


def _same_grid(run_t: NDArray[np.float64], ref_t: NDArray[np.float64]) -> None:
    if run_t.shape[0] < 2 or ref_t.shape[0] < 2:  # noqa: PLR2004
        msg = "run and reference need at least two samples"
        raise ValueError(msg)
    run_dt = float(run_t[1] - run_t[0])
    ref_dt = float(ref_t[1] - ref_t[0])
    if abs(run_dt - ref_dt) > _GRID_TOLERANCE or abs(float(run_t[0])) > _GRID_TOLERANCE:
        msg = f"run grid (start {float(run_t[0])!r}, period {run_dt!r}) does not match the reference period {ref_dt!r}"
        raise ValueError(msg)
    n = min(run_t.shape[0], ref_t.shape[0])
    if np.max(np.abs(run_t[:n] - ref_t[:n])) > _GRID_TOLERANCE:
        msg = "run and reference timestamps diverge on the shared grid"
        raise ValueError(msg)


def build_report(
    run: LoadedRun,
    reference: SampleSet,
    reference_artifact: str,
    *,
    tolerance: float,
    torque_limits: tuple[float, ...],
    policy: JointAnglePolicy,
) -> RunReport:
    """Evaluate ``run`` against ``reference`` with the pure metric functions."""
    arrays = run.arrays.arrays
    run_t = cast("NDArray[np.float64]", arrays["t"])
    _same_grid(run_t, reference.t)
    intervals = intervals_from_phases(reference.t, reference.phase)
    windows = ReportWindows(move=intervals.move, dwell=intervals.dwell)

    n = min(run_t.shape[0], reference.n_samples)
    move_mask = (reference.t[:n] >= intervals.move[0]) & (reference.t[:n] < intervals.move[1])
    move_total = int(np.count_nonzero((reference.t >= intervals.move[0]) & (reference.t < intervals.move[1])))
    dwell_total = int(np.count_nonzero((reference.t >= intervals.dwell[0]) & (reference.t <= intervals.dwell[1])))
    dwell_in_run = int(np.count_nonzero((run_t >= intervals.dwell[0]) & (run_t <= intervals.dwell[1])))

    rmse: JointRmse | None = None
    if move_mask.any():
        q_run = cast("NDArray[np.float64]", arrays["q"])[:n][move_mask]
        q_ref = reference.q[:n][move_mask]
        rmse = joint_rmse(q_run, q_ref, policy)
    dwell: DwellMetrics | None = None
    if dwell_in_run:
        dwell = dwell_metrics(
            run_t,
            cast("NDArray[np.float64]", arrays["tip"]),
            cast("NDArray[np.float64]", arrays["dq"]),
            np.asarray(run.summary.target, dtype=np.float64),
            tolerance,
            window=(intervals.dwell[0], intervals.dwell[1]),
        )
    effort = effort_metrics(run_t, cast("NDArray[np.float64]", arrays["tau_requested"]), torque_limits)
    return RunReport(
        run_id=run.pointer.artifact.artifact_id,
        method=run.summary.method,
        scenario=run.summary.scenario,
        reference_artifact=reference_artifact,
        termination_kind=run.summary.termination.kind,
        success=run.summary.outcome.success,
        failed_criteria=run.summary.outcome.failed_criteria,
        windows=windows,
        move_coverage=float(np.count_nonzero(move_mask)) / move_total if move_total else 0.0,
        dwell_coverage=dwell_in_run / dwell_total if dwell_total else 0.0,
        joint_rmse=rmse,
        dwell=dwell,
        effort=effort,
    )


def report_to_json(report: RunReport) -> str:
    """Canonical JSON of the report (loads back with ``from_mapping``)."""
    return canonical_json(to_mapping(report))


def report_from_json(text: str) -> RunReport:
    """Strictly rebuild a report from JSON."""
    return from_mapping(cast("dict[str, object]", json.loads(text)), RunReport)


def _flatten(prefix: str, value: object, out: dict[str, object]) -> None:
    if isinstance(value, dict):
        for key, item in cast("dict[str, object]", value).items():
            _flatten(f"{prefix}.{key}" if prefix else key, item, out)
    elif isinstance(value, list):
        items = cast("list[object]", value)
        if all(isinstance(v, (int, float, str, bool)) for v in items):
            out[prefix] = ";".join(str(v) for v in items)
        else:
            for i, item in enumerate(items):
                _flatten(f"{prefix}[{i}]", item, out)
    else:
        out[prefix] = value


def report_to_csv(reports: list[RunReport]) -> str:
    """One CSV row per report with dotted column names; lists are ``;``-joined, ``None`` is empty."""
    rows: list[dict[str, object]] = []
    for report in reports:
        flat: dict[str, object] = {}
        _flatten("", to_mapping(report), flat)
        rows.append(flat)
    columns: list[str] = []
    for row in rows:
        columns.extend(key for key in row if key not in columns)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: ("" if row.get(key) is None else row.get(key)) for key in columns})
    return buffer.getvalue()

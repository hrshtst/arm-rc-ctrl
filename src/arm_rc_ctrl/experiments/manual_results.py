# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-010: the derived evidence of the manual study (plan sections 6 and 7.1).

The sweep's manifests say how every run was judged; this module turns them
into the tables a reader works from. One row per run, with its verdict, its
terminal state and the diagnostic metrics the plan lists, read from the run's
own stored trajectories after the same digest checks a resume applies. The
paired comparisons and class summaries of :mod:`manual_contrasts` are computed
from those rows, so every aggregate traces back to runs that were verified.

Metrics are measured over the active segment, from activation to the last
sample, which is the segment the sweep judged its dwell over; the negative-time
warm-up never contributes a peak or an error. They are descriptive diagnostics
defined for this handoff, not outcomes: the verdicts are the sweep's own.

The two large tables, one row per run and one row per paired comparison, go to
the external store behind digest-and-size pointers (the owner's decision of
2026-09-19); the summaries, the accounting, the illustration selections and the
figure inputs are small enough to commit. Every output is versioned and written
once: an existing file is refused rather than replaced.

Command line::

    python -m arm_rc_ctrl.experiments.manual_results derive
        --study docs/experiments/task_1a_manual_demonstration/study_manifest_v1.json
        --evaluation configs/evaluations/task_1a_manual_dev_v1.toml
        --evidence-dir docs/experiments/task_1a_manual_demonstration/evidence
        --run-ordering docs/experiments/task_1a_manual_demonstration/run_ordering_v1.json
        --representative-rule docs/experiments/task_1a_manual_demonstration/representative_rule_v1.json
        --result-schema docs/experiments/task_1a_manual_demonstration/result_schema_v3.json
        --output docs/experiments/task_1a_manual_demonstration/results
"""

from __future__ import annotations

import argparse
import csv
import dataclasses as dc
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.experiments.manual_accounting import StudyAccounting, account_study
from arm_rc_ctrl.experiments.manual_contrasts import (
    ManualArmSummary,
    ManualContrastRow,
    ManualContrastSummary,
    ScenarioVerdict,
    arm_label,
    arm_summaries,
    contrast_rows,
    contrast_summaries,
)
from arm_rc_ctrl.experiments.manual_evaluation import (
    GRID_TOLERANCE_S,
    bank_identity,
    load_manual_model_evidence,
    load_manual_pointer,
    load_manual_replay_bank,
    load_verified_run,
    manual_pointer_name,
    prepare_runner,
)
from arm_rc_ctrl.experiments.manual_figures import ManualFigureInputs, figure_inputs, figure_inputs_to_json
from arm_rc_ctrl.experiments.manual_handoff import RepresentativeRule, RunOrdering, load_handoff
from arm_rc_ctrl.experiments.manual_representative import ArmVerdict, Selection, representative_cases
from arm_rc_ctrl.experiments.manual_schema import load_schema
from arm_rc_ctrl.experiments.manual_study import EXPERIMENT_LABEL
from arm_rc_ctrl.experiments.recovery_search import RECOVERY_TRACKERS
from arm_rc_ctrl.provenance import (
    ArtifactReference,
    ProvenanceRecord,
    canonical_json,
    sha256_bytes,
    sha256_file,
    verify_artifact,
)
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import ArtifactUri, StorageRoot

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    from arm_rc_ctrl.experiments.manual_evaluation import (
        ManualEvaluationRunner,
        ManualModelEvidence,
        ManualPairRecord,
        ManualReplayBank,
    )
    from arm_rc_ctrl.experiments.manual_study import StudyManifest
    from arm_rc_ctrl.experiments.run_record import RunSummary

__all__ = [
    "DEPARTURE_RADIUS_M",
    "RESULTS_PREFIX",
    "RESULTS_SCHEMA_VERSION",
    "RESULTS_VERSION",
    "RESULT_DOCUMENTS",
    "RUN_STATUSES",
    "ManualResultDocument",
    "ManualResultInputs",
    "ManualResultTable",
    "ManualResults",
    "ManualRunMetrics",
    "ManualRunRow",
    "ManualSelection",
    "ManualSelections",
    "evidence_digest",
    "load_results",
    "main",
    "render_results_markdown",
    "results_to_json",
    "run_metrics",
    "selections_of",
    "table_columns",
    "table_from_csv",
    "table_to_csv",
    "verdicts_of",
]

DEPARTURE_RADIUS_M: Final = 0.01
"""The manual task's dwell radius (1 cm), reused as the distance that counts as leaving the start."""
RESULTS_VERSION: Final = 1
"""Version of this derivation's outputs; a changed output is a new version beside this one."""
RESULTS_SCHEMA_VERSION: Final = 1
"""Version of the :class:`ManualResults` record itself."""
REFERENCED_RESULT_SCHEMA: Final = 3
"""The result schema version whose records these outputs are."""
RESULTS_PREFIX: Final = "armrc://reports/task_1a_manual_v1/results"
_MODULE: Final = "arm_rc_ctrl.experiments.manual_results"
RUN_STATUSES: Final = ("completed", "infeasible", "unexecuted", "unavailable")
"""A pair's recorded status, or ``unavailable`` for a run whose model has no evidence at all."""
_SIMULATED: Final = ("completed", "infeasible")
_RC, _REPLAY, _ALL_TEN = "rc", "replay", "M10"

RESULT_DOCUMENTS: Final = {
    "accounting": f"accounting_v{RESULTS_VERSION}.json",
    "arm_summary": f"arm_summary_v{RESULTS_VERSION}.csv",
    "contrast_summary": f"contrast_summary_v{RESULTS_VERSION}.csv",
    "selections": f"selections_v{RESULTS_VERSION}.json",
    "figure_inputs": f"figure_inputs_v{RESULTS_VERSION}.json",
}
"""The committed outputs beside ``results_v1.json``, by role."""


# --- per-run metrics -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ManualRunMetrics:
    """The diagnostic metrics of one run over its active segment; all absent when it never activated."""

    n_active_samples: int
    final_endpoint_error_m: float | None
    time_to_final_dwell_s: float | None
    departure_latency_s: float | None
    peak_speed_rad_s: float | None
    peak_acceleration_rad_s2: float | None
    peak_reference_speed_rad_s: float | None
    peak_reference_acceleration_rad_s2: float | None
    activation_jump_rad: float | None
    tracking_error_rms_rad: float | None
    tracking_error_max_rad: float | None
    torque_peak_nm: float | None


def _peak(values: NDArray[np.float64]) -> float | None:
    """The largest magnitude, ignoring the NaN a channel carries where it is inactive."""
    finite = np.abs(values[np.isfinite(values)])
    return float(np.max(finite)) if finite.size else None


def _array(arrays: Mapping[str, NDArray[Any]], name: str) -> NDArray[np.float64]:
    return np.asarray(arrays[name], dtype=np.float64)


def run_metrics(
    arrays: Mapping[str, NDArray[Any]],
    *,
    activation_s: float,
    target: Sequence[float],
    final_dwell_samples: int | None,
    departure_radius_m: float = DEPARTURE_RADIUS_M,
) -> ManualRunMetrics:
    """Measure one run's diagnostics from its stored trajectories.

    ``final_dwell_samples`` is the length of the dwell that ends at the last
    sample, as the sweep recorded it when the run's dwell succeeded, and
    ``None`` otherwise; it is measured over the same active segment, so it
    cannot be longer than that segment.
    """
    t = _array(arrays, "t")
    active = t >= activation_s - GRID_TOLERANCE_S
    first = int(np.argmax(active)) if bool(np.any(active)) else t.shape[0]
    n_active = t.shape[0] - first
    if final_dwell_samples is not None and not 0 < final_dwell_samples <= n_active:
        msg = f"a final dwell of {final_dwell_samples} samples cannot end a {n_active}-sample active segment"
        raise ValueError(msg)
    if n_active == 0:
        return ManualRunMetrics(0, *([None] * 11))
    window = slice(first, None)
    tip = _array(arrays, "tip")[window]
    goal = np.asarray(target, dtype=np.float64)
    moved = np.hypot(tip[:, 0] - tip[0, 0], tip[:, 1] - tip[0, 1]) > departure_radius_m
    t_active = t[window]
    dq = _array(arrays, "dq")[window]
    acceleration = np.diff(dq, axis=0) / np.diff(t_active)[:, None] if n_active > 1 else np.empty((0, dq.shape[1]))
    jump = _array(arrays, "q_desired")[first] - _array(arrays, "q")[first]
    error = _array(arrays, "tracking_error")[window]
    torque = _array(arrays, "tau_applied" if "tau_applied" in arrays else "tau_requested")[window]
    return ManualRunMetrics(
        n_active_samples=n_active,
        final_endpoint_error_m=float(np.hypot(tip[-1, 0] - goal[0], tip[-1, 1] - goal[1])),
        time_to_final_dwell_s=(
            None if final_dwell_samples is None else float(t[t.shape[0] - final_dwell_samples] - activation_s)
        ),
        departure_latency_s=float(t_active[int(np.argmax(moved))] - activation_s) if bool(np.any(moved)) else None,
        peak_speed_rad_s=_peak(dq),
        peak_acceleration_rad_s2=_peak(acceleration),
        peak_reference_speed_rad_s=_peak(_array(arrays, "dq_desired")[window]),
        peak_reference_acceleration_rad_s2=_peak(_array(arrays, "ddq_desired")[window]),
        activation_jump_rad=float(np.sqrt(np.sum(np.asarray(jump * jump, dtype=np.float64)))),
        tracking_error_rms_rad=float(np.sqrt(np.mean(error * error))),
        tracking_error_max_rad=_peak(error),
        torque_peak_nm=_peak(torque),
    )


# --- one row per run -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ManualRunRow:
    """One run of the study: what ran, how it was judged, where it stopped, what it measured, where it lives."""

    source: str
    configuration: str
    arm: str
    arm_kind: str
    parent: str | None
    model_label: str
    evidence_identity: str | None
    tracker: str
    scenario_id: str
    scenario_class: str
    scenario_index: int
    status: str
    success: bool | None
    reason: str | None
    initial_q: tuple[float, ...] | None
    horizon_completed: bool | None
    termination_kind: str | None
    termination_time_s: float | None
    termination_limit: str | None
    termination_joint: int | None
    termination_value: float | None
    termination_bound: float | None
    termination_failure: str | None
    termination_detail: str | None
    dwell_ok: bool | None
    dwell_final_s: float | None
    dwell_longest_s: float | None
    dwell_earliest_start_s: float | None
    departures_after_hold: int | None
    post_pulse_dwell_ok: bool | None
    post_pulse_dwell_final_s: float | None
    trigger_ok: bool | None
    pulse_triggered: bool | None
    pulse_start_s: float | None
    pulse_end_s: float | None
    trigger_reason: str | None
    generated_within_position_limits: bool | None
    generated_within_speed_limits: bool | None
    generated_within_workspace: bool | None
    generated_dwell_ok: bool | None
    generated_dwell_final_s: float | None
    saturation_fraction: float | None
    torque_rms_nm: float | None
    activation_s: float | None
    duration_s: float | None
    n_samples: int | None
    n_active_samples: int | None
    final_endpoint_error_m: float | None
    time_to_final_dwell_s: float | None
    departure_latency_s: float | None
    peak_speed_rad_s: float | None
    peak_acceleration_rad_s2: float | None
    peak_reference_speed_rad_s: float | None
    peak_reference_acceleration_rad_s2: float | None
    activation_jump_rad: float | None
    tracking_error_rms_rad: float | None
    tracking_error_max_rad: float | None
    torque_peak_nm: float | None
    run_artifact_id: str | None
    run_uri: str | None
    run_sha256: str | None
    run_size: int | None
    arrays_sha256: str | None
    sources: tuple[str, ...] | None
    fit_identity: str | None

    def __post_init__(self) -> None:
        """A simulated run carries its verdict and its payload; any other run carries neither."""
        if self.source not in (_RC, _REPLAY) or self.status not in RUN_STATUSES:
            msg = f"{self.model_label} {self.scenario_id}: unknown source {self.source!r} or status {self.status!r}"
            raise ValueError(msg)
        simulated = self.status in _SIMULATED
        stated = self.success is not None and self.run_uri is not None and self.n_active_samples is not None
        if simulated != stated or (simulated and self.success != (self.status == "completed")):
            msg = f"{self.model_label} {self.scenario_id} [{self.tracker}]: status {self.status} contradicts the row"
            raise ValueError(msg)
        if self.success is not None and self.success != (self.reason is None):
            msg = f"{self.model_label} {self.scenario_id}: success must mean no reason"
            raise ValueError(msg)


@dataclass(frozen=True)
class _Origin:
    """What every run of one model, or of one replay bank, has in common."""

    source: str
    configuration: str
    arm: str
    arm_kind: str
    parent: str | None
    model_label: str
    evidence_identity: str | None
    fit_identity: str | None


def _row(
    origin: _Origin,
    pair: ManualPairRecord,
    loaded: tuple[RunSummary, Mapping[str, NDArray[Any]]] | None,
    *,
    radius_m: float,
) -> ManualRunRow:
    """Assemble one run's row from its pair record and, when it ran, its verified payload."""
    outcome, run = pair.outcome, pair.run
    summary = None if loaded is None else loaded[0]
    metrics: ManualRunMetrics | None = None
    if loaded is not None:
        stored = loaded[0]
        if stored.activation_s is None or outcome is None:
            msg = f"{origin.model_label} {pair.scenario_id}: a simulated run records its activation and outcome"
            raise ValueError(msg)
        metrics = run_metrics(
            loaded[1],
            activation_s=stored.activation_s,
            target=stored.target,
            final_dwell_samples=outcome.dwell.final_samples if outcome.dwell.ok else None,
            departure_radius_m=radius_m,
        )
    termination = None if summary is None else summary.termination
    generated = None if outcome is None else outcome.generated
    trigger = None if outcome is None else outcome.trigger
    post = None if outcome is None else outcome.post_pulse_dwell
    m = metrics
    return ManualRunRow(
        source=origin.source,
        configuration=origin.configuration,
        arm=origin.arm,
        arm_kind=origin.arm_kind,
        parent=origin.parent,
        model_label=origin.model_label,
        evidence_identity=origin.evidence_identity,
        tracker=pair.tracker,
        scenario_id=pair.scenario_id,
        scenario_class=pair.kind,
        scenario_index=pair.index,
        status=pair.status,
        success=None if outcome is None or loaded is None else outcome.success,
        reason=None if outcome is None or loaded is None else outcome.reason,
        initial_q=pair.initial_q,
        horizon_completed=None if outcome is None else outcome.completed,
        termination_kind=None if termination is None else str(termination.kind),
        termination_time_s=None if termination is None else termination.time_s,
        termination_limit=None if termination is None or termination.limit is None else str(termination.limit),
        termination_joint=None if termination is None else termination.joint,
        termination_value=None if termination is None else termination.value,
        termination_bound=None if termination is None else termination.bound,
        termination_failure=None if termination is None or termination.failure is None else str(termination.failure),
        termination_detail=None if termination is None or not termination.detail else termination.detail,
        dwell_ok=None if outcome is None else outcome.dwell.ok,
        dwell_final_s=None if outcome is None else outcome.dwell.final_duration_s,
        dwell_longest_s=None if outcome is None else outcome.dwell.longest_duration_s,
        dwell_earliest_start_s=None if outcome is None else outcome.dwell.earliest_start_s,
        departures_after_hold=None if outcome is None else outcome.dwell.departures_after_hold,
        post_pulse_dwell_ok=None if post is None else post.ok,
        post_pulse_dwell_final_s=None if post is None else post.final_duration_s,
        trigger_ok=None if trigger is None else trigger.ok,
        pulse_triggered=None if trigger is None else trigger.triggered,
        pulse_start_s=None if trigger is None else trigger.pulse_start_s,
        pulse_end_s=None if trigger is None else trigger.pulse_end_s,
        trigger_reason=None if trigger is None else trigger.reason,
        generated_within_position_limits=None if generated is None else generated.within_position_limits,
        generated_within_speed_limits=None if generated is None else generated.within_speed_limits,
        generated_within_workspace=None if generated is None else generated.within_workspace,
        generated_dwell_ok=None if generated is None or generated.dwell is None else generated.dwell.ok,
        generated_dwell_final_s=(
            None if generated is None or generated.dwell is None else generated.dwell.final_duration_s
        ),
        saturation_fraction=None if outcome is None else outcome.saturation_fraction,
        torque_rms_nm=None if outcome is None else outcome.torque_rms,
        activation_s=None if summary is None else summary.activation_s,
        duration_s=None if summary is None else summary.duration_s,
        n_samples=None if loaded is None else int(np.asarray(loaded[1]["t"]).shape[0]),
        n_active_samples=None if m is None else m.n_active_samples,
        final_endpoint_error_m=None if m is None else m.final_endpoint_error_m,
        time_to_final_dwell_s=None if m is None else m.time_to_final_dwell_s,
        departure_latency_s=None if m is None else m.departure_latency_s,
        peak_speed_rad_s=None if m is None else m.peak_speed_rad_s,
        peak_acceleration_rad_s2=None if m is None else m.peak_acceleration_rad_s2,
        peak_reference_speed_rad_s=None if m is None else m.peak_reference_speed_rad_s,
        peak_reference_acceleration_rad_s2=None if m is None else m.peak_reference_acceleration_rad_s2,
        activation_jump_rad=None if m is None else m.activation_jump_rad,
        tracking_error_rms_rad=None if m is None else m.tracking_error_rms_rad,
        tracking_error_max_rad=None if m is None else m.tracking_error_max_rad,
        torque_peak_nm=None if m is None else m.torque_peak_nm,
        run_artifact_id=None if run is None else run.artifact_id,
        run_uri=None if run is None else run.uri,
        run_sha256=None if run is None else run.sha256,
        run_size=None if run is None else run.size,
        arrays_sha256=None if run is None else run.arrays_sha256,
        sources=None if run is None else run.sources,
        fit_identity=origin.fit_identity,
    )


def _unavailable_row(origin: _Origin, index: int, scenario_id: str, scenario_class: str, tracker: str) -> ManualRunRow:
    """A run the protocol names whose model has no evidence: every measured field is absent."""
    empty = {f.name: None for f in dc.fields(ManualRunRow)}
    fixed: dict[str, object] = {
        **dc.asdict(origin),
        "tracker": tracker,
        "scenario_id": scenario_id,
        "scenario_class": scenario_class,
        "scenario_index": index,
        "status": "unavailable",
    }
    return ManualRunRow(**cast("dict[str, Any]", empty | fixed))


@dataclass(frozen=True)
class _Job:
    """The runs of one model or one replay bank, read and measured together."""

    origin: _Origin
    pairs: tuple[ManualPairRecord, ...]
    radius_m: float


def _rows_of(store: StorageRoot, job: _Job) -> list[ManualRunRow]:
    """Every run of one job: each simulated run verified and measured, every other one carried as recorded."""
    rows: list[ManualRunRow] = []
    for pair in job.pairs:
        loaded = load_verified_run(store, pair) if pair.status in _SIMULATED else None
        rows.append(_row(job.origin, pair, loaded, radius_m=job.radius_m))
    return rows


# --- strict CSV tables ---------------------------------------------------------------------------


def table_columns(cls: type[Any]) -> tuple[str, ...]:
    """The columns of a table of ``cls`` rows: its fields, in declaration order."""
    return tuple(f.name for f in dc.fields(cls))


def _base(declared: str) -> tuple[str, bool]:
    optional = declared.endswith(" | None")
    return (declared.removesuffix(" | None") if optional else declared), optional


def _json_item(value: object, declared: str) -> object:
    base, optional = _base(declared)
    if value is None:
        if not optional:
            msg = f"a {declared} item cannot be absent"
            raise ValueError(msg)
        return None
    return float(cast("float", value)) if base == "float" else value


def _cell(value: object, declared: str) -> str:
    """One cell: absent values empty, booleans ``true``/``false``, floats exact, tuples as JSON arrays."""
    base, optional = _base(declared)
    if value is None:
        if not optional:
            msg = f"a {declared} cell cannot be empty"
            raise ValueError(msg)
        return ""
    if base.startswith("tuple["):
        inner = base.removeprefix("tuple[").removesuffix(", ...]")
        items = [_json_item(item, inner) for item in cast("tuple[object, ...]", value)]
        return json.dumps(items, separators=(",", ":"), allow_nan=False)
    if base == "bool":
        return "true" if value else "false"
    if base == "float":
        return repr(float(cast("float", value)))
    text = str(value)
    if not text:
        msg = f"an empty string is indistinguishable from an absent value in a {declared} cell"
        raise ValueError(msg)
    return text


def _parse_item(value: object, declared: str) -> object:
    base, optional = _base(declared)
    if value is None:
        if not optional:
            msg = f"a {declared} item cannot be absent"
            raise ValueError(msg)
        return None
    expected: dict[str, type[Any] | tuple[type[Any], ...]] = {
        "int": int,
        "float": (int, float),
        "str": str,
        "bool": bool,
    }
    if base not in expected or isinstance(value, bool) != (base == "bool") or not isinstance(value, expected[base]):
        msg = f"{value!r} is not of type {declared}"
        raise ValueError(msg)
    return float(cast("float", value)) if base == "float" else value


def _parse(text: str, declared: str) -> object:
    base, optional = _base(declared)
    if text == "":
        if optional:
            return None
        msg = f"a {declared} cell cannot be empty"
        raise ValueError(msg)
    if base.startswith("tuple["):
        inner = base.removeprefix("tuple[").removesuffix(", ...]")
        items = json.loads(text)
        if not isinstance(items, list):
            msg = f"{text!r} is not a JSON array"
            raise ValueError(msg)
        return tuple(_parse_item(item, inner) for item in cast("list[object]", items))
    if base == "bool":
        if text not in ("true", "false"):
            msg = f"{text!r} is not a boolean"
            raise ValueError(msg)
        return text == "true"
    if base == "int":
        return int(text)
    if base == "float":
        return float(text)
    return text


def table_to_csv(rows: Sequence[object], cls: type[Any]) -> str:
    """Write rows as CSV: one column per field, absent values empty, tuples as JSON arrays, floats exact."""
    fields = dc.fields(cls)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([f.name for f in fields])
    for row in rows:
        if type(row) is not cls:
            msg = f"a {cls.__name__} table cannot hold a {type(row).__name__}"
            raise TypeError(msg)
        writer.writerow([_cell(getattr(row, f.name), str(f.type)) for f in fields])
    return buffer.getvalue()


def table_from_csv[T](text: str, cls: type[T]) -> tuple[T, ...]:
    """Strictly read a table written by :func:`table_to_csv`, refusing any other header."""
    fields = dc.fields(cast("Any", cls))
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    if header != [f.name for f in fields]:
        msg = f"the table's header is not the {cls.__name__} columns"
        raise ValueError(msg)
    rows: list[T] = []
    for cells in reader:
        if len(cells) != len(fields):
            msg = f"a {cls.__name__} row has {len(cells)} cells, not {len(fields)}"
            raise ValueError(msg)
        rows.append(cls(**{f.name: _parse(cell, str(f.type)) for f, cell in zip(fields, cells, strict=True)}))
    return tuple(rows)


# --- selections ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class ManualSelection:
    """The frozen representative rule applied to one configuration under one tracker."""

    configuration: str
    tracker: str
    arms: tuple[str, ...]
    omitted: tuple[str, ...]
    """Scenarios left out because one of the compared arms has no verdict there."""
    selection: Selection


@dataclass(frozen=True)
class ManualSelections:
    """Every application of the frozen representative rule, in the rule's configuration and tracker order."""

    experiment: str
    rule_sha256: str
    ordering_sha256: str
    applications: tuple[ManualSelection, ...]
    schema_version: int = field(default=RESULTS_SCHEMA_VERSION)


def selections_of(
    rows: Sequence[ManualRunRow],
    rule: RepresentativeRule,
    *,
    order: Sequence[str],
    rule_sha256: str,
    ordering_sha256: str,
) -> ManualSelections:
    """Apply the frozen rule to each configuration under each tracker, exactly as it was frozen.

    Public because the clean-checkout audit rebuilds the selections from the
    committed per-run table and compares them with the committed ones: a second
    implementation of the rule would be a second thing to keep right.
    """
    kinds = dict(zip(rule.arm_labels, rule.arms, strict=True))
    verdicts: dict[tuple[str, str, str], dict[str, bool | None]] = {}
    for row in rows:
        if row.source == _RC and row.arm in kinds:
            verdicts.setdefault((row.configuration, row.tracker, row.arm), {})[row.scenario_id] = row.success
    applications: list[ManualSelection] = []
    for configuration in rule.configurations:
        for tracker in rule.trackers:
            by_arm = {label: verdicts.get((configuration, tracker, label), {}) for label in rule.arm_labels}
            complete = [sid for sid in order if all(by_arm[label].get(sid) is not None for label in rule.arm_labels)]
            chosen = [
                ArmVerdict(scenario_id=sid, tracker=tracker, arm=kinds[label], succeeded=bool(by_arm[label][sid]))
                for sid in complete
                for label in rule.arm_labels
            ]
            applications.append(
                ManualSelection(
                    configuration=configuration,
                    tracker=tracker,
                    arms=rule.arm_labels,
                    omitted=tuple(sid for sid in order if sid not in complete),
                    selection=representative_cases(chosen, order=order),
                )
            )
    return ManualSelections(
        experiment=EXPERIMENT_LABEL,
        rule_sha256=rule_sha256,
        ordering_sha256=ordering_sha256,
        applications=tuple(applications),
    )


# --- the results manifest ------------------------------------------------------------------------


@dataclass(frozen=True)
class ManualResultInputs:
    """The frozen inputs the derivation read, bound by digest."""

    study_manifest_sha256: str
    evaluation_sha256: str
    run_ordering_sha256: str
    representative_rule_sha256: str
    result_schema_sha256: str
    evidence_sha256: str
    n_pointers: int


@dataclass(frozen=True)
class ManualResultTable:
    """One table kept in the external store, cited by its digest and size."""

    name: str
    record: str
    payload: ArtifactReference
    n_rows: int
    columns: tuple[str, ...]


@dataclass(frozen=True)
class ManualResultDocument:
    """One committed output beside the results manifest, cited by its digest and size."""

    name: str
    record: str
    sha256: str
    size: int
    n_entries: int


@dataclass(frozen=True)
class ManualResults:
    """The index of the derived evidence: what was read, what was written, and the totals it all rests on."""

    experiment: str
    version: int
    result_schema_version: int
    inputs: ManualResultInputs
    tables: tuple[ManualResultTable, ...]
    documents: tuple[ManualResultDocument, ...]
    n_models: int
    n_replay_banks: int
    n_rc_runs: int
    n_replay_runs: int
    n_unavailable_runs: int
    n_rc_successes: int
    n_replay_successes: int
    departure_radius_m: float
    command: str
    provenance: ProvenanceRecord
    schema_version: int = field(default=RESULTS_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        """The index names this experiment, its own version, and the result schema its records follow."""
        if (self.experiment, self.schema_version, self.result_schema_version) != (
            EXPERIMENT_LABEL,
            RESULTS_SCHEMA_VERSION,
            REFERENCED_RESULT_SCHEMA,
        ):
            msg = f"unsupported results index {self.schema_version} of {self.experiment!r}"
            raise ValueError(msg)
        if self.n_rc_successes > self.n_rc_runs or self.n_replay_successes > self.n_replay_runs:
            msg = "more successes than runs"
            raise ValueError(msg)


def results_to_json(results: ManualResults) -> str:
    """Canonical JSON of the results index."""
    return canonical_json(to_mapping(results))


def load_results(path: Path) -> ManualResults:
    """Strictly rebuild the results index from its JSON."""
    return from_mapping(cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), ManualResults)


def evidence_digest(evidence_dir: Path) -> tuple[str, int]:
    """One digest over every pointer in the evidence directory: its name and its file digest, in name order."""
    pointers = sorted(evidence_dir.glob("*.toml"))
    listing = [[path.name, sha256_file(path)] for path in pointers]
    return sha256_bytes(canonical_json(listing).encode("utf-8")), len(pointers)


def render_results_markdown(results: ManualResults) -> str:
    """The index as Markdown: the bound inputs, every output with its digest, and the totals."""
    lines = [
        f"# Task 1-a manual-demonstration derived evidence (v{results.version})",
        "",
        (
            "Generated by `manual_results derive`; regenerate rather than edit. Record definitions and units are "
            f"in result schema v{results.result_schema_version}; `usage_v{results.version}.md` explains how to "
            "read and reproduce every output."
        ),
        "",
        "| bound input | sha256 |",
        "| --- | --- |",
        f"| study manifest | `{results.inputs.study_manifest_sha256}` |",
        f"| evaluation configuration | `{results.inputs.evaluation_sha256}` |",
        f"| run ordering | `{results.inputs.run_ordering_sha256}` |",
        f"| representative rule | `{results.inputs.representative_rule_sha256}` |",
        f"| result schema v{results.result_schema_version} | `{results.inputs.result_schema_sha256}` |",
        f"| evidence pointers ({results.inputs.n_pointers}) | `{results.inputs.evidence_sha256}` |",
        "",
        "## Stored tables",
        "",
        "| table | record | rows | size (bytes) | sha256 | store location |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {t.name} | `{t.record}` | {t.n_rows} | {t.payload.size} | `{t.payload.sha256}` | `{t.payload.uri}` |"
        for t in results.tables
    ]
    lines += [
        "",
        "## Committed documents",
        "",
        "| file | record | entries | size (bytes) | sha256 |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [f"| `{d.name}` | `{d.record}` | {d.n_entries} | {d.size} | `{d.sha256}` |" for d in results.documents]
    lines += [
        "",
        "## Totals",
        "",
        "| quantity | value |",
        "| --- | --- |",
        f"| models | {results.n_models} |",
        f"| replay banks | {results.n_replay_banks} |",
        f"| RC runs | {results.n_rc_runs} |",
        f"| replay runs | {results.n_replay_runs} |",
        f"| unavailable runs | {results.n_unavailable_runs} |",
        f"| RC runs that met every criterion | {results.n_rc_successes} |",
        f"| replay runs that met every criterion | {results.n_replay_successes} |",
        f"| departure radius (m) | {results.departure_radius_m} |",
        "",
        "Generated by:",
        "",
        "```text",
        results.command,
        "```",
        "",
    ]
    return "\n".join(lines)


# --- the derivation ------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Loaded:
    """The study's evidence, read through its pointers and verified by digest."""

    models: dict[str, ManualModelEvidence]
    banks: dict[str, ManualReplayBank]
    bank_of: dict[tuple[str, str], str]
    """(configuration, parent) -> the bank identity the run ordering keys there."""


def _load_models(
    evidence_dir: Path, store: StorageRoot, manifest: StudyManifest, runner: ManualEvaluationRunner
) -> dict[str, ManualModelEvidence]:
    """Every study model a pointer names, verified by digest and then checked the way a resume checks it."""
    models: dict[str, ManualModelEvidence] = {}
    for entry in manifest.entries:
        path = evidence_dir / manual_pointer_name("model", entry.label)
        if not path.exists():
            continue
        pointer = load_manual_pointer(path)
        evidence = load_manual_model_evidence(verify_artifact(store, pointer.payload))
        if evidence.identity != pointer.identity:
            msg = f"{path.name} resolves to {evidence.label} {evidence.identity[:12]}, not its own"
            raise ValueError(msg)
        runner.verify_stored_model(entry, evidence, where=path.name)
        models[entry.label] = evidence
    return models


def _key_banks(
    banks: Mapping[str, tuple[str, ManualReplayBank]],
    ordering: RunOrdering,
    manifest: StudyManifest,
    runner: ManualEvaluationRunner,
) -> dict[tuple[str, str], str]:
    """Map each (configuration, parent) the run ordering keys to the bank its trusted conditions identify.

    The expected identity is derived from the evaluation configuration and the
    study's configuration, never from a stored bank, and a bank found under it
    is checked the way a resume checks it. A pointer naming a bank that no
    configuration's protocol produces is refused rather than left unused.
    """
    bank_of: dict[tuple[str, str], str] = {}
    for key in ordering.replay_banks:
        configuration = manifest.configuration(key.configuration)
        cutoffs = (key.velocity_cutoff_hz, key.acceleration_cutoff_hz)
        if (key.warmup_s, *cutoffs) != (
            configuration.warmup_s,
            configuration.velocity_cutoff_hz,
            configuration.acceleration_cutoff_hz,
        ):
            msg = f"the run ordering keys {key.configuration}/{key.assignment} under another protocol than the study's"
            raise ValueError(msg)
        identity = bank_identity(runner.conditions(key.warmup_s, cutoffs), key.assignment)
        found = banks.get(identity)
        if found is None:
            continue
        name, bank = found
        runner.verify_stored_bank(
            bank, assignment=key.assignment, warmup_s=key.warmup_s, replay_cutoffs=cutoffs, where=name
        )
        bank_of[key.configuration, key.assignment] = identity
    unkeyed = sorted(name for identity, (name, _) in banks.items() if identity not in bank_of.values())
    if unkeyed:
        msg = (
            f"{unkeyed} hold replay banks the study's protocol keys nowhere: no configuration's conditions and "
            f"parent produce them"
        )
        raise ValueError(msg)
    if len(set(bank_of.values())) != len(bank_of):
        msg = "a replay bank is keyed to more than one configuration; each row must belong to one"
        raise ValueError(msg)
    return bank_of


def _load_evidence(
    evidence_dir: Path,
    store: StorageRoot,
    manifest: StudyManifest,
    ordering: RunOrdering,
    runner: ManualEvaluationRunner,
) -> _Loaded:
    """Every model and bank the pointers name, each checked against the study's trusted inputs and keyed."""
    models = _load_models(evidence_dir, store, manifest, runner)
    banks: dict[str, tuple[str, ManualReplayBank]] = {}
    for path in sorted(evidence_dir.glob("replay__*.toml")):
        bank = load_manual_replay_bank(verify_artifact(store, load_manual_pointer(path).payload))
        if bank.identity in banks:
            msg = f"{path.name} and {banks[bank.identity][0]} point at the same replay bank"
            raise ValueError(msg)
        banks[bank.identity] = (path.name, bank)
    bank_of = _key_banks(banks, ordering, manifest, runner)
    return _Loaded(models=models, banks={identity: bank for identity, (_, bank) in banks.items()}, bank_of=bank_of)


def _jobs(
    loaded: _Loaded, manifest: StudyManifest, scenarios: Sequence[tuple[str, str]], trackers: Sequence[str]
) -> tuple[list[_Job], list[ManualRunRow]]:
    """One job per model and bank with evidence; unavailable rows for every model without it."""
    jobs: list[_Job] = []
    missing: list[ManualRunRow] = []
    for entry in manifest.entries:
        parent = entry.arm.assignment
        evidence = loaded.models.get(entry.label)
        origin = _Origin(
            source=_RC,
            configuration=entry.configuration,
            arm=entry.arm.label,
            arm_kind=entry.arm.arm,
            parent=parent,
            model_label=entry.label,
            evidence_identity=None if evidence is None else evidence.identity,
            fit_identity=None if evidence is None or evidence.fit is None else evidence.fit.identity,
        )
        if evidence is None:
            missing.extend(
                _unavailable_row(origin, index, sid, kind, tracker)
                for index, (sid, kind) in enumerate(scenarios)
                for tracker in trackers
            )
        else:
            jobs.append(_Job(origin, evidence.pairs, evidence.conditions.dwell_tolerance_m))
    for (configuration, parent), identity in loaded.bank_of.items():
        bank = loaded.banks[identity]
        origin = _Origin(
            source=_REPLAY,
            configuration=configuration,
            arm=arm_label(_REPLAY, parent),
            arm_kind=_REPLAY,
            parent=parent,
            model_label=f"{configuration}/{arm_label(_REPLAY, parent)}",
            evidence_identity=identity,
            fit_identity=None,
        )
        jobs.append(_Job(origin, bank.pairs, bank.conditions.dwell_tolerance_m))
    return jobs, missing


def verdicts_of(rows: Sequence[ManualRunRow]) -> list[ScenarioVerdict]:
    """One verdict per row for the comparisons: a run that was not simulated has none, never a failure."""
    return [
        ScenarioVerdict(
            configuration=row.configuration,
            tracker=row.tracker,
            arm_kind=row.arm_kind,
            parent=None if row.arm_kind == _ALL_TEN else row.parent,
            scenario_id=row.scenario_id,
            scenario_class=row.scenario_class,
            success=row.success if row.status in _SIMULATED else None,
        )
        for row in rows
    ]


def _store_table(
    store: StorageRoot, name: str, record: str, text: str, n_rows: int, columns: Sequence[str]
) -> ManualResultTable:
    """Write one table to the store under a content-addressed name, or verify the identical one already there."""
    data = text.encode("utf-8")
    digest = sha256_bytes(data)
    uri = ArtifactUri.parse(f"{RESULTS_PREFIX}/{name}-{digest[:12]}.csv")
    target = store.path(uri, mode="write")
    if target.exists():
        if sha256_file(target) != digest:
            msg = f"{uri} already holds other content"
            raise FileExistsError(msg)
    else:
        staged = target.with_name(f".{target.name}.staging")
        staged.write_bytes(data)
        staged.replace(target)
    return ManualResultTable(
        name=name,
        record=record,
        payload=ArtifactReference(uri=str(uri), sha256=digest, size=len(data)),
        n_rows=n_rows,
        columns=tuple(columns),
    )


@dataclass(frozen=True)
class _Derived:
    rows: tuple[ManualRunRow, ...]
    contrasts: tuple[ManualContrastRow, ...]
    contrast_summaries: tuple[ManualContrastSummary, ...]
    arm_summaries: tuple[ManualArmSummary, ...]
    selections: ManualSelections
    figures: ManualFigureInputs
    radius_m: float


def _derive(
    *,
    store: StorageRoot,
    manifest: StudyManifest,
    loaded: _Loaded,
    scenarios: Sequence[tuple[str, str]],
    trackers: Sequence[str],
    rule: RepresentativeRule,
    rule_sha256: str,
    ordering_sha256: str,
    scenario_file: str,
    root: Path,
    workers: int,
) -> _Derived:
    """Read and measure every run, then compute the comparisons, summaries, selections and figure inputs."""
    jobs, missing = _jobs(loaded, manifest, scenarios, trackers)
    radii = {job.radius_m for job in jobs}
    if len(radii) > 1:
        msg = f"the evidence was judged under several dwell radii {sorted(radii)}"
        raise ValueError(msg)

    if workers == 1:
        measured = [_rows_of(store, job) for job in jobs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            measured = list(pool.map(partial(_rows_of, store), jobs))
    rows = (*[row for rows in measured for row in rows], *missing)
    verdicts = verdicts_of(rows)
    contrasts = contrast_rows(verdicts, scenarios=scenarios)
    selections = selections_of(
        rows,
        rule,
        order=[sid for sid, _ in scenarios],
        rule_sha256=rule_sha256,
        ordering_sha256=ordering_sha256,
    )
    return _Derived(
        rows=rows,
        contrasts=contrasts,
        contrast_summaries=contrast_summaries(contrasts),
        arm_summaries=arm_summaries(verdicts, scenarios=scenarios),
        selections=selections,
        figures=figure_inputs(selections, rows, manifest=manifest, rule=rule, scenario_file=scenario_file, root=root),
        radius_m=radii.pop() if radii else DEPARTURE_RADIUS_M,
    )


def _write_documents(output: Path, derived: _Derived, accounting: StudyAccounting) -> tuple[ManualResultDocument, ...]:
    """Write the committed outputs, refusing to replace any that exists."""
    texts = {
        "accounting": (canonical_json(to_mapping(accounting)) + "\n", "StudyAccounting", len(accounting.models)),
        "arm_summary": (
            table_to_csv(derived.arm_summaries, ManualArmSummary),
            "ManualArmSummary",
            len(derived.arm_summaries),
        ),
        "contrast_summary": (
            table_to_csv(derived.contrast_summaries, ManualContrastSummary),
            "ManualContrastSummary",
            len(derived.contrast_summaries),
        ),
        "selections": (
            canonical_json(to_mapping(derived.selections)) + "\n",
            "ManualSelections",
            len(derived.selections.applications),
        ),
        "figure_inputs": (
            figure_inputs_to_json(derived.figures),
            "ManualFigureInputs",
            len(derived.figures.cases),
        ),
    }
    documents: list[ManualResultDocument] = []
    for role, (text, record, entries) in texts.items():
        target = output / RESULT_DOCUMENTS[role]
        data = text.encode("utf-8")
        target.write_bytes(data)
        documents.append(
            ManualResultDocument(
                name=target.name, record=record, sha256=sha256_bytes(data), size=len(data), n_entries=entries
            )
        )
    return tuple(documents)


def _refuse_existing(output: Path) -> None:
    targets = [output / name for name in RESULT_DOCUMENTS.values()]
    targets += [output / f"results_v{RESULTS_VERSION}.json", output / f"results_v{RESULTS_VERSION}.md"]
    existing = [path.name for path in targets if path.exists()]
    if existing:
        msg = f"refusing to overwrite {existing}: derived evidence is versioned, so derive a new version beside it"
        raise FileExistsError(msg)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        msg = f"{path} lies outside the repository; only repository files are cited"
        raise ValueError(msg) from error


def _derive_command(args: argparse.Namespace) -> int:
    """Derive every output from the verified evidence and write it once."""
    root = repository_root()
    output = Path(cast("str", args.output))
    _refuse_existing(output)
    evaluation_file = Path(cast("str", args.evaluation))
    ordering_file, rule_file = Path(cast("str", args.run_ordering)), Path(cast("str", args.representative_rule))
    schema_file, evidence_dir = Path(cast("str", args.result_schema)), Path(cast("str", args.evidence_dir))
    workers = int(cast("int", args.workers))
    if workers < 1:
        msg = f"workers must be at least 1, got {workers}"
        raise ValueError(msg)
    schema = load_schema(schema_file)
    if schema.schema_version != REFERENCED_RESULT_SCHEMA:
        msg = f"{schema_file.name} is result schema {schema.schema_version}, not {REFERENCED_RESULT_SCHEMA}"
        raise ValueError(msg)
    ordering = load_handoff(ordering_file, RunOrdering)
    rule = load_handoff(rule_file, RepresentativeRule)
    ordering_sha256 = sha256_file(ordering_file)
    if rule.ordering_sha256 != ordering_sha256:
        msg = "the representative rule was frozen against another run ordering"
        raise ValueError(msg)
    # The trusted inputs are the ones the sweep itself ran from: the canonical environment, the frozen
    # study with its verified demonstrations, the evaluation configuration and its locked scenarios.
    prepared = prepare_runner(args, role="main", root=root, module=_MODULE)
    runner, config, manifest = prepared.runner, prepared.config, prepared.context.manifest
    scenarios = tuple((case.scenario_id, str(case.kind)) for case in runner.scenarios)
    trackers = RECOVERY_TRACKERS
    if (
        tuple(sid for sid, _ in scenarios) != ordering.scenarios
        or tuple(trackers) != ordering.trackers
        or tuple(entry.label for entry in manifest.entries) != ordering.models
    ):
        msg = "the evaluation, trackers or study do not reproduce the frozen run ordering"
        raise ValueError(msg)
    evidence_sha256, n_pointers = evidence_digest(evidence_dir)
    store = runner.store
    loaded = _load_evidence(evidence_dir, store, manifest, ordering, runner)
    inputs = ManualResultInputs(
        study_manifest_sha256=prepared.context.manifest_sha256,
        evaluation_sha256=sha256_file(evaluation_file),
        run_ordering_sha256=ordering_sha256,
        representative_rule_sha256=sha256_file(rule_file),
        result_schema_sha256=sha256_file(schema_file),
        evidence_sha256=evidence_sha256,
        n_pointers=n_pointers,
    )
    command, provenance = runner.command, runner.provenance
    accounting = account_study(store=store, evidence_dir=evidence_dir, manifest=manifest, provenance=provenance)
    derived = _derive(
        store=store,
        manifest=manifest,
        loaded=loaded,
        scenarios=scenarios,
        trackers=trackers,
        rule=rule,
        rule_sha256=inputs.representative_rule_sha256,
        ordering_sha256=ordering_sha256,
        scenario_file=_relative(config.scenario, root),
        root=root,
        workers=workers,
    )
    tables = (
        _store_table(
            store,
            f"runs_v{RESULTS_VERSION}",
            "ManualRunRow",
            table_to_csv(derived.rows, ManualRunRow),
            len(derived.rows),
            table_columns(ManualRunRow),
        ),
        _store_table(
            store,
            f"contrasts_v{RESULTS_VERSION}",
            "ManualContrastRow",
            table_to_csv(derived.contrasts, ManualContrastRow),
            len(derived.contrasts),
            table_columns(ManualContrastRow),
        ),
    )
    output.mkdir(parents=True, exist_ok=True)
    documents = _write_documents(output, derived, accounting)
    rc = [row for row in derived.rows if row.source == _RC and row.status in _SIMULATED]
    replay = [row for row in derived.rows if row.source == _REPLAY and row.status in _SIMULATED]
    results = ManualResults(
        experiment=EXPERIMENT_LABEL,
        version=RESULTS_VERSION,
        result_schema_version=REFERENCED_RESULT_SCHEMA,
        inputs=inputs,
        tables=tables,
        documents=documents,
        n_models=len(loaded.models),
        n_replay_banks=len(loaded.bank_of),
        n_rc_runs=len(rc),
        n_replay_runs=len(replay),
        n_unavailable_runs=sum(1 for row in derived.rows if row.status not in _SIMULATED),
        n_rc_successes=sum(1 for row in rc if row.success),
        n_replay_successes=sum(1 for row in replay if row.success),
        departure_radius_m=derived.radius_m,
        command=command,
        provenance=provenance,
    )
    (output / f"results_v{RESULTS_VERSION}.json").write_text(results_to_json(results) + "\n", encoding="utf-8")
    (output / f"results_v{RESULTS_VERSION}.md").write_text(render_results_markdown(results), encoding="utf-8")
    print(
        json.dumps(
            {
                "rows": len(derived.rows),
                "rc_runs": results.n_rc_runs,
                "replay_runs": results.n_replay_runs,
                "unavailable_runs": results.n_unavailable_runs,
                "rc_successes": results.n_rc_successes,
                "replay_successes": results.n_replay_successes,
                "contrasts": len(derived.contrasts),
                "figure_cases": len(derived.figures.cases),
                "tables": [table.payload.uri for table in tables],
                "exploratory": bool(args.exploratory),
            },
            indent=2,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Derive the machine-readable evidence of the manual study.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    derive = subparsers.add_parser("derive", help="derive the tables, summaries, selections and figure inputs")
    for name, text in (
        ("--study", "the frozen study manifest"),
        ("--evaluation", "the evaluation configuration the sweep ran under"),
        ("--evidence-dir", "the directory of Git pointers the sweep wrote"),
        ("--run-ordering", "the frozen run ordering"),
        ("--representative-rule", "the frozen representative-case rule"),
        ("--result-schema", "the result schema the outputs follow"),
        ("--output", "the directory the committed outputs are written to"),
    ):
        derive.add_argument(name, dest=name.removeprefix("--").replace("-", "_"), required=True, help=text)
    derive.add_argument("--workers", type=int, default=1, help="threads reading and measuring runs")
    derive.add_argument(
        "--exploratory", action="store_true", help="tolerate a dirty worktree and mark the outputs exploratory"
    )
    args = parser.parse_args(argv)
    args.argv = argv
    return _derive_command(args)


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())

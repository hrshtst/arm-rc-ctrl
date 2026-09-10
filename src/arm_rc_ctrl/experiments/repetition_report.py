# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Report of the repeated-demonstration pilot (M3REP-007; repetition plan section 7.3, C4, C5).

From the executed pilot's evidence (the execution accounting, every model
manifest and replay bank, the numerical validation, and the timing smoke
check) this module derives the machine-readable report and its assets:

- the paired outcome table of all 120 behavioral configurations;
- the paired comparisons of plan section 7.2 (R and R-scaled against S within
  each formulation, both augmented arms against R at the same count, and each
  residual arm against its absolute counterpart), as medians of both arms and
  the median signed difference over the scenario/tracker pairs both arms
  completed, with the numbers of shared pairs stated;
- the numerical-equivalence errors of the validation, the C11 exception included;
- the per-configuration speed diagnostics under the 12 rad/s abort with the
  historical 6 rad/s crossings apart;
- measured fit and evaluation cost by arm and count, and the smoke check's
  peak memory;
- task-clock trajectory figures of the predeclared representatives and paired
  animations exported on the task clock (C5).

Every time axis is task-relative (run time minus the warm-up): the warm-up is
shaded and negative, activation is 0 s, and the ESN readout is missing before
activation while the tracker command (the hold, then the readout) is drawn
apart. The narrative ``overview.html`` is hand-written against this report
and bound to it by regression tests (C4).

Command line::

    python -m arm_rc_ctrl.experiments.repetition_report render --docs <docs> [--skip-animations] [--exploratory]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any, Final, cast

import matplotlib as mpl

mpl.use("Agg")  # headless rendering; the backend is fixed before pyplot is imported
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.data.records import write_record
from arm_rc_ctrl.execution import ExecutionRecord, load_execution
from arm_rc_ctrl.experiments.repetition_accounting import PilotAccounting, load_accounting
from arm_rc_ctrl.experiments.repetition_evaluation import (
    C11_CAVEAT,
    ModelEvidence,
    PairRecord,
    ReplayBank,
    load_model_evidence,
    load_pointer,
    load_replay_bank,
    pointer_name,
)
from arm_rc_ctrl.experiments.repetition_fits import FitStore
from arm_rc_ctrl.experiments.repetition_numerics import NumericalValidation, load_validation
from arm_rc_ctrl.experiments.repetition_panel import EXPERIMENT_LABEL, PanelManifest, load_panel
from arm_rc_ctrl.experiments.repetition_recipes import ArmSpec, panel_arms
from arm_rc_ctrl.experiments.repetition_timing import TimingReport, load_timing
from arm_rc_ctrl.experiments.run_record import LoadedRun, load_run, pointer_from_summary
from arm_rc_ctrl.provenance import (
    ProvenanceRecord,
    canonical_json,
    collect_provenance,
    command_line,
    require_clean_for_confirmatory,
    sha256_file,
    verify_artifact,
)
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageRoot, open_storage
from arm_rc_ctrl.validation import is_hex

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from arm_rc_ctrl.experiments.recovery_objective import RecoveryComponent

__all__ = [
    "ANIMATION_DIR",
    "ANIMATION_RULE",
    "PLOT_DIR",
    "REPORT_SCHEMA_VERSION",
    "REPRESENTATIVE_ARMS",
    "REPRESENTATIVE_RULE",
    "CostRow",
    "EquivalenceRow",
    "OutcomeRow",
    "PairedRow",
    "RepetitionReport",
    "ReportInputs",
    "Representative",
    "SpeedRow",
    "build_report",
    "build_report_inputs",
    "cost_rows",
    "equivalence_rows",
    "load_report",
    "main",
    "outcome_rows",
    "paired_rows",
    "plot_outcome_grid",
    "plot_peak_speeds",
    "plot_trajectory",
    "plot_worst_cell_by_count",
    "render_report_markdown",
    "representatives",
    "speed_rows",
    "write_animations",
    "write_plots",
]

REPORT_SCHEMA_VERSION: Final = 1
PLOT_DIR: Final = "plots/repetition_report_v1"
ANIMATION_DIR: Final = "animations/repetition_v1"
REPRESENTATIVE_ARMS: Final = (
    ArmSpec("absolute", "S"),
    ArmSpec("absolute", "R", 64),
    ArmSpec("absolute", "R-scaled", 64),
    ArmSpec("absolute", "A-contractive", 64),
    ArmSpec("residual", "S"),
)
REPRESENTATIVE_RULE: Final = (
    "For every panel entry, the first pair of the fixed evaluation order (the nominal scenario under pd_v2) of the "
    "absolute S, R/K65, R-scaled/K65, and A-contractive/K65 arms and of the residual S arm, whatever its outcome. "
    "Declared at the start of M3REP-007, after the execution accounting was visible; the rule selects by position "
    "in the panel, never by result, so failures are shown as often as successes."
)
ANIMATION_RULE: Final = (
    "The feasible-best entry's representative pair of the absolute S and absolute R/K65 arms (their RC runs) and "
    "the replay run they share, exported on the task clock so matching frames show matching task times."
)
_PAIRS_WITHIN: Final = {
    "absolute": (("R", "S"), ("R-scaled", "S"), ("A-non-decaying", "R"), ("A-contractive", "R")),
    "residual": (("R", "S"), ("R-scaled", "S")),
}
_ACROSS: Final = ("S", "R", "R-scaled")
_METRIC_FIELDS: Final = ("early_gap_integral", "activation_jump_rad", "gap_ratio", "settling_time_s", "torque_rms")
_SHA256_HEX: Final = 64
_MODULE: Final = "arm_rc_ctrl.experiments.repetition_report"
_CURVES: Final[dict[str, tuple[str, str, float, str]]] = {
    "replay_command": ("black", "--", 1.8, "replay tracker command (hold, then reference)"),
    "replay_actual": ("tab:blue", "-", 1.6, "replay actual"),
    "rc_command": ("tab:gray", "-", 1.0, "RC tracker command (hold, then readout)"),
    "rc_readout": ("tab:green", "--", 1.8, "ESN readout (absent before activation)"),
    "rc_actual": ("tab:orange", "-", 1.5, "RC actual"),
}
_STATUS_COLORS: Final = {
    "feasible": "tab:green",
    "rc_gate_failure": "tab:red",
    "replay_blocked": "tab:purple",
    "training_failure": "black",
}


# --- inputs -------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReportInputs:
    """Everything the report derives from, loaded and digest-verified."""

    docs: Path
    store: StorageRoot
    root: Path
    manifest: PanelManifest
    accounting: PilotAccounting
    validation: NumericalValidation
    timing: TimingReport
    execution: ExecutionRecord
    models: dict[str, ModelEvidence]
    """By ``<entry>/<arm label>``."""
    banks: dict[float, ReplayBank]
    """By warm-up."""
    fit_seconds: dict[str, float]
    """Fit time by fit identity, as the fit cache recorded it."""
    sources: dict[str, str]
    """SHA-256 of every committed evidence file the report binds."""


def build_report_inputs(docs: Path, *, store: StorageRoot, root: Path) -> ReportInputs:
    """Load the pilot's committed evidence and the manifests it points to (payloads digest-verified)."""
    manifest = load_panel(docs / "panel_manifest_v1.json")
    accounting = load_accounting(docs / "pilot_execution_v1.json")
    validation = load_validation(docs / "numerical_validation_v1.json")
    timing = load_timing(docs / "timing_smoke_check_v1.json")
    execution = load_execution(docs / "execution_environment_v1.json")
    if accounting.n_missing:
        msg = f"the accounting lists {accounting.n_missing} missing configurations; the report needs a complete panel"
        raise ValueError(msg)
    models: dict[str, ModelEvidence] = {}
    fits = FitStore(store)
    fit_seconds: dict[str, float] = {}
    for line in accounting.models:
        label = f"{line.panel_label}/{line.arm}"
        pointer = load_pointer(docs / "evidence" / pointer_name("model", label))
        evidence = load_model_evidence(verify_artifact(store, pointer.payload))
        models[label] = evidence
        if evidence.fit is not None and fits.exists(evidence.fit.identity):
            fit_seconds[evidence.fit.identity] = fits.read_record(evidence.fit.identity).fit_seconds
    for summary in validation.fits:
        if fits.exists(summary.identity):
            fit_seconds[summary.identity] = fits.read_record(summary.identity).fit_seconds
    banks: dict[float, ReplayBank] = {}
    for line in accounting.banks:
        pointer = load_pointer(docs / "evidence" / pointer_name("replay", f"warmup-{line.warmup_s:g}s"))
        banks[line.warmup_s] = load_replay_bank(verify_artifact(store, pointer.payload))
    sources = {
        name: sha256_file(docs / name)
        for name in (
            "panel_manifest_v1.json",
            "pilot_execution_v1.json",
            "numerical_validation_v1.json",
            "timing_smoke_check_v1.json",
            "execution_environment_v1.json",
        )
    }
    return ReportInputs(
        docs, store, root, manifest, accounting, validation, timing, execution, models, banks, fit_seconds, sources
    )


# --- tables -------------------------------------------------------------------------------


@dataclass(frozen=True)
class OutcomeRow:
    """One configuration's paired outcome."""

    panel_label: str
    formulation: str
    arm: str
    count: int
    status: str
    first_failure: str | None
    n_completed: int
    n_unexecuted: int
    crossed_historical: bool
    peak_speed: float | None
    """Largest measured joint speed over the executed RC runs (rad/s)."""
    worst_cell: float | None
    """The largest class-by-tracker median early-gap ratio (the recovery objective), feasible models only."""
    cells: dict[str, float]


def _peak_speed(evidence: ModelEvidence) -> float | None:
    peaks = [j.peak_abs_speed for p in evidence.pairs if p.velocity is not None for j in p.velocity.joints]
    return max(peaks) if peaks else None


def outcome_rows(inputs: ReportInputs) -> tuple[OutcomeRow, ...]:
    """The paired outcome table in panel and report order."""
    rows: list[OutcomeRow] = []
    for label, evidence in inputs.models.items():
        entry, arm_label = label.split("/", 1)
        arm = _arm_of(arm_label)
        rows.append(
            OutcomeRow(
                panel_label=entry,
                formulation=arm.formulation,
                arm=arm_label,
                count=arm.count,
                status=evidence.status,
                first_failure=evidence.first_failure,
                n_completed=evidence.n_completed,
                n_unexecuted=evidence.n_unexecuted,
                crossed_historical=evidence.crossed_historical,
                peak_speed=_peak_speed(evidence),
                worst_cell=max(evidence.cells.values()) if evidence.cells else None,
                cells=dict(evidence.cells),
            )
        )
    return tuple(rows)


def _arm_of(label: str) -> ArmSpec:
    for arm in panel_arms():
        if arm.label == label:
            return arm
    msg = f"unknown arm label {label!r}"
    raise ValueError(msg)


@dataclass(frozen=True)
class PairedRow:
    """One paired comparison of plan section 7.2 at one entry and count."""

    panel_label: str
    comparison: str
    """``<left> vs <right>`` in arm labels."""
    count: int
    left: str
    right: str
    left_status: str
    right_status: str
    verdict_changed: bool
    shared_pairs: int
    """Scenario/tracker pairs both arms completed feasibly (the comparison's denominator)."""
    left_median: dict[str, float]
    right_median: dict[str, float]
    signed_difference_median: dict[str, float]
    """Median over the shared pairs of ``left - right`` per metric (original units)."""
    left_worst_cell: float | None
    right_worst_cell: float | None


def _completed(evidence: ModelEvidence) -> dict[tuple[str, str], RecoveryComponent]:
    out: dict[tuple[str, str], RecoveryComponent] = {}
    for pair in evidence.pairs:
        if pair.status == "completed" and pair.rc is not None:
            out[(pair.scenario_id, pair.tracker)] = pair.rc
    return out


def _paired(entry: str, left_label: str, right_label: str, count: int, inputs: ReportInputs) -> PairedRow | None:
    left = inputs.models.get(f"{entry}/{left_label}")
    right = inputs.models.get(f"{entry}/{right_label}")
    if left is None or right is None:
        return None
    a, b = _completed(left), _completed(right)
    shared = sorted(set(a) & set(b))
    left_median: dict[str, float] = {}
    right_median: dict[str, float] = {}
    difference: dict[str, float] = {}
    for metric in _METRIC_FIELDS:
        pairs = [
            (cast("float", getattr(a[key], metric)), cast("float", getattr(b[key], metric)))
            for key in shared
            if getattr(a[key], metric) is not None and getattr(b[key], metric) is not None
        ]
        if pairs:
            left_median[metric] = float(median(x for x, _ in pairs))
            right_median[metric] = float(median(y for _, y in pairs))
            difference[metric] = float(median(x - y for x, y in pairs))
    return PairedRow(
        panel_label=entry,
        comparison=f"{left_label} vs {right_label}",
        count=count,
        left=left_label,
        right=right_label,
        left_status=left.status,
        right_status=right.status,
        verdict_changed=left.status != right.status,
        shared_pairs=len(shared),
        left_median=left_median,
        right_median=right_median,
        signed_difference_median=difference,
        left_worst_cell=max(left.cells.values()) if left.cells else None,
        right_worst_cell=max(right.cells.values()) if right.cells else None,
    )


def paired_rows(inputs: ReportInputs) -> tuple[PairedRow, ...]:
    """Every comparison of plan section 7.2 at every entry and count (within and across formulations)."""
    rows: list[PairedRow] = []
    counts = (17, 33, 65)
    for entry in inputs.manifest.rule.labels:
        for formulation, pairs in _PAIRS_WITHIN.items():
            for count in counts:
                for left_arm, right_arm in pairs:
                    left = ArmSpec(formulation, left_arm, count - 1).label
                    right = ArmSpec(formulation, right_arm, None if right_arm == "S" else count - 1).label
                    row = _paired(entry, left, right, count, inputs)
                    if row is not None:
                        rows.append(row)
        for arm_name in _ACROSS:
            for count in (1,) if arm_name == "S" else counts:
                residual = ArmSpec("residual", arm_name, None if arm_name == "S" else count - 1).label
                absolute = ArmSpec("absolute", arm_name, None if arm_name == "S" else count - 1).label
                row = _paired(entry, residual, absolute, count, inputs)
                if row is not None:
                    rows.append(row)
    return tuple(rows)


@dataclass(frozen=True)
class EquivalenceRow:
    """One numerical-equivalence comparison of the validation (M3REP-003)."""

    panel_label: str
    formulation: str
    count: int
    candidate: str
    reference: str
    quantity: str
    max_abs: float
    max_rel: float
    coefficient_fro_rel: float
    passed: bool
    accepted_exception: bool


def equivalence_rows(inputs: ReportInputs) -> tuple[EquivalenceRow, ...]:
    """The validation's 72 comparisons, one row per compared quantity, with the C11 exception flagged."""
    return tuple(
        EquivalenceRow(
            panel_label=c.panel_label,
            formulation=c.formulation,
            count=c.count,
            candidate=c.candidate,
            reference=c.reference,
            quantity=d.quantity,
            max_abs=d.max_abs,
            max_rel=d.max_rel,
            coefficient_fro_rel=c.coefficient_fro_rel,
            passed=d.passed,
            accepted_exception=not d.passed,
        )
        for c in inputs.validation.comparisons
        for d in c.differences
    )


@dataclass(frozen=True)
class SpeedRow:
    """One configuration's joint-speed diagnostics over its executed RC runs."""

    panel_label: str
    arm: str
    executed_runs: int
    crossed_runs: int
    """Executed RC runs in which any joint exceeded the historical 6 rad/s limit."""
    peak_speed: float | None
    aborts: int
    """RC runs stopped by the 12 rad/s abort."""
    time_above_historical_s: dict[str, float]
    """Seconds above the historical limit summed over the executed runs and joints, by phase."""


def speed_rows(inputs: ReportInputs) -> tuple[SpeedRow, ...]:
    """Per configuration: crossings, peaks, aborts, and time above 6 rad/s by phase."""
    rows: list[SpeedRow] = []
    for label, evidence in inputs.models.items():
        entry, arm_label = label.split("/", 1)
        executed = [p for p in evidence.pairs if p.velocity is not None]
        phases: dict[str, float] = {"warmup": 0.0, "movement": 0.0, "dwell": 0.0}
        for pair in executed:
            for joint in cast("Any", pair.velocity).joints:
                for phase, seconds in joint.time_above_historical_s.items():
                    phases[phase] += seconds
        rows.append(
            SpeedRow(
                panel_label=entry,
                arm=arm_label,
                executed_runs=len(executed),
                crossed_runs=sum(1 for p in executed if p.crossed_historical),
                peak_speed=_peak_speed(evidence),
                aborts=sum(1 for p in executed if cast("Any", p.velocity).abort is not None),
                time_above_historical_s=phases,
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class CostRow:
    """Measured cost of one arm at one count, averaged over the entries."""

    formulation: str
    arm: str
    count: int
    models: int
    fits_timed: int
    fit_seconds_mean: float | None
    evaluation_seconds_mean: float
    """Mean over the entries of the sum of the model's run times (simulation plus persistence)."""
    executed_runs_mean: float


def cost_rows(inputs: ReportInputs) -> tuple[CostRow, ...]:
    """Fit and evaluation cost by arm and count (S-effective from the numerical validation's fits)."""
    rows: list[CostRow] = []
    for arm in panel_arms():
        models = [e for label, e in inputs.models.items() if label.endswith(f"/{arm.label}")]
        if arm.behavioral:
            fit_ids = [e.fit.identity for e in models if e.fit is not None]
            evaluation = [
                sum(p.timing.simulate_seconds + p.timing.persist_seconds for p in e.pairs if p.timing is not None)
                for e in models
            ]
            runs = [float(e.n_completed + e.n_infeasible) for e in models]
        else:
            fit_ids = [f.identity for f in inputs.validation.fits if f.arm == arm]
            evaluation, runs = [0.0] * len(fit_ids), [0.0] * len(fit_ids)
        timed = [inputs.fit_seconds[i] for i in fit_ids if i in inputs.fit_seconds]
        rows.append(
            CostRow(
                formulation=arm.formulation,
                arm=arm.label,
                count=arm.count,
                models=len(fit_ids),
                fits_timed=len(timed),
                fit_seconds_mean=float(np.mean(timed)) if timed else None,
                evaluation_seconds_mean=float(np.mean(evaluation)) if evaluation else 0.0,
                executed_runs_mean=float(np.mean(runs)) if runs else 0.0,
            )
        )
    return tuple(rows)


# --- representatives -----------------------------------------------------------------------


@dataclass(frozen=True)
class Representative:
    """One predeclared representative pair and its assets."""

    panel_label: str
    arm: str
    scenario_id: str
    tracker: str
    status: str
    """The pair's status (``completed`` or ``infeasible``; ``unexecuted`` when the sweep never reached it)."""
    reason: str | None
    rc_run: str | None
    replay_run: str | None
    warmup_s: float
    plot: str | None
    rc_animation: str | None = None
    replay_animation: str | None = None


def _first_pair(evidence: ModelEvidence) -> PairRecord:
    return evidence.pairs[0]


def representatives(inputs: ReportInputs) -> tuple[Representative, ...]:
    """The representatives the predeclared rule selects, with the runs they refer to (assets attached later)."""
    out: list[Representative] = []
    for entry in inputs.manifest.rule.labels:
        warmup = inputs.manifest.entry(entry).warmup_s
        for arm in REPRESENTATIVE_ARMS:
            evidence = inputs.models.get(f"{entry}/{arm.label}")
            if evidence is None:  # an entry without evidence (a partial fixture); the accounting reports it
                continue
            bank = inputs.banks[warmup]
            pair = _first_pair(evidence)
            replay = bank.get(pair.scenario_id, pair.tracker)
            out.append(
                Representative(
                    panel_label=entry,
                    arm=arm.label,
                    scenario_id=pair.scenario_id,
                    tracker=pair.tracker,
                    status=pair.status,
                    reason=None if pair.rc is None else pair.rc.reason,
                    rc_run=None if pair.run is None else pair.run.artifact_id,
                    replay_run=None if replay is None or replay.run is None else replay.run.artifact_id,
                    warmup_s=warmup,
                    plot=None,
                )
            )
    return tuple(out)


def _slug(label: str) -> str:
    return label.replace("/", "__")


# --- figures --------------------------------------------------------------------------------


def _save(fig: Any, out: Path) -> Path:  # noqa: ANN401 - a matplotlib figure
    if out.exists():
        msg = f"refusing to overwrite {out}"
        raise FileExistsError(msg)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=out.stem, suffix=".tmp.png", dir=out.parent, delete=False) as handle:
        staged = Path(handle.name)
    try:
        fig.savefig(staged, dpi=100, format="png", metadata={"Software": None}, pil_kwargs={"optimize": True})
        staged.replace(out)
    finally:
        plt.close(fig)
        staged.unlink(missing_ok=True)
    return out


def plot_outcome_grid(rows: Sequence[OutcomeRow], out: Path, *, formulation: str, entries: Sequence[str]) -> Path:
    """Entries by arms, colored by status, annotated with the number of completed pairs."""
    selected = [r for r in rows if r.formulation == formulation]
    arms = list(dict.fromkeys(r.arm for r in selected))
    width = max(6.0, 1.5 + 0.9 * len(arms))
    height = max(3.0, 1.6 + 0.55 * len(entries))
    fig, axis = cast("tuple[Any, Any]", plt.subplots(figsize=(width, height), constrained_layout=True))
    for y, entry in enumerate(entries):
        for x, arm in enumerate(arms):
            row = next((r for r in selected if r.panel_label == entry and r.arm == arm), None)
            if row is None:
                continue
            axis.add_patch(Rectangle((x - 0.45, y - 0.4), 0.9, 0.8, color=_STATUS_COLORS[row.status], alpha=0.75))
            text = f"{row.n_completed}" + ("*" if row.crossed_historical else "")
            axis.text(x, y, text, ha="center", va="center", fontsize=8, color="white")
    axis.set_xlim(-0.6, max(len(arms), 1) - 0.4)
    axis.set_ylim(-0.6, max(len(entries), 1) - 0.4)
    axis.set_xticks(range(len(arms)), [a.split("/", 1)[1] for a in arms], rotation=45, ha="right", fontsize=8)
    axis.set_yticks(range(len(entries)), list(entries), fontsize=8)
    axis.invert_yaxis()
    axis.set_title(
        f"{formulation} output: completed pairs of 130 (green feasible, red RC-gate failure; * crossed 6 rad/s)",
        fontsize=8,
    )
    return _save(fig, out)


def plot_worst_cell_by_count(rows: Sequence[OutcomeRow], out: Path, *, entries: Sequence[str]) -> Path:
    """The recovery objective (worst class-by-tracker median early-gap ratio) against K for S, R, and R-scaled."""
    fig, axes = cast(
        "tuple[Any, Any]",
        plt.subplots(2, 3, figsize=(12, 6.5), sharex=True, sharey=True, squeeze=False, constrained_layout=True),
    )
    styles = {"S": ("black", "o"), "R": ("tab:orange", "s"), "R-scaled": ("tab:blue", "^")}
    for axis, entry in zip(axes.flat, entries, strict=False):
        for arm_name, (color, marker) in styles.items():
            points = [
                (r.count, r.worst_cell)
                for r in rows
                if r.panel_label == entry and r.formulation == "absolute" and r.arm.split("/")[1] == arm_name
            ]
            xs = [k for k, v in points if v is not None]
            ys = [v for _, v in points if v is not None]
            axis.plot(xs, ys, marker=marker, color=color, label=arm_name)
            for k, v in points:
                if v is None:
                    axis.plot([k], [1.0], marker="x", color=color, markersize=9, linestyle="none")
        axis.axhline(1.0, color="0.6", linewidth=0.8, linestyle=":")
        axis.set_title(entry, fontsize=9)
        axis.set_xticks([1, 17, 33, 65])
        axis.grid(visible=True, alpha=0.3)
    axes[0, 0].legend(fontsize="small")
    for axis in axes[-1]:
        axis.set_xlabel("episode count K")
    for axis in axes[:, 0]:
        axis.set_ylabel("worst cell median ratio (x = infeasible)")
    return _save(fig, out)


def plot_peak_speeds(rows: Sequence[SpeedRow], out: Path, *, historical: float, abort: float) -> Path:
    """Peak measured joint speed of every configuration against the historical and the abort limits."""
    fig, axis = cast("tuple[Any, Any]", plt.subplots(figsize=(14, 4.5), constrained_layout=True))
    labels = [f"{r.panel_label[:14]}/{r.arm.split('/', 1)[1]}" for r in rows]
    values = [0.0 if r.peak_speed is None else r.peak_speed for r in rows]
    colors = ["tab:red" if r.aborts else ("tab:orange" if r.crossed_runs else "tab:green") for r in rows]
    axis.bar(range(len(rows)), values, color=colors)
    axis.axhline(historical, color="0.3", linestyle="--", linewidth=1.0, label=f"historical limit {historical:g} rad/s")
    axis.axhline(abort, color="tab:red", linestyle=":", linewidth=1.0, label=f"evaluation abort {abort:g} rad/s")
    axis.set_xticks(range(len(rows)), labels, rotation=90, fontsize=5)
    axis.set_ylabel("peak measured joint speed (rad/s)")
    axis.set_title("Peak speed over executed RC runs (green: below 6; orange: crossed 6; red: aborted at 12)")
    axis.legend(fontsize="small", loc="upper right")
    return _save(fig, out)


def _curve(axis: Any, t: NDArray[np.float64], values: NDArray[np.float64], name: str) -> None:  # noqa: ANN401
    color, style, width, label = _CURVES[name]
    axis.plot(t, values, color=color, linestyle=style, linewidth=width, label=label)


def _shifted(run: LoadedRun, warmup_s: float) -> NDArray[np.float64]:
    return cast("NDArray[np.float64]", run.arrays.arrays["t"]) - warmup_s


def _column(run: LoadedRun, name: str, joint: int) -> NDArray[np.float64]:
    return cast("NDArray[np.float64]", run.arrays.arrays[name])[:, joint]


def _positions(axis: Any, joint: int, rc: LoadedRun | None, replay: LoadedRun, warmup_s: float) -> None:  # noqa: ANN401
    _curve(axis, _shifted(replay, warmup_s), _column(replay, "q_desired", joint), "replay_command")
    _curve(axis, _shifted(replay, warmup_s), _column(replay, "q", joint), "replay_actual")
    if rc is not None:
        _curve(axis, _shifted(rc, warmup_s), _column(rc, "q_desired", joint), "rc_command")
        if "generator_output_q" in rc.arrays.arrays:
            _curve(axis, _shifted(rc, warmup_s), _column(rc, "generator_output_q", joint), "rc_readout")
        _curve(axis, _shifted(rc, warmup_s), _column(rc, "q", joint), "rc_actual")
    axis.set_ylabel(f"q{joint + 1} (rad)")


def _speeds(
    axis: Any,  # noqa: ANN401
    joint: int,
    rc: LoadedRun | None,
    replay: LoadedRun,
    warmup_s: float,
    historical: float,
    abort: float,
) -> None:
    axis.plot(
        _shifted(replay, warmup_s), _column(replay, "dq", joint), color="tab:blue", linewidth=1.2, label="replay actual"
    )
    if rc is not None:
        axis.plot(
            _shifted(rc, warmup_s), _column(rc, "dq", joint), color="tab:orange", linewidth=1.2, label="RC actual"
        )
    for bound, color, style in ((historical, "0.3", "--"), (abort, "tab:red", ":")):
        axis.axhline(bound, color=color, linestyle=style, linewidth=0.8)
        axis.axhline(-bound, color=color, linestyle=style, linewidth=0.8)
    axis.set_ylabel(f"dq{joint + 1} (rad/s)")


def _torques(axis: Any, rc: LoadedRun | None, replay: LoadedRun, warmup_s: float, dof: int) -> None:  # noqa: ANN401
    name = "tau_applied" if "tau_applied" in replay.arrays.arrays else "tau_requested"
    for joint in range(dof):
        alpha = 0.5 + 0.5 * joint / max(dof - 1, 1)
        axis.plot(
            _shifted(replay, warmup_s),
            _column(replay, name, joint),
            color="tab:blue",
            linewidth=1.0,
            alpha=alpha,
            label=f"replay tau{joint + 1}",
        )
        if rc is not None:
            axis.plot(
                _shifted(rc, warmup_s),
                _column(rc, name, joint),
                color="tab:orange",
                linewidth=1.0,
                alpha=alpha,
                label=f"RC tau{joint + 1}",
            )
    axis.set_ylabel("torque (N m)")


def _increments(axis: Any, joint: int, rc: LoadedRun, warmup_s: float) -> None:  # noqa: ANN401
    axis.plot(
        _shifted(rc, warmup_s),
        _column(rc, "generator_increment_q", joint),
        color="tab:green",
        linewidth=1.2,
        label="predicted increment (absent before 0)",
    )
    axis.set_ylabel(f"dq{joint + 1} increment (rad)")


def plot_trajectory(
    rc: LoadedRun | None,
    replay: LoadedRun,
    out: Path,
    *,
    title: str,
    warmup_s: float,
    dwell_start_s: float,
    historical: Sequence[float],
    abort: Sequence[float],
    force_window: tuple[float, float] | None = None,
) -> Path:
    """Joint positions, speeds, and torques on the task clock with the hold, readout, and commands apart.

    The replay's tracker command (the held start, then the task reference) is
    the black dashed reference; the RC tracker command is drawn thin and gray
    from the hold on; the ESN readout is drawn only from activation (it is
    absent before); a residual model gets one more panel per joint with its
    predicted increment. Warm-up ``[-T_w, 0)`` is shaded; activation is at 0 s.
    """
    dof = replay.arrays.dof
    residual = rc is not None and "generator_increment_q" in rc.arrays.arrays
    rows = dof * 2 + 1 + (dof if residual else 0)
    fig, axes = cast(
        "tuple[Any, Any]",
        plt.subplots(rows, 1, figsize=(10, 2.1 * rows), sharex=True, squeeze=False, constrained_layout=True),
    )
    panels = axes[:, 0]
    for joint in range(dof):
        _positions(panels[joint], joint, rc, replay, warmup_s)
        _speeds(panels[dof + joint], joint, rc, replay, warmup_s, historical[joint], abort[joint])
    _torques(panels[2 * dof], rc, replay, warmup_s, dof)
    if residual and rc is not None:
        for joint in range(dof):
            _increments(panels[2 * dof + 1 + joint], joint, rc, warmup_s)
    for axis in panels:
        if warmup_s > 0:
            axis.axvspan(-warmup_s, 0.0, color="0.85", alpha=0.6, label="warm-up / hold")
        axis.axvline(0.0, color="0.2", linewidth=0.9)
        axis.axvline(dwell_start_s, color="0.65", linewidth=0.8, linestyle=":")
        if force_window is not None:
            axis.axvspan(force_window[0], force_window[1], color="tab:red", alpha=0.12)
        axis.grid(visible=True, alpha=0.3)
    panels[0].set_title(title)
    panels[0].legend(loc="best", fontsize="x-small", ncol=3)
    panels[dof].legend(loc="best", fontsize="x-small", ncol=2)
    panels[2 * dof].legend(loc="best", fontsize="x-small", ncol=4)
    panels[-1].set_xlabel(
        f"Task time (s)   |   warm-up: [-{warmup_s:g}, 0)   activation: 0   dwell from {dwell_start_s:g}"
    )
    return _save(fig, out)


def _load(inputs: ReportInputs, artifact_id: str) -> LoadedRun:
    return load_run(inputs.store, pointer_from_summary(inputs.store, artifact_id))


def write_plots(
    inputs: ReportInputs, reps: Sequence[Representative], out_dir: Path
) -> tuple[list[str], list[Representative]]:
    """Write every figure of the report; returns the figure names and the representatives with their plots attached."""
    entries = list(inputs.manifest.rule.labels)
    outcomes = outcome_rows(inputs)
    names = ["outcomes_absolute.png", "outcomes_residual.png", "worst_cell_by_count.png", "peak_speed.png"]
    plot_outcome_grid(outcomes, out_dir / names[0], formulation="absolute", entries=entries)
    plot_outcome_grid(outcomes, out_dir / names[1], formulation="residual", entries=entries)
    plot_worst_cell_by_count(outcomes, out_dir / names[2], entries=entries)
    replay_conditions = next(iter(inputs.banks.values())).conditions
    plot_peak_speeds(
        speed_rows(inputs),
        out_dir / names[3],
        historical=replay_conditions.historical_velocity_limit[0],
        abort=replay_conditions.velocity_abort[0],
    )
    dwell_start = _dwell_start(inputs)
    attached: list[Representative] = []
    for rep in reps:
        if rep.replay_run is None:
            attached.append(rep)
            continue
        replay = _load(inputs, rep.replay_run)
        rc = None if rep.rc_run is None else _load(inputs, rep.rc_run)
        name = f"trajectory_{_slug(rep.panel_label)}__{_slug(rep.arm)}.png"
        plot_trajectory(
            rc,
            replay,
            out_dir / name,
            title=(
                f"{rep.panel_label} / {rep.arm}: {rep.scenario_id} [{rep.tracker}] "
                f"({rep.status}{'' if rep.reason is None else ': ' + rep.reason})"
            ),
            warmup_s=rep.warmup_s,
            dwell_start_s=dwell_start,
            historical=replay_conditions.historical_velocity_limit,
            abort=replay_conditions.velocity_abort,
        )
        names.append(name)
        attached.append(replace(rep, plot=name))
    return names, attached


def _dwell_start(inputs: ReportInputs) -> float:
    """The task-clock dwell start every run's speed diagnostics recorded (identical across the panel)."""
    starts = {p.velocity.dwell_start_s for bank in inputs.banks.values() for p in bank.pairs if p.velocity is not None}
    if len(starts) != 1:
        msg = f"the replay banks record {len(starts)} distinct dwell starts; expected exactly one"
        raise ValueError(msg)
    return starts.pop()


def write_animations(
    inputs: ReportInputs,
    reps: Sequence[Representative],
    out_dir: Path,
    *,
    fps: float = 12.0,
    player: Callable[[Sequence[str]], int] | None = None,
) -> list[Representative]:
    """Export the animation rule's runs on the task clock (C5); returns the representatives with their GIFs attached."""
    from arm_rc_ctrl.experiments.playback import main_play

    play = main_play if player is None else player
    scenario_file = inputs.root / inputs.manifest.configs.scenario_file
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = {
        (r.panel_label, r.arm): r
        for r in reps
        if r.panel_label == "feasible-best"
        and r.arm in (ArmSpec("absolute", "S").label, ArmSpec("absolute", "R", 64).label)
    }
    updated = list(reps)
    with tempfile.TemporaryDirectory(prefix="arm-rc-ctrl-pointers-") as scratch:
        exported: dict[str, str] = {}

        def export(run_id: str, name: str) -> str:
            if run_id in exported:
                return exported[run_id]
            pointer_file = Path(scratch) / f"{run_id}.toml"
            write_record(pointer_file, pointer_from_summary(inputs.store, run_id))
            target = out_dir / name
            if target.exists():
                msg = f"refusing to overwrite {target}"
                raise FileExistsError(msg)
            status = play(
                [
                    "--pointer",
                    str(pointer_file),
                    "--scenario",
                    str(scenario_file),
                    "--export",
                    str(target),
                    "--fps",
                    f"{fps:g}",
                    "--task-clock",
                ]
            )
            if status != 0:
                msg = f"the player failed to export {name} (status {status})"
                raise RuntimeError(msg)
            exported[run_id] = name
            return name

        for key, rep in selected.items():
            rc_name = (
                None
                if rep.rc_run is None
                else export(rep.rc_run, f"{_slug(rep.panel_label)}__{_slug(rep.arm)}__rc.gif")
            )
            replay_name = (
                None
                if rep.replay_run is None
                else export(rep.replay_run, f"{_slug(rep.panel_label)}__replay_{rep.tracker}.gif")
            )
            index = next(i for i, r in enumerate(updated) if (r.panel_label, r.arm) == key)
            updated[index] = replace(rep, rc_animation=rc_name, replay_animation=replay_name)
    return updated


# --- the report -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RepetitionReport:
    """The committed machine-readable report."""

    experiment: str
    sources: dict[str, str]
    canonical_execution_identity: str
    c11_caveat: str
    representative_rule: str
    animation_rule: str
    outcomes: tuple[OutcomeRow, ...]
    paired: tuple[PairedRow, ...]
    equivalence: tuple[EquivalenceRow, ...]
    speeds: tuple[SpeedRow, ...]
    costs: tuple[CostRow, ...]
    representatives: tuple[Representative, ...]
    peak_rss_bytes: int
    """Process-cumulative peak RSS of the timing smoke check (the panel run recorded none per model)."""
    velocity_abort: tuple[float, ...]
    historical_velocity_limit: tuple[float, ...]
    plots: tuple[str, ...]
    animations: tuple[str, ...]
    n_feasible: int
    n_rc_gate_failure: int
    n_crossed_historical: int
    provenance: ProvenanceRecord
    schema_version: int = field(default=REPORT_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        """Counts re-derive from the outcome rows; the caveat is the C11 text."""
        if self.schema_version != REPORT_SCHEMA_VERSION or self.experiment != EXPERIMENT_LABEL:
            msg = "unsupported report schema or experiment"
            raise ValueError(msg)
        if not is_hex(self.canonical_execution_identity, _SHA256_HEX):
            msg = "canonical_execution_identity must be 64 lowercase hex characters"
            raise ValueError(msg)
        counts = (
            sum(1 for r in self.outcomes if r.status == "feasible"),
            sum(1 for r in self.outcomes if r.status == "rc_gate_failure"),
            sum(1 for r in self.outcomes if r.crossed_historical),
        )
        if (self.n_feasible, self.n_rc_gate_failure, self.n_crossed_historical) != counts:
            msg = "the recorded counts contradict the outcome rows"
            raise ValueError(msg)
        if self.c11_caveat != C11_CAVEAT:
            msg = "the report carries the C11 caveat verbatim"
            raise ValueError(msg)


def build_report(
    inputs: ReportInputs,
    *,
    reps: Sequence[Representative],
    plots: Sequence[str],
    animations: Sequence[str],
    provenance: ProvenanceRecord,
) -> RepetitionReport:
    """Assemble the report from the inputs and the assets written for it."""
    outcomes = outcome_rows(inputs)
    conditions = next(iter(inputs.banks.values())).conditions
    return RepetitionReport(
        experiment=EXPERIMENT_LABEL,
        sources=dict(inputs.sources),
        canonical_execution_identity=inputs.execution.identity,
        c11_caveat=C11_CAVEAT,
        representative_rule=REPRESENTATIVE_RULE,
        animation_rule=ANIMATION_RULE,
        outcomes=outcomes,
        paired=paired_rows(inputs),
        equivalence=equivalence_rows(inputs),
        speeds=speed_rows(inputs),
        costs=cost_rows(inputs),
        representatives=tuple(reps),
        peak_rss_bytes=inputs.timing.peak_rss_bytes,
        velocity_abort=conditions.velocity_abort,
        historical_velocity_limit=conditions.historical_velocity_limit,
        plots=tuple(plots),
        animations=tuple(animations),
        n_feasible=sum(1 for r in outcomes if r.status == "feasible"),
        n_rc_gate_failure=sum(1 for r in outcomes if r.status == "rc_gate_failure"),
        n_crossed_historical=sum(1 for r in outcomes if r.crossed_historical),
        provenance=provenance,
    )


def report_to_json(report: RepetitionReport) -> str:
    """Canonical JSON."""
    return canonical_json(to_mapping(report))


def load_report(path: Path) -> RepetitionReport:
    """Strictly rebuild the report from JSON."""
    return from_mapping(cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), RepetitionReport)


def _f(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}g}"


def _outcome_line(o: OutcomeRow) -> str:
    return (
        f"| {o.panel_label} | {o.formulation} | {o.arm.split('/', 1)[1]} | {o.count} | {o.status} | {o.n_completed} "
        f"| {o.n_unexecuted} | {'yes' if o.crossed_historical else 'no'} | {_f(o.peak_speed, 3)} "
        f"| {_f(o.worst_cell)} | {o.first_failure or ''} |"
    )


def _paired_line(p: PairedRow) -> str:
    gap, jump = "early_gap_integral", "activation_jump_rad"
    return (
        f"| {p.panel_label} | {p.comparison} | {p.count} | {p.left_status} | {p.right_status} | {p.shared_pairs} "
        f"| {_f(p.left_median.get(gap))} | {_f(p.right_median.get(gap))} | {_f(p.signed_difference_median.get(gap))} "
        f"| {_f(p.left_median.get(jump))} | {_f(p.right_median.get(jump))} "
        f"| {_f(p.signed_difference_median.get(jump))} | {_f(p.left_worst_cell)} | {_f(p.right_worst_cell)} |"
    )


def _equivalence_line(e: EquivalenceRow) -> str:
    decision = "pass" if e.passed else "**accepted exception (C11)**"
    return (
        f"| {e.panel_label} | {e.formulation} | {e.count} | {e.candidate.split('/', 1)[1]} "
        f"| {e.reference.split('/', 1)[1]} | {e.quantity} | {e.max_abs:.3e} | {e.max_rel:.3e} "
        f"| {e.coefficient_fro_rel:.3e} | {decision} |"
    )


def _speed_line(s: SpeedRow) -> str:
    above = s.time_above_historical_s
    return (
        f"| {s.panel_label} | {s.arm} | {s.executed_runs} | {s.crossed_runs} | {_f(s.peak_speed, 3)} | {s.aborts} "
        f"| {above['warmup']:.2f} | {above['movement']:.2f} | {above['dwell']:.2f} |"
    )


def _cost_line(c: CostRow) -> str:
    return (
        f"| {c.formulation} | {c.arm.split('/', 1)[1]} | {c.count} | {c.models} | {c.fits_timed} "
        f"| {_f(c.fit_seconds_mean, 3)} | {c.evaluation_seconds_mean:.1f} | {c.executed_runs_mean:.1f} |"
    )


def _representative_line(x: Representative) -> str:
    animations = ", ".join(a for a in (x.rc_animation, x.replay_animation) if a)
    return (
        f"| {x.panel_label} | {x.arm} | {x.scenario_id} [{x.tracker}] | {x.status} | {x.reason or ''} "
        f"| {x.rc_run or ''} | {x.replay_run or ''} | {x.plot or ''} | {animations} |"
    )


def render_report_markdown(report: RepetitionReport) -> str:
    """The Markdown rendering of every table of the report."""
    r = report
    lines = [
        "# Task 1-a repetition pilot report tables (v1)",
        "",
        (
            f"Experiment `{r.experiment}`; canonical execution identity `{r.canonical_execution_identity[:12]}`; "
            f"project commit `{r.provenance.project_commit[:12]}`{' (dirty)' if r.provenance.project_dirty else ''}; "
            f"evaluation abort {list(r.velocity_abort)} rad/s with the historical limit "
            f"{list(r.historical_velocity_limit)} rad/s reported as a diagnostic."
        ),
        "",
        (
            f"- Feasible configurations: {r.n_feasible} of {len(r.outcomes)}; RC-gate failures: "
            f"{r.n_rc_gate_failure}; configurations that crossed the historical limit in an executed run: "
            f"{r.n_crossed_historical}."
        ),
        f"- Representative rule: {r.representative_rule}",
        f"- Animation rule: {r.animation_rule}",
        f"- {r.c11_caveat}",
        "",
        "## Paired outcomes",
        "",
        (
            "| entry | formulation | arm | K | status | completed | unexecuted | crossed 6 rad/s | peak rad/s "
            "| worst cell | first failure |"
        ),
        "| --- | --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | --- |",
        *(_outcome_line(o) for o in r.outcomes),
        "",
        "## Paired comparisons (medians over the scenario/tracker pairs both arms completed)",
        "",
        (
            "| entry | comparison | K | left status | right status | shared pairs | early gap left "
            "| early gap right | early gap diff | jump left | jump right | jump diff | worst cell left "
            "| worst cell right |"
        ),
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *(_paired_line(p) for p in r.paired),
        "",
        "## Numerical equivalence (M3REP-003)",
        "",
        "| entry | formulation | K | candidate | reference | quantity | max abs | max rel | coef fro rel | decision |",
        "| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- |",
        *(_equivalence_line(e) for e in r.equivalence),
        "",
        "## Speed diagnostics",
        "",
        (
            "| entry | arm | executed | crossed 6 rad/s | peak rad/s | aborts at 12 | above 6 warm-up s "
            "| above 6 movement s | above 6 dwell s |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *(_speed_line(s) for s in r.speeds),
        "",
        "## Measured cost by arm and count",
        "",
        (
            f"Peak resident set size of the timing smoke check (process-cumulative, 20 models): "
            f"{r.peak_rss_bytes / 2**20:.1f} MiB; the panel run recorded no per-model memory figure."
        ),
        "",
        "| formulation | arm | K | models | fits timed | fit s mean | evaluation s mean | executed runs mean |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *(_cost_line(c) for c in r.costs),
        "",
        "## Representatives",
        "",
        "| entry | arm | pair | status | reason | RC run | replay run | plot | animations |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        *(_representative_line(x) for x in r.representatives),
        "",
        "## Limitations",
        "",
        (
            "- The panel is six historical configurations of one demonstration; the augmentation anchor and seed "
            "bank are fixed; first-failure censoring hides later behavior (plan section 7.3)."
        ),
        (
            "- Feasibility is under the 12 rad/s evaluation abort; crossings of the historical 6 rad/s limit are "
            "reported apart and are not recovery-v1 successes."
        ),
        (
            "- Signed differences use only the pairs both arms completed; comparisons with few shared pairs "
            "describe those pairs only."
        ),
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Render the repeated-demonstration pilot's report tables and assets.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    render = subparsers.add_parser("render", help="derive the report tables and write the figures and animations")
    render.add_argument("--docs", type=str, required=True, help="the experiment's docs directory")
    render.add_argument("--skip-animations", action="store_true", help="do not export the GIF animations")
    render.add_argument("--exploratory", action="store_true", help="allow a dirty worktree")
    args = parser.parse_args(argv)
    docs = Path(cast("str", args.docs))
    output, markdown = docs / "repetition_report_v1.json", docs / "repetition_report_v1.md"
    for target in (output, markdown):
        if target.exists():
            msg = f"refusing to overwrite {target}"
            raise FileExistsError(msg)
    root = repository_root()
    store = open_storage()
    inputs = build_report_inputs(docs, store=store, root=root)
    resolved = {"sources": inputs.sources, "command": command_line(_MODULE, argv)}
    provenance = collect_provenance(
        resolved, seeds={}, artifacts=[], exploratory=bool(args.exploratory), now=datetime.now(tz=UTC)
    )
    require_clean_for_confirmatory(provenance)
    reps = representatives(inputs)
    plots, reps = write_plots(inputs, reps, docs / PLOT_DIR)
    animations: list[str] = []
    if not bool(args.skip_animations):
        reps = write_animations(inputs, reps, docs / ANIMATION_DIR)
        animations = sorted({a for r in reps for a in (r.rc_animation, r.replay_animation) if a})
    report = build_report(inputs, reps=reps, plots=plots, animations=animations, provenance=provenance)
    output.write_text(report_to_json(report) + "\n", encoding="utf-8")
    markdown.write_text(render_report_markdown(report), encoding="utf-8")
    print(json.dumps({"feasible": report.n_feasible, "plots": len(plots), "animations": len(animations)}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())

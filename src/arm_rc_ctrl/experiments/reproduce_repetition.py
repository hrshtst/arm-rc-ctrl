# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Reproduce the task 1-a repeated-demonstration pilot from the committed records (M3REP-008; C3, C9).

One command resolves the external records the committed evidence points at and,
without touching them, re-resolves the panel manifest from the source study
payload, checks every prescribed recipe identity, refits all 156 fits in fresh
processes, verifies every stored evidence payload (the 123 manifests, every
run they name, the fit cache, and the dataset), recomputes every recorded
metric and diagnostic from the stored arrays and terminal-state records,
re-derives the report tables and the accounting, re-simulates a deterministic
subset of runs under the rule declared in this module, and regenerates every
figure and animation byte for byte. Everything it writes goes to a scratch
store outside the repository; nothing is deleted and no mismatch is accepted.
The historical DOC-005 task 1-a reproduction is rerun separately, on request,
and reported apart with its own tolerance untouched.

Command line::

    python scripts/reproduce_repetition.py [--scratch DIR] [--summary FILE] [--audit FILE]
        [--keep-going] [--exploratory] [--skip-resimulation] [--skip-fits]
        [--doc005] [--doc005-cpus LIST] [--gates] [--from-checkout [COMMIT]]

Launch it through the canonical execution environment
(``python -m arm_rc_ctrl.execution run --policy p-cores -- ...``): the
environment step requires the execution identity the evidence binds. The
re-simulation step requires a clean checkout; ``--exploratory`` lets the other
steps run in a dirty worktree and makes that step fail clearly.
``--from-checkout`` reproduces inside a fresh detached worktree at the given
commit (default ``HEAD``) after a fresh ``uv sync --locked``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any, Final, Literal, cast

import numpy as np

from arm_rc_ctrl.config import to_mapping
from arm_rc_ctrl.data.arrays import array_digest
from arm_rc_ctrl.data.recovery import RecoveryDatasetRecord, load_processed_record, task_intervals_from_phases
from arm_rc_ctrl.dependencies import submodule_revisions, verify_builds
from arm_rc_ctrl.execution import ExecutionRecord, collect_execution, load_execution, parse_cpu_list, require_canonical
from arm_rc_ctrl.experiments.baselines import frozen_baseline_digest, load_frozen_baseline
from arm_rc_ctrl.experiments.evidence import load_report_pointer, open_stored_report
from arm_rc_ctrl.experiments.perturbations import RobustnessScenario, load_development_robustness, robustness_scenarios
from arm_rc_ctrl.experiments.recovery_ablation import load_ablation
from arm_rc_ctrl.experiments.recovery_objective import (
    RecoveryComponent,
    ReplayComponent,
    blocked_component,
    recovery_component,
    replay_component,
)
from arm_rc_ctrl.experiments.recovery_search import RECOVERY_TRACKERS, load_recovery_search
from arm_rc_ctrl.experiments.repetition_accounting import (
    PilotAccounting,
    account_pilot,
    load_accounting,
    render_accounting_markdown,
)
from arm_rc_ctrl.experiments.repetition_diagnosis import load_diagnosis, render_diagnosis_markdown
from arm_rc_ctrl.experiments.repetition_evaluation import (
    ModelEvidence,
    PairRecord,
    PilotRunner,
    ReplayBank,
    load_evaluation_config,
    load_pointer,
    numerical_binding,
    pointer_name,
)
from arm_rc_ctrl.experiments.repetition_fits import FitStore
from arm_rc_ctrl.experiments.repetition_numerics import (
    FreshRefit,
    NumericalValidation,
    PanelContext,
    load_validation,
    refit_in_subprocess,
    render_validation_markdown,
)
from arm_rc_ctrl.experiments.repetition_panel import (
    PanelManifest,
    build_panel_manifest,
    load_panel,
    render_panel_markdown,
    resolve_panel,
)
from arm_rc_ctrl.experiments.repetition_recipes import ArmSpec, panel_arms, readout_for_arm, training_spec_for_arm
from arm_rc_ctrl.experiments.repetition_report import (
    ANIMATION_DIR,
    PLOT_DIR,
    RepetitionReport,
    ReportInputs,
    Representative,
    build_report,
    build_report_inputs,
    load_report,
    render_report_markdown,
    representatives,
    write_animations,
    write_plots,
)
from arm_rc_ctrl.experiments.repetition_timing import load_timing, render_timing_markdown
from arm_rc_ctrl.experiments.reproduce_1a import Check, ReproductionError, prepare_scratch
from arm_rc_ctrl.experiments.reproduce_recovery import compare_evidence
from arm_rc_ctrl.experiments.run_record import RUN_ARRAYS_FILE, LoadedRun, load_run, pointer_from_summary
from arm_rc_ctrl.experiments.simulation import CheckedState
from arm_rc_ctrl.experiments.velocity_diagnostics import VelocityDiagnostics, velocity_diagnostics
from arm_rc_ctrl.provenance import (
    ArtifactReference,
    collect_provenance,
    command_line,
    require_clean_for_confirmatory,
    sha256_file,
    verify_artifact,
    worktree_state,
)
from arm_rc_ctrl.rc.esn import ensure_single_thread
from arm_rc_ctrl.rc.recipe import solver_alpha
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageError, StorageRoot, open_storage

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from arm_rc_ctrl.experiments.baselines import TrackerConfig
    from arm_rc_ctrl.experiments.repetition_evaluation import SimulateFn
    from arm_rc_ctrl.experiments.repetition_panel import PanelEntry

__all__ = [
    "DECLARED_COMPARISONS",
    "DOC005_TOLERANCE",
    "RESIMULATION_RULE",
    "STEPS",
    "DeclaredComparison",
    "Doc005Rerun",
    "GateResult",
    "PrescribedFit",
    "RepetitionReproduction",
    "Reproducer",
    "Selection",
    "SubsetCoverage",
    "audit_markdown",
    "boundary_jump_from_arrays",
    "checked_states_from_run",
    "main",
    "prescribed_fits",
    "reproduce",
    "resimulation_subset",
    "run_doc005",
    "run_from_checkout",
    "run_gates",
    "subset_coverage",
    "verify_fit_cache",
    "verify_run_payloads",
]

_MODULE: Final = "arm_rc_ctrl.experiments.reproduce_repetition"
REPO: Final = repository_root()
DOCS: Final = REPO / "docs" / "experiments" / "task_1a_repeated_demonstration"
RECOVERY_DOCS: Final = REPO / "docs" / "experiments" / "task_1a_state_conditioned_recovery"
RETENTION_INCIDENT: Final = "retention_incident_2026-09-10.md"
STEPS: Final = (
    "environment",
    "storage",
    "records",
    "payloads",
    "recipes",
    "fits",
    "metrics",
    "tables",
    "resimulation",
    "assets",
)
_REPRODUCTION_COMMAND: Final = f"python -m {_MODULE} (reproduction rerun)"
DOC005_TOLERANCE: Final = 0.0
"""The historical task 1-a reproduction's exact-zero tolerance; never changed here (plan section 8)."""

RESIMULATION_RULE: Final = (
    "Deterministic re-simulation subset (C3), declared in this module at the start of M3REP-008 after the pilot "
    "had executed and its results were visible; it selects by position in the fixed evaluation order and by the "
    "recorded first-failure position, never by a metric value: (1) every behavioral configuration's first pair "
    "(the nominal scenario under pd_v2), which covers both formulations, every arm and count, every warm-up, "
    "and the failure example of every configuration that failed at its first pair; (2) the first infeasible "
    "pair of every configuration whose first failure came later; (3) for every configuration that completed "
    "every pair, the first pair of each perturbation class under each tracker; (4) for every replay bank, the "
    "first pair of each perturbation class under each tracker. Every selected run is re-simulated into the "
    "scratch store with a refitted model and compared bitwise (array digests) and exactly (recomputed component, "
    "diagnostics, and status) against the stored run; coverage that the evidence cannot offer is reported."
)


@dataclass(frozen=True)
class DeclaredComparison:
    """One comparison the reproduction performs, with its kind, units, and reason (plan section 8)."""

    name: str
    kind: Literal["bitwise", "exact", "tolerance"]
    units: str
    reason: str
    tolerance: float | None = None


DECLARED_COMPARISONS: Final = (
    DeclaredComparison(
        "panel manifest",
        "exact",
        "records",
        "the manifest is a deterministic function of the digest-verified study payload and configs",
    ),
    DeclaredComparison(
        "recipe identities and prescriptions",
        "exact",
        "digests, ridge parameters",
        "fit identities and solver rules are prescribed by the plan and bind the canonical environment",
    ),
    DeclaredComparison(
        "refitted readout weights and reservoir states",
        "bitwise",
        "float64 arrays",
        "the fit cache identity binds the execution environment and rclib commit; a same-environment refit must "
        "reproduce the cached arrays bit for bit (M3REP-003 fresh-process rule)",
    ),
    DeclaredComparison(
        "stored payload digests",
        "bitwise",
        "sha256, bytes",
        "every manifest, run summary, array file, fit-cache entry, and the dataset must match its recorded digest",
    ),
    DeclaredComparison(
        "recomputed pair components, diagnostics, and cells",
        "exact",
        "rad, rad/s, s, ratios, counts",
        "metrics are pure functions of the stored arrays, the terminal-state record, and the recorded conditions",
    ),
    DeclaredComparison(
        "report tables and accounting",
        "exact",
        "records",
        "tables are pure functions of the verified manifests",
    ),
    DeclaredComparison(
        "re-simulated runs",
        "bitwise",
        "float64 arrays (array digests), records",
        "the canonical single-threaded execution environment is deterministic; the evidence binds its identity",
    ),
    DeclaredComparison(
        "figures and animations",
        "bitwise",
        "bytes",
        "the generators are deterministic given the verified runs and the pinned libraries",
    ),
    DeclaredComparison(
        "historical task 1-a reproduction (DOC-005)",
        "tolerance",
        "metric units of the confirmatory suite",
        "the historical exact-zero tolerance is reported as recorded and never changed (plan section 8)",
        DOC005_TOLERANCE,
    ),
)


@dataclass(frozen=True)
class Selection:
    """One run the re-simulation subset selects."""

    kind: Literal["rc", "replay"]
    entry: str
    arm: str | None
    scenario_id: str
    tracker: str
    reason: str
    """Which clause of the rule selected it."""


@dataclass(frozen=True)
class SubsetCoverage:
    """What the selected subset covers, and what the evidence could not offer."""

    runs: int
    rc_runs: int
    replay_runs: int
    formulations: tuple[str, ...]
    arms: tuple[str, ...]
    warmups_s: tuple[float, ...]
    trackers: tuple[str, ...]
    classes: tuple[str, ...]
    first_pair_failures: int
    later_failures: int
    unavailable: tuple[str, ...]


@dataclass(frozen=True)
class Doc005Rerun:
    """One rerun of the historical task 1-a reproduction, reported apart from this pilot."""

    label: str
    cpus: tuple[int, ...]
    command: str
    returncode: int
    ok: bool | None
    max_deviation: float | None
    tolerance: float
    elapsed_s: float
    detail: str


@dataclass(frozen=True)
class GateResult:
    """One repository quality gate run from the reproduction."""

    command: str
    returncode: int
    elapsed_s: float


@dataclass(frozen=True)
class RepetitionReproduction:
    """The whole reproduction: checks, environment, inputs, coverage, and the separate DOC-005 status."""

    started_at: str
    checks: tuple[Check, ...]
    inputs: dict[str, str]
    environment: dict[str, str]
    max_deviation: float | None
    elapsed_s: float
    comparisons: tuple[DeclaredComparison, ...]
    resimulation_rule: str
    coverage: SubsetCoverage | None
    doc005: tuple[Doc005Rerun, ...]
    gates: tuple[GateResult, ...]
    schema_version: int = field(default=1)

    @property
    def ok(self) -> bool:
        """Whether every step ran and passed, every requested gate passed, and every DOC-005 rerun completed.

        A DOC-005 rerun counts as completed when it wrote its summary; its own
        verdict is reported apart and never decides this reproduction's.
        """
        steps = len(self.checks) == len(STEPS) and all(c.ok for c in self.checks)
        gates = all(g.returncode == 0 for g in self.gates)
        reruns = all(r.ok is not None for r in self.doc005)
        return steps and gates and reruns


# --- helpers ------------------------------------------------------------------------------


def _need[T](value: T | None, step: str) -> T:
    if value is None:
        msg = f"step {step!r} requires an earlier step that did not run"
        raise ReproductionError(msg)
    return value


def _exact(name: str, committed: object, rebuilt: object) -> None:
    """Fail unless two records are identical (categorically and to the last float)."""
    differences: list[str] = []
    worst = compare_evidence(name, committed, rebuilt, differences)
    if differences or worst > 0.0:
        first = differences[0] if differences else f"largest float deviation {worst:.3e}"
        msg = f"{name}: {len(differences)} categorical differences, largest deviation {worst:.3e} ({first})"
        raise ReproductionError(msg)


def checked_states_from_run(run: LoadedRun, pair: PairRecord) -> list[CheckedState]:
    """Rebuild the states the simulator checked from the telemetry rows plus the recorded terminal state.

    Every telemetry row is a checked state (the simulator checks before it
    commands); a terminal offending state never enters the telemetry, so it is
    taken from the pair's recorded diagnostics.
    """
    arrays = run.arrays.arrays
    t, q, dq = arrays["t"], arrays["q"], arrays["dq"]
    states = [CheckedState(t=float(t[i]), step=i, q=np.array(q[i]), dq=np.array(dq[i])) for i in range(t.shape[0])]
    diagnostics = pair.velocity
    if not run.summary.termination.is_completed:
        if diagnostics is None or diagnostics.terminal_state is None:
            msg = f"{pair.scenario_id}/{pair.tracker}: a non-completed run needs its recorded terminal state"
            raise ReproductionError(msg)
        terminal = diagnostics.terminal_state
        states.append(
            CheckedState(
                t=terminal.run_time_s,
                step=terminal.step,
                q=np.array(terminal.q, dtype=np.float64),
                dq=np.array(terminal.dq, dtype=np.float64),
            )
        )
    return states


def boundary_jump_from_arrays(run: LoadedRun) -> float | None:
    """The controller's boundary jump from the telemetry: the first generated target against the held posture."""
    arrays = run.arrays.arrays
    phase = arrays.get("phase")
    if phase is None:
        return None
    generating = np.flatnonzero(np.asarray(phase) > 0)
    if generating.size == 0:
        return None
    first = int(generating[0])
    return float(np.abs(arrays["q_desired"][first] - arrays["q"][0]).max())


def _select_model(label: str, evidence: ModelEvidence, add: Callable[[Selection], None]) -> None:
    entry, arm = label.split("/", 1)
    executed = [p for p in evidence.pairs if p.run is not None]
    if not executed:
        return
    first = executed[0]
    add(Selection("rc", entry, arm, first.scenario_id, first.tracker, "first pair"))
    if evidence.status == "rc_gate_failure":
        failed = next((p for p in evidence.pairs if p.status == "infeasible"), None)
        if failed is not None and failed is not first:
            add(Selection("rc", entry, arm, failed.scenario_id, failed.tracker, "first infeasible pair"))
    if evidence.status == "feasible" and evidence.n_completed == evidence.n_pairs:
        seen: set[tuple[str, str]] = set()
        for pair in executed:
            key = (pair.kind, pair.tracker)
            if key not in seen:
                seen.add(key)
                add(Selection("rc", entry, arm, pair.scenario_id, pair.tracker, "first pair of class and tracker"))


def _select_bank(entry: str, bank: ReplayBank, add: Callable[[Selection], None]) -> None:
    seen: set[tuple[str, str]] = set()
    for pair in bank.pairs:
        key = (pair.kind, pair.tracker)
        if key not in seen and pair.run is not None:
            seen.add(key)
            add(Selection("replay", entry, None, pair.scenario_id, pair.tracker, "first pair of class and tracker"))


def _bank_entry(inputs: ReportInputs, warmup: float) -> str:
    return next((e.label for e in inputs.manifest.entries if e.warmup_s == warmup), f"warmup-{warmup:g}s")


def resimulation_subset(inputs: ReportInputs) -> tuple[Selection, ...]:
    """Apply :data:`RESIMULATION_RULE` to the verified evidence."""
    selected: dict[tuple[str, str, str | None, str, str], Selection] = {}

    def add(selection: Selection) -> None:
        key = (selection.kind, selection.entry, selection.arm, selection.scenario_id, selection.tracker)
        selected.setdefault(key, selection)

    for label, evidence in inputs.models.items():
        _select_model(label, evidence, add)
    for warmup, bank in sorted(inputs.banks.items()):
        _select_bank(_bank_entry(inputs, warmup), bank, add)
    return tuple(selected.values())


def subset_coverage(selections: Sequence[Selection], inputs: ReportInputs) -> SubsetCoverage:
    """What the selections cover and what the evidence could not offer (reported explicitly, C3)."""
    rc = [s for s in selections if s.kind == "rc"]
    replay = [s for s in selections if s.kind == "replay"]
    arms = sorted({s.arm for s in rc if s.arm is not None})
    kinds = {p.kind for bank in inputs.banks.values() for p in bank.pairs}
    selected_rc = {(s.entry, s.arm, s.scenario_id, s.tracker) for s in rc}
    covered_kinds = {
        pair.kind
        for label, evidence in inputs.models.items()
        for pair in evidence.pairs
        if (*label.split("/", 1), pair.scenario_id, pair.tracker) in selected_rc
    }
    first_pair_failures = sum(
        1
        for e in inputs.models.values()
        if e.status == "rc_gate_failure" and e.pairs and e.pairs[0].status == "infeasible"
    )
    unavailable: list[str] = []
    if not any(p.status == "replay_blocked" for e in inputs.models.values() for p in e.pairs):
        unavailable.append("replay-blocked pairs: none exist in the evidence")
    if not any(e.status == "training_failure" for e in inputs.models.values()):
        unavailable.append("training failures: none exist in the evidence")
    missing_kinds = sorted(kinds - covered_kinds)
    if missing_kinds:
        unavailable.append(f"perturbation classes without an executed RC run: {missing_kinds}")
    return SubsetCoverage(
        runs=len(selections),
        rc_runs=len(rc),
        replay_runs=len(replay),
        formulations=tuple(sorted({a.split("/", 1)[0] for a in arms})),
        arms=tuple(arms),
        warmups_s=tuple(sorted({e.warmup_s for e in inputs.manifest.entries})),
        trackers=tuple(sorted({s.tracker for s in selections})),
        classes=tuple(sorted(covered_kinds)),
        first_pair_failures=first_pair_failures,
        later_failures=sum(1 for s in rc if s.reason == "first infeasible pair"),
        unavailable=tuple(unavailable),
    )


def _strip_assets(reps: Sequence[Representative]) -> list[dict[str, object]]:
    stripped: list[dict[str, object]] = []
    for rep in reps:
        row = to_mapping(rep)
        for key in ("plot", "rc_animation", "replay_animation"):
            row.pop(key, None)
        stripped.append(row)
    return stripped


@dataclass(frozen=True)
class PrescribedFit:
    """One of the prescribed fits: where the evidence records it and what its cache entry must hold."""

    identity: str
    panel_label: str
    arm: ArmSpec
    solver_alpha: float
    weights_sha256: str
    source: Literal["validation", "evidence"]


def prescribed_fits(validation: NumericalValidation, inputs: ReportInputs) -> dict[str, PrescribedFit]:
    """The validation's fits plus the evidence-bound fits it does not cover (the augmented arms)."""
    fits = {
        f.identity: PrescribedFit(f.identity, f.panel_label, f.arm, f.solver_alpha, f.weights_sha256, "validation")
        for f in validation.fits
    }
    for evidence in inputs.models.values():
        binding = evidence.fit
        if binding is not None and binding.identity not in fits:
            fits[binding.identity] = PrescribedFit(
                binding.identity,
                binding.panel_label,
                binding.arm,
                binding.solver_alpha,
                binding.weights_sha256,
                "evidence",
            )
    return fits


def _verify_run_payload(store: StorageRoot, pair: PairRecord) -> int:
    run = cast("Any", pair.run)
    summary_file = verify_artifact(store, ArtifactReference(uri=run.uri, sha256=run.sha256, size=run.size))
    arrays_file = summary_file.parent / RUN_ARRAYS_FILE
    if not arrays_file.is_file():
        msg = f"{pair.scenario_id}/{pair.tracker}: {run.artifact_id} has no arrays file"
        raise ReproductionError(msg)
    digest = sha256_file(arrays_file)
    if digest != run.arrays_sha256:
        msg = f"{pair.scenario_id}/{pair.tracker}: arrays digest {digest[:12]} != recorded {run.arrays_sha256[:12]}"
        raise ReproductionError(msg)
    return arrays_file.stat().st_size


def verify_run_payloads(store: StorageRoot, inputs: ReportInputs) -> tuple[int, int]:
    """Every run the manifests name resolves with its recorded summary and array digests; (runs, array bytes)."""
    runs = 0
    arrays_bytes = 0
    pairs = [p for e in inputs.models.values() for p in e.pairs]
    pairs += [p for b in inputs.banks.values() for p in b.pairs]
    for pair in pairs:
        if pair.run is not None:
            arrays_bytes += _verify_run_payload(store, pair)
            runs += 1
    return runs, arrays_bytes


def verify_fit_cache(fits: FitStore, prescribed: Mapping[str, PrescribedFit], inputs: ReportInputs) -> int:
    """Every prescribed fit is cached with its recorded weight digest, and every evidence binding matches its record."""
    for fit in prescribed.values():
        if not fits.exists(fit.identity):
            msg = f"fit {fit.identity[:12]} ({fit.panel_label}/{fit.arm.label}) is not cached"
            raise ReproductionError(msg)
        record = fits.read_record(fit.identity)
        weights = fits.read_weights(record)
        if array_digest(weights) != record.weights_sha256 or record.weights_sha256 != fit.weights_sha256:
            msg = f"fit {fit.identity[:12]}: cached weights do not match the recorded digest"
            raise ReproductionError(msg)
    for label, evidence in inputs.models.items():
        binding = evidence.fit
        if binding is None:
            continue
        record = fits.read_record(binding.identity)
        recorded = (record.recipe_sha256, record.weights_sha256, record.fit.rmse, record.fit.loss_rows)
        bound = (binding.recipe_sha256, binding.weights_sha256, binding.fit_rmse, binding.loss_rows)
        if recorded != bound:
            msg = f"{label}: the evidence's fit binding differs from the cached fit record"
            raise ReproductionError(msg)
    return len(prescribed)


# --- the steps ----------------------------------------------------------------------------


@dataclass
class _Context:
    """What the metric recomputation and the re-simulation share (rebuilt, never read from the evidence)."""

    scenarios: tuple[RobustnessScenario, ...]
    dwell_start_s: float
    settling_band_rad: float
    saturation_bound: float
    velocity_abort: tuple[float, ...]
    trackers: dict[str, TrackerConfig]
    tracker_digests: dict[str, str]
    evaluation_file: Path
    development_sha256: str

    def case(self, scenario_id: str) -> tuple[int, RobustnessScenario]:
        """The evaluation-order index and the scenario of ``scenario_id``."""
        for index, case in enumerate(self.scenarios):
            if case.scenario_id == scenario_id:
                return index, case
        msg = f"unknown scenario {scenario_id!r}"
        raise ReproductionError(msg)


@dataclass
class Reproducer:
    """The steps, in order, sharing what they resolve (public so tests can exercise one step directly)."""

    scratch: Path
    exploratory: bool
    configured_store: StorageRoot | None
    docs: Path
    now: datetime | None
    skip_resimulation: bool = False
    skip_fits: bool = False
    simulate_fn: SimulateFn | None = None
    refit: Callable[..., FreshRefit] = refit_in_subprocess
    root: Path = REPO
    """The repository the evidence's relative paths resolve against (a fixture root in tests)."""
    scenarios: tuple[RobustnessScenario, ...] | None = None
    """Injected development scenarios (tests); otherwise derived from the bound development file."""
    trackers: dict[str, TrackerConfig] | None = None
    tracker_digests: dict[str, str] | None = None
    prescribed: dict[str, PrescribedFit] | None = None
    entries: dict[str, PanelEntry] | None = None
    """Injected panel entries (tests); otherwise the manifest's."""
    inputs: dict[str, str] = field(default_factory=dict)
    max_deviation: float | None = None
    coverage: SubsetCoverage | None = None
    store: StorageRoot | None = None
    scratch_store: StorageRoot | None = None
    report: RepetitionReport | None = None
    execution: ExecutionRecord | None = None
    current_execution: ExecutionRecord | None = None
    context: PanelContext | None = None
    report_inputs: ReportInputs | None = None
    metrics_context: _Context | None = None
    reps: tuple[Representative, ...] | None = None

    # -- environment and storage ------------------------------------------------------------

    def environment(self) -> str:
        """Pins, build identities, the lock digest, and the execution identity match the committed evidence."""
        report = load_report(self.docs / "repetition_report_v1.json")
        self.report = report
        canonical = load_execution(self.docs / "execution_environment_v1.json")
        self.execution = canonical
        if report.canonical_execution_identity != canonical.identity:
            msg = "the report's canonical execution identity is not the committed execution record's"
            raise ReproductionError(msg)
        builds = verify_builds(REPO)
        current = {s.name: (s.checked_out or s.recorded) for s in submodule_revisions(REPO)}
        recorded = {s.name: (s.checked_out or s.recorded) for s in report.provenance.submodules}
        mismatched = sorted(name for name in recorded if current.get(name) != recorded[name])
        if mismatched:
            msg = f"submodule pins differ from the committed evidence: {mismatched}"
            raise ReproductionError(msg)
        lock = sha256_file(REPO / "uv.lock")
        if lock != report.provenance.lock_sha256:
            msg = f"uv.lock digest {lock[:12]} differs from the evidence's {report.provenance.lock_sha256[:12]}"
            raise ReproductionError(msg)
        require_canonical()
        for name in ("numpy", "rclib"):
            importlib.import_module(name)
        execution = collect_execution(command=_REPRODUCTION_COMMAND, role="main", now=self.now or datetime.now(tz=UTC))
        execution.check_canonical()
        self.current_execution = execution
        commit, dirty = worktree_state(REPO)
        self.inputs.update(
            {
                "evidence_project_commit": report.provenance.project_commit,
                "reproduction_project_commit": commit,
                "reproduction_project_dirty": str(dirty),
                "canonical_execution_identity": canonical.identity,
                "reproduction_execution_identity": execution.identity,
            }
        )
        if execution.identity != canonical.identity:
            msg = (
                f"this process runs in execution environment {execution.identity[:12]}, not the canonical "
                f"{canonical.identity[:12]} the evidence binds (C10); launch through the pinned launcher"
            )
            raise ReproductionError(msg)
        return (
            f"{len(builds)} build identities verified; submodules {sorted(recorded)}, uv.lock, and the canonical "
            f"execution identity {canonical.identity[:12]} match the evidence"
        )

    def storage(self) -> str:
        """The external storage root resolves (never the repository) and the scratch store is created."""
        self.store = open_storage() if self.configured_store is None else self.configured_store
        store_dir = self.scratch / "store"
        store_dir.mkdir()
        self.scratch_store = StorageRoot(store_dir, repositories=(REPO,))
        self.inputs["storage_root"] = "<configured external root>"
        return "external storage root resolved; scratch store created"

    # -- records ------------------------------------------------------------------------------

    def _check_bindings(
        self, report: RepetitionReport, validation: NumericalValidation, accounting: PilotAccounting
    ) -> None:
        for name, digest in report.sources.items():
            actual = sha256_file(self.docs / name)
            if actual != digest:
                msg = f"report source {name} has digest {actual[:12]}, the report recorded {digest[:12]}"
                raise ReproductionError(msg)
        diagnosis = load_diagnosis(self.docs / "numerical_validation_v1_diagnosis.json")
        timing = load_timing(self.docs / "timing_smoke_check_v1.json")
        manifest_sha = sha256_file(self.docs / "panel_manifest_v1.json")
        validation_sha = sha256_file(self.docs / "numerical_validation_v1.json")
        if validation.panel_manifest_sha256 != manifest_sha or accounting.panel_manifest_sha256 != manifest_sha:
            msg = "the validation or the accounting does not bind the committed panel manifest"
            raise ReproductionError(msg)
        if accounting.numerical_validation_sha256 != validation_sha or diagnosis.validation_sha256 != validation_sha:
            msg = "the accounting or the diagnosis does not bind the committed numerical validation"
            raise ReproductionError(msg)
        if timing.panel_manifest_sha256 != manifest_sha:
            msg = "the timing smoke check does not bind the committed panel manifest"
            raise ReproductionError(msg)
        self.inputs.update({"panel_manifest_sha256": manifest_sha, "numerical_validation_sha256": validation_sha})

    def _check_c11_and_incident(self, report: RepetitionReport, validation: NumericalValidation) -> str:
        binding = numerical_binding(self.docs / "numerical_validation_v1.json", root=REPO)
        exception = [e for e in report.equivalence if e.accepted_exception]
        if binding.all_passed or len(exception) != 1 or binding.exception_candidate_identity is None:
            msg = "the C11 exception must be exactly one failed comparison, retained as failed"
            raise ReproductionError(msg)
        failed = [c for c in validation.comparisons if not c.passed]
        if len(failed) != 1 or failed[0].candidate_identity != binding.exception_candidate_identity:
            msg = "the validation's retained failure is not the C11-bound comparison"
            raise ReproductionError(msg)
        incident = self.docs / RETENTION_INCIDENT
        if not incident.is_file() or "Retention incident" not in incident.read_text(encoding="utf-8"):
            msg = f"the retention incident record {RETENTION_INCIDENT} is missing or unnamed"
            raise ReproductionError(msg)
        self.inputs.update(
            {
                "c11_exception_candidate": binding.exception_candidate_identity,
                "c11_exception_reference": str(binding.exception_reference_identity),
                "retention_incident_sha256": sha256_file(incident),
            }
        )
        return binding.exception_candidate_identity

    def _rebuild_panel(self, manifest: PanelManifest, store: StorageRoot) -> None:
        pointer = load_report_pointer(RECOVERY_DOCS / manifest.source.pointer_file)
        study = open_stored_report(store, pointer)
        protocol = load_recovery_search(REPO / study.protocol_file)
        ablation = load_ablation(RECOVERY_DOCS / manifest.ablation.file)
        dataset_file = REPO / manifest.configs.dataset_record_file
        dataset = load_processed_record(dataset_file)
        if not isinstance(dataset, RecoveryDatasetRecord):
            msg = f"{manifest.configs.dataset_record_file} is not a recovery dataset record"
            raise ReproductionError(msg)
        entries = resolve_panel(study, protocol, ablation)
        rebuilt = build_panel_manifest(
            entries,
            report=study,
            pointer=pointer,
            pointer_file=RECOVERY_DOCS / manifest.source.pointer_file,
            protocol=protocol,
            protocol_file=REPO / study.protocol_file,
            ablation=ablation,
            ablation_file=RECOVERY_DOCS / manifest.ablation.file,
            dataset=dataset,
            dataset_file=dataset_file,
            provenance=manifest.provenance,
        )
        if rebuilt != manifest:
            msg = "the panel manifest re-resolved from the study payload differs from the committed one"
            raise ReproductionError(msg)
        self.inputs["source_study_payload_sha256"] = manifest.source.payload.sha256

    def _check_pointers(self, accounting: PilotAccounting) -> int:
        pointers = 0
        for line in accounting.models:
            file = self.docs / "evidence" / pointer_name("model", f"{line.panel_label}/{line.arm}")
            pointer = load_pointer(file)
            if pointer.identity != line.evaluation_identity or pointer.payload != line.payload:
                msg = f"the pointer of {line.panel_label}/{line.arm} disagrees with the accounting"
                raise ReproductionError(msg)
            pointers += 1
        for bank in accounting.banks:
            pointer = load_pointer(self.docs / "evidence" / pointer_name("replay", f"warmup-{bank.warmup_s:g}s"))
            if pointer.identity != bank.identity or pointer.payload != bank.payload:
                msg = f"the replay bank pointer at warm-up {bank.warmup_s:g} s disagrees with the accounting"
                raise ReproductionError(msg)
            pointers += 1
        return pointers

    def records(self) -> str:
        """Every committed record loads strictly, binds its neighbours by digest, and the panel re-resolves."""
        report = _need(self.report, "records")
        store = _need(self.store, "records")
        manifest = load_panel(self.docs / "panel_manifest_v1.json")
        validation = load_validation(self.docs / "numerical_validation_v1.json")
        accounting = load_accounting(self.docs / "pilot_execution_v1.json")
        self._check_bindings(report, validation, accounting)
        candidate = self._check_c11_and_incident(report, validation)
        self._rebuild_panel(manifest, store)
        pointers = self._check_pointers(accounting)
        return (
            f"{len(report.sources)} report sources, the panel manifest (re-resolved identically from the study "
            f"payload), validation, diagnosis, timing, accounting, {pointers} evidence pointers, the C11 binding "
            f"({candidate[:12]}), and the retention incident record verified"
        )

    # -- payloads -----------------------------------------------------------------------------

    def payloads(self) -> str:
        """Every payload resolves with its recorded digest: dataset, manifests, runs, and the fit cache."""
        store = _need(self.store, "payloads")
        execution = _need(self.current_execution, "payloads")
        manifest_file = self.docs / "panel_manifest_v1.json"
        context = PanelContext.load(manifest_file, store=store, root=self.root, execution=execution)
        self.context = context
        inputs = build_report_inputs(self.docs, store=store, root=self.root)
        self.report_inputs = inputs
        runs, arrays_bytes = verify_run_payloads(store, inputs)
        validation = load_validation(self.docs / "numerical_validation_v1.json")
        prescribed = prescribed_fits(validation, inputs)
        self.prescribed = prescribed
        cached = verify_fit_cache(FitStore(store), prescribed, inputs)
        self.inputs["dataset"] = context.manifest.configs.dataset
        augmented = sum(1 for f in prescribed.values() if f.source == "evidence")
        return (
            f"dataset {context.manifest.configs.dataset}, {len(inputs.models)} model manifests, {len(inputs.banks)} "
            f"replay banks, {runs} run payloads ({arrays_bytes / 2**20:.1f} MiB of arrays), and {cached} cached fits "
            f"({len(validation.fits)} of the validation, {augmented} augmented fits bound by the evidence) verified "
            "by digest"
        )

    # -- recipes and fits -------------------------------------------------------------------

    def recipes(self) -> str:
        """Every prescribed fit identity and ridge rule re-derives from the manifest; the cache holds them."""
        context = _need(self.context, "recipes")
        store = _need(self.store, "recipes")
        prescribed = _need(self.prescribed, "recipes")
        fits = FitStore(store)
        checked = 0
        for entry in context.manifest.entries:
            for arm in panel_arms():
                label = f"{entry.label}/{arm.label}"
                identity = context.inputs.identity(entry, arm)
                fit = prescribed.get(identity)
                if fit is None or fit.panel_label != entry.label or fit.arm != arm:
                    msg = f"{label}: the re-derived fit identity {identity[:12]} is not recorded by the evidence"
                    raise ReproductionError(msg)
                expected_alpha = solver_alpha(entry.base_alpha, arm.regularization_rule, arm.count)
                recipe = fits.read_recipe(fits.read_record(identity))
                config = entry.point.esn.model_config(context.inputs.base, name=label)
                expected_readout = readout_for_arm(config.esn.readout, arm, base_alpha=entry.base_alpha)
                if recipe.training != training_spec_for_arm(arm, warmup_s=entry.warmup_s, base_alpha=entry.base_alpha):
                    msg = f"{label}: the cached recipe's training construction is not the arm's"
                    raise ReproductionError(msg)
                if fit.solver_alpha != expected_alpha or recipe.esn.readout != expected_readout:
                    msg = f"{label}: the cached readout settings are not the arm's prescription ({expected_alpha!r})"
                    raise ReproductionError(msg)
                if recipe.esn.reservoir != config.esn.reservoir:
                    msg = f"{label}: the cached reservoir configuration is not the entry's"
                    raise ReproductionError(msg)
                checked += 1
        if checked != len(prescribed):
            msg = f"{checked} prescribed fits re-derived but the evidence records {len(prescribed)}"
            raise ReproductionError(msg)
        validated = sum(1 for f in prescribed.values() if f.source == "validation")
        return (
            f"{checked} prescribed fit identities, solver rules, training constructions, and reservoirs re-derived "
            f"({validated} recorded by the validation, {checked - validated} augmented fits by the evidence)"
        )

    def fits(self) -> str:
        """Every cached fit refits bitwise in a fresh process of the canonical environment."""
        if self.skip_fits:
            return "skipped on request (--skip-fits)"
        execution = _need(self.current_execution, "fits")
        prescribed = _need(self.prescribed, "fits")
        out_dir = self.scratch / "refits"
        out_dir.mkdir()
        worst = 0.0
        for fit in prescribed.values():
            output = out_dir / f"{fit.identity}.json"
            result = self.refit(fit.identity, root=self.root, parent_identity=execution.identity, output=output)
            worst = max(worst, result.max_abs_weight_diff)
            if not result.passed:
                msg = (
                    f"fit {fit.identity[:12]} ({fit.panel_label}/{fit.arm.label}) did not refit bitwise: "
                    f"weights {result.weights_bitwise_equal}, states {result.states_bitwise_equal}, report "
                    f"{result.fit_report_equal}, environment {result.environment_match}"
                )
                raise ReproductionError(msg)
        self.max_deviation = worst if self.max_deviation is None else max(self.max_deviation, worst)
        return (
            f"{len(prescribed)} fits refitted bitwise in fresh pinned processes (largest weight deviation {worst:.3e})"
        )

    # -- metrics -----------------------------------------------------------------------------

    def _metrics_context(self) -> _Context:
        if self.metrics_context is not None:
            return self.metrics_context
        context = _need(self.context, "metrics")
        inputs = _need(self.report_inputs, "metrics")
        conditions = next(iter(inputs.banks.values())).conditions
        evaluation_file = self.root / conditions.evaluation_file
        evaluation = load_evaluation_config(evaluation_file)
        development_sha256 = sha256_file(evaluation.development)
        if (development_sha256, sha256_file(evaluation_file)) != (
            conditions.development_sha256,
            conditions.evaluation_sha256,
        ):
            msg = "the evaluation or development configuration no longer hashes to what the replay banks bound"
            raise ReproductionError(msg)
        levels = load_development_robustness(evaluation.development)
        scenario = context.inputs.scenario
        lower = tuple(link.q_min for link in scenario.robot.links)
        upper = tuple(link.q_max for link in scenario.robot.links)
        scenarios = self.scenarios
        if scenarios is None:
            scenarios = robustness_scenarios(levels, nominal=context.dataset.q0_ref, lower=lower, upper=upper)
        if tuple(s.scenario_id for s in scenarios) != conditions.scenario_ids:
            msg = "the development scenarios no longer resolve to the ones the replay banks bound"
            raise ReproductionError(msg)
        trackers = self.trackers or {name: load_frozen_baseline(name) for name in RECOVERY_TRACKERS}
        digests = self.tracker_digests or {name: frozen_baseline_digest(name) for name in RECOVERY_TRACKERS}
        if digests != conditions.trackers:
            msg = "the frozen trackers differ from the ones the replay banks bound"
            raise ReproductionError(msg)
        task = task_intervals_from_phases(context.inputs.samples.t, context.inputs.samples.phase)
        settling = next(iter(inputs.models.values())).conditions.settling_band_rad
        self.metrics_context = _Context(
            scenarios=scenarios,
            dwell_start_s=task.dwell[0],
            settling_band_rad=settling,
            saturation_bound=conditions.saturation_bound,
            velocity_abort=conditions.velocity_abort,
            trackers=trackers,
            tracker_digests=digests,
            evaluation_file=evaluation_file,
            development_sha256=development_sha256,
        )
        return self.metrics_context

    def _load(self, store: StorageRoot, pair: PairRecord) -> LoadedRun:
        run = cast("Any", pair.run)
        return load_run(store, pointer_from_summary(store, run.artifact_id))

    def _recompute_diagnostics(self, loaded: LoadedRun, pair: PairRecord, activation: float) -> VelocityDiagnostics:
        ctx = self._metrics_context()
        scenario = _need(self.context, "metrics").inputs.scenario
        return velocity_diagnostics(
            checked_states_from_run(loaded, pair),
            loaded.summary.termination,
            historical=scenario.limits.velocity,
            abort_limit=ctx.velocity_abort,
            activation_s=activation,
            dwell_start_s=ctx.dwell_start_s,
            dt=scenario.timing.dt,
        )

    def _recompute_replay(
        self, store: StorageRoot, pair: PairRecord, activation: float
    ) -> tuple[ReplayComponent, VelocityDiagnostics]:
        ctx = self._metrics_context()
        context = _need(self.context, "metrics")
        index, case = ctx.case(pair.scenario_id)
        loaded = self._load(store, pair)
        component = replay_component(
            index,
            case,
            pair.tracker,
            case.initial_q(context.dataset.q0_ref),
            context.inputs.scenario,
            context.inputs.samples,
            loaded.arrays,
            loaded.summary.termination,
            activation_s=activation,
            bound=ctx.saturation_bound,
        )
        return component, self._recompute_diagnostics(loaded, pair, activation)

    def _recompute_rc(
        self, store: StorageRoot, pair: PairRecord, activation: float, replay: ReplayComponent
    ) -> tuple[RecoveryComponent, VelocityDiagnostics]:
        ctx = self._metrics_context()
        context = _need(self.context, "metrics")
        index, case = ctx.case(pair.scenario_id)
        loaded = self._load(store, pair)
        component = recovery_component(
            index,
            case,
            pair.tracker,
            case.initial_q(context.dataset.q0_ref),
            (loaded.arrays, loaded.summary.termination),
            scenario=context.inputs.scenario,
            reference=context.inputs.samples,
            activation_s=activation,
            bound=ctx.saturation_bound,
            replay=replay,
            boundary_jump=boundary_jump_from_arrays(loaded),
            settling_band_rad=ctx.settling_band_rad,
        )
        return component, self._recompute_diagnostics(loaded, pair, activation)

    def _metrics_of_banks(
        self, store: StorageRoot, inputs: ReportInputs
    ) -> tuple[dict[str, dict[tuple[str, str], PairRecord]], int]:
        by_bank: dict[str, dict[tuple[str, str], PairRecord]] = {}
        runs = 0
        for warmup, bank in inputs.banks.items():
            if bank.conditions.warmup_s != warmup:
                msg = f"the replay bank filed under warm-up {warmup:g} s records warm-up {bank.conditions.warmup_s:g} s"
                raise ReproductionError(msg)
            table: dict[tuple[str, str], PairRecord] = {}
            for pair in bank.pairs:
                component, diagnostics = self._recompute_replay(store, pair, bank.conditions.warmup_s)
                label = f"replay {warmup:g}s {pair.scenario_id}/{pair.tracker}"
                _exact(f"{label} component", to_mapping(cast("Any", pair.replay)), to_mapping(component))
                _exact(f"{label} diagnostics", to_mapping(cast("Any", pair.velocity)), to_mapping(diagnostics))
                if pair.status != ("completed" if component.feasible else "infeasible"):
                    msg = f"{label}: status {pair.status!r} contradicts the recomputed component"
                    raise ReproductionError(msg)
                table[(pair.scenario_id, pair.tracker)] = pair
                runs += 1
            by_bank[bank.identity] = table
        return by_bank, runs

    def _metrics_of_model(
        self, store: StorageRoot, label: str, evidence: ModelEvidence, replays: dict[tuple[str, str], PairRecord]
    ) -> int:
        context = _need(self.context, "metrics")
        activation = evidence.conditions.warmup_s
        ratios: dict[str, list[float]] = {}
        pairs = 0
        for pair in evidence.pairs:
            name = f"{label} {pair.scenario_id}/{pair.tracker}"
            if pair.status == "unexecuted":
                if pair.run is not None or pair.rc is not None:
                    msg = f"{name}: an unexecuted pair carries data"
                    raise ReproductionError(msg)
                continue
            replay_record = cast("ReplayComponent", replays[(pair.scenario_id, pair.tracker)].replay)
            if pair.status == "replay_blocked":
                index, case = self._metrics_context().case(pair.scenario_id)
                start = case.initial_q(context.dataset.q0_ref)
                rebuilt = blocked_component(index, case, pair.tracker, start, replay_record)
                _exact(f"{name} blocked component", to_mapping(cast("Any", pair.rc)), to_mapping(rebuilt))
                pairs += 1
                continue
            component, diagnostics = self._recompute_rc(store, pair, activation, replay_record)
            _exact(f"{name} component", to_mapping(cast("Any", pair.rc)), to_mapping(component))
            _exact(f"{name} diagnostics", to_mapping(cast("Any", pair.velocity)), to_mapping(diagnostics))
            if pair.status != ("completed" if component.feasible else "infeasible"):
                msg = f"{name}: status {pair.status!r} contradicts the recomputed component"
                raise ReproductionError(msg)
            if pair.crossed_historical != diagnostics.crossed_historical:
                msg = f"{name}: the recorded historical crossing contradicts the recomputed diagnostics"
                raise ReproductionError(msg)
            if pair.status == "completed" and component.gap_ratio is not None:
                ratios.setdefault(f"{pair.kind}:{pair.tracker}", []).append(component.gap_ratio)
            pairs += 1
        cells = {k: float(median(v)) for k, v in sorted(ratios.items())} if evidence.status == "feasible" else {}
        _exact(f"{label} cells", dict(evidence.cells), cells)
        return pairs

    def metrics(self) -> str:
        """Every recorded component, diagnostic, status, and cell recomputes exactly from the stored runs."""
        store = _need(self.store, "metrics")
        inputs = _need(self.report_inputs, "metrics")
        by_bank, replay_runs = self._metrics_of_banks(store, inputs)
        pairs = 0
        for label, evidence in inputs.models.items():
            replays = by_bank.get(evidence.replay_bank)
            if replays is None:
                msg = f"{label}: its replay bank {evidence.replay_bank[:12]} is not among the committed banks"
                raise ReproductionError(msg)
            pairs += self._metrics_of_model(store, label, evidence, replays)
        self.max_deviation = 0.0 if self.max_deviation is None else self.max_deviation
        return (
            f"{replay_runs} replay runs and the runs of {pairs} executed model pairs reloaded with their array "
            "digests; every component, diagnostic, status, crossing, and cell recomputed exactly"
        )

    # -- tables ------------------------------------------------------------------------------

    def tables(self) -> str:
        """The report tables, the accounting, and every Markdown rendering re-derive identically."""
        report = _need(self.report, "tables")
        inputs = _need(self.report_inputs, "tables")
        store = _need(self.store, "tables")
        reps = representatives(inputs)
        self.reps = tuple(reps)
        rebuilt = build_report(
            inputs, reps=reps, plots=report.plots, animations=report.animations, provenance=report.provenance
        )
        committed = to_mapping(report)
        rebuilt_map = to_mapping(rebuilt)
        committed.pop("representatives")
        rebuilt_map.pop("representatives")
        _exact("report tables", committed, rebuilt_map)
        _exact("representatives", _strip_assets(report.representatives), _strip_assets(reps))
        if render_report_markdown(report) != (self.docs / "repetition_report_v1.md").read_text(encoding="utf-8"):
            msg = "the rendered report differs from the committed repetition_report_v1.md"
            raise ReproductionError(msg)
        accounting = load_accounting(self.docs / "pilot_execution_v1.json")
        rebuilt_accounting = account_pilot(
            store=store,
            evidence_dir=self.docs / "evidence",
            manifest_file=self.docs / "panel_manifest_v1.json",
            validation_file=self.docs / "numerical_validation_v1.json",
            provenance=accounting.provenance,
        )
        _exact("accounting", to_mapping(accounting), to_mapping(rebuilt_accounting))
        renderings = (
            ("pilot_execution_v1.md", render_accounting_markdown(accounting)),
            ("panel_manifest_v1.md", render_panel_markdown(load_panel(self.docs / "panel_manifest_v1.json"))),
            (
                "numerical_validation_v1.md",
                render_validation_markdown(load_validation(self.docs / "numerical_validation_v1.json")),
            ),
            (
                "numerical_validation_v1_diagnosis.md",
                render_diagnosis_markdown(load_diagnosis(self.docs / "numerical_validation_v1_diagnosis.json")),
            ),
            ("timing_smoke_check_v1.md", render_timing_markdown(load_timing(self.docs / "timing_smoke_check_v1.json"))),
        )
        for name, text in renderings:
            if text != (self.docs / name).read_text(encoding="utf-8"):
                msg = f"the rendered {name} differs from the committed file"
                raise ReproductionError(msg)
        return (
            f"{len(report.outcomes)} outcome, {len(report.paired)} paired, {len(report.equivalence)} equivalence, "
            f"{len(report.speeds)} speed, {len(report.costs)} cost, and {len(report.eligibility)} eligibility rows, "
            f"{len(reps)} representatives, the accounting, and {len(renderings) + 1} Markdown renderings re-derived "
            "identically"
        )

    # -- re-simulation -----------------------------------------------------------------------

    def _scratch_runner(self) -> PilotRunner:
        context = _need(self.context, "resimulation")
        scratch_store = _need(self.scratch_store, "resimulation")
        execution = _need(self.current_execution, "resimulation")
        ctx = self._metrics_context()
        evaluation = load_evaluation_config(ctx.evaluation_file)
        numerical = numerical_binding(self.docs / "numerical_validation_v1.json", root=self.root)
        resolved = {
            "manifest": context.manifest_sha256,
            "evaluation": {ctx.evaluation_file.name: sha256_file(ctx.evaluation_file)},
            "development_sha256": ctx.development_sha256,
            "execution_identity": execution.identity,
            "command": _REPRODUCTION_COMMAND,
        }
        provenance = collect_provenance(
            resolved,
            seeds={},
            artifacts=[context.payload],
            exploratory=self.exploratory,
            now=self.now or datetime.now(tz=UTC),
        )
        require_clean_for_confirmatory(provenance)
        return PilotRunner(
            store=scratch_store,
            inputs=context.inputs,
            dataset=context.dataset,
            evaluation=evaluation,
            evaluation_file=ctx.evaluation_file,
            root=self.root,
            execution=execution,
            provenance=provenance,
            numerical=numerical,
            scenarios=ctx.scenarios,
            trackers=ctx.trackers,
            tracker_digests=ctx.tracker_digests,
            development_sha256=ctx.development_sha256,
            settling_band_rad=ctx.settling_band_rad,
            saturation_bound=ctx.saturation_bound,
            command=_REPRODUCTION_COMMAND,
            simulate_fn=self.simulate_fn,
            log=lambda _message: None,
        )

    def _check_conditions(self, runner: PilotRunner, inputs: ReportInputs, entries: dict[str, PanelEntry]) -> None:
        for warmup, bank in inputs.banks.items():
            entry = next(e for e in entries.values() if e.warmup_s == warmup)
            if runner.replay_conditions(entry) != bank.conditions:
                msg = f"the rebuilt replay conditions at warm-up {warmup:g} s are not the committed bank's"
                raise ReproductionError(msg)
        for label, evidence in inputs.models.items():
            if runner.conditions(entries[label.split("/", 1)[0]]) != evidence.conditions:
                msg = f"{label}: the rebuilt evaluation conditions are not the committed evidence's"
                raise ReproductionError(msg)

    def resimulation(self) -> str:
        """The declared subset re-simulates bitwise into the scratch store with refitted models."""
        if self.skip_resimulation:
            return "skipped on request (--skip-resimulation)"
        if self.exploratory:
            msg = "the re-simulation requires a clean checkout; run without --exploratory from a clean tree"
            raise ReproductionError(msg)
        return self.resimulate_with(self._scratch_runner())

    def resimulate_with(self, runner: PilotRunner) -> str:
        """Re-simulate the declared subset with ``runner`` (which writes to its own store) and compare every run."""
        store = _need(self.store, "resimulation")
        inputs = _need(self.report_inputs, "resimulation")
        context = _need(self.context, "resimulation")
        entries = self.entries or {e.label: e for e in context.manifest.entries}
        arms = {a.label: a for a in panel_arms()}
        self._check_conditions(runner, inputs, entries)
        selections = resimulation_subset(inputs)
        self.coverage = subset_coverage(selections, inputs)
        for selection in selections:
            entry = entries[selection.entry]
            bank = inputs.banks[entry.warmup_s]
            stored_replay = next(
                p for p in bank.pairs if p.scenario_id == selection.scenario_id and p.tracker == selection.tracker
            )
            name = f"{selection.entry}/{selection.arm or 'replay'} {selection.scenario_id}/{selection.tracker}"
            if selection.kind == "replay":
                rerun = runner.rerun_replay_pair(entry, selection.scenario_id, selection.tracker)
                self._compare_rerun(store, runner.store, stored_replay, rerun, name)
                continue
            evidence = inputs.models[f"{selection.entry}/{selection.arm}"]
            stored = next(
                p for p in evidence.pairs if p.scenario_id == selection.scenario_id and p.tracker == selection.tracker
            )
            arm = arms[cast("str", selection.arm)]
            rerun = runner.rerun_rc_pair(entry, arm, selection.scenario_id, selection.tracker, replay=stored_replay)
            self._compare_rerun(store, runner.store, stored, rerun, name)
        coverage = self.coverage
        return (
            f"{coverage.runs} runs re-simulated bitwise into the scratch store ({coverage.rc_runs} RC, "
            f"{coverage.replay_runs} replay; {len(coverage.arms)} arms, warm-ups {list(coverage.warmups_s)}, classes "
            f"{list(coverage.classes)}, {coverage.first_pair_failures} first-pair and {coverage.later_failures} later "
            f"failure examples); unavailable: {'; '.join(coverage.unavailable) or 'none'}"
        )

    def _compare_rerun(
        self, store: StorageRoot, scratch_store: StorageRoot, stored: PairRecord, rerun: PairRecord, name: str
    ) -> None:
        if stored.run is None or rerun.run is None:
            msg = f"{name}: both the stored and the re-simulated pair must carry a run"
            raise ReproductionError(msg)
        if rerun.run.arrays_sha256 != stored.run.arrays_sha256:
            loaded = self._load(store, stored)
            rebuilt = load_run(scratch_store, pointer_from_summary(scratch_store, rerun.run.artifact_id))
            specs = rebuilt.arrays.specs()
            differing = sorted(key for key, spec in loaded.arrays.specs().items() if specs.get(key) != spec)
            msg = f"{name}: re-simulated arrays differ from the stored run (arrays {differing})"
            raise ReproductionError(msg)
        if rerun.status != stored.status:
            msg = f"{name}: re-simulated status {rerun.status!r} != stored {stored.status!r}"
            raise ReproductionError(msg)
        component_stored = stored.rc if stored.arm == "rc" else stored.replay
        component_rerun = rerun.rc if rerun.arm == "rc" else rerun.replay
        _exact(f"{name} component", to_mapping(cast("Any", component_stored)), to_mapping(cast("Any", component_rerun)))
        _exact(f"{name} diagnostics", to_mapping(cast("Any", stored.velocity)), to_mapping(cast("Any", rerun.velocity)))

    # -- assets ------------------------------------------------------------------------------

    def assets(self) -> str:
        """Every committed figure and animation regenerates byte for byte and the full report re-assembles."""
        report = _need(self.report, "assets")
        inputs = _need(self.report_inputs, "assets")
        reps = list(self.reps) if self.reps is not None else list(representatives(inputs))
        plots_dir = self.scratch / "plots"
        names, reps = write_plots(inputs, reps, plots_dir)
        if tuple(names) != report.plots:
            msg = f"regenerated figures {names} differ from the committed set {list(report.plots)}"
            raise ReproductionError(msg)
        for name in names:
            if (plots_dir / name).read_bytes() != (self.docs / PLOT_DIR / name).read_bytes():
                msg = f"figure {name} does not regenerate byte-for-byte"
                raise ReproductionError(msg)
        animations_dir = self.scratch / "animations"
        reps = write_animations(inputs, reps, animations_dir)
        animations = sorted({a for r in reps for a in (r.rc_animation, r.replay_animation) if a})
        if tuple(animations) != report.animations:
            msg = f"regenerated animations {animations} differ from the committed set {list(report.animations)}"
            raise ReproductionError(msg)
        for name in animations:
            if (animations_dir / name).read_bytes() != (self.docs / ANIMATION_DIR / name).read_bytes():
                msg = f"animation {name} does not regenerate byte-for-byte"
                raise ReproductionError(msg)
        rebuilt = build_report(inputs, reps=reps, plots=names, animations=animations, provenance=report.provenance)
        _exact("report", to_mapping(report), to_mapping(rebuilt))
        return (
            f"{len(names)} figures and {len(animations)} animations regenerated byte-for-byte; the report re-assembles"
        )

    def step(self, name: str) -> Check:
        """Run one step and record its outcome."""
        action = cast("Callable[[], str]", getattr(self, name))
        t0 = time.perf_counter()
        try:
            detail = action()
        except (
            ReproductionError,
            StorageError,
            FileNotFoundError,
            ValueError,
            RuntimeError,
            TypeError,
            KeyError,
        ) as exc:
            return Check(name, ok=False, detail=f"{type(exc).__name__}: {exc}", elapsed_s=time.perf_counter() - t0)
        return Check(name, ok=True, detail=detail, elapsed_s=time.perf_counter() - t0)


# --- the separate DOC-005 rerun and the gates ---------------------------------------------


def _affinity() -> tuple[int, ...]:
    getter = getattr(os, "sched_getaffinity", None)
    return tuple(sorted(cast("set[int]", getter(0)))) if getter is not None else ()


def run_doc005(
    scratch: Path,
    *,
    label: str,
    cpus: Sequence[int] | None = None,
    runner: Callable[..., Any] | None = None,
) -> Doc005Rerun:
    """Rerun the historical task 1-a reproduction (``--from-evidence``) and report its status apart.

    ``cpus`` pins the rerun to an explicit CPU list through ``taskset``; the
    default inherits this process's affinity. The tolerance is the historical
    exact zero and is reported, never changed.
    """
    scratch.mkdir(parents=True, exist_ok=False)
    summary = scratch / "reproduce_1a.json"
    arguments = ["--from-evidence", "--scratch", str(scratch / "work"), "--summary", str(summary)]
    command = [sys.executable, "scripts/reproduce_1a.py", *arguments]
    recorded = command_line("arm_rc_ctrl.experiments.reproduce_1a", arguments)
    if cpus is not None:
        pinned = ",".join(str(c) for c in cpus)
        command = ["taskset", "-c", pinned, *command]
        recorded = f"taskset -c {pinned} {recorded}"
    started = time.perf_counter()
    run = subprocess.run if runner is None else runner
    completed = run(command, check=False, cwd=REPO, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    ok: bool | None = None
    deviation: float | None = None
    detail = f"exit {completed.returncode}"
    if summary.is_file():
        data = cast("dict[str, object]", json.loads(summary.read_text(encoding="utf-8")))
        ok = bool(data.get("ok"))
        raw = data.get("max_deviation")
        deviation = float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else None
        checks = cast("list[dict[str, object]]", data.get("checks", []))
        failed = [str(c.get("name")) for c in checks if not c.get("ok")]
        detail = f"ok {ok}; largest deviation {deviation!r}; failed steps {failed}"
    else:
        detail += f"; no summary written; stderr {str(completed.stderr).strip()[-300:]!r}"
    return Doc005Rerun(
        label=label,
        cpus=tuple(cpus) if cpus is not None else _affinity(),
        command=recorded,
        returncode=int(completed.returncode),
        ok=ok,
        max_deviation=deviation,
        tolerance=DOC005_TOLERANCE,
        elapsed_s=elapsed,
        detail=detail,
    )


_GATE_COMMANDS: Final = (
    ("uv", "run", "--locked", "nox"),
    ("uv", "run", "--locked", "nox", "-s", "pre_commit"),
)


def run_gates(
    *, runner: Callable[..., Any] | None = None, commands: Sequence[Sequence[str]] = _GATE_COMMANDS
) -> tuple[GateResult, ...]:
    """Run the repository quality gates and record their exit statuses."""
    run = subprocess.run if runner is None else runner
    results: list[GateResult] = []
    for command in commands:
        started = time.perf_counter()
        completed = run(list(command), check=False, cwd=REPO)
        results.append(GateResult(" ".join(command), int(completed.returncode), time.perf_counter() - started))
    return tuple(results)


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
        "affinity": ",".join(str(c) for c in _affinity()),
    }


def reproduce(
    *,
    scratch: Path,
    keep_going: bool = False,
    exploratory: bool = False,
    skip_resimulation: bool = False,
    skip_fits: bool = False,
    store: StorageRoot | None = None,
    docs: Path = DOCS,
    now: datetime | None = None,
    doc005: Sequence[tuple[str, Sequence[int] | None]] = (),
    gates: bool = False,
    simulate_fn: SimulateFn | None = None,
) -> RepetitionReproduction:
    """Run every reproduction step.

    ``scratch`` receives the rebuilt store, the refits, and the regenerated assets (outside the repository).
    """
    started = time.perf_counter()
    started_at = (now or datetime.now(tz=UTC)).isoformat()
    reproducer = Reproducer(
        prepare_scratch(scratch),
        exploratory,
        store,
        docs,
        now,
        skip_resimulation=skip_resimulation,
        skip_fits=skip_fits,
        simulate_fn=simulate_fn,
    )
    checks: list[Check] = []
    for name in STEPS:
        check = reproducer.step(name)
        checks.append(check)
        if not check.ok and not keep_going:
            break
    reruns = tuple(run_doc005(reproducer.scratch / f"doc005-{label}", label=label, cpus=cpus) for label, cpus in doc005)
    gate_results = run_gates() if gates else ()
    return RepetitionReproduction(
        started_at=started_at,
        checks=tuple(checks),
        inputs=dict(reproducer.inputs),
        environment=_environment(),
        max_deviation=reproducer.max_deviation,
        elapsed_s=time.perf_counter() - started,
        comparisons=DECLARED_COMPARISONS,
        resimulation_rule=RESIMULATION_RULE,
        coverage=reproducer.coverage,
        doc005=reruns,
        gates=gate_results,
    )


def _coverage_lines(coverage: SubsetCoverage | None) -> list[str]:
    if coverage is None:
        return ["- The re-simulation step did not run."]
    return [
        (
            f"- Selected runs: {coverage.runs} ({coverage.rc_runs} RC, {coverage.replay_runs} replay); formulations "
            f"{list(coverage.formulations)}; {len(coverage.arms)} arms; warm-ups {list(coverage.warmups_s)} s; "
            f"trackers "
            f"{list(coverage.trackers)}; classes {list(coverage.classes)}; failure examples: "
            f"{coverage.first_pair_failures} at the first pair, {coverage.later_failures} later."
        ),
        f"- Unavailable coverage: {'; '.join(coverage.unavailable) or 'none'}.",
    ]


def _doc005_lines(reruns: Sequence[Doc005Rerun]) -> list[str]:
    if not reruns:
        return ["- Not rerun in this invocation (pass --doc005)."]
    lines = [
        "| rerun | CPUs | exit | ok | largest deviation | tolerance | elapsed (s) | detail |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        (
            f"| {r.label} | {','.join(str(c) for c in r.cpus)} | {r.returncode} | {r.ok} | {r.max_deviation!r} "
            f"| {r.tolerance!r} | {r.elapsed_s:.1f} | {r.detail} |"
        )
        for r in reruns
    )
    return lines


def audit_markdown(
    result: RepetitionReproduction, *, command: str, auditor: str = "(to be filled by the auditor)"
) -> str:
    """A compact audit note of one reproduction invocation."""
    machine = f"{result.environment.get('platform', '')} / {result.environment.get('machine', '')}"
    lines = [
        "# Task 1-a repeated-demonstration pilot reproduction",
        "",
        f"- Started: {result.started_at}; elapsed {result.elapsed_s:.1f} s; ok: {result.ok}.",
        f"- Command: `{command}`",
        f"- Executor machine: `{machine}`; CPU affinity `{result.environment.get('affinity', '')}`.",
        f"- Auditor: {auditor}.",
        f"- Largest deviation over the exact and bitwise comparisons: {result.max_deviation!r}.",
        "",
        "| step | ok | elapsed (s) | detail |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(f"| {c.name} | {c.ok} | {c.elapsed_s:.1f} | {c.detail} |" for c in result.checks)
    lines += ["", "## Declared comparisons", ""]
    lines += ["| comparison | kind | units | tolerance | reason |", "| --- | --- | --- | --- | --- |"]
    lines.extend(
        f"| {c.name} | {c.kind} | {c.units} | {'n/a' if c.tolerance is None else c.tolerance!r} | {c.reason} |"
        for c in result.comparisons
    )
    lines += ["", "## Re-simulation subset", "", result.resimulation_rule, "", *_coverage_lines(result.coverage)]
    lines += ["", "## Historical task 1-a reproduction (DOC-005), reported apart", "", *_doc005_lines(result.doc005)]
    lines += ["", "## Repository gates", ""]
    if result.gates:
        lines += ["| command | exit | elapsed (s) |", "| --- | --- | --- |"]
        lines.extend(f"| `{g.command}` | {g.returncode} | {g.elapsed_s:.1f} |" for g in result.gates)
    else:
        lines.append("- Not run in this invocation (pass --gates).")
    lines += ["", "## Inputs", ""]
    lines.extend(f"- {key}: `{value}`" for key, value in sorted(result.inputs.items()))
    lines += ["", "## Environment", ""]
    lines.extend(f"- {key}: `{value}`" for key, value in sorted(result.environment.items()))
    return "\n".join(lines) + "\n"


def _git(*args: str) -> str:
    completed = subprocess.run(["git", *args], check=False, capture_output=True, text=True, cwd=REPO)
    if completed.returncode != 0:
        msg = f"git {' '.join(args[:3])} failed: {completed.stderr.strip()[:300]}"
        raise ReproductionError(msg)
    return completed.stdout


def run_from_checkout(scratch: Path, commit: str, forwarded: Sequence[str]) -> int:
    """Reproduce inside a fresh detached worktree at ``commit``: sync, stamp the build manifest, run pinned, keep it."""
    resolved = _git("rev-parse", "--verify", f"{commit}^{{commit}}").strip()
    checkout = prepare_scratch(scratch) / "checkout"
    _git("worktree", "add", "--detach", str(checkout), resolved)
    _git("-C", str(checkout), "submodule", "update", "--init", "third_party/skelarm", "third_party/rtctrl")
    _git("-C", str(checkout), "submodule", "update", "--init", "--recursive", "third_party/rclib")
    for stage in (
        ["uv", "sync", "--locked"],
        ["uv", "run", "--locked", "python", "-m", "arm_rc_ctrl.dependencies", "rebuild"],
    ):
        completed = subprocess.run(stage, check=False, cwd=checkout)
        if completed.returncode != 0:
            print(f"'{' '.join(stage)}' failed in {checkout}")
            return int(completed.returncode)
    inner_scratch = scratch / "inner"
    command = [
        "uv", "run", "--locked", "python", "-m", "arm_rc_ctrl.execution", "run", "--policy", "p-cores", "--",
        "uv", "run", "--locked", "python", "scripts/reproduce_repetition.py",
        "--scratch", str(inner_scratch), *forwarded,
    ]  # fmt: skip
    print(f"reproducing at {resolved[:12]} in {checkout} (remove with: git worktree remove --force {checkout})")
    inner = subprocess.run(command, check=False, cwd=checkout)
    return int(inner.returncode)


def _forwarded(args: argparse.Namespace) -> list[str]:
    """The inner command line of ``--from-checkout`` (output paths made absolute; the scratch is the caller's)."""
    flags = ("keep_going", "exploratory", "skip_resimulation", "skip_fits", "doc005", "gates")
    forwarded: list[str] = ["--" + flag.replace("_", "-") for flag in flags if bool(getattr(args, flag))]
    if args.doc005_cpus is not None:
        forwarded += ["--doc005-cpus", str(args.doc005_cpus)]
    for option in ("summary", "audit"):
        value = cast("Path | None", getattr(args, option))
        if value is not None:
            forwarded += [f"--{option}", str(Path(value).resolve())]
    return forwarded


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point (exit status 1 when any step, gate, or DOC-005 rerun fails)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Reproduce the task 1-a repeated-demonstration pilot from the committed records."
    )
    parser.add_argument("--scratch", type=Path, default=None, help="scratch directory outside the repository")
    parser.add_argument("--summary", type=Path, default=None, help="write the machine-readable summary here")
    parser.add_argument("--audit", type=Path, default=None, help="write the Markdown audit note here")
    parser.add_argument("--keep-going", action="store_true", help="run every step even after a failure")
    parser.add_argument(
        "--exploratory", action="store_true", help="allow a dirty worktree (the re-simulation then fails clearly)"
    )
    parser.add_argument("--skip-resimulation", action="store_true", help="do not re-simulate the declared subset")
    parser.add_argument("--skip-fits", action="store_true", help="do not refit the 156 fits in fresh processes")
    parser.add_argument(
        "--doc005", action="store_true", help="also rerun the historical task 1-a reproduction under this affinity"
    )
    parser.add_argument("--doc005-cpus", type=str, default=None, help="also rerun it pinned to this CPU list (16-31)")
    parser.add_argument("--gates", action="store_true", help="run the repository quality gates and record them")
    parser.add_argument(
        "--from-checkout",
        nargs="?",
        const="HEAD",
        default=None,
        metavar="COMMIT",
        help="reproduce inside a fresh worktree at COMMIT (default HEAD)",
    )
    args = parser.parse_args(argv)
    if args.scratch is None:
        scratch = Path(tempfile.mkdtemp(prefix="arm-rc-ctrl-reproduce-repetition-"))
    else:
        scratch = Path(cast("Path", args.scratch))
    if args.from_checkout is not None:
        return run_from_checkout(scratch, str(args.from_checkout), _forwarded(args))
    ensure_single_thread()
    reruns: list[tuple[str, Sequence[int] | None]] = []
    if bool(args.doc005):
        reruns.append(("canonical-affinity", None))
    if args.doc005_cpus is not None:
        cpus = parse_cpu_list(str(args.doc005_cpus))
        reruns.append((f"cpus-{args.doc005_cpus}", cpus))
    result = reproduce(
        scratch=scratch,
        keep_going=bool(args.keep_going),
        exploratory=bool(args.exploratory),
        skip_resimulation=bool(args.skip_resimulation),
        skip_fits=bool(args.skip_fits),
        doc005=reruns,
        gates=bool(args.gates),
    )
    summary = to_mapping(result)
    summary["ok"] = result.ok
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.summary is not None:
        Path(cast("Path", args.summary)).write_text(text + "\n", encoding="utf-8")
    if args.audit is not None:
        note = audit_markdown(result, command=command_line(_MODULE, argv))
        Path(cast("Path", args.audit)).write_text(note, encoding="utf-8")
    print(text)
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())

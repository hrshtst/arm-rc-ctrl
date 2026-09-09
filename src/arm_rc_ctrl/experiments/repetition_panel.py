# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""The frozen six-configuration panel of the repeated-demonstration pilot (M3REP-001; repetition plan section 5.1).

``task_1a_repetition_v1`` reuses six trial configurations of the completed
timing-only recovery search. The plan fixes their identities (trials 17, 136,
53, 1, 0, 28) and the historical rule that picked them: feasible ranks 1, 67,
and 134 by ``(objective, trial_number)`` over the development ablation's
candidates, and the lowest-numbered infeasible trial of three first-failure
categories. This module resolves every source from digest-verified inputs
(the committed study pointer and its external report payload, the ablation
record, the protocol, model, scenario, development, and dataset files, and the
frozen tracker gains), re-derives the selection, refuses any resolution that
differs from the approved identities, and freezes the complete parameters into
an immutable manifest. The manifest keeps the source study's historical
provenance apart from the implementation revision that wrote it
(clarification C1), records the evaluation-side estimator cutoffs next to the
reservoir/readout point, and names the source ridge parameter that the pilot's
scaled arms derive from.

Command line::

    python -m arm_rc_ctrl.experiments.repetition_panel
        --recovery-docs docs/experiments/task_1a_state_conditioned_recovery
        --dataset data/records/processed/<id>.toml
        --output docs/experiments/task_1a_repeated_demonstration/panel_manifest_v1.json
        --markdown docs/experiments/task_1a_repeated_demonstration/panel_manifest_v1.md [--exploratory]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.data.recovery import RecoveryDatasetRecord, load_processed_record
from arm_rc_ctrl.experiments.baselines import frozen_baseline_digest
from arm_rc_ctrl.experiments.closed_loop import EstimatorSpec
from arm_rc_ctrl.experiments.evidence import load_report_pointer, open_stored_report
from arm_rc_ctrl.experiments.recovery_ablation import AblationReport, CandidateTrial, load_ablation, reason_head
from arm_rc_ctrl.experiments.recovery_search import (
    RECOVERY_TRACKERS,
    RecoverySearchProtocol,
    RecoveryTrialPoint,
    load_recovery_search,
    point_from_params,
    recovery_protocol_digest,
)
from arm_rc_ctrl.provenance import (
    ArtifactReference,
    ProvenanceRecord,
    canonical_json,
    collect_provenance,
    command_line,
    require_clean_for_confirmatory,
    sha256_file,
)
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import open_storage
from arm_rc_ctrl.validation import COMMIT_HEX_LENGTH, SHA256_HEX_LENGTH, is_hex

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arm_rc_ctrl.experiments.evidence import StoredReport
    from arm_rc_ctrl.experiments.recovery_study import RecoveryStudyReport
    from arm_rc_ctrl.experiments.studies import TrialRecord

__all__ = [
    "APPROVED_RULE",
    "EXPERIMENT_LABEL",
    "PANEL_SCHEMA_VERSION",
    "SOURCE_ABLATION_FILE",
    "SOURCE_POINTER_FILE",
    "PanelEntry",
    "PanelManifest",
    "PanelMismatchError",
    "SelectionRule",
    "SourceAblation",
    "SourceConfigs",
    "SourceProvenance",
    "SourceStudy",
    "build_panel_manifest",
    "load_panel",
    "main",
    "panel_to_json",
    "render_panel_markdown",
    "resolve_panel",
]

PANEL_SCHEMA_VERSION: Final = 1
EXPERIMENT_LABEL: Final = "task_1a_repetition_v1"
SOURCE_POINTER_FILE: Final = "recovery_search_no_augmentation_v1.toml"
"""The committed pointer of the timing-only study the panel is drawn from (recovery docs directory)."""
SOURCE_ABLATION_FILE: Final = "development_ablation_v2.json"
"""The committed development ablation whose candidate list fixes the feasible ranking."""
SOURCE_FORMULATION: Final = "no_augmentation"
_SHORT: Final = 12


class PanelMismatchError(ValueError):
    """The re-derived panel differs from the approved identities (decision D3: no post-result substitution)."""


def _require_sha256(name: str, value: str) -> None:
    if not is_hex(value, SHA256_HEX_LENGTH):
        msg = f"{name} must be 64 lowercase hex characters, got {value!r}"
        raise ValueError(msg)


@dataclass(frozen=True)
class SelectionRule:
    """The deterministic historical rule that fixed the panel, with the approved identities it must reproduce."""

    labels: tuple[str, ...]
    """Panel labels in report order."""
    feasible_ranking: str
    feasible_ranks: dict[str, int]
    """Panel label to its 1-based rank among the source study's feasible trials."""
    failure_rule: str
    failure_categories: dict[str, str]
    """Panel label to the first-failure reason head it selects (see :func:`reason_head`)."""
    approved_trials: dict[str, int]
    """Panel label to the approved source trial number (D3); a resolution that differs is refused."""

    def __post_init__(self) -> None:
        """The ranked and failure groups partition the labels, and every label has an approved trial."""
        _check_rule_partition(self)
        _check_rule_values(self)


def _check_rule_partition(rule: SelectionRule) -> None:
    if not rule.labels or len(set(rule.labels)) != len(rule.labels):
        msg = "selection labels must be non-empty and distinct"
        raise ValueError(msg)
    ranked = set(rule.feasible_ranks)
    failing = set(rule.failure_categories)
    if ranked & failing or (ranked | failing) != set(rule.labels):
        msg = "feasible ranks and failure categories must partition the labels"
        raise ValueError(msg)
    if set(rule.approved_trials) != set(rule.labels):
        msg = "every label needs exactly one approved trial number"
        raise ValueError(msg)


def _check_rule_values(rule: SelectionRule) -> None:
    ranks = list(rule.feasible_ranks.values())
    if any(rank < 1 for rank in ranks) or len(set(ranks)) != len(ranks):
        msg = "feasible ranks must be distinct positive integers"
        raise ValueError(msg)
    heads = list(rule.failure_categories.values())
    if len(set(heads)) != len(heads) or not all(head.strip() for head in heads):
        msg = "failure categories must be distinct non-empty reason heads"
        raise ValueError(msg)
    trials = list(rule.approved_trials.values())
    if any(number < 0 for number in trials) or len(set(trials)) != len(trials):
        msg = "approved trials must be distinct non-negative trial numbers"
        raise ValueError(msg)
    if not rule.feasible_ranking.strip() or not rule.failure_rule.strip():
        msg = "the rule text must not be empty"
        raise ValueError(msg)


APPROVED_RULE: Final = SelectionRule(
    labels=(
        "feasible-best",
        "feasible-middle",
        "feasible-worst",
        "failure-actual-dwell",
        "failure-joint-velocity",
        "failure-generated-dwell",
    ),
    feasible_ranking=(
        "Feasible trials of the source study sorted by (objective, trial_number) ascending over the development "
        "ablation's candidates; rank r takes the r-th trial (plan section 5.1)."
    ),
    feasible_ranks={"feasible-best": 1, "feasible-middle": 67, "feasible-worst": 134},
    failure_rule=(
        "Lowest trial number among the study's infeasible trials whose top-level first-failure reason head is the "
        "category; the head is the gate before the first colon, keeping the limit name of a violation, so every "
        "dwell:* head (dwell_stationary, dwell_in_tolerance, or both) is the actual-motion dwell category "
        "(plan sections 5.1 and 12)."
    ),
    failure_categories={
        "failure-actual-dwell": "dwell",
        "failure-joint-velocity": "limit_violation:joint_velocity",
        "failure-generated-dwell": "generated_dwell",
    },
    approved_trials={
        "feasible-best": 17,
        "feasible-middle": 136,
        "feasible-worst": 53,
        "failure-actual-dwell": 1,
        "failure-joint-velocity": 0,
        "failure-generated-dwell": 28,
    },
)
"""The approved panel and the historical rule that produced it (plan section 5.1, decision D3)."""


@dataclass(frozen=True)
class SourceStudy:
    """The digest-verified timing-only study the panel is drawn from."""

    pointer_file: str
    pointer_sha256: str
    study: str
    formulation: str
    protocol_file: str
    protocol_file_sha256: str
    protocol_sha256: str
    """Portable digest of the resolved protocol as the stored report records it."""
    payload: ArtifactReference
    trials_stored: int
    n_feasible: int

    def __post_init__(self) -> None:
        """Digests are well-formed, the formulation is the timing-only arm, and the counts are consistent."""
        for name in ("pointer_sha256", "protocol_file_sha256", "protocol_sha256"):
            _require_sha256(f"source.{name}", getattr(self, name))
        if self.formulation != SOURCE_FORMULATION:
            msg = f"the panel is drawn from the {SOURCE_FORMULATION!r} study, got {self.formulation!r}"
            raise ValueError(msg)
        if not 0 <= self.n_feasible <= self.trials_stored:
            msg = f"inconsistent trial counts: {self.n_feasible} feasible of {self.trials_stored} stored"
            raise ValueError(msg)
        if not (self.pointer_file.strip() and self.study.strip() and self.protocol_file.strip()):
            msg = "the source study needs its pointer, study, and protocol file names"
            raise ValueError(msg)


@dataclass(frozen=True)
class SourceAblation:
    """The committed development ablation whose candidate list defines the feasible ranking."""

    file: str
    sha256: str
    n_candidates: int
    study_candidates: int
    """Candidates belonging to the source study (its feasible trials)."""

    def __post_init__(self) -> None:
        """The digest is well-formed and the study's candidates are a subset."""
        _require_sha256("ablation.sha256", self.sha256)
        if not self.file.strip():
            msg = "the ablation needs its file name"
            raise ValueError(msg)
        if not 0 <= self.study_candidates <= self.n_candidates:
            msg = f"{self.study_candidates} study candidates exceed the {self.n_candidates} recorded"
            raise ValueError(msg)


@dataclass(frozen=True)
class SourceConfigs:
    """Every versioned file the source trials were evaluated under, by repository-relative path and digest."""

    model_file: str
    model_sha256: str
    scenario_file: str
    scenario_sha256: str
    development_file: str
    development_sha256: str
    dataset: str
    dataset_record_file: str
    dataset_record_sha256: str
    dataset_payload_sha256: str
    trackers: dict[str, str]
    """SHA-256 of each frozen tracker's gains, by baseline name."""

    def __post_init__(self) -> None:
        """Digests are well-formed and both frozen trackers are bound."""
        for name in (
            "model_sha256",
            "scenario_sha256",
            "development_sha256",
            "dataset_record_sha256",
            "dataset_payload_sha256",
        ):
            _require_sha256(f"configs.{name}", getattr(self, name))
        if set(self.trackers) != set(RECOVERY_TRACKERS):
            msg = f"configs.trackers must bind exactly {RECOVERY_TRACKERS}, got {sorted(self.trackers)}"
            raise ValueError(msg)
        for name, digest in self.trackers.items():
            _require_sha256(f"configs.trackers.{name}", digest)
        if not self.dataset.strip():
            msg = "configs.dataset must name the dataset artifact"
            raise ValueError(msg)


@dataclass(frozen=True)
class SourceProvenance:
    """Historical provenance of the source study, kept apart from the manifest's own implementation revision (C1)."""

    created_at: str
    project_commit: str
    project_dirty: bool
    lock_sha256: str
    submodules: dict[str, str]
    """Submodule name to the commit checked out when the study ran."""
    builds: dict[str, str]
    """Built package name to ``version@source_commit``."""
    python: str

    def __post_init__(self) -> None:
        """Identity fields are well-formed and present."""
        if not is_hex(self.project_commit, COMMIT_HEX_LENGTH):
            msg = f"source_provenance.project_commit must be a 40-hex commit, got {self.project_commit!r}"
            raise ValueError(msg)
        _require_sha256("source_provenance.lock_sha256", self.lock_sha256)
        if not self.submodules or not self.builds or not self.python.strip() or not self.created_at.strip():
            msg = "source_provenance needs its submodules, builds, interpreter, and timestamp"
            raise ValueError(msg)

    @classmethod
    def from_record(cls, record: ProvenanceRecord) -> SourceProvenance:
        """Reduce a full provenance record to the identity figures the manifest keeps."""
        return cls(
            created_at=record.created_at,
            project_commit=record.project_commit,
            project_dirty=record.project_dirty,
            lock_sha256=record.lock_sha256,
            submodules={s.name: s.checked_out or s.recorded for s in record.submodules},
            builds={b.name: f"{b.version}@{b.source_commit}" for b in record.builds},
            python=record.platform.python,
        )


@dataclass(frozen=True)
class PanelEntry:
    """One frozen source configuration with its complete parameters and the record that selected it."""

    label: str
    source_trial: int
    role: str
    selection: str
    """How the historical rule reached this trial (rank among feasible trials, or the failure category)."""
    warmup_s: float
    point: RecoveryTrialPoint
    """Reservoir, readout, and warm-up parameters exactly as the source study stored them (no augmentation)."""
    estimator: EstimatorSpec
    """Evaluation-side causal derivative-estimator settings; the cutoffs are trial parameters, never recipe fields."""
    base_alpha: float
    """The source ridge parameter alpha_0 that the pilot's R-scaled and S-effective arms scale."""
    feasible: bool
    objective: float | None
    first_failure: str | None
    """The stored top-level first-failure reason of an infeasible source trial."""
    reason_head: str | None
    comparison_label: str | None = None
    """The study's anchor label when the trial was a queued comparison point."""

    def __post_init__(self) -> None:
        """The redundant fields agree with the point, and feasibility decides which figures are present."""
        if not (self.label.strip() and self.role.strip() and self.selection.strip()) or self.source_trial < 0:
            msg = "a panel entry needs its label, role, selection, and a non-negative trial number"
            raise ValueError(msg)
        _check_entry_point(self)
        _check_entry_outcome(self)


def _check_entry_point(entry: PanelEntry) -> None:
    if entry.point.augmentation is not None:
        msg = f"{entry.label}: the timing-only source has no augmentation parameters"
        raise ValueError(msg)
    esn = entry.point.esn
    if entry.warmup_s != entry.point.warmup_s or entry.base_alpha != esn.alpha:
        msg = f"{entry.label}: warmup_s and base_alpha must equal the point's values"
        raise ValueError(msg)
    cutoffs = (entry.estimator.velocity_cutoff_hz, entry.estimator.acceleration_cutoff_hz)
    if cutoffs != (esn.velocity_cutoff_hz, esn.acceleration_cutoff_hz):
        msg = f"{entry.label}: the estimator cutoffs must equal the point's trial parameters"
        raise ValueError(msg)


def _check_entry_outcome(entry: PanelEntry) -> None:
    if entry.feasible:
        if entry.objective is None or entry.first_failure is not None or entry.reason_head is not None:
            msg = f"{entry.label}: a feasible entry records its objective and no failure"
            raise ValueError(msg)
        return
    if not entry.first_failure or entry.reason_head != reason_head(entry.first_failure):
        msg = f"{entry.label}: an infeasible entry records its first failure and the matching reason head"
        raise ValueError(msg)


@dataclass(frozen=True)
class PanelManifest:
    """The immutable panel: sources, rule, six entries, and both provenance revisions."""

    experiment: str
    rule: SelectionRule
    source: SourceStudy
    ablation: SourceAblation
    configs: SourceConfigs
    source_provenance: SourceProvenance
    entries: tuple[PanelEntry, ...]
    provenance: ProvenanceRecord
    """The implementation revision that resolved and wrote the manifest (distinct from ``source_provenance``)."""
    schema_version: int = field(default=PANEL_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        """Entries follow the rule's labels, carry the approved trials, and stay consistent with the sources."""
        if self.schema_version != PANEL_SCHEMA_VERSION:
            msg = f"unsupported panel schema_version {self.schema_version}"
            raise ValueError(msg)
        if self.experiment != EXPERIMENT_LABEL:
            msg = f"the manifest belongs to {EXPERIMENT_LABEL!r}, got {self.experiment!r}"
            raise ValueError(msg)
        if tuple(e.label for e in self.entries) != self.rule.labels:
            msg = f"entries must follow the rule's labels {self.rule.labels} in order"
            raise ValueError(msg)
        for entry in self.entries:
            if entry.source_trial != self.rule.approved_trials[entry.label]:
                msg = (
                    f"{entry.label}: trial {entry.source_trial} is not the approved trial "
                    f"{self.rule.approved_trials[entry.label]} (no post-result substitution, D3)"
                )
                raise PanelMismatchError(msg)
            if entry.feasible != (entry.label in self.rule.feasible_ranks):
                msg = f"{entry.label}: feasibility contradicts its selection group"
                raise ValueError(msg)
        if max(self.rule.feasible_ranks.values()) > self.source.n_feasible:
            msg = "a feasible rank exceeds the source study's feasible count"
            raise ValueError(msg)
        if self.ablation.study_candidates != self.source.n_feasible:
            msg = "the ablation's candidates for the source study must equal its feasible trials"
            raise ValueError(msg)

    def entry(self, label: str) -> PanelEntry:
        """The entry with ``label``."""
        for entry in self.entries:
            if entry.label == label:
                return entry
        msg = f"no panel entry labelled {label!r}"
        raise KeyError(msg)


def _check_sources(report: RecoveryStudyReport, protocol: RecoverySearchProtocol, ablation: AblationReport) -> None:
    if report.formulation != SOURCE_FORMULATION:
        msg = f"the panel is drawn from the {SOURCE_FORMULATION!r} study, got {report.formulation!r}"
        raise ValueError(msg)
    if protocol.name != report.protocol:
        msg = f"protocol {protocol.name!r} is not the report's study {report.protocol!r}"
        raise ValueError(msg)
    digest = recovery_protocol_digest(protocol)
    if digest != report.protocol_sha256:
        msg = (
            f"protocol {protocol.name!r} resolves to digest {digest[:_SHORT]}, the stored report records "
            f"{report.protocol_sha256[:_SHORT]}"
        )
        raise ValueError(msg)
    if ablation.dataset != report.dataset:
        msg = f"the ablation evaluates dataset {ablation.dataset!r}, the study {report.dataset!r}"
        raise ValueError(msg)


def _ranked_candidates(report: RecoveryStudyReport, ablation: AblationReport) -> list[CandidateTrial]:
    candidates = [c for c in ablation.candidates if c.study == report.protocol]
    feasible = {t.number: t for t in report.summary.trials if t.flags.get("feasible") is True}
    if len(candidates) != report.n_feasible or {c.number for c in candidates} != set(feasible):
        msg = f"the ablation's candidates for {report.protocol!r} do not match the study's feasible trials"
        raise ValueError(msg)
    for candidate in candidates:
        if feasible[candidate.number].value != candidate.value:
            msg = f"candidate {candidate.number}: ablation value {candidate.value!r} contradicts the stored trial"
            raise ValueError(msg)
    return sorted(candidates, key=lambda c: (c.value, c.number))


def _entry(
    label: str,
    trial: TrialRecord,
    protocol: RecoverySearchProtocol,
    *,
    role: str,
    selection: str,
    feasible: bool,
) -> PanelEntry:
    point = point_from_params(protocol, trial.params)
    anchor = trial.labels.get("armrc.comparison")
    if anchor is not None:
        role = f"{role}; also the historical anchor {anchor!r}"
    reason = trial.labels.get("reason") or None
    return PanelEntry(
        label=label,
        source_trial=trial.number,
        role=role,
        selection=selection,
        warmup_s=point.warmup_s,
        point=point,
        estimator=point.esn.estimator(max_dt_ratio=protocol.max_dt_ratio),
        base_alpha=point.esn.alpha,
        feasible=feasible,
        objective=trial.value,
        first_failure=None if feasible else reason,
        reason_head=None if feasible else reason_head(reason or ""),
        comparison_label=anchor,
    )


def resolve_panel(
    report: RecoveryStudyReport,
    protocol: RecoverySearchProtocol,
    ablation: AblationReport,
    *,
    rule: SelectionRule = APPROVED_RULE,
) -> tuple[PanelEntry, ...]:
    """Re-derive the panel from the study report and ablation; a resolution that differs from the approval fails."""
    _check_sources(report, protocol, ablation)
    candidates = _ranked_candidates(report, ablation)
    by_number = {t.number: t for t in report.summary.trials}
    entries: dict[str, PanelEntry] = {}
    for label, rank in rule.feasible_ranks.items():
        if rank > len(candidates):
            msg = f"{label}: rank {rank} exceeds the {len(candidates)} feasible candidates"
            raise ValueError(msg)
        candidate = candidates[rank - 1]
        entries[label] = _entry(
            label,
            by_number[candidate.number],
            protocol,
            role=f"Rank {rank} of the {len(candidates)} feasible trials by early-gap objective",
            selection=f"feasible rank {rank} of {len(candidates)} by (objective, trial_number) ascending",
            feasible=True,
        )
    infeasible = [t for t in report.summary.trials if t.flags.get("feasible") is not True]
    for label, category in rule.failure_categories.items():
        matches = sorted(
            (t for t in infeasible if reason_head(t.labels.get("reason", "")) == category), key=lambda t: t.number
        )
        if not matches:
            msg = f"{label}: no infeasible trial of {report.protocol!r} has first-failure category {category!r}"
            raise ValueError(msg)
        entries[label] = _entry(
            label,
            matches[0],
            protocol,
            role=f"Lowest-numbered trial whose first failure is {category}",
            selection=f"lowest-numbered of {len(matches)} infeasible trials whose first failure is {category!r}",
            feasible=False,
        )
    ordered = tuple(entries[label] for label in rule.labels)
    resolved = {e.label: e.source_trial for e in ordered}
    if resolved != rule.approved_trials:
        msg = (
            f"the resolved panel {resolved} differs from the approved identities {rule.approved_trials}; the panel "
            "is fixed by decision D3 and is never replaced from new outcomes"
        )
        raise PanelMismatchError(msg)
    return ordered


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    base = root.resolve()
    return resolved.relative_to(base).as_posix() if resolved.is_relative_to(base) else path.name


def build_panel_manifest(
    entries: Sequence[PanelEntry],
    *,
    report: RecoveryStudyReport,
    pointer: StoredReport,
    pointer_file: Path,
    protocol: RecoverySearchProtocol,
    protocol_file: Path,
    ablation: AblationReport,
    ablation_file: Path,
    dataset: RecoveryDatasetRecord,
    dataset_file: Path,
    provenance: ProvenanceRecord,
    rule: SelectionRule = APPROVED_RULE,
    root: Path | None = None,
) -> PanelManifest:
    """Bind the resolved entries to every source identity; a source that moved since the study fails."""
    root = repository_root() if root is None else root
    if pointer.study != report.protocol or pointer.protocol_sha256 != report.protocol_sha256:
        msg = f"pointer {pointer_file.name} does not name the study {report.protocol!r} at its recorded digest"
        raise ValueError(msg)
    if _relative(protocol_file, root) != report.protocol_file:
        msg = f"protocol file {protocol_file} is not the report's {report.protocol_file!r}"
        raise ValueError(msg)
    if dataset.artifact.artifact_id != report.dataset:
        msg = f"dataset record {dataset.artifact.artifact_id!r} is not the study's dataset {report.dataset!r}"
        raise ValueError(msg)
    dataset.check_scenario(protocol.scenario)
    trackers = dict(report.trackers)
    for name, digest in sorted(trackers.items()):
        current = frozen_baseline_digest(name)
        if current != digest:
            msg = (
                f"frozen tracker {name!r} gains digest {current[:_SHORT]} differs from the source study's "
                f"{digest[:_SHORT]}"
            )
            raise ValueError(msg)
    source = SourceStudy(
        pointer_file=pointer_file.name,
        pointer_sha256=sha256_file(pointer_file),
        study=report.protocol,
        formulation=report.formulation,
        protocol_file=report.protocol_file,
        protocol_file_sha256=sha256_file(protocol_file),
        protocol_sha256=report.protocol_sha256,
        payload=pointer.payload,
        trials_stored=len(report.summary.trials),
        n_feasible=report.n_feasible,
    )
    ablation_source = SourceAblation(
        file=ablation_file.name,
        sha256=sha256_file(ablation_file),
        n_candidates=len(ablation.candidates),
        study_candidates=sum(1 for c in ablation.candidates if c.study == report.protocol),
    )
    configs = SourceConfigs(
        model_file=_relative(protocol.model, root),
        model_sha256=sha256_file(protocol.model),
        scenario_file=_relative(protocol.scenario, root),
        scenario_sha256=sha256_file(protocol.scenario),
        development_file=_relative(protocol.development, root),
        development_sha256=sha256_file(protocol.development),
        dataset=report.dataset,
        dataset_record_file=_relative(dataset_file, root),
        dataset_record_sha256=sha256_file(dataset_file),
        dataset_payload_sha256=dataset.artifact.payload.sha256,
        trackers=trackers,
    )
    return PanelManifest(
        experiment=EXPERIMENT_LABEL,
        rule=rule,
        source=source,
        ablation=ablation_source,
        configs=configs,
        source_provenance=SourceProvenance.from_record(report.provenance),
        entries=tuple(entries),
        provenance=provenance,
    )


def panel_to_json(manifest: PanelManifest) -> str:
    """Canonical JSON of the manifest."""
    return canonical_json(to_mapping(manifest))


def load_panel(path: Path) -> PanelManifest:
    """Strictly rebuild a manifest from JSON (re-deriving every invariant, including the approved identities)."""
    return from_mapping(cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), PanelManifest)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4g}"


def _short(digest: str) -> str:
    return digest[:_SHORT]


def _params_row(entry: PanelEntry) -> str:
    esn, estimator = entry.point.esn, entry.estimator
    cells = (
        entry.label,
        str(esn.n_neurons),
        repr(esn.spectral_radius),
        repr(esn.sparsity),
        repr(esn.leak_rate),
        repr(esn.input_scaling),
        str(esn.seed),
        repr(esn.alpha),
        repr(estimator.velocity_cutoff_hz),
        repr(estimator.acceleration_cutoff_hz),
        repr(estimator.max_dt_ratio),
    )
    return "| " + " | ".join(cells) + " |"


def render_panel_markdown(manifest: PanelManifest) -> str:
    """The Markdown rendering of the manifest (the JSON stays the exact record)."""
    source, ablation, configs, rule = manifest.source, manifest.ablation, manifest.configs, manifest.rule
    historical, current = manifest.source_provenance, manifest.provenance
    trackers = ", ".join(f"`{name}` (`{_short(digest)}`)" for name, digest in sorted(configs.trackers.items()))
    approved = ", ".join(f"{label} = trial {rule.approved_trials[label]}" for label in rule.labels)
    lines = [
        "# Task 1-a repetition panel manifest (v1)",
        "",
        (
            f"Experiment `{manifest.experiment}`: the six fixed source configurations of the timing-only recovery "
            "search (repetition plan section 5.1, decision D3), frozen before any pilot arm is trained."
        ),
        "",
        "## Sources",
        "",
        (
            f"- Study `{source.study}` via pointer `{source.pointer_file}` (sha256 `{_short(source.pointer_sha256)}`); "
            f"report payload `{source.payload.uri}` (sha256 `{_short(source.payload.sha256)}`, "
            f"{source.payload.size} bytes); {source.trials_stored} trials stored, {source.n_feasible} feasible."
        ),
        (
            f"- Protocol `{source.protocol_file}` (file sha256 `{_short(source.protocol_file_sha256)}`, "
            f"portable digest `{_short(source.protocol_sha256)}`)."
        ),
        (
            f"- Ablation `{ablation.file}` (sha256 `{_short(ablation.sha256)}`): {ablation.study_candidates} of "
            f"{ablation.n_candidates} candidates belong to the source study."
        ),
        (
            f"- Model `{configs.model_file}` (`{_short(configs.model_sha256)}`), scenario `{configs.scenario_file}` "
            f"(`{_short(configs.scenario_sha256)}`), development levels `{configs.development_file}` "
            f"(`{_short(configs.development_sha256)}`)."
        ),
        (
            f"- Dataset `{configs.dataset}`: record `{configs.dataset_record_file}` "
            f"(`{_short(configs.dataset_record_sha256)}`), payload sha256 `{_short(configs.dataset_payload_sha256)}`."
        ),
        f"- Frozen trackers: {trackers}.",
        "",
        "## Selection rule",
        "",
        f"- Feasible ranking: {rule.feasible_ranking}",
        f"- Failure categories: {rule.failure_rule}",
        (
            f"- Approved identities (D3): {approved}. A resolution that differs fails; no configuration is replaced "
            "because its repeated or augmented variant performs poorly."
        ),
        "",
        "## Panel",
        "",
        "| label | trial | role | warm-up (s) | objective | first failure |",
        "| --- | ---: | --- | ---: | ---: | --- |",
    ]
    lines.extend(
        f"| {e.label} | {e.source_trial} | {e.role} | {e.warmup_s:g} | {_fmt(e.objective)} | {e.first_failure or ''} |"
        for e in manifest.entries
    )
    lines.extend(
        [
            "",
            "## Parameters",
            "",
            (
                "Reservoir/readout parameters and the warm-up are recipe inputs; the estimator cutoffs are "
                "evaluation-side settings, never recipe fields; `alpha` is the source ridge parameter alpha_0 that "
                "the pilot's R-scaled and S-effective arms scale."
            ),
            "",
            (
                "| label | n_neurons | spectral_radius | sparsity | leak_rate | input_scaling | seed | alpha "
                "| velocity_cutoff_hz | acceleration_cutoff_hz | max_dt_ratio |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(_params_row(e) for e in manifest.entries)
    submodules = ", ".join(f"{name} `{_short(commit)}`" for name, commit in sorted(historical.submodules.items()))
    builds = ", ".join(f"{name} {build}" for name, build in sorted(historical.builds.items()))
    historical_dirty = " (dirty)" if historical.project_dirty else ""
    current_dirty = " (dirty)" if current.project_dirty else ""
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            (
                f"- Source study: commit `{_short(historical.project_commit)}`"
                f"{historical_dirty}, created {historical.created_at}, Python "
                f"{historical.python}, lock `{_short(historical.lock_sha256)}`; submodules {submodules}; "
                f"builds {builds}."
            ),
            (
                f"- This manifest: commit `{_short(current.project_commit)}`{current_dirty}, "
                f"created {current.created_at}, Python {current.platform.python}, lock `{_short(current.lock_sha256)}`."
            ),
            "",
            (
                "The two revisions are recorded separately (clarification C1): the parameters come from the "
                "historical study; the manifest comes from the pilot implementation. No new reservoir seed is "
                "introduced by this pilot."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Resolve and freeze the repeated-demonstration pilot's panel.")
    parser.add_argument(
        "--recovery-docs", type=Path, required=True, help="recovery experiment docs directory (pointer and ablation)"
    )
    parser.add_argument(
        "--pointer", type=str, default=SOURCE_POINTER_FILE, help="study pointer file in --recovery-docs"
    )
    parser.add_argument("--ablation", type=str, default=SOURCE_ABLATION_FILE, help="ablation JSON in --recovery-docs")
    parser.add_argument("--dataset", type=Path, required=True, help="recovery dataset record (TOML)")
    parser.add_argument("--output", type=Path, required=True, help="panel manifest JSON to write (must not exist)")
    parser.add_argument("--markdown", type=Path, required=True, help="panel Markdown to write (must not exist)")
    parser.add_argument("--exploratory", action="store_true", help="allow a dirty worktree")
    args = parser.parse_args(argv)
    for target in (args.output, args.markdown):
        if Path(target).exists():
            msg = f"refusing to overwrite {target}"
            raise FileExistsError(msg)
    root = repository_root()
    docs = Path(args.recovery_docs)
    pointer_file = docs / str(args.pointer)
    ablation_file = docs / str(args.ablation)
    dataset_file = Path(args.dataset)
    store = open_storage()
    pointer = load_report_pointer(pointer_file)
    report = open_stored_report(store, pointer)
    protocol_file = root / report.protocol_file
    protocol = load_recovery_search(protocol_file)
    ablation = load_ablation(ablation_file)
    dataset = load_processed_record(dataset_file)
    if not isinstance(dataset, RecoveryDatasetRecord):
        msg = f"{dataset_file} is not a recovery dataset record, got {type(dataset).__name__}"
        raise TypeError(msg)
    entries = resolve_panel(report, protocol, ablation)
    resolved = {
        "experiment": EXPERIMENT_LABEL,
        "pointer": {pointer_file.name: sha256_file(pointer_file)},
        "report_payload": pointer.payload.sha256,
        "ablation": {ablation_file.name: sha256_file(ablation_file)},
        "protocol_file": report.protocol_file,
        "dataset": report.dataset,
        "rule": to_mapping(APPROVED_RULE),
        "command": command_line("arm_rc_ctrl.experiments.repetition_panel", sys.argv[1:] if argv is None else argv),
    }
    provenance = collect_provenance(
        resolved,
        seeds={},
        artifacts=[pointer.payload],
        exploratory=bool(args.exploratory),
        now=datetime.now(tz=UTC),
    )
    require_clean_for_confirmatory(provenance)
    manifest = build_panel_manifest(
        entries,
        report=report,
        pointer=pointer,
        pointer_file=pointer_file,
        protocol=protocol,
        protocol_file=protocol_file,
        ablation=ablation,
        ablation_file=ablation_file,
        dataset=dataset,
        dataset_file=dataset_file,
        provenance=provenance,
        root=root,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(panel_to_json(manifest) + "\n", encoding="utf-8")
    Path(args.markdown).write_text(render_panel_markdown(manifest), encoding="utf-8")
    print(
        json.dumps(
            {
                "experiment": manifest.experiment,
                "study": manifest.source.study,
                "panel": {e.label: e.source_trial for e in manifest.entries},
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())

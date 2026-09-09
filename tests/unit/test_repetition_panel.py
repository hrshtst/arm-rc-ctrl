# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3REP-001: the panel resolver re-derives the approved six configurations and refuses every substitution."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.data.recovery import RecoveryDatasetRecord, load_processed_record
from arm_rc_ctrl.experiments.baselines import frozen_baseline_digest
from arm_rc_ctrl.experiments.esn_search import TrialPoint
from arm_rc_ctrl.experiments.evidence import (
    REPORT_POINTER_SCHEMA,
    StoredReport,
    report_pointer,
    store_report_payload,
    write_report_pointer,
)
from arm_rc_ctrl.experiments.recovery_ablation import (
    AblationReport,
    ArmSummary,
    CandidateCell,
    CandidateTrial,
    ablation_to_json,
)
from arm_rc_ctrl.experiments.recovery_search import (
    RECOVERY_TRACKERS,
    AugmentationPoint,
    RecoverySearchProtocol,
    RecoveryTrialPoint,
    load_recovery_search,
    recovery_protocol_digest,
)
from arm_rc_ctrl.experiments.recovery_study import RecoveryStudyReport, report_to_json
from arm_rc_ctrl.experiments.repetition_panel import (
    APPROVED_RULE,
    EXPERIMENT_LABEL,
    PanelManifest,
    PanelMismatchError,
    SelectionRule,
    SourceAblation,
    SourceConfigs,
    SourceProvenance,
    SourceStudy,
    build_panel_manifest,
    load_panel,
    main,
    panel_to_json,
    render_panel_markdown,
    resolve_panel,
)
from arm_rc_ctrl.experiments.studies import StudySummary, TrialRecord
from arm_rc_ctrl.provenance import ArtifactReference, ProvenanceRecord, collect_provenance
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageRoot

if TYPE_CHECKING:
    from pathlib import Path

REPO_ROOT = repository_root()
PROTOCOL_FILE = "configs/studies/recovery_search_1a_no_augmentation_v1.toml"
DATASET_FILE = "data/records/processed/processed-20260903-ce343c8ce6a5.toml"
DATASET = "processed-20260903-ce343c8ce6a5"
CELLS = (
    "posture_small:pd_v2",
    "posture_small:computed_torque",
    "posture_large:pd_v2",
    "posture_large:computed_torque",
)
INFEASIBLE_REASONS = {
    0: "scenario 23 [pd_v2]: limit_violation:joint_velocity",
    1: "scenario 0 [pd_v2]: dwell:dwell_stationary",
    2: "scenario 0 [pd_v2]: dwell:dwell_in_tolerance,dwell_stationary",
    3: "scenario 4 [computed_torque]: limit_violation:joint_velocity",
    28: "scenario 1 [pd_v2]: generated_dwell:generated_dwell_stationary",
    29: "scenario 1 [pd_v2]: generated_dwell:generated_dwell_stationary",
}
FEASIBLE_NUMBERS = tuple(n for n in range(4, 140) if n not in INFEASIBLE_REASONS)  # 134 numbers, 17/53/136 included
WARMUPS = (0.25, 0.0, 1.0, 0.5, 2.0)


@pytest.fixture(scope="module")
def protocol() -> RecoverySearchProtocol:
    """The real timing-only protocol; the synthetic report claims its name and digest."""
    return load_recovery_search(REPO_ROOT / PROTOCOL_FILE)


@pytest.fixture(scope="module")
def provenance() -> ProvenanceRecord:
    """One exploratory provenance record shared by the synthetic sources."""
    return collect_provenance({}, seeds={}, artifacts=[], exploratory=True)


def _point(number: int) -> RecoveryTrialPoint:
    """A distinct in-space point per trial (the panel must carry each trial's own parameters)."""
    return RecoveryTrialPoint(
        esn=TrialPoint(
            n_neurons=100 + 50 * (number % 7),
            spectral_radius=0.8 + 0.001 * (number % 400),
            sparsity=0.5 + 0.001 * (number % 400),
            leak_rate=0.01 + 0.0005 * (number % 400),
            input_scaling=0.02 + 0.001 * (number % 400),
            seed=1 + number,
            alpha=0.001 + 0.001 * (number % 900),
            velocity_cutoff_hz=5.0 + 0.05 * (number % 400),
            acceleration_cutoff_hz=5.0 + 0.05 * (number % 400),
        ),
        warmup_s=WARMUPS[number % len(WARMUPS)],
        augmentation=None,
    )


def _ordered_feasible() -> list[int]:
    """Feasible trials by rank: 17 first, 136 at rank 67, 53 last."""
    others = [n for n in FEASIBLE_NUMBERS if n not in (17, 136, 53)]
    ranked = [17, *others]
    ranked.insert(66, 136)
    return [*ranked, 53]


def _trials(*, infeasible: dict[int, str] | None = None, anchor: int | None = 0) -> tuple[TrialRecord, ...]:
    infeasible = INFEASIBLE_REASONS if infeasible is None else infeasible
    values = {number: 0.5 + 0.001 * rank for rank, number in enumerate(_ordered_feasible())}
    records: list[TrialRecord] = []
    for number in sorted([*values, *infeasible]):
        params = {k: float(v) for k, v in _point(number).params().items()}
        feasible = number in values
        labels: dict[str, str] = {}
        if not feasible:
            labels["reason"] = infeasible[number]
        if number == anchor:
            labels["armrc.comparison"] = "anchor-v4-tw1"
        records.append(
            TrialRecord(
                number=number,
                state="COMPLETE",
                value=values[number] if feasible else 10.0,
                params=params,
                flags={"feasible": feasible},
                labels=labels,
            )
        )
    return tuple(records)


def _report(
    protocol: RecoverySearchProtocol,
    provenance: ProvenanceRecord,
    trials: tuple[TrialRecord, ...],
    *,
    digest: str | None = None,
    trackers: dict[str, str] | None = None,
) -> RecoveryStudyReport:
    feasible = [t for t in trials if t.flags.get("feasible") is True]
    best = min(feasible, key=lambda t: (t.value if t.value is not None else float("inf"), t.number))
    summary = StudySummary(
        name=protocol.name,
        storage=f"armrc://optuna/{protocol.name}.db",
        direction="minimize",
        identity={},
        trials=trials,
        n_complete=len(trials),
        n_pruned=0,
        best_number=best.number,
        best_value=best.value,
        selection_rule="feasible",
    )
    return RecoveryStudyReport(
        protocol=protocol.name,
        protocol_file=PROTOCOL_FILE,
        protocol_sha256=recovery_protocol_digest(protocol) if digest is None else digest,
        formulation="no_augmentation",
        dataset=DATASET,
        trackers={name: frozen_baseline_digest(name) for name in RECOVERY_TRACKERS} if trackers is None else trackers,
        budget=500,
        trials_run=0,
        summary=summary,
        best_point=_point(best.number),
        n_feasible=len(feasible),
        provenance=provenance,
    )


def _ablation(report: RecoveryStudyReport, provenance: ProvenanceRecord, *, drop: int | None = None) -> AblationReport:
    cell = CandidateCell(gap_median=0.5, jump_median=1.5, improving_both=0, n=20, passes=False)
    candidates = tuple(
        CandidateTrial(
            study=report.protocol,
            number=t.number,
            value=t.value if t.value is not None else 0.0,
            warmup_s=float(t.params["warmup_s"]),
            cells=dict.fromkeys(CELLS, cell),
            eligible=False,
        )
        for t in report.summary.trials
        if t.flags.get("feasible") is True and t.number != drop
    )
    infeasible = [t for t in report.summary.trials if t.flags.get("feasible") is not True]
    reasons: dict[str, int] = {}
    for trial in infeasible:
        head = trial.labels["reason"].partition("]: ")[2].split(":")[0]
        head = "limit_violation:joint_velocity" if head == "limit_violation" else head
        reasons[head] = reasons.get(head, 0) + 1
    by_warmup: dict[str, int] = {}
    for candidate in candidates:
        key = f"{candidate.warmup_s:g}"
        by_warmup[key] = by_warmup.get(key, 0) + 1
    arm = ArmSummary(
        study=report.protocol,
        formulation="no_augmentation",
        file="recovery_search_no_augmentation_v1.toml",
        protocol_sha256=report.protocol_sha256,
        budget=report.budget,
        trials_stored=len(candidates) + len(infeasible),
        n_feasible=len(candidates),
        best_number=report.summary.best_number,
        best_value=report.summary.best_value,
        reasons=reasons,
        feasible_by_warmup=by_warmup,
    )
    return AblationReport(
        dataset=DATASET,
        arms=(arm,),
        candidates=candidates,
        n_eligible=0,
        improving_rule="15 of 20",
        provenance=provenance,
    )


def _sources(tmp_path: Path, report: RecoveryStudyReport) -> tuple[StoredReport, Path, Path, RecoveryDatasetRecord]:
    pointer = StoredReport(
        schema=REPORT_POINTER_SCHEMA,
        study=report.protocol,
        formulation=report.formulation,
        protocol_sha256=report.protocol_sha256,
        dataset=report.dataset,
        budget=report.budget,
        trials_stored=len(report.summary.trials),
        n_feasible=report.n_feasible,
        best_number=report.summary.best_number,
        best_value=report.summary.best_value,
        payload=ArtifactReference("armrc://reports/task_1a_state_conditioned_recovery/x-abc.json", "a" * 64, 10),
    )
    pointer_file = tmp_path / "recovery_search_no_augmentation_v1.toml"
    pointer_file.write_text("pointer\n", encoding="utf-8")
    ablation_file = tmp_path / "development_ablation_v2.json"
    ablation_file.write_text("{}\n", encoding="utf-8")
    dataset = load_processed_record(REPO_ROOT / DATASET_FILE)
    assert isinstance(dataset, RecoveryDatasetRecord)
    return pointer, pointer_file, ablation_file, dataset


def _manifest(tmp_path: Path, protocol: RecoverySearchProtocol, provenance: ProvenanceRecord) -> PanelManifest:
    report = _report(protocol, provenance, _trials())
    ablation = _ablation(report, provenance)
    pointer, pointer_file, ablation_file, dataset = _sources(tmp_path, report)
    entries = resolve_panel(report, protocol, ablation)
    return build_panel_manifest(
        entries,
        report=report,
        pointer=pointer,
        pointer_file=pointer_file,
        protocol=protocol,
        protocol_file=REPO_ROOT / PROTOCOL_FILE,
        ablation=ablation,
        ablation_file=ablation_file,
        dataset=dataset,
        dataset_file=REPO_ROOT / DATASET_FILE,
        provenance=provenance,
    )


def test_approved_rule_fixes_the_six_identities() -> None:
    """The module constant carries the plan's labels, ranks, categories, and trial numbers."""
    assert APPROVED_RULE.labels == (
        "feasible-best",
        "feasible-middle",
        "feasible-worst",
        "failure-actual-dwell",
        "failure-joint-velocity",
        "failure-generated-dwell",
    )
    assert APPROVED_RULE.feasible_ranks == {"feasible-best": 1, "feasible-middle": 67, "feasible-worst": 134}
    assert APPROVED_RULE.failure_categories["failure-actual-dwell"] == "dwell"
    assert APPROVED_RULE.approved_trials == {
        "feasible-best": 17,
        "feasible-middle": 136,
        "feasible-worst": 53,
        "failure-actual-dwell": 1,
        "failure-joint-velocity": 0,
        "failure-generated-dwell": 28,
    }
    assert len(FEASIBLE_NUMBERS) == 134


def test_resolves_the_approved_panel_with_complete_parameters(
    protocol: RecoverySearchProtocol, provenance: ProvenanceRecord
) -> None:
    """Ranks and failure categories reproduce 17/136/53/1/0/28 and every entry carries its own trial's point."""
    report = _report(protocol, provenance, _trials())
    entries = resolve_panel(report, protocol, _ablation(report, provenance))
    assert [e.label for e in entries] == list(APPROVED_RULE.labels)
    assert {e.label: e.source_trial for e in entries} == APPROVED_RULE.approved_trials
    for entry in entries:
        assert entry.point == _point(entry.source_trial)
        assert entry.warmup_s == entry.point.warmup_s
        assert entry.base_alpha == entry.point.esn.alpha
        assert entry.estimator.velocity_cutoff_hz == entry.point.esn.velocity_cutoff_hz
        assert entry.estimator.max_dt_ratio == protocol.max_dt_ratio
    best, middle, worst = entries[:3]
    assert best.feasible
    assert middle.feasible
    assert worst.feasible
    assert best.objective is not None
    assert worst.objective is not None
    assert best.objective < worst.objective
    assert "rank 1 of 134" in best.selection
    assert "rank 67 of 134" in middle.selection
    dwell, velocity, generated = entries[3:]
    assert (dwell.reason_head, velocity.reason_head, generated.reason_head) == (
        "dwell",
        "limit_violation:joint_velocity",
        "generated_dwell",
    )
    assert velocity.comparison_label == "anchor-v4-tw1"
    assert "historical anchor" in velocity.role
    assert dwell.first_failure == INFEASIBLE_REASONS[1]


def test_every_dwell_head_counts_as_the_actual_dwell_category(
    protocol: RecoverySearchProtocol, provenance: ProvenanceRecord
) -> None:
    """A combined ``dwell:dwell_in_tolerance,dwell_stationary`` head still selects trial 1 for the dwell category."""
    reasons = dict(INFEASIBLE_REASONS)
    reasons[1] = "scenario 0 [computed_torque]: dwell:dwell_in_tolerance,dwell_stationary"
    report = _report(protocol, provenance, _trials(infeasible=reasons))
    entries = resolve_panel(report, protocol, _ablation(report, provenance))
    dwell = entries[3]
    assert dwell.source_trial == 1
    assert dwell.reason_head == "dwell"


def test_a_different_resolution_is_a_substitution_error(
    protocol: RecoverySearchProtocol, provenance: ProvenanceRecord
) -> None:
    """When the sources would pick another trial, the approved identities are never quietly replaced."""
    reasons = {k: v for k, v in INFEASIBLE_REASONS.items() if k != 1}  # trial 2 now has the lowest dwell failure
    report = _report(protocol, provenance, _trials(infeasible=reasons))
    with pytest.raises(PanelMismatchError, match="approved identities"):
        resolve_panel(report, protocol, _ablation(report, provenance))
    rule = SelectionRule(
        labels=APPROVED_RULE.labels,
        feasible_ranking=APPROVED_RULE.feasible_ranking,
        feasible_ranks=APPROVED_RULE.feasible_ranks,
        failure_rule=APPROVED_RULE.failure_rule,
        failure_categories=APPROVED_RULE.failure_categories,
        approved_trials={**APPROVED_RULE.approved_trials, "feasible-best": 18},
    )
    full = _report(protocol, provenance, _trials())
    with pytest.raises(PanelMismatchError, match="D3"):
        resolve_panel(full, protocol, _ablation(full, provenance), rule=rule)


def test_missing_or_mismatched_sources_fail(protocol: RecoverySearchProtocol, provenance: ProvenanceRecord) -> None:
    """Protocol digest drift, ablation/study disagreement, and a missing failure category are all refused."""
    report = _report(protocol, provenance, _trials())
    ablation = _ablation(report, provenance)
    drifted = _report(protocol, provenance, _trials(), digest="f" * 64)
    with pytest.raises(ValueError, match="digest"):
        resolve_panel(drifted, protocol, _ablation(drifted, provenance))
    with pytest.raises(ValueError, match="do not match the study's feasible trials"):
        resolve_panel(report, protocol, _ablation(report, provenance, drop=17))
    reasons = {k: v for k, v in INFEASIBLE_REASONS.items() if k not in (28, 29)}
    without_generated = _report(protocol, provenance, _trials(infeasible=reasons))
    with pytest.raises(ValueError, match="generated_dwell"):
        resolve_panel(without_generated, protocol, _ablation(without_generated, provenance))
    assert ablation.n_eligible == 0


def test_manifest_binds_sources_and_roundtrips_strictly(
    tmp_path: Path, protocol: RecoverySearchProtocol, provenance: ProvenanceRecord
) -> None:
    """The manifest hashes every source, separates the two provenance revisions, and reloads byte-exactly."""
    manifest = _manifest(tmp_path, protocol, provenance)
    assert manifest.experiment == EXPERIMENT_LABEL
    assert manifest.source.protocol_file == PROTOCOL_FILE
    assert manifest.configs.dataset == DATASET
    assert manifest.configs.scenario_file == "configs/tasks/task_1a.toml"
    assert manifest.configs.model_file == "configs/models/esn_task_1a_v4.toml"
    assert set(manifest.configs.trackers) == set(RECOVERY_TRACKERS)
    assert manifest.ablation.study_candidates == manifest.source.n_feasible == 134
    assert manifest.source_provenance.project_commit == provenance.project_commit
    assert manifest.provenance == provenance
    assert manifest.entry("failure-joint-velocity").source_trial == 0
    with pytest.raises(KeyError):
        manifest.entry("missing")
    file = tmp_path / "panel.json"
    file.write_text(panel_to_json(manifest) + "\n", encoding="utf-8")
    assert load_panel(file) == manifest
    tampered = json.loads(file.read_text(encoding="utf-8"))
    tampered["entries"][0]["source_trial"] = 18
    file.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="substitution"):
        load_panel(file)
    reordered = json.loads(panel_to_json(manifest))
    reordered["entries"].reverse()
    file.write_text(json.dumps(reordered), encoding="utf-8")
    with pytest.raises(ValueError, match="in order"):
        load_panel(file)


def test_manifest_refuses_moved_sources(
    tmp_path: Path, protocol: RecoverySearchProtocol, provenance: ProvenanceRecord
) -> None:
    """Changed tracker gains, another dataset, or a pointer for another digest fail the binding."""
    report = _report(protocol, provenance, _trials())
    ablation = _ablation(report, provenance)
    pointer, pointer_file, ablation_file, dataset = _sources(tmp_path, report)
    entries = resolve_panel(report, protocol, ablation)

    def build(report_: RecoveryStudyReport, pointer_: StoredReport) -> PanelManifest:
        return build_panel_manifest(
            entries,
            report=report_,
            pointer=pointer_,
            pointer_file=pointer_file,
            protocol=protocol,
            protocol_file=REPO_ROOT / PROTOCOL_FILE,
            ablation=ablation,
            ablation_file=ablation_file,
            dataset=dataset,
            dataset_file=REPO_ROOT / DATASET_FILE,
            provenance=provenance,
        )

    changed_gains = _report(protocol, provenance, _trials(), trackers=dict.fromkeys(RECOVERY_TRACKERS, "e" * 64))
    with pytest.raises(ValueError, match="frozen tracker"):
        build(changed_gains, pointer)
    other_pointer = StoredReport(
        schema=REPORT_POINTER_SCHEMA,
        study=pointer.study,
        formulation=pointer.formulation,
        protocol_sha256="d" * 64,
        dataset=pointer.dataset,
        budget=pointer.budget,
        trials_stored=pointer.trials_stored,
        n_feasible=pointer.n_feasible,
        best_number=pointer.best_number,
        best_value=pointer.best_value,
        payload=pointer.payload,
    )
    with pytest.raises(ValueError, match="recorded digest"):
        build(report, other_pointer)


def test_source_records_validate_their_identity_fields(provenance: ProvenanceRecord) -> None:
    """Malformed digests, the wrong formulation, unknown trackers, and inconsistent counts are rejected."""
    payload = ArtifactReference("armrc://reports/x/y-abc.json", "a" * 64, 1)
    with pytest.raises(ValueError, match="hex"):
        SourceStudy("p.toml", "zz", "s", "no_augmentation", "c.toml", "b" * 64, "c" * 64, payload, 10, 5)
    with pytest.raises(ValueError, match="no_augmentation"):
        SourceStudy("p.toml", "a" * 64, "s", "contractive", "c.toml", "b" * 64, "c" * 64, payload, 10, 5)
    with pytest.raises(ValueError, match="counts"):
        SourceStudy("p.toml", "a" * 64, "s", "no_augmentation", "c.toml", "b" * 64, "c" * 64, payload, 4, 5)
    with pytest.raises(ValueError, match="exceed"):
        SourceAblation("a.json", "a" * 64, 3, 4)
    with pytest.raises(ValueError, match="trackers"):
        SourceConfigs("m", "a" * 64, "s", "a" * 64, "d", "a" * 64, "ds", "r", "a" * 64, "a" * 64, {"pd_v2": "a" * 64})
    reduced = SourceProvenance.from_record(provenance)
    assert reduced.project_commit == provenance.project_commit
    assert set(reduced.submodules) == {s.name for s in provenance.submodules}
    with pytest.raises(ValueError, match="40-hex"):
        SourceProvenance(
            created_at="t",
            project_commit="abc",
            project_dirty=False,
            lock_sha256="a" * 64,
            submodules={"rclib": "x"},
            builds={"rclib": "1@x"},
            python="3.12",
        )
    with pytest.raises(ValueError, match="partition"):
        SelectionRule(("a", "b"), "r", {"a": 1}, "f", {"a": "dwell"}, {"a": 0, "b": 1})
    with pytest.raises(ValueError, match="distinct positive"):
        SelectionRule(("a", "b"), "r", {"a": 0}, "f", {"b": "dwell"}, {"a": 0, "b": 1})
    with pytest.raises(ValueError, match="approved trial"):
        SelectionRule(("a", "b"), "r", {"a": 1}, "f", {"b": "dwell"}, {"a": 0})
    with pytest.raises(ValueError, match="partition"):
        SelectionRule(("a", "b"), "r", {"a": 1}, "f", {"a": "dwell", "b": "x"}, {"a": 0, "b": 1})
    with pytest.raises(ValueError, match="non-empty and distinct"):
        SelectionRule(("a", "a"), "r", {"a": 1}, "f", {}, {"a": 0})
    with pytest.raises(ValueError, match="distinct non-negative"):
        SelectionRule(("a", "b"), "r", {"a": 1}, "f", {"b": "dwell"}, {"a": 0, "b": 0})
    with pytest.raises(ValueError, match="reason heads"):
        SelectionRule(("a", "b"), "r", {}, "f", {"a": "dwell", "b": "dwell"}, {"a": 0, "b": 1})
    with pytest.raises(ValueError, match="rule text"):
        SelectionRule(("a",), " ", {"a": 1}, "f", {}, {"a": 0})
    with pytest.raises(ValueError, match="needs its pointer"):
        SourceStudy(" ", "a" * 64, "s", "no_augmentation", "c.toml", "b" * 64, "c" * 64, payload, 10, 5)
    with pytest.raises(ValueError, match="needs its file name"):
        SourceAblation(" ", "a" * 64, 4, 3)
    with pytest.raises(ValueError, match="must name the dataset"):
        SourceConfigs(
            "m",
            "a" * 64,
            "s",
            "a" * 64,
            "d",
            "a" * 64,
            " ",
            "r",
            "a" * 64,
            "a" * 64,
            dict.fromkeys(RECOVERY_TRACKERS, "a" * 64),
        )
    with pytest.raises(ValueError, match="needs its submodules"):
        SourceProvenance(
            created_at="t",
            project_commit="a" * 40,
            project_dirty=False,
            lock_sha256="a" * 64,
            submodules={},
            builds={"rclib": "1@x"},
            python="3.12",
        )


def test_markdown_carries_sources_rule_panel_parameters_and_provenance(
    tmp_path: Path, protocol: RecoverySearchProtocol, provenance: ProvenanceRecord
) -> None:
    """The rendering names every section the acceptance criteria ask for."""
    text = render_panel_markdown(_manifest(tmp_path, protocol, provenance))
    for required in (
        "# Task 1-a repetition panel manifest (v1)",
        "## Sources",
        "## Selection rule",
        "feasible-best = trial 17",
        "failure-generated-dwell = trial 28",
        "## Panel",
        "| feasible-best | 17 |",
        "| failure-joint-velocity | 0 |",
        "## Parameters",
        "velocity_cutoff_hz",
        "evaluation-side settings",
        "## Provenance",
        "- Source study: commit",
        "- This manifest: commit",
        "clarification C1",
    ):
        assert required in text


def test_main_writes_the_manifest_once_from_a_stored_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: RecoverySearchProtocol, provenance: ProvenanceRecord
) -> None:
    """The command resolves pointer, payload, ablation, and dataset, writes JSON and Markdown, and never overwrites."""
    root = tmp_path / "store"
    root.mkdir()
    store = StorageRoot(root, repositories=(REPO_ROOT,))
    monkeypatch.setenv("ARM_RC_CTRL_STORAGE_ROOT", str(root))
    report = _report(protocol, provenance, _trials())
    payload = store_report_payload(store, report_to_json(report) + "\n", name="recovery_search_no_augmentation_v1")
    docs = tmp_path / "docs"
    docs.mkdir()
    write_report_pointer(docs / "recovery_search_no_augmentation_v1.toml", report_pointer(report, payload))
    ablation_text = ablation_to_json(_ablation(report, provenance)) + "\n"
    (docs / "development_ablation_v2.json").write_text(ablation_text, encoding="utf-8")
    output = tmp_path / "panel_manifest_v1.json"
    markdown = tmp_path / "panel_manifest_v1.md"
    argv = [
        "--recovery-docs",
        str(docs),
        "--dataset",
        str(REPO_ROOT / DATASET_FILE),
        "--output",
        str(output),
        "--markdown",
        str(markdown),
        "--exploratory",
    ]
    assert main(argv) == 0
    manifest = load_panel(output)
    assert {e.label: e.source_trial for e in manifest.entries} == APPROVED_RULE.approved_trials
    assert manifest.source.payload == payload
    assert manifest.provenance.exploratory is True
    assert manifest.provenance.artifacts == (payload,)
    assert manifest.source_provenance.created_at == provenance.created_at
    assert render_panel_markdown(manifest) == markdown.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing"):
        main(argv)
    other_record = next(
        p for p in sorted((REPO_ROOT / "data" / "records" / "processed").glob("*.toml")) if p.name != f"{DATASET}.toml"
    )
    argv[3] = str(other_record)
    argv[5] = str(tmp_path / "other.json")
    argv[7] = str(tmp_path / "other.md")
    with pytest.raises(TypeError, match="not a recovery dataset record"):
        main(argv)


def test_entry_and_manifest_invariants_reject_inconsistent_records(
    tmp_path: Path, protocol: RecoverySearchProtocol, provenance: ProvenanceRecord
) -> None:
    """Every redundant field of an entry and every cross-check of the manifest is re-derived on construction."""
    manifest = _manifest(tmp_path, protocol, provenance)
    best = manifest.entry("feasible-best")
    with pytest.raises(ValueError, match="augmentation"):
        replace(best, point=replace(best.point, augmentation=AugmentationPoint(16, 0.05, 0.99, 1.0)))
    with pytest.raises(ValueError, match="warmup_s and base_alpha"):
        replace(best, base_alpha=best.base_alpha * 2)
    with pytest.raises(ValueError, match="estimator cutoffs"):
        replace(best, estimator=replace(best.estimator, velocity_cutoff_hz=1.0))
    with pytest.raises(ValueError, match="feasible entry"):
        replace(best, objective=None)
    with pytest.raises(ValueError, match="needs its label"):
        replace(best, label=" ")
    failed = manifest.entry("failure-actual-dwell")
    with pytest.raises(ValueError, match="infeasible entry"):
        replace(failed, reason_head="generated_dwell")
    with pytest.raises(ValueError, match="schema_version"):
        replace(manifest, schema_version=2)
    with pytest.raises(ValueError, match="belongs to"):
        replace(manifest, experiment="other")
    flipped = replace(
        best, feasible=False, first_failure="scenario 0 [pd_v2]: dwell:dwell_stationary", reason_head="dwell"
    )
    with pytest.raises(ValueError, match="contradicts its selection group"):
        replace(manifest, entries=tuple(flipped if e.label == best.label else e for e in manifest.entries))
    with pytest.raises(ValueError, match="rank exceeds"):
        replace(
            manifest,
            source=replace(manifest.source, n_feasible=100),
            ablation=replace(manifest.ablation, study_candidates=100),
        )
    with pytest.raises(ValueError, match="candidates for the source study"):
        replace(manifest, ablation=replace(manifest.ablation, study_candidates=133))


def test_source_cross_checks_refuse_other_studies_and_moved_files(
    tmp_path: Path, protocol: RecoverySearchProtocol, provenance: ProvenanceRecord
) -> None:
    """Formulation, study name, dataset, candidate values, ranks, and file identities are all cross-checked."""
    report = _report(protocol, provenance, _trials())
    ablation = _ablation(report, provenance)
    with pytest.raises(ValueError, match="no_augmentation"):
        resolve_panel(replace(report, formulation="contractive"), protocol, ablation)
    with pytest.raises(ValueError, match="is not the report's study"):
        resolve_panel(report, replace(protocol, name="other-study"), ablation)
    with pytest.raises(ValueError, match="evaluates dataset"):
        resolve_panel(report, protocol, replace(ablation, dataset="processed-other"))
    tampered = tuple(replace(c, value=c.value + 1.0) if c.number == 17 else c for c in ablation.candidates)
    with pytest.raises(ValueError, match="contradicts the stored trial"):
        resolve_panel(report, protocol, replace(ablation, candidates=tampered))
    deep = SelectionRule(("a",), "r", {"a": 999}, "f", {}, {"a": 17})
    with pytest.raises(ValueError, match="exceeds the 134"):
        resolve_panel(report, protocol, ablation, rule=deep)
    pointer, pointer_file, ablation_file, dataset = _sources(tmp_path, report)
    entries = resolve_panel(report, protocol, ablation)
    with pytest.raises(ValueError, match="is not the report's"):
        build_panel_manifest(
            entries,
            report=report,
            pointer=pointer,
            pointer_file=pointer_file,
            protocol=protocol,
            protocol_file=tmp_path / "elsewhere.toml",
            ablation=ablation,
            ablation_file=ablation_file,
            dataset=dataset,
            dataset_file=REPO_ROOT / DATASET_FILE,
            provenance=provenance,
        )
    with pytest.raises(ValueError, match="is not the study's dataset"):
        build_panel_manifest(
            entries,
            report=replace(report, dataset="processed-other"),
            pointer=replace(pointer, dataset="processed-other"),
            pointer_file=pointer_file,
            protocol=protocol,
            protocol_file=REPO_ROOT / PROTOCOL_FILE,
            ablation=ablation,
            ablation_file=ablation_file,
            dataset=dataset,
            dataset_file=REPO_ROOT / DATASET_FILE,
            provenance=provenance,
        )

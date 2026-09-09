# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3REP-001: the committed panel manifest binds its sources and rebuilds exactly from the study payload."""

from __future__ import annotations

import pytest

from arm_rc_ctrl.data.recovery import RecoveryDatasetRecord, load_processed_record
from arm_rc_ctrl.experiments.baselines import frozen_baseline_digest
from arm_rc_ctrl.experiments.evidence import load_report_pointer, open_stored_report
from arm_rc_ctrl.experiments.recovery_ablation import load_ablation
from arm_rc_ctrl.experiments.recovery_search import load_recovery_search
from arm_rc_ctrl.experiments.repetition_panel import (
    APPROVED_RULE,
    EXPERIMENT_LABEL,
    SOURCE_ABLATION_FILE,
    SOURCE_POINTER_FILE,
    build_panel_manifest,
    load_panel,
    render_panel_markdown,
    resolve_panel,
)
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageAccessError, open_storage

pytestmark = pytest.mark.regression

REPO_ROOT = repository_root()
RECOVERY_DOCS = REPO_ROOT / "docs" / "experiments" / "task_1a_state_conditioned_recovery"
DOCS = REPO_ROOT / "docs" / "experiments" / "task_1a_repeated_demonstration"
MANIFEST = DOCS / "panel_manifest_v1.json"
MARKDOWN = DOCS / "panel_manifest_v1.md"
DATASET = REPO_ROOT / "data" / "records" / "processed" / "processed-20260903-ce343c8ce6a5.toml"
APPROVED_WARMUPS = {
    "feasible-best": 0.25,
    "feasible-middle": 0.0,
    "feasible-worst": 1.0,
    "failure-actual-dwell": 0.25,
    "failure-joint-velocity": 1.0,
    "failure-generated-dwell": 0.0,
}


def test_panel_manifest_binds_the_committed_sources() -> None:
    """The strict load re-derives the approved identities, and every recorded digest matches its committed file."""
    manifest = load_panel(MANIFEST)
    assert manifest.experiment == EXPERIMENT_LABEL
    assert manifest.rule == APPROVED_RULE
    assert {e.label: e.source_trial for e in manifest.entries} == APPROVED_RULE.approved_trials
    assert {e.label: e.warmup_s for e in manifest.entries} == APPROVED_WARMUPS
    pointer = load_report_pointer(RECOVERY_DOCS / SOURCE_POINTER_FILE)
    assert manifest.source.pointer_file == SOURCE_POINTER_FILE
    assert manifest.source.pointer_sha256 == sha256_file(RECOVERY_DOCS / SOURCE_POINTER_FILE)
    assert manifest.source.payload == pointer.payload
    assert manifest.source.study == pointer.study
    assert manifest.source.protocol_sha256 == pointer.protocol_sha256
    assert manifest.source.trials_stored == pointer.trials_stored == 500
    assert manifest.source.n_feasible == pointer.n_feasible == 134
    assert manifest.configs.dataset == pointer.dataset
    assert manifest.ablation.file == SOURCE_ABLATION_FILE
    assert manifest.ablation.sha256 == sha256_file(RECOVERY_DOCS / SOURCE_ABLATION_FILE)
    assert manifest.ablation.study_candidates == 134
    for file, digest in (
        (manifest.source.protocol_file, manifest.source.protocol_file_sha256),
        (manifest.configs.model_file, manifest.configs.model_sha256),
        (manifest.configs.scenario_file, manifest.configs.scenario_sha256),
        (manifest.configs.development_file, manifest.configs.development_sha256),
        (manifest.configs.dataset_record_file, manifest.configs.dataset_record_sha256),
    ):
        assert sha256_file(REPO_ROOT / file) == digest, file
    for name, digest in manifest.configs.trackers.items():
        assert frozen_baseline_digest(name) == digest
    dataset = load_processed_record(DATASET)
    assert isinstance(dataset, RecoveryDatasetRecord)
    assert manifest.configs.dataset_payload_sha256 == dataset.artifact.payload.sha256
    # Clarification C1: the historical study revision and the implementation revision are distinct records.
    assert manifest.source_provenance.project_commit != manifest.provenance.project_commit
    assert not manifest.provenance.project_dirty
    assert not manifest.provenance.exploratory
    assert manifest.entry("failure-joint-velocity").comparison_label == "anchor-v4-tw1"
    assert render_panel_markdown(manifest) == MARKDOWN.read_text(encoding="utf-8")


def test_committed_panel_rebuilds_exactly() -> None:
    """The digest-verified study payload, ablation, and configs reproduce the committed manifest byte for byte."""
    committed = load_panel(MANIFEST)
    try:
        store = open_storage()
    except Exception as exc:  # noqa: BLE001 - any setup failure just means no store on this runner
        pytest.skip(f"external storage unavailable: {exc}")
    pointer = load_report_pointer(RECOVERY_DOCS / SOURCE_POINTER_FILE)
    try:
        report = open_stored_report(store, pointer)
    except StorageAccessError as exc:
        pytest.skip(f"external payload unavailable: {exc}")
    protocol = load_recovery_search(REPO_ROOT / report.protocol_file)
    ablation = load_ablation(RECOVERY_DOCS / SOURCE_ABLATION_FILE)
    dataset = load_processed_record(DATASET)
    assert isinstance(dataset, RecoveryDatasetRecord)
    entries = resolve_panel(report, protocol, ablation)
    rebuilt = build_panel_manifest(
        entries,
        report=report,
        pointer=pointer,
        pointer_file=RECOVERY_DOCS / SOURCE_POINTER_FILE,
        protocol=protocol,
        protocol_file=REPO_ROOT / report.protocol_file,
        ablation=ablation,
        ablation_file=RECOVERY_DOCS / SOURCE_ABLATION_FILE,
        dataset=dataset,
        dataset_file=DATASET,
        provenance=committed.provenance,
    )
    assert rebuilt == committed
    assert render_panel_markdown(rebuilt) == MARKDOWN.read_text(encoding="utf-8")

# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-016 finding 1: study-report payloads live externally behind verified, content-addressed pointers."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from arm_rc_ctrl.experiments.evidence import (
    load_report_pointer,
    open_stored_report,
    report_pointer,
    store_report_payload,
    write_report_pointer,
)
from arm_rc_ctrl.experiments.recovery_study import RecoveryStudyReport, report_to_json
from arm_rc_ctrl.experiments.studies import StudySummary, TrialRecord
from arm_rc_ctrl.provenance import ArtifactMismatchError, collect_provenance
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import BUCKETS, StorageRoot

REPO_ROOT = repository_root()


def _report() -> RecoveryStudyReport:
    trials = (TrialRecord(number=0, state="COMPLETE", value=10.0, params={"warmup_s": 0.0}, flags={"feasible": False}),)
    summary = StudySummary(
        name="pointer-fixture",
        storage="armrc://optuna/pointer-fixture.db",
        direction="minimize",
        identity={},
        trials=trials,
        n_complete=1,
        n_pruned=0,
        best_number=None,
        best_value=None,
        selection_rule="feasible",
    )
    return RecoveryStudyReport(
        protocol="pointer-fixture",
        protocol_file="configs/studies/pointer_fixture.toml",
        protocol_sha256="c" * 64,
        formulation="no_augmentation",
        dataset="processed-test",
        trackers={"pd_v2": "a" * 64, "computed_torque": "b" * 64},
        budget=1,
        trials_run=1,
        summary=summary,
        best_point=None,
        n_feasible=0,
        provenance=collect_provenance({}, seeds={}, artifacts=[], exploratory=True),
    )


def _store(tmp_path: Path) -> StorageRoot:
    root = tmp_path / "store"
    root.mkdir()
    return StorageRoot(root, repositories=(REPO_ROOT,))


def test_reports_bucket_exists() -> None:
    """The external reports bucket is a first-class storage location."""
    assert "reports" in BUCKETS


def test_payload_pointer_roundtrip_is_content_addressed(tmp_path: Path) -> None:
    """Store, point, write, load, and open reproduce the report; identical payloads are reused."""
    store = _store(tmp_path)
    report = _report()
    text = report_to_json(report) + "\n"
    payload = store_report_payload(store, text, name="pointer_fixture_v1")
    assert payload.sha256[:12] in payload.uri
    assert payload.uri.startswith("armrc://reports/")
    assert payload.size == len(text.encode("utf-8"))
    again = store_report_payload(store, text, name="pointer_fixture_v1")
    assert again == payload
    pointer = report_pointer(report, payload)
    file = tmp_path / "pointer_fixture_v1.toml"
    write_report_pointer(file, pointer)
    with pytest.raises(FileExistsError, match="immutable"):
        write_report_pointer(file, pointer)
    loaded = load_report_pointer(file)
    assert loaded == pointer
    assert open_stored_report(store, loaded) == report
    target = store.path(payload.uri, mode="read")
    assert not list(target.parent.glob("*.staging-*"))  # staging files never survive


def test_tampered_payloads_and_pointers_are_refused(tmp_path: Path) -> None:
    """A modified payload fails digest verification; a drifted pointer field fails the cross-check."""
    store = _store(tmp_path)
    report = _report()
    payload = store_report_payload(store, report_to_json(report) + "\n", name="pointer_fixture_v1")
    pointer = report_pointer(report, payload)
    target = store.path(payload.uri, mode="read")
    target.write_text(target.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ArtifactMismatchError, match=r"size|sha256"):
        open_stored_report(store, pointer)
    target.write_text(report_to_json(report) + "\n", encoding="utf-8")
    from dataclasses import replace

    drifted = replace(pointer, n_feasible=1, trials_stored=1)
    with pytest.raises(ValueError, match="contradicts"):
        open_stored_report(store, drifted)


def test_a_racing_identical_payload_is_reused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A file that appears between the existence check and installation survives when identical."""
    store = _store(tmp_path)
    report = _report()
    text = report_to_json(report) + "\n"
    real_link = os.link
    raced: list[str] = []

    def racing_link(src: str, dst: str) -> None:
        Path(dst).write_bytes(text.encode("utf-8"))  # the racer installs the identical payload first
        raced.append(dst)
        real_link(src, dst)

    monkeypatch.setattr("os.link", racing_link)
    payload = store_report_payload(store, text, name="pointer_fixture_v1")
    assert raced  # the race actually happened
    target = store.path(payload.uri, mode="read")
    assert target.read_text(encoding="utf-8") == text
    assert not list(target.parent.glob("*.staging-*"))


def test_a_racing_divergent_payload_is_never_clobbered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A different file that races in is preserved byte for byte and the installation fails."""
    store = _store(tmp_path)
    report = _report()
    text = report_to_json(report) + "\n"
    racer = b"racer content that must survive"
    real_link = os.link

    def racing_link(src: str, dst: str) -> None:
        Path(dst).write_bytes(racer)
        real_link(src, dst)

    monkeypatch.setattr("os.link", racing_link)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        store_report_payload(store, text, name="pointer_fixture_v1")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    target = store.root / "reports" / "task_1a_state_conditioned_recovery" / f"pointer_fixture_v1-{digest[:12]}.json"
    assert target.read_bytes() == racer  # the racer's file is untouched
    assert not list(target.parent.glob("*.staging-*"))


def test_an_existing_divergent_payload_is_refused(tmp_path: Path) -> None:
    """A pre-existing different file at the content-addressed path fails cleanly and is preserved."""
    store = _store(tmp_path)
    report = _report()
    text = report_to_json(report) + "\n"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    target_dir = store.root / "reports" / "task_1a_state_conditioned_recovery"
    target_dir.mkdir(parents=True)
    target = target_dir / f"pointer_fixture_v1-{digest[:12]}.json"
    target.write_bytes(b"already here")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        store_report_payload(store, text, name="pointer_fixture_v1")
    assert target.read_bytes() == b"already here"
    assert not list(target_dir.glob("*.staging-*"))

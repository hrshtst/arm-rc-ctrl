# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-014/M3R-016: the residual study pointer binds its config and records the exploratory negative."""

from __future__ import annotations

import pytest

from arm_rc_ctrl.experiments.evidence import StoredReport, load_report_pointer, open_stored_report
from arm_rc_ctrl.experiments.recovery_search import load_recovery_search, recovery_protocol_digest
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageAccessError, open_storage

pytestmark = pytest.mark.regression

REPO_ROOT = repository_root()
DOCS = REPO_ROOT / "docs" / "experiments" / "task_1a_state_conditioned_recovery"
CONFIG = REPO_ROOT / "configs" / "studies" / "recovery_search_1a_residual_v1.toml"


def _pointer() -> StoredReport:
    return load_report_pointer(DOCS / "residual_search_1a_v1.toml")


def test_residual_pointer_is_bound_and_negative() -> None:
    """The pointer matches the committed protocol and records the D4 exploratory negative."""
    pointer = _pointer()
    protocol = load_recovery_search(CONFIG)
    assert pointer.formulation == "residual"
    assert pointer.study == protocol.name
    assert pointer.protocol_sha256 == recovery_protocol_digest(protocol)
    assert pointer.dataset == "processed-20260903-ce343c8ce6a5"
    assert pointer.budget == 500
    assert pointer.trials_stored == 500
    # D4 disposition: retained as an exploratory negative; never predeclared for confirmatory
    # inclusion, and the seed panel does not apply without a feasible trial.
    assert pointer.n_feasible == 0
    assert pointer.best_number is None
    assert pointer.best_value is None


def test_large_report_payloads_never_sit_in_git() -> None:
    """Git carries pointers and markdown only; the full per-trial reports live in the external store."""
    assert not list(DOCS.glob("recovery_search_*_v1.json"))
    assert not (DOCS / "residual_search_1a_v1.json").exists()
    assert len(list(DOCS.glob("recovery_search_*_v1.toml"))) == 3


def test_residual_payload_matches_its_pointer_when_available() -> None:
    """With the external store present, the full report verifies against and re-derives the pointer."""
    pointer = _pointer()
    try:
        store = open_storage()
    except Exception as exc:  # noqa: BLE001 - any setup failure just means no store on this runner
        pytest.skip(f"external storage unavailable: {exc}")
    try:
        report = open_stored_report(store, pointer)
    except StorageAccessError as exc:
        pytest.skip(f"external payload unavailable: {exc}")
    assert report.formulation == "residual"
    assert report.n_feasible == 0
    assert report.summary.best_number is None


def test_residual_markdown_records_the_failure_taxonomy() -> None:
    """The rendered study report keeps the dominant joint-velocity failure mode visible."""
    text = (DOCS / "residual_search_1a_v1.md").read_text(encoding="utf-8")
    assert text.startswith("# Recovery search `recovery-search-1a-residual-v1`")
    assert "limit_violation:joint_velocity" in text

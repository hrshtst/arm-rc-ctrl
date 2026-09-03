# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-014: the committed residual study evidence binds its config and records the exploratory negative."""

from __future__ import annotations

import pytest

from arm_rc_ctrl.experiments.recovery_search import load_recovery_search, recovery_protocol_digest
from arm_rc_ctrl.experiments.recovery_study import load_report
from arm_rc_ctrl.repo import repository_root

pytestmark = pytest.mark.regression

REPO_ROOT = repository_root()
DOCS = REPO_ROOT / "docs" / "experiments" / "task_1a_state_conditioned_recovery"
CONFIG = REPO_ROOT / "configs" / "studies" / "recovery_search_1a_residual_v1.toml"


def test_residual_study_evidence_is_bound_and_negative() -> None:
    """The report re-derives, matches the committed protocol, and records the D4 exploratory negative."""
    report = load_report(DOCS / "residual_search_1a_v1.json")
    protocol = load_recovery_search(CONFIG)
    assert report.formulation == "residual"
    assert report.protocol == protocol.name
    assert report.protocol_sha256 == recovery_protocol_digest(protocol)
    assert report.dataset == "processed-20260903-ce343c8ce6a5"
    assert report.budget == 500
    assert len(report.summary.trials) == 500
    # D4 disposition: retained as an exploratory negative; never predeclared for confirmatory
    # inclusion, and the seed panel does not apply without a feasible trial.
    assert report.n_feasible == 0
    assert report.best_point is None
    assert report.summary.best_number is None
    # The development ablation compares exactly the three absolute arms; the residual evidence
    # deliberately lives under its own name so the recovery_search_*_v1 glob stays three-wide.
    assert len(sorted(DOCS.glob("recovery_search_*_v1.json"))) == 3


def test_residual_markdown_records_the_failure_taxonomy() -> None:
    """The rendered study report keeps the dominant joint-velocity failure mode visible."""
    text = (DOCS / "residual_search_1a_v1.md").read_text(encoding="utf-8")
    assert text.startswith("# Recovery search `recovery-search-1a-residual-v1`")
    assert "limit_violation:joint_velocity" in text

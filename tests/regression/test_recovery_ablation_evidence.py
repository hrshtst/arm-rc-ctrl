# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-013: the committed ablation evidence re-derives from the committed study reports."""

from __future__ import annotations

import pytest

from arm_rc_ctrl.experiments.recovery_ablation import load_ablation
from arm_rc_ctrl.experiments.recovery_study import load_report
from arm_rc_ctrl.repo import repository_root

pytestmark = pytest.mark.regression

DOCS = repository_root() / "docs" / "experiments" / "task_1a_state_conditioned_recovery"


def test_ablation_matches_the_committed_studies() -> None:
    """Arms bind the study digests and counts; candidates cover every feasible trial; the negative is recorded."""
    ablation = load_ablation(DOCS / "development_ablation_v1.json")  # strict load re-derives its counts
    studies = {f.name: load_report(f) for f in sorted(DOCS.glob("recovery_search_*_v1.json"))}
    assert len(studies) == 3
    by_file = {arm.file: arm for arm in ablation.arms}
    assert set(by_file) == set(studies)
    for name, report in studies.items():
        arm = by_file[name]
        assert arm.study == report.protocol
        assert arm.formulation == report.formulation
        assert arm.protocol_sha256 == report.protocol_sha256
        assert arm.n_feasible == report.n_feasible
        assert arm.budget == report.budget
        assert arm.trials_stored == len(report.summary.trials)
    assert len(ablation.candidates) == sum(r.n_feasible for r in studies.values())
    assert ablation.n_eligible == 0  # the recorded negative: small-posture cells miss the 15-of-20 rule
    assert ablation.dataset == "processed-20260903-ce343c8ce6a5"


def test_ablation_markdown_carries_the_required_sections() -> None:
    """The rendered report keeps the acceptance-required comparisons and limitations."""
    text = (DOCS / "development_ablation_v1.md").read_text(encoding="utf-8")
    for required in (
        "## Arms",
        "## Failure taxonomy",
        "## Timing-only arm",
        "## Eligible candidates (section 7.3)",
        "Synthetic-sample-count confound",
        "First-infeasible censoring",
    ):
        assert required in text

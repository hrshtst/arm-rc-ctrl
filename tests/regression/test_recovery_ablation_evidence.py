# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-013/M3R-016: the committed ablation re-derives from the committed study pointers."""

from __future__ import annotations

import pytest

from arm_rc_ctrl.experiments.evidence import load_report_pointer
from arm_rc_ctrl.experiments.recovery_ablation import load_ablation
from arm_rc_ctrl.repo import repository_root

pytestmark = pytest.mark.regression

DOCS = repository_root() / "docs" / "experiments" / "task_1a_state_conditioned_recovery"


def test_ablation_matches_the_committed_pointers() -> None:
    """Arms bind the pointer digests and counts; candidates cover every feasible trial; the negative holds."""
    ablation = load_ablation(DOCS / "development_ablation_v2.json")  # strict load re-derives its counts
    pointers = {f.name: load_report_pointer(f) for f in sorted(DOCS.glob("recovery_search_*_v1.toml"))}
    assert len(pointers) == 3
    by_file = {arm.file: arm for arm in ablation.arms}
    assert set(by_file) == set(pointers)
    for name, pointer in pointers.items():
        arm = by_file[name]
        assert arm.study == pointer.study
        assert arm.formulation == pointer.formulation
        assert arm.protocol_sha256 == pointer.protocol_sha256
        assert arm.n_feasible == pointer.n_feasible
        assert arm.budget == pointer.budget
        assert arm.trials_stored == pointer.trials_stored
        if arm.formulation == "no_augmentation":
            assert arm.d1_sampled is None
        else:
            assert arm.d1_sampled is not None
            assert arm.d1xd2_sampled is not None
    assert len(ablation.candidates) == sum(p.n_feasible for p in pointers.values()) == 134
    assert ablation.n_eligible == 0  # the recorded negative: small-posture cells miss the 15-of-20 rule
    assert ablation.dataset == "processed-20260903-ce343c8ce6a5"


def test_ablation_markdown_carries_the_required_sections() -> None:
    """The rendered report keeps the comparisons, coverage, and acceptance-required limitations."""
    text = (DOCS / "development_ablation_v2.md").read_text(encoding="utf-8")
    for required in (
        "## Arms",
        "## Failure taxonomy",
        "## Timing-only arm",
        "## Eligible candidates (section 7.3)",
        "Synthetic-sample-count confound",
        "First-infeasible censoring",
        "Flat infeasible objective",
        "Sampled, not exhaustive",
        "No feasible model was found among the 500 sampled trials",
        "D1 / D1 x T_w sampled",
    ):
        assert required in text

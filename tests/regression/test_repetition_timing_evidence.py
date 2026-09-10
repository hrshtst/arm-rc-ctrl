# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3REP-005: the committed timing smoke check covers feasible-best across every behavioral arm, pinned."""

from __future__ import annotations

import pytest

from arm_rc_ctrl.execution import load_execution
from arm_rc_ctrl.experiments.repetition_evaluation import load_evaluation_config, load_pointer, pointer_name
from arm_rc_ctrl.experiments.repetition_recipes import panel_arms
from arm_rc_ctrl.experiments.repetition_timing import load_timing, render_timing_markdown
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.repo import repository_root

pytestmark = pytest.mark.regression

REPO_ROOT = repository_root()
DOCS = REPO_ROOT / "docs" / "experiments" / "task_1a_repeated_demonstration"
EVIDENCE = DOCS / "timing_smoke_check_v1.json"
MARKDOWN = DOCS / "timing_smoke_check_v1.md"
POINTERS = DOCS / "evidence"
MANIFEST = DOCS / "panel_manifest_v1.json"
EVALUATION = REPO_ROOT / "configs" / "evaluations" / "task_1a_repetition_dev_v1.toml"
EXECUTION = DOCS / "execution_environment_v1.json"


def test_smoke_check_measured_every_behavioral_arm_of_trial_17() -> None:
    """Twenty models over 130 pairs each, one replay bank, in the canonical environment, with the projection."""
    report = load_timing(EVIDENCE)
    behavioral = [arm for arm in panel_arms() if arm.behavioral]
    assert report.panel_label == "feasible-best"
    assert report.panel_manifest_sha256 == sha256_file(MANIFEST)
    assert report.evaluation_sha256 == sha256_file(EVALUATION)
    assert report.execution.identity == load_execution(EXECUTION).identity
    assert report.execution.canonical
    assert not report.provenance.project_dirty
    assert not report.provenance.exploratory
    assert [m.label for m in report.models] == [f"feasible-best/{arm.label}" for arm in behavioral]
    assert len(report.models) == 20
    assert report.replay_bank_runs == 130
    assert all(m.runs + m.unexecuted <= 130 for m in report.models)
    assert all(m.fit_seconds is not None and m.fit_seconds >= 0 for m in report.models)
    assert report.wall_seconds > 0
    assert report.runs_this_invocation == len(report.runs)  # one uninterrupted invocation measured everything
    assert len(report.runs) == report.replay_bank_runs + sum(m.runs for m in report.models)
    assert report.peak_rss_bytes > 0
    assert report.storage_bytes > 0
    assert report.projection.entries == 6
    assert report.projection.models_per_entry == 20
    assert report.projection.pairs_per_model == 130
    assert report.projection.replay_banks == 3
    assert report.projection.completed_models == 20
    assert report.projection.total_seconds >= report.projection.remaining_seconds > 0
    assert "M3REP-006 waits for the owner's budget approval" in report.revised_estimate
    assert render_timing_markdown(report) == MARKDOWN.read_text(encoding="utf-8")
    assert load_evaluation_config(EVALUATION).simulation.velocity_abort == (12.0, 12.0)


def test_smoke_check_pointers_are_committed_for_every_model_and_the_replay_bank() -> None:
    """One Git pointer per model configuration and one for the 0.25 s replay bank, consistent with the report."""
    report = load_timing(EVIDENCE)
    for model in report.models:
        pointer = load_pointer(POINTERS / pointer_name("model", model.label))
        assert pointer.kind == "model"
        assert pointer.status == model.status
        assert pointer.n_completed + pointer.n_unexecuted <= pointer.n_pairs == 130
        assert pointer.n_unexecuted == model.unexecuted
    bank = load_pointer(POINTERS / pointer_name("replay", "warmup-0.25s"))
    assert bank.kind == "replay"
    assert bank.n_pairs == 130
    assert bank.status == "complete"

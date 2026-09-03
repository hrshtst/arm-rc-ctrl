# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-015: the committed freeze record re-derives, binds its inputs, and keeps the negative explicit."""

from __future__ import annotations

import pytest

from arm_rc_ctrl.experiments.recovery_freeze import load_freeze
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.repo import repository_root

pytestmark = pytest.mark.regression

DOCS = repository_root() / "docs" / "experiments" / "task_1a_state_conditioned_recovery"


def test_freeze_record_binds_the_committed_evidence() -> None:
    """The record's digests and counts match the committed ablation and study reports byte for byte."""
    record = load_freeze(DOCS / "model_freeze_v1.json")  # strict load re-derives the outcome invariants
    assert record.outcome == "negative"
    assert record.selection is None
    assert record.panel is None
    assert record.n_candidates == 134
    assert record.n_eligible == 0
    assert record.eligible_trials == ()
    assert record.dataset == "processed-20260903-ce343c8ce6a5"
    assert record.ablation_sha256 == sha256_file(DOCS / record.ablation_file)
    by_formulation = {s.formulation: s for s in record.studies}
    assert set(by_formulation) == {"no_augmentation", "non_decaying", "contractive", "residual"}
    assert by_formulation["no_augmentation"].n_feasible == 134
    residual = by_formulation["residual"]
    assert not residual.included
    assert residual.note is not None
    assert "D4" in residual.note
    for study in record.studies:
        assert (DOCS / study.file).exists()


def test_freeze_markdown_states_the_closed_confirmatory_gate() -> None:
    """The rendered record keeps the negative outcome and its consequence explicit for the review."""
    text = (DOCS / "model_freeze_v1.md").read_text(encoding="utf-8")
    assert "NEGATIVE RESULT" in text
    assert "confirmatory gate stays closed" in text
    assert "reservoir-seed-panel stability precedes any freeze" in text

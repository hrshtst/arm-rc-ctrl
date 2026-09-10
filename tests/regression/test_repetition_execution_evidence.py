# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3REP-006: the committed accounting covers all 120 configurations, three replay banks, and 36 references."""

from __future__ import annotations

import pytest

from arm_rc_ctrl.execution import load_execution
from arm_rc_ctrl.experiments.repetition_accounting import load_accounting, render_accounting_markdown
from arm_rc_ctrl.experiments.repetition_evaluation import load_pointer, pointer_name
from arm_rc_ctrl.experiments.repetition_panel import APPROVED_RULE
from arm_rc_ctrl.experiments.repetition_recipes import panel_arms
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.repo import repository_root

pytestmark = pytest.mark.regression

REPO_ROOT = repository_root()
DOCS = REPO_ROOT / "docs" / "experiments" / "task_1a_repeated_demonstration"
EVIDENCE = DOCS / "pilot_execution_v1.json"
MARKDOWN = DOCS / "pilot_execution_v1.md"
POINTERS = DOCS / "evidence"
MANIFEST = DOCS / "panel_manifest_v1.json"
VALIDATION = DOCS / "numerical_validation_v1.json"
EXECUTION = DOCS / "execution_environment_v1.json"


def test_accounting_is_complete_and_bound() -> None:
    """Every configuration has evidence in the canonical environment with the C11 caveat; nothing is missing."""
    a = load_accounting(EVIDENCE)
    assert a.complete
    assert a.n_configurations == 120
    assert a.n_present == 120
    assert a.n_missing == 0
    assert a.missing == ()
    assert a.numerical_reference_fits == 36
    assert len(a.banks) == 3
    assert sorted(b.warmup_s for b in a.banks) == [0.0, 0.25, 1.0]
    assert all(b.n_pairs == 130 for b in a.banks)
    assert a.n_replay_runs == 390
    assert a.n_rc_runs <= 15600
    assert a.all_bind_canonical_execution
    assert a.all_carry_c11
    assert a.canonical_execution_identity == load_execution(EXECUTION).identity
    assert a.panel_manifest_sha256 == sha256_file(MANIFEST)
    assert a.numerical_validation_sha256 == sha256_file(VALIDATION)
    assert not a.provenance.project_dirty
    assert not a.provenance.exploratory
    behavioral = [arm for arm in panel_arms() if arm.behavioral]
    assert [(m.panel_label, m.arm) for m in a.models] == [
        (label, arm.label) for label in APPROVED_RULE.labels for arm in behavioral
    ]
    assert sum(a.statuses.values()) == 120
    for m in a.models:
        assert m.present
        assert m.n_completed + m.n_infeasible + m.n_replay_blocked + m.n_unexecuted == 130
        if m.status == "feasible":
            assert m.n_completed == 130
            assert set(m.cells) == {
                "posture_small:pd_v2",
                "posture_small:computed_torque",
                "posture_large:pd_v2",
                "posture_large:computed_torque",
            }
        else:
            assert m.first_failure is not None
            assert m.cells == {}
    assert render_accounting_markdown(a) == MARKDOWN.read_text(encoding="utf-8")


def test_every_configuration_and_bank_has_its_pointer() -> None:
    """One Git pointer per model configuration and per replay bank, consistent with the accounting."""
    a = load_accounting(EVIDENCE)
    for m in a.models:
        pointer = load_pointer(POINTERS / pointer_name("model", f"{m.panel_label}/{m.arm}"))
        assert pointer.identity == m.evaluation_identity
        assert pointer.status == m.status
        assert pointer.payload == m.payload
        assert pointer.n_unexecuted == m.n_unexecuted
    for b in a.banks:
        pointer = load_pointer(POINTERS / pointer_name("replay", f"warmup-{b.warmup_s:g}s"))
        assert pointer.identity == b.identity
        assert pointer.payload == b.payload
    assert len(list(POINTERS.glob("model__*.toml"))) == 120
    assert len(list(POINTERS.glob("replay__*.toml"))) == 3

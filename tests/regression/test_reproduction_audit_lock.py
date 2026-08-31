# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3-013: the committed reproduction audit passed every step on the committed evidence without machine paths."""

from __future__ import annotations

import json
import re

import pytest

from arm_rc_ctrl.config import from_mapping
from arm_rc_ctrl.experiments.reproduce_1a import STEPS, Reproduction, audit_markdown
from arm_rc_ctrl.experiments.robustness import load_suite
from arm_rc_ctrl.repo import repository_root

pytestmark = pytest.mark.regression

REPO_ROOT = repository_root()
DOCS = REPO_ROOT / "docs" / "experiments" / "task_1a"
SUMMARY = DOCS / "reproduction_audit.json"
NOTE = DOCS / "reproduction_audit.md"
CONFIRMATORY = DOCS / "robustness_confirmatory_v2_recipe_v4.json"
AUDITOR = "OpenAI Codex (independent same-host rerun at `3ecb548`)"


def test_audit_passed_every_step_exactly_on_the_committed_evidence() -> None:
    """All steps ran and passed, the rerun deviated by zero, and the inputs are the confirmatory evidence's."""
    raw = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert raw.pop("ok") is True
    result = from_mapping(raw, Reproduction)
    assert [c.name for c in result.checks] == list(STEPS)
    assert all(c.ok for c in result.checks)
    assert result.max_deviation == 0.0
    suite = load_suite(CONFIRMATORY)
    assert result.inputs["evidence_project_commit"] == suite.provenance.project_commit
    assert len(result.inputs["reproduction_project_commit"]) == 40
    assert result.inputs["reproduction_project_dirty"] == "False"
    assert result.inputs["lock_sha256"] == suite.provenance.lock_sha256
    assert result.inputs["recipe"] == suite.recipe_file
    assert result.inputs["confirmatory_report"] == CONFIRMATORY.name
    assert result.environment["OMP_NUM_THREADS"] == "1"
    assert "260 runs re-evaluated" in next(c.detail for c in result.checks if c.name == "evaluation")


def test_audit_note_is_rendered_from_the_summary_and_names_no_machine_path() -> None:
    """The committed note equals the rendering of the summary and carries no absolute path."""
    raw = json.loads(SUMMARY.read_text(encoding="utf-8"))
    raw.pop("ok")
    result = from_mapping(raw, Reproduction)
    note = NOTE.read_text(encoding="utf-8")
    command = re.search(r"- Command: `([^`]+)`", note)
    assert command is not None
    assert f"- Auditor: {AUDITOR}" in note
    assert note == audit_markdown(result, command=command.group(1), auditor=AUDITOR)
    assert not re.search(r"(?<![\w./])/(?:home|tmp|Users|mnt)/", note + SUMMARY.read_text(encoding="utf-8"))

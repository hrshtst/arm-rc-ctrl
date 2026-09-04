# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-020: the signed recovery audit matches its machine-readable executor record."""

from __future__ import annotations

import json
import re

import pytest

from arm_rc_ctrl.config import from_mapping
from arm_rc_ctrl.experiments.reproduce_recovery import (
    STEPS,
    RecoveryReproduction,
    audit_markdown,
)
from arm_rc_ctrl.repo import repository_root

pytestmark = pytest.mark.regression

DOCS = repository_root() / "docs" / "experiments" / "task_1a_state_conditioned_recovery"
SUMMARY = DOCS / "reproduction_audit_v1.json"
NOTE = DOCS / "reproduction_audit_v1.md"
AUDITOR = "OpenAI Codex (independent same-host rerun at `0517fd7`; 2026-09-04)"


def _result() -> RecoveryReproduction:
    raw = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert raw.pop("ok") is True
    return from_mapping(raw, RecoveryReproduction)


def test_executor_audit_passed_every_recovery_step_exactly() -> None:
    """The committed executor record covers the complete negative-path reproduction with zero deviation."""
    result = _result()
    assert [check.name for check in result.checks] == list(STEPS)
    assert all(check.ok for check in result.checks)
    assert result.ok
    assert result.max_deviation == 0.0
    assert result.inputs["dataset"] == "processed-20260903-ce343c8ce6a5"
    assert result.inputs["reproduction_project_commit"] == "7818c52db6e4b2e5fc887b8c2f9ffba4bc4a4471"
    assert result.inputs["reproduction_project_dirty"] == "False"
    assert result.environment["OMP_NUM_THREADS"] == "1"
    assert "130 pairs" in next(check.detail for check in result.checks if check.name == "model")
    assert "16 runs rerun" in next(check.detail for check in result.checks if check.name == "pairs")


def test_signed_note_is_rendered_from_the_executor_record_without_machine_paths() -> None:
    """The signature is the only human addition to the canonical audit rendering."""
    result = _result()
    note = NOTE.read_text(encoding="utf-8")
    command = re.search(r"- Command: `([^`]+)`", note)
    assert command is not None
    assert f"- Auditor: {AUDITOR}" in note
    assert note == audit_markdown(result, command=command.group(1), auditor=AUDITOR)
    assert not re.search(r"(?<![\w./])/(?:home|tmp|Users|mnt)/", note + SUMMARY.read_text(encoding="utf-8"))

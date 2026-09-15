# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3REP-008: the reproduction's verification steps run against the committed pilot evidence.

The fresh-process refits and the re-simulation are exercised by the audit
run (they need a clean checkout and minutes of wall time); the remaining
steps re-verify every committed record, payload, metric, table, and asset
here whenever the configured store and the canonical execution environment
are available.
"""

from __future__ import annotations

import importlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.dependencies import submodule_revisions
from arm_rc_ctrl.execution import ExecutionEnvironmentError, collect_execution, load_execution, require_canonical
from arm_rc_ctrl.experiments.repetition_report import load_report
from arm_rc_ctrl.experiments.reproduce_repetition import DOCS, STEPS, main
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageError, open_storage

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
_ENVIRONMENT_DEPENDENT = ("payloads", "recipes", "metrics", "tables", "assets")


def at_evidence_pins() -> bool:
    """Whether the checked-out submodules match the pilot evidence (required by the environment step)."""
    report = load_report(DOCS / "repetition_report_v1.json")
    recorded = {s.name: (s.checked_out or s.recorded) for s in report.provenance.submodules}
    current = {s.name: (s.checked_out or s.recorded) for s in submodule_revisions(REPO_ROOT)}
    return all(current.get(name) == revision for name, revision in recorded.items())


def _require_evidence_environment() -> None:
    try:
        open_storage()
    except (StorageError, FileNotFoundError, ValueError, RuntimeError) as exc:
        pytest.skip(f"configured external store not available: {exc}")
    try:
        require_canonical()
    except ExecutionEnvironmentError as exc:
        pytest.skip(f"not launched in the canonical execution environment: {exc}")
    for name in ("numpy", "rclib"):
        importlib.import_module(name)
    current = collect_execution(command="pytest", role="main", now=datetime.now(tz=UTC))
    canonical = load_execution(DOCS / "execution_environment_v1.json")
    if current.identity != canonical.identity:
        pytest.skip("the committed evidence binds another canonical execution environment")


def test_verification_steps_reproduce_the_committed_evidence(tmp_path: Path) -> None:
    """Records, payloads, recipes, metrics, tables, and assets all re-verify; the summary and audit are written.

    On a checkout whose submodule pins differ from the evidence (after a pin
    advance) the environment step must refuse, naming the submodules, and every
    step that depends on it must decline to run; the committed evidence is then
    reproduced from its audit checkout (``--from-checkout``), never from the new pins.
    """
    _require_evidence_environment()
    summary = tmp_path / "summary.json"
    audit = tmp_path / "audit.md"
    scratch = tmp_path / "scratch"
    status = main(
        [
            "--scratch",
            str(scratch),
            "--summary",
            str(summary),
            "--audit",
            str(audit),
            "--skip-fits",
            "--skip-resimulation",
            "--keep-going",
        ]
    )
    data = json.loads(summary.read_text(encoding="utf-8"))
    checks = data["checks"]
    assert [c["name"] for c in checks] == list(STEPS)  # keep_going runs every step
    details = {c["name"]: c["detail"] for c in checks}
    if not at_evidence_pins():
        assert status != 0
        assert data["ok"] is False
        outcomes = {c["name"]: c["ok"] for c in checks}
        assert not outcomes["environment"]
        assert "submodule pins differ from the committed evidence" in details["environment"]
        assert outcomes["storage"]  # independent of the environment
        assert outcomes["records"]
        for name in _ENVIRONMENT_DEPENDENT:
            assert not outcomes[name]
            assert "requires an earlier step that did not run" in details[name]
        assert details["fits"].startswith("skipped on request")
        assert details["resimulation"].startswith("skipped on request")
        assert audit.is_file()
        return
    assert status == 0
    assert data["ok"] is True
    assert all(c["ok"] for c in checks)
    assert details["fits"].startswith("skipped on request")
    assert details["resimulation"].startswith("skipped on request")
    assert "123 evidence pointers" in details["records"]
    assert "4790 run payloads" in details["payloads"]
    assert "156 cached fits" in details["payloads"]
    assert details["recipes"].startswith("156 prescribed fit identities")
    assert details["metrics"].startswith("390 replay runs and the runs of 4400 executed model pairs")
    assert "120 eligibility rows" in details["tables"]
    assert "34 figures and 3 animations regenerated byte-for-byte" in details["assets"]
    assert data["max_deviation"] == 0.0
    report = load_report(DOCS / "repetition_report_v1.json")
    assert data["inputs"]["evidence_project_commit"] == report.provenance.project_commit
    assert data["inputs"]["canonical_execution_identity"] == report.canonical_execution_identity
    assert data["inputs"]["c11_exception_candidate"].startswith("6912bffcc9a5")
    assert data["doc005"] == []
    assert data["gates"] == []
    text = audit.read_text(encoding="utf-8")
    assert text.startswith("# Task 1-a repeated-demonstration pilot reproduction")
    assert "## Declared comparisons" in text
    assert "## Historical task 1-a reproduction (DOC-005), reported apart" in text
    assert (scratch / "plots").is_dir()
    assert (scratch / "animations").is_dir()
    assert (scratch / "store").is_dir()

# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Reproduce all expert-report presentation bytes from the configured read-only store."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arm_rc_ctrl.experiments.manual_expert_report import build_report, verify_report
from arm_rc_ctrl.experiments.manual_report import DOCUMENT_LOCATION
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageError, open_storage


@pytest.mark.integration
def test_complete_expert_presentation_reproduces_from_stored_evidence(tmp_path: Path) -> None:
    """All figures, playback, metrics and prose reproduce; no simulation or store write is involved."""
    try:
        open_storage()
    except StorageError as exc:
        pytest.skip(f"configured evidence store unavailable: {exc}")
    docs = repository_root() / DOCUMENT_LOCATION
    expected = json.loads((docs / "expert_report/manifest.json").read_text())
    output = tmp_path / "report"
    build_report(output)
    verify_report(output)
    actual = json.loads((output / "manifest.json").read_text())
    assert actual["outputs"] == expected["outputs"]
    assert actual["inputs"] == expected["inputs"]
    assert actual["sources"] == expected["sources"]
    for name, record in expected["outputs"].items():
        assert sha256_file(output / name) == record["sha256"], name

# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only
"""Read-only reproduction of the complete expert report, with no simulation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arm_rc_ctrl.experiments.manual_search_report import DOCS, build_report, verify_report
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageError, open_storage


@pytest.mark.integration
def test_report_reproduces_every_plot_animation_and_prose_byte(tmp_path: Path) -> None:
    """Report reproduces every plot animation and prose byte."""
    try:
        open_storage()
    except StorageError as exc:
        pytest.skip(f"external evidence store unavailable: {exc}")
    expected = json.loads((repository_root() / DOCS / "report/manifest.json").read_text())
    output = tmp_path / "report"
    build_report(output)
    verify_report(output)
    actual = json.loads((output / "manifest.json").read_text())
    for key in ("outputs", "inputs", "sources", "cases"):
        assert actual[key] == expected[key], key

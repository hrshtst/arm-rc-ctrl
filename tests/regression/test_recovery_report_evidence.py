# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-018: the committed report binds its statements, embedded assets, and tracked runs."""

from __future__ import annotations

import re

import pytest

from arm_rc_ctrl.experiments.recovery_freeze import load_freeze
from arm_rc_ctrl.experiments.recovery_representative import load_representatives
from arm_rc_ctrl.repo import repository_root

pytestmark = pytest.mark.regression

REPO_ROOT = repository_root()
DOCS = REPO_ROOT / "docs" / "experiments" / "task_1a_state_conditioned_recovery"


def _report_text() -> str:
    return (DOCS / "recovery_report_v1.md").read_text(encoding="utf-8")


def test_report_states_the_accepted_negative() -> None:
    """The headline, gate closure, and shared limitations render from the committed evidence."""
    text = _report_text()
    freeze = load_freeze(DOCS / "model_freeze_v2.json")
    for required in (
        "**Accepted negative result.**",
        "no model is frozen and the",
        "not authorized under protocol v1",
        "Flat infeasible objective",
        "Sampled, not exhaustive",
        "15-of-20 consistency requirement",
        f"freeze commit `{freeze.provenance.project_commit[:12]}`",
        f"{freeze.n_candidates} feasible development trials",
    ):
        assert required in text


def test_every_embedded_asset_exists() -> None:
    """Every plot and animation the report embeds is a committed, non-empty file."""
    text = _report_text()
    links = re.findall(r"!\[[^]]*\]\(([^)]+)\)", text)
    assert len(links) >= 15  # 7 plots + 8 animations
    for link in links:
        target = DOCS / link
        assert target.exists(), link
        assert target.stat().st_size > 0, link


def test_representative_runs_are_tracked_and_cited() -> None:
    """Every representative run ID appears in the report and has a Git-tracked pointer record."""
    text = _report_text()
    record = load_representatives(DOCS / "recovery_representative_v1.json")
    for pair in record.pairs:
        for run_id in (pair.replay_run, pair.rc_run):
            assert (REPO_ROOT / "data" / "records" / "runs" / f"{run_id}.toml").exists(), run_id
        assert pair.rc_run in text
        assert pair.replay_run in text

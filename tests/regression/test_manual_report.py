# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-012: presentation uses audited counts without treating M10 as ten independent models."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from arm_rc_ctrl.experiments.manual_report import load_report_data, main, render_summaries
from arm_rc_ctrl.repo import repository_root

pytestmark = pytest.mark.regression
DOCS = repository_root() / "docs/experiments/task_1a_manual_demonstration"


def test_report_keeps_the_single_all_ten_model_and_every_parent() -> None:
    """The plotted M10 point is one model; each singleton contributes its own point."""
    data = load_report_data(DOCS)
    chosen = [r for r in data.arms if r.configuration == "feasible-middle" and r.tracker == "pd_v2"]
    all_ten = next(r for r in chosen if r.arm_kind == "M10" and r.scenario_class == "all")
    singletons = next(r for r in chosen if r.arm_kind == "S" and r.scenario_class == "all")
    assert all_ten.per_model == (14,)
    assert singletons.per_model == (59, 65, 65, 65, 65, 65, 65, 65, 65, 65)
    assert (all_ten.n_runs, singletons.n_runs) == (65, 650)


@pytest.mark.parametrize("name", ["arm_summary_v1.csv", "figure_inputs_v1.json"])
def test_report_refuses_a_summary_that_no_longer_matches_the_audit(tmp_path: Path, name: str) -> None:
    """A cosmetic or numerical edit to a source file requires a newly verified source."""
    docs = tmp_path / "experiment"
    shutil.copytree(DOCS, docs)
    path = docs / "results" / name
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        load_report_data(docs)


def test_both_summary_figures_render_from_the_audited_tables(tmp_path: Path) -> None:
    """Render the real 12 configuration/tracker cells without external payloads."""
    render_summaries(load_report_data(DOCS), tmp_path)
    for name in ("success_counts.png", "class_comparisons.png"):
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_the_presentation_command_records_inputs_and_refuses_to_overwrite(tmp_path: Path) -> None:
    """A reproducible render keeps its input bindings and preserves earlier exports."""
    out = tmp_path / "assets"
    assert main(["--docs", str(DOCS), "--output", str(out)]) == 0
    assert (out / "render_manifest.json").is_file()
    with pytest.raises(FileExistsError):
        main(["--docs", str(DOCS), "--output", str(out)])

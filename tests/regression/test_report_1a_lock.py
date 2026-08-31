# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""The task 1-a report renders from the committed evidence and is locked to the committed Markdown (M3-011)."""

from __future__ import annotations

from pathlib import Path

import pytest

from arm_rc_ctrl.experiments.report_1a import PLOT_FILES, load_inputs, main, render_report, write_plots
from arm_rc_ctrl.repo import repository_root

pytestmark = pytest.mark.regression

REPO_ROOT = repository_root()
DOCS = REPO_ROOT / "docs" / "experiments" / "task_1a"
REPORT = DOCS / "report.md"
PLOTS = DOCS / "plots"


def test_report_covers_metrics_distributions_failures_and_paired_comparisons() -> None:
    """Every section the plan asks for is present and derived from the confirmatory suite."""
    inputs = load_inputs(DOCS)
    text = render_report(inputs)
    for heading in (
        "## Summary",
        "## Primary metric",
        "### Joint trajectory RMSE over the movement window (rad)",
        "Per-joint RMSE medians",
        "## Secondary metrics",
        "### Dwell-window metrics",
        "### Effort over the whole run",
        "## Paired comparisons",
        "Distribution of the per-scenario joint RMSE difference",
        "## Failures",
        "## Development evidence (not confirmatory)",
        "### ESN searches",
        "### Reservoir-seed panel",
        "### Training reports",
        "### Development robustness suites",
        "## Limitations",
    ):
        assert heading in text, heading
    assert f"`{inputs.confirmatory_file}`" in text
    assert "| median | q25 | q75 | min | max |" in text
    assert "endpoint p95 (m)" in text
    assert inputs.confirmatory.recipe in text


def test_committed_report_is_the_rendered_evidence() -> None:
    """Regenerating from the committed JSON reproduces the committed report (figures referenced by name)."""
    inputs = load_inputs(DOCS)
    plots = [f"plots/{name}" for name in PLOT_FILES]
    assert all((PLOTS / name).is_file() for name in PLOT_FILES)
    assert render_report(inputs, plots=plots) == REPORT.read_text(encoding="utf-8")


def test_command_writes_report_and_figures(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The command renders into a fresh directory, refuses to overwrite, and regenerates with --force."""
    output = tmp_path / "out" / "report.md"
    argv = ["--docs", str(DOCS), "--output", str(output), "--plots", str(tmp_path / "out" / "plots")]
    assert main(argv) == 0
    assert "3 figures" in capsys.readouterr().out
    assert output.read_text(encoding="utf-8").endswith("![search_objectives](plots/search_objectives.png)\n")
    assert sorted(p.name for p in (tmp_path / "out" / "plots").glob("*.png")) == [
        "paired_differences.png",
        "rmse_by_class.png",
        "search_objectives.png",
    ]
    with pytest.raises(FileExistsError, match="refusing"):
        main(argv)
    assert main([*argv, "--force"]) == 0
    written = write_plots(load_inputs(DOCS), tmp_path / "again")
    assert all(p.stat().st_size > 1000 for p in written)

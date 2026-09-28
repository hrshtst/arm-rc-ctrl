# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only
"""Expert presentation binds evidence, denominators, and every produced byte."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from arm_rc_ctrl.experiments.manual_search_report import DOCS, load_data, summary_table, verify_report
from arm_rc_ctrl.experiments.manual_search_report_plots import frame_indices
from arm_rc_ctrl.repo import repository_root


def test_paired_table_is_generated_from_all_six_strata() -> None:
    """Paired table is generated from all six strata."""
    data = load_data(repository_root())
    table = summary_table(data)
    assert table.count('data-result="') == 6
    for row in data.contrasts:
        if row.scenario_class == "all" and row.contrast == "M10-S":
            assert f'data-result="{row.configuration}/{row.tracker}"' in table
            assert f"{row.parents_improved} / {row.parents_worsened} / {row.parents_tied}" in table
    assert "53.5" in table
    assert "\u221250.5" in table


def test_loader_refuses_a_summary_not_bound_by_the_audit(tmp_path: Path) -> None:
    """Loader refuses a summary not bound by the audit."""
    root = repository_root()
    shutil.copytree(root / DOCS, tmp_path / DOCS)
    path = tmp_path / DOCS / "results/arm_summary_v1.csv"
    path.write_bytes(path.read_bytes().replace(b"53.5", b"53.6"))
    with pytest.raises(ValueError, match="fingerprint"):
        load_data(tmp_path)


def test_sampling_preserves_abort_endpoint_and_force_events() -> None:
    """Sampling preserves abort endpoint and force events."""
    t = np.arange(-0.25, 1.084, 0.01, dtype=np.float64)
    ix = frame_indices(t, stride=10, events=(0, 0.53, 0.73))
    assert ix[0] == 0
    assert ix[-1] == len(t) - 1
    assert np.all(np.diff(ix) > 0)
    for value in (0, 0.53, 0.73):
        assert np.argmin(np.abs(t - value)) in ix
    with pytest.raises(ValueError, match="stride"):
        frame_indices(t, stride=0)


def test_presentation_inventory_and_source_are_bound(tmp_path: Path) -> None:
    """Presentation inventory and source are bound."""
    root = repository_root()
    output = root / DOCS / "report"
    verify_report(output, root=root)
    copied = tmp_path / "report"
    shutil.copytree(output, copied)
    asset = copied / "assets/success_counts.png"
    raw = asset.read_bytes()
    asset.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match="fingerprint"):
        verify_report(copied, root=root)
    asset.write_bytes(raw)
    (copied / "extra.txt").write_text("extra")
    with pytest.raises(ValueError, match="inventory"):
        verify_report(copied, root=root)


def test_committed_prose_contains_the_generated_table_and_posthoc_labels() -> None:
    """Committed prose contains the generated table and posthoc labels."""
    root = repository_root()
    data = load_data(root)
    output = root / DOCS / "report"
    html = (output / "index.html").read_text()
    assert summary_table(data) in html
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["project_dirty"] is False
    assert "post-hoc" in html
    assert "highest nominal scores" in html
    assert "separately optimized" in html
    assert "not newly drawn" in html
    assert len(manifest["cases"]) == 10


def test_source_inventory_cannot_be_shortened(tmp_path: Path) -> None:
    """Removing a binding does not turn verification into a weaker check."""
    root = repository_root()
    copied = tmp_path / "report"
    shutil.copytree(root / DOCS / "report", copied)
    path = copied / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["sources"].pop(next(iter(manifest["sources"])))
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inventory"):
        verify_report(copied, root=root)


def test_summary_figures_reproduce_without_the_store(tmp_path: Path) -> None:
    """Every summary plot is an exact rendering of its committed audited tables."""
    from arm_rc_ctrl.experiments.manual_search_report import render_summaries
    from arm_rc_ctrl.provenance import sha256_file

    root = repository_root()
    render_summaries(load_data(root), tmp_path)
    for path in tmp_path.glob("*.png"):
        assert sha256_file(path) == sha256_file(root / DOCS / "report/assets" / path.name)

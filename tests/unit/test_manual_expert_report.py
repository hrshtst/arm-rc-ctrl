# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Scientific presentation contracts for the standalone expert report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from arm_rc_ctrl.experiments.manual_expert_report import (
    frame_indices,
    save_indexed_plot,
    summary_table,
    verify_outputs,
    verify_report,
)
from arm_rc_ctrl.experiments.manual_report import DOCUMENT_LOCATION, load_report_data
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.repo import repository_root


def test_animation_sampling_keeps_activation_pulse_and_aborted_endpoint() -> None:
    """Display decimation cannot omit an event or manufacture a completed horizon."""
    t = np.arange(-0.25, 1.084, 0.01, dtype=np.float64)
    indices = frame_indices(t, stride=10, events=(0.0, 0.53, 0.73))
    assert indices[0] == 0
    assert indices[-1] == len(t) - 1
    assert np.all(np.diff(indices) > 0)
    for event in (0.0, 0.53, 0.73):
        assert int(np.argmin(np.abs(t - event))) in indices
    assert t[indices[-1]] < 1.09
    with pytest.raises(ValueError, match="stride"):
        frame_indices(t, stride=0)


def test_summary_table_uses_every_configuration_and_tracker() -> None:
    """The human-facing main table is generated from the audited paired summaries."""
    data = load_report_data(repository_root() / DOCUMENT_LOCATION)
    table = summary_table(data)
    assert table.count('data-result="') == 12
    assert 'data-result="feasible-middle/pd_v2"' in table
    assert "\u221251" in table
    assert "2 / 7 / 1" in table


def test_asset_verification_refuses_same_size_corruption_and_extra_files(tmp_path: Path) -> None:
    """An asset fingerprint and the complete output inventory are both checked."""
    asset = tmp_path / "a.js"
    asset.write_bytes(b"first")
    outputs = {"a.js": {"sha256": sha256_file(asset), "size": 5}}
    verify_outputs(tmp_path, outputs)
    asset.write_bytes(b"other")
    with pytest.raises(ValueError, match=r"a\.js"):
        verify_outputs(tmp_path, outputs)
    asset.write_bytes(b"first")
    (tmp_path / "extra.js").write_bytes(b"x")
    with pytest.raises(ValueError, match="inventory"):
        verify_outputs(tmp_path, outputs)


def test_report_verification_binds_source_and_evidence_bytes(tmp_path: Path) -> None:
    """A correct output cannot hide a changed evidence table or renderer."""
    source = tmp_path / "renderer.py"
    evidence = tmp_path / "table.csv"
    output = tmp_path / "report"
    output.mkdir()
    source.write_bytes(b"source")
    evidence.write_bytes(b"data")
    manifest = {
        "sources": {"renderer.py": sha256_file(source)},
        "inputs": {"table.csv": sha256_file(evidence)},
        "outputs": {},
    }
    (output / "manifest.json").write_text(json.dumps(manifest))
    verify_report(output, root=tmp_path)
    for path in (source, evidence):
        original = path.read_bytes()
        path.write_bytes(b"altered")
        with pytest.raises(ValueError, match="input or source"):
            verify_report(output, root=tmp_path)
        path.write_bytes(original)


def test_indexed_figures_preserve_sparse_legend_colours(tmp_path: Path) -> None:
    """A mostly overplotted S curve must still have a blue legend, not a green approximation."""
    pixels = np.random.default_rng(17).integers(0, 256, size=(100, 100, 3), dtype=np.uint8)
    blue = (31, 119, 180)
    pixels[0, 0] = blue
    path = tmp_path / "plot.png"
    Image.fromarray(pixels).save(path)
    save_indexed_plot(path)
    with Image.open(path) as figure:
        assert figure.convert("RGB").getpixel((0, 0)) == blue

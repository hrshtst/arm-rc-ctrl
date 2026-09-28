# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Portable GIF exports preserve playback semantics and their matching figures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import GifImagePlugin, Image

from arm_rc_ctrl.experiments import manual_expert_export as exporter
from arm_rc_ctrl.experiments.manual_expert_export import export_case, load_animation, sample_index
from arm_rc_ctrl.experiments.manual_expert_report import CASES
from arm_rc_ctrl.experiments.manual_report import DOCUMENT_LOCATION
from arm_rc_ctrl.repo import repository_root


def test_sampling_holds_the_preceding_state_and_aborted_endpoint() -> None:
    """No interpolation or extrapolation is added by the GIF export."""
    assert sample_index((0.0, 0.1, 0.17), 0.15) == 1
    assert sample_index((0.0, 0.1, 0.17), 30.0) == 2
    assert sample_index((0.0, 0.1, 0.17), -1.0) == 0


def test_animation_loader_reads_data_without_executing_javascript(tmp_path: Path) -> None:
    """Only the known data wrapper is accepted, never arbitrary executable JavaScript."""
    p = tmp_path / "bad.js"
    p.write_text('alert("not data");')
    with pytest.raises(ValueError, match="wrapper"):
        load_animation(p)


def test_export_pairs_a_looping_gif_with_unchanged_matching_plots(tmp_path: Path) -> None:
    """A short test export checks GIF timing, provenance and the three paired figure files."""
    report = repository_root() / DOCUMENT_LOCATION / "expert_report"
    spec = CASES[0]
    info = export_case(report, tmp_path / spec.slug, spec, stop_s=0.2)
    with Image.open(tmp_path / spec.slug / "animation.gif") as gif:
        assert isinstance(gif, GifImagePlugin.GifImageFile)
        assert gif.size == (1500, 450)
        assert gif.n_frames == 6  # starts at -0.25 s, includes the exact +0.2 s endpoint
        assert gif.info["loop"] == 0
        gif.seek(gif.n_frames - 1)
        assert gif.info["duration"] == 1000
    for source, target in [("space", "task-space.png"), ("time", "time-series.png"), ("joints", "joint-angles.png")]:
        assert (tmp_path / spec.slug / target).read_bytes() == (
            report / "assets" / f"{spec.slug}-{source}.png"
        ).read_bytes()
    recorded = json.loads((tmp_path / spec.slug / "case.json").read_text())
    assert recorded["case_id"] == spec.case_id
    assert recorded["playback_speed"] == 2.0
    assert recorded == info
    with pytest.raises(FileExistsError):
        export_case(report, tmp_path / spec.slug, spec, stop_s=0.2)


def test_bundle_refuses_an_existing_archive_before_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An earlier ZIP must never be silently replaced by a fresh directory export."""
    from arm_rc_ctrl.experiments.manual_expert_export import export_bundle

    def unexpected_export(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Existing archive was not refused before rendering")

    monkeypatch.setattr(exporter, "export_case", unexpected_export)
    report = repository_root() / DOCUMENT_LOCATION / "expert_report"
    output = tmp_path / "media"
    archive = tmp_path / "media.zip"
    archive.write_bytes(b"existing export")
    with pytest.raises(FileExistsError):
        export_bundle(report, output)
    assert archive.read_bytes() == b"existing export"
    assert not output.exists()


def test_bundle_inventories_all_case_files_and_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use short frames to exercise the complete packaging path without a long render."""
    import zipfile

    def short_case(report: Path, output: Path, spec: exporter.Illustration) -> dict[str, object]:
        return export_case(report, output, spec, stop_s=0.2)

    monkeypatch.setattr(exporter, "export_case", short_case)
    output = tmp_path / "bundle"
    exporter.export_bundle(repository_root() / DOCUMENT_LOCATION / "expert_report", output)
    manifest = json.loads((output / "manifest.json").read_text())
    for relative, digest in manifest["outputs"].items():
        assert exporter.sha256_file(output / relative) == digest
    assert len(list(output.glob("*/animation.gif"))) == len(CASES)
    with zipfile.ZipFile(tmp_path / "bundle.zip") as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) >= {"bundle/" + path for path in manifest["outputs"]}

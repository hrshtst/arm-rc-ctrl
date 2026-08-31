# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M2-018: the input-scale pilot runs end to end on the fixture dataset and writes a loadable report.

The fixture reference is a fast 0.3 s reach, so the cells here are physically
meaningless; the test exercises the sweep, the selection, the outputs, and the CLI.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from arm_rc_ctrl.data.preprocess import preprocess_demonstration
from arm_rc_ctrl.data.records import ProcessedDatasetRecord, RawDemonstrationRecord, load_record, verify_payload
from arm_rc_ctrl.data.samples import load_samples
from arm_rc_ctrl.experiments import scale_pilot
from arm_rc_ctrl.experiments.scale_pilot import (
    load_protocol,
    load_scale_pilot_report,
    main,
    render_markdown,
    run_scale_pilot,
)
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageRoot

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
RAW_RECORD = REPO_ROOT / "tests" / "fixtures" / "records" / "raw-20260830-287036d83d46.toml"
RAW_LOG = REPO_ROOT / "tests" / "fixtures" / "raw" / "demo.sklog.npz"
SCENARIO = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
MODEL = REPO_ROOT / "tests" / "fixtures" / "configs" / "esn_fixture.toml"
PREPROCESS = REPO_ROOT / "configs" / "preprocessing" / "default.toml"
FIXED_TIME = datetime(2026, 9, 2, 9, 0, 0, tzinfo=UTC)


def _protocol_file(directory: Path) -> Path:
    file = directory / "scale_pilot_fixture.toml"
    file.write_text(
        f'name = "scale-pilot-fixture"\nscenario = "{SCENARIO.as_posix()}"\nmodel = "{MODEL.as_posix()}"\n'
        'trackers = ["pd", "computed_torque"]\n'
        "[grid]\nq_scales = [0.3, 0.5]\ndq_scales = [4.0, 8.0]\n"
        "[variants]\nridge_alphas = [1e-2]\nreservoir_seeds = [31]\n"
        "[estimator]\nvelocity_cutoff_hz = 20.0\nacceleration_cutoff_hz = 20.0\n"
        "[selection]\nmin_feasible_fraction = 0.5\n"
    )
    return file


@pytest.fixture
def prepared(tmp_path: Path) -> tuple[StorageRoot, Path, Path]:
    """A store with the fixture dataset, its record file, and a fixture-scale protocol."""
    root = tmp_path / "store"
    root.mkdir()
    store = StorageRoot(root, repositories=(REPO_ROOT,))
    raw = load_record(RAW_RECORD, RawDemonstrationRecord)
    store.path(raw.artifact.payload.uri, mode="write").write_bytes(RAW_LOG.read_bytes())
    records = tmp_path / "repo"
    (records / "data" / "records" / "processed").mkdir(parents=True)
    processed = preprocess_demonstration(
        RAW_RECORD, SCENARIO, PREPROCESS, store=store, records_root=records, exploratory=True, now=FIXED_TIME
    )
    return store, processed.record_file, _protocol_file(tmp_path)


def test_pilot_sweeps_cells_and_selects_an_anchor(prepared: tuple[StorageRoot, Path, Path]) -> None:
    """Every (cell, variant, tracker) is evaluated and the report carries cells, selection, and provenance."""
    store, record_file, protocol_file = prepared
    protocol = load_protocol(protocol_file)
    dataset = load_record(record_file, ProcessedDatasetRecord)
    samples = load_samples(verify_payload(store, dataset.artifact))
    report = run_scale_pilot(protocol, protocol_file, dataset, samples, exploratory=True, now=FIXED_TIME)
    assert len(report.variants) == 2 * 2 * 1 * 1 * 2
    assert {v.tracker for v in report.variants} == {"pd", "computed_torque"}
    assert len(report.cells) == 4
    assert all(c.variants == 2 for c in report.cells)
    assert report.selection is not None
    assert report.selection.q_scale in protocol.grid.q_scales
    assert report.selection.dq_scale in protocol.grid.dq_scales
    assert report.trackers["pd"].type == "pd"
    assert report.provenance.seeds == {"reservoir_0": 31}
    markdown = render_markdown(report)
    assert "| q \\ dq | 4 | 8 |" in markdown
    assert "## Selection" in markdown


def test_command_line_entry_point(
    prepared: tuple[StorageRoot, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI writes a report that loads back and a Markdown summary, and refuses to overwrite."""
    store, record_file, protocol_file = prepared
    monkeypatch.setenv("ARM_RC_CTRL_STORAGE_ROOT", str(store.root))
    report_file = tmp_path / "out" / "pilot.json"
    markdown_file = tmp_path / "out" / "pilot.md"
    argv = [
        "--protocol",
        str(protocol_file),
        "--dataset",
        str(record_file),
        "--report",
        str(report_file),
        "--markdown",
        str(markdown_file),
        "--exploratory",
    ]
    assert main(argv) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["variants"] == 8
    report = load_scale_pilot_report(report_file)
    assert len(report.cells) == 4
    assert str(tmp_path) not in report_file.read_text()
    assert markdown_file.read_text().startswith("# Input-scale pilot `scale-pilot-fixture`")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(argv)


def test_report_without_a_selection_is_still_written(
    prepared: tuple[StorageRoot, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When no anchor qualifies, the report and table are still written and the command fails loudly."""
    store, record_file, protocol_file = prepared

    def no_anchor(*_args: object, **_kwargs: object) -> object:
        msg = "no feasible cell has only feasible grid neighbours; widen the grid or revisit the settings"
        raise ValueError(msg)

    monkeypatch.setattr(scale_pilot, "select_anchor", no_anchor)
    monkeypatch.setenv("ARM_RC_CTRL_STORAGE_ROOT", str(store.root))
    report_file = tmp_path / "none.json"
    markdown_file = tmp_path / "none.md"
    argv = [
        "--protocol",
        str(protocol_file),
        "--dataset",
        str(record_file),
        "--report",
        str(report_file),
        "--markdown",
        str(markdown_file),
        "--exploratory",
    ]
    with pytest.raises(RuntimeError, match="report was written but nothing is selected"):
        main(argv)
    printed = json.loads(capsys.readouterr().out)
    assert printed["selection"] is None
    report = load_scale_pilot_report(report_file)
    assert report.selection is None
    assert len(report.cells) == 4
    assert "nothing is selected" in markdown_file.read_text()

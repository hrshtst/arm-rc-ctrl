# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-028: the pilot runs end to end on the fixture dataset and writes a loadable report.

The fixture reference is a fast 0.3 s reach on which the frozen task 1-a gains
saturate, so the levels below are physically meaningless; the test exercises
the case sweep, the selection, the JSON/Markdown outputs, and the CLI.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from arm_rc_ctrl.data.preprocess import preprocess_demonstration
from arm_rc_ctrl.data.records import ProcessedDatasetRecord, RawDemonstrationRecord, load_record, verify_payload
from arm_rc_ctrl.data.samples import load_samples
from arm_rc_ctrl.experiments.perturbation_pilot import (
    load_pilot_report,
    load_protocol,
    main,
    render_markdown,
    run_pilot,
)
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageRoot

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
RAW_RECORD = REPO_ROOT / "tests" / "fixtures" / "records" / "raw-20260830-287036d83d46.toml"
RAW_LOG = REPO_ROOT / "tests" / "fixtures" / "raw" / "demo.sklog.npz"
SCENARIO = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
PREPROCESS = REPO_ROOT / "configs" / "preprocessing" / "default.toml"
FIXED_TIME = datetime(2026, 8, 31, 12, 0, 0, tzinfo=UTC)


def _protocol_file(directory: Path) -> Path:
    file = directory / "pilot_fixture.toml"
    file.write_text(
        f'name = "pilot-fixture"\nscenario = "{SCENARIO.as_posix()}"\nbaselines = ["pd", "computed_torque"]\n'
        "[posture]\nmagnitudes = [0.005, 0.01, 0.02]\ndirections = [[1.0, 0.0], [0.0, -1.0]]\n"
        "[force]\nmagnitudes = [0.5, 2.0]\ndirections_deg = [0.0, 180.0]\nstart_s = 0.12\nduration_s = 0.05\n"
        "[selection]\nposture_recovery_min_s = 0.02\nposture_recovery_max_s = 0.5\nforce_recovery_max_s = 0.5\n"
        "force_deviation_min_m = 0.002\nforce_max_saturation_fraction = 1.0\n"
    )
    return file


@pytest.fixture
def prepared(tmp_path: Path) -> tuple[StorageRoot, Path, Path]:
    """A store with the fixture dataset, the processed record file, and a fixture-scale protocol."""
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


def test_pilot_sweeps_every_case_and_selects_levels(prepared: tuple[StorageRoot, Path, Path]) -> None:
    """Every (baseline, kind, magnitude, direction) case is kept; levels and the selection follow the rules."""
    store, record_file, protocol_file = prepared
    protocol = load_protocol(protocol_file)
    dataset = load_record(record_file, ProcessedDatasetRecord)
    samples = load_samples(verify_payload(store, dataset.artifact))
    report = run_pilot(protocol, protocol_file, dataset, samples, exploratory=True, now=FIXED_TIME)
    assert len(report.cases) == 2 * (3 * 2 + 2 * 2)
    assert {c.baseline for c in report.cases} == {"pd", "computed_torque"}
    posture = [c for c in report.cases if c.kind == "posture"]
    assert all(c.recovery_time_s is not None and c.termination == "completed" for c in posture)
    force = [c for c in report.cases if c.kind == "force"]
    assert all(c.initial_q == (0.3, 0.6) for c in force)
    assert all(c.direction in {(0.0,), (180.0,)} for c in force)
    assert [(lv.kind, lv.magnitude) for lv in report.levels] == [
        ("posture", 0.005),
        ("posture", 0.01),
        ("posture", 0.02),
        ("force", 0.5),
        ("force", 2.0),
    ]
    assert all(lv.safe for lv in report.levels)
    assert report.selection.posture_small_rad == 0.005
    assert report.selection.posture_large_rad == 0.02
    assert report.selection.force_magnitude_n == 2.0
    assert report.scenario_file == "tests/fixtures/configs/planar_2dof_fixture.toml"
    assert report.provenance.config["protocol"]["scenario"] == report.scenario_file  # type: ignore[index]
    assert report.baselines["pd"].type == "pd"
    markdown = render_markdown(report)
    assert "## Posture levels" in markdown
    assert "| 0.02 | computed_torque |" in markdown
    assert "- endpoint force pulse: 2 N for 0.05 s from t = 0.12 s" in markdown


def test_command_line_entry_point(
    prepared: tuple[StorageRoot, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI writes a report that loads back, a Markdown summary, and refuses to overwrite either."""
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
    assert printed["cases"] == 20
    assert printed["selection"]["force_magnitude_n"] == 2.0
    report = load_pilot_report(report_file)
    assert len(report.cases) == 20
    assert str(tmp_path) not in report_file.read_text()  # no machine paths in the report
    assert markdown_file.read_text().startswith("# Perturbation pilot `pilot-fixture`")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(argv)

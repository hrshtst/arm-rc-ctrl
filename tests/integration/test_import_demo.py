# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-013: demonstrations are imported transactionally, verified against their record, and catalogued."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from skelarm import StateLog

from arm_rc_ctrl.data.import_demo import import_demonstration, main, sampling_from_log
from arm_rc_ctrl.data.raw import load_raw_demonstration
from arm_rc_ctrl.data.records import Intervals, RawDemonstrationRecord, load_catalog, load_record
from arm_rc_ctrl.data.teacher import record_demonstration
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import load_scenario
from arm_rc_ctrl.storage import StorageRoot

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
TASK_1A = REPO_ROOT / "configs" / "tasks" / "task_1a.toml"
INTERVALS = Intervals((0.0, 1.0), (1.0, 4.0), (4.0, 5.0))
FIXED_TIME = datetime(2026, 8, 30, 12, 0, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def teacher_log(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The scripted demonstration written once per module."""
    path = tmp_path_factory.mktemp("teacher") / "demo.sklog.npz"
    log, _ = record_demonstration(load_scenario(TASK_1A))
    log.save(path)
    return path


@pytest.fixture
def store(tmp_path: Path) -> StorageRoot:
    """Empty storage root."""
    root = tmp_path / "store"
    root.mkdir()
    return StorageRoot(root, repositories=(REPO_ROOT,))


@pytest.fixture
def records_root(tmp_path: Path) -> Path:
    """Repository-like root holding a copy of the scenario so record paths stay repository-relative."""
    root = tmp_path / "repo"
    (root / "configs" / "tasks").mkdir(parents=True)
    (root / "configs" / "tasks" / "task_1a.toml").write_text(TASK_1A.read_text())
    return root


def _import(store: StorageRoot, records_root: Path, log: Path, **overrides: object):  # noqa: ANN202
    kwargs: dict[str, object] = {
        "store": store,
        "records_root": records_root,
        "session": "scripted-minjerk-01",
        "license_label": "CC0-1.0",
        "access": "public",
        "intervals": INTERVALS,
        "clock": "simulated",
        "exploratory": True,
        "now": FIXED_TIME,
        "notes": "test import",
    }
    kwargs.update(overrides)
    return import_demonstration(log, records_root / "configs" / "tasks" / "task_1a.toml", **kwargs)  # type: ignore[arg-type]


def test_scripted_demonstration_is_imported_verified_and_catalogued(
    store: StorageRoot, records_root: Path, teacher_log: Path
) -> None:
    """The payload lands at its content-addressed URI, the record validates, and the catalog lists it."""
    result = _import(store, records_root, teacher_log)
    record = result.record
    assert record.artifact.artifact_id.startswith("raw-20260830-")
    assert record.artifact.license == "CC0-1.0"
    assert record.artifact.access == "public"
    assert record.session == "scripted-minjerk-01"
    assert record.intervals == INTERVALS
    assert record.duration_s == 5.0
    assert record.sampling.period_s == pytest.approx(0.01)
    assert record.sampling.units == {"t": "s", "q": "rad", "dq": "rad/s", "tau": "N*m"}
    assert record.scenario.config_path == "configs/tasks/task_1a.toml"
    assert record.scenario.target == (0.10, 0.45)
    assert result.payload_file.read_bytes() == teacher_log.read_bytes()
    assert result.resumed is False
    assert load_record(result.record_file, RawDemonstrationRecord) == record
    assert load_catalog(records_root / "data" / "catalog.toml").find(record.artifact.artifact_id) is not None
    demo = load_raw_demonstration(store, record)
    assert demo.n_samples == 501
    assert not any(p.name.startswith("staging-") for p in (store.root / "raw").iterdir())


def test_reimport_is_a_verified_no_op(store: StorageRoot, records_root: Path, teacher_log: Path) -> None:
    """Importing the same bytes again resumes without rewriting anything."""
    first = _import(store, records_root, teacher_log)
    before = (first.payload_file.read_bytes(), first.record_file.read_text())
    again = _import(store, records_root, teacher_log)
    assert again.resumed is True
    assert again.record == first.record
    assert (first.payload_file.read_bytes(), first.record_file.read_text()) == before


def test_human_recorder_layout_imports_with_proposed_intervals(
    store: StorageRoot, records_root: Path, tmp_path: Path
) -> None:
    """A q/tip-only log (as skelarm's IK recorder writes) imports once units and intervals are supplied."""
    scenario = load_scenario(TASK_1A)
    teacher, plan = record_demonstration(scenario)
    log = StateLog(teacher.build_skeleton(), producer="trajectory_recorder", channel_meta={"q": {"unit": "rad"}})
    q = teacher.channel("q")
    for k in range(0, len(teacher), 2):  # 20 ms, like the mouse recorder
        log.record(float(teacher.times[k]), q=q[k], tip=np.zeros(2))
    path = tmp_path / "human.sklog.npz"
    log.save(path)
    with pytest.raises(ValueError, match=r"no unit for channel\(s\) \['tip'\]"):
        _import(store, records_root, path, clock="wall")
    result = _import(store, records_root, path, clock="wall", units_override={"tip": "m"}, session="teacher-01")
    assert result.record.sampling.clock == "wall"
    assert result.record.sampling.period_s == pytest.approx(0.02)
    assert load_raw_demonstration(store, result.record).dq is None
    assert sampling_from_log(StateLog.load(path), "wall", {"tip": "m"}).units["tip"] == "m"
    del plan


def test_failures_leave_no_payload_or_record(
    store: StorageRoot, records_root: Path, teacher_log: Path, tmp_path: Path
) -> None:
    """Bad intervals or a record that disagrees with the log abort before anything is registered."""
    with pytest.raises(ValueError, match="recording ends at"):
        _import(store, records_root, teacher_log, intervals=Intervals((0.0, 1.0), (1.0, 4.0), (4.0, 6.0)))
    other = tmp_path / "other.toml"
    other.write_text(TASK_1A.read_text())
    with pytest.raises(ValueError, match="must live inside the repository"):
        import_demonstration(
            teacher_log,
            other,
            store=store,
            records_root=records_root,
            session="s-01",
            license_label="CC0-1.0",
            access="public",
            intervals=INTERVALS,
            clock="simulated",
            exploratory=True,
            now=FIXED_TIME,
        )
    assert not (store.root / "raw").exists() or not any((store.root / "raw").iterdir())
    assert not (records_root / "data").exists()


def test_command_line_proposal_then_confirm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], teacher_log: Path
) -> None:
    """--propose-intervals prints and plots without importing; --confirm imports through the configured store."""
    root = tmp_path / "store"
    root.mkdir()
    monkeypatch.setenv("ARM_RC_CTRL_STORAGE_ROOT", str(root))
    fake_repo = tmp_path / "repo"
    (fake_repo / "configs" / "tasks").mkdir(parents=True)
    (fake_repo / "configs" / "tasks" / "task_1a.toml").write_text(TASK_1A.read_text())
    (fake_repo / "src" / "arm_rc_ctrl").mkdir(parents=True)
    (fake_repo / "pyproject.toml").write_text((REPO_ROOT / "pyproject.toml").read_text())
    monkeypatch.setattr("arm_rc_ctrl.data.import_demo.repository_root", lambda: fake_repo)
    plot = tmp_path / "review.png"
    base = [
        "--log",
        str(teacher_log),
        "--scenario",
        str(fake_repo / "configs" / "tasks" / "task_1a.toml"),
        "--session",
        "teacher-01",
        "--license",
        "LicenseRef-Private",
        "--access",
        "private",
        "--clock",
        "simulated",
        "--propose-intervals",
        "--speed-threshold",
        "0.01",
        "--plot",
        str(plot),
        "--exploratory",
    ]
    assert main(base) == 0
    out = capsys.readouterr().out
    assert "proposed intervals" in out
    assert "re-run with --confirm" in out
    assert plot.is_file()
    assert not (root / "raw").exists()
    assert main([*base, "--confirm"]) == 0
    printed = json.loads(capsys.readouterr().out.strip().split("\n", 2)[-1])
    assert printed["frames"] == 501
    record = load_record(fake_repo / printed["record"], RawDemonstrationRecord)
    assert abs(record.intervals.move[0] - 1.0) < 0.15

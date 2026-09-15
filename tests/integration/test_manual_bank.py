# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-002: batches of saved takes are validated offline, the first ten accepted in order form the bank."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from arm_rc_ctrl.data.manual import ManualDatasetRecord, ManualTakeRecord
from arm_rc_ctrl.data.manual_scenario import load_manual_scenario
from arm_rc_ctrl.data.records import load_record
from arm_rc_ctrl.data.synthetic import synthetic_manual_take_log
from arm_rc_ctrl.experiments import manual_bank
from arm_rc_ctrl.experiments.manual_bank import (
    BankManifest,
    BatchReport,
    attempt_number,
    load_bank_manifest,
    main,
    validate_batch,
)
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageRoot

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "configs"
GOAL_Q = (0.8, 0.4)
# attempt -> take settings; 3 has too short a dwell, 7 duplicates 2 byte for byte, 9 is malformed
SETTINGS: dict[int, dict[str, float]] = {
    1: {"hold_s": 0.5, "move_s": 0.6, "dwell_s": 0.3},
    2: {"hold_s": 0.6, "move_s": 0.5, "dwell_s": 0.3},
    3: {"hold_s": 0.5, "move_s": 0.6, "dwell_s": 0.05},
    4: {"hold_s": 0.7, "move_s": 0.6, "dwell_s": 0.3},
    5: {"hold_s": 0.5, "move_s": 0.8, "dwell_s": 0.3},
    6: {"hold_s": 0.8, "move_s": 0.7, "dwell_s": 0.4},
    7: {"hold_s": 0.6, "move_s": 0.5, "dwell_s": 0.3},
    8: {"hold_s": 0.5, "move_s": 0.9, "dwell_s": 0.3},
    9: {"hold_s": 0.5, "move_s": 0.6, "dwell_s": 0.3},
    10: {"hold_s": 0.9, "move_s": 0.6, "dwell_s": 0.3},
    11: {"hold_s": 0.5, "move_s": 0.65, "dwell_s": 0.35},
    12: {"hold_s": 0.55, "move_s": 0.6, "dwell_s": 0.3},
    13: {"hold_s": 0.5, "move_s": 0.75, "dwell_s": 0.3},
    14: {"hold_s": 0.65, "move_s": 0.6, "dwell_s": 0.3},
}
GOOD_ORDER = [1, 2, 4, 5, 6, 8, 10, 11, 12, 13, 14]
MALFORMED_ATTEMPT = 9


@dataclass(frozen=True)
class Workspace:
    """Store, fake records root with the fixture configs, take files, and the manifest path."""

    store: StorageRoot
    records_root: Path
    scenario: Path
    derive: Path
    takes: Path
    manifest: Path


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    """Build the store, the records root, and the numbered take files of every attempt."""
    root = tmp_path / "store"
    root.mkdir()
    store = StorageRoot(root, repositories=(REPO_ROOT,))
    records_root = tmp_path / "repo"
    (records_root / "configs" / "tasks").mkdir(parents=True)
    (records_root / "configs" / "preprocessing").mkdir(parents=True)
    scenario = records_root / "configs" / "tasks" / "manual_fixture.toml"
    shutil.copyfile(FIXTURES / "planar_2dof_manual_fixture.toml", scenario)
    derive = records_root / "configs" / "preprocessing" / "manual_fixture.toml"
    shutil.copyfile(FIXTURES / "manual_derive_fixture.toml", derive)
    takes = tmp_path / "takes"
    takes.mkdir()
    config = load_manual_scenario(scenario)
    for attempt, settings in SETTINGS.items():
        log = synthetic_manual_take_log(
            config, goal_q=GOAL_Q, hold_s=settings["hold_s"], move_s=settings["move_s"], dwell_s=settings["dwell_s"]
        )
        if attempt == MALFORMED_ATTEMPT:
            del log.extra["acquisition"]
        log.save(takes / f"reach_{attempt:03d}.sklog.npz")
    manifest = records_root / "docs" / "bank" / "task_bank_v1.json"
    return Workspace(store, records_root, scenario, derive, takes, manifest)


def _validate(ws: Workspace, batch: int, attempts: list[int], scenario_file: Path | None = None) -> BatchReport:
    files = [ws.takes / f"reach_{a:03d}.sklog.npz" for a in attempts]
    return validate_batch(
        files,
        scenario_file=scenario_file or ws.scenario,
        config_file=ws.derive,
        store=ws.store,
        records_root=ws.records_root,
        session="fixture-session",
        batch=batch,
        manifest_file=ws.manifest,
        required=10,
        license_label="proprietary",
        access="private",
        exploratory=True,
    )


def test_attempt_numbers_come_from_the_recorder_file_names() -> None:
    """Numbered recorder outputs identify attempts; anything else is refused rather than guessed."""
    assert attempt_number(Path("out/reach_007.sklog.npz")) == 7
    assert attempt_number(Path("take_12.npz")) == 12
    with pytest.raises(ValueError, match="numbered"):
        attempt_number(Path("teach.sklog.npz"))


def test_first_batch_accepts_in_order_and_requests_the_shortfall(workspace: Workspace) -> None:
    """Every attempt is imported and retained; accepted ones derive datasets; the shortfall is requested."""
    report = _validate(workspace, 1, [1, 2, 3, 4, 5, 6])
    assert [v.attempt for v in report.verdicts] == [1, 2, 3, 4, 5, 6]
    assert [v.accepted for v in report.verdicts] == [True, True, False, True, True, True]
    rejected = report.verdicts[2]
    assert rejected.raw_artifact_id is not None  # the failed study take is retained
    assert rejected.processed_artifact_id is None
    assert any("dwell" in reason for reason in rejected.reasons)
    assert all(v.processed_artifact_id is not None for v in report.verdicts if v.accepted)
    assert report.accepted_total == 5
    assert report.shortfall == 5
    assert not report.complete
    assert "5 more" in report.next_action

    manifest = load_bank_manifest(workspace.manifest)
    assert isinstance(manifest, BankManifest)
    assert [t.attempt for t in manifest.takes] == [1, 2, 3, 4, 5, 6]
    assert manifest.accepted_attempts == (1, 2, 4, 5, 6)
    assert manifest.assignments == {}
    assert manifest.scenario_path == "configs/tasks/manual_fixture.toml"
    text = workspace.manifest.read_text(encoding="utf-8")
    assert str(workspace.store.root) not in text  # portable: no machine paths
    assert str(workspace.takes) not in text
    reports = sorted(p.name for p in workspace.manifest.parent.iterdir())
    assert reports == ["task_bank_v1.json", "task_bank_v1_batch_001.json", "task_bank_v1_batch_001.md"]
    markdown = (workspace.manifest.parent / "task_bank_v1_batch_001.md").read_text(encoding="utf-8")
    assert "| 3 |" in markdown
    assert "collect at least 5 more" in markdown


def test_later_batches_complete_the_bank_with_the_first_ten_in_order(workspace: Workspace) -> None:
    """Duplicates and malformed takes are rejected and kept; the first ten accepted takes become D01..D10."""
    _validate(workspace, 1, [1, 2, 3, 4, 5, 6])
    report = _validate(workspace, 2, [7, 8, 9, 10, 11, 12, 13, 14])
    by_attempt = {v.attempt: v for v in report.verdicts}
    assert not by_attempt[7].accepted
    assert any("duplicate of attempt 2" in reason for reason in by_attempt[7].reasons)
    assert by_attempt[7].raw_artifact_id is not None  # retained even as a duplicate
    assert not by_attempt[9].accepted
    assert any("malformed" in reason for reason in by_attempt[9].reasons)
    assert by_attempt[9].raw_artifact_id is None  # nothing importable to retain
    assert report.accepted_total == 11
    assert report.shortfall == 0
    assert report.complete
    assert "complete" in report.next_action

    manifest = load_bank_manifest(workspace.manifest)
    assert manifest.accepted_attempts == tuple(GOOD_ORDER)
    assert list(manifest.assignments) == [f"D{i:02d}" for i in range(1, 11)]
    assigned_attempts = [
        next(t.attempt for t in manifest.takes if t.raw_artifact_id == raw) for raw in manifest.assignments.values()
    ]
    assert assigned_attempts == GOOD_ORDER[:10]
    extra = next(t for t in manifest.takes if t.attempt == GOOD_ORDER[-1])
    assert extra.accepted  # accepted and retained ...
    assert extra.assignment is None  # ... but beyond the required ten
    for take in manifest.takes:
        if take.raw_artifact_id is not None:
            raw = load_record(
                workspace.records_root / "data" / "records" / "raw" / f"{take.raw_artifact_id}.toml",
                ManualTakeRecord,
            )
            assert raw.take == take.attempt
        if take.processed_artifact_id is not None:
            processed = load_record(
                workspace.records_root / "data" / "records" / "processed" / f"{take.processed_artifact_id}.toml",
                ManualDatasetRecord,
            )
            assert processed.take == take.attempt
    assert manifest.complete


def test_a_batch_is_versioned_never_edited(workspace: Workspace) -> None:
    """Re-running a validated batch number is refused; the manifest keeps its history."""
    _validate(workspace, 1, [1, 2])
    with pytest.raises(ValueError, match="batch 1"):
        _validate(workspace, 1, [3])
    with pytest.raises(ValueError, match="scenario"):
        _validate(workspace, 2, [3], scenario_file=REPO_ROOT / "configs" / "tasks" / "task_1a_manual_v1.toml")
    assert [t.attempt for t in load_bank_manifest(workspace.manifest).takes] == [1, 2]


def test_cli_validates_a_batch_and_signals_the_shortfall(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command validates through the configured store and exits 2 while the bank is incomplete, 0 once done."""
    monkeypatch.setenv("ARM_RC_CTRL_STORAGE_ROOT", str(workspace.store.root))
    monkeypatch.setattr(manual_bank, "repository_root", lambda: workspace.records_root)
    common = [
        "--scenario", str(workspace.scenario), "--config", str(workspace.derive),
        "--session", "fixture-session", "--manifest", str(workspace.manifest), "--exploratory",
    ]  # fmt: skip
    files = [str(workspace.takes / f"reach_{a:03d}.sklog.npz") for a in [1, 2, 3, 4, 5, 6]]
    assert main([*common, "--batch", "1", "--takes", *files]) == 2
    out = capsys.readouterr().out
    assert "accepted 5" in out
    assert "collect at least 5 more" in out
    files = [str(workspace.takes / f"reach_{a:03d}.sklog.npz") for a in [7, 8, 10, 11, 12, 13]]
    assert main([*common, "--batch", "2", "--takes", *files]) == 0
    assert "complete" in capsys.readouterr().out

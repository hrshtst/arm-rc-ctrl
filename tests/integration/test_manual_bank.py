# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-002: batches of saved takes are validated offline, the first ten accepted in order form the bank."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from arm_rc_ctrl import provenance
from arm_rc_ctrl.data.manual import ManualDatasetRecord, ManualTakeRecord
from arm_rc_ctrl.data.manual_scenario import ManualScenarioConfig, load_manual_scenario
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
from arm_rc_ctrl.provenance import DirtyWorktreeError
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


def _validate(
    ws: Workspace, batch: int, attempts: list[int], scenario_file: Path | None = None, *, exploratory: bool = True
) -> BatchReport:
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
        exploratory=exploratory,
    )


def _git(root: Path, *args: str) -> str:
    return subprocess.run(  # trusted: git on the fixture directory
        ["git", *args], cwd=root, text=True, check=True, capture_output=True
    ).stdout.strip()


def _make_clean_worktree(ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn the fixture records root into a committed Git worktree whose dirty state the provenance reads."""
    root = ws.records_root
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
    real = provenance.worktree_state

    def state(path: Path) -> tuple[str, bool]:
        commit, _ = real(path)
        return commit, bool(_git(root, "status", "--porcelain"))

    monkeypatch.setattr(provenance, "worktree_state", state)


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


# ----------------------------------------------------------------------------------------------
# Review round 2026-09-15
# ----------------------------------------------------------------------------------------------


def test_a_confirmatory_batch_registers_its_records_only_at_the_end(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without --exploratory a batch must not trip over its own records: they are registered after the takes."""
    _make_clean_worktree(workspace, monkeypatch)
    report = _validate(workspace, 1, [1, 2], exploratory=False)
    assert [v.accepted for v in report.verdicts] == [True, True]
    for verdict in report.verdicts:
        raw = load_record(
            workspace.records_root / "data" / "records" / "raw" / f"{verdict.raw_artifact_id}.toml", ManualTakeRecord
        )
        processed = load_record(
            workspace.records_root / "data" / "records" / "processed" / f"{verdict.processed_artifact_id}.toml",
            ManualDatasetRecord,
        )
        assert raw.artifact.origin.project_dirty is False
        assert processed.artifact.origin.project_dirty is False
    _git(workspace.records_root, "add", ".")
    _git(workspace.records_root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "b1")
    (workspace.records_root / "scratch.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(DirtyWorktreeError):
        _validate(workspace, 2, [4], exploratory=False)
    assert load_bank_manifest(workspace.manifest).batches == (1,)


def test_a_corrupt_archive_is_rejected_without_aborting_the_batch(workspace: Workspace) -> None:
    """An unreadable file becomes a per-take rejection; the later takes of the batch are still assessed."""
    (workspace.takes / "reach_002.sklog.npz").write_bytes(b"not an npz archive")
    report = _validate(workspace, 1, [1, 2, 4])
    assert [v.accepted for v in report.verdicts] == [True, False, True]
    assert any("malformed" in reason for reason in report.verdicts[1].reasons)
    assert report.verdicts[1].raw_artifact_id is None
    with pytest.raises(ValueError, match="not found"):
        _validate(workspace, 2, [99])


def test_a_failed_report_write_leaves_the_batch_unrecorded_and_retryable(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manifest is published last: a report failure records nothing, and the retry rebuilds identical verdicts."""
    original = Path.write_text

    def failing(path: Path, *args: object, **kwargs: object) -> int:
        if path.name.endswith("_batch_001.json"):
            msg = "disk full while writing the batch report"
            raise OSError(msg)
        return original(path, *args, **kwargs)  # type: ignore[arg-type]

    with monkeypatch.context() as context:
        context.setattr(Path, "write_text", failing)
        with pytest.raises(OSError, match="disk full"):
            _validate(workspace, 1, [1, 2])
    assert not workspace.manifest.exists() or 1 not in load_bank_manifest(workspace.manifest).batches
    report = _validate(workspace, 1, [1, 2])
    assert [v.accepted for v in report.verdicts] == [True, True]
    assert report.report_json.exists()
    assert load_bank_manifest(workspace.manifest).batches == (1,)


def test_selection_follows_acquisition_order_and_locks_once_complete(workspace: Workspace) -> None:
    """Attempt numbers, not batch numbers, order the bank; a complete bank refuses earlier attempts."""
    _validate(workspace, 1, [4, 5, 6, 8, 10, 11])
    report = _validate(workspace, 2, [1, 2, 12, 13, 14])
    assert report.complete
    manifest = load_bank_manifest(workspace.manifest)
    assert manifest.accepted_attempts == (1, 2, 4, 5, 6, 8, 10, 11, 12, 13, 14)
    by_raw = {t.raw_artifact_id: t.attempt for t in manifest.takes}
    assert [by_raw[raw] for raw in manifest.assignments.values()] == [1, 2, 4, 5, 6, 8, 10, 11, 12, 13]
    with pytest.raises(ValueError, match="locked"):
        _validate(workspace, 3, [3])
    assert load_bank_manifest(workspace.manifest).batches == (1, 2)


def test_a_byte_identical_file_is_a_duplicate_without_a_record_of_its_own(workspace: Workspace) -> None:
    """Payload deduplication never lends another attempt's record to a copied file."""
    shutil.copyfile(workspace.takes / "reach_001.sklog.npz", workspace.takes / "reach_002.sklog.npz")
    report = _validate(workspace, 1, [1, 2])
    first, second = report.verdicts
    assert first.accepted
    assert not second.accepted
    assert second.raw_artifact_id is None
    assert second.duplicate_of == first.raw_artifact_id
    assert any("byte-identical" in reason and "attempt 1" in reason for reason in second.reasons)
    raw = load_record(
        workspace.records_root / "data" / "records" / "raw" / f"{first.raw_artifact_id}.toml", ManualTakeRecord
    )
    assert raw.take == 1
    shutil.copyfile(workspace.takes / "reach_001.sklog.npz", workspace.takes / "reach_003.sklog.npz")
    later = _validate(workspace, 2, [3])
    assert later.verdicts[0].duplicate_of == first.raw_artifact_id
    assert later.verdicts[0].raw_artifact_id is None


def test_manifest_loading_is_strict_and_checks_consistency(workspace: Workspace) -> None:
    """A tampered manifest is refused: wrong types, unknown keys, and assignments that contradict the verdicts."""
    _validate(workspace, 1, [1, 2, 3])
    data = json.loads(workspace.manifest.read_text(encoding="utf-8"))

    def write(document: dict[str, object]) -> None:
        workspace.manifest.write_text(json.dumps(document), encoding="utf-8")

    tampered = json.loads(json.dumps(data))
    tampered["takes"][2]["accepted"] = "false"
    write(tampered)
    with pytest.raises(ValueError, match="accepted"):
        load_bank_manifest(workspace.manifest)
    write({**data, "extra": 1})
    with pytest.raises(ValueError, match="extra"):
        load_bank_manifest(workspace.manifest)
    tampered = json.loads(json.dumps(data))
    tampered["takes"][2]["assignment"] = "D01"
    write(tampered)
    with pytest.raises(ValueError, match="assignment"):
        load_bank_manifest(workspace.manifest)
    write(data)
    assert load_bank_manifest(workspace.manifest).batches == (1,)


# ----------------------------------------------------------------------------------------------
# Review round 2 (2026-09-15)
# ----------------------------------------------------------------------------------------------


def test_a_partial_publication_resumes_without_exploratory_and_refuses_unrelated_edits(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A batch whose Markdown report failed after its JSON report completes on retry from its journal."""
    _make_clean_worktree(workspace, monkeypatch)
    original = Path.write_text

    def failing(path: Path, *args: object, **kwargs: object) -> int:
        if path.name.endswith("_batch_001.md"):
            msg = "disk full while writing the Markdown report"
            raise OSError(msg)
        return original(path, *args, **kwargs)  # type: ignore[arg-type]

    with monkeypatch.context() as context:
        context.setattr(Path, "write_text", failing)
        with pytest.raises(OSError, match="Markdown"):
            _validate(workspace, 1, [1, 2], exploratory=False)
    assert not workspace.manifest.exists()
    assert workspace.manifest.with_name("task_bank_v1_batch_001.json").exists()
    with pytest.raises(ValueError, match="pending publication"):
        _validate(workspace, 2, [4], exploratory=False)
    with pytest.raises(ValueError, match="same takes"):
        _validate(workspace, 1, [1], exploratory=False)
    unrelated = workspace.records_root / "notes.txt"
    unrelated.write_text("unrelated", encoding="utf-8")
    with pytest.raises(DirtyWorktreeError, match=r"notes\.txt"):
        _validate(workspace, 1, [1, 2], exploratory=False)
    unrelated.unlink()
    report = _validate(workspace, 1, [1, 2], exploratory=False)
    assert [v.accepted for v in report.verdicts] == [True, True]
    assert report.report_markdown.exists()
    assert load_bank_manifest(workspace.manifest).batches == (1,)
    assert not list((workspace.store.root / "reports" / "manual_bank" / "pending").glob("*.json"))
    with pytest.raises(ValueError, match="batch 1"):
        _validate(workspace, 1, [1, 2], exploratory=False)


def test_mismatched_channel_lengths_are_a_per_take_rejection(workspace: Workspace) -> None:
    """An archive whose joint rows disagree with its timestamps is malformed; the batch goes on."""
    bad = workspace.takes / "reach_002.sklog.npz"
    with np.load(bad, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["q"] = arrays["q"][:-1]
    np.savez_compressed(bad, **arrays)
    report = _validate(workspace, 1, [1, 2, 4])
    assert [v.accepted for v in report.verdicts] == [True, False, True]
    assert any("malformed" in reason and "rows" in reason for reason in report.verdicts[1].reasons)


@pytest.mark.parametrize("target", ["raw", "processed", "catalog"])
def test_an_interrupted_atomic_record_write_resumes_without_exploratory(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    """A temporary record or catalog file left by an interrupted atomic write is the batch's own and is cleaned up."""
    _make_clean_worktree(workspace, monkeypatch)
    original = Path.replace

    def interrupted(path: Path, destination: Path) -> Path:
        if target == "catalog":
            hit = path.name == "catalog.toml.tmp"
        else:
            hit = path.parent.name == target and path.name.endswith(".toml.tmp")
        if hit:
            msg = "publication interrupted after the temporary file was written"
            raise OSError(msg)
        return original(path, destination)

    with monkeypatch.context() as context:
        context.setattr(Path, "replace", interrupted)
        with pytest.raises(OSError, match="interrupted"):
            _validate(workspace, 1, [1, 2], exploratory=False)
    assert not workspace.manifest.exists()
    assert list(workspace.records_root.rglob("*.toml.tmp"))
    report = _validate(workspace, 1, [1, 2], exploratory=False)
    assert [v.accepted for v in report.verdicts] == [True, True]
    assert not list(workspace.records_root.rglob("*.tmp"))
    assert load_bank_manifest(workspace.manifest).batches == (1,)


# ----------------------------------------------------------------------------------------------
# M3MAN-013: a v2 batch (50 Hz acquisition, natural pre-roll, first-sample reflection)
# ----------------------------------------------------------------------------------------------


def _goal_posture(config: ManualScenarioConfig) -> tuple[float, float]:
    """Joint angles that put the endpoint on the target, on the elbow branch of the reset posture."""
    l1 = config.robot.links[0].length
    l2 = config.robot.links[1].length
    x, y = config.task.target
    elbow = math.acos((x * x + y * y - l1 * l1 - l2 * l2) / (2.0 * l1 * l2))
    return math.atan2(y, x) - math.atan2(l2 * math.sin(elbow), l1 + l2 * math.cos(elbow)), elbow


def test_a_v2_batch_of_50hz_takes_that_move_at_once_is_validated_end_to_end(tmp_path: Path) -> None:
    """Under the v2 configurations, takes moving from the second sample are accepted and a 100 Hz take is not."""
    root = tmp_path / "store"
    root.mkdir()
    store = StorageRoot(root, repositories=(REPO_ROOT,))
    records_root = tmp_path / "repo"
    (records_root / "configs" / "tasks").mkdir(parents=True)
    (records_root / "configs" / "preprocessing").mkdir(parents=True)
    scenario = records_root / "configs" / "tasks" / "task_1a_manual_v2.toml"
    shutil.copyfile(REPO_ROOT / "configs" / "tasks" / "task_1a_manual_v2.toml", scenario)
    derive = records_root / "configs" / "preprocessing" / "manual_v2.toml"
    shutil.copyfile(REPO_ROOT / "configs" / "preprocessing" / "manual_v2.toml", derive)
    config = load_manual_scenario(scenario)
    takes = tmp_path / "takes"
    takes.mkdir()
    shapes: dict[int, tuple[float, float, float]] = {1: (0.02, 1.0, 0.004), 2: (0.02, 1.3, 0.003), 3: (0.01, 1.0, 0.0)}
    for attempt, (period, move_s, jitter_s) in shapes.items():
        log = synthetic_manual_take_log(
            config,
            goal_q=_goal_posture(config),
            sample_period_s=period,
            hold_s=period,
            move_s=move_s,
            dwell_s=2.0,
            preroll_amplitude_rad=0.005,
            jitter_s=jitter_s,
            seed=attempt,
        )
        log.save(takes / f"reach_{attempt:03d}.sklog.npz")
    manifest = records_root / "docs" / "bank" / "task_bank_v2.json"
    report = validate_batch(
        [takes / f"reach_{attempt:03d}.sklog.npz" for attempt in shapes],
        scenario_file=scenario,
        config_file=derive,
        store=store,
        records_root=records_root,
        session="practice-v2",
        batch=1,
        manifest_file=manifest,
        required=2,
        license_label="proprietary",
        access="private",
        exploratory=True,
    )
    assert [v.accepted for v in report.verdicts] == [True, True, False]
    assert any("100 Hz" in reason and "50 Hz" in reason for reason in report.verdicts[2].reasons)
    assert report.verdicts[2].raw_artifact_id is not None  # the rejected take is retained
    assert report.complete
    assert list(load_bank_manifest(manifest).assignments) == ["D01", "D02"]
    for verdict in report.verdicts[:2]:
        measurements = verdict.measurements
        assert measurements.start_shift_rad is not None
        assert measurements.start_shift_rad <= 1e-12
        assert measurements.hold_end_s is not None
        assert measurements.hold_end_s < 0.05  # movement from the second raw sample on
        assert measurements.median_interval_s == pytest.approx(0.02, abs=0.005)
        processed = load_record(
            records_root / "data" / "records" / "processed" / f"{verdict.processed_artifact_id}.toml",
            ManualDatasetRecord,
        )
        assert processed.manual_schema_version == 2
        assert processed.raw_timing.acquisition_period_s == 0.02

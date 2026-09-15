# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-002: manual takes import unchanged, are assessed against the frozen rules, and derive full datasets."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from skelarm import StateLog

from arm_rc_ctrl.data.manual import (
    MANUAL_PHASE_CODES,
    DuplicatePayloadError,
    ManualDatasetRecord,
    ManualImportResult,
    ManualTakeError,
    ManualTakeRecord,
    assess_take,
    derive_manual_dataset,
    import_manual_take,
    load_manual_derive_config,
    load_manual_take,
    register_manual_records,
)
from arm_rc_ctrl.data.manual_scenario import load_manual_scenario
from arm_rc_ctrl.data.records import load_catalog, load_record
from arm_rc_ctrl.data.recovery import load_processed_record
from arm_rc_ctrl.data.samples import load_samples
from arm_rc_ctrl.data.synthetic import synthetic_manual_take_log
from arm_rc_ctrl.experiments.repetition_numerics import require_recipe_dataset
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageRoot

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "configs"
GOAL_Q = (0.8, 0.4)


@dataclass(frozen=True)
class Workspace:
    """A store outside the repository plus a fake records root holding the fixture configs."""

    store: StorageRoot
    records_root: Path
    scenario: Path
    derive: Path


@dataclass(frozen=True)
class TakeSettings:
    """Shape of a synthetic take and the defect it carries, if any."""

    hold_s: float = 0.5
    move_s: float = 0.6
    dwell_s: float = 0.3
    gap_at_s: float | None = None
    jump_at_s: float | None = None
    start_offset_rad: float = 0.0


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    """Build the store and the fake records root with copies of the fixture configs."""
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
    return Workspace(store, records_root, scenario, derive)


DEFAULT_SETTINGS = TakeSettings()


def _log(scenario: Path, path: Path, settings: TakeSettings = DEFAULT_SETTINGS) -> Path:
    config = load_manual_scenario(scenario)
    log = synthetic_manual_take_log(
        config,
        goal_q=GOAL_Q,
        hold_s=settings.hold_s,
        move_s=settings.move_s,
        dwell_s=settings.dwell_s,
        gap_at_s=settings.gap_at_s,
        jump_at_s=settings.jump_at_s,
        start_offset_rad=settings.start_offset_rad,
    )
    log.save(path)
    return path


def _import(workspace: Workspace, log_file: Path, take: int = 1) -> ManualImportResult:
    return import_manual_take(
        log_file,
        workspace.scenario,
        store=workspace.store,
        records_root=workspace.records_root,
        session="manual-fixture-session",
        take=take,
        license_label="proprietary",
        access="private",
        exploratory=True,
    )


def test_import_keeps_the_log_unchanged_and_registers_its_record(workspace: Workspace, tmp_path: Path) -> None:
    """The take's bytes, timing summary, reset posture, and recorder metadata are bound into an immutable record."""
    store, records_root, scenario = workspace.store, workspace.records_root, workspace.scenario
    log_file = _log(scenario, tmp_path / "reach_003.sklog.npz")
    result = _import(workspace, log_file, take=3)
    record = result.record
    assert isinstance(record, ManualTakeRecord)
    assert record.artifact.kind == "raw"
    assert record.artifact.payload.sha256 == sha256_file(log_file) == sha256_file(result.payload_file)
    assert record.take == 3
    assert record.session == "manual-fixture-session"
    assert record.source_file == "reach_003.sklog.npz"
    assert record.sampling.clock == "wall"
    assert set(record.sampling.units) == {"t", "q", "tip", "nominal_time"}
    assert record.n_frames == 141  # 1.4 s at 100 Hz plus the t = 0 frame
    assert record.duration_s == pytest.approx(1.4)
    assert record.reset_q == load_manual_scenario(scenario).task.initial_q
    assert record.acquisition.clock == "wall-clock"
    assert record.acquisition.ticks == 140
    assert record.display.history_takes == (1, 2)
    assert record.scenario.config_path == "configs/tasks/manual_fixture.toml"
    assert not result.resumed
    assert load_record(result.record_file, ManualTakeRecord) == record
    assert load_catalog(records_root / "data" / "catalog.toml").find(record.artifact.artifact_id) is not None

    take = load_manual_take(store, record)
    original = StateLog.load(log_file)
    assert np.array_equal(take.q, original.channel("q"))
    assert np.array_equal(take.times, original.times)
    assert np.array_equal(take.tip, original.channel("tip"))

    again = _import(workspace, log_file, take=3)
    assert again.resumed
    assert again.record == record
    assert len(list((records_root / "data" / "records" / "raw").glob("*.toml"))) == 1


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("no_acquisition", "acquisition"),
        ("time_offset", "t = 0"),
        ("non_monotonic", "increasing"),
        ("missing_tip", "tip"),
        ("wrong_dof", "joints"),
    ],
)
def test_malformed_logs_are_refused_before_anything_is_written(
    workspace: Workspace, tmp_path: Path, defect: str, message: str
) -> None:
    """Structural defects reject the take with a clear reason and leave the store and records untouched."""
    store, records_root, scenario = workspace.store, workspace.records_root, workspace.scenario
    config = load_manual_scenario(scenario)
    log = synthetic_manual_take_log(config, goal_q=GOAL_Q, hold_s=0.5, move_s=0.6, dwell_s=0.3)
    log_file = tmp_path / "bad.sklog.npz"
    if defect == "no_acquisition":
        del log.extra["acquisition"]
        log.save(log_file)
    else:
        times = log.times.copy()
        q = log.channel("q").copy()
        tip = log.channel("tip").copy()
        nominal = log.channel("nominal_time").copy()
        if defect == "time_offset":
            times += 0.5
        elif defect == "non_monotonic":
            times[10] = times[12]
        elif defect == "wrong_dof":
            q = np.concatenate([q, q[:, :1]], axis=1)
        meta = dict(log.channel_meta)
        if defect == "missing_tip":
            meta.pop("tip")
        rebuilt = StateLog(log.build_skeleton(), producer=log.producer, channel_meta=meta, extra=log.extra)
        for k in range(len(times)):
            channels = {"q": q[k], "nominal_time": nominal[k]}
            if defect != "missing_tip":
                channels["tip"] = tip[k]
            rebuilt.record(float(times[k]), **channels)
        rebuilt.save(log_file)
    with pytest.raises(ManualTakeError, match=message):
        _import(workspace, log_file)
    assert not list((store.root / "raw").glob("raw-*"))
    assert not (records_root / "data").exists()


def test_a_clean_take_is_accepted_with_its_measurements(workspace: Workspace, tmp_path: Path) -> None:
    """Exact start, no gaps, bounded speeds, an anchored hold, and a full final dwell: accepted, fully measured."""
    store, scenario, derive_file = workspace.store, workspace.scenario, workspace.derive
    config = load_manual_scenario(scenario)
    log_file = _log(scenario, tmp_path / "good.sklog.npz")
    take = load_manual_take(store, _import(workspace, log_file).record)
    assessment = assess_take(take, config, load_manual_derive_config(derive_file))
    assert assessment.accepted, assessment.problems
    assert assessment.problems == ()
    assert assessment.start.ok
    assert assessment.start.max_deviation_rad == 0.0
    assert assessment.raw_timing.n_frames == 141
    assert assessment.raw_timing.max_interval_s == pytest.approx(0.01)
    assert assessment.raw_timing.late_frames == 0
    assert all(v < config.acquisition.velocity_bound_rad_s for v in assessment.raw_timing.max_raw_speed_rad_s)
    smoothing = assessment.smoothing
    assert smoothing is not None
    assert smoothing.ok
    assert smoothing.start_shift_rad == 0.0
    assert smoothing.onset_s == pytest.approx(0.5)
    assert smoothing.anchored_from_s == pytest.approx(0.3)
    dwell = assessment.dwell
    assert dwell is not None
    assert dwell.ok
    assert dwell.final_samples >= config.dwell_min_samples
    motion = assessment.motion
    assert motion is not None
    assert motion.hold_end_s == pytest.approx(0.5)
    assert 0.3 < motion.movement_duration_s < 0.7  # the reach settles inside the tolerance early
    assert motion.final_endpoint_error_m < config.task.tolerance
    samples = assessment.samples
    assert samples is not None
    assert samples.t[0] == 0.0
    assert np.array_equal(samples.q[0], np.asarray(config.task.initial_q))
    assert samples.n_samples == 141  # the whole recording, hold included
    codes = samples.phase.tolist()
    assert codes[0] == MANUAL_PHASE_CODES["hold"]
    assert codes[-1] == MANUAL_PHASE_CODES["dwell"]
    assert codes == sorted(codes)  # hold, then move, then dwell


@pytest.mark.parametrize(
    ("settings", "reason"),
    [
        (TakeSettings(gap_at_s=0.7), "gap"),
        (TakeSettings(jump_at_s=0.7), "increment"),
        (TakeSettings(start_offset_rad=1e-3), "reset"),
        (TakeSettings(dwell_s=0.05), "dwell"),
        (TakeSettings(move_s=0.0, dwell_s=0.0), "movement"),
        (TakeSettings(hold_s=0.05), "start shift"),
    ],
)
def test_rule_violations_are_rejected_with_their_reason(
    workspace: Workspace, tmp_path: Path, settings: TakeSettings, reason: str
) -> None:
    """Every offline rule names what it measured; a rejected take keeps its raw record and payload."""
    config = load_manual_scenario(workspace.scenario)
    log_file = _log(workspace.scenario, tmp_path / "flawed.sklog.npz", settings)
    take = load_manual_take(workspace.store, _import(workspace, log_file).record)
    assessment = assess_take(take, config, load_manual_derive_config(workspace.derive))
    assert not assessment.accepted
    assert any(reason in problem for problem in assessment.problems), assessment.problems
    assert (workspace.records_root / "data" / "records" / "raw").exists()


def test_derivation_writes_the_full_dataset_once_and_dispatches_its_schema(
    workspace: Workspace, tmp_path: Path
) -> None:
    """An accepted take becomes a processed record with the measured dwell; rejected takes derive nothing."""
    store, records_root, scenario, derive_file = (
        workspace.store,
        workspace.records_root,
        workspace.scenario,
        workspace.derive,
    )
    imported = _import(workspace, _log(scenario, tmp_path / "good.sklog.npz"))
    result = derive_manual_dataset(
        imported.record_file, scenario, derive_file, store=store, records_root=records_root, exploratory=True
    )
    record = result.record
    assert isinstance(record, ManualDatasetRecord)
    assert record.artifact.kind == "processed"
    assert record.phases == MANUAL_PHASE_CODES
    assert record.artifact.origin.sources == (imported.record.artifact.artifact_id,)
    assert record.take == 1
    assert record.dwell.ok
    assert record.start.ok
    assert record.smoothing.start_shift_rad == 0.0
    assert record.preprocessing.smoothing == "butterworth-zero-phase-hold-anchored"
    assert record.n_samples == 141
    samples = load_samples(result.payload_file)
    record.check_samples(samples)
    assert samples.t[-1] == pytest.approx(1.4)
    loaded = load_processed_record(result.record_file)
    assert isinstance(loaded, ManualDatasetRecord)
    with pytest.raises(TypeError, match="M3MAN-005"):
        require_recipe_dataset(loaded)  # repetition recipes never bind a manual take
    assert not result.resumed

    again = derive_manual_dataset(
        imported.record_file, scenario, derive_file, store=store, records_root=records_root, exploratory=True
    )
    assert again.resumed
    assert again.record.artifact.artifact_id == record.artifact.artifact_id

    rejected = _import(workspace, _log(scenario, tmp_path / "short.sklog.npz", TakeSettings(dwell_s=0.05)), take=2)
    with pytest.raises(ManualTakeError, match="dwell"):
        derive_manual_dataset(
            rejected.record_file, scenario, derive_file, store=store, records_root=records_root, exploratory=True
        )
    assert len(list((records_root / "data" / "records" / "processed").glob("*.toml"))) == 1


def test_the_same_bytes_under_another_attempt_are_a_duplicate_at_import(workspace: Workspace, tmp_path: Path) -> None:
    """A copied file is never lent the original attempt's record; the same identity resumes instead."""
    log_file = _log(workspace.scenario, tmp_path / "reach_001.sklog.npz")
    first = _import(workspace, log_file, take=1)
    copy = tmp_path / "reach_002.sklog.npz"
    shutil.copyfile(log_file, copy)
    with pytest.raises(DuplicatePayloadError, match="attempt 1") as info:
        _import(workspace, copy, take=2)
    assert info.value.existing.artifact.artifact_id == first.record.artifact.artifact_id
    assert _import(workspace, log_file, take=1).resumed


def test_registration_can_be_deferred_and_is_idempotent(workspace: Workspace, tmp_path: Path) -> None:
    """A batch registers its records once all takes are processed; registering twice changes nothing."""
    log_file = _log(workspace.scenario, tmp_path / "reach_001.sklog.npz")
    result = import_manual_take(
        log_file,
        workspace.scenario,
        store=workspace.store,
        records_root=workspace.records_root,
        session="manual-fixture-session",
        take=1,
        license_label="proprietary",
        access="private",
        exploratory=True,
        register=False,
    )
    assert result.payload_file.exists()
    assert not result.record_file.exists()
    assert not (workspace.records_root / "data").exists()
    written = register_manual_records(workspace.records_root, [result.record])
    assert written == [result.record_file]
    assert load_record(result.record_file, ManualTakeRecord) == result.record
    assert register_manual_records(workspace.records_root, [result.record]) == [result.record_file]
    assert load_catalog(workspace.records_root / "data" / "catalog.toml").find(result.record.artifact.artifact_id)


def test_an_archive_with_inconsistent_rows_is_refused_as_malformed(workspace: Workspace, tmp_path: Path) -> None:
    """Rows that disagree with the timestamps are refused before the state-log loader can fail on them."""
    log_file = _log(workspace.scenario, tmp_path / "rows.sklog.npz")
    with np.load(log_file, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["tip"] = arrays["tip"][:-3]
    np.savez_compressed(log_file, **arrays)
    with pytest.raises(ManualTakeError, match="rows"):
        _import(workspace, log_file)
    assert not (workspace.records_root / "data").exists()

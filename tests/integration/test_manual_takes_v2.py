# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-013 (I10, I11, I13): 50 Hz takes that move at once are reconstructed on the 0.01 s grid and assessed."""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from skelarm import StateLog

from arm_rc_ctrl.config import ConfigError, from_mapping, to_mapping
from arm_rc_ctrl.data.manual import (
    MANUAL_PHASE_CODES,
    ManualDatasetRecord,
    ManualTake,
    TakeAssessment,
    assess_take,
    derive_manual_dataset,
    import_manual_take,
    load_manual_derive_config,
    load_manual_take,
    reconstruct_on_grid,
)
from arm_rc_ctrl.data.manual_scenario import ManualScenarioConfig, load_manual_scenario, manual_endpoint_positions
from arm_rc_ctrl.data.recovery import load_processed_record
from arm_rc_ctrl.data.samples import load_samples
from arm_rc_ctrl.data.synthetic import synthetic_manual_take_log
from arm_rc_ctrl.experiments.repetition_numerics import require_recipe_dataset
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageRoot

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
SCENARIO_V2 = REPO_ROOT / "configs" / "tasks" / "task_1a_manual_v2.toml"
DERIVE_V2 = REPO_ROOT / "configs" / "preprocessing" / "manual_v2.toml"
DERIVE_V1 = REPO_ROOT / "configs" / "preprocessing" / "manual_v1.toml"
ACQUISITION_S = 0.02
GRID_S = 0.01
BOUND = 1e-12  # max_start_shift_rad of manual_v2.toml
REFLECTED = "butterworth-zero-phase-first-sample-reflected"


@dataclass(frozen=True)
class Workspace:
    """A store outside the repository plus a fake records root holding copies of the v2 configurations."""

    store: StorageRoot
    records_root: Path
    takes: Path
    scenario: Path
    derive: Path
    derive_v1: Path
    config: ManualScenarioConfig


@dataclass(frozen=True)
class Shape:
    """A synthetic take: 50 Hz, only the t = 0 frame at rest, a 5 mrad pre-roll wobble, a 1 s reach, a 2 s dwell."""

    sample_period_s: float = ACQUISITION_S
    hold_s: float = ACQUISITION_S
    move_s: float = 1.0
    dwell_s: float = 2.0
    preroll_amplitude_rad: float = 0.005
    jitter_s: float = 0.0
    seed: int = 0
    gap_at_s: float | None = None
    gap_s: float = 0.05


DEFAULT_SHAPE = Shape()


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    """Build the store, the records root with the v2 configurations (and the v1 derivation), and a take folder."""
    root = tmp_path / "store"
    root.mkdir()
    records_root = tmp_path / "repo"
    (records_root / "configs" / "tasks").mkdir(parents=True)
    (records_root / "configs" / "preprocessing").mkdir(parents=True)
    scenario = records_root / "configs" / "tasks" / SCENARIO_V2.name
    shutil.copyfile(SCENARIO_V2, scenario)
    derive = records_root / "configs" / "preprocessing" / DERIVE_V2.name
    shutil.copyfile(DERIVE_V2, derive)
    derive_v1 = records_root / "configs" / "preprocessing" / DERIVE_V1.name
    shutil.copyfile(DERIVE_V1, derive_v1)
    takes = tmp_path / "takes"
    takes.mkdir()
    store = StorageRoot(root, repositories=(REPO_ROOT,))
    return Workspace(store, records_root, takes, scenario, derive, derive_v1, load_manual_scenario(scenario))


def goal_posture(config: ManualScenarioConfig) -> tuple[float, float]:
    """Joint angles that put the endpoint on the target, on the elbow branch of the reset posture."""
    l1 = config.robot.links[0].length
    l2 = config.robot.links[1].length
    x, y = config.task.target
    elbow = math.acos((x * x + y * y - l1 * l1 - l2 * l2) / (2.0 * l1 * l2))
    shoulder = math.atan2(y, x) - math.atan2(l2 * math.sin(elbow), l1 + l2 * math.cos(elbow))
    return shoulder, elbow


def _log(ws: Workspace, name: str, shape: Shape = DEFAULT_SHAPE) -> Path:
    log = synthetic_manual_take_log(
        ws.config,
        goal_q=goal_posture(ws.config),
        sample_period_s=shape.sample_period_s,
        hold_s=shape.hold_s,
        move_s=shape.move_s,
        dwell_s=shape.dwell_s,
        preroll_amplitude_rad=shape.preroll_amplitude_rad,
        jitter_s=shape.jitter_s,
        seed=shape.seed,
        gap_at_s=shape.gap_at_s,
        gap_s=shape.gap_s,
    )
    path = ws.takes / name
    log.save(path)
    return path


def _load(ws: Workspace, log_file: Path, take: int = 1) -> ManualTake:
    result = import_manual_take(
        log_file,
        ws.scenario,
        store=ws.store,
        records_root=ws.records_root,
        session="manual-v2-session",
        take=take,
        license_label="proprietary",
        access="private",
        exploratory=True,
    )
    return load_manual_take(ws.store, result.record)


def _assess(ws: Workspace, shape: Shape = DEFAULT_SHAPE, take: int = 1) -> tuple[ManualTake, TakeAssessment]:
    loaded = _load(ws, _log(ws, f"reach_{take:03d}.sklog.npz", shape), take)
    return loaded, assess_take(loaded, ws.config, load_manual_derive_config(ws.derive))


def test_the_goal_posture_reaches_the_target(workspace: Workspace) -> None:
    """The synthetic reaches end on the configured target."""
    tip = manual_endpoint_positions(workspace.config, np.asarray([goal_posture(workspace.config)]))
    assert np.allclose(tip[0], workspace.config.task.target, rtol=0.0, atol=1e-12)


def test_movement_immediately_after_the_first_sample_is_accepted(workspace: Workspace) -> None:
    """Natural pre-roll from the second 50 Hz sample on passes, and the first processed sample is the reset posture.

    The same take fails the v1 hold-anchored derivation on its start shift, as every pilot-1 take did.
    """
    take, assessment = _assess(workspace)
    assert assessment.accepted, assessment.problems
    assert take.record.acquisition.sample_period_s == ACQUISITION_S
    assert np.all(take.q[0] == np.asarray(workspace.config.task.initial_q))
    assert np.any(take.q[1] != take.q[0])  # departs at the second raw sample
    timing = assessment.raw_timing
    assert timing.acquisition_period_s == ACQUISITION_S
    assert timing.median_interval_s == pytest.approx(ACQUISITION_S)
    smoothing = assessment.smoothing
    assert smoothing is not None
    assert smoothing.method == REFLECTED
    assert smoothing.extension_s == 5.0
    assert smoothing.hold_margin_s is None
    assert smoothing.anchored_from_s is None
    assert smoothing.start_shift_rad <= BOUND
    assert smoothing.onset_s == pytest.approx(GRID_S)  # the first grid sample after t = 0 already departs
    samples = assessment.samples
    assert samples is not None
    assert float(np.max(np.abs(samples.q[0] - np.asarray(workspace.config.task.initial_q)))) <= BOUND
    assert np.allclose(np.diff(samples.t), GRID_S, rtol=0.0, atol=1e-12)
    assert samples.t[-1] <= take.times[-1] < samples.t[-1] + GRID_S
    motion = assessment.motion
    assert motion is not None
    assert motion.hold_end_s == pytest.approx(float(take.times[1]))
    codes = samples.phase.tolist()
    assert codes[0] == MANUAL_PHASE_CODES["hold"]
    assert codes[-1] == MANUAL_PHASE_CODES["dwell"]
    assert codes == sorted(codes)
    assert set(codes) == set(MANUAL_PHASE_CODES.values())

    hold_anchored = assess_take(take, workspace.config, load_manual_derive_config(workspace.derive_v1))
    assert not hold_anchored.accepted
    assert any("start shift" in problem for problem in hold_anchored.problems), hold_anchored.problems


def test_jittered_50hz_timestamps_are_reconstructed_on_the_100hz_grid(workspace: Workspace) -> None:
    """Grid values interpolate the actual timestamps; nominal tick times would misplace the motion."""
    jittered = _load(workspace, _log(workspace, "reach_001.sklog.npz", Shape(jitter_s=0.005, seed=3)), take=1)
    clean = _load(workspace, _log(workspace, "reach_002.sklog.npz"), take=2)
    grid, q_grid = reconstruct_on_grid(jittered.times, jittered.q, GRID_S, "linear")
    assert grid[0] == 0.0
    assert np.allclose(np.diff(grid), GRID_S, rtol=0.0, atol=1e-12)
    assert grid[-1] <= jittered.times[-1] < grid[-1] + GRID_S
    assert np.max(np.abs(np.diff(jittered.times) - ACQUISITION_S)) > 0.004
    for joint in range(q_grid.shape[1]):
        assert np.array_equal(q_grid[:, joint], np.interp(grid, jittered.times, jittered.q[:, joint]))
    _, reference = reconstruct_on_grid(clean.times, clean.q, GRID_S, "linear")
    n = min(reference.shape[0], q_grid.shape[0])
    actual_error = float(np.max(np.abs(q_grid[:n] - reference[:n])))
    nominal = np.column_stack(
        [np.interp(grid, jittered.nominal_times, jittered.q[:, joint]) for joint in range(q_grid.shape[1])]
    )
    nominal_error = float(np.max(np.abs(nominal[:n] - reference[:n])))
    assert actual_error < 1e-3
    assert nominal_error > 3.0 * actual_error
    assessment = assess_take(jittered, workspace.config, load_manual_derive_config(workspace.derive))
    assert assessment.accepted, assessment.problems


@pytest.mark.parametrize(("gap_s", "rejected"), [(0.04, False), (0.042, True)])
def test_the_gap_limit_is_three_acquisition_periods_of_actual_time(
    workspace: Workspace, gap_s: float, *, rejected: bool
) -> None:
    """An interval of exactly 0.06 s (one period plus a 40 ms stall) passes; 0.062 s is a missing interval."""
    _, assessment = _assess(workspace, Shape(gap_at_s=2.5, gap_s=gap_s))
    assert assessment.raw_timing.max_interval_s == pytest.approx(ACQUISITION_S + gap_s)
    gaps = [problem for problem in assessment.problems if "sample gap" in problem]
    assert bool(gaps) == rejected
    assert assessment.accepted != rejected, assessment.problems


@pytest.mark.parametrize(("dwell_s", "duration_s", "short"), [(0.68, 1.0, False), (0.66, 0.98, True)])
def test_the_minimum_duration_is_actual_time_from_the_first_sample(
    workspace: Workspace, dwell_s: float, duration_s: float, *, short: bool
) -> None:
    """A recording of exactly min_duration_s (50 intervals at 50 Hz) is long enough; one interval less is not."""
    _, assessment = _assess(workspace, Shape(move_s=0.3, dwell_s=dwell_s))
    assert assessment.raw_timing.duration_s == pytest.approx(duration_s)
    assert assessment.raw_timing.n_frames == round(duration_s / ACQUISITION_S) + 1
    too_short = [problem for problem in assessment.problems if "min_duration_s" in problem]
    assert bool(too_short) == short


def test_a_take_recorded_at_100hz_is_rejected_under_the_50hz_rules(workspace: Workspace) -> None:
    """The recorder's declared sample period must be the configured acquisition period."""
    take, assessment = _assess(workspace, Shape(sample_period_s=GRID_S, hold_s=GRID_S))
    assert take.record.acquisition.sample_period_s == GRID_S
    assert not assessment.accepted
    assert any("100 Hz" in problem and "50 Hz" in problem for problem in assessment.problems), assessment.problems


def _truncated(take: ManualTake, end_s: float, path: Path) -> Path:
    """The take cut so that its last raw frame, still at rest on the target, is stamped ``end_s``."""
    keep = int(np.searchsorted(take.times, end_s - 1e-9))
    times = np.append(take.times[:keep], end_s)
    q = np.vstack([take.q[:keep], take.q[-1]])
    tip = np.vstack([take.tip[:keep], take.tip[-1]])
    nominal = np.arange(times.shape[0], dtype=np.float64) * ACQUISITION_S
    log = StateLog(
        take.log.build_skeleton(),
        producer=take.log.producer,
        channel_meta=dict(take.log.channel_meta),
        extra=take.log.extra,
    )
    for k in range(times.shape[0]):
        log.record(float(times[k]), q=q[k], tip=tip[k], nominal_time=np.asarray(nominal[k], dtype=np.float64))
    log.save(path)
    return path


def test_the_final_dwell_is_one_actual_second_at_50hz_acquisition(workspace: Workspace) -> None:
    """Ending the recording 1.00 s after the dwell starts gives 101 grid samples and passes; 0.99 s gives 100."""
    derive = load_manual_derive_config(workspace.derive)
    long_take, long = _assess(workspace, Shape(dwell_s=3.0))
    assert long.accepted, long.problems
    assert long.dwell is not None
    start = long.dwell.start_s
    assert start is not None
    outcomes: dict[float, TakeAssessment] = {}
    for take, duration in ((2, 1.0), (3, 0.99)):
        cut = _truncated(long_take, start + duration, workspace.takes / f"reach_{take:03d}.sklog.npz")
        outcomes[duration] = assess_take(_load(workspace, cut, take), workspace.config, derive)
    exact, short = outcomes[1.0], outcomes[0.99]
    assert exact.accepted, exact.problems
    assert exact.dwell is not None
    assert exact.dwell.final_samples == 101
    assert exact.dwell.final_duration_s == pytest.approx(1.0)
    assert exact.dwell.start_s == pytest.approx(start)
    assert not short.accepted
    assert short.dwell is not None
    assert short.dwell.final_samples == 100
    assert any("final dwell of 0.99 s" in problem for problem in short.problems), short.problems


def test_v2_derivation_writes_schema_2_records_that_keep_the_schemas_apart(workspace: Workspace) -> None:
    """The processed record of a v2 derivation is manual schema 2; its raw take record stays schema 1."""
    imported = import_manual_take(
        _log(workspace, "reach_001.sklog.npz"),
        workspace.scenario,
        store=workspace.store,
        records_root=workspace.records_root,
        session="manual-v2-session",
        take=1,
        license_label="proprietary",
        access="private",
        exploratory=True,
    )
    assert imported.record.manual_schema_version == 1
    result = derive_manual_dataset(
        imported.record_file,
        workspace.scenario,
        workspace.derive,
        store=workspace.store,
        records_root=workspace.records_root,
        exploratory=True,
    )
    record = result.record
    assert record.manual_schema_version == 2
    assert record.preprocessing.resample_period_s == GRID_S
    assert record.preprocessing.smoothing == REFLECTED
    assert record.preprocessing.smoothing_params["extension_s"] == 5.0
    assert record.preprocessing.smoothing_params["max_start_shift_rad"] == BOUND
    assert record.smoothing.extension_s == 5.0
    assert record.smoothing.start_shift_rad <= BOUND
    assert record.raw_timing.acquisition_period_s == ACQUISITION_S
    record.check_samples(load_samples(result.payload_file))
    loaded = load_processed_record(result.record_file)
    assert isinstance(loaded, ManualDatasetRecord)
    assert loaded == record
    with pytest.raises(TypeError, match="M3MAN-005"):
        require_recipe_dataset(loaded)

    as_v1 = to_mapping(record)
    as_v1["manual_schema_version"] = 1
    with pytest.raises(ConfigError, match="hold"):
        from_mapping(as_v1, ManualDatasetRecord)
    unknown = to_mapping(record)
    unknown["manual_schema_version"] = 3
    with pytest.raises(ConfigError, match="unsupported manual_schema_version"):
        from_mapping(unknown, ManualDatasetRecord)

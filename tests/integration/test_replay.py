# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-022/M1-023: direct-replay PD and computed-torque runs terminate normally, log everything, respect limits."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from arm_rc_ctrl.config import load_config
from arm_rc_ctrl.controllers.tracking import TrackerConfig
from arm_rc_ctrl.data.preprocess import preprocess_demonstration
from arm_rc_ctrl.data.records import ProcessedDatasetRecord, RawDemonstrationRecord, load_record
from arm_rc_ctrl.data.samples import SampleSet
from arm_rc_ctrl.experiments.disturbances import ForcePulse
from arm_rc_ctrl.experiments.replay import main, run_replay
from arm_rc_ctrl.experiments.run_record import REQUIRED_ARRAYS
from arm_rc_ctrl.metrics.report import report_from_json
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import ScenarioConfig, load_scenario
from arm_rc_ctrl.storage import StorageRoot

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
RAW_RECORD = REPO_ROOT / "tests" / "fixtures" / "records" / "raw-20260830-287036d83d46.toml"
RAW_LOG = REPO_ROOT / "tests" / "fixtures" / "raw" / "demo.sklog.npz"
SCENARIO = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
PREPROCESS = REPO_ROOT / "configs" / "preprocessing" / "default.toml"
PD = REPO_ROOT / "configs" / "controllers" / "pd.toml"
CT = REPO_ROOT / "configs" / "controllers" / "computed_torque.toml"
FIXED_TIME = datetime(2026, 8, 30, 10, 0, 0, tzinfo=UTC)
_CLOCK = [0]


def _now() -> datetime:
    """Distinct timestamps keep content-addressed run IDs unique within the shared store."""
    _CLOCK[0] += 1
    return FIXED_TIME + timedelta(minutes=_CLOCK[0])


@pytest.fixture(scope="module")
def dataset(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[StorageRoot, SampleSet, ProcessedDatasetRecord, ScenarioConfig]:
    """A processed dataset built from the committed raw fixture in a module-scoped store."""
    base = tmp_path_factory.mktemp("replay")
    root = base / "store"
    root.mkdir()
    store = StorageRoot(root, repositories=(REPO_ROOT,))
    raw = load_record(RAW_RECORD, RawDemonstrationRecord)
    store.path(raw.artifact.payload.uri, mode="write").write_bytes(RAW_LOG.read_bytes())
    records = base / "repo"
    (records / "data" / "records" / "processed").mkdir(parents=True)
    result = preprocess_demonstration(
        RAW_RECORD, SCENARIO, PREPROCESS, store=store, records_root=records, exploratory=True, now=FIXED_TIME
    )
    return store, result.samples, result.record, load_scenario(SCENARIO)


@pytest.mark.parametrize(("controller", "method"), [(PD, "replay+pd"), (CT, "replay+computed_torque")])
def test_replay_terminates_normally_logs_channels_and_respects_limits(
    dataset: tuple[StorageRoot, SampleSet, ProcessedDatasetRecord, ScenarioConfig], controller: Path, method: str
) -> None:
    """Both trackers complete the demonstration, record every channel, and never exceed the torque limits."""
    store, samples, record, scenario = dataset
    artifact_id = record.artifact.artifact_id
    tracker = load_config(controller, TrackerConfig)
    result = run_replay(scenario, SCENARIO, record, samples, tracker, store=store, exploratory=True, now=_now())
    summary, arrays = result.summary, result.run.arrays.arrays

    assert summary.method == method
    assert summary.termination.kind == "completed"
    assert summary.outcome.criteria["completed"] is True
    assert set(summary.outcome.criteria) == {"completed", "dwell_in_tolerance", "dwell_stationary"}
    assert set(arrays) == {*REQUIRED_ARRAYS, "tau_applied"}  # ext_force only exists under a disturbance
    assert arrays["t"].shape[0] == samples.n_samples
    assert np.allclose(arrays["t"], samples.t)
    assert np.array_equal(arrays["q_desired"], samples.q)  # the reference is the demonstration itself
    assert np.array_equal(arrays["dq_desired_raw"], arrays["dq_desired"])
    assert np.allclose(arrays["tracking_error"], arrays["q_desired"] - arrays["q"])
    limits = np.asarray(scenario.limits.torque)
    assert np.all(np.abs(arrays["tau_applied"]) <= limits + 1e-12)
    saturated = np.any(np.abs(arrays["tau_requested"]) >= limits, axis=1).astype(np.int64)
    assert np.array_equal(arrays["saturation"], saturated)
    assert np.all(np.abs(arrays["dq"]) <= np.asarray(scenario.limits.velocity))
    assert np.all(np.hypot(arrays["tip"][:, 0], arrays["tip"][:, 1]) <= scenario.limits.endpoint_radius)
    assert result.pointer.artifact.origin.sources == (artifact_id,)
    assert [a.uri for a in summary.provenance.artifacts] == [record.artifact.payload.uri]
    assert summary.provenance.config["interpolation"] == record.preprocessing.interpolation
    assert cast("dict[str, object]", summary.provenance.config["tracker"])["kp"] == list(tracker.kp)

    report = result.report
    assert report.move_coverage == 1.0
    assert report.dwell_coverage == 1.0
    assert report.joint_rmse is not None
    assert report.joint_rmse.aggregate < 0.2  # placeholder gains; M1-025 tunes them
    assert report.dwell is not None
    assert report.effort is not None
    assert report.effort.samples == samples.n_samples


def test_computed_torque_shows_the_dynamics_feedforward(
    dataset: tuple[StorageRoot, SampleSet, ProcessedDatasetRecord, ScenarioConfig],
) -> None:
    """With an exact model the computed-torque replay tracks much better than PD and its torque differs."""
    store, samples, record, scenario = dataset
    pd = run_replay(
        scenario,
        SCENARIO,
        record,
        samples,
        load_config(PD, TrackerConfig),
        store=store,
        exploratory=True,
        now=FIXED_TIME,
    )
    ct = run_replay(
        scenario,
        SCENARIO,
        record,
        samples,
        load_config(CT, TrackerConfig),
        store=store,
        exploratory=True,
        now=FIXED_TIME,
    )
    assert ct.report.joint_rmse is not None
    assert pd.report.joint_rmse is not None
    assert ct.report.joint_rmse.aggregate < 0.25 * pd.report.joint_rmse.aggregate
    assert not np.allclose(ct.run.arrays.arrays["tau_requested"], pd.run.arrays.arrays["tau_requested"])
    assert ct.pointer.artifact.artifact_id != pd.pointer.artifact.artifact_id


def test_velocity_limit_violation_terminates_early(
    dataset: tuple[StorageRoot, SampleSet, ProcessedDatasetRecord, ScenarioConfig],
) -> None:
    """A scenario with an unreachable velocity bound stops with a typed limit violation and a truncated log."""
    store, samples, record, scenario = dataset
    strict = dataclasses.replace(scenario, limits=dataclasses.replace(scenario.limits, velocity=(0.3, 0.3)))
    result = run_replay(
        strict, SCENARIO, record, samples, load_config(PD, TrackerConfig), store=store, exploratory=True, now=FIXED_TIME
    )
    termination = result.summary.termination
    assert termination.kind == "limit_violation"
    assert termination.limit == "joint_velocity"
    assert termination.bound == 0.3
    assert result.summary.outcome.success is False
    assert result.summary.outcome.criteria == {
        "completed": False,
        "dwell_in_tolerance": False,
        "dwell_stationary": False,
    }
    assert result.run.arrays.n_samples == termination.step
    assert result.report.move_coverage < 1.0
    assert result.report.termination_kind == "limit_violation"
    assert result.report.effort is None or result.report.effort.samples == result.run.arrays.n_samples


def test_initial_posture_override_and_reruns_are_content_addressed(
    dataset: tuple[StorageRoot, SampleSet, ProcessedDatasetRecord, ScenarioConfig],
) -> None:
    """A different initial posture yields a different run; identical inputs map to an existing run ID."""
    store, samples, record, scenario = dataset
    tracker = load_config(PD, TrackerConfig)
    shifted = run_replay(
        scenario,
        SCENARIO,
        record,
        samples,
        tracker,
        store=store,
        exploratory=True,
        now=FIXED_TIME,
        initial_q=(0.35, 0.55),
    )
    assert np.allclose(shifted.run.arrays.arrays["q"][0], [0.35, 0.55])
    with pytest.raises(FileExistsError, match="runs are immutable"):
        run_replay(
            scenario,
            SCENARIO,
            record,
            samples,
            tracker,
            store=store,
            exploratory=True,
            now=FIXED_TIME,
            initial_q=(0.35, 0.55),
        )


def test_dataset_from_another_scenario_is_refused(
    dataset: tuple[StorageRoot, SampleSet, ProcessedDatasetRecord, ScenarioConfig], tmp_path: Path
) -> None:
    """A same-dof dataset derived under a different scenario file cannot be replayed silently."""
    store, samples, record, scenario = dataset
    other = tmp_path / "other_scenario.toml"
    other.write_text(SCENARIO.read_text().replace('name = "pd-reach-fixture"', 'name = "another-task"'))
    tracker = load_config(PD, TrackerConfig)
    with pytest.raises(ValueError, match="was derived under scenario digest"):
        run_replay(load_scenario(other), other, record, samples, tracker, store=store, exploratory=True, now=_now())
    drifted = SampleSet.from_arrays({**samples.arrays(), "q": samples.q + 1e-6})
    with pytest.raises(ValueError, match="samples do not match the record"):
        run_replay(scenario, SCENARIO, record, drifted, tracker, store=store, exploratory=True, now=_now())


def test_command_line_entry_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI loads the dataset through the store, runs, and writes the report JSON."""
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
    monkeypatch.setenv("ARM_RC_CTRL_STORAGE_ROOT", str(root))
    report_file = tmp_path / "report.json"
    assert (
        main(
            [
                "--scenario",
                str(SCENARIO),
                "--dataset",
                str(processed.record_file),
                "--controller",
                str(PD),
                "--exploratory",
                "--no-pointer",
                "--report",
                str(report_file),
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["termination"] == "completed"
    assert printed["run_dir"].startswith("runs/run-")
    report = report_from_json(report_file.read_text())
    assert report.run_id == printed["run_id"]
    assert report.method == "replay+pd"


def test_endpoint_force_pulse_deflects_the_tip_and_is_recorded(
    dataset: tuple[StorageRoot, SampleSet, ProcessedDatasetRecord, ScenarioConfig],
) -> None:
    """A pulse acts only inside its window, is logged as ``ext_force``, and appears in the run summary."""
    store, samples, record, scenario = dataset
    tracker = load_config(PD, TrackerConfig)
    pulse = ForcePulse(start_s=0.12, duration_s=0.05, force=(0.0, -3.0))
    quiet = run_replay(scenario, SCENARIO, record, samples, tracker, store=store, exploratory=True, now=_now())
    pushed = run_replay(
        scenario, SCENARIO, record, samples, tracker, store=store, exploratory=True, now=_now(), force=pulse
    )
    zero = run_replay(
        scenario,
        SCENARIO,
        record,
        samples,
        tracker,
        store=store,
        exploratory=True,
        now=_now(),
        force=ForcePulse(start_s=0.12, duration_s=0.05, force=(0.0, 0.0)),
    )
    arrays, t = pushed.run.arrays.arrays, pushed.run.arrays.arrays["t"]
    assert "ext_force" not in quiet.run.arrays.arrays
    active = (t >= 0.12) & (t < 0.17)
    assert active.sum() == 5
    assert np.array_equal(arrays["ext_force"][active], np.tile([0.0, -3.0], (5, 1)))
    assert not arrays["ext_force"][~active].any()
    before = t < 0.13  # the first pushed step is integrated after the state was logged
    assert np.array_equal(arrays["q"][before], quiet.run.arrays.arrays["q"][before])
    assert not np.array_equal(arrays["q"][~before], quiet.run.arrays.arrays["q"][~before])
    assert np.abs(arrays["tip"] - quiet.run.arrays.arrays["tip"]).max() > 1e-4
    assert np.array_equal(zero.run.arrays.arrays["q"], quiet.run.arrays.arrays["q"])
    assert pushed.summary.termination.kind == "completed"
    (disturbance,) = pushed.summary.disturbances
    assert disturbance.kind == "endpoint_force_pulse"
    assert (disturbance.start_s, disturbance.end_s) == (0.12, pytest.approx(0.17))
    assert disturbance.parameters == pytest.approx({"fx": 0.0, "fy": -3.0, "magnitude_n": 3.0})
    assert pushed.summary.provenance.config["force"] == {"start_s": 0.12, "duration_s": 0.05, "force": [0.0, -3.0]}
    assert quiet.summary.provenance.config["force"] is None
    assert pushed.summary.arrays["ext_force"].shape == (samples.n_samples, 2)

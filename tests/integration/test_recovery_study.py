# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-012: one recovery formulation study — Optuna plus MLflow mirror, anchors, deterministic resume, CLI."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from arm_rc_ctrl.config import from_mapping
from arm_rc_ctrl.data.derivatives import DerivativeConfig, differentiate
from arm_rc_ctrl.data.normalization import fit_normalization
from arm_rc_ctrl.data.records import (
    CANONICAL_UNITS,
    ArtifactRecord,
    Origin,
    Payload,
    Preprocessing,
    Scenario,
    array_specs,
    load_record,
    make_artifact_id,
    write_record,
)
from arm_rc_ctrl.data.recovery import (
    TASK_PHASE_CODES,
    BaselineCheck,
    CropWindow,
    OnsetAnnotation,
    RecoveryDatasetRecord,
    TaskIntervals,
)
from arm_rc_ctrl.data.samples import SampleSet, save_samples
from arm_rc_ctrl.experiments.esn_study import STUDY_TAG, TRIAL_TAG
from arm_rc_ctrl.experiments.perturbations import load_development_robustness, robustness_scenarios
from arm_rc_ctrl.experiments.recovery_objective import RecoveryTrialEvaluation
from arm_rc_ctrl.experiments.recovery_search import load_recovery_search
from arm_rc_ctrl.experiments.recovery_study import (
    RecoveryStudyResult,
    load_report,
    main,
    report_to_json,
    run_recovery_study,
)
from arm_rc_ctrl.experiments.tracking import MlflowTracker
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import endpoint_positions, load_scenario
from arm_rc_ctrl.storage import StorageRoot

if TYPE_CHECKING:
    from arm_rc_ctrl.experiments.recovery_search import RecoverySearchProtocol

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
SCENARIO_FILE = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
SCENARIO = load_scenario(SCENARIO_FILE)
MODEL = REPO_ROOT / "tests" / "fixtures" / "configs" / "esn_fixture.toml"
DERIVATIVES = DerivativeConfig(method="central")
RAW_SOURCE = "raw-20260830-2a97516c354b"
CREATED = "2026-09-04T10:00:00+00:00"
N = 101
DT = 0.01
TASK = TaskIntervals(move=(0.0, 0.8), dwell=(0.8, 1.0))

_DEVELOPMENT = """
# Fixture development levels (physically meaningless magnitudes; structure mirrors the locked v1 file).
name = "recovery-dev-fixture"
scenario = "{scenario}"
seeds = [11, 12]

[posture]
small_magnitude_rad = 0.05
large_magnitude_rad = 0.1
draws_per_seed = 1

[force]
magnitude_n = 3.0
start_s = 0.3
duration_s = 0.2
directions_deg = [0.0, 90.0]
"""

_PROTOCOL = """
name = "{name}"
scenario = "{scenario}"
model = "{model}"
formulation = "no_augmentation"
budget = {budget}
seed_bank = 3
attempt_factor = 4
development = "{development}"

[sampler]
kind = "tpe"
seed = 17
n_startup_trials = 2

[pruner]
kind = "none"

[objective]
kind = "worst_cell_median_gap_ratio"
infeasible_penalty = 10.0

[esn]
n_neurons = {{ low = 50, high = 150, step = 50 }}
spectral_radius = {{ low = 0.8, high = 1.3 }}
sparsity = {{ low = 0.5, high = 0.98 }}
leak_rate = {{ low = 0.01, high = 0.3, log = true }}
input_scaling = {{ low = 0.02, high = 0.5, log = true }}
seed = {{ low = 1, high = 1000 }}
alpha = {{ low = 1e-3, high = 1.0, log = true }}
velocity_cutoff_hz = {{ low = 5.0, high = 30.0, log = true }}
acceleration_cutoff_hz = {{ low = 5.0, high = 30.0, log = true }}

[space]
warmups_s = [0.0, 1.0]

[[comparison]]
label = "anchor"

[comparison.point]
warmup_s = 1.0

[comparison.point.esn]
n_neurons = 100
spectral_radius = 0.9
sparsity = 0.9
leak_rate = 0.1
input_scaling = 0.1
seed = 31
alpha = 0.01
velocity_cutoff_hz = 20.0
acceleration_cutoff_hz = 20.0
"""


def _samples() -> SampleSet:
    t = np.arange(N, dtype=np.float64) * DT
    start = np.array(SCENARIO.task.initial_q)
    goal = np.array([0.8, 0.4])
    s = np.clip(t / TASK.move[1], 0.0, 1.0)
    blend = s * s * (3.0 - 2.0 * s)
    q = start[None, :] + blend[:, None] * (goal - start)[None, :]
    dq, ddq = differentiate(q, DT, DERIVATIVES)
    tip = endpoint_positions(SCENARIO, q)
    dtip, ddtip = differentiate(tip, DT, DERIVATIVES)
    phase = np.where(t < TASK.dwell[0], 1, 2).astype(np.int64)
    return SampleSet(t, q, dq, ddq, tip, dtip, ddtip, np.zeros((N, 0)), phase)


def _record(samples: SampleSet, payload_sha: str, size: int) -> RecoveryDatasetRecord:
    artifact_id = make_artifact_id("processed", CREATED, payload_sha)
    normalization = fit_normalization(
        samples.arrays(),
        ("q", "dq", "ddq", "tip", "dtip", "ddtip"),
        fitted_on=(artifact_id,),
        training_rows=np.ones(samples.n_samples, dtype=np.bool_),
    )
    return RecoveryDatasetRecord(
        artifact=ArtifactRecord(
            artifact_id=artifact_id,
            kind="processed",
            created_at=CREATED,
            license="LicenseRef-Private",
            access="private",
            payload=Payload(f"armrc://processed/{artifact_id}/samples.npz", payload_sha, size, "samples.npz", 1),
            origin=Origin(
                command="synthetic recovery study fixture",
                config_sha256="2" * 64,
                project_commit="a" * 40,
                project_dirty=False,
                dependency_commits={},
                sources=(RAW_SOURCE,),
            ),
        ),
        scenario=Scenario(
            config_path="tests/fixtures/configs/planar_2dof_fixture.toml",
            config_sha256=sha256_file(SCENARIO_FILE),
            robot="planar-2dof-fixture",
            task="pd-reach-fixture",
            dof=2,
            initial_q=tuple(SCENARIO.task.initial_q),
            target=tuple(SCENARIO.task.target),
        ),
        n_samples=samples.n_samples,
        dof=samples.dof,
        task_dim=samples.task_dim,
        task_code_dim=samples.task_code_dim,
        units=dict(CANONICAL_UNITS),
        phases=dict(TASK_PHASE_CODES),
        preprocessing=Preprocessing(
            resample_period_s=DT, smoothing="none", smoothing_params={}, derivative_method="central-difference"
        ),
        onset=OnsetAnnotation(
            kind="scripted",
            raw_artifact_id=RAW_SOURCE,
            raw_payload_sha256="b" * 64,
            detector="programmed",
            detector_params={},
            sampling_period_s=DT,
            proposed_onset_sample=100,
            proposed_onset_s=1.0,
            confirmed_onset_sample=100,
            confirmed_onset_s=1.0,
            confirmed_by="script",
        ),
        baseline=BaselineCheck(
            q_pre=tuple(float(v) for v in samples.q[0]), tolerance_rad=0.05, max_deviation_rad=0.0, status="passed"
        ),
        crop=CropWindow(pre_roll=(0.0, 1.0), source_duration_s=2.0, task=TASK),
        q0_ref=tuple(float(v) for v in samples.q[0]),
        arrays=array_specs(samples),
        normalization=normalization,
    )


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> tuple[StorageRoot, Path, Path, Path, Path]:
    """A store holding the fixture recovery dataset, its record file, the dev levels, and a 3-trial protocol."""
    base = tmp_path_factory.mktemp("recovery-study")
    root = base / "store"
    root.mkdir()
    store = StorageRoot(root, repositories=(REPO_ROOT,))
    samples = _samples()
    staged = base / "samples.npz"
    save_samples(staged, samples)
    record = _record(samples, sha256_file(staged), staged.stat().st_size)
    store.path(record.artifact.payload.uri, mode="write").write_bytes(staged.read_bytes())
    records = base / "repo"
    (records / "data" / "records" / "processed").mkdir(parents=True)
    record_file = records / "data" / "records" / "processed" / f"{record.artifact.artifact_id}.toml"
    write_record(record_file, record)
    development = base / "recovery_dev_fixture.toml"
    development.write_text(_DEVELOPMENT.format(scenario=SCENARIO_FILE.as_posix()), encoding="utf-8")
    protocol_file = base / "recovery_search_fixture.toml"
    protocol_file.write_text(
        _PROTOCOL.format(
            name="recovery-search-fixture",
            scenario=SCENARIO_FILE.as_posix(),
            model=MODEL.as_posix(),
            budget=3,
            development=development.as_posix(),
        ),
        encoding="utf-8",
    )
    return store, records, record_file, development, protocol_file


def _pairs_total(protocol: RecoverySearchProtocol, record_file: Path) -> int:
    record = load_record(record_file, RecoveryDatasetRecord)
    levels = load_development_robustness(protocol.development)
    lower = tuple(link.q_min for link in SCENARIO.robot.links)
    upper = tuple(link.q_max for link in SCENARIO.robot.links)
    return 2 * len(robustness_scenarios(levels, nominal=record.q0_ref, lower=lower, upper=upper))


def test_study_mirrors_anchors_resumes_and_reports(
    prepared: tuple[StorageRoot, Path, Path, Path, Path], tmp_path: Path
) -> None:
    """Two invocations reach the budget: the anchor runs first, every trial is mirrored, resume adds nothing."""
    store, records, record_file, _development, protocol_file = prepared
    protocol = load_recovery_search(protocol_file)
    tracker = MlflowTracker(store)

    def run(max_trials: int | None = None) -> RecoveryStudyResult:
        return run_recovery_study(
            protocol,
            protocol_file,
            store=store,
            dataset_file=record_file,
            records_root=records,
            exploratory=True,
            max_trials=max_trials,
            tracker=tracker,
        )

    first = run(max_trials=2)
    assert first.report.trials_run == 2
    assert first.report.formulation == "no_augmentation"
    assert len(first.report.summary.trials) == 2
    assert first.report.summary.trials[0].labels["armrc.comparison"] == "anchor"
    assert first.report.summary.trials[0].params["warmup_s"] == 1.0
    parent = first.report.mlflow_parent_run
    assert parent is not None
    experiment = tracker.experiment_id(protocol.name)
    assert tracker.find_by_tags(experiment, {STUDY_TAG: protocol.name, "armrc.kind": "study"}) == parent
    params = tracker.params(parent)
    assert params["protocol.formulation"] == "no_augmentation"
    assert params["protocol.budget"] == "3"
    assert params["protocol.sha256"] == first.report.protocol_sha256
    assert set(first.report.trackers) == {"pd_v2", "computed_torque"}
    assert params["tracker.pd_v2.sha256"] == first.report.trackers["pd_v2"]
    assert params["tracker.computed_torque.sha256"] == first.report.trackers["computed_torque"]

    pairs = _pairs_total(protocol, record_file)
    child = first.child_runs[0]
    tags = tracker.tags(child)
    assert tags["mlflow.parentRunId"] == parent
    assert tags[TRIAL_TAG] == "0"
    assert tags["armrc.comparison"] == "anchor"
    metrics = tracker.metrics(child)
    evaluation = first.evaluations[0]
    assert metrics["pairs_total"] == float(pairs)
    assert metrics["objective"] == evaluation.objective
    assert metrics["feasible"] == float(evaluation.feasible)
    downloaded = Path(tracker.client.download_artifacts(child, "evaluation.json", str(tmp_path / "dl")))
    stored = from_mapping(json.loads(downloaded.read_text()), RecoveryTrialEvaluation)
    assert stored == evaluation

    second = run()
    assert second.report.trials_run == 1
    assert len(second.report.summary.trials) == 3
    assert second.report.mlflow_parent_run == parent
    third = run()
    assert third.report.trials_run == 0
    assert third.child_runs == {}
    assert third.evaluations == ()
    assert third.report.summary.selection_rule == "feasible"
    if third.report.n_feasible:
        assert third.report.best_point is not None
        best = third.report.summary.best_number
        assert best is not None
        assert {k: float(v) for k, v in third.report.best_point.params().items()} == {
            k: float(v) for k, v in third.study.trials[best].params.items()
        }
    else:
        assert third.report.best_point is None
        assert third.report.summary.best_number is None
    for evaluation in (*first.evaluations, *second.evaluations):
        assert evaluation.reason is None or isinstance(evaluation.reason, str)
        assert math.isfinite(evaluation.objective)
    report_file = tmp_path / "report.json"
    report_file.write_text(report_to_json(third.report) + "\n", encoding="utf-8")
    assert load_report(report_file) == third.report


def test_command_runs_a_bounded_number_of_trials_and_writes_the_report(
    prepared: tuple[StorageRoot, Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """The command runs one formulation study from files, honours --max-trials, and refuses to overwrite."""
    store, records, record_file, development, _protocol_file = prepared
    monkeypatch.setenv("ARM_RC_CTRL_STORAGE_ROOT", str(store.root))
    protocol_file = tmp_path / "recovery_search_cli.toml"
    protocol_file.write_text(
        _PROTOCOL.format(
            name="recovery-search-cli",
            scenario=SCENARIO_FILE.as_posix(),
            model=MODEL.as_posix(),
            budget=2,
            development=development.as_posix(),
        ),
        encoding="utf-8",
    )
    report = tmp_path / "recovery_search.json"
    argv = [
        "--protocol",
        str(protocol_file),
        "--dataset",
        str(record_file),
        "--report",
        str(report),
        "--records-root",
        str(records),
        "--exploratory",
        "--no-mlflow",
        "--max-trials",
        "1",
    ]
    assert main([*argv, "--markdown", str(tmp_path / "recovery_search.md")]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["trials_run"] == 1
    assert printed["trials_stored"] == 1
    assert printed["budget"] == 2
    assert printed["formulation"] == "no_augmentation"
    assert printed["mlflow_parent_run"] is None
    loaded = load_report(report)
    assert loaded.protocol == "recovery-search-cli"
    assert loaded.summary.trials[0].labels["armrc.comparison"] == "anchor"
    assert loaded.provenance.exploratory
    markdown = (tmp_path / "recovery_search.md").read_text(encoding="utf-8")
    assert markdown.startswith("# Recovery search `recovery-search-cli`")
    assert "## Comparison points" in markdown
    with pytest.raises(FileExistsError, match="refusing"):
        main(argv)
    with pytest.raises(ValueError, match="max-trials"):
        main([*argv[:-2], "--max-trials", "0", "--report", str(tmp_path / "other.json")])

# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Mandatory MLflow logging of curated runs into the external store (M3-001)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.config import load_config
from arm_rc_ctrl.controllers.tracking import TrackerConfig
from arm_rc_ctrl.data.preprocess import PreprocessResult, preprocess_demonstration
from arm_rc_ctrl.data.records import RawDemonstrationRecord, load_record
from arm_rc_ctrl.experiments import replay
from arm_rc_ctrl.experiments.closed_loop import EstimatorSpec, run_nominal
from arm_rc_ctrl.experiments.run_record import record_run_pointer
from arm_rc_ctrl.experiments.tracking import RUN_ID_TAG, MlflowTracker, flatten_scalars
from arm_rc_ctrl.metrics.report import report_from_json, report_to_json
from arm_rc_ctrl.rc.recipe import ModelRecipe, write_recipe
from arm_rc_ctrl.rc.runtime import load_training_samples
from arm_rc_ctrl.rc.train import load_model_config, train_task
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import load_scenario
from arm_rc_ctrl.storage import StorageRoot

if TYPE_CHECKING:
    from arm_rc_ctrl.experiments.closed_loop import ClosedLoopResult

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
MODEL = REPO_ROOT / "configs" / "models" / "esn_task_1a.toml"
DEV_PD = REPO_ROOT / "configs" / "controllers" / "pd.toml"
RAW_RECORD = REPO_ROOT / "tests" / "fixtures" / "records" / "raw-20260830-287036d83d46.toml"
RAW_LOG = REPO_ROOT / "tests" / "fixtures" / "raw" / "demo.sklog.npz"
SCENARIO = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
PREPROCESS = REPO_ROOT / "configs" / "preprocessing" / "default.toml"
FIXED_TIME = datetime(2026, 9, 3, 9, 0, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[StorageRoot, Path, PreprocessResult, ModelRecipe, Path]:
    """The fixture dataset in a store, a small recipe trained on it, and the written recipe file."""
    base = tmp_path_factory.mktemp("tracking")
    root = base / "store"
    root.mkdir()
    store = StorageRoot(root, repositories=(REPO_ROOT,))
    raw = load_record(RAW_RECORD, RawDemonstrationRecord)
    store.path(raw.artifact.payload.uri, mode="write").write_bytes(RAW_LOG.read_bytes())
    records = base / "repo"
    (records / "data" / "records" / "processed").mkdir(parents=True)
    processed = preprocess_demonstration(
        RAW_RECORD, SCENARIO, PREPROCESS, store=store, records_root=records, exploratory=True, now=FIXED_TIME
    )
    config_file = base / "esn_small.toml"
    config_file.write_text(MODEL.read_text().replace("n_neurons = 200", "n_neurons = 30"))
    result = train_task(
        load_model_config(config_file),
        config_file,
        [processed.record_file],
        store=store,
        exploratory=True,
        now=FIXED_TIME,
        records_root=records,
    )
    recipe_file = base / "recipe.toml"
    write_recipe(recipe_file, result.recipe)
    return store, records, processed, result.recipe, recipe_file


@pytest.fixture(scope="module")
def closed_loop(trained: tuple[StorageRoot, Path, PreprocessResult, ModelRecipe, Path]) -> ClosedLoopResult:
    """One nominal RC run in the module store."""
    store, records, processed, recipe, _ = trained
    scenario = load_scenario(SCENARIO)
    return run_nominal(
        scenario,
        SCENARIO,
        processed.record,
        processed.samples,
        recipe,
        load_config(DEV_PD, TrackerConfig),
        store=store,
        estimator=EstimatorSpec(20.0, 20.0).config(scenario.timing.dt),
        training_samples=load_training_samples(recipe, store, records_root=records),
        exploratory=True,
        now=FIXED_TIME,
    )


def test_flatten_scalars_uses_dotted_keys_and_keeps_scalar_types() -> None:
    """Nested mappings and sequences flatten to dotted keys; scalars keep their type."""
    out: dict[str, object] = {}
    flatten_scalars("config", {"a": {"b": 1, "c": [0.5, "x"]}, "d": None, "e": True}, out)
    assert out == {"config.a.b": 1, "config.a.c.0": 0.5, "config.a.c.1": "x", "config.d": None, "config.e": True}


def test_log_run_records_everything_in_the_external_store(
    trained: tuple[StorageRoot, Path, PreprocessResult, ModelRecipe, Path],
    closed_loop: ClosedLoopResult,
    tmp_path: Path,
) -> None:
    """Config, revisions, hashes, seeds, metrics, recipe, and plots of an RC run and a replay land under armrc://mlflow/."""
    store, records, processed, recipe, _ = trained
    scenario = load_scenario(SCENARIO)
    replayed = replay.run_replay(
        scenario,
        SCENARIO,
        processed.record,
        processed.samples,
        load_config(DEV_PD, TrackerConfig),
        store=store,
        exploratory=True,
        now=FIXED_TIME,
    )
    pointer_file = record_run_pointer(records, closed_loop.pointer)
    tracker = MlflowTracker(store)
    rc = tracker.log_run(
        closed_loop.run, closed_loop.report, experiment="fixture", recipe=recipe, pointer_file=pointer_file
    )
    rp = tracker.log_run(replayed.run, replayed.report, experiment="fixture")
    assert rc.created
    assert rp.created
    assert rc.experiment_id == rp.experiment_id
    assert rc.mlflow_run_id != rp.mlflow_run_id
    tracking = store.root / "mlflow" / "tracking"
    assert tracker.tracking_uri == f"sqlite:///{tracking / 'mlflow.db'}"
    assert (tracking / "mlflow.db").is_file()
    assert (tracking / "artifacts" / "fixture" / rc.mlflow_run_id / "artifacts" / "report.json").is_file()
    assert not list(REPO_ROOT.glob("mlruns"))

    provenance = closed_loop.summary.provenance
    params = tracker.params(rc.mlflow_run_id)
    assert params["provenance.project_commit"] == provenance.project_commit
    assert params["provenance.project_dirty"] == str(provenance.project_dirty)
    assert params["provenance.lock_sha256"] == provenance.lock_sha256
    assert params["provenance.config_sha256"] == provenance.config_sha256
    assert params["config.tracker.type"] == "pd"
    assert params["config.recipe.esn.reservoir.n_neurons"] == "30"
    assert params["seeds.reservoir"] == str(recipe.esn.reservoir.seed)
    for submodule in provenance.submodules:
        assert params[f"revisions.{submodule.name}"] == (submodule.checked_out or submodule.recorded)
    assert {key for key in params if key.startswith("builds.")} == {f"builds.{b.name}" for b in provenance.builds}
    assert provenance.seeds
    for name, seed in provenance.seeds.items():
        assert params[f"seeds.{name}"] == str(seed)
    payload = processed.record.artifact.payload
    assert params["artifacts.0.uri"] == payload.uri
    assert params["artifacts.0.sha256"] == payload.sha256
    assert params["run.arrays_sha256"] == closed_loop.summary.arrays_sha256
    assert params["run.n_samples"] == str(closed_loop.run.arrays.n_samples)

    metrics = tracker.metrics(rc.mlflow_run_id)
    report = closed_loop.report
    assert report.joint_rmse is not None
    assert metrics["joint_rmse.aggregate"] == report.joint_rmse.aggregate
    assert metrics["move_coverage"] == report.move_coverage
    assert report.effort is not None
    assert metrics["effort.saturation_fraction"] == report.effort.saturation_fraction
    assert "success" not in metrics  # booleans are tags/params, not metrics

    tags = tracker.tags(rc.mlflow_run_id)
    assert tags[RUN_ID_TAG] == closed_loop.pointer.artifact.artifact_id
    assert tags["armrc.method"] == "rc+pd"
    assert tags["armrc.termination"] == closed_loop.summary.termination.kind
    assert tags["armrc.reference_artifact"] == processed.record.artifact.artifact_id
    assert tags["mlflow.runName"] == closed_loop.pointer.artifact.artifact_id

    assert tracker.logged_artifacts(rc.mlflow_run_id) == [
        "plots/tracking.png",
        "pointer.toml",
        "pointer_record.toml",
        "provenance.json",
        "recipe.toml",
        "report.json",
        "run.json",
    ]
    assert tracker.logged_artifacts(rp.mlflow_run_id) == [
        "plots/tracking.png",
        "pointer.toml",
        "provenance.json",
        "report.json",
        "run.json",
    ]
    artifact_dir = Path(tracker.client.download_artifacts(rc.mlflow_run_id, "", str(tmp_path / "dl")))
    assert report_from_json((artifact_dir / "report.json").read_text()) == report
    assert (artifact_dir / "recipe.toml").read_text() == (tmp_path / "dl" / "recipe.toml").read_text()
    assert (artifact_dir / "pointer_record.toml").read_bytes() == pointer_file.read_bytes()
    assert (artifact_dir / "run.json").read_bytes() == (closed_loop.directory / "run.json").read_bytes()
    assert (artifact_dir / "plots" / "tracking.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert json.loads((artifact_dir / "provenance.json").read_text())["project_commit"] == provenance.project_commit

    # Idempotent per run ID: a second call finds the existing MLflow run and logs nothing new.
    again = tracker.log_run(closed_loop.run, closed_loop.report, experiment="fixture", recipe=recipe)
    assert again.mlflow_run_id == rc.mlflow_run_id
    assert not again.created
    assert tracker.find("run-20260903-000000000000") is None
    experiment = tracker.client.get_experiment_by_name("fixture")
    assert experiment is not None
    assert len(tracker.client.search_runs([experiment.experiment_id])) == 2


def test_replay_command_logs_by_default_and_reports_the_mlflow_run(
    trained: tuple[StorageRoot, Path, PreprocessResult, ModelRecipe, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """The run commands log to MLflow unless ``--no-mlflow`` is given, and print the MLflow run ID."""
    store, _, processed, _, _ = trained
    monkeypatch.setenv("ARM_RC_CTRL_STORAGE_ROOT", str(store.root))
    records = tmp_path / "repo"
    (records / "data" / "records" / "runs").mkdir(parents=True)
    argv = [
        "--scenario",
        str(SCENARIO),
        "--dataset",
        str(processed.record_file),
        "--controller",
        str(DEV_PD),
        "--exploratory",
        "--records-root",
        str(records),
        "--report",
        str(tmp_path / "report.json"),
    ]
    assert replay.main([*argv, "--experiment", "cli"]) == 0
    printed = json.loads(capsys.readouterr().out)
    tracker = MlflowTracker(store)
    assert printed["mlflow_run_id"] == tracker.find(printed["run_id"])
    assert tracker.tags(printed["mlflow_run_id"])["armrc.method"] == "replay+pd"
    experiment = tracker.client.get_experiment_by_name("cli")
    assert experiment is not None
    assert report_to_json(report_from_json((tmp_path / "report.json").read_text()))
    assert "pointer_record.toml" in tracker.logged_artifacts(printed["mlflow_run_id"])

    # A different tracker gives a different (content-addressed) run; opting out logs nothing.
    argv[argv.index(str(DEV_PD))] = str(REPO_ROOT / "configs" / "controllers" / "task_1a_pd_v2.toml")
    assert replay.main([*argv[:-2], "--no-pointer", "--no-mlflow"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["mlflow_run_id"] is None
    assert printed["pointer"] is None
    assert tracker.find(printed["run_id"]) is None

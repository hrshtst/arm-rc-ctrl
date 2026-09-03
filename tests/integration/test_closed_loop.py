# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M2-012: the nominal RC closed-loop runner records the whole loop with provenance and evaluates it."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from arm_rc_ctrl.config import load_config
from arm_rc_ctrl.controllers.tracking import TrackerConfig
from arm_rc_ctrl.data.preprocess import PreprocessResult, preprocess_demonstration
from arm_rc_ctrl.data.records import RawDemonstrationRecord, load_record
from arm_rc_ctrl.experiments.closed_loop import EstimatorSpec, NominalConfig, load_nominal_config, main, run_nominal
from arm_rc_ctrl.experiments.run_record import OPTIONAL_ARRAYS, REQUIRED_ARRAYS
from arm_rc_ctrl.metrics.report import report_from_json
from arm_rc_ctrl.rc.recipe import ModelRecipe, write_recipe
from arm_rc_ctrl.rc.runtime import load_training_samples
from arm_rc_ctrl.rc.train import load_model_config, train_task
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import load_scenario
from arm_rc_ctrl.storage import StorageRoot

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
NOMINAL = REPO_ROOT / "configs" / "evaluations" / "task_1a_nominal.toml"
MODEL = REPO_ROOT / "configs" / "models" / "esn_task_1a.toml"
DEV_PD = REPO_ROOT / "configs" / "controllers" / "pd.toml"
RAW_RECORD = REPO_ROOT / "tests" / "fixtures" / "records" / "raw-20260830-287036d83d46.toml"
RAW_LOG = REPO_ROOT / "tests" / "fixtures" / "raw" / "demo.sklog.npz"
SCENARIO = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
PREPROCESS = REPO_ROOT / "configs" / "preprocessing" / "default.toml"
FIXED_TIME = datetime(2026, 9, 1, 16, 0, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[StorageRoot, Path, PreprocessResult, ModelRecipe, Path]:
    """The fixture dataset in a store, a small recipe trained on it, and the written recipe file."""
    base = tmp_path_factory.mktemp("closed-loop")
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


def test_committed_nominal_config_uses_the_pd_v2_baseline() -> None:
    """The nominal evaluation tracks with the robustness-constrained PD v2 gains and filtered derivatives."""
    config = load_nominal_config(NOMINAL)
    assert config.name == "task-1a-nominal"
    assert config.tracker == REPO_ROOT / "configs" / "controllers" / "task_1a_pd_v2.toml"
    assert config.estimator == EstimatorSpec(20.0, 20.0, 3.0)
    assert config.estimator.config(0.01).nominal_dt_s == 0.01
    with pytest.raises(ValueError, match="name must not be empty"):
        NominalConfig(" ", config.tracker, config.estimator)


def test_nominal_run_records_everything(trained: tuple[StorageRoot, Path, PreprocessResult, ModelRecipe, Path]) -> None:
    """The run holds measured/generated states, raw and filtered derivatives, torque, phase, ESN state, termination."""
    store, records, processed, recipe, _ = trained
    scenario = load_scenario(SCENARIO)
    training = load_training_samples(recipe, store, records_root=records)
    result = run_nominal(
        scenario,
        SCENARIO,
        processed.record,
        processed.samples,
        recipe,
        load_config(DEV_PD, TrackerConfig),
        store=store,
        estimator=EstimatorSpec(20.0, 20.0).config(scenario.timing.dt),
        training_samples=training,
        exploratory=True,
        now=FIXED_TIME,
    )
    arrays = result.run.arrays.arrays
    assert set(arrays) == {
        *REQUIRED_ARRAYS,
        "tau_applied",
        "phase",
        "esn_state_norm",
        "generator_output_q",
        "warmup_state_norm",
        "warmup_esn_input",
    }
    assert set(OPTIONAL_ARRAYS) - set(arrays) == {"ext_force", "generator_increment_q"}
    hold_rows = arrays["phase"] == 0
    assert np.all(np.isnan(arrays["generator_output_q"][hold_rows]))
    assert np.all(np.isfinite(arrays["generator_output_q"][~hold_rows]))
    assert np.all(np.isfinite(arrays["warmup_esn_input"][hold_rows]))
    assert np.all(np.isnan(arrays["warmup_state_norm"][~hold_rows]))
    assert result.summary.activation_s == scenario.timing.intervals.prime[1]
    assert result.summary.method == "rc+pd"
    assert result.summary.termination.kind == "completed"
    assert arrays["t"].shape[0] == processed.samples.n_samples
    boundary = scenario.timing.intervals.prime[1]
    hold = arrays["t"] < boundary
    assert np.array_equal(arrays["phase"][hold], np.zeros(hold.sum(), dtype=np.int64))
    assert np.array_equal(arrays["phase"][~hold], np.ones((~hold).sum(), dtype=np.int64))
    assert np.allclose(arrays["q_desired"][hold], scenario.task.initial_q)
    assert not arrays["dq_desired_raw"][hold].any()
    assert np.all(np.isfinite(arrays["esn_state_norm"]))
    assert arrays["esn_state_norm"][1:].max() > 0
    assert not np.array_equal(arrays["dq_desired_raw"], arrays["dq_desired"])  # filtered differs from raw
    assert np.all(np.abs(arrays["tau_applied"]) <= np.asarray(scenario.limits.torque) + 1e-12)
    assert result.boundary_jump is not None
    assert result.summary.provenance.seeds == {"reservoir": recipe.esn.reservoir.seed}
    assert result.summary.provenance.config["recipe"]["name"] == recipe.name  # type: ignore[index]
    assert result.summary.provenance.config["hold_until_s"] == boundary
    assert result.pointer.artifact.origin.sources == (processed.record.artifact.artifact_id,)
    assert "boundary jump" in result.summary.notes
    assert result.report.method == "rc+pd"
    assert result.report.joint_rmse is not None
    assert result.report.move_coverage == 1.0


def test_command_line_entry_point(
    trained: tuple[StorageRoot, Path, PreprocessResult, ModelRecipe, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI rebuilds the generator from the recipe through the store, runs, and writes the report."""
    store, records, processed, _, recipe_file = trained
    monkeypatch.setenv("ARM_RC_CTRL_STORAGE_ROOT", str(store.root))
    config = tmp_path / "nominal_fixture.toml"
    config.write_text(
        NOMINAL.read_text().replace('tracker = "../controllers/task_1a_pd_v2.toml"', f'tracker = "{DEV_PD.as_posix()}"')
    )
    report_file = tmp_path / "report.json"
    argv = [
        "--config",
        str(config),
        "--scenario",
        str(SCENARIO),
        "--dataset",
        str(processed.record_file),
        "--recipe",
        str(recipe_file),
        "--records-root",
        str(records),
        "--report",
        str(report_file),
        "--exploratory",
        "--no-mlflow",
    ]
    assert main(argv) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["method"] == "rc+pd"
    assert printed["termination"] == "completed"
    assert printed["run_dir"].startswith("runs/run-")
    report = report_from_json(report_file.read_text())
    assert report.run_id == printed["run_id"]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(argv)


def test_nominal_ct_config_and_run(trained: tuple[StorageRoot, Path, PreprocessResult, ModelRecipe, Path]) -> None:
    """The computed-torque evaluation config differs from the PD one only by the tracker."""
    pd_config = load_nominal_config(NOMINAL)
    ct_config = load_nominal_config(REPO_ROOT / "configs" / "evaluations" / "task_1a_nominal_ct.toml")
    assert ct_config.tracker == REPO_ROOT / "configs" / "controllers" / "task_1a_computed_torque.toml"
    assert ct_config.estimator == pd_config.estimator
    store, records, processed, recipe, _ = trained
    scenario = load_scenario(SCENARIO)
    result = run_nominal(
        scenario,
        SCENARIO,
        processed.record,
        processed.samples,
        recipe,
        load_config(REPO_ROOT / "configs" / "controllers" / "computed_torque.toml", TrackerConfig),
        store=store,
        estimator=ct_config.estimator.config(scenario.timing.dt),
        training_samples=load_training_samples(recipe, store, records_root=records),
        exploratory=True,
        now=FIXED_TIME,
    )
    assert result.summary.method == "rc+computed_torque"
    assert result.summary.termination.kind == "completed"
    assert result.report.method == "rc+computed_torque"
    assert result.summary.provenance.config["tracker"]["type"] == "computed_torque"  # type: ignore[index]

# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M2-011: the training command validates its inputs and emits a recipe, training metrics, and provenance."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from arm_rc_ctrl.config import ConfigError
from arm_rc_ctrl.controllers.estimator import EstimatorConfig
from arm_rc_ctrl.data.preprocess import PreprocessResult, preprocess_demonstration
from arm_rc_ctrl.data.records import RawDemonstrationRecord, load_record
from arm_rc_ctrl.rc.recipe import RecipeMismatchError, load_recipe
from arm_rc_ctrl.rc.runtime import generator_from_recipe, load_training_samples
from arm_rc_ctrl.rc.train import InputTransformSpec, ModelConfig, load_model_config, main, recipe_id, train_task
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageRoot

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
MODEL = REPO_ROOT / "configs" / "models" / "esn_task_1a.toml"
RAW_RECORD = REPO_ROOT / "tests" / "fixtures" / "records" / "raw-20260830-287036d83d46.toml"
RAW_LOG = REPO_ROOT / "tests" / "fixtures" / "raw" / "demo.sklog.npz"
SCENARIO = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
PREPROCESS = REPO_ROOT / "configs" / "preprocessing" / "default.toml"
FIXED_TIME = datetime(2026, 9, 1, 14, 0, 0, tzinfo=UTC)


@pytest.fixture
def prepared(tmp_path: Path) -> tuple[StorageRoot, Path, PreprocessResult]:
    """A store holding the fixture dataset, its records root, and the preprocessing result."""
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
    return store, records, processed


def _small_model(tmp_path: Path) -> Path:
    """The committed model config with a small reservoir so the fixture trains quickly."""
    text = MODEL.read_text().replace("n_neurons = 200", "n_neurons = 30")
    file = tmp_path / "esn_small.toml"
    file.write_text(text)
    return file


def test_committed_model_config_is_the_development_anchor() -> None:
    """The committed config declares the fixed-scale transform at the owner's anchor values."""
    config = load_model_config(MODEL)
    assert config.name == "esn-task-1a-dev"
    assert config.input_transform == InputTransformSpec("fixed_scale", 0.3, 4.0)
    assert config.input_transform.fixed_scales == {"q": 0.3, "dq": 4.0}
    assert config.esn.reservoir.n_neurons == 200
    assert config.esn.readout.solver == "cholesky"


def test_transform_spec_validation() -> None:
    """Fixed scales are required for fixed_scale and forbidden otherwise."""
    with pytest.raises(ValueError, match="must be positive for the fixed_scale policy"):
        InputTransformSpec("fixed_scale", 0.3, None)
    with pytest.raises(ValueError, match="only meaningful for the fixed_scale policy"):
        InputTransformSpec("training_std", 0.3, 4.0)
    assert InputTransformSpec("training_std").fixed_scales is None
    with pytest.raises(ValueError, match="name must not be empty"):
        ModelConfig(" ", load_model_config(MODEL).esn, InputTransformSpec("training_std"))


def test_train_task_builds_a_verified_recipe(
    prepared: tuple[StorageRoot, Path, PreprocessResult], tmp_path: Path
) -> None:
    """The recipe binds the dataset, carries the fit report, and is verified by a refit before it is reported."""
    store, records, processed = prepared
    config_file = _small_model(tmp_path)
    result = train_task(
        load_model_config(config_file),
        config_file,
        [processed.record_file],
        store=store,
        exploratory=True,
        now=FIXED_TIME,
        records_root=records,
    )
    recipe, report = result.recipe, result.report
    assert recipe.datasets[0].artifact_id == processed.record.artifact.artifact_id
    assert recipe.datasets[0].record == "data/records/processed/" + processed.record_file.name
    assert recipe.transform.policy == "fixed_scale"
    assert recipe.transform.derived_from == (processed.record.artifact.artifact_id,)
    assert report.refit_verified is True
    assert report.fit == recipe.fit
    assert report.recipe_id.startswith("model-20260901-")
    assert report.recipe_file == f"data/records/models/{report.recipe_id}.toml"
    assert report.model_config == config_file.name  # outside the repository: only the name is recorded
    assert report.provenance.seeds == {"reservoir": recipe.esn.reservoir.seed}
    assert [a.sha256 for a in report.provenance.artifacts] == [processed.record.artifact.payload.sha256]
    assert recipe_id(recipe, report.provenance.created_at) == report.recipe_id
    assert result.model.fitted
    # the runtime path rebuilds a generator from the recipe through the same store
    samples = load_training_samples(recipe, store, records_root=records)
    generator = generator_from_recipe(recipe, samples, estimator=EstimatorConfig(0.01))
    assert generator.model.fitted


def test_train_task_validates_inputs(prepared: tuple[StorageRoot, Path, PreprocessResult], tmp_path: Path) -> None:
    """Empty, duplicate, or inconsistent datasets are refused before anything is trained."""
    store, records, processed = prepared
    config_file = _small_model(tmp_path)
    config = load_model_config(config_file)
    with pytest.raises(ValueError, match="at least one processed dataset record"):
        train_task(config, config_file, [], store=store, exploratory=True, now=FIXED_TIME, records_root=records)
    with pytest.raises(ValueError, match="datasets must be distinct"):
        train_task(
            config,
            config_file,
            [processed.record_file, processed.record_file],
            store=store,
            exploratory=True,
            now=FIXED_TIME,
            records_root=records,
        )
    other = tmp_path / "other.toml"
    other.write_text(processed.record_file.read_text().replace('interpolation = "linear"', 'interpolation = "cubic"'))
    with pytest.raises(ValueError, match="does not share the scenario, widths, and preprocessing"):
        train_task(
            config,
            config_file,
            [processed.record_file, other],
            store=store,
            exploratory=True,
            now=FIXED_TIME,
            records_root=records,
        )
    outside = tmp_path / "elsewhere" / "x.toml"
    outside.parent.mkdir()
    outside.write_text(processed.record_file.read_text())
    with pytest.raises(ValueError, match="lies outside the records root"):
        train_task(config, config_file, [outside], store=store, exploratory=True, now=FIXED_TIME, records_root=records)
    bad = tmp_path / "bad_model.toml"
    bad.write_text(config_file.read_text().replace('policy = "fixed_scale"', 'policy = "median"'))
    with pytest.raises(ConfigError):
        load_model_config(bad)


def test_command_line_writes_recipe_and_report(
    prepared: tuple[StorageRoot, Path, PreprocessResult],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI writes a loadable recipe and a report with provenance, and refuses to overwrite either."""
    store, records, processed = prepared
    config_file = _small_model(tmp_path)
    monkeypatch.setenv("ARM_RC_CTRL_STORAGE_ROOT", str(store.root))
    report_file = tmp_path / "out" / "training.json"
    recipe_file = tmp_path / "out" / "recipe.toml"
    argv = [
        "--model",
        str(config_file),
        "--dataset",
        str(processed.record_file),
        "--report",
        str(report_file),
        "--recipe",
        str(recipe_file),
        "--records-root",
        str(records),
        "--exploratory",
    ]
    assert main(argv) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["refit_verified"] is True
    assert printed["datasets"] == [processed.record.artifact.artifact_id]
    recipe = load_recipe(recipe_file)
    assert recipe.fit.rmse == printed["rmse"]
    report = json.loads(report_file.read_text())
    assert report["recipe_id"] == printed["recipe_id"]
    assert report["recipe_file"] == "recipe.toml"
    assert json.loads(report["provenance"]["config_json"])["datasets"] == [processed.record.artifact.artifact_id]
    assert str(tmp_path) not in report_file.read_text()
    assert str(tmp_path) not in recipe_file.read_text()
    samples = load_training_samples(recipe, store, records_root=records)
    recipe.refit(samples)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(argv)
    tampered = dataclasses.replace(recipe, fit=dataclasses.replace(recipe.fit, rmse=recipe.fit.rmse * 2))
    with pytest.raises(RecipeMismatchError):
        tampered.refit(samples)

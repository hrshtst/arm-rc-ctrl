# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M2-014: the paired RC+PD / replay+PD report uses identical windows and metric definitions."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from arm_rc_ctrl.config import load_config
from arm_rc_ctrl.controllers.tracking import TrackerConfig
from arm_rc_ctrl.data.preprocess import PreprocessResult, preprocess_demonstration
from arm_rc_ctrl.data.records import RawDemonstrationRecord, load_record
from arm_rc_ctrl.experiments.closed_loop import EstimatorSpec
from arm_rc_ctrl.experiments.paired import (
    MetricComparison,
    PairedReport,
    PairedResult,
    compare_reports,
    load_paired_report,
    main,
    paired_to_markdown,
    run_paired_nominal,
)
from arm_rc_ctrl.experiments.paired_suite import EffectDecomposition, PairedSuite, load_paired_suite, suite_to_markdown
from arm_rc_ctrl.experiments.paired_suite import main as suite_main
from arm_rc_ctrl.metrics.joint import JointAnglePolicy
from arm_rc_ctrl.metrics.report import build_report
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
FIXED_TIME = datetime(2026, 9, 1, 18, 0, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[StorageRoot, Path, PreprocessResult, ModelRecipe, Path]:
    """The fixture dataset in a store, a small recipe trained on it, and the recipe file."""
    base = tmp_path_factory.mktemp("paired")
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


def test_paired_runs_share_windows_and_metric_definitions(
    trained: tuple[StorageRoot, Path, PreprocessResult, ModelRecipe, Path],
) -> None:
    """Both reports come from build_report on the same reference; every metric is compared side by side."""
    store, records, processed, recipe, _ = trained
    scenario = load_scenario(SCENARIO)
    tracker = load_config(DEV_PD, TrackerConfig)
    result = run_paired_nominal(
        scenario,
        SCENARIO,
        processed.record,
        processed.samples,
        recipe,
        "recipe.toml",
        tracker,
        store=store,
        estimator=EstimatorSpec(20.0, 20.0).config(scenario.timing.dt),
        training_samples=load_training_samples(recipe, store, records_root=records),
        exploratory=True,
        now=FIXED_TIME,
    )
    paired = result.paired
    assert (paired.rc.method, paired.replay.method) == ("rc+pd", "replay+pd")
    assert paired.tracker == "pd"
    assert paired.rc.windows == paired.replay.windows
    assert paired.rc.reference_artifact == paired.replay.reference_artifact == processed.record.artifact.artifact_id
    names = [m.name for m in paired.metrics]
    assert names[:3] == ["joint_rmse", "dwell_in_tolerance_fraction", "dwell_longest_in_tolerance_s"]
    assert len(names) == len(set(names)) == 12
    rmse = next(m for m in paired.metrics if m.name == "joint_rmse")
    assert rmse.rc is not None
    assert rmse.replay is not None
    assert rmse.rc > 0
    assert rmse.replay > 0
    assert rmse.difference == pytest.approx(rmse.rc - rmse.replay)
    assert rmse.ratio == pytest.approx(rmse.rc / rmse.replay)
    # the replay report equals a fresh build_report of the same run: one metric implementation for both arms
    rebuilt = build_report(
        result.replay.run,
        processed.samples,
        processed.record.artifact.artifact_id,
        tolerance=scenario.task.tolerance,
        torque_limits=scenario.limits.torque,
        policy=JointAnglePolicy.limited(scenario.dof),
    )
    assert rebuilt == result.replay.report
    assert result.rc.pointer.artifact.artifact_id != result.replay.pointer.artifact.artifact_id
    markdown = paired_to_markdown(paired)
    assert "| joint_rmse | rad |" in markdown
    assert f"`{paired.rc.run_id}`" not in markdown  # run IDs appear as plain cells
    assert paired.rc.run_id in markdown


def test_paired_report_refuses_mismatched_runs(
    trained: tuple[StorageRoot, Path, PreprocessResult, ModelRecipe, Path],
) -> None:
    """Different trackers, scenarios, references, or windows cannot be paired."""
    store, records, processed, recipe, _ = trained
    scenario = load_scenario(SCENARIO)
    result = run_paired_nominal(
        scenario,
        SCENARIO,
        processed.record,
        processed.samples,
        recipe,
        "recipe.toml",
        load_config(DEV_PD, TrackerConfig),
        store=store,
        estimator=EstimatorSpec(20.0, 20.0).config(scenario.timing.dt),
        training_samples=load_training_samples(recipe, store, records_root=records),
        exploratory=True,
        now=FIXED_TIME.replace(hour=19),
    )
    rc, replay = result.rc.report, result.replay.report
    good = PairedReport(scenario.name, processed.record.artifact.artifact_id, "pd", "recipe.toml", rc, replay)
    assert good.metrics == compare_reports(rc, replay)
    with pytest.raises(ValueError, match="not the RC/replay pair of tracker 'computed_torque'"):
        PairedReport(scenario.name, processed.record.artifact.artifact_id, "computed_torque", "r", rc, replay)
    with pytest.raises(ValueError, match="describe scenarios"):
        PairedReport("other", processed.record.artifact.artifact_id, "pd", "r", rc, replay)
    with pytest.raises(ValueError, match="reference artifact"):
        PairedReport(scenario.name, "processed-20260830-000000000000", "pd", "r", rc, replay)
    shifted = dataclasses.replace(replay, windows=dataclasses.replace(replay.windows, move=(0.0, 0.2)))
    with pytest.raises(ValueError, match="metric windows differ"):
        PairedReport(scenario.name, processed.record.artifact.artifact_id, "pd", "r", rc, shifted)
    with pytest.raises(ValueError, match="recipe must name"):
        PairedReport(scenario.name, processed.record.artifact.artifact_id, "pd", " ", rc, replay)
    assert MetricComparison("x", "", None, 1.0).difference is None
    assert MetricComparison("x", "", 1.0, 0.0).ratio is None


def test_command_line_writes_json_and_markdown(
    trained: tuple[StorageRoot, Path, PreprocessResult, ModelRecipe, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI runs both arms, writes a loadable paired report and its table, and refuses to overwrite."""
    store, records, processed, _, recipe_file = trained
    monkeypatch.setenv("ARM_RC_CTRL_STORAGE_ROOT", str(store.root))
    config = tmp_path / "nominal_fixture.toml"
    config.write_text(
        NOMINAL.read_text().replace('tracker = "../controllers/task_1a_pd_v2.toml"', f'tracker = "{DEV_PD.as_posix()}"')
    )
    report_file = tmp_path / "paired.json"
    markdown_file = tmp_path / "paired.md"
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
        "--markdown",
        str(markdown_file),
        "--exploratory",
    ]
    assert main(argv) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["rc_termination"] == "completed"
    loaded = load_paired_report(report_file)
    assert loaded.rc.run_id == printed["rc_run"]
    assert loaded.replay.run_id == printed["replay_run"]
    assert loaded.recipe == "recipe.toml"
    assert markdown_file.read_text().startswith("# Paired nominal evaluation")
    assert str(tmp_path) not in report_file.read_text()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(argv)


# --- M2-017: the two-tracker suite ---------------------------------------------------------------


def _pair(
    trained: tuple[StorageRoot, Path, PreprocessResult, ModelRecipe, Path], tracker_file: Path, hour: int
) -> PairedResult:
    store, records, processed, recipe, _ = trained
    scenario = load_scenario(SCENARIO)
    return run_paired_nominal(
        scenario,
        SCENARIO,
        processed.record,
        processed.samples,
        recipe,
        "recipe.toml",
        load_config(tracker_file, TrackerConfig),
        store=store,
        estimator=EstimatorSpec(20.0, 20.0).config(scenario.timing.dt),
        training_samples=load_training_samples(recipe, store, records_root=records),
        exploratory=True,
        now=FIXED_TIME.replace(hour=hour),
    )


def _without_tracker(config: dict[str, object]) -> dict[str, object]:
    return {k: v for k, v in config.items() if k != "tracker"}


def test_suite_separates_generator_and_tracker_effects(
    trained: tuple[StorageRoot, Path, PreprocessResult, ModelRecipe, Path],
) -> None:
    """Four runs of one reference and recipe share windows, metric definitions, and provenance structure."""
    pd = _pair(trained, DEV_PD, 20)
    ct = _pair(trained, REPO_ROOT / "configs" / "controllers" / "computed_torque.toml", 21)
    suite = PairedSuite(pd.paired.scenario, pd.paired.reference_artifact, "recipe.toml", pd.paired, ct.paired)
    assert [e.name for e in suite.effects] == [m.name for m in pd.paired.metrics]
    rmse = next(e for e in suite.effects if e.name == "joint_rmse")
    assert rmse.generator_effect_pd == pytest.approx(cast("float", rmse.rc_pd) - cast("float", rmse.replay_pd))
    assert rmse.tracker_effect_replay == pytest.approx(cast("float", rmse.replay_ct) - cast("float", rmse.replay_pd))
    assert rmse.tracker_effect_rc == pytest.approx(cast("float", rmse.rc_ct) - cast("float", rmse.rc_pd))
    reports = [pd.paired.rc, pd.paired.replay, ct.paired.rc, ct.paired.replay]
    assert len({r.windows for r in reports}) == 1
    assert len({r.reference_artifact for r in reports}) == 1
    # identical provenance structure: the RC runs differ only in the tracker, likewise the replay runs
    rc_pd_config = dict(pd.rc.summary.provenance.config)
    rc_ct_config = dict(ct.rc.summary.provenance.config)
    assert rc_pd_config.keys() == rc_ct_config.keys()
    assert _without_tracker(rc_pd_config) == _without_tracker(rc_ct_config)
    assert pd.rc.summary.provenance.seeds == ct.rc.summary.provenance.seeds
    assert pd.rc.summary.provenance.artifacts == ct.rc.summary.provenance.artifacts
    replay_pd_config = dict(pd.replay.summary.provenance.config)
    replay_ct_config = dict(ct.replay.summary.provenance.config)
    assert _without_tracker(replay_pd_config) == _without_tracker(replay_ct_config)
    markdown = suite_to_markdown(suite)
    assert "| rc+computed_torque |" in markdown
    assert "| joint_rmse | rad |" in markdown
    with pytest.raises(ValueError, match="needs a pd pair and a computed_torque pair"):
        PairedSuite(pd.paired.scenario, pd.paired.reference_artifact, "recipe.toml", pd.paired, pd.paired)
    with pytest.raises(ValueError, match="recipe differs"):
        PairedSuite(pd.paired.scenario, pd.paired.reference_artifact, "other", pd.paired, ct.paired)
    assert EffectDecomposition("x", "", None, 1.0, 2.0, 3.0).generator_effect_pd is None


def test_suite_command_line(
    trained: tuple[StorageRoot, Path, PreprocessResult, ModelRecipe, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The suite CLI combines two paired reports into a loadable suite and a Markdown table."""
    store, records, processed, _, recipe_file = trained
    monkeypatch.setenv("ARM_RC_CTRL_STORAGE_ROOT", str(store.root))
    outputs: dict[str, Path] = {}
    for label, tracker in (("pd", DEV_PD), ("ct", REPO_ROOT / "configs" / "controllers" / "computed_torque.toml")):
        config = tmp_path / f"nominal_{label}.toml"
        text = NOMINAL.read_text().replace(
            'tracker = "../controllers/task_1a_pd_v2.toml"', f'tracker = "{tracker.as_posix()}"'
        )
        config.write_text(text)
        outputs[label] = tmp_path / f"paired_{label}.json"
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
            str(outputs[label]),
            "--exploratory",
        ]
        assert main(argv) == 0
        capsys.readouterr()
    suite_file = tmp_path / "suite.json"
    markdown_file = tmp_path / "suite.md"
    argv = ["--pd", str(outputs["pd"]), "--computed-torque", str(outputs["ct"])]
    argv += ["--report", str(suite_file), "--markdown", str(markdown_file)]
    assert suite_main(argv) == 0
    printed = json.loads(capsys.readouterr().out)
    assert set(printed["joint_rmse"]) == {"replay_pd", "rc_pd", "replay_ct", "rc_ct"}
    suite = load_paired_suite(suite_file)
    assert suite.pd.tracker == "pd"
    assert suite.computed_torque.tracker == "computed_torque"
    assert markdown_file.read_text().startswith("# Paired suite")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        suite_main(argv)

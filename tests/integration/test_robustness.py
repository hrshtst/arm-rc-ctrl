# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""The paired robustness suite runs every arm on identical scenarios and persists a validated report (M3-009)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from arm_rc_ctrl.config import load_config
from arm_rc_ctrl.controllers.tracking import TrackerConfig
from arm_rc_ctrl.data.preprocess import PreprocessResult, preprocess_demonstration
from arm_rc_ctrl.data.records import RawDemonstrationRecord, load_record
from arm_rc_ctrl.experiments.closed_loop import EstimatorSpec
from arm_rc_ctrl.experiments.confirmatory import ForcePulseLevels, PostureLevels, load_confirmatory
from arm_rc_ctrl.experiments.perturbations import DevelopmentRobustness
from arm_rc_ctrl.experiments.robustness import Arm, load_suite, main, run_robustness, suite_to_json, suite_to_markdown
from arm_rc_ctrl.rc.recipe import ModelRecipe, write_recipe
from arm_rc_ctrl.rc.runtime import load_training_samples
from arm_rc_ctrl.rc.train import load_model_config, train_task
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import load_scenario
from arm_rc_ctrl.storage import StorageRoot

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
MODEL = REPO_ROOT / "tests" / "fixtures" / "configs" / "esn_fixture.toml"
DEV_PD = REPO_ROOT / "configs" / "controllers" / "pd.toml"
RAW_RECORD = REPO_ROOT / "tests" / "fixtures" / "records" / "raw-20260830-287036d83d46.toml"
RAW_LOG = REPO_ROOT / "tests" / "fixtures" / "raw" / "demo.sklog.npz"
SCENARIO = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
PREPROCESS = REPO_ROOT / "configs" / "preprocessing" / "default.toml"
FIXED_TIME = datetime(2026, 9, 3, 14, 0, 0, tzinfo=UTC)
LEVELS = DevelopmentRobustness(
    name="fixture-robustness",
    scenario=SCENARIO,
    seeds=(7,),
    posture=PostureLevels(0.02, 0.04, 1),
    force=ForcePulseLevels(2.0, 0.1, 0.05, (0.0, 90.0)),
)


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[StorageRoot, Path, PreprocessResult, ModelRecipe, Path]:
    """The fixture dataset in a store and a small recipe trained on it."""
    base = tmp_path_factory.mktemp("robustness")
    root = base / "store"
    root.mkdir()
    store = StorageRoot(root, repositories=(REPO_ROOT,))
    raw = load_record(RAW_RECORD, RawDemonstrationRecord)
    store.path(raw.artifact.payload.uri, mode="write").write_bytes(RAW_LOG.read_bytes())
    records = base / "repo"
    (records / "data" / "records" / "processed").mkdir(parents=True)
    (records / "data" / "records" / "runs").mkdir(parents=True)
    processed = preprocess_demonstration(
        RAW_RECORD, SCENARIO, PREPROCESS, store=store, records_root=records, exploratory=True, now=FIXED_TIME
    )
    result = train_task(
        load_model_config(MODEL), MODEL, [processed.record_file], store=store, exploratory=True, now=FIXED_TIME,
        records_root=records,
    )  # fmt: skip
    recipe_file = base / "recipe.toml"
    write_recipe(recipe_file, result.recipe)
    return store, records, processed, result.recipe, recipe_file


def test_every_arm_runs_identical_scenarios_and_the_suite_round_trips(
    trained: tuple[StorageRoot, Path, PreprocessResult, ModelRecipe, Path], tmp_path: Path
) -> None:
    """Both arms see the same IDs, postures, and pulses; the report validates, renders, and reloads."""
    store, records, processed, recipe, recipe_file = trained
    scenario = load_scenario(SCENARIO)
    protocol_file = tmp_path / "levels.toml"
    protocol_file.write_text("# fixture levels\n", encoding="utf-8")
    seen: list[tuple[str, str]] = []
    runs_dir = store.root / "runs"

    def after_all(run: object, _result: object) -> object:
        # The callback must only run once every simulation has been persisted (a clean-worktree guard).
        assert len(list(runs_dir.iterdir())) >= 12
        seen.append((getattr(run, "arm"), getattr(run, "scenario_id")))  # noqa: B009
        return run

    suite = run_robustness(
        LEVELS,
        protocol_file,
        label="development",
        scenario=scenario,
        scenario_file=SCENARIO,
        dataset=processed.record,
        reference=processed.samples,
        recipe=recipe,
        recipe_file=recipe_file,
        estimator=EstimatorSpec(20.0, 20.0),
        trackers={"pd": load_config(DEV_PD, TrackerConfig)},
        training_samples=load_training_samples(recipe, store, records_root=records),
        store=store,
        exploratory=True,
        now=FIXED_TIME,
        on_run=after_all,  # type: ignore[arg-type]
    )
    assert [a.name for a in suite.arms] == ["rc+pd", "replay+pd"]
    ids = [s.scenario_id for s in suite.scenarios]
    assert ids == [
        "nominal",
        "posture-small-7-00",
        "posture-large-7-00",
        "force-2N-000deg",
        "force-2N-090deg",
        "combined-7-00-000deg",
    ]
    assert len(suite.runs) == 12
    assert seen == [(arm, i) for i in ids for arm in ("rc+pd", "replay+pd")]
    for scenario_id in ids:
        rc, rp = (r for r in suite.runs if r.scenario_id == scenario_id)
        assert rc.arm == "rc+pd"
        assert rp.arm == "replay+pd"
        assert rc.run_id != rp.run_id
        assert rc.report.reference_artifact == rp.report.reference_artifact == processed.record.artifact.artifact_id
    assert {a.kind for a in suite.aggregates} == {"nominal", "posture_small", "posture_large", "force", "combined"}
    assert sum(a.n for a in suite.aggregates) == 12
    assert all(a.n == a.successes + sum(a.failures.values()) for a in suite.aggregates)
    assert {e.metric for e in suite.effects} >= {"joint_rmse", "effort_torque_rms"}
    assert suite.provenance.seeds == {"protocol.0": 7}
    assert suite.provenance.exploratory
    report = tmp_path / "suite.json"
    report.write_text(suite_to_json(suite) + "\n", encoding="utf-8")
    assert load_suite(report) == suite
    text = suite_to_markdown(suite)
    assert text.startswith("# Robustness suite `fixture-robustness` (development)")
    assert "6 scenarios x 2 arms = 12 runs" in text
    tampered = json.loads(report.read_text(encoding="utf-8"))
    tampered["aggregates"][0]["successes"] = 99
    report.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="stored aggregates"):
        load_suite(report)
    with pytest.raises(ValueError, match="cannot be run under the development label"):
        run_robustness(
            load_confirmatory(REPO_ROOT / "configs" / "evaluations" / "task_1a_confirmatory_v2.toml"),
            protocol_file,
            label="development",
            scenario=scenario,
            scenario_file=SCENARIO,
            dataset=processed.record,
            reference=processed.samples,
            recipe=recipe,
            recipe_file=recipe_file,
            estimator=EstimatorSpec(20.0, 20.0),
            trackers={"pd": load_config(DEV_PD, TrackerConfig)},
            training_samples=load_training_samples(recipe, store, records_root=records),
            store=store,
            exploratory=True,
        )
    with pytest.raises(ValueError, match="needs the locked confirmatory protocol"):
        run_robustness(
            LEVELS,
            protocol_file,
            label="confirmatory",
            scenario=scenario,
            scenario_file=SCENARIO,
            dataset=processed.record,
            reference=processed.samples,
            recipe=recipe,
            recipe_file=recipe_file,
            estimator=EstimatorSpec(20.0, 20.0),
            trackers={"pd": load_config(DEV_PD, TrackerConfig)},
            training_samples=load_training_samples(recipe, store, records_root=records),
            store=store,
            exploratory=True,
        )
    with pytest.raises(ValueError, match="without a configuration"):
        run_robustness(
            LEVELS,
            protocol_file,
            label="development",
            scenario=scenario,
            scenario_file=SCENARIO,
            dataset=processed.record,
            reference=processed.samples,
            recipe=recipe,
            recipe_file=recipe_file,
            estimator=EstimatorSpec(20.0, 20.0),
            trackers={"pd": load_config(DEV_PD, TrackerConfig)},
            training_samples=load_training_samples(recipe, store, records_root=records),
            store=store,
            exploratory=True,
            arms=(Arm("rc+ct", "rc", "computed_torque"),),
        )


def test_command_writes_report_pointers_and_refuses_overwrites(
    trained: tuple[StorageRoot, Path, PreprocessResult, ModelRecipe, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """The command runs a class subset with registry trackers, tracks pointers, and prints per-class outcomes."""
    store, records, processed, _, recipe_file = trained
    monkeypatch.setenv("ARM_RC_CTRL_STORAGE_ROOT", str(store.root))
    levels = tmp_path / "levels.toml"
    levels.write_text(
        "\n".join(
            [
                'name = "fixture-robustness-cli"',
                f'scenario = "{SCENARIO}"',
                "seeds = [7]",
                "[posture]",
                "small_magnitude_rad = 0.02",
                "large_magnitude_rad = 0.04",
                "draws_per_seed = 1",
                "[force]",
                "magnitude_n = 2.0",
                "start_s = 0.1",
                "duration_s = 0.05",
                "directions_deg = [0.0]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    evaluation = tmp_path / "evaluation.toml"
    evaluation.write_text(
        f'name = "fixture"\ntracker = "{DEV_PD}"\n[estimator]\n'
        "velocity_cutoff_hz = 20.0\nacceleration_cutoff_hz = 20.0\n",
        encoding="utf-8",
    )
    report = tmp_path / "suite.json"
    argv = [
        "--development",
        str(levels),
        "--dataset",
        str(processed.record_file),
        "--recipe",
        str(recipe_file),
        "--evaluation",
        str(evaluation),
        "--trackers",
        "pd",
        "--classes",
        "nominal",
        "force",
        "--label",
        "development",
        "--report",
        str(report),
        "--markdown",
        str(tmp_path / "suite.md"),
        "--records-root",
        str(records),
        "--exploratory",
        "--no-mlflow",
    ]
    assert main(argv) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["runs"] == 4
    assert set(printed["arms"]) == {"rc+pd/nominal", "rc+pd/force", "replay+pd/nominal", "replay+pd/force"}
    suite = load_suite(report)
    assert [s.kind for s in suite.scenarios] == ["nominal", "force"]
    assert all(r.pointer is not None and (records / r.pointer).is_file() for r in suite.runs)
    assert all(r.mlflow_run_id is None for r in suite.runs)
    assert (tmp_path / "suite.md").read_text(encoding="utf-8").startswith("# Robustness suite `fixture-robustness-cli`")
    with pytest.raises(FileExistsError, match="refusing"):
        main(argv)

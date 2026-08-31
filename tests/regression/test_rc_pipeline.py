# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M2-015: the raw fixture reproduces, through training and closed-loop evaluation, within declared tolerances.

The pipeline is deterministic in one process (UP-005 guard) and its committed
snapshot under ``tests/fixtures/regression`` is compared with tolerances that
cover platform differences in floating-point evaluation; regenerate it with
``pytest --update-baselines`` after an intentional change.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from arm_rc_ctrl.config import load_config, to_mapping
from arm_rc_ctrl.controllers.tracking import TrackerConfig
from arm_rc_ctrl.data.preprocess import preprocess_demonstration
from arm_rc_ctrl.data.records import RawDemonstrationRecord, load_record
from arm_rc_ctrl.experiments.baselines import (
    PipelineExpectations,
    PipelineSnapshot,
    Tolerances,
    compare_pipeline,
    load_pipeline_expectations,
    snapshot,
    write_pipeline_expectations,
)
from arm_rc_ctrl.experiments.closed_loop import EstimatorSpec
from arm_rc_ctrl.experiments.paired import PairedResult, run_paired_nominal
from arm_rc_ctrl.provenance import canonical_json
from arm_rc_ctrl.rc.runtime import load_training_samples
from arm_rc_ctrl.rc.train import TrainingResult, load_model_config, train_task
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import load_scenario
from arm_rc_ctrl.storage import StorageRoot

pytestmark = [pytest.mark.regression, pytest.mark.integration]

REPO_ROOT = repository_root()
EXPECTATIONS = REPO_ROOT / "tests" / "fixtures" / "regression" / "rc_pipeline_fixture.toml"
MODEL = REPO_ROOT / "tests" / "fixtures" / "configs" / "esn_fixture.toml"
DEV_PD = REPO_ROOT / "configs" / "controllers" / "pd.toml"
RAW_RECORD = REPO_ROOT / "tests" / "fixtures" / "records" / "raw-20260830-287036d83d46.toml"
RAW_LOG = REPO_ROOT / "tests" / "fixtures" / "raw" / "demo.sklog.npz"
SCENARIO = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
PREPROCESS = REPO_ROOT / "configs" / "preprocessing" / "default.toml"
FIXED_TIME = datetime(2026, 9, 1, 20, 0, 0, tzinfo=UTC)
TOLERANCES = Tolerances(metric_rel=1e-6, state_abs=1e-6)


def _pipeline(base: Path, run_time: datetime) -> tuple[TrainingResult, PairedResult]:
    """Raw fixture -> fresh store -> preprocessing -> training -> paired nominal evaluation."""
    root = base / "store"
    root.mkdir(parents=True)
    store = StorageRoot(root, repositories=(REPO_ROOT,))
    raw = load_record(RAW_RECORD, RawDemonstrationRecord)
    store.path(raw.artifact.payload.uri, mode="write").write_bytes(RAW_LOG.read_bytes())
    records = base / "repo"
    (records / "data" / "records" / "processed").mkdir(parents=True)
    processed = preprocess_demonstration(
        RAW_RECORD, SCENARIO, PREPROCESS, store=store, records_root=records, exploratory=True, now=FIXED_TIME
    )
    training = train_task(
        load_model_config(MODEL),
        MODEL,
        [processed.record_file],
        store=store,
        exploratory=True,
        now=FIXED_TIME,
        records_root=records,
    )
    scenario = load_scenario(SCENARIO)
    paired = run_paired_nominal(
        scenario,
        SCENARIO,
        processed.record,
        processed.samples,
        training.recipe,
        "fixture",
        load_config(DEV_PD, TrackerConfig),
        store=store,
        estimator=EstimatorSpec(20.0, 20.0).config(scenario.timing.dt),
        training_samples=load_training_samples(training.recipe, store, records_root=records),
        exploratory=True,
        now=run_time,
    )
    return training, paired


def _snapshot(training: TrainingResult, paired: PairedResult) -> PipelineSnapshot:
    fit = training.recipe.fit
    scalars = {"rmse": fit.rmse, "constant_rmse": fit.constant_rmse, "max_abs_error": fit.max_abs_error}
    scalars.update({f"rmse_joint_{i}": v for i, v in enumerate(fit.rmse_per_joint)})
    assert paired.rc.boundary_jump is not None
    return PipelineSnapshot(
        recipe_id=training.report.recipe_id,
        fit=scalars,
        boundary_jump=paired.rc.boundary_jump,
        rc=snapshot(cast("Any", paired.rc)),
        replay=snapshot(paired.replay),
    )


def _expectations(training: TrainingResult, paired: PairedResult) -> PipelineExpectations:
    gains = load_config(DEV_PD, TrackerConfig)
    return PipelineExpectations(
        scenario=paired.paired.scenario,
        dataset=paired.paired.reference_artifact,
        model_config_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        gains=hashlib.sha256(canonical_json(to_mapping(gains)).encode("utf-8")).hexdigest(),
        tolerances=TOLERANCES,
        snapshot=_snapshot(training, paired),
    )


def test_pipeline_is_deterministic_in_process(tmp_path: Path) -> None:
    """Two complete runs from the raw fixture give identical recipes, fit reports, and telemetry."""
    first_training, first = _pipeline(tmp_path / "a", FIXED_TIME)
    second_training, second = _pipeline(tmp_path / "b", FIXED_TIME)
    assert first_training.recipe == second_training.recipe
    assert first_training.report.recipe_id == second_training.report.recipe_id
    for name, array in first.rc.run.arrays.arrays.items():
        assert np.array_equal(array, second.rc.run.arrays.arrays[name]), name
    assert first.rc.summary.arrays == second.rc.summary.arrays
    assert first.replay.summary.arrays == second.replay.summary.arrays
    assert _snapshot(first_training, first) == _snapshot(second_training, second)


def test_pipeline_matches_committed_expectations(tmp_path: Path, *, update_baselines: bool) -> None:
    """The committed snapshot of recipe, fit, boundary jump, and both runs reproduces within tolerance."""
    training, paired = _pipeline(tmp_path, FIXED_TIME)
    actual = _expectations(training, paired)
    if update_baselines:
        write_pipeline_expectations(EXPECTATIONS, actual)
        pytest.skip(
            f"rewrote {EXPECTATIONS.relative_to(REPO_ROOT)}; review the diff and re-run without --update-baselines"
        )
    expected = load_pipeline_expectations(EXPECTATIONS)
    assert expected.scenario == actual.scenario
    assert expected.dataset == actual.dataset
    assert expected.model_config_sha256 == actual.model_config_sha256, "the fixture ESN config changed; regenerate"
    assert expected.gains == actual.gains, "the development PD gains changed; regenerate"
    mismatches = compare_pipeline(actual.snapshot, expected.snapshot, expected.tolerances)
    assert not mismatches, "\n".join(mismatches)
    assert paired.rc.report.termination_kind == "completed"
    assert json.loads(json.dumps(actual.snapshot.fit))  # plain numbers only

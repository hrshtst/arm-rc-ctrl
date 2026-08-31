# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M2-013: invalid generated commands end the run safely with a structured, categorized termination."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from arm_rc_ctrl.config import load_config
from arm_rc_ctrl.controllers.contracts import DesiredJointState, GeneratorError, RobotState
from arm_rc_ctrl.controllers.estimator import CausalDerivativeEstimator, EstimatorConfig, EstimatorError
from arm_rc_ctrl.controllers.tracking import TrackerConfig
from arm_rc_ctrl.data.preprocess import PreprocessResult, preprocess_demonstration
from arm_rc_ctrl.data.records import RawDemonstrationRecord, load_record
from arm_rc_ctrl.experiments.closed_loop import ClosedLoopResult, EstimatorSpec, run_nominal
from arm_rc_ctrl.experiments.run_record import load_run
from arm_rc_ctrl.experiments.termination import FAILURE_KINDS, Termination, invalid_output
from arm_rc_ctrl.rc import generator as generator_module
from arm_rc_ctrl.rc.recipe import ModelRecipe
from arm_rc_ctrl.rc.runtime import generator_from_recipe, load_training_samples
from arm_rc_ctrl.rc.train import load_model_config, train_task
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import load_scenario
from arm_rc_ctrl.storage import StorageRoot

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
MODEL = REPO_ROOT / "configs" / "models" / "esn_task_1a.toml"
DEV_PD = REPO_ROOT / "configs" / "controllers" / "pd.toml"
RAW_RECORD = REPO_ROOT / "tests" / "fixtures" / "records" / "raw-20260830-287036d83d46.toml"
RAW_LOG = REPO_ROOT / "tests" / "fixtures" / "raw" / "demo.sklog.npz"
SCENARIO = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
PREPROCESS = REPO_ROOT / "configs" / "preprocessing" / "default.toml"
FIXED_TIME = datetime(2026, 9, 1, 17, 0, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[StorageRoot, Path, PreprocessResult, ModelRecipe]:
    """The fixture dataset in a store and a small recipe trained on it."""
    base = tmp_path_factory.mktemp("safety")
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
    return store, records, processed, result.recipe


def _run(
    trained: tuple[StorageRoot, Path, PreprocessResult, ModelRecipe],
    monkeypatch: pytest.MonkeyPatch,
    fault: Callable[[np.ndarray], np.ndarray] | None,
    *,
    at_step: int = 10,
    estimator: EstimatorConfig | None = None,
) -> tuple[Termination, int, ClosedLoopResult]:
    """Run the nominal loop with ``EsnModel.step`` replaced from ``at_step`` on."""
    store, records, processed, recipe = trained
    scenario = load_scenario(SCENARIO)
    training = load_training_samples(recipe, store, records_root=records)
    original = generator_module.RcTargetGenerator._step  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    calls = {"n": 0}

    def patched(self: generator_module.RcTargetGenerator, state: RobotState, code: np.ndarray) -> DesiredJointState:
        calls["n"] += 1
        if fault is None or calls["n"] <= at_step:
            return original(self, state, code)
        model = self.model
        model.step = fault  # type: ignore[method-assign] - instance-level override of the fitted model's step
        try:
            return original(self, state, code)
        finally:
            del model.step

    monkeypatch.setattr(generator_module.RcTargetGenerator, "_step", patched)
    config = EstimatorSpec(20.0, 20.0).config(scenario.timing.dt) if estimator is None else estimator
    result = run_nominal(
        scenario,
        SCENARIO,
        processed.record,
        processed.samples,
        recipe,
        load_config(DEV_PD, TrackerConfig),
        store=store,
        estimator=config,
        training_samples=training,
        exploratory=True,
        now=FIXED_TIME,
    )
    monkeypatch.undo()
    return result.summary.termination, result.run.arrays.n_samples, result


def _returning(value: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    def fault(_u: np.ndarray) -> np.ndarray:
        return value

    return fault


def _raise(exc: Exception) -> Callable[[np.ndarray], np.ndarray]:
    def fault(_u: np.ndarray) -> np.ndarray:
        raise exc

    return fault


@pytest.mark.parametrize(
    ("fault", "failure", "fragment"),
    [
        (_returning(np.array([np.nan, 0.0])), "non_finite", "non-finite target"),
        (_returning(np.array([0.0, 0.0, 0.0])), "shape", "target of shape (3,)"),
        (_returning(np.array([5.0, 0.0])), "bounds", "leaves the joint bounds"),
        (_raise(RuntimeError("reservoir exploded")), "model_exception", "RuntimeError: reservoir exploded"),
    ],
    ids=["nan", "shape", "bounds", "model-exception"],
)
def test_invalid_commands_end_the_run_safely(
    trained: tuple[StorageRoot, Path, PreprocessResult, ModelRecipe],
    monkeypatch: pytest.MonkeyPatch,
    fault: Callable[[np.ndarray], np.ndarray],
    failure: str,
    fragment: str,
) -> None:
    """A NaN, mis-shaped, out-of-bounds, or raising model ends the run with an invalid_output termination."""
    termination, n_samples, result = _run(trained, monkeypatch, fault)
    assert termination.kind == "invalid_output"
    assert termination.failure == failure
    assert fragment in termination.detail
    assert termination.step == n_samples  # every sample before the faulty command is logged
    assert n_samples > 10  # the hold and the first generated samples were healthy
    assert result.summary.outcome.success is False
    assert result.summary.outcome.criteria["completed"] is False
    assert result.report.termination_kind == "invalid_output"
    assert result.report.move_coverage < 1.0
    assert np.all(np.isfinite(result.run.arrays.arrays["q_desired"]))  # nothing invalid reached the record
    reloaded = load_run(_store_of(result), result.pointer)
    assert reloaded.summary.termination == termination


def _store_of(result: ClosedLoopResult) -> StorageRoot:
    return StorageRoot(result.run.directory.parent.parent, repositories=(REPO_ROOT,))


def test_stale_time_is_a_structured_failure(
    trained: tuple[StorageRoot, Path, PreprocessResult, ModelRecipe], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An estimator that rejects the control interval, or a generator fed non-advancing time, reports stale_time."""
    termination, n_samples, _ = _run(trained, monkeypatch, None, estimator=EstimatorConfig(0.01, max_dt_ratio=1.5))
    assert termination.kind == "completed"  # regular intervals within the bound are accepted
    termination, n_samples, _ = _run(trained, monkeypatch, None, estimator=EstimatorConfig(0.004, max_dt_ratio=2.0))
    assert termination.kind == "invalid_output"  # a 10 ms step exceeds the 8 ms bound: the estimator refuses it
    assert termination.failure == "stale_time"
    assert "EstimatorError" in termination.detail
    assert n_samples == 1
    store, records, processed, recipe = trained
    training = load_training_samples(recipe, store, records_root=records)
    generator = generator_from_recipe(recipe, training, estimator=EstimatorConfig(0.01, max_dt_ratio=1.0))
    state = RobotState(0.0, processed.samples.q[0], processed.samples.dq[0])
    generator.reset(state)
    generator.step(state)
    with pytest.raises(GeneratorError, match="time must advance") as excinfo:
        generator.step(state)
    assert excinfo.value.category == "stale_time"
    later = RobotState(0.05, processed.samples.q[0], processed.samples.dq[0])
    with pytest.raises(EstimatorError, match="exceeds the accepted maximum") as est:
        generator.step(later)
    assert est.value.category == "stale_time"
    estimator = CausalDerivativeEstimator(EstimatorConfig(0.01), 2)
    estimator.update(0.0, np.zeros(2))
    with pytest.raises(EstimatorError) as backwards:
        estimator.update(0.0, np.zeros(2))
    assert backwards.value.category == "stale_time"
    del n_samples


def test_failure_category_is_part_of_the_termination_schema() -> None:
    """The category is validated, restricted to invalid_output, and round-trips through the mapping."""
    assert set(FAILURE_KINDS) == {"non_finite", "shape", "bounds", "stale_time", "model_exception"}
    termination = invalid_output(0.5, 50, "ESN returned NaN", "non_finite")
    assert termination.failure == "non_finite"
    with pytest.raises(ValueError, match="failure must be one of"):
        invalid_output(0.5, 50, "x", cast("Any", "cosmic_ray"))
    with pytest.raises(ValueError, match="failure is only valid for invalid_output"):
        Termination("completed", 1.0, 100, failure="shape")
    assert invalid_output(0.5, 50, "plain").failure is None

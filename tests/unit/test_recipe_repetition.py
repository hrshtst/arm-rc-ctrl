# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3REP-002: schema 2 recipes bind their validation, repeat one episode literally, and derive the solver alpha."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from arm_rc_ctrl.controllers.contracts import RobotState
from arm_rc_ctrl.controllers.estimator import EstimatorConfig
from arm_rc_ctrl.data.derivatives import DerivativeConfig, differentiate
from arm_rc_ctrl.data.normalization import fit_normalization
from arm_rc_ctrl.data.records import Preprocessing
from arm_rc_ctrl.data.samples import SampleSet
from arm_rc_ctrl.rc.esn import EsnConfig, EsnModel, ReadoutConfig, ReservoirConfig
from arm_rc_ctrl.rc.recipe import (
    APPROVED_ADDITIONAL_REPEATS,
    BINDING_SCHEMA_VERSION,
    AugmentationTrainingSpec,
    DatasetSource,
    ModelRecipe,
    TrainingSpec,
    TrainingValidation,
    create_recipe,
    expected_episode_labels,
    load_recipe,
    solver_alpha,
    write_recipe,
)
from arm_rc_ctrl.rc.runtime import generator_from_recipe
from arm_rc_ctrl.rc.teacher_forcing import InputTransform
from arm_rc_ctrl.rc.training import harvest_episode
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import ScenarioConfig, endpoint_positions, load_scenario

REPO_ROOT = repository_root()
SCENARIO_FILE = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
SCENARIO = load_scenario(SCENARIO_FILE)
DERIVATIVES = DerivativeConfig(method="central")
SOURCE = DatasetSource("processed-20260830-555555555555", "ab" * 32, "data/records/processed/x.toml")
N = 101
DT = 0.01
ROWS = N - 1
ALPHA = 0.02
ESN = EsnConfig(
    reservoir=ReservoirConfig(
        n_neurons=40, spectral_radius=0.85, sparsity=0.9, leak_rate=0.4, input_scaling=0.4, seed=23
    ),
    readout=ReadoutConfig(alpha=ALPHA),
)
PREPROCESSING = Preprocessing(
    resample_period_s=DT, smoothing="none", smoothing_params={}, derivative_method="central-difference"
)
AUGMENTATION = AugmentationTrainingSpec(
    family="contractive", n_synthetic=16, sigma_rad=0.025, phi=0.99, gamma=1.0, seed_bank=1, attempt_budget=64
)


def _samples() -> SampleSet:
    t = np.arange(N, dtype=np.float64) * DT
    start = np.array(SCENARIO.task.initial_q)
    goal = np.array([0.8, 0.4])
    s = np.clip(t / 0.8, 0.0, 1.0)
    blend = s * s * (3.0 - 2.0 * s)
    q = start[None, :] + blend[:, None] * (goal - start)[None, :]
    dq, ddq = differentiate(q, DT, DERIVATIVES)
    tip = endpoint_positions(SCENARIO, q)
    dtip, ddtip = differentiate(tip, DT, DERIVATIVES)
    phase = np.where(t < 0.8, 1, 2).astype(np.int64)
    return SampleSet(t, q, dq, ddq, tip, dtip, ddtip, np.zeros((N, 0)), phase)


def _transform(samples: SampleSet) -> InputTransform:
    normalization = fit_normalization(
        samples.arrays(), ("q", "dq"), fitted_on=(SOURCE.artifact_id,), training_rows=np.ones(N, dtype=np.bool_)
    )
    return InputTransform.derive("fixed_scale", normalization, fixed_scales={"q": 0.3, "dq": 4.0})


def _validation() -> TrainingValidation:
    return TrainingValidation.from_scenario(SCENARIO, SCENARIO_FILE, root=REPO_ROOT)


def _build(
    samples: SampleSet,
    spec: TrainingSpec,
    *,
    esn: EsnConfig = ESN,
    scenario: ScenarioConfig | None = SCENARIO,
    validation: TrainingValidation | None = None,
) -> tuple[ModelRecipe, EsnModel]:
    return create_recipe(
        "repetition-test",
        esn,
        sources=[SOURCE],
        samples={SOURCE.artifact_id: samples},
        dof=2,
        task_code_dim=0,
        preprocessing=PREPROCESSING,
        transform=_transform(samples),
        training=spec,
        scenario=scenario,
        validation=validation,
    )


def _relaxed(scenario: ScenarioConfig) -> ScenarioConfig:
    limits = dataclasses.replace(scenario.limits, velocity=tuple(2.0 * v for v in scenario.limits.velocity))
    return dataclasses.replace(scenario, limits=limits)


def test_solver_alpha_rules_and_their_validation() -> None:
    """The three rules give alpha_0, K alpha_0, and alpha_0 / K; nonsense inputs are refused."""
    assert solver_alpha(0.01, "base", 17) == 0.01
    assert solver_alpha(0.01, "count_scaled", 17) == 0.01 * 17
    assert solver_alpha(0.01, "count_divided", 65) == 0.01 / 65
    assert solver_alpha(0.5, "count_scaled", 65) > 1.0  # above the historical search bound, deliberately allowed
    assert solver_alpha(0.001, "count_divided", 65) < 0.001  # below it, deliberately allowed
    with pytest.raises(ValueError, match="regularization_rule"):
        solver_alpha(0.01, "halved", 2)
    with pytest.raises(ValueError, match="base_alpha"):
        solver_alpha(0.0, "base", 1)
    with pytest.raises(ValueError, match="count"):
        solver_alpha(0.01, "base", 0)


def test_repetition_spec_validation() -> None:
    """Approved counts only, warm-up washout required, no augmentation alongside, rule and base recorded together."""
    assert frozenset({16, 32, 64}) == APPROVED_ADDITIONAL_REPEATS
    with pytest.raises(ValueError, match="approved counts"):
        TrainingSpec(washout="warmup_hold", warmup_s=0.25, additional_repeats=8)
    with pytest.raises(ValueError, match="warmup_hold"):
        TrainingSpec(additional_repeats=16)
    with pytest.raises(ValueError, match="mutually exclusive"):
        TrainingSpec(washout="warmup_hold", warmup_s=0.25, additional_repeats=16, augmentation=AUGMENTATION)
    with pytest.raises(ValueError, match="recorded together"):
        TrainingSpec(washout="warmup_hold", warmup_s=0.25, base_alpha=0.01)
    with pytest.raises(ValueError, match="needs base_alpha"):
        TrainingSpec(washout="warmup_hold", warmup_s=0.25, regularization_count=17)
    with pytest.raises(ValueError, match="regularization_rule"):
        TrainingSpec(washout="warmup_hold", warmup_s=0.25, base_alpha=0.01, regularization_rule="halved")
    spec = TrainingSpec(
        washout="warmup_hold", warmup_s=0.25, additional_repeats=16, base_alpha=ALPHA, regularization_rule="base"
    )
    assert spec.episode_count == 17
    assert spec.formulation == "absolute"
    assert spec.uses_binding_features
    residual = TrainingSpec(washout="warmup_hold", warmup_s=0.0, target="increment_q", additional_repeats=32)
    assert residual.episode_count == 33
    assert residual.formulation == "residual"
    assert TrainingSpec().episode_count == 1
    assert not TrainingSpec().uses_binding_features
    assert TrainingSpec(washout="warmup_hold", warmup_s=0.25, augmentation=AUGMENTATION).episode_count == 17


def test_repeat_labels_and_literal_copies_with_identical_states() -> None:
    """Sixteen copies carry deterministic labels, identical arrays and masks, and identical harvested states."""
    samples = _samples()
    spec = TrainingSpec(washout="warmup_hold", warmup_s=0.25, additional_repeats=16)
    labels = expected_episode_labels(spec, (SOURCE.artifact_id,))
    assert labels[0] == SOURCE.artifact_id
    assert labels[1] == f"{SOURCE.artifact_id}#repeat-001"
    assert labels[-1] == f"{SOURCE.artifact_id}#repeat-016"
    assert len(labels) == 17
    with pytest.raises(ValueError, match="exactly one dataset"):
        expected_episode_labels(spec, (SOURCE.artifact_id, "processed-20260830-666666666666"))
    recipe, model = _build(samples, spec, validation=_validation())
    episodes = recipe.episodes({SOURCE.artifact_id: samples}, scenario=SCENARIO)
    assert [e.source for e in episodes] == list(labels)
    original = episodes[0]
    reference = harvest_episode(model, original)
    for copy in episodes[1:]:
        assert np.array_equal(copy.inputs, original.inputs)
        assert np.array_equal(copy.targets, original.targets)
        assert np.array_equal(copy.loss_rows, original.loss_rows)
        assert np.array_equal(copy.t, original.t)
        assert copy.washout_len == original.washout_len == 25
        harvested = harvest_episode(model, copy)
        assert np.array_equal(harvested.states, reference.states)
    assert recipe.fit.loss_rows == 17 * ROWS
    assert recipe.fit.washout_rows == 17 * 25
    assert recipe.fit.episodes == labels


def test_repeated_recipe_is_schema_two_bound_and_refits_exactly(tmp_path: Path) -> None:
    """The recipe writes schema 2 with its validation, reloads equal, refits bitwise, and refuses relaxed limits."""
    samples = _samples()
    spec = TrainingSpec(
        washout="warmup_hold",
        warmup_s=0.25,
        additional_repeats=16,
        base_alpha=ALPHA,
        regularization_rule="count_scaled",
        regularization_count=17,
    )
    scaled = dataclasses.replace(
        ESN, readout=dataclasses.replace(ESN.readout, alpha=solver_alpha(ALPHA, "count_scaled", 17))
    )
    recipe, _ = _build(samples, spec, esn=scaled, validation=_validation())
    assert recipe.schema_version == BINDING_SCHEMA_VERSION
    assert recipe.validation is not None
    assert recipe.validation.scenario_file == "tests/fixtures/configs/planar_2dof_fixture.toml"
    assert recipe.validation.velocity_limit == tuple(SCENARIO.limits.velocity)
    assert recipe.solver_alpha == ALPHA * 17
    assert recipe.formulation == "absolute"
    assert recipe.output == "absolute"
    file = tmp_path / "repeated.toml"
    write_recipe(file, recipe)
    text = file.read_text(encoding="utf-8")
    assert "schema_version = 2" in text
    assert "additional_repeats = 16" in text
    assert "regularization_rule" in text
    assert "[validation]" in text
    loaded = load_recipe(file)
    assert loaded == recipe
    _model, report = loaded.refit({SOURCE.artifact_id: samples}, scenario=SCENARIO)
    assert report == recipe.fit
    with pytest.raises(ValueError, match="training-validation limits"):
        loaded.refit({SOURCE.artifact_id: samples}, scenario=_relaxed(SCENARIO))
    with pytest.raises(ValueError, match="training-validation limits"):
        _build(samples, spec, esn=scaled, scenario=_relaxed(SCENARIO), validation=_validation())


def test_schema_rules_keep_legacy_recipes_legacy() -> None:
    """Schema 1 carries no validation and no repetition; schema 2 requires the validation; alpha follows the rule."""
    samples = _samples()
    plain = TrainingSpec(washout="warmup_hold", warmup_s=0.25)
    legacy, _ = _build(samples, plain)
    assert legacy.schema_version == 1
    assert legacy.validation is None
    with pytest.raises(ValueError, match="schema 1 recipes carry no training validation"):
        dataclasses.replace(legacy, validation=_validation())
    with pytest.raises(ValueError, match="schema 2 recipes bind"):
        dataclasses.replace(legacy, schema_version=2)
    repeated = TrainingSpec(washout="warmup_hold", warmup_s=0.25, additional_repeats=16)
    with pytest.raises(ValueError, match="need a schema 2 recipe"):
        dataclasses.replace(
            legacy,
            training=repeated,
            fit=dataclasses.replace(legacy.fit, episodes=expected_episode_labels(repeated, (SOURCE.artifact_id,))),
        )
    bound, _ = _build(samples, plain, validation=_validation())
    assert bound.schema_version == 2
    with pytest.raises(ValueError, match="unsupported recipe schema version 4"):
        dataclasses.replace(bound, schema_version=4)
    ruled = TrainingSpec(
        washout="warmup_hold",
        warmup_s=0.25,
        base_alpha=ALPHA,
        regularization_rule="count_divided",
        regularization_count=17,
    )
    with pytest.raises(ValueError, match="must equal"):
        dataclasses.replace(bound, training=ruled)  # readout alpha is still alpha_0, not alpha_0 / 17
    effective = dataclasses.replace(
        ESN, readout=dataclasses.replace(ESN.readout, alpha=solver_alpha(ALPHA, "count_divided", 17))
    )
    reference, _ = _build(samples, ruled, esn=effective, validation=_validation())
    assert reference.solver_alpha == ALPHA / 17
    assert reference.fit.loss_rows == ROWS  # S-effective fits the original once


def test_training_validation_binding_and_checks() -> None:
    """The binding names the scenario file and limits; malformed bindings and other roots are refused."""
    validation = _validation()
    assert validation.joint_lower < validation.joint_upper
    assert validation.endpoint_radius == SCENARIO.limits.endpoint_radius
    validation.check(SCENARIO)
    with pytest.raises(ValueError, match="training-validation limits"):
        validation.check(_relaxed(SCENARIO))
    with pytest.raises(ValueError, match="outside the repository root"):
        TrainingValidation.from_scenario(SCENARIO, SCENARIO_FILE, root=Path("/nonexistent-root"))
    with pytest.raises(ValueError, match="hex"):
        dataclasses.replace(validation, scenario_sha256="zz")
    with pytest.raises(ValueError, match="same non-empty joint set"):
        dataclasses.replace(validation, velocity_limit=(1.0,))
    with pytest.raises(ValueError, match="positive"):
        dataclasses.replace(validation, endpoint_radius=0.0)
    with pytest.raises(ValueError, match="lower < upper"):
        dataclasses.replace(validation, joint_lower=validation.joint_upper)
    with pytest.raises(ValueError, match="finite"):
        dataclasses.replace(validation, endpoint_radius=float("inf"))


def test_residual_repetition_reconstructs_commands_from_the_measured_posture() -> None:
    """A residual repeated recipe trains on increments, refits exactly, and commands q_measured + increment."""
    samples = _samples()
    spec = TrainingSpec(
        washout="warmup_hold",
        warmup_s=0.25,
        target="increment_q",
        additional_repeats=16,
        base_alpha=ALPHA,
        regularization_rule="base",
    )
    recipe, _ = _build(samples, spec, validation=_validation())
    assert recipe.formulation == "residual"
    assert recipe.output == "increment"
    assert recipe.training.augmentation is None  # no contractive requirement is inherited
    episodes = recipe.episodes({SOURCE.artifact_id: samples}, scenario=SCENARIO)
    first = episodes[0]
    assert np.array_equal(first.targets[first.washout_len :], np.diff(samples.q, axis=0))
    assert np.array_equal(episodes[5].targets, first.targets)
    _, report = recipe.refit({SOURCE.artifact_id: samples}, scenario=SCENARIO)
    assert report == recipe.fit
    generator = generator_from_recipe(
        recipe, {SOURCE.artifact_id: samples}, estimator=EstimatorConfig(nominal_dt_s=DT), scenario=SCENARIO
    )
    start = RobotState(0.0, samples.q[0].copy(), samples.dq[0].copy())
    generator.reset(start)
    measured = RobotState(DT, samples.q[3].copy(), samples.dq[3].copy())
    desired = generator.step(measured)
    telemetry = generator.last
    increment = telemetry["generator_increment_q"]
    assert np.all(np.isfinite(increment))
    assert np.array_equal(desired.q, measured.q + increment)
    assert np.array_equal(telemetry["generator_output_q"], desired.q)

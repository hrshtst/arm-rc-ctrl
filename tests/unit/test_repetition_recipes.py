# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3REP-002: each pilot arm maps to one recipe with its prescribed ridge parameter and environment-bound identity."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from arm_rc_ctrl.data.derivatives import DerivativeConfig, differentiate
from arm_rc_ctrl.data.normalization import fit_normalization
from arm_rc_ctrl.data.records import Preprocessing
from arm_rc_ctrl.data.samples import SampleSet
from arm_rc_ctrl.experiments.closed_loop import EstimatorSpec
from arm_rc_ctrl.experiments.esn_search import TrialPoint
from arm_rc_ctrl.experiments.recovery_search import RecoveryTrialPoint
from arm_rc_ctrl.experiments.repetition_panel import PanelEntry
from arm_rc_ctrl.experiments.repetition_recipes import (
    ARMS,
    AUGMENTATION_ANCHOR,
    RESIDUAL_ARMS,
    ROWS_PER_EPISODE,
    ArmSpec,
    AugmentationAnchor,
    expected_loss_rows,
    fit_identity,
    panel_arms,
    readout_for_arm,
    recipe_for_arm,
    training_spec_for_arm,
)
from arm_rc_ctrl.rc.esn import EsnConfig, ReadoutConfig, ReservoirConfig
from arm_rc_ctrl.rc.recipe import DatasetSource, ModelRecipe, RclibIdentity, TrainingValidation, solver_alpha
from arm_rc_ctrl.rc.teacher_forcing import InputTransform
from arm_rc_ctrl.rc.train import InputTransformSpec, ModelConfig
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import endpoint_positions, load_scenario

REPO_ROOT = repository_root()
SCENARIO_FILE = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
SCENARIO = load_scenario(SCENARIO_FILE)
DERIVATIVES = DerivativeConfig(method="central")
SOURCE = DatasetSource("processed-20260830-555555555555", "ab" * 32, "data/records/processed/x.toml")
N = 101
DT = 0.01
BASE = ModelConfig(
    name="fixture",
    esn=EsnConfig(
        reservoir=ReservoirConfig(
            n_neurons=40, spectral_radius=0.85, sparsity=0.9, leak_rate=0.4, input_scaling=0.4, seed=23
        ),
        readout=ReadoutConfig(alpha=0.5),
    ),
    input_transform=InputTransformSpec(policy="fixed_scale", q_scale=0.3, dq_scale=4.0),
)
DEFAULT_ARM = ArmSpec("absolute", "R", 16)
PREPROCESSING = Preprocessing(
    resample_period_s=DT, smoothing="none", smoothing_params={}, derivative_method="central-difference"
)
POINT = RecoveryTrialPoint(
    esn=TrialPoint(
        n_neurons=40,
        spectral_radius=0.85,
        sparsity=0.9,
        leak_rate=0.4,
        input_scaling=0.4,
        seed=23,
        alpha=0.02,
        velocity_cutoff_hz=20.0,
        acceleration_cutoff_hz=10.0,
    ),
    warmup_s=0.25,
    augmentation=None,
)
ENTRY = PanelEntry(
    label="feasible-best",
    source_trial=17,
    role="rank 1",
    selection="feasible rank 1 of 134",
    warmup_s=0.25,
    point=POINT,
    estimator=EstimatorSpec(velocity_cutoff_hz=20.0, acceleration_cutoff_hz=10.0, max_dt_ratio=3.0),
    base_alpha=0.02,
    feasible=True,
    objective=0.67,
    first_failure=None,
    reason_head=None,
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


def test_arm_specs_enumerate_the_approved_panel() -> None:
    """Sixteen absolute and ten residual arms per entry (26, of which 20 behavioral); invalid combinations fail."""
    arms = panel_arms()
    assert len(arms) == 26
    assert sum(1 for a in arms if a.formulation == "absolute") == 1 + 3 * 5
    assert sum(1 for a in arms if a.formulation == "residual") == 1 + 3 * 3
    assert sum(1 for a in arms if a.behavioral) == 20  # 120 behavioral configurations over six entries
    assert sum(1 for a in arms if not a.behavioral) == 6  # 36 S-effective reference fits over six entries
    labels = [a.label for a in arms]
    assert labels[0] == "absolute/S"
    assert "absolute/R/K17" in labels
    assert "absolute/A-contractive/K65" in labels
    assert "residual/S-effective/K33" in labels
    assert len(set(labels)) == len(labels)
    assert ArmSpec("absolute", "R", 16).count == 17
    assert ArmSpec("absolute", "S").count == 1
    assert ArmSpec("absolute", "S-effective", 64).episodes_trained == 1
    assert ArmSpec("absolute", "A-non-decaying", 64).episodes_trained == 65
    assert ArmSpec("residual", "R-scaled", 32).regularization_rule == "count_scaled"
    assert ArmSpec("residual", "S-effective", 16).regularization_rule == "count_divided"
    assert ArmSpec("residual", "R", 16).target == "increment_q"
    with pytest.raises(ValueError, match="residual arms"):
        ArmSpec("residual", "A-contractive", 16)
    with pytest.raises(ValueError, match="no episode count"):
        ArmSpec("absolute", "S", 16)
    with pytest.raises(ValueError, match="additional_repeats"):
        ArmSpec("absolute", "R", 8)
    with pytest.raises(ValueError, match="additional_repeats"):
        ArmSpec("absolute", "R")
    with pytest.raises(ValueError, match="formulation"):
        ArmSpec("hybrid", "R", 16)
    assert set(RESIDUAL_ARMS) < set(ARMS)


def test_training_specs_readouts_and_loss_rows_per_arm() -> None:
    """Each arm yields the construction, solver parameter, and row count the plan prescribes."""
    alpha0 = 0.02
    for arm in panel_arms():
        spec = training_spec_for_arm(arm, warmup_s=0.25, base_alpha=alpha0)
        readout = readout_for_arm(BASE.esn.readout, arm, base_alpha=alpha0)
        assert spec.washout == "warmup_hold"
        assert spec.warmup_s == 0.25
        assert spec.target == arm.target
        assert spec.base_alpha == alpha0
        assert spec.regularization_rule == arm.regularization_rule
        assert spec.regularization_count == arm.count
        assert readout.alpha == solver_alpha(alpha0, arm.regularization_rule, arm.count)
        assert readout.solver == BASE.esn.readout.solver
        if arm.arm in ("R", "R-scaled"):
            assert spec.additional_repeats == arm.count - 1
            assert spec.augmentation is None
        elif arm.arm.startswith("A-"):
            assert spec.additional_repeats is None
            assert spec.augmentation is not None
            assert spec.augmentation.n_synthetic == arm.count - 1
            assert spec.augmentation.attempt_budget == 4 * (arm.count - 1)
            assert (spec.augmentation.sigma_rad, spec.augmentation.phi, spec.augmentation.gamma) == (0.05, 0.99, 1.0)
            assert spec.augmentation.seed_bank == 1
        else:
            assert spec.additional_repeats is None
            assert spec.augmentation is None
    assert expected_loss_rows(ArmSpec("absolute", "R", 16)) == 6800
    assert expected_loss_rows(ArmSpec("absolute", "R-scaled", 32)) == 13200
    assert expected_loss_rows(ArmSpec("absolute", "A-contractive", 64)) == 26000
    assert expected_loss_rows(ArmSpec("absolute", "S-effective", 64)) == ROWS_PER_EPISODE
    assert expected_loss_rows(ArmSpec("absolute", "S")) == ROWS_PER_EPISODE
    with pytest.raises(ValueError, match="positive"):
        expected_loss_rows(ArmSpec("absolute", "S"), rows_per_episode=0)
    assert readout_for_arm(BASE.esn.readout, ArmSpec("absolute", "R-scaled", 64), base_alpha=0.5).alpha == 32.5
    assert (
        readout_for_arm(BASE.esn.readout, ArmSpec("absolute", "S-effective", 64), base_alpha=0.001).alpha == 0.001 / 65
    )
    with pytest.raises(ValueError, match="family"):
        AugmentationAnchor().spec("both", 16)
    assert AugmentationAnchor() == AUGMENTATION_ANCHOR


def test_recipes_for_arms_fit_from_a_panel_entry() -> None:
    """S, R, R-scaled, and S-effective recipes derive from one entry; R and its copies share the entry's reservoir."""
    samples = _samples()
    normalization = fit_normalization(
        samples.arrays(), ("q", "dq"), fitted_on=(SOURCE.artifact_id,), training_rows=np.ones(N, dtype=np.bool_)
    )

    def fit(arm: ArmSpec) -> ModelRecipe:
        recipe, _model = recipe_for_arm(
            ENTRY,
            arm,
            base=BASE,
            source=SOURCE,
            samples={SOURCE.artifact_id: samples},
            dof=2,
            task_code_dim=0,
            preprocessing=PREPROCESSING,
            normalization=normalization,
            scenario=SCENARIO,
            scenario_file=SCENARIO_FILE,
            root=REPO_ROOT,
        )
        return recipe

    single = fit(ArmSpec("absolute", "S"))
    repeated = fit(ArmSpec("absolute", "R", 16))
    scaled = fit(ArmSpec("absolute", "R-scaled", 16))
    effective = fit(ArmSpec("residual", "S-effective", 16))
    assert single.name == "feasible-best/absolute/S"
    assert repeated.name == "feasible-best/absolute/R/K17"
    assert single.esn.reservoir == repeated.esn.reservoir == scaled.esn.reservoir == effective.esn.reservoir
    assert single.esn.reservoir.seed == POINT.esn.seed
    assert single.solver_alpha == 0.02
    assert repeated.solver_alpha == 0.02
    assert scaled.solver_alpha == 0.02 * 17
    assert effective.solver_alpha == 0.02 / 17
    assert effective.formulation == "residual"
    assert effective.fit.loss_rows == N - 1
    assert repeated.fit.loss_rows == 17 * (N - 1)
    assert repeated.schema_version == 2
    assert repeated.validation == TrainingValidation.from_scenario(SCENARIO, SCENARIO_FILE, root=REPO_ROOT)
    assert repeated.transform.policy == "fixed_scale"
    assert repeated.transform.fixed_scales == {"q": 0.3, "dq": 4.0}
    assert single.training.warmup_s == ENTRY.warmup_s
    _, report = repeated.refit({SOURCE.artifact_id: samples}, scenario=SCENARIO)
    assert report == repeated.fit


def test_fit_identity_binds_every_input_including_the_execution_environment() -> None:
    """The identity changes with the arm, count, formulation, alpha, transform, validation, and environment."""
    validation = TrainingValidation.from_scenario(SCENARIO, SCENARIO_FILE, root=REPO_ROOT)
    samples = _samples()
    normalization = fit_normalization(
        samples.arrays(), ("q", "dq"), fitted_on=(SOURCE.artifact_id,), training_rows=np.ones(N, dtype=np.bool_)
    )
    transform = InputTransform.derive("fixed_scale", normalization, fixed_scales={"q": 0.3, "dq": 4.0})
    rclib = RclibIdentity.current()

    def identity(
        *,
        arm: ArmSpec = DEFAULT_ARM,
        warmup_s: float = 0.25,
        solver_alpha: float = 0.02,
        dataset: DatasetSource = SOURCE,
        bound: TrainingValidation = validation,
        execution_identity: str = "a" * 64,
    ) -> str:
        return fit_identity(
            panel_label="feasible-best",
            source_trial=17,
            arm=arm,
            warmup_s=warmup_s,
            base_alpha=0.02,
            solver_alpha=solver_alpha,
            dataset=dataset,
            transform=transform,
            validation=bound,
            rclib_commit=rclib.commit,
            execution_identity=execution_identity,
        )

    base = identity()
    assert base == identity()
    assert len(base) == 64
    assert base != identity(arm=ArmSpec("absolute", "R", 32))
    assert base != identity(arm=ArmSpec("residual", "R", 16))
    assert base != identity(arm=ArmSpec("absolute", "R-scaled", 16), solver_alpha=0.02 * 17)
    assert base != identity(execution_identity="b" * 64)
    assert base != identity(bound=replace(validation, velocity_limit=(1.0, 1.0)))
    assert base != identity(dataset=replace(SOURCE, payload_sha256="cd" * 32))
    assert base != identity(warmup_s=1.0)

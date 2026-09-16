# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-005: the approved manual arms map the locked bank onto schema 3 recipes with matched counts and weights.

``S_i`` trains on one demonstration, ``M10`` on all ten, and ``R10_i`` on ten
exact copies of one; the last two match ``M10`` on episode count and total loss
weight, not on raw sample count. Every arm copies its input transform from the
historical scripted dataset and never from a take (clarification I8).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import pytest

from arm_rc_ctrl.data.derivatives import DerivativeConfig, differentiate
from arm_rc_ctrl.data.manual import ManualDatasetRecord
from arm_rc_ctrl.data.records import Normalization, Preprocessing, ProcessedDatasetRecord, load_record
from arm_rc_ctrl.data.samples import SampleSet
from arm_rc_ctrl.experiments.manual_bank import load_bank_manifest
from arm_rc_ctrl.experiments.manual_recipes import (
    ARMS,
    ASSIGNMENTS,
    COPIES,
    MANUAL_ANCHOR,
    ROWS_REFERENCE,
    TRANSFORM_SOURCE,
    ManualAnchor,
    ManualArmSpec,
    arm_accounting,
    arm_sources,
    bank_sources,
    esn_for_arm,
    fit_identity,
    manual_arms,
    readout_for_arm,
    recipe_for_arm,
    training_spec_for_arm,
)
from arm_rc_ctrl.rc.esn import EsnConfig, ReadoutConfig, ReservoirConfig
from arm_rc_ctrl.rc.recipe import (
    EQUAL_EPISODE_WEIGHTING,
    WEIGHTED_SCHEMA_VERSION,
    DatasetSource,
    RclibIdentity,
    TrainingValidation,
)
from arm_rc_ctrl.rc.teacher_forcing import InputTransform
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import endpoint_positions, load_scenario

if TYPE_CHECKING:
    from arm_rc_ctrl.rc.esn import EsnModel
    from arm_rc_ctrl.rc.recipe import ModelRecipe


def _statistics(record: ProcessedDatasetRecord) -> Normalization:
    """The record's recorded statistics; the frozen scripted dataset always carries them."""
    if record.normalization is None:
        msg = f"{record.artifact.artifact_id} records no normalization statistics"
        raise ValueError(msg)
    return record.normalization


REPO_ROOT = repository_root()
BANK_FILE = REPO_ROOT / "docs" / "experiments" / "task_1a_manual_demonstration" / "bank" / "bank_v1.json"
BANK = load_bank_manifest(BANK_FILE)
SOURCES = bank_sources(BANK, root=REPO_ROOT)
SCENARIO_FILE = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
SCENARIO = load_scenario(SCENARIO_FILE)
VALIDATION = TrainingValidation.from_scenario(SCENARIO, SCENARIO_FILE, root=REPO_ROOT)
DERIVATIVES = DerivativeConfig(method="central")
HISTORICAL = load_record(REPO_ROOT / TRANSFORM_SOURCE.record, ProcessedDatasetRecord)
HISTORICAL_NORMALIZATION = _statistics(HISTORICAL)  # the frozen physical statistics live in the scripted record
TRANSFORM = InputTransform.derive("fixed_scale", HISTORICAL_NORMALIZATION, fixed_scales={"q": 0.3, "dq": 12.0})
DT = 0.01
WARMUP_S = 0.25
BASE_ALPHA = 0.02
BASE_ESN = EsnConfig(
    reservoir=ReservoirConfig(
        n_neurons=20, spectral_radius=0.85, sparsity=0.9, leak_rate=0.4, input_scaling=0.4, seed=23
    ),
    readout=ReadoutConfig(alpha=0.5),
)
PREPROCESSING = Preprocessing(
    resample_period_s=DT, smoothing="none", smoothing_params={}, derivative_method="central-difference"
)
DEFAULT_ARM = ManualArmSpec("S", "D01")


def _samples(n: int) -> SampleSet:
    """A complete recording of ``n`` samples: a short hold, a smooth reach, and a final dwell."""
    t = np.arange(n, dtype=np.float64) * DT
    start = np.array(SCENARIO.task.initial_q)
    hold_end, move_end = 0.1, float(t[-1]) - 0.1
    ramp = np.clip((t - hold_end) / (move_end - hold_end), 0.0, 1.0)
    blend = ramp * ramp * (3.0 - 2.0 * ramp)
    q = start[None, :] + blend[:, None] * (np.array([0.8, 0.4]) - start)[None, :]
    dq, ddq = differentiate(q, DT, DERIVATIVES)
    tip = endpoint_positions(SCENARIO, q)
    dtip, ddtip = differentiate(tip, DT, DERIVATIVES)
    phase = np.where(t < hold_end, 0, np.where(t < move_end, 1, 2)).astype(np.int64)
    return SampleSet(t, q, dq, ddq, tip, dtip, ddtip, np.zeros((n, 0)), phase)


SAMPLES = {SOURCES[name].artifact_id: _samples(41 + 4 * index) for index, name in enumerate(ASSIGNMENTS)}
LOSS_ROWS = {name: SAMPLES[SOURCES[name].artifact_id].n_samples - 1 for name in ASSIGNMENTS}


def _build(
    arm: ManualArmSpec, *, transform: InputTransform = TRANSFORM, preprocessing: Preprocessing = PREPROCESSING
) -> tuple[ModelRecipe, EsnModel]:
    return recipe_for_arm(
        arm,
        esn=BASE_ESN,
        sources=SOURCES,
        samples=SAMPLES,
        dof=2,
        task_code_dim=0,
        preprocessing=preprocessing,
        transform=transform,
        validation=VALIDATION,
        warmup_s=WARMUP_S,
        base_alpha=BASE_ALPHA,
        scenario=SCENARIO,
    )


def test_manual_arms_enumerate_the_approved_scope() -> None:
    """Ten singletons, the all-ten model, and ten duplication controls; invalid combinations are refused."""
    arms = manual_arms()
    labels = [arm.label for arm in arms]
    assert len(arms) == 2 * len(ASSIGNMENTS) + 1
    assert labels[0] == "S/D01"
    assert labels[9] == "S/D10"
    assert labels[10] == "M10"
    assert labels[-1] == "R10/D10"
    assert len(set(labels)) == len(labels)
    assert set(ARMS) == {"S", "M10", "R10"}

    single = ManualArmSpec("S", "D01")
    assert (single.count, single.unique_sources, single.copies) == (1, 1, 0)
    assert single.source_counts == (1,)
    everything = ManualArmSpec("M10")
    assert (everything.count, everything.unique_sources, everything.copies) == (10, 10, 0)
    assert everything.source_counts == (1,) * 10
    assert everything.assignments == ASSIGNMENTS
    control = ManualArmSpec("R10", "D05")
    assert (control.count, control.unique_sources, control.copies) == (COPIES, 1, COPIES - 1)
    assert control.source_counts == (COPIES,)

    with pytest.raises(ValueError, match="carries no assignment"):
        ManualArmSpec("M10", "D01")
    with pytest.raises(ValueError, match="needs an assignment"):
        ManualArmSpec("S")
    with pytest.raises(ValueError, match="needs an assignment"):
        ManualArmSpec("R10", "D11")
    with pytest.raises(ValueError, match="arm must be one of"):
        ManualArmSpec("C10", "D01")  # the contractive arm is the augmentation task's


def test_training_specs_and_readouts_follow_the_frozen_anchor() -> None:
    """Every arm gets the anchor's weighting and warm-up, ``K alpha_0``, and the explicit-bias readout."""
    for arm in manual_arms():
        spec = training_spec_for_arm(arm, warmup_s=WARMUP_S, base_alpha=BASE_ALPHA)
        assert spec.washout == "warmup_hold"
        assert spec.warmup_s == WARMUP_S
        assert spec.target == "next_q"
        assert spec.episode_weighting == EQUAL_EPISODE_WEIGHTING
        assert spec.weight_reference_rows == ROWS_REFERENCE
        assert spec.source_counts == arm.source_counts
        assert spec.episode_count == arm.count
        assert spec.base_alpha == BASE_ALPHA
        assert spec.regularization_rule == "count_scaled"
        readout = readout_for_arm(BASE_ESN.readout, arm, base_alpha=BASE_ALPHA)
        assert readout.alpha == BASE_ALPHA * arm.count
        assert readout.include_bias is False
        assert readout.explicit_bias is True
        assert readout.solver == BASE_ESN.readout.solver
        assert esn_for_arm(BASE_ESN, arm, base_alpha=BASE_ALPHA).reservoir == BASE_ESN.reservoir
    assert readout_for_arm(BASE_ESN.readout, ManualArmSpec("S", "D01"), base_alpha=BASE_ALPHA).alpha == BASE_ALPHA
    assert readout_for_arm(BASE_ESN.readout, ManualArmSpec("M10"), base_alpha=BASE_ALPHA).alpha == 10 * BASE_ALPHA
    assert MANUAL_ANCHOR.regularization_lambda(BASE_ALPHA) == BASE_ALPHA / ROWS_REFERENCE
    assert ManualAnchor() == MANUAL_ANCHOR
    with pytest.raises(ValueError, match="approved construction"):
        ManualAnchor(episode_weighting="per_row")
    with pytest.raises(ValueError, match="must be positive"):
        ManualAnchor(weight_reference_rows=0)


def test_accounting_records_rows_weights_and_ridge_scales() -> None:
    """Unequal recordings get weights ``400 / L_i``; copies get one weight; the matched arms share total weight."""
    single = arm_accounting(ManualArmSpec("S", "D02"), base_alpha=BASE_ALPHA, loss_rows=LOSS_ROWS)
    assert single.loss_rows == (LOSS_ROWS["D02"],)
    assert single.row_weights == (ROWS_REFERENCE / LOSS_ROWS["D02"],)
    assert single.solver_alpha == BASE_ALPHA
    assert single.regularization_lambda == BASE_ALPHA / ROWS_REFERENCE
    assert single.total_loss_weight == pytest.approx(ROWS_REFERENCE)

    bank = arm_accounting(ManualArmSpec("M10"), base_alpha=BASE_ALPHA, loss_rows=LOSS_ROWS)
    assert bank.loss_rows == tuple(LOSS_ROWS[name] for name in ASSIGNMENTS)
    assert bank.row_weights == tuple(ROWS_REFERENCE / LOSS_ROWS[name] for name in ASSIGNMENTS)
    assert len(set(bank.row_weights)) == len(ASSIGNMENTS)  # unequal lengths give unequal per-row weights
    assert bank.total_loss_weight == pytest.approx(ROWS_REFERENCE * 10)
    assert (bank.unique_sources, bank.copies) == (10, 0)

    control = arm_accounting(ManualArmSpec("R10", "D02"), base_alpha=BASE_ALPHA, loss_rows=LOSS_ROWS)
    assert control.loss_rows == (LOSS_ROWS["D02"],) * COPIES
    assert len(set(control.row_weights)) == 1  # equal lengths give equal weights
    assert (control.unique_sources, control.copies) == (1, COPIES - 1)
    assert control.solver_alpha == bank.solver_alpha == 10 * BASE_ALPHA
    # The controls match M10 on episode count and total loss weight, not on raw sample count.
    assert control.episodes == bank.episodes == 10
    assert control.total_loss_weight == pytest.approx(bank.total_loss_weight)
    assert sum(control.loss_rows) != sum(bank.loss_rows)
    with pytest.raises(ValueError, match="needs the loss rows"):
        arm_accounting(ManualArmSpec("S", "D01"), base_alpha=BASE_ALPHA, loss_rows={})


def test_bank_sources_bind_the_locked_takes_by_digest() -> None:
    """Each assignment resolves to its committed processed record, digest-bound and repository-relative."""
    assert sorted(SOURCES) == list(ASSIGNMENTS)
    assigned = {take.assignment: take for take in BANK.takes if take.assignment is not None}
    for name, source in SOURCES.items():
        record = load_record(REPO_ROOT / source.record, ManualDatasetRecord)
        assert source.artifact_id == assigned[name].processed_artifact_id == record.artifact.artifact_id
        assert source.payload_sha256 == record.artifact.payload.sha256
        assert source.record == f"data/records/processed/{source.artifact_id}.toml"
        assert not source.record.startswith("/")  # records never carry machine paths
        assert record.normalization is None  # a manual take records no statistics by design (I8)
    assert arm_sources(ManualArmSpec("M10"), SOURCES) == tuple(SOURCES[name] for name in ASSIGNMENTS)
    assert arm_sources(ManualArmSpec("R10", "D03"), SOURCES) == (SOURCES["D03"],)
    with pytest.raises(ValueError, match="needs datasets for"):
        arm_sources(ManualArmSpec("S", "D04"), {})


def test_recipe_for_arm_builds_the_weighted_schema_three_recipe() -> None:
    """M10 and R10 are schema 3 recipes at ``10 alpha_0``; the control's copies are multiplicities, not payloads."""
    recipe, model = _build(ManualArmSpec("M10"))
    assert recipe.schema_version == WEIGHTED_SCHEMA_VERSION
    assert recipe.name == "M10"
    assert recipe.solver_alpha == 10 * BASE_ALPHA
    assert recipe.transform_source == TRANSFORM_SOURCE
    assert [d.artifact_id for d in recipe.datasets] == [SOURCES[name].artifact_id for name in ASSIGNMENTS]
    assert recipe.training.source_counts == (1,) * 10
    assert recipe.fit.episode_loss_rows == tuple(LOSS_ROWS[name] for name in ASSIGNMENTS)
    assert recipe.fit.episode_weights == tuple(ROWS_REFERENCE / LOSS_ROWS[name] for name in ASSIGNMENTS)
    assert model.explicit_bias
    assert model.readout_weights().shape == (BASE_ESN.reservoir.n_neurons + 1, 2)

    control, _model = _build(ManualArmSpec("R10", "D01"))
    assert control.name == "R10/D01"
    assert control.training.source_counts == (COPIES,)
    assert len(control.datasets) == 1
    assert control.fit.episodes[1] == f"{SOURCES['D01'].artifact_id}#copy-001"
    assert control.fit.loss_rows == COPIES * LOSS_ROWS["D01"]
    assert control.solver_alpha == recipe.solver_alpha
    _refit, report = control.refit(SAMPLES, scenario=SCENARIO)
    assert report == control.fit

    single, _single_model = _build(ManualArmSpec("S", "D01"))
    assert single.solver_alpha == BASE_ALPHA
    assert single.fit.episodes == (SOURCES["D01"].artifact_id,)
    assert single.esn.reservoir == recipe.esn.reservoir  # one reservoir across the arms of a configuration


def test_a_transform_computed_from_a_take_is_refused() -> None:
    """Source isolation: the frozen transform is copied from the scripted dataset, never from a demonstration."""
    from_take = dataclasses.replace(TRANSFORM, derived_from=(SOURCES["D01"].artifact_id,))
    with pytest.raises(ValueError, match="never compute statistics from a take"):
        _build(ManualArmSpec("S", "D01"), transform=from_take)


def test_a_manual_take_is_accepted_as_training_data_without_normalization() -> None:
    """The recipe binds the committed manual record although it records no statistics of its own (I8)."""
    record = load_record(REPO_ROOT / SOURCES["D01"].record, ManualDatasetRecord)
    recipe, _model = _build(ManualArmSpec("S", "D01"), preprocessing=record.preprocessing)
    recipe.check_dataset_record(SOURCES["D01"], record)
    assert recipe.check_transform_record(HISTORICAL) == HISTORICAL_NORMALIZATION
    recipe.check_transform_source({TRANSFORM_SOURCE.artifact_id: HISTORICAL_NORMALIZATION})


def test_fit_identity_binds_every_input_including_the_environment() -> None:
    """The identity changes with the arm, the demonstration, the ridge scale, the transform, and the environment."""
    rclib = RclibIdentity.current()

    def identity(
        *,
        arm: ManualArmSpec = DEFAULT_ARM,
        warmup_s: float = WARMUP_S,
        base_alpha: float = BASE_ALPHA,
        esn: EsnConfig = BASE_ESN,
        datasets: tuple[DatasetSource, ...] = (),
        transform: InputTransform = TRANSFORM,
        execution_identity: str = "a" * 64,
    ) -> str:
        return fit_identity(
            configuration="trial-17",
            arm=arm,
            warmup_s=warmup_s,
            base_alpha=base_alpha,
            esn=esn,
            datasets=datasets or (SOURCES["D01"],),
            transform=transform,
            validation=VALIDATION,
            rclib_commit=rclib.commit,
            execution_identity=execution_identity,
        )

    base = identity()
    assert base == identity()
    assert len(base) == 64
    assert base != identity(arm=ManualArmSpec("R10", "D01"))
    assert base != identity(arm=ManualArmSpec("S", "D02"))
    assert base != identity(datasets=(SOURCES["D02"],))
    assert base != identity(base_alpha=0.03)
    assert base != identity(warmup_s=1.0)
    assert base != identity(execution_identity="b" * 64)
    assert base != identity(esn=esn_for_arm(BASE_ESN, ManualArmSpec("S", "D01"), base_alpha=BASE_ALPHA))
    assert base != identity(transform=dataclasses.replace(TRANSFORM, derived_from=("processed-20260830-999999999999",)))

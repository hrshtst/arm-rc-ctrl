# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-005: schema 3 recipes weight episodes equally, carry source multiplicities, and copy a frozen transform.

The manual-demonstration arms train on complete recordings of unequal length
whose recorded pre-roll stays inside the loss, express exact copies as
per-source multiplicities instead of duplicated payloads, and take their input
transform from the historical scripted dataset (clarification I8) rather than
from any manual take.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import pytest

from arm_rc_ctrl.data.derivatives import DerivativeConfig, differentiate
from arm_rc_ctrl.data.manual_scenario import load_manual_scenario
from arm_rc_ctrl.data.records import Normalization, Preprocessing, ProcessedDatasetRecord, load_record
from arm_rc_ctrl.data.samples import SampleSet
from arm_rc_ctrl.rc.augment import MANUAL_N_SYNTHETIC
from arm_rc_ctrl.rc.esn import EsnConfig, ReadoutConfig, ReservoirConfig
from arm_rc_ctrl.rc.recipe import (
    EQUAL_EPISODE_WEIGHTING,
    WEIGHTED_SCHEMA_VERSION,
    AugmentationTrainingSpec,
    ContractiveTrainingSpec,
    DatasetSource,
    ModelRecipe,
    RecipeMismatchError,
    TrainingSpec,
    TrainingValidation,
    create_recipe,
    expected_episode_labels,
    load_recipe,
    solver_alpha,
    write_recipe,
)
from arm_rc_ctrl.rc.teacher_forcing import InputTransform
from arm_rc_ctrl.rc.training import harvest_episode
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import endpoint_positions, load_scenario

if TYPE_CHECKING:
    from pathlib import Path

    from arm_rc_ctrl.rc.augment import TaskGeometry
    from arm_rc_ctrl.rc.esn import EsnModel


def _statistics(record: ProcessedDatasetRecord) -> Normalization:
    """The record's recorded statistics; the frozen scripted dataset always carries them."""
    if record.normalization is None:
        msg = f"{record.artifact.artifact_id} records no normalization statistics"
        raise ValueError(msg)
    return record.normalization


REPO_ROOT = repository_root()
SCENARIO_FILE = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
SCENARIO = load_scenario(SCENARIO_FILE)
DERIVATIVES = DerivativeConfig(method="central")
HISTORICAL_RELATIVE = "data/records/processed/processed-20260830-feaf73e6663c.toml"
HISTORICAL = load_record(REPO_ROOT / HISTORICAL_RELATIVE, ProcessedDatasetRecord)
HISTORICAL_NORMALIZATION = _statistics(HISTORICAL)  # the scripted dataset carries the frozen statistics (I8)
FROZEN_SOURCE = DatasetSource(HISTORICAL.artifact.artifact_id, HISTORICAL.artifact.payload.sha256, HISTORICAL_RELATIVE)
"""The historical scripted dataset the input transform is copied from; never a training source of these arms."""
TRANSFORM = InputTransform.derive("fixed_scale", HISTORICAL_NORMALIZATION, fixed_scales={"q": 0.3, "dq": 12.0})
FIRST = DatasetSource("processed-20260916-111111111111", "ab" * 32, "data/records/processed/first.toml")
SECOND = DatasetSource("processed-20260916-222222222222", "cd" * 32, "data/records/processed/second.toml")
DT = 0.01
WARMUP_S = 0.25
WARMUP_ROWS = 25
ROWS_REFERENCE = 400
BASE_ALPHA = 0.01
RESERVOIR = ReservoirConfig(n_neurons=40, spectral_radius=0.85, sparsity=0.9, leak_rate=0.4, input_scaling=0.4, seed=23)
PREPROCESSING = Preprocessing(
    resample_period_s=DT, smoothing="none", smoothing_params={}, derivative_method="central-difference"
)


def _samples(n: int, goal: tuple[float, float]) -> SampleSet:
    """A complete recording: a short initial hold, a smooth reach, and a final dwell (manual phase codes)."""
    t = np.arange(n, dtype=np.float64) * DT
    start = np.array(SCENARIO.task.initial_q)
    hold_end, move_end = 0.1, float(t[-1]) - 0.1
    ramp = np.clip((t - hold_end) / (move_end - hold_end), 0.0, 1.0)
    blend = ramp * ramp * (3.0 - 2.0 * ramp)
    q = start[None, :] + blend[:, None] * (np.array(goal) - start)[None, :]
    dq, ddq = differentiate(q, DT, DERIVATIVES)
    tip = endpoint_positions(SCENARIO, q)
    dtip, ddtip = differentiate(tip, DT, DERIVATIVES)
    phase = np.where(t < hold_end, 0, np.where(t < move_end, 1, 2)).astype(np.int64)
    return SampleSet(t, q, dq, ddq, tip, dtip, ddtip, np.zeros((n, 0)), phase)


PARENT = DatasetSource("processed-20260916-333333333333", "ef" * 32, "data/records/processed/parent.toml")
"""The contractive arm's parent: a 1.50 s recording with room for the frozen 0.5 s ramp before the taper window."""
FIRST_SAMPLES = _samples(61, (0.8, 0.4))
SECOND_SAMPLES = _samples(41, (0.7, 0.5))
PARENT_SAMPLES = _samples(151, (0.8, 0.4))
SAMPLES = {
    FIRST.artifact_id: FIRST_SAMPLES,
    SECOND.artifact_id: SECOND_SAMPLES,
    PARENT.artifact_id: PARENT_SAMPLES,
}
FIRST_ROWS = FIRST_SAMPLES.n_samples - 1
SECOND_ROWS = SECOND_SAMPLES.n_samples - 1
PARENT_ROWS = PARENT_SAMPLES.n_samples - 1
DWELL_START_S = float(PARENT_SAMPLES.t[-1]) - 0.1
CONTRACTIVE = ContractiveTrainingSpec(seed_bank=5, dwell_start_s=DWELL_START_S)
MANUAL_SCENARIO_FILE = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_manual_fixture.toml"
MANUAL_SCENARIO = load_manual_scenario(MANUAL_SCENARIO_FILE)
AUGMENTATION = AugmentationTrainingSpec(
    family="contractive", n_synthetic=16, sigma_rad=0.025, phi=0.99, gamma=1.0, seed_bank=1, attempt_budget=64
)


def _spec(
    counts: tuple[int, ...],
    *,
    weighting: str | None = EQUAL_EPISODE_WEIGHTING,
    reference: int | None = ROWS_REFERENCE,
    contractive: ContractiveTrainingSpec | None = None,
) -> TrainingSpec:
    return TrainingSpec(
        washout="warmup_hold",
        warmup_s=WARMUP_S,
        episode_weighting=weighting,
        weight_reference_rows=reference,
        source_counts=counts,
        base_alpha=BASE_ALPHA,
        regularization_rule="count_scaled",
        contractive=contractive,
    )


def _esn(count: int, *, explicit_bias: bool | None = True, include_bias: bool = False) -> EsnConfig:
    readout = ReadoutConfig(
        alpha=solver_alpha(BASE_ALPHA, "count_scaled", count), include_bias=include_bias, explicit_bias=explicit_bias
    )
    return EsnConfig(reservoir=RESERVOIR, readout=readout)


def _validation() -> TrainingValidation:
    return TrainingValidation.from_scenario(SCENARIO, SCENARIO_FILE, root=REPO_ROOT)


def _build(
    sources: list[DatasetSource],
    counts: tuple[int, ...],
    *,
    contractive: ContractiveTrainingSpec | None = None,
    scenario: TaskGeometry | None = SCENARIO,
) -> tuple[ModelRecipe, EsnModel]:
    spec = _spec(counts, contractive=contractive)
    return create_recipe(
        "manual-weighted-test",
        _esn(spec.episode_count),
        sources=sources,
        samples=SAMPLES,
        dof=2,
        task_code_dim=0,
        preprocessing=PREPROCESSING,
        transform=TRANSFORM,
        training=spec,
        scenario=scenario,
        validation=_validation(),
        transform_source=FROZEN_SOURCE,
    )


def test_weighted_recipe_binds_its_construction_and_refits_exactly(tmp_path: Path) -> None:
    """Two takes of unequal length are weighted 400 / L_i at K alpha_0; the written recipe reloads and refits."""
    recipe, _model = _build([FIRST, SECOND], (1, 1))
    assert recipe.schema_version == WEIGHTED_SCHEMA_VERSION
    assert recipe.training.episode_count == 2
    assert recipe.solver_alpha == 2 * BASE_ALPHA
    assert recipe.fit.episodes == (FIRST.artifact_id, SECOND.artifact_id)
    assert recipe.fit.episode_loss_rows == (FIRST_ROWS, SECOND_ROWS)
    assert recipe.fit.episode_weights == (ROWS_REFERENCE / FIRST_ROWS, ROWS_REFERENCE / SECOND_ROWS)
    assert recipe.fit.loss_rows == FIRST_ROWS + SECOND_ROWS
    assert recipe.fit.washout_rows == 2 * WARMUP_ROWS

    file = tmp_path / "weighted.toml"
    write_recipe(file, recipe)
    text = file.read_text(encoding="utf-8")
    assert "schema_version = 3" in text
    assert 'episode_weighting = "equal_episode"' in text
    assert "weight_reference_rows = 400" in text
    assert "source_counts = [" in text
    assert "explicit_bias = true" in text
    assert "include_bias = false" in text
    assert "[transform_source]" in text
    loaded = load_recipe(file)
    assert loaded == recipe
    _refit, report = loaded.refit(SAMPLES, scenario=SCENARIO)
    assert report == recipe.fit


def test_literal_copies_share_the_parents_arrays_states_and_weight() -> None:
    """A source at count ten yields ten episodes whose harvested states are bitwise the parent's."""
    recipe, model = _build([FIRST], (10,))
    labels = expected_episode_labels(recipe.training, (FIRST.artifact_id,))
    assert labels[0] == FIRST.artifact_id
    assert labels[1] == f"{FIRST.artifact_id}#copy-001"
    assert labels[-1] == f"{FIRST.artifact_id}#copy-009"
    assert len(labels) == 10
    assert recipe.fit.episodes == labels
    assert recipe.solver_alpha == 10 * BASE_ALPHA
    assert recipe.fit.loss_rows == 10 * FIRST_ROWS
    assert recipe.fit.episode_weights == (ROWS_REFERENCE / FIRST_ROWS,) * 10

    episodes = recipe.episodes(SAMPLES)
    original = episodes[0]
    reference = harvest_episode(model, original)
    for copy in episodes[1:]:
        assert np.array_equal(copy.inputs, original.inputs)
        assert np.array_equal(copy.targets, original.targets)
        assert np.array_equal(copy.loss_rows, original.loss_rows)
        assert np.array_equal(copy.t, original.t)
        assert np.array_equal(harvest_episode(model, copy).states, reference.states)
    with pytest.raises(ValueError, match="one multiplicity per dataset"):
        expected_episode_labels(recipe.training, (FIRST.artifact_id, SECOND.artifact_id))


def test_no_target_pair_crosses_an_episode_boundary() -> None:
    """Every row pairs sample k with k + 1 of its own take; the recorded hold trains, the warm-up does not."""
    recipe, _model = _build([FIRST, SECOND], (1, 1))
    episodes = recipe.episodes(SAMPLES)
    assert [e.source for e in episodes] == [FIRST.artifact_id, SECOND.artifact_id]
    encoder = recipe.encoder()
    for episode, samples in zip(episodes, (FIRST_SAMPLES, SECOND_SAMPLES), strict=True):
        assert episode.washout_len == WARMUP_ROWS
        assert episode.n_rows == WARMUP_ROWS + samples.n_samples - 1
        assert int(episode.loss_rows.sum()) == samples.n_samples - 1  # every recorded sample trains (I4, I10)
        assert np.array_equal(episode.targets[WARMUP_ROWS:], samples.q[1:])
        assert np.array_equal(episode.targets[:WARMUP_ROWS], np.tile(samples.q[0], (WARMUP_ROWS, 1)))
        assert np.array_equal(episode.inputs[WARMUP_ROWS], encoder.encode(samples.q[0], samples.dq[0]))
    first, second = episodes
    assert np.array_equal(first.targets[-1], FIRST_SAMPLES.q[-1])  # not SECOND's first sample
    assert not np.array_equal(first.targets[-1], SECOND_SAMPLES.q[0])
    assert np.array_equal(second.inputs[WARMUP_ROWS], encoder.encode(SECOND_SAMPLES.q[0], SECOND_SAMPLES.dq[0]))


def test_transform_is_copied_from_outside_the_training_data() -> None:
    """The frozen transform re-derives bitwise from the bound source, which is never one of the takes (I8)."""
    recipe, _model = _build([FIRST], (10,))
    assert recipe.transform_source == FROZEN_SOURCE
    assert recipe.transform.derived_from == (HISTORICAL.artifact.artifact_id,)
    assert FROZEN_SOURCE.artifact_id not in [d.artifact_id for d in recipe.datasets]
    recipe.check_transform_source({HISTORICAL.artifact.artifact_id: HISTORICAL_NORMALIZATION})
    assert recipe.check_transform_record(HISTORICAL) == HISTORICAL_NORMALIZATION

    with pytest.raises(ValueError, match="must name exactly the transform source"):
        dataclasses.replace(recipe, transform=dataclasses.replace(TRANSFORM, derived_from=(FIRST.artifact_id,)))
    with pytest.raises(ValueError, match="one of the training datasets"):
        dataclasses.replace(recipe, transform_source=dataclasses.replace(FROZEN_SOURCE, artifact_id=FIRST.artifact_id))
    with pytest.raises(ValueError, match="records no normalization statistics"):
        recipe.check_transform_record(dataclasses.replace(HISTORICAL, normalization=None))
    tampered = dataclasses.replace(
        recipe, transform_source=dataclasses.replace(FROZEN_SOURCE, payload_sha256="ab" * 32)
    )
    with pytest.raises(ValueError, match="not the transform source"):
        tampered.check_transform_record(HISTORICAL)


def test_recorded_normalization_stays_required_for_the_historical_schemas() -> None:
    """Schema 1 derives its transform from a training dataset, so that dataset must record its statistics."""
    legacy, _model = create_recipe(
        "legacy",
        EsnConfig(reservoir=RESERVOIR, readout=ReadoutConfig(alpha=BASE_ALPHA)),
        sources=[FROZEN_SOURCE],
        samples={FROZEN_SOURCE.artifact_id: FIRST_SAMPLES},
        dof=2,
        task_code_dim=0,
        preprocessing=HISTORICAL.preprocessing,
        transform=TRANSFORM,
    )
    assert legacy.schema_version == 1
    assert legacy.transform_source is None
    legacy.check_dataset_record(FROZEN_SOURCE, HISTORICAL)
    with pytest.raises(ValueError, match="records no normalization statistics"):
        legacy.check_dataset_record(FROZEN_SOURCE, dataclasses.replace(HISTORICAL, normalization=None))


def test_schema_gating_keeps_the_historical_schemas_unweighted() -> None:
    """Schemas 1 and 2 refuse every schema 3 field, schema 3 requires them, and an unknown version is refused."""
    recipe, _model = _build([FIRST, SECOND], (1, 1))
    with pytest.raises(ValueError, match="unsupported recipe schema version 4"):
        dataclasses.replace(recipe, schema_version=4)
    with pytest.raises(ValueError, match="need a schema 3 recipe"):
        dataclasses.replace(recipe, schema_version=2)
    unbound = dataclasses.replace(recipe.training, base_alpha=None, regularization_rule=None)
    with pytest.raises(ValueError, match="need a schema 3 recipe"):
        dataclasses.replace(recipe, schema_version=1, validation=None, training=unbound)
    with pytest.raises(ValueError, match="bind the dataset their input transform was copied from"):
        dataclasses.replace(recipe, transform_source=None)
    with pytest.raises(ValueError, match="explicit-bias readout"):
        dataclasses.replace(recipe, esn=_esn(2, explicit_bias=None, include_bias=True))
    with pytest.raises(ValueError, match="one multiplicity per dataset"):
        dataclasses.replace(recipe, training=dataclasses.replace(recipe.training, source_counts=(1, 1, 1)))
    with pytest.raises(ValueError, match="must equal"):
        dataclasses.replace(recipe, esn=_esn(1))  # alpha_0 instead of K alpha_0
    with pytest.raises(ValueError, match="'count_scaled'"):
        dataclasses.replace(
            recipe, training=dataclasses.replace(recipe.training, regularization_rule="base"), esn=_esn(1)
        )


def test_the_weighting_fields_are_one_construction() -> None:
    """The three weighting fields are recorded together, hold approved values, and exclude the other constructions."""
    with pytest.raises(ValueError, match="recorded together"):
        TrainingSpec(washout="warmup_hold", warmup_s=WARMUP_S, episode_weighting=EQUAL_EPISODE_WEIGHTING)
    with pytest.raises(ValueError, match="episode_weighting must be"):
        _spec((1,), weighting="per_row")
    with pytest.raises(ValueError, match="weight_reference_rows must be >= 1"):
        _spec((1,), reference=0)
    with pytest.raises(ValueError, match="at least one episode"):
        _spec((1, 0))
    with pytest.raises(ValueError, match="warmup_hold"):
        dataclasses.replace(_spec((1,)), washout="prime_phase", warmup_s=None)
    with pytest.raises(ValueError, match="separate repetition and varied-data constructions"):
        dataclasses.replace(_spec((1,)), additional_repeats=16)
    spec = _spec((1, 10))
    assert spec.episode_count == 11
    assert spec.uses_weighted_features
    assert not TrainingSpec().uses_weighted_features


def test_a_regularization_override_cannot_understate_the_episode_count() -> None:
    """A ten-copy recipe stacks ten episodes, so its ridge parameter is ten alpha_0 and no override may claim fewer."""
    recipe, _model = _build([FIRST], (10,))
    assert recipe.solver_alpha == 10 * BASE_ALPHA
    understated = dataclasses.replace(recipe.training, regularization_count=1)
    with pytest.raises(ValueError, match="regularization_count"):
        # Without this rule the copy control would train at alpha_0, the regularization the singleton uses.
        dataclasses.replace(recipe, training=understated, esn=_esn(1))
    matched = dataclasses.replace(recipe.training, regularization_count=10)
    assert dataclasses.replace(recipe, training=matched).solver_alpha == 10 * BASE_ALPHA


def test_a_refit_verifies_the_recorded_per_episode_weighting() -> None:
    """Equal weights recorded for unequal-length episodes fail the refit instead of passing on the errors alone."""
    recipe, _model = _build([FIRST, SECOND], (1, 1))
    assert recipe.fit.episode_weights != (1.0, 1.0)
    tampered = dataclasses.replace(recipe, fit=dataclasses.replace(recipe.fit, episode_weights=(1.0, 1.0)))
    with pytest.raises(RecipeMismatchError, match="episode_weights"):
        tampered.refit(SAMPLES, scenario=SCENARIO)
    rows = dataclasses.replace(recipe, fit=dataclasses.replace(recipe.fit, episode_loss_rows=(1, 1)))
    with pytest.raises(RecipeMismatchError, match="episode_loss_rows"):
        rows.refit(SAMPLES, scenario=SCENARIO)


def test_schema_three_records_the_weighting_its_fit_used() -> None:
    """A schema 3 recipe carries the per-episode rows and weights, so a refit can verify the weighting at all."""
    recipe, _model = _build([FIRST, SECOND], (1, 1))
    unweighted = dataclasses.replace(recipe.fit, episode_loss_rows=None, episode_weights=None)
    with pytest.raises(ValueError, match="per-episode loss rows and weights"):
        dataclasses.replace(recipe, fit=unweighted)


def test_the_contractive_construction_is_one_parent_and_its_frozen_bank() -> None:
    """Nine synthetic episodes join one parent at multiplicity one; copies and the inherited augmentation are out."""
    spec = _spec((1,), contractive=CONTRACTIVE)
    assert spec.contractive == CONTRACTIVE
    assert CONTRACTIVE.n_synthetic == MANUAL_N_SYNTHETIC
    assert spec.episode_count == 1 + MANUAL_N_SYNTHETIC == 10
    assert spec.uses_weighted_features
    assert TrainingSpec(washout="warmup_hold", warmup_s=WARMUP_S).contractive is None

    with pytest.raises(ValueError, match="exactly one dataset at multiplicity one"):
        _spec((1, 1), contractive=CONTRACTIVE)  # a contractive recipe grows exactly one parent's bank
    with pytest.raises(ValueError, match="exactly one dataset at multiplicity one"):
        _spec((10,), contractive=CONTRACTIVE)  # copies are the separate duplication control
    with pytest.raises(ValueError, match="exactly one dataset at multiplicity one"):
        TrainingSpec(washout="warmup_hold", warmup_s=WARMUP_S, contractive=CONTRACTIVE)
    with pytest.raises(ValueError, match="separate repetition and varied-data constructions"):
        dataclasses.replace(_spec((1,), contractive=CONTRACTIVE), augmentation=AUGMENTATION)
    with pytest.raises(ValueError, match="separate repetition and varied-data constructions"):
        dataclasses.replace(_spec((1,), contractive=CONTRACTIVE), additional_repeats=16)
    with pytest.raises(ValueError, match="seed_bank"):
        ContractiveTrainingSpec(seed_bank=-1, dwell_start_s=DWELL_START_S)
    with pytest.raises(ValueError, match="dwell_start_s"):
        ContractiveTrainingSpec(seed_bank=1, dwell_start_s=0.0)


def test_a_contractive_recipe_trains_its_parent_with_the_regenerated_bank(tmp_path: Path) -> None:
    """Schema 3 with a contractive bank: ten episodes at ten alpha_0, written, reloaded, and refitted exactly."""
    recipe, _model = _build([PARENT], (1,), contractive=CONTRACTIVE, scenario=MANUAL_SCENARIO)
    assert recipe.schema_version == WEIGHTED_SCHEMA_VERSION
    assert recipe.training.contractive == CONTRACTIVE
    assert recipe.training.episode_count == 10
    assert recipe.solver_alpha == 10 * BASE_ALPHA
    labels = expected_episode_labels(recipe.training, (PARENT.artifact_id,))
    assert labels == recipe.fit.episodes
    assert labels[0] == PARENT.artifact_id
    assert labels[1] == f"{PARENT.artifact_id}#contractive-001"
    assert labels[-1] == f"{PARENT.artifact_id}#contractive-{MANUAL_N_SYNTHETIC:03d}"
    assert recipe.fit.episode_loss_rows == (PARENT_ROWS,) * 10
    assert recipe.fit.episode_weights == (ROWS_REFERENCE / PARENT_ROWS,) * 10
    assert recipe.fit.loss_rows == 10 * PARENT_ROWS

    file = tmp_path / "contractive.toml"
    write_recipe(file, recipe)
    text = file.read_text(encoding="utf-8")
    assert "[training.contractive]" in text
    assert "seed_bank = 5" in text
    loaded = load_recipe(file)
    assert loaded == recipe
    _refit, report = loaded.refit(SAMPLES, scenario=MANUAL_SCENARIO)
    assert report == recipe.fit


def test_the_historical_schemas_refuse_the_contractive_bank() -> None:
    """Only schema 3 may carry the manual protocol's contractive construction."""
    recipe, _model = _build([PARENT], (1,), contractive=CONTRACTIVE, scenario=MANUAL_SCENARIO)
    with pytest.raises(ValueError, match="need a schema 3 recipe"):
        dataclasses.replace(recipe, schema_version=2)
    unbound = dataclasses.replace(recipe.training, base_alpha=None, regularization_rule=None)
    with pytest.raises(ValueError, match="need a schema 3 recipe"):
        dataclasses.replace(recipe, schema_version=1, validation=None, transform_source=None, training=unbound)


def test_a_contractive_bank_that_cannot_be_grown_names_its_parent() -> None:
    """A generation failure is reported with the parent it belongs to, never as a silently short bank."""
    unreachable = dataclasses.replace(CONTRACTIVE, dwell_start_s=10.0)  # past the end of the recording
    with pytest.raises(ValueError, match=PARENT.artifact_id) as excinfo:
        _build([PARENT], (1,), contractive=unreachable, scenario=MANUAL_SCENARIO)
    assert "dwell onset" in str(excinfo.value.__cause__)  # the generator's own diagnosis stays attached
    with pytest.raises(ValueError, match="scenario"):
        _build([PARENT], (1,), contractive=CONTRACTIVE, scenario=None)


def test_training_validation_accepts_both_task_schemas() -> None:
    """The manual protocol binds a ManualScenarioConfig; both schemas report the same limits for the same arm."""
    scripted = _validation()
    manual = TrainingValidation.from_scenario(MANUAL_SCENARIO, MANUAL_SCENARIO_FILE, root=REPO_ROOT)
    assert manual.scenario_file == "tests/fixtures/configs/planar_2dof_manual_fixture.toml"
    assert manual.scenario_sha256 != scripted.scenario_sha256  # a different file is bound
    limits = ("velocity_limit", "joint_lower", "joint_upper", "endpoint_radius")
    assert [getattr(manual, name) for name in limits] == [getattr(scripted, name) for name in limits]
    # Both schemas describe the same fixture arm, so either validation accepts either configuration.
    manual.check(MANUAL_SCENARIO)
    manual.check(SCENARIO)
    scripted.check(MANUAL_SCENARIO)
    assert scripted == _validation()  # existing callers are unchanged

    slower = dataclasses.replace(
        MANUAL_SCENARIO,
        limits=dataclasses.replace(MANUAL_SCENARIO.limits, velocity=(1.0, 1.0)),
        acquisition=dataclasses.replace(MANUAL_SCENARIO.acquisition, velocity_bound_rad_s=1.0),
    )
    with pytest.raises(ValueError, match="training-validation limits"):
        manual.check(slower)

# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-012: recipes train through deterministic augmentation and refit exactly from config + dataset."""

from __future__ import annotations

import numpy as np
import pytest

from arm_rc_ctrl.data.derivatives import DerivativeConfig, differentiate
from arm_rc_ctrl.data.normalization import fit_normalization
from arm_rc_ctrl.data.records import Preprocessing
from arm_rc_ctrl.data.samples import SampleSet
from arm_rc_ctrl.rc.esn import EsnConfig, EsnModel, ReadoutConfig, ReservoirConfig
from arm_rc_ctrl.rc.recipe import (
    AugmentationTrainingSpec,
    DatasetSource,
    ModelRecipe,
    TrainingSpec,
    create_recipe,
    expected_episode_labels,
)
from arm_rc_ctrl.rc.teacher_forcing import InputTransform
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import ScenarioConfig, endpoint_positions, load_scenario

SCENARIO_FILE = repository_root() / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
SCENARIO = load_scenario(SCENARIO_FILE)
DERIVATIVES = DerivativeConfig(method="central")
SOURCE_ID = "processed-20260830-555555555555"
N = 101
DT = 0.01
ESN = EsnConfig(
    reservoir=ReservoirConfig(
        n_neurons=40, spectral_radius=0.85, sparsity=0.9, leak_rate=0.4, input_scaling=0.4, seed=23
    ),
    readout=ReadoutConfig(alpha=1e-6),
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
        samples.arrays(), ("q", "dq"), fitted_on=(SOURCE_ID,), training_rows=np.ones(N, dtype=np.bool_)
    )
    return InputTransform.derive("fixed_scale", normalization, fixed_scales={"q": 0.3, "dq": 4.0})


def _build(
    samples: SampleSet, *, spec: TrainingSpec, scenario: ScenarioConfig | None = None
) -> tuple[ModelRecipe, EsnModel]:
    return create_recipe(
        "augmented-test",
        ESN,
        sources=[DatasetSource(SOURCE_ID, "ab" * 32, "data/records/processed/x.toml")],
        samples={SOURCE_ID: samples},
        dof=2,
        task_code_dim=0,
        preprocessing=PREPROCESSING,
        transform=_transform(samples),
        training=spec,
        scenario=scenario,
    )


def test_augmentation_spec_is_validated_and_requires_warmup_hold() -> None:
    """The augmentation section holds approved values only and never combines with the M3 washout."""
    with pytest.raises(ValueError, match="approved"):
        AugmentationTrainingSpec(
            family="contractive", n_synthetic=8, sigma_rad=0.025, phi=0.99, gamma=1.0, seed_bank=1, attempt_budget=64
        )
    with pytest.raises(ValueError, match="family"):
        AugmentationTrainingSpec(
            family="both",  # type: ignore[arg-type]
            n_synthetic=16,
            sigma_rad=0.025,
            phi=0.99,
            gamma=1.0,
            seed_bank=1,
            attempt_budget=64,
        )
    with pytest.raises(ValueError, match="warmup_hold"):
        TrainingSpec(washout="prime_phase", augmentation=AUGMENTATION)
    spec = TrainingSpec(washout="warmup_hold", warmup_s=0.25, augmentation=AUGMENTATION)
    assert spec.augmentation is AUGMENTATION


def test_expected_episode_labels_cover_the_synthetic_set() -> None:
    """One label per source without augmentation; source plus one per synthetic episode with it."""
    plain = TrainingSpec(washout="warmup_hold", warmup_s=0.25)
    assert expected_episode_labels(plain, (SOURCE_ID,)) == (SOURCE_ID,)
    augmented = TrainingSpec(washout="warmup_hold", warmup_s=0.25, augmentation=AUGMENTATION)
    labels = expected_episode_labels(augmented, (SOURCE_ID,))
    assert labels[0] == SOURCE_ID
    assert len(labels) == 1 + AUGMENTATION.n_synthetic
    assert labels[1] == f"{SOURCE_ID}#contractive-001"
    assert labels[-1] == f"{SOURCE_ID}#contractive-{AUGMENTATION.n_synthetic:03d}"


def test_augmented_recipe_trains_and_refits_exactly() -> None:
    """Training regenerates the synthetic episodes deterministically; refit reproduces the fit bitwise."""
    samples = _samples()
    spec = TrainingSpec(washout="warmup_hold", warmup_s=0.25, augmentation=AUGMENTATION)
    recipe, _ = _build(samples, spec=spec, scenario=SCENARIO)
    assert recipe.fit.episodes == expected_episode_labels(spec, (SOURCE_ID,))
    assert len(recipe.fit.episodes) == 1 + AUGMENTATION.n_synthetic
    model, report = recipe.refit({SOURCE_ID: samples}, scenario=SCENARIO)
    assert report == recipe.fit
    assert model.fitted


def test_augmented_episodes_require_the_scenario() -> None:
    """Without the scenario the envelope and validity limits are undefined; the error says so."""
    samples = _samples()
    spec = TrainingSpec(washout="warmup_hold", warmup_s=0.25, augmentation=AUGMENTATION)
    with pytest.raises(ValueError, match="scenario"):
        _build(samples, spec=spec)
    recipe, _ = _build(samples, spec=spec, scenario=SCENARIO)
    with pytest.raises(ValueError, match="scenario"):
        recipe.refit({SOURCE_ID: samples})


def test_synthetic_episodes_are_deterministic_and_warmup_shaped() -> None:
    """Two builds give byte-identical episode inputs; every episode carries the warm-up washout."""
    samples = _samples()
    spec = TrainingSpec(washout="warmup_hold", warmup_s=0.25, augmentation=AUGMENTATION)
    first, _ = _build(samples, spec=spec, scenario=SCENARIO)
    episodes_a = first.episodes({SOURCE_ID: samples}, scenario=SCENARIO)
    episodes_b = first.episodes({SOURCE_ID: samples}, scenario=SCENARIO)
    assert len(episodes_a) == 1 + AUGMENTATION.n_synthetic
    for a, b in zip(episodes_a, episodes_b, strict=True):
        assert a.source == b.source
        assert np.array_equal(a.inputs, b.inputs)
        assert np.array_equal(a.targets, b.targets)
        assert a.washout_len == 25
    original, synthetic = episodes_a[0], episodes_a[1]
    assert original.source == SOURCE_ID
    assert not np.array_equal(original.targets, synthetic.targets)  # the perturbation reaches the targets

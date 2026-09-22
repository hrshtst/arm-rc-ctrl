# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MS-002: one sampled configuration, its four learned arms, and what sampling may not touch.

A trial samples the ESN and the warm-up; everything else is inherited. These
tests hold that boundary: the reservoir carries the sampled parameters with the
fixed seed, the fixed filter policy is the same whatever the trial draws, zero
warm-up is a real choice rather than an unhandled edge, the four arms keep the
closed experiment's weighting and multiplicities at the sampled regularization,
and no sampled configuration can collide with the closed study's own six or
change a committed identity.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.experiments.manual_recipes import (
    manual_arms,
    recipe_for_arm,
    solver_alpha,
    training_spec_for_arm,
)
from arm_rc_ctrl.experiments.manual_sampled import (
    LEARNED_ARMS,
    SampledPoint,
    configuration_label,
    contractive_bank,
    fixed_policy_mismatches,
    sampled_configuration,
    sampled_entry,
    sampled_esn,
    sampled_point,
)
from arm_rc_ctrl.experiments.manual_search import load_manual_search
from arm_rc_ctrl.experiments.manual_study import load_study
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from arm_rc_ctrl.experiments.manual_fixture import ManualFixture
    from arm_rc_ctrl.experiments.manual_recipes import ManualArmSpec
    from arm_rc_ctrl.experiments.manual_search import ManualSearchProtocol
    from arm_rc_ctrl.experiments.manual_study import StudyConfiguration, StudyManifest
    from arm_rc_ctrl.rc.recipe import TrainingSpec

ROOT = repository_root()
PROTOCOL_FILE = ROOT / "configs/studies/manual_esn_search_v1.toml"
POINT = SampledPoint(
    n_neurons=150,
    spectral_radius=1.05,
    sparsity=0.8,
    leak_rate=0.05,
    input_scaling=0.4,
    alpha_0=0.01,
    warmup_s=0.5,
)


@pytest.fixture(scope="module")
def protocol() -> ManualSearchProtocol:
    """The committed search protocol; sampling is always read against a protocol."""
    return load_manual_search(PROTOCOL_FILE)


@pytest.fixture(scope="module")
def study(protocol: ManualSearchProtocol) -> StudyManifest:
    """The closed study the search inherits its data, banks and construction from."""
    return load_study(protocol.study)


def _spec(study: StudyManifest, arm: ManualArmSpec, configuration: StudyConfiguration) -> TrainingSpec:
    """One arm's training construction at a sampled configuration, with the frozen bank where it applies."""
    assignment = arm.assignment
    construction = contractive_bank(study, assignment).spec if arm.arm == "C10" and assignment else None
    return training_spec_for_arm(
        arm, warmup_s=configuration.warmup_s, base_alpha=configuration.base_alpha, contractive=construction
    )


# --- what a trial samples ----------------------------------------------------------------------


def test_a_sampled_configuration_carries_the_point_and_the_fixed_seed(protocol: ManualSearchProtocol) -> None:
    """The reservoir is the trial's, except the seed, which the owner fixed."""
    configuration = sampled_configuration(protocol, POINT, trial=7)
    reservoir = configuration.reservoir
    assert (reservoir.n_neurons, reservoir.spectral_radius, reservoir.sparsity) == (150, 1.05, 0.8)
    assert (reservoir.leak_rate, reservoir.input_scaling) == (0.05, 0.4)
    assert reservoir.seed == protocol.fixed.reservoir_seed == 896
    assert configuration.base_alpha == POINT.alpha_0
    assert configuration.warmup_s == POINT.warmup_s
    assert configuration.source_trial == 7


def test_the_fixed_policy_is_the_same_whatever_the_trial_draws(protocol: ManualSearchProtocol) -> None:
    """Sampling moves the ESN and the warm-up; it never moves the filters, seed or trackers."""
    first = sampled_configuration(protocol, POINT, trial=1)
    other = sampled_configuration(protocol, replace(POINT, n_neurons=400, warmup_s=0.0, alpha_0=0.9), trial=2)
    for configuration in (first, other):
        assert configuration.velocity_cutoff_hz == protocol.fixed.velocity_cutoff_hz
        assert configuration.acceleration_cutoff_hz == protocol.fixed.acceleration_cutoff_hz
        assert configuration.reservoir.seed == protocol.fixed.reservoir_seed
        assert fixed_policy_mismatches(protocol, configuration) == []


def test_a_configuration_that_drifted_from_the_fixed_policy_is_reported(protocol: ManualSearchProtocol) -> None:
    """The invariance is checkable after the fact, not only at construction."""
    configuration = sampled_configuration(protocol, POINT, trial=3)
    drifted = replace(configuration, velocity_cutoff_hz=1.0)
    reseeded = replace(configuration, reservoir=replace(configuration.reservoir, seed=897))
    assert any("velocity_cutoff_hz" in text for text in fixed_policy_mismatches(protocol, drifted))
    assert any("seed" in text for text in fixed_policy_mismatches(protocol, reseeded))


def test_zero_warm_up_is_a_choice_the_training_construction_accepts(
    protocol: ManualSearchProtocol, study: StudyManifest
) -> None:
    """Zero means no warm-up phase, and it must reach every arm's training spec unchanged."""
    configuration = sampled_configuration(protocol, replace(POINT, warmup_s=0.0), trial=4)
    assert configuration.warmup_s == 0.0
    for arm in LEARNED_ARMS:
        assert _spec(study, arm, configuration).warmup_s == 0.0


@pytest.mark.parametrize("warmup_s", [0.0, 0.25, 0.5, 1.0, 2.0])
def test_every_approved_warm_up_is_accepted(protocol: ManualSearchProtocol, warmup_s: float) -> None:
    """The five approved choices all build a configuration."""
    assert sampled_configuration(protocol, replace(POINT, warmup_s=warmup_s), trial=5).warmup_s == warmup_s


# --- a point must lie in the approved space -----------------------------------------------------


def test_a_point_is_read_from_the_trial_parameters_and_checked(protocol: ManualSearchProtocol) -> None:
    """Optuna hands back a mapping; it is validated against the approved space before use."""
    params = {
        "n_neurons": 200,
        "spectral_radius": 1.0,
        "sparsity": 0.7,
        "leak_rate": 0.1,
        "input_scaling": 0.5,
        "alpha_0": 0.05,
        "warmup_s": 1.0,
    }
    point = sampled_point(protocol.space, params)
    assert (point.n_neurons, point.warmup_s) == (200, 1.0)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("n_neurons", 90),
        ("n_neurons", 420),
        ("n_neurons", 125),
        ("spectral_radius", 1.4),
        ("sparsity", 0.4),
        ("leak_rate", 0.5),
        ("input_scaling", 2.0),
        ("alpha_0", 2.0),
        ("warmup_s", 0.75),
    ],
)
def test_a_point_outside_the_approved_space_is_refused(protocol: ManualSearchProtocol, name: str, value: float) -> None:
    """A parameter off the approved grid or outside its bounds is never trained on."""
    params = {
        "n_neurons": 200,
        "spectral_radius": 1.0,
        "sparsity": 0.7,
        "leak_rate": 0.1,
        "input_scaling": 0.5,
        "alpha_0": 0.05,
        "warmup_s": 1.0,
    }
    with pytest.raises(ValueError, match=name):
        sampled_point(protocol.space, {**params, name: value})


def test_missing_or_unknown_parameters_are_refused(protocol: ManualSearchProtocol) -> None:
    """A point is exactly the searched parameters: no silent default, no extra."""
    with pytest.raises(ValueError, match="n_neurons"):
        sampled_point(protocol.space, {"spectral_radius": 1.0})
    complete = {
        "n_neurons": 200,
        "spectral_radius": 1.0,
        "sparsity": 0.7,
        "leak_rate": 0.1,
        "input_scaling": 0.5,
        "alpha_0": 0.05,
        "warmup_s": 1.0,
    }
    with pytest.raises(ValueError, match="velocity_cutoff_hz"):
        sampled_point(protocol.space, {**complete, "velocity_cutoff_hz": 30.0})


# --- the four learned arms at a sampled configuration -------------------------------------------


def test_the_learned_arms_are_the_closed_experiment_s() -> None:
    """S, M10, R10 and C10, with the same parents; replay carries no ESN and is not an arm here."""
    labels = [arm.label for arm in LEARNED_ARMS]
    assert labels == [arm.label for arm in manual_arms()]
    assert len(labels) == 31
    assert sum(1 for arm in LEARNED_ARMS if arm.arm == "M10") == 1
    assert {arm.arm for arm in LEARNED_ARMS} == {"S", "M10", "R10", "C10"}


@pytest.mark.parametrize(("kind", "count", "recorded"), [("S", 1, 1), ("M10", 10, 10), ("R10", 10, 10), ("C10", 10, 1)])
def test_each_arm_keeps_its_weighting_and_multiplicity_at_the_sampled_alpha(
    protocol: ManualSearchProtocol, study: StudyManifest, kind: str, count: int, recorded: int
) -> None:
    """The sampled alpha_0 scales with the arm's episode count, as the closed experiment's does."""
    configuration = sampled_configuration(protocol, POINT, trial=6)
    arm = next(item for item in LEARNED_ARMS if item.arm == kind)
    spec = _spec(study, arm, configuration)
    rule, counts = spec.regularization_rule, spec.source_counts
    assert rule is not None
    assert counts is not None
    assert arm.count == count
    assert spec.base_alpha == POINT.alpha_0
    assert solver_alpha(POINT.alpha_0, rule, arm.count) == pytest.approx(POINT.alpha_0 * count)
    assert sum(counts) == recorded, "synthetic episodes are a construction, not stored sources"
    assert spec.episode_weighting == "equal_episode"


def test_copies_are_multiplicities_and_synthetic_episodes_are_a_construction() -> None:
    """R10 records ten copies of one source; C10 records nine synthetic episodes of its parent."""
    copies = next(arm for arm in LEARNED_ARMS if arm.arm == "R10")
    synthetic = next(arm for arm in LEARNED_ARMS if arm.arm == "C10")
    assert copies.source_counts == (10,)
    assert synthetic.source_counts == (1,)
    assert synthetic.count == 10


# --- the recipes those arms actually build ------------------------------------------------------


@pytest.mark.parametrize("kind", ["S", "M10", "R10", "C10"])
def test_a_sampled_configuration_builds_each_arm_s_recipe(
    protocol: ManualSearchProtocol, manual_fixture: ManualFixture, kind: str
) -> None:
    """The sampled reservoir, warm-up and regularization reach the recipe the arm is fitted from.

    This runs the closed experiment's own recipe constructor over the fixture
    bank, so what is tested is that a sampled configuration is an ordinary
    input to it, not that a second training path agrees with the first.
    """
    fixture = manual_fixture
    configuration = sampled_configuration(protocol, replace(POINT, warmup_s=0.0), trial=11)
    arm = next(item for item in fixture.manifest.entries if item.arm.arm == kind).arm
    assignment = arm.assignment
    construction = (
        contractive_bank(fixture.manifest, assignment).spec if kind == "C10" and assignment is not None else None
    )
    recipe, _model = recipe_for_arm(
        arm,
        esn=sampled_esn(configuration, arm, readout=fixture.manifest.readout, anchor=fixture.manifest.anchor),
        sources=fixture.inputs.sources,
        samples=fixture.samples,
        dof=fixture.inputs.dof,
        task_code_dim=fixture.inputs.task_code_dim,
        preprocessing=fixture.inputs.preprocessing,
        transform=fixture.manifest.transform.transform,
        validation=fixture.manifest.validation,
        warmup_s=configuration.warmup_s,
        base_alpha=configuration.base_alpha,
        scenario=fixture.inputs.scenario,
        anchor=fixture.manifest.anchor,
        contractive=construction,
    )
    assert recipe.esn.reservoir.n_neurons == configuration.reservoir.n_neurons
    assert recipe.esn.reservoir.seed == protocol.fixed.reservoir_seed
    assert recipe.esn.readout.alpha == pytest.approx(configuration.base_alpha * arm.count)
    assert recipe.training.warmup_s == 0.0
    assert recipe.training.base_alpha == configuration.base_alpha


# --- what sampling may never touch --------------------------------------------------------------


def test_a_sampled_label_can_never_be_one_of_the_closed_study_s(study: StudyManifest) -> None:
    """The closed study's six configurations keep their names and their identities."""
    closed = {configuration.label for configuration in study.configurations}
    labels = {configuration_label(trial) for trial in range(100)}
    assert not labels & closed
    assert len(labels) == 100


def test_the_closed_study_manifest_is_untouched_by_sampling(protocol: ManualSearchProtocol) -> None:
    """Reading the study to build a sampled configuration must not rewrite or re-derive it."""
    before = protocol.study.read_bytes()
    sampled_configuration(protocol, POINT, trial=8)
    assert protocol.study.read_bytes() == before


def test_two_trials_at_the_same_point_are_different_configurations(protocol: ManualSearchProtocol) -> None:
    """A configuration is identified by its trial, so evidence of one never serves another."""
    first = sampled_configuration(protocol, POINT, trial=9)
    second = sampled_configuration(protocol, POINT, trial=10)
    assert first.label != second.label
    assert replace(first, label=second.label, source_trial=second.source_trial) == second


# --- what the owner's review of 2026-09-23 found ------------------------------------------------


@pytest.mark.parametrize("n_neurons", [100.9, 200.5, 400.9, 149.5])
def test_a_fractional_reservoir_size_is_refused_rather_than_truncated(
    protocol: ManualSearchProtocol, n_neurons: float
) -> None:
    """Truncating would train a different point from the one the trial supplied."""
    params = {
        "n_neurons": n_neurons,
        "spectral_radius": 1.0,
        "sparsity": 0.7,
        "leak_rate": 0.1,
        "input_scaling": 0.5,
        "alpha_0": 0.05,
        "warmup_s": 1.0,
    }
    with pytest.raises(ValueError, match="n_neurons"):
        sampled_point(protocol.space, params)


def test_an_integral_reservoir_size_given_as_a_float_is_accepted(protocol: ManualSearchProtocol) -> None:
    """A resumed study may hand back 200.0; that is the same point, and it stays on the grid."""
    params = {
        "n_neurons": 200.0,
        "spectral_radius": 1.0,
        "sparsity": 0.7,
        "leak_rate": 0.1,
        "input_scaling": 0.5,
        "alpha_0": 0.05,
        "warmup_s": 1.0,
    }
    assert sampled_point(protocol.space, params).n_neurons == 200


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_neurons", 125),
        ("n_neurons", 90),
        ("input_scaling", 2.0),
        ("spectral_radius", 1.4),
        ("alpha_0", 2.0),
        ("warmup_s", 0.75),
    ],
)
def test_a_point_built_directly_is_still_checked_against_the_approved_space(
    protocol: ManualSearchProtocol, field: str, value: float
) -> None:
    """A configuration is where a point becomes trainable, so that is where the space is enforced."""
    with pytest.raises(ValueError, match=field):
        sampled_configuration(protocol, replace(POINT, **{field: value}), trial=12)


def test_an_approved_point_built_directly_still_builds_a_configuration(protocol: ManualSearchProtocol) -> None:
    """The check refuses what the owner did not approve, and nothing else."""
    assert sampled_configuration(protocol, POINT, trial=13).reservoir.n_neurons == POINT.n_neurons


# --- a sampled entry, bound as the frozen study binds its own -----------------------------------


def test_the_public_entry_builder_reproduces_a_frozen_entry_s_identity(study: StudyManifest) -> None:
    """Publishing the study's own derivation must not have changed it."""
    entry = next(item for item in study.entries if item.arm.arm == "C10")
    configuration = study.configuration(entry.configuration)
    rebuilt = sampled_entry(study, configuration, entry.arm)
    assert rebuilt.fit_identity == entry.fit_identity
    assert rebuilt.contractive == entry.contractive


@pytest.mark.parametrize("kind", ["S", "M10", "R10", "C10"])
def test_a_sampled_entry_is_keyed_apart_from_every_frozen_one(
    protocol: ManualSearchProtocol, study: StudyManifest, kind: str
) -> None:
    """The configuration label is part of the fit key, so sampled work never serves frozen work."""
    configuration = sampled_configuration(protocol, POINT, trial=14)
    arm = next(item.arm for item in study.entries if item.arm.arm == kind)
    entry = sampled_entry(study, configuration, arm)
    assert entry.configuration == configuration.label
    assert entry.warmup_s == configuration.warmup_s
    assert entry.fit_identity not in {frozen.fit_identity for frozen in study.entries}
    assert entry.execution_identity == study.execution.identity


def test_a_sampled_configuration_resolves_through_the_fit_inputs(
    protocol: ManualSearchProtocol, manual_fixture: ManualFixture
) -> None:
    """The existing fit and evaluation path reads a sampled configuration without a manifest change."""
    configuration = sampled_configuration(protocol, POINT, trial=15)
    inputs = replace(manual_fixture.inputs, sampled=(configuration,))
    arm = next(item.arm for item in manual_fixture.manifest.entries if item.arm.arm == "M10")
    entry = sampled_entry(manual_fixture.manifest, configuration, arm)
    assert inputs.configuration(entry) == configuration
    assert inputs.base_esn(entry).reservoir == configuration.reservoir
    frozen = next(item for item in manual_fixture.manifest.entries if item.arm.arm == "M10")
    assert inputs.configuration(frozen) == manual_fixture.manifest.configuration(frozen.configuration)


def test_a_sampled_configuration_may_not_shadow_a_frozen_one(manual_fixture: ManualFixture) -> None:
    """A label collision would let sampled work answer for a frozen model."""
    frozen = manual_fixture.manifest.configurations[0]
    with pytest.raises(ValueError, match="shadow"):
        replace(manual_fixture.inputs, sampled=(frozen,))

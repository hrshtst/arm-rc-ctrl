# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-007 (stage 3): the numerical copy controls of the manual-demonstration study.

The weighted ridge system is checked against an independently assembled
``sqrt(W)``-scaled solve, the literal copies of a duplication control are
verified to share every input the fit sees, ``R10_i`` is compared with ``S_i``
on the fixed manual probes, and every fit is refitted in a fresh process. The
fixture is hermetic: a temporary repository root and storage root, never the
external store.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.data.arrays import array_digest
from arm_rc_ctrl.experiments import manual_numerics
from arm_rc_ctrl.experiments.manual_fits import (
    FIT_FILE,
    RECIPE_FILE,
    WEIGHTS_FILE,
    EpisodeIdentity,
    EvidenceIntegrityError,
    ManualFitRecord,
    ManualFitStore,
    cache_uri,
    episode_identities,
    fit_entry,
    fit_record_from,
    recipe_text_of,
    states_witness,
)
from arm_rc_ctrl.experiments.manual_fixture import ManualFixture
from arm_rc_ctrl.experiments.manual_numerics import (
    COMPARISON_PAIR,
    LARGE_FILE_LIMIT_BYTES,
    NUMERICAL_ARMS,
    PROBE_BANKS,
    TOLERANCES,
    Comparison,
    FreshRefit,
    ManualNumericalValidation,
    ManualStudyContext,
    NormalEquations,
    ProbeBank,
    RepeatIdentity,
    Tolerances,
    build_probes,
    fresh_refit,
    load_validation,
    main,
    numerical_entries,
    predict_probes,
    refit_in_subprocess,
    render_validation_markdown,
    run_validation,
    validation_to_json,
    weighted_normal_equations,
)
from arm_rc_ctrl.experiments.manual_recipes import ManualArmSpec, fit_identity, recipe_for_arm
from arm_rc_ctrl.experiments.manual_study import load_study
from arm_rc_ctrl.rc.recipe import DatasetSource, RclibIdentity, load_recipe
from arm_rc_ctrl.rc.training import FitReport, harvest_episode

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from arm_rc_ctrl.experiments.manual_study import ContractiveBank, StudyManifest, StudyModel
    from arm_rc_ctrl.rc.recipe import ContractiveTrainingSpec

FULL_CONFIGURATIONS = (
    "feasible-best",
    "feasible-middle",
    "feasible-worst",
    "failure-actual-dwell",
    "failure-joint-velocity",
    "failure-generated-dwell",
)
"""The six inherited configurations of the frozen panel; the size projection replicates the fixture over them."""
CONFIGURATION = "feasible-best"
"""The fixture configuration the focused tests fit (the smallest inherited reservoir)."""
PAIR_ASSIGNMENTS = ("D01", "D02")


def _entries(fixture: ManualFixture, *, assignments: tuple[str, ...] = PAIR_ASSIGNMENTS) -> tuple[StudyModel, ...]:
    """The ``S`` and ``R10`` entries of one configuration over ``assignments``, in manifest order."""
    wanted = {(CONFIGURATION, f"{arm}/{name}") for arm in NUMERICAL_ARMS for name in assignments}
    return tuple(e for e in fixture.manifest.entries if (e.configuration, e.arm.label) in wanted)


def _refitter(fixture: ManualFixture) -> manual_numerics.Refitter:
    """Run the fresh refit in this process (the real subprocess has its own test)."""

    def refit(identity: str, parent_identity: str) -> FreshRefit:
        return fresh_refit(
            identity,
            store=fixture.store,
            root=fixture.root,
            parent_identity=parent_identity,
            execution=fixture.execution,
        )

    return refit


@pytest.fixture(scope="module")
def validation(manual_fixture: ManualFixture) -> ManualNumericalValidation:
    """The validation of four fits (``S`` and ``R10`` over two demonstrations) of one configuration."""
    return run_validation(
        _entries(manual_fixture),
        manual_fixture.inputs,
        store=manual_fixture.store,
        execution=manual_fixture.execution,
        refit=_refitter(manual_fixture),
        manifest_file="docs/experiments/task_1a_manual_demonstration/study_manifest_v1.json",
        manifest_sha256="d" * 64,
        provenance=manual_fixture.provenance,
        now=manual_fixture.now,
        log=lambda _message: None,
    )


# --- tolerances and the weighted ridge system -----------------------------------------------


def test_the_frozen_tolerances_refuse_any_other_triple() -> None:
    """The predecessor's prediction and residual bounds are prescribed by the manual plan; nothing else loads."""
    assert Tolerances() == TOLERANCES
    assert (TOLERANCES.prediction_atol_rad, TOLERANCES.prediction_rtol) == (1e-8, 1e-8)
    assert TOLERANCES.residual_max == 1e-10
    for changes in (
        {"prediction_atol_rad": 1e-6},
        {"prediction_rtol": 1e-6},
        {"residual_max": 1e-8},
        {"prediction_atol_rad": 1e-10},
    ):
        with pytest.raises(ValueError, match="prescribed by the manual plan"):
            dataclasses.replace(TOLERANCES, **changes)
    assert NUMERICAL_ARMS == ("S", "R10")
    assert COMPARISON_PAIR == ("R10", "S")
    assert PROBE_BANKS == ("manual", "contractive")


def test_weighted_normal_system_matches_an_independently_scaled_solve() -> None:
    """``A = Xw' Xw + alpha I`` on the sqrt-weighted design, with the bias column appended once, not twice."""
    rng = np.random.default_rng(17)
    rows, neurons, dof = 240, 12, 2
    states = rng.standard_normal((rows, neurons))
    targets = rng.standard_normal((rows, dof))
    row_weights = np.concatenate([np.full(140, 400.0 / 140.0), np.full(100, 400.0 / 100.0)])
    alpha = 0.25

    # Independently assembled: scale the whole row, its ones entry, and its target by sqrt(w).
    scale = np.sqrt(row_weights)[:, None]
    design = np.hstack([states, np.ones((rows, 1))]) * scale
    normal = design.T @ design + alpha * np.eye(neurons + 1)
    rhs = design.T @ (targets * scale)
    exact = np.linalg.solve(normal, rhs)

    diagnostic = weighted_normal_equations(states, targets, row_weights, exact, alpha=alpha)
    assert diagnostic.rows == rows
    assert diagnostic.columns == neurons + 1  # the explicit bias column is appended exactly once
    assert diagnostic.alpha == alpha
    assert diagnostic.within
    assert diagnostic.residual < 1e-14
    assert diagnostic.cond2 == pytest.approx(diagnostic.eigenvalue_max / diagnostic.eigenvalue_min)
    assert diagnostic.a_norm == pytest.approx(float(np.linalg.norm(normal)))
    assert diagnostic.b_norm == pytest.approx(float(np.linalg.norm(rhs)))

    # The unweighted system is a different problem: using it would hide the weighting entirely.
    unweighted = np.hstack([states, np.ones((rows, 1))])
    plain = unweighted.T @ unweighted + alpha * np.eye(neurons + 1)
    assert not np.allclose(normal, plain)
    assert not weighted_normal_equations(states, targets, row_weights, np.linalg.solve(plain, rhs), alpha=alpha).within

    perturbed = weighted_normal_equations(states, targets, row_weights, exact * (1.0 + 1e-6), alpha=alpha)
    assert not perturbed.within
    assert perturbed.residual > TOLERANCES.residual_max
    zero = weighted_normal_equations(np.zeros((4, 3)), np.zeros((4, 1)), np.ones(4), np.zeros((4, 1)), alpha=0.0)
    assert (zero.residual, zero.within, zero.cond2) == (0.0, True, float("inf"))
    with pytest.raises(ValueError, match="do not match"):
        weighted_normal_equations(states, targets, row_weights, exact[:-1], alpha=alpha)
    with pytest.raises(ValueError, match="one positive weight per row"):
        weighted_normal_equations(states, targets, np.zeros(rows), exact, alpha=alpha)
    with pytest.raises(ValueError, match="within contradicts"):
        dataclasses.replace(diagnostic, within=False)


# --- the fit cache --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "content"),
    [
        (WEIGHTS_FILE, b"this is not an npy file at all"),
        (WEIGHTS_FILE, b"\x93NUMPY\x01\x00truncated"),
        (FIT_FILE, b'{"identity": "trunca'),
        (RECIPE_FILE, b"[not valid toml"),
    ],
)
def test_a_cache_payload_that_cannot_be_read_is_an_integrity_failure(
    manual_fixture: ManualFixture, name: str, content: bytes
) -> None:
    """A payload that cannot be decoded is a fault in the store, exactly like one whose digest disagrees.

    The distinction matters to a caller: a reader that saw a plain
    ``ValueError`` here would be free to treat a truncated cache as a verdict
    on the candidate it was cached for, spend the trial and move on.
    """
    f = manual_fixture
    fits = ManualFitStore(f.store)
    entry = f.manifest.entry(CONFIGURATION, "S/D04")
    record = fits.fit_or_load(entry, f.inputs, now=f.now).record
    path = fits.directory(record.identity) / name
    original = path.read_bytes()
    path.write_bytes(content)
    try:
        with pytest.raises(EvidenceIntegrityError, match=r"cannot be read|digest"):
            _read_payload(fits, record, name)
    finally:
        path.write_bytes(original)


def _read_payload(fits: ManualFitStore, record: ManualFitRecord, name: str) -> object:
    """Read one cached payload through the verified reader that serves it."""
    if name == WEIGHTS_FILE:
        return fits.read_weights(record)
    if name == FIT_FILE:
        return fits.read_record(record.identity)
    return fits.read_recipe(record)


def test_the_fit_cache_serves_verifies_and_refuses_a_stale_fit(manual_fixture: ManualFixture) -> None:
    """A miss fits and writes recipe, record, and weights under its own prefix; a hit must refit bitwise."""
    f = manual_fixture
    fits = ManualFitStore(f.store)
    entry = f.manifest.entry(CONFIGURATION, "S/D03")
    identity = entry.fit_identity
    assert str(cache_uri(identity)) == f"armrc://models/task_1a_manual_v1/{identity}"
    assert not fits.exists(identity)

    first = fits.fit_or_load(entry, f.inputs, now=f.now)
    assert first.cache_hit is False
    assert first.record.identity == identity
    assert first.record.execution_identity == f.execution.identity
    assert first.record.configuration == CONFIGURATION
    assert first.record.arm == entry.arm
    assert len(first.record.episodes) == 1
    directory = fits.directory(identity)
    assert {p.name for p in directory.iterdir()} == {RECIPE_FILE, FIT_FILE, WEIGHTS_FILE}
    assert load_recipe(directory / RECIPE_FILE) == first.recipe
    assert recipe_text_of(first.recipe) == (directory / RECIPE_FILE).read_text(encoding="utf-8")
    assert from_mapping(json.loads((directory / FIT_FILE).read_text(encoding="utf-8")), ManualFitRecord) == first.record

    second = fits.fit_or_load(entry, f.inputs, now=f.now)
    assert second.cache_hit is True
    assert second.record == first.record
    assert array_digest(second.weights) == first.record.weights_sha256

    with pytest.raises(FileExistsError, match="immutable"):
        fits.write(first.record, recipe_text_of(first.recipe), first.weights)
    np.save(directory / WEIGHTS_FILE, first.weights + 1e-9)
    with pytest.raises(ValueError, match="recorded digest"):
        fits.read_weights(first.record)
    tampered = dataclasses.replace(first.record, weights_sha256=array_digest(first.weights + 1e-9))
    (directory / FIT_FILE).write_text(json.dumps(to_mapping(tampered)), encoding="utf-8")
    with pytest.raises(ValueError, match="did not reproduce its weights bitwise"):
        fits.fit_or_load(entry, f.inputs, now=f.now)
    np.save(directory / WEIGHTS_FILE, first.weights)
    (directory / FIT_FILE).write_text(json.dumps(to_mapping(first.record)), encoding="utf-8")
    assert fits.fit_or_load(entry, f.inputs, now=f.now).cache_hit

    with (directory / RECIPE_FILE).open("a", encoding="utf-8") as handle:
        handle.write("# tampered\n")
    with pytest.raises(ValueError, match="recorded digest"):
        fits.read_recipe(first.record)
    (directory / RECIPE_FILE).write_text(recipe_text_of(first.recipe), encoding="utf-8")
    assert fits.read_recipe(first.record) == first.recipe

    misfiled = fits.directory("0" * 64, mode="write")
    misfiled.mkdir(parents=True, exist_ok=True)
    (misfiled / FIT_FILE).write_text(json.dumps(to_mapping(first.record)), encoding="utf-8")
    with pytest.raises(ValueError, match="carries the identity"):
        fits.read_record("0" * 64)

    with pytest.raises(ValueError, match="64 lowercase hex"):
        cache_uri("nope")
    with pytest.raises(ValueError, match="one identity per training episode"):
        dataclasses.replace(first.record, episodes=())


def _identity(source: str, **changes: object) -> EpisodeIdentity:
    """One synthetic episode identity of a 200-row recording behind a 25-row warm-up."""
    fields: dict[str, object] = {
        "source": source,
        "rows": 225,
        "washout_rows": 25,
        "loss_rows": 200,
        "weight": 2.0,
        "inputs_sha256": "1" * 64,
        "targets_sha256": "2" * 64,
        "mask_sha256": "3" * 64,
        "states_sha256": "4" * 64,
    }
    return EpisodeIdentity(**{**fields, **changes})  # type: ignore[arg-type]


def _fit_report(sources: tuple[str, ...]) -> FitReport:
    """A weighted fit report over ``sources``, as a schema 3 recipe records one."""
    return FitReport(
        episodes=sources,
        loss_rows=200 * len(sources),
        washout_rows=25 * len(sources),
        rmse_per_joint=(0.01, 0.02),
        rmse=0.015,
        constant_rmse=0.5,
        max_abs_error=0.04,
        episode_loss_rows=(200,) * len(sources),
        episode_weights=(2.0,) * len(sources),
    )


def _record(**changes: object) -> ManualFitRecord:
    """A synthetic cache record of a two-episode duplication control."""
    sources = ("processed-20260916-a00000000000", "processed-20260916-a00000000000#copy-001")
    episodes = tuple(_identity(source) for source in sources)
    fields: dict[str, object] = {
        "identity": "a" * 64,
        "configuration": "feasible-best",
        "source_trial": 17,
        "arm": ManualArmSpec("R10", "D01"),
        "recipe_sha256": "b" * 64,
        "fit": _fit_report(sources),
        "weights_sha256": "c" * 64,
        "weights_shape": (101, 2),
        "episodes": episodes,
        "states_sha256": states_witness(episodes),
        "execution_identity": "d" * 64,
        "rclib": RclibIdentity("0.1.0", "e" * 40),
        "fit_seconds": 0.25,
        "created_at": "2026-09-16T12:00:00+00:00",
    }
    return ManualFitRecord(**{**fields, **changes})  # type: ignore[arg-type]


def test_cache_records_refuse_an_inconsistent_episode_or_witness() -> None:
    """The record's own fields are checked, not trusted: counts, order, weights, digests, and the state witness."""
    record = _record()
    assert record.label == "feasible-best/R10/D01"
    assert record.schema_version == 1

    with pytest.raises(ValueError, match="warm-up plus"):
        _identity("x", rows=224)
    with pytest.raises(ValueError, match="at least one row"):
        _identity("x", loss_rows=0, rows=25, washout_rows=25)
    with pytest.raises(ValueError, match="weight must be positive"):
        _identity("x", weight=0.0)
    with pytest.raises(ValueError, match="targets_sha256"):
        _identity("x", targets_sha256="short")

    with pytest.raises(ValueError, match="unsupported fit schema_version"):
        dataclasses.replace(record, schema_version=2)
    with pytest.raises(ValueError, match="one identity per training episode"):
        dataclasses.replace(record, episodes=record.episodes[:1])
    reordered = (record.episodes[1], record.episodes[0])
    with pytest.raises(ValueError, match="training order"):
        dataclasses.replace(record, episodes=reordered, states_sha256=states_witness(reordered))
    with pytest.raises(ValueError, match="does not witness"):
        dataclasses.replace(record, states_sha256="f" * 64)
    with pytest.raises(ValueError, match="positive 2-D shape"):
        dataclasses.replace(record, weights_shape=(101, 0))
    with pytest.raises(ValueError, match="fit_seconds"):
        dataclasses.replace(record, fit_seconds=float("inf"))
    with pytest.raises(ValueError, match="non-negative source trial"):
        dataclasses.replace(record, source_trial=-1)
    with pytest.raises(ValueError, match="recipe_sha256"):
        dataclasses.replace(record, recipe_sha256="short")

    # A singleton has nothing to compare, while the two copies above repeat each other exactly.
    verdict = manual_numerics.repeat_identity(record)
    assert verdict is not None
    assert verdict.identical
    assert manual_numerics.repeat_identity(dataclasses.replace(record, arm=ManualArmSpec("S", "D01"))) is None


def test_fit_inputs_refuse_a_library_or_payload_the_study_was_not_frozen_with(
    manual_fixture: ManualFixture,
) -> None:
    """The installed rclib and every locked payload must be the ones the manifest's identities were hashed with."""
    f = manual_fixture
    with pytest.raises(ValueError, match="is not the manifest's"):
        dataclasses.replace(f.inputs, rclib=RclibIdentity("0.0.1", "f" * 40))
    with pytest.raises(ValueError, match="samples are missing"):
        dataclasses.replace(f.inputs, samples={})


def test_literal_copies_share_inputs_targets_weights_masks_and_harvested_states(
    manual_fixture: ManualFixture,
) -> None:
    """Every copy of a ``R10`` fit repeats its parent bitwise: the control differs only in episode count."""
    f = manual_fixture
    entry = f.manifest.entry(CONFIGURATION, "R10/D01")
    cached = ManualFitStore(f.store).fit_or_load(entry, f.inputs, now=f.now)
    identities = cached.record.episodes
    assert len(identities) == 10
    for name in ("inputs_sha256", "targets_sha256", "mask_sha256", "states_sha256"):
        assert len({getattr(episode, name) for episode in identities}) == 1, name
    assert len({episode.weight for episode in identities}) == 1
    assert len({episode.washout_rows for episode in identities}) == 1
    assert len({episode.loss_rows for episode in identities}) == 1
    assert identities[0].source == f.manifest.sources["D01"].artifact_id
    assert identities[1].source.endswith("#copy-001")

    repeat = manual_numerics.repeat_identity(cached.record)
    assert repeat is not None
    assert repeat.identical
    assert repeat.arm == "R10/D01"
    assert repeat.episodes == 10
    broken = dataclasses.replace(identities[1], targets_sha256="ab" * 32)
    changed = dataclasses.replace(cached.record, episodes=(identities[0], broken, *identities[2:]))
    altered = manual_numerics.repeat_identity(changed)
    assert altered is not None
    assert not altered.identical
    with pytest.raises(ValueError, match="identical contradicts"):
        dataclasses.replace(repeat, identical=False)

    # The reservoir is driven from a reset for every episode, so the copies are bitwise equal in this process too.
    harvested = [harvest_episode(cached.model, episode) for episode in cached.episodes]
    assert len({array_digest(h.states) for h in harvested}) == 1

    single = ManualFitStore(f.store).fit_or_load(f.manifest.entry(CONFIGURATION, "S/D01"), f.inputs, now=f.now)
    assert manual_numerics.repeat_identity(single.record) is None  # a singleton has no copies to compare


# --- probes ---------------------------------------------------------------------------------


def test_probes_stack_the_ten_manual_trajectories_and_never_enter_the_fit(manual_fixture: ManualFixture) -> None:
    """Every readout is probed on all ten demonstrations; nine of them are probe-only for a singleton."""
    f = manual_fixture
    entry = f.manifest.entry(CONFIGURATION, "S/D01")
    cached = ManualFitStore(f.store).fit_or_load(entry, f.inputs, now=f.now)
    probes = build_probes(cached.model, cached.recipe.encoder(), entry, f.inputs)
    (bank,) = probes.banks
    assert bank.bank == "manual"
    assert bank.episodes == 10
    assert bank.episode_labels == tuple(f"D{i:02d}" for i in range(1, 11))
    assert bank.first_row == 0
    assert bank.rows == probes.states.shape[0] == sum(f.manifest.loss_rows[n] for n in bank.episode_labels)
    assert len(set(bank.episode_rows)) > 1  # the recordings have deliberately unequal lengths
    assert bank.states_sha256 == array_digest(probes.states)
    assert probes.bank_of(0) == "manual"
    assert probes.bank_of(probes.states.shape[0] - 1) == "manual"
    with pytest.raises(ValueError, match="precedes"):
        probes.bank_of(-1)

    # The fit trained on D01 alone; the other nine probe trajectories never entered it or the transform.
    trained = {d.artifact_id for d in cached.recipe.datasets}
    assert trained == {f.manifest.sources["D01"].artifact_id}
    probe_only = {f.manifest.sources[name].artifact_id for name in bank.episode_labels} - trained
    assert len(probe_only) == 9
    assert not probe_only & {episode.source.split("#")[0] for episode in cached.record.episodes}
    assert cached.recipe.transform_source is not None
    assert cached.recipe.transform.derived_from == (cached.recipe.transform_source.artifact_id,)
    assert cached.recipe.transform_source.artifact_id not in probe_only | trained

    prediction = predict_probes(cached.model, probes)
    assert prediction.shape == (probes.states.shape[0], 2)
    weights = cached.model.readout_weights()
    assert np.allclose(prediction, probes.states @ weights[:-1] + weights[-1], atol=1e-12, rtol=0.0)
    with pytest.raises(ValueError, match="malformed probe bank"):
        dataclasses.replace(bank, bank="other")
    with pytest.raises(ValueError, match="one label and one row count"):
        dataclasses.replace(bank, episode_labels=())


def test_a_contractive_fit_probes_its_own_bank_beside_the_ten_demonstrations(
    manual_fixture: ManualFixture,
) -> None:
    """The bank a fit trains on is probed with it; the ten recorded trajectories stay the shared matched block."""
    f = manual_fixture
    entry = f.manifest.entry(CONFIGURATION, "C10/D01")
    recipe, model, _episodes = fit_entry(entry, f.inputs)
    probes = build_probes(model, recipe.encoder(), entry, f.inputs)
    assert [bank.bank for bank in probes.banks] == list(PROBE_BANKS)
    manual, contractive = probes.banks
    assert manual.episodes == 10
    assert contractive.episodes == 9
    assert contractive.first_row == manual.rows
    assert contractive.episode_labels[0].endswith("#contractive-001")
    assert probes.bank_of(contractive.first_row) == "contractive"
    assert probes.matched.shape == (manual.rows, model.n_neurons)
    assert array_digest(probes.matched) == manual.states_sha256


# --- comparisons ----------------------------------------------------------------------------


def test_the_ten_copy_control_agrees_with_its_singleton_on_the_fixed_probes(
    validation: ManualNumericalValidation,
) -> None:
    """``R10_i`` at ``10 alpha_0`` is the ridge-equivalent of ``S_i`` at ``alpha_0`` within the frozen tolerance."""
    v = validation
    assert v.n_comparisons == len(PAIR_ASSIGNMENTS)
    assert v.n_comparisons_passed == v.n_comparisons
    for comparison in v.comparisons:
        assert comparison.configuration == CONFIGURATION
        assert comparison.candidate.startswith("R10/")
        assert comparison.reference.startswith("S/")
        assert comparison.candidate.split("/")[1] == comparison.reference.split("/")[1] == comparison.assignment
        assert comparison.rows > 0
        assert comparison.passed
        bound = TOLERANCES.prediction_atol_rad + TOLERANCES.prediction_rtol * max(1.0, comparison.max_abs)
        assert comparison.max_abs <= bound
        assert comparison.worst_bank == "manual"
        assert comparison.coefficient_fro_rel < 1e-6
    assert v.all_passed
    assert v.n_fits == 2 * len(PAIR_ASSIGNMENTS)
    assert v.n_residuals_within == v.n_fits
    assert v.n_repeat_identities_identical == len(PAIR_ASSIGNMENTS)
    assert v.n_fresh_refits_passed == v.n_fits
    assert v.probe_states_identical_across_arms == {CONFIGURATION: True}
    for summary in v.fits:
        assert summary.weighting_matches_accounting
        assert summary.training_sources_expected
        assert summary.transform_isolated
        assert summary.normal.within
        assert summary.accessor_max_abs_diff < 1e-12
        assert summary.loss_rows == summary.loss_rows_expected
    with pytest.raises(ValueError, match="approved comparison"):
        dataclasses.replace(v.comparisons[0], candidate="M10")


# --- fresh refits ---------------------------------------------------------------------------


def test_fit_inputs_refuse_an_environment_the_study_was_not_frozen_in(manual_fixture: ManualFixture) -> None:
    """Every fit identity hashes the execution environment, so the inputs refuse a foreign one outright (C10)."""
    f = manual_fixture
    with pytest.raises(ValueError, match="never this study's"):
        dataclasses.replace(f.inputs, execution_identity="e" * 64)
    with pytest.raises(ValueError, match="execution_identity"):
        dataclasses.replace(f.inputs, execution_identity="short")


def test_a_foreign_execution_identity_is_a_reported_failed_refit(manual_fixture: ManualFixture) -> None:
    """A worker whose environment differs from the parent's is recorded as a failure, never raised."""
    f = manual_fixture
    entry = f.manifest.entry(CONFIGURATION, "S/D01")
    cached = ManualFitStore(f.store).fit_or_load(entry, f.inputs, now=f.now)
    foreign = fresh_refit(
        cached.record.identity,
        store=f.store,
        root=f.root,
        parent_identity="f" * 64,
        execution=f.execution,
    )
    assert not foreign.passed
    assert not foreign.environment_match
    assert foreign.weights_bitwise_equal
    assert foreign.states_bitwise_equal
    assert foreign.fit_report_equal
    assert foreign.worker_execution_identity == f.execution.identity  # recorded because it differs from the parent
    with pytest.raises(ValueError, match="passed contradicts"):
        dataclasses.replace(foreign, passed=True)
    with pytest.raises(ValueError, match="worker_execution_identity"):
        dataclasses.replace(foreign, worker_execution_identity="short")


def test_a_changed_cache_record_is_a_reported_failed_refit(manual_fixture: ManualFixture) -> None:
    """The refit is computed before the cache is read, so a tampered record fails instead of being served."""
    f = manual_fixture
    entry = f.manifest.entry(CONFIGURATION, "S/D02")
    fits = ManualFitStore(f.store)
    cached = fits.fit_or_load(entry, f.inputs, now=f.now)
    directory = fits.directory(cached.record.identity)
    tampered = dataclasses.replace(
        cached.record,
        weights_sha256=array_digest(cached.weights + 1e-9),
        fit=dataclasses.replace(cached.record.fit, rmse=cached.record.fit.rmse + 1e-9),
    )
    np.save(directory / WEIGHTS_FILE, cached.weights + 1e-9)
    (directory / FIT_FILE).write_text(json.dumps(to_mapping(tampered)), encoding="utf-8")
    changed = fresh_refit(
        cached.record.identity,
        store=f.store,
        root=f.root,
        parent_identity=f.execution.identity,
        execution=f.execution,
    )
    assert changed.environment_match
    assert not changed.weights_bitwise_equal
    assert not changed.fit_report_equal
    assert changed.max_abs_weight_diff == pytest.approx(1e-9, rel=1e-3)
    assert not changed.passed
    np.save(directory / WEIGHTS_FILE, cached.weights)
    (directory / FIT_FILE).write_text(json.dumps(to_mapping(cached.record)), encoding="utf-8")


def test_the_worker_refits_in_a_fresh_interpreter(manual_fixture: ManualFixture, tmp_path: Path) -> None:
    """A new pinned process rebuilds the fit from the recipe and the digest-verified payloads alone.

    Bitwise reproduction holds within one execution environment, so this needs
    the canonical launcher (``python -m arm_rc_ctrl.execution run --policy
    p-cores -- ...``): on a hybrid CPU an unpinned worker may land on another
    core type and differ by an ulp, which is the recorded C10 behavior rather
    than a defect of the fit.
    """
    f = manual_fixture
    entry = f.manifest.entry(CONFIGURATION, "R10/D02")
    cached = ManualFitStore(f.store).fit_or_load(entry, f.inputs, now=f.now)
    result = refit_in_subprocess(
        cached.record.identity,
        root=f.root,
        parent_identity=f.execution.identity,
        output=tmp_path / "refit.json",
        env=f.env,
    )
    assert result.passed
    assert result.identity == cached.record.identity
    assert result.worker_execution_identity is None  # the worker ran in the parent's own environment
    assert result.seconds > 0.0


# --- the record -----------------------------------------------------------------------------


def test_the_record_round_trips_and_re_derives_its_counts(
    validation: ManualNumericalValidation, tmp_path: Path
) -> None:
    """Counts and the overall decision rebuild from the retained records; an edited count never loads."""
    v = validation
    file = tmp_path / "numerical_validation.json"
    file.write_text(validation_to_json(v) + "\n", encoding="utf-8")
    assert load_validation(file) == v
    assert isinstance(v.fits[0].normal, NormalEquations)
    assert isinstance(v.comparisons[0], Comparison)
    assert isinstance(v.repeat_identities[0], RepeatIdentity)
    assert isinstance(v.probes[CONFIGURATION][0], ProbeBank)
    with pytest.raises(ValueError, match="recorded counts"):
        dataclasses.replace(v, n_fits=v.n_fits - 1)
    with pytest.raises(ValueError, match="all_passed contradicts"):
        dataclasses.replace(v, all_passed=False)
    with pytest.raises(ValueError, match="exactly one fresh-process refit"):
        dataclasses.replace(v, fresh_refits=v.fresh_refits[:-1], n_fresh_refits_passed=v.n_fits - 1)
    with pytest.raises(ValueError, match="unsupported numerics schema_version"):
        dataclasses.replace(v, schema_version=2)
    with pytest.raises(ValueError, match="study_manifest_sha256"):
        dataclasses.replace(v, study_manifest_sha256="short")
    stray = dataclasses.replace(v.fresh_refits[0], worker_execution_identity=v.execution.identity)
    with pytest.raises(ValueError, match="foreign worker environment"):
        dataclasses.replace(v, fresh_refits=(stray, *v.fresh_refits[1:]))
    assert v.fits[0].normal.denominator == pytest.approx(
        v.fits[0].normal.a_norm * v.fits[0].normal.w_norm + v.fits[0].normal.b_norm
    )
    with pytest.raises(ValueError, match="cond2 contradicts"):
        dataclasses.replace(v.fits[0].normal, cond2=1.0)


def test_a_retained_failure_keeps_all_passed_false_and_stays_visible(
    validation: ManualNumericalValidation,
) -> None:
    """A failed comparison or refit is kept with its figures and rendered loudly; nothing is repaired."""
    v = validation
    failing = dataclasses.replace(v.comparisons[0], passed=False, max_abs=1e-3, max_rel=1e-2)
    refit = dataclasses.replace(v.fresh_refits[0], weights_bitwise_equal=False, max_abs_weight_diff=1e-9, passed=False)
    retained = dataclasses.replace(
        v,
        comparisons=(failing, *v.comparisons[1:]),
        fresh_refits=(refit, *v.fresh_refits[1:]),
        n_comparisons_passed=v.n_comparisons_passed - 1,
        n_fresh_refits_passed=v.n_fresh_refits_passed - 1,
        all_passed=False,
    )
    assert not retained.all_passed
    rendered = render_validation_markdown(retained)
    assert "FAILURES RETAINED" in rendered
    assert "**FAIL**" in rendered
    assert "**DIFFER**" in rendered
    assert "1.000e-03" in rendered
    passing = render_validation_markdown(v)
    assert "all checks passed" in passing
    assert "FAIL" not in passing
    assert CONFIGURATION in passing
    assert "R10/D01" in passing


def _full_scope(validation: ManualNumericalValidation) -> ManualNumericalValidation:
    """Replicate the fixture's records to the whole approved scope: 120 fits over six configurations."""
    configurations = FULL_CONFIGURATIONS
    fits: list[object] = []
    comparisons: list[object] = []
    repeats: list[object] = []
    refits: list[object] = []
    probes = dict.fromkeys(configurations, validation.probes[CONFIGURATION])
    identical = dict.fromkeys(configurations, True)
    index = 0
    for label in configurations:
        for assignment in (f"D{i:02d}" for i in range(1, 11)):
            for arm in NUMERICAL_ARMS:
                index += 1
                identity = f"{index:064x}"
                template = next(f for f in validation.fits if f.arm.startswith(arm))
                fits.append(
                    dataclasses.replace(template, configuration=label, arm=f"{arm}/{assignment}", identity=identity)
                )
                refits.append(dataclasses.replace(validation.fresh_refits[0], identity=identity))
                if arm == "R10":
                    repeats.append(
                        dataclasses.replace(
                            validation.repeat_identities[0], configuration=label, arm=f"{arm}/{assignment}"
                        )
                    )
                    comparisons.append(
                        dataclasses.replace(
                            validation.comparisons[0],
                            configuration=label,
                            assignment=assignment,
                            candidate=f"R10/{assignment}",
                            reference=f"S/{assignment}",
                        )
                    )
    return dataclasses.replace(
        validation,
        configurations=configurations,
        probes=probes,
        probe_states_identical_across_arms=identical,
        fits=tuple(fits),
        comparisons=tuple(comparisons),
        repeat_identities=tuple(repeats),
        fresh_refits=tuple(refits),
        n_fits=len(fits),
        n_comparisons=len(comparisons),
        n_comparisons_passed=len(comparisons),
        n_residuals_within=len(fits),
        n_weighting_matches=len(fits),
        n_repeat_identities_identical=len(repeats),
        n_fresh_refits_passed=len(fits),
    )


def test_the_full_scope_record_stays_inside_the_repository_size_limit(
    validation: ManualNumericalValidation,
) -> None:
    """The evidence of the whole approved scope is committable: entries reference the study, never copy it."""
    full = _full_scope(validation)
    assert full.n_fits == 120  # ten S and ten R10 per configuration, over the six inherited configurations
    assert full.n_comparisons == 60
    size = len(validation_to_json(full).encode("utf-8"))
    assert size < LARGE_FILE_LIMIT_BYTES, f"projected evidence {size} bytes exceeds {LARGE_FILE_LIMIT_BYTES}"


# --- the command ----------------------------------------------------------------------------


def test_numerical_entries_select_the_approved_copy_controls(manual_fixture: ManualFixture) -> None:
    """The committed scope is the ten singletons and the ten duplication controls of every configuration."""
    entries = numerical_entries(manual_fixture.manifest)
    assert len(entries) == 120
    assert {entry.arm.arm for entry in entries} == set(NUMERICAL_ARMS)
    assert [e.arm.label for e in entries[:2]] == ["S/D01", "S/D02"]


def test_the_command_guards_its_outputs_and_its_environment(
    manual_fixture: ManualFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The validate command refuses to overwrite evidence and writes both files from the study manifest."""
    f = manual_fixture
    for name, value in f.env.items():
        monkeypatch.setenv(name, value)

    def selected(manifest: StudyManifest, arms: tuple[str, ...] = NUMERICAL_ARMS) -> tuple[StudyModel, ...]:
        """The four fixture entries, in place of the committed 120-model scope."""
        del manifest, arms
        return _entries(f)

    def in_process(identity: str, **kwargs: object) -> FreshRefit:
        """Refit in this process instead of spawning a worker (the real subprocess has its own test)."""
        del kwargs
        return fresh_refit(
            identity,
            store=f.store,
            root=f.root,
            parent_identity=f.execution.identity,
            execution=f.execution,
        )

    monkeypatch.setattr(manual_numerics, "repository_root", lambda: f.root)
    monkeypatch.setattr(manual_numerics, "numerical_entries", selected)
    monkeypatch.setattr(manual_numerics, "refit_in_subprocess", in_process)
    output, markdown = tmp_path / "numerics.json", tmp_path / "numerics.md"
    argv = [
        "validate",
        "--study",
        str(f.manifest_file),
        "--output",
        str(output),
        "--markdown",
        str(markdown),
        "--workspace",
        str(tmp_path / "workspace"),
        "--exploratory",
    ]
    assert main(argv) == 0
    written = load_validation(output)
    assert written.all_passed
    assert written.n_fits == 2 * len(PAIR_ASSIGNMENTS)
    assert written.study_manifest_sha256 == manual_numerics.sha256_file(f.manifest_file)
    assert written.provenance.exploratory
    assert markdown.read_text(encoding="utf-8") == render_validation_markdown(written)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(argv)


def _foreign_fit(
    f: ManualFixture,
    entry: StudyModel,
    *,
    base_alpha: float | None = None,
    sources: dict[str, DatasetSource] | None = None,
) -> ManualFitRecord:
    """Cache a fit of a construction that is *not* ``entry``'s under ``entry``'s own expected identity.

    This is the reproduction: recipe, weights, and refit all agree with each
    other, so every check the store makes of a cached fit against its own
    record passes. Only a comparison against the frozen study entry can see
    that the construction is another one.
    """
    configuration = f.inputs.configuration(entry)
    recipe, model = recipe_for_arm(
        entry.arm,
        esn=f.inputs.base_esn(entry),
        sources=f.manifest.sources if sources is None else sources,
        samples=f.inputs.samples,
        dof=f.inputs.dof,
        task_code_dim=f.inputs.task_code_dim,
        preprocessing=f.inputs.preprocessing,
        transform=f.manifest.transform.transform,
        validation=f.manifest.validation,
        warmup_s=entry.warmup_s,
        base_alpha=configuration.base_alpha if base_alpha is None else base_alpha,
        scenario=f.inputs.scenario,
        anchor=f.manifest.anchor,
        name=entry.label,
    )
    episodes = tuple(recipe.episodes(f.inputs.samples, scenario=f.inputs.scenario))
    weights = model.readout_weights()
    text = recipe_text_of(recipe)
    record = fit_record_from(
        identity=entry.fit_identity,
        entry=entry,
        recipe_text=text,
        recipe=recipe,
        weights=weights,
        episodes=episode_identities(model, episodes, cast("tuple[float, ...]", recipe.fit.episode_weights)),
        execution_identity=f.inputs.execution_identity,
        fit_seconds=0.5,
        now=f.now,
    )
    ManualFitStore(f.store).write(record, text, weights)
    return record


def _assert_the_cache_verifies_itself(f: ManualFixture, record: ManualFitRecord) -> None:
    """Every check the store makes of a cached fit against its own record passes for ``record``."""
    fits = ManualFitStore(f.store)
    assert fits.exists(record.identity)
    assert fits.read_record(record.identity) == record
    recipe = fits.read_recipe(record)  # the stored recipe matches its recorded digest
    assert array_digest(fits.read_weights(record)) == record.weights_sha256
    model, _report = recipe.refit(f.inputs.samples, scenario=f.inputs.scenario)
    assert array_digest(model.readout_weights()) == record.weights_sha256  # and refitting reproduces them bitwise


def test_a_cached_fit_of_another_ridge_parameter_is_never_served(manual_fixture: ManualFixture) -> None:
    """A fit trained at twice the frozen regularization is refused, although it verifies against its own record."""
    f = manual_fixture
    entry = f.manifest.entry(CONFIGURATION, "S/D05")
    doubled = 2 * f.inputs.configuration(entry).base_alpha
    record = _foreign_fit(f, entry, base_alpha=doubled)
    fits = ManualFitStore(f.store)
    try:
        _assert_the_cache_verifies_itself(f, record)
        assert fits.read_recipe(record).esn.readout.alpha == doubled
        assert f.manifest.esn(entry).readout.alpha != doubled  # the entry demands alpha_0, not 2 alpha_0
        with pytest.raises(ValueError, match="esn") as excinfo:
            fits.fit_or_load(entry, f.inputs, now=f.now)
        message = str(excinfo.value)
        assert entry.label in message
        assert repr(doubled) in message  # the refusal names the construction it found
    finally:
        shutil.rmtree(fits.directory(record.identity))


def test_a_cached_fit_trained_on_another_recording_is_never_served(manual_fixture: ManualFixture) -> None:
    """The guard is not alpha-specific: a fit of the wrong demonstration is refused under the right identity."""
    f = manual_fixture
    entry = f.manifest.entry(CONFIGURATION, "S/D06")
    record = _foreign_fit(f, entry, sources={**f.manifest.sources, "D06": f.manifest.sources["D07"]})
    fits = ManualFitStore(f.store)
    try:
        _assert_the_cache_verifies_itself(f, record)
        assert [d.artifact_id for d in fits.read_recipe(record).datasets] == [f.manifest.sources["D07"].artifact_id]
        with pytest.raises(ValueError, match="datasets") as excinfo:
            fits.fit_or_load(entry, f.inputs, now=f.now)
        message = str(excinfo.value)
        assert entry.label in message
        assert f.manifest.sources["D06"].artifact_id in message  # the recording the entry actually demands
    finally:
        shutil.rmtree(fits.directory(record.identity))


def test_the_study_context_refuses_a_contractive_bank_that_does_not_regenerate(
    manual_fixture: ManualFixture, tmp_path: Path
) -> None:
    """A recorded bank digest is verified by growing the bank again, never by trusting the manifest."""
    f = manual_fixture
    assignment = "D04"
    forged = "ab" * 32
    recorded = cast("ContractiveBank", f.manifest.entry(CONFIGURATION, f"C10/{assignment}").contractive).bank_sha256
    document = json.loads(f.manifest_file.read_text(encoding="utf-8"))
    for entry in document["entries"]:
        construction = entry["contractive"]
        if construction is None or construction["assignment"] != assignment:
            continue
        construction["bank_sha256"] = forged
        model = f.manifest.entry(entry["configuration"], f"C10/{assignment}")
        entry["fit_identity"] = fit_identity(
            configuration=model.configuration,
            arm=model.arm,
            warmup_s=model.warmup_s,
            base_alpha=f.inputs.configuration(model).base_alpha,
            esn=f.manifest.esn(model),
            datasets=f.manifest.datasets(model),
            transform=f.manifest.transform.transform,
            validation=f.manifest.validation,
            anchor=f.manifest.anchor,
            rclib_commit=f.manifest.rclib.commit,
            execution_identity=f.manifest.execution.identity,
            contractive=cast("ContractiveBank", model.contractive).spec,
            bank_sha256=forged,
        )
    file = tmp_path / "study_manifest_v1.json"
    file.write_text(json.dumps(document), encoding="utf-8")
    # The document is internally consistent: every forged key re-derives from the forged digest.
    assert len(load_study(file).entries) == len(f.manifest.entries)
    with pytest.raises(ValueError, match=assignment) as excinfo:
        ManualStudyContext.load(file, store=f.store, root=f.root, execution=f.execution)
    message = str(excinfo.value)
    assert forged in message
    assert recorded in message  # both digests are named, so the failure can be diagnosed from the message alone


def test_the_study_context_binds_the_manifest_its_sources_and_the_environment(
    manual_fixture: ManualFixture,
) -> None:
    """The context verifies every recorded source file, the payloads, and the canonical environment."""
    f = manual_fixture
    context = ManualStudyContext.load(f.manifest_file, store=f.store, root=f.root, execution=f.execution)
    assert context.manifest == f.manifest
    assert context.manifest_sha256 == manual_numerics.sha256_file(f.manifest_file)
    assert set(context.inputs.samples) == {d.dataset.artifact_id for d in f.manifest.demonstrations}
    assert len(context.payloads) == 10
    assert context.inputs.dof == 2
    # The fixture is module scoped, so the change must be undone: every later test loads this same scenario.
    scenario_file = f.root / f.manifest.scenario.path
    original = scenario_file.read_text(encoding="utf-8")
    try:
        with scenario_file.open("a", encoding="utf-8") as handle:
            handle.write("# changed\n")
        with pytest.raises(ValueError, match="study manifest recorded"):
            ManualStudyContext.load(f.manifest_file, store=f.store, root=f.root, execution=f.execution)
    finally:
        scenario_file.write_text(original, encoding="utf-8")
    assert manual_numerics.sha256_file(scenario_file) == f.manifest.scenario.sha256


def _doctored_source_fit(f: ManualFixture, entry: StudyModel, assignment: str, **changes: str) -> ManualFitRecord:
    """Cache a fit whose dataset differs from the demanded one only in ``changes``, under the entry's identity."""
    demanded = f.manifest.sources[assignment]
    doctored = dataclasses.replace(demanded, **changes)
    assert doctored != demanded
    return _foreign_fit(f, entry, sources={**f.manifest.sources, assignment: doctored})


def test_a_cached_fit_whose_payload_digest_differs_late_is_never_served(manual_fixture: ManualFixture) -> None:
    """Datasets are compared whole: a payload digest agreeing only in its first twelve characters is another one."""
    f = manual_fixture
    entry = f.manifest.entry(CONFIGURATION, "S/D08")
    demanded = f.manifest.sources["D08"]
    tail = "0" * (len(demanded.payload_sha256) - 12)
    record = _doctored_source_fit(f, entry, "D08", payload_sha256=demanded.payload_sha256[:12] + tail)
    fits = ManualFitStore(f.store)
    try:
        _assert_the_cache_verifies_itself(f, record)
        assert fits.read_recipe(record).datasets[0].payload_sha256[:12] == demanded.payload_sha256[:12]
        with pytest.raises(ValueError, match="datasets") as excinfo:
            fits.fit_or_load(entry, f.inputs, now=f.now)
        assert entry.label in str(excinfo.value)
    finally:
        shutil.rmtree(fits.directory(record.identity))


def test_a_cached_fit_whose_record_path_differs_is_never_served(manual_fixture: ManualFixture) -> None:
    """The record a dataset is bound to is part of its identity, so a fit bound to another record is refused."""
    f = manual_fixture
    entry = f.manifest.entry(CONFIGURATION, "S/D09")
    record = _doctored_source_fit(f, entry, "D09", record="data/records/processed/processed-20260916-000000000000.toml")
    fits = ManualFitStore(f.store)
    try:
        _assert_the_cache_verifies_itself(f, record)
        assert fits.read_recipe(record).datasets[0].payload_sha256 == f.manifest.sources["D09"].payload_sha256
        with pytest.raises(ValueError, match="datasets") as excinfo:
            fits.fit_or_load(entry, f.inputs, now=f.now)
        assert entry.label in str(excinfo.value)
    finally:
        shutil.rmtree(fits.directory(record.identity))


def _forged_study(
    f: ManualFixture,
    tmp_path: Path,
    assignment: str,
    edit: Callable[[dict[str, object], str], ContractiveTrainingSpec | None],
) -> Path:
    """Write a study document whose ``assignment`` constructions are edited and whose keys re-derive from them."""
    document = json.loads(f.manifest_file.read_text(encoding="utf-8"))
    for entry in document["entries"]:
        construction = entry["contractive"]
        if construction is None or construction["assignment"] != assignment:
            continue
        spec = edit(construction, cast("str", entry["configuration"]))
        if spec is None:
            continue
        model = f.manifest.entry(entry["configuration"], f"C10/{assignment}")
        entry["fit_identity"] = fit_identity(
            configuration=model.configuration,
            arm=model.arm,
            warmup_s=model.warmup_s,
            base_alpha=f.inputs.configuration(model).base_alpha,
            esn=f.manifest.esn(model),
            datasets=f.manifest.datasets(model),
            transform=f.manifest.transform.transform,
            validation=f.manifest.validation,
            anchor=f.manifest.anchor,
            rclib_commit=f.manifest.rclib.commit,
            execution_identity=f.manifest.execution.identity,
            contractive=spec,
            bank_sha256=cast("str", construction["bank_sha256"]),
        )
    file = tmp_path / "study_manifest_v1.json"
    file.write_text(json.dumps(document), encoding="utf-8")
    assert len(load_study(file).entries) == len(f.manifest.entries)  # internally consistent
    return file


def test_the_study_context_verifies_every_configurations_contractive_construction(
    manual_fixture: ManualFixture, tmp_path: Path
) -> None:
    """All six configurations record the bank, so a forgery in a later one is caught, not skipped."""
    f = manual_fixture
    assignment, forged = "D04", "ab" * 32
    late = FULL_CONFIGURATIONS[-1]

    def edit(construction: dict[str, object], configuration: str) -> ContractiveTrainingSpec | None:
        if configuration != late:
            return None
        construction["bank_sha256"] = forged
        return cast("ContractiveBank", f.manifest.entry(configuration, f"C10/{assignment}").contractive).spec

    file = _forged_study(f, tmp_path, assignment, edit)
    with pytest.raises(ValueError, match=assignment) as excinfo:
        ManualStudyContext.load(file, store=f.store, root=f.root, execution=f.execution)
    assert late in str(excinfo.value)


def test_the_study_context_refuses_a_changed_contractive_dwell_onset(
    manual_fixture: ManualFixture, tmp_path: Path
) -> None:
    """The onset places the envelope and so decides the episodes; a recorded onset is verified, never trusted."""
    f = manual_fixture
    assignment = "D02"
    recorded = cast("ContractiveBank", f.manifest.entry(CONFIGURATION, f"C10/{assignment}").contractive)
    shifted = recorded.dwell_start_s + 0.01

    def edit(construction: dict[str, object], configuration: str) -> ContractiveTrainingSpec | None:
        construction["dwell_start_s"] = shifted  # the bank digest is left correct
        model = cast("ContractiveBank", f.manifest.entry(configuration, f"C10/{assignment}").contractive)
        return dataclasses.replace(model.spec, dwell_start_s=shifted)

    file = _forged_study(f, tmp_path, assignment, edit)
    with pytest.raises(ValueError, match=assignment) as excinfo:
        ManualStudyContext.load(file, store=f.store, root=f.root, execution=f.execution)
    assert repr(shifted) in str(excinfo.value) or str(shifted) in str(excinfo.value)

# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3REP-003: the fit cache, the ridge-equivalence checks, and fresh-process refits on the planar fixture."""

from __future__ import annotations

import importlib
import json
import os
import shutil
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.data.arrays import array_digest
from arm_rc_ctrl.data.derivatives import DerivativeConfig, differentiate
from arm_rc_ctrl.data.normalization import fit_normalization
from arm_rc_ctrl.data.records import (
    CANONICAL_UNITS,
    ArtifactRecord,
    Origin,
    Preprocessing,
    Scenario,
    array_specs,
    make_artifact_id,
    payload_from_store,
    write_record,
)
from arm_rc_ctrl.data.recovery import (
    TASK_PHASE_CODES,
    BaselineCheck,
    CropWindow,
    OnsetAnnotation,
    RecoveryDatasetRecord,
    TaskIntervals,
)
from arm_rc_ctrl.data.samples import SampleSet, save_samples
from arm_rc_ctrl.execution import AffinityRequest, ExecutionRecord, collect_execution
from arm_rc_ctrl.experiments import repetition_numerics
from arm_rc_ctrl.experiments.closed_loop import EstimatorSpec
from arm_rc_ctrl.experiments.esn_search import TrialPoint
from arm_rc_ctrl.experiments.recovery_search import RecoveryTrialPoint
from arm_rc_ctrl.experiments.repetition_fits import (
    FIT_FILE,
    RECIPE_FILE,
    WEIGHTS_FILE,
    FitInputs,
    FitRecord,
    FitStore,
    cache_uri,
    fit_arm,
    harvested_states,
    recipe_text_of,
)
from arm_rc_ctrl.experiments.repetition_numerics import (
    COMPARISON_PAIRS,
    PROBE_BANKS,
    TOLERANCES,
    Comparison,
    FreshRefit,
    NormalEquations,
    NumericalValidation,
    PanelContext,
    ProbeBank,
    QuantityDifference,
    Tolerances,
    build_probes,
    compare_fits,
    fresh_refit,
    load_recipe_samples,
    load_validation,
    main,
    normal_equations,
    numerical_arms,
    predict_probes,
    refit_in_subprocess,
    render_validation_markdown,
    run_validation,
    validation_to_json,
)
from arm_rc_ctrl.experiments.repetition_panel import PanelEntry
from arm_rc_ctrl.experiments.repetition_recipes import ArmSpec
from arm_rc_ctrl.provenance import ArtifactReference, collect_provenance, sha256_bytes, sha256_file
from arm_rc_ctrl.rc.esn import EsnConfig, ReadoutConfig, ReservoirConfig
from arm_rc_ctrl.rc.recipe import DatasetSource, RclibIdentity, load_recipe
from arm_rc_ctrl.rc.train import InputTransformSpec, ModelConfig
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import endpoint_positions, load_scenario
from arm_rc_ctrl.storage import ENV_VAR, StorageRoot

REPO_ROOT = repository_root()
SCENARIO_RELATIVE = Path("tests") / "fixtures" / "configs" / "planar_2dof_fixture.toml"
FIXTURE_CREATED = "2026-09-09T10:00:00+00:00"
NOW = datetime(2026, 9, 9, 12, 0, tzinfo=UTC)
N = 101
DT = 0.01
MOVE_END_S = 0.8
BANK_COUNT = 16
TASK = TaskIntervals(move=(0.0, MOVE_END_S), dwell=(MOVE_END_S, 1.0))
BASE_MODEL = ModelConfig(
    name="fixture",
    esn=EsnConfig(
        reservoir=ReservoirConfig(
            n_neurons=40, spectral_radius=0.85, sparsity=0.9, leak_rate=0.4, input_scaling=0.4, seed=23
        ),
        readout=ReadoutConfig(alpha=0.5),
    ),
    input_transform=InputTransformSpec(policy="fixed_scale", q_scale=0.3, dq_scale=4.0),
)
ENTRY = PanelEntry(
    label="feasible-best",
    source_trial=17,
    role="rank 1",
    selection="feasible rank 1 of 134",
    warmup_s=0.25,
    point=RecoveryTrialPoint(
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
    ),
    estimator=EstimatorSpec(velocity_cutoff_hz=20.0, acceleration_cutoff_hz=10.0, max_dt_ratio=3.0),
    base_alpha=0.02,
    feasible=True,
    objective=0.67,
    first_failure=None,
    reason_head=None,
)


def _planar_samples(scenario_file: Path) -> SampleSet:
    scenario = load_scenario(scenario_file)
    derivatives = DerivativeConfig(method="central")
    t = np.arange(N, dtype=np.float64) * DT
    start = np.array(scenario.task.initial_q)
    goal = np.array([0.8, 0.4])
    s = np.clip(t / MOVE_END_S, 0.0, 1.0)
    blend = s * s * (3.0 - 2.0 * s)
    q = start[None, :] + blend[:, None] * (goal - start)[None, :]
    dq, ddq = differentiate(q, DT, derivatives)
    tip = endpoint_positions(scenario, q)
    dtip, ddtip = differentiate(tip, DT, derivatives)
    phase = np.where(t < MOVE_END_S, 1, 2).astype(np.int64)
    return SampleSet(t, q, dq, ddq, tip, dtip, ddtip, np.zeros((N, 0)), phase)


@dataclass(frozen=True)
class Fixture:
    """A temporary repository root and store holding one recovery dataset, plus one panel entry to fit."""

    root: Path
    store: StorageRoot
    record: RecoveryDatasetRecord
    samples: SampleSet
    inputs: FitInputs
    entry: PanelEntry
    env: dict[str, str]
    """The environment a pinned launcher would have exported (plus the store root) for worker processes."""
    execution: ExecutionRecord

    @property
    def scenario_file(self) -> Path:
        """The scenario copy under the temporary root."""
        return self.root / SCENARIO_RELATIVE

    @property
    def payload(self) -> ArtifactReference:
        """The dataset payload reference."""
        payload = self.record.artifact.payload
        return ArtifactReference(payload.uri, payload.sha256, payload.size)


def _record(
    artifact_id: str, payload: object, samples: SampleSet, scenario_file: Path, preprocessing: Preprocessing
) -> RecoveryDatasetRecord:
    scenario = load_scenario(scenario_file)
    normalization = fit_normalization(
        samples.arrays(), ("q", "dq"), fitted_on=(artifact_id,), training_rows=np.ones(N, dtype=np.bool_)
    )
    return RecoveryDatasetRecord(
        artifact=ArtifactRecord(
            artifact_id=artifact_id,
            kind="processed",
            created_at=FIXTURE_CREATED,
            license="LicenseRef-Private",
            access="private",
            payload=payload,  # type: ignore[arg-type]
            origin=Origin(
                command="synthetic repetition fixture",
                config_sha256="2" * 64,
                project_commit="a" * 40,
                project_dirty=False,
                dependency_commits={},
                sources=("raw-20260830-2a97516c354b",),
            ),
        ),
        scenario=Scenario(
            config_path=SCENARIO_RELATIVE.as_posix(),
            config_sha256=sha256_file(scenario_file),
            robot="planar-2dof-fixture",
            task="pd-reach-fixture",
            dof=2,
            initial_q=tuple(scenario.task.initial_q),
            target=tuple(scenario.task.target),
        ),
        n_samples=samples.n_samples,
        dof=samples.dof,
        task_dim=samples.task_dim,
        task_code_dim=samples.task_code_dim,
        units=dict(CANONICAL_UNITS),
        phases=dict(TASK_PHASE_CODES),
        preprocessing=preprocessing,
        onset=OnsetAnnotation(
            kind="scripted",
            raw_artifact_id="raw-20260830-2a97516c354b",
            raw_payload_sha256="b" * 64,
            detector="programmed",
            detector_params={},
            sampling_period_s=DT,
            proposed_onset_sample=100,
            proposed_onset_s=1.0,
            confirmed_onset_sample=100,
            confirmed_onset_s=1.0,
            confirmed_by="script",
        ),
        baseline=BaselineCheck(
            q_pre=tuple(float(v) for v in samples.q[0]), tolerance_rad=0.05, max_deviation_rad=0.0, status="passed"
        ),
        crop=CropWindow(pre_roll=(0.0, 1.0), source_duration_s=2.0, task=TASK),
        q0_ref=tuple(float(v) for v in samples.q[0]),
        arrays=array_specs(samples),
        normalization=normalization,
    )


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> Fixture:
    """Build the planar recovery dataset in a temporary root/store and declare this process's affinity canonical."""
    base = tmp_path_factory.mktemp("repetition")
    root = base / "repo"
    scenario_file = root / SCENARIO_RELATIVE
    scenario_file.parent.mkdir(parents=True)
    shutil.copyfile(REPO_ROOT / SCENARIO_RELATIVE, scenario_file)
    store_root = base / "store"
    store_root.mkdir()
    store = StorageRoot(store_root, repositories=(REPO_ROOT,))
    samples = _planar_samples(scenario_file)
    staged = base / "samples.npz"
    save_samples(staged, samples)
    artifact_id = make_artifact_id("processed", FIXTURE_CREATED, sha256_file(staged))
    uri = f"armrc://processed/{artifact_id}/samples.npz"
    shutil.move(staged, store.path(uri, mode="write"))
    payload = payload_from_store(store, uri, format="samples.npz", schema_version=1)
    preprocessing = Preprocessing(
        resample_period_s=DT, smoothing="none", smoothing_params={}, derivative_method="central-difference"
    )
    record = _record(artifact_id, payload, samples, scenario_file, preprocessing)
    record_file = root / "data" / "records" / "processed" / f"{artifact_id}.toml"
    record_file.parent.mkdir(parents=True)
    write_record(record_file, record)
    importlib.import_module("rclib")  # the OpenMP probe must see the runtime the workers will also load
    request = AffinityRequest("explicit", tuple(sorted(os.sched_getaffinity(0))))
    env = {**os.environ, **request.environment(), ENV_VAR: str(store_root)}
    execution = collect_execution(
        command="python -m arm_rc_ctrl.experiments.repetition_numerics validate",
        env=env,
        effective_cpus=request.cpus,
        now=NOW,
    )
    if not execution.canonical:  # pragma: no cover - only on a multithreaded test runner
        pytest.skip("the test process's numerical runtimes are not single-threaded")
    assert record.normalization is not None
    inputs = FitInputs(
        base=BASE_MODEL,
        source=DatasetSource(artifact_id, payload.sha256, record_file.relative_to(root).as_posix()),
        samples=samples,
        dof=2,
        task_code_dim=0,
        preprocessing=preprocessing,
        normalization=record.normalization,
        scenario=load_scenario(scenario_file),
        scenario_file=scenario_file,
        root=root,
        execution_identity=execution.identity,
        rclib=RclibIdentity.current(),
    )
    return Fixture(root, store, record, samples, inputs, ENTRY, env, execution)


@pytest.fixture
def pinned_environment(fixture: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make this process look launched through the pinned launcher with the fixture's store."""
    for name, value in fixture.env.items():
        monkeypatch.setenv(name, value)


# --- fit cache ----------------------------------------------------------------------------
def test_fit_store_caches_verifies_and_serves(fixture: Fixture, tmp_path: Path) -> None:
    """A miss fits and writes recipe, record, and weights; a hit verifies digests and refits bitwise."""
    f = fixture
    fits = FitStore(f.store)
    arm = ArmSpec("absolute", "R", 16)
    identity = f.inputs.identity(f.entry, arm)
    assert not fits.exists(identity)
    first = fits.fit_or_load(f.entry, arm, f.inputs)
    assert first.cache_hit is False
    assert first.record.identity == identity
    assert first.record.arm == arm
    assert first.record.execution_identity == f.execution.identity
    assert first.record.rclib == f.inputs.rclib
    assert first.record.weights_shape == (41, 2)
    assert first.record.fit.loss_rows == 17 * 100
    assert len(first.record.state_digests) == 17
    assert len(set(first.record.state_digests)) == 1  # literal copies harvest bitwise identical states
    directory = fits.directory(identity)
    assert {p.name for p in directory.iterdir()} == {RECIPE_FILE, FIT_FILE, WEIGHTS_FILE}
    assert fits.exists(identity)
    assert fits.store is f.store
    second = fits.fit_or_load(f.entry, arm, f.inputs)
    assert second.cache_hit is True
    assert array_digest(second.weights) == first.record.weights_sha256
    assert array_digest(second.model.readout_weights()) == first.record.weights_sha256
    assert second.record == first.record
    assert second.recipe == first.recipe
    assert len(second.episodes) == 17
    # The stored recipe is a loadable recipe that binds the environment-independent identity of the fit.
    assert load_recipe(directory / RECIPE_FILE) == first.recipe
    assert recipe_text_of(first.recipe) == (directory / RECIPE_FILE).read_text(encoding="utf-8")
    # Records roundtrip strictly.
    loaded = from_mapping(json.loads((directory / FIT_FILE).read_text(encoding="utf-8")), FitRecord)
    assert loaded == first.record
    # Writes are immutable and identities are checked.
    with pytest.raises(FileExistsError, match="immutable"):
        fits.write(first.record, recipe_text_of(first.recipe), first.weights)
    with pytest.raises(ValueError, match="carries the identity"):
        _misfiled(fits, first.record)
    # Tampered weights and recipes are refused.
    np.save(directory / WEIGHTS_FILE, first.weights + 1e-9)
    with pytest.raises(ValueError, match="recorded digest"):
        fits.read_weights(first.record)
    np.save(directory / WEIGHTS_FILE, first.weights)
    with (directory / RECIPE_FILE).open("a", encoding="utf-8") as handle:
        handle.write("# tampered\n")
    with pytest.raises(ValueError, match="recorded digest"):
        fits.read_recipe(first.record)
    (directory / RECIPE_FILE).write_text(recipe_text_of(first.recipe), encoding="utf-8")
    assert fits.read_recipe(first.record) == first.recipe
    # A staged directory left behind by an interrupted write is replaced, never merged.
    other = f.inputs.identity(f.entry, ArmSpec("residual", "S"))
    staging = directory.parent / f".staging-{other}"
    staging.mkdir()
    (staging / "junk").write_text("x", encoding="utf-8")
    recipe, model, episodes = fit_arm(f.entry, ArmSpec("residual", "S"), f.inputs)
    record = replace(
        first.record,
        identity=other,
        arm=ArmSpec("residual", "S"),
        fit=recipe.fit,
        recipe_sha256=sha256_bytes(recipe_text_of(recipe).encode("utf-8")),
        weights_sha256=array_digest(model.readout_weights()),
        state_digests=tuple(array_digest(s) for s in harvested_states(model, episodes)),
    )
    final = fits.write(record, recipe_text_of(recipe), model.readout_weights())
    assert not staging.exists()
    assert {p.name for p in final.iterdir()} == {RECIPE_FILE, FIT_FILE, WEIGHTS_FILE}
    assert fits.fit_or_load(f.entry, ArmSpec("residual", "S"), f.inputs).cache_hit
    del tmp_path


def _misfiled(fits: FitStore, record: FitRecord) -> None:
    """Write a record under a directory whose name is not its identity, then read it back."""
    wrong = "0" * 64
    directory = fits.directory(wrong, mode="write")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / FIT_FILE).write_text(json.dumps(to_mapping(record)), encoding="utf-8")
    fits.read_record(wrong)


def test_cache_hit_refuses_a_fit_that_no_longer_reproduces(fixture: Fixture) -> None:
    """A cached fit whose refit in the same environment differs bitwise is refused (the identity is stale)."""
    f = fixture
    fits = FitStore(f.store)
    arm = ArmSpec("residual", "R-scaled", 16)
    cached = fits.fit_or_load(f.entry, arm, f.inputs)
    directory = fits.directory(cached.record.identity)
    tampered = replace(cached.record, weights_sha256=array_digest(cached.weights + 1e-12))
    np.save(directory / WEIGHTS_FILE, cached.weights + 1e-12)
    (directory / FIT_FILE).write_text(json.dumps(to_mapping(tampered)), encoding="utf-8")
    with pytest.raises(ValueError, match="did not reproduce its weights bitwise"):
        fits.fit_or_load(f.entry, arm, f.inputs)
    np.save(directory / WEIGHTS_FILE, cached.weights)
    (directory / FIT_FILE).write_text(json.dumps(to_mapping(cached.record)), encoding="utf-8")
    assert fits.fit_or_load(f.entry, arm, f.inputs).cache_hit


def test_fit_record_and_uri_validation(fixture: Fixture) -> None:
    """Malformed digests, shapes, counts, and timings are rejected; the cache URI is fixed."""
    f = fixture
    recipe, model, episodes = fit_arm(f.entry, ArmSpec("absolute", "S"), f.inputs)
    weights = model.readout_weights()
    states = harvested_states(model, episodes)
    good = FitRecord(
        identity="a" * 64,
        panel_label="feasible-best",
        source_trial=17,
        arm=ArmSpec("absolute", "S"),
        recipe_sha256="b" * 64,
        fit=recipe.fit,
        weights_sha256=array_digest(weights),
        weights_shape=(41, 2),
        state_digests=tuple(array_digest(s) for s in states),
        execution_identity="c" * 64,
        rclib=recipe.rclib,
        fit_seconds=0.5,
        created_at="2026-09-09T12:00:00+00:00",
    )
    assert good.schema_version == 1
    with pytest.raises(ValueError, match="identity must be 64"):
        replace(good, identity="xyz")
    with pytest.raises(ValueError, match="one 64-hex digest per training episode"):
        replace(good, state_digests=())
    with pytest.raises(ValueError, match="positive 2-D shape"):
        replace(good, weights_shape=(41, 0))
    with pytest.raises(ValueError, match="fit_seconds"):
        replace(good, fit_seconds=float("inf"))
    with pytest.raises(ValueError, match="non-negative source trial"):
        replace(good, source_trial=-1)
    with pytest.raises(ValueError, match="unsupported fit schema_version"):
        replace(good, schema_version=2)
    assert str(cache_uri("a" * 64)) == f"armrc://models/task_1a_repetition_v1/{'a' * 64}"
    with pytest.raises(ValueError, match="64 lowercase hex"):
        cache_uri("nope")
    with pytest.raises(ValueError, match="execution_identity"):
        replace(f.inputs, execution_identity="short")


# --- numerics -----------------------------------------------------------------------------


def test_numerical_arms_and_tolerances_are_fixed() -> None:
    """Twenty fits per entry (S plus three arms at three counts, both formulations); D6 tolerances are immutable."""
    arms = numerical_arms()
    assert len(arms) == 20
    assert [a.label for a in arms[:4]] == [
        "absolute/S",
        "absolute/R/K17",
        "absolute/R-scaled/K17",
        "absolute/S-effective/K17",
    ]
    assert sum(1 for a in arms if a.formulation == "residual") == 10
    assert all(a.arm not in {"A-contractive", "A-non-decaying"} for a in arms)
    assert Tolerances() == TOLERANCES
    with pytest.raises(ValueError, match="approved under D6"):
        Tolerances(prediction_atol_rad=1e-6)
    assert COMPARISON_PAIRS == (("R", "S-effective"), ("R-scaled", "S"))
    assert PROBE_BANKS == ("original", "non_decaying", "contractive")


def test_normal_equations_diagnose_the_solve() -> None:
    """An exact ridge solution sits at roundoff; a perturbed one exceeds 1e-10; the all-zero problem is zero."""
    rng = np.random.default_rng(7)
    states = rng.standard_normal((300, 12))
    targets = rng.standard_normal((300, 2))
    alpha = 0.3
    x = np.hstack([states, np.ones((300, 1))])
    a = x.T @ x + alpha * np.eye(13)
    exact = np.linalg.solve(a, x.T @ targets)
    diagnostic = normal_equations(states, targets, exact, alpha=alpha)
    assert diagnostic.within
    assert diagnostic.residual < 1e-14
    assert diagnostic.rows == 300
    assert diagnostic.columns == 13
    assert diagnostic.cond2 == pytest.approx(diagnostic.eigenvalue_max / diagnostic.eigenvalue_min)
    assert diagnostic.cond2 > 1.0
    perturbed = normal_equations(states, targets, exact * (1.0 + 1e-6), alpha=alpha)
    assert not perturbed.within
    assert perturbed.residual > 1e-10
    zero = normal_equations(np.zeros((5, 3)), np.zeros((5, 1)), np.zeros((4, 1)), alpha=0.0)
    assert zero.residual == 0.0
    assert zero.within
    assert zero.cond2 == float("inf")
    with pytest.raises(ValueError, match="do not match"):
        normal_equations(states, targets, exact[:-1], alpha=alpha)
    with pytest.raises(ValueError, match="within contradicts"):
        replace(diagnostic, within=False)
    with pytest.raises(ValueError, match="all-zero problem"):
        replace(zero, numerator=1.0)


def test_probes_stack_the_original_and_both_banks_with_anchors(fixture: Fixture) -> None:
    """The probe matrix holds the task-row states of the original and both banks, with per-row provenance."""
    f = fixture
    recipe, model, _episodes = fit_arm(f.entry, ArmSpec("absolute", "S"), f.inputs)
    probes = build_probes(
        model,
        recipe.encoder(),
        f.samples,
        scenario=f.inputs.scenario,
        warmup_s=f.entry.warmup_s,
        period_s=DT,
        derivative_method="central-difference",
        task_code=f.samples.task_code,
        bank_count=BANK_COUNT,
    )
    assert [b.bank for b in probes.banks] == list(PROBE_BANKS)
    assert [b.episodes for b in probes.banks] == [1, BANK_COUNT, BANK_COUNT]
    assert all(b.rows_per_episode == N - 1 for b in probes.banks)
    assert probes.states.shape == ((1 + 2 * BANK_COUNT) * (N - 1), 40)
    assert probes.anchors.shape == ((1 + 2 * BANK_COUNT) * (N - 1), 2)
    assert [b.first_row for b in probes.banks] == [0, N - 1, (1 + BANK_COUNT) * (N - 1)]
    assert probes.banks[0].attempts_used is None
    assert probes.banks[1].attempts_used is not None
    assert probes.banks[1].attempts_used >= BANK_COUNT
    assert probes.banks[1].episode_labels[0] == "non_decaying-001"
    assert np.array_equal(probes.anchors[: N - 1], f.samples.q[:-1])  # the original's measured postures
    assert probes.bank_of(0) == "original"
    assert probes.bank_of(N - 1) == "non_decaying"
    assert probes.bank_of(probes.states.shape[0] - 1) == "contractive"
    with pytest.raises(ValueError, match="precedes"):
        probes.bank_of(-1)
    prediction = predict_probes(model, probes)
    assert prediction.shape == (probes.states.shape[0], 2)
    weights = model.readout_weights()
    assert np.allclose(prediction, probes.states @ weights[:-1] + weights[-1], atol=1e-12, rtol=0.0)
    with pytest.raises(ValueError, match="malformed probe bank"):
        replace(probes.banks[0], bank="other")
    with pytest.raises(ValueError, match="labels for"):
        replace(probes.banks[0], episode_labels=())
    with pytest.raises(ValueError, match="64 lowercase hex"):
        replace(probes.banks[0], states_sha256="x")
    with pytest.raises(ValueError, match="unsupported derivative method"):
        build_probes(
            model,
            recipe.encoder(),
            f.samples,
            scenario=f.inputs.scenario,
            warmup_s=0.25,
            period_s=DT,
            derivative_method="savitzky",
            task_code=f.samples.task_code,
            bank_count=BANK_COUNT,
        )


def _refit_in_process(f: Fixture) -> repetition_numerics.Refitter:
    def refit(identity: str, parent_identity: str) -> FreshRefit:
        return fresh_refit(identity, store=f.store, root=f.root, parent_identity=parent_identity, execution=f.execution)

    return refit


def _provenance() -> object:
    return collect_provenance({"kind": "test"}, seeds={}, exploratory=True)


@pytest.fixture(scope="module")
def validation(fixture: Fixture) -> NumericalValidation:
    """The full validation of the fixture entry at a 16-episode probe bank with in-process fresh refits."""
    f = fixture
    return run_validation(
        [f.entry],
        f.inputs,
        store=f.store,
        execution=f.execution,
        refit=_refit_in_process(f),
        manifest_file="docs/panel.json",
        manifest_sha256="d" * 64,
        dataset_payload=f.payload,
        provenance=_provenance(),  # type: ignore[arg-type]
        bank_count=BANK_COUNT,
        now=NOW,
        log=lambda _message: None,
    )


def test_run_validation_checks_every_equivalence_on_the_fixture(
    fixture: Fixture, validation: NumericalValidation
) -> None:
    """Twenty fits, twelve comparisons, ten state identities, and twenty refits; every check passes on the fixture."""
    v = validation
    assert v.n_fits == 20
    assert v.n_comparisons == 12
    assert len(v.state_identities) == 10
    assert len(v.fresh_refits) == 20
    assert v.all_passed
    assert v.n_comparisons_passed == 12
    assert v.n_residuals_within == 20
    assert v.n_state_identities_identical == 10
    assert v.n_fresh_refits_passed == 20
    assert v.probe_bank_count == BANK_COUNT
    assert v.probe_states_identical_across_formulations == {"feasible-best": True}
    assert set(v.probes) == {"feasible-best"}
    assert v.execution == fixture.execution
    assert v.dataset == fixture.inputs.source
    assert v.experiment == "task_1a_repetition_v1"
    assert all(f.copies_identical for f in v.fits)
    assert all(f.accessor_max_abs_diff < 1e-12 for f in v.fits)
    # Earlier tests of this module cached three of the arms; the rest were fitted here.
    assert {f.arm.label for f in v.fits if f.cache_hit} <= {"absolute/R/K17", "residual/S", "residual/R-scaled/K17"}
    assert sum(1 for f in v.fits if not f.cache_hit) >= 17
    labels = {f.arm.label: f for f in v.fits}
    assert labels["absolute/R/K65"].fit.loss_rows == 65 * (N - 1)
    assert labels["absolute/S-effective/K65"].solver_alpha == pytest.approx(0.02 / 65)
    assert labels["residual/R-scaled/K33"].solver_alpha == pytest.approx(0.02 * 33)
    for c in v.comparisons:
        quantities = [d.quantity for d in c.differences]
        assert quantities == (["prediction"] if c.formulation == "absolute" else ["increment", "command"])
        assert all(d.max_abs <= TOLERANCES.prediction_atol_rad + TOLERANCES.prediction_rtol for d in c.differences)
        assert all(d.rows == (1 + 2 * BANK_COUNT) * (N - 1) for d in c.differences)
        assert c.coefficient_fro_rel < 1e-6
    # R versus S-effective is the (K alpha_0)/K identity; R-scaled versus S the K-copies identity.
    assert {(c.candidate.split("/")[1], c.reference.split("/")[1]) for c in v.comparisons} == set(COMPARISON_PAIRS)
    for r in v.fresh_refits:
        assert r.environment_match
        assert r.weights_bitwise_equal
        assert r.states_bitwise_equal
        assert r.fit_report_equal
        assert r.max_abs_weight_diff == 0.0
        assert r.rmse_abs_diff == 0.0


def test_validation_roundtrips_renders_and_rederives_its_invariants(
    validation: NumericalValidation, tmp_path: Path
) -> None:
    """JSON and Markdown are complete; counts and the decision cannot be edited independently of the records."""
    v = validation
    file = tmp_path / "numerical_validation.json"
    file.write_text(validation_to_json(v) + "\n", encoding="utf-8")
    assert load_validation(file) == v
    markdown = render_validation_markdown(v)
    assert "all checks passed" in markdown
    assert "FAIL" not in markdown
    assert markdown.count("| feasible-best | absolute |") == 6  # six absolute comparison rows
    assert "absolute/R/K65" in markdown
    with pytest.raises(ValueError, match="recorded counts"):
        replace(v, n_fits=19)
    with pytest.raises(ValueError, match="all_passed contradicts"):
        replace(v, all_passed=False)
    with pytest.raises(ValueError, match="exactly one fresh-process refit"):
        replace(v, fresh_refits=v.fresh_refits[:-1], n_fresh_refits_passed=19)
    with pytest.raises(ValueError, match="unsupported numerics schema_version"):
        replace(v, schema_version=2)
    with pytest.raises(ValueError, match="panel_manifest_sha256"):
        replace(v, panel_manifest_sha256="short")
    foreign = replace(v.fits[0], execution_identity="e" * 64)
    with pytest.raises(ValueError, match="binds the validation's execution identity"):
        replace(v, fits=(foreign, *v.fits[1:]))
    # A failure is retained with its figures and rendered as such.
    failing_difference = replace(v.comparisons[0].differences[0], passed=False, max_abs=1e-3)
    failing = replace(v.comparisons[0], differences=(failing_difference,), passed=False)
    bad_refit = replace(v.fresh_refits[0], weights_bitwise_equal=False, max_abs_weight_diff=1e-9, passed=False)
    retained = replace(
        v,
        comparisons=(failing, *v.comparisons[1:]),
        fresh_refits=(bad_refit, *v.fresh_refits[1:]),
        n_comparisons_passed=v.n_comparisons_passed - 1,
        n_fresh_refits_passed=v.n_fresh_refits_passed - 1,
        all_passed=False,
    )
    rendered = render_validation_markdown(retained)
    assert "FAILURES RETAINED" in rendered
    assert "**FAIL**" in rendered
    assert "**DIFFER**" in rendered
    with pytest.raises(ValueError, match="passed contradicts"):
        replace(v.comparisons[0], passed=False)
    with pytest.raises(ValueError, match="hold the quantities"):
        replace(v.comparisons[0], formulation="residual")
    with pytest.raises(ValueError, match="passed contradicts"):
        replace(v.fresh_refits[0], passed=False)
    with pytest.raises(ValueError, match="loss rows"):
        replace(v.fits[0], loss_rows_expected=1)
    assert isinstance(v.comparisons[0], Comparison)
    assert isinstance(v.comparisons[0].differences[0], QuantityDifference)
    assert isinstance(v.fits[0].normal, NormalEquations)
    assert isinstance(v.probes["feasible-best"][0], ProbeBank)


def test_compare_fits_refuses_pairs_outside_the_plan(fixture: Fixture, validation: NumericalValidation) -> None:
    """Only R against S-effective and R-scaled against S within one formulation are comparisons."""
    f = fixture
    fits = FitStore(f.store)
    absolute = fits.fit_or_load(f.entry, ArmSpec("absolute", "R", 16), f.inputs)
    residual = fits.fit_or_load(f.entry, ArmSpec("residual", "R", 16), f.inputs)
    single = fits.fit_or_load(f.entry, ArmSpec("absolute", "S"), f.inputs)
    assert absolute.cache_hit
    assert residual.cache_hit
    assert single.cache_hit
    recipe, model, _episodes = fit_arm(f.entry, ArmSpec("absolute", "S"), f.inputs)
    probes = build_probes(
        model,
        recipe.encoder(),
        f.samples,
        scenario=f.inputs.scenario,
        warmup_s=f.entry.warmup_s,
        period_s=DT,
        derivative_method="central-difference",
        task_code=f.samples.task_code,
        bank_count=BANK_COUNT,
    )
    predictions = {c.record.identity: predict_probes(c.model, probes) for c in (absolute, residual, single)}
    with pytest.raises(ValueError, match="is not compared against"):
        compare_fits(f.entry, absolute, residual, probes, predictions)
    with pytest.raises(ValueError, match="is not compared against"):
        compare_fits(f.entry, absolute, single, probes, predictions)
    assert validation.n_fits == 20


def test_fresh_refit_in_a_real_subprocess(fixture: Fixture, tmp_path: Path) -> None:
    """A worker interpreter verifies its pinned environment, refits from recipe and dataset, and matches bitwise."""
    f = fixture
    fits = FitStore(f.store)
    cached = fits.fit_or_load(f.entry, ArmSpec("residual", "S-effective", 32), f.inputs)
    result = refit_in_subprocess(
        cached.record.identity,
        root=f.root,
        parent_identity=f.execution.identity,
        output=tmp_path / "refit.json",
        env=f.env,
    )
    assert result.passed
    assert result.worker_execution_identity == f.execution.identity
    assert result.identity == cached.record.identity
    assert result.seconds > 0.0


def test_fresh_refit_reports_a_foreign_environment_and_a_changed_fit(fixture: Fixture) -> None:
    """A parent identity that differs from the worker's, or a cache record that differs from the refit, fails."""
    f = fixture
    fits = FitStore(f.store)
    cached = fits.fit_or_load(f.entry, ArmSpec("absolute", "R-scaled", 32), f.inputs)
    foreign = fresh_refit(
        cached.record.identity, store=f.store, root=f.root, parent_identity="f" * 64, execution=f.execution
    )
    assert not foreign.passed
    assert not foreign.environment_match
    assert foreign.weights_bitwise_equal
    assert foreign.states_bitwise_equal
    assert foreign.fit_report_equal
    directory = fits.directory(cached.record.identity)
    tampered = replace(
        cached.record,
        weights_sha256=array_digest(cached.weights + 1e-9),
        fit=replace(cached.record.fit, rmse=cached.record.fit.rmse + 1e-9),
    )
    np.save(directory / WEIGHTS_FILE, cached.weights + 1e-9)
    (directory / FIT_FILE).write_text(json.dumps(to_mapping(tampered)), encoding="utf-8")
    changed = fresh_refit(
        cached.record.identity, store=f.store, root=f.root, parent_identity=f.execution.identity, execution=f.execution
    )
    assert not changed.passed
    assert changed.environment_match
    assert not changed.weights_bitwise_equal
    assert changed.max_abs_weight_diff == pytest.approx(1e-9, rel=1e-3)
    assert not changed.fit_report_equal
    assert changed.rmse_abs_diff == pytest.approx(1e-9, rel=1e-3)
    np.save(directory / WEIGHTS_FILE, cached.weights)
    (directory / FIT_FILE).write_text(json.dumps(to_mapping(cached.record)), encoding="utf-8")


def test_load_recipe_samples_binds_the_record_and_payload(fixture: Fixture) -> None:
    """The recipe's dataset resolves through its record and digest-verified payload; a foreign digest is refused."""
    f = fixture
    recipe, _model, _episodes = fit_arm(f.entry, ArmSpec("absolute", "S"), f.inputs)
    samples = load_recipe_samples(recipe, f.store, root=f.root)
    assert set(samples) == {f.inputs.source.artifact_id}
    assert samples[f.inputs.source.artifact_id].digests() == f.samples.digests()
    foreign = replace(recipe, datasets=(replace(recipe.datasets[0], payload_sha256="0" * 64),))
    with pytest.raises(ValueError, match="describes"):
        load_recipe_samples(foreign, f.store, root=f.root)


@pytest.mark.usefixtures("pinned_environment")
def test_worker_and_validate_commands(fixture: Fixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both subcommands run under the pinned environment; the validate command writes the evidence files."""
    f = fixture
    fits = FitStore(f.store)
    cached = fits.fit_or_load(f.entry, ArmSpec("absolute", "S"), f.inputs)
    output = tmp_path / "worker.json"
    argv = [
        "refit-worker",
        "--fit",
        cached.record.identity,
        "--root",
        str(f.root),
        "--parent-identity",
        f.execution.identity,
        "--output",
        str(output),
    ]
    assert main(argv) == 0
    result = from_mapping(json.loads(output.read_text(encoding="utf-8")), FreshRefit)
    assert result.passed
    # validate: the panel context and the subprocess refits are replaced by the fixture's in-process equivalents.
    manifest = tmp_path / "panel.json"
    manifest.write_text("{}", encoding="utf-8")

    def fake_load(manifest_file: Path, *, store: StorageRoot, root: Path, execution: ExecutionRecord) -> PanelContext:
        del store, root
        return PanelContext(
            manifest=_FakeManifest(entries=(f.entry,)),  # type: ignore[arg-type]
            manifest_sha256=sha256_file(manifest_file),
            inputs=replace(f.inputs, execution_identity=execution.identity),
            dataset=f.record,
            payload=f.payload,
        )

    def fake_refit(
        identity: str, *, root: Path, parent_identity: str, output: Path, python: str = "", env: object = None
    ) -> FreshRefit:
        del root, output, python, env
        return fresh_refit(identity, store=f.store, root=f.root, parent_identity=parent_identity, execution=f.execution)

    monkeypatch.setattr(PanelContext, "load", fake_load)
    monkeypatch.setattr(repetition_numerics, "refit_in_subprocess", fake_refit)
    monkeypatch.setattr(repetition_numerics, "repository_root", lambda: f.root)
    evidence, markdown = tmp_path / "numerical_validation.json", tmp_path / "numerical_validation.md"
    argv = [
        "validate",
        "--manifest",
        str(manifest),
        "--output",
        str(evidence),
        "--markdown",
        str(markdown),
        "--workspace",
        str(tmp_path / "workspace"),
        "--bank-count",
        str(BANK_COUNT),
        "--exploratory",
    ]
    assert main(argv) == 0
    written = load_validation(evidence)
    assert written.n_fits == 20
    assert written.all_passed
    assert written.provenance.exploratory
    assert written.panel_manifest_sha256 == sha256_file(manifest)
    assert markdown.read_text(encoding="utf-8") == render_validation_markdown(written)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(argv)


@dataclass(frozen=True)
class _FakeManifest:
    entries: tuple[PanelEntry, ...]


def test_panel_context_refuses_a_changed_source_file(fixture: Fixture, tmp_path: Path) -> None:
    """A configured file whose digest differs from the frozen manifest's record stops the validation."""
    docs = REPO_ROOT / "docs" / "experiments" / "task_1a_repeated_demonstration"
    manifest = docs / "panel_manifest_v1.json"
    recorded = json.loads(manifest.read_text(encoding="utf-8"))["configs"]
    root = tmp_path / "repo"
    for key in ("model_file", "scenario_file", "dataset_record_file"):
        target = root / recorded[key]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / recorded[key], target)
    with (root / recorded["model_file"]).open("a", encoding="utf-8") as handle:
        handle.write("# changed\n")
    with pytest.raises(ValueError, match="panel manifest recorded"):
        PanelContext.load(manifest, store=fixture.store, root=root, execution=fixture.execution)


def test_cache_uri_and_recipe_text(fixture: Fixture) -> None:
    """Cached recipes are loadable TOML with the project's recipe header."""
    recipe, _model, _episodes = fit_arm(fixture.entry, ArmSpec("residual", "S"), fixture.inputs)
    text = recipe_text_of(recipe)
    assert text.startswith("# Deterministic model recipe")
    path = fixture.root / "recipe.toml"
    path.write_text(text, encoding="utf-8")
    assert load_recipe(path) == recipe
    assert str(cache_uri(recipe.rclib.commit + "0" * 24)).startswith("armrc://models/task_1a_repetition_v1/")

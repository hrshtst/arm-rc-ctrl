# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-009: the no-augmentation timing vertical slice — common hold, simultaneous activation, full telemetry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arm_rc_ctrl.config import load_config
from arm_rc_ctrl.controllers.tracking import TrackerConfig
from arm_rc_ctrl.data.derivatives import DerivativeConfig, differentiate
from arm_rc_ctrl.data.normalization import fit_normalization
from arm_rc_ctrl.data.records import (
    CANONICAL_UNITS,
    ArtifactRecord,
    Origin,
    Payload,
    Preprocessing,
    Scenario,
    array_specs,
    make_artifact_id,
)
from arm_rc_ctrl.data.recovery import (
    TASK_PHASE_CODES,
    BaselineCheck,
    CropWindow,
    OnsetAnnotation,
    RecoveryDatasetRecord,
    TaskIntervals,
)
from arm_rc_ctrl.data.samples import SampleSet
from arm_rc_ctrl.experiments.disturbances import ForcePulse
from arm_rc_ctrl.experiments.recovery_slice import HeldTaskReference, run_recovery_pair
from arm_rc_ctrl.metrics.recovery import RecoveryMetricsReport
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.rc.esn import EsnConfig, ReadoutConfig, ReservoirConfig
from arm_rc_ctrl.rc.recipe import DatasetSource, ModelRecipe, TrainingSpec, create_recipe
from arm_rc_ctrl.rc.teacher_forcing import InputTransform
from arm_rc_ctrl.rc.warmup import APPROVED_WARMUPS_S, WarmupConfig
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import endpoint_positions, load_scenario
from arm_rc_ctrl.storage import StorageRoot

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
SCENARIO_FILE = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
SCENARIO = load_scenario(SCENARIO_FILE)
TRACKER = load_config(REPO_ROOT / "configs" / "controllers" / "pd.toml", TrackerConfig)
DERIVATIVES = DerivativeConfig(method="central")
RAW_SOURCE = "raw-20260830-2a97516c354b"
CREATED = "2026-09-03T10:00:00+00:00"
N = 101
DT = 0.01
TASK = TaskIntervals(move=(0.0, 0.8), dwell=(0.8, 1.0))
ESN = EsnConfig(
    reservoir=ReservoirConfig(
        n_neurons=80, spectral_radius=0.85, sparsity=0.9, leak_rate=0.4, input_scaling=0.4, seed=11
    ),
    readout=ReadoutConfig(alpha=1e-6),
)


def _samples() -> SampleSet:
    t = np.arange(N, dtype=np.float64) * DT
    start = np.array(SCENARIO.task.initial_q)
    goal = np.array([0.8, 0.4])
    s = np.clip(t / TASK.move[1], 0.0, 1.0)
    blend = s * s * (3.0 - 2.0 * s)
    q = start[None, :] + blend[:, None] * (goal - start)[None, :]
    dq, ddq = differentiate(q, DT, DERIVATIVES)
    tip = endpoint_positions(SCENARIO, q)
    dtip, ddtip = differentiate(tip, DT, DERIVATIVES)
    phase = np.where(t < TASK.dwell[0], 1, 2).astype(np.int64)
    return SampleSet(t, q, dq, ddq, tip, dtip, ddtip, np.zeros((N, 0)), phase)


def _record(samples: SampleSet, payload_sha: str) -> RecoveryDatasetRecord:
    artifact_id = make_artifact_id("processed", CREATED, payload_sha)
    normalization = fit_normalization(
        samples.arrays(),
        ("q", "dq", "ddq", "tip", "dtip", "ddtip"),
        fitted_on=(artifact_id,),
        training_rows=np.ones(samples.n_samples, dtype=np.bool_),
    )
    return RecoveryDatasetRecord(
        artifact=ArtifactRecord(
            artifact_id=artifact_id,
            kind="processed",
            created_at=CREATED,
            license="LicenseRef-Private",
            access="private",
            payload=Payload(f"armrc://processed/{artifact_id}/samples.npz", payload_sha, 2048, "samples.npz", 1),
            origin=Origin(
                command="synthetic recovery slice fixture",
                config_sha256="2" * 64,
                project_commit="a" * 40,
                project_dirty=False,
                dependency_commits={},
                sources=(RAW_SOURCE,),
            ),
        ),
        scenario=Scenario(
            config_path="tests/fixtures/configs/planar_2dof_fixture.toml",
            config_sha256=sha256_file(SCENARIO_FILE),
            robot="planar-2dof-fixture",
            task="pd-reach-fixture",
            dof=2,
            initial_q=tuple(SCENARIO.task.initial_q),
            target=tuple(SCENARIO.task.target),
        ),
        n_samples=samples.n_samples,
        dof=samples.dof,
        task_dim=samples.task_dim,
        task_code_dim=samples.task_code_dim,
        units=dict(CANONICAL_UNITS),
        phases=dict(TASK_PHASE_CODES),
        preprocessing=Preprocessing(
            resample_period_s=DT, smoothing="none", smoothing_params={}, derivative_method="central-difference"
        ),
        onset=OnsetAnnotation(
            kind="scripted",
            raw_artifact_id=RAW_SOURCE,
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
            q_pre=tuple(SCENARIO.task.initial_q), tolerance_rad=0.05, max_deviation_rad=0.0, status="passed"
        ),
        crop=CropWindow(pre_roll=(0.0, 1.0), source_duration_s=2.0, task=TASK),
        q0_ref=tuple(float(v) for v in samples.q[0]),
        arrays=array_specs(samples),
        normalization=normalization,
    )


@pytest.fixture(scope="module")
def fixture_dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[RecoveryDatasetRecord, SampleSet]:
    """A recovery dataset record whose payload digest matches the synthetic samples."""
    from arm_rc_ctrl.data.samples import save_samples

    samples = _samples()
    staged = tmp_path_factory.mktemp("slice") / "samples.npz"
    save_samples(staged, samples)
    return _record(samples, sha256_file(staged)), samples


def _recipe(record: RecoveryDatasetRecord, samples: SampleSet, warmup_s: float) -> ModelRecipe:
    assert record.normalization is not None
    transform = InputTransform.derive("fixed_scale", record.normalization, fixed_scales={"q": 0.3, "dq": 4.0})
    recipe, _ = create_recipe(
        "slice-test",
        ESN,
        sources=[
            DatasetSource(
                record.artifact.artifact_id,
                record.artifact.payload.sha256,
                f"data/records/processed/{record.artifact.artifact_id}.toml",
            )
        ],
        samples={record.artifact.artifact_id: samples},
        dof=2,
        task_code_dim=0,
        preprocessing=record.preprocessing,
        transform=transform,
        training=TrainingSpec(washout="warmup_hold", warmup_s=warmup_s),
    )
    return recipe


def test_warmup_training_specs_are_versioned() -> None:
    """The M3 spec is unchanged; the warm-up spec requires an approved duration and builds task episodes."""
    assert TrainingSpec() == TrainingSpec(washout="prime_phase")
    with pytest.raises(ValueError, match="warmup_s"):
        TrainingSpec(washout="warmup_hold")
    with pytest.raises(ValueError, match=r"approved|warmup_s"):
        TrainingSpec(washout="warmup_hold", warmup_s=0.3)
    with pytest.raises(ValueError, match="warmup_s"):
        TrainingSpec(washout="prime_phase", warmup_s=1.0)


@pytest.mark.parametrize("warmup_s", sorted(APPROVED_WARMUPS_S))
def test_recipe_episodes_cover_every_warmup_duration(
    fixture_dataset: tuple[RecoveryDatasetRecord, SampleSet], warmup_s: float
) -> None:
    """Each approved T_w yields a recipe whose single episode has exactly T_w/dt washout rows."""
    record, samples = fixture_dataset
    recipe = _recipe(record, samples, warmup_s)
    assert recipe.training.warmup_s == warmup_s
    (episode,) = recipe.episodes({record.artifact.artifact_id: samples})
    assert episode.washout_len == WarmupConfig(warmup_s).n_rows(DT)
    assert int(episode.loss_rows.sum()) == samples.n_samples - 1


def test_held_reference_holds_then_plays_the_task_reference(
    fixture_dataset: tuple[RecoveryDatasetRecord, SampleSet],
) -> None:
    """Before activation the reference is the initial posture at rest; after, the shifted task reference."""
    _, samples = fixture_dataset
    reference = HeldTaskReference.from_samples(samples, activation_s=0.5, interpolation="linear")
    q_hold, dq_hold, ddq_hold = reference.sample(0.2)
    assert np.array_equal(q_hold, samples.q[0])
    assert not dq_hold.any()
    assert not ddq_hold.any()
    q_active, _, _ = reference.sample(0.5 + 0.4)
    expected = samples.q[np.argmin(np.abs(samples.t - 0.4))]
    assert np.allclose(q_active, expected)
    zero = HeldTaskReference.from_samples(samples, activation_s=0.0, interpolation="linear")
    q0, _, _ = zero.sample(0.0)
    assert np.allclose(q0, samples.q[0])
    perturbed = np.array([0.35, 0.55])
    held = HeldTaskReference.from_samples(samples, activation_s=0.5, interpolation="linear", hold=perturbed)
    q_perturbed, dq_perturbed, _ = held.sample(0.25)
    assert np.array_equal(q_perturbed, perturbed)
    assert not dq_perturbed.any()
    q_task, _, _ = held.sample(0.5)
    assert np.allclose(q_task, samples.q[0])
    with pytest.raises(ValueError, match="hold"):
        HeldTaskReference.from_samples(samples, activation_s=0.5, hold=np.array([0.1]))


@pytest.mark.parametrize("warmup_s", [0.0, 0.25, 1.0])
def test_paired_arms_hold_and_activate_simultaneously(
    fixture_dataset: tuple[RecoveryDatasetRecord, SampleSet], tmp_path: Path, warmup_s: float
) -> None:
    """Replay and RC share the initial posture, the pre-task clock, and the activation boundary."""
    record, samples = fixture_dataset
    store_root = tmp_path / "store"
    store_root.mkdir()
    store = StorageRoot(store_root, repositories=(REPO_ROOT,))
    pair = run_recovery_pair(
        SCENARIO,
        SCENARIO_FILE,
        record,
        samples,
        _recipe(record, samples, warmup_s),
        TRACKER,
        store=store,
        warmup=WarmupConfig(warmup_s),
        exploratory=True,
    )
    assert pair.activation_s == warmup_s
    for result in (pair.replay, pair.rc):
        assert result.summary.activation_s == warmup_s
        assert result.summary.duration_s == pytest.approx(warmup_s + float(samples.t[-1]))
        assert result.summary.termination.kind == "completed"
    replay_arrays = pair.replay.run.arrays.arrays
    rc_arrays = pair.rc.run.arrays.arrays
    assert np.array_equal(replay_arrays["q"][0], rc_arrays["q"][0])
    hold = replay_arrays["t"] < warmup_s
    if warmup_s > 0.0:
        assert np.allclose(replay_arrays["q_desired"][hold], samples.q[0])
        assert not replay_arrays["dq_desired"][hold].any()
        assert np.all(rc_arrays["phase"][hold] == 0)
    assert np.all(rc_arrays["phase"][~hold] == 1)
    active_start = int(np.count_nonzero(hold))
    assert float(rc_arrays["t"][active_start]) == pytest.approx(warmup_s)
    assert np.all(np.isnan(rc_arrays["generator_output_q"][hold]))
    assert np.all(np.isfinite(rc_arrays["generator_output_q"][~hold]))
    reference_after = samples.q[: np.count_nonzero(~hold)]
    assert np.allclose(replay_arrays["q_desired"][~hold], reference_after, atol=1e-9)
    assert isinstance(pair.recovery, RecoveryMetricsReport)
    assert pair.recovery.command_gap_early.samples > 0


def test_perturbed_start_is_shared_and_measured(
    fixture_dataset: tuple[RecoveryDatasetRecord, SampleSet], tmp_path: Path
) -> None:
    """A perturbed initial posture applies identically to both arms and yields a nonzero activation jump."""
    record, samples = fixture_dataset
    store_root = tmp_path / "store"
    store_root.mkdir()
    store = StorageRoot(store_root, repositories=(REPO_ROOT,))
    initial = tuple(float(v) + d for v, d in zip(record.q0_ref, (0.05, -0.04), strict=True))
    pair = run_recovery_pair(
        SCENARIO,
        SCENARIO_FILE,
        record,
        samples,
        _recipe(record, samples, 0.25),
        TRACKER,
        store=store,
        warmup=WarmupConfig(0.25),
        exploratory=True,
        initial_q=initial,
    )
    for result in (pair.replay, pair.rc):
        arrays = result.run.arrays.arrays
        assert np.allclose(arrays["q"][0], initial)
        hold = arrays["t"] < 0.25
        assert bool(hold.any())
        # Both arms hold the *perturbed* posture: the desired command never drifts toward the
        # reference before activation (the protocol forbids correcting the offset during the hold).
        assert np.allclose(arrays["q_desired"][hold], initial)
        assert not arrays["dq_desired"][hold].any()
    assert pair.recovery is not None
    assert pair.recovery.activation_jump_rad > 0.0
    assert pair.recovery.smoothness_actual.samples == samples.n_samples


@pytest.mark.parametrize("warmup_s", [0.0, 1.0])
def test_force_pulses_are_scheduled_on_the_task_clock(
    fixture_dataset: tuple[RecoveryDatasetRecord, SampleSet], tmp_path: Path, warmup_s: float
) -> None:
    """A task-relative pulse lands at activation_s + start for every warm-up and never inside the hold."""
    record, samples = fixture_dataset
    store_root = tmp_path / "store"
    store_root.mkdir()
    store = StorageRoot(store_root, repositories=(REPO_ROOT,))
    pulse = ForcePulse(start_s=0.2, duration_s=0.1, force=(0.0, -3.0))
    pair = run_recovery_pair(
        SCENARIO,
        SCENARIO_FILE,
        record,
        samples,
        _recipe(record, samples, warmup_s),
        TRACKER,
        store=store,
        warmup=WarmupConfig(warmup_s),
        exploratory=True,
        force=pulse,
    )
    for result in (pair.replay, pair.rc):
        arrays = result.run.arrays.arrays
        t = arrays["t"]
        applied = np.abs(arrays["ext_force"]).sum(axis=1) > 0
        assert bool(applied.any())
        applied_times = t[applied]
        # Boundary-robust: the pulse starts at activation + task start, lasts its duration to
        # within one control period, is contiguous, and never overlaps the hold.
        assert float(applied_times[0]) == pytest.approx(warmup_s + pulse.start_s, abs=1e-9)
        assert float(applied_times[-1] - applied_times[0]) == pytest.approx(pulse.duration_s, abs=DT + 1e-9)
        assert int(np.count_nonzero(applied)) == int(np.count_nonzero(np.diff(applied_times) > 0)) + 1
        assert bool(np.all(np.diff(np.argwhere(applied).ravel()) == 1))  # contiguous block
        assert not bool(applied[t < warmup_s].any())  # never during the hold
        (disturbance,) = result.summary.disturbances
        assert disturbance.start_s == pytest.approx(warmup_s + pulse.start_s)
        force_config = result.summary.provenance.config["force"]
        assert isinstance(force_config, dict)
        assert force_config["task"]["start_s"] == pulse.start_s
        assert force_config["run"]["start_s"] == pytest.approx(warmup_s + pulse.start_s)


def test_provenance_records_the_common_schedule(
    fixture_dataset: tuple[RecoveryDatasetRecord, SampleSet], tmp_path: Path
) -> None:
    """Both runs bind the dataset, warm-up, tracker, and activation boundary into provenance."""
    record, samples = fixture_dataset
    store_root = tmp_path / "store"
    store_root.mkdir()
    store = StorageRoot(store_root, repositories=(REPO_ROOT,))
    pair = run_recovery_pair(
        SCENARIO,
        SCENARIO_FILE,
        record,
        samples,
        _recipe(record, samples, 0.5),
        TRACKER,
        store=store,
        warmup=WarmupConfig(0.5),
        exploratory=True,
    )
    for result in (pair.replay, pair.rc):
        config = result.summary.provenance.config
        assert config["reference_artifact"] == record.artifact.artifact_id
        assert config["activation_s"] == 0.5
        warm = config["warmup"]
        assert isinstance(warm, dict)
        assert warm["duration_s"] == 0.5
        assert result.summary.provenance.artifacts[0].sha256 == record.artifact.payload.sha256
        assert result.pointer.artifact.origin.sources == (record.artifact.artifact_id,)
    assert pair.rc.summary.provenance.seeds == {"reservoir": ESN.reservoir.seed}

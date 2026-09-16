# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""A hermetic fixture store of the manual-demonstration study (M3MAN-007; manual plan sections 4 and 5).

Ten synthetic manual recordings of deliberately unequal length are written as
digest-verified manual dataset records and payloads into a temporary repository
root and storage root, their contractive banks are grown, and the frozen study
manifest of the 186 approved models is built over them with the committed
repetition panel and model configuration. Tests of the fit cache and of the
numerical copy controls build on this instead of the private external store.

The recordings are short and the reservoirs are the panel's own, so a handful
of arms can be fitted in a unit test; nothing here depends on a machine path.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from arm_rc_ctrl.data.derivatives import DerivativeConfig, differentiate
from arm_rc_ctrl.data.manual import (
    MANUAL_PHASE_CODES,
    MANUAL_SCHEMA_VERSION,
    DwellMeasurement,
    DwellPredicate,
    ManualDatasetRecord,
    MotionSummary,
    RawTiming,
    SmoothingCheck,
    StartCheck,
)
from arm_rc_ctrl.data.manual_scenario import load_manual_scenario, manual_endpoint_positions
from arm_rc_ctrl.data.records import (
    CANONICAL_UNITS,
    PROCESSED_PAYLOAD_FORMAT,
    PROCESSED_PAYLOAD_NAME,
    ArtifactRecord,
    Origin,
    Payload,
    Preprocessing,
    Scenario,
    array_specs,
    make_artifact_id,
    payload_from_store,
    write_record,
)
from arm_rc_ctrl.data.samples import SAMPLES_SCHEMA_VERSION, SampleSet, save_samples
from arm_rc_ctrl.experiments.manual_augmentation import ManualParent
from arm_rc_ctrl.experiments.manual_bank import BankManifest, TakeMeasurements, TakeVerdict, write_bank_manifest
from arm_rc_ctrl.experiments.manual_fits import ManualFitInputs
from arm_rc_ctrl.experiments.manual_recipes import ASSIGNMENTS
from arm_rc_ctrl.experiments.manual_study import (
    EXPERIMENT_LABEL,
    build_study_manifest,
    contractive_banks,
    frozen_transform,
    study_to_json,
)
from arm_rc_ctrl.experiments.repetition_panel import load_panel
from arm_rc_ctrl.provenance import collect_provenance, sha256_file
from arm_rc_ctrl.rc.recipe import RclibIdentity, TrainingValidation
from arm_rc_ctrl.rc.train import load_model_config
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageRoot

if TYPE_CHECKING:
    from arm_rc_ctrl.data.manual_scenario import ManualScenarioConfig
    from arm_rc_ctrl.execution import ExecutionRecord
    from arm_rc_ctrl.experiments.manual_study import StudyManifest
    from arm_rc_ctrl.provenance import ArtifactReference, ProvenanceRecord

__all__ = [
    "DT",
    "DWELL_START_S",
    "GOAL_Q",
    "HOLD_S",
    "NOW",
    "SEED_BANK",
    "SESSION",
    "ManualFixture",
    "build_manual_fixture",
    "fixture_samples",
]

REPO_ROOT = repository_root()
PANEL_RELATIVE = Path("docs") / "experiments" / "task_1a_repeated_demonstration" / "panel_manifest_v1.json"
BANK_RELATIVE = Path("docs") / "experiments" / "task_1a_manual_demonstration" / "bank" / "bank_v1.json"
STUDY_RELATIVE = Path("docs") / "experiments" / "task_1a_manual_demonstration" / "study_manifest_v1.json"
SCENARIO_RELATIVE = Path("configs") / "tasks" / "manual_fixture.toml"
PREPROCESSING_RELATIVE = Path("configs") / "preprocessing" / "manual_fixture.toml"
FIXTURE_CONFIGS = REPO_ROOT / "tests" / "fixtures" / "configs"

NOW = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)
CREATED = "2026-09-16T12:00:00+00:00"
DT = 0.01
HOLD_S = 0.2
DWELL_START_S = 1.5
"""Recorded final-dwell onset: the frozen 0.5 s contractive ramp closes well before the taper window opens."""
GOAL_Q: tuple[float, float] = (0.8, 0.4)
SEED_BANK = 3
SESSION = "fixture-01"
BASE_SAMPLES = 201
"""Samples of the shortest recording; each further demonstration adds four, so every weight ``400 / L_i`` differs."""
DERIVATIVES = DerivativeConfig(method="central")
_LICENSE = "proprietary"
_COMMAND = "python -m arm_rc_ctrl.experiments.manual_fixture"


def fixture_samples(scenario: ManualScenarioConfig, n_samples: int) -> SampleSet:
    """A complete manual recording: a recorded pre-roll, a smooth reach, and the final dwell at the target."""
    t = np.arange(n_samples, dtype=np.float64) * DT
    start = np.asarray(scenario.task.initial_q, dtype=np.float64)
    ramp = np.clip((t - HOLD_S) / (DWELL_START_S - HOLD_S), 0.0, 1.0)
    blend = ramp * ramp * (3.0 - 2.0 * ramp)
    q = start[None, :] + blend[:, None] * (np.asarray(GOAL_Q, dtype=np.float64) - start)[None, :]
    dq, ddq = differentiate(q, DT, DERIVATIVES)
    tip = manual_endpoint_positions(scenario, q)
    dtip, ddtip = differentiate(tip, DT, DERIVATIVES)
    phase = np.full(n_samples, MANUAL_PHASE_CODES["move"], dtype=np.int64)
    phase[t < HOLD_S - 1e-9] = MANUAL_PHASE_CODES["hold"]
    phase[t >= DWELL_START_S - 1e-9] = MANUAL_PHASE_CODES["dwell"]
    return SampleSet(t, q, dq, ddq, tip, dtip, ddtip, np.zeros((n_samples, 0), dtype=np.float64), phase)


@dataclass(frozen=True)
class ManualFixture:
    """A temporary repository root and store holding the ten demonstrations and the frozen study manifest."""

    root: Path
    store: StorageRoot
    manifest: StudyManifest
    manifest_file: Path
    samples: dict[str, SampleSet]
    payloads: tuple[ArtifactReference, ...]
    inputs: ManualFitInputs
    provenance: ProvenanceRecord
    env: dict[str, str]
    """The environment a pinned launcher would have exported (plus the store root) for worker processes."""
    execution: ExecutionRecord
    now: datetime = NOW

    @property
    def scenario_file(self) -> Path:
        """The manual task configuration under the temporary root."""
        return self.root / SCENARIO_RELATIVE


def _preprocessing() -> Preprocessing:
    return Preprocessing(
        resample_period_s=DT,
        smoothing="none",
        smoothing_params={},
        derivative_method="central-difference",
        interpolation="linear",
    )


def _scenario_section(scenario: ManualScenarioConfig, scenario_file: Path, root: Path) -> Scenario:
    return Scenario(
        config_path=scenario_file.relative_to(root).as_posix(),
        config_sha256=sha256_file(scenario_file),
        robot=scenario.robot.name,
        task=scenario.name,
        dof=scenario.dof,
        initial_q=scenario.task.initial_q,
        target=scenario.task.target,
    )


def _dwell(scenario: ManualScenarioConfig, samples: SampleSet) -> DwellMeasurement:
    """The measured final dwell of a fixture recording (it rests exactly on the target from ``DWELL_START_S``)."""
    predicate = DwellPredicate(
        tolerance_m=scenario.task.tolerance,
        max_velocity_rad_s=scenario.task.dwell_max_velocity,
        min_duration_s=scenario.task.dwell_min_duration_s,
        min_samples=scenario.dwell_min_samples,
    )
    held = int(np.count_nonzero(samples.phase == MANUAL_PHASE_CODES["dwell"]))
    return DwellMeasurement(
        predicate=predicate,
        ok=True,
        final_samples=held,
        final_duration_s=(held - 1) * DT,
        start_s=DWELL_START_S,
        end_s=float(samples.t[-1]),
        max_endpoint_error_m=0.0,
        max_joint_speed_rad_s=0.0,
        longest_samples=held,
        earliest_start_s=DWELL_START_S,
    )


def _motion(scenario: ManualScenarioConfig, samples: SampleSet) -> MotionSummary:
    target = np.asarray(scenario.task.target, dtype=np.float64)
    return MotionSummary(
        hold_end_s=HOLD_S,
        dwell_start_s=DWELL_START_S,
        movement_duration_s=DWELL_START_S - HOLD_S,
        time_to_dwell_s=DWELL_START_S,
        path_length_rad=float(np.sum(np.abs(np.diff(samples.q, axis=0)))),
        peak_speed_rad_s=float(np.max(np.abs(samples.dq))),
        final_q=tuple(float(v) for v in samples.q[-1]),
        final_tip=tuple(float(v) for v in samples.tip[-1]),
        final_endpoint_error_m=float(np.hypot(*(np.asarray(samples.tip[-1], dtype=np.float64) - target))),
    )


def _record(
    scenario: ManualScenarioConfig,
    scenario_file: Path,
    samples: SampleSet,
    *,
    root: Path,
    artifact_id: str,
    payload: Payload,
    take: int,
    raw_artifact_id: str,
) -> ManualDatasetRecord:
    """The committed record of one fixture demonstration (no smoothing, so the recording is its own dataset)."""
    intervals = np.diff(samples.t)
    speeds = np.abs(np.diff(samples.q, axis=0)) / intervals[:, None]
    return ManualDatasetRecord(
        artifact=ArtifactRecord(
            artifact_id=artifact_id,
            kind="processed",
            created_at=CREATED,
            license=_LICENSE,
            access="private",
            payload=payload,
            origin=Origin(
                command=_COMMAND,
                config_sha256="2" * 64,
                project_commit="a" * 40,
                project_dirty=False,
                dependency_commits={},
                sources=(raw_artifact_id,),
            ),
        ),
        scenario=_scenario_section(scenario, scenario_file, root),
        n_samples=samples.n_samples,
        dof=samples.dof,
        task_dim=samples.task_dim,
        task_code_dim=samples.task_code_dim,
        units=dict(CANONICAL_UNITS),
        phases=dict(MANUAL_PHASE_CODES),
        preprocessing=_preprocessing(),
        session=SESSION,
        take=take,
        raw_timing=RawTiming(
            n_frames=samples.n_samples,
            duration_s=float(samples.t[-1]),
            median_interval_s=DT,
            max_interval_s=DT,
            late_frames=0,
            max_raw_speed_rad_s=tuple(float(v) for v in np.max(speeds, axis=0)),
        ),
        start=StartCheck(
            expected=scenario.task.initial_q,
            observed=tuple(float(v) for v in samples.q[0]),
            max_deviation_rad=0.0,
            tolerance_rad=scenario.acquisition.start_tolerance_rad,
            ok=True,
        ),
        smoothing=SmoothingCheck(
            method="none",
            onset_s=HOLD_S,
            start_shift_rad=0.0,
            tolerance_rad=scenario.acquisition.start_tolerance_rad,
            ok=True,
            hold_margin_s=0.1,
            anchored_from_s=0.0,
        ),
        dwell=_dwell(scenario, samples),
        motion=_motion(scenario, samples),
        arrays=array_specs(samples),
        manual_schema_version=MANUAL_SCHEMA_VERSION,
    )


def _publish(
    scenario: ManualScenarioConfig,
    scenario_file: Path,
    samples: SampleSet,
    *,
    root: Path,
    store: StorageRoot,
    staging: Path,
    take: int,
) -> ManualDatasetRecord:
    """Write one demonstration's payload into the store and its record into the temporary repository."""
    staged = staging / f"take_{take:03d}.npz"
    save_samples(staged, samples)
    artifact_id = make_artifact_id("processed", CREATED, sha256_file(staged))
    uri = f"armrc://processed/{artifact_id}/{PROCESSED_PAYLOAD_NAME}"
    shutil.move(staged, store.path(uri, mode="write"))
    payload = payload_from_store(store, uri, format=PROCESSED_PAYLOAD_FORMAT, schema_version=SAMPLES_SCHEMA_VERSION)
    raw_artifact_id = make_artifact_id("raw", CREATED, payload.sha256)
    record = _record(
        scenario,
        scenario_file,
        samples,
        root=root,
        artifact_id=artifact_id,
        payload=payload,
        take=take,
        raw_artifact_id=raw_artifact_id,
    )
    record_file = root / "data" / "records" / "processed" / f"{artifact_id}.toml"
    record_file.parent.mkdir(parents=True, exist_ok=True)
    write_record(record_file, record)
    return record


def _bank_manifest(
    records: list[ManualDatasetRecord], *, scenario_file: Path, preprocessing_file: Path, root: Path
) -> BankManifest:
    """The locked bank the ten fixture demonstrations came from, with one retained rejection."""
    takes = [
        TakeVerdict(
            batch=1,
            attempt=index + 1,
            source_file=f"reach_{index + 1:03d}.sklog.npz",
            accepted=True,
            reasons=(),
            raw_artifact_id=record.artifact.origin.sources[0],
            processed_artifact_id=record.artifact.artifact_id,
            payload_sha256=record.artifact.payload.sha256,
            q_sha256=record.arrays["q"].sha256,
            duplicate_of=None,
            assignment=ASSIGNMENTS[index],
            measurements=TakeMeasurements(),
        )
        for index, record in enumerate(records)
    ]
    takes.append(
        TakeVerdict(
            batch=2,
            attempt=len(records) + 1,
            source_file=f"reach_{len(records) + 1:03d}.sklog.npz",
            accepted=False,
            reasons=("final dwell shorter than 1.00 s",),
            raw_artifact_id=make_artifact_id("raw", CREATED, "cd" * 32),
            processed_artifact_id=None,
            payload_sha256="cd" * 32,
            q_sha256="ef" * 32,
            duplicate_of=None,
            assignment=None,
            measurements=TakeMeasurements(),
        )
    )
    return BankManifest(
        bank_schema_version=1,
        protocol=EXPERIMENT_LABEL,
        session=SESSION,
        scenario_path=scenario_file.relative_to(root).as_posix(),
        scenario_sha256=sha256_file(scenario_file),
        derive_config_path=preprocessing_file.relative_to(root).as_posix(),
        derive_config_sha256=sha256_file(preprocessing_file),
        required=len(ASSIGNMENTS),
        takes=tuple(takes),
        updated_at=CREATED,
    )


def _parent(assignment: str, record: ManualDatasetRecord, samples: SampleSet) -> ManualParent:
    from arm_rc_ctrl.rc.recipe import DatasetSource

    return ManualParent(
        assignment=assignment,
        dataset=DatasetSource(
            record.artifact.artifact_id,
            record.artifact.payload.sha256,
            f"data/records/processed/{record.artifact.artifact_id}.toml",
        ),
        dwell_start_s=DWELL_START_S,
        n_samples=samples.n_samples,
        period_s=DT,
        derivative_method=DERIVATIVES.label,
        q_sha256=record.arrays["q"].sha256,
        dq_sha256=record.arrays["dq"].sha256,
    )


def _copy_sources(root: Path) -> tuple[Path, Path, Path, Path]:
    """Place the committed panel, model configuration, transform-source record, and fixture configs under ``root``."""
    from arm_rc_ctrl.experiments.manual_recipes import TRANSFORM_SOURCE

    panel_file = root / PANEL_RELATIVE
    panel_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO_ROOT / PANEL_RELATIVE, panel_file)
    model_relative = load_panel(panel_file).configs.model_file
    model_file = root / model_relative
    model_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO_ROOT / model_relative, model_file)
    transform_record = root / TRANSFORM_SOURCE.record
    transform_record.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO_ROOT / TRANSFORM_SOURCE.record, transform_record)
    scenario_file = root / SCENARIO_RELATIVE
    scenario_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FIXTURE_CONFIGS / "planar_2dof_manual_fixture.toml", scenario_file)
    preprocessing_file = root / PREPROCESSING_RELATIVE
    preprocessing_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FIXTURE_CONFIGS / "manual_derive_fixture.toml", preprocessing_file)
    return panel_file, model_file, scenario_file, preprocessing_file


def build_manual_fixture(base: Path, *, execution: ExecutionRecord, env: dict[str, str]) -> ManualFixture:
    """Build the ten demonstrations, their contractive banks, and the frozen study manifest under ``base``.

    ``env`` is the environment worker processes inherit (the pinned launcher's
    variables plus the store root); it is recorded verbatim.
    """
    from arm_rc_ctrl.provenance import ArtifactReference

    root = base / "repo"
    root.mkdir(parents=True)
    store_root = base / "store"
    store_root.mkdir()
    store = StorageRoot(store_root, repositories=(REPO_ROOT,))
    staging = base / "staging"
    staging.mkdir()
    panel_file, model_file, scenario_file, preprocessing_file = _copy_sources(root)
    scenario = load_manual_scenario(scenario_file)

    records: list[ManualDatasetRecord] = []
    samples: dict[str, SampleSet] = {}
    for index in range(len(ASSIGNMENTS)):
        recording = fixture_samples(scenario, BASE_SAMPLES + 4 * index)
        record = _publish(scenario, scenario_file, recording, root=root, store=store, staging=staging, take=index + 1)
        records.append(record)
        samples[record.artifact.artifact_id] = recording

    bank = _bank_manifest(records, scenario_file=scenario_file, preprocessing_file=preprocessing_file, root=root)
    bank_file = root / BANK_RELATIVE
    bank_file.parent.mkdir(parents=True, exist_ok=True)
    write_bank_manifest(bank_file, bank)
    parents = tuple(
        _parent(name, record, samples[record.artifact.artifact_id])
        for name, record in zip(ASSIGNMENTS, records, strict=True)
    )
    parent_samples = {parent.identifier: samples[parent.identifier] for parent in parents}
    banks = contractive_banks(parents, parent_samples, scenario, seed_bank=SEED_BANK)
    model = load_model_config(model_file)
    provenance = collect_provenance({"kind": "manual-fixture"}, seeds={"seed_bank": SEED_BANK}, exploratory=True)
    manifest = build_study_manifest(
        panel=load_panel(panel_file),
        panel_file=panel_file,
        bank=bank,
        bank_file=bank_file,
        scenario_file=scenario_file,
        preprocessing_file=preprocessing_file,
        model=model,
        model_file=model_file,
        parents=parents,
        banks=banks,
        transform=frozen_transform(model, root=root),
        validation=TrainingValidation.from_scenario(scenario, scenario_file, root=root),
        execution=execution,
        provenance=provenance,
        seed_bank=SEED_BANK,
        rclib=RclibIdentity.current(),
        root=root,
    )
    manifest_file = root / STUDY_RELATIVE
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(study_to_json(manifest) + "\n", encoding="utf-8")
    payloads = tuple(
        ArtifactReference(r.artifact.payload.uri, r.artifact.payload.sha256, r.artifact.payload.size) for r in records
    )
    inputs = ManualFitInputs(
        manifest=manifest,
        samples=samples,
        dof=records[0].dof,
        task_code_dim=records[0].task_code_dim,
        preprocessing=records[0].preprocessing,
        scenario=scenario,
        root=root,
        execution_identity=execution.identity,
        rclib=RclibIdentity.current(),
    )
    return ManualFixture(
        root=root,
        store=store,
        manifest=manifest,
        manifest_file=manifest_file,
        samples=samples,
        payloads=payloads,
        inputs=inputs,
        provenance=provenance,
        env=env,
        execution=execution,
    )

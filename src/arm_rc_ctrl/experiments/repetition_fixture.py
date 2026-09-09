# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""A small deterministic fixture store of the repeated-demonstration pilot (clarification C9).

The planar 2-DOF fixture scenario, one synthetic recovery dataset written as a
digest-verified record and payload into a temporary repository root and
storage root, one panel entry, and the fit inputs the pilot's modules take.
Tests of the fit cache, the numerical validation, the paired evaluation, and
the reproduction logic build on it instead of the private external store.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

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
from arm_rc_ctrl.experiments.closed_loop import EstimatorSpec
from arm_rc_ctrl.experiments.esn_search import TrialPoint
from arm_rc_ctrl.experiments.recovery_search import RecoveryTrialPoint
from arm_rc_ctrl.experiments.repetition_fits import FitInputs
from arm_rc_ctrl.experiments.repetition_panel import PanelEntry
from arm_rc_ctrl.provenance import ArtifactReference, sha256_file
from arm_rc_ctrl.rc.esn import EsnConfig, ReadoutConfig, ReservoirConfig
from arm_rc_ctrl.rc.recipe import DatasetSource, RclibIdentity
from arm_rc_ctrl.rc.train import InputTransformSpec, ModelConfig
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import endpoint_positions, load_scenario
from arm_rc_ctrl.storage import StorageRoot

if TYPE_CHECKING:
    from arm_rc_ctrl.execution import ExecutionRecord

__all__ = [
    "BANK_COUNT",
    "BASE_MODEL",
    "DT",
    "ENTRY",
    "FIXTURE_CREATED",
    "GOAL_Q",
    "MOVE_END_S",
    "NOW",
    "SCENARIO_RELATIVE",
    "TASK",
    "N",
    "PlanarFixture",
    "build_planar_fixture",
    "planar_samples",
]

SCENARIO_RELATIVE = Path("tests") / "fixtures" / "configs" / "planar_2dof_fixture.toml"
"""The planar fixture scenario, relative to the repository root (copied into every fixture root)."""
FIXTURE_CREATED = "2026-09-09T10:00:00+00:00"
NOW = datetime(2026, 9, 9, 12, 0, tzinfo=UTC)
N = 101
DT = 0.01
MOVE_END_S = 0.8
BANK_COUNT = 16
"""Synthetic episodes per probe bank in tests (the evidence uses 64)."""
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


GOAL_Q: tuple[float, float] = (0.8, 0.4)
"""The joint goal of the fixture reach; its endpoint is the fixture scenario's task target."""


def planar_samples(scenario_file: Path, *, goal_q: tuple[float, float] = GOAL_Q) -> SampleSet:
    """A smooth 101-sample reach on the planar fixture (move until 0.8 s, dwell after)."""
    scenario = load_scenario(scenario_file)
    derivatives = DerivativeConfig(method="central")
    t = np.arange(N, dtype=np.float64) * DT
    start = np.array(scenario.task.initial_q)
    goal = np.array(goal_q, dtype=np.float64)
    s = np.clip(t / MOVE_END_S, 0.0, 1.0)
    blend = s * s * (3.0 - 2.0 * s)
    q = start[None, :] + blend[:, None] * (goal - start)[None, :]
    dq, ddq = differentiate(q, DT, derivatives)
    tip = endpoint_positions(scenario, q)
    dtip, ddtip = differentiate(tip, DT, derivatives)
    phase = np.where(t < MOVE_END_S, 1, 2).astype(np.int64)
    return SampleSet(t, q, dq, ddq, tip, dtip, ddtip, np.zeros((N, 0)), phase)


@dataclass(frozen=True)
class PlanarFixture:
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
    artifact_id: str, payload: Payload, samples: SampleSet, scenario_file: Path, preprocessing: Preprocessing
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
            payload=payload,
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


def build_planar_fixture(
    base: Path,
    *,
    execution: ExecutionRecord,
    env: dict[str, str],
    goal_q: tuple[float, float] = GOAL_Q,
    entry: PanelEntry = ENTRY,
) -> PlanarFixture:
    """Build the fixture under ``base`` (``repo`` and ``store`` subdirectories) bound to ``execution``.

    ``env`` is the environment the pilot's worker processes inherit (the
    launcher's variables plus the store root); it is recorded verbatim. A
    ``goal_q`` other than the default rewrites the scenario copy's task target
    to that posture's endpoint, so the reach still ends on target.
    """
    root = base / "repo"
    scenario_file = root / SCENARIO_RELATIVE
    scenario_file.parent.mkdir(parents=True)
    text = (repository_root() / SCENARIO_RELATIVE).read_text(encoding="utf-8")
    if goal_q != GOAL_Q:
        tip = endpoint_positions(load_scenario(repository_root() / SCENARIO_RELATIVE), np.array([goal_q]))[0]
        text = text.replace("target = [0.2996, 0.4482]", f"target = [{float(tip[0])!r}, {float(tip[1])!r}]")
    scenario_file.write_text(text, encoding="utf-8")
    store_root = base / "store"
    store_root.mkdir()
    store = StorageRoot(store_root, repositories=(repository_root(),))
    samples = planar_samples(scenario_file, goal_q=goal_q)
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
    if record.normalization is None:  # pragma: no cover - the record is built with its statistics
        msg = "the fixture record lacks normalization statistics"
        raise ValueError(msg)
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
    return PlanarFixture(root, store, record, samples, inputs, entry, env, execution)

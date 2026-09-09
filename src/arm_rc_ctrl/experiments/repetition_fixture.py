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
from typing import TYPE_CHECKING, Any, cast

import numpy as np

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
from arm_rc_ctrl.experiments.perturbations import RobustnessScenario
from arm_rc_ctrl.experiments.recovery_search import RecoveryTrialPoint
from arm_rc_ctrl.experiments.repetition_evaluation import (
    NumericalExceptionBinding,
    PilotRunner,
    RepetitionEvaluationConfig,
    load_evaluation_config,
    numerical_binding,
)
from arm_rc_ctrl.experiments.repetition_fits import FitInputs
from arm_rc_ctrl.experiments.repetition_panel import PanelEntry
from arm_rc_ctrl.experiments.run_record import RunArrays
from arm_rc_ctrl.experiments.simulation import CheckedState
from arm_rc_ctrl.experiments.termination import completed
from arm_rc_ctrl.provenance import ArtifactReference, collect_provenance, sha256_file
from arm_rc_ctrl.rc.esn import EsnConfig, ReadoutConfig, ReservoirConfig
from arm_rc_ctrl.rc.recipe import DatasetSource, RclibIdentity
from arm_rc_ctrl.rc.train import InputTransformSpec, ModelConfig
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import endpoint_positions, load_scenario
from arm_rc_ctrl.storage import StorageRoot

if TYPE_CHECKING:
    from collections.abc import Callable

    from arm_rc_ctrl.execution import ExecutionRecord
    from arm_rc_ctrl.experiments.termination import Termination
    from arm_rc_ctrl.provenance import ProvenanceRecord

__all__ = [
    "BANK_COUNT",
    "BASE_MODEL",
    "DOCS",
    "DT",
    "ENTRY",
    "FIXTURE_CREATED",
    "GOAL_Q",
    "MOVE_END_S",
    "NOW",
    "PLANAR_DIGESTS",
    "PLANAR_SCENARIOS",
    "PLANAR_TRACKER",
    "SCENARIO_RELATIVE",
    "TASK",
    "WARMUP_ROWS",
    "CraftedSimulator",
    "N",
    "PlanarFixture",
    "build_pilot_runner",
    "build_planar_fixture",
    "committed_numerical_binding",
    "crafted_run",
    "exploratory_provenance",
    "planar_samples",
    "silent",
    "write_pilot_evaluation_config",
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


# --- pilot test support (M3REP-004/005) ------------------------------------------------------

DOCS = repository_root() / "docs" / "experiments" / "task_1a_repeated_demonstration"
REPO_ROOT = repository_root()
WARMUP_ROWS = 25
"""Hold rows before activation at the fixture entry's 0.25 s warm-up."""
PLANAR_TRACKER = TrackerConfig(type="pd", kp=(10.0, 5.0), kd=(1.5, 0.8))
"""Gains under which the fixture's direct replay tracks without torque saturation."""
PLANAR_DIGESTS = {"pd_v2": "a" * 64, "computed_torque": "b" * 64}
PLANAR_SCENARIOS = (
    RobustnessScenario("nominal", "nominal", (0.0, 0.0)),
    RobustnessScenario("small-1", "posture_small", (0.02, 0.0), seed=1, draw=0, magnitude_rad=0.05),
    RobustnessScenario("small-2", "posture_small", (-0.02, 0.01), seed=1, draw=1, magnitude_rad=0.05),
    RobustnessScenario("large-1", "posture_large", (0.05, -0.02), seed=2, draw=0, magnitude_rad=0.1),
    RobustnessScenario(
        "force-000deg",
        "force",
        (0.0, 0.0),
        force_magnitude_n=3.0,
        force_start_s=0.3,
        force_duration_s=0.1,
        direction_deg=0.0,
    ),
)
_RAMP = 1.0 + 0.05 * np.linspace(0.0, 1.0, N)


def crafted_run(
    samples: SampleSet, *, rc: bool, error: float, saturation: float = 0.0, residual: bool = False, n: int | None = None
) -> RunArrays:
    """Crafted run arrays on the pilot schedule: a 0.25 s hold, then the reference plus a gently ramped offset."""
    dof = samples.dof
    rows = WARMUP_ROWS + samples.n_samples
    t = np.arange(rows, dtype=np.float64) * DT
    ramp = _RAMP[:, None]
    q = np.vstack([np.tile(samples.q[0], (WARMUP_ROWS, 1)), samples.q + error * ramp])
    dq = np.vstack([np.zeros((WARMUP_ROWS, dof)), samples.dq])
    tip = np.vstack([np.tile(samples.tip[0], (WARMUP_ROWS, 1)), samples.tip])
    saturated = np.zeros(rows, dtype=np.int64)
    saturated[: round(saturation * rows)] = 1
    zeros = np.zeros((rows, dof), dtype=np.float64)
    data: dict[str, Any] = {
        "t": t,
        "q": q,
        "dq": dq,
        "tip": tip,
        "q_desired": np.vstack([np.tile(samples.q[0], (WARMUP_ROWS, 1)), samples.q]),
        "dq_desired": dq.copy(),
        "dq_desired_raw": dq.copy(),
        "ddq_desired": zeros.copy(),
        "ddq_desired_raw": zeros.copy(),
        "tracking_error": zeros.copy(),
        "task_code": np.zeros((rows, 0), dtype=np.float64),
        "saturation": saturated,
        "tau_requested": zeros.copy(),
    }
    if rc:
        data["generator_output_q"] = np.vstack([np.full((WARMUP_ROWS, dof), np.nan), samples.q + 0.5 * error * ramp])
        data["phase"] = np.concatenate(
            [np.zeros(WARMUP_ROWS, dtype=np.int64), np.ones(samples.n_samples, dtype=np.int64)]
        )
        if residual:
            data["generator_increment_q"] = np.vstack(
                [np.full((WARMUP_ROWS, dof), np.nan), np.diff(samples.q, axis=0, append=samples.q[-1:])]
            )
    if n is not None:
        data = {name: values[:n] for name, values in data.items()}
    return RunArrays(data)


class CraftedSimulator:
    """A ``simulate`` stand-in that crafts feasible runs and records what it was asked to do.

    The planar fixture cannot run a feasible real RC closed loop (its torque limits saturate any tracker fast
    enough to follow the reach), so the pilot's sweep logic is tested on crafted runs; the real simulator covers
    the replay bank and the velocity abort.
    """

    def __init__(
        self,
        samples: SampleSet,
        *,
        blocked_replays: tuple[str, ...] = (),
        interrupt_after_rc: int | None = None,
    ) -> None:
        self.samples = samples
        self.blocked_replays = blocked_replays
        self.interrupt_after_rc = interrupt_after_rc
        self.rc_calls = 0
        self.replay_calls = 0
        self.aborts: list[tuple[float, ...]] = []

    def __call__(self, scenario: object, controller: object, **kwargs: object) -> tuple[RunArrays, Termination]:
        """Craft the run of one call; ``channels`` marks an rc run, everything else is a replay."""
        del scenario, controller
        rc = kwargs.get("channels") is not None
        residual = rc and getattr(kwargs.get("channels"), "generator_increment_q", None) is not None
        self.aborts.append(cast("tuple[float, ...]", kwargs["velocity_abort"]))
        if rc:
            if self.interrupt_after_rc is not None and self.rc_calls >= self.interrupt_after_rc:
                msg = "interrupted"
                raise KeyboardInterrupt(msg)
            self.rc_calls += 1
            arrays = crafted_run(self.samples, rc=True, error=0.01, residual=residual)
        else:
            self.replay_calls += 1
            start = cast("tuple[float, ...]", kwargs["initial_q"])
            blocked = any(np.allclose(start, self._start(label)) for label in self.blocked_replays)
            arrays = crafted_run(self.samples, rc=False, error=0.02, saturation=1.0 if blocked else 0.0)
        termination = completed(float(arrays.arrays["t"][-1]), arrays.n_samples - 1)
        sink = cast("list[CheckedState] | None", kwargs.get("checked_states"))
        if sink is not None:
            sink.extend(
                CheckedState(t=float(t), step=k, q=q, dq=dq)
                for k, (t, q, dq) in enumerate(
                    zip(arrays.arrays["t"], arrays.arrays["q"], arrays.arrays["dq"], strict=True)
                )
            )
        return arrays, termination

    def _start(self, label: str) -> tuple[float, ...]:
        case = next(s for s in PLANAR_SCENARIOS if s.scenario_id == label)
        return case.initial_q(tuple(float(v) for v in self.samples.q[0]))


def silent(message: str) -> None:
    """A log sink that drops its messages."""
    del message


def exploratory_provenance() -> ProvenanceRecord:
    """An exploratory provenance record of the current checkout (tests tolerate a dirty tree)."""
    return collect_provenance({"kind": "test"}, seeds={}, exploratory=True)


def committed_numerical_binding() -> NumericalExceptionBinding:
    """The committed M3REP-003 validation's binding."""
    return numerical_binding(DOCS / "numerical_validation_v1.json", root=REPO_ROOT)


def write_pilot_evaluation_config(
    root: Path, *, velocity_abort: tuple[float, ...] = (40.0, 40.0)
) -> tuple[RepetitionEvaluationConfig, Path]:
    """Write a pilot evaluation config under ``root`` naming the fixture's development levels."""
    development = root / "configs" / "evaluations" / "task_1a_recovery_dev_v1.toml"
    development.parent.mkdir(parents=True, exist_ok=True)
    if not development.exists():
        development.write_text(
            (REPO_ROOT / "configs" / "evaluations" / "task_1a_recovery_dev_v1.toml").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    stem = "_".join(f"{v:g}" for v in velocity_abort)
    file = root / "configs" / "evaluations" / f"task_1a_repetition_dev_v1_{stem}.toml"
    bounds = ", ".join(repr(float(v)) for v in velocity_abort)
    file.write_text(
        'name = "task-1a-repetition-dev-fixture"\n'
        'development = "task_1a_recovery_dev_v1.toml"\n\n'
        f"[simulation]\nvelocity_abort = [{bounds}]\n",
        encoding="utf-8",
    )
    return load_evaluation_config(file), file


def build_pilot_runner(
    f: PlanarFixture,
    *,
    velocity_abort: tuple[float, ...] = (40.0, 40.0),
    scenarios: tuple[RobustnessScenario, ...] = PLANAR_SCENARIOS,
    simulate_fn: Callable[..., Any] | None = None,
    log: list[str] | None = None,
) -> PilotRunner:
    """A pilot runner over the fixture with the planar tracker under both frozen names (exploratory provenance)."""
    evaluation, file = write_pilot_evaluation_config(f.root, velocity_abort=velocity_abort)
    return PilotRunner(
        store=f.store,
        inputs=f.inputs,
        dataset=f.record,
        evaluation=evaluation,
        evaluation_file=file,
        root=f.root,
        execution=f.execution,
        provenance=exploratory_provenance(),
        numerical=committed_numerical_binding(),
        scenarios=scenarios,
        trackers={"pd_v2": PLANAR_TRACKER, "computed_torque": PLANAR_TRACKER},
        tracker_digests=PLANAR_DIGESTS,
        development_sha256=sha256_file(evaluation.development),
        simulate_fn=simulate_fn,
        log=silent if log is None else log.append,
    )

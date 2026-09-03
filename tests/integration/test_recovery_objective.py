# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-012: paired recovery trial evaluation — replay baselines, feasibility reasons, and the worst-cell objective."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import numpy as np
import optuna
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
from arm_rc_ctrl.data.samples import SampleSet, save_samples
from arm_rc_ctrl.experiments import recovery_objective
from arm_rc_ctrl.experiments.esn_search import TrialPoint
from arm_rc_ctrl.experiments.perturbations import RobustnessScenario
from arm_rc_ctrl.experiments.recovery_objective import (
    RecoveryTrialContext,
    evaluate_recovery_point,
    make_recovery_objective,
    train_recovery_point,
)
from arm_rc_ctrl.experiments.recovery_search import (
    AugmentationPoint,
    RecoverySearchProtocol,
    RecoveryTrialPoint,
    load_recovery_search,
)
from arm_rc_ctrl.experiments.run_record import RunArrays
from arm_rc_ctrl.experiments.termination import completed, divergence, limit_violation
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.rc.recipe import DatasetSource
from arm_rc_ctrl.rc.train import load_model_config
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import endpoint_positions, load_scenario
from arm_rc_ctrl.storage import StorageRoot

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from arm_rc_ctrl.experiments.termination import Termination

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
SCENARIO_FILE = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
SCENARIO = load_scenario(SCENARIO_FILE)
MODEL = REPO_ROOT / "tests" / "fixtures" / "configs" / "esn_fixture.toml"
TRACKER = load_config(REPO_ROOT / "configs" / "controllers" / "pd.toml", TrackerConfig)
DEVELOPMENT = REPO_ROOT / "configs" / "evaluations" / "task_1a_recovery_dev_v1.toml"
CONFIRMATORY = REPO_ROOT / "configs" / "evaluations" / "task_1a_recovery_confirmatory_v1.toml"
DERIVATIVES = DerivativeConfig(method="central")
RAW_SOURCE = "raw-20260830-2a97516c354b"
CREATED = "2026-09-03T10:00:00+00:00"
N = 101
DT = 0.01
TASK = TaskIntervals(move=(0.0, 0.8), dwell=(0.8, 1.0))
POINT = RecoveryTrialPoint(
    esn=TrialPoint(80, 0.9, 0.9, 0.3, 0.5, 31, 1e-2, 20.0, 20.0), warmup_s=0.0, augmentation=None
)
AUG_POINT = replace(POINT, augmentation=AugmentationPoint(n_synthetic=16, sigma_rad=0.025, phi=0.98, gamma=1.0))

_BASE = """
name = "{name}"
scenario = "{scenario}"
model = "{model}"
formulation = "{formulation}"
budget = 4
seed_bank = 3
attempt_factor = 4
development = "{development}"

[sampler]
kind = "tpe"
seed = 20260903
n_startup_trials = 2

[pruner]
kind = "none"

[objective]
kind = "worst_cell_median_gap_ratio"
infeasible_penalty = 10.0

[esn]
n_neurons = {{ low = 50, high = 200, step = 50 }}
spectral_radius = {{ low = 0.8, high = 1.3 }}
sparsity = {{ low = 0.5, high = 0.98 }}
leak_rate = {{ low = 0.01, high = 0.3, log = true }}
input_scaling = {{ low = 0.02, high = 0.5, log = true }}
seed = {{ low = 1, high = 1000 }}
alpha = {{ low = 1e-3, high = 1.0, log = true }}
velocity_cutoff_hz = {{ low = 5.0, high = 30.0, log = true }}
acceleration_cutoff_hz = {{ low = 5.0, high = 30.0, log = true }}

[space]
warmups_s = [0.0, 0.25, 1.0]
{augmentation}
"""

_AUGMENTATION = """n_synthetic = [16, 32]
sigma_rad = [0.025, 0.05]
phi = [0.98, 0.99]
gamma = [0.5, 1.0]
"""


def _protocol(
    tmp_path: Path, *, formulation: str = "no_augmentation", development: Path = DEVELOPMENT
) -> RecoverySearchProtocol:
    file = tmp_path / f"{formulation}.toml"
    file.write_text(
        _BASE.format(
            name=f"recovery-objective-{formulation}",
            scenario=SCENARIO_FILE.as_posix(),
            model=MODEL.as_posix(),
            formulation=formulation,
            development=development.as_posix(),
            augmentation="" if formulation == "no_augmentation" else _AUGMENTATION,
        ),
        encoding="utf-8",
    )
    return load_recovery_search(file)


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
                command="synthetic recovery objective fixture",
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
            q_pre=tuple(float(v) for v in samples.q[0]), tolerance_rad=0.05, max_deviation_rad=0.0, status="passed"
        ),
        crop=CropWindow(pre_roll=(0.0, 1.0), source_duration_s=2.0, task=TASK),
        q0_ref=tuple(float(v) for v in samples.q[0]),
        arrays=array_specs(samples),
        normalization=normalization,
    )


@pytest.fixture(scope="module")
def fixture_dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[RecoveryDatasetRecord, SampleSet]:
    """A recovery dataset record whose payload digest matches the synthetic samples."""
    samples = _samples()
    staged = tmp_path_factory.mktemp("objective") / "samples.npz"
    save_samples(staged, samples)
    return _record(samples, sha256_file(staged)), samples


def _scenarios() -> tuple[RobustnessScenario, ...]:
    """One nominal, two small-posture, two large-posture, and one force scenario (protocol class order)."""
    return (
        RobustnessScenario("nominal", "nominal", (0.0, 0.0)),
        RobustnessScenario("small-1", "posture_small", (0.02, 0.0), seed=1, draw=0, magnitude_rad=0.05),
        RobustnessScenario("small-2", "posture_small", (-0.02, 0.01), seed=1, draw=1, magnitude_rad=0.05),
        RobustnessScenario("large-1", "posture_large", (0.05, -0.02), seed=2, draw=0, magnitude_rad=0.1),
        RobustnessScenario("large-2", "posture_large", (-0.04, 0.04), seed=2, draw=1, magnitude_rad=0.1),
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


def make_context(
    record: RecoveryDatasetRecord, samples: SampleSet, scenarios: tuple[RobustnessScenario, ...] | None = None
) -> RecoveryTrialContext:
    """A directly constructed trial context (fixture trackers under the frozen tracker names)."""
    artifact_id = record.artifact.artifact_id
    return RecoveryTrialContext(
        scenario=SCENARIO,
        scenario_file=SCENARIO_FILE,
        reference=samples,
        dataset=record,
        source=DatasetSource(artifact_id, record.artifact.payload.sha256, f"data/records/processed/{artifact_id}.toml"),
        trackers={"pd_v2": TRACKER, "computed_torque": TRACKER},
        base_model=load_model_config(MODEL),
        scenarios=_scenarios() if scenarios is None else scenarios,
    )


_RAMP = 1.0 + 0.05 * np.linspace(0.0, 1.0, N)
"""Per-sample scaling of crafted offsets (see the comment in ``crafted``)."""


def crafted(
    samples: SampleSet,
    *,
    duration_s: float = 1.0,
    rc: bool = False,
    error: float = 0.0,
    gen_offset: float = 0.0,
    tip_offset: float = 0.0,
    saturation: float = 0.0,
    n: int | None = None,
) -> RunArrays:
    """Run arrays on the recovery schedule: held rows before activation, then the reference plus the given offsets."""
    dof = samples.dof
    rows = round(duration_s / DT) + 1
    warm = rows - samples.n_samples
    t = np.arange(rows, dtype=np.float64) * DT
    # A gentle ramp keeps offset series non-constant: real dynamics never produce bitwise-constant
    # nonzero gaps, and the strict WindowSummary validation rejects their mean/max float-ulp inversion.
    ramp = _RAMP[:, None]
    q = np.vstack([np.tile(samples.q[0], (warm, 1)), samples.q + error * ramp])
    dq = np.vstack([np.zeros((warm, dof)), samples.dq])
    tip = np.vstack([np.tile(samples.tip[0], (warm, 1)), samples.tip + tip_offset])
    saturated = np.zeros(rows, dtype=np.int64)
    saturated[: round(saturation * rows)] = 1
    zeros = np.zeros((rows, dof), dtype=np.float64)
    data = {
        "t": t,
        "q": q,
        "dq": dq,
        "tip": tip,
        "q_desired": np.vstack([np.tile(samples.q[0], (warm, 1)), samples.q]),
        "dq_desired": dq.copy(),
        "dq_desired_raw": dq.copy(),
        "ddq_desired": zeros.copy(),
        "ddq_desired_raw": zeros.copy(),
        "tracking_error": zeros.copy(),
        "task_code": np.zeros((rows, samples.task_code.shape[1]), dtype=np.float64),
        "saturation": saturated,
        "tau_requested": zeros.copy(),
    }
    if rc:
        data["generator_output_q"] = np.vstack([np.full((warm, dof), np.nan), samples.q + gen_offset * ramp])
        data["phase"] = np.concatenate([np.zeros(warm, dtype=np.int64), np.ones(samples.n_samples, dtype=np.int64)])
    if n is not None:
        data = {name: values[:n] for name, values in data.items()}
    return RunArrays(data)


class FakeSimulate:
    """A ``simulate`` stand-in returning queued outcomes in the evaluator's documented order."""

    def __init__(self, outcomes: list[tuple[RunArrays, Termination]]) -> None:
        self.queue = list(outcomes)
        self.calls = 0

    def __call__(self, *_args: object, **_kwargs: object) -> tuple[RunArrays, Termination]:
        """Pop and return the next queued outcome."""
        self.calls += 1
        return self.queue.pop(0)


DONE = completed(1.0, N - 1)


def replay_queue(samples: SampleSet, *, error: float = 0.02) -> list[tuple[RunArrays, Termination]]:
    """Feasible replay outcomes for both tracker sweeps over the six fixture scenarios."""
    return [(crafted(samples, error=error), DONE) for _ in range(12)]


def test_context_load_refuses_the_confirmatory_lock(tmp_path: Path) -> None:
    """The confirmatory evaluation lock is unreachable from the search: loading such a protocol fails first."""
    protocol = _protocol(tmp_path, development=CONFIRMATORY)
    (tmp_path / "store").mkdir()
    store = StorageRoot(tmp_path / "store", repositories=(REPO_ROOT,))
    with pytest.raises(ValueError, match="confirmatory"):
        RecoveryTrialContext.load(protocol, store=store, dataset_file=tmp_path / "x.toml", records_root=tmp_path)


def test_replay_baselines_are_computed_once_per_tracker_and_warmup(
    fixture_dataset: tuple[RecoveryDatasetRecord, SampleSet], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The replay sweep runs one simulation per scenario and is cached across trials."""
    record, samples = fixture_dataset
    context = make_context(record, samples)
    fake = FakeSimulate([(crafted(samples, error=0.02), DONE) for _ in range(6)])
    monkeypatch.setattr(recovery_objective, "simulate", fake)
    first = context.replay_components("pd_v2", 0.0)
    again = context.replay_components("pd_v2", 0.0)
    assert fake.calls == 6
    assert again is first
    assert all(c.feasible for c in first)
    early = samples.t <= 0.5
    gap = 0.02 * _RAMP * math.sqrt(2.0)
    expected = float(np.trapezoid(gap[early], samples.t[early]))
    for component in first:
        assert component.early_gap_integral == pytest.approx(expected, rel=1e-9)
        assert component.activation_jump_rad == pytest.approx(0.02 * math.sqrt(2.0), rel=1e-12)
    with pytest.raises(ValueError, match="approved"):
        context.replay_components("pd_v2", 0.3)


def _diverging(samples: SampleSet) -> tuple[RunArrays, Termination]:
    return crafted(samples, rc=True, n=20), divergence(0.08, 19, "state grew without bound")


def _torque_limit(samples: SampleSet) -> tuple[RunArrays, Termination]:
    return crafted(samples, rc=True, n=20), limit_violation(0.08, 19, "torque", 90.0, 60.0, joint=0)


def _dwell_miss(samples: SampleSet) -> tuple[RunArrays, Termination]:
    return crafted(samples, rc=True, tip_offset=0.5), DONE


def _saturated(samples: SampleSet) -> tuple[RunArrays, Termination]:
    return crafted(samples, rc=True, saturation=0.5), DONE


def _generated_miss(samples: SampleSet) -> tuple[RunArrays, Termination]:
    return crafted(samples, rc=True, gen_offset=0.5), DONE


INFEASIBLE: dict[str, tuple[Callable[[SampleSet], tuple[RunArrays, Termination]], str, bool]] = {
    "divergence": (_diverging, "divergence", False),
    "torque-limit": (_torque_limit, "limit_violation:torque", False),
    "dwell": (_dwell_miss, "dwell:", True),
    "saturation": (_saturated, "saturation", False),
    "generated-dwell": (_generated_miss, "generated_dwell:generated_dwell_in_tolerance", False),
}
"""Infeasible RC outcomes: the crafted run, its reason prefix, and whether dwell rules must be tightened."""


@pytest.mark.parametrize("case", sorted(INFEASIBLE))
def test_an_infeasible_component_receives_the_penalty(
    fixture_dataset: tuple[RecoveryDatasetRecord, SampleSet],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: str,
) -> None:
    """The documented penalty replaces the objective; the reason names the scenario, tracker, and cause."""
    record, samples = fixture_dataset
    make, prefix, strict = INFEASIBLE[case]
    context = make_context(record, samples)
    if strict:
        task = replace(context.scenario.task, dwell_min_fraction=1.0)
        context = replace(context, scenario=replace(context.scenario, task=task))
    protocol = _protocol(tmp_path)
    outcomes = replay_queue(samples)
    outcomes.append(make(samples))
    fake = FakeSimulate(outcomes)
    monkeypatch.setattr(recovery_objective, "simulate", fake)
    evaluation = evaluate_recovery_point(protocol, context, POINT)
    assert not fake.queue
    assert evaluation.objective == protocol.objective.infeasible_penalty
    assert not evaluation.feasible
    assert evaluation.penalized
    assert evaluation.cells == {}
    assert evaluation.reason is not None
    assert evaluation.reason.startswith(f"scenario 0 [pd_v2]: {prefix}")
    assert len(evaluation.components) == 1
    assert evaluation.components[0].gap_ratio is None
    assert evaluation.running == (protocol.objective.infeasible_penalty,)
    assert evaluation.scenarios_total == 12
    assert evaluation.fit_rmse is not None
    assert math.isfinite(evaluation.fit_rmse)


def test_a_feasible_trial_scores_the_worst_cell_median_ratio(
    fixture_dataset: tuple[RecoveryDatasetRecord, SampleSet], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Both trackers run every scenario; posture ratios fill the four cells and the worst median is the objective."""
    record, samples = fixture_dataset
    context = make_context(record, samples)
    protocol = _protocol(tmp_path)
    outcomes = replay_queue(samples)
    for _scenario in range(6):
        outcomes.append((crafted(samples, rc=True, error=0.01), DONE))
        outcomes.append((crafted(samples, rc=True, error=0.03), DONE))
    fake = FakeSimulate(outcomes)
    monkeypatch.setattr(recovery_objective, "simulate", fake)
    evaluation = evaluate_recovery_point(protocol, context, POINT)
    assert not fake.queue
    assert evaluation.feasible
    assert not evaluation.penalized
    assert evaluation.reason is None
    assert len(evaluation.components) == 12
    assert evaluation.objective == pytest.approx(1.5, rel=1e-9)
    assert set(evaluation.cells) == {
        "posture_small:pd_v2",
        "posture_small:computed_torque",
        "posture_large:pd_v2",
        "posture_large:computed_torque",
    }
    assert evaluation.cells["posture_small:pd_v2"] == pytest.approx(0.5, rel=1e-9)
    assert evaluation.cells["posture_large:computed_torque"] == pytest.approx(1.5, rel=1e-9)
    ratios = [c.gap_ratio for c in evaluation.components]
    assert ratios[:2] == [None, None]  # nominal
    assert ratios[10:] == [None, None]  # force
    assert all(r is not None for r in ratios[2:10])
    assert len(evaluation.running) == 5
    assert all(value == pytest.approx(1.5, rel=1e-9) for value in evaluation.running)
    for component in evaluation.components:
        assert component.settling_time_s == 0.0
        assert component.torque_rms == 0.0
        assert component.generated_criteria == {
            "generated_dwell_in_tolerance": True,
            "generated_dwell_stationary": True,
        }
    attrs = evaluation.attrs()
    assert attrs["feasible"] is True
    assert attrs["components_evaluated"] == 12
    assert isinstance(attrs["cells"], dict)


def test_the_pruner_can_stop_a_partial_evaluation(
    fixture_dataset: tuple[RecoveryDatasetRecord, SampleSet], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Stopping at the first reported value keeps the partial objective without a penalty."""
    record, samples = fixture_dataset
    context = make_context(record, samples)
    protocol = _protocol(tmp_path)
    outcomes = replay_queue(samples)
    for _scenario in range(2):
        outcomes.append((crafted(samples, rc=True, error=0.01), DONE))
        outcomes.append((crafted(samples, rc=True, error=0.03), DONE))
    monkeypatch.setattr(recovery_objective, "simulate", FakeSimulate(outcomes))
    evaluation = evaluate_recovery_point(protocol, context, POINT, report=lambda _step, _value: True)
    assert evaluation.stopped_early
    assert not evaluation.feasible
    assert not evaluation.penalized
    assert evaluation.reason == "stopped by the pruner"
    assert len(evaluation.components) == 4
    assert evaluation.objective == pytest.approx(1.5, rel=1e-9)
    assert evaluation.running == (pytest.approx(1.5, rel=1e-9),)


def test_an_infeasible_replay_baseline_blocks_the_pair(
    fixture_dataset: tuple[RecoveryDatasetRecord, SampleSet], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A posture pair without a feasible replay baseline is infeasible and its RC run is never simulated."""
    record, samples = fixture_dataset
    context = make_context(record, samples)
    protocol = _protocol(tmp_path)
    outcomes: list[tuple[RunArrays, Termination]] = []
    for index in range(6):  # pd_v2 sweep: the first small-posture scenario diverges
        if index == 1:
            outcomes.append((crafted(samples, n=20), divergence(0.08, 19, "diverged")))
        else:
            outcomes.append((crafted(samples, error=0.02), DONE))
    outcomes.extend((crafted(samples, error=0.02), DONE) for _ in range(6))  # computed_torque sweep
    outcomes.append((crafted(samples, rc=True, error=0.01), DONE))  # nominal pd_v2
    outcomes.append((crafted(samples, rc=True, error=0.01), DONE))  # nominal computed_torque
    fake = FakeSimulate(outcomes)
    monkeypatch.setattr(recovery_objective, "simulate", fake)
    evaluation = evaluate_recovery_point(protocol, context, POINT)
    assert not fake.queue
    assert not evaluation.feasible
    assert evaluation.reason == "scenario 1 [pd_v2]: replay_infeasible:divergence"
    blocked = evaluation.components[2]
    assert blocked.termination == "not_simulated"
    assert not blocked.feasible
    assert blocked.replay_early_gap_integral is None


def test_a_training_failure_is_penalized_without_simulation(
    fixture_dataset: tuple[RecoveryDatasetRecord, SampleSet], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed fit yields the penalty, the named reason, and no components."""
    record, samples = fixture_dataset
    context = make_context(record, samples)
    protocol = _protocol(tmp_path)

    def boom(*_args: object, **_kwargs: object) -> object:
        msg = "synthetic training failure"
        raise ValueError(msg)

    monkeypatch.setattr(recovery_objective, "create_recipe", boom)
    evaluation = evaluate_recovery_point(protocol, context, POINT)
    assert evaluation.reason == "training_failure:ValueError"
    assert evaluation.objective == protocol.objective.infeasible_penalty
    assert evaluation.components == ()
    assert evaluation.fit_rmse is None
    assert evaluation.running == (protocol.objective.infeasible_penalty,)


def test_development_scenarios_must_cover_both_posture_classes(
    fixture_dataset: tuple[RecoveryDatasetRecord, SampleSet], tmp_path: Path
) -> None:
    """The four-cell objective is undefined without both posture classes."""
    record, samples = fixture_dataset
    context = make_context(record, samples, scenarios=_scenarios()[:3])
    with pytest.raises(ValueError, match="posture classes"):
        evaluate_recovery_point(_protocol(tmp_path), context, POINT)


def test_an_augmented_point_trains_through_its_formulation(
    fixture_dataset: tuple[RecoveryDatasetRecord, SampleSet], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The contractive formulation binds the family, shared seed bank, and attempt budget into the recipe."""
    record, samples = fixture_dataset
    context = make_context(record, samples)
    protocol = _protocol(tmp_path, formulation="contractive")
    trained = train_recovery_point(protocol, context, AUG_POINT)
    assert not isinstance(trained, str)
    recipe, _model = trained
    augmentation = recipe.training.augmentation
    assert augmentation is not None
    assert augmentation.family == "contractive"
    assert augmentation.seed_bank == protocol.seed_bank
    assert augmentation.attempt_budget == protocol.attempt_factor * 16
    outcomes = replay_queue(samples)
    for _scenario in range(6):
        outcomes.append((crafted(samples, rc=True, error=0.01), DONE))
        outcomes.append((crafted(samples, rc=True, error=0.03), DONE))
    monkeypatch.setattr(recovery_objective, "simulate", FakeSimulate(outcomes))
    evaluation = evaluate_recovery_point(protocol, context, AUG_POINT)
    assert evaluation.feasible
    assert evaluation.point.augmentation == AUG_POINT.augmentation
    assert evaluation.fit_rmse is not None
    assert math.isfinite(evaluation.fit_rmse)


def test_make_recovery_objective_runs_a_sampled_trial(
    fixture_dataset: tuple[RecoveryDatasetRecord, SampleSet], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A sampled Optuna trial trains, evaluates against schedule-matched fakes, and records every attribute."""
    record, samples = fixture_dataset
    context = make_context(record, samples)
    protocol = _protocol(tmp_path)

    def fake(*_args: object, **kwargs: object) -> tuple[RunArrays, Termination]:
        duration = float(kwargs["duration_s"])  # pyright: ignore[reportArgumentType]
        rc = "channels" in kwargs
        arrays = crafted(samples, duration_s=duration, rc=rc, error=0.01 if rc else 0.02)
        rows = round(duration / DT)
        return arrays, completed(duration, rows)

    monkeypatch.setattr(recovery_objective, "simulate", fake)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler(seed=7))
    study.optimize(make_recovery_objective(protocol, context), n_trials=1)
    (trial,) = study.trials
    assert trial.value == pytest.approx(0.5, rel=1e-9)
    assert trial.user_attrs["feasible"] is True
    assert trial.user_attrs["components_evaluated"] == 12
    cells = cast("dict[str, float]", trial.user_attrs["cells"])
    assert set(cells) == {
        "posture_small:pd_v2",
        "posture_small:computed_torque",
        "posture_large:pd_v2",
        "posture_large:computed_torque",
    }
    assert "n_synthetic" not in trial.params
    assert trial.params["warmup_s"] in {0.0, 0.25, 1.0}

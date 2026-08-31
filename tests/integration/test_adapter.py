# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M2-009/M2-010: the generator + tracker adapter runs in skelarm, holds, then generates without a jump."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from skelarm import Skeleton, Task, build_controller, compute_forward_kinematics, integrate_with_limits

from arm_rc_ctrl.config import load_config
from arm_rc_ctrl.controllers.adapter import (
    CONTROLLER_TYPE,
    GeneratorTrackingController,
    LatestTargetReference,
    Phase,
    register_with_skelarm,
)
from arm_rc_ctrl.controllers.contracts import DesiredJointState
from arm_rc_ctrl.controllers.estimator import CausalDerivativeEstimator, EstimatorConfig
from arm_rc_ctrl.controllers.tracking import TrackerConfig
from arm_rc_ctrl.data.normalization import fit_normalization
from arm_rc_ctrl.data.phases import annotate_phases
from arm_rc_ctrl.data.preprocess import preprocess_demonstration
from arm_rc_ctrl.data.records import Preprocessing, RawDemonstrationRecord, load_record
from arm_rc_ctrl.data.samples import SampleSet
from arm_rc_ctrl.data.teacher import plan_reach
from arm_rc_ctrl.experiments.baselines import load_frozen_baseline
from arm_rc_ctrl.rc.esn import EsnConfig, ReadoutConfig, ReservoirConfig
from arm_rc_ctrl.rc.generator import RcTargetGenerator
from arm_rc_ctrl.rc.recipe import DatasetSource, RclibIdentity, create_recipe, write_recipe
from arm_rc_ctrl.rc.teacher_forcing import InputTransform
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import ScenarioConfig, build_skeleton, endpoint_positions, load_scenario
from arm_rc_ctrl.storage import StorageRoot

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
TASK_1A = REPO_ROOT / "configs" / "tasks" / "task_1a.toml"
FIXTURE_SCENARIO = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
FIXTURE_RAW_RECORD = REPO_ROOT / "tests" / "fixtures" / "records" / "raw-20260830-287036d83d46.toml"
FIXTURE_RAW_LOG = REPO_ROOT / "tests" / "fixtures" / "raw" / "demo.sklog.npz"
PREPROCESS = REPO_ROOT / "configs" / "preprocessing" / "default.toml"
SOURCE_ID = "processed-20260830-555555555555"
ESN = EsnConfig(
    reservoir=ReservoirConfig(
        n_neurons=200, spectral_radius=0.9, sparsity=0.9, leak_rate=0.3, input_scaling=0.5, seed=31
    ),
    readout=ReadoutConfig(alpha=1e-6),
)
RCLIB = RclibIdentity.current()


def _planned_samples(scenario: ScenarioConfig) -> SampleSet:
    """The scripted task 1-a demonstration as a canonical dataset (no simulation, no store)."""
    plan = plan_reach(scenario)
    tip = endpoint_positions(scenario, plan.q)
    n = plan.t.shape[0]
    return SampleSet(
        plan.t,
        plan.q,
        plan.dq,
        plan.ddq,
        tip,
        np.gradient(tip, scenario.timing.dt, axis=0),
        np.zeros((n, 2)),
        np.zeros((n, 0)),
        annotate_phases(plan.t, scenario.timing.intervals),
    )


@pytest.fixture(scope="module")
def generator_and_scenario() -> tuple[RcTargetGenerator, ScenarioConfig, SampleSet]:
    """A generator trained on the planned task 1-a demonstration."""
    scenario = load_scenario(TASK_1A)
    samples = _planned_samples(scenario)
    normalization = fit_normalization(
        samples.arrays(), ("q", "dq"), fitted_on=(SOURCE_ID,), training_rows=np.ones(samples.n_samples, dtype=np.bool_)
    )
    recipe, model = create_recipe(
        "adapter-test",
        ESN,
        sources=[DatasetSource(SOURCE_ID, "ab" * 32, "data/records/processed/planned.toml")],
        samples={SOURCE_ID: samples},
        dof=2,
        task_code_dim=0,
        preprocessing=Preprocessing(scenario.timing.dt, "none", {}, "planned"),
        transform=InputTransform.derive("fixed_scale", normalization, fixed_scales={"q": 0.3, "dq": 4.0}),
        rclib=RCLIB,
    )
    assert recipe.fit.rmse < 1e-3
    lower = np.array([link.q_min for link in scenario.robot.links])
    upper = np.array([link.q_max for link in scenario.robot.links])
    estimator = CausalDerivativeEstimator(EstimatorConfig(scenario.timing.dt, velocity_cutoff_hz=20.0), 2)
    return RcTargetGenerator(model, recipe.encoder(), estimator, position_bounds=(lower, upper)), scenario, samples


def _simulate(
    controller: GeneratorTrackingController, scenario: ScenarioConfig, duration_s: float
) -> dict[str, np.ndarray]:
    """Drive skelarm with the adapter and collect every logged channel plus the measured state."""
    dt = scenario.timing.dt
    skeleton = build_skeleton(scenario)
    controller.reset(skeleton)
    lower = np.array([link.q_min for link in scenario.robot.links])
    upper = np.array([link.q_max for link in scenario.robot.links])
    gravity = np.asarray(scenario.robot.gravity, dtype=np.float64)
    rows: dict[str, list[np.ndarray]] = {}
    steps = round(duration_s / dt)
    for step in range(steps + 1):
        t = step * dt
        tau = controller.control(t, skeleton)
        assert np.all(np.isfinite(tau))
        logged = {"t": np.array([t]), "q": skeleton.q.copy(), "dq": skeleton.dq.copy(), "tau": tau}
        logged.update({k: np.asarray(v, dtype=np.float64) for k, v in controller.log_channels().items()})
        for name, value in logged.items():
            rows.setdefault(name, []).append(np.atleast_1d(value))
        integrate_with_limits(skeleton, tau, dt, lower, upper, gravity)
        compute_forward_kinematics(skeleton)
    return {name: np.vstack(values) for name, values in rows.items()}


def test_adapter_holds_then_tracks_generated_targets_with_finite_torque(
    generator_and_scenario: tuple[RcTargetGenerator, ScenarioConfig, SampleSet],
) -> None:
    """Every torque is finite and within limits; the hold posture is kept while priming; every channel is logged."""
    generator, scenario, samples = generator_and_scenario
    gains = load_frozen_baseline("pd")
    boundary = scenario.timing.intervals.prime[1]
    controller = GeneratorTrackingController(generator, gains, scenario.limits.torque, hold_until_s=boundary)
    log = _simulate(controller, scenario, duration_s=scenario.timing.intervals.duration_s)
    t = log["t"][:, 0]
    hold = t < boundary
    assert np.array_equal(log["phase"][hold, 0], np.full(hold.sum(), Phase.HOLD))
    assert np.array_equal(log["phase"][~hold, 0], np.full((~hold).sum(), Phase.GENERATE))
    assert np.allclose(log["q_desired"][hold], scenario.task.initial_q)
    assert not log["dq_desired"][hold].any()
    limits = np.asarray(scenario.limits.torque)
    assert np.all(np.abs(log["tau"]) <= limits + 1e-12)
    assert np.all(np.isfinite(log["esn_state_norm"]))
    assert np.all(np.isfinite(log["q_generated"]))
    lower = np.array([link.q_min for link in scenario.robot.links])
    upper = np.array([link.q_max for link in scenario.robot.links])
    assert np.all((log["q_generated"] >= lower) & (log["q_generated"] <= upper))  # bounds enforced by the generator
    assert log["q_generated"].shape == (samples.n_samples, 2)
    for channel in (
        "esn_input",
        "q_generated",
        "dq_desired_raw",
        "ddq_desired",
        "q_ref",
        "tau_requested",
        "tau_applied",
    ):
        assert channel in log, channel
    # Reach quality of the closed loop is a scientific result evaluated by the nominal runner and its report
    # (M2-012/M2-014), not an adapter contract: with the frozen v1 PD gains this loop does not reach the target.


def test_generation_starts_at_the_boundary_without_a_command_jump(
    generator_and_scenario: tuple[RcTargetGenerator, ScenarioConfig, SampleSet],
) -> None:
    """The first generated target lies within a small distance of the held posture."""
    generator, scenario, _ = generator_and_scenario
    boundary = scenario.timing.intervals.prime[1]
    controller = GeneratorTrackingController(
        generator, load_frozen_baseline("pd"), scenario.limits.torque, hold_until_s=boundary
    )
    assert controller.boundary_jump is None
    log = _simulate(controller, scenario, duration_s=boundary + 0.2)
    t = log["t"][:, 0]
    first = int(np.argmax(t >= boundary))
    assert log["phase"][first - 1, 0] == Phase.HOLD
    assert log["phase"][first, 0] == Phase.GENERATE
    assert controller.boundary_jump is not None
    assert controller.boundary_jump < 5e-3
    assert np.abs(log["q_desired"][first] - log["q_desired"][first - 1]).max() < 5e-3
    assert np.abs(log["dq_desired"][first]).max() < 0.5  # no derivative spike either
    # priming alone does not move the arm
    assert np.abs(log["q"][first] - np.asarray(scenario.task.initial_q)).max() < 1e-3


def test_adapter_requires_reset_and_validates_the_boundary(
    generator_and_scenario: tuple[RcTargetGenerator, ScenarioConfig, SampleSet],
) -> None:
    """control() before reset() and a malformed boundary are errors; the reference returns the latest target."""
    generator, scenario, _ = generator_and_scenario
    controller = GeneratorTrackingController(
        generator, load_frozen_baseline("pd"), scenario.limits.torque, hold_until_s=1.0
    )
    with pytest.raises(RuntimeError, match=r"reset\(\) must be called before control\(\)"):
        controller.control(0.0, build_skeleton(scenario))
    with pytest.raises(ValueError, match="hold_until_s must be finite and non-negative"):
        GeneratorTrackingController(generator, load_frozen_baseline("pd"), scenario.limits.torque, hold_until_s=-1.0)
    reference = LatestTargetReference(DesiredJointState.hold(np.array([0.1, 0.2])))
    q, dq, _ = reference.sample(3.0)
    assert np.array_equal(q, [0.1, 0.2])
    assert not dq.any()
    reference.set(DesiredJointState(np.array([0.3, 0.4]), np.array([1.0, 1.0]), np.zeros(2)))
    assert np.array_equal(reference.sample(0.0)[0], [0.3, 0.4])


def test_registered_builder_constructs_the_adapter_through_skelarm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Skelarm's config-driven build_controller returns the adapter from a recipe resolved through the store."""
    root = tmp_path / "store"
    root.mkdir()
    store = StorageRoot(root, repositories=(REPO_ROOT,))
    raw = load_record(FIXTURE_RAW_RECORD, RawDemonstrationRecord)
    store.path(raw.artifact.payload.uri, mode="write").write_bytes(FIXTURE_RAW_LOG.read_bytes())
    records = tmp_path / "repo"
    (records / "data" / "records" / "processed").mkdir(parents=True)
    processed = preprocess_demonstration(
        FIXTURE_RAW_RECORD,
        FIXTURE_SCENARIO,
        PREPROCESS,
        store=store,
        records_root=records,
        exploratory=True,
        now=datetime(2026, 9, 1, 8, 0, 0, tzinfo=UTC),
    )
    record = processed.record
    assert record.normalization is not None
    small = EsnConfig(
        reservoir=ReservoirConfig(
            n_neurons=30, spectral_radius=0.9, sparsity=0.8, leak_rate=0.5, input_scaling=0.5, seed=2
        ),
        readout=ReadoutConfig(alpha=1e-3),
    )
    recipe, _ = create_recipe(
        "fixture",
        small,
        sources=[
            DatasetSource(
                record.artifact.artifact_id,
                record.artifact.payload.sha256,
                processed.record_file.relative_to(records).as_posix(),
            )
        ],
        samples={record.artifact.artifact_id: processed.samples},
        dof=record.dof,
        task_code_dim=record.task_code_dim,
        preprocessing=record.preprocessing,
        transform=InputTransform.derive("training_std", record.normalization),
        rclib=RCLIB,
    )
    recipe_file = tmp_path / "recipe.toml"
    write_recipe(recipe_file, recipe)
    monkeypatch.setenv("ARM_RC_CTRL_STORAGE_ROOT", str(root))
    assert register_with_skelarm() == CONTROLLER_TYPE
    scenario = load_scenario(FIXTURE_SCENARIO)
    skeleton: Skeleton = build_skeleton(scenario)
    params: dict[str, Any] = {
        "type": CONTROLLER_TYPE,
        "recipe": str(recipe_file),
        "tracker": str(REPO_ROOT / "configs" / "controllers" / "pd.toml"),
        "torque_limits": list(scenario.limits.torque),
        "hold_until_s": scenario.timing.intervals.prime[1],
        "estimator": {"velocity_cutoff_hz": 20.0},
        "records_root": str(records),
    }
    task = Task(type="reaching", target=np.asarray(scenario.task.target, dtype=np.float64))
    controller = build_controller(params, skeleton=skeleton, task=task, dt=scenario.timing.dt)
    assert isinstance(controller, GeneratorTrackingController)
    assert controller.tracker_config == load_config(REPO_ROOT / "configs" / "controllers" / "pd.toml", TrackerConfig)
    controller.reset(skeleton)
    tau = controller.control(0.0, skeleton)
    assert np.all(np.isfinite(tau))
    assert controller.phase == Phase.HOLD
    channels = controller.log_channels()
    assert {"phase", "q_desired", "esn_input", "esn_state_norm", "tau_requested", "tau_applied", "saturation"} <= set(
        channels
    )
    with pytest.raises(ValueError, match="controller params are missing"):
        build_controller({"type": CONTROLLER_TYPE, "recipe": str(recipe_file)}, skeleton=skeleton, task=task)
    cast("Any", None)  # keep the typing import used

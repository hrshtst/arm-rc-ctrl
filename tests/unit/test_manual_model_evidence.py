# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-008: evaluating one model over every scenario, paired against its parent's replay.

Unlike the predecessor's model sweep, which stopped at the first infeasible
pair and marked the rest unexecuted, D6 evaluates every scenario
independently: an unsafe run aborts alone and the next scenario still runs
from a fresh reset, so per-class success rates exist. The evidence is keyed by
the fit together with the conditions, because the same fit under another
protocol is a different experiment.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, cast

import numpy as np

from arm_rc_ctrl.controllers.tracking import TrackerConfig
from arm_rc_ctrl.data.manual_scenario import load_manual_scenario, manual_endpoint_positions
from arm_rc_ctrl.experiments.manual_evaluation import (
    ManualEvaluationConfig,
    ManualEvaluationRunner,
    ManualModelEvidence,
    load_manual_evaluation_config,
)
from arm_rc_ctrl.experiments.perturbations import RobustnessScenario
from arm_rc_ctrl.experiments.run_record import RunArrays
from arm_rc_ctrl.experiments.termination import completed, limit_violation
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import joint_target

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from arm_rc_ctrl.data.manual_scenario import ManualScenarioConfig
    from arm_rc_ctrl.experiments.manual_evaluation import SimulateFn
    from arm_rc_ctrl.experiments.manual_fixture import ManualFixture
    from arm_rc_ctrl.experiments.manual_study import StudyModel
    from arm_rc_ctrl.experiments.termination import Termination

REPO_ROOT = repository_root()
DEVELOPMENT_SOURCE = REPO_ROOT / "configs" / "evaluations" / "task_1a_recovery_dev_v1.toml"
TRACKER = TrackerConfig(type="pd", kp=(10.0, 5.0), kd=(1.5, 0.8))

HOLD_S, PULSE_S, HORIZON_S, WARMUP_S = 0.05, 0.02, 1.0, 0.25
SWEEP_WARMUP_S = WARMUP_S + 0.75
"""A warm-up of its own for the sweep test: completed evidence is served rather than re-simulated,
so a test that counts simulations must not depend on what an earlier test left in the shared store."""
CONFIGURATION = "feasible-best"
ARM_LABEL = "S/D01"
FAILING = "small-1"
"""The one scenario the crafted simulator aborts; every later scenario must still run."""

SCENARIOS = (
    RobustnessScenario("nominal", "nominal", (0.0, 0.0)),
    RobustnessScenario(FAILING, "posture_small", (0.02, -0.01), seed=1, draw=0, magnitude_rad=0.05),
    RobustnessScenario("large-1", "posture_large", (0.05, -0.02), seed=2, draw=0, magnitude_rad=0.1),
)


def _evaluation(f: ManualFixture) -> tuple[ManualEvaluationConfig, Path]:
    """A manual evaluation configuration inside the fixture root (written identically on every call)."""
    evaluations = f.root / "configs" / "evaluations"
    evaluations.mkdir(parents=True, exist_ok=True)
    development = evaluations / DEVELOPMENT_SOURCE.name
    shutil.copyfile(DEVELOPMENT_SOURCE, development)
    scenario = f.scenario_file
    limits = ", ".join(f"{v}" for v in load_manual_scenario(scenario).limits.velocity)
    target = evaluations / "task_1a_manual_dev_fixture.toml"
    target.write_text(
        f'name = "task-1a-manual-dev-fixture"\n'
        f'development = "{development.as_posix()}"\n'
        f'scenario = "{scenario.as_posix()}"\n'
        f"horizon_s = {HORIZON_S}\n\n"
        f"[trigger]\nhold_s = {HOLD_S}\nduration_s = {PULSE_S}\nmagnitude_n = 3.0\n\n"
        f"[simulation]\nvelocity_abort = [{limits}]\n",
        encoding="utf-8",
    )
    return load_manual_evaluation_config(target), target


class _CraftedSimulator:
    """Returns feasible runs, aborting only the scenario whose start matches ``FAILING``.

    The fixture arm cannot sustain a feasible real closed loop, so the sweep's
    own logic -- what runs, what is recorded, what continues after a failure --
    is tested on crafted runs, while the replay bank's tests cover the real
    simulator.
    """

    def __init__(self, scenario_config: ManualScenarioConfig, failing_start: tuple[float, ...]) -> None:
        self.scenario = scenario_config
        self.failing_start = failing_start
        self.calls: list[tuple[float, ...]] = []

    def __call__(self, scenario: object, controller: object, **kwargs: object) -> tuple[RunArrays, Termination]:
        del scenario, controller
        start = tuple(float(v) for v in cast("tuple[float, ...]", kwargs["initial_q"]))
        self.calls.append(start)
        rows = round(cast("float", kwargs["duration_s"]) / self.scenario.timing.dt) + 1
        rc = kwargs.get("channels") is not None
        aborted = np.allclose(start, self.failing_start)
        n = rows // 2 if aborted else rows
        arrays = _crafted(self.scenario, n, start, rc=rc)
        t_last = float(arrays.arrays["t"][-1])
        termination = (
            limit_violation(t_last, n - 1, "joint_velocity", 99.0, 20.0, joint=0)
            if aborted
            else completed(t_last, n - 1)
        )
        return arrays, termination


def _crafted(scenario_config: ManualScenarioConfig, rows: int, start: tuple[float, ...], *, rc: bool) -> RunArrays:
    """A run holding the target from a quarter of the way in, with a readout when it is an RC run."""
    dt = scenario_config.timing.dt
    on_target = np.asarray(joint_target(scenario_config), dtype=np.float64)
    t: NDArray[np.float64] = np.arange(rows, dtype=np.float64) * dt
    q = np.tile(np.asarray(start, dtype=np.float64), (rows, 1))
    hold_from = max(1, rows // 4)
    q[hold_from:] = on_target
    dq = np.zeros((rows, 2), dtype=np.float64)
    zeros = np.zeros((rows, 2), dtype=np.float64)
    data: dict[str, NDArray[np.float64] | NDArray[np.int64]] = {
        "t": t,
        "q": q,
        "dq": dq,
        "tip": manual_endpoint_positions(scenario_config, q),
        "q_desired": q.copy(),
        "dq_desired": dq.copy(),
        "dq_desired_raw": dq.copy(),
        "ddq_desired": zeros.copy(),
        "ddq_desired_raw": zeros.copy(),
        "tracking_error": zeros.copy(),
        "tau_requested": zeros.copy(),
        "task_code": np.zeros((rows, 0), dtype=np.float64),
        "saturation": np.zeros(rows, dtype=np.int64),
    }
    if rc:
        active = t >= WARMUP_S - 1e-9
        readout = np.full((rows, 2), np.nan, dtype=np.float64)
        readout[active] = q[active]
        data["generator_output_q"] = readout
        data["phase"] = active.astype(np.int64)
    return RunArrays(dict(data))


def _entry(f: ManualFixture) -> StudyModel:
    return next(e for e in f.manifest.entries if (e.configuration, e.arm.label) == (CONFIGURATION, ARM_LABEL))


def _runner(f: ManualFixture, *, simulate_fn: SimulateFn | None = None) -> ManualEvaluationRunner:
    config, file = _evaluation(f)
    return ManualEvaluationRunner(
        store=f.store,
        inputs=f.inputs,
        config=config,
        evaluation_file=file,
        scenarios=SCENARIOS,
        trackers={"pd_v2": TRACKER, "computed_torque": TRACKER},
        root=f.root,
        execution=f.execution,
        provenance=f.provenance,
        simulate_fn=simulate_fn,
    )


def _crafted_runner(f: ManualFixture) -> tuple[ManualEvaluationRunner, _CraftedSimulator]:
    scenario_config = load_manual_scenario(f.scenario_file)
    failing = next(c for c in SCENARIOS if c.scenario_id == FAILING).initial_q(scenario_config.task.initial_q)
    crafted = _CraftedSimulator(scenario_config, failing)
    return _runner(f, simulate_fn=crafted), crafted


# --- what one model's evidence contains ------------------------------------------------------


def test_a_model_is_evaluated_over_every_scenario_and_tracker(manual_fixture: ManualFixture) -> None:
    """Every case under both trackers, in evaluation order, each carrying its own RC run."""
    runner, _ = _crafted_runner(manual_fixture)
    evidence = runner.evaluate(_entry(manual_fixture), warmup_s=WARMUP_S)
    assert isinstance(evidence, ManualModelEvidence)
    assert [(p.scenario_id, p.tracker) for p in evidence.pairs] == list(evidence.conditions.pairs)
    assert all(p.arm == "rc" for p in evidence.pairs)
    assert evidence.n_pairs == len(SCENARIOS) * 2


def test_an_infeasible_scenario_does_not_stop_the_sweep(manual_fixture: ManualFixture) -> None:
    """D6: the unsafe run aborts alone and every later scenario is still attempted from a fresh reset."""
    runner, crafted = _crafted_runner(manual_fixture)
    evidence = runner.evaluate(_entry(manual_fixture), warmup_s=SWEEP_WARMUP_S)
    failed = [p for p in evidence.pairs if p.scenario_id == FAILING]
    later = [p for p in evidence.pairs if p.scenario_id == "large-1"]
    assert all(p.status == "infeasible" for p in failed)
    assert all(p.status == "completed" for p in later)
    assert evidence.n_unexecuted == 0
    assert evidence.n_infeasible == len(failed)
    # Every scenario was actually simulated, including the ones after the failure.
    assert len(crafted.calls) >= len(SCENARIOS) * 2


def test_the_model_identity_binds_the_fit_and_the_conditions(manual_fixture: ManualFixture) -> None:
    """The same fit under another protocol keys different evidence; under the same protocol, the same."""
    runner, _ = _crafted_runner(manual_fixture)
    entry = _entry(manual_fixture)
    first = runner.evaluate(entry, warmup_s=WARMUP_S)
    again = runner.evaluate(entry, warmup_s=WARMUP_S)
    other = runner.evaluate(entry, warmup_s=WARMUP_S + 0.25)
    assert first.identity == again.identity
    assert first.identity != other.identity
    assert first.fit is not None
    assert first.fit.identity == entry.fit_identity


def test_each_pair_names_the_replay_baseline_it_is_compared_against(manual_fixture: ManualFixture) -> None:
    """A model is paired against replay of its own parent, so the bank it used is recorded with it."""
    runner, _ = _crafted_runner(manual_fixture)
    entry = _entry(manual_fixture)
    evidence = runner.evaluate(entry, warmup_s=WARMUP_S)
    bank = runner.replay_bank(
        entry.arm.assignment or "D01", warmup_s=WARMUP_S, replay_cutoffs=runner.replay_cutoffs(entry)
    )
    assert evidence.replay_bank == bank.identity
    assert evidence.assignment == entry.arm.assignment

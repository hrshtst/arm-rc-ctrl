# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-009: what an interrupted sweep records, and what it must never invent.

Resuming is covered elsewhere. What matters before the full execution is the
shape of the record an interruption leaves behind: the runs that never happened
are absent from it, and they never become recorded not-executed pairs. A
not-executed pair would be a claim that the sweep considered a scenario and
declined it, which is exactly what D6 says it does not do -- every scenario is
attempted from a fresh reset, so nothing is skipped by an earlier failure.

The accounting must say the same thing from outside: a model interrupted
partway has no manifest, so it is missing, not present-with-gaps.
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from arm_rc_ctrl.controllers.tracking import TrackerConfig
from arm_rc_ctrl.data.manual_scenario import load_manual_scenario, manual_endpoint_positions
from arm_rc_ctrl.experiments.manual_accounting import account_study
from arm_rc_ctrl.experiments.manual_evaluation import (
    PROGRESS_FILE,
    ManualEvaluationConfig,
    ManualEvaluationRunner,
    load_manual_evaluation_config,
    model_uri,
)
from arm_rc_ctrl.experiments.perturbations import RobustnessScenario
from arm_rc_ctrl.experiments.run_record import RunArrays
from arm_rc_ctrl.experiments.termination import completed
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
HOLD_S, PULSE_S, HORIZON_S = 0.05, 0.02, 1.0
CONFIGURATION, ARM_LABEL = "feasible-best", "S/D01"

WARMUP_PARTIAL = 7.0
WARMUP_ACCOUNTED = 7.25
WARMUP_RAISED = 7.5
"""A warm-up of its own per test, so each starts from a store holding no evidence of its protocol."""

SCENARIOS = (
    RobustnessScenario("nominal", "nominal", (0.0, 0.0)),
    RobustnessScenario("small-1", "posture_small", (0.02, -0.01), seed=1, draw=0, magnitude_rad=0.05),
    RobustnessScenario("large-1", "posture_large", (0.05, -0.02), seed=2, draw=0, magnitude_rad=0.1),
)


def _evaluation(f: ManualFixture) -> tuple[ManualEvaluationConfig, Path]:
    """The fixture's own evaluation configuration (written identically on every call)."""
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


def _crafted(scenario_config: ManualScenarioConfig, rows: int, start: tuple[float, ...], *, rc: bool) -> RunArrays:
    """A feasible run that reaches the target early and holds it."""
    dt = scenario_config.timing.dt
    on_target = np.asarray(joint_target(scenario_config), dtype=np.float64)
    t: NDArray[np.float64] = np.arange(rows, dtype=np.float64) * dt
    q = np.tile(np.asarray(start, dtype=np.float64), (rows, 1))
    q[max(1, rows // 4) :] = on_target
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
        active = t >= 0.0
        readout = np.full((rows, 2), np.nan, dtype=np.float64)
        readout[active] = q[active]
        data["generator_output_q"] = readout
        data["phase"] = active.astype(np.int64)
    return RunArrays(data)


class _Stopping:
    """Crafts feasible runs and stops the sweep after ``stop_after`` RC runs, by interrupt or by error."""

    def __init__(
        self, scenario_config: ManualScenarioConfig, *, stop_after: int | None = None, error: bool = False
    ) -> None:
        self.scenario = scenario_config
        self.stop_after = stop_after
        self.error = error
        self.rc_calls = 0
        self.replay_calls = 0

    def __call__(self, scenario: object, controller: object, **kwargs: object) -> tuple[RunArrays, Termination]:
        del scenario, controller
        rc = kwargs.get("channels") is not None
        if rc and self.stop_after is not None and self.rc_calls >= self.stop_after:
            msg = "the simulator failed"
            if self.error:
                raise RuntimeError(msg)
            raise KeyboardInterrupt(msg)
        if rc:
            self.rc_calls += 1
        else:
            self.replay_calls += 1
        rows = round(cast("float", kwargs["duration_s"]) / self.scenario.timing.dt) + 1
        start = tuple(float(v) for v in cast("tuple[float, ...]", kwargs["initial_q"]))
        arrays = _crafted(self.scenario, rows, start, rc=rc)
        return arrays, completed(float(arrays.arrays["t"][-1]), rows - 1)


def _entry(f: ManualFixture) -> StudyModel:
    return next(e for e in f.manifest.entries if (e.configuration, e.arm.label) == (CONFIGURATION, ARM_LABEL))


def _runner(f: ManualFixture, simulate_fn: SimulateFn) -> ManualEvaluationRunner:
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


def _progress(
    f: ManualFixture, runner: ManualEvaluationRunner, entry: StudyModel, warmup_s: float
) -> dict[str, object]:
    """The progress an interrupted sweep left for one model."""
    identity = runner.model_identity(entry, warmup_s=warmup_s)
    path = f.store.path(f"{model_uri(identity)}/{PROGRESS_FILE}", mode="write")
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))


# --- what the record holds, and what it does not ------------------------------------------------


def test_the_runs_that_never_happened_are_absent_not_recorded_as_unexecuted(
    manual_fixture: ManualFixture,
) -> None:
    """An interruption leaves fewer pairs, never pairs claiming the sweep declined to run them.

    Recording a not-executed pair would assert that the scenario was considered
    and skipped. D6 says the opposite: every scenario is attempted from a fresh
    reset, so a run that did not happen is simply not in the record yet.
    """
    f = manual_fixture
    scenario_config = load_manual_scenario(f.scenario_file)
    entry = _entry(f)
    stopped = _Stopping(scenario_config, stop_after=2)
    runner = _runner(f, stopped)
    with pytest.raises(KeyboardInterrupt):
        runner.evaluate(entry, warmup_s=WARMUP_PARTIAL)
    document = _progress(f, runner, entry, WARMUP_PARTIAL)
    pairs = cast("list[dict[str, object]]", document["pairs"])
    assert len(pairs) == 2, "only the runs that completed are recorded"
    assert len(pairs) < len(SCENARIOS) * 2, "the sweep really was cut short"
    assert {pair["status"] for pair in pairs} == {"completed"}
    assert not any(pair["status"] == "unexecuted" for pair in pairs)


def test_an_interrupted_model_is_missing_from_the_accounting_not_present_with_gaps(
    manual_fixture: ManualFixture, tmp_path: Path
) -> None:
    """From outside, a model with no manifest is missing; a partial execution reads as partial."""
    f = manual_fixture
    scenario_config = load_manual_scenario(f.scenario_file)
    entry = _entry(f)
    runner = _runner(f, _Stopping(scenario_config, stop_after=1))
    with pytest.raises(KeyboardInterrupt):
        runner.evaluate(entry, warmup_s=WARMUP_ACCOUNTED)
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    runner.write_pointers(evidence_dir)
    accounting = account_study(store=f.store, evidence_dir=evidence_dir, manifest=f.manifest, provenance=f.provenance)
    line = next(line for line in accounting.models if line.label == f"{CONFIGURATION}/{ARM_LABEL}")
    assert line.present is False
    assert line.n_unexecuted == 0, "a missing model reports nothing, not gaps"
    assert line.label in accounting.missing
    assert accounting.complete is False


def test_a_failing_run_leaves_recoverable_progress(manual_fixture: ManualFixture) -> None:
    """A run that raises is not different from an interruption: what succeeded is kept and resuming finishes."""
    f = manual_fixture
    scenario_config = load_manual_scenario(f.scenario_file)
    entry = _entry(f)
    failing = _Stopping(scenario_config, stop_after=2, error=True)
    with pytest.raises(RuntimeError, match="the simulator failed"):
        _runner(f, failing).evaluate(entry, warmup_s=WARMUP_RAISED)
    recovered = _Stopping(scenario_config)
    evidence = _runner(f, recovered).evaluate(entry, warmup_s=WARMUP_RAISED)
    assert evidence.n_pairs == len(SCENARIOS) * 2
    assert evidence.n_completed == evidence.n_pairs
    assert evidence.n_unexecuted == 0
    assert recovered.rc_calls == evidence.n_pairs - 2, "only the runs the failure left undone were simulated again"
    assert recovered.replay_calls == 0, "the replay baselines were already complete and stored"

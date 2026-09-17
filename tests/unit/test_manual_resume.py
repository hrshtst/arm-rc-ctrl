# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-008: a sweep interrupted partway resumes at run granularity.

The full execution is tens of thousands of runs, so an interruption must not
cost the runs already paid for, and a resumed sweep must produce exactly what
an uninterrupted one would. Completed runs are re-verified against the store
before they are trusted: a progress file from another protocol, or a run whose
stored arrays no longer match what the record claims, is refused rather than
served.
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from arm_rc_ctrl.config import to_mapping
from arm_rc_ctrl.controllers.tracking import TrackerConfig
from arm_rc_ctrl.data.manual_scenario import load_manual_scenario, manual_endpoint_positions
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
CONFIGURATION, ARM_LABEL = "feasible-best", "S/D02"

WARMUP_RESUME = 2.0
WARMUP_WHOLE = 2.25
WARMUP_BROKEN = 2.5
WARMUP_FOREIGN = 2.75
WARMUP_CORRUPT = 3.0
"""A warm-up of its own per test.

Completed evidence is served rather than recomputed, so each test needs a cold
store to exercise the path it is about. The comparison test deliberately uses
two protocols for the same reason: run under one warm-up, an uninterrupted
sweep and a resumed one would be the same stored experiment, and the second
would prove nothing.
"""

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
        active = t >= (rows // 4) * dt - 1e-9
        readout = np.full((rows, 2), np.nan, dtype=np.float64)
        readout[active] = q[active]
        data["generator_output_q"] = readout
        data["phase"] = active.astype(np.int64)
    return RunArrays(data)


class _Interrupting:
    """Crafts feasible runs, counting them, and stops the sweep after ``interrupt_after`` RC runs."""

    def __init__(self, scenario_config: ManualScenarioConfig, *, interrupt_after: int | None = None) -> None:
        self.scenario = scenario_config
        self.interrupt_after = interrupt_after
        self.rc_calls = 0
        self.replay_calls = 0

    @property
    def calls(self) -> int:
        """Every run this simulator produced, of either arm."""
        return self.rc_calls + self.replay_calls

    def __call__(self, scenario: object, controller: object, **kwargs: object) -> tuple[RunArrays, Termination]:
        del scenario, controller
        rc = kwargs.get("channels") is not None
        if rc and self.interrupt_after is not None and self.rc_calls >= self.interrupt_after:
            msg = "interrupted"
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


# --- resuming ---------------------------------------------------------------------------------


def test_an_interrupted_sweep_keeps_its_completed_runs_and_resumes(manual_fixture: ManualFixture) -> None:
    """The runs already paid for are kept: resuming simulates only what the interruption left undone."""
    scenario_config = load_manual_scenario(manual_fixture.scenario_file)
    entry = _entry(manual_fixture)
    stopped = _Interrupting(scenario_config, interrupt_after=3)
    with pytest.raises(KeyboardInterrupt):
        _runner(manual_fixture, stopped).evaluate(entry, warmup_s=WARMUP_RESUME)
    finished = stopped.rc_calls
    assert finished == 3
    resumed = _Interrupting(scenario_config)
    evidence = _runner(manual_fixture, resumed).evaluate(entry, warmup_s=WARMUP_RESUME)
    assert evidence.n_pairs == len(SCENARIOS) * 2
    assert evidence.n_completed == evidence.n_pairs
    # Exactly the model pairs the interruption left undone were simulated again, and the replay
    # baselines, already complete and stored, were not touched at all.
    assert resumed.rc_calls == evidence.n_pairs - finished
    assert resumed.replay_calls == 0


def test_a_resumed_sweep_matches_an_uninterrupted_one(manual_fixture: ManualFixture) -> None:
    """Resuming is not a different experiment: the evidence is what an unbroken sweep would have produced."""
    scenario_config = load_manual_scenario(manual_fixture.scenario_file)
    entry = _entry(manual_fixture)
    whole = _runner(manual_fixture, _Interrupting(scenario_config)).evaluate(entry, warmup_s=WARMUP_WHOLE)
    stopped = _Interrupting(scenario_config, interrupt_after=2)
    with pytest.raises(KeyboardInterrupt):
        _runner(manual_fixture, stopped).evaluate(entry, warmup_s=WARMUP_BROKEN)
    resumed = _runner(manual_fixture, _Interrupting(scenario_config)).evaluate(entry, warmup_s=WARMUP_BROKEN)
    mapped_whole = to_mapping(whole)
    mapped_resumed = to_mapping(resumed)
    for key in ("n_pairs", "n_completed", "n_infeasible", "n_unexecuted", "status"):
        assert mapped_resumed[key] == mapped_whole[key]
    assert [p.scenario_id for p in resumed.pairs] == [p.scenario_id for p in whole.pairs]


def test_progress_from_another_protocol_is_refused(manual_fixture: ManualFixture) -> None:
    """A progress file belonging to other conditions is a different experiment, not a resumable one."""
    scenario_config = load_manual_scenario(manual_fixture.scenario_file)
    entry = _entry(manual_fixture)
    runner = _runner(manual_fixture, _Interrupting(scenario_config, interrupt_after=1))
    with pytest.raises(KeyboardInterrupt):
        runner.evaluate(entry, warmup_s=WARMUP_FOREIGN)
    identity = runner.model_identity(entry, warmup_s=WARMUP_FOREIGN)
    path = manual_fixture.store.path(f"{model_uri(identity)}/{PROGRESS_FILE}", mode="write")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["identity"] = "f" * 64
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="another evaluation"):
        _runner(manual_fixture, _Interrupting(scenario_config)).evaluate(entry, warmup_s=WARMUP_FOREIGN)


def test_a_completed_run_whose_arrays_changed_is_refused(manual_fixture: ManualFixture) -> None:
    """A stored run is re-verified before it is trusted; a changed payload is never served as evidence."""
    scenario_config = load_manual_scenario(manual_fixture.scenario_file)
    entry = _entry(manual_fixture)
    runner = _runner(manual_fixture, _Interrupting(scenario_config, interrupt_after=2))
    with pytest.raises(KeyboardInterrupt):
        runner.evaluate(entry, warmup_s=WARMUP_CORRUPT)
    identity = runner.model_identity(entry, warmup_s=WARMUP_CORRUPT)
    path = manual_fixture.store.path(f"{model_uri(identity)}/{PROGRESS_FILE}", mode="write")
    document = json.loads(path.read_text(encoding="utf-8"))
    pairs = cast("list[dict[str, object]]", document["pairs"])
    recorded = next(p for p in pairs if p.get("run") is not None)
    run_uri = cast("str", cast("dict[str, object]", recorded["run"])["uri"])
    stored = manual_fixture.store.path(run_uri, mode="write")
    stored.write_bytes(stored.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="no longer matches"):
        _runner(manual_fixture, _Interrupting(scenario_config)).evaluate(entry, warmup_s=WARMUP_CORRUPT)

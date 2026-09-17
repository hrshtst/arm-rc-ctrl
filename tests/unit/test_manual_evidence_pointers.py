# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-008: the Git-tracked pointers to the sweep's stored evidence.

Payloads live in the external store; Git holds portable pointers to them. A
pointer names what was produced, where it lives, and its digest, so the
evidence a report cites can be found and verified from the repository alone.
Pointers are immutable: writing the same one again is a no-op, and writing a
different one under the same name is refused.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from arm_rc_ctrl.controllers.tracking import TrackerConfig
from arm_rc_ctrl.data.manual_scenario import load_manual_scenario, manual_endpoint_positions
from arm_rc_ctrl.experiments.manual_evaluation import (
    ManualEvaluationConfig,
    ManualEvaluationRunner,
    load_manual_evaluation_config,
    load_manual_pointer,
    manual_pointer_name,
)
from arm_rc_ctrl.experiments.manual_study import EXPERIMENT_LABEL
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
CONFIGURATION, ARM_LABEL = "feasible-best", "S/D05"

WARMUP_POINTERS = 3.25
WARMUP_IDEMPOTENT = 3.5
WARMUP_CONFLICT = 3.75
"""A warm-up of its own per test, so each starts from a store with no evidence of its own kind."""

SCENARIOS = (
    RobustnessScenario("nominal", "nominal", (0.0, 0.0)),
    RobustnessScenario("small-1", "posture_small", (0.02, -0.01), seed=1, draw=0, magnitude_rad=0.05),
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


class _Feasible:
    """Crafts feasible runs for either arm."""

    def __init__(self, scenario_config: ManualScenarioConfig) -> None:
        self.scenario = scenario_config

    def __call__(self, scenario: object, controller: object, **kwargs: object) -> tuple[RunArrays, Termination]:
        del scenario, controller
        rows = round(cast("float", kwargs["duration_s"]) / self.scenario.timing.dt) + 1
        start = tuple(float(v) for v in cast("tuple[float, ...]", kwargs["initial_q"]))
        arrays = _crafted(self.scenario, rows, start, rc=kwargs.get("channels") is not None)
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


def _evaluated(f: ManualFixture, warmup_s: float) -> ManualEvaluationRunner:
    """A runner that has produced one model's evidence and the replay bank it was paired against."""
    runner = _runner(f, _Feasible(load_manual_scenario(f.scenario_file)))
    runner.evaluate(_entry(f), warmup_s=warmup_s)
    return runner


# --- what a completed sweep points at ----------------------------------------------------------


def test_a_completed_sweep_points_at_its_model_and_its_replay_bank(
    manual_fixture: ManualFixture, tmp_path: Path
) -> None:
    """Both manifests get a pointer: the model's evidence and the baselines it was compared against."""
    runner = _evaluated(manual_fixture, WARMUP_POINTERS)
    written = runner.write_pointers(tmp_path)
    assert len(written) == 2
    pointers = {path: load_manual_pointer(path) for path in written}
    assert sorted(p.kind for p in pointers.values()) == ["model", "replay"]
    for path, pointer in pointers.items():
        assert pointer.experiment == EXPERIMENT_LABEL
        assert path.name == manual_pointer_name(pointer.kind, pointer.label)
        assert len(pointer.identity) == 64
        assert pointer.payload.sha256
        # The pointer names a payload that really is in the store, at the digest it claims.
        stored = manual_fixture.store.path(pointer.payload.uri, mode="read")
        assert stored.is_file()


def test_a_model_pointer_names_the_label_and_counts_of_its_evidence(
    manual_fixture: ManualFixture, tmp_path: Path
) -> None:
    """A reader can tell from the pointer alone which model this is and how it came out."""
    runner = _evaluated(manual_fixture, WARMUP_POINTERS)
    written = runner.write_pointers(tmp_path)
    model = next(p for p in (load_manual_pointer(path) for path in written) if p.kind == "model")
    assert model.label == f"{CONFIGURATION}/{ARM_LABEL}"
    assert model.status == "feasible"
    assert model.n_pairs == len(SCENARIOS) * 2
    assert model.n_completed == model.n_pairs


# --- pointers are immutable --------------------------------------------------------------------


def test_writing_the_same_pointers_again_changes_nothing(manual_fixture: ManualFixture, tmp_path: Path) -> None:
    """Re-running the command over finished work is a no-op, not an error and not a rewrite."""
    runner = _evaluated(manual_fixture, WARMUP_IDEMPOTENT)
    first = runner.write_pointers(tmp_path)
    before = {path: path.read_bytes() for path in first}
    again = runner.write_pointers(tmp_path)
    assert again == []
    assert {path: path.read_bytes() for path in first} == before


def test_a_different_pointer_under_the_same_name_is_refused(manual_fixture: ManualFixture, tmp_path: Path) -> None:
    """Completed evidence is never overwritten: a conflicting pointer is an error, not a silent replacement."""
    runner = _evaluated(manual_fixture, WARMUP_CONFLICT)
    written = runner.write_pointers(tmp_path)
    target = next(path for path in written if load_manual_pointer(path).kind == "model")
    text = target.read_text(encoding="utf-8")
    assert '"feasible"' in text
    target.write_text(text.replace('"feasible"', '"infeasible"'), encoding="utf-8")
    assert load_manual_pointer(target).status == "infeasible"
    with pytest.raises(FileExistsError, match="another pointer"):
        runner.write_pointers(tmp_path)

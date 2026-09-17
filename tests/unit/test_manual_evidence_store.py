# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-008: the sweep's evidence is written once, served again, and rebuilt strictly.

A completed bank or model manifest is immutable evidence: a second sweep under
the same protocol must serve what is stored rather than simulate again, and a
manifest must rebuild from its own JSON without losing a field. Both matter
before the full execution, where re-simulating would be both expensive and a
different experiment.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, cast

import numpy as np

from arm_rc_ctrl.config import to_mapping
from arm_rc_ctrl.controllers.tracking import TrackerConfig
from arm_rc_ctrl.data.manual_scenario import load_manual_scenario, manual_endpoint_positions
from arm_rc_ctrl.experiments.manual_evaluation import (
    ManualEvaluationConfig,
    ManualEvaluationRunner,
    ManualModelEvidence,
    ManualReplayBank,
    load_manual_evaluation_config,
    load_manual_model_evidence,
    load_manual_replay_bank,
    manual_bank_to_json,
    manual_evidence_to_json,
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
HOLD_S, PULSE_S, HORIZON_S, WARMUP_S = 0.05, 0.02, 1.0, 0.25
CONFIGURATION, ARM_LABEL = "feasible-best", "S/D01"

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


class _CountingSimulator:
    """Crafts feasible runs and counts how many times it was asked to simulate."""

    def __init__(self, scenario_config: ManualScenarioConfig) -> None:
        self.scenario = scenario_config
        self.calls = 0

    def __call__(self, scenario: object, controller: object, **kwargs: object) -> tuple[RunArrays, Termination]:
        del scenario, controller
        self.calls += 1
        rows = round(cast("float", kwargs["duration_s"]) / self.scenario.timing.dt) + 1
        start = tuple(float(v) for v in cast("tuple[float, ...]", kwargs["initial_q"]))
        arrays = _crafted(self.scenario, rows, start, rc=kwargs.get("channels") is not None)
        return arrays, completed(float(arrays.arrays["t"][-1]), rows - 1)


def _crafted(scenario_config: ManualScenarioConfig, rows: int, start: tuple[float, ...], *, rc: bool) -> RunArrays:
    """A run that reaches the target early and holds it to the last sample."""
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
        active = t >= WARMUP_S - 1e-9
        readout = np.full((rows, 2), np.nan, dtype=np.float64)
        readout[active] = q[active]
        data["generator_output_q"] = readout
        data["phase"] = active.astype(np.int64)
    return RunArrays(data)


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


def _rebuilt_bank(tmp_path: Path, bank: ManualReplayBank) -> ManualReplayBank:
    """Write the bank's manifest and read it back through the loader the sweep itself uses."""
    path = tmp_path / "bank.json"
    path.write_text(manual_bank_to_json(bank), encoding="utf-8")
    return load_manual_replay_bank(path)


def _rebuilt_evidence(tmp_path: Path, evidence: ManualModelEvidence) -> ManualModelEvidence:
    """Write the model manifest and read it back through the shipped loader."""
    path = tmp_path / "model.json"
    path.write_text(manual_evidence_to_json(evidence), encoding="utf-8")
    return load_manual_model_evidence(path)


# --- written once, served again ---------------------------------------------------------------


def test_a_stored_bank_is_served_without_simulating_again(manual_fixture: ManualFixture) -> None:
    """A completed bank is immutable evidence: a fresh runner over the same store reuses it."""
    scenario_config = load_manual_scenario(manual_fixture.scenario_file)
    first_sim = _CountingSimulator(scenario_config)
    first = _runner(manual_fixture, first_sim).replay_bank("D03", warmup_s=WARMUP_S)
    assert first_sim.calls == len(SCENARIOS) * 2
    second_sim = _CountingSimulator(scenario_config)
    second = _runner(manual_fixture, second_sim).replay_bank("D03", warmup_s=WARMUP_S)
    assert second_sim.calls == 0
    assert second.identity == first.identity
    assert [(p.scenario_id, p.tracker) for p in second.pairs] == [(p.scenario_id, p.tracker) for p in first.pairs]


def test_stored_model_evidence_is_served_without_simulating_again(manual_fixture: ManualFixture) -> None:
    """The same model under the same protocol is one experiment, run once."""
    scenario_config = load_manual_scenario(manual_fixture.scenario_file)
    entry = _entry(manual_fixture)
    first_sim = _CountingSimulator(scenario_config)
    first = _runner(manual_fixture, first_sim).evaluate(entry, warmup_s=WARMUP_S)
    ran = first_sim.calls
    assert ran > 0
    second_sim = _CountingSimulator(scenario_config)
    second = _runner(manual_fixture, second_sim).evaluate(entry, warmup_s=WARMUP_S)
    assert second_sim.calls == 0
    assert second.identity == first.identity
    assert second.n_completed == first.n_completed


# --- rebuilt strictly from their own JSON ------------------------------------------------------


def test_a_bank_rebuilds_from_its_manifest(manual_fixture: ManualFixture, tmp_path: Path) -> None:
    """Every field survives the round trip, including the ones that are unset on a replay run."""
    scenario_config = load_manual_scenario(manual_fixture.scenario_file)
    bank = _runner(manual_fixture, _CountingSimulator(scenario_config)).replay_bank("D04", warmup_s=WARMUP_S)
    rebuilt = _rebuilt_bank(tmp_path, bank)
    assert to_mapping(rebuilt) == to_mapping(bank)
    assert rebuilt.identity == bank.identity


def test_model_evidence_rebuilds_from_its_manifest(manual_fixture: ManualFixture, tmp_path: Path) -> None:
    """The nested verdicts survive too: the dwell reports, the generated reference, and the trigger."""
    scenario_config = load_manual_scenario(manual_fixture.scenario_file)
    entry = _entry(manual_fixture)
    evidence = _runner(manual_fixture, _CountingSimulator(scenario_config)).evaluate(entry, warmup_s=WARMUP_S)
    rebuilt = _rebuilt_evidence(tmp_path, evidence)
    assert to_mapping(rebuilt) == to_mapping(evidence)
    assert rebuilt.identity == evidence.identity

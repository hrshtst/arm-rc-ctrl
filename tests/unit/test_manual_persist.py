# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-008: what a persisted run records about the model that produced it.

Two properties the sweep depends on and that the pair record alone does not
guarantee: a run names every demonstration its model trained on, and the stored
summary's verdict is the same verdict the pair record carries.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, cast

import numpy as np

from arm_rc_ctrl.controllers.tracking import TrackerConfig
from arm_rc_ctrl.data.manual_scenario import load_manual_scenario, manual_endpoint_positions
from arm_rc_ctrl.experiments.manual_evaluation import (
    REPORTS_PREFIX,
    ManualEvaluationConfig,
    ManualEvaluationRunner,
    load_manual_evaluation_config,
    load_manual_model_evidence,
)
from arm_rc_ctrl.experiments.manual_recipes import ASSIGNMENTS
from arm_rc_ctrl.experiments.perturbations import RobustnessScenario
from arm_rc_ctrl.experiments.run_record import RunArrays, RunSummary
from arm_rc_ctrl.experiments.termination import completed
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import joint_target

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from arm_rc_ctrl.data.manual_scenario import ManualScenarioConfig
    from arm_rc_ctrl.experiments.manual_evaluation import ManualModelEvidence
    from arm_rc_ctrl.experiments.manual_fixture import ManualFixture
    from arm_rc_ctrl.experiments.manual_study import StudyModel
    from arm_rc_ctrl.experiments.termination import Termination

REPO_ROOT = repository_root()
DEVELOPMENT_SOURCE = REPO_ROOT / "configs" / "evaluations" / "task_1a_recovery_dev_v1.toml"
TRACKER = TrackerConfig(type="pd", kp=(10.0, 5.0), kd=(1.5, 0.8))
HOLD_S, PULSE_S, HORIZON_S = 0.05, 0.02, 1.0
CONFIGURATION = "feasible-best"

WARMUP_ALL_TEN = 6.0
WARMUP_SATURATED = 6.25
WARMUP_SINGLETON = 6.5
"""A warm-up of its own per test, so each starts from a store holding no evidence of its protocol."""

SCENARIOS = (RobustnessScenario("nominal", "nominal", (0.0, 0.0)),)


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


def _crafted(
    scenario_config: ManualScenarioConfig, rows: int, start: tuple[float, ...], *, rc: bool, saturation: float = 0.0
) -> RunArrays:
    """A feasible run holding the target, optionally reporting saturated samples after activation.

    Saturation is judged over the active segment, so samples inside the warm-up
    hold would report nothing at all.
    """
    dt = scenario_config.timing.dt
    on_target = np.asarray(joint_target(scenario_config), dtype=np.float64)
    t: NDArray[np.float64] = np.arange(rows, dtype=np.float64) * dt
    q = np.tile(np.asarray(start, dtype=np.float64), (rows, 1))
    q[max(1, rows // 4) :] = on_target
    dq = np.zeros((rows, 2), dtype=np.float64)
    zeros = np.zeros((rows, 2), dtype=np.float64)
    saturated = np.zeros(rows, dtype=np.int64)
    if saturation:
        active_from = int(np.searchsorted(t, WARMUP_SATURATED - 1e-9, side="left"))
        active = rows - active_from
        saturated[active_from : active_from + max(1, round(saturation * active))] = 1
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
        "saturation": saturated,
    }
    if rc:
        active = t >= (rows // 4) * dt - 1e-9
        readout = np.full((rows, 2), np.nan, dtype=np.float64)
        readout[active] = q[active]
        data["generator_output_q"] = readout
        data["phase"] = active.astype(np.int64)
    return RunArrays(data)


class _Crafted:
    """Crafts feasible runs for either arm, with a fixed saturated fraction."""

    def __init__(self, scenario_config: ManualScenarioConfig, *, saturation: float = 0.0) -> None:
        self.scenario = scenario_config
        self.saturation = saturation

    def __call__(self, scenario: object, controller: object, **kwargs: object) -> tuple[RunArrays, Termination]:
        del scenario, controller
        rows = round(cast("float", kwargs["duration_s"]) / self.scenario.timing.dt) + 1
        start = tuple(float(v) for v in cast("tuple[float, ...]", kwargs["initial_q"]))
        arrays = _crafted(self.scenario, rows, start, rc=kwargs.get("channels") is not None, saturation=self.saturation)
        return arrays, completed(float(arrays.arrays["t"][-1]), rows - 1)


def _entry(f: ManualFixture, arm: str) -> StudyModel:
    return next(e for e in f.manifest.entries if (e.configuration, e.arm.label) == (CONFIGURATION, arm))


def _runner(f: ManualFixture, *, saturation: float = 0.0) -> ManualEvaluationRunner:
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
        simulate_fn=_Crafted(load_manual_scenario(f.scenario_file), saturation=saturation),
    )


def _summary(f: ManualFixture, uri: str) -> RunSummary:
    """The stored run summary, rebuilt through the loader the rest of the repository uses."""
    path = f.store.path(uri, mode="read")
    return RunSummary.from_json(path.read_text(encoding="utf-8"))


def _stored_evidence(f: ManualFixture, identity: str) -> ManualModelEvidence:
    """One model's evidence loaded back from the manifest it was written to."""
    root = f.store.path(f"{REPORTS_PREFIX}/model/{identity}/manifest.json", mode="write").parent
    manifests = sorted(root.glob("manifest-*.json"))
    assert len(manifests) == 1, manifests
    return load_manual_model_evidence(manifests[0])


# --- the all-ten arm ----------------------------------------------------------------------------


def test_the_all_ten_arm_records_every_demonstration_it_trained_on(manual_fixture: ManualFixture) -> None:
    """M10 trains on the whole bank, so its runs name all ten demonstrations, not one and not a bank label.

    The binding is asserted by reading the stored evidence back: what a run
    trained on has to survive in the manifest, not only in the object this
    process happened to build.
    """
    f = manual_fixture
    runner = _runner(f)
    evidence = runner.evaluate(_entry(f, "M10"), warmup_s=WARMUP_ALL_TEN)
    assert evidence.assignment is None
    assert evidence.n_pairs == len(SCENARIOS) * 2
    expected = tuple(f.manifest.sources[name].artifact_id for name in ASSIGNMENTS)
    stored = _stored_evidence(f, evidence.identity)
    for pair in stored.pairs:
        assert pair.run is not None
        assert pair.run.sources == expected


def test_a_singleton_records_only_the_demonstration_it_trained_on(manual_fixture: ManualFixture) -> None:
    """The same binding for a one-demonstration arm: exactly its own parent, read back from the manifest."""
    f = manual_fixture
    runner = _runner(f)
    evidence = runner.evaluate(_entry(f, "S/D03"), warmup_s=WARMUP_SINGLETON)
    stored = _stored_evidence(f, evidence.identity)
    expected = (f.manifest.sources["D03"].artifact_id,)
    assert all(pair.run is not None and pair.run.sources == expected for pair in stored.pairs)


# --- the stored verdict agrees with the pair record -----------------------------------------------


def test_a_saturated_run_is_infeasible_in_its_stored_summary_too(manual_fixture: ManualFixture) -> None:
    """One verdict, two places: a run the pair calls infeasible is not a success in its own record."""
    f = manual_fixture
    evidence = _runner(f, saturation=0.5).evaluate(_entry(f, "S/D01"), warmup_s=WARMUP_SATURATED)
    pair = evidence.pairs[0]
    assert pair.outcome is not None
    assert pair.outcome.saturation_fraction > 0.005
    assert pair.status == "infeasible"
    assert pair.run is not None
    summary = _summary(f, pair.run.uri)
    assert "saturation" in summary.outcome.criteria
    assert summary.outcome.criteria["saturation"] is False
    assert summary.outcome.success is False
    assert "saturation" in summary.outcome.failed_criteria

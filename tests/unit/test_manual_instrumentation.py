# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-009: what the sweep measures about its own cost, and what that must not disturb.

The timing smoke check needs per-run and per-model costs, so the runner records
them. It records them in memory only: a run's wall time is a fact about this
machine, not about the experiment, and putting it inside content-addressed
evidence would make the same sweep produce different manifests on different
machines and on every re-run. The tests below pin both halves -- that the
measurements are taken, and that taking them leaves the evidence untouched.
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, cast

import numpy as np

from arm_rc_ctrl.controllers.tracking import TrackerConfig
from arm_rc_ctrl.data.manual_scenario import load_manual_scenario, manual_endpoint_positions
from arm_rc_ctrl.experiments.manual_evaluation import (
    PHASE_BUILD_BANK,
    PHASE_FIT,
    PHASE_SERVE_FIT,
    PHASE_SWEEP,
    PHASE_VERIFY_MODEL,
    PROGRESS_FILE,
    ManualEvaluationConfig,
    ManualEvaluationRunner,
    load_manual_evaluation_config,
    replay_bank_uri,
)
from arm_rc_ctrl.experiments.perturbations import RobustnessScenario
from arm_rc_ctrl.experiments.run_record import RunArrays
from arm_rc_ctrl.experiments.termination import completed
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import joint_target
from arm_rc_ctrl.storage import StorageRoot

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from arm_rc_ctrl.data.manual_scenario import ManualScenarioConfig
    from arm_rc_ctrl.experiments.manual_fixture import ManualFixture
    from arm_rc_ctrl.experiments.manual_study import StudyModel
    from arm_rc_ctrl.experiments.termination import Termination

REPO_ROOT = repository_root()
DEVELOPMENT_SOURCE = REPO_ROOT / "configs" / "evaluations" / "task_1a_recovery_dev_v1.toml"
TRACKER = TrackerConfig(type="pd", kp=(10.0, 5.0), kd=(1.5, 0.8))
REPLAY_CUTOFFS = (20.0, 20.0)
HOLD_S, PULSE_S, HORIZON_S, WARMUP_S = 0.05, 0.02, 1.0, 0.25
CONFIGURATION, ARM_LABEL = "feasible-best", "S/D01"

WARMUP_MEASURED = 0.5
WARMUP_SERVED = 0.55
WARMUP_UNCHANGED = 0.6
WARMUP_PHASES = 0.65
WARMUP_SERVE_FIT = 0.7
WARMUP_VERIFY = 0.75
"""A warm-up of its own per test. The phase tests additionally take a store of their own, because a
warm-up alone leaves one test able to serve another's evidence."""

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


class _Crafted:
    """Crafts feasible runs; the sweep times whatever this returns."""

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


def _runner(f: ManualFixture, store: StorageRoot | None = None) -> ManualEvaluationRunner:
    config, file = _evaluation(f)
    return ManualEvaluationRunner(
        store=f.store if store is None else store,
        inputs=f.inputs,
        config=config,
        evaluation_file=file,
        scenarios=SCENARIOS,
        trackers={"pd_v2": TRACKER, "computed_torque": TRACKER},
        root=f.root,
        execution=f.execution,
        provenance=f.provenance,
        simulate_fn=_Crafted(load_manual_scenario(f.scenario_file)),
    )


# --- what the sweep measures ------------------------------------------------------------------


def test_every_simulated_run_is_measured_once(manual_fixture: ManualFixture) -> None:
    """One timing per run this invocation simulated, naming the arm, scenario and tracker it belongs to."""
    f = manual_fixture
    runner = _runner(f)
    bank = runner.replay_bank("D01", warmup_s=WARMUP_MEASURED, replay_cutoffs=REPLAY_CUTOFFS)
    timings = runner.run_timings
    assert len(timings) == len(bank.pairs) == len(SCENARIOS) * 2
    assert {t.arm for t in timings} == {"replay"}
    assert {(t.scenario_id, t.tracker) for t in timings} == {(p.scenario_id, p.tracker) for p in bank.pairs}
    assert all(t.simulate_seconds >= 0 and t.persist_seconds >= 0 for t in timings)
    assert all(t.run_bytes > 0 and t.rows > 0 for t in timings)


def test_a_model_records_its_fit_cost_and_whether_the_cache_served_it(manual_fixture: ManualFixture) -> None:
    """The fit is the other half of a model's cost, and a cache hit reports what the fit originally took."""
    f = manual_fixture
    entry = _entry(f)
    first = _runner(f)
    first.evaluate(entry, warmup_s=WARMUP_MEASURED)
    fitted = first.model_timings[entry.label]
    assert fitted.fit_cache_hit is False
    assert fitted.fit_seconds > 0
    assert fitted.sweep_seconds >= 0
    assert fitted.runs == len(SCENARIOS) * 2
    second = _runner(f)
    second.evaluate(entry, warmup_s=WARMUP_SERVED)
    served = second.model_timings[entry.label]
    assert served.fit_cache_hit is True
    assert served.fit_seconds == fitted.fit_seconds, "a cache hit reports the cost of the fit it serves"


def test_runs_served_from_the_store_are_not_measured_again(manual_fixture: ManualFixture) -> None:
    """Only this invocation's own work is timed, so a projection is never inflated by served evidence."""
    f = manual_fixture
    _runner(f).replay_bank("D02", warmup_s=WARMUP_MEASURED, replay_cutoffs=REPLAY_CUTOFFS)
    second = _runner(f)
    second.replay_bank("D02", warmup_s=WARMUP_MEASURED, replay_cutoffs=REPLAY_CUTOFFS)
    assert second.run_timings == (), "a served bank simulated nothing, so it measured nothing"


def test_manifest_bytes_are_accumulated(manual_fixture: ManualFixture) -> None:
    """The evidence a sweep installs is part of what it costs to store."""
    f = manual_fixture
    runner = _runner(f)
    assert runner.manifest_bytes == 0
    runner.replay_bank("D03", warmup_s=WARMUP_MEASURED, replay_cutoffs=REPLAY_CUTOFFS)
    assert runner.manifest_bytes > 0


# --- and what measuring must not disturb -------------------------------------------------------


PAIR_FIELDS = {
    "index",
    "scenario_id",
    "kind",
    "tracker",
    "arm",
    "status",
    "initial_q",
    "outcome",
    "run",
    "pulse_start_s",
}
"""Exactly what a pair records; a measurement must never appear among them."""


def test_measuring_does_not_change_the_evidence_it_measures(manual_fixture: ManualFixture) -> None:
    """Timings stay out of the evidence: the stored manifest is what it would be unmeasured.

    Wall time is a fact about this machine. If it entered a content-addressed
    manifest, the same experiment would produce different evidence on different
    machines, and a re-run would collide with its own stored digest.
    """
    f = manual_fixture
    runner = _runner(f)
    bank = runner.replay_bank("D04", warmup_s=WARMUP_UNCHANGED, replay_cutoffs=REPLAY_CUTOFFS)
    assert runner.run_timings, "the sweep must actually have measured something"
    uri = replay_bank_uri(runner.conditions(WARMUP_UNCHANGED, REPLAY_CUTOFFS), "D04")
    directory = f.store.path(f"{uri}/{PROGRESS_FILE}", mode="write").parent
    manifests = sorted(directory.glob("manifest-*.json"))
    assert len(manifests) == 1, manifests
    text = manifests[0].read_text(encoding="utf-8")
    assert "seconds" not in text, "no measurement may reach the evidence"
    document = cast("dict[str, object]", json.loads(text))
    for pair in cast("list[dict[str, object]]", document["pairs"]):
        assert set(pair) == PAIR_FIELDS, f"an unexpected field entered the evidence: {sorted(pair)}"
    assert bank.identity == runner.replay_bank("D04", warmup_s=WARMUP_UNCHANGED, replay_cutoffs=REPLAY_CUTOFFS).identity


# --- the phases a sweep spends its time in ------------------------------------------------------


def _isolated(f: ManualFixture, path: Path) -> ManualEvaluationRunner:
    """A runner over a store of its own, so nothing another test produced can be served into this one."""
    path.mkdir(parents=True)
    return _runner(f, StorageRoot(path, repositories=(REPO_ROOT,)))


def test_a_span_that_encloses_other_measurements_says_so(manual_fixture: ManualFixture, tmp_path: Path) -> None:
    """Phases are only meaningful if it is stated which of them contain the others.

    The sweep encloses every run it simulates, so adding the sweep to those runs
    would count the same seconds twice. Each timing therefore declares whether it
    is an inclusive span or a disjoint phase, and the enclosure is asserted here
    rather than merely labelled.
    """
    f = manual_fixture
    runner = _isolated(f, tmp_path / "phases")
    runner.evaluate(_entry(f), warmup_s=WARMUP_PHASES)
    phases = runner.phase_timings
    assert {p.phase for p in phases} >= {PHASE_FIT, PHASE_SWEEP}
    sweep = next(p for p in phases if p.phase == PHASE_SWEEP)
    assert sweep.inclusive is True
    assert next(p for p in phases if p.phase == PHASE_FIT).inclusive is False
    enclosed = sum(t.simulate_seconds + t.persist_seconds for t in runner.run_timings)
    assert enclosed <= sweep.seconds + 1e-6, "the sweep span contains the runs it measured"
    assert all(p.seconds >= 0 for p in phases)


def test_building_a_replay_bank_is_a_span_over_the_runs_it_simulates(
    manual_fixture: ManualFixture, tmp_path: Path
) -> None:
    """The other enclosing span, measured the same way and for the same reason."""
    f = manual_fixture
    runner = _isolated(f, tmp_path / "bank")
    runner.replay_bank("D01", warmup_s=WARMUP_PHASES, replay_cutoffs=REPLAY_CUTOFFS)
    built = next(p for p in runner.phase_timings if p.phase == PHASE_BUILD_BANK)
    assert built.inclusive is True
    enclosed = sum(t.simulate_seconds + t.persist_seconds for t in runner.run_timings)
    assert enclosed <= built.seconds + 1e-6


def test_serving_a_cached_fit_is_measured_apart_from_the_fit_it_reports(
    manual_fixture: ManualFixture, tmp_path: Path
) -> None:
    """A cache hit refits the model to verify it, and that work is nowhere in ``fit_seconds``.

    ``fit_seconds`` reports what the fit originally cost, which is the contract
    a projection scales. The refit a hit performs is real work this invocation
    does, so it is measured as its own phase instead of being folded into a
    number that means something else.
    """
    f = manual_fixture
    entry = _entry(f)
    first = _isolated(f, tmp_path / "fitcache")
    first.evaluate(entry, warmup_s=WARMUP_SERVE_FIT)
    assert not [p for p in first.phase_timings if p.phase == PHASE_SERVE_FIT], "the first fit was no cache hit"

    # A different warm-up is a different protocol, so the MODEL is not served; its fit still is.
    second = _runner(f, first.store)
    second.evaluate(entry, warmup_s=WARMUP_SERVE_FIT + 0.01)
    served = second.model_timings[entry.label]
    assert served.fit_cache_hit is True
    assert served.fit_seconds == first.model_timings[entry.label].fit_seconds
    serving = next(p for p in second.phase_timings if p.phase == PHASE_SERVE_FIT)
    assert serving.inclusive is False
    assert serving.seconds > 0, "serving a cached fit refits it, and the refit is this invocation's own cost"


def test_verifying_served_evidence_is_measured(manual_fixture: ManualFixture, tmp_path: Path) -> None:
    """Serving a stored model re-reads and re-digests every run behind it; that is not free."""
    f = manual_fixture
    entry = _entry(f)
    first = _isolated(f, tmp_path / "verify")
    first.evaluate(entry, warmup_s=WARMUP_VERIFY)
    second = _runner(f, first.store)
    second.evaluate(entry, warmup_s=WARMUP_VERIFY)
    assert second.run_timings == (), "the second invocation simulated nothing"
    verifying = next(p for p in second.phase_timings if p.phase == PHASE_VERIFY_MODEL)
    assert verifying.inclusive is False
    assert verifying.seconds > 0

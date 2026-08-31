# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Feasibility rules and the documented penalty of the ESN trial objective (M3-004)."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import optuna
import pytest

from arm_rc_ctrl.data.preprocess import PreprocessResult, preprocess_demonstration
from arm_rc_ctrl.data.records import RawDemonstrationRecord, load_record
from arm_rc_ctrl.experiments import esn_objective
from arm_rc_ctrl.experiments.esn_objective import (
    TrialContext,
    TrialEvaluation,
    classify,
    development_cases,
    evaluate_point,
    make_objective,
)
from arm_rc_ctrl.experiments.esn_search import EsnSearchProtocol, TrialPoint, load_esn_search
from arm_rc_ctrl.experiments.run_record import RunArrays
from arm_rc_ctrl.experiments.studies import summarize
from arm_rc_ctrl.experiments.termination import Termination, completed, divergence, invalid_output, limit_violation
from arm_rc_ctrl.experiments.tuning import DevelopmentPulse, DevelopmentScenarios
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageRoot

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
PROTOCOL = REPO_ROOT / "configs" / "studies" / "esn_search_1a.toml"
MODEL = REPO_ROOT / "tests" / "fixtures" / "configs" / "esn_fixture.toml"
RAW_RECORD = REPO_ROOT / "tests" / "fixtures" / "records" / "raw-20260830-287036d83d46.toml"
RAW_LOG = REPO_ROOT / "tests" / "fixtures" / "raw" / "demo.sklog.npz"
SCENARIO = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
PREPROCESS = REPO_ROOT / "configs" / "preprocessing" / "default.toml"
FIXED_TIME = datetime(2026, 9, 3, 10, 0, 0, tzinfo=UTC)
POINT = TrialPoint(100, 0.9, 0.9, 0.3, 0.5, 31, 1e-2, 20.0, 20.0)
PULSE = DevelopmentPulse(2.5, 0.2, 3.0, 45.0)

type Simulate = Callable[..., tuple[RunArrays, Termination]]


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> tuple[EsnSearchProtocol, TrialContext]:
    """The committed protocol re-targeted at the fixture scenario/dataset with three development scenarios."""
    base = tmp_path_factory.mktemp("objective")
    root = base / "store"
    root.mkdir()
    store = StorageRoot(root, repositories=(REPO_ROOT,))
    raw = load_record(RAW_RECORD, RawDemonstrationRecord)
    store.path(raw.artifact.payload.uri, mode="write").write_bytes(RAW_LOG.read_bytes())
    records = base / "repo"
    (records / "data" / "records" / "processed").mkdir(parents=True)
    processed: PreprocessResult = preprocess_demonstration(
        RAW_RECORD, SCENARIO, PREPROCESS, store=store, records_root=records, exploratory=True, now=FIXED_TIME
    )
    protocol = replace(
        load_esn_search(PROTOCOL),
        scenario=SCENARIO,
        model=MODEL,
        comparison=(),
        development=DevelopmentScenarios(((0.0, 0.0), (0.02, 0.0)), (PULSE,)),
    )
    context = TrialContext.load(protocol, store=store, dataset_file=processed.record_file, records_root=records)
    return protocol, context


def crafted(
    context: TrialContext,
    *,
    error: float = 0.0,
    n: int | None = None,
    tip_offset: float = 0.0,
    saturation: float = 0.0,
) -> RunArrays:
    """Run arrays derived from the reference: joint error, optional truncation, tip offset, saturation share."""
    ref = context.reference
    count = ref.n_samples if n is None else min(n, ref.n_samples)
    dof = ref.q.shape[1]
    q = ref.q[:count] + error
    zeros = np.zeros((count, dof), dtype=np.float64)
    saturated = np.zeros(count, dtype=np.int64)
    saturated[: round(saturation * count)] = 1
    return RunArrays(
        {
            "t": ref.t[:count].copy(),
            "q": q,
            "dq": ref.dq[:count].copy(),
            "tip": ref.tip[:count] + tip_offset,
            "q_desired": ref.q[:count].copy(),
            "dq_desired_raw": ref.dq[:count].copy(),
            "dq_desired": ref.dq[:count].copy(),
            "ddq_desired_raw": zeros,
            "ddq_desired": zeros,
            "tracking_error": zeros - error,
            "tau_requested": zeros,
            "task_code": np.zeros((count, ref.task_code.shape[1]), dtype=np.float64),
            "saturation": saturated,
        }
    )


def fake_simulate(outcomes: list[tuple[RunArrays, Termination]]) -> Simulate:
    """A ``simulate`` stand-in returning the queued outcomes in order."""
    queue = list(outcomes)

    def simulate(*_args: object, **_kwargs: object) -> tuple[RunArrays, Termination]:
        return queue.pop(0)

    return simulate


def done(context: TrialContext) -> Termination:
    """A completed termination at the reference's end."""
    ref = context.reference
    return completed(float(ref.t[-1]), ref.n_samples - 1)


def test_classify_names_every_infeasibility_reason() -> None:
    """Divergence, limits, other early terminations, missed dwell criteria, and saturation are classified."""
    ok = {"completed": True, "in_tolerance": True}
    assert classify(divergence(1.0, 250, "blew up"), ok, 0.0, 0.05) == "divergence"
    assert classify(limit_violation(1.0, 250, "torque", 90.0, 60.0, joint=1), ok, 0.0, 0.05) == "limit_violation:torque"
    assert (
        classify(limit_violation(1.0, 250, "joint_velocity", 7.0, 6.0), ok, 0.0, 0.05)
        == "limit_violation:joint_velocity"
    )
    early = invalid_output(1.0, 250, "nan target", "non_finite")
    assert classify(early, ok, 0.0, 0.05) == "early_termination:invalid_output:non_finite"
    missed = {"completed": True, "in_tolerance": False, "stationary": False}
    assert classify(completed(5.0, 1250), missed, 0.0, 0.05) == "dwell:in_tolerance,stationary"
    assert classify(completed(5.0, 1250), ok, 0.2, 0.05) == "saturation"
    assert classify(completed(5.0, 1250), ok, None, 0.05) == "saturation"
    assert classify(completed(5.0, 1250), ok, 0.05, 0.05) is None


def test_development_cases_follow_the_protocol(prepared: tuple[EsnSearchProtocol, TrialContext]) -> None:
    """Posture offsets come first (added to the scenario's initial posture), then the force pulses."""
    protocol, context = prepared
    cases = development_cases(protocol, context.scenario)
    q0 = context.scenario.task.initial_q
    assert cases[0] == (tuple(q0), None)
    assert cases[1] == ((q0[0] + 0.02, q0[1]), None)
    assert cases[2] == (tuple(q0), PULSE)
    with pytest.raises(ValueError, match="entries"):
        development_cases(replace(protocol, development=DevelopmentScenarios(((0.0, 0.0, 0.0),))), context.scenario)


def _diverging(c: TrialContext) -> tuple[RunArrays, Termination]:
    return crafted(c, n=20), divergence(0.08, 19, "state grew without bound")


def _torque_limit(c: TrialContext) -> tuple[RunArrays, Termination]:
    return crafted(c, n=20), limit_violation(0.08, 19, "torque", 90.0, 60.0, joint=0)


def _early(c: TrialContext) -> tuple[RunArrays, Termination]:
    return crafted(c, n=20), invalid_output(0.08, 19, "nan", "non_finite")


def _dwell_miss(c: TrialContext) -> tuple[RunArrays, Termination]:
    return crafted(c, tip_offset=0.5), done(c)


def _saturated(c: TrialContext) -> tuple[RunArrays, Termination]:
    return crafted(c, saturation=0.5), done(c)


INFEASIBLE: dict[str, tuple[Callable[[TrialContext], tuple[RunArrays, Termination]], str, bool]] = {
    "divergence": (_diverging, "divergence", False),
    "torque-limit": (_torque_limit, "limit_violation:torque", False),
    "early-termination": (_early, "early_termination:invalid_output:non_finite", False),
    "dwell": (_dwell_miss, "dwell:", True),
    "saturation": (_saturated, "saturation", False),
}
"""Infeasible outcomes, their reason prefix, and whether the fixture's lenient dwell criteria must be tightened."""


def strict_dwell(context: TrialContext) -> TrialContext:
    """The context with a dwell criterion the fixture demonstration would meet but a 0.5 m tip offset cannot."""
    task = replace(context.scenario.task, dwell_min_fraction=1.0)
    return replace(context, scenario=replace(context.scenario, task=task))


@pytest.mark.parametrize("case", sorted(INFEASIBLE))
def test_an_infeasible_scenario_receives_the_penalty(
    prepared: tuple[EsnSearchProtocol, TrialContext], monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """The documented penalty replaces the objective; the reason and the components so far are kept."""
    protocol, context = prepared
    make, reason, strict = INFEASIBLE[case]
    if strict:
        context = strict_dwell(context)
    monkeypatch.setattr(esn_objective, "simulate", fake_simulate([make(context)]))
    evaluation = evaluate_point(protocol, context, POINT)
    assert evaluation.objective == protocol.objective.infeasible_penalty
    assert not evaluation.feasible
    assert evaluation.penalized
    assert evaluation.reason is not None
    assert evaluation.reason.startswith(f"scenario 0: {reason}")
    assert len(evaluation.components) == 1
    assert evaluation.components[0].reason is not None
    assert evaluation.components[0].reason.startswith(reason)
    assert evaluation.running == (protocol.objective.infeasible_penalty,)
    assert evaluation.scenarios_total == 3
    assert evaluation.fit_rmse is not None
    assert math.isfinite(evaluation.fit_rmse)


def test_a_feasible_trial_scores_the_median_movement_rmse(
    prepared: tuple[EsnSearchProtocol, TrialContext], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every scenario completes within the rules: the objective is the median RMSE and all components are kept."""
    protocol, context = prepared
    errors = (0.01, 0.03, 0.02)
    monkeypatch.setattr(
        esn_objective, "simulate", fake_simulate([(crafted(context, error=e), done(context)) for e in errors])
    )
    evaluation = evaluate_point(protocol, context, POINT)
    assert evaluation.feasible
    assert not evaluation.penalized
    assert evaluation.reason is None
    rmses = [c.move_joint_rmse for c in evaluation.components]
    assert all(r is not None and math.isclose(r, e, rel_tol=1e-6) for r, e in zip(rmses, errors, strict=True))
    assert math.isclose(evaluation.objective, 0.02, rel_tol=1e-6)
    assert len(evaluation.running) == 3
    assert math.isclose(evaluation.running[0], 0.01, rel_tol=1e-6)
    assert math.isclose(evaluation.running[1], 0.02, rel_tol=1e-6)
    assert [c.kind for c in evaluation.components] == ["posture", "posture", "force"]
    assert evaluation.components[2].pulse == PULSE
    assert all(c.saturation_fraction == 0.0 for c in evaluation.components)
    attrs = evaluation.attrs()
    assert attrs["feasible"] is True
    assert attrs["scenarios_evaluated"] == 3
    assert isinstance(attrs["components"], list)


def test_evaluation_stops_at_the_first_infeasible_scenario(
    prepared: tuple[EsnSearchProtocol, TrialContext], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once a scenario is infeasible the objective is decided; later scenarios are not simulated."""
    protocol, context = prepared
    outcomes = [
        (crafted(context, error=0.01), done(context)),
        (crafted(context, n=20), divergence(0.08, 19, "diverged")),
    ]
    monkeypatch.setattr(esn_objective, "simulate", fake_simulate(outcomes))
    evaluation = evaluate_point(protocol, context, POINT)
    assert len(evaluation.components) == 2
    assert evaluation.components[0].feasible
    assert not evaluation.components[1].feasible
    assert evaluation.reason == "scenario 1: divergence"
    assert evaluation.running[1] == protocol.objective.infeasible_penalty
    assert evaluation.objective == protocol.objective.infeasible_penalty


def test_a_training_failure_is_penalized_without_scenarios(
    prepared: tuple[EsnSearchProtocol, TrialContext], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A point the model cannot be trained on is infeasible with a training reason."""
    protocol, context = prepared

    def failing(*_args: object, **_kwargs: object) -> object:
        msg = "singular ridge system"
        raise ValueError(msg)

    monkeypatch.setattr(esn_objective, "create_recipe", failing)
    evaluation = evaluate_point(protocol, context, POINT)
    assert evaluation.reason == "training_failure:ValueError"
    assert evaluation.components == ()
    assert evaluation.fit_rmse is None
    assert evaluation.objective == protocol.objective.infeasible_penalty


def test_objective_reports_for_pruning_and_stores_every_component(
    prepared: tuple[EsnSearchProtocol, TrialContext], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trials report the running objective; a pruned trial is stopped and everything is stored as attributes."""
    protocol, context = prepared
    per_trial = iter([0.01, 0.5])
    calls = {"count": 0}

    def simulate(*_args: object, **_kwargs: object) -> tuple[RunArrays, Termination]:
        if calls["count"] % 3 == 0:
            simulate.error = next(per_trial)  # type: ignore[attr-defined]
        calls["count"] += 1
        return crafted(context, error=simulate.error), done(context)  # type: ignore[attr-defined]

    monkeypatch.setattr(esn_objective, "simulate", simulate)
    seen: list[TrialEvaluation] = []
    study = optuna.create_study(
        sampler=optuna.samplers.RandomSampler(seed=0), pruner=optuna.pruners.MedianPruner(n_startup_trials=1)
    )
    study.optimize(make_objective(protocol, context, on_evaluation=lambda _t, e: seen.append(e)), n_trials=2)
    first, second = study.trials
    assert first.state.name == "COMPLETE"
    assert math.isclose(first.value or 0.0, 0.01, rel_tol=1e-6)
    assert second.state.name == "PRUNED"
    assert seen[1].stopped_early
    assert seen[1].reason == "stopped by the pruner"
    assert calls["count"] == 4  # three scenarios, then one before the prune
    summary = summarize(study)
    assert summary.trials[0].flags["feasible"] is True
    assert (
        summary.trials[0].metrics["components.0.move_joint_rmse"]
        == first.user_attrs["components"][0]["move_joint_rmse"]
    )
    assert summary.trials[0].labels["components.2.kind"] == "force"
    assert summary.trials[1].flags["stopped_early"] is True
    assert summary.trials[1].intermediate_values == {"0": pytest.approx(0.5)}


def test_a_real_point_runs_the_fixture_closed_loop(prepared: tuple[EsnSearchProtocol, TrialContext]) -> None:
    """Without stand-ins the point is trained and simulated; the objective and components are consistent."""
    protocol, context = prepared
    nominal_only = replace(protocol, development=DevelopmentScenarios(((0.0, 0.0),)))
    evaluation = evaluate_point(nominal_only, context, POINT)
    assert evaluation.scenarios_total == 1
    assert len(evaluation.components) == 1
    component = evaluation.components[0]
    assert math.isfinite(evaluation.objective)
    if evaluation.feasible:
        assert component.move_joint_rmse is not None
        assert evaluation.objective == component.move_joint_rmse
    else:
        assert evaluation.objective == protocol.objective.infeasible_penalty
        assert evaluation.reason is not None
    assert component.boundary_jump is not None
    assert "completed" in component.criteria

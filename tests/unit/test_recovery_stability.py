# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-014: the recovery seed panel — leading trials, per-seed re-evaluation, summaries, and strict IO."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from arm_rc_ctrl.experiments import recovery_stability
from arm_rc_ctrl.experiments.esn_search import TrialPoint
from arm_rc_ctrl.experiments.recovery_objective import RecoveryTrialEvaluation
from arm_rc_ctrl.experiments.recovery_search import (
    RecoveryTrialPoint,
    load_recovery_search,
    recovery_protocol_digest,
)
from arm_rc_ctrl.experiments.recovery_stability import (
    ConfigurationStability,
    SeedOutcome,
    leading_trials,
    load_stability,
    render_markdown,
    run_stability,
    stability_to_json,
)
from arm_rc_ctrl.experiments.recovery_study import RecoveryStudyReport
from arm_rc_ctrl.experiments.studies import StudySummary, TrialRecord
from arm_rc_ctrl.provenance import collect_provenance
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from pathlib import Path

    from arm_rc_ctrl.experiments.recovery_objective import RecoveryTrialContext
    from arm_rc_ctrl.experiments.recovery_search import RecoverySearchProtocol

REPO_ROOT = repository_root()

_PROTOCOL = """
name = "stability-fixture"
scenario = "{scenario}"
model = "{model}"
formulation = "no_augmentation"
budget = 4
seed_bank = 1
attempt_factor = 4
development = "{development}"

[sampler]
kind = "tpe"
seed = 9
n_startup_trials = 2

[pruner]
kind = "none"

[objective]
kind = "worst_cell_median_gap_ratio"
infeasible_penalty = 10.0

[esn]
n_neurons = {{ low = 50, high = 150, step = 50 }}
spectral_radius = {{ low = 0.8, high = 1.3 }}
sparsity = {{ low = 0.5, high = 0.98 }}
leak_rate = {{ low = 0.01, high = 0.3, log = true }}
input_scaling = {{ low = 0.02, high = 0.5, log = true }}
seed = {{ low = 1, high = 1000 }}
alpha = {{ low = 1e-3, high = 1.0, log = true }}
velocity_cutoff_hz = {{ low = 5.0, high = 30.0, log = true }}
acceleration_cutoff_hz = {{ low = 5.0, high = 30.0, log = true }}

[space]
warmups_s = [0.0, 1.0]
"""


def _stub_context() -> RecoveryTrialContext:
    """A context stub carrying only the dataset payload the provenance step reads."""
    payload = SimpleNamespace(uri="armrc://processed/processed-test/samples.npz", sha256="0" * 64, size=1)
    artifact = SimpleNamespace(artifact_id="processed-test", payload=payload)
    return cast("RecoveryTrialContext", SimpleNamespace(dataset=SimpleNamespace(artifact=artifact)))


POINT = RecoveryTrialPoint(
    esn=TrialPoint(100, 0.9, 0.9, 0.1, 0.1, 31, 0.01, 20.0, 20.0), warmup_s=0.0, augmentation=None
)
CELLS = {
    "posture_small:pd_v2": 0.5,
    "posture_small:computed_torque": 0.5,
    "posture_large:pd_v2": 0.5,
    "posture_large:computed_torque": 0.5,
}


def _protocol(tmp_path: Path) -> RecoverySearchProtocol:
    file = tmp_path / "stability_fixture.toml"
    file.write_text(
        _PROTOCOL.format(
            scenario=(REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml").as_posix(),
            model=(REPO_ROOT / "tests" / "fixtures" / "configs" / "esn_fixture.toml").as_posix(),
            development=(REPO_ROOT / "configs" / "evaluations" / "task_1a_recovery_dev_v1.toml").as_posix(),
        ),
        encoding="utf-8",
    )
    return load_recovery_search(file)


def _trial(number: int, *, value: float | None, feasible: bool) -> TrialRecord:
    params = {k: float(v) for k, v in POINT.params().items()}
    return TrialRecord(
        number=number,
        state="COMPLETE",
        value=value,
        params=params,
        flags={"feasible": feasible},
        labels={} if feasible else {"reason": "scenario 0 [pd_v2]: dwell:dwell_stationary"},
    )


def _report(protocol: RecoverySearchProtocol, trials: tuple[TrialRecord, ...]) -> RecoveryStudyReport:
    feasible = [t for t in trials if t.flags.get("feasible") is True and t.value is not None]
    best = min(feasible, key=lambda t: (t.value, t.number)) if feasible else None
    summary = StudySummary(
        name=protocol.name,
        storage=f"armrc://optuna/{protocol.name}.db",
        direction="minimize",
        identity={},
        trials=trials,
        n_complete=len(trials),
        n_pruned=0,
        best_number=None if best is None else best.number,
        best_value=None if best is None else best.value,
        selection_rule="feasible",
    )
    return RecoveryStudyReport(
        protocol=protocol.name,
        protocol_file="configs/studies/stability_fixture.toml",
        protocol_sha256=recovery_protocol_digest(protocol),
        formulation="no_augmentation",
        dataset="processed-test",
        trackers={"pd_v2": "a" * 64, "computed_torque": "b" * 64},
        budget=4,
        trials_run=len(trials),
        summary=summary,
        best_point=None if best is None else POINT,
        n_feasible=len(feasible),
        provenance=collect_provenance({}, seeds={}, artifacts=[], exploratory=True),
    )


def _evaluation(point: RecoveryTrialPoint, *, feasible: bool, objective: float) -> RecoveryTrialEvaluation:
    if feasible:
        return RecoveryTrialEvaluation(
            point=point,
            objective=objective,
            feasible=True,
            penalized=False,
            reason=None,
            fit_rmse=0.001,
            scenarios_total=8,
            components=(),
            cells=dict(CELLS),
            running=(objective,),
        )
    return RecoveryTrialEvaluation(
        point=point,
        objective=objective,
        feasible=False,
        penalized=True,
        reason="scenario 1 [pd_v2]: limit_violation:joint_velocity",
        fit_rmse=0.001,
        scenarios_total=8,
        components=(),
        cells={},
        running=(objective,),
    )


def test_panel_reevaluates_leaders_with_every_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each leader runs once per panel seed with only the reservoir seed replaced; summaries re-derive."""
    protocol = _protocol(tmp_path)
    report = _report(protocol, (_trial(0, value=0.7, feasible=True), _trial(1, value=0.5, feasible=True)))
    assert leading_trials(report, 1) == [1]
    assert leading_trials(report, 5) == [1, 0]
    seen: list[tuple[int, int]] = []

    def fake_evaluate(
        _protocol: RecoverySearchProtocol, _context: object, point: RecoveryTrialPoint
    ) -> RecoveryTrialEvaluation:
        seen.append((point.esn.seed, point.esn.n_neurons))
        feasible = point.esn.seed != 303
        return _evaluation(point, feasible=feasible, objective=0.6 if feasible else 10.0)

    monkeypatch.setattr(recovery_stability, "evaluate_recovery_point", fake_evaluate)
    result = run_stability(
        report,
        tmp_path / "study.json",
        protocol,
        context=_stub_context(),
        top=2,
        seeds=[101, 202, 303],
        exploratory=True,
    )
    assert [c.trial for c in result.configurations] == [1, 0]
    assert seen == [(101, 100), (202, 100), (303, 100)] * 2
    first = result.configurations[0]
    assert first.point == POINT
    assert first.feasible_seeds == 2
    assert first.objective_median == pytest.approx(0.6)
    assert [o.feasible for o in first.outcomes] == [True, True, False]
    assert first.outcomes[2].reason is not None
    markdown = render_markdown(result)
    assert markdown.startswith("# Reservoir-seed sensitivity panel of `stability-fixture`")
    assert "worst class-by-tracker cell median" in markdown
    file = tmp_path / "stability.json"
    file.write_text(stability_to_json(result) + "\n", encoding="utf-8")
    assert load_stability(file) == result


def test_mismatched_protocols_and_empty_studies_are_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The panel binds to its study's protocol digest and needs at least one feasible trial."""
    protocol = _protocol(tmp_path)
    report = _report(protocol, (_trial(0, value=10.0, feasible=False),))

    def _must_not_evaluate(*_args: object, **_kwargs: object) -> RecoveryTrialEvaluation:
        pytest.fail("must not evaluate")

    monkeypatch.setattr(recovery_stability, "evaluate_recovery_point", _must_not_evaluate)
    with pytest.raises(ValueError, match="no feasible trial"):
        run_stability(
            report,
            tmp_path / "study.json",
            protocol,
            context=cast("RecoveryTrialContext", object()),
            top=3,
            seeds=[101],
            exploratory=True,
        )
    tampered = replace(report, protocol="other-study")
    with pytest.raises(ValueError, match="was not produced"):
        run_stability(
            tampered,
            tmp_path / "study.json",
            protocol,
            context=cast("RecoveryTrialContext", object()),
            top=1,
            seeds=[101],
            exploratory=True,
        )
    with pytest.raises(ValueError, match="distinct"):
        run_stability(
            report,
            tmp_path / "study.json",
            protocol,
            context=cast("RecoveryTrialContext", object()),
            top=1,
            seeds=[101, 101],
            exploratory=True,
        )


def test_summary_tampering_is_detected() -> None:
    """Stored configuration summaries must re-derive from their outcomes."""
    outcomes = (
        SeedOutcome(seed=101, feasible=True, objective=0.6, reason=None, pairs_evaluated=8),
        SeedOutcome(
            seed=202,
            feasible=False,
            objective=10.0,
            reason="scenario 0 [pd_v2]: dwell:dwell_stationary",
            pairs_evaluated=1,
        ),
    )
    good = ConfigurationStability(trial=1, label=None, point=POINT, own_objective=0.5, outcomes=outcomes)
    assert good.feasible_seeds == 1
    assert good.objective_median == pytest.approx(0.6)
    with pytest.raises(ValueError, match="does not match"):
        ConfigurationStability(
            trial=1,
            label=None,
            point=POINT,
            own_objective=0.5,
            outcomes=outcomes,
            feasible_seeds=2,
            objective_median=0.6,
            objective_min=0.6,
            objective_max=0.6,
        )

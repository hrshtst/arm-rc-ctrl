# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-009: the deterministic re-simulation subset, enumerated before the study runs.

M3MAN-011 re-simulates a sample from a clean checkout to show the runs reproduce
bitwise. Sampling only works if the sample is fixed in advance: chosen
afterwards, it could be the runs that happened to reproduce. So the subset is
the 24 models the budget subset measures, both frozen trackers, and five
scenarios taken as the first of each perturbation class in the study's own
order.

Only re-simulation is sampled. Artifact verification and metric recomputation
still cover the complete evidence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.data.manual_scenario import load_manual_scenario
from arm_rc_ctrl.experiments.manual_evaluation import evaluation_scenarios
from arm_rc_ctrl.experiments.manual_resimulation import (
    RESIMULATION_CLASSES,
    ResimulationSubset,
    resimulation_subset,
)
from arm_rc_ctrl.experiments.perturbations import load_development_robustness
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arm_rc_ctrl.experiments.manual_fixture import ManualFixture
    from arm_rc_ctrl.experiments.perturbations import RobustnessScenario

TRACKERS = ("pd_v2", "computed_torque")
DEVELOPMENT = repository_root() / "configs" / "evaluations" / "task_1a_recovery_dev_v1.toml"


def _cases(f: ManualFixture) -> tuple[RobustnessScenario, ...]:
    """The locked development cases at the fixture's own task, resolved as a caller resolves them.

    The development draws are named by the evaluation configuration rather than
    by the study manifest, so the caller supplies them; the subset never
    rediscovers a path of its own.
    """
    return evaluation_scenarios(load_development_robustness(DEVELOPMENT), load_manual_scenario(f.scenario_file))


def _subset(
    f: ManualFixture, *, classes: Sequence[str] = RESIMULATION_CLASSES, trackers: Sequence[str] = TRACKERS
) -> ResimulationSubset:
    """The frozen sample over the fixture's study and its locked cases."""
    return resimulation_subset(f.manifest, scenarios=_cases(f), trackers=trackers, classes=classes)


def test_the_subset_is_the_budget_models_over_five_frozen_scenarios(manual_fixture: ManualFixture) -> None:
    """24 models, both trackers, five scenarios: 240 model runs to re-simulate."""
    subset = _subset(manual_fixture)
    assert len(subset.models) == 24
    assert len(subset.scenarios) == 5
    assert subset.n_rc_runs == 24 * 2 * 5 == 240


def test_the_scenarios_are_the_first_of_each_class_in_frozen_order(manual_fixture: ManualFixture) -> None:
    """Chosen by position within each perturbation class, never by what a run produced."""
    subset = _subset(manual_fixture)
    assert tuple(scenario.kind for scenario in subset.scenarios) == RESIMULATION_CLASSES
    assert subset.scenarios[0].kind == "nominal"
    every = _cases(manual_fixture)
    for scenario in subset.scenarios:
        first = next(case for case in every if case.kind == scenario.kind)
        assert scenario.scenario_id == first.scenario_id, "the first of its class, not a later one"


def test_the_replay_side_is_counted_per_bank(manual_fixture: ManualFixture) -> None:
    """Six banks over the same trackers and scenarios: 60 replay runs, 300 in total."""
    subset = _subset(manual_fixture)
    assert subset.n_banks == 6
    assert subset.n_replay_runs == 6 * 2 * 5 == 60
    assert subset.n_runs == 300


def test_the_identities_are_enumerated_explicitly(manual_fixture: ManualFixture) -> None:
    """The audit needs to know which runs to redo, so every triple is named rather than derived later."""
    subset = _subset(manual_fixture)
    triples = subset.run_identities()
    assert len(triples) == subset.n_rc_runs
    assert len(set(triples)) == len(triples), "no run is listed twice"
    assert len({label for label, _scenario, _tracker in triples}) == 24
    assert {tracker for _label, _scenario, tracker in triples} == set(TRACKERS)
    assert {scenario for _label, scenario, _tracker in triples} == {
        scenario.scenario_id for scenario in subset.scenarios
    }


def test_the_subset_is_deterministic(manual_fixture: ManualFixture) -> None:
    """Frozen means frozen: the same manifest and cases yield the same enumeration every time."""
    assert _subset(manual_fixture).run_identities() == _subset(manual_fixture).run_identities()


def test_a_missing_perturbation_class_is_refused(manual_fixture: ManualFixture) -> None:
    """A class the study does not contain would silently shrink the sample the audit rests on."""
    with pytest.raises(ValueError, match="no scenario"):
        _subset(manual_fixture, classes=("nominal", "no_such_class"))

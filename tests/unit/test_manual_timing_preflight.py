# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-009: the benchmark refuses a shape it was not authorized to run, before it runs anything.

The authorized measurement is a precise size: 24 models over six replay banks,
one nominal scenario, two frozen trackers, 60 runs per invocation. An earlier
invocation of this command evaluated every locked scenario instead of the
nominal one and executed 3,900 runs, so the shape is now resolved and checked
before a single fit or simulation begins rather than discovered from the
summary afterwards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.experiments.manual_evaluation import load_manual_evaluation_config, manual_conditions
from arm_rc_ctrl.experiments.manual_timing import (
    AUTHORIZED_SHAPE,
    BudgetShape,
    budget_entries,
    preflight_budget,
)

if TYPE_CHECKING:
    from arm_rc_ctrl.experiments.manual_fixture import ManualFixture

TRACKERS = ("pd_v2", "computed_torque")
NOMINAL = ("nominal",)


def test_the_authorized_shape_is_the_one_that_was_sanctioned() -> None:
    """24 models, six banks, one scenario, two trackers, 60 runs -- and the arithmetic agrees."""
    shape = AUTHORIZED_SHAPE
    assert (shape.models, shape.banks, shape.scenarios, shape.trackers) == (24, 6, 1, 2)
    assert shape.runs == 60
    assert shape.models * shape.trackers * shape.scenarios == 48, "model runs"
    assert shape.banks * shape.trackers * shape.scenarios == 12, "replay runs"


def test_the_preflight_accepts_the_authorized_shape(manual_fixture: ManualFixture) -> None:
    """The frozen budget subset over one nominal scenario resolves to exactly what was authorized."""
    resolved = preflight_budget(budget_entries(manual_fixture.manifest), scenarios=NOMINAL, trackers=TRACKERS)
    assert resolved == AUTHORIZED_SHAPE


def test_the_preflight_refuses_more_scenarios_than_authorized(manual_fixture: ManualFixture) -> None:
    """The mistake that produced 3,900 runs: every locked scenario instead of the nominal one."""
    every = tuple(f"case-{index}" for index in range(65))
    with pytest.raises(ValueError, match="scenarios"):
        preflight_budget(budget_entries(manual_fixture.manifest), scenarios=every, trackers=TRACKERS)


def test_the_preflight_refuses_a_short_model_subset(manual_fixture: ManualFixture) -> None:
    """Fewer models would measure less than the budget rests on, and divide the projection by less."""
    entries = budget_entries(manual_fixture.manifest)[:4]
    with pytest.raises(ValueError, match="models"):
        preflight_budget(entries, scenarios=NOMINAL, trackers=TRACKERS)


def test_the_preflight_refuses_a_single_tracker(manual_fixture: ManualFixture) -> None:
    """Both frozen trackers are part of the authorized measurement."""
    with pytest.raises(ValueError, match="trackers"):
        preflight_budget(budget_entries(manual_fixture.manifest), scenarios=NOMINAL, trackers=("pd_v2",))


def test_the_preflight_names_what_it_found_against_what_it_expected(manual_fixture: ManualFixture) -> None:
    """A refusal has to say which figure was wrong, or the operator cannot tell what to change."""
    with pytest.raises(ValueError, match="authorized measurement") as failure:
        preflight_budget(budget_entries(manual_fixture.manifest), scenarios=("a", "b"), trackers=TRACKERS)
    message = str(failure.value)
    assert "2" in message, "the count it found"
    assert "1" in message, "the count it expected"


def test_an_explicit_expectation_can_be_stated(manual_fixture: ManualFixture) -> None:
    """A different sanctioned size is checked the same way rather than by disabling the check."""
    entries = budget_entries(manual_fixture.manifest)
    expected = BudgetShape(models=24, banks=6, scenarios=2, trackers=2, runs=120)
    assert preflight_budget(entries, scenarios=("a", "b"), trackers=TRACKERS, expected=expected) == expected


# --- the selection is part of what the evidence is keyed by -------------------------------------


def test_selected_scenario_ids_enter_the_evidence_identity() -> None:
    """A restricted sweep keys its evidence differently, so it can never serve a broader sweep's runs.

    This characterises behaviour that already holds: the conditions carry the
    scenario ids, and their identity is a digest over every field. It is pinned
    because the benchmark depends on it -- the nominal measurement and the
    broader sweep must not be able to reuse each other's evidence.
    """
    from arm_rc_ctrl.repo import repository_root

    root = repository_root()
    config_file = root / "configs" / "evaluations" / "task_1a_manual_dev_v1.toml"
    config = load_manual_evaluation_config(config_file)
    common = {
        "warmup_s": 0.25,
        "replay_cutoffs": (6.67, 5.59),
        "execution_identity": "a" * 64,
        "root": root,
    }
    one = manual_conditions(config, config_file, scenario_ids=("nominal",), **common)  # type: ignore[arg-type]
    two = manual_conditions(config, config_file, scenario_ids=("nominal", "force-12N-000deg"), **common)  # type: ignore[arg-type]
    assert one.scenario_ids == ("nominal",)
    assert one.identity != two.identity, "a different selection is a different protocol"

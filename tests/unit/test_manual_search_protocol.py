# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MS-001: the frozen nominal-only search protocol and its selection rule.

The protocol is what keeps the search honest: it binds the approved ranges and
the fixed conditions, it keeps perturbed scenarios out of the optimizer, and it
fixes how three configurations are chosen before any of them is evaluated on a
perturbed case. Everything here is pure configuration and arithmetic; nothing
fits a model or simulates a run.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.config import ConfigError
from arm_rc_ctrl.experiments.manual_search import (
    ScoredTrial,
    filter_mismatches,
    load_manual_search,
    nominal_success_fraction,
    protocol_digest,
    scope_mismatches,
    select_configurations,
)
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from collections.abc import Callable

PROTOCOL = repository_root() / "configs/studies/manual_esn_search_v1.toml"


def _edited(tmp_path: Path, replace: tuple[str, str]) -> Path:
    """A copy of the committed protocol with one edit, so a refusal is about that edit alone.

    The whole configuration tree is copied, because the protocol resolves the
    policy it inherits relative to its own directory.
    """
    target = tmp_path / "configs"
    shutil.copytree(PROTOCOL.parent.parent, target)
    copy = target / "studies" / PROTOCOL.name
    text = copy.read_text(encoding="utf-8")
    assert replace[0] in text, replace[0]
    copy.write_text(text.replace(*replace, 1), encoding="utf-8")
    return copy


# --- what the owner approved -----------------------------------------------------------------


def test_the_protocol_binds_every_approved_range_and_fixed_condition() -> None:
    """The committed protocol is the owner's approved domain, not a restatement of it."""
    protocol = load_manual_search(PROTOCOL)
    space = protocol.space
    assert (space.n_neurons.low, space.n_neurons.high, space.n_neurons.step) == (100, 400, 50)
    assert (space.spectral_radius.low, space.spectral_radius.high, space.spectral_radius.log) == (0.8, 1.3, False)
    assert (space.sparsity.low, space.sparsity.high, space.sparsity.log) == (0.5, 0.98, False)
    assert (space.leak_rate.low, space.leak_rate.high, space.leak_rate.log) == (0.01, 0.3, True)
    assert (space.input_scaling.low, space.input_scaling.high, space.input_scaling.log) == (0.02, 1.5, True)
    assert (space.alpha_0.low, space.alpha_0.high, space.alpha_0.log) == (0.001, 1.0, True)
    assert space.warmup_s == (0.0, 0.25, 0.5, 1.0, 2.0), "the inherited categorical set, zero included"
    assert protocol.fixed.reservoir_seed == 896, "fixed by the owner, never an Optuna parameter"
    assert protocol.fixed.trackers == ("pd_v2", "computed_torque")


def test_the_fixed_filter_policy_is_compared_with_the_configuration_it_names() -> None:
    """Restating the v4 cutoffs would let the protocol drift from the policy it claims to inherit."""
    protocol = load_manual_search(PROTOCOL)
    assert filter_mismatches(protocol) == []
    assert protocol.fixed.velocity_cutoff_hz == 29.980411525699598
    assert protocol.fixed.acceleration_cutoff_hz == 10.938122239871603
    assert protocol.fixed.max_dt_ratio == 3.0


def test_a_filter_value_that_no_longer_matches_its_source_is_reported(tmp_path: Path) -> None:
    """A drifted cutoff is named rather than silently searched under."""
    copy = _edited(tmp_path, ("velocity_cutoff_hz = 29.980411525699598", "velocity_cutoff_hz = 29.98"))
    mismatches = filter_mismatches(load_manual_search(copy))
    assert any("velocity_cutoff_hz" in text for text in mismatches), mismatches


# --- the optimizer never sees a perturbed case -------------------------------------------------


def test_the_optimizer_scope_is_exactly_the_nominal_case() -> None:
    """The whole design rests on perturbations being unknown to the optimizer."""
    protocol = load_manual_search(PROTOCOL)
    assert protocol.objective.scenarios == ("nominal",)
    assert protocol.objective.runs_per_candidate == len(protocol.fixed.trackers) == 2
    assert protocol.objective.direction == "maximize"
    assert scope_mismatches(protocol) == []


@pytest.mark.parametrize(
    "edit",
    [
        ('scenarios = ["nominal"]', 'scenarios = ["nominal", "posture-small-20261201-01"]'),
        ('scenarios = ["nominal"]', 'scenarios = ["posture-small-20261201-01"]'),
    ],
)
def test_a_protocol_that_widens_the_optimizer_scope_is_refused(tmp_path: Path, edit: tuple[str, str]) -> None:
    """Widening the optimizer's scenarios is the failure this experiment cannot tolerate."""
    with pytest.raises(ConfigError, match="nominal"):
        load_manual_search(_edited(tmp_path, edit))


@pytest.mark.parametrize(
    "parameter",
    ["velocity_cutoff_hz = { low = 1.0, high = 2.0 }", "seed = { low = 1, high = 2 }"],
)
def test_the_search_space_cannot_tune_a_fixed_condition(tmp_path: Path, parameter: str) -> None:
    """Trackers, filters and the reservoir seed are fixed; an unknown search key is refused."""
    with pytest.raises(ConfigError):
        load_manual_search(_edited(tmp_path, ("[space]", f"[space]\n{parameter}")))


# --- the objective --------------------------------------------------------------------------


@pytest.mark.parametrize(("successes", "expected"), [(0, 0.0), (1, 0.5), (2, 1.0)])
def test_the_objective_is_the_nominal_success_fraction(successes: int, expected: float) -> None:
    """Two tracker runs give a coarse score with exactly three values."""
    assert nominal_success_fraction(successes, runs=2) == expected


@pytest.mark.parametrize(
    ("successes", "runs", "complaint"),
    [(3, 2, "successes"), (-1, 2, "successes"), (0, 0, "at least one run")],
)
def test_an_impossible_success_count_is_refused(successes: int, runs: int, complaint: str) -> None:
    """A score outside its own denominator would silently rank a candidate that was never evaluated."""
    with pytest.raises(ValueError, match=complaint):
        nominal_success_fraction(successes, runs=runs)


# --- the selection rule ----------------------------------------------------------------------


def _trial(number: int, score: float | None, *, n_neurons: int = 100, warmup_s: float = 0.0) -> ScoredTrial:
    return ScoredTrial(
        number=number,
        score=score,
        point={"n_neurons": n_neurons, "warmup_s": warmup_s},
    )


def test_selection_takes_the_highest_scores_and_breaks_ties_by_trial_number() -> None:
    """Descending nominal score, then the earliest trial: the rule the owner recorded."""
    trials = [
        _trial(0, 0.5, n_neurons=100),
        _trial(1, 1.0, n_neurons=150),
        _trial(2, 1.0, n_neurons=200),
        _trial(3, 0.0, n_neurons=250),
        _trial(4, 1.0, n_neurons=300),
    ]
    selection = select_configurations(trials, n_configurations=3)
    assert [chosen.number for chosen in selection.chosen] == [1, 2, 4]
    assert selection.shortfall == 0
    assert selection.label == "highest nominal scores"


def test_a_repeated_parameter_point_is_represented_by_its_earliest_trial() -> None:
    """Ten trials at one point are one configuration, not three."""
    trials = [
        _trial(0, 1.0, n_neurons=100),
        _trial(1, 1.0, n_neurons=100),
        _trial(2, 1.0, n_neurons=150),
        _trial(3, 0.5, n_neurons=200),
    ]
    selection = select_configurations(trials, n_configurations=3)
    assert [chosen.number for chosen in selection.chosen] == [0, 2, 3]


@pytest.mark.parametrize("score", [None])
def test_an_unscored_candidate_is_never_selected(score: float | None) -> None:
    """A fit failure or an interrupted run is retained, but it is not a scored configuration."""
    trials = [_trial(0, score, n_neurons=100), _trial(1, 1.0, n_neurons=150)]
    selection = select_configurations(trials, n_configurations=3)
    assert [chosen.number for chosen in selection.chosen] == [1]
    assert selection.shortfall == 2


def test_a_shortfall_is_reported_rather_than_filled_or_raised() -> None:
    """Fewer than three distinct scored candidates is a reported result, not a reason to search more."""
    selection = select_configurations([_trial(0, 1.0)], n_configurations=3)
    assert len(selection.chosen) == 1
    assert selection.shortfall == 2


def test_selection_is_stable_under_the_order_trials_arrive_in() -> None:
    """A resumed study may report trials out of order; the frozen rule must not depend on that."""
    trials = [_trial(4, 1.0, n_neurons=300), _trial(1, 1.0, n_neurons=150), _trial(2, 1.0, n_neurons=200)]
    assert [c.number for c in select_configurations(trials, n_configurations=3).chosen] == [1, 2, 4]


# --- budgets and the comparison scope ----------------------------------------------------------


def test_the_budget_and_comparison_scope_are_the_approved_ones() -> None:
    """The caps and the final scope are frozen here so execution cannot quietly enlarge them."""
    protocol = load_manual_search(PROTOCOL)
    budget = protocol.budget
    assert budget.trials == 100
    assert (budget.hours, budget.gib) == (10.0, 20.0)
    assert budget.shared_with_comparison is True
    assert budget.parallel_trials == 1, "serial trials to begin with"
    comparison = protocol.comparison
    assert comparison.arms == ("S", "M10", "R10", "C10", "replay")
    assert comparison.models_per_configuration == 31
    assert comparison.replay_banks_per_configuration == 10
    assert comparison.scenarios == 65
    assert comparison.sequential is True
    assert protocol.selection.n_configurations == 3


def test_the_protocol_digest_is_stable_and_changes_with_the_protocol(tmp_path: Path) -> None:
    """The digest is the study identity: equal protocols agree, and one edited value does not."""
    protocol = load_manual_search(PROTOCOL)
    assert protocol_digest(protocol) == protocol_digest(load_manual_search(PROTOCOL))
    edited = load_manual_search(_edited(tmp_path, ("trials = 100", "trials = 99")))
    assert protocol_digest(edited) != protocol_digest(protocol)


@pytest.mark.parametrize(
    "edit",
    [
        ("trials = 100", "trials = 101"),
        ("hours = 10.0", "hours = 12.0"),
        ("gib = 20.0", "gib = 24.0"),
    ],
)
def test_an_enlarged_budget_is_refused(tmp_path: Path, edit: tuple[str, str]) -> None:
    """The caps are the owner's; a protocol may tighten them, never exceed them."""
    with pytest.raises(ConfigError, match="approved"):
        load_manual_search(_edited(tmp_path, edit))


def test_the_sampler_seed_is_recorded_in_a_namespace_of_its_own() -> None:
    """A reused seed would couple this search's draws to another study's."""
    protocol = load_manual_search(PROTOCOL)
    assert protocol.sampler.seed == 20270301
    assert protocol.sampler.kind == "tpe"
    others: set[int] = set()
    for path in sorted((repository_root() / "configs").rglob("*.toml")):
        if path == PROTOCOL:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("seed = ") and stripped.removeprefix("seed = ").strip().isdigit():
                others.add(int(stripped.removeprefix("seed = ").strip()))
    assert protocol.sampler.seed not in others, "the sampler seed is a new namespace"


def _rule_text() -> str:
    return (repository_root() / "docs/experiments/task_1a_manual_esn_search/plan.md").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "promise",
    [
        "sequentially, in their selection order",
        "highest *nominal* scores",
        "process-isolated",
    ],
)
def test_the_protocol_freezes_what_the_plan_promises(promise: str) -> None:
    """The frozen protocol and the approved plan say the same thing about scope and execution."""
    assert promise in _rule_text()


def test_the_loader_refuses_an_unknown_key(tmp_path: Path) -> None:
    """Unknown keys are rejected rather than silently ignored, as everywhere else in this project."""
    with pytest.raises(ConfigError):
        load_manual_search(_edited(tmp_path, ("[budget]", "[budget]\nextra_allowance = 5")))


def _callables() -> tuple[Callable[..., object], ...]:
    return (load_manual_search, protocol_digest, select_configurations, nominal_success_fraction)


def test_the_module_exposes_the_pieces_the_later_tasks_need() -> None:
    """M3MS-002 and M3MS-003 build on these; they are public on purpose."""
    assert all(callable(item) for item in _callables())

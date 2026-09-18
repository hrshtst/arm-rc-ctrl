# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-009: the frozen rule choosing which cases the final report illustrates (plan section 7.1).

Selecting the study's models or evaluations on outcome would bias the
experiment. Selecting which of its results to *show* does not, and a report that
never showed a failure would misrepresent what happened. So this rule reads
verdicts, but it is frozen before execution and applied afterwards, and what it
yields are illustrations: aggregate conclusions come from the whole study.

A category the rule looked for and did not find is reported as absent rather
than quietly dropped, so a reader can tell the difference between "this contrast
did not occur" and "nobody looked".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "ILLUSTRATION_ARMS",
    "ILLUSTRATION_CATEGORIES",
    "ArmVerdict",
    "IllustrationCase",
    "Selection",
    "representative_cases",
]

ILLUSTRATION_ARMS: Final = ("S", "M10", "R10", "C10")
"""The four comparison arms; a case is only a comparison if all of them are present."""
ILLUSTRATION_CATEGORIES: Final = ("nominal", "all_ten_wins", "singleton_wins", "first_failure")
"""Every category the rule looks for, in the order the report presents them."""


@dataclass(frozen=True)
class ArmVerdict:
    """Whether one arm succeeded on one scenario under one tracker."""

    scenario_id: str
    tracker: str
    arm: str
    succeeded: bool


@dataclass(frozen=True)
class IllustrationCase:
    """One scenario the report illustrates, with the categories that chose it."""

    scenario_id: str
    categories: tuple[str, ...]
    arms: dict[str, bool]
    """Each comparison arm's verdict, so the figure shows all four beside the replay baseline."""


@dataclass(frozen=True)
class Selection:
    """The cases the frozen rule chose, and the categories that found nothing."""

    cases: tuple[IllustrationCase, ...]
    categories: dict[str, tuple[str, ...]]
    absent: tuple[str, ...]
    """Categories that occurred nowhere; stated so a reader sees the rule looked."""


def _by_scenario(verdicts: Sequence[ArmVerdict], order: Sequence[str]) -> dict[str, dict[str, bool]]:
    """Group verdicts by scenario, refusing any that is not one complete four-arm comparison under one tracker.

    Grouping is by scenario, so verdicts from two trackers -- or two verdicts for
    one arm -- would collapse into a single entry and silently keep whichever came
    last. One application of the rule covers one configuration under one tracker,
    and that is enforced here rather than left to the caller.
    """
    trackers = sorted({verdict.tracker for verdict in verdicts})
    if len(trackers) > 1:
        msg = f"the rule is applied to one tracker at a time, got verdicts for {trackers}"
        raise ValueError(msg)
    grouped: dict[str, dict[str, bool]] = {}
    for verdict in verdicts:
        arms = grouped.setdefault(verdict.scenario_id, {})
        if verdict.arm in arms:
            msg = f"{verdict.scenario_id}: arm {verdict.arm} has a verdict more than once"
            raise ValueError(msg)
        arms[verdict.arm] = verdict.succeeded
    for scenario_id, arms in grouped.items():
        if set(arms) != set(ILLUSTRATION_ARMS):
            msg = f"{scenario_id}: an illustrated case needs all four arms, got {sorted(arms)}"
            raise ValueError(msg)
    unknown = sorted(set(grouped) - set(order))
    if unknown:
        msg = f"scenarios outside the frozen order: {unknown}"
        raise ValueError(msg)
    return grouped


def representative_cases(verdicts: Sequence[ArmVerdict], *, order: Sequence[str]) -> Selection:
    """Choose the cases to illustrate, by a rule fixed before any of these verdicts existed.

    Always the nominal comparison; then the first scenario in frozen order where
    the all-ten arm succeeds and the singleton fails, the first showing the
    reverse, and the first in which any comparison arm fails. Cases are
    deduplicated in frozen order and each records every category that chose it.
    """
    grouped = _by_scenario(verdicts, order)
    ranked = [scenario_id for scenario_id in order if scenario_id in grouped]

    def first(predicate: Callable[[dict[str, bool]], bool]) -> tuple[str, ...]:
        for scenario_id in ranked:
            if predicate(grouped[scenario_id]):
                return (scenario_id,)
        return ()

    found: dict[str, tuple[str, ...]] = {
        "nominal": tuple(scenario_id for scenario_id in ranked if scenario_id == "nominal"),
        "all_ten_wins": first(lambda arms: arms["M10"] and not arms["S"]),
        "singleton_wins": first(lambda arms: arms["S"] and not arms["M10"]),
        "first_failure": first(lambda arms: not all(arms.values())),
    }
    chosen: dict[str, list[str]] = {}
    for category in ILLUSTRATION_CATEGORIES:
        for scenario_id in found[category]:
            chosen.setdefault(scenario_id, []).append(category)
    cases = tuple(
        IllustrationCase(
            scenario_id=scenario_id,
            categories=tuple(chosen[scenario_id]),
            arms=dict(grouped[scenario_id]),
        )
        for scenario_id in ranked
        if scenario_id in chosen
    )
    return Selection(
        cases=cases,
        categories=found,
        absent=tuple(category for category in ILLUSTRATION_CATEGORIES if not found[category]),
    )
